"""Replay routes: real MAST shots (FAIR-MAST cache) served beside the physics model."""

from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from zarr.errors import GroupNotFoundError

from fusionlab import mast
from fusionlab.compute import warm_up as compute_warm_up
from fusionlab.compute.status import capabilities as compute_capabilities
from fusionlab.compute.status import provenance, run_task
from fusionlab.physics import (
    BETA_N_LIMIT,
    LIMIT_NAMES,
    M_D,
    replay,
    tau_coeff_H,
    volume,
)

router = APIRouter()

ATTRIBUTION = "MAST data: UKAEA FAIR-MAST (mastapp.site), CC BY-SA 4.0"
_TRACES = ("t_s", "Ip_MA", "B_T", "n_e20", "P_ohm_MW", "P_nbi_MW", "P_rad_MW", "W_MJ", "q95", "beta_N", "kappa",
           "a_m", "R_m", "Te0_keV", "R_mag_m", "Z_mag_m")

logger = logging.getLogger(__name__)


def _j(a, nd=4):
    """numpy -> JSON-safe nested lists: rounded, NaN/inf as null."""
    a = np.asarray(a)
    if a.dtype == bool or a.dtype.kind in "iu":
        return a.tolist()
    a = np.round(a.astype(float), nd)
    return np.where(np.isfinite(a), a, None).tolist()


@lru_cache(maxsize=16)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _shot_cached(shot_id: int) -> dict:
    return mast.load_shot(shot_id)


# One archive fetch per shot id, however many requests want it: compare fans out over ids in parallel.
_inflight: dict[int, dict] = {}
_inflight_lock = threading.Lock()
_FETCH_WAIT_S = 600   # a fetch takes seconds to minutes; a waiter gives up and asks the client to retry


def _shot(shot_id: int) -> dict:
    """A cached shot from the process cache; an uncached catalog shot through load_shot()'s fetch-and-cache path."""
    if shot_id in mast.cached_shots():
        return _shot_cached(shot_id)
    if not bool(np.any(_catalog()["shot_id"] == shot_id)):
        raise HTTPException(404, f"shot {shot_id} is not in the MAST catalog (data/mast_db.npz)")
    with _inflight_lock:
        job = _inflight.get(shot_id)
        if job is None:
            job = _inflight[shot_id] = {"done": threading.Event(), "err": None}
            owner = True
        else:
            owner = False
    if not owner:
        if not job["done"].wait(_FETCH_WAIT_S):
            raise HTTPException(503, f"shot {shot_id} is still being fetched from FAIR-MAST — retry shortly")
        if job["err"] == "unpublished":
            raise HTTPException(404, f"shot {shot_id} is in the catalog but its level-2 data is not published in FAIR-MAST")
        if job["err"]:
            raise HTTPException(502, f"shot {shot_id} could not be fetched from FAIR-MAST: {job['err']}")
    else:
        try:
            mast.load_shot(shot_id)   # fetches from S3 + REST, writes the local cache atomically (fusionlab.mast)
        except GroupNotFoundError as err:
            job["err"] = "unpublished"   # waiters answer with the same 404, not a 502
            raise HTTPException(
                404, f"shot {shot_id} is in the catalog but its level-2 data is not published in FAIR-MAST"
            ) from err
        except Exception as e:
            job["err"] = type(e).__name__
            logger.warning("fetch of shot %s from FAIR-MAST failed", shot_id, exc_info=True)
        finally:
            with _inflight_lock:
                _inflight.pop(shot_id, None)
            job["done"].set()
    if shot_id in mast.cached_shots():
        return _shot_cached(shot_id)
    raise HTTPException(502, f"shot {shot_id} could not be fetched from FAIR-MAST")


@router.get("/shots")
def shots():
    """Cached shots with their logbook text. Uncached catalog shots are fetched on demand by /replay/{id}."""
    out = []
    for sid in mast.cached_shots():
        s = _shot(sid)
        out.append({**s["meta"], "n_slices": int(s["t_s"].size), "t_start_s": float(s["t_s"][0]),
                    "t_end_s": float(s["t_s"][-1]), "Ip_max_MA": round(float(np.nanmax(s["Ip_MA"])), 3),
                    "P_nbi_max_MW": round(float(np.nanmax(s["P_nbi_MW"])), 2)})
    return {"shots": out, "attribution": ATTRIBUTION}


@router.get("/replay/{shot_id}")
def replay_shot(shot_id: int, H: float = 1.0):
    """Measured traces, model traces on the same time base, limits, and the geometry to draw the shot."""
    s = _shot(shot_id)
    r = replay(s, H=H)
    summary = _summary(s, r)
    _add_hybrid(s, r, summary)
    eq = _eq_psi(shot_id)
    if eq is not None:
        summary["eq_surrogate"] = {"held_out": eq["held_out"], "median_rel_l2": round(float(np.median(eq["rel_l2"])), 4),
                                   "p95_rel_l2": round(float(np.percentile(eq["rel_l2"], 95)), 4),
                                   "computed_on": eq.get("computed_on")}
    return {
        "meta": s["meta"], "attribution": ATTRIBUTION, "limit_names": LIMIT_NAMES, "summary": summary,
        "measured": {k: _j(s[k]) for k in _TRACES if k in s},
        "model": {k: _j(v) for k, v in r.items()},
        "lcfs": {"R": _j(s["lcfs_R"], 3), "Z": _j(s["lcfs_Z"], 3)},
        "wall": {"R": _j(s["wall_R"], 3), "Z": _j(s["wall_Z"], 3)} if "wall_R" in s else None,
        "coils": {k[5:]: _j(s[k], 3) for k in ("coil_R", "coil_Z", "coil_W", "coil_H")} if "coil_R" in s else None,
        "psi_grid": {"R": _j(s["psi_R"], 3), "Z": _j(s["psi_Z"], 3)},
    }


@router.get("/replay/{shot_id}/psi/{i}")
def replay_psi(shot_id: int, i: int):
    """Normalised poloidal flux (0 axis, 1 separatrix) on the EFIT grid for one time slice, rows = Z."""
    s = _shot(shot_id)
    if not 0 <= i < s["t_s"].size:
        raise HTTPException(404, "time index out of range")
    out = {"i": i, "t_s": float(s["t_s"][i]), "psi_n": _j(s["psi_n"][i], 3)}
    eq = _eq_psi(shot_id)
    if eq is not None:
        out["surrogate"] = {"psi_n": _j(eq["psi_n"][i], 3), "rel_l2": _j(eq["rel_l2"][i], 4),
                            "computed_on": eq.get("computed_on")}
    if "ts_R" in s:
        out["thomson"] = {"R": _j(s["ts_R"], 3), "Te_keV": _j(s["ts_Te_keV"][i], 3), "ne_e20": _j(s["ts_ne_e20"][i], 3)}
    return out


@router.post("/replay/{shot_id}/usd")
def replay_usd(shot_id: int, fmt: str = "usdc"):
    """OpenUSD export (Omniverse-compatible): vessel, PF coils and the plasma boundary time-sampled over the shot.

    POST, not GET: the export recomputes the stage and writes out/ on every hit — a state-changing
    operation must not be triggerable by navigation, prefetch or crawlers (audit F1)."""
    from fusionlab.usd_export import FORMATS, export_shot
    if f".{fmt}" not in FORMATS:
        raise HTTPException(422, f"fmt must be one of {sorted(x.lstrip('.') for x in FORMATS)}")
    out = Path(__file__).resolve().parent.parent / "out" / f"mast_{shot_id}.{fmt}"
    tmp = out.with_name(f".{out.stem}.tmp{os.getpid()}{out.suffix}")   # dotfile, pid-unique, extension kept (export_shot validates the suffix)
    try:
        try:   # with Warp-traced field lines when Warp can run here; the plain stage otherwise
            from fusionlab.fieldlines import export_shot_with_field_lines
            export_shot_with_field_lines(_shot(shot_id), tmp)
        except Exception:
            logger.warning("USD field-line export failed; falling back to plain stage", exc_info=True)
            export_shot(_shot(shot_id), tmp)
        os.replace(tmp, out)   # atomic same-directory publish: readers only ever see a complete stage
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return FileResponse(out, filename=out.name, media_type="application/octet-stream")


@lru_cache(maxsize=8)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _lines(shot_id: int):
    """Field lines for every slice of a shot in one Warp launch, plus traced q at psi_N = 0.95 beside EFIT's q95."""
    s = _shot(shot_id)
    out = run_task("trace_fieldlines", {"shot": s})
    where = provenance("trace_fieldlines").get("host")   # captured now: the cache keeps it honest
    pts, qc = out["pts"], out["q_check"]                    # (time, line, point, xyz)
    q = np.full(s["t_s"].size, np.nan)
    q[np.asarray(qc["slice"], dtype=int)] = qc["q_traced"]
    return pts, q, qc["summary"], out["device"], out["psi_n_start"], where


@router.get("/replay/{shot_id}/fieldlines/{i}")
def replay_fieldlines(shot_id: int, i: int):
    """Field lines of the EFIT reconstruction at one time slice (axisymmetric; no islands or 3D fields)."""
    s = _shot(shot_id)
    if not 0 <= i < s["t_s"].size:
        raise HTTPException(404, "time index out of range")
    try:
        pts, q, summary, device, psi_n_start, computed_on = _lines(shot_id)
    except Exception as e:   # no Warp / no usable device: the rest of the replay still works
        raise HTTPException(503, f"field-line tracing unavailable: {type(e).__name__}") from e
    return {"i": i, "t_s": float(s["t_s"][i]), "psi_n_start": list(psi_n_start), "device": device,
            "computed_on": computed_on,
            "lines": [{"x": _j(line[:, 0], 3), "y": _j(line[:, 1], 3), "z": _j(line[:, 2], 3)} for line in pts[i]],
            "q95_traced": _j(q[i], 3), "q95_efit": _j(s["q95"][i], 3),
            "shot_check": {k: summary[k] for k in ("n_compared", "median_rel_err", "p95_rel_err", "psi_n_drift_max")}}


@lru_cache(maxsize=1)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _eq_model():
    """The equilibrium surrogate's holdout shots. None when no trained model is on disk."""
    if not compute_capabilities().get("eq_surrogate_available"):
        return None
    metrics = run_task("model_metrics", {"model": "eq_surrogate"})
    return set(metrics["test_shot_ids"])


@lru_cache(maxsize=8)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _eq_psi(shot_id: int):
    """PhysicsNeMo reconstruction of psi from the magnetic sensors, every slice of the shot in one batch.
    Normalised with EFIT's axis and boundary flux so the two maps share contour levels; error is on psi itself."""
    s = _shot(shot_id)
    try:
        held_out = _eq_model()
        if held_out is None or "eq_inputs" not in s:
            return None
        psi = run_task("run_eq_surrogate", {"eq_inputs": s["eq_inputs"]})   # one batch, where it ran last
    except Exception:   # the replay works without the surrogate
        logger.warning("equilibrium surrogate overlay failed; serving the replay without it", exc_info=True)
        return None
    ax, bd = s["psi_axis_Wb"][:, None, None], s["psi_bnd_Wb"][:, None, None]
    psi_efit = ax + s["psi_n"] * (bd - ax)
    def flat(a):
        return a.reshape(len(a), -1)
    rel = np.linalg.norm(flat(psi - psi_efit), axis=1) / np.linalg.norm(flat(psi_efit), axis=1)
    return {"psi_n": (psi - ax) / (bd - ax), "rel_l2": rel, "held_out": shot_id in held_out,
            "computed_on": provenance("run_eq_surrogate").get("host")}


def _summary(s: dict, r: dict) -> dict:
    """Numbers for the insight panel, taken over near-steady slices only. Medians are null when the model
    has nothing to say (a shot without line-average density has an all-NaN model) — never NaN: the JSON
    encoder rejects it and the panel shows an honest gap."""
    ok = r["steady"] & np.isfinite(r["H98"]) & np.isfinite(r["H89"])
    worst = np.nan_to_num(r["worst_limit"], nan=0.0)
    k = int(worst.argmax())
    def med(a):
        if not ok.any():
            return None
        m = float(np.median(a[ok]))
        return round(m, 2) if np.isfinite(m) else None
    p_lh = float(np.nanmedian(r["P_LH_MW"])) if np.isfinite(r["P_LH_MW"]).any() else None
    return {"n_steady": int(ok.sum()), "H98_median": med(r["H98"]), "H89_median": med(r["H89"]),
            "peak_limit": LIMIT_NAMES[int(r["binding"][k])], "peak_limit_fraction": round(float(worst[k]), 2),
            "peak_limit_t_s": round(float(s["t_s"][k]), 3),
            "P_LH_median_MW": round(p_lh, 2) if p_lh is not None else None, "P_loss_median_MW": med(r["P_loss_MW"])}


def _add_hybrid(s: dict, r: dict, summary: dict) -> None:
    """IPB98 x learned correction on this shot, with its error beside plain IPB98 so it is never trusted blindly.
    Skipped when no trained model is on disk: the replay works without it."""
    if not compute_capabilities().get("surrogate_available"):
        return
    Hc = run_task("run_surrogate_shot", {"shot": s, "P_loss_MW": r["P_loss_MW"]})
    summary["learned_computed_on"] = provenance("run_surrogate_shot").get("host")
    r["H_learned"], r["W_hybrid_MJ"] = Hc, r["W_H_MJ"] * Hc
    m = run_task("model_metrics", {"model": "surrogate"})
    summary["tau_correction_held_out"] = s["meta"]["shot_id"] // m.get("block_size", 100) in m.get("held_out_blocks", [])
    ok = r["steady"] & np.isfinite(r["H98"]) & (r["H98"] > 0)
    if ok.any():
        def rms(a):
            return round(float(np.sqrt(np.mean(np.log(a) ** 2))), 2)
        summary["rmse_ln_tau_ipb98"], summary["rmse_ln_tau_hybrid"] = rms(r["H98"][ok]), rms(r["H98"][ok] / Hc[ok])


def _warm_up():
    compute_warm_up()   # the correction model (and torch) load here rather than on the first replay request
    try:   # builds/loads the Warp kernels and traces the landing shot
        _lines(30166)
    except Exception:
        logger.warning("warm-up field-line trace failed; the first replay request pays for it instead", exc_info=True)
    _eq_psi(30166)


def start_warm_up():
    """Importing torch takes ~5 s; do it while the server starts so the first replay request is not the one that pays.

    Called from the app's lifespan (api.py), never at import: a thread importing torch and Warp while pytest imports
    them on the main thread crashed or hung about one test run in eight.
    """
    threading.Thread(target=_warm_up, daemon=True).start()


@router.get("/eq_surrogate")
def eq_surrogate_metrics():
    """Holdout metrics of the equilibrium surrogate, as written by scripts/train_eq_surrogate.py."""
    try:
        return run_task("model_metrics", {"model": "eq_surrogate"})
    except FileNotFoundError:
        raise HTTPException(404, "no metrics: run scripts/train_eq_surrogate.py") from None


@router.get("/surrogate")
def surrogate_metrics():
    """Holdout errors of the learned correction, as written by `python -m fusionlab.surrogate`."""
    try:
        return run_task("model_metrics", {"model": "surrogate"})
    except FileNotFoundError:
        raise HTTPException(404, "no trained model: run `make train`") from None


@lru_cache(maxsize=1)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _db_view() -> dict:
    """The usable shot table in operating-space coordinates, with IPB98(y,2) evaluated on every row at once."""
    c = mast.clean_db(mast.load_db())
    g = {"R": c["R_m"], "a": c["a_m"], "kappa": c["V_m3"] / volume(c["R_m"], c["a_m"], 1.0)}
    C, alpha = tau_coeff_H(g, c["Ip_MA"], c["B_T"], c["n_e20"], 1.0, M_D)
    beta_N = c["beta_t_pct"] * c["a_m"] * c["B_T"] / c["Ip_MA"]
    return {"shot_id": c["shot_id"], "campaign": c["campaign"], "P_nbi_MW": c["P_nbi_MW"], "Ip_MA": c["Ip_MA"],
            "f_greenwald": c["n_e20"] / (c["Ip_MA"] / (np.pi * c["a_m"] ** 2)), "q95": c["q95"], "beta_N": beta_N,
            "tau_E_s": c["tau_E_s"], "tau_98_s": C * c["P_loss_MW"] ** -alpha,
            "H98": c["tau_E_s"] / (C * c["P_loss_MW"] ** -alpha)}


@router.get("/db")
def db():
    """~6k real MAST shots at peak current, for the operating-space scatter."""
    v = _db_view()
    return {"n": int(v["shot_id"].size), "attribution": ATTRIBUTION, "beta_N_limit": BETA_N_LIMIT, "q95_limit": 2.0,
            "H98_median": round(float(np.nanmedian(v["H98"])), 3), "columns": {k: _j(a) for k, a in v.items()}}


# ---- catalog search over the full shot table (the db view above is the cleaned analysis subset)

_ROW_FIELDS = ("shot_id", "campaign", "Ip_MA", "B_T", "n_e20", "P_nbi_MW", "P_ohm_MW", "P_rad_MW", "W_MJ",
               "Te0_keV", "q95", "kappa", "beta_t_pct", "a_m", "R_m", "V_m3", "tau_E_s", "dWdt_MW",
               "t_ipmax_s", "useful", "abort")


@lru_cache(maxsize=1)
def _catalog() -> dict:
    """The full FAIR-MAST shot table (data/mast_db.npz), read once per process and shared by every search."""
    return mast.load_db()


def _q_mask(c: dict, q: str | None) -> np.ndarray | None:
    """Mask for one shot number or an inclusive range like 30400-30500."""
    if q is None:
        return None
    if not (m := re.fullmatch(r"(\d{1,6})(?:-(\d{1,6}))?", q.strip())):
        raise HTTPException(422, "q must be a shot number or a range like 30400-30500")
    lo, hi = int(m.group(1)), m.group(2)
    if hi is not None:
        if int(hi) < lo:
            raise HTTPException(422, "q range must go from the lower shot number to the higher")
        return (c["shot_id"] >= lo) & (c["shot_id"] <= int(hi))
    return c["shot_id"] == lo


def _campaign_mask(c: dict, campaign: str | None) -> np.ndarray | None:
    """Mask for one campaign, named like the archive names them (M9) or as a bare number (9)."""
    if campaign is None:
        return None
    m = re.fullmatch(r"M?(\d+)", campaign.strip(), re.IGNORECASE)
    if not m:
        raise HTTPException(422, "campaign must look like M9 (bare digits work too)")
    return c["campaign"] == int(m.group(1))


def _range_mask(c: dict, field: str, lo: float | None, hi: float | None) -> np.ndarray | None:
    """Inclusive min/max mask on one numeric field; shots missing that field (NaN) never match."""
    if lo is None and hi is None:
        return None
    a, m = c[field], np.ones(c["shot_id"].size, dtype=bool)
    if lo is not None:
        m &= a >= lo
    if hi is not None:
        m &= a <= hi
    return m


def _flag_mask(c: dict, field: str, flag: bool | None) -> np.ndarray | None:
    """useful/abort toggle: True keeps rows the archive marked with 1, False the rest; None does not filter."""
    if flag is None:
        return None
    marked = c[field] == 1
    return marked if flag else ~marked


@router.get("/db/search")
def db_search(q: str | None = None, campaign: str | None = None,
              ip_min: float | None = None, ip_max: float | None = None,   # MA
              bt_min: float | None = None, bt_max: float | None = None,   # T
              pnbi_min: float | None = None, pnbi_max: float | None = None,   # MW
              ne_min: float | None = None, ne_max: float | None = None,   # 1e20 m^-3
              w_min: float | None = None, w_max: float | None = None,   # MJ
              q95_min: float | None = None, q95_max: float | None = None,
              useful: bool | None = None, abort: bool | None = None,
              limit: int = Query(default=200, ge=1, le=1000), offset: int = Query(default=0, ge=0)) -> dict:
    """Vectorized filter over the full FAIR-MAST catalog (15,969 rows), beside /db's cleaned analysis view.

    Filters combine (AND); the UI mirrors them into the URL so any result set is shareable. total counts
    the whole match set, independent of the limit/offset window. Units are the repo's: MA, T, MW,
    1e20 m^-3, MJ. There is no H-mode filter: the archive's metadata table has no H-mode flag.
    """
    c = _catalog()
    mask = np.ones(c["shot_id"].size, dtype=bool)
    for m in (_q_mask(c, q), _campaign_mask(c, campaign),
              _range_mask(c, "Ip_MA", ip_min, ip_max), _range_mask(c, "B_T", bt_min, bt_max),
              _range_mask(c, "P_nbi_MW", pnbi_min, pnbi_max), _range_mask(c, "n_e20", ne_min, ne_max),
              _range_mask(c, "W_MJ", w_min, w_max), _range_mask(c, "q95", q95_min, q95_max),
              _flag_mask(c, "useful", useful), _flag_mask(c, "abort", abort)):
        if m is not None:
            mask &= m
    total = int(mask.sum())
    if offset and offset >= total:   # offset 0 stays valid on an empty match: that is the "no shots" state
        raise HTTPException(422, f"offset {offset} is beyond the {total} matching shots")
    idx = np.where(mask)[0]
    idx = idx[np.argsort(c["shot_id"][idx], kind="stable")][offset:offset + limit]   # ascending ids: stable pages
    cols = {k: _j(c[k][idx]) for k in _ROW_FIELDS}
    rows = [dict(zip(_ROW_FIELDS, row, strict=True)) for row in zip(*cols.values(), strict=True)]
    return {"total": total, "limit": limit, "offset": offset, "rows": rows, "attribution": ATTRIBUTION}
