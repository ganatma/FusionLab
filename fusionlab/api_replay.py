"""Replay routes: shots from the data seam — the FAIR-MAST archive and imported bundles — served beside the physics model."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import threading
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from zarr.errors import GroupNotFoundError

from fusionlab import mast
from fusionlab.compute import warm_up as compute_warm_up
from fusionlab.compute.status import capabilities as compute_capabilities
from fusionlab.compute.status import (
    provenance,
    run_compute_async,
    run_task,
    run_task_async,
)
from fusionlab.data import sources
from fusionlab.data.bundle import SPEC, BundleValidationError
from fusionlab.physics import (
    BETA_N_LIMIT,
    LIMIT_NAMES,
    replay,
    tau_coeff_H,
    volume,
)
from fusionlab.virtual import Edit, domain_bands, in_distribution

router = APIRouter()

# Machine-identity values (attribution, warm-up shot, isotope) come from the configured source.
ATTRIBUTION = mast.CONFIG.attribution
_TRACES = ("t_s", "Ip_MA", "B_T", "n_e20", "P_ohm_MW", "P_nbi_MW", "P_rad_MW", "W_MJ", "q95", "beta_N", "kappa",
           "a_m", "R_m", "Te0_keV", "R_mag_m", "Z_mag_m")

logger = logging.getLogger(__name__)


def _j(a, nd=4):
    """numpy -> JSON-safe nested lists: rounded, NaN/inf as null."""
    a = np.asarray(a)
    if a.dtype == bool or a.dtype.kind in "iu":
        return a.tolist()
    if a.dtype.kind == "U":   # string ids / campaigns (imported catalogs)
        return a.tolist()
    a = np.round(a.astype(float), nd)
    return np.where(np.isfinite(a), a, None).tolist()


@lru_cache(maxsize=16)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _shot_cached(shot_id: int) -> dict:
    return mast.load_shot(shot_id)


# One archive fetch per shot id, however many requests want it: compare fans out over ids in parallel.
_inflight: dict[str, dict] = {}
_inflight_lock = threading.Lock()
_FETCH_WAIT_S = 600   # a fetch takes seconds to minutes; a waiter gives up and asks the client to retry


def _shot(shot_id: str) -> tuple[dict, sources.DataSource]:
    """A shot bundle and the source that served it. An imported-bundle id (anything that is not a
    plain shot number) resolves through the user-file source; shot numbers go to the archive —
    cached shots from the process cache, uncached catalog shots through load_shot()'s
    fetch-and-cache path. Returns the source too, so routes follow its machine config (attribution,
    USD naming, isotope) instead of assuming MAST."""
    shot_id = str(shot_id)
    if not shot_id.isdigit():
        if not sources.valid_bundle_id(shot_id):
            raise HTTPException(404, f"shot id {shot_id!r} is neither a shot number nor a well-formed bundle id")
        src = sources.get_source("user-file")
        try:
            return src.load_shot(shot_id), src
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except BundleValidationError as e:
            raise HTTPException(400, f"the bundle does not satisfy the ShotBundle v1 contract: {e}") from e
    if shot_id in {str(i) for i in mast.cached_shots()}:
        return _shot_cached(int(shot_id)), sources.get_source("fair-mast")
    cat_src = sources.get_source()   # membership is asked of the configured source's own catalog
    if not bool(np.any(_catalog(cat_src.name)["shot_id"] == int(shot_id))):
        tail = " (data/mast_db.npz)" if cat_src.name == "fair-mast" else ""
        raise HTTPException(404, f"shot {shot_id} is not in the {cat_src.name} catalog{tail}")
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
            mast.load_shot(int(shot_id))   # fetches from S3 + REST, writes the local cache atomically (fusionlab.mast)
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
    if shot_id in {str(i) for i in mast.cached_shots()}:
        return _shot_cached(int(shot_id)), sources.get_source("fair-mast")
    raise HTTPException(502, f"shot {shot_id} could not be fetched from FAIR-MAST")


def _shot_row(s: dict, source: str) -> dict:
    """One /shots entry: the bundle's logbook meta plus the picker's scalars, tagged with its source."""
    return {**s["meta"], "source": source, "n_slices": int(s["t_s"].size), "t_start_s": float(s["t_s"][0]),
            "t_end_s": float(s["t_s"][-1]), "Ip_max_MA": round(float(np.nanmax(s["Ip_MA"])), 3),
            "P_nbi_max_MW": round(float(np.nanmax(s["P_nbi_MW"])), 2)}


@router.get("/shots")
def shots():
    """Cached shots with their logbook text: the archive's cache plus any imported bundles.
    Uncached catalog shots are fetched on demand by /replay/{id}."""
    out = []
    for sid in mast.cached_shots():
        s, _ = _shot(str(sid))
        out.append(_shot_row(s, "fair-mast"))
    src = sources.get_source("user-file")
    for sid in src.cached_shots():
        out.append(_shot_row(src.load_shot(sid), "user-file"))
    return {"shots": out, "attribution": ATTRIBUTION}


def _attribution(s: dict, src: sources.DataSource) -> str:
    """The bundle's own attribution when it carries one; the source machine's otherwise."""
    return str(s["meta"].get("attribution") or src.config.attribution)


def _replay_bundle(shot_id: str, H: float = 1.0) -> dict:
    """The full replay response: measured traces, model traces on the same time base, limits, geometry.
    /export serves the very same dict, so exported values equal replayed values by construction.

    Equilibrium surfaces (LCFS, psi grid) are served only when the bundle carries them — a machine
    without equilibrium reconstructions replays its traces with "equilibrium_absent" set instead of
    failing the route.
    """
    shot_id = str(shot_id)   # archive callers still pass the shot number as an int
    s, src = _shot(shot_id)
    r = replay(s, H=H, config=src.config)
    r["in_distribution"] = in_distribution(s, [Edit()])[0]   # the virtual path's own guard, on the measured inputs
    summary = _summary(s, r)
    _add_hybrid(s, r, summary)
    eq = _eq_psi(shot_id)
    if eq is not None:
        summary["eq_surrogate"] = {"held_out": eq["held_out"], "median_rel_l2": round(float(np.median(eq["rel_l2"])), 4),
                                   "p95_rel_l2": round(float(np.percentile(eq["rel_l2"], 95)), 4),
                                   "computed_on": eq.get("computed_on")}
    payload = {
        "meta": s["meta"], "attribution": _attribution(s, src), "limit_names": LIMIT_NAMES, "summary": summary,
        # The archive's own bookkeeping (FEAT-3): the abort flag the limit panel overlays as context.
        "archive": _archive_flags(shot_id, src.name),
        "measured": {k: _j(s[k]) for k in _TRACES if k in s},
        "model": {k: _j(v) for k, v in r.items()},
        "lcfs": {"R": _j(s["lcfs_R"], 3), "Z": _j(s["lcfs_Z"], 3)} if "lcfs_R" in s else None,
        "wall": {"R": _j(s["wall_R"], 3), "Z": _j(s["wall_Z"], 3)} if "wall_R" in s else None,
        "coils": {k[5:]: _j(s[k], 3) for k in ("coil_R", "coil_Z", "coil_W", "coil_H")} if "coil_R" in s else None,
        "psi_grid": {"R": _j(s["psi_R"], 3), "Z": _j(s["psi_Z"], 3)} if "psi_R" in s else None,
        "domain_bands": _j(domain_bands(s["t_s"], r["in_distribution"])),   # the shading coordinates, from the flags
    }
    if "lcfs_R" not in s or "psi_R" not in s:
        payload["equilibrium_absent"] = (
            "no equilibrium reconstruction in this bundle: the 3D plasma surface, psi view and "
            "field lines are unavailable (trace-only replay)")
    return payload


@router.get("/replay/{shot_id}")
async def replay_shot(shot_id: str, H: float = 1.0):
    """Measured traces, model traces on the same time base, limits, and the geometry to draw the shot."""
    return await run_compute_async(partial(_replay_bundle, shot_id, H))


def _replay_psi_slice(shot_id: str, i: int) -> dict:
    """One psi slice's response body — the blocking part the async route below runs on the compute pool."""
    s, _ = _shot(shot_id)
    if "psi_n" not in s:
        raise HTTPException(404, f"shot {shot_id}'s bundle has no equilibrium reconstruction (psi)")
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


@router.get("/replay/{shot_id}/psi/{i}")
async def replay_psi(shot_id: str, i: int):
    """Normalised poloidal flux (0 axis, 1 separatrix) on the EFIT grid for one time slice, rows = Z."""
    return await run_compute_async(partial(_replay_psi_slice, shot_id, i))


@router.post("/replay/{shot_id}/usd")
def replay_usd(shot_id: str, fmt: str = "usdc"):
    """OpenUSD export (Omniverse-compatible): vessel, PF coils and the plasma boundary time-sampled over the shot.

    POST, not GET: the export recomputes the stage and writes out/ on every hit — a state-changing
    operation must not be triggerable by navigation, prefetch or crawlers (audit F1)."""
    from fusionlab.usd_export import FORMATS, export_shot
    if f".{fmt}" not in FORMATS:
        raise HTTPException(422, f"fmt must be one of {sorted(x.lstrip('.') for x in FORMATS)}")
    s, src = _shot(shot_id)
    if "lcfs_R" not in s:   # the plasma boundary is the one geometry export_shot requires
        raise HTTPException(422, f"shot {shot_id}'s bundle has no equilibrium boundary (lcfs); USD export needs it")
    stem = src.config.usd_prim_root.lstrip("/").lower() or "bundle"   # mast_30420.usdc, user_1.usdc, ...
    out = Path(__file__).resolve().parent.parent / "out" / f"{stem}_{shot_id}.{fmt}"
    tmp = out.with_name(f".{out.stem}.tmp{os.getpid()}{out.suffix}")   # dotfile, pid-unique, extension kept (export_shot validates the suffix)
    try:
        try:   # with Warp-traced field lines when Warp can run here; the plain stage otherwise
            from fusionlab.fieldlines import export_shot_with_field_lines
            export_shot_with_field_lines(s, tmp, machine=src.config)
        except Exception:
            logger.warning("USD field-line export failed; falling back to plain stage", exc_info=True)
            export_shot(s, tmp, machine=src.config)
        os.replace(tmp, out)   # atomic same-directory publish: readers only ever see a complete stage
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return FileResponse(out, filename=out.name, media_type="application/octet-stream")


@lru_cache(maxsize=8)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _lines(shot_id: str):
    """Field lines for every slice of a shot in one Warp launch, plus traced q at psi_N = 0.95 beside EFIT's q95."""
    s, _ = _shot(shot_id)
    out = run_task("trace_fieldlines", {"shot": s})
    where = provenance("trace_fieldlines").get("host")   # captured now: the cache keeps it honest
    pts, qc = out["pts"], out["q_check"]                    # (time, line, point, xyz)
    q = np.full(s["t_s"].size, np.nan)
    q[np.asarray(qc["slice"], dtype=int)] = qc["q_traced"]
    return pts, q, qc["summary"], out["device"], out["psi_n_start"], where


def _replay_fieldlines_slice(shot_id: str, i: int) -> dict:
    """One field-lines response body — the blocking part (a Warp trace through run_task) the async
    route below runs on the compute pool."""
    s, _ = _shot(shot_id)
    if not 0 <= i < s["t_s"].size:
        raise HTTPException(404, "time index out of range")
    if "psi_n" not in s:
        raise HTTPException(503, f"field-line tracing unavailable: shot {shot_id}'s bundle has no equilibrium (psi)")
    try:
        pts, q, summary, device, psi_n_start, computed_on = _lines(shot_id)
    except Exception as e:   # no Warp / no usable device: the rest of the replay still works
        raise HTTPException(503, f"field-line tracing unavailable: {type(e).__name__}") from e
    return {"i": i, "t_s": float(s["t_s"][i]), "psi_n_start": list(psi_n_start), "device": device,
            "computed_on": computed_on,
            "lines": [{"x": _j(line[:, 0], 3), "y": _j(line[:, 1], 3), "z": _j(line[:, 2], 3)} for line in pts[i]],
            "q95_traced": _j(q[i], 3), "q95_efit": _j(s["q95"][i], 3),
            "shot_check": {k: summary[k] for k in ("n_compared", "median_rel_err", "p95_rel_err", "psi_n_drift_max")}}


@router.get("/replay/{shot_id}/fieldlines/{i}")
async def replay_fieldlines(shot_id: str, i: int):
    """Field lines of the EFIT reconstruction at one time slice (axisymmetric; no islands or 3D fields)."""
    return await run_compute_async(partial(_replay_fieldlines_slice, shot_id, i))


@lru_cache(maxsize=1)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _eq_model():
    """The equilibrium surrogate's holdout shots. None when no trained model is on disk."""
    if not compute_capabilities().get("eq_surrogate_available"):
        return None
    metrics = run_task("model_metrics", {"model": "eq_surrogate"})
    return set(metrics["test_shot_ids"])


@lru_cache(maxsize=8)   # process-lifetime cache: a refetched or retrained artifact on disk is not seen until restart
def _eq_psi(shot_id: str):
    """PhysicsNeMo reconstruction of psi from the magnetic sensors, every slice of the shot in one batch.
    Normalised with the equilibrium reconstruction's axis and boundary flux so the two maps share contour
    levels; error is on psi itself."""
    s = _shot(shot_id)[0]
    try:
        held_out = _eq_model()
        if held_out is None or "eq_inputs" not in s or "psi_n" not in s:
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
    return {"psi_n": (psi - ax) / (bd - ax), "rel_l2": rel,
            "held_out": (int(shot_id) in held_out) if shot_id.isdigit() else False,
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
    sid = s["meta"]["shot_id"]
    # Session-block held-out split is a MAST training artifact; imported ids are not block numbers.
    summary["tau_correction_held_out"] = (
        sid // m.get("block_size", 100) in m.get("held_out_blocks", []) if isinstance(sid, int) else False)
    ok = r["steady"] & np.isfinite(r["H98"]) & (r["H98"] > 0)
    if ok.any():
        def rms(a):
            return round(float(np.sqrt(np.mean(np.log(a) ** 2))), 2)
        summary["rmse_ln_tau_ipb98"], summary["rmse_ln_tau_hybrid"] = rms(r["H98"][ok]), rms(r["H98"][ok] / Hc[ok])


def _warm_up():
    compute_warm_up()   # the correction model (and torch) load here rather than on the first replay request
    if (warm := mast.CONFIG.warmup_shot) is not None:
        try:   # builds/loads the Warp kernels and traces the landing shot
            _lines(warm)
        except Exception:
            logger.warning("warm-up field-line trace failed; the first replay request pays for it instead", exc_info=True)
        _eq_psi(warm)


def start_warm_up():
    """Importing torch takes ~5 s; do it while the server starts so the first replay request is not the one that pays.

    Called from the app's lifespan (api.py), never at import: a thread importing torch and Warp while pytest imports
    them on the main thread crashed or hung about one test run in eight.
    """
    threading.Thread(target=_warm_up, daemon=True).start()


@router.get("/eq_surrogate")
async def eq_surrogate_metrics():
    """Holdout metrics of the equilibrium surrogate, as written by scripts/train_eq_surrogate.py."""
    try:
        return await run_task_async("model_metrics", {"model": "eq_surrogate"})
    except FileNotFoundError:
        raise HTTPException(404, "no metrics: run scripts/train_eq_surrogate.py") from None


@router.get("/surrogate")
async def surrogate_metrics():
    """Holdout errors of the learned correction, as written by `python -m fusionlab.surrogate`."""
    try:
        return await run_task_async("model_metrics", {"model": "surrogate"})
    except FileNotFoundError:
        raise HTTPException(404, "no trained model: run `make train`") from None


@lru_cache(maxsize=2)   # per source: a different FUSIONLAB_DATA_SOURCE takes a fresh entry (or a restart)
def _db_view(source: str) -> dict:
    """The usable shot table in operating-space coordinates, with IPB98(y,2) evaluated on every row at once."""
    src = sources.get_source(source)
    c = src.clean_db(src.load_db())
    g = {"R": c["R_m"], "a": c["a_m"], "kappa": c["V_m3"] / volume(c["R_m"], c["a_m"], 1.0)}
    C, alpha = tau_coeff_H(g, c["Ip_MA"], c["B_T"], c["n_e20"], 1.0, src.config.isotope_md)
    beta_N = c["beta_t_pct"] * c["a_m"] * c["B_T"] / c["Ip_MA"]
    return {"shot_id": c["shot_id"], "campaign": c["campaign"], "P_nbi_MW": c["P_nbi_MW"], "Ip_MA": c["Ip_MA"],
            "f_greenwald": c["n_e20"] / (c["Ip_MA"] / (np.pi * c["a_m"] ** 2)), "q95": c["q95"], "beta_N": beta_N,
            "tau_E_s": c["tau_E_s"], "tau_98_s": C * c["P_loss_MW"] ** -alpha,
            "H98": c["tau_E_s"] / (C * c["P_loss_MW"] ** -alpha)}


@router.get("/db")
def db():
    """The configured source's usable shots at peak current, for the operating-space scatter."""
    src = sources.get_source()
    v = _db_view(src.name)
    n = int(v["shot_id"].size)
    return {"n": n, "attribution": src.config.attribution, "beta_N_limit": BETA_N_LIMIT, "q95_limit": 2.0,
            "H98_median": round(float(np.nanmedian(v["H98"])), 3) if n else None,
            "columns": {k: _j(a) for k, a in v.items()}}


# ---- catalog search over the full shot table (the db view above is the cleaned analysis subset)

_ROW_FIELDS = ("shot_id", "campaign", "Ip_MA", "B_T", "n_e20", "P_nbi_MW", "P_ohm_MW", "P_rad_MW", "W_MJ",
               "Te0_keV", "q95", "kappa", "beta_t_pct", "a_m", "R_m", "V_m3", "tau_E_s", "dWdt_MW",
               "t_ipmax_s", "useful", "abort", "mode_derived")


@lru_cache(maxsize=2)
def _catalog(source: str) -> dict:
    """The configured source's full shot table (data/mast_db.npz for the archive), read once per
    process per source and shared by every search."""
    return sources.get_source(source).load_db()


def _archive_flags(shot_id: str, source: str) -> dict[str, bool | None]:
    """The archive's own bookkeeping for one shot — the catalog's cpf_useful/cpf_abort columns.
    None where the row is missing or a flag is NaN (the archive never graded it), and for
    imported bundles (they carry no archive bookkeeping): the UI degrades silently, never
    invents a flag. These are archive bookkeeping, not physics."""
    if not shot_id.isdigit():
        return {"useful": None, "abort": None}
    c = _catalog(source)
    hits = np.nonzero(c["shot_id"] == int(shot_id))[0]
    if not hits.size:
        return {"useful": None, "abort": None}
    def flag(col: str) -> bool | None:
        v = float(c[col][int(hits[0])])
        return None if not np.isfinite(v) else bool(v)
    return {"useful": flag("useful"), "abort": flag("abort")}


def _q_mask(c: dict, q: str | None) -> np.ndarray | None:
    """Mask for one shot number or an inclusive range like 30400-30500."""
    if q is None:
        return None
    if not (m := re.fullmatch(r"(\d{1,6})(?:-(\d{1,6}))?", q.strip())):
        raise HTTPException(422, "q must be a shot number or a range like 30400-30500")
    lo, hi = int(m.group(1)), m.group(2)
    if c["shot_id"].dtype.kind not in "iu":   # imported catalogs carry free-form string ids
        raise HTTPException(422, "q filters shot numbers; the imported catalog's ids are not numbers — filter by campaign or ranges instead")
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


def _mode_mask(c: dict, mode: str | None) -> np.ndarray | None:
    """Filter on `mode_derived` — FusionLab's derived confinement-regime flag (fusionlab.data.lh_mode,
    docs/PHYSICS.md). 'H' keeps H-regime rows, 'L' the rest; rows the catalog scalars cannot flag
    (NaN, 'unknown') match neither — they are not evidence for either regime."""
    if mode is None:
        return None
    m = mode.strip().upper()
    if m not in ("H", "L"):
        raise HTTPException(422, "mode must be 'H' or 'L' — the derived confinement-regime flag (docs/PHYSICS.md)")
    flag = c["mode_derived"]
    return flag == 1 if m == "H" else flag == 0


@router.get("/db/search")
def db_search(q: str | None = None, campaign: str | None = None,
              ip_min: float | None = None, ip_max: float | None = None,   # MA
              bt_min: float | None = None, bt_max: float | None = None,   # T
              pnbi_min: float | None = None, pnbi_max: float | None = None,   # MW
              ne_min: float | None = None, ne_max: float | None = None,   # 1e20 m^-3
              w_min: float | None = None, w_max: float | None = None,   # MJ
              q95_min: float | None = None, q95_max: float | None = None,
              useful: bool | None = None, abort: bool | None = None,
              mode: str | None = None,   # derived regime flag ('H'/'L'), never an archive field
              limit: int = Query(default=200, ge=1, le=1000), offset: int = Query(default=0, ge=0)) -> dict:
    """Vectorized filter over the configured source's full shot table, beside /db's cleaned analysis view.

    Filters combine (AND); the UI mirrors them into the URL so any result set is shareable. total counts
    the whole match set, independent of the limit/offset window. Units are the repo's: MA, T, MW,
    1e20 m^-3, MJ. mode filters on `mode_derived` — FusionLab's derived confinement-regime flag
    (docs/PHYSICS.md): 'H' or 'L'; rows the scalars cannot flag stay unknown and match neither.
    Derived, never measured.
    """
    src = sources.get_source()
    c = _catalog(src.name)
    mask = np.ones(c["shot_id"].size, dtype=bool)
    for m in (_q_mask(c, q), _campaign_mask(c, campaign),
              _range_mask(c, "Ip_MA", ip_min, ip_max), _range_mask(c, "B_T", bt_min, bt_max),
              _range_mask(c, "P_nbi_MW", pnbi_min, pnbi_max), _range_mask(c, "n_e20", ne_min, ne_max),
              _range_mask(c, "W_MJ", w_min, w_max), _range_mask(c, "q95", q95_min, q95_max),
              _flag_mask(c, "useful", useful), _flag_mask(c, "abort", abort),
              _mode_mask(c, mode)):
        if m is not None:
            mask &= m
    total = int(mask.sum())
    if offset and offset >= total:   # offset 0 stays valid on an empty match: that is the "no shots" state
        raise HTTPException(422, f"offset {offset} is beyond the {total} matching shots")
    idx = np.where(mask)[0]
    idx = idx[np.argsort(c["shot_id"][idx], kind="stable")][offset:offset + limit]   # ascending ids: stable pages
    cols = {k: _j(c[k][idx]) for k in _ROW_FIELDS}
    rows = [dict(zip(_ROW_FIELDS, row, strict=True)) for row in zip(*cols.values(), strict=True)]
    return {"total": total, "limit": limit, "offset": offset, "rows": rows,
            "attribution": src.config.attribution}


# ---- take-away export (FEAT-1): the replay bundle's numbers, downloadable with their provenance.
# The UI tags every panel measured / model / learned / computed; the export carries the same labels so
# derived quantities stay honest outside the app (in a notebook or a group meeting, not just on screen).

_LEARNED = frozenset({"H_learned", "W_hybrid_MJ"})   # the learned correction's outputs (_add_hybrid)
_MODEL = frozenset({                                 # replay()'s published-scaling outputs (IPB98/ITER89/Martin)
    "P_LH_MW", "W_model_MJ", "W_L_MJ", "W_H_MJ", "tau_98_s", "tau_89_s",
    "T_avg_model_keV", "Te0_model_keV", "P_ohm_model_MW", "hmode_model",
})
_MEASURED_PASSTHROUGH = frozenset({"t_s", "W_meas_MJ"})   # archive diagnostics inside replay()'s dict

_UNIT = {   # the repo's fixed units (CLAUDE.md); "" is dimensionless
    "t_s": "s", "Ip_MA": "MA", "B_T": "T", "n_e20": "1e20 m^-3", "P_ohm_MW": "MW", "P_nbi_MW": "MW",
    "P_rad_MW": "MW", "W_MJ": "MJ", "q95": "", "beta_N": "", "kappa": "", "a_m": "m", "R_m": "m",
    "Te0_keV": "keV", "R_mag_m": "m", "Z_mag_m": "m",
    "P_in_MW": "MW", "P_loss_MW": "MW", "P_LH_MW": "MW", "W_meas_MJ": "MJ", "W_model_MJ": "MJ",
    "W_L_MJ": "MJ", "W_H_MJ": "MJ", "W_hybrid_MJ": "MJ", "tau_meas_s": "s", "tau_98_s": "s",
    "tau_89_s": "s", "T_avg_model_keV": "keV", "Te0_model_keV": "keV", "P_ohm_model_MW": "MW",
    "f_greenwald": "", "troyon": "", "kink": "", "worst_limit": "", "steady": "", "hmode_model": "",
    "binding": "", "H98": "", "H89": "", "H_learned": "", "in_distribution": "",
    "n_steady": "", "H98_median": "", "H89_median": "", "peak_limit_fraction": "", "peak_limit_t_s": "s",
    "P_LH_median_MW": "MW", "P_loss_median_MW": "MW", "rmse_ln_tau_ipb98": "", "rmse_ln_tau_hybrid": "",
    "tau_correction_held_out": "", "eq_surrogate_held_out": "", "eq_surrogate_median_rel_l2": "",
    "eq_surrogate_p95_rel_l2": "",
}

_EXPORT_COLUMNS = ("shot_id", "t_s", "quantity", "value", "unit", "provenance")
_EXPORT_MAX_SHOTS = 6   # the compare view's limit (SRCH_MAX in web/replay.js); each id is a full replay


def _provenance(name: str, group: str) -> str:
    """One series' label in the measured/model/learned/computed vocabulary. `group` is the bundle's
    series group. Unknown model keys answer 'computed' — the conservative label — and the coverage
    tripwire in test_api.py pins every real series to its honest label."""
    if group == "measured" or name in _MEASURED_PASSTHROUGH:
        return "measured"
    if name in _LEARNED:
        return "learned"
    if name in _MODEL:
        return "model"
    return "computed"


def _summary_provenance(name: str) -> str:
    """Summary numbers are computed here, except the learned models' own metadata (the correction's
    holdout flag, the equilibrium surrogate's errors), which carry the learned label."""
    return "learned" if name.startswith("eq_surrogate") or name == "tau_correction_held_out" else "computed"


def _flat_summary(summary: dict) -> list[tuple[str, object]]:
    """(name, value) pairs of the summary panel: nested dicts (the eq-surrogate block) get a name prefix."""
    out: list[tuple[str, object]] = []
    for k, v in summary.items():
        if isinstance(v, dict):
            out += [(f"{k}_{sk}", sv) for sk, sv in v.items()]
        else:
            out.append((k, v))
    return out


def _export_provenance_map(bundle: dict) -> dict[str, str]:
    """quantity -> provenance for every series and summary number a bundle exports. The CSV's
    provenance column and the JSON export block share this one source of truth."""
    prov = {k: _provenance(k, "measured") for k in bundle["measured"]}
    prov.update({k: _provenance(k, "model") for k in bundle["model"] if k != "t_s"})
    prov.update({name: _summary_provenance(name) for name, _ in _flat_summary(bundle["summary"])})
    return prov


def _export_rows(bundles: list[dict]) -> list[dict]:
    """The tidy take-away table: one row per (shot, time slice, quantity), every row carrying its
    provenance and the repo's fixed unit. Values are the replay bundle's own JSON-safe numbers
    (rounded, NaN empty), so the CSV equals the API response by construction. Summary scalars ride
    once per shot with an empty t_s; string-valued summary entries stay in the JSON export only."""
    rows = []
    for b in bundles:
        sid = b["meta"]["shot_id"]
        series = [(k, v, "measured") for k, v in b["measured"].items()] \
            + [(k, v, "model") for k, v in b["model"].items() if k != "t_s"]   # model shares the measured time base
        for i, t in enumerate(b["measured"]["t_s"]):
            for name, vals, group in series:
                rows.append({"shot_id": sid, "t_s": t, "quantity": name, "value": vals[i],
                             "unit": _UNIT.get(name, ""), "provenance": _provenance(name, group)})
        for name, v in _flat_summary(b["summary"]):
            if isinstance(v, (bool, int, float)):
                rows.append({"shot_id": sid, "t_s": "", "quantity": name, "value": v,
                             "unit": _UNIT.get(name, ""), "provenance": _summary_provenance(name)})
    return rows


def _export_ids(shots: str) -> list[str]:
    """Comma-separated shot ids for the compare export — archive shot numbers or imported-bundle
    ids: deduped, request order kept, capped (each id is a full replay, the cap keeps an accidental
    URL cheap)."""
    ids: list[str] = []
    for part in shots.split(","):
        part = part.strip()
        if not sources.valid_bundle_id(part):   # covers plain shot numbers and bundle ids
            raise HTTPException(422, "shots must be comma-separated shot numbers or imported-bundle ids, "
                                     "e.g. shots=30166,30420 or shots=user_bundle_1")
        if part not in ids:
            ids.append(part)
    if len(ids) > _EXPORT_MAX_SHOTS:
        raise HTTPException(422, f"export covers at most {_EXPORT_MAX_SHOTS} shots — the compare view's limit")
    return ids


def _export(bundles: list[dict], fmt: str, H: float):
    """CSV as an attachment download, or JSON: the full replay bundles, each plus its provenance map."""
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=_EXPORT_COLUMNS)
        w.writeheader()
        w.writerows(_export_rows(bundles))
        ids = [b["meta"]["shot_id"] for b in bundles]
        name = f"fusionlab_shot_{ids[0]}.csv" if len(ids) == 1 else f"fusionlab_shots_{'_'.join(map(str, ids))}.csv"
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})
    if fmt == "json":
        return {"shots": [{**b, "export": {"provenance": _export_provenance_map(b), "H": H}} for b in bundles]}
    raise HTTPException(422, 'fmt must be "csv" or "json"')


def _replay_bundles(shot_ids: list[str], H: float) -> list[dict]:
    """Full replay bundles in request order — one blocking call for the compute pool."""
    return [_replay_bundle(i, H) for i in shot_ids]


@router.get("/export/{shot_id}")
async def export_shot(shot_id: str, H: float = 1.0, fmt: str = "csv"):
    """Take-away export of one shot's derived quantities: every measured trace, model trace and
    summary number the replay serves, as a tidy CSV (one row per shot/slice/quantity, each row
    labelled with its provenance and unit) or as JSON (the full replay bundle + the provenance map).
    The numbers are the replay bundle's own, never a recomputation."""
    bundle = await run_compute_async(partial(_replay_bundle, shot_id, H))
    return _export([bundle], fmt, H)


@router.get("/export")
async def export_compare(shots: str, H: float = 1.0, fmt: str = "csv"):
    """Compare-variant export: the same tidy table over several shots (the compare view's cap),
    a `shot_id` column so any spreadsheet or notebook can group them. Order follows the request."""
    bundles = await run_compute_async(partial(_replay_bundles, _export_ids(shots), H))
    return _export(bundles, fmt, H)


# ---- bring-your-own-data (DATA-2): imported shot bundles replay, compare and export beside the archive.

@router.post("/data/bundles/{shot_id}")
def upload_bundle(shot_id: str, file: UploadFile | None = File(None),
                  meta_json: str | None = Form(None)) -> dict:
    """Import a shot so it replays, compares and exports beside the archive's cache.

    Two shapes on one route: a multipart POST with a `file` field imports an npz bundle (the
    ShotBundle v1 contract, docs/DATA_CONTRACT.md) — validated in full before anything is written,
    so a malformed upload fails with the first bad field named and leaves no partial file. The id
    must be letters/digits/dots/dashes/underscores, not a bare shot number (those are the
    archive's). A bare POST (no file) on a shot number imports that archive shot through the
    loader's cache-then-S3 path — the UI's self-import when a catalog shot is not cached yet.
    The optional meta_json form field overrides/extends the bundle's own meta."""
    if file is None or not file.filename:
        if file is None and shot_id.isdigit():
            s, src = _shot(shot_id)   # the same fetch-and-cache path a replay takes
            return {**_shot_row(s, src.name), "equilibrium": "psi_R" in s and "lcfs_R" in s,
                    "fields": sorted(k for k in SPEC if k in s), "attribution": _attribution(s, src)}
        raise HTTPException(422, "send the bundle as multipart form data (field name: file); a bare "
                                 "POST imports an archive shot number")
    src = sources.get_source("user-file")
    raw = file.file.read()
    meta = None
    if meta_json is not None:
        try:
            meta = json.loads(meta_json)
        except json.JSONDecodeError as e:
            raise HTTPException(422, f"meta_json is not valid JSON: {e}") from e
        if not isinstance(meta, dict):
            raise HTTPException(422, "meta_json must be a JSON object")
    try:
        s = src.import_bundle(shot_id, raw, meta)
    except BundleValidationError as e:   # contract violations: the first bad field, named
        raise HTTPException(400, f"the bundle does not satisfy the ShotBundle v1 contract: {e}") from e
    except ValueError as e:             # bad id shape or over the size cap
        raise HTTPException(422, str(e)) from e
    return {**_shot_row(s, src.name), "equilibrium": "psi_R" in s and "lcfs_R" in s,
            "fields": sorted(k for k in SPEC if k in s), "attribution": _attribution(s, src)}


@router.delete("/data/bundles/{shot_id}")
def delete_bundle(shot_id: str) -> dict:
    """Remove an imported bundle. The archive's cache is managed by the loader, not this route."""
    src = sources.get_source("user-file")
    if shot_id.isdigit() or not sources.valid_bundle_id(shot_id):
        raise HTTPException(404, f"no imported bundle named {shot_id!r}")
    f = src.bundles_dir / f"{shot_id}.npz"
    if not f.is_file():
        raise HTTPException(404, f"no imported bundle named {shot_id!r}")
    f.unlink()
    return {"deleted": shot_id}
