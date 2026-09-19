"""Replay routes: real MAST shots (FAIR-MAST cache) served beside the physics model."""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from fusionlab import mast
from fusionlab.physics import BETA_N_LIMIT, LIMIT_NAMES, M_D, replay, tau_coeff_H, volume

router = APIRouter()

ATTRIBUTION = "MAST data: UKAEA FAIR-MAST (mastapp.site), CC BY-SA 4.0"
_TRACES = ("t_s", "Ip_MA", "B_T", "n_e20", "P_ohm_MW", "P_nbi_MW", "P_rad_MW", "W_MJ", "q95", "beta_N", "kappa",
           "a_m", "R_m", "Te0_keV", "R_mag_m", "Z_mag_m")


def _j(a, nd=4):
    """numpy -> JSON-safe nested lists: rounded, NaN/inf as null."""
    a = np.asarray(a)
    if a.dtype == bool or a.dtype.kind in "iu":
        return a.tolist()
    a = np.round(a.astype(float), nd)
    return np.where(np.isfinite(a), a, None).tolist()


@lru_cache(maxsize=16)
def _shot(shot_id: int) -> dict:
    if shot_id not in mast.cached_shots():
        raise HTTPException(404, f"shot {shot_id} is not in the local cache (scripts/fetch_mast.py {shot_id})")
    return mast.load_shot(shot_id)


@router.get("/shots")
def shots():
    """Cached shots with their logbook text. The app never fetches from the network."""
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
                                   "p95_rel_l2": round(float(np.percentile(eq["rel_l2"], 95)), 4)}
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
        out["surrogate"] = {"psi_n": _j(eq["psi_n"][i], 3), "rel_l2": _j(eq["rel_l2"][i], 4)}
    if "ts_R" in s:
        out["thomson"] = {"R": _j(s["ts_R"], 3), "Te_keV": _j(s["ts_Te_keV"][i], 3), "ne_e20": _j(s["ts_ne_e20"][i], 3)}
    return out


@router.get("/replay/{shot_id}/usd")
def replay_usd(shot_id: int, fmt: str = "usdc"):
    """OpenUSD export (Omniverse-compatible): vessel, PF coils and the plasma boundary time-sampled over the shot."""
    from fusionlab.usd_export import FORMATS, export_shot
    if f".{fmt}" not in FORMATS:
        raise HTTPException(422, f"fmt must be one of {sorted(x.lstrip('.') for x in FORMATS)}")
    out = Path(__file__).resolve().parent.parent / "out" / f"mast_{shot_id}.{fmt}"
    try:   # with Warp-traced field lines when Warp can run here; the plain stage otherwise
        from fusionlab.fieldlines import export_shot_with_field_lines
        export_shot_with_field_lines(_shot(shot_id), out)
    except Exception:
        export_shot(_shot(shot_id), out)
    return FileResponse(out, filename=out.name, media_type="application/octet-stream")


@lru_cache(maxsize=8)
def _lines(shot_id: int):
    """Field lines for every slice of a shot in one Warp launch, plus traced q at psi_N = 0.95 beside EFIT's q95."""
    from fusionlab import fieldlines
    s = _shot(shot_id)
    pts, tr = fieldlines.usd_lines(s)                       # (time, line, point, xyz)
    qc = fieldlines.q_check(s)
    q = np.full(s["t_s"].size, np.nan)
    q[np.asarray(qc["slice"], dtype=int)] = qc["q_traced"]
    return pts, q, qc["summary"], str(tr.get("device", fieldlines.default_device()))


@router.get("/replay/{shot_id}/fieldlines/{i}")
def replay_fieldlines(shot_id: int, i: int):
    """Field lines of the EFIT reconstruction at one time slice (axisymmetric; no islands or 3D fields)."""
    s = _shot(shot_id)
    if not 0 <= i < s["t_s"].size:
        raise HTTPException(404, "time index out of range")
    try:
        from fusionlab.fieldlines import USD_PSI_N
        pts, q, summary, device = _lines(shot_id)
    except Exception as e:   # no Warp / no usable device: the rest of the replay still works
        raise HTTPException(503, f"field-line tracing unavailable: {type(e).__name__}") from e
    return {"i": i, "t_s": float(s["t_s"][i]), "psi_n_start": list(USD_PSI_N), "device": device,
            "lines": [{"x": _j(l[:, 0], 3), "y": _j(l[:, 1], 3), "z": _j(l[:, 2], 3)} for l in pts[i]],
            "q95_traced": _j(q[i], 3), "q95_efit": _j(s["q95"][i], 3),
            "shot_check": {k: summary[k] for k in ("n_compared", "median_rel_err", "p95_rel_err", "psi_n_drift_max")}}


@lru_cache(maxsize=1)
def _eq_model():
    """Equilibrium surrogate + the shots it never saw. None when no trained model is on disk."""
    from fusionlab import eq_surrogate
    if not eq_surrogate.MODEL_FILE.exists():
        return None
    held_out = set(json.loads(eq_surrogate.METRICS_FILE.read_text())["test_shot_ids"])
    return eq_surrogate.load(), held_out


@lru_cache(maxsize=8)
def _eq_psi(shot_id: int):
    """PhysicsNeMo reconstruction of psi from the magnetic sensors, every slice of the shot in one batch.
    Normalised with EFIT's axis and boundary flux so the two maps share contour levels; error is on psi itself."""
    s = _shot(shot_id)
    try:
        loaded = _eq_model()
        if loaded is None or "eq_inputs" not in s:
            return None
        import torch
        model, held_out = loaded
        with torch.no_grad():
            x = torch.as_tensor(s["eq_inputs"], dtype=torch.float32, device=model.x_mean.device)
            psi = model(x).cpu().numpy()
    except Exception:   # the replay works without the surrogate
        return None
    ax, bd = s["psi_axis_Wb"][:, None, None], s["psi_bnd_Wb"][:, None, None]
    psi_efit = ax + s["psi_n"] * (bd - ax)
    flat = lambda a: a.reshape(len(a), -1)
    rel = np.linalg.norm(flat(psi - psi_efit), axis=1) / np.linalg.norm(flat(psi_efit), axis=1)
    return {"psi_n": (psi - ax) / (bd - ax), "rel_l2": rel, "held_out": shot_id in held_out}


def _summary(s: dict, r: dict) -> dict:
    """Numbers for the insight panel, taken over near-steady slices only."""
    ok = r["steady"] & np.isfinite(r["H98"]) & np.isfinite(r["H89"])
    worst = np.nan_to_num(r["worst_limit"], nan=0.0)
    k = int(worst.argmax())
    med = lambda a: round(float(np.median(a[ok])), 2) if ok.any() else None
    return {"n_steady": int(ok.sum()), "H98_median": med(r["H98"]), "H89_median": med(r["H89"]),
            "peak_limit": LIMIT_NAMES[int(r["binding"][k])], "peak_limit_fraction": round(float(worst[k]), 2),
            "peak_limit_t_s": round(float(s["t_s"][k]), 3),
            "P_LH_median_MW": round(float(np.nanmedian(r["P_LH_MW"])), 2), "P_loss_median_MW": med(r["P_loss_MW"])}


def _add_hybrid(s: dict, r: dict, summary: dict) -> None:
    """IPB98 x learned correction on this shot, with its error beside plain IPB98 so it is never trusted blindly.
    Skipped when no trained model is on disk: the replay works without it."""
    from fusionlab import surrogate   # imports torch; keep it off the import path of the rest of the API
    if not surrogate.available():
        return
    Hc = surrogate.correction(surrogate.shot_features(s, r["P_loss_MW"]))
    r["H_learned"], r["W_hybrid_MJ"] = Hc, r["W_H_MJ"] * Hc
    m = json.loads(surrogate.METRICS_FILE.read_text())
    summary["tau_correction_held_out"] = s["meta"]["shot_id"] // m.get("block_size", 100) in m.get("held_out_blocks", [])
    ok = r["steady"] & np.isfinite(r["H98"]) & (r["H98"] > 0)
    if ok.any():
        rms = lambda a: round(float(np.sqrt(np.mean(np.log(a) ** 2))), 2)
        summary["rmse_ln_tau_ipb98"], summary["rmse_ln_tau_hybrid"] = rms(r["H98"][ok]), rms(r["H98"][ok] / Hc[ok])


def _warm_up():
    from fusionlab import surrogate
    if surrogate.available():
        surrogate.load()
    try:   # builds/loads the Warp kernels and traces the landing shot
        _lines(30166)
    except Exception:
        pass
    _eq_psi(30166)


# importing torch takes ~5 s; do it while the server starts so the first replay request is not the one that pays
threading.Thread(target=_warm_up, daemon=True).start()


@router.get("/eq_surrogate")
def eq_surrogate_metrics():
    """Holdout metrics of the equilibrium surrogate, as written by scripts/train_eq_surrogate.py."""
    from fusionlab.eq_surrogate import METRICS_FILE
    if not METRICS_FILE.exists():
        raise HTTPException(404, "no metrics: run scripts/train_eq_surrogate.py")
    return json.loads(METRICS_FILE.read_text())


@router.get("/surrogate")
def surrogate_metrics():
    """Holdout errors of the learned correction, as written by `python -m fusionlab.surrogate`."""
    from fusionlab.surrogate import METRICS_FILE
    if not METRICS_FILE.exists():
        raise HTTPException(404, "no trained model: run `make train`")
    return json.loads(METRICS_FILE.read_text())


@lru_cache(maxsize=1)
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
