"""FAIR-MAST client: real MAST shots -> arrays in FusionLab units, cached under data/.

Data: UKAEA FAIR-MAST (https://mastapp.site), licence CC BY-SA 4.0. Cite Jackson et al.,
SoftwareX 27 (2024) 101869. Signal -> quantity table and conversions: docs/PHYSICS.md.

Two anonymous sources:
  * shot table (ndjson, one request)   -> load_db():   one row per shot at the time of peak current
  * level-2 Zarr on S3 (one per shot)  -> load_shot(): traces on the EFIT time base + geometry

Everything leaves this module in project units: m, T, MA, 1e20 m^-3, keV, MW (plus MJ, s).
The app only ever reads the cache, so the demo does not need the network.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REST = "https://mastapp.site"
S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
DATA = Path(__file__).resolve().parent.parent / "data"
SHOTS = DATA / "shots"
DB_FILE = DATA / "mast_db.npz"

IP_MIN_MA = 0.05  # below this there is no plasma worth comparing against


# ---------------------------------------------------------------- shot table
# cpf_* = MAST "central physics file" scalars. "_ipmax" means "at the time of peak plasma current".
# column -> (cpf key, factor to project units)
_DB_COLS = {
    "Ip_MA": ("cpf_ip_max", 1e-3),            # kA
    "B_T": ("cpf_bt_ipmax", 1.0),             # T, signed; abs() taken below
    "n_e20": ("cpf_ne_bar_ipmax", 1e-20),     # m^-3, line average
    "P_ohm_MW": ("cpf_pohm_ipmax", 1.0),
    "P_nbi_MW": ("cpf_pnbi_ipmax", 1.0),
    "P_rad_MW": ("cpf_prad_ipmax", 1e-6),     # W
    "W_MJ": ("cpf_wmhd_ipmax", 1e-6),         # J, EFIT total stored energy
    "dWdt_MW": ("cpf_dwmhd_ipmax", 1e-6),     # W
    "tau_E_s": ("cpf_tautot_ipmax", 1.0),     # = W / (P_ohm + P_nbi - dW/dt)
    "q95": ("cpf_q95_ipmax", 1.0),
    "kappa": ("cpf_kappa_ipmax", 1.0),
    "a_m": ("cpf_amin_ipmax", 1.0),
    "R_m": ("cpf_rgeo_ipmax", 1.0),
    "V_m3": ("cpf_vol_ipmax", 1.0),
    "beta_t_pct": ("cpf_betmhd_ipmax", 1.0),
    "Te0_keV": ("cpf_te0_ipmax", 1e-3),       # eV
    "t_ipmax_s": ("cpf_tipmax", 1.0),
    "useful": ("cpf_useful", 1.0),
    "abort": ("cpf_abort", 1.0),
}


def fetch_db() -> dict:
    """Download the whole shot table (~64 MB, one request) and keep the numeric columns."""
    raw = urllib.request.urlopen(f"{REST}/ndjson/shots", timeout=180).read()
    rows = [json.loads(line) for line in raw.split(b"\n") if line.strip()]
    num = lambda v: float(v) if isinstance(v, (int, float)) else np.nan
    db = {name: np.array([num(r.get(key)) for r in rows]) * k for name, (key, k) in _DB_COLS.items()}
    db["B_T"] = np.abs(db["B_T"])
    db["P_nbi_MW"] = np.nan_to_num(db["P_nbi_MW"])    # null means no beams
    db["abort"] = np.nan_to_num(db["abort"])
    db["shot_id"] = np.array([r["shot_id"] for r in rows])
    db["campaign"] = np.array([int(str(r.get("campaign") or "M0")[1:]) for r in rows])
    return db


def load_db(refresh: bool = False) -> dict:
    """Shot table as a dict of equal-length arrays. Cached in data/mast_db.npz."""
    if refresh or not DB_FILE.exists():
        DATA.mkdir(exist_ok=True)
        np.savez_compressed(DB_FILE, **fetch_db())
    with np.load(DB_FILE) as z:
        return {k: z[k] for k in z.files}


def clean_db(db: dict) -> dict:
    """Rows usable for confinement analysis: flat-top-ish, not aborted, every input finite and sane."""
    need = ("Ip_MA", "B_T", "n_e20", "P_ohm_MW", "W_MJ", "dWdt_MW", "tau_E_s", "q95", "kappa", "a_m", "R_m", "V_m3")
    ok = np.all([np.isfinite(db[k]) for k in need], axis=0) & (db["abort"] == 0)
    P_loss = db["P_ohm_MW"] + db["P_nbi_MW"] - db["dWdt_MW"]
    ok &= (db["Ip_MA"] > 0.3) & (db["B_T"] > 0.2) & (db["n_e20"] > 0.05) & (db["W_MJ"] > 0.005)
    ok &= (P_loss > 0.1) & (np.abs(db["dWdt_MW"]) < 0.3 * (db["P_ohm_MW"] + db["P_nbi_MW"]))  # near steady state
    ok &= (db["tau_E_s"] > 1e-3) & (db["tau_E_s"] < 0.5) & (db["kappa"] > 1.2) & (db["a_m"] > 0.3)
    out = {k: v[ok] for k, v in db.items()}
    out["P_loss_MW"] = P_loss[ok]
    return out


# ---------------------------------------------------------------- one shot
def _open(shot_id: int):
    import zarr
    so = dict(anon=True, client_kwargs={"endpoint_url": S3_ENDPOINT})
    return zarr.open_consolidated(f"s3://mast/level2/shots/{shot_id}.zarr", storage_options=so, mode="r")


def _on(t, ts, y):
    """Linear interpolation of (ts, y) onto t, ignoring non-finite samples. NaN outside the source range."""
    ok = np.isfinite(ts) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full(t.shape, np.nan)
    return np.interp(t, ts[ok], y[ok], left=np.nan, right=np.nan)


def _pf_coils(g) -> dict:
    """PF coil cross-sections (one rectangle per winding-pack element) from the pf_active geometry arrays."""
    names = {k for k, _ in g.arrays()}
    stems = sorted(k[:-2] for k in names if k.endswith("_r") and {k[:-2] + s for s in ("_z", "_width", "_height")} <= names)
    cat = lambda s: np.concatenate([np.atleast_1d(g[st + s][:]).astype(float) for st in stems]) if stems else np.zeros(0)
    return {"coil_R": cat("_r"), "coil_Z": cat("_z"), "coil_W": cat("_width"), "coil_H": cat("_height")}


def fetch_shot(shot_id: int) -> dict:
    """Pull one level-2 shot from S3 and return it on the EFIT time base in project units."""
    g = _open(shot_id)
    groups = {k for k, _ in g.groups()}
    eq, summ = g["equilibrium"], g["summary"]

    eq_keys = ["time", "ip", "bvac_rmag", "q95", "beta_tor_normal", "elongation", "triangularity_upper",
               "triangularity_lower", "minor_radius", "geometric_axis_r", "magnetic_axis_r", "magnetic_axis_z",
               "plasma_energy", "volume", "li", "lcfs_r", "lcfs_z", "psi", "psi_axis", "psi_boundary", "major_radius", "z",
               "f", "q", "psi_norm", "profile_r",
               "b_field_pol_probe_measured", "flux_loop_measured", "pf_current", "ip_measured"]
    have = {k for k, _ in summ.arrays()}   # older campaigns lack some summary signals (M8 has no power_radiated)
    jobs = [("eq", k) for k in eq_keys] + [("summary", k) for k in ("time", "line_average_n_e", "power_ohm", "power_radiated") if k in have]
    if "thomson_scattering" in groups:
        jobs += [("thomson_scattering", k) for k in ("time", "major_radius", "t_e", "n_e", "t_e_core", "n_e_core")]
    if "wall" in groups:
        jobs += [("wall", k) for k in ("limiter_r", "limiter_z")]
    src = {"eq": eq, "summary": summ, **{n: g[n] for n in ("thomson_scattering", "wall") if n in groups}}
    with ThreadPoolExecutor(16) as pool:  # S3 latency dominates; one request per array
        raw = dict(zip(jobs, pool.map(lambda j: np.asarray(src[j[0]][j[1]][:], dtype=float), jobs)))
    e = lambda k: raw[("eq", k)]

    # EFIT arrays are (..., time). Keep slices with a plasma and a closed boundary.
    t_all, ip = e("time"), np.abs(e("ip")) * 1e-6
    keep = np.isfinite(ip) & (ip > IP_MIN_MA) & np.isfinite(e("q95")) & (np.isfinite(e("lcfs_r")).mean(0) > 0.5)
    keep &= e("plasma_energy") > 0   # EFIT returns negative energy on a few start-up slices
    t = t_all[keep]
    sl = lambda k: e(k)[..., keep]

    # Stored energy is EFM_PLASMA_ENERGY (3/2 ∫p dV), which matches the logged cpf_wmhd. The level-2 array
    # named "wmhd" is EFM_WPLASMD, the diamagnetic energy, ~3.5x larger on shot 30420. Do not use it.
    W = sl("plasma_energy") * 1e-6
    out = {
        "t_s": t, "Ip_MA": ip[keep], "B_T": np.abs(sl("bvac_rmag")), "q95": np.abs(sl("q95")),
        "beta_N": sl("beta_tor_normal"), "kappa": sl("elongation"),
        "delta": 0.5 * (sl("triangularity_upper") + sl("triangularity_lower")),
        "a_m": sl("minor_radius"), "R_m": sl("geometric_axis_r"), "V_m3": sl("volume"), "li": sl("li"),
        "R_mag_m": sl("magnetic_axis_r"), "Z_mag_m": sl("magnetic_axis_z"),
        "W_MJ": W, "dWdt_MW": np.gradient(W, t) if t.size > 2 else np.zeros_like(W),
        "lcfs_R": sl("lcfs_r").T, "lcfs_Z": sl("lcfs_z").T,            # (time, boundary point)
        "psi_R": e("major_radius"), "psi_Z": e("z"),
    }
    # Normalised poloidal flux: 0 on axis, 1 at the separatrix. Stored (time, Z, R) as float32.
    psi, ax, bd = sl("psi"), sl("psi_axis"), sl("psi_boundary")
    psi_n = (psi - ax) / np.where(bd - ax == 0, np.nan, bd - ax)
    out["psi_n"] = _psi_tzr(psi_n, eq, out["psi_R"].size, out["psi_Z"].size).astype(np.float32)
    # What field-line tracing needs: psi = psi_axis + psi_n (psi_bnd - psi_axis) [Wb/rad] and F = R B_phi [T m] as a
    # profile over psi_norm. EFIT's q profile is stored as found: q_mid(time, x) on q_mid_R = the archive's
    # "profile_r", a normalised 0-1 midplane coordinate whose mapping to metres we have not verified. Use q95 for checks.
    out.update(psi_axis_Wb=ax, psi_bnd_Wb=bd, psi_norm=e("psi_norm"), F_Tm=sl("f").T,
               q_mid=np.abs(sl("q")).T, q_mid_R=e("profile_r"))

    # Raw magnetics EFIT was given, in archive units and in the order fusionlab.eq_surrogate.INPUT_NAMES expects:
    # 78 poloidal field probes [T], 46 flux loops [Wb], 101 PF/passive currents [A], Rogowski Ip [A]. (time, 226)
    by_time = lambda k: (e(k) if e(k).shape[0] == t_all.size else e(k).T)[keep]
    out["eq_inputs"] = np.concatenate([by_time("b_field_pol_probe_measured"), by_time("flux_loop_measured"),
                                       by_time("pf_current"), e("ip_measured")[keep, None]], axis=1).astype(np.float32)

    ts = raw[("summary", "time")]
    summary = lambda k, scale: _on(t, ts, raw[("summary", k)]) * scale if ("summary", k) in raw else np.full(t.shape, np.nan)
    out["n_e20"] = summary("line_average_n_e", 1e-20)
    out["P_ohm_MW"] = summary("power_ohm", 1e-6)
    out["P_rad_MW"] = summary("power_radiated", 1e-6)

    if ("thomson_scattering", "time") in raw:
        tt = raw[("thomson_scattering", "time")]
        out["Te0_keV"] = _on(t, tt, raw[("thomson_scattering", "t_e_core")]) * 1e-3
        out["ne0_e20"] = _on(t, tt, raw[("thomson_scattering", "n_e_core")]) * 1e-20
        near = np.abs(tt[None, :] - t[:, None]).argmin(1)          # nearest laser pulse per EFIT slice
        out["ts_R"] = raw[("thomson_scattering", "major_radius")]
        out["ts_Te_keV"] = (raw[("thomson_scattering", "t_e")][:, near].T * 1e-3).astype(np.float32)
        out["ts_ne_e20"] = (raw[("thomson_scattering", "n_e")][:, near].T * 1e-20).astype(np.float32)
    if ("wall", "limiter_r") in raw:
        out["wall_R"], out["wall_Z"] = raw[("wall", "limiter_r")], raw[("wall", "limiter_z")]
    if "pf_active" in groups:
        out.update(_pf_coils(g["pf_active"]))

    meta = fetch_meta(shot_id)
    out["P_nbi_MW"] = _nbi_trace(t, meta)
    out["meta"] = meta
    return out


def _psi_tzr(psi_n, eq, nr, nz):
    """Reorder psi from the archive layout (dims read from the Zarr metadata, time last) to (time, Z, R)."""
    dims = list(getattr(eq["psi"].metadata, "dimension_names", None) or [])
    r_first = bool(dims) and "z" not in str(dims[0]).lower()
    if not dims and nr != nz:
        r_first = psi_n.shape[0] == nr
    return np.transpose(psi_n, (2, 1, 0) if r_first else (2, 0, 1))


_META_KEYS = {"campaign": "campaign", "heating": "heating", "timestamp": "timestamp", "divertor_config": "divertor_config",
              "plasma_shape": "plasma_shape", "current_range": "current_range", "objective": "shot_objective",
              "preshot": "preshot_description", "postshot": "postshot_description", "summary": "shot_summary",
              "t_nbi_start_s": "nbi_start_time", "t_nbi_end_s": "nbi_end_time", "P_nbi_max_MW": "nbi_power_max_power"}


def fetch_meta(shot_id: int) -> dict:
    """Logbook text and beam timing for one shot from the REST API."""
    d = json.load(urllib.request.urlopen(f"{REST}/json/shots/{shot_id}", timeout=60))
    clean = lambda v: v.strip().strip("'").strip() if isinstance(v, str) else v
    return {"shot_id": int(shot_id), **{k: clean(d.get(src)) for k, src in _META_KEYS.items()}}


def _nbi_trace(t, meta):
    """Level 2 carries no beam power trace, so NBI is a box: peak power between beam start and end (logged scalars)."""
    p, t0, t1 = (meta.get(k) for k in ("P_nbi_max_MW", "t_nbi_start_s", "t_nbi_end_s"))
    if not all(isinstance(v, (int, float)) for v in (p, t0, t1)):
        return np.zeros_like(t)
    return np.where((t >= t0) & (t <= t1), float(p), 0.0)


def load_shot(shot_id: int, refresh: bool = False) -> dict:
    """One shot as a dict of arrays (+ "meta"). Reads data/shots/{id}.npz; fetches from S3 only if missing."""
    f = SHOTS / f"{int(shot_id)}.npz"
    if refresh or not f.exists():
        s = fetch_shot(int(shot_id))
        SHOTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, meta=json.dumps(s.pop("meta")), **s)
    with np.load(f) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def cached_shots() -> list[int]:
    return sorted(int(p.stem) for p in SHOTS.glob("*.npz"))
