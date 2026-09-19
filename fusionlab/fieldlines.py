"""GPU magnetic field-line tracing through a real EFIT reconstruction, with NVIDIA Warp.

    uv run python -m fusionlab.fieldlines 30166     # q check vs EFIT + throughput + out/mast_30166_fieldlines.usdc

What it does: follows field lines of the EFIT equilibrium of a cached MAST shot (mast.load_shot), one GPU
thread per field line, RK4 in toroidal angle:

    dR/dphi = R B_R / B_phi,   dZ/dphi = R B_Z / B_phi
    B_R = -(1/R) dpsi/dZ,  B_Z = (1/R) dpsi/dR,  B_phi = F(psi_N) / R          (EFIT convention)
    psi = psi_axis + psi_N (psi_bnd - psi_axis)   [Wb/rad]

Units: R, Z in m; phi in rad; psi in Wb/rad; F = R B_phi in T m; q dimensionless. Grid layout is psi_n[time, Z, R].
Signs: F is negative on MAST and psi falls from axis to edge; both only set the direction of travel, so q is
reported as |q|.

Interpolation: psi_N and its gradient come from ONE bicubic (Catmull-Rom) interpolant of the 65 x 65 EFIT grid,
evaluated in the kernel. The grid is coarse (3 cm x 6.25 cm); taking the gradient analytically from the same C1
interpolant makes the traced field exactly tangent to that interpolant's contours, so a line stays on its flux
surface up to RK4 error. max |psi_N - psi_N,start| along the line is reported as the integrator check.

Safety factor from the lines: q = toroidal angle / poloidal angle about the magnetic axis, measured over whole
poloidal transits (the toroidal angle at the last completed transit is interpolated inside the step), so there is
no partial-transit bias. The check is this q at psi_N = 0.95 against EFIT's own q95 on every cached slice.

What this is not: these are field lines of the axisymmetric EFIT reconstruction. No magnetic islands, no 3D
fields, no error fields, no ripple: every line closes on a nested flux surface (or, outside the separatrix,
runs to the wall). It is an educational twin, not a new physics result.

Data: UKAEA FAIR-MAST (https://mastapp.site), CC BY-SA 4.0.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

wp.config.log_level = wp.LOG_WARNING      # no init banner in the API logs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
CHECK_FILE = ROOT / "models" / "fieldlines_check.json"
PROVENANCE = ("Field lines traced with NVIDIA Warp through the EFIT reconstruction "
              "(UKAEA FAIR-MAST, CC BY-SA 4.0)")

STEPS_PER_TURN = 256            # RK4 steps per toroidal turn for pictures (psi_N drift ~1e-5 over 5 turns)
Q_STEPS_PER_TURN = 512          # for q: >100 turns per line. On MAST B_pol ~ B_phi at the outboard edge, so a 1/256-turn
                                # step crosses most of a 3 cm grid cell and RK4 loses order at the cell edges (the
                                # interpolant is C1). 512 is where q stops changing in the 4th digit (see "convergence").
Q_MIN_TRANSITS = 12             # poloidal transits wanted for a q measurement (the turn count follows from EFIT q95)
USD_PSI_N = (0.3, 0.6, 0.9, 1.02, 1.02)       # the last two are the same scrape-off-layer line, traced both ways
USD_DIRECTION = (1.0, 1.0, 1.0, 1.0, -1.0)    # so it reaches both divertor ends
USD_TURNS = 5
USD_STORE_EVERY = 4             # 256 steps/turn -> 64 stored points/turn
LINE_COLOR = (0.25, 0.85, 1.0)  # cyan: cool, distinct from the red-yellow plasma ramp
LINE_WIDTH_M = 0.006
_N_ROW = 512                    # midplane samples used to bracket the start radius


def default_device() -> str:
    """cuda:0 when a GPU is visible, else Warp's CPU device. Everything in FusionLab shares that one GPU
    (PhysicsNeMo models and Warp kernels), so the app never occupies more than one. Override with
    FUSIONLAB_WARP_DEVICE=cpu|cuda:N.
    """
    env = os.environ.get("FUSIONLAB_WARP_DEVICE")
    if env:
        return env
    wp.init()
    n = wp.get_cuda_device_count()
    return "cuda:0" if n > 0 else "cpu"


# ---------------------------------------------------------------- Warp kernel
@wp.func
def _cr_w(t: float):
    """Catmull-Rom weights for the 4 neighbours at fractional position t in [0, 1)."""
    return wp.vec4(((-0.5 * t + 1.0) * t - 0.5) * t, (1.5 * t - 2.5) * t * t + 1.0,
                   ((-1.5 * t + 2.0) * t + 0.5) * t, (0.5 * t - 0.5) * t * t)


@wp.func
def _cr_d(t: float):
    """d(weights)/dt."""
    return wp.vec4((-1.5 * t + 2.0) * t - 0.5, (4.5 * t - 5.0) * t, (-4.5 * t + 4.0) * t + 0.5, (1.5 * t - 1.0) * t)


@wp.func
def _psi(psi_n: wp.array3d(dtype=wp.float32), k: int, R: float, Z: float,
         R0: float, Z0: float, inv_dR: float, inv_dZ: float):
    """(psi_N, dpsi_N/dR, dpsi_N/dZ) from one bicubic interpolant. Indices are clamped; the caller tests bounds."""
    x = (R - R0) * inv_dR
    y = (Z - Z0) * inv_dZ
    ix = wp.clamp(int(wp.floor(x)), 1, psi_n.shape[2] - 3)
    iy = wp.clamp(int(wp.floor(y)), 1, psi_n.shape[1] - 3)
    tx = x - float(ix)
    ty = y - float(iy)
    wx = _cr_w(tx)
    dx = _cr_d(tx)
    wy = _cr_w(ty)
    dy = _cr_d(ty)
    val = float(0.0)
    gR = float(0.0)
    gZ = float(0.0)
    for a in range(4):
        row_v = float(0.0)
        row_d = float(0.0)
        for b in range(4):
            p = psi_n[k, iy - 1 + a, ix - 1 + b]
            row_v += wx[b] * p
            row_d += dx[b] * p
        val += wy[a] * row_v
        gR += wy[a] * row_d
        gZ += dy[a] * row_v
    return wp.vec3(val, gR * inv_dR, gZ * inv_dZ)


@wp.func
def _rhs(psi_n: wp.array3d(dtype=wp.float32), F_Tm: wp.array2d(dtype=wp.float32), k: int, dpsi_Wb: float,
         R: float, Z: float, R0: float, Z0: float, inv_dR: float, inv_dZ: float):
    """(dR/dphi, dZ/dphi). F is linear in psi_N on a uniform 0-1 grid; outside the separatrix it is the edge value."""
    p = _psi(psi_n, k, R, Z, R0, Z0, inv_dR, inv_dZ)
    nF = F_Tm.shape[1]
    u = wp.clamp(p[0], 0.0, 1.0) * float(nF - 1)
    j = wp.min(int(u), nF - 2)
    F = F_Tm[k, j] + (u - float(j)) * (F_Tm[k, j + 1] - F_Tm[k, j])
    # R B_R / B_phi = -R psi_Z / F ;  R B_Z / B_phi = R psi_R / F
    return wp.vec2(-R * dpsi_Wb * p[2] / F, R * dpsi_Wb * p[1] / F)


@wp.kernel
def _trace_kernel(psi_n: wp.array3d(dtype=wp.float32), F_Tm: wp.array2d(dtype=wp.float32),
                  dpsi_Wb: wp.array(dtype=wp.float32), R_mag: wp.array(dtype=wp.float32),
                  Z_mag: wp.array(dtype=wp.float32), line_slice: wp.array(dtype=wp.int32),
                  R_start: wp.array(dtype=wp.float32), Z_start: wp.array(dtype=wp.float32),
                  direction: wp.array(dtype=wp.float32),
                  R0: float, Z0: float, dR: float, dZ: float, dphi: float, n_steps: int, store_every: int,
                  traj_R: wp.array2d(dtype=wp.float32), traj_Z: wp.array2d(dtype=wp.float32),
                  out_psi0: wp.array(dtype=wp.float32), out_drift: wp.array(dtype=wp.float32),
                  out_transits: wp.array(dtype=wp.int32), out_cross_step: wp.array(dtype=wp.float32),
                  out_valid: wp.array(dtype=wp.int32)):
    i = wp.tid()
    k = line_slice[i]
    R = R_start[i]
    Z = Z_start[i]
    h = dphi * direction[i]
    inv_dR = 1.0 / dR
    inv_dZ = 1.0 / dZ
    R_hi = R0 + dR * float(psi_n.shape[2] - 2)      # the 4x4 stencil needs one cell of margin
    Z_hi = Z0 + dZ * float(psi_n.shape[1] - 2)
    dp = dpsi_Wb[k]
    Rm = R_mag[k]
    Zm = Z_mag[k]

    psi0 = _psi(psi_n, k, R, Z, R0, Z0, inv_dR, inv_dZ)[0]
    drift = float(0.0)
    theta_prev = wp.atan2(Z - Zm, R - Rm)
    theta = float(0.0)          # poloidal angle inside the current transit (kept small: float32 stays exact enough)
    transits = int(0)
    cross_step = float(0.0)
    valid = int(n_steps)
    alive = int(1)
    store = int(store_every > 0)
    if store == 1:
        traj_R[i, 0] = R
        traj_Z[i, 0] = Z

    for s in range(n_steps):
        if alive == 1:
            k1 = _rhs(psi_n, F_Tm, k, dp, R, Z, R0, Z0, inv_dR, inv_dZ)
            k2 = _rhs(psi_n, F_Tm, k, dp, R + 0.5 * h * k1[0], Z + 0.5 * h * k1[1], R0, Z0, inv_dR, inv_dZ)
            k3 = _rhs(psi_n, F_Tm, k, dp, R + 0.5 * h * k2[0], Z + 0.5 * h * k2[1], R0, Z0, inv_dR, inv_dZ)
            k4 = _rhs(psi_n, F_Tm, k, dp, R + h * k3[0], Z + h * k3[1], R0, Z0, inv_dR, inv_dZ)
            Rn = R + h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
            Zn = Z + h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
            if Rn > R0 + dR and Rn < R_hi and Zn > Z0 + dZ and Zn < Z_hi:
                R = Rn
                Z = Zn
                drift = wp.max(drift, wp.abs(_psi(psi_n, k, R, Z, R0, Z0, inv_dR, inv_dZ)[0] - psi0))
                th = wp.atan2(Z - Zm, R - Rm)
                d = th - theta_prev
                if d > wp.pi:
                    d -= 2.0 * wp.pi
                if d < -wp.pi:
                    d += 2.0 * wp.pi
                theta_prev = th
                a_prev = wp.abs(theta)
                theta += d
                a_new = wp.abs(theta)
                if a_new >= 2.0 * wp.pi:          # one more whole poloidal transit: note where inside the step
                    cross_step = float(s) + (2.0 * wp.pi - a_prev) / (a_new - a_prev)
                    transits += 1
                    theta -= wp.sign(theta) * 2.0 * wp.pi
            else:                                   # left the grid: freeze here
                alive = 0
                valid = s
        if store == 1:
            if (s + 1) % store_every == 0:
                traj_R[i, (s + 1) / store_every] = R
                traj_Z[i, (s + 1) / store_every] = Z

    out_psi0[i] = psi0
    out_drift[i] = drift
    out_transits[i] = transits
    out_cross_step[i] = cross_step
    out_valid[i] = valid


# ---------------------------------------------------------------- numpy side
def _cr_weights(t):
    return np.stack([((-0.5 * t + 1.0) * t - 0.5) * t, (1.5 * t - 2.5) * t * t + 1.0,
                     ((-1.5 * t + 2.0) * t + 0.5) * t, (0.5 * t - 0.5) * t * t], -1)


def psi_n_at(shot: dict, slice_idx, R_m, Z_m):
    """psi_N at (R, Z) on EFIT slice(s) slice_idx: the kernel's bicubic interpolant, in numpy (broadcasts)."""
    psi, Rg, Zg = shot["psi_n"], np.asarray(shot["psi_R"], float), np.asarray(shot["psi_Z"], float)
    k, R, Z = np.broadcast_arrays(np.asarray(slice_idx), np.asarray(R_m, float), np.asarray(Z_m, float))
    x, y = (R - Rg[0]) / (Rg[1] - Rg[0]), (Z - Zg[0]) / (Zg[1] - Zg[0])
    ix = np.clip(np.floor(x), 1, Rg.size - 3).astype(int)
    iy = np.clip(np.floor(y), 1, Zg.size - 3).astype(int)
    wx, wy = _cr_weights(x - ix), _cr_weights(y - iy)
    o = np.arange(4) - 1
    patch = psi[k[..., None, None], (iy[..., None] + o)[..., :, None], (ix[..., None] + o)[..., None, :]]
    return np.einsum("...a,...ab,...b->...", wy, patch, wx)


def start_points(shot: dict, slice_idx, psi_n_starts):
    """Outboard-midplane start radius per line: R > R_mag on Z = Z_mag where psi_N(R) = the requested value.

    Bracket on a fine row, then a few regula-falsi steps on the bicubic interpolant. Returns (R_m, Z_m, found).
    """
    k = np.asarray(slice_idx, int)
    target = np.asarray(psi_n_starts, float)
    Rm, Zm = np.asarray(shot["R_mag_m"], float)[k], np.asarray(shot["Z_mag_m"], float)[k]
    R_edge = float(shot["psi_R"][-3])
    uk, inv = np.unique(k, return_inverse=True)          # one midplane row per distinct slice, shared by its lines
    Rm_u, Zm_u = np.asarray(shot["R_mag_m"], float)[uk], np.asarray(shot["Z_mag_m"], float)[uk]
    row_R = Rm_u[:, None] + (R_edge - Rm_u)[:, None] * np.linspace(0.0, 1.0, _N_ROW)       # (n_distinct, _N_ROW)
    row = psi_n_at(shot, uk[:, None], row_R, Zm_u[:, None])
    above = row[inv] >= target[:, None]
    hit = np.clip(above.argmax(1), 1, _N_ROW - 1)                  # first sample at or past the target
    found = above.any(1) & np.isfinite(target) & np.isfinite(Rm)
    lo, hi, f_lo, f_hi = row_R[inv, hit - 1], row_R[inv, hit], row[inv, hit - 1] - target, row[inv, hit] - target
    for _ in range(4):
        R = lo - f_lo * (hi - lo) / np.where(f_hi - f_lo == 0, 1.0, f_hi - f_lo)
        f = psi_n_at(shot, k, R, Zm) - target
        left = f < 0
        lo, f_lo = np.where(left, R, lo), np.where(left, f, f_lo)
        hi, f_hi = np.where(left, hi, R), np.where(left, f_hi, f)
    return np.where(found, R, Rm + 0.05), Zm, found


def trace_lines(shot: dict, slice_idx, psi_n_starts, *, n_turns: float = 5, steps_per_turn: int = STEPS_PER_TURN,
                store_every: int = 1, direction=1.0, device: str | None = None) -> dict:
    """Trace one field line per (slice_idx[j], psi_n_starts[j]) pair in ONE kernel launch (one thread per line).

    store_every = n keeps every n-th step of the trajectory; 0 keeps none (q and drift only, for big batches).
    direction = +1 / -1 per line: sense of travel in phi.

    Returns a dict of per-line arrays:
      R_m, Z_m (n_lines, n_stored)   trajectory; a line that leaves the grid is frozen at its last point
      phi_rad  (n_lines, n_stored)   toroidal angle of the stored points (signed by direction)
      slice, psi_n_start (requested), psi_n_start_interp (interpolant value at the start point), R_start_m
      psi_n_drift                    max |psi_N - psi_N,start| along the line
      q                              |q| over whole poloidal transits (NaN if the line made none)
      n_transits, n_valid_steps, left_grid, device, n_steps, launch_s
    """
    device = device or default_device()
    k = np.atleast_1d(np.asarray(slice_idx, np.int32))
    target = np.atleast_1d(np.asarray(psi_n_starts, float))
    k, target = (np.ascontiguousarray(a) for a in np.broadcast_arrays(k, target))
    n = k.size
    sense = np.ascontiguousarray(np.broadcast_to(np.asarray(direction, np.float32), (n,)))
    R0_m, Z0_m, found = start_points(shot, k, target)

    Rg, Zg = np.asarray(shot["psi_R"], float), np.asarray(shot["psi_Z"], float)
    n_steps = int(round(n_turns * steps_per_turn))
    n_store = n_steps // store_every + 1 if store_every > 0 else 1
    f32 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float32), dtype=wp.float32, device=device)
    dev_in = [wp.array(np.ascontiguousarray(shot["psi_n"], dtype=np.float32), dtype=wp.float32, device=device),
              f32(shot["F_Tm"]), f32(np.asarray(shot["psi_bnd_Wb"], float) - np.asarray(shot["psi_axis_Wb"], float)),
              f32(shot["R_mag_m"]), f32(shot["Z_mag_m"]), wp.array(k.astype(np.int32), dtype=wp.int32, device=device),
              f32(R0_m), f32(Z0_m), f32(sense)]
    traj_R = wp.zeros((n, n_store), dtype=wp.float32, device=device)
    traj_Z = wp.zeros((n, n_store), dtype=wp.float32, device=device)
    outs = [wp.zeros(n, dtype=t, device=device) for t in (wp.float32, wp.float32, wp.int32, wp.float32, wp.int32)]

    wp.synchronize_device(device)
    t0 = time.perf_counter()
    wp.launch(_trace_kernel, dim=n, device=device,
              inputs=dev_in + [float(Rg[0]), float(Zg[0]), float(Rg[1] - Rg[0]), float(Zg[1] - Zg[0]),
                               2.0 * np.pi / steps_per_turn, n_steps, int(store_every), traj_R, traj_Z] + outs)
    wp.synchronize_device(device)
    launch_s = time.perf_counter() - t0

    psi0, drift, transits, cross_step, valid = (o.numpy() for o in outs)
    dphi = 2.0 * np.pi / steps_per_turn
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(transits > 0, cross_step.astype(float) * dphi / (2.0 * np.pi * transits), np.nan)
    q = np.where(found, q, np.nan)
    stride = max(store_every, 1)
    phi = sense[:, None].astype(float) * (np.arange(n_store) * stride * dphi)[None, :]
    return {"R_m": traj_R.numpy(), "Z_m": traj_Z.numpy(), "phi_rad": phi, "slice": k, "psi_n_start": target,
            "psi_n_start_interp": psi0, "R_start_m": R0_m, "psi_n_drift": np.where(found, drift, np.nan), "q": q,
            "n_transits": transits, "n_valid_steps": valid, "left_grid": valid < n_steps, "found_start": found,
            "device": device, "n_steps": n_steps, "launch_s": launch_s}


def trace(shot: dict, i: int, psi_n_starts, n_turns: float = 5, steps_per_turn: int = STEPS_PER_TURN, **kw) -> dict:
    """Field lines of EFIT slice i, one per requested psi_N, started on the outboard midplane. See trace_lines."""
    starts = np.atleast_1d(np.asarray(psi_n_starts, float))
    return trace_lines(shot, np.full(starts.size, int(i)), starts, n_turns=n_turns, steps_per_turn=steps_per_turn, **kw)


def xyz(tr: dict) -> np.ndarray:
    """Trajectories as Cartesian points (n_lines, n_stored, 3): x = R cos phi, y = R sin phi, z = Z (Z up, metres)."""
    return np.stack([tr["R_m"] * np.cos(tr["phi_rad"]), tr["R_m"] * np.sin(tr["phi_rad"]), tr["Z_m"]], -1)


# ---------------------------------------------------------------- safety factor + the check against EFIT
def _turns_for_q(q_max: float) -> int:
    return int(np.ceil(Q_MIN_TRANSITS * max(float(q_max), 1.0)))


def q_traced(shot: dict, i: int, psi_n: float = 0.95, *, n_turns: float | None = None,
             steps_per_turn: int = Q_STEPS_PER_TURN, device: str | None = None) -> float:
    """|q| of the flux surface psi_N on EFIT slice i, measured from a traced line over whole poloidal transits.

    n_turns defaults to enough toroidal turns for ~12 poloidal transits, sized from EFIT's q95 (length only).
    """
    if n_turns is None:
        q_ref = float(shot["q95"][i]) if np.isfinite(shot["q95"][i]) else 10.0
        n_turns = _turns_for_q(1.5 * q_ref)
    tr = trace_lines(shot, [i], [psi_n], n_turns=n_turns, steps_per_turn=steps_per_turn, store_every=0, device=device)
    return float(tr["q"][0])


def q_check(shot: dict, psi_n: float = 0.95, *, steps_per_turn: int = Q_STEPS_PER_TURN, device: str | None = None) -> dict:
    """Traced q at psi_N vs EFIT q95 on every slice with a finite q95: all slices in one batched kernel launch."""
    q_efit = np.asarray(shot["q95"], float)
    idx = np.flatnonzero(np.isfinite(q_efit))
    n_turns = _turns_for_q(np.max(q_efit[idx]))
    tr = trace_lines(shot, idx, np.full(idx.size, psi_n), n_turns=n_turns, steps_per_turn=steps_per_turn,
                     store_every=0, device=device)
    signed = (tr["q"] - q_efit[idx]) / q_efit[idx]
    rel = np.abs(signed)
    ok = np.isfinite(rel)
    summary = {"n_slices": int(idx.size), "n_compared": int(ok.sum()),
               "median_rel_err": float(np.median(rel[ok])), "p95_rel_err": float(np.percentile(rel[ok], 95)),
               "max_rel_err": float(rel[ok].max()), "median_signed_rel_err": float(np.median(signed[ok])),
               "psi_n_drift_max": float(np.nanmax(tr["psi_n_drift"])),
               "psi_n_drift_median": float(np.nanmedian(tr["psi_n_drift"])),
               "min_poloidal_transits": int(tr["n_transits"][ok].min()), "n_turns": n_turns,
               "steps_per_turn": int(steps_per_turn), "lines_x_steps": int(idx.size * tr["n_steps"]),
               "launch_s": float(tr["launch_s"])}
    return {"slice": idx, "t_s": np.asarray(shot["t_s"], float)[idx], "q_traced": tr["q"], "q_efit": q_efit[idx],
            "rel_err": rel, "signed_rel_err": signed, "psi_n_drift": tr["psi_n_drift"], "n_transits": tr["n_transits"],
            "n_turns": n_turns, "steps_per_turn": steps_per_turn, "device": tr["device"], "launch_s": tr["launch_s"],
            "summary": summary}


# ---------------------------------------------------------------- throughput
def first_launch_s(device: str, cold: bool) -> float:
    """Wall time of the first tiny trace in a fresh process: kernel build + device context + launch.

    cold=True points Warp at an empty throw-away kernel cache, so the kernel is compiled from source;
    cold=False uses the normal cache (what a user sees on every run after the first). The difference is the compile.
    """
    import subprocess
    import tempfile

    code = ("import time\nfrom fusionlab import mast, fieldlines as fl\ns = mast.load_shot(mast.cached_shots()[0])\n"
            f"t0 = time.perf_counter()\nfl.trace(s, 0, [0.5], n_turns=1, steps_per_turn=16, device={device!r})\n"
            "print(time.perf_counter() - t0)")
    with tempfile.TemporaryDirectory() as tmp:
        # cold = no Warp kernel cache AND no NVIDIA driver JIT cache (which otherwise hides ~2 s of the CUDA build)
        env = {**os.environ, "WARP_CACHE_PATH": tmp, "CUDA_CACHE_DISABLE": "1"} if cold else dict(os.environ)
        for _ in range(1 if cold else 2):          # cached: the first run primes the cache, the second is the number
            res = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, check=True)
    return float(res.stdout.strip().splitlines()[-1])


def benchmark(shot: dict, *, n_lines: int = 4096, n_turns: int = 20, steps_per_turn: int = 256, store_every: int = 1,
              device: str | None = None, repeats: int = 3) -> dict:
    """RK4 field-line steps per second, wp.synchronize around the launch, after a warm-up launch.

    Times the kernel launch only (host<->device copies and the numpy start-point search are outside the timer).
    The first launch in a process includes kernel compilation or a kernel-cache load; see first_launch_s.
    """
    device = device or default_device()
    i = shot["t_s"].size // 2
    trace_lines(shot, [i], [0.5], n_turns=1, steps_per_turn=16, device=device)      # warm-up: compile / load
    starts = np.linspace(0.05, 0.98, n_lines)
    runs, left = [], 0
    for _ in range(repeats):
        tr = trace(shot, i, starts, n_turns=n_turns, steps_per_turn=steps_per_turn, store_every=store_every, device=device)
        runs.append(tr["launch_s"])
        left = int(tr["left_grid"].sum())
    steps = n_lines * n_turns * steps_per_turn
    return {"device": device, "device_name": wp.get_device(device).name, "n_lines": n_lines, "n_turns": n_turns,
            "steps_per_turn": steps_per_turn, "store_every": store_every, "steps": steps, "lines_stopped_early": left,
            "launch_s_best": float(min(runs)), "launch_s_all": [float(r) for r in runs],
            "steps_per_s": float(steps / min(runs))}


# ---------------------------------------------------------------- OpenUSD
def _inside_wall(R, Z, wall_R, wall_Z):
    """Even-odd point-in-polygon test, vectorised over points x wall segments."""
    ok = np.isfinite(wall_R) & np.isfinite(wall_Z)
    wr, wz = np.asarray(wall_R, float)[ok], np.asarray(wall_Z, float)[ok]
    r1, z1, r2, z2 = wr, wz, np.roll(wr, -1), np.roll(wz, -1)
    Rp, Zp = R[..., None], Z[..., None]
    span = (z1 > Zp) != (z2 > Zp)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_cross = r1 + (Zp - z1) * (r2 - r1) / (z2 - z1)
    return (np.sum(span & (Rp < r_cross), -1) % 2) == 1


def _stop_at_wall(tr: dict, shot: dict) -> dict:
    """Hold each line at its last point inside the vessel (a scrape-off-layer line ends on the divertor)."""
    if "wall_R" not in shot:
        return tr
    inside = _inside_wall(tr["R_m"], tr["Z_m"], shot["wall_R"], shot["wall_Z"])
    n_pts = inside.shape[1]
    first_out = np.where(inside.all(1), n_pts, (~inside).argmax(1))
    last = np.clip(np.minimum(np.arange(n_pts)[None, :], first_out[:, None] - 1), 0, None)
    out = dict(tr)
    for key in ("R_m", "Z_m", "phi_rad"):
        out[key] = np.take_along_axis(tr[key], last, 1)
    out["n_inside_wall"] = first_out
    return out


def usd_lines(shot: dict, psi_n=USD_PSI_N, direction=USD_DIRECTION, *, n_turns: float = USD_TURNS,
              steps_per_turn: int = STEPS_PER_TURN, store_every: int = USD_STORE_EVERY, device: str | None = None):
    """Cartesian points (nt, n_lines, n_pts, 3) for every EFIT slice: all slices x lines in one kernel launch."""
    nt, psi_n = shot["t_s"].size, np.asarray(psi_n, float)
    k = np.repeat(np.arange(nt), psi_n.size)
    tr = trace_lines(shot, k, np.tile(psi_n, nt), n_turns=n_turns, steps_per_turn=steps_per_turn,
                     store_every=store_every, direction=np.tile(np.asarray(direction, np.float32), nt), device=device)
    pts = xyz(_stop_at_wall(tr, shot))
    return pts.reshape(nt, psi_n.size, -1, 3), tr


def add_field_lines(stage_or_path, shot: dict, *, psi_n=USD_PSI_N, direction=USD_DIRECTION, n_turns: float = USD_TURNS,
                    steps_per_turn: int = STEPS_PER_TURN, store_every: int = USD_STORE_EVERY,
                    color=LINE_COLOR, width_m: float = LINE_WIDTH_M, device: str | None = None):
    """Add /MAST/FieldLines (UsdGeom.BasisCurves, linear) to a stage written by usd_export.export_shot.

    stage_or_path: an open Usd.Stage (edited in place, not saved) or a path (opened, edited, saved). Points are
    time-sampled, one sample per EFIT slice (time code = slice index, as in the rest of the stage), with a
    constant vertex count per line. Returns the BasisCurves prim's schema object.
    """
    from pxr import Sdf, Usd, UsdGeom, Vt

    from fusionlab import usd_export

    stage = stage_or_path if isinstance(stage_or_path, Usd.Stage) else Usd.Stage.Open(str(stage_or_path))
    pts, tr = usd_lines(shot, psi_n, direction, n_turns=n_turns, steps_per_turn=steps_per_turn,
                        store_every=store_every, device=device)
    nt, n_lines, n_pts, _ = pts.shape
    pts = np.round(pts.reshape(nt, n_lines * n_pts, 3), usd_export.ROUND_M).astype(np.float32)
    ext = usd_export._extent(pts)

    curves = UsdGeom.BasisCurves.Define(stage, "/MAST/FieldLines")
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n_lines, n_pts, dtype=np.int32)))
    curves.CreateWidthsAttr(Vt.FloatArray([float(width_m)]))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(usd_export._vec3f([color]))
    p_attr, e_attr = curves.CreatePointsAttr(), curves.CreateExtentAttr()
    k0 = int(np.nanargmax(shot["Ip_MA"])) if "Ip_MA" in shot else nt // 2      # default value, as for the plasma
    p_attr.Set(usd_export._vec3f(pts[k0]))
    e_attr.Set(usd_export._vec3f(ext[k0]))
    for k in range(nt):
        p_attr.Set(usd_export._vec3f(pts[k]), k)
        e_attr.Set(usd_export._vec3f(ext[k]), k)
    shader = usd_export._material(stage, curves, color)        # Blender / USD Composer shade from materials
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(tuple(float(c) for c in color))

    prim = curves.GetPrim()
    prim.SetCustomDataByKey("provenance", PROVENANCE)
    prim.SetCustomDataByKey("psi_n_start", ", ".join(f"{p:g}" for p in np.asarray(psi_n, float)))
    prim.SetCustomDataByKey("direction", ", ".join(f"{d:+g}" for d in np.asarray(direction, float)))
    prim.SetCustomDataByKey("scope_note", "Field lines of the axisymmetric EFIT reconstruction: no islands, no 3D "
                                          "or error fields. Lines outside the separatrix stop at the limiter outline.")
    prim.SetCustomDataByKey("trace", f"RK4 in toroidal angle, {steps_per_turn} steps/turn, {n_turns:g} turns, "
                                     f"every {store_every}th step stored, device {tr['device']}")
    if not isinstance(stage_or_path, Usd.Stage):
        stage.GetRootLayer().Save()
    return curves


def export_shot_with_field_lines(shot: dict, path: str | Path, *, device: str | None = None, **export_kw) -> Path:
    """usd_export.export_shot (unchanged), then /MAST/FieldLines added to the same file."""
    from fusionlab import usd_export

    path = usd_export.export_shot(shot, path, **export_kw)
    add_field_lines(path, shot, device=device)
    return Path(path)


# ---------------------------------------------------------------- CLI
def run_checks(shot_ids, device: str | None = None) -> dict:
    """q check on every listed shot. Returns {shot_id: summary} with python floats, ready for JSON."""
    from fusionlab import mast

    return {int(sid): q_check(mast.load_shot(sid), device=device)["summary"] for sid in shot_ids}


def main(argv=None) -> int:
    from fusionlab import mast

    args = list(sys.argv[1:] if argv is None else argv)
    shot_id = next((int(a) for a in args if a.isdigit()), 30166)
    device = default_device()
    shot = mast.load_shot(shot_id)

    trace(shot, 0, [0.5], n_turns=1, steps_per_turn=16, device=device)     # warm-up, so no timing below includes the build
    print(f"Warp {wp.__version__} on {device} ({wp.get_device(device).name})")

    print("\nq from traced field lines at psi_N = 0.95 vs EFIT q95 (field lines of the EFIT reconstruction)")
    print(f"{'shot':>6} {'slices':>6} {'median':>8} {'p95':>8} {'max':>8} {'bias':>8} {'drift_max':>10} {'transits':>8} {'launch_s':>8}")
    checks = run_checks(mast.cached_shots(), device)
    for sid, c in checks.items():
        print(f"{sid:>6} {c['n_compared']:>6} {c['median_rel_err']:>8.2%} {c['p95_rel_err']:>8.2%} {c['max_rel_err']:>8.2%} "
              f"{c['median_signed_rel_err']:>+8.2%} {c['psi_n_drift_max']:>10.1e} {c['min_poloidal_transits']:>8d} {c['launch_s']:>8.2f}"
              + ("   <-" if sid == shot_id else ""))

    print(f"\nstep-size convergence, shot {shot_id} (integrator accuracy; independent of EFIT's q)")
    conv = {}
    for spt in (128, 256, 512, 1024):
        c = q_check(shot, steps_per_turn=spt, device=device)["summary"]
        conv[str(spt)] = {k: c[k] for k in ("median_rel_err", "p95_rel_err", "max_rel_err", "psi_n_drift_max", "launch_s")}
        print(f"  {spt:>5} steps/turn: median {c['median_rel_err']:.2%}  p95 {c['p95_rel_err']:.2%}  max {c['max_rel_err']:.2%}  "
              f"psi_N drift max {c['psi_n_drift_max']:.1e}  ({c['launch_s']:.2f} s)")

    print("\nthroughput (RK4 steps/s, trajectory stored, synchronised, after warm-up)")
    gpu = device != "cpu"
    bench = {"gpu": benchmark(shot, device=device) if gpu else None,
             "gpu_large": benchmark(shot, n_lines=65536, store_every=16, device=device) if gpu else None,
             "cpu": benchmark(shot, n_lines=256, device="cpu", repeats=1)}
    for b in filter(None, bench.values()):
        print(f"  {b['device']:>7} {b['device_name']}: {b['n_lines']} lines x {b['n_turns']} turns x {b['steps_per_turn']} steps "
              f"= {b['steps']:.3g} steps in {b['launch_s_best']:.3f} s -> {b['steps_per_s']:.3g} steps/s "
              f"(every {b['store_every']} step stored)")
    first = {f"{d}_{'cold' if cold else 'cached'}_s": first_launch_s(d, cold)
             for d in dict.fromkeys([device, "cpu"]) for cold in (True, False)}
    print("first launch in a fresh process (kernel build + context + launch): "
          + ", ".join(f"{k} {v:.2f}" for k, v in first.items()))

    out = export_shot_with_field_lines(shot, OUT / f"mast_{shot_id}_fieldlines.usdc", device=device)
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(out))          # keep the stage alive while its prims are read
    fl = UsdGeom.BasisCurves(stage.GetPrimAtPath("/MAST/FieldLines"))
    counts = [int(c) for c in fl.GetCurveVertexCountsAttr().Get()]
    usd = {"path": str(out.relative_to(ROOT)), "size_MB": out.stat().st_size / 1e6, "n_curves": len(counts),
           "vertices_per_curve": counts[0], "time_samples": fl.GetPointsAttr().GetNumTimeSamples(),
           "psi_n_start": list(USD_PSI_N), "direction": list(USD_DIRECTION)}
    print(f"\n{out}: {usd['size_MB']:.2f} MB, {usd['n_curves']} curves x {usd['vertices_per_curve']} vertices, "
          f"{usd['time_samples']} time samples")

    CHECK_FILE.parent.mkdir(exist_ok=True)
    CHECK_FILE.write_text(json.dumps({
        "what": "Field lines of the EFIT reconstruction (axisymmetric) traced with NVIDIA Warp; q at psi_N = 0.95 "
                "from the traced lines vs EFIT q95, per cached FAIR-MAST shot. Relative errors are |q_traced - q95| / q95.",
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "warp_version": wp.__version__,
        "device": device, "gpu_name": wp.get_device(device).name if device != "cpu" else None,
        "first_launch": first, "interpolation": "bicubic Catmull-Rom on the 65x65 EFIT grid, analytic gradient",
        "integrator": f"RK4 in toroidal angle, float32, {Q_STEPS_PER_TURN} steps per turn for q, {STEPS_PER_TURN} for the USD lines",
        "q_check": {str(k): v for k, v in checks.items()}, "convergence_shot": shot_id, "convergence": conv,
        "benchmark": bench, "usd": usd}, indent=2) + "\n")
    print(f"wrote {CHECK_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
