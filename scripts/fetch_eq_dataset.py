"""Download the EFIT magnetics -> psi(R,Z) training set from FAIR-MAST level 2 (anonymous S3).

Every array comes from one group, `equilibrium`, so inputs and target already share the EFIT time base:
  inputs   b_field_pol_probe_measured (time, 78) [T], flux_loop_measured (time, 46) [Wb], pf_current (time, 101) [A],
           ip_measured [A] (the Rogowski value EFIT was given; `ip` is EFIT's fitted current and is only used to filter slices)
  weights  *_weight arrays: 0 means EFIT ignored that sensor. Stored so the training script can decide.
  target   psi (z, major_radius, time) [Wb/rad], plus psi_axis, psi_boundary, magnetic_axis_r/z, lcfs_r/z for the metrics.

One float32 .npz per shot under data/eq/shots/ (git-ignored), so a rerun resumes. Only slices with a plasma
are kept: finite q95, |ip| > 0.1 MA, psi finite on the whole grid.

Usage: for i in 0 1 2 3; do uv run python scripts/fetch_eq_dataset.py --n 1767 --minutes 12 --shard $i/4 & done; wait
(1767 = every usable M8+M9 shot in the table; reruns skip files that exist, so the set can be grown in steps.)
Data: UKAEA FAIR-MAST, CC BY-SA 4.0 (Jackson et al., SoftwareX 27 (2024) 101869).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusionlab import mast  # noqa: E402

OUT = mast.DATA / "eq" / "shots"
IP_MIN_A = 1e5
PER_TIME = ["time", "ip", "ip_measured", "q95", "psi_axis", "psi_boundary", "magnetic_axis_r", "magnetic_axis_z", "lcfs_r", "lcfs_z"]
SENSORS = ["b_field_pol_probe_measured", "b_field_pol_probe_weight", "flux_loop_measured", "flux_loop_weight",
           "pf_current", "pf_current_weight"]
GRID = ["major_radius", "z"]
TOPUP = ["ip_measured"]   # added after the first download: `ip` is EFIT's fitted current, the input must be the measured one

# Politeness budget for the public endpoint: run as 4 shards (--shard i/4), each 4 shot opens + 8 array reads
# in flight, so 48 requests at most. Separate processes because zarr/s3fs funnel all I/O through one event loop.
SHOT_WORKERS, ARRAY_WORKERS = 4, 8
_arrays = ThreadPoolExecutor(ARRAY_WORKERS)


def pick_shots(n: int, seed: int = 0) -> np.ndarray:
    """Usable shots from the table: a seeded random draw from campaigns M8+M9 first, older campaigns after."""
    c = mast.clean_db(mast.load_db())
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(c["shot_id"].size), c["campaign"] < 8))
    return c["shot_id"][order][:n].astype(int)


def _time_first(eq, key: str, x: np.ndarray) -> np.ndarray:
    """Move the time axis to the front using the Zarr dimension names (archive layouts differ per array)."""
    dims = [str(d) for d in (getattr(eq[key].metadata, "dimension_names", None) or [])]
    return np.moveaxis(x, dims.index("time"), 0) if "time" in dims else x


def _topup(shot_id: int, f: Path) -> tuple[int, int, str]:
    """Add the TOPUP arrays to a shot file written by an earlier version, on the slices it already holds."""
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    if all(k in d for k in TOPUP):
        return shot_id, -1, "cached"
    try:
        eq = mast._open(shot_id)["equilibrium"]
        t = np.asarray(eq["time"][:], dtype=np.float32)
        idx = np.flatnonzero(np.isin(t, d["time"]))                               # same float32 time stamps as stored
        if idx.size != d["time"].size:
            return shot_id, 0, "topup: time base mismatch"
        for k in TOPUP:
            d[k] = _time_first(eq, k, np.asarray(eq[k][:], dtype=np.float32))[idx]
    except Exception as ex:
        return shot_id, 0, f"topup {type(ex).__name__}: {str(ex)[:80]}"
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.rename(f)
    return shot_id, -1, "topped up"


def fetch_one(shot_id: int, deadline: float) -> tuple[int, int, str]:
    f = OUT / f"{shot_id}.npz"
    if f.exists():
        return _topup(shot_id, f)
    if time.time() > deadline:
        return shot_id, 0, "deadline"
    try:
        eq = mast._open(shot_id)["equilibrium"]
        keys = PER_TIME + SENSORS + GRID + ["psi"]
        futs = {k: _arrays.submit(lambda k=k: np.asarray(eq[k][:], dtype=np.float32)) for k in keys}
        raw = {k: _time_first(eq, k, fu.result()) for k, fu in futs.items()}      # time first from here on
    except Exception as ex:  # not every table shot exists in level 2; no retry, move on
        return shot_id, 0, f"{type(ex).__name__}: {str(ex)[:80]}"

    dims = [str(d) for d in eq["psi"].metadata.dimension_names]                   # ('z','major_radius','time')
    psi = raw["psi"]                                                              # (time, a, b)
    if [d for d in dims if d != "time"][0] != "z":
        psi = psi.transpose(0, 2, 1)                                              # always store (time, Z, R)
    keep = np.isfinite(raw["q95"]) & (np.abs(raw["ip"]) > IP_MIN_A) & np.isfinite(psi).all(axis=(1, 2))
    if keep.sum() == 0:
        return shot_id, 0, "no plasma slices"
    out = {k: raw[k][keep] for k in PER_TIME + SENSORS}
    out.update(psi=psi[keep], major_radius=raw["major_radius"], z=raw["z"])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.rename(f)
    return shot_id, int(keep.sum()), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes shots[i::n]")
    a = ap.parse_args()

    import zarr
    zarr.config.set({"async.concurrency": 4})       # psi is 8 chunks; keep per-array fan-out small
    i_shard, n_shard = map(int, a.shard.split("/"))
    shots = pick_shots(a.n)[i_shard::n_shard]
    t0 = time.time()
    deadline = t0 + 60 * a.minutes
    n_ok = n_slices = n_fail = 0
    with ThreadPoolExecutor(SHOT_WORKERS) as pool:
        futs = [pool.submit(fetch_one, int(s), deadline) for s in shots]
        for i, fu in enumerate(as_completed(futs), 1):
            sid, n, msg = fu.result()
            if n != 0:
                n_ok += 1
                n_slices += max(n, 0)
            else:
                n_fail += 1
                if msg != "deadline":
                    print(f"  skip {sid}: {msg}", flush=True)
            if i % 20 == 0 or i == len(futs):
                print(f"[{time.time() - t0:6.0f}s] {i}/{len(futs)} done, {n_ok} shots ok, {n_fail} skipped, "
                      f"{n_slices} new slices", flush=True)
    print(f"download wall time {time.time() - t0:.0f} s; {len(list(OUT.glob('*.npz')))} shot files in {OUT}")


if __name__ == "__main__":
    main()
