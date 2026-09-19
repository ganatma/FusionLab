"""Cache real MAST shots (and optionally the shot table) from FAIR-MAST into data/.

    uv run python scripts/fetch_mast.py                 # the showcase shots
    uv run python scripts/fetch_mast.py 30166 30192     # any level-2 shot ids
    uv run python scripts/fetch_mast.py --db            # also refresh data/mast_db.npz

Data: UKAEA FAIR-MAST, CC BY-SA 4.0 (see data/README.md). The app reads only this cache.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusionlab import mast  # noqa: E402

# Chosen from the session leaders' logbook comments:
SHOWCASE = [
    30420,  # ohmic, 700 kA: "runs out of flux before reaches required density"
    30166,  # NBI, 900 kA: "Good shot, H-mode from 223 ms"
    30192,  # NBI, 920 kA: "plasma locks ... disrupts at 300ms"
    29823,  # NBI, 900 kA: "Disrupts at the end of the current ramp because of locked mode"
    27257,  # NBI, 810 kA, campaign M8: in the equilibrium surrogate's held-out test sessions
]


def _one(shot_id: int) -> str:
    t0 = time.time()
    try:
        s = mast.load_shot(shot_id, refresh=True)
    except Exception as e:  # a missing level-2 shot must not stop the others
        return f"{shot_id}: FAILED {type(e).__name__}: {str(e)[:120]}"
    size = (mast.SHOTS / f"{shot_id}.npz").stat().st_size / 1e6
    return (f"{shot_id}: {s['t_s'].size} slices, t = {s['t_s'][0]:.3f}-{s['t_s'][-1]:.3f} s, "
            f"Ip max {s['Ip_MA'].max():.2f} MA, {s['meta']['heating']}, {size:.2f} MB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--db" in args:
        t0 = time.time()
        db = mast.load_db(refresh=True)
        print(f"shot table: {db['shot_id'].size} rows, {mast.clean_db(db)['shot_id'].size} usable, {time.time() - t0:.0f} s")
    ids = [int(a) for a in args if a.isdigit()] or SHOWCASE
    with ThreadPoolExecutor(len(ids)) as pool:
        for line in pool.map(_one, ids):
            print(line)
