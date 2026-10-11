"""Catalog mode_derived report — the docs' numbers for the derived L/H flag (FEAT-4, #38).

Reads the cached catalog (no network), applies the exact derivation rule FusionLab ships
(`lh_mode.derive_mode` with MAST_CONFIG's cut), and writes data/mode_flag_report.json for the
anchor test in tests/test_lh_mode.py. Run: `uv run python scripts/mode_flag_report.py`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fusionlab import mast  # noqa: E402
from fusionlab.data import lh_mode  # noqa: E402
from fusionlab.data.machine import MAST_CONFIG  # noqa: E402

SHOWCASE = [30420, 30166, 30192, 29823, 27257]   # the shipped showcase shots (scripts/fetch_mast.py)


def build_report() -> dict:
    db = mast.load_db()
    flag = db["mode_derived"]
    ids = db["shot_id"]
    anchors = {}
    for shot in SHOWCASE:
        i = int(np.flatnonzero(ids == shot)[0])
        anchors[str(shot)] = {
            "flag": "unknown" if np.isnan(flag[i]) else ("H" if flag[i] == 1 else "L"),
            "tau_E_s": float(db["tau_E_s"][i]),
            "enhancement_ratio": float(lh_mode.enhancement_ratio(db, isotope_md=MAST_CONFIG.isotope_md)[i]),
        }
    return {
        "rule": (f"mode_derived = H if tau_E_s > {MAST_CONFIG.lh_enhancement_min} * tau_89 (ITER89-P L-mode law) "
                 "at the peak-current snapshot; L if <=; NaN/unknown when the scalars are missing or invalid "
                 "(fusionlab/data/lh_mode.py, docs/PHYSICS.md)"),
        "enhancement_min": MAST_CONFIG.lh_enhancement_min,
        "counts": {"H": int(np.sum(flag == 1)), "L": int(np.sum(flag == 0)),
                   "unknown": int(np.sum(np.isnan(flag))), "rows": int(flag.size)},
        "anchors": anchors,
    }


if __name__ == "__main__":
    report = build_report()
    out = Path("data/mode_flag_report.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
