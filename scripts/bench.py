"""Timings for the README. Every number printed here is measured on this machine; none are estimates.

    make bench

The measurement lives in fusionlab/bench.py so the worker's `benchmark` task returns the same rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusionlab.bench import timings  # noqa: E402


def main():
    rows = timings(profile="full")["rows"]
    w = max(len(r["label"]) for r in rows)
    print(f"| {'Task':{w}} | Time | Throughput / note |\n|{'-' * (w + 2)}|---|---|")
    for r in rows:
        print(f"| {r['label']:{w}} | {r['time']} | {r['note']} |")


if __name__ == "__main__":
    main()
