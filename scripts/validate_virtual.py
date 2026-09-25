"""Honesty gates for the virtual shot (fusionlab/virtual.py). Writes models/virtual_metrics.json and prints README tables.

    uv run python scripts/validate_virtual.py               # fetch up to 120 held-out shots (resumable), then score
    uv run python scripts/validate_virtual.py --n 40        # fewer shots
    uv run python scripts/validate_virtual.py --no-fetch    # score what is already under data/refly/

Gate 1, re-fly: shots from the sessions the tau_E correction never saw (its `held_out_blocks`) are re-flown blind with each
closure and compared with the measured W(t), on the flat top and, separately, in the 50 ms after beam-on (a quasi-static
flat top cannot see the integrator). The closure that wins here is the UI's default; nothing is chosen by decree.

Gate 2, matched pairs: a what-if is an *edit*, and holdout error on whole shots does not test edits. From the 6,353-shot
table, pairs of shots in the same session block with one input differing by > 20% and the others within 5% are the
closest thing the archive has to a controlled scan. For each actuator we compare the measured change in ln tau_E with
what each response law says, against the null "nothing changes". A law is *admissible* for an actuator when the pairs do
not contradict it (its error is within 5% of the null's, or better). An actuator gets a slider only with >= 30 pairs and
an admissible law, and the what-if never uses an exponent the pairs contradict (fusionlab.virtual.gate).

Data: UKAEA FAIR-MAST, CC BY-SA 4.0. Slim bundles (time traces only) are cached under data/refly/, which is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusionlab import mast, surrogate  # noqa: E402
from fusionlab.virtual import CLOSURES, baseline, refly_error, response_laws  # noqa: E402

REFLY = mast.DATA / "refly"
OUT = surrogate.MODELS / "virtual_metrics.json"
KEEP = ("t_s", "Ip_MA", "B_T", "n_e20", "R_m", "a_m", "V_m3", "W_MJ", "dWdt_MW", "P_ohm_MW", "P_nbi_MW", "beta_N", "q95", "kappa")
WORKERS = 4            # the archive is a public service: a few requests at a time
MIN_SLICES = 25
PAIR_BIG, PAIR_SMALL = 0.20, 0.05
MIN_PAIRS, NULL_SLACK = 30, 1.05
SEED = 0


# ---------------------------------------------------------------- gate 1: re-fly held-out shots
def pick_shots(n: int) -> list[int]:
    m = json.loads(surrogate.METRICS_FILE.read_text())
    c = surrogate.table()
    held = np.isin(c["shot_id"] // m["block_size"], m["held_out_blocks"])
    ids = np.asarray(c["shot_id"])[held]
    return sorted(int(i) for i in np.random.default_rng(SEED).permutation(ids)[:n])


def fetch(shot_id: int) -> str:
    f = REFLY / f"{shot_id}.npz"
    if f.exists():
        return "cached"
    try:
        s = mast.fetch_shot(shot_id)
    except Exception as e:   # level 2 does not hold every shot of the table
        return f"FAILED {type(e).__name__}: {str(e)[:80]}"
    np.savez_compressed(f, meta=json.dumps(s["meta"]), **{k: s[k] for k in KEEP})
    return f"{s['t_s'].size} slices"


def load(f: Path) -> dict:
    with np.load(f) as z:
        s = {k: z[k] for k in KEEP}
        s["meta"] = json.loads(str(z["meta"]))
    return s


def refly_table() -> dict:
    rows = {c: {"flat": [], "rise": [], "flat_beam": [], "flat_ohmic": []} for c in CLOSURES}
    used = []
    for f in sorted(REFLY.glob("*.npz")):
        s = load(f)
        if s["t_s"].size < MIN_SLICES or not np.isfinite(s["W_MJ"]).all():
            continue
        used.append(int(f.stem))
        beam = bool(s["meta"].get("P_nbi_max_MW"))
        ts = np.asarray(s["t_s"], float)
        for c in CLOSURES:
            b = baseline(s, c)
            idx = np.clip(np.rint((ts - b["p"]["t_s"][0]) / 1e-3).astype(int), 0, b["W_blind"].size - 1)
            e = refly_error(s, b["W_blind"][idx])
            if e["flat_top_median_abs_ln"] is not None:
                rows[c]["flat"].append(e["flat_top_median_abs_ln"])
                rows[c]["flat_beam" if beam else "flat_ohmic"].append(e["flat_top_median_abs_ln"])
            if e["beam_rise_median_abs_ln"] is not None:
                rows[c]["rise"].append(e["beam_rise_median_abs_ln"])
    med = lambda v: round(float(np.median(v)), 3) if v else None
    table = {c: {k: med(v) for k, v in r.items()} | {"n_flat": len(r["flat"]), "n_rise": len(r["rise"]),
                                                     "n_beam": len(r["flat_beam"]), "n_ohmic": len(r["flat_ohmic"])} for c, r in rows.items()}
    best = min((c for c in CLOSURES if table[c]["flat"] is not None), key=lambda c: table[c]["flat"], default=None)
    return {"metric": "median over shots of the shot's median |ln(W_refly / W_measured)|", "n_shots": len(used), "shots": used,
            "closures": table, "default_closure": best}


# ---------------------------------------------------------------- gate 2: matched pairs from the shot table
ACTUATORS = {"Ip": "Ip_MA", "B": "B_T", "n": "n_e20", "P": "P_loss_MW"}
OTHERS = ("Ip_MA", "B_T", "n_e20", "P_loss_MW", "R_m", "a_m", "kappa_a")


def matched_pairs() -> dict:
    c, m = surrogate.table(), json.loads(surrogate.METRICS_FILE.read_text())
    ln = {k: np.log(np.asarray(c[k], float)) for k in OTHERS} | {"tau": np.log(np.asarray(c["tau_E_s"], float))}
    block = np.asarray(c["shot_id"]) // m["block_size"]
    laws = {name: {"Ip_MA": e[0], "B_T": e[1], "n_e20": e[2], "P_loss_MW": -e[3]} for name, e in response_laws().items()}
    out = {}
    for name, key in ACTUATORS.items():
        d_tau, d_x = [], {k: [] for k in ACTUATORS.values()}
        for b in np.unique(block):                                  # pairs only inside a session block: same machine state
            i = np.flatnonzero(block == b)
            if i.size < 2:
                continue
            D = {k: ln[k][i][:, None] - ln[k][i][None, :] for k in OTHERS}
            ok = np.abs(D[key]) > np.log1p(PAIR_BIG)
            for k in OTHERS:
                if k != key:
                    ok &= np.abs(D[k]) < np.log1p(PAIR_SMALL)
            ok &= np.triu(np.ones_like(ok, bool), 1)                # each pair once
            if ok.any():
                d_tau.append((ln["tau"][i][:, None] - ln["tau"][i][None, :])[ok])
                for k in d_x:
                    d_x[k].append(D[k][ok])
        if not d_tau:
            out[name] = {"n_pairs": 0, "admissible": [], "best": None, "slider": False}
            continue
        d_tau = np.concatenate(d_tau)
        d_x = {k: np.concatenate(v) for k, v in d_x.items()}
        rmse = lambda pred: round(float(np.sqrt(np.mean((d_tau - pred) ** 2))), 3)
        slope = float(np.sum(d_tau * d_x[key]) / np.sum(d_x[key] ** 2))   # measured exponent in these pairs, through the origin
        row = {"n_pairs": int(d_tau.size), "measured_exponent": round(slope, 2), "rmse_null": rmse(0.0)}
        for law, e in laws.items():
            row[f"rmse_{law}"] = rmse(sum(e[k] * d_x[k] for k in d_x))
            row[f"exponent_{law}"] = round(e[key], 2)
        row["admissible"] = [law for law in laws if row[f"rmse_{law}"] <= NULL_SLACK * row["rmse_null"]]
        row["best"] = min(laws, key=lambda law: row[f"rmse_{law}"])
        row["slider"] = bool(d_tau.size >= MIN_PAIRS and row["admissible"])
        out[name] = row
    return {"rule": f"same block of {m['block_size']} shots, one input differs by > {PAIR_BIG:.0%}, the other six within {PAIR_SMALL:.0%}",
            "gate": f"slider needs >= {MIN_PAIRS} pairs and a law whose RMSE is <= {NULL_SLACK} x the no-change RMSE",
            "metric": "RMSE of the change in ln tau_E between the two shots of a pair", "actuators": out}


# ---------------------------------------------------------------- report
def report(m: dict) -> str:
    r, p = m["refly"], m["matched_pairs"]
    NAMES = {"learned": "IPB98(y,2) × PhysicsNeMo correction", "ipb98": "IPB98(y,2)", "iter89": "ITER89-P"}
    f = lambda v: "–" if v is None else f"{v:.3f}"
    lines = [f"Blind re-fly of {r['n_shots']} held-out shots, median |ln(W_refly / W_measured)| (0.10 ≈ 10%):", "",
             "| Closure | Flat top, all | beam-heated | ohmic | 50 ms after beam-on |", "|---|---|---|---|---|"]
    for c, row in r["closures"].items():
        lines.append(f"| {NAMES[c]}{' **(default)**' if c == r['default_closure'] else ''} | {f(row['flat'])} ({row['n_flat']}) | "
                     f"{f(row['flat_beam'])} ({row['n_beam']}) | {f(row['flat_ohmic'])} ({row['n_ohmic']}) | {f(row['rise'])} ({row['n_rise']}) |")
    lines += ["", f"Matched pairs ({p['rule']}), RMSE of Δln τ_E:", "",
              "| Edited input | Pairs | Measured exponent | No change | IPB98(y,2) | Valovič 2009 | Refit power law | Slider |", "|---|---|---|---|---|---|---|---|"]
    why = lambda row: "yes: " + ", ".join(row["admissible"]) if row["slider"] else ("no: too few pairs" if row["n_pairs"] < MIN_PAIRS else "no: every law contradicted")
    LABEL = {"Ip": "Plasma current", "B": "Toroidal field", "n": "Density", "P": "Loss power"}
    for a, row in p["actuators"].items():
        if not row["n_pairs"]:
            lines.append(f"| {LABEL[a]} | 0 | – | – | – | – | – | no: no pairs |")
            continue
        lines.append(f"| {LABEL[a]} | {row['n_pairs']} | {row['measured_exponent']} | {row['rmse_null']} | {row['rmse_ipb98']} ({row['exponent_ipb98']}) | "
                     f"{row['rmse_valovic']} ({row['exponent_valovic']}) | {row['rmse_refit']} ({row['exponent_refit']}) | {why(row)} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=120, help="held-out shots to fetch (already cached ones are free)")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    REFLY.mkdir(parents=True, exist_ok=True)
    if not args.no_fetch:
        ids, t0 = pick_shots(args.n), time.time()
        with ThreadPoolExecutor(WORKERS) as pool:
            for k, (i, msg) in enumerate(zip(ids, pool.map(fetch, ids)), 1):
                print(f"[{k}/{len(ids)}] {i}: {msg}  ({time.time() - t0:.0f} s)", flush=True)
    m = {"refly": refly_table(), "matched_pairs": matched_pairs(),
         "data": "UKAEA FAIR-MAST level 2, CC BY-SA 4.0; shots drawn from the tau_E correction's held_out_blocks (seed 0)"}
    OUT.write_text(json.dumps(m, indent=1) + "\n")
    print("\n" + report(m) + f"\n\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
