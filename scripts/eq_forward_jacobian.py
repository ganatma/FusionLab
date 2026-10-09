"""Intervention check for the forward equilibrium spike: does the coils-only model respond to a coil the way that coil's
field says it should?  Writes the result into models/eq_forward_spike_metrics.json ("jacobian_check") and prints a table.

    uv run python scripts/eq_forward_jacobian.py                 # data/eq/runs/forward.pt, held-out shots only
    uv run python scripts/eq_forward_jacobian.py --verify-map    # also recompute the column -> circuit map from the archive

Why: held-out error on whole shots says the model reproduces EFIT on shots like the training ones. It says nothing about an
*edit*, because the coil currents in real shots are set by feedback on the plasma: part of the coil <-> shape correlation
runs from plasma to coil. The one thing a coil current change must do, whatever the plasma does, is add that coil's own
vacuum field. So: perturb each circuit's EFIT columns by dI, take the model's finite-difference dpsi/dI on held-out slices,
and compare it with the circuit's vacuum Green's function sum_elements psi_filament(R, Z) outside the plasma. Sign, shape
(Pearson r) and magnitude (regression slope) are reported per circuit; the plasma's own response is in the model and not in
the vacuum field, so agreement is expected to be good near the coil and partial near the plasma.

Column map (EFIT `pf_current`, 101 columns; the first 21 are live): regressing each column on the 13 measured circuit
currents of pf_active/coil_current gives scale 1.0 for exactly one family per column on shots 30166, 30192 and 27257:
col 0 = solenoid; 1, 3 = P2 inner (upper/lower, in series, 1.5x the outer current); 2, 4 = P2 outer; 5, 6 = P3 (r = 0.73
only: its column carries more than the feed current); 7, 8 = P4; 9, 10 = P5; 11, 12 = P6 (small currents, r ~ 0.6-0.8).
Columns 13-20 are small, anti-correlated with P4/P5/solenoid and match no feed: passive structure, not operator knobs.
Geometry: pf_active winding-pack elements (one per turn: P4 has 23), in the order fusionlab.mast._pf_coils concatenates them.

Units of a column, settled two ways (the first run of this script had them wrong by the turn count): a P4 feed of 195 kA
can only be ampere-turns (as per-turn current over 23 turns it would put ~4 T of vertical field on a plasma that needs
~0.17 T), and the feeds match EFIT's columns at scale 1.0, so the P2-P6 columns are ampere-turns per coil, spread over
that coil's turns. The solenoid column (46 kA, 656 turns) is per-turn current: regressing EFIT's own psi outside the
plasma on the vacuum fields gives 0.37x the 656-turn field for it (not 1/656), and 1/(2N) per coil for P3, P4, P5,
exactly what ampere-turns over N turns predict. The vacuum reference here uses those conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusionlab import eq_surrogate as eqs  # noqa: E402
from fusionlab import mast  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FORWARD = ROOT / "data" / "eq" / "runs" / "forward.pt"
FORWARD_JSON = ROOT / "data" / "eq" / "runs" / "forward.json"
OUT = ROOT / "models" / "eq_forward_spike_metrics.json"
MU0 = 4e-7 * np.pi

# pf_active geometry stems in the order _pf_coils concatenates them (sorted), with element counts
STEMS = [("p2_inner_lower", 12), ("p2_inner_upper", 12), ("p2_outer_lower", 8), ("p2_outer_upper", 8), ("p3_lower", 8),
         ("p3_upper", 8), ("p4_lower", 23), ("p4_upper", 23), ("p5_lower", 23), ("p5_upper", 23), ("p6_lower", 4),
         ("p6_upper", 4), ("sol", 656)]
TURNS_DIVISOR = {stem: n for stem, n in STEMS} | {"sol": 1}   # column = ampere-turns per PF coil, per-turn amperes for the solenoid
CIRCUITS = {   # EFIT columns (0-based of 101) -> coils they drive, with how sure the map is
    "solenoid": {"cols": [0], "coils": ["sol"], "map": "scale 1.00 to SOL, r > 0.99"},
    "P2 inner": {"cols": [1, 3], "coils": ["p2_inner_upper", "p2_inner_lower"], "map": "scale 1.00 to P2IU/P2IL, r > 0.99"},
    "P2 outer": {"cols": [2, 4], "coils": ["p2_outer_upper", "p2_outer_lower"], "map": "scale 0.99 to P2OU/P2OL, r > 0.99"},
    "P3": {"cols": [5, 6], "coils": ["p3_upper", "p3_lower"], "map": "r = 0.73 to P3U/P3L only: weak"},
    "P4": {"cols": [7, 8], "coils": ["p4_upper", "p4_lower"], "map": "scale 1.00 to P4U/P4L, r > 0.99"},
    "P5": {"cols": [9, 10], "coils": ["p5_upper", "p5_lower"], "map": "scale 1.00 to P5U/P5L, r > 0.99"},
    "P6": {"cols": [11, 12], "coils": ["p6_upper", "p6_lower"], "map": "r = 0.6-0.8 to P6U/P6L: weak, 1-9 kA currents"},
}
MAIN = ("solenoid", "P2 inner", "P2 outer", "P4", "P5")   # circuits whose map is certain; the gate is judged on these
N_PF0 = eqs.N_PROBE + eqs.N_LOOP                          # first pf_current column in the raw input
D_I = 2000.0                                              # A, central difference (columns have std 20-60 kA; P6: 200 A)
EXCLUDE_M = 0.08                                          # no comparison within this distance of a perturbed element
PSI_N_OUT = 1.05
N_SHOTS, N_SLICES = 24, 8
PASS = {"r": 0.8, "slope": (0.5, 2.0)}


def coil_elements() -> dict:
    """(R, Z) of every winding-pack element per stem, from a cached replay bundle (no network)."""
    s = mast.load_shot(mast.cached_shots()[0])
    R, Z, out, k = s["coil_R"], s["coil_Z"], {}, 0
    assert R.size == sum(n for _, n in STEMS), R.size
    for stem, n in STEMS:
        out[stem] = (R[k:k + n].astype(float), Z[k:k + n].astype(float))
        k += n
    return out


def ellipke(m):
    """Complete elliptic integrals K(m), E(m) by the arithmetic-geometric mean (no scipy in the project)."""
    a, b, c = np.ones_like(m), np.sqrt(1 - m), np.sqrt(m)
    acc = 0.5 * c**2
    for n in range(1, 12):
        a, b, c = 0.5 * (a + b), np.sqrt(a * b), 0.5 * (a - b)
        acc = acc + 2 ** (n - 1) * c**2
    K = np.pi / (2 * a)
    return K, K * (1 - acc)


def green(R, Z, Rc, Zc, w=1.0):
    """Poloidal flux per radian [Wb/rad] at (R, Z) from w A in circular filaments at (Rc, Zc), summed over filaments."""
    R, Z = R[..., None], Z[..., None]
    k2 = 4 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 0, 1 - 1e-12)
    k = np.sqrt(k2)
    K, E = ellipke(k2)
    return (w * MU0 / (2 * np.pi) * np.sqrt(R * Rc) / k * ((2 - k2) * K - 2 * E)).sum(-1)


def load_forward(dev):
    ck = torch.load(FORWARD, map_location="cpu", weights_only=True)
    m = eqs.EqSurrogate(ck)
    m.net.load_state_dict(ck["state_dict"])
    return m.to(dev).eval()


def held_out_slices():
    ids = json.loads(FORWARD_JSON.read_text())["test_shot_ids"]
    rng = np.random.default_rng(0)
    for sid in sorted(int(i) for i in rng.permutation(ids)[:N_SHOTS]):
        f = ROOT / "data" / "eq" / "shots" / f"{sid}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            d = {k: z[k] for k in ("ip_measured", "pf_current", "psi", "psi_axis", "psi_boundary", "magnetic_axis_r", "magnetic_axis_z", "major_radius", "z")}
        ok = np.flatnonzero(np.isfinite(d["ip_measured"]) & (np.abs(d["ip_measured"]) > 3e5) & np.isfinite(d["pf_current"]).all(1)
                            & np.isfinite(d["psi"]).reshape(d["psi"].shape[0], -1).all(1))
        for i in ok[np.linspace(0, ok.size - 1, min(N_SLICES, ok.size)).astype(int)] if ok.size else []:
            yield sid, int(i), d


def verify_map(shot_id: int = 30166) -> dict:
    """Recompute the scale of each live EFIT column on each measured circuit current (needs the archive)."""
    g = mast._open(shot_id)
    pa, eq = g["pf_active"], g["equilibrium"]
    names = [str(s).replace(" FEED", "") for s in pa["current_channel"][:]]
    cc, tt, pf, te = pa["coil_current"][:], pa["time"][:], eq["pf_current"][:], eq["time"][:]
    C = np.stack([mast._on(te, tt, cc[k]) for k in range(len(names))])
    ok = np.isfinite(C).all(0) & np.isfinite(pf).all(1)
    pf, C = pf[ok], C[:, ok]
    out = {}
    for j in range(21):
        x = pf[:, j] - pf[:, j].mean()
        best = max(range(len(names)), key=lambda k: abs(np.corrcoef(x, C[k])[0, 1]) if C[k].std() > 0 else 0)
        c = C[best] - C[best].mean()
        out[j] = {"channel": names[best], "r": round(float(np.corrcoef(x, c)[0, 1]), 3), "scale": round(float(x @ c / (c @ c)), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-map", action="store_true")
    a = ap.parse_args()
    if a.verify_map:
        for j, v in verify_map().items():
            print(f"col {j:2d}: {v}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_forward(dev)
    R_m, Z_m = model.R_m.cpu().numpy(), model.Z_m.cpu().numpy()
    RR, ZZ = np.meshgrid(R_m, Z_m)                                                   # (Z, R), the model's output layout
    elements = coil_elements()
    G, dist = {}, {}
    for name, c in CIRCUITS.items():
        Rc = np.concatenate([elements[s][0] for s in c["coils"]])
        Zc = np.concatenate([elements[s][1] for s in c["coils"]])
        w = np.concatenate([np.full(elements[s][0].size, 1.0 / TURNS_DIVISOR[s]) for s in c["coils"]])
        G[name] = green(RR, ZZ, Rc, Zc, w)                                           # Wb/rad per unit of the column, on every coil of the circuit
        dist[name] = np.sqrt((RR[..., None] - Rc) ** 2 + (ZZ[..., None] - Zc) ** 2).min(-1)

    rows = {name: {"r_all": [], "r_near": [], "slope": [], "ratio": []} for name in CIRCUITS}
    n_slices, axis_err = 0, []
    for _sid, i, d in held_out_slices():
        x = np.zeros((1, eqs.N_RAW), np.float32)
        x[0, N_PF0:N_PF0 + eqs.N_PF] = np.nan_to_num(d["pf_current"][i])
        x[0, -1] = d["ip_measured"][i]
        psi_efit = d["psi"][i]
        psi_n = (psi_efit - d["psi_axis"][i]) / (d["psi_boundary"][i] - d["psi_axis"][i])
        # self-check of the (Z, R) layout: the file's own axis must sit where its psi extremum is
        sign = torch.as_tensor([np.sign(d["psi_axis"][i] - d["psi_boundary"][i])], dtype=torch.float32, device=dev)
        ar, az = eqs.magnetic_axis(torch.as_tensor(psi_efit[None], dtype=torch.float32, device=dev), sign, model.R_m, model.Z_m)
        axis_err.append(float(np.hypot(ar.item() - d["magnetic_axis_r"][i], az.item() - d["magnetic_axis_z"][i])))
        for name, c in CIRCUITS.items():
            dI = D_I if name != "P6" else 200.0
            xp, xm = x.copy(), x.copy()
            xp[0, N_PF0 + np.array(c["cols"])] += dI
            xm[0, N_PF0 + np.array(c["cols"])] -= dI
            with torch.no_grad():
                pp, pm = model(torch.as_tensor(np.concatenate([xp, xm]), device=dev)).cpu().numpy()
            dpsi = (pp - pm) / (2 * dI)                                                # Wb/rad per A, model
            mask = (psi_n > PSI_N_OUT) & (dist[name] > EXCLUDE_M) & np.isfinite(psi_n)
            if mask.sum() < 50:
                continue
            m, g = dpsi[mask], G[name][mask]
            near = dist[name][mask] <= np.median(dist[name][mask])
            rows[name]["r_all"].append(float(np.corrcoef(m, g)[0, 1]))
            rows[name]["r_near"].append(float(np.corrcoef(m[near], g[near])[0, 1]))
            rows[name]["slope"].append(float(m @ g / (g @ g)))
            rows[name]["ratio"].append(float(np.sqrt((m ** 2).mean() / (g ** 2).mean())))
        n_slices += 1
    assert n_slices > 20 and np.median(axis_err) < 0.05, (n_slices, np.median(axis_err))

    def q(v):
        return {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3), "p90": round(float(np.percentile(v, 90)), 3)}
    table = {}
    for name, r in rows.items():
        s = {k: q(v) for k, v in r.items()} | {"n": len(r["r_all"]), "map": CIRCUITS[name]["map"], "n_turns": int(sum(dict(STEMS)[s] for s in CIRCUITS[name]["coils"]))}
        s["sign"] = "same" if s["r_all"]["median"] > 0 else "opposite"
        s["passes"] = bool(abs(s["r_all"]["median"]) >= PASS["r"] and PASS["slope"][0] <= abs(s["slope"]["median"]) <= PASS["slope"][1])
        table[name] = s
    signs = {table[n]["sign"] for n in MAIN}
    verdict = {"consistent_sign_on_main_circuits": len(signs) == 1, "main_circuits_passing": [n for n in MAIN if table[n]["passes"]],
               "go": bool(len(signs) == 1 and all(table[n]["passes"] for n in MAIN))}
    result = {"what": "finite-difference dpsi/dI of the coils-only model vs the circuit's vacuum Green's function, outside the plasma "
                      f"(psi_N > {PSI_N_OUT}, > {EXCLUDE_M} m from the perturbed turns), on {n_slices} held-out slices of "
                      f"{len({s for s, _, _ in held_out_slices()})} shots; r = Pearson over those points, slope = <model, vacuum>/<vacuum, vacuum>, "
                      "near = the half of the points closest to the coil",
              "pass_rule": f"|median r| >= {PASS['r']} and {PASS['slope'][0]} <= |median slope| <= {PASS['slope'][1]} on every main circuit, one sign convention",
              "column_units": "P2-P6 columns are ampere-turns per coil (spread over its turns); the solenoid column is per-turn amperes (656 turns)",
              "expected_slope": "1 for the bare vacuum field; below 1 where the plasma and the vessel's induced currents respond against the coil "
                                "(EFIT's own psi outside the plasma follows the solenoid at 0.37x its vacuum field)",
              "main_circuits": list(MAIN), "dI_A": D_I, "circuits": table, "verdict": verdict,
              "layout_self_check_axis_err_m_median": round(float(np.median(axis_err)), 4)}
    m = json.loads(OUT.read_text()) if OUT.exists() else {}
    m["jacobian_check"] = result
    OUT.write_text(json.dumps(m, indent=1) + "\n")

    print(f"\n{'circuit':10s} {'turns':>5s} {'n':>3s} {'r all':>7s} {'r near':>7s} {'slope':>7s} {'rms ratio':>9s}  sign      pass  map")
    for name, s in table.items():
        print(f"{name:10s} {s['n_turns']:5d} {s['n']:3d} {s['r_all']['median']:7.3f} {s['r_near']['median']:7.3f} {s['slope']['median']:7.3f} "
              f"{s['ratio']['median']:9.3f}  {s['sign']:9s} {'yes' if s['passes'] else 'no ':4s}  {s['map']}")
    print("verdict:", verdict, "\nwrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
