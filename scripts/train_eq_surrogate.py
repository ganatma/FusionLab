"""Train and score the EFIT surrogate (fusionlab/eq_surrogate.py) on real MAST shots.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_eq_surrogate.py

Reads data/eq/shots/*.npz (scripts/fetch_eq_dataset.py), splits BY SHOT NUMBER BLOCK (never by time slice),
and scores three models on the held-out test shots:

    mean     the training-set mean psi
    ridge    closed-form ridge regression, standardised magnetics -> psi on the full 65x65 grid
    network  PhysicsNeMo FullyConnected -> k PCA coefficients of psi (candidates: direct, and ridge + learned
             residual); the candidate is chosen on the VALIDATION shots, the test shots are only scored

Writes models/eq_surrogate.pt and models/eq_surrogate_metrics.json (every number in it is measured here).
EFIT is itself a fit to these magnetics, so a small error means "reproduces EFIT", nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fusionlab import eq_surrogate as eqs  # noqa: E402

SHOTS = ROOT / "data" / "eq" / "shots"
SEED = 0
BLOCK = 100          # consecutive shot numbers share a split: MAST sessions repeat near-identical discharges
LIVE_MIN = 0.90      # a sensor is an input if it is alive in at least this fraction of training slices
K_PCA = 64
RIDGE_LAMBDAS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


# ---------------------------------------------------------------- data
def load_dataset(ip_source: str = "ip_measured") -> dict:
    """All shot files concatenated along time. One loop over files (I/O), none over slices."""
    files = sorted(p for p in SHOTS.glob("*.npz") if ".tmp" not in p.name)
    cols = {k: [] for k in ("x", "w", "psi", "psi_axis", "psi_boundary", "axis_r", "axis_z", "ip", "box", "shot")}
    grid = None
    for p in files:
        with np.load(p) as z:
            g = (z["major_radius"], z["z"])
            if grid is None:
                grid = g
            if z["psi"].shape[1:] != (eqs.NZ, eqs.NR) or not all(np.allclose(a, b) for a, b in zip(g, grid)):
                print(f"  skip {p.name}: different grid")
                continue
            n = z["ip"].size
            cols["x"].append(np.concatenate([z["b_field_pol_probe_measured"], z["flux_loop_measured"], z["pf_current"], z[ip_source][:, None]], 1))
            cols["w"].append(np.concatenate([z["b_field_pol_probe_weight"], z["flux_loop_weight"], z["pf_current_weight"], np.ones((n, 1), np.float32)], 1))
            cols["psi"].append(z["psi"].reshape(n, -1))
            for k, src in (("psi_axis", "psi_axis"), ("psi_boundary", "psi_boundary"), ("axis_r", "magnetic_axis_r"), ("axis_z", "magnetic_axis_z"), ("ip", "ip")):
                cols[k].append(z[src])
            with warnings.catch_warnings():                       # all-NaN boundary on a slice -> NaN box, dropped later
                warnings.simplefilter("ignore")
                lr, lz = z["lcfs_r"].reshape(n, -1), z["lcfs_z"].reshape(n, -1)
                cols["box"].append(np.stack([np.nanmin(lr, 1), np.nanmax(lr, 1), np.nanmin(lz, 1), np.nanmax(lz, 1)], 1))
            cols["shot"].append(np.full(n, int(p.stem)))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    ok = np.isfinite(d["x"][:, -1]) & (d["x"][:, -1] != 0)                 # the measured plasma current must be there
    d = {k: v[ok] for k, v in d.items()}
    d["n_dropped_no_ip_measured"] = int((~ok).sum())
    d["R_m"], d["Z_m"] = grid[0].astype(np.float32), grid[1].astype(np.float32)
    return d


def split_by_block(shot: np.ndarray) -> dict:
    """80/10/10 over blocks of BLOCK consecutive shot numbers, fixed seed. Returns boolean masks over slices."""
    blocks = np.unique(shot // BLOCK)
    rng = np.random.default_rng(SEED)
    blocks = blocks[rng.permutation(blocks.size)]
    n_val = n_test = max(1, round(0.1 * blocks.size))
    parts = {"test": blocks[:n_test], "val": blocks[n_test:n_test + n_val], "train": blocks[n_test + n_val:]}
    return {k: np.isin(shot // BLOCK, b) for k, b in parts.items()}


def gram(A: torch.Tensor, B: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """A^T B in float64, accumulated over row chunks so the float64 copies never exist in full on the GPU."""
    out = torch.zeros(A.shape[1], B.shape[1], dtype=torch.float64, device=A.device)
    for i in range(0, A.shape[0], chunk):
        out += A[i:i + chunk].double().T @ B[i:i + chunk].double()
    return out


# ---------------------------------------------------------------- metrics
def pct(v: torch.Tensor) -> dict:
    v = v[torch.isfinite(v)].double()
    return {"median": float(v.median()), "p95": float(v.quantile(0.95)), "mean": float(v.mean())}


def score(psi_hat: torch.Tensor, t: dict, orient: float) -> dict:
    """psi_hat, t['psi']: (n, 4225) [Wb/rad]. All vectorised over slices on the GPU."""
    psi = t["psi"]
    out = {"rel_l2": pct((psi_hat - psi).norm(dim=1) / psi.norm(dim=1))}
    # B = grad(psi) x grad(phi) / R does not see a uniform offset of psi, so also score the maps with their means removed.
    e, p0 = psi_hat - psi, psi - psi.mean(1, keepdim=True)
    out["rel_l2_offset_removed"] = pct((e - e.mean(1, keepdim=True)).norm(dim=1) / p0.norm(dim=1))
    out["uniform_offset_share_of_squared_error"] = float((e.mean(1) ** 2).sum() * e.shape[1] / (e ** 2).sum())
    # Normalised flux with EFIT's own axis/boundary values; "inside the plasma" = psi_N <= 1 inside the LCFS bounding box
    # (the box removes the private-flux and coil regions where psi_N also drops below 1).
    den = (t["psi_boundary"] - t["psi_axis"])[:, None]
    pn, pn_hat = (psi - t["psi_axis"][:, None]) / den, (psi_hat - t["psi_axis"][:, None]) / den
    R, Z = t["R_m"].repeat(eqs.NZ), t["Z_m"].repeat_interleave(eqs.NR)           # flattened (Z, R) grid
    b = t["box"]
    inside = (pn <= 1) & (R >= b[:, 0:1]) & (R <= b[:, 1:2]) & (Z >= b[:, 2:3]) & (Z <= b[:, 3:4])
    se, cnt = ((pn_hat - pn) ** 2 * inside).sum(1), inside.sum(1)
    ok = (cnt > 0) & torch.isfinite(se)
    out["psi_n_rmse_in_plasma"] = {"pooled": float((se[ok].sum() / cnt[ok].sum()).sqrt()), "per_slice": pct((se[ok] / cnt[ok]).sqrt()),
                                   "n_slices": int(ok.sum())}
    sign = orient * torch.sign(t["ip"])                                          # t["ip"] is the measured current (an input)
    r, z = eqs.magnetic_axis(psi_hat.reshape(-1, eqs.NZ, eqs.NR), sign, t["R_m"], t["Z_m"])
    out["axis_error_cm"] = pct(100 * torch.hypot(r - t["axis_r"], z - t["axis_z"]))
    return out


# ---------------------------------------------------------------- training
def train_net(cfg: dict, xs: dict, target: dict, out_std: torch.Tensor, max_epochs: int, dev) -> tuple:
    """AdamW + cosine schedule, loss = psi-space MSE (coefficient errors weighted by their variance), early stopping on val."""
    torch.manual_seed(SEED)
    net = eqs.build_net(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = xs["train"].shape[0], cfg["batch_size"]
    steps = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * steps)
    w = out_std ** 2 / (out_std ** 2).sum()
    y = {k: v / out_std for k, v in target.items()}
    best, best_state, best_epoch, t0 = float("inf"), None, 0, time.time()
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(steps):                                              # loop over minibatches, not samples
            idx = perm[i * bs:(i + 1) * bs]
            xb = xs["train"][idx] + cfg["input_noise"] * torch.randn((idx.numel(), xs["train"].shape[1]), device=dev)
            loss = (((net(xb) - y["train"][idx]) ** 2) * w).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            val = float((((net(xs["val"]) - y["val"]) ** 2) * w).sum(1).mean())
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if epoch % 25 == 0:
            print(f"    epoch {epoch:4d}  train {loss.item():.3e}  val {val:.3e}  best {best:.3e} @ {best_epoch}", flush=True)
        if epoch - best_epoch >= cfg["patience"]:
            break
    net.load_state_dict(best_state)
    return net.eval(), {"epochs_run": epoch, "best_epoch": best_epoch, "train_time_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60, help="length of the cosine schedule; 60 had the lowest VALIDATION error "
                    "of 30/45/60/120/400 (sweep quoted in the metrics file). The spread between them is ~0.1-0.2 % absolute.")
    ap.add_argument("--layer-size", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--input-noise", type=float, default=0.0, help="std of Gaussian noise on the standardised inputs (training only)")
    ap.add_argument("--min-efit-weight", type=float, default=0.0,
                    help="keep a probe/flux loop only if EFIT's mean weight for it on the training slices is at least this")
    ap.add_argument("--only", default="", help="comma-separated candidate names (default: all)")
    ap.add_argument("--ip-source", default="ip_measured", choices=("ip_measured", "ip"),
                    help="'ip' feeds EFIT's own fitted current: a leak, for the ablation only")
    ap.add_argument("--tag", default="", help="ablation run: write to data/eq/runs/<tag>.* instead of models/")
    a = ap.parse_args()
    if a.tag:
        (ROOT / "data" / "eq" / "runs").mkdir(parents=True, exist_ok=True)
        eqs.MODEL_FILE, eqs.METRICS_FILE = (ROOT / "data" / "eq" / "runs" / f"{a.tag}{ext}" for ext in (".pt", ".json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=dev)

    d = load_dataset(a.ip_source)
    masks = split_by_block(d["shot"])
    counts = {k: {"n_shots": int(np.unique(d["shot"][m]).size), "n_slices": int(m.sum()),
                  "n_shot_number_blocks": int(np.unique(d["shot"][m] // BLOCK).size)} for k, m in masks.items()}
    counts["n_slices_dropped_without_ip_measured"] = d.pop("n_dropped_no_ip_measured")
    d["ip"] = d["x"][:, -1]                                                       # from here on "ip" is the measured current
    print("split", counts)

    # ---- sensors: alive = finite, and non-zero for probes/loops (the archive writes 0.0 where EFIT's weight is 0)
    n_mag = eqs.N_PROBE + eqs.N_LOOP
    zero_missing = np.arange(eqs.N_RAW) < n_mag
    xtr = d["x"][masks["train"]]
    miss = ~np.isfinite(xtr) | ((xtr == 0) & zero_missing)
    live = 1 - miss.mean(0)
    xm = np.ma.masked_array(np.nan_to_num(xtr), miss)
    x_mean, x_std = xm.mean(0).filled(0).astype(np.float32), xm.std(0).filled(0).astype(np.float32)
    w_mean = np.nan_to_num(d["w"][masks["train"]]).mean(0)
    used = (w_mean >= a.min_efit_weight) | ~zero_missing                         # the weight rule applies to probes and loops only
    keep = np.flatnonzero((live >= LIVE_MIN) & (x_std > 0) & used)
    grp = lambda lo, hi: int(((keep >= lo) & (keep < hi)).sum())
    sensors = {
        "rule": f"input = sensor finite and (for probes/flux loops) non-zero in >= {LIVE_MIN:.0%} of training slices, with non-zero spread; "
                "remaining dropouts (NaN, or exactly 0 on a probe/loop) are imputed with the training mean; EFIT weights are never network inputs"
                + (f"; probes/loops also need a mean EFIT weight >= {a.min_efit_weight} on the training slices" if a.min_efit_weight > 0 else ""),
        "n_inputs": int(keep.size), "probes_kept": grp(0, eqs.N_PROBE), "flux_loops_kept": grp(eqs.N_PROBE, n_mag),
        "pf_currents_kept": grp(n_mag, n_mag + eqs.N_PF), "ip_kept": grp(n_mag + eqs.N_PF, eqs.N_RAW),
        "kept_inputs_with_mean_efit_weight_below_0.5": int((w_mean[keep] < 0.5).sum()),
        "mean_efit_weight_of_all_flux_loops_train": float(w_mean[eqs.N_PROBE:n_mag].mean()),
        "imputed_fraction_of_kept_train_values": float(miss[:, keep].mean()),
    }
    print("sensors", {k: v for k, v in sensors.items() if k != "rule"})

    model_bufs = {"keep": torch.as_tensor(keep), "zero_is_missing": torch.as_tensor(zero_missing[keep]),
                  "x_mean": torch.as_tensor(x_mean[keep]), "x_std": torch.as_tensor(x_std[keep])}

    def feats(m):
        x = T(d["x"][m][:, keep])
        x = torch.where(eqs.missing(x, model_bufs["zero_is_missing"].to(dev)), model_bufs["x_mean"].to(dev), x)
        return (x - model_bufs["x_mean"].to(dev)) / model_bufs["x_std"].to(dev)

    xs = {k: feats(m) for k, m in masks.items()}
    psi = {k: T(d["psi"][m]) for k, m in masks.items()}
    R_m, Z_m = T(d["R_m"]), T(d["Z_m"])
    tt = {k: {"psi": psi[k], "R_m": R_m, "Z_m": Z_m, "box": T(d["box"][m]),
              **{q: T(d[q][m]) for q in ("psi_axis", "psi_boundary", "axis_r", "axis_z", "ip")}} for k, m in masks.items()}

    # Sign convention, from the training shots: is the axis a maximum or a minimum of psi for positive ip?
    s = np.sign((d["psi_axis"] - d["psi_boundary"]) * np.sign(d["ip"]))[masks["train"]]
    orient = float(np.sign(np.median(s)))
    print(f"axis is a {'max' if orient > 0 else 'min'} of psi*sign(ip) in {np.mean(s == orient):.4f} of training slices")

    # ---- PCA of psi on the training shots (eigh of the 4225x4225 covariance, float64)
    psi_mean = psi["train"].mean(0)
    psi_scale = (psi["train"] - psi_mean).std()
    Y = {k: (v - psi_mean) / psi_scale for k, v in psi.items()}
    cov = gram(Y["train"], Y["train"]) / Y["train"].shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    del cov
    basis = evecs[:, -K_PCA:].flip(1).T.float().contiguous()                     # (k, 4225), orthonormal rows
    explained = float(evals[-K_PCA:].sum() / evals.sum())
    coef = {k: v @ basis.T for k, v in Y.items()}
    decode = lambda c: psi_mean + psi_scale * (c @ basis)
    print(f"PCA k={K_PCA}: explained variance {explained:.6f}")

    # ---- baselines
    results = {"mean": score(psi_mean.expand_as(psi["test"]), tt["test"], orient)}
    results["efit_psi_axis_finder_floor_cm"] = pct(100 * torch.hypot(*[p - q for p, q in zip(
        eqs.magnetic_axis(psi["test"].reshape(-1, eqs.NZ, eqs.NR), orient * torch.sign(tt["test"]["ip"]), R_m, Z_m),
        (tt["test"]["axis_r"], tt["test"]["axis_z"]))]))
    results[f"pca_truncation_floor_k{K_PCA}"] = score(decode(coef["test"]), tt["test"], orient)

    X1 = {k: torch.cat([v, torch.ones_like(v[:, :1])], 1) for k, v in xs.items()}
    XtX, XtY = gram(X1["train"], X1["train"]), gram(X1["train"], Y["train"])
    n_tr, eye = X1["train"].shape[0], torch.eye(X1["train"].shape[1], device=dev, dtype=torch.float64)
    eye[-1, -1] = 0                                                              # the bias is not penalised
    ridge_val = {}
    for lam in RIDGE_LAMBDAS:
        W = torch.linalg.solve(XtX + lam * n_tr * eye, XtY)
        ridge_val[lam] = float(((X1["val"].float() @ W.float() - Y["val"]) ** 2).mean())
    lam = min(ridge_val, key=ridge_val.get)
    W_ridge = torch.linalg.solve(XtX + lam * n_tr * eye, XtY).float()            # (d+1, 4225)
    ridge_psi = lambda k: psi_mean + psi_scale * (X1[k].float() @ W_ridge)
    results["ridge"] = score(ridge_psi("test"), tt["test"], orient)
    results["ridge"]["lambda"] = lam
    val_rel = lambda p: float(((p - psi["val"]).norm(dim=1) / psi["val"].norm(dim=1)).median())
    print("ridge lambda", lam, "test", results["ridge"]["rel_l2"], "val median rel L2", val_rel(ridge_psi("val")))
    lin_W = (W_ridge @ basis.T).contiguous()                                     # the same ridge fit in PCA space

    # ---- PhysicsNeMo candidates, chosen on the validation shots
    base = {"in_features": int(keep.size), "k": K_PCA, "layer_size": a.layer_size, "num_layers": a.num_layers,
            "activation_fn": "silu", "skip_connections": True, "lr": a.lr, "weight_decay": a.weight_decay,
            "batch_size": a.batch_size, "patience": 60, "input_noise": a.input_noise}
    cands = {"mlp_direct": {**base, "residual": False}, "ridge_plus_mlp_residual": {**base, "residual": True}}
    cands = {k: v for k, v in cands.items() if not a.only or k in a.only.split(",")}
    trained = {}
    for name, cfg in cands.items():
        print(f"  training {name}", flush=True)
        lin = {k: (X1[k].float() @ lin_W if cfg["residual"] else torch.zeros_like(coef[k])) for k in coef}
        target = {k: coef[k] - lin[k] for k in coef}
        out_std = target["train"].std(0)
        net, info = train_net(cfg, xs, target, out_std, a.epochs, dev)
        with torch.no_grad():
            pred = {k: decode(net(xs[k]) * out_std + lin[k]) for k in ("val", "test")}
        info.update(val_rel_l2_median=val_rel(pred["val"]), test=score(pred["test"], tt["test"], orient),
                    n_parameters=sum(p.numel() for p in net.parameters()))
        trained[name] = (cfg, net, out_std, info)
        print(f"    {name}: val median rel L2 {info['val_rel_l2_median']:.5f}  test {info['test']['rel_l2']}", flush=True)
    chosen = min(trained, key=lambda k: trained[k][3]["val_rel_l2_median"])
    cfg, net, out_std, info = trained[chosen]
    results["physicsnemo"] = info["test"]

    # ---- save the model, then time the saved model through the public entry point
    ck = {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "config": cfg, **model_bufs,
          "out_std": out_std.cpu(), "basis": basis.cpu(), "psi_mean": psi_mean.cpu(), "psi_scale": psi_scale.cpu(),
          "lin_W": lin_W.cpu(), "R_m": R_m.cpu(), "Z_m": Z_m.cpu()}
    eqs.MODELS.mkdir(exist_ok=True)
    torch.save(ck, eqs.MODEL_FILE)
    eqs.load.cache_clear()
    model = eqs.load(str(dev))
    raw_test = T(d["x"][masks["test"]])
    with torch.no_grad():
        roundtrip = float((model(raw_test).reshape(-1, eqs.NZ * eqs.NR) - decode(net(xs["test"]) * out_std + (X1["test"].float() @ lin_W if cfg["residual"] else 0))).abs().max())
        xb = raw_test[torch.randint(0, raw_test.shape[0], (4096,), device=dev)]

        def clock(x, reps):
            for _ in range(5):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / reps

        t_batch, t_one = clock(xb, 50), clock(xb[:1], 500)
    throughput = {"batch_size": 4096, "slices_per_s": round(4096 / t_batch), "batch_ms": round(1e3 * t_batch, 3),
                  "single_slice_ms": round(1e3 * t_one, 3),
                  "note": "GPU-resident float32 inputs -> psi on the GPU, torch.cuda.synchronize around the timed loop, eager mode. "
                          "No comparison with EFIT run time: EFIT was not run here."}
    print("saved-model round trip max |dpsi|", roundtrip, "throughput", throughput)

    from fusionlab import mast
    db = mast.load_db()                                                          # campaign of every shot actually used
    shots_used = np.unique(d["shot"])
    camp, n_camp = np.unique(db["campaign"][np.isin(db["shot_id"], shots_used)], return_counts=True)
    campaigns = {f"M{c}": int(n) for c, n in zip(camp, n_camp)}
    beats = results["physicsnemo"]["rel_l2"]["median"] < results["ridge"]["rel_l2"]["median"]
    metrics = {
        "what": "Surrogate of EFIT's equilibrium reconstruction on MAST: magnetic measurements -> psi(R,Z) on EFIT's 65x65 grid. "
                "Target and inputs are EFIT's own stored arrays (FAIR-MAST level 2, group 'equilibrium'). EFIT is a fit to these "
                "same magnetics, so low error means the surrogate reproduces EFIT; it is not validated against anything else and is not predictive.",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "data": {"source": "UKAEA FAIR-MAST level 2 (CC BY-SA 4.0), s3://mast/level2/shots/{id}.zarr", "campaigns": campaigns,
                 "slice_filter": "finite q95, |EFIT ip| > 1e5 A, psi finite on the whole grid, finite measured ip",
                 "plasma_current_input": {"ip_measured": "ip_measured (the measurement), not EFIT's fitted ip",
                                          "ip": "EFIT's fitted ip (ABLATION ONLY: leaks an EFIT output into the inputs)"}[a.ip_source],
                 "split": f"by blocks of {BLOCK} consecutive shot numbers (never by time slice), 80/10/10 of blocks, seed {SEED}",
                 **counts, "psi_units": "Wb/rad (as archived)",
                 "grid": f"{eqs.NZ} x {eqs.NR}, axes (Z, R); R {d['R_m'][0]:.3f} to {d['R_m'][-1]:.3f} m, Z {d['Z_m'][0]:.3f} to {d['Z_m'][-1]:.3f} m"},
        "inputs": sensors, "input_names": [eqs.INPUT_NAMES[i] for i in keep],
        "pca": {"k": K_PCA, "explained_variance_train": explained},
        "architecture": {"chosen": chosen, "class": "physicsnemo.models.mlp.FullyConnected", **{k: cfg[k] for k in
                         ("in_features", "layer_size", "num_layers", "activation_fn", "skip_connections", "k", "residual", "input_noise")},
                         "decoder": "linear PCA decoder (fixed)", "n_parameters": info["n_parameters"],
                         "optimiser": f"AdamW lr {cfg['lr']} wd {cfg['weight_decay']}, cosine schedule, batch {cfg['batch_size']}, "
                                      f"psi-space MSE, early stopping on val (patience {cfg['patience']})"},
        "training": {"max_epochs": a.epochs, **{k: info[k] for k in ("epochs_run", "best_epoch", "train_time_s")}},
        "candidates": {k: {"val_rel_l2_median": v[3]["val_rel_l2_median"], "test_rel_l2": v[3]["test"]["rel_l2"],
                           "train_time_s": v[3]["train_time_s"], "best_epoch": v[3]["best_epoch"]} for k, v in trained.items()},
        "selection": "candidate with the lowest validation median relative L2; test shots were never used to choose",
        "metrics_definition": {
            "rel_l2": "per test slice ||psi_hat - psi_efit||_2 / ||psi_efit||_2 over the 65x65 grid",
            "rel_l2_offset_removed": "the same after subtracting each map's grid mean from error and reference "
                                     "(the field depends on grad psi only, so a uniform offset of psi has no effect on B)",
            "psi_n_rmse_in_plasma": "psi_N = (psi - psi_axis)/(psi_boundary - psi_axis) with EFIT's psi_axis/psi_boundary for both maps; "
                                    "grid points with EFIT psi_N <= 1 inside the bounding box of EFIT's LCFS",
            "axis_error_cm": f"distance between EFIT magnetic_axis_r/z and the extremum of predicted psi (max of psi*sign(ip)*{orient:+.0f}) inside "
                             f"R {eqs.AXIS_BOX_R_M} m, Z {eqs.AXIS_BOX_Z_M} m, parabolic sub-grid refinement; grid spacing is 3.0 cm in R, 6.25 cm in Z",
            "efit_psi_axis_finder_floor_cm": "the same axis finder applied to EFIT's own psi: the error of the finder itself"},
        "caveats": [
            "Surrogate of EFIT's reconstruction: the target is EFIT output, so this measures agreement with EFIT, not with the plasma.",
            "Held-out shots come from the same two campaigns (M8, M9) and the same machine; nothing here says how it does elsewhere.",
            "The test set is a small number of shot-number blocks (see n_shot_number_blocks), so the numbers move with the split seed.",
            "EFIT's own fitted plasma current is NOT an input. The ablation that feeds it shows a much lower error: most of the "
            "remaining error is tied to how EFIT's fit departs from the measured current, which these inputs do not determine.",
            "No run-time comparison with EFIT is made: EFIT was not run here."],
        "test_metrics": results,
        "beats_linear": bool(beats),
        "beats_linear_basis": "median relative L2 of psi on the test shots, PhysicsNeMo model vs ridge regression",
        "throughput": throughput,
        "test_shot_ids": np.unique(d["shot"][masks["test"]]).tolist(),
    }
    if not a.tag:   # ablations (tagged runs of this script on the same files) are quoted next to the headline numbers
        for p in sorted((ROOT / "data" / "eq" / "runs").glob("ablation_*.json")):
            m = json.loads(p.read_text())
            metrics.setdefault("ablations", {})[p.stem] = {
                "plasma_current_input": m["data"]["plasma_current_input"], "inputs": {k: v for k, v in m["inputs"].items() if k != "rule"},
                "same_test_shots_as_headline": m["test_shot_ids"] == metrics["test_shot_ids"], "date": m["date"],
                "physicsnemo_test": {k: m["test_metrics"]["physicsnemo"][k] for k in ("rel_l2", "rel_l2_offset_removed", "axis_error_cm")},
                "ridge_test_rel_l2": m["test_metrics"]["ridge"]["rel_l2"]}
        for p in sorted((ROOT / "data" / "eq" / "runs").glob("e_ep*.json")):      # cosine-length sweep, judged on validation only
            m = json.loads(p.read_text())
            c = m["candidates"]["mlp_direct"]
            metrics.setdefault("schedule_sweep_mlp_direct", {})[p.stem] = {
                "epochs_run": m["training"]["epochs_run"], "val_rel_l2_median": c["val_rel_l2_median"],
                "test_rel_l2_median": c["test_rel_l2"]["median"], "same_test_shots_as_headline": m["test_shot_ids"] == metrics["test_shot_ids"]}
    eqs.METRICS_FILE.write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: metrics[k] for k in ("test_metrics", "beats_linear", "throughput")}, indent=1))


if __name__ == "__main__":
    main()
