"""Learned correction to IPB98(y,2) on real MAST shots (NVIDIA PhysicsNeMo).

Grey-box model: the scaling law stays as the prior and a small PhysicsNeMo MLP learns
log(tau_E measured / tau_E IPB98) from the shot table (fusionlab.mast.load_db, one row per shot
at peak current). The model is never trusted blindly: every run writes holdout errors for

    ipb98      IPB98(y,2) as published
    ipb98_x_H  IPB98(y,2) times one fitted constant
    power_law  refit power law, least squares in log space (what a physicist would try first)
    power_law+ the same with the beam power fraction as an extra term (same inputs as the network)
    hybrid     IPB98(y,2) x exp(PhysicsNeMo residual)

on two splits, neither of which is a random split by shot. MAST sessions repeat near-identical
discharges, so a random split leaks: the network then looks ~40% better than the power law, and the
gain vanishes on unseen sessions. We split by blocks of 100 consecutive shot numbers (about one
session) and, as the hard test, train on campaigns M5-M8 and test on M9.
Errors are RMSE of ln(tau_E); 0.10 is roughly 10% scatter.

    uv run python -m fusionlab.surrogate        # trains, writes models/surrogate.pt + surrogate_metrics.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from fusionlab import mast
from fusionlab.physics import M_D, tau_coeff_H, volume

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE, METRICS_FILE = MODELS / "surrogate.pt", MODELS / "surrogate_metrics.json"

VARS = ("Ip_MA", "B_T", "n_e20", "P_loss_MW", "R_m", "a_m", "kappa_a")   # power-law variables (log)
IPB98_EXP = {"Ip_MA": 0.93, "B_T": 0.15, "n_e20": 0.41, "P_loss_MW": -0.69}
VALOVIC_EXP = {"Ip_MA": 0.59, "B_T": 1.4, "n_e20": 0.00, "P_loss_MW": -0.73}  # Valovič 2009 eq. (1), W -> tau = W / P
BLOCK = 100            # consecutive shot numbers that share a split (about one experimental session)
RESIDUAL_CLIP = 1.0   # |ln correction| allowed when applied outside the training distribution (shot ramps)
SEED = 0


# ---------------------------------------------------------------- features
def tau_ipb98(x: dict):
    """IPB98(y,2) tau_E [s] for deuterium, with kappa_a = V / (2 pi^2 R a^2). Vectorized over rows."""
    g = {"R": x["R_m"], "a": x["a_m"], "kappa": x["kappa_a"]}
    C, alpha = tau_coeff_H(g, x["Ip_MA"], x["B_T"], x["n_e20"], 1.0, M_D)
    return C * x["P_loss_MW"] ** -alpha


def features(x: dict) -> np.ndarray:
    """(n, 8): log of the seven power-law variables + beam fraction of the heating power."""
    logs = [np.log(np.maximum(x[k], 1e-6)) for k in VARS]
    return np.stack(logs + [x["f_nbi"]], axis=-1).astype(np.float32)


def table() -> dict:
    """Usable shot-table rows with the derived columns the models need."""
    c = mast.clean_db(mast.load_db())
    c["kappa_a"] = c["V_m3"] / volume(c["R_m"], c["a_m"], 1.0)
    c["f_nbi"] = c["P_nbi_MW"] / (c["P_ohm_MW"] + c["P_nbi_MW"])
    return c


def shot_features(shot: dict, P_loss) -> dict:
    """The same columns from a replay bundle (fusionlab.mast.load_shot), one row per time slice."""
    P_in = shot["P_ohm_MW"] + shot["P_nbi_MW"]
    return {"Ip_MA": shot["Ip_MA"], "B_T": shot["B_T"], "n_e20": shot["n_e20"], "P_loss_MW": P_loss,
            "R_m": shot["R_m"], "a_m": shot["a_m"], "kappa_a": shot["V_m3"] / volume(shot["R_m"], shot["a_m"], 1.0),
            "f_nbi": np.where(P_in > 0, shot["P_nbi_MW"] / np.maximum(P_in, 1e-9), 0.0)}


# ---------------------------------------------------------------- model
def _net(n_in: int):
    from physicsnemo.models.mlp import FullyConnected
    return FullyConnected(in_features=n_in, layer_size=32, out_features=1, num_layers=2, activation_fn="silu")


def _fit_net(X, y, Xv, yv, device, epochs=2000, noise=0.1):
    """Full-batch AdamW, early stopping on validation sessions. Small net, weight decay and input noise keep the
    correction smooth: the table is routine operation, not a designed scan. Returns (net, mu, sd, seconds)."""
    torch.manual_seed(SEED)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    Xt, yt, Xvt, yvt = t((X - mu) / sd), t(y)[:, None], t((Xv - mu) / sd), t(yv)[:, None]
    net = _net(X.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, t0 = np.inf, None, time.time()
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(Xt + noise * torch.randn_like(Xt)), yt).backward()
        opt.step(); sched.step()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(net(Xvt), yvt).item()
        if v < best:
            best, best_state = v, {k: p.detach().clone() for k, p in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.eval(), mu, sd, time.time() - t0


def _lstsq(A, y):
    """Least squares with standard errors. A already carries its intercept column."""
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    cov = np.linalg.inv(A.T @ A) * (res @ res) / max(len(y) - A.shape[1], 1)
    return coef, np.sqrt(np.diag(cov))


def _rmse(a):
    return float(np.sqrt(np.mean(np.square(a))))


def evaluate(c: dict, train, val, test, device) -> tuple[dict, dict]:
    """Fit every model on train(+val for early stopping), score ln(tau) RMSE on the test shots."""
    X, ln_tau, ln_98 = features(c), np.log(c["tau_E_s"]), np.log(tau_ipb98(c))
    y = ln_tau - ln_98
    fit = np.concatenate([train, val])            # the closed-form baselines get the validation shots too
    one = np.ones((len(X), 1), dtype=np.float32)
    A7, A8 = np.hstack([one, X[:, :7]]), np.hstack([one, X])

    lnH = y[fit].mean()
    b7, se7 = _lstsq(A7[fit].astype(float), ln_tau[fit])
    b8, _ = _lstsq(A8[fit].astype(float), ln_tau[fit])
    net, mu, sd, secs = _fit_net(X[train], y[train], X[val], y[val], device)
    with torch.no_grad():
        r_net = net(torch.as_tensor((X[test] - mu) / sd, device=device)).cpu().numpy()[:, 0]

    rmse = {"ipb98": _rmse(y[test]), "ipb98_x_H": _rmse(y[test] - lnH),
            "power_law": _rmse(ln_tau[test] - A7[test] @ b7), "power_law+f_nbi": _rmse(ln_tau[test] - A8[test] @ b8),
            "hybrid_physicsnemo": _rmse(y[test] - r_net)}
    info = {"n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)), "rmse_ln_tau": rmse,
            "H_fit": float(np.exp(lnH)), "train_seconds": round(secs, 2),
            "power_law_exponents": {k: {"value": round(float(b7[i + 1]), 3), "std_err": round(float(se7[i + 1]), 3)}
                                    for i, k in enumerate(VARS)}}
    return info, {"net": net, "mu": mu, "sd": sd}


def local_exponents(bundle, X, device) -> dict:
    """Mean d ln(tau_hybrid) / d ln(x) over rows: IPB98 exponent + autograd through the residual network."""
    mu, sd = (torch.as_tensor(bundle[k], dtype=torch.float32, device=device) for k in ("mu", "sd"))
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device).requires_grad_(True)
    bundle["net"]((Xt - mu) / sd).sum().backward()
    g = Xt.grad.mean(0).cpu().numpy()
    return {k: round(float(IPB98_EXP[k] + g[i]), 3) for i, k in enumerate(VARS) if k in IPB98_EXP}


def train(showcase=(27257, 29823, 30166, 30192, 30420)) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = table()
    n = c["shot_id"].size
    rng = np.random.default_rng(SEED)

    # split A: by session block; blocks holding the shots shown in the app are forced into the test set
    block = c["shot_id"] // BLOCK
    ids = rng.permutation(np.unique(block))
    shown = np.isin(ids, np.asarray(showcase) // BLOCK)
    ids = np.concatenate([ids[~shown], ids[shown]])
    n_test, n_val = int(0.15 * ids.size), int(0.10 * ids.size)
    rows = lambda b: np.flatnonzero(np.isin(block, b))
    A = (rows(ids[: ids.size - n_test - n_val]), rows(ids[ids.size - n_test - n_val: ids.size - n_test]), rows(ids[ids.size - n_test:]))
    # split B: temporal. Train on campaigns M5-M8 (validation = held-out session blocks), test on M9 (2013)
    old = rng.permutation(np.unique(block[c["campaign"] < 9]))
    in_old = lambda b: np.flatnonzero(np.isin(block, b) & (c["campaign"] < 9))
    B = (in_old(old[int(0.1 * old.size):]), in_old(old[: int(0.1 * old.size)]), np.flatnonzero(c["campaign"] == 9))

    info_A, bundle = evaluate(c, *A, device)
    info_B, _ = evaluate(c, *B, device)
    X_test = features(c)[A[2]]
    metrics = {
        "data": "UKAEA FAIR-MAST shot table, cpf_* scalars at peak current (CC BY-SA 4.0)", "n_shots": int(n),
        "target": "ln(tau_E measured / tau_E IPB98(y,2)); tau_E = W_mhd / (P_ohm + P_nbi - dW/dt), W includes beam fast ions",
        "inputs": list(VARS) + ["f_nbi"], "model": "physicsnemo.models.mlp.FullyConnected 8-32-32-1, SiLU, input noise 0.1 sigma, weight decay 1e-2",
        "n_params": int(sum(p.numel() for p in bundle["net"].parameters())),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "split_by_session_block": info_A, "split_temporal_M9_held_out": info_B,
        "block_size": BLOCK, "held_out_blocks": sorted(int(b) for b in ids[ids.size - n_test:]),   # of the shipped model
        "exponents": {"ipb98": IPB98_EXP, "valovic_2009_mast": VALOVIC_EXP,
                      "power_law_refit": {k: info_A["power_law_exponents"][k] for k in IPB98_EXP},
                      "hybrid_mean_local": local_exponents(bundle, X_test, device)},
        "ranges": {k: [round(float(np.percentile(c[k], 5)), 3), round(float(np.percentile(c[k], 95)), 3)] for k in VARS},
    }
    best = min(info_A["rmse_ln_tau"], key=info_A["rmse_ln_tau"].get)
    metrics["best_on_session_split"] = best
    metrics["hybrid_gain_vs_power_law_same_inputs"] = {
        name: round(1 - i["rmse_ln_tau"]["hybrid_physicsnemo"] / i["rmse_ln_tau"]["power_law+f_nbi"], 3)
        for name, i in (("session_block", info_A), ("temporal_M9", info_B))}

    MODELS.mkdir(exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in bundle["net"].state_dict().items()}, "mu": bundle["mu"], "sd": bundle["sd"],
                "n_in": len(VARS) + 1}, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=1))
    return metrics


# ---------------------------------------------------------------- inference (used by the API)
_loaded = None


def available() -> bool:
    return MODEL_FILE.exists()


def load():
    """Trained network + input normalisation, read once."""
    global _loaded
    if _loaded is None:
        ck = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
        net = _net(ck["n_in"])
        net.load_state_dict(ck["state"])
        _loaded = (net.eval(), ck["mu"], ck["sd"])
    return _loaded


def correction(x: dict) -> np.ndarray:
    """Multiplier on tau_E IPB98 for each row of x (see shot_features). CPU, a few hundred rows at most."""
    net, mu, sd = load()
    X = np.nan_to_num((features(x) - mu) / sd, nan=0.0)
    with torch.no_grad():
        r = net(torch.as_tensor(X, dtype=torch.float32)).numpy()[:, 0]
    return np.exp(np.clip(r, -RESIDUAL_CLIP, RESIDUAL_CLIP))


if __name__ == "__main__":
    m = train()
    for name in ("split_by_session_block", "split_temporal_M9_held_out"):
        s = m[name]
        print(f"\n{name}: train {s['n_train']}  val {s['n_val']}  test {s['n_test']}  (net trained in {s['train_seconds']} s)")
        for k, v in s["rmse_ln_tau"].items():
            print(f"  {k:20s} RMSE ln(tau_E) = {v:.3f}")
    print("\nexponents:", json.dumps(m["exponents"], indent=1))
    print("hybrid gain over refit power law with the same inputs:", m["hybrid_gain_vs_power_law_same_inputs"])
    print("5-95% ranges:", m["ranges"])
