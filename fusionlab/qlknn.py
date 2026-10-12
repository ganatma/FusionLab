"""QLKNN7_11 turbulent-flux net, reduced to a confinement level for the virtual shot.

QLKNN7_11 (vendored from DeepMind's fusion_surrogates, Apache-2.0 — provenance in
models/qlknn7_11_ATTRIBUTION.md, rebuilt by scripts/convert_qlknn_weights.py) is an MLP over
10 local plasma inputs that predicts turbulent heat and particle fluxes in gyro-Bohm units.
It outputs FLUXES, not a confinement time. Turning them into the tau_E = K · P_loss^-alpha
interface the virtual-shot closures consume takes a reduction — the capped power-balance
fixed point below — whose profile shapes and power routing are assumptions, not measurements.
The re-fly evidence (models/virtual_metrics.json, via scripts/validate_virtual.py) measures
how that reduction compares with MAST data; nothing here is validated or predictive.

The reduction, one fixed point per time slice (vectorized across slices):

1. Fixed profile shapes: T = T0 (1-x^2)^A_T and n = n0 (1-x^2)^A_N, with A_T = 2.0 placing
   R/L_T at the balance surface in the canonical stiff band (~6-8 over MAST-like aspect
   ratios); below about A_T 1.8 the net sees only sub-critical gradients and no balance
   exists at any temperature. The balance is taken at x = X_EVAL = 0.7 (mid-radius). No
   per-shot Ti or composition measurement exists, so Ti = Te and the main-ion content is
   the quasineutral value at the 0D engine's default effective charge (ZEFF, carried by a
   carbon-like impurity). These shape and composition choices are assumptions — the re-fly
   evidence measures their consequence, nothing here is a fit.
2. The net's inputs follow from the shapes and the shot globals: R/L_T, R/L_n from the
   shape derivatives, q anchored on the shot's own q95 under q ~ x^2 (cylindrical
   flat-current), shear smag = 2 from that same shape, and LogNuStar from the QuaLiKiz
   collisionality definition (nu_e * tau_bounce, see the QuaLiKiz wiki's input table).
   Inputs outside the training grid are clipped to it and counted.
3. A capped bisection finds the local temperature at which the net's total conductive
   heat flux — converted from gyro-Bohm to W/m^2 with the QuaLiKiz chi_GB scaling —
   balances the replayed loss power crossing the evaluation surface. All of the loss
   power is assumed to cross that surface (no deposition profile is modelled).
4. The two-point reduction repeats the solve at P_RATIO x the loss power and reads off
   the local power law: alpha = -ln(tau2/tau1)/ln(P_RATIO), K = tau1 * P_loss^alpha.
   The iteration cap is BISECTION_STEPS halvings of T_BRACKET_KEV; slices the bracket
   cannot resolve are counted and reported, never hidden.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from fusionlab import mast
from fusionlab.physics import KEV_J, stored_energy, surface, volume

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE, ATTRIBUTION_FILE = MODELS / "qlknn.pt", MODELS / "qlknn7_11_ATTRIBUTION.md"

# ---- the 0D reduction's documented constants ------------------------------------------------
X_EVAL = 0.7        # the balance surface: mid-radius
A_T, A_N = 2.0, 0.3  # temperature / density profile shape exponents (A_T = 2 keeps R/L_T at the
#                     balance surface in the ~6-8 stiff band; the balance is sub-critical below ~1.8)
Q_EDGE_X = 0.95     # the shot's q95 anchors q(x) ~ x^2 here
SMAG = 2.0          # magnetic shear of the same q ~ x^2 shape
ZEFF = 1.7          # physics.Controls().Zeff — the 0D engine's default effective charge (test-pinned)
Z_IMP = 6           # the impurity charge assumed to carry ZEFF (carbon-like)
NORMNI = 1.0 - (ZEFF - 1.0) / (Z_IMP - 1.0)  # main-ion density ratio, quasineutral
TI_TE = 1.0         # no per-shot Ti measurement: Ti = Te
P_RATIO = 1.25      # the second power point the (K, alpha) two-point reduction solves at
T_BRACKET_KEV = (0.02, 50.0)
BISECTION_STEPS = 32  # the fixed point's iteration cap: 32 halvings resolve the bracket to ~1e-8
ALPHA_LIMITS = (-1.0, 0.95)
P_FLOOR_MW = 1e-6

# chi_GB needs the SI charge/mass constants (exact 2019 SI values); FusionLab's own geometry
# and stored-energy helpers come from physics.py.
Q_E, M_P, M_E = 1.602176634e-19, 1.67262192387e-27, 9.1093837015e-31

# Chord average of the density shape: the archive's line-average density is a central chord.
_CHORD_N = 0.5 * math.gamma(0.5) * math.gamma(1.0 + A_N) / math.gamma(1.5 + A_N)
# Volume averages of the shapes over dV ~ x dx (exact: 2*int_0^1 x (1-x^2)^a dx = 1/(a+1)).
_I_N, _I_T = 1.0 / (1.0 + A_N), 1.0 / (1.0 + A_T)


# ---------------------------------------------------------------- model (the vendored net)
def _build_net(state: dict, n_hidden: int, activation: str):
    """The upstream MLP in torch. Lives in a factory so importing fusionlab.qlknn never imports
    torch — the house pattern (surrogate.py); torch loads when the model is first used."""
    import torch
    from torch import nn

    acts = {"tanh": torch.tanh, "relu": torch.relu}
    if activation not in acts:
        raise RuntimeError(f"checkpoint activation {activation!r} has no torch mapping")
    act = acts[activation]
    layers = []
    for i in range(n_hidden + 1):
        w, b = state[f"w{i}"], state[f"b{i}"]
        lin = nn.Linear(w.shape[0], w.shape[1])
        with torch.no_grad():
            lin.weight.copy_(w.T)
            lin.bias.copy_(b)
        layers.append(lin)

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(layers)   # registered: .to()/.eval() reach the weights

        def forward(self, x):
            for lin in self.layers[:-1]:
                x = act(lin(x))
            return self.layers[-1](x)

    return _Net().eval()


_loaded: _Model | None = None


class _Model(NamedTuple):
    """Everything the net needs at inference, read once per process."""

    net: Any
    i_mean: np.ndarray
    i_sd: np.ndarray
    t_mean: np.ndarray
    t_sd: np.ndarray
    i_min: np.ndarray
    i_max: np.ndarray
    input_names: list[str]
    target_names: list[str]
    flux_map: dict


def available() -> bool:
    return MODEL_FILE.exists()


def load() -> _Model:
    """The vendored net + its stats, read once per process, float64 for eval."""
    global _loaded
    if _loaded is None:
        import torch

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"{MODEL_FILE.name} is missing on this host — run scripts/convert_qlknn_weights.py "
                "to vendor the QLKNN7_11 weights (see models/qlknn7_11_ATTRIBUTION.md)"
            )
        ck = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)  # checkpoint is tensor-only (B614)
        state = {k: v.to(torch.float64) for k, v in ck["state"].items()}
        net = _build_net(state, int(ck["num_hiddens"]), str(ck["activation"]))
        net = net.to(torch.float64)
        _loaded = _Model(
            net=net,
            i_mean=ck["input_mean"].to(torch.float64).numpy(),
            i_sd=ck["input_stddev"].to(torch.float64).numpy(),
            t_mean=ck["target_mean"].to(torch.float64).numpy(),
            t_sd=ck["target_stddev"].to(torch.float64).numpy(),
            i_min=ck["input_min"].to(torch.float64).numpy(),
            i_max=ck["input_max"].to(torch.float64).numpy(),
            input_names=list(ck["input_names"]),
            target_names=list(ck["target_names"]),
            flux_map=dict(ck["flux_map"]),
        )
    return _loaded


def predict_targets(x: np.ndarray) -> np.ndarray:
    """Unnormalized net targets (n, len(target_names)) for (n, len(input_names)) inputs — the
    upstream predict_targets, in torch. The reference pairs in
    tests/fixtures/qlknn_7_11_test_data.npz pin this to the upstream implementation."""
    m = load()
    import torch

    z = torch.as_tensor((x - m.i_mean) / np.where(m.i_sd == 0, 1, m.i_sd), dtype=torch.float64)
    with torch.no_grad():
        y = m.net(z).numpy()
    return y * np.where(m.t_sd == 0, 1, m.t_sd) + m.t_mean


def fluxes(x: np.ndarray) -> dict[str, np.ndarray]:
    """Gyro-Bohm heat/particle fluxes by name, composed from the targets exactly as the
    upstream get_flux_from_targets does (leading flux clipped at 0; a denominator flux is
    clipped at 0 and multiplies)."""
    m = load()
    targets = predict_targets(x)
    out = {}
    for flux_name, spec in m.flux_map.items():
        tgt = targets[:, m.target_names.index(str(spec["target"]))]
        denom = spec.get("denominator")
        if denom is not None:
            out[flux_name] = tgt * np.clip(targets[:, m.target_names.index(str(denom))], 0.0, None)
        else:
            out[flux_name] = np.clip(tgt, 0.0, None)
    return out


# ---------------------------------------------------------------- the 0D reduction
def _nustar_log(n_e19: np.ndarray, T_keV: np.ndarray, q: np.ndarray, R0: np.ndarray, Rmin: np.ndarray) -> np.ndarray:
    """LogNuStar per the QuaLiKiz wiki: log10(nu_e * tau_bounce), n in 1e19 m^-3, T in keV."""
    lam = 15.2 - 0.5 * np.log(np.maximum(0.1 * n_e19, 1e-12)) + np.log(np.maximum(T_keV, 1e-6))
    nu_e = 914.7 * ZEFF * n_e19 * lam / np.maximum(T_keV, 1e-6) ** 1.5
    v_e = np.sqrt(np.maximum(T_keV, 1e-6) * 1e3 * Q_E / M_E)          # the wiki's sqrt(T_e * 1e3 * q_e / m_e)
    tau_bounce = q * R0 / ((X_EVAL * Rmin / R0) ** 1.5 * v_e)
    return np.log10(np.maximum(nu_e * tau_bounce, 1e-30))


def _globals(shot: dict) -> dict[str, np.ndarray]:
    """Per-slice globals the reduction reads, NaN-sanitized like surrogate.shot_features."""
    g = {k: np.nan_to_num(np.asarray(shot[k], dtype=float).ravel()) for k in
         ("Ip_MA", "B_T", "n_e20", "R_m", "a_m", "V_m3", "q95")}
    g["kappa_a"] = g["V_m3"] / volume(g["R_m"], g["a_m"], 1.0)
    g["q"] = g["q95"] * (X_EVAL / Q_EDGE_X) ** 2                      # q ~ x^2 anchored on the shot's q95
    return g


def _net_inputs(g: dict[str, np.ndarray], T_keV: np.ndarray) -> np.ndarray:
    """(n, 10) net inputs in checkpoint order at the evaluation point."""
    names = load().input_names
    n = g["Ip_MA"].size
    r_over_a = g["R_m"] / np.maximum(g["a_m"], 1e-6)
    grad_t = r_over_a * 2.0 * A_T * X_EVAL / (1.0 - X_EVAL**2)        # R/L_T of the fixed shape
    grad_n = r_over_a * 2.0 * A_N * X_EVAL / (1.0 - X_EVAL**2)
    n_local = g["n_e20"] / _CHORD_N * (1.0 - X_EVAL**2) ** A_N * 10.0  # 1e19 m^-3 at the eval point
    cols: dict[str, np.ndarray] = {
        "Ati": np.full(n, grad_t), "Ate": np.full(n, grad_t),
        "Ane": np.full(n, grad_n), "Ani": np.full(n, grad_n),
        "q": g["q"], "smag": np.full(n, SMAG), "x": np.full(n, X_EVAL), "Ti_Te": np.full(n, TI_TE),
        "LogNuStar": _nustar_log(n_local, T_keV, g["q"], g["R_m"], g["a_m"]),
        "normni": np.full(n, NORMNI),
    }
    missing = [k for k in names if k not in cols]
    if missing:
        raise RuntimeError(f"QLKNN7_11 input {missing[0]!r} has no 0D mapping — extend the reduction in qlknn.py")
    return np.stack([np.broadcast_to(np.atleast_1d(cols[k]), (n,)) for k in names], axis=1)


def _heat_flux_si(g: dict[str, np.ndarray], T_keV: np.ndarray, i_min: np.ndarray, i_max: np.ndarray) -> np.ndarray:
    """Total conducted heat flux [W/m^2] at the evaluation point: (efe + efi) in gyro-Bohm,
    converted with the QuaLiKiz chi_GB = sqrt(m_i) T_e^1.5 / (e^2 B^2 a) scaling."""
    f = fluxes(np.clip(_net_inputs(g, T_keV), i_min, i_max))
    q_gb = f["efeITG"] + f["efeTEM"] + f["efeETG"] + f["efiITG"] + f["efiTEM"]
    n_local = g["n_e20"] / _CHORD_N * (1.0 - X_EVAL**2) ** A_N * 1e20  # m^-3
    t_j = np.maximum(T_keV, 1e-6) * KEV_J
    m_i = mast.CONFIG.isotope_md * M_P
    chi_gb = np.sqrt(m_i) * t_j**1.5 / (Q_E**2 * np.maximum(g["B_T"], 1e-6) ** 2 * np.maximum(g["a_m"], 1e-6))
    return q_gb * n_local * t_j * chi_gb / np.maximum(g["a_m"], 1e-6)


def _tau_e(T_keV: np.ndarray, P_loss_MW: np.ndarray, g: dict[str, np.ndarray]) -> np.ndarray:
    """Energy confinement time [s] of the assumed shapes at the solved temperature."""
    t0 = T_keV / (1.0 - X_EVAL**2) ** A_T
    w = stored_energy(g["n_e20"] * (1.0 - X_EVAL**2) ** A_N / _CHORD_N * _I_N, t0 * _I_T, g["V_m3"])
    return w / np.maximum(P_loss_MW, P_FLOOR_MW)


def confinement_coeff(shot: dict, P_loss_MW: float | np.ndarray) -> dict:
    """(K, alpha) per slice such that tau_E = K * P_loss^-alpha, from a capped power-balance
    fixed point over the vendored QLKNN7_11 net. Assumptions and the iteration cap are
    documented on the module; compared with MAST data by the re-fly gate, never predictive.

    Returns {"K", "alpha", "n_clipped", "n_unresolved"} where the counts report how many
    slices needed input clipping and how many the T bracket could not resolve.
    """
    m = load()
    g = _globals(shot)
    P = np.maximum(np.asarray(P_loss_MW, dtype=float).ravel(), P_FLOOR_MW)
    s_eval = surface(g["R_m"], X_EVAL * g["a_m"], g["kappa_a"])

    def solve(required_wm2: np.ndarray) -> np.ndarray:
        lo = np.full(P.size, T_BRACKET_KEV[0])
        hi = np.full(P.size, T_BRACKET_KEV[1])
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            hotter = _heat_flux_si(g, mid, m.i_min, m.i_max) < required_wm2
            lo, hi = np.where(hotter, mid, lo), np.where(hotter, hi, mid)
        return 0.5 * (lo + hi)

    req = P * 1e6 / np.maximum(s_eval, 1e-6)
    t1, t2 = solve(req), solve(req * P_RATIO)
    tau1, tau2 = _tau_e(t1, P, g), _tau_e(t2, P * P_RATIO, g)

    alpha = np.clip(-np.log(np.maximum(tau2, 1e-12) / np.maximum(tau1, 1e-12)) / np.log(P_RATIO), *ALPHA_LIMITS)
    k = np.maximum(tau1, 1e-12) * P**alpha
    raw = _net_inputs(g, t1)
    n_clipped = int(((raw < m.i_min) | (raw > m.i_max)).any(axis=1).sum())
    unresolved = int(((_heat_flux_si(g, np.full(P.size, T_BRACKET_KEV[1]), m.i_min, m.i_max) < req)
                      | (_heat_flux_si(g, np.full(P.size, T_BRACKET_KEV[0]), m.i_min, m.i_max) > req)).sum())
    return {"K": k, "alpha": alpha, "n_clipped": n_clipped, "n_unresolved": unresolved}


if __name__ == "__main__":
    from fusionlab import mast
    from fusionlab.physics import replay

    shot = mast.load_shot(30166)
    out = confinement_coeff(shot, replay(shot)["P_loss_MW"])
    print(f"shot 30166: {out['K'].size} slices, K median {np.nanmedian(out['K']):.3g} s*MW^alpha, "
          f"alpha median {np.nanmedian(out['alpha']):.3f}, clipped {out['n_clipped']}, unresolved {out['n_unresolved']}")
