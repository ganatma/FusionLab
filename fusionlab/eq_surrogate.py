"""Surrogate of EFIT's equilibrium reconstruction on MAST: magnetics -> poloidal flux map psi(R,Z).

EFIT fits a Grad-Shafranov equilibrium to the magnetic measurements of one time slice. This module learns
that map from EFIT's own stored output (FAIR-MAST level 2, group `equilibrium`), so it is a surrogate of
EFIT's reconstruction, not an independent measurement and not a predictive code: it can only be as right
as EFIT was, and only on discharges that look like the training campaigns. Holdout errors (by shot) live
in models/eq_surrogate_metrics.json, next to a mean and a ridge-regression baseline.

    raw inputs (n, 226), archive units: 78 poloidal field probes [T], 46 flux loops [Wb],
                                        101 PF/passive circuit currents [A], measured plasma current [A]
    -> keep the sensors that are alive in the training shots, mean-impute dropouts, standardise
    -> physicsnemo FullyConnected -> k PCA coefficients of psi (optionally on top of a ridge fit)
    -> psi (n, 65, 65) [Wb/rad], axes (Z, R), grid in EqSurrogate.R_m / .Z_m

Train: scripts/fetch_eq_dataset.py, then scripts/train_eq_surrogate.py. Use: load().predict_psi(x).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE, METRICS_FILE = MODELS / "eq_surrogate.pt", MODELS / "eq_surrogate_metrics.json"

N_PROBE, N_LOOP, N_PF = 78, 46, 101
INPUT_NAMES = ([f"b_field_pol_probe_{i}" for i in range(N_PROBE)] + [f"flux_loop_{i}" for i in range(N_LOOP)]
               + [f"pf_current_{i}" for i in range(N_PF)] + ["ip_measured"])
N_RAW = len(INPUT_NAMES)      # 226
NZ = NR = 65

# The axis search stays inside this box so a PF coil (a local flux extremum too) is never picked.
AXIS_BOX_R_M, AXIS_BOX_Z_M = (0.3, 1.5), (-0.8, 0.8)


def build_net(cfg: dict) -> torch.nn.Module:
    """The PhysicsNeMo MLP. cfg is stored in the checkpoint so load() rebuilds the same network."""
    from physicsnemo.models.mlp import FullyConnected
    return FullyConnected(in_features=cfg["in_features"], layer_size=cfg["layer_size"], out_features=cfg["k"],
                          num_layers=cfg["num_layers"], activation_fn=cfg["activation_fn"],
                          skip_connections=cfg["skip_connections"])


def missing(x: torch.Tensor, zero_is_missing: torch.Tensor) -> torch.Tensor:
    """A dropped probe or flux loop is archived as NaN or exactly 0.0 (EFIT weight 0). A coil current of 0 A is real."""
    return ~torch.isfinite(x) | ((x == 0) & zero_is_missing)


class EqSurrogate(torch.nn.Module):
    """Normalisation + MLP + linear PCA decoder in one module, so inference is a single batched call."""

    def __init__(self, ck: dict):
        super().__init__()
        self.cfg = ck["config"]
        self.net = build_net(self.cfg)
        for name in ("keep", "zero_is_missing", "x_mean", "x_std", "out_std", "basis", "psi_mean", "psi_scale", "lin_W", "R_m", "Z_m"):
            self.register_buffer(name, ck[name])
        self.input_names = [INPUT_NAMES[i] for i in ck["keep"].tolist()]

    def features(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw[:, self.keep]
        x = torch.where(missing(x, self.zero_is_missing), self.x_mean, x)                      # mean-impute -> 0 after standardising
        return (x - self.x_mean) / self.x_std

    def coefficients(self, xs: torch.Tensor) -> torch.Tensor:
        c = self.net(xs) * self.out_std
        if self.cfg["residual"]:                                          # ridge fit + learned correction
            c = c + xs @ self.lin_W[:-1] + self.lin_W[-1]
        return c

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        c = self.coefficients(self.features(raw))
        return (self.psi_mean + self.psi_scale * (c @ self.basis)).reshape(-1, NZ, NR)


@lru_cache(maxsize=2)
def load(device: str | None = None) -> EqSurrogate:
    """The trained surrogate from models/eq_surrogate.pt (raises FileNotFoundError if it was never trained)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    model = EqSurrogate(ck)
    model.net.load_state_dict(ck["state_dict"])
    return model.to(dev).eval()


def predict_psi(inputs: np.ndarray, device: str | None = None) -> np.ndarray:
    """(n, 226) raw magnetics in archive units (order: INPUT_NAMES) -> psi (n, 65, 65) [Wb/rad], axes (Z, R)."""
    model = load(device)
    x = torch.as_tensor(np.atleast_2d(inputs), dtype=torch.float32, device=model.x_mean.device)
    if x.shape[1] != N_RAW:
        raise ValueError(f"expected (n, {N_RAW}) inputs, got {tuple(x.shape)}")
    with torch.no_grad():
        return model(x).cpu().numpy()


def magnetic_axis(psi: torch.Tensor, sign: torch.Tensor, R_m: torch.Tensor, Z_m: torch.Tensor):
    """Axis (R, Z) [m] of each flux map: extremum of sign*psi inside the search box, parabolic sub-grid refinement.

    psi (n, Z, R); sign (n,) is +1 where the axis is a maximum of psi, -1 where it is a minimum.
    """
    n = psi.shape[0]
    box = ((Z_m >= AXIS_BOX_Z_M[0]) & (Z_m <= AXIS_BOX_Z_M[1]))[:, None] & ((R_m >= AXIS_BOX_R_M[0]) & (R_m <= AXIS_BOX_R_M[1]))[None, :]
    f = psi * sign[:, None, None]
    flat = torch.where(box, f, torch.full_like(f, -torch.inf)).reshape(n, -1).argmax(1)
    iz, ir = flat // NR, flat % NR                                        # strictly inside the grid thanks to the box
    i = torch.arange(n, device=psi.device)

    def refine(lo, mid, hi):                                              # vertex of the parabola through 3 points
        den = lo - 2 * mid + hi
        return torch.where(den.abs() > 0, 0.5 * (lo - hi) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)

    dr = refine(f[i, iz, ir - 1], f[i, iz, ir], f[i, iz, ir + 1])
    dz = refine(f[i, iz - 1, ir], f[i, iz, ir], f[i, iz + 1, ir])
    return R_m[ir] + dr * (R_m[1] - R_m[0]), Z_m[iz] + dz * (Z_m[1] - Z_m[0])
