"""ShotBundle v1: the versioned contract every FusionLab data source must produce.

`mast.load_shot()` has always returned a plain dict of arrays in fixed project units —
that dict is the app's de facto data contract (physics replay, the virtual shot, field-line
tracing, USD export, the UI and the agent all consume it). This module writes the contract
down and versions it:

  * SPEC records every array's units, group and shape obligation;
  * `ShotBundle.from_dict` validates a shot dict against it (presence, shape, coherence);
  * `ShotBundle.to_dict` round-trips to exactly the dict the app passes around.

Fields are flat, mirroring the on-disk npz bundle — the contract is what the ingestor
already produces, written down, not a new container. Sources beyond FAIR-MAST (issue #34,
bring-your-own-data) produce the same bundle; consumers never learn which ingestor ran.
Units are fixed house-wide: m, T, MA, 1e20 m^-3, keV, MW, MJ, s.

The equilibrium group (LCFS boundary, ψ grid, magnetic axis) is optional as of v1: a
machine without equilibrium reconstructions still replays its traces; `/replay` marks the
surfaces absent instead of failing (`api_replay.replay_shot`), and 3D/ψ/field-line
consumers degrade gracefully.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np

BUNDLE_VERSION = 1

# Fields every bundle must carry: the time base plus the 13 operating traces physics
# replay needs (W_MJ is MHD stored energy 3/2∫p dV, never the diamagnetic wmhd).
REQUIRED: tuple[str, ...] = (
    "t_s", "Ip_MA", "B_T", "n_e20", "P_ohm_MW", "P_nbi_MW", "W_MJ", "dWdt_MW",
    "q95", "beta_N", "kappa", "a_m", "R_m", "V_m3",
)


class BundleValidationError(ValueError):
    """A shot dict violates the ShotBundle contract; the message names the field."""


@dataclass(frozen=True)
class FieldSpec:
    """One bundle array: fixed unit, group (for docs and UI grouping), shape obligation."""

    units: str
    group: str  # "core" | "equilibrium" | "tracing" | "optional"
    note: str


# The v1 field table — units are fixed (m, T, MA, 1e20 m^-3, keV, MW, MJ, s); shapes are
# (nt,) on the equilibrium time base unless noted. This table drives validation, the
# contract tests and the docs; a field not here is not in the bundle.
SPEC: dict[str, FieldSpec] = {
    # --- time base -------------------------------------------------------
    "t_s": FieldSpec("s", "core", "equilibrium time base, strictly increasing; every trace shares it"),
    # --- 13 operating traces ---------------------------------------------
    "Ip_MA": FieldSpec("MA", "core", "plasma current (magnitude)"),
    "B_T": FieldSpec("T", "core", "vacuum toroidal field (magnitude, per slice)"),
    "n_e20": FieldSpec("1e20 m^-3", "core", "line-average electron density; all-NaN tolerated (model traces go null)"),
    "P_ohm_MW": FieldSpec("MW", "core", "ohmic heating power"),
    "P_nbi_MW": FieldSpec("MW", "core", "neutral-beam power (from logged scalars where no trace exists)"),
    "W_MJ": FieldSpec("MJ", "core", "MHD stored energy 3/2∫p dV — never the diamagnetic wmhd"),
    "dWdt_MW": FieldSpec("MW", "core", "dW/dt; the ingestor computes it by np.gradient when the archive lacks it"),
    "q95": FieldSpec("", "core", "edge safety factor; NaN → limit panels show gaps"),
    "beta_N": FieldSpec("", "core", "normalized beta"),
    "kappa": FieldSpec("", "core", "elongation"),
    "a_m": FieldSpec("m", "core", "minor radius"),
    "R_m": FieldSpec("m", "core", "major radius"),
    "V_m3": FieldSpec("m^3", "core", "plasma volume"),
    # --- equilibrium (optional in v1; /replay degrades gracefully) --------
    "lcfs_R": FieldSpec("m", "equilibrium", "last-closed-flux-surface R, (nt, n_pts) closed outline"),
    "lcfs_Z": FieldSpec("m", "equilibrium", "last-closed-flux-surface Z, (nt, n_pts)"),
    "psi_n": FieldSpec("", "equilibrium", "normalized ψ (0 axis, 1 separatrix), (nt, NZ, NR), rows = Z, float32 on a uniform grid"),
    "psi_R": FieldSpec("m", "equilibrium", "ψ grid R axis, (NR,), uniform spacing"),
    "psi_Z": FieldSpec("m", "equilibrium", "ψ grid Z axis, (NZ,), uniform spacing"),
    "R_mag_m": FieldSpec("m", "equilibrium", "magnetic axis R, (nt,)"),
    "Z_mag_m": FieldSpec("m", "equilibrium", "magnetic axis Z, (nt,)"),
    # --- tracing extras (field-line tracing only) --------------------------
    "F_Tm": FieldSpec("T m", "tracing", "F = R·B_φ over normalized ψ, (nt, n_f)"),
    "psi_axis_Wb": FieldSpec("Wb/rad", "tracing", "axis flux, (nt,)"),
    "psi_bnd_Wb": FieldSpec("Wb/rad", "tracing", "boundary flux, (nt,)"),
    "psi_norm": FieldSpec("", "tracing", "normalized-ψ sample axis, (n_f,)"),
    "q_mid": FieldSpec("", "tracing", "mid-plane safety profile, (nt, n_f)"),
    "q_mid_R": FieldSpec("", "tracing", "mid-plane radius per ψ sample, (n_f,)"),
    # --- optional, already-guarded consumers -------------------------------
    "delta": FieldSpec("", "optional", "triangularity"),
    "li": FieldSpec("", "optional", "internal inductance"),
    "P_rad_MW": FieldSpec("MW", "optional", "radiated power"),
    "Te0_keV": FieldSpec("keV", "optional", "core electron temperature (USD Te colour ramp)"),
    "ne0_e20": FieldSpec("1e20 m^-3", "optional", "core electron density"),
    "ts_R": FieldSpec("m", "optional", "Thomson major-radius axis, (n_ts,)"),
    "ts_Te_keV": FieldSpec("keV", "optional", "Thomson Te profile, (nt, n_ts)"),
    "ts_ne_e20": FieldSpec("1e20 m^-3", "optional", "Thomson ne profile, (nt, n_ts)"),
    "wall_R": FieldSpec("m", "optional", "vessel limiter outline R, 1-D closed contour"),
    "wall_Z": FieldSpec("m", "optional", "vessel limiter outline Z"),
    "coil_R": FieldSpec("m", "optional", "PF-coil filament R centres"),
    "coil_Z": FieldSpec("m", "optional", "PF-coil filament Z centres"),
    "coil_W": FieldSpec("m", "optional", "PF-coil filament widths"),
    "coil_H": FieldSpec("m", "optional", "PF-coil filament heights"),
    "eq_inputs": FieldSpec("", "optional", "magnetics for the eq-surrogate, (nt, n_sensors) in the machine's own layout"),
}

# Groups that must be carried together: a bundle has all of a group or none of it.
_GROUPS: tuple[tuple[str, ...], ...] = (
    ("lcfs_R", "lcfs_Z"),
    ("psi_n", "psi_R", "psi_Z"),
    ("R_mag_m", "Z_mag_m"),
    ("psi_axis_Wb", "psi_bnd_Wb"),
    ("q_mid", "q_mid_R"),
    ("ts_Te_keV", "ts_ne_e20"),
    ("wall_R", "wall_Z"),
    ("coil_R", "coil_Z", "coil_W", "coil_H"),
)

# 1-D fields whose length is the number of EFIT slices (nt).
_NT_ONE_D: tuple[str, ...] = (
    "Ip_MA", "B_T", "n_e20", "P_ohm_MW", "P_nbi_MW", "W_MJ", "dWdt_MW", "q95", "beta_N",
    "kappa", "a_m", "R_m", "V_m3", "R_mag_m", "Z_mag_m", "psi_axis_Wb", "psi_bnd_Wb",
    "delta", "li", "P_rad_MW", "Te0_keV", "ne0_e20",
)

# 1-D fields with their OWN axis (contours, profile axes) — length is free, coherence
# within their group is checked below.
_FREE_1D: tuple[str, ...] = (
    "psi_norm", "q_mid_R", "ts_R", "wall_R", "wall_Z", "coil_R", "coil_Z", "coil_W", "coil_H",
)

# 2-D fields whose first axis is the t_s base (n = free, cross-checked in the group).
_NT_2D: tuple[str, ...] = ("F_Tm", "q_mid", "eq_inputs", "ts_Te_keV", "ts_ne_e20")


@dataclass(frozen=True)
class ShotBundle:
    """The versioned shot contract (v1): identity, time base, traces, geometry, groups.

    `load_shot()` keeps returning the plain dict (runtime interchange); the bundle is the
    schema carrier and validator at the data door. Optional fields default to None and
    consumers degrade gracefully — only the 14 REQUIRED core fields must exist.
    """

    version: int
    shot_id: str          # archive-native id, e.g. "30166"
    source: str           # ingestor name: "fair-mast", later "user-file", ...
    t_s: np.ndarray
    Ip_MA: np.ndarray
    B_T: np.ndarray
    n_e20: np.ndarray
    P_ohm_MW: np.ndarray
    P_nbi_MW: np.ndarray
    W_MJ: np.ndarray
    dWdt_MW: np.ndarray
    q95: np.ndarray
    beta_N: np.ndarray
    kappa: np.ndarray
    a_m: np.ndarray
    R_m: np.ndarray
    V_m3: np.ndarray
    meta: dict = field(default_factory=dict)
    lcfs_R: np.ndarray | None = None
    lcfs_Z: np.ndarray | None = None
    psi_n: np.ndarray | None = None
    psi_R: np.ndarray | None = None
    psi_Z: np.ndarray | None = None
    R_mag_m: np.ndarray | None = None
    Z_mag_m: np.ndarray | None = None
    F_Tm: np.ndarray | None = None
    psi_axis_Wb: np.ndarray | None = None
    psi_bnd_Wb: np.ndarray | None = None
    psi_norm: np.ndarray | None = None
    q_mid: np.ndarray | None = None
    q_mid_R: np.ndarray | None = None
    delta: np.ndarray | None = None
    li: np.ndarray | None = None
    P_rad_MW: np.ndarray | None = None
    Te0_keV: np.ndarray | None = None
    ne0_e20: np.ndarray | None = None
    ts_R: np.ndarray | None = None
    ts_Te_keV: np.ndarray | None = None
    ts_ne_e20: np.ndarray | None = None
    wall_R: np.ndarray | None = None
    wall_Z: np.ndarray | None = None
    coil_R: np.ndarray | None = None
    coil_Z: np.ndarray | None = None
    coil_W: np.ndarray | None = None
    coil_H: np.ndarray | None = None
    eq_inputs: np.ndarray | None = None

    @property
    def has_equilibrium(self) -> bool:
        """True when ψ surfaces and the LCFS boundary are present (the replay surfaces)."""
        return self.psi_n is not None and self.lcfs_R is not None

    @classmethod
    def from_dict(cls, d: Mapping, *, source: str = "unknown") -> ShotBundle:
        """Validate a shot dict against the v1 contract, naming the first bad field."""
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            raise BundleValidationError(f"missing required field(s): {', '.join(missing)}")
        meta = d.get("meta")
        if meta is not None and not isinstance(meta, dict):
            raise BundleValidationError("meta: expected a dict of logbook/scalars")
        if meta is not None and "shot_id" not in meta:
            raise BundleValidationError("meta: missing 'shot_id'")
        arrays = {k: d[k] for k in SPEC if k in d}
        bundle = cls(version=BUNDLE_VERSION, shot_id=str(meta["shot_id"]) if meta else "",
                     source=source, meta=dict(meta) if meta else {}, **arrays)
        _validate(bundle)
        return bundle

    def to_dict(self) -> dict:
        """The runtime dict exactly as load_shot() returns it (arrays + meta, nothing else)."""
        out = {k: getattr(self, k) for k in SPEC if getattr(self, k) is not None}
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    def __eq__(self, other: object) -> bool:
        """Field-wise equality (array-aware); the generated tuple eq would break on ndarrays."""
        if not isinstance(other, ShotBundle):
            return NotImplemented
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)):
                    return False
            elif a != b:
                return False
        return True


def _validate(b: ShotBundle) -> None:
    """Shape and coherence rules of the v1 contract. Vectorised; raises BundleValidationError."""

    def fail(name: str, why: str) -> None:
        raise BundleValidationError(f"{name}: {why}")

    nt = b.t_s.shape[0]
    if b.t_s.ndim != 1:
        fail("t_s", f"expected 1-D, got shape {b.t_s.shape}")
    if not np.issubdtype(b.t_s.dtype, np.floating):
        fail("t_s", f"expected a floating dtype, got {b.t_s.dtype}")
    if not np.all(np.isfinite(b.t_s)):
        fail("t_s", "time base has non-finite entries")
    if nt > 1 and not np.all(np.diff(b.t_s) > 0):
        fail("t_s", "time base is not strictly increasing")

    for name in SPEC:
        if name == "t_s":
            continue
        a = getattr(b, name)
        if a is None:
            continue
        if not isinstance(a, np.ndarray):
            fail(name, f"expected a numpy array, got {type(a).__name__}")
        if not np.issubdtype(a.dtype, np.floating):
            fail(name, f"expected a floating dtype, got {a.dtype}")
        if name in _NT_ONE_D and (a.ndim != 1 or a.shape[0] != nt):
            fail(name, f"expected ({nt},) on the t_s base, got {a.shape}")
        if name in _FREE_1D and a.ndim != 1:
            fail(name, f"expected a 1-D array, got shape {a.shape}")
        if name in _NT_2D and (a.ndim != 2 or a.shape[0] != nt):
            fail(name, f"expected (nt, n) with nt={nt}, got {a.shape}")
        if name == "psi_n" and (a.ndim != 3 or a.shape[0] != nt):
            fail("psi_n", f"expected (nt, NZ, NR) with nt={nt}, got {a.shape}")

    for group in _GROUPS:
        absent = [k for k in group if getattr(b, k) is None]
        if absent and len(absent) != len(group):
            fail(absent[0], f"incomplete group: needs all of {group}")

    psi_n, psi_Z, psi_R = b.psi_n, b.psi_Z, b.psi_R
    if psi_n is not None and psi_Z is not None and psi_R is not None and (
            psi_n.shape[1] != psi_Z.shape[0] or psi_n.shape[2] != psi_R.shape[0]):
        fail("psi_n", f"grid mismatch: psi_n {psi_n.shape} vs psi_Z {psi_Z.shape} x psi_R {psi_R.shape}")
    lcfs_R, lcfs_Z = b.lcfs_R, b.lcfs_Z
    if lcfs_R is not None and lcfs_Z is not None and lcfs_R.shape != lcfs_Z.shape:
        fail("lcfs_R", f"outline mismatch: lcfs_R {lcfs_R.shape} vs lcfs_Z {lcfs_Z.shape}")
    ts_Te, ts_R = b.ts_Te_keV, b.ts_R
    if ts_Te is not None and ts_R is not None and ts_Te.shape[-1] != ts_R.shape[0]:
        fail("ts_Te_keV", f"profile axis mismatch: {ts_Te.shape[-1]} columns vs ts_R {ts_R.shape[0]}")
    wall_R, wall_Z = b.wall_R, b.wall_Z
    if wall_R is not None and wall_Z is not None and wall_R.shape != wall_Z.shape:
        fail("wall_R", f"outline mismatch: wall_R {wall_R.shape} vs wall_Z {wall_Z.shape}")
    coil_R, coil_Z, coil_W, coil_H = b.coil_R, b.coil_Z, b.coil_W, b.coil_H
    if coil_R is not None and coil_Z is not None and coil_W is not None and coil_H is not None \
            and len({coil_R.shape, coil_Z.shape, coil_W.shape, coil_H.shape}) != 1:
        fail("coil_R", f"filament arrays must share one shape, got {coil_R.shape} {coil_Z.shape} {coil_W.shape} {coil_H.shape}")
