"""IMAS-shaped export of a replay bundle (export-only): the ShotBundle v1 arrays at IMAS
data-dictionary paths, plus a FusionLab annotations sidecar for everything the dictionary
has no home for. Served by `GET /replay/{shot_id}/imas` (fusionlab.api_replay).

FusionLab's internal data spine is the ShotBundle v1 (docs/DATA_CONTRACT.md) — IMAS is never
the internal format. This module is a naming discipline at the export edge, not a schema
enforcement: a static, in-repo table maps each bundle array to its IMAS path, converting
units once, here (house units m, T, MA, 1e20 m^-3, keV, MW, MJ, s -> the dictionary's
A, eV, m^-3, W, J). Values are the bundle's own — no physics is recomputed; the psi map and
the profiles_1d coordinate are un-normalised with the bundle's own axis/boundary fluxes (a
lossless linear re-labelling of psi_n, not an interpolation). Groups the bundle lacks are
omitted, never emitted as empty IDS stubs — the same graceful degradation /replay applies.

Dictionary conventions were verified against the IMAS Data Dictionary
(imas-data-dictionary.readthedocs.io, DD 4.2.x build, accessed 2026-10-12) before this
conversion table was locked: equilibrium/global_quantities ip in A, q_95, beta_tor_norm
(defined as 100*beta_tor*a[m]*B0[T]/ip[MA] — i.e. beta_N), li_3, volume in m^3,
magnetic_axis.{r,z} in m, psi_magnetic_axis in Wb (DD 4.1 renamed psi_axis);
boundary/{outline.{r,z} in m, psi in Wb, minor_radius in m, elongation, triangularity};
vacuum_toroidal_field/b0 in T (b0 is defined at r0 — the archive does not record the
reference radius of its B_T, so r0 is omitted rather than invented);
profiles_1d/{psi in Wb, psi_norm, f in T.m ("Diamagnetic function (F=R B_Phi)")};
profiles_2d/psi in Wb on grid_type index 1 ("rectangular": R=dim1, Z=dim2, and the
position arrays r/z "should not be filled since they are redundant with grid/dim1 and
dim2"); pf_active.coil[i]/geometry/rectangle/{r,z,width,height} in m with geometry_type
index 2 ("rectangle"); wall/description_2d[i]/limiter/unit[j]/outline/{r,z} in m; and the
common ids_properties structure, whose `comment` is the free-text slot describing the
origin and content of the IDS.

What has no dictionary home rides the sidecar (fusionlab_annotations): the catalog's
useful/abort bookkeeping, the per-series provenance labels (measured/model/learned/computed),
the logbook meta, and the measured series the dictionary cannot hold without bending
(line-average density, heating powers, stored energy, Thomson profiles on a major-radius
axis, the machine-specific magnetics block). The IMAS payload mentions them only in
ids_properties.comment, as a pointer to the sidecar.

What this is not: an IMAS import, a schema-validating writer, or a forecast. The twin is
compared with MAST data — nothing here is validated or predictive.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fusionlab.data.bundle import BUNDLE_VERSION, SPEC
from fusionlab.data.machine import MAST_CONFIG, MachineConfig

GENERATOR = "FusionLab IMAS-shaped export (export-only; IMAS is never the internal data spine)"

# House unit -> (IMAS unit, factor). The conversion happens once, here, at the export edge.
# Units and spellings follow the DD's own unit column ("T.m", "Wb", "m^3", "1" for
# dimensionless); the factors are the test oracle's business (tests/test_imas_export.py,
# with api_replay._UNIT as the oracle for the trace names it covers).
_TO_IMAS: dict[str, tuple[str, float]] = {
    "MA": ("A", 1e6),
    "T": ("T", 1.0),
    "1e20 m^-3": ("m^-3", 1e20),
    "MW": ("W", 1e6),
    "MJ": ("J", 1e6),
    "keV": ("eV", 1e3),
    "m": ("m", 1.0),
    "m^3": ("m^3", 1.0),
    "T m": ("T.m", 1.0),
    "Wb/rad": ("Wb", 1.0),   # poloidal flux per toroidal radian: the DD's psi convention
    "s": ("s", 1.0),
    "": ("1", 1.0),
}

# Per-slice 1-D scalars: bundle field -> path inside time_slice[i]. Every path is a verified
# DD path (module docstring); the comments record the definitional caveats the mapping accepts.
_TIME_SLICE: dict[str, str] = {
    "Ip_MA": "global_quantities/ip",
    "q95": "global_quantities/q_95",
    "beta_N": "global_quantities/beta_tor_norm",   # DD definition == beta_N (renamed from beta_normal, DD 4.0)
    "li": "global_quantities/li_3",                # the archive's EFIT li, at the DD's only internal-inductance home
    "V_m3": "global_quantities/volume",
    "a_m": "boundary/minor_radius",                # archive scalar; the DD defines it (Rmax-Rmin)/2 of the boundary
    "kappa": "boundary/elongation",
    "delta": "boundary/triangularity",
    "R_mag_m": "global_quantities/magnetic_axis/r",
    "Z_mag_m": "global_quantities/magnetic_axis/z",
    "psi_axis_Wb": "global_quantities/psi_magnetic_axis",   # DD 4.1 renamed psi_axis -> psi_magnetic_axis
    "psi_bnd_Wb": "boundary/psi",                  # global_quantities/psi_boundary is deprecated since DD 4.1
}

# Bundle fields with no honest IMAS home: they ride the annotations sidecar, in house units.
# Each reason is the mapping's own record — a physics field is never bent to fit.
_NO_IMAS_HOME: dict[str, str] = {
    "n_e20": "line-average density: the DD holds densities as core_profiles profiles, not scalars",
    "P_ohm_MW": "heating powers: equilibrium carries no power balance",
    "P_nbi_MW": "heating powers: equilibrium carries no power balance",
    "P_rad_MW": "heating powers: equilibrium carries no power balance",
    "W_MJ": "MHD stored energy: no equilibrium home",
    "dWdt_MW": "MHD stored energy: no equilibrium home",
    "R_m": "geometric major radius; magnetic_axis.r carries the axis position, not this scalar",
    "Te0_keV": "core scalars: core_profiles is profile-shaped, and t_e[0] would invent a grid",
    "ne0_e20": "core scalars: core_profiles is profile-shaped, and n_e[0] would invent a grid",
    "ts_R": "Thomson profiles are on a major-radius axis; core_profiles needs rho_tor (a recomputation)",
    "ts_Te_keV": "Thomson profiles are on a major-radius axis; core_profiles needs rho_tor (a recomputation)",
    "ts_ne_e20": "Thomson profiles are on a major-radius axis; core_profiles needs rho_tor (a recomputation)",
    "q_mid": "q on a midplane-radius axis; profiles_1d.q's coordinate is psi, which the bundle lacks here",
    "q_mid_R": "q on a midplane-radius axis; profiles_1d.q's coordinate is psi, which the bundle lacks here",
    "eq_inputs": "magnetics in the machine's own sensor layout — no dictionary home",
}


def _nan_to_none(x: Any) -> Any:
    """JSON has no NaN or Infinity: replace non-finite float leaves a tolist leaves behind with None."""
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _j(a: Any, nd: int = 4) -> list:
    """numpy -> JSON-safe nested lists: rounded to nd decimals, NaN/inf as null. The same
    rule as api_replay._j, kept local so this module stays free of the API stack."""
    a = np.round(np.asarray(a).astype(float), nd)
    return _nan_to_none(a.tolist())


def _set_at(d: dict, path: str, value: Any) -> None:
    """value at an IMAS path like 'global_quantities/ip' inside one time-slice dict."""
    head, _, tail = path.partition("/")
    if not tail:
        d[head] = value
    else:
        _set_at(d.setdefault(head, {}), tail, value)


def _slice_scalars(shot: dict, i: int) -> dict:
    """One time-slice's scalar fields from the static table; NaN slices are omitted, not nulled."""
    out: dict = {}
    for field, path in _TIME_SLICE.items():
        if field not in shot:
            continue
        v = float(shot[field][i])
        if np.isfinite(v):
            unit, factor = _TO_IMAS[SPEC[field].units]
            _set_at(out, path, round(v * factor, 4))
    return out


def _psi_profiles(shot: dict, i: int, ax: float, bd: float) -> dict:
    """profiles_2d + profiles_1d for slice i: the bundle's own map and F profile, un-normalised
    with its own axis/boundary fluxes (lossless linear re-labelling — not an interpolation)."""
    out: dict = {}
    if "psi_n" in shot and np.isfinite(ax) and np.isfinite(bd):
        out["profiles_2d"] = [{
            "grid_type": {"name": "rectangular", "index": 1},   # DD index 1: cylindrical R,Z, R=dim1, Z=dim2
            "grid": {"dim1": _j(shot["psi_R"]), "dim2": _j(shot["psi_Z"])},
            "psi": _j(ax + shot["psi_n"][i] * (bd - ax)),
        }]
    if "F_Tm" in shot and np.isfinite(ax) and np.isfinite(bd):
        out["profiles_1d"] = [{
            "psi": _j(ax + shot["psi_norm"] * (bd - ax)),
            "psi_norm": _j(shot["psi_norm"]),
            "f": _j(shot["F_Tm"][i]),   # DD: f = R B_phi, the diamagnetic function, in T.m
        }]
    return out


def export_imas(shot: dict, machine: MachineConfig = MAST_CONFIG) -> dict:
    """IMAS-path-shaped export of one shot (the dict load_shot returns).

    Compares with MAST data; carries no forecast. Units convert once, here, at the export
    edge; absent groups are omitted, never emitted as empty IDS stubs. See the module
    docstring for the dictionary verification behind the mapping table.
    """
    t = np.asarray(shot["t_s"], float)
    nt = int(t.size)
    if nt == 0:
        raise ValueError("shot has no time slices")

    time_slice: list[dict] = []
    for i in range(nt):
        sl: dict = {"time": round(float(t[i]), 4)}
        sl.update(_slice_scalars(shot, i))
        if "lcfs_R" in shot:   # the boundary group: LCFS outline (+ scalar shape factors above)
            sl.setdefault("boundary", {})["outline"] = {
                "r": _j(shot["lcfs_R"][i]), "z": _j(shot["lcfs_Z"][i]),
            }
        ax = float(shot["psi_axis_Wb"][i]) if "psi_axis_Wb" in shot else float("nan")
        bd = float(shot["psi_bnd_Wb"][i]) if "psi_bnd_Wb" in shot else float("nan")
        sl.update(_psi_profiles(shot, i, ax, bd))
        time_slice.append(sl)

    eq: dict[str, Any] = {"time": _j(t), "time_slice": time_slice}
    if "B_T" in shot:
        # b0 is defined at the reference radius r0; the archive does not record that radius
        # for its B_T, so r0 is omitted rather than invented (noted in ids_properties.comment).
        eq["vacuum_toroidal_field"] = {"b0": _j(shot["B_T"])}

    out: dict[str, Any] = {"ids_properties": {"comment": _comment(shot, machine)}, "equilibrium": eq}

    if "coil_R" in shot and np.size(shot["coil_R"]):
        coils = []
        for r, z, w, h in zip(shot["coil_R"], shot["coil_Z"], shot["coil_W"], shot["coil_H"], strict=True):
            if np.isfinite([r, z, w, h]).all():
                coils.append({"name": "", "geometry": {
                    "geometry_type": {"name": "rectangle", "index": 2},   # DD: 2 = rectangle
                    "rectangle": {"r": round(float(r), 4), "z": round(float(z), 4),
                                  "width": round(float(w), 4), "height": round(float(h), 4)},
                }})
        if coils:
            out["pf_active"] = {"coil": coils}
    if "wall_R" in shot:
        # The DD's description_2d.type has no enum value for a single limiter unit without a
        # filled vessel, so type is omitted rather than mislabelled.
        out["wall"] = {"description_2d": [{"limiter": {"unit": [{"name": "vessel_limiter", "outline": {
            "r": _j(shot["wall_R"]), "z": _j(shot["wall_Z"])}}]}}]}
    return out


def _comment(shot: dict, machine: MachineConfig) -> str:
    """The ids_properties.comment: origin, content, honesty rule, and the sidecar pointer.
    Catalog bookkeeping and provenance labels have no IMAS home — the comment only points."""
    meta = shot.get("meta", {}) or {}
    sid = meta.get("shot_id", "?")
    campaign = f", campaign {meta['campaign']}" if meta.get("campaign") else ""
    return (
        f"FusionLab replay export of {machine.device} shot {sid} "
        f"({machine.reconstruction_name} reconstruction{campaign}). An educational digital twin "
        "compared with MAST data - carries no forecast; nothing here is validated or predictive. "
        "Export-only: IMAS is never FusionLab's internal data spine (docs/DATA_CONTRACT.md). "
        "Catalog bookkeeping (useful/abort), per-series provenance labels, logbook meta and the "
        "series the dictionary has no home for are in the sibling fusionlab_annotations.json "
        "sidecar of the same download. Units converted once at the export edge (house MA, "
        "1e20 m^-3, keV, MW, MJ -> A, m^-3, eV, W, J); vacuum_toroidal_field/b0's reference "
        "radius is not recorded by the archive and is left unset. "
        f"Generated by {GENERATOR}."
    )


def fusionlab_annotations(shot: dict, *, archive_flags: dict, provenance: dict,
                          attribution: str | None = None) -> dict:
    """The sibling sidecar: what the IMAS dictionary has no home for, in the house's own units.

    archive_flags: the catalog's useful/abort bookkeeping (api_replay._archive_flags) —
    archive bookkeeping, not physics. provenance: the per-series labels of the replay bundle
    (api_replay._export_provenance_map), the measured/model/learned/computed vocabulary.
    quantities_without_imas_home: the bundle arrays the dictionary cannot hold (see
    _NO_IMAS_HOME), values exactly as the bundle carries them.
    """
    meta = shot.get("meta", {}) or {}
    return {
        "shot_id": meta.get("shot_id"),
        "bundle_version": BUNDLE_VERSION,
        "attribution": attribution,
        "archive_flags": archive_flags,
        "meta": meta,   # the logbook text, verbatim — no IMAS home for it either
        "provenance": provenance,
        "quantities_without_imas_home": {
            name: {"unit": SPEC[name].units, "values": _j(shot[name]), "note": _NO_IMAS_HOME[name]}
            for name in _NO_IMAS_HOME if name in shot
        },
        "note": (
            "FusionLab annotations for the sibling IMAS export. The twin is compared with "
            "MAST data; nothing is validated or predictive."
        ),
    }
