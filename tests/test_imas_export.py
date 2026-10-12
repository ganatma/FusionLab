"""Offline tests for the IMAS-shaped export (issue #49) on the committed FAIR-MAST cache.

The golden snapshot pins the export's shape and values; a snapshot diff is the failure mode.
When the export changes deliberately, regenerate the snapshot and review the diff — never
hand-edit it:

    uv run python -c "import json; from fusionlab import imas_export, mast; \
print(json.dumps(imas_export.export_imas(mast.load_shot(30166)), indent=1))" \
        > tests/fixtures/imas/golden_30166.json
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fusionlab import imas_export, mast
from fusionlab.data.bundle import BUNDLE_VERSION, SPEC

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "imas" / "golden_30166.json"
SHOT = 30166

# The mapping's non-scalar groups: fields export_imas places by hand (everything else with an
# IMAS home is in _TIME_SLICE).
_HAND_MAPPED = {"t_s", "B_T", "lcfs_R", "lcfs_Z", "psi_n", "psi_R", "psi_Z",
                "F_Tm", "psi_norm", "wall_R", "wall_Z", "coil_R", "coil_Z", "coil_W", "coil_H"}


@pytest.fixture(scope="module")
def shot():
    if SHOT not in mast.cached_shots():
        pytest.skip("shot cache missing")
    return mast.load_shot(SHOT)


@pytest.fixture(scope="module")
def payload(shot):
    return imas_export.export_imas(shot)


def test_mapping_table_partitions_the_spec():
    """Every SPEC field is classified exactly once — an IMAS home or an honest sidecar entry.
    A future SPEC field fails here until its home is decided, never silently unexported."""
    mapped = set(imas_export._TIME_SLICE) | _HAND_MAPPED
    assert mapped | set(imas_export._NO_IMAS_HOME) == set(SPEC)
    assert not mapped & set(imas_export._NO_IMAS_HOME)


def test_conversion_table_against_the_house_oracle():
    """api_replay._UNIT is the house's own record of its units (the oracle the spec named):
    the trace units it records agree with SPEC, and each maps to the IMAS unit the DD defines."""
    from fusionlab import api_replay
    for name, house in api_replay._UNIT.items():
        if name in SPEC:
            assert SPEC[name].units == house, name
    expected = {   # bundle field -> (IMAS unit, factor), verified against the IMAS DD
        "Ip_MA": ("A", 1e6), "B_T": ("T", 1.0), "q95": ("1", 1.0), "beta_N": ("1", 1.0),
        "kappa": ("1", 1.0), "a_m": ("m", 1.0), "V_m3": ("m^3", 1.0), "R_mag_m": ("m", 1.0),
        "Z_mag_m": ("m", 1.0), "Te0_keV": ("eV", 1e3), "ne0_e20": ("m^-3", 1e20),
        "P_ohm_MW": ("W", 1e6), "W_MJ": ("J", 1e6), "F_Tm": ("T.m", 1.0),
        "psi_axis_Wb": ("Wb", 1.0), "psi_bnd_Wb": ("Wb", 1.0), "t_s": ("s", 1.0),
    }
    for name, want in expected.items():
        assert imas_export._TO_IMAS[SPEC[name].units] == want, name


def test_golden_snapshot_30166(payload):
    """The committed snapshot pins the export of the demo shot; a diff is the failure mode
    (see the module docstring for the regeneration one-liner)."""
    assert payload == json.loads(GOLDEN.read_text())


def test_imas_payload_structure(payload):
    eq = payload["equilibrium"]
    assert set(payload) == {"ids_properties", "equilibrium", "pf_active", "wall"}
    assert set(eq) == {"time", "time_slice", "vacuum_toroidal_field"}
    k = eq["time_slice"][0]
    assert set(k) == {"time", "global_quantities", "boundary", "profiles_1d", "profiles_2d"}
    assert set(k["global_quantities"]) == {"ip", "q_95", "beta_tor_norm", "li_3", "volume",
                                           "psi_magnetic_axis", "magnetic_axis"}
    assert set(k["boundary"]) == {"outline", "psi", "minor_radius", "elongation", "triangularity"}


def test_values_are_the_bundles_own_converted_once(shot, payload):
    """Spot-check the conversion edge: house values in, IMAS units out, nothing recomputed."""
    k = shot["t_s"].size // 2
    ts = payload["equilibrium"]["time_slice"][k]
    assert ts["global_quantities"]["ip"] == round(float(shot["Ip_MA"][k]) * 1e6, 4)   # MA -> A
    assert ts["global_quantities"]["q_95"] == round(float(shot["q95"][k]), 4)         # dimensionless
    assert ts["boundary"]["minor_radius"] == round(float(shot["a_m"][k]), 4)          # m -> m
    assert payload["equilibrium"]["vacuum_toroidal_field"]["b0"][k] == round(float(shot["B_T"][k]), 4)
    assert "r0" not in payload["equilibrium"]["vacuum_toroidal_field"]   # never invented (see module docstring)


def test_psi_profiles_are_the_bundles_own_map_relabelled(shot, payload):
    """profiles_2d/psi is psi_n un-normalised with the bundle's own axis/boundary fluxes — a
    lossless linear re-labelling, not an interpolation; grid_type 1 = rectangular (R=dim1, Z=dim2)."""
    k = shot["t_s"].size // 2
    ts = payload["equilibrium"]["time_slice"][k]
    ax, bd = float(shot["psi_axis_Wb"][k]), float(shot["psi_bnd_Wb"][k])
    psi = np.round((ax + shot["psi_n"][k] * (bd - ax)).astype(float), 4)
    got = np.array([[v if v is not None else np.nan for v in row] for row in ts["profiles_2d"][0]["psi"]])
    assert np.array_equal(got, psi)
    assert ts["profiles_2d"][0]["grid_type"] == {"name": "rectangular", "index": 1}
    assert np.array_equal(np.asarray(ts["profiles_2d"][0]["grid"]["dim1"]), np.round(shot["psi_R"].astype(float), 4))
    assert np.array_equal(np.asarray(ts["profiles_1d"][0]["f"]), np.round(shot["F_Tm"][k].astype(float), 4))


def test_geometry_groups_map_to_verified_paths(shot, payload):
    """Coils land on pf_active rectangle geometry, the vessel outline on the wall limiter
    outline — the two DD paths the mapping table cites."""
    coil = payload["pf_active"]["coil"][0]
    assert coil["geometry"]["geometry_type"] == {"name": "rectangle", "index": 2}
    assert coil["geometry"]["rectangle"]["r"] == round(float(shot["coil_R"][0]), 4)
    assert coil["geometry"]["rectangle"]["height"] == round(float(shot["coil_H"][0]), 4)
    wall = payload["wall"]["description_2d"][0]["limiter"]["unit"][0]
    assert np.array_equal(np.asarray(wall["outline"]["r"]), np.round(shot["wall_R"].astype(float), 4))


def test_minimal_bundle_exports_core_traces_only():
    """A shot missing equilibrium, tracing, Thomson, vessel and coils exports core traces only —
    no empty IDS stubs (spec criterion): no pf_active/wall, no boundary outline, no profiles."""
    nt = 5
    b = {"t_s": np.linspace(0.1, 0.5, nt), "Ip_MA": np.full(nt, 0.4), "B_T": np.full(nt, 0.6),
         "n_e20": np.full(nt, 1.0), "P_ohm_MW": np.full(nt, 0.5), "P_nbi_MW": np.full(nt, 2.0),
         "W_MJ": np.full(nt, 0.1), "dWdt_MW": np.full(nt, np.nan), "q95": np.full(nt, 3.0),
         "beta_N": np.full(nt, 1.5), "kappa": np.full(nt, 1.6), "a_m": np.full(nt, 0.5),
         "R_m": np.full(nt, 0.85), "V_m3": np.full(nt, 8.0), "meta": {"shot_id": 1}}
    out = imas_export.export_imas(b)
    assert set(out) == {"ids_properties", "equilibrium"}   # no pf_active, no wall
    eq = out["equilibrium"]
    assert set(eq) == {"time", "time_slice", "vacuum_toroidal_field"}
    ts = eq["time_slice"][0]
    assert set(ts) == {"time", "global_quantities", "boundary"}
    assert "outline" not in ts["boundary"]   # scalars only — never an empty stub


def test_boundary_without_psi_fluxes_still_exports_the_outline():
    """A bundle carrying the LCFS but no axis/boundary fluxes exports the outline and omits
    profiles_2d/1d and boundary/psi rather than emitting un-normalisable or null data."""
    nt = 3
    b = {"t_s": np.linspace(0.1, 0.3, nt), "Ip_MA": np.full(nt, 0.4), "B_T": np.full(nt, 0.6),
         "n_e20": np.full(nt, 1.0), "P_ohm_MW": np.full(nt, 0.5), "P_nbi_MW": np.full(nt, 2.0),
         "W_MJ": np.full(nt, 0.1), "dWdt_MW": np.full(nt, np.nan), "q95": np.full(nt, 3.0),
         "beta_N": np.full(nt, 1.5), "kappa": np.full(nt, 1.6), "a_m": np.full(nt, 0.5),
         "R_m": np.full(nt, 0.85), "V_m3": np.full(nt, 8.0), "meta": {},
         "lcfs_R": np.tile(np.linspace(0.3, 1.2, 8), (nt, 1)),
         "lcfs_Z": np.tile(np.linspace(-0.6, 0.6, 8), (nt, 1))}
    ts = imas_export.export_imas(b)["equilibrium"]["time_slice"][0]
    assert set(ts["boundary"]) == {"outline", "minor_radius", "elongation"}
    assert "profiles_2d" not in ts and "profiles_1d" not in ts


def test_sidecar_carries_what_the_dictionary_has_no_home_for(shot):
    """Archive flags, provenance labels, meta and the unmapped measured series ride the
    sidecar in house units; mapped fields never appear there."""
    flags, prov = {"useful": 1, "abort": 0}, {"Ip_MA": "measured", "H98": "computed"}
    side = imas_export.fusionlab_annotations(shot, archive_flags=flags, provenance=prov, attribution="attr")
    assert side["archive_flags"] == flags and side["provenance"] == prov
    assert side["attribution"] == "attr" and side["bundle_version"] == BUNDLE_VERSION
    assert side["shot_id"] == SHOT and side["meta"] == shot["meta"]
    q = side["quantities_without_imas_home"]
    assert q["n_e20"]["unit"] == "1e20 m^-3" and len(q["n_e20"]["values"]) == shot["t_s"].size
    assert q["W_MJ"]["unit"] == "MJ" and q["ts_R"]["unit"] == "m"
    assert set(q) == set(imas_export._NO_IMAS_HOME)   # shot 30166 carries them all
    assert "Ip_MA" not in q and "lcfs_R" not in q and "wall_R" not in q   # mapped: never sidecar'd


def _keys(node) -> set:
    """Every dict key anywhere in a nested structure (structure only — never builds a big string)."""
    if isinstance(node, dict):
        out = set(node)
        for v in node.values():
            out |= _keys(v)
        return out
    if isinstance(node, list):
        out: set = set()
        for v in node:
            out |= _keys(v)
        return out
    return set()


def test_imas_payload_carries_the_sidecars_semantics_only_in_the_comment(payload):
    """Catalog bookkeeping and provenance have no IMAS home: no dict key anywhere in the
    payload is a flag or a label — the comment is the pointer (spec criterion)."""
    assert not {"useful", "abort", "provenance"} & _keys(payload)
    comment = payload["ids_properties"]["comment"]
    assert "fusionlab_annotations.json" in comment and "30166" in comment
    assert "compared with MAST data" in comment and "validated or predictive" in comment
