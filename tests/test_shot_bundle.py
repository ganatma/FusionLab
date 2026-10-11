"""ShotBundle v1 contract tests (DATA-1).

The contract is asserted against every cached fixture shot, discovered via
`mast.cached_shots()` — no hard-coded ids — and against synthetic bundles that
exercise the failure modes a second machine's first export will hit. The
validators are the same ones the reader applies on every `load_shot()` read.
"""

import numpy as np
import pytest

from fusionlab import mast
from fusionlab.data.bundle import (
    BUNDLE_VERSION,
    REQUIRED,
    SPEC,
    BundleValidationError,
    ShotBundle,
)


@pytest.fixture(params=mast.cached_shots(), ids=lambda s: f"shot-{s}")
def cached_shot(request):
    return mast.load_shot(request.param)


def _core_only_bundle(nt: int = 50) -> dict:
    """The 14 required fields and nothing else — a machine with no equilibrium
    reconstructions, no wall data, no profiles. Values are contract units, plausible scale."""
    return {
        "t_s": np.linspace(0.0, 0.5, nt),
        "Ip_MA": np.linspace(0.4, 0.8, nt),
        "B_T": np.full(nt, 0.5),
        "n_e20": np.linspace(0.2, 0.4, nt),
        "P_ohm_MW": np.full(nt, 0.5),
        "P_nbi_MW": np.zeros(nt),
        "W_MJ": np.linspace(0.05, 0.1, nt),
        "dWdt_MW": np.full(nt, 0.01),
        "q95": np.full(nt, 5.0),
        "beta_N": np.full(nt, 1.5),
        "kappa": np.full(nt, 1.8),
        "a_m": np.full(nt, 0.6),
        "R_m": np.full(nt, 0.9),
        "V_m3": np.full(nt, 8.0),
        "meta": {"shot_id": "K1", "campaign": "K1"},
    }


def test_every_cached_fixture_is_a_valid_v1_bundle(cached_shot):
    """Every cached fixture validates against the v1 contract, discovered via cached_shots()."""
    b = ShotBundle.from_dict(cached_shot, source="fair-mast")
    assert b.version == BUNDLE_VERSION == 1
    assert b.has_equilibrium                        # the shipped fixtures carry reconstructions
    assert str(cached_shot["meta"]["shot_id"]) == b.shot_id


def test_cached_fixture_traces_share_the_time_base(cached_shot):
    nt = cached_shot["t_s"].size
    assert np.all(np.diff(cached_shot["t_s"]) > 0)
    for name in REQUIRED:
        assert isinstance(cached_shot[name], np.ndarray) and cached_shot[name].shape == (nt,), name


def test_unknown_fields_do_not_break_the_contract(cached_shot):
    """A field not in SPEC is not part of the bundle: from_dict ignores it (forward-compatible reads)."""
    b = ShotBundle.from_dict({**cached_shot, "not_in_the_contract": 1}, source="fair-mast")
    assert not hasattr(b, "not_in_the_contract")
    assert len([f for f in dir(b) if f in SPEC]) == len(SPEC)


def test_a_core_only_bundle_is_valid_without_equilibrium():
    """No equilibrium, no optional groups — still a valid v1 bundle (the ψ-free machine case)."""
    b = ShotBundle.from_dict(_core_only_bundle(), source="user-file")
    assert not b.has_equilibrium
    assert b.source == "user-file" and b.shot_id == "K1"


def test_missing_required_fields_fail_by_name():
    d = _core_only_bundle()
    del d["W_MJ"], d["kappa"]
    with pytest.raises(BundleValidationError, match=r"W_MJ[\s\S]*kappa|kappa[\s\S]*W_MJ"):
        ShotBundle.from_dict(d, source="user-file")


def test_wrong_shape_fails_by_name():
    d = _core_only_bundle()
    d["R_m"] = np.full(7, 0.9)
    with pytest.raises(BundleValidationError, match="R_m"):
        ShotBundle.from_dict(d, source="user-file")


def test_non_monotonic_time_base_fails():
    d = _core_only_bundle()
    d["t_s"][3] = d["t_s"][2]
    with pytest.raises(BundleValidationError, match="strictly increasing"):
        ShotBundle.from_dict(d, source="user-file")


def test_half_an_equilibrium_group_fails_by_name():
    """Groups travel together: a bundle with one half of (lcfs_R, lcfs_Z) names the field."""
    d = _core_only_bundle()
    d["lcfs_R"] = np.zeros((5, 8))
    with pytest.raises(BundleValidationError, match="lcfs"):
        ShotBundle.from_dict(d, source="user-file")


def test_non_floating_traces_fail():
    d = _core_only_bundle()
    d["Ip_MA"] = d["Ip_MA"].astype(np.int64)
    with pytest.raises(BundleValidationError, match="Ip_MA"):
        ShotBundle.from_dict(d, source="user-file")


def test_unit_sanity_table_holds_on_every_cached_fixture():
    """The configured machine's unit ranges hold mid-shot on every cached fixture — the
    SI-conversion guard the MAST test reframe asserts, here over the whole cache."""
    for sid in mast.cached_shots():
        s = mast.load_shot(sid)
        i = s["t_s"].size // 2
        for field, lo, hi in mast.CONFIG.unit_sanity:
            v = float(s[field][i])
            assert lo < v < hi, (sid, field, v)
