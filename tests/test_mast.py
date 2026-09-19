"""Offline tests on the committed FAIR-MAST cache (no network)."""

import numpy as np
import pytest

from fusionlab import mast

SHOT = 30420  # ohmic, 700 kA


@pytest.fixture(scope="module")
def shot():
    return mast.load_shot(SHOT)


def test_showcase_shots_are_cached():
    assert SHOT in mast.cached_shots() and len(mast.cached_shots()) >= 3


def test_time_base_and_shapes(shot):
    t = shot["t_s"]
    assert t.size > 20 and np.all(np.diff(t) > 0)
    for k in ("Ip_MA", "B_T", "n_e20", "P_ohm_MW", "P_nbi_MW", "W_MJ", "q95", "kappa", "a_m", "R_m", "V_m3"):
        assert shot[k].shape == t.shape, k
    assert shot["lcfs_R"].shape == shot["lcfs_Z"].shape and shot["lcfs_R"].shape[0] == t.size
    assert shot["psi_n"].shape == (t.size, shot["psi_Z"].size, shot["psi_R"].size)


def test_project_units_are_sane_for_mast(shot):
    """MAST: ~0.4-1 MA, ~0.45 T, few 1e19 m^-3, tens of kJ, R ~0.85 m, a ~0.6 m. Catches a missed SI conversion."""
    i = shot["t_s"].size // 2
    assert 0.3 < shot["Ip_MA"][i] < 1.5
    assert 0.3 < shot["B_T"][i] < 0.7            # sign removed
    assert 0.05 < shot["n_e20"][i] < 1.0
    assert 0.005 < shot["W_MJ"][i] < 0.3         # plasma_energy, not the diamagnetic "wmhd"
    assert 0.1 < shot["P_ohm_MW"][i] < 5.0
    assert 0.1 < shot["Te0_keV"][i] < 3.0
    assert 0.6 < shot["R_m"][i] < 1.0 and 0.4 < shot["a_m"][i] < 0.7 and 1.4 < shot["kappa"][i] < 2.5


def test_flux_map_is_oriented_time_z_r(shot):
    """psi_n is 0 on the magnetic axis: its minimum must sit on EFIT's own axis position."""
    i = shot["t_s"].size // 2
    iz, ir = np.unravel_index(np.nanargmin(shot["psi_n"][i]), shot["psi_n"][i].shape)
    assert abs(shot["psi_R"][ir] - shot["R_mag_m"][i]) < 0.06
    assert abs(shot["psi_Z"][iz] - shot["Z_mag_m"][i]) < 0.06


def test_ohmic_shot_has_no_beams_and_keeps_its_logbook(shot):
    assert shot["meta"]["heating"] == "Ohmic" and np.all(shot["P_nbi_MW"] == 0)
    assert "flux" in shot["meta"]["postshot"]


def test_shot_table_cleaning():
    db = mast.load_db()
    c = mast.clean_db(db)
    assert db["shot_id"].size > 10_000 and 1_000 < c["shot_id"].size < db["shot_id"].size
    assert np.all(np.isfinite(c["tau_E_s"])) and np.all(c["P_loss_MW"] > 0)
    # stored tau_E is W / (P_ohm + P_nbi - dW/dt)
    assert np.median(np.abs(c["W_MJ"] / c["P_loss_MW"] / c["tau_E_s"] - 1)) < 0.05
