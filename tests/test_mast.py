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
    """Mid-shot values sit inside the configured machine's unit ranges — catches a missed SI conversion.
    The ranges live in MachineConfig.unit_sanity (formerly hard-coded here): a second machine carries
    its own bounds, and the contract test asserts this table against every cached fixture."""
    i = shot["t_s"].size // 2
    for field, lo, hi in mast.CONFIG.unit_sanity:
        v = shot[field][i]
        assert lo < v < hi, (field, float(v))


def test_flux_map_is_oriented_time_z_r(shot):
    """psi_n is 0 on the magnetic axis: its minimum must sit on EFIT's own axis position."""
    i = shot["t_s"].size // 2
    iz, ir = np.unravel_index(np.nanargmin(shot["psi_n"][i]), shot["psi_n"][i].shape)
    assert abs(shot["psi_R"][ir] - shot["R_mag_m"][i]) < 0.06
    assert abs(shot["psi_Z"][iz] - shot["Z_mag_m"][i]) < 0.06


def test_ohmic_shot_has_no_beams_and_keeps_its_logbook(shot):
    assert shot["meta"]["heating"] == "Ohmic" and np.all(shot["P_nbi_MW"] == 0)
    assert "flux" in shot["meta"]["postshot"]


def test_savez_writes_the_exact_temp_path(tmp_path):
    """np.savez_compressed(Path) would append .npz and rename the temp file out from under atomic_write."""
    t = tmp_path / ".shot.npz.tmp123"
    mast._savez(t, a=np.arange(3))
    assert t.exists() and not (tmp_path / ".shot.npz.tmp123.npz").exists()
    with np.load(t) as z:
        assert z["a"].tolist() == [0, 1, 2]


def test_load_shot_fetches_uncached_catalog_shot_and_caches_it(monkeypatch, tmp_path):
    """A2 criterion 4: load_shot() is the app's only fetch path — cache-then-archive, mocked offline here."""
    uncached = next(i for i in sorted(int(x) for x in mast.load_db()["shot_id"]) if i > 30000 and i not in mast.cached_shots())
    real = mast.load_shot(30420)
    calls = []

    def fake_fetch(shot_id):
        calls.append(int(shot_id))
        return {**real, "meta": {**real["meta"], "shot_id": int(shot_id)}}

    monkeypatch.setattr(mast, "SHOTS", tmp_path)
    monkeypatch.setattr(mast, "fetch_shot", fake_fetch)
    s = mast.load_shot(uncached)
    assert calls == [uncached]
    assert s["meta"]["shot_id"] == uncached
    assert np.array_equal(s["Ip_MA"], real["Ip_MA"])          # fetched shots carry identical provenance and units
    assert uncached in mast.cached_shots() and (tmp_path / f"{uncached}.npz").exists()
    assert mast.load_shot(uncached)["meta"]["shot_id"] == uncached and calls == [uncached]   # second call: pure cache hit


def test_shot_table_cleaning():
    db = mast.load_db()
    c = mast.clean_db(db)
    assert db["shot_id"].size > 10_000 and 1_000 < c["shot_id"].size < db["shot_id"].size
    assert np.all(np.isfinite(c["tau_E_s"])) and np.all(c["P_loss_MW"] > 0)
    # stored tau_E is W / (P_ohm + P_nbi - dW/dt)
    assert np.median(np.abs(c["W_MJ"] / c["P_loss_MW"] / c["tau_E_s"] - 1)) < 0.05
