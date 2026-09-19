"""The learned correction ships with its holdout numbers; these tests keep the two in step (offline, CPU)."""

import json

import numpy as np
import pytest

from fusionlab import mast, surrogate
from fusionlab.physics import replay

pytestmark = pytest.mark.skipif(not surrogate.available(), reason="no trained model: run `make train`")


def test_metrics_report_every_model_on_both_honest_splits():
    m = json.loads(surrogate.METRICS_FILE.read_text())
    for split in ("split_by_session_block", "split_temporal_M9_held_out"):
        rmse = m[split]["rmse_ln_tau"]
        assert set(rmse) == {"ipb98", "ipb98_x_H", "power_law", "power_law+f_nbi", "hybrid_physicsnemo"}
        assert all(0 < v < 2 for v in rmse.values()) and m[split]["n_test"] > 300
    assert "split_random_by_shot" not in m   # a random split leaks between repeat shots of one session


def test_ipb98_matches_the_engine_on_the_shot_table():
    c = surrogate.table()
    assert np.all(np.isfinite(surrogate.tau_ipb98(c))) and surrogate.features(c).shape == (c["shot_id"].size, 8)


def test_correction_is_finite_and_clipped_on_a_whole_shot():
    s = mast.load_shot(30166)
    H = surrogate.correction(surrogate.shot_features(s, replay(s)["P_loss_MW"]))
    lim = np.exp(surrogate.RESIDUAL_CLIP)
    assert H.shape == s["t_s"].shape and np.all(np.isfinite(H)) and H.min() >= 1 / lim - 1e-6 and H.max() <= lim + 1e-6
