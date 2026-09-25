"""Virtual shot (fusionlab/virtual.py): the integrator against an analytic solution, and the what-if on cached shots."""

import numpy as np
import pytest

from fusionlab import mast
from fusionlab.virtual import CLOSURES, RESPONSE, Edit, baseline, gate, gated_laws, in_distribution, integrate, response_laws, run


def test_integrator_matches_the_analytic_rise_and_converges_first_order():
    P, tau, T = 2.0, 0.03, 0.3                      # alpha = 0 makes K the confinement time: W = P tau (1 - exp(-t/tau))
    err = []
    for dt in (2e-3, 1e-3, 5e-4):
        nt = int(round(T / dt)) + 1
        W = integrate(np.full((1, nt), P), np.full((1, nt), tau), 0.0, 0.0, dt)[0]
        exact = P * tau * (1 - np.exp(-np.arange(nt) * dt / tau))
        err.append(np.max(np.abs(W - exact)))
        assert abs(W[-1] / (P * tau) - 1) < 1e-3     # settles on P tau
    assert err[1] < 0.02 * P * tau                  # within 2% of the final level everywhere at the 1 ms default
    assert 1.7 < err[0] / err[1] < 2.3 and 1.7 < err[1] / err[2] < 2.3   # backward Euler: halve the step, halve the error


def test_nonlinear_loss_settles_on_the_scaling_law():
    K, alpha, P = 0.05, 0.69, 3.0                   # tau = K P^-alpha  =>  steady state W = K P^(1 - alpha)
    W = integrate(np.full((1, 400), P), np.full((1, 400), K), alpha, 1e-3)[0]
    assert abs(W[-1] / (K * P ** (1 - alpha)) - 1) < 1e-6
    assert np.all(np.diff(W) >= -1e-12)             # monotone rise, no overshoot from the stiff loss term


@pytest.fixture(scope="module")
def shot():
    return mast.load_shot(30166)


@pytest.fixture(scope="module")
def out(shot):
    return run(shot, [Edit(), Edit(P_nbi=1.3), Edit(P_nbi=0.7), Edit(Ip=1.1), Edit(B=1.1), Edit(nbi_shift_s=-0.03)], closure="ipb98")


def test_shapes_and_no_nan_on_the_efit_time_base(shot, out):
    n, E, L = len(shot["t_s"]), 6, len(response_laws())
    assert out["laws"][:2] == list(RESPONSE)
    assert out["W_blind_MJ"].shape == out["W_anchored_MJ"].shape == (E, L, n)
    assert out["troyon"].shape == out["in_distribution"].shape == (E, n)
    for k in ("W_refly_MJ", "W_blind_MJ", "W_anchored_MJ", "P_nbi_MW", "dW_flat_anchored"):
        assert np.all(np.isfinite(out[k])), k


def test_unedited_run_returns_its_own_baseline(shot, out):
    assert np.allclose(out["W_blind_MJ"][0, 0], out["W_refly_MJ"], rtol=1e-3)             # blind what-if with no edit == re-fly
    assert np.allclose(out["W_blind_MJ"][0, 1], out["W_refly_MJ"], rtol=1e-3)             # ... whichever law carries the response
    flat = out["flat_top"]
    dev = np.abs(out["W_anchored_MJ"][0, 0][flat] / shot["W_MJ"][flat] - 1)
    assert np.median(dev) < 0.02                                                          # anchored no-edit lies on the measurement
    assert np.all(np.abs(out["dW_flat_anchored"][0]) < 0.01)


def test_edits_move_the_stored_energy_the_way_the_laws_say(out):
    dW = out["dW_flat_anchored"]                                                          # (edit, law)
    assert np.all(dW[1] > 0.02) and np.all(dW[2] < -0.02)                                 # more beam power, more energy; less, less
    assert np.all(dW[1] < 0.3 * 0.31 * 1.5)                                               # ... but far less than in proportion (tau ~ P^-0.7)
    assert np.all(dW[3] > 0.03)                                                           # more current confines better under both laws
    assert dW[4, 1] > dW[4, 0] > 0                                                        # field: Valovic (B^1.4) responds more than IPB98 (B^0.15)
    assert np.nanmax(out["troyon"][1]) > np.nanmax(out["troyon"][0])                      # more beam power pushes on the Troyon limit
    assert np.allclose(out["kink"][3], out["kink"][0] * 1.1)                              # q95 ~ B / Ip
    assert np.allclose(out["f_greenwald"][3], out["f_greenwald"][0] / 1.1)                # n_GW ~ Ip


def test_a_batch_equals_its_scenarios_run_one_at_a_time(shot, out):
    one = run(shot, Edit(P_nbi=1.3), closure="ipb98")
    assert np.allclose(one["W_anchored_MJ"][0], out["W_anchored_MJ"][1], rtol=1e-9)
    assert np.allclose(one["W_blind_MJ"][0], out["W_blind_MJ"][1], rtol=1e-9)


def test_earlier_beams_raise_the_energy_early_not_on_the_flat_top(shot, out):
    t, W0, W5 = np.asarray(shot["t_s"]), out["W_anchored_MJ"][0, 0], out["W_anchored_MJ"][5, 0]
    early = (t > 0.10) & (t < 0.14)
    assert np.all(W5[early] > W0[early]) and abs(out["dW_flat_anchored"][5, 0]) < 0.03


def test_ramps_and_big_edits_are_flagged_out_of_distribution(shot):
    ok = in_distribution(shot, [Edit(), Edit(Ip=1.5)])
    t = np.asarray(shot["t_s"])
    assert not ok[0][t < 0.08].any() and ok[0].any()      # the current ramp is outside the training range, the flat top inside
    assert ok[1].sum() < ok[0].sum()                      # 1.5x the current leaves the range the correction ever saw


def test_every_closure_reflies_and_an_ohmic_shot_has_no_beam_window():
    ohmic = mast.load_shot(30420)
    for c in CLOSURES:
        b = baseline(ohmic, c)
        assert np.all(np.isfinite(b["W_blind"])) and b["W_blind"].max() > 0
    err = run(ohmic, closure="iter89")["refly_error"]
    assert err["beam_rise_median_abs_ln"] is None and err["flat_top_median_abs_ln"] < 0.5
    with pytest.raises(ValueError):
        baseline(ohmic, "nope")


# ---- API: the gate is enforced server side, the wording never forecasts a disruption
def test_virtual_endpoint_shapes_wording_and_gate():
    from fastapi.testclient import TestClient

    from fusionlab.api import app
    client = TestClient(app)
    g = client.get("/virtual/gate").json()
    assert set(g["sliders"]) == {"Ip", "B", "n", "P_nbi", "nbi_shift_s"} and g["default_closure"] in CLOSURES
    r = client.post("/virtual/30192", json={"edit": {"P_nbi": 1.3}})
    assert r.status_code == 200 and "NaN" not in r.text
    j = r.json()
    n = len(j["t_s"])
    assert len(j["anchored"]["lo"]) == len(j["blind"]["hi"]) == len(j["in_distribution"]) == len(j["limits"]["troyon"]) == n
    assert all(lo <= hi for lo, hi in zip(j["anchored"]["lo"], j["anchored"]["hi"]))
    assert j["dW_flat_pct"]["anchored"][0] > 0 and j["would_cross_at_s"]["troyon"] is not None   # more power on a shot already over the limit
    text = (r.text + client.get("/virtual/gate").text).lower()
    assert "would cross" in text and "will disrupt" not in text and "predicts a disruption" not in text
    assert client.post("/virtual/30192", json={}).json()["is_identity"] is True
    assert client.post("/virtual/1", json={}).status_code == 404
    assert client.post("/virtual/30192", json={"edit": {"P_nbi": 9}}).status_code == 422
    assert client.post("/virtual/30192", json={"closure": "nope"}).status_code == 422
    for slider, info in g["sliders"].items():   # a slider the matched pairs cannot support is refused, not just greyed out
        if not info["enabled"]:
            assert client.post("/virtual/30192", json={"edit": {slider: 1.1 if slider != "nbi_shift_s" else 0.01}}).status_code == 422, slider


def test_gated_laws_never_use_an_exponent_the_matched_pairs_contradict():
    g, laws, raw = gate(), gated_laws(), response_laws()
    assert set(laws) == set(raw)
    for act, k in {"Ip": 0, "B": 1, "n": 2, "P": 3}.items():
        ok = g.get(act, {}).get("admissible") or []
        for name in laws:
            if ok and name not in ok:
                assert laws[name][k] == raw[g[act]["best"]][k], (act, name)
            else:
                assert laws[name][k] == raw[name][k], (act, name)
