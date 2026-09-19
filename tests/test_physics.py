import numpy as np
import pytest

from fusionlab.physics import DEVICES, Controls, simulate, sigmav_dt


def test_iter_baseline_is_calibrated():
    r = simulate(Controls(device="iter"))
    assert 7 <= r["Q"] <= 13
    assert 350 <= r["P_fus_MW"] <= 650
    assert r["hmode"] and not r["disruption"]


def test_dt_reactivity_peaks_near_65_keV():
    T = np.linspace(10, 200, 400)
    assert 50 < T[np.argmax(sigmav_dt(np.clip(T, 0, 100)))] <= 100


def test_greenwald_violation_disrupts():
    r = simulate(Controls(device="iter", n=1.5))
    assert r["f_greenwald"] > 1 and r["disruption"]


def test_low_q95_disrupts():
    r = simulate(Controls(device="diiid", Ip=2.0, B=0.8, n=0.5, P_aux=10))
    assert r["q95"] < 2 and r["disruption"]


def test_more_heating_crosses_lh_threshold():
    lo = simulate(Controls(device="diiid", Ip=1.2, B=2.0, n=0.5, P_aux=0.5))
    hi = simulate(Controls(device="diiid", Ip=1.2, B=2.0, n=0.5, P_aux=15))
    assert not lo["hmode"] and hi["hmode"]


@pytest.mark.parametrize("key", list(DEVICES))
def test_vectorized_sweep_matches_scalar(key):
    d = DEVICES[key]
    P = np.linspace(1, d.P_aux_max, 25)
    sweep = simulate(Controls(device=key, Ip=0.8 * d.Ip_max, B=d.B_max, n=0.5), P_aux=P)
    one = simulate(Controls(device=key, Ip=0.8 * d.Ip_max, B=d.B_max, n=0.5, P_aux=float(P[7])))
    assert sweep["T_keV"].shape == (25,)
    assert np.isclose(sweep["T_keV"][7], one["T_keV"])
    assert np.all(np.isfinite(sweep["P_fus_MW"]))
