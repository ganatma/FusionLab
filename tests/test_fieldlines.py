"""Offline tests for Warp field-line tracing through the EFIT reconstruction (committed FAIR-MAST cache).

Runs on Warp's CPU device everywhere; the same checks run on CUDA when a GPU is present.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fusionlab import fieldlines as fl, mast  # noqa: E402

SHOT = 30420  # ohmic, 700 kA
wp.init()
DEVICES = ["cpu"] + ([fl.default_device()] if wp.get_cuda_device_count() and fl.default_device() != "cpu" else [])


@pytest.fixture(scope="module")
def shot():
    if SHOT not in mast.cached_shots():
        pytest.skip("shot cache missing")
    return mast.load_shot(SHOT)


@pytest.mark.parametrize("device", DEVICES)
def test_line_stays_on_its_flux_surface(shot, device):
    mid = shot["t_s"].size // 2
    tr = fl.trace(shot, mid, [0.5], n_turns=5, steps_per_turn=256, device=device)
    assert tr["R_m"].shape == tr["Z_m"].shape == tr["phi_rad"].shape == (1, 5 * 256 + 1)
    assert np.isfinite(tr["R_m"]).all() and not tr["left_grid"][0]
    assert abs(tr["psi_n_start_interp"][0] - 0.5) < 1e-4                   # start point really is on psi_N = 0.5
    assert tr["psi_n_drift"][0] < 0.01                                     # the spec; measured ~1e-5
    # independent of the kernel's own bookkeeping: re-evaluate psi_N along the stored trajectory in numpy
    psi_n = fl.psi_n_at(shot, mid, tr["R_m"][0], tr["Z_m"][0])
    assert np.abs(psi_n - 0.5).max() < 1e-3
    assert tr["R_m"][0, 0] > shot["R_mag_m"][mid] and abs(tr["Z_m"][0, 0] - shot["Z_mag_m"][mid]) < 1e-6
    assert tr["phi_rad"][0, -1] == pytest.approx(5 * 2 * np.pi)


@pytest.mark.parametrize("device", DEVICES)
def test_traced_q95_matches_efit(shot, device):
    mid = shot["t_s"].size // 2
    q = fl.q_traced(shot, mid, 0.95, device=device)
    assert q == pytest.approx(shot["q95"][mid], rel=0.10)                  # the spec; measured 0.1 % on this slice
    assert q == pytest.approx(shot["q95"][mid], rel=0.02)
    q_core = fl.q_traced(shot, mid, 0.3, n_turns=40, device=device)       # q rises from core to edge
    assert 0.5 < q_core < q


def test_whole_shot_check_is_one_batched_launch(shot):
    c = fl.q_check(shot, device="cpu", steps_per_turn=64)                  # coarse steps keep the CPU run short
    s = c["summary"]
    assert s["n_compared"] == s["n_slices"] == int(np.isfinite(shot["q95"]).sum())
    assert s["min_poloidal_transits"] >= 10
    assert s["median_rel_err"] < 0.02 and s["p95_rel_err"] < 0.05
    assert s["psi_n_drift_max"] < 0.01


@pytest.mark.skipif(len(DEVICES) < 2, reason="no CUDA device")
def test_gpu_and_cpu_agree(shot):
    mid = shot["t_s"].size // 2
    a, b = (fl.trace(shot, mid, [0.3, 0.9], n_turns=3, device=d) for d in DEVICES)
    assert np.abs(a["R_m"] - b["R_m"]).max() < 1e-3 and np.abs(a["Z_m"] - b["Z_m"]).max() < 1e-3


def test_scrape_off_layer_line_stops_at_the_wall(shot):
    mid = shot["t_s"].size // 2
    pts, tr = fl.usd_lines(shot, psi_n=(1.02,), direction=(1.0,), device="cpu")
    assert np.isnan(tr["q"][mid])                                          # open line: no whole poloidal transit
    R, Z = np.hypot(pts[mid, 0, :, 0], pts[mid, 0, :, 1]), pts[mid, 0, :, 2]
    assert fl._inside_wall(R, Z, shot["wall_R"], shot["wall_Z"]).all()
    assert np.hypot(np.diff(R), np.diff(Z))[-10:].max() == 0               # held at the strike point
    assert np.abs(Z[-1]) > 1.0                                             # and that is up/down at the divertor


def test_usd_stage_has_time_sampled_field_lines(shot, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    path = fl.export_shot_with_field_lines(shot, tmp_path / f"mast_{SHOT}_fieldlines.usdc", device="cpu")
    stage = Usd.Stage.Open(str(path))
    nt = shot["t_s"].size
    assert {"Vessel", "Coils", "Plasma", "FieldLines"} <= {p.GetName() for p in stage.GetDefaultPrim().GetChildren()}
    curves = UsdGeom.BasisCurves(stage.GetPrimAtPath("/MAST/FieldLines"))
    assert curves.GetTypeAttr().Get() == UsdGeom.Tokens.linear
    assert curves.GetPointsAttr().GetNumTimeSamples() == nt == curves.GetExtentAttr().GetNumTimeSamples()
    assert not curves.GetCurveVertexCountsAttr().GetNumTimeSamples()      # vertex counts written once
    counts = np.array(curves.GetCurveVertexCountsAttr().Get())
    assert counts.size == len(fl.USD_PSI_N) and (counts == counts[0]).all()
    for k in (0, nt // 2, nt - 1):
        p = np.array(curves.GetPointsAttr().Get(k))
        assert p.shape == (counts.sum(), 3) and np.isfinite(p).all()
        R = np.hypot(p[:, 0], p[:, 1])
        assert 0.15 < R.min() and R.max() < 2.0 and np.abs(p[:, 2]).max() < 2.1
    cd = curves.GetPrim().GetCustomData()
    assert cd["provenance"] == fl.PROVENANCE and "NVIDIA Warp" in cd["provenance"] and "CC BY-SA 4.0" in cd["provenance"]
    assert UsdGeom.Mesh(stage.GetPrimAtPath("/MAST/Plasma")).GetPointsAttr().GetNumTimeSamples() == nt   # untouched
