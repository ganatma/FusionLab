"""Offline tests for the OpenUSD export (Omniverse-compatible) on the committed FAIR-MAST cache."""

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom  # noqa: E402

from fusionlab import mast, usd_export  # noqa: E402

SHOT = 30420  # ohmic, 700 kA


@pytest.fixture(scope="module")
def shot():
    if SHOT not in mast.cached_shots():
        pytest.skip("shot cache missing")
    return mast.load_shot(SHOT)


@pytest.fixture(scope="module")
def stage(shot, tmp_path_factory):
    path = usd_export.export_shot(shot, tmp_path_factory.mktemp("usd") / f"mast_{SHOT}.usda")
    assert path.exists() and path.stat().st_size > 0
    return Usd.Stage.Open(str(path))


def test_stage_metadata_and_provenance(stage, shot):
    root = stage.GetDefaultPrim()
    assert root.GetPath() == "/MAST"
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert (stage.GetStartTimeCode(), stage.GetEndTimeCode()) == (0, shot["t_s"].size - 1)   # one code per slice
    cd = root.GetCustomData()
    assert "CC BY-SA 4.0" in cd["attribution"] and "FAIR-MAST" in cd["attribution"]
    assert cd["shot_id"] == SHOT and cd["postshot"] == shot["meta"]["postshot"]
    assert "CC BY-SA 4.0" in stage.GetRootLayer().comment
    assert {"Vessel", "Coils", "Plasma"} <= {p.GetName() for p in root.GetChildren()}


def test_plasma_is_time_sampled_with_constant_topology(stage, shot):
    nt = shot["t_s"].size
    plasma = UsdGeom.Mesh(stage.GetPrimAtPath("/MAST/Plasma"))
    pts_attr = plasma.GetPointsAttr()
    assert pts_attr.GetNumTimeSamples() == nt
    assert plasma.GetExtentAttr().GetNumTimeSamples() == nt
    assert plasma.GetDisplayColorAttr().GetNumTimeSamples() == nt
    assert not plasma.GetFaceVertexIndicesAttr().GetNumTimeSamples()        # topology written once
    frames = [np.array(pts_attr.Get(k)) for k in (0, nt // 2, nt - 1)]
    assert len({f.shape for f in frames}) == 1 and frames[0].shape == (usd_export.N_POL * usd_export.N_TOR, 3)
    assert np.array(plasma.GetFaceVertexIndicesAttr().Get()).max() == frames[0].shape[0] - 1
    for f in frames:
        assert np.isfinite(f).all()
        R = np.hypot(f[:, 0], f[:, 1])
        assert 0.15 < R.min() and R.max() < 1.6 and np.abs(f[:, 2]).max() < 1.5    # inside the MAST vessel


def test_vertices_do_not_swim(shot):
    """Vertex 0 stays at the outboard midplane and the loop runs counter-clockwise on every slice."""
    for k in (0, shot["t_s"].size // 2, shot["t_s"].size - 1):
        r, z = usd_export.resample_outline(shot["lcfs_R"][k], shot["lcfs_Z"][k], 96, shot["Z_mag_m"][k])
        assert r.shape == z.shape == (96,) and np.isfinite(r).all() and np.isfinite(z).all()
        assert r[0] > 0.98 * r.max() and abs(z[0] - shot["Z_mag_m"][k]) < 1e-6
        assert np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z) > 0


def test_scalars_ride_along(stage, shot):
    prim = stage.GetPrimAtPath("/MAST/Plasma")
    mid = shot["t_s"].size // 2
    assert prim.GetAttribute("fusionlab:Ip_MA").Get(mid) == pytest.approx(shot["Ip_MA"][mid])
    assert prim.GetAttribute("fusionlab:t_s").Get(mid) == pytest.approx(shot["t_s"][mid])
    for name in ("B_T", "n_e20", "W_MJ", "q95"):
        attr = prim.GetAttribute(f"fusionlab:{name}")
        assert attr.GetNumTimeSamples() == int(np.isfinite(shot[name]).sum()), name     # NaN slices are skipped


def test_static_meshes_and_bad_extension(stage, shot, tmp_path):
    for name in ("Vessel", "Coils"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/MAST/{name}"))
        pts, idx = np.array(mesh.GetPointsAttr().Get()), np.array(mesh.GetFaceVertexIndicesAttr().Get())
        assert np.isfinite(pts).all() and idx.min() == 0 and idx.max() == len(pts) - 1
    with pytest.raises(ValueError):
        usd_export.export_shot(shot, tmp_path / "shot.obj")
