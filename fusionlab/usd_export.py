"""OpenUSD export (Omniverse-compatible): one real MAST shot as a time-sampled USD stage.

    uv run python -m fusionlab.usd_export 30166            # -> out/mast_30166.usda (text, diffable)
    uv run python -m fusionlab.usd_export 30166 --usdc     # -> out/mast_30166.usdc (binary crate, smaller)

Stage layout (Z up, metres, one time code per EFIT slice):
  /MAST          Xform; provenance in customData (shot id, campaign, heating, logbook text, data licence)
  /MAST/Vessel   limiter outline (wall_R, wall_Z) revolved about Z; grey, translucent, cut away
  /MAST/Coils    every PF winding-pack rectangle revolved about Z, merged into ONE mesh
  /MAST/Plasma   EFIT last closed flux surface revolved about Z. points, extent and displayColor (from
                 Te0_keV) are time-sampled, so the shot plays in usdview / USD Composer / Blender.
                 Measured scalars ride along as time-sampled fusionlab:* attributes, unit in the name.
  /MAST/Looks    UsdPreviewSurface materials repeating the display colours, for viewers that shade from
                 materials rather than displayColor

Units are the project's: m, T, MA, 1e20 m^-3, keV, MW (plus MJ, s). Time codes count EFIT slices, not
seconds (slices are ~5 ms apart and not always evenly spaced); the real time is fusionlab:t_s.

What this is not: the plasma surface is the axisymmetric EFIT boundary swept around the torus. Nothing 3D
is measured here, and it is an educational twin, not a predictive code.

Data: UKAEA FAIR-MAST (https://mastapp.site), licence CC BY-SA 4.0. The exported stage is derived from
it and carries the same licence; the attribution string is written into the file.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

ATTRIBUTION = "MAST data: UKAEA FAIR-MAST (mastapp.site), CC BY-SA 4.0"
GENERATOR = "FusionLab OpenUSD export (Omniverse-compatible)"
OUT = Path(__file__).resolve().parent.parent / "out"
FORMATS = (".usda", ".usdc", ".usd")

N_POL = 96             # poloidal points on the plasma outline after resampling (fixed -> constant topology)
N_TOR = 64             # toroidal segments for a full turn (vessel, plasma)
N_TOR_COIL = 36        # coils are thin rings, fewer segments is enough and there are 812 of them
CUTAWAY_DEG = 90.0     # vessel and coils leave this wedge open so the plasma is visible even in viewers
                       # that ignore displayOpacity (Blender's importer does). 0 = full 360 degrees.
PLAYBACK_FPS = 12.0    # time codes per second: a 0.35 s shot plays in ~6 s instead of a blink
ROUND_M = 3            # write geometry to 1 mm; EFIT is good to ~1 cm, and the .usda shrinks by a third (23 -> 16 MB)

VESSEL_COLOR, VESSEL_OPACITY = (0.62, 0.64, 0.67), 0.15
COIL_COLOR = (0.72, 0.45, 0.20)   # copper

# Core electron temperature -> colour. Fixed absolute scale so two shots can be compared by eye.
TE_RAMP_MAX_KEV = 1.5
_RAMP_X = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_RAMP_RGB = np.array([[0.35, 0.02, 0.02],    # dark red
                      [0.80, 0.16, 0.04],
                      [1.00, 0.50, 0.08],    # orange
                      [1.00, 0.82, 0.30],
                      [1.00, 0.97, 0.80]])   # yellow-white

# Time-sampled scalars on /MAST/Plasma, written as fusionlab:<name>. Missing keys are skipped.
_SCALARS = ("t_s", "Ip_MA", "B_T", "n_e20", "W_MJ", "q95", "beta_N", "Te0_keV", "P_nbi_MW")


# ---------------------------------------------------------------- outlines (R, Z plane)
def _ccw(r, z):
    """Counter-clockwise order (R to the right, Z up), so the revolved faces get outward normals."""
    area2 = np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z)
    return (r, z) if area2 >= 0 else (r[::-1], z[::-1])


def _open_loop(R, Z):
    """Drop NaNs, repeated points and the closing point: a closed outline stored as distinct vertices, CCW."""
    ok = np.isfinite(R) & np.isfinite(Z)
    r, z = np.asarray(R, float)[ok], np.asarray(Z, float)[ok]
    keep = np.r_[True, np.hypot(np.diff(r), np.diff(z)) > 1e-9]   # zero-length segments break arc length
    r, z = r[keep], z[keep]
    if r.size > 1 and np.hypot(r[0] - r[-1], z[0] - z[-1]) < 1e-9:
        r, z = r[:-1], z[:-1]
    return _ccw(r, z)


def resample_outline(R, Z, n: int = N_POL, z_mid: float | None = None):
    """One closed outline -> n points at equal arc length, CCW, starting at the outboard midplane.

    EFIT returns a different number of boundary points on every slice and starts on the inboard side.
    A fixed count, start point and direction keep vertex k at the same place on the surface from frame
    to frame, which is what lets USD animate one mesh instead of swapping meshes.
    """
    r, z = _open_loop(R, Z)
    if r.size < 3:
        raise ValueError("outline has fewer than 3 finite points")
    rc, zc = np.r_[r, r[0]], np.r_[z, z[0]]
    seg = np.hypot(np.diff(rc), np.diff(zc))
    s = np.r_[0.0, np.cumsum(seg)]

    # Start where the outline crosses the midplane going up: on a CCW loop that is the outboard side.
    z0 = float(z_mid) if z_mid is not None and np.isfinite(z_mid) else 0.5 * (z.min() + z.max())
    d = zc - z0
    up = np.flatnonzero((d[:-1] <= 0) & (d[1:] > 0))
    if up.size:
        k = up[np.argmax(rc[up])]
        s0 = s[k] + seg[k] * (-d[k] / (d[k + 1] - d[k]))
    else:                                   # outline never crosses z0 (should not happen): largest R
        s0 = s[np.argmax(r)]
    target = (s0 + s[-1] * np.arange(n) / n) % s[-1]
    return np.interp(target, s, rc), np.interp(target, s, zc)


# ---------------------------------------------------------------- revolve (vectorised)
def _angles(n_full: int, cutaway_deg: float):
    """Toroidal angles and whether the sweep closes on itself. The open wedge sits at x > 0, y < 0."""
    if cutaway_deg <= 0:
        return np.arange(n_full) * (2 * np.pi / n_full), True
    sweep = 2 * np.pi * (1 - cutaway_deg / 360.0)
    m = max(2, int(round(n_full * sweep / (2 * np.pi))))
    return np.linspace(0.0, sweep, m + 1), False


def _revolve(R, Z, phi):
    """Outline(s) (..., n) x angles (m,) -> points (..., m*n, 3). Vertex (toroidal j, poloidal i) is j*n + i."""
    R, Z = np.asarray(R, float), np.asarray(Z, float)
    x = R[..., None, :] * np.cos(phi)[:, None]
    y = R[..., None, :] * np.sin(phi)[:, None]
    z = np.broadcast_to(Z[..., None, :], x.shape)
    return np.stack([x, y, z], axis=-1).reshape(*R.shape[:-1], -1, 3)


def _quads(n_pol: int, n_tor: int, closed_tor: bool):
    """Quad vertex indices (n_faces, 4) for a closed poloidal loop swept over n_tor angle rows.

    Order (j,i) -> (j+1,i) -> (j+1,i+1) -> (j,i+1): toroidal edge first, then poloidal. For a CCW outline
    that is e_phi x t_pol = outward, matching USD's default right-handed orientation.
    """
    j = np.arange(n_tor if closed_tor else n_tor - 1)[:, None]
    i = np.arange(n_pol)[None, :]
    j1, i1 = (j + 1) % n_tor, (i + 1) % n_pol
    return np.stack(np.broadcast_arrays(j * n_pol + i, j1 * n_pol + i, j1 * n_pol + i1, j * n_pol + i1), -1).reshape(-1, 4)


def _coil_mesh(shot: dict, cutaway_deg: float):
    """All PF winding-pack rectangles revolved and concatenated: (points (N,3), quads (F,4))."""
    R, Z, W, H = (np.asarray(shot[k], float) for k in ("coil_R", "coil_Z", "coil_W", "coil_H"))
    ok = np.isfinite(R) & np.isfinite(Z) & (W > 0) & (H > 0) & (R - W / 2 > 0)
    R, Z, W, H = R[ok], Z[ok], W[ok], H[ok]
    sx, sz = np.array([-0.5, 0.5, 0.5, -0.5]), np.array([-0.5, -0.5, 0.5, 0.5])    # corners, CCW
    phi, closed = _angles(N_TOR_COIL, cutaway_deg)
    pts = _revolve(R[:, None] + W[:, None] * sx, Z[:, None] + H[:, None] * sz, phi)  # (n_coil, m*4, 3)
    per = pts.shape[1]
    faces = _quads(4, phi.size, closed)
    if not closed:   # cap the two cut ends so the rings do not look hollow
        last = (phi.size - 1) * 4
        faces = np.vstack([faces, [0, 1, 2, 3], [last + 3, last + 2, last + 1, last]])
    offset = (np.arange(R.size) * per)[:, None, None]
    return pts.reshape(-1, 3), (faces[None] + offset).reshape(-1, 4)


# ---------------------------------------------------------------- colour
def _hold_last_finite(x):
    """Forward-fill NaN with the last finite value (leading NaNs take the first finite one)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.full(x.shape, np.nan)
    idx = np.maximum.accumulate(np.where(ok, np.arange(x.size), 0))
    idx = np.where(np.arange(x.size) < np.argmax(ok), np.argmax(ok), idx)
    return x[idx]


def te_color(Te0_keV):
    """Core T_e (keV) -> RGB (nt, 3): dark red -> orange -> yellow-white. All-NaN falls back to mid-ramp."""
    x = np.nan_to_num(_hold_last_finite(Te0_keV) / TE_RAMP_MAX_KEV, nan=0.5).clip(0, 1)
    return np.stack([np.interp(x, _RAMP_X, _RAMP_RGB[:, c]) for c in range(3)], -1)


# ---------------------------------------------------------------- USD helpers
def _vec3f(a):
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a, dtype=np.float32))


def _extent(p):
    """(..., N, 3) -> (..., 2, 3) axis-aligned bounds."""
    return np.stack([p.min(-2), p.max(-2)], -2)


def _static_mesh(stage, path, pts, quads, color, opacity=None, double_sided=False, materials=True):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = np.round(pts, ROUND_M)
    mesh.CreatePointsAttr(_vec3f(pts))
    _topology(mesh, quads)
    mesh.CreateExtentAttr(_vec3f(_extent(pts)))
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(_vec3f([color]))
    if opacity is not None:
        mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set(Vt.FloatArray([float(opacity)]))
    if double_sided:
        mesh.CreateDoubleSidedAttr(True)
    if materials:
        _material(stage, mesh, color, 1.0 if opacity is None else opacity)
    return mesh


def _material(stage, mesh, color, opacity=1.0):
    """Bind a UsdPreviewSurface with the same colour as displayColor. Returns the shader (to animate it).

    usdview falls back to displayColor on its own; Blender and USD Composer shade from bound materials,
    and without one every mesh imports as the same grey.
    """
    UsdGeom.Scope.Define(stage, "/MAST/Looks")
    base = f"/MAST/Looks/{mesh.GetPrim().GetName()}"
    mat = UsdShade.Material.Define(stage, base)
    shader = UsdShade.Shader.Define(stage, f"{base}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return shader


def _topology(mesh, quads):
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(quads), 4, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(quads.reshape(-1), dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)   # polygons as given; the USD default would smooth them


# ---------------------------------------------------------------- export
def export_shot(shot: dict, path: str | Path, *, n_pol: int = N_POL, n_tor: int = N_TOR,
                cutaway_deg: float = CUTAWAY_DEG, fps: float = PLAYBACK_FPS, materials: bool = True) -> Path:
    """Write one shot (a mast.load_shot dict) as a USD stage. Format follows the extension: .usda/.usdc/.usd.

    OpenUSD export (Omniverse-compatible). Time code k is EFIT slice k; the real time is fusionlab:t_s.
    The stage is built in memory and exported, so calling this repeatedly for the same path is safe.
    """
    path = Path(path)
    if path.suffix.lower() not in FORMATS:
        raise ValueError(f"extension must be one of {FORMATS}, got {path.suffix!r}")
    t = np.asarray(shot["t_s"], float)
    nt = t.size
    if nt == 0:
        raise ValueError("shot has no time slices")
    meta = shot.get("meta", {}) or {}

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(nt - 1)

    root = UsdGeom.Xform.Define(stage, "/MAST").GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind("component")
    prov = {"shot_id": meta.get("shot_id"), "campaign": meta.get("campaign"), "heating": meta.get("heating"),
            "timestamp": meta.get("timestamp"), "preshot": meta.get("preshot"), "postshot": meta.get("postshot"),
            "attribution": ATTRIBUTION, "generator": GENERATOR,
            "time_note": f"time code = EFIT slice index ({nt} slices, t = {t[0]:.4f}-{t[-1]:.4f} s); "
                         "real time in seconds is /MAST/Plasma.fusionlab:t_s",
            "scope_note": "Educational twin. Plasma surface is the axisymmetric EFIT boundary revolved about Z."}
    for k, v in prov.items():
        if v is not None:
            root.SetCustomDataByKey(k, v if isinstance(v, (int, float)) else str(v))
    stage.GetRootLayer().comment = (f"MAST shot {meta.get('shot_id', '?')} ({meta.get('campaign', '?')}, "
                                    f"{meta.get('heating', '?')}). {ATTRIBUTION}. {GENERATOR}.")

    if "wall_R" in shot:
        wr, wz = _open_loop(shot["wall_R"], shot["wall_Z"])     # corners kept as they are, no resampling
        phi, closed = _angles(n_tor, cutaway_deg)
        # double-sided: with the wedge open you look at the inside of the wall
        _static_mesh(stage, "/MAST/Vessel", _revolve(wr, wz, phi), _quads(wr.size, phi.size, closed),
                     VESSEL_COLOR, VESSEL_OPACITY, double_sided=True, materials=materials)
    if "coil_R" in shot and np.size(shot["coil_R"]):
        _static_mesh(stage, "/MAST/Coils", *_coil_mesh(shot, cutaway_deg), COIL_COLOR, materials=materials)

    # Plasma: resample per slice (a loop over ~70 slices; point counts differ), revolve all slices at once.
    z_mid = shot.get("Z_mag_m", np.full(nt, np.nan))
    outl = np.array([resample_outline(shot["lcfs_R"][k], shot["lcfs_Z"][k], n_pol, z_mid[k]) for k in range(nt)])
    phi = np.arange(n_tor) * (2 * np.pi / n_tor)
    pts = np.round(_revolve(outl[:, 0], outl[:, 1], phi), ROUND_M).astype(np.float32)   # (nt, n_tor*n_pol, 3)
    ext = _extent(pts)
    rgb = te_color(shot.get("Te0_keV", np.full(nt, np.nan)))

    plasma = UsdGeom.Mesh.Define(stage, "/MAST/Plasma")
    _topology(plasma, _quads(n_pol, n_tor, True))
    p_attr, e_attr = plasma.CreatePointsAttr(), plasma.CreateExtentAttr()
    c_attr = plasma.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    # Default (un-timed) value = the peak-current slice, for tools that read a stage without a time.
    k0 = int(np.nanargmax(shot["Ip_MA"])) if "Ip_MA" in shot else nt // 2
    p_attr.Set(_vec3f(pts[k0])); e_attr.Set(_vec3f(ext[k0])); c_attr.Set(_vec3f(rgb[k0:k0 + 1]))
    glow = []
    if materials:   # the plasma glows: same colour on diffuse and emissive, animated with the mesh
        shader = _material(stage, plasma, rgb[k0])
        glow = [shader.GetInput("diffuseColor"), shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)]
        glow[1].Set(Gf.Vec3f(*map(float, rgb[k0])))
    for k in range(nt):
        p_attr.Set(_vec3f(pts[k]), k)
        e_attr.Set(_vec3f(ext[k]), k)
        c_attr.Set(_vec3f(rgb[k:k + 1]), k)
        for inp in glow:
            inp.Set(Gf.Vec3f(*map(float, rgb[k])), k)

    prim = plasma.GetPrim()
    for name in _SCALARS:
        if name not in shot:
            continue
        v = np.asarray(shot[name], float)
        attr = prim.CreateAttribute(f"fusionlab:{name}", Sdf.ValueTypeNames.Double, custom=True)
        for k in np.flatnonzero(np.isfinite(v)):      # NaN = not measured on this slice: leave it out
            attr.Set(float(v[k]), int(k))

    path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(path)):
        raise OSError(f"USD export failed: {path}")
    return path


def describe(path: str | Path) -> str:
    """One line per mesh: vertex and face counts and number of time samples on points."""
    stage = Usd.Stage.Open(str(path))
    rows = [f"time codes {stage.GetStartTimeCode():.0f}-{stage.GetEndTimeCode():.0f} at {stage.GetTimeCodesPerSecond():g}/s"]
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            tc = Usd.TimeCode(stage.GetStartTimeCode())
            rows.append(f"{prim.GetPath()}: {len(m.GetPointsAttr().Get(tc))} vertices, "
                        f"{len(m.GetFaceVertexCountsAttr().Get())} faces, "
                        f"{m.GetPointsAttr().GetNumTimeSamples()} time samples")
    return "\n".join(rows)


if __name__ == "__main__":
    from fusionlab import mast

    args = sys.argv[1:]
    ext = ".usdc" if "--usdc" in args else ".usda"
    ids = [int(a) for a in args if a.isdigit()] or [30166]
    for shot_id in ids:
        t0 = time.time()
        out = export_shot(mast.load_shot(shot_id), OUT / f"mast_{shot_id}{ext}")
        print(f"{out}: {out.stat().st_size / 1e6:.2f} MB, {time.time() - t0:.2f} s")
        print(describe(out))
