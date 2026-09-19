"""FastAPI app: serves the physics engine (and later the surrogate + coach) and the static web UI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fusionlab.api_replay import router as replay_router
from fusionlab.physics import DEVICES, Controls, simulate

WEB = Path(__file__).resolve().parent.parent / "web"
MAP_MAX = 100  # cap on grid points per axis for /map
app = FastAPI(title="FusionLab", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/devices")
def devices():
    return {k: d.to_dict() for k, d in DEVICES.items()}


@app.get("/simulate")
def simulate_point(device: str = "iter", Ip: float | None = None, B: float | None = None, n: float = 1.0,
                   P_aux: float = 50.0, H: float = 1.0, Zeff: float = 1.7):
    if device not in DEVICES:
        raise HTTPException(404, f"unknown device '{device}'")
    d = DEVICES[device]
    c = Controls(device=device, Ip=Ip if Ip is not None else d.Ip_max, B=B if B is not None else d.B_max,
                 n=n, P_aux=P_aux, H=H, Zeff=Zeff)
    return {"controls": c.__dict__, "result": simulate(c)}


def _grid_json(x, shape):
    """Broadcast to the grid and make it strict-JSON safe: NaN/inf → null (vectorized)."""
    x = np.broadcast_to(np.asarray(x), shape)
    if x.dtype == bool:
        return x.tolist()
    out = x.astype(object)
    out[~np.isfinite(x)] = None
    return out.tolist()


MAP_KEYS = ("Q", "T_keV", "P_fus_MW", "tau_E_s", "f_greenwald", "beta_N", "q95", "worst_limit",
            "hmode", "disruption", "ignited", "radiative_collapse")


@app.get("/map")
def operating_map(device: str = "iter", Ip: float | None = None, B: float | None = None, H: float = 1.0,
                  Zeff: float = 1.7, nx: int = 30, ny: int = 30):
    """Sweep density (x) × heating (y) at fixed Ip, B, H, Zeff. One vectorized engine call."""
    if device not in DEVICES:
        raise HTTPException(404, f"unknown device '{device}'")
    d = DEVICES[device]
    Ip, B = Ip if Ip is not None else d.Ip_max, B if B is not None else d.B_max
    if not (Ip > 0 and B > 0 and H > 0 and Zeff >= 1):
        raise HTTPException(422, "need Ip > 0, B > 0, H > 0, Zeff >= 1")
    nx, ny = int(np.clip(nx, 2, MAP_MAX)), int(np.clip(ny, 2, MAP_MAX))
    n_gw = Ip / (np.pi * d.a**2)  # Greenwald density [1e20 m^-3]; sweep past it so the limit shows
    n_axis = np.linspace(0.05 * n_gw, 1.3 * n_gw, nx)
    P_axis = np.linspace(0.0, d.P_aux_max, ny)
    n, P_aux = np.meshgrid(n_axis, P_axis)  # grids are [ny][nx]: row = P_aux, column = n
    r = simulate(None, device=device, Ip=Ip, B=B, n=n, P_aux=P_aux, H=H, Zeff=Zeff)
    phys = [r["limits"][k] for k in ("greenwald", "troyon", "kink")]
    out = {
        "device": device, "Ip": Ip, "B": B, "H": H, "Zeff": Zeff, "nx": nx, "ny": ny, "n_GW": float(n_gw),
        "n": n_axis.tolist(), "P_aux": P_axis.tolist(),
        "limits": {k: _grid_json(r["limits"][k], n.shape) for k in ("greenwald", "troyon", "kink")},
        # worst_limit also holds the engineering limits (== 1.0 at full field/current), so the
        # disruption boundary is drawn from the three stability limits only.
        "stability_limit": _grid_json(np.max(np.stack(np.broadcast_arrays(*phys)), axis=0), n.shape),
    }
    out.update({k: _grid_json(r[k], n.shape) for k in MAP_KEYS})
    return out


# TODO(P1+): /map from the surrogate, /missions, /missions/{id}/check, /coach


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.include_router(replay_router)
app.mount("/static", StaticFiles(directory=WEB), name="static")
