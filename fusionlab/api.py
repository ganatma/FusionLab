"""FastAPI app: serves the physics engine (and later the surrogate + coach) and the static web UI."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from fusionlab.agent_config import agent_settings
from fusionlab.api_agent import router as agent_router
from fusionlab.api_replay import router as replay_router
from fusionlab.api_replay import start_warm_up
from fusionlab.api_virtual import router as virtual_router
from fusionlab.physics import DEVICES, Controls, sigmav_dt, simulate

WEB = Path(__file__).resolve().parent.parent / "web"
MAP_MAX = 100  # cap on grid points per axis for /map


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_warm_up()   # a real server start only; TestClient(app) without a context never enters the lifespan
    yield


app = FastAPI(title="FusionLab", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/compute")
def compute_status():
    """Where compute runs right now: the selected provider, the SSH worker's handshake when configured,
    and the fallback state (design §7). Built against a short TTL cache with a single in-flight probe,
    so the UI can poll it without hammering the tunnel — and an unreachable worker is a status here,
    never an error."""
    from fusionlab.compute import (
        status,  # lazy: pulls LocalProvider (numpy), never asyncssh
    )

    return status.snapshot()


@app.get("/devices")
def devices():
    return {k: d.to_dict() for k, d in DEVICES.items()}


@app.get("/simulate")
def simulate_point(device: str = "iter", Ip: float | None = None, B: float | None = None, n: float = 1.0,
                   P_aux: float = 50.0, H: float = 1.0, Zeff: float = 1.7):
    if device not in DEVICES:
        raise HTTPException(404, f"unknown device '{device}'")
    d = DEVICES[device]
    Ip, B = Ip if Ip is not None else d.Ip_max, B if B is not None else d.B_max
    # Generous device-relative floors and caps: exclude overflow-scale inputs, never the UI's
    # operating range (sliders span 0.1-1x Ip_max, 0.2-1x B_max, 0-1x P_aux_max, H 0.5-2, Zeff 1-4).
    # Caps alone miss degenerate small inputs (n = 1e-300), floors alone miss 1e308-scale overflows.
    if not (1e-3 * d.Ip_max <= Ip <= 10 * d.Ip_max and 1e-3 * d.B_max <= B <= 10 * d.B_max
            and 1e-6 <= n <= 1e6 and 0 <= P_aux <= 10 * d.P_aux_max
            and 1e-3 <= H <= 100 and 1 <= Zeff <= 1000):
        raise HTTPException(422, "controls out of physical range")
    c = Controls(device=device, Ip=Ip, B=B, n=n, P_aux=P_aux, H=H, Zeff=Zeff)
    return {"controls": c.__dict__, "result": _strict_json(simulate(c))}


def _grid_json(x, shape):
    """Broadcast to the grid and make it strict-JSON safe: NaN/inf → null (vectorized)."""
    x = np.broadcast_to(np.asarray(x), shape)
    if x.dtype == bool:
        return x.tolist()
    out = x.astype(object)
    out[~np.isfinite(x)] = None
    return out.tolist()


def _strict_json(x):
    """Scalar counterpart of _grid_json: a non-finite number must reach the client as a 422,
    never as Starlette's allow_nan=False encoder raising at render time (an unhandled 500)."""
    if isinstance(x, dict):
        return {k: _strict_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict_json(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        raise HTTPException(422, "result contains a non-finite value")
    return x


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
    # Finiteness belongs in the guard: inf passes every positivity check, but the raw Ip/B/H/Zeff
    # scalars below are not grid-nulled, so they crashed the strict JSON encoder at render (a 500).
    if not (Ip > 0 and B > 0 and H > 0 and Zeff >= 1
            and np.isfinite([Ip, B, H, Zeff]).all()):
        raise HTTPException(422, "need finite Ip > 0, B > 0, H > 0, Zeff >= 1")
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


@app.get("/reactivity")
def reactivity(n: int = 80):
    """D-T reactivity <sigma v>(T) from the engine's Bosch-Hale fit, for the guided study's first figure."""
    T_keV = np.geomspace(1.0, 100.0, int(np.clip(n, 8, 400)))
    return {"T_keV": T_keV.tolist(), "sigmav_m3_s": sigmav_dt(T_keV).tolist(), "source": "Bosch & Hale, Nucl. Fusion 32 (1992) 611"}


# TODO(P1+): /map from the surrogate, /coach


@app.get("/")
def index():
    # The agent panel is additive. With the agent off, the page is the file on disk, byte for byte —
    # zero agent markup, and /static/agent.js below answers 404 (the blueprint's verification row:
    # "`/` renders without the agent panel and agent.js is not served"). With it on, the only delta
    # is one script tag; the panel itself is built by agent.js after /agent/health confirms.
    if not agent_settings().enabled:
        return FileResponse(WEB / "index.html")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    if "</head>" not in html:
        return FileResponse(WEB / "index.html")   # unexpected shape: fail without the panel, never half-served
    tag = '  <script src="/static/agent.js" defer></script>\n</head>'
    return Response(html.replace("</head>", tag, 1), media_type="text/html")


app.include_router(replay_router)
app.include_router(virtual_router)
app.include_router(agent_router)


@app.get("/static/agent.js", include_in_schema=False)
def agent_js():
    """The panel script, served only when the agent is enabled — the /static mount would otherwise
    serve it unconditionally. Registered before the mount, so this exact path wins; the offline core
    loop's other static files are untouched."""
    if not agent_settings().enabled:
        raise HTTPException(404, "the agent is disabled — the assistant panel does not exist")
    return FileResponse(WEB / "agent.js")


class NoDotfiles(StaticFiles):
    """Serve no dotfiles: any dotted path component reports "not found" (404, never 403),
    so a dotfile that ever lands under web/ is inert rather than served."""

    def lookup_path(self, path: str):
        if any(part.startswith(".") for part in PurePosixPath(path).parts):
            return "", None  # same shape as Starlette's own not-found
        return super().lookup_path(path)


app.mount("/static", NoDotfiles(directory=WEB), name="static")
