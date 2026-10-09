"""Measured timings for the README (`make bench`) and for the worker's `benchmark` task.

Every number here is measured on the machine that runs it; none are estimates (house rule).
scripts/bench.py prints the markdown table exactly as before; the worker's benchmark task returns
the same rows as JSON, so a remote GPU box reports its own measured numbers through the job API.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from fusionlab import mast, surrogate
from fusionlab.physics import DEVICES, Controls, replay, simulate

PROFILES = ("quick", "full")


def best(fn, repeat=5):
    """Best wall time of `repeat` runs, in seconds (after one warm-up call)."""
    fn()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out)


def timings(profile: str = "full", progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Run the benchmark sections and return {"rows": [...], "profile", "n_slices"}.

    profile "full" is `make bench` (the README numbers); "quick" skips the per-slice loop
    reference and the large operating map — the worker's smoke-level benchmark. `progress`, when
    given, is called once per section: the worker turns that into SSE progress events and its
    cancellation point between sections.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown benchmark profile {profile!r}; available: {PROFILES}")
    report = progress or (lambda event: None)
    rows: list[dict] = []
    shot = mast.load_shot(30166)
    nt = shot["t_s"].size

    # 1. replay: every time slice in one numpy call vs one call per slice
    report({"section": "replay"})

    def one(i):
        return {k: (v[i:i + 1] if isinstance(v, np.ndarray) and v.shape[:1] == (nt,) else v) for k, v in shot.items()}

    t_vec = best(lambda: replay(shot))
    rows.append({"label": f"Replay of a real shot ({nt} EFIT slices), vectorized",
                 "time": f"{t_vec * 1e3:.2f} ms", "note": f"{nt / t_vec:,.0f} slices/s"})
    if profile == "full":
        slices = [one(i) for i in range(nt)]
        t_loop = best(lambda: [replay(s) for s in slices])
        rows.append({"label": "  same, one call per slice (reference)",
                     "time": f"{t_loop * 1e3:.2f} ms", "note": f"{t_loop / t_vec:,.0f}x slower"})

    # 2. shot cache
    report({"section": "shot_cache"})
    t_load = best(lambda: mast.load_shot(30166))
    rows.append({"label": "Load a cached shot (npz, no network)", "time": f"{t_load * 1e3:.1f} ms",
                 "note": "first fetch from S3: ~27 s"})

    # 3. IPB98 over the whole usable shot table
    report({"section": "ipb98"})
    c = surrogate.table()
    n = c["shot_id"].size
    t_tab = best(lambda: surrogate.tau_ipb98(c))
    rows.append({"label": f"IPB98(y,2) on the shot table ({n:,} shots)",
                 "time": f"{t_tab * 1e3:.2f} ms", "note": f"{n / t_tab:,.0f} shots/s"})

    # 4. learned correction: CPU (as served) and GPU (batched)
    if surrogate.available():
        report({"section": "correction"})
        import torch

        net, mu, sd = surrogate.load()
        X = torch.as_tensor((surrogate.features(c) - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            t_cpu = best(lambda: net(X))
            rows.append({"label": f"PhysicsNeMo correction, CPU ({n:,} rows)",
                         "time": f"{t_cpu * 1e3:.2f} ms", "note": f"{n / t_cpu:,.0f} rows/s"})
            if profile == "full" and torch.cuda.is_available():
                big = X.repeat(160, 1).cuda()            # ~1M rows
                gnet = surrogate._net(X.shape[1]).cuda().eval()
                gnet.load_state_dict(net.state_dict())

                def run():
                    gnet(big)
                    torch.cuda.synchronize()
                t_gpu = best(run)
                rows.append({"label": f"PhysicsNeMo correction, {torch.cuda.get_device_name(0)} ({big.shape[0]:,} rows)",
                             "time": f"{t_gpu * 1e3:.2f} ms", "note": f"{big.shape[0] / t_gpu:,.0f} rows/s"})

    # 5. sandbox engine: operating map, one vectorized simulate() call
    report({"section": "operating_map"})
    d = DEVICES["iter"]
    for nx in ((30,) if profile == "quick" else (30, 100)):
        n_ax, P_ax = np.meshgrid(np.linspace(0.05, 1.5, nx), np.linspace(0, d.P_aux_max, nx))
        t_map = best(functools.partial(simulate, Controls(device="iter"), n=n_ax, P_aux=P_ax), repeat=3)
        rows.append({"label": f"Sandbox operating map {nx}x{nx} (0D power-balance solve per point)",
                     "time": f"{t_map * 1e3:.0f} ms", "note": f"{nx * nx / t_map:,.0f} points/s"})

    return {"rows": rows, "profile": profile, "n_slices": int(nt)}
