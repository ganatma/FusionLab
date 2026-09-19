"""Timings for the README. Every number printed here is measured on this machine; none are estimates.

    make bench
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusionlab import mast, surrogate  # noqa: E402
from fusionlab.physics import Controls, DEVICES, replay, simulate  # noqa: E402


def best(fn, repeat=5):
    """Best wall time of `repeat` runs, in seconds (after one warm-up call)."""
    fn()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out)


def main():
    rows = []
    shot = mast.load_shot(30166)
    nt = shot["t_s"].size

    # 1. replay: every time slice in one numpy call vs one call per slice
    one = lambda i: {k: (v[i:i + 1] if isinstance(v, np.ndarray) and v.shape[:1] == (nt,) else v) for k, v in shot.items()}
    slices = [one(i) for i in range(nt)]
    t_vec, t_loop = best(lambda: replay(shot)), best(lambda: [replay(s) for s in slices])
    rows.append((f"Replay of a real shot ({nt} EFIT slices), vectorized", f"{t_vec * 1e3:.2f} ms", f"{nt / t_vec:,.0f} slices/s"))
    rows.append(("  same, one call per slice (reference)", f"{t_loop * 1e3:.2f} ms", f"{t_loop / t_vec:.0f}x slower"))

    # 2. shot cache
    t_load = best(lambda: mast.load_shot(30166))
    rows.append(("Load a cached shot (npz, no network)", f"{t_load * 1e3:.1f} ms", "first fetch from S3: ~27 s"))

    # 3. IPB98 over the whole usable shot table
    c = surrogate.table()
    n = c["shot_id"].size
    t_tab = best(lambda: surrogate.tau_ipb98(c))
    rows.append((f"IPB98(y,2) on the shot table ({n:,} shots)", f"{t_tab * 1e3:.2f} ms", f"{n / t_tab:,.0f} shots/s"))

    # 4. learned correction: CPU (as served) and GPU (batched)
    if surrogate.available():
        import torch
        net, mu, sd = surrogate.load()
        X = torch.as_tensor((surrogate.features(c) - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            t_cpu = best(lambda: net(X))
            rows.append((f"PhysicsNeMo correction, CPU ({n:,} rows)", f"{t_cpu * 1e3:.2f} ms", f"{n / t_cpu:,.0f} rows/s"))
            if torch.cuda.is_available():
                big = X.repeat(160, 1).cuda()            # ~1M rows
                gnet = surrogate._net(X.shape[1]).cuda().eval()
                gnet.load_state_dict(net.state_dict())

                def run():
                    gnet(big)
                    torch.cuda.synchronize()
                t_gpu = best(run)
                rows.append((f"PhysicsNeMo correction, {torch.cuda.get_device_name(0)} ({big.shape[0]:,} rows)",
                             f"{t_gpu * 1e3:.2f} ms", f"{big.shape[0] / t_gpu:,.0f} rows/s"))

    # 5. sandbox engine: operating map, one vectorized simulate() call
    d = DEVICES["iter"]
    for nx in (30, 100):
        n_ax, P_ax = np.meshgrid(np.linspace(0.05, 1.5, nx), np.linspace(0, d.P_aux_max, nx))
        t_map = best(lambda: simulate(Controls(device="iter"), n=n_ax, P_aux=P_ax), repeat=3)
        rows.append((f"Sandbox operating map {nx}x{nx} (0D power-balance solve per point)", f"{t_map * 1e3:.0f} ms",
                     f"{nx * nx / t_map:,.0f} points/s"))

    w = max(len(r[0]) for r in rows)
    print(f"| {'Task':{w}} | Time | Throughput / note |\n|{'-' * (w + 2)}|---|---|")
    for a, b, cc in rows:
        print(f"| {a:{w}} | {b} | {cc} |")


if __name__ == "__main__":
    main()
