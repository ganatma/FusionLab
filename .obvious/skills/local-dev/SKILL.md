---
name: local-dev
description: How to get FusionLab running locally (FastAPI + static web UI, uv-managed Python 3.12, no external services)
---

# local-dev — FusionLab

Recorded 2026-10-09 from a fresh-checkout onboarding run that reached `dev_stack_healthy: true`.

## Prerequisites

1. **uv** — not preinstalled. `curl -LsSf https://astral.sh/uv/install.sh | sh` puts it in `~/.local/bin/uv`;
   add that to PATH (Makefile calls bare `uv`).
2. **Python 3.12** — system python is 3.13, outside `requires-python = ">=3.11,<3.13"`. Do NOT pip-install deps
   globally; `uv sync` provisions and caches 3.12 itself.
3. **No GPU needed.** Without CUDA the server logs
   `Warp CUDA warning: Could not find or load the NVIDIA CUDA driver` once at startup — benign; replay and tests
   are CPU/vectorized. Force a device with `FUSIONLAB_DEVICE=cpu` only if torch autodetect misbehaves.

## Steps that worked

```bash
make setup    # uv sync (torch+physicsnemo+warp download ~3-4 min), sets git hooks, copies .env
make dev      # uvicorn fusionlab.api:app --reload --port 8000
```

- Parse the real port from the log line `Uvicorn running on http://127.0.0.1:8000` — do not assume it.
- No lock files to clear; no services to start; data and trained models ship in the repo. Offline after setup.

## Verify (the primary loop)

1. `curl http://127.0.0.1:8000/health` → `{"ok": true}`.
2. `curl .../shots` then `.../replay/30166` → 200 with `measured/model/psi_grid/lcfs/summary` keys (~170 KB, <1 s).
3. `.../simulate?device=iter&n=1.2&P_aux=120` → sanity: `Q` ≈ 9.2 at those controls.
4. Browser: `/` shows the first-visit gate ("New to fusion" / "I know tokamaks"); after choosing Lab, the Replay tab
   loads shot 30166. Headless-chromium screenshots from the run: `/tmp/fl_firstvisit.png`, `/tmp/fl_replay.png`.
5. `make test` → 88 passed, 1 skipped, ~19 s. There is no separate lint/typecheck config; pytest is the gate.

## Gotchas

- Playwright `wait_until="networkidle"` intermittently times out on this app (server-sent updates keep sockets
  busy); prefer `domcontentloaded` + a fixed ~8 s wait.
- `uv run` inside tmux needs `PATH="$HOME/.local/bin:$PATH"` exported in the same shell.
- `make validate-virtual` and `scripts/fetch_eq_dataset.py` fetch from FAIR-MAST S3 (~120 shots / 2.7 GB) — not
  part of getting the dev stack healthy; `make test` does not need them.
