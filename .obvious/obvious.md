# FusionLab — obvious contract

Open-data tokamak digital twin: replays real MAST shots beside a 0D physics model, learns a
confinement correction with NVIDIA PhysicsNeMo, traces field lines with NVIDIA Warp, and exports
time-sampled OpenUSD stages. FastAPI backend serves a static, no-build web UI.

## Stack

| Layer | Tech |
|---|---|
| Language / runtime | Python 3.11–3.12 (NOT system 3.13 — uv provisions 3.12 from `.python-version`/uv.lock) |
| Package manager | [uv](https://docs.astral.sh/uv/) (`uv.lock`); installed at `~/.local/bin/uv` |
| API | FastAPI + Uvicorn (`fusionlab.api:app`), port **8000** |
| Frontend | Static `web/` (HTML/JS, Plotly + three.js vendored, no build step, offline) |
| ML / GPU | torch, nvidia-physicsnemo, warp-lang — GPU **optional**; runs CPU-only with a benign Warp CUDA warning |
| Data | UKAEA FAIR-MAST cache ships in-repo (`data/shots/*.npz`, `data/mast_db.npz`) — no network, no DB, no secrets; compare fetches uncached catalog shots on demand via `/replay/{id}` (cache-then-S3) |

## Commands (all verified 2026-10-09)

```bash
make setup   # uv sync + git core.hooksPath=.githooks + cp .env.example .env  (install uv first if missing)
make dev     # uvicorn fusionlab.api:app --reload --port 8000 -> http://localhost:8000
make test    # uv run pytest -q  (88 passed, 1 skipped, ~19 s)
make train   # retrain IPB98 correction -> models/surrogate*
make bench   # timings
make usd     # OpenUSD export of shot 30166 -> out/
make fieldlines  # Warp field-line q check + benchmark (needs GPU for speed, works CPU)
make data    # re-download FAIR-MAST cache (repo already ships it)
```

No external services (no Postgres/Redis/etc.). The only setup prerequisite is `uv`
(`curl -LsSf https://astral.sh/uv/install.sh | sh` if `~/.local/bin/uv` is absent).

## Environment

- `.env` (copied from `.env.example`): `FUSIONLAB_PORT=8000`, `FUSIONLAB_DEVICE=` (force `mps|cuda|cpu`; default auto).
- Optional agent keys (`FUSIONLAB_AGENT_ENABLED`, `FUSIONLAB_AGENT_MODEL`, `ANTHROPIC_API_KEY`): all three are
  unnecessary for the offline demo; the in-app agent stays off unless the flag and a key are both set.
- No API keys are needed. The demo runs fully offline after `make setup`.

## Codebase map

See [codebase-map.md](codebase-map.md) for the folder-level table.

Key entry points: `fusionlab/api.py` (app + `/health /devices /simulate /map /reactivity /`),
`fusionlab/api_replay.py` (`/shots /replay/{id}[/psi|/usd|/fieldlines] /db /surrogate /eq_surrogate`),
`fusionlab/api_virtual.py` (`/virtual/gate`, `POST /virtual/{id}`), `fusionlab/physics.py` (0D engine).

## Local verification (Validation Summary)

- **dev_stack_healthy: true** (2026-10-09, fresh checkout)
- `make setup` EXIT=0; `make dev` up on parsed port **8000**; `/health` → `{"ok": true}`.
- Primary flow exercised end-to-end over HTTP: `/shots` → `/replay/30166` (200, full measured+model+EFIT bundle),
  `/simulate?device=iter` (Q ≈ 9.2), `/map`, `/devices`, and `/` (UI renders; all 7 expected UI tokens found
  on the live page; screenshots in the skill record; console clean apart from WebGL performance warnings).
- `make test`: **88 passed, 1 skipped** (18.46 s). No lint/typecheck config exists in the repo (pytest is the gate,
  also run by the `.githooks/pre-push` hook).
- Known-benign on GPU-less machines: `Warp CUDA warning: Could not find or load the NVIDIA CUDA driver.`
- Snapshot: `b0m8lu2i3g9bglil5io5:default` captured 2026-10-09T16:25:46Z from this working session.

## House rules (from CLAUDE.md — read before changing physics or docs)

- Keep `main` runnable; run `make test` after every change (pre-push hook re-runs it).
- Units are fixed: m, T, MA, 1e20 m^-3, keV, MW, MJ. No Python loops over operating points.
- Never "validated" or "predictive" — this twin is *compared with* MAST data; limits are distances, never forecasts.
- Docs numbers come from the scripts that compute them, never typed by hand.
