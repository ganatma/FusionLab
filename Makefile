.PHONY: setup dev test data train bench usd fieldlines

PORT ?= 8000
# One GPU for everything this Makefile starts (server, training, Warp, tests).
# Pick another with `make dev CUDA_VISIBLE_DEVICES=1`.
export CUDA_VISIBLE_DEVICES ?= 0

setup:            ## install everything
	uv sync
	@test -f .env || cp .env.example .env

dev:              ## run the app at http://localhost:$(PORT)
	uv run uvicorn fusionlab.api:app --reload --port $(PORT)

test:             ## unit + API tests
	uv run pytest -q

data:             ## re-download the FAIR-MAST cache (showcase shots + shot table). The repo already ships it.
	uv run python scripts/fetch_mast.py --db

train:            ## PhysicsNeMo correction to IPB98 on the real shot table -> models/surrogate*.{pt,json}
	uv run python -m fusionlab.surrogate

bench:            ## timings -> paste into README
	uv run python scripts/bench.py

usd:              ## OpenUSD export (Omniverse-compatible) of a cached shot -> out/
	uv run python -m fusionlab.usd_export 30166

fieldlines:       ## NVIDIA Warp field-line tracing: q check vs EFIT q95, benchmark, USD with field lines -> out/
	uv run python -m fusionlab.fieldlines 30166
