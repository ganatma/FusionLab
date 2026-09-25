# Agent instructions (Claude Code, Cursor, any AI pair)

Read first: `README.md` (what it is), `docs/RESULTS.md` (numbers and limitations), `docs/PHYSICS.md` (the model and its
sources), `docs/CURRICULUM.md` (the guided study).

## Workflow
1. Keep `main` runnable. After every change run `make test`; the pre-push hook runs it again.
2. One small vertical slice at a time. Commit small with clear messages.
3. Never break the core loop: pick a real shot → replay it → model beside the measurement → limits from real data
   (`/shots`, `/replay/{id}`, `/simulate`, `/map`). The guided study (`web/guide.js`) and the virtual shot
   (`fusionlab/virtual.py`) ride on top of it and must never be needed for it.

## Code conventions
- Python 3.12, numpy-vectorized physics. **No Python loops over operating points**; a time integration loops over time
  steps with every scenario advanced together as arrays.
- Units are fixed: m, T, MA, 1e20 m^-3, keV, MW, MJ. Name quantities with units (`P_fus_MW`, `T_keV`).
- Physics changes cite a source in `docs/PHYSICS.md` and keep `tests/test_physics.py` green, especially the ITER
  calibration test.
- A learned model is never trusted blindly: every new output head gets a holdout number in `docs/RESULTS.md`, on a split
  by session block or campaign, never a random split by shot. Numbers in the docs come from the scripts that compute
  them (`make train`, `make bench`, `make validate-virtual`), never typed by hand.
- Frontend is static (`web/`), served by FastAPI. No bundler, no framework install, no CDN (the demo runs offline;
  `tests/test_api.py` checks). Lessons are data (`web/lessons.json`); checks are data, never code.
- No secrets in the repo. If a key is ever needed, it goes in `.env` and is documented in `.env.example`.
- One GPU by default (`CUDA_VISIBLE_DEVICES=0` in the Makefile).

## Honesty rules
- It is an educational twin on a reduced (0D) model: "compared with MAST data", never "validated" or "predictive".
- A limit is a distance: "would cross the Troyon limit at t = …", never "would disrupt".
- Say "OpenUSD export (Omniverse-compatible)", never "runs in Omniverse", unless it actually does.
- Where a model loses (the correction on an unseen campaign, the forward-equilibrium spike), the docs say so.

## Data and licence
FAIR-MAST data is CC BY-SA 4.0 with two requested citations (`data/README.md`). `data/eq/` (equilibrium training set,
2.7 GB) and `data/refly/` (re-fly cache) are git-ignored; `scripts/fetch_eq_dataset.py` and `make validate-virtual`
rebuild them from the archive.
