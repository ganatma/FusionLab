# FusionLab codebase map (depth 2)

| Path | What it is |
|---|---|
| `fusionlab/` | Python package: the twin itself |
| `fusionlab/api.py` | FastAPI app, lifespan warm-up, `/health /devices /simulate /map /reactivity`, serves `web/` |
| `fusionlab/api_replay.py` | Replay router: `/shots`, `/replay/{id}` (+`/psi/{i}`, `/usd`, `/fieldlines/{i}`), `/db`, `/surrogate`, `/eq_surrogate` |
| `fusionlab/api_virtual.py` | Virtual-shot router: `/virtual/gate`, `POST /virtual/{id}` |
| `fusionlab/physics.py` | 0D power-balance engine, scaling laws (IPB98, ITER89-P), limits, DEVICES table |
| `fusionlab/mast.py` | FAIR-MAST loader: `load_shot`, `clean_db`, npz caching, SI → project units |
| `fusionlab/surrogate.py` | PhysicsNeMo τ_E correction to IPB98 (session/campaign holdout) |
| `fusionlab/eq_surrogate.py` | PhysicsNeMo flux-map surrogate: 93 magnetics signals → ψ(R,Z) 65×65 |
| `fusionlab/fieldlines.py` | NVIDIA Warp RK4 field-line tracer + q95 check vs EFIT |
| `fusionlab/usd_export.py` | Time-sampled OpenUSD stage export (vessel, PF coils, plasma, field lines) |
| `fusionlab/virtual.py` | Virtual-shot engine: blind + anchored 0D re-fly, matched-pair gating |
| `web/` | Static frontend, no bundler, works offline |
| `web/index.html`, `app.js`, `replay.js` | App shell, tabs, replay panel |
| `web/guide.js`, `lessons.json`, `glossary.json` | Guided study: chapters/tasks are data; `tests/test_guide.py` proves tasks doable |
| `web/figures.js`, `vessel3d.js` | Plotly figures; three.js vessel view with field lines |
| `web/vendor/` | Vendored Plotly, three.js, OrbitControls (no CDN) |
| `tests/` | pytest suite: `test_api, _physics, _mast, _surrogate, _eq_surrogate, _virtual, _fieldlines, _usd, _guide` |
| `scripts/` | `fetch_mast.py`, `fetch_eq_dataset.py`, `train_eq_surrogate.py`, `validate_virtual.py`, `bench.py`, `eq_forward_jacobian.py` |
| `data/` | Shipped FAIR-MAST cache: 5 showcase shots (npz) + `mast_db.npz` (shot table); CC BY-SA 4.0 |
| `models/` | Trained artifacts: `surrogate.pt`, `eq_surrogate.pt` + per-model metrics JSON |
| `docs/` | `PHYSICS.md` (equations/sources), `RESULTS.md` (numbers/limitations), `CURRICULUM.md`, `PRIOR_ART.md` |
| `.githooks/` | `pre-push` runs `make test` |
