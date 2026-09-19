# Credit, and how this differs from fusionsimulator.io

[fusionsimulator.io](https://fusionsimulator.io) (Daniel Burgess, Columbia Fusion Research Center; repo
[d-burg/fusion-sim](https://github.com/d-burg/fusion-sim)) is a polished open-source tokamak teaching simulator. It was our
reference for what a clear fusion UI feels like: one screen, a cross-section beside time traces, a view of the machine.
**We copied no code, shaders, geometry files or assets from it.** Its `LICENSE` file is GPL-3.0 (its README says MIT), this
repo is MIT, so we kept a clean separation: FusionLab is written from scratch in a different stack. The comparison below is
our reading of its README and architecture doc on 2026-09-19.

| | fusionsimulator.io | FusionLab |
|---|---|---|
| Question | "What would a pulse look like if I programmed these waveforms?" | "What did this real shot do, and where do the models miss it?" |
| Data | none: every signal is simulated (synthetic diagnostics) | measured: the 15,969-shot FAIR-MAST table; EFIT, Thomson and magnetics for the replayed shots |
| Machines | DIII-D, JET, ITER, CENTAUR (a concept) | MAST, a spherical tokamak, replayed from data; ITER / SPARC-class / JET / DIII-D / MAST in the sandbox |
| Equilibrium | analytic Cerfon–Freidberg solution | EFIT's reconstruction from the archive, plus a PhysicsNeMo surrogate from the magnetic sensors with its error on held-out sessions |
| Engine | Rust → WebAssembly in the browser; time-dependent 0D with ELM and disruption models | Python/NumPy on the server; scaling laws and limits evaluated on the measured inputs, every time slice in one vectorized call |
| Learning | none | two PhysicsNeMo models, each with a leak-free holdout table |
| GPU | WebGL shaders for rendering | NVIDIA Warp field-line tracing checked against EFIT's q95; CUDA training |
| 3D | hand-built in-vessel scene (tiles, ports) on CAD limiter polygons | only what the archive holds, revolved: limiter contour, PF filaments, last closed flux surface, traced field lines |
| Takes away | the web app | `load_shot()` Python API, cleaned shot table, time-sampled OpenUSD stage |
| UI idea | a control room for running a pulse | every panel tagged **measured / model / learned / computed**, so data is never mistaken for model |

Where it is ahead of us: its time-dependent dynamics (ELMs, disruptions, a pulse planner) are far richer than our steady-state
sandbox, and it needs no server. The two are complementary: a simulator shows what the textbook says should happen; a twin
shows what did.
