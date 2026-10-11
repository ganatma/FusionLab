# Bring your own machine: the shot-bundle contract

FusionLab replays MAST because MAST's open archive (FAIR-MAST) is what it ships with — not because
the physics or the UI know it. Everything the app consumes per shot is one plain dictionary of
arrays in fixed units, and this page writes that dictionary down as **ShotBundle v1**, the versioned
contract any data source must produce. Convert your own shots to it — from DIII-D, JET, AUG, EAST,
your lab's machine, wherever you hold data you are entitled to hold — and FusionLab runs on them
unchanged: the same replay strips, multi-shot compare, take-away export, USD stage, and compute
harness. FAIR-MAST stays exactly as it was; your bundles replay beside the archive's shots, never
instead of them.

Two honesty rules up front, and they hold for every number in this pipeline:

- **Units are fixed house-wide** — m, T, MA, 1e20 m⁻³, keV, MW, MJ, s. Convert at the edge, once,
  in your ingestor; nothing downstream re-scales.
- **Nothing here "validates" or "predicts" anything.** The twin is *compared with* experiment.
  Limits are distances, never forecasts. Imported catalog numbers are labelled `computed`, and
  quantities FusionLab derives (catalog scalars, the L/H flag where present) are labelled derived —
  never presented as measurements.

## The two ways to bring data

**A directory of bundles** (good for many shots): one `.npz` file per shot plus one `machine.json`,
in any directory; point `FUSIONLAB_USER_BUNDLES` at it. Or just drop `.npz` files into
`data/user_bundles/`, the default.

**One upload** (good for a first look): `POST /data/bundles/<id>` with the `.npz` as multipart form
data — the UI's **⬆ Import bundle** button in the replay toolbar does exactly this. The file's name
( minus `.npz`) becomes the shot's id; ids may use letters, digits, dots, dashes and underscores,
and must not collide with a bare shot number (those are the archive's). `DELETE /data/bundles/<id>`
removes an import; the archive's cache is never touched by that route.

Either way the file is validated against the contract in full before anything is published: an
invalid bundle fails with `400` and a message naming the first bad field, and leaves no partial
file behind.

## The bundle: one npz per shot

Each `.npz` holds a `meta` entry (a JSON string) plus one array per field. Every trace lives on a
single common time base `t_s` — the equilibrium-reconstruction slice times if you have them; the
slow diagnostics' values are interpolated onto it before saving. Required of every bundle
(`nt` = number of time slices):

| Field | Unit | Shape | Meaning, and what happens without it |
|---|---|---|---|
| `t_s` | s | `(nt,)` | strictly increasing time base; every trace shares it |
| `Ip_MA` | MA | `(nt,)` | plasma current (magnitude) |
| `B_T` | T | `(nt,)` | vacuum toroidal field (magnitude) |
| `n_e20` | 1e20 m⁻³ | `(nt,)` | line-average electron density; all-NaN tolerated — model traces go null, the summary shows gaps |
| `P_ohm_MW` | MW | `(nt,)` | ohmic heating power |
| `P_nbi_MW` | MW | `(nt,)` | neutral-beam power; zeros make the beam what-ifs inert |
| `W_MJ` | MJ | `(nt,)` | **MHD stored energy** `3/2 ∫p dV` — *not* the diamagnetic `wmhd`, which can differ by ~3.5× |
| `dWdt_MW` | MW | `(nt,)` | dW/dt; your ingestor may compute it by `np.gradient` if the archive lacks it |
| `q95` | – | `(nt,)` | edge safety factor; NaN → limit panels show gaps |
| `beta_N` | – | `(nt,)` | normalized beta |
| `kappa` | – | `(nt,)` | elongation |
| `a_m` | m | `(nt,)` | minor radius |
| `R_m` | m | `(nt,)` | major radius |
| `V_m3` | m³ | `(nt,)` | plasma volume |

The **equilibrium group** is optional in v1 — a machine without equilibrium reconstructions still
replays its traces; `/replay` answers normally with the surfaces marked absent and the 3D/ψ views
degrade gracefully:

| Field | Unit | Shape | Meaning |
|---|---|---|---|
| `lcfs_R`, `lcfs_Z` | m | `(nt, n_pts)` | last-closed-flux-surface outline, closed |
| `psi_n` | – | `(nt, NZ, NR)` f32 | normalized ψ (0 axis, 1 separatrix), rows = Z, **uniform grid** |
| `psi_R`, `psi_Z` | m | `(NR,)`, `(NZ,)` | the ψ grid axes |
| `R_mag_m`, `Z_mag_m` | m | `(nt,)` | magnetic axis |

Also optional, and each consumed only when present: `delta`, `li`, `P_rad_MW`, `Te0_keV`,
`ne0_e20`, the Thomson profiles `ts_R` / `ts_Te_keV` / `ts_ne_e20` `(nt, n_ts)`, the vessel outline
`wall_R` / `wall_Z`, the PF-coil filaments `coil_R` / `coil_Z` / `coil_W` / `coil_H`, and
`eq_inputs` `(nt, n_sensors)` for the equilibrium surrogate — in **your machine's own sensor
layout**; the shipped surrogate model is trained on MAST's 226-channel layout and flags foreign
input as out-of-distribution rather than pretend. Field-line tracing additionally consumes
`F_Tm` `(nt, n_f)` (F = R·B_φ over normalized ψ), `psi_axis_Wb` / `psi_bnd_Wb` `(nt,)`, and
`psi_norm` `(n_f,)`; without them tracing is simply absent.

`meta` is a JSON string (an object). Recognised keys — all optional, all read defensively:
`shot_id`, `campaign`, `heating`, `timestamp`, `divertor_config`, `plasma_shape`, `current_range`,
`shot_objective`, `preshot`, `postshot`, `summary`, `t_nbi_start_s`, `t_nbi_end_s`,
`P_nbi_max_MW`. The upload route's `meta_json` form field merges into it (your values win).

Writing one, concretely:

```python
import json
import numpy as np

np.savez_compressed(
    "out.npz",
    meta=json.dumps({"heating": "NBI", "campaign": "MYRUN1"}),  # a JSON *string*
    t_s=np.arange(nt) * 0.01, Ip_MA=ip, B_T=0.4, n_e20=ne,
    P_ohm_MW=pohm, P_nbi_MW=pnbi, W_MJ=w, dWdt_MW=dwdt,
    q95=q95, beta_N=bn, kappa=kappa, a_m=a, R_m=R, V_m3=vol,
    # optional: lcfs_R=..., psi_n=..., psi_R=..., psi_Z=..., R_mag_m=..., Z_mag_m=...,
)
```

## `machine.json`: the machine's identity

The per-machine values that are *not* data — display scales, naming, attribution, calibration
thresholds — come from a `MachineConfig` in `machine.json` inside the bundles directory. Every
field is optional; the defaults are deliberately machine-neutral. Unknown keys fail loudly.

| Field | Type | Meaning |
|---|---|---|
| `device` | str | key into `physics.DEVICES` presets (R, a, κ, B/I/P maxima); omit for a machine without a preset |
| `isotope_md` | float | isotope mass for the confinement scalings (2.0 = deuterium) |
| `alpha_particles` | bool | whether alpha heating enters the model (MAST: false, it ran deuterium) |
| `attribution` | str | provenance string carried on every replay/USD payload for these bundles — put your data's licence and source here |
| `campaign_pattern` | str | regex your campaign labels follow (the archive's look like `M9`) |
| `usd_prim_root` | str | stage root prim for USD exports (the archive's is `/MAST`) |
| `reconstruction_name` | str | what your archive calls its equilibrium code (shown in the UI) |
| `te_display_range` | [lo, hi] | 3D plasma Te colour range, keV |
| `te_usd_ramp_max_kev` | float | USD export Te→colour ramp ceiling, keV |
| `warmup_shot` | str\|null | bundle loaded at server start to warm the caches |
| `clean_limits` | object | catalog filter thresholds `{tau_E_max_s, kappa_min, a_min_m}` — imported catalogs default to *no* filtering; never MAST's machine-calibrated values |
| `bench_shot` | str\|null | the bundle the benchmark task replays (else the first cached) |
| `unit_sanity` | list | `[field, lo, hi]` mid-shot guards against a missed unit conversion — cheap, and it catches everything |

## What runs on a ψ-free bundle

With only the 14 required fields: replay strips, limits, summary, the virtual what-if (beam box
from `meta`), compare, CSV/JSON take-away export, catalog + search, and every harness task that
consumes traces. Without equilibrium you lose: the 3D vessel/plasma view, the ψ map, field-line
tracing, the USD stage, and the eq-surrogate overlay — each absent, never wrong.

With the equilibrium group added, those come back on your geometry. The two *learned* models
(confinement correction, equilibrium surrogate) remain MAST-trained data: they run on your bundles
with honest out-of-distribution shading and flags, or not at all — they never silently claim your
machine.

## The harness on imported data

The compute tasks (benchmarks, surrogates, tracing) resolve their inputs through the same seam —
`bench.timings(source=..., shot=...)`, the benchmark route's `source`/`shot` fields, `GET
/replay/{id}` with your id. No worker-registry changes, no MAST-shaped assumptions: a task says
"needs a shot bundle", the seam hands it yours.

```bash
# point the app at your bundles
FUSIONLAB_USER_BUNDLES=~/my_machine_bundles make dev

# or select the source explicitly
FUSIONLAB_DATA_SOURCE=user-file make dev      # default is fair-mast; /replay serves both regardless

# benchmark the harness on your bundle
curl -s -X POST 'localhost:8000/compute/benchmark?source=user-file&shot=my_machine_0001' | jq .
```

Example `machine.json`, for an imaginary AUG-like machine:

```json
{
  "device": "jet",
  "isotope_md": 2.0,
  "alpha_particles": false,
  "attribution": "My machine runs: internal laboratory use only, do not redistribute",
  "campaign_pattern": "RUN\\d+",
  "usd_prim_root": "/MyMachine",
  "reconstruction_name": "LELIN",
  "te_display_range": [0.1, 5.0],
  "te_usd_ramp_max_kev": 6.0,
  "warmup_shot": "my_machine_0001",
  "unit_sanity": [["R_m", 1.5, 2.0], ["a_m", 0.4, 0.6], ["B_T", 2.0, 3.5]]
}
```

## Versioning and the future

The bundle schema is versioned (`ShotBundle v1`, `fusionlab/data/bundle.py` — the field table in
code is authoritative and drives both validation and this page). Later versions add fields and
degrade old readers gracefully; the `omas`/IMAS path is a future *adapter* that produces bundles,
not a new internal format. If your archive can emit these ~45 arrays per shot, FusionLab is
yours.

One last restatement of the no-validation rule, because it is the contract's soul: numbers from
your bundles are replayed, compared with the 0D model and the learned corrections, and exported
with their provenance labels (`measured` / `model` / `learned` / `computed`). Nothing that comes
out is called a validated or predictive result — it is the model *beside* your data, honestly
labelled, everywhere.
