# User-provided compute over SSH

FusionLab runs everything on the CPU of whatever machine starts it, and that is enough for most of
the app. The heavy parts — Warp field-line tracing, the PhysicsNeMo surrogates, benchmarks — go
faster on a GPU. This is a way to use one you already have: a lab box, a university login node, a
gaming PC. FusionLab never provisions or pays for compute; you point it at a machine you control, a
small worker process from this same repo runs there, and GPU-bound work executes remotely and
streams results back.

The compute side of the connection is your machine, your account, your SSH setup. FusionLab adds no
credentials of its own: it uses your keys, agent, and `~/.ssh/config` exactly as they are, and
neither `fusionlab.toml` nor `.env` ever holds a key, password, or token.

## Quick start

On the machine with the GPU (any host you can SSH to and install software on):

```bash
git clone https://github.com/ganatma/FusionLab.git ~/FusionLab
cd ~/FusionLab
uv sync --extra remote      # or: curl -LsSf https://astral.sh/uv/install.sh | sh first
```

On the machine with the browser (your laptop):

1. Make sure `ssh <host>` works from it, without an interactive password prompt.
2. Tell FusionLab about the host — `fusionlab.toml` at the repo root, beside `.env`
   (`cp fusionlab.toml.example fusionlab.toml` to start):

   ```toml
   [compute]
   provider = "ssh"            # "local" (default) — zero config, current behaviour

   [compute.ssh]
   host = "gpu-lab"            # any ~/.ssh/config alias resolves as-is
   remote_dir = "~/FusionLab"  # the clone + uv sync from above
   autostart = true            # run the worker over SSH when none is answering
   # port = 8420               # local end of the tunnel
   ```

3. `make dev`, open the Replay tab, and watch the header chip. Within a few seconds it should
   read **GPU: gpu-lab (…)** — the worker's own handshake answering through the tunnel.

Every value has an environment-variable form — `FUSIONLAB_COMPUTE_PROVIDER=ssh`,
`FUSIONLAB_SSH_HOST=gpu-lab`, `FUSIONLAB_SSH_REMOTE_DIR=…`, `FUSIONLAB_SSH_AUTOSTART=…`,
`FUSIONLAB_SSH_PORT=…` (full list in `.env.example`) — and the environment wins over the TOML
file. Nothing loads `.env` automatically — not the app, not `make dev` — so these take effect
when they are in the actual environment: export them, or set them in your service manager.

## What the UI tells you

- **Compute chip** (Replay header): where compute runs. `Local CPU` is the default.
  `GPU: <host> (<device>)` means the remote worker answered a health probe within the last few
  seconds. `Local CPU · <host> unreachable` means the SSH provider is configured but not
  answering — work falls back to this machine. Hover for the handshake detail (worker version,
  torch/warp versions, CUDA, device name); a version mismatch with your local install is called
  out there rather than hidden.
- **Provenance**: results computed remotely are labelled `computed on <host>` in the notes under
  the plots. This joins the app's existing measured / model / learned / computed labels — it says
  *where* compute ran, never that the number is better.
- **Fallback banner**: when the remote side fails (connect, autostart, or mid-job) and the work
  ran locally instead, a dismissible banner says so with the reason. Nothing in the UI blocks on
  the remote: replay interactions proceed on whatever compute answered, and the next remote run
  tries the SSH provider again on its own.

## First-run checklist (against a real host)

Work through these in order; the first failure is the one to fix.

- [ ] `ssh <host>` connects without an interactive password prompt. The backend cannot type into
      one mid-request — keys or agent, as your own setup already does it.
- [ ] The host has `git` and `uv` on the PATH that a **non-interactive** SSH command sees
      (`ssh <host> 'which git uv'` — login shells can hide a missing PATH entry).
- [ ] The clone exists at `remote_dir`, and `cd <remote_dir> && uv sync --extra remote` has
      completed there.
- [ ] `ssh <host> 'cd <remote_dir> && timeout 5 uv run fusionlab-worker'` starts and logs that it
      binds 127.0.0.1:8420 — this is exactly what autostart runs for you.
- [ ] `make dev` on the laptop with `provider = "ssh"`: the chip shows the handshake within a few
      seconds, and the tip mentions no version mismatch.
- [ ] Trace field lines on a shot; the note under the plot says `computed on <host>`.
- [ ] Kill the worker (`pkill -f fusionlab-worker` on the host) and run again: the run falls back
      to local, the banner says why, and nothing in the UI hangs.

The last two checks are the honest test of the feature. Note what we could not do: this guide was
written in a development environment with no SSH credentials for any GPU host, so nothing here has
run against a real one yet — the loopback and mocked-transport tests cover the protocol, and this
checklist is the real-host path that remains.

## What it costs, and what it does not do

- **First job on a fresh worker** uploads model weights by content hash (sha256): 9.0 KB for the
  confinement-correction surrogate, 4.6 MB for the equilibrium surrogate. One time per weight
  version; after that the worker's cache answers and weight traffic is zero.
- **One job at a time** per worker — it is a single-user GPU box, not a cluster, and a queue of
  depth 1 is honest about that. Jobs can be cancelled.
- **The worker executes a fixed registry of named tasks** — field-line tracing, the two
  surrogates, the model-metrics readout, benchmarks. It is not a shell: schema-checked inputs, no
  command execution, payload-size caps. It binds loopback only, reachable through the tunnel.
- **Shot data** is read by the worker through the same FAIR-MAST path as everywhere else, from its
  own clone (which ships the same small showcase cache; other shots fetch on demand). Compact
  inputs — EFIT equilibria, signal arrays — ride inside the job itself. So the compute host needs
  ordinary internet, not a copy of the archive.
- **Version drift** between your laptop's clone and the worker's is surfaced, not hidden: the
  handshake reports the worker's version and the chip warns on mismatch. The fix is the usual one
  on the remote side — `git pull` and `uv sync`.
- **Not in this version**: SLURM/PBS submission, multi-user queues or auth, training jobs, Windows
  remotes, and anything that opens a port on your network. If you cannot install software on the
  compute target, this design does not fit it — that is its one real prerequisite.
