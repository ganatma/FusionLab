"""The fusionlab worker: the GPU-side half of the SSH compute mode (design outline §4-8).

A small FastAPI app from this same repo, run on the user's own compute host with

    uv run fusionlab-worker            # binds 127.0.0.1:8420 by default

It binds the loopback interface only — the backend reaches it through the SSH tunnel SshProvider
opens, never over the network. Its whole surface is the fixed, versioned task registry below plus
an artifact cache for trained weights: no shell, no eval, no filesystem passthrough. Worst case if
an exposed worker were found by a stranger is compute theft, not host compromise (design §8).

    POST /v1/jobs               {task, inputs, artifact_refs} -> {job_id, state}; 409 when busy
    GET  /v1/jobs/{id}/events   SSE: queued / running / progress / completed / error / cancelled
    GET  /v1/jobs/{id}/result   application/x-fusionlab-frames (fusionlab.frames), consumed once
    POST /v1/jobs/{id}/cancel   cooperative: immediate while queued; benchmark tasks check between sections
    POST /v1/artifacts/{sha256} upload one trained weight into the cache (hash-verified, capped)
    GET  /v1/capabilities       handshake: worker/torch/warp versions, CUDA, tasks, weight hashes
    GET  /health                {ok: true} — the tunnel/autostart probe

One job at a time (a single-user GPU box is honest about itself); a second submission gets a clean
409 instead of a hidden queue. Jobs live in memory only — a worker restart forgets them, and the
backend's local-fallback path never depends on remembering. Task bodies are the same functions
LocalProvider runs (fusionlab.compute.local.TASKS), so both providers execute one implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from fusionlab import frames
from fusionlab.compute.local import TASKS as _LOCAL_TASKS

WORKER_PROTOCOL_VERSION = 1      # bumped when the wire contract (framing, events) changes
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Payload caps (design §4: pick them from measured sizes — the numbers live in the PR body).
# Largest shipped weight: eq_surrogate.pt at 4,607,723 B; largest measured job input: one loaded
# shot (~1.1 MB as npz, ~1.5 MB frames-encoded). Both caps leave >10x headroom.
MAX_ARTIFACT_BYTES = 64 * 2**20   # one uploaded weight
MAX_INPUT_BYTES = 64 * 2**20      # one job's inputs
MAX_JOBS_REMEMBERED = 16          # terminal jobs kept for result collection before eviction

RESULT_MEDIA_TYPE = "application/x-fusionlab-frames"
TERMINAL_STATES = ("completed", "error", "cancelled")


# ---------------------------------------------------------------- task registry (named, never a shell)
class TraceFieldlinesInputs(BaseModel):
    """A loaded FAIR-MAST shot (fusionlab.mast.load_shot), arrays frames-encoded."""

    model_config = ConfigDict(extra="forbid")

    shot: dict[str, Any]


class RunSurrogateInputs(BaseModel):
    """Feature columns for surrogate.correction (surrogate.shot_features output)."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]


class RunSurrogateShotInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    shot: dict[str, Any]
    P_loss_MW: float | np.ndarray   # the replayed loss trace is per-slice (physics.replay)


class RunEqSurrogateInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    eq_inputs: np.ndarray   # (n, 226) magnetics signals, archive units


class ModelMetricsInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["surrogate", "eq_surrogate"]


class BenchmarkInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["quick", "full"] = "quick"


class ArtifactRef(BaseModel):
    """One trained weight the job will read, referenced by content hash, never carried in the job."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    role: Literal["surrogate.pt", "eq_surrogate.pt"]


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    inputs: dict[str, Any] = {}
    artifact_refs: list[ArtifactRef] = []


@dataclass(frozen=True)
class Task:
    """One registry entry: a pydantic input schema and the runner (validated inputs, progress hook)."""

    schema: type[BaseModel]
    run: Callable[[BaseModel, Callable[[dict], None]], Any]


def _local_task(name: str) -> Callable[[BaseModel, Callable[[dict], None]], Any]:
    """Run a PR-1 task body so both providers execute the same single implementation."""
    fn = _LOCAL_TASKS[name]
    return lambda inputs, _progress: fn(inputs.model_dump())


def _run_benchmark(inputs: BaseModel, progress: Callable[[dict], None]) -> dict:
    from fusionlab import bench  # lazy: pulls torch via surrogate

    if not isinstance(inputs, BenchmarkInputs):   # registry dispatch guarantees this; keep it honest
        raise TypeError(f"benchmark inputs must be BenchmarkInputs, got {type(inputs).__name__}")
    return bench.timings(profile=inputs.profile, progress=progress)


TASKS: dict[str, Task] = {
    "trace_fieldlines": Task(TraceFieldlinesInputs, _local_task("trace_fieldlines")),
    "run_surrogate": Task(RunSurrogateInputs, _local_task("run_surrogate")),
    "run_surrogate_shot": Task(RunSurrogateShotInputs, _local_task("run_surrogate_shot")),
    "run_eq_surrogate": Task(RunEqSurrogateInputs, _local_task("run_eq_surrogate")),
    "model_metrics": Task(ModelMetricsInputs, _local_task("model_metrics")),
    "benchmark": Task(BenchmarkInputs, _run_benchmark),
}


# ---------------------------------------------------------------- artifacts (sha256 cache)
def artifact_dir() -> Path:
    """Cache root for uploaded weights; FUSIONLAB_WORKER_ARTIFACTS overrides (tests, sandboxes)."""
    env = os.environ.get("FUSIONLAB_WORKER_ARTIFACTS", "")
    return Path(env) if env else Path.home() / ".cache" / "fusionlab" / "worker" / "artifacts"


def cached_artifacts() -> dict[str, int]:
    """sha256 -> size for the uploads in the cache dir. A missing dir means nothing cached yet."""
    root = artifact_dir()
    if not root.is_dir():
        return {}
    return {p.name: p.stat().st_size for p in root.iterdir() if _SHA256_RE.fullmatch(p.name)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def shipped_models() -> dict[str, dict[str, Any]]:
    """The trained weights this checkout ships (models/*.pt), by content hash — the drift signal."""
    from fusionlab import (  # module attrs; no torch import needed yet
        eq_surrogate,
        surrogate,
    )

    out: dict[str, dict[str, Any]] = {}
    for module in (surrogate, eq_surrogate):
        model_file: Path = module.MODEL_FILE
        if model_file.exists():
            out[model_file.name] = {"sha256": sha256_file(model_file), "bytes": model_file.stat().st_size}
    return out


def resolve_artifacts(refs: list[ArtifactRef]) -> dict[str, Path]:
    """Map each ref to a concrete weight file: a cached upload wins; otherwise the ref must match the
    sha256 of a weight the checkout ships. Anything else is rejected — never silently run on drift."""
    shipped = shipped_models()
    by_hash = {info["sha256"]: name for name, info in shipped.items()}
    resolved: dict[str, Path] = {}
    for ref in refs:
        cached = artifact_dir() / ref.sha256
        if cached.is_file():
            resolved[ref.role] = cached
        elif by_hash.get(ref.sha256) == ref.role:
            from fusionlab import eq_surrogate, surrogate

            resolved[ref.role] = (surrogate if ref.role == "surrogate.pt" else eq_surrogate).MODEL_FILE
        else:
            raise HTTPException(
                422, f"artifact ref not satisfied: {ref.role} @ {ref.sha256[:12]}... is neither cached on this "
                     f"worker nor shipped in models/ — upload it first (POST /v1/artifacts/{ref.sha256})")
    return resolved


@contextlib.contextmanager
def _weights_for_job(resolved: dict[str, Path]):
    """Point the model modules' weight paths at job-resolved files for exactly one job.

    Safe because the depth-1 queue runs one job at a time; paths (and the surrogate's loaded-net
    cache) are restored on exit, so a later job without refs reads its own checkout again.
    """
    from fusionlab import eq_surrogate, surrogate

    target = {"surrogate.pt": (surrogate, "MODEL_FILE"), "eq_surrogate.pt": (eq_surrogate, "MODEL_FILE")}
    saved = [(module, attr, getattr(module, attr)) for role, path in resolved.items()
             for module, attr in [target[role]]]
    for role, path in resolved.items():
        module, attr = target[role]
        setattr(module, attr, path)
    surrogate._loaded = None   # a net cached from a redirected weight must not leak into the next job
    try:
        yield
    finally:
        for module, attr, old in saved:
            setattr(module, attr, old)
        surrogate._loaded = None


# ---------------------------------------------------------------- job book (in-memory, depth-1)
@dataclass
class _Job:
    job_id: str
    task: str
    state: str = "queued"
    events: list[dict] = field(default_factory=list)
    result: bytes | None = None
    error: str | None = None
    error_type: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    runner: asyncio.Task | None = None


class _JobBook:
    """One job at a time; terminal jobs kept until MAX_JOBS_REMEMBERED lets them be evicted."""

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def create(self, task: str) -> _Job:
        with self._lock:
            active = [j for j in self._jobs.values() if j.state not in TERMINAL_STATES]
            if active:
                raise HTTPException(409, f"worker busy: job {active[0].job_id} is {active[0].state}; this worker "
                                         "runs one job at a time — cancel it or wait, then resubmit")
            job = _Job(job_id=uuid4().hex, task=task)
            self._jobs[job.job_id] = job
            self._evict_locked()
        return job

    def _evict_locked(self) -> None:
        terminal = sorted((j for j in self._jobs.values() if j.state in TERMINAL_STATES),
                          key=lambda j: j.job_id)
        for job in terminal[:max(len(terminal) - MAX_JOBS_REMEMBERED + 1, 0)]:
            del self._jobs[job.job_id]

    def get(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id!r} (the worker may have restarted)")
        return job

    def append(self, job: _Job, event: dict) -> None:
        with self._lock:
            job.events.append(event)


class _JobCancelled(Exception):
    """Raised from the progress hook between task sections to unwind a cancelled job."""


_JOBS = _JobBook()


# ---------------------------------------------------------------- the runner
async def _run_job(job: _Job, task: Task, inputs: BaseModel, resolved: dict[str, Path]) -> None:
    def progress(event: dict) -> None:
        if job.cancel.is_set():
            raise _JobCancelled
        _JOBS.append(job, {"event": "progress", **event})

    def execute() -> bytes:
        with _weights_for_job(resolved):
            return frames.pack_result(task.run(inputs, progress))

    try:
        if job.cancel.is_set():   # cancelled while queued
            raise _JobCancelled
        job.state = "running"
        _JOBS.append(job, {"event": "running", "task": job.task})
        blob = await asyncio.to_thread(execute)
        if job.cancel.is_set():   # the task finished after a cancel arrived: discard the result
            raise _JobCancelled
        job.result, job.state = blob, "completed"
        _JOBS.append(job, {"event": "completed", "job_id": job.job_id, "task": job.task})
    except _JobCancelled:
        job.state = "cancelled"
        _JOBS.append(job, {"event": "cancelled", "job_id": job.job_id})
    except Exception as e:   # a task failure is a job outcome, not a worker crash
        job.state, job.error, job.error_type = "error", str(e), type(e).__name__
        _JOBS.append(job, {"event": "error", "error": str(e), "error_type": type(e).__name__})


# ---------------------------------------------------------------- handshake
def worker_version() -> str:
    try:
        return importlib.metadata.version("fusionlab")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0+unknown"


_HANDSHAKE: dict[str, Any] | None = None
_HANDSHAKE_LOCK = threading.Lock()


def handshake() -> dict[str, Any]:
    """Version/device identity (computed once — a running box does not change its GPU), then the
    per-call parts: tasks, shipped weight hashes, and what the artifact cache already holds."""
    global _HANDSHAKE
    with _HANDSHAKE_LOCK:
        if _HANDSHAKE is None:
            import torch  # the worker host always has torch; the first handshake pays the import

            cuda = bool(torch.cuda.is_available())
            try:
                warp_version: str | None = importlib.metadata.version("warp-lang")
            except importlib.metadata.PackageNotFoundError:
                warp_version = None
            _HANDSHAKE = {"worker_protocol_version": WORKER_PROTOCOL_VERSION, "worker_version": worker_version(),
                          "torch": torch.__version__, "warp": warp_version, "cuda_available": cuda,
                          "device_name": torch.cuda.get_device_name(0) if cuda else (platform.processor() or "CPU")}
    return {**_HANDSHAKE, "tasks": sorted(TASKS), "models": shipped_models(), "artifacts": cached_artifacts()}


# ---------------------------------------------------------------- the app
app = FastAPI(title="fusionlab worker", version=worker_version(), docs_url=None, redoc_url=None)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/v1/capabilities")
async def capabilities() -> dict:
    return handshake()


@app.post("/v1/jobs", status_code=201)
async def create_job(request: Request) -> dict:
    """Validate, then queue. Depth-1: a second concurrent submission is a clean 409, not a hidden queue."""
    body = await request.body()
    if len(body) > MAX_INPUT_BYTES:
        raise HTTPException(413, f"job inputs exceed {MAX_INPUT_BYTES} bytes")
    try:
        req = JobRequest.model_validate_json(body)
    except ValidationError as e:
        raise HTTPException(422, f"invalid job request: {e}") from None
    task = TASKS.get(req.task)
    if task is None:
        raise HTTPException(422, f"unknown task {req.task!r}; available: {sorted(TASKS)}. "
                                 "The registry is the worker's whole surface — there is no shell.")
    try:
        inputs = task.schema.model_validate(frames.decode_inputs(req.inputs))
    except ValidationError as e:
        raise HTTPException(422, f"invalid inputs for {req.task!r}: {e}") from None
    except frames.FrameError as e:
        raise HTTPException(422, f"malformed job inputs: {e}") from None

    job = _JOBS.create(req.task)   # raises 409 when busy
    _JOBS.append(job, {"event": "queued", "task": req.task})
    job.runner = asyncio.create_task(_run_job(job, task, inputs, resolve_artifacts(req.artifact_refs)))
    return {"job_id": job.job_id, "state": job.state, "task": req.task}


@app.get("/v1/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """SSE over the job's event list; the stream ends with the terminal event."""
    job = _JOBS.get(job_id)

    async def stream() -> AsyncIterator[str]:
        index = 0
        while True:
            while index < len(job.events):
                event = job.events[index]
                index += 1
                yield f"event: {event['event']}\ndata: {json.dumps(event, allow_nan=True)}\n\n"
                if event["event"] in TERMINAL_STATES:
                    return
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    """Job state without the payload — SshProvider.result() polls this while the job runs."""
    job = _JOBS.get(job_id)
    return {"job_id": job_id, "task": job.task, "state": job.state,
            "error": job.error, "error_type": job.error_type}


@app.get("/v1/jobs/{job_id}/result")
async def job_result(job_id: str) -> Response:
    """The FLF1-framed result, consumed once — the same pop-once contract as LocalProvider."""
    job = _JOBS.get(job_id)
    if job.state in ("queued", "running"):
        raise HTTPException(409, f"job {job_id} is {job.state}")
    if job.state == "cancelled":
        raise HTTPException(409, f"job {job_id} was cancelled")
    if job.state == "error":
        return JSONResponse(status_code=500, content={"detail": f"remote task failed: {job.error}",
                                                      "error_type": job.error_type})
    if job.result is None:   # completed, but the payload was already fetched once
        raise HTTPException(409, f"job {job_id}'s result was already consumed")
    payload, job.result = job.result, None
    return Response(content=payload, media_type=RESULT_MEDIA_TYPE)


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job.state in TERMINAL_STATES:
        raise HTTPException(409, f"job {job_id} is already {job.state}")
    job.cancel.set()
    return {"job_id": job_id, "state": job.state, "cancelling": True,
            "note": "cancellation is cooperative: immediate while queued, checked between benchmark sections"}


@app.post("/v1/artifacts/{sha256}", status_code=201)
async def upload_artifact(sha256: str, request: Request) -> dict:
    """One weight into the cache, verified byte-for-byte against the hash in the path."""
    if not _SHA256_RE.fullmatch(sha256):
        raise HTTPException(422, "artifact id must be a lowercase 64-hex sha256")
    body = await request.body()
    if len(body) > MAX_ARTIFACT_BYTES:
        raise HTTPException(413, f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    if hashlib.sha256(body).hexdigest() != sha256:
        raise HTTPException(422, "sha256 mismatch: the uploaded bytes do not match the requested hash")
    root = artifact_dir()
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{sha256}.tmp"
    tmp.write_bytes(body)
    os.replace(tmp, root / sha256)   # atomic: a reader never sees a half-written weight
    return {"sha256": sha256, "bytes": len(body)}


def main(argv: list[str] | None = None) -> int:
    """Console entry point (pyproject [project.scripts]): `uv run fusionlab-worker` on the GPU host."""
    import argparse

    parser = argparse.ArgumentParser(description="fusionlab compute worker (binds loopback; reach it over SSH)")
    parser.add_argument("--host", default="127.0.0.1", help="bind interface; keep the loopback default")
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUSIONLAB_WORKER_PORT", "8420")))
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
