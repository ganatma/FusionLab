"""LocalProvider: today's in-process execution, behind the ComputeProvider seam.

Every task body lazy-imports the module it wraps (fieldlines pulls in Warp, the surrogates pull in torch),
exactly as the routers did before the seam: a broken or missing torch/Warp install degrades only the endpoints
that need it, never the API's import. The SSH provider (PR 2) implements the same task registry remotely; these
task functions are the reference implementation of its payload contract.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

import numpy as np

from fusionlab.compute.provider import JobHandle


# ---------------------------------------------------------------- the tasks (named, never a shell)
def _trace_fieldlines(inputs: dict) -> dict:
    """Both Warp traces /_lines runs for one shot — usd_lines (the polylines) and q_check (traced q vs EFIT) —
    plus the device and psi_N start values the field-lines endpoint reports beside them."""
    from fusionlab import fieldlines  # imports warp

    shot = inputs["shot"]
    pts, tr = fieldlines.usd_lines(shot)
    qc = fieldlines.q_check(shot)
    return {"pts": pts, "trace": tr, "q_check": qc, "device": str(tr.get("device", fieldlines.default_device())),
            "psi_n_start": list(fieldlines.USD_PSI_N)}


def _run_surrogate(inputs: dict) -> np.ndarray:
    """surrogate.correction on the model's feature columns (the operating point the caller built)."""
    from fusionlab import surrogate  # imports torch

    return surrogate.correction(inputs["features"])


def _run_surrogate_shot(inputs: dict) -> np.ndarray:
    """surrogate.correction(shot_features(shot, P_loss)): the composite /replay runs, so the feature definition
    keeps exactly one home (fusionlab.surrogate.shot_features) and the routers never import torch."""
    from fusionlab import surrogate  # imports torch

    return surrogate.correction(surrogate.shot_features(inputs["shot"], inputs["P_loss_MW"]))


def _run_eq_surrogate(inputs: dict) -> np.ndarray:
    """eq_surrogate.predict_psi: magnetics signals -> the psi(R, Z) map."""
    from fusionlab import eq_surrogate  # imports torch

    return eq_surrogate.predict_psi(inputs["eq_inputs"])


def _model_metrics(inputs: dict) -> dict:
    """Holdout metrics JSON for one trained model ("surrogate" | "eq_surrogate"), read beside its checkpoint."""
    from fusionlab import eq_surrogate, surrogate  # import torch

    files = {"surrogate": surrogate.METRICS_FILE, "eq_surrogate": eq_surrogate.METRICS_FILE}
    return json.loads(files[inputs["model"]].read_text())


def _benchmark(inputs: dict, progress: Callable[[dict], None] | None = None) -> dict:
    """bench.timings for one profile — the worker's `benchmark` task, in-process. The section hook is
    what progress subscribers see between sections (the same events the worker's SSE emits)."""
    from fusionlab import bench  # lazy: pulls torch via surrogate

    return bench.timings(profile=inputs.get("profile", "quick"), progress=progress)


# Task bodies take the inputs dict. Bodies that report section progress (declared here) additionally
# take the event hook: the worker passes its SSE hook, LocalProvider appends to the job's event list —
# one event vocabulary, both backends.
_PROGRESS_TASKS = {"benchmark"}

TASKS: dict[str, Callable[..., Any]] = {
    "trace_fieldlines": _trace_fieldlines,
    "run_surrogate": _run_surrogate,
    "run_surrogate_shot": _run_surrogate_shot,
    "run_eq_surrogate": _run_eq_surrogate,
    "model_metrics": _model_metrics,
    "benchmark": _benchmark,
}


# ---------------------------------------------------------------- the provider
_TERMINAL_EVENTS = ("completed", "error", "cancelled")
MAX_JOBS_REMEMBERED = 16   # terminal jobs kept for late progress subscribers (mirrors the worker)


class LocalProvider:
    """In-process execution on this machine's CPU/GPU — the behavior the routers had before the seam."""

    name = "local"

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: dict[str, JobHandle] = {}   # task -> the job submit() is running right now

    def capabilities(self) -> dict:
        """Tasks and trained-model availability. The lazy imports are the torch cost the routers used to pay
        at first use; nothing here touches Warp (tracing imports stay inside the trace task)."""
        from fusionlab import eq_surrogate, surrogate  # import torch

        return {"provider": self.name, "tasks": sorted(TASKS),
                "surrogate_available": surrogate.available(),
                "eq_surrogate_available": eq_surrogate.MODEL_FILE.exists()}

    def active_job(self, task: str) -> JobHandle | None:
        """The job of `task` this provider is running right now, or None. submit() runs synchronously,
        so a local job is discoverable mid-run — that is what lets the progress route stream its event
        list while the body works."""
        return self._active.get(task)

    def submit(self, task: str, inputs: dict) -> JobHandle:
        """Run the task in-process, now: a local job is complete when submit returns, and a failure raises
        here — the same call stack the routers had. The payload is held until result() consumes it once.

        The job's event list grows while the body runs (queued → running → progress? → terminal, the
        worker's exact vocabulary), so a progress subscriber can watch a local job live.
        """
        try:
            run = TASKS[task]
        except KeyError:
            raise ValueError(f"unknown compute task {task!r}; available: {sorted(TASKS)}") from None
        job = JobHandle(job_id=uuid4().hex, task=task, provider=self.name)
        record: dict[str, Any] = {"task": task, "result": None,
                                  "events": [{"event": "queued", "task": task},
                                             {"event": "running", "task": task}]}
        self._jobs[job.job_id] = record
        self._prune()
        self._active[task] = job

        def progress(event: dict) -> None:
            record["events"].append({"event": "progress", **event})

        try:
            record["result"] = run(inputs, progress) if task in _PROGRESS_TASKS else run(inputs)
            record["events"].append({"event": "completed", "job_id": job.job_id, "task": task})
        except Exception as e:
            record["events"].append({"event": "error", "error": str(e), "error_type": type(e).__name__})
            raise
        finally:
            if self._active.get(task) is job:   # a concurrent same-task submit owns the slot now
                del self._active[task]
        return job

    async def stream(self, job: JobHandle) -> AsyncIterator[dict]:
        """The job's events in the worker's vocabulary, replayed from zero and — while the body runs —
        live: this polls the record exactly like the worker's SSE endpoint does. Local jobs have no
        remote queue, but the event sequence is identical, so subscribers render both the same."""
        record = self._jobs.get(job.job_id)
        if record is None:
            raise KeyError(f"unknown job {job.job_id}")
        index = 0
        while True:
            while index < len(record["events"]):
                event = record["events"][index]
                index += 1
                yield event
                if event["event"] in _TERMINAL_EVENTS:
                    return
            if self._jobs.get(job.job_id) is not record:   # pruned while nobody was watching
                return
            await asyncio.sleep(0.1)

    def result(self, job: JobHandle) -> Any:
        try:
            return self._jobs[job.job_id].pop("result")
        except KeyError:
            raise KeyError(f"job {job.job_id} has no result: unknown id, or already consumed") from None

    def _prune(self) -> None:
        """Keep only the most recent MAX_JOBS_REMEMBERED records: the shared provider instance must not
        accumulate old jobs (results leave when result() consumes them; event lists are small)."""
        excess = list(self._jobs)[:-MAX_JOBS_REMEMBERED]
        for job_id in excess:
            del self._jobs[job_id]

    def health(self) -> None:
        """Always healthy: this provider is the current process."""

    def warm_up(self) -> None:
        """Load the trained correction model (and torch) at server start — the ~5 s import the first replay
        request otherwise pays. Reached through fusionlab.compute.warm_up(), never at import time."""
        from fusionlab import surrogate  # imports torch

        if surrogate.available():
            surrogate.load()
