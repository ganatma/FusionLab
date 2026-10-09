"""LocalProvider: today's in-process execution, behind the ComputeProvider seam.

Every task body lazy-imports the module it wraps (fieldlines pulls in Warp, the surrogates pull in torch),
exactly as the routers did before the seam: a broken or missing torch/Warp install degrades only the endpoints
that need it, never the API's import. The SSH provider (PR 2) implements the same task registry remotely; these
task functions are the reference implementation of its payload contract.
"""

from __future__ import annotations

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


TASKS: dict[str, Callable[[dict], Any]] = {
    "trace_fieldlines": _trace_fieldlines,
    "run_surrogate": _run_surrogate,
    "run_surrogate_shot": _run_surrogate_shot,
    "run_eq_surrogate": _run_eq_surrogate,
    "model_metrics": _model_metrics,
}


# ---------------------------------------------------------------- the provider
class LocalProvider:
    """In-process execution on this machine's CPU/GPU — the behavior the routers had before the seam."""

    name = "local"

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> dict:
        """Tasks and trained-model availability. The lazy imports are the torch cost the routers used to pay
        at first use; nothing here touches Warp (tracing imports stay inside the trace task)."""
        from fusionlab import eq_surrogate, surrogate  # import torch

        return {"provider": self.name, "tasks": sorted(TASKS),
                "surrogate_available": surrogate.available(),
                "eq_surrogate_available": eq_surrogate.MODEL_FILE.exists()}

    def submit(self, task: str, inputs: dict) -> JobHandle:
        """Run the task in-process, now: a local job is complete when submit returns, and a failure raises
        here — the same call stack the routers had. The payload is held until result() consumes it once."""
        try:
            run = TASKS[task]
        except KeyError:
            raise ValueError(f"unknown compute task {task!r}; available: {sorted(TASKS)}") from None
        job = JobHandle(job_id=uuid4().hex, task=task, provider=self.name)
        self._jobs[job.job_id] = {"task": task, "result": run(inputs)}
        return job

    async def stream(self, job: JobHandle) -> AsyncIterator[dict]:
        """A local job has no queued/running phase (submit already ran it): one completed event."""
        record = self._jobs.get(job.job_id)
        if record is None:
            raise KeyError(f"unknown job {job.job_id}")
        yield {"event": "completed", "job_id": job.job_id, "task": record["task"]}

    def result(self, job: JobHandle) -> Any:
        try:
            return self._jobs[job.job_id].pop("result")
        except KeyError:
            raise KeyError(f"job {job.job_id} has no result: unknown id, or already consumed") from None

    def warm_up(self) -> None:
        """Load the trained correction model (and torch) at server start — the ~5 s import the first replay
        request otherwise pays. Reached through fusionlab.compute.warm_up(), never at import time."""
        from fusionlab import surrogate  # imports torch

        if surrogate.available():
            surrogate.load()
