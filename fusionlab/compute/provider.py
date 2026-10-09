"""The ComputeProvider seam: how the API asks for GPU-bound compute without importing torch or Warp.

The routers talk to a provider and nothing else (design outline §3). LocalProvider (fusionlab.compute.local)
runs the existing in-process functions, which is today's behavior; PR 2 adds SshProvider, which submits the
same named tasks to a worker on the user's own GPU host over an SSH tunnel. The interface is the four methods
below, kept minimal on purpose so PR 2 can implement it without renegotiation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class JobHandle:
    """Handle to one submitted job. LocalProvider finishes the job inside submit(); a remote provider returns
    at once and fills it asynchronously — stream() reports progress, result() collects the payload."""

    job_id: str
    task: str
    provider: str


class ComputeProvider(Protocol):
    """One execution backend for the named compute tasks (trace_fieldlines, run_surrogate, run_eq_surrogate, ...).

    Errors surface where today's call stack raises them: LocalProvider executes inside submit(), so a task
    failure raises there; a remote provider reports failures through stream() and result() instead.
    """

    name: str

    def capabilities(self) -> dict:
        """What this backend can run: tasks, trained models on disk, device — the design's version handshake."""
        ...

    def submit(self, task: str, inputs: dict) -> JobHandle:
        """Run/queue one named task on schema-checked inputs. An unknown task is a programming error (ValueError)."""
        ...

    def stream(self, job: JobHandle) -> AsyncIterator[dict]:
        """Progress events for one job (queued / running / progress / completed — whatever this backend has)."""
        ...

    def result(self, job: JobHandle) -> Any:
        """The job's payload, consumed once."""
        ...
