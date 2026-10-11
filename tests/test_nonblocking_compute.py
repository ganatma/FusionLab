"""HARN-4: async routes await compute on the dedicated pool — the shared request threadpool stays free.

Three layers, mirroring the issue's acceptance criteria:

- ``run_task_async`` — the awaited entry point keeps run_task's provider selection, local fallback
  and provenance exactly.
- Router-level remote coverage (the audit's gap 1): a configured remote through real routes,
  including the fallback labels the responses and provenance carry.
- The concurrency property: parallel replays against a slow remote never starve ``/health``.

No SSH: the provider is a fake whose ``result()`` takes its time (the exact wait a sync router
used to pin a threadpool token for).
"""

import asyncio
import threading
import time

import anyio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fusionlab import api_replay, compute
from fusionlab.api import app
from fusionlab.compute import status
from fusionlab.compute.provider import JobHandle
from fusionlab.compute.ssh import RemoteComputeError

client = TestClient(app)


# ---------------------------------------------------------------- fakes and helpers
def shaped_payload(task: str, inputs: dict, psi_shape: tuple[int, ...]) -> dict | np.ndarray:
    """Task-shaped results the replay path's consumers accept (the bodies index and scale arrays,
    they never inspect provenance of the fake's values)."""
    if task == "model_metrics":
        return {"block_size": 100, "held_out_blocks": [], "test_shot_ids": []}
    if task == "run_surrogate_shot":
        return np.ones(np.shape(inputs["shot"]["t_s"]))
    if task == "run_eq_surrogate":
        return np.zeros(psi_shape)
    if task == "run_surrogate":
        return np.ones(np.shape(inputs["features"])[:1])
    raise AssertionError(f"unexpected task {task!r}")


class SlowRemote:
    """A provider-shaped remote: submit queues instantly, result() takes delay_s to land.

    ``stream`` matches the worker's SSE vocabulary (queued/running/terminal); with ``fail`` every
    job dies the mid-job way run_task's fallback is built on; ``cancel`` makes result() raise the
    same cancelled error HARN-2's route expects.
    """

    name, host = "ssh", "slowsrvr"

    def __init__(self, delay_s: float = 0.0, fail: str | None = None,
                 psi_shape: tuple[int, ...] = (1, 1, 1)) -> None:
        self.delay_s, self.fail, self.psi_shape = delay_s, fail, psi_shape
        self._inputs: dict[str, dict] = {}
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self.cancel_calls = 0

    def capabilities(self) -> dict:
        return {"provider": self.name, "host": self.host, "surrogate_available": True,
                "eq_surrogate_available": True, "version_mismatch": None, "remote": {}}

    def health(self) -> None:
        return None

    def submit(self, task: str, inputs: dict) -> JobHandle:
        self._inputs[task] = inputs
        return JobHandle(job_id=f"j-{task}", task=task, provider=self.name)

    async def stream(self, job: JobHandle):
        yield {"event": "queued", "task": job.task}
        yield {"event": "running", "task": job.task}
        while not self._done.is_set():
            await asyncio.sleep(0.02)
        if self._cancelled.is_set():
            yield {"event": "cancelled", "task": job.task}
        elif self.fail:
            yield {"event": "error", "error": self.fail}
        else:
            yield {"event": "completed", "job_id": job.job_id, "task": job.task}

    def cancel(self, job: JobHandle) -> None:
        self.cancel_calls += 1
        self._cancelled.set()

    def result(self, job: JobHandle):
        deadline = time.monotonic() + self.delay_s
        while time.monotonic() < deadline and not self._cancelled.is_set():
            time.sleep(0.01)
        self._done.set()
        if self._cancelled.is_set():
            raise RemoteComputeError("job", "remote job was cancelled")
        if self.fail:
            raise RemoteComputeError("job", self.fail)
        return shaped_payload(job.task, self._inputs[job.task], self.psi_shape)


@pytest.fixture()
def clean(monkeypatch):
    """No selected provider, no status state leaking between tests (same as test_compute_status),
    plus the replay-path caches: a stub answer must never be served to another test."""
    monkeypatch.setenv("FUSIONLAB_COMPUTE_PROVIDER", "local")
    monkeypatch.setattr(compute, "_provider", None)
    status._reset_for_tests()
    api_replay._eq_psi.cache_clear()
    api_replay._eq_model.cache_clear()
    yield
    status._reset_for_tests()
    api_replay._eq_psi.cache_clear()
    api_replay._eq_model.cache_clear()


# ---------------------------------------------------------------- run_task_async
def test_run_task_async_matches_run_task_on_remote_success(clean):
    """Remote success through the awaited entry point: same payload, same provenance as run_task."""
    stub = SlowRemote(delay_s=0.05)
    compute._provider = stub
    out = asyncio.run(status.run_task_async("model_metrics", {"model": "surrogate"}))
    assert out == {"block_size": 100, "held_out_blocks": [], "test_shot_ids": []}
    assert status.provenance("model_metrics") == {"provider": "ssh", "host": "slowsrvr"}


def test_run_task_async_falls_back_and_records(clean):
    """Remote failure through the awaited entry point: the local answer comes back, the fallback
    is recorded with the remote's reason, and provenance flips to local."""
    stub = SlowRemote(fail="worker died mid-job")
    compute._provider = stub
    out = asyncio.run(status.run_task_async("model_metrics", {"model": "surrogate"}))
    assert "block_size" in out   # the local metrics file answered, not the fake
    fb = status.fallback_state()
    assert fb["fallback"] is True and "mid-job" in fb["reason"]
    assert status.provenance("model_metrics") == {"provider": "local", "host": None}


# ---------------------------------------------------------------- router-level remote (audit gap 1)
def test_router_serves_the_remote_answer_with_remote_provenance(clean):
    """A configured remote through a real route: the response body is the remote's own answer and
    the provenance names it — run_task-through-router, exercised for the first time here."""
    stub = SlowRemote(delay_s=0.05)
    compute._provider = stub
    r = client.get("/surrogate")
    assert r.status_code == 200
    assert r.json() == {"block_size": 100, "held_out_blocks": [], "test_shot_ids": []}
    assert status.provenance("model_metrics") == {"provider": "ssh", "host": "slowsrvr"}


def test_replay_route_labels_remote_provenance(clean):
    """The replay bundle says where its learned correction ran: the remote host, with no fallback."""
    stub = SlowRemote(delay_s=0.02)
    compute._provider = stub
    body = client.get("/replay/30166").json()
    assert body["summary"]["learned_computed_on"] == "slowsrvr"
    assert status.fallback_state()["fallback"] is False


def test_replay_route_falls_back_with_honest_labels(clean):
    """A dying remote mid-replay: the bundle is still served, built locally — learned_computed_on
    is None (no host), the fallback reason is recorded, and the provenance map says local."""
    stub = SlowRemote(fail="worker died mid-job")
    compute._provider = stub
    body = client.get("/replay/30166").json()
    assert body["summary"]["learned_computed_on"] is None
    fb = status.fallback_state()
    assert fb["fallback"] is True and "mid-job" in fb["reason"]
    assert status.provenance("run_surrogate_shot")["provider"] == "local"


# ---------------------------------------------------------------- the concurrency property (HARN-4)
async def _health_stays_responsive_while_replays_wait() -> None:
    """Four parallel replays against a slow remote; a watcher polls /health throughout.

    The shared default limiter is shrunk to 2 tokens first: pre-HARN-4 (sync routers), each replay
    held one of those 2 tokens for its whole ~2 s of remote waiting and /health starved past 1 s;
    post-HARN-4 the replays wait on the compute pool and /health keeps its token.
    """
    anyio.to_thread.current_default_thread_limiter().total_tokens = 2
    compute._provider = SlowRemote(delay_s=1.0, psi_shape=api_replay._shot(30166)["psi_n"].shape)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:

        async def one_replay() -> None:
            r = await c.get("/replay/30166")
            assert r.status_code == 200

        async def watch_health(stop: asyncio.Event) -> list[float]:
            latencies = []
            while not stop.is_set():
                t0 = time.monotonic()
                r = await c.get("/health")
                latencies.append(time.monotonic() - t0)
                assert r.json() == {"ok": True}
                await asyncio.sleep(0.05)
            return latencies

        stop = asyncio.Event()
        watcher = asyncio.create_task(watch_health(stop))
        await asyncio.gather(*(one_replay() for _ in range(4)))
        stop.set()
        latencies = await watcher
    assert latencies and max(latencies) < 1.0, max(latencies)


def test_health_stays_responsive_while_replays_wait(clean):
    asyncio.run(_health_stays_responsive_while_replays_wait())


# ---------------------------------------------------------------- HARN-1/HARN-2 through the new dispatch
async def _cancel_stops_an_awaited_remote_job() -> None:
    """HARN-2's route must reach a job launched through run_task_async (an async route in flight),
    and the request that was waiting on it must come back as the local fallback answer."""
    compute._provider = SlowRemote(delay_s=3.0)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        pending = asyncio.create_task(c.get("/surrogate"))
        for _ in range(250):
            if status.active_job("model_metrics") is not None:
                break
            await asyncio.sleep(0.02)
        assert status.active_job("model_metrics") is not None
        cancel_r = await c.post("/compute/cancel?task=model_metrics")
        out = await pending
    stub = compute._provider
    assert cancel_r.status_code == 200 and cancel_r.json()["cancelling"] is True
    assert stub.cancel_calls == 1   # the remote was told, not just abandoned client-side
    assert out.status_code == 200 and "block_size" in out.json()   # the fallback answered locally
    fb = status.fallback_state()
    assert fb["fallback"] is True and "cancel" in fb["reason"]
    assert status.provenance("model_metrics")["provider"] == "local"


def test_cancel_stops_an_awaited_remote_job(clean):
    asyncio.run(_cancel_stops_an_awaited_remote_job())


async def _progress_streams_an_awaited_remote_job() -> None:
    """HARN-1's SSE route must relay a job launched through run_task_async, labelled with the
    remote provider it ran on."""
    compute._provider = SlowRemote(delay_s=1.0)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        pending = asyncio.create_task(c.get("/surrogate"))
        for _ in range(250):
            if status.active_job("model_metrics") is not None:
                break
            await asyncio.sleep(0.02)
        body = ""
        async with c.stream("GET", "/compute/progress?task=model_metrics") as r:
            assert r.status_code == 200
            async for chunk in r.aiter_raw():
                body += chunk.decode()
                if "event: running" in body:
                    break   # the live prefix is the assertion; leaving early cancels the stream
        out = await pending
    assert "event: queued" in body and "event: running" in body
    assert '"provider": "ssh"' in body   # the relay names where the job actually ran
    assert out.status_code == 200   # the compute request itself completes remotely, untouched


def test_progress_streams_an_awaited_remote_job(clean):
    asyncio.run(_progress_streams_an_awaited_remote_job())
