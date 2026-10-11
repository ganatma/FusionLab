"""GET /compute and the fallback runner (PR 3): the local shape, a mocked-SSH handshake, an unreachable
worker as a status — never a 5xx — and provenance. No SSH; HTTP stays on httpx.MockTransport fakes."""

import asyncio
import sys
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from fusionlab import compute
from fusionlab.api import app
from fusionlab.compute import ssh as ssh_mod
from fusionlab.compute import status
from fusionlab.compute.provider import JobHandle
from fusionlab.compute.ssh import RemoteComputeError, SshProvider

client = TestClient(app)


# ---------------------------------------------------------------- fakes and helpers
class FakeSession:
    """The WorkerSession seam with the SSH parts stripped (same trick as test_ssh_provider)."""

    def __init__(self, base_url: str = "http://worker.test", host: str = "ousaisrvr",
                 fail: str | None = None) -> None:
        self.base_url, self.host, self._fail = base_url, host, fail
        self.ensure_calls, self.closed = 0, 0

    def ensure(self) -> None:
        self.ensure_calls += 1
        if self._fail:
            raise RemoteComputeError("connect", self._fail)

    def close(self) -> None:
        self.closed += 1


def make_provider(handler, **session_kw) -> SshProvider:
    """A provider over a MockTransport-backed fake session (no SSH, no real network)."""
    p = SshProvider(session=FakeSession(**session_kw))
    p._http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://worker.test",
                           timeout=30.0)
    return p


def handshake_response(version: str | None = None) -> dict:
    from fusionlab import worker

    return {"worker_version": version or ssh_mod._local_version(), "worker_protocol_version": 1,
            "tasks": sorted(worker.TASKS), "artifacts": {}, "torch": "3.0", "warp": "1.0",
            "cuda_available": False, "device_name": "cpu"}


def json_handler(routes: dict):
    def handle(request: httpx.Request) -> httpx.Response:
        entry = routes.get(f"{request.method} {request.url.path}")
        if entry is None:
            return httpx.Response(404, json={"detail": "Not Found"})
        return entry(request) if isinstance(entry, httpx.Response) else httpx.Response(200, json=entry)
    return handle


class StubRemote:
    """A provider-shaped stand-in: run_task and snapshot only need name/host/submit/result/capabilities/health.
    With fail_first, the first remote run dies mid-job and the next one succeeds (recovery)."""

    name, host = "ssh", "ousaisrvr"

    def __init__(self, fail_first: bool = False) -> None:
        self.calls, self.fail_first = 0, fail_first

    def submit(self, task: str, inputs: dict) -> JobHandle:
        return JobHandle(job_id="j1", task=task, provider=self.name)

    def result(self, job: JobHandle) -> dict:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RemoteComputeError("job", "worker died mid-job")
        return {"ok": True}

    def capabilities(self) -> dict:
        return {"provider": self.name, "host": self.host, "surrogate_available": True,
                "eq_surrogate_available": True, "version_mismatch": None, "remote": handshake_response()}

    def health(self) -> None:
        return None


@pytest.fixture()
def clean(monkeypatch):
    """No selected provider, no status state leaking between tests (and out of them). The env pin
    keeps the suite hermetic: a user's repo-root fusionlab.toml must not flip these tests to ssh."""
    monkeypatch.setenv("FUSIONLAB_COMPUTE_PROVIDER", "local")
    monkeypatch.setattr(compute, "_provider", None)
    status._reset_for_tests()
    yield
    status._reset_for_tests()


# ---------------------------------------------------------------- GET /compute
def test_local_shape(clean):
    body = client.get("/compute").json()
    assert body["provider"] == "local" and body["reachable"] is True
    assert body["fallback"] is False and body["reason"] is None
    assert body["host"] is None and body["worker"] is None and body["version_mismatch"] is None


def test_ssh_handshake_surfaced(clean, monkeypatch):
    monkeypatch.setattr(compute, "_provider",
                        make_provider(json_handler({"GET /v1/capabilities": handshake_response(),
                                                    "GET /health": {"ok": True}})))
    body = client.get("/compute").json()
    assert body["provider"] == "ssh" and body["reachable"] is True and body["host"] == "ousaisrvr"
    assert body["worker"]["worker_version"] == ssh_mod._local_version()
    assert body["worker"]["cuda_available"] is False and body["worker"]["device_name"] == "cpu"
    assert body["version_mismatch"] is None and body["fallback"] is False


def test_version_mismatch_is_surfaced(clean, monkeypatch):
    monkeypatch.setattr(compute, "_provider",
                        make_provider(json_handler({"GET /v1/capabilities": handshake_response("9.9.9"),
                                                    "GET /health": {"ok": True}})))
    body = client.get("/compute").json()
    assert body["version_mismatch"] == {"local": ssh_mod._local_version(), "remote": "9.9.9"}


def test_unreachable_worker_is_a_status_not_an_error(clean, monkeypatch):
    """The brief's no-5xx rule: connect, autostart or handshake failure reads as fallback + reason."""
    monkeypatch.setattr(compute, "_provider",
                        make_provider(json_handler({}), fail="ssh to 'ousaisrvr' failed: no route"))
    r = client.get("/compute")
    assert r.status_code == 200
    body = r.json()
    assert body["reachable"] is False and body["fallback"] is True and body["worker"] is None
    assert "no route" in body["reason"]


def test_missing_remote_extra_raises_the_install_hint(monkeypatch):
    """Without the optional extra, configuring provider="ssh" is a status that names the fix —
    not a bare ModuleNotFoundError and not a 5xx."""
    monkeypatch.setitem(sys.modules, "asyncssh", None)   # forces `import asyncssh` to ImportError
    session = object.__new__(ssh_mod._SshSession)        # no loop thread needed for this path
    session.host = "gpu-lab"
    with pytest.raises(ssh_mod.RemoteComputeError, match=r"uv sync --extra remote"):
        asyncio.run(session._ensure_async())


def test_dead_worker_between_handshakes_is_unreachable(clean, monkeypatch):
    """A worker that answered the handshake but died since: the live health probe reports it and drops
    the stale session so the next use reconnects instead of trusting the cache."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/capabilities":
            return httpx.Response(200, json=handshake_response())
        raise httpx.ConnectError("refused")

    p = make_provider(handler)
    monkeypatch.setattr(compute, "_provider", p)
    p.capabilities()   # a stale handshake exists
    body = client.get("/compute").json()
    assert body["reachable"] is False and body["fallback"] is True and "/health" in body["reason"]
    assert p._remote is None and p._session.closed == 1


def test_snapshot_is_cached_within_the_ttl(clean, monkeypatch):
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, json=handshake_response())

    monkeypatch.setattr(compute, "_provider", make_provider(handler))
    a, b = status.snapshot(), status.snapshot()
    assert hits["n"] == 2 and a == b   # one handshake + one health probe for two polls


# ---------------------------------------------------------------- the fallback runner
def test_run_task_falls_back_and_records(clean, monkeypatch):
    monkeypatch.setattr(compute, "_provider", make_provider(lambda request: (_ for _ in ()).throw(
        httpx.ConnectError("refused"))))
    out = status.run_task("model_metrics", {"model": "surrogate"})   # the local re-run answers
    assert "block_size" in out            # the local metrics JSON, not a remote error
    fb = status.fallback_state()
    assert fb["fallback"] is True and "refused" in fb["reason"]
    assert status.provenance("model_metrics") == {"provider": "local", "host": None}


def test_run_task_remote_success_labels_host_and_clears(clean):
    stub = StubRemote()
    compute._provider = stub   # fixture already reset it; replaced directly for clarity
    status.record_fallback("earlier failure")
    assert status.run_task("model_metrics", {"model": "surrogate"}) == {"ok": True}
    assert status.fallback_state()["fallback"] is False
    assert status.provenance("model_metrics") == {"provider": "ssh", "host": "ousaisrvr"}


def test_run_fallback_survives_a_healthy_probe_until_a_remote_run_succeeds(clean):
    """A mid-job remote failure with the worker healthy: the banner state stays until a remote run
    actually succeeds — 'running on Local CPU' stays true until then."""
    stub = StubRemote(fail_first=True)
    compute._provider = stub
    assert status.snapshot(0.0)["fallback"] is False
    status.run_task("model_metrics", {"model": "surrogate"})   # first call dies mid-job
    body = status.snapshot(0.0)
    assert body["fallback"] is True and "mid-job" in body["reason"]
    status.run_task("model_metrics", {"model": "surrogate"})   # the next run tries remote again
    assert status.snapshot(0.0)["fallback"] is False


# ---------------------------------------------------------------- job progress (HARN-1)
def test_run_task_exposes_the_remote_job_while_it_runs(clean, monkeypatch):
    """run_task's remote job is visible to active_job() from submit until result returns — the seam
    the progress route watches — and is dropped once collected."""
    stub = StubRemote()
    seen = []

    real_result = stub.result

    def slow_result(job):
        seen.append(status.active_job("model_metrics"))
        return real_result(job)

    monkeypatch.setattr(stub, "result", slow_result)
    compute._provider = stub
    assert status.run_task("model_metrics", {"model": "surrogate"}) == {"ok": True}
    assert seen == [(stub, JobHandle(job_id="j1", task="model_metrics", provider="ssh"))]
    assert status.active_job("model_metrics") is None


def _drain_sse(response) -> str:
    return b"".join(response.iter_raw()).decode()


def test_progress_route_ends_quietly_with_no_job(clean, monkeypatch):
    """No job of that task is running: a bounded wait, then a clean stream end — 200 and no error,
    never a reconnect bait for the browser."""
    monkeypatch.setattr(status, "PROGRESS_WAIT_S", 0.3)
    with client.stream("GET", "/compute/progress?task=model_metrics") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert _drain_sse(r) == ""


def test_progress_route_streams_a_live_local_job(clean, monkeypatch):
    """A local job mid-run when the route opens: the relay replays queued/running from zero, then
    streams the terminal event live — the worker's exact vocabulary."""
    monkeypatch.setattr(status, "PROGRESS_GRACE_S", 0.2)
    from fusionlab.compute import local as local_mod

    started, release = threading.Event(), threading.Event()

    def slow(inputs):
        started.set()
        release.wait(timeout=10)
        return {"ok": True}

    monkeypatch.setitem(local_mod.TASKS, "slow_task", slow)
    runner = threading.Thread(target=lambda: status.run_task("slow_task", {}))
    runner.start()
    assert started.wait(timeout=5)   # the body is running: the job is discoverable
    try:
        with client.stream("GET", "/compute/progress?task=slow_task") as r:
            assert r.status_code == 200
            body = _drain_sse(r)
    finally:
        release.set()
        runner.join(timeout=10)
    assert "event: queued\ndata: " in body
    assert "event: running\ndata: " in body
    assert "event: completed\ndata: " in body


def test_progress_route_follows_the_fallback_rerun(clean, monkeypatch):
    """A remote job that dies mid-job: the relay streams the remote's terminal error, then follows the
    local re-run the fallback performs — the UI keeps ticking instead of freezing at an error."""
    monkeypatch.setattr(status, "PROGRESS_GRACE_S", 1.0)
    from fusionlab.compute import local as local_mod

    local_go = threading.Event()

    def slow_local(inputs):
        local_go.wait(timeout=10)
        return {"ok": True, "where": "local"}

    monkeypatch.setitem(local_mod.TASKS, "slow_task", slow_local)

    class DyingRemote:
        """A remote whose job streams events, then dies — held open until the relay has watched it."""

        name, host = "ssh", "ousaisrvr"

        def __init__(self) -> None:
            self.stream_seen = threading.Event()

        def submit(self, task: str, inputs: dict) -> JobHandle:
            return JobHandle(job_id="j1", task=task, provider=self.name)

        def result(self, job: JobHandle) -> dict:
            self.stream_seen.wait(timeout=5)
            raise RemoteComputeError("job", "worker died mid-job")

        async def stream(self, job: JobHandle):
            yield {"event": "queued", "task": job.task}
            yield {"event": "running", "task": job.task}
            self.stream_seen.set()   # the relay is attached: result() may die now
            yield {"event": "error", "error": "worker died mid-job"}

    compute._provider = DyingRemote()
    runner = threading.Thread(target=lambda: status.run_task("slow_task", {}))
    runner.start()
    threading.Timer(1.0, local_go.set).start()   # let the re-run's progress be visible, then finish it
    try:
        with client.stream("GET", "/compute/progress?task=slow_task") as r:
            assert r.status_code == 200
            body = _drain_sse(r)
    finally:
        runner.join(timeout=10)
    assert "event: error" in body and "worker died mid-job" in body
    assert "event: completed" in body   # the fallback re-run's terminal, same vocabulary


# ---------------------------------------------------------------- cancellation (HARN-2)
def test_cancel_route_stops_the_remote_job_and_falls_back(clean, monkeypatch):
    """The full HARN-2 acceptance through the app's HTTP surface: POST /compute/cancel cancels the
    named task's in-flight remote job, result() raises the typed cancelled error, and run_task's
    EXISTING fallback re-runs the work locally — the fallback reason and the task's provenance say
    honestly where the answer came from (remote cancelled, answer local)."""
    from fusionlab.compute.ssh import RemoteComputeError

    class CancellingRemote:
        """A remote whose result() blocks until cancel() arrives, then dies exactly the way a
        cancelled job does — the signal run_task's fallback is built on."""
        name = "ssh"
        host = "cancelsrvr"

        def __init__(self) -> None:
            self.cancel_calls = 0
            self._cancelled = threading.Event()

        def submit(self, task: str, inputs: dict) -> JobHandle:
            return JobHandle(job_id="j1", task=task, provider=self.name)

        def result(self, job: JobHandle) -> dict:
            self._cancelled.wait(timeout=10)
            raise RemoteComputeError("job", "remote job was cancelled")

        def cancel(self, job: JobHandle) -> None:
            self.cancel_calls += 1
            self._cancelled.set()

    from fusionlab.compute import local as local_mod

    def local_run(inputs):
        return {"where": "local", "ok": True}

    monkeypatch.setitem(local_mod.TASKS, "slow_task", local_run)

    fake = CancellingRemote()
    monkeypatch.setattr(compute, "_provider", fake)
    out: dict = {}
    runner = threading.Thread(target=lambda: out.update(result=status.run_task("slow_task", {})))
    runner.start()
    for _ in range(100):                     # wait until the remote job is registered in-flight
        if status.active_job("slow_task") is not None:
            break
        time.sleep(0.05)
    assert status.active_job("slow_task") is not None

    r = client.post("/compute/cancel?task=slow_task")
    assert r.status_code == 200
    assert r.json() == {"job_id": "j1", "task": "slow_task", "provider": "ssh", "cancelling": True}

    runner.join(timeout=10)
    assert out["result"] == {"where": "local", "ok": True}   # the fallback re-run's answer
    assert fake.cancel_calls == 1
    state = status.fallback_state()
    assert state["fallback"] and "cancelled" in state["reason"]
    assert status.provenance("slow_task")["provider"] == "local"


def test_cancel_route_on_a_local_job_is_honest_about_the_noop(clean, monkeypatch):
    """Local jobs run synchronously in-process: the route answers 'cancelling': false with the
    note, never pretends to stop anything, and the job finishes normally with local provenance."""
    from fusionlab.compute import local as local_mod

    started, release = threading.Event(), threading.Event()

    def slow_local(inputs):
        started.set()
        release.wait(timeout=10)
        return {"where": "local"}

    monkeypatch.setitem(local_mod.TASKS, "slow_task", slow_local)
    out: dict = {}
    runner = threading.Thread(target=lambda: out.update(result=status.run_task("slow_task", {})))
    runner.start()
    for _ in range(100):                     # wait until the local job is registered in-flight
        if status.active_job("slow_task") is not None:
            break
        time.sleep(0.05)
    assert status.active_job("slow_task") is not None

    r = client.post("/compute/cancel?task=slow_task")
    assert r.status_code == 200
    body = r.json()
    assert body["cancelling"] is False and body["provider"] == "local" and "note" in body

    release.set()
    runner.join(timeout=10)
    assert out["result"] == {"where": "local"}   # untouched by the no-op cancel
    assert status.provenance("slow_task")["provider"] == "local"


def test_cancel_route_404s_when_nothing_is_in_flight(clean):
    """With no job running the route is a plain 404 — the UI affordance is only ever shown for a
    job the progress relay is streaming, so this path is the stale-tab / already-finished case."""
    r = client.post("/compute/cancel?task=benchmark")
    assert r.status_code == 404
    assert "benchmark" in r.json()["detail"] and "no" in r.json()["detail"]
