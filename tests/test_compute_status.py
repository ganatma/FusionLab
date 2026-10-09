"""GET /compute and the fallback runner (PR 3): the local shape, a mocked-SSH handshake, an unreachable
worker as a status — never a 5xx — and provenance. No SSH; HTTP stays on httpx.MockTransport fakes."""

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
    """No selected provider, no status state leaking between tests (and out of them)."""
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
