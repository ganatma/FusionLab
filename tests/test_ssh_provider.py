"""The SshProvider: mocked-transport unit tests, mocked-asyncssh tunnel tests, and loopback
end-to-end tests (backend → SshProvider → the real worker app on 127.0.0.1).

No SSH server, no GPU, no network beyond loopback. The unit tests replace the HTTP layer with
httpx.MockTransport and the SSH layer with a fake asyncssh module; the loopback tests run the
actual worker under uvicorn so the wire contract — framing, SSE, artifact cache — is exercised
for real, minus only the tunnel (a loopback port stands in for it).
"""

import asyncio
import hashlib
import json
import socket
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from fusionlab import frames, worker
from fusionlab.compute import ssh as ssh_mod
from fusionlab.compute.provider import JobHandle
from fusionlab.compute.ssh import RemoteComputeError, SshProvider

needs_surrogate = pytest.mark.skipif(not worker.shipped_models().get("surrogate.pt"),
                                     reason="models/surrogate.pt missing: run `make train`")


# ---------------------------------------------------------------- fakes and helpers
class FakeSession:
    """The WorkerSession seam with the SSH parts stripped: the provider's HTTP just works."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.ensure_calls = 0
        self.closed = False

    def ensure(self) -> None:
        self.ensure_calls += 1

    def close(self) -> None:
        self.closed = True


def make_provider(handler, **kw) -> SshProvider:
    """A provider over a MockTransport-backed fake session (no SSH, no real network)."""
    p = SshProvider(session=FakeSession("http://worker.test"), **kw)
    p._http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://worker.test",
                           timeout=30.0)
    return p


def handshake_response(**overrides):
    """A worker capabilities payload matching THIS install's weights, unless overridden.
    Only existing files are hashed — a missing local weight set is reported honestly."""
    if "models" not in overrides:
        overrides["models"] = {name: {"sha256": ssh_mod.sha256_file(path), "bytes": path.stat().st_size}
                               for name, path in ssh_mod.local_weight_paths().items() if path.exists()}
    body = {"worker_version": ssh_mod._local_version(), "worker_protocol_version": 1,
            "tasks": sorted(worker.TASKS), "artifacts": {},
            "torch": "3.0", "warp": "1.0", "cuda_available": False, "device_name": "cpu"}
    body.update(overrides)
    return body


def json_handler(routes: dict):
    """Route table for MockTransport: 'METHOD /path' -> response value, Response, or handler."""
    def handle(request: httpx.Request) -> httpx.Response:
        entry = routes.get(f"{request.method} {request.url.path}")
        if entry is None:
            return httpx.Response(404, json={"detail": "Not Found"})
        out = entry(request) if callable(entry) else entry
        if isinstance(out, httpx.Response):
            return out
        return httpx.Response(200, json=out)
    return handle


async def collect(stream) -> list[dict]:
    return [e async for e in stream]


# ---------------------------------------------------------------- capabilities (handshake + drift)
def test_capabilities_reports_remote_tasks_and_model_availability():
    p = make_provider(json_handler({"GET /v1/capabilities": handshake_response()}))
    caps = p.capabilities()
    assert caps["provider"] == "ssh" and caps["host"] == "127.0.0.1"
    assert "benchmark" in caps["tasks"]
    assert caps["surrogate_available"] and caps["eq_surrogate_available"]
    assert caps["version_mismatch"] is None            # handshake_response mirrors this install


def test_capabilities_surfaces_version_drift_as_data_and_warning(caplog):
    p = make_provider(json_handler({"GET /v1/capabilities": handshake_response(worker_version="0.0.0-test")}))
    with caplog.at_level("WARNING", logger="fusionlab.compute"):
        caps = p.capabilities()
    assert caps["version_mismatch"] == {"local": ssh_mod._local_version(), "remote": "0.0.0-test"}
    assert any("version" in r.message.lower() for r in caplog.records)


def test_capabilities_survives_a_worker_without_version_keys():
    p = make_provider(json_handler({"GET /v1/capabilities": {"tasks": ["benchmark"], "models": {}}}))
    caps = p.capabilities()
    assert caps["surrogate_available"] is False        # an unknown remote reports honestly, not optimistically
    assert caps["version_mismatch"]["remote"] == ""


# ---------------------------------------------------------------- submit (refs, upload, rejections)
def test_submit_sends_refs_without_uploading_when_weights_match():
    """The common case: worker ships the same commit's weights — zero weight bytes over the wire."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, request.read()
        if request.url.path == "/v1/capabilities":
            return httpx.Response(200, json=handshake_response())
        return httpx.Response(201, json={"job_id": "j1", "state": "queued"})

    p = make_provider(handler)
    job = p.submit("run_surrogate_shot", {"shot": {}, "P_loss_MW": 1.0})
    assert (job.job_id, job.task, job.provider) == ("j1", "run_surrogate_shot", "ssh")
    payload = json.loads(seen["body"])
    assert payload["artifact_refs"] == [{"sha256": ssh_mod.sha256_file(ssh_mod.local_weight_paths()["surrogate.pt"]),
                                         "role": "surrogate.pt"}]


def test_submit_uploads_weights_the_worker_lacks(tmp_path, monkeypatch):
    fake = tmp_path / "surrogate.pt"
    fake.write_bytes(b"weights-the-worker-never-saw")
    monkeypatch.setattr(ssh_mod, "local_weight_paths", lambda: {"surrogate.pt": fake})
    uploads = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/capabilities":
            return httpx.Response(200, json=handshake_response(models={}, artifacts={}))
        if request.method == "POST" and request.url.path.startswith("/v1/artifacts/"):
            uploads.append(request.url.path.rsplit("/", 1)[-1])
            return httpx.Response(201, json={"sha256": uploads[-1], "bytes": len(request.content)})
        return httpx.Response(201, json={"job_id": "j2", "state": "queued"})

    p = make_provider(handler)
    p.submit("run_surrogate", {"features": {"Ip_MA": [0.8]}})
    assert uploads == [hashlib.sha256(fake.read_bytes()).hexdigest()]


def test_submit_of_a_weight_task_with_missing_local_weights_is_a_clean_error(tmp_path, monkeypatch):
    missing = tmp_path / "surrogate.pt"        # never created
    monkeypatch.setattr(ssh_mod, "local_weight_paths", lambda: {"surrogate.pt": missing})
    p = make_provider(json_handler({"GET /v1/capabilities": handshake_response(models={}, artifacts={})}))
    with pytest.raises(RemoteComputeError, match=r"models/surrogate.pt is missing.*make train"):
        p.submit("run_surrogate", {"features": {}})


def test_worker_is_the_single_registry_enforcement_point():
    """The provider does not second-guess task names; the worker's registry rejects them."""
    p = make_provider(json_handler({"GET /v1/capabilities": handshake_response(),
                                    "POST /v1/jobs": httpx.Response(
                                        422, json={"detail": "unknown task 'deploy ransomware'. The registry "
                                                           "is the worker's whole surface"})}))
    with pytest.raises(RemoteComputeError, match="unknown task"):
        p.submit("deploy ransomware", {})


def test_busy_worker_raises_clean_submit_error():
    p = make_provider(json_handler({"GET /v1/capabilities": handshake_response(),
                                    "POST /v1/jobs": httpx.Response(
                                        409, json={"detail": "worker busy: job x is running"})}))
    with pytest.raises(RemoteComputeError, match="worker busy"):
        p.submit("benchmark", {"profile": "quick"})


def test_validation_rejections_become_submit_errors():
    p = make_provider(json_handler({"GET /v1/capabilities": handshake_response(),
                                    "POST /v1/jobs": httpx.Response(422, json={"detail": "invalid inputs"})}))
    with pytest.raises(RemoteComputeError, match="worker rejected the job"):
        p.submit("benchmark", {"profile": "bogus"})


def test_connection_failure_becomes_a_submit_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    p = make_provider(handler)
    with pytest.raises(RemoteComputeError, match="ConnectError"):
        p.submit("benchmark", {"profile": "quick"})


# ---------------------------------------------------------------- result (poll, decode, failure)
def test_result_polls_then_decodes_the_framed_payload():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/result"):
            return httpx.Response(200, content=frames.pack_result({"value": 42}),
                                  headers={"content-type": worker.RESULT_MEDIA_TYPE})
        calls["n"] += 1
        return httpx.Response(200, json={"state": "completed" if calls["n"] >= 3 else "running"})

    p = make_provider(handler, result_timeout_s=30.0)
    out = p.result(JobHandle(job_id="j3", task="benchmark", provider="ssh"))
    assert out == {"value": 42} and calls["n"] >= 3


def test_result_surfaces_remote_task_failure_with_the_worker_text():
    p = make_provider(json_handler({"GET /v1/jobs/j4": {"state": "error", "error": "CUDA out of memory"}}))
    with pytest.raises(RemoteComputeError, match="CUDA out of memory") as ei:
        p.result(JobHandle(job_id="j4", task="benchmark", provider="ssh"))
    assert ei.value.phase == "job"


def test_result_on_a_cancelled_job():
    p = make_provider(json_handler({"GET /v1/jobs/j5": {"state": "cancelled"}}))
    with pytest.raises(RemoteComputeError, match="cancelled"):
        p.result(JobHandle(job_id="j5", task="benchmark", provider="ssh"))


def test_result_on_an_unknown_job():
    p = make_provider(json_handler({}))
    with pytest.raises(RemoteComputeError, match="unknown job"):
        p.result(JobHandle(job_id="j6", task="benchmark", provider="ssh"))


def test_result_times_out_cleanly(monkeypatch):
    monkeypatch.setattr(ssh_mod, "_POLL_S", 0.01)
    p = make_provider(json_handler({"GET /v1/jobs/j7": {"state": "running"}}), result_timeout_s=0.05)
    with pytest.raises(RemoteComputeError, match="still running"):
        p.result(JobHandle(job_id="j7", task="benchmark", provider="ssh"))


def test_result_with_a_corrupt_envelope_is_a_collect_error():
    def handler(request):
        if request.url.path.endswith("/result"):
            return httpx.Response(200, content=b"NOPE-not-frames")
        return httpx.Response(200, json={"state": "completed"})

    p = make_provider(handler)
    with pytest.raises(RemoteComputeError, match="corrupt result envelope"):
        p.result(JobHandle(job_id="j8", task="benchmark", provider="ssh"))


# ---------------------------------------------------------------- the SSH layer (mocked asyncssh)
class FakeResult:
    def __init__(self, exit_status: int = 0, stderr: str = "", stdout: str = "") -> None:
        self.exit_status, self.stderr, self.stdout = exit_status, stderr, stdout


class FakeConn:
    def __init__(self, fail_autostart: bool = False) -> None:
        self.run_commands: list[str] = []
        self.closed = False
        self._fail = fail_autostart

    async def run(self, cmd: str) -> FakeResult:
        self.run_commands.append(cmd)
        if self._fail:
            return FakeResult(exit_status=1, stderr="cd: no such directory")
        return FakeResult()

    async def forward_local_port(self, local_host, local_port, remote_host, remote_port):
        return SimpleNamespace(close=lambda: None)

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_asyncssh(monkeypatch):
    """asyncssh replaced wholesale; hosts pick the scenario: 'unreachable.test' fails to connect,
    'baddir.test' autostarts into a missing directory, anything else connects cleanly."""
    conns: list[FakeConn] = []

    async def connect(host, **kw):
        if host == "unreachable.test":
            raise OSError("no route to host")
        conn = FakeConn(fail_autostart=host == "baddir.test")
        conns.append(conn)
        return conn

    monkeypatch.setitem(sys.modules, "asyncssh", SimpleNamespace(connect=connect))
    return conns


def _session(host: str = "unit.test", local_port: int = 0, **kw) -> ssh_mod._SshSession:
    kw.setdefault("start_timeout_s", 1.5)
    return ssh_mod._SshSession(host, local_port=local_port, **kw)


def test_ssh_session_leaves_a_listening_worker_alone(fake_asyncssh, worker_url):
    """Worker already up (someone started it, or a previous job): tunnel + health, zero autostart."""
    session = _session(local_port=_free_url_port(worker_url))
    session.ensure()
    assert session.base_url == worker_url and session._conn is not None
    assert fake_asyncssh[0].run_commands == []          # nothing needed autostarting
    session.close()


def test_ssh_session_autostarts_when_nothing_is_listening(fake_asyncssh):
    session = _session(local_port=_free_port(), remote_dir="~/FusionLab", autostart=True)
    with pytest.raises(RemoteComputeError, match="did not come up within"):
        session.ensure()                       # nothing will listen on this port; that is the point
    assert len(fake_asyncssh) == 1 and len(fake_asyncssh[0].run_commands) == 1
    cmd = fake_asyncssh[0].run_commands[0]
    assert "cd ~/FusionLab" in cmd and "uv run fusionlab-worker" in cmd and "nohup" in cmd


def test_ssh_session_reports_a_failed_autostart_immediately(fake_asyncssh):
    session = _session("baddir.test", local_port=_free_port(), start_timeout_s=5.0)
    with pytest.raises(RemoteComputeError, match="autostart failed"):
        session.ensure()
    assert len(fake_asyncssh[0].run_commands) == 1


def test_ssh_session_without_autostart_fails_fast_with_instructions(fake_asyncssh):
    session = _session(local_port=_free_port(), autostart=False, start_timeout_s=5.0)
    with pytest.raises(RemoteComputeError, match="autostart is off"):
        session.ensure()
    assert all("fusionlab-worker" not in c for c in fake_asyncssh[0].run_commands)


def test_ssh_session_connect_failure_is_typed(fake_asyncssh):
    session = _session("unreachable.test", local_port=_free_port(), start_timeout_s=1.0)
    with pytest.raises(RemoteComputeError, match="ssh to 'unreachable.test' failed"):
        session.ensure()


# ---------------------------------------------------------------- loopback end-to-end (real worker)
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _free_url_port(url: str) -> int:
    return int(url.rsplit(":", 1)[-1])


@pytest.fixture(scope="module")
def worker_url(tmp_path_factory):
    """The real worker app under uvicorn on 127.0.0.1 — the tunnel's local end in the e2e tests."""
    import os

    import uvicorn

    artifact_dir = tmp_path_factory.mktemp("worker-artifacts")
    old = os.environ.get("FUSIONLAB_WORKER_ARTIFACTS")
    os.environ["FUSIONLAB_WORKER_ARTIFACTS"] = str(artifact_dir)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(worker.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "uvicorn worker did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
    if old is None:
        os.environ.pop("FUSIONLAB_WORKER_ARTIFACTS", None)
    else:
        os.environ["FUSIONLAB_WORKER_ARTIFACTS"] = old


def test_loopback_end_to_end_benchmark(worker_url):
    """backend → SshProvider → real worker on 127.0.0.1: handshake, submit, framed result, close."""
    p = SshProvider(session=FakeSession(worker_url))
    caps = p.capabilities()
    assert caps["provider"] == "ssh" and "benchmark" in caps["tasks"]
    assert caps["version_mismatch"] is None    # same install, same commit, same weights
    out = p.result(p.submit("benchmark", {"profile": "quick"}))
    assert out["profile"] == "quick" and len(out["rows"]) == 5
    p.close()


def test_loopback_stream_relays_worker_events(worker_url):
    p = SshProvider(session=FakeSession(worker_url))
    job = p.submit("model_metrics", {"model": "surrogate"})
    events = asyncio.run(collect(p.stream(job)))
    names = [e["event"] for e in events]
    assert names[0] == "queued" and names[-1] == "completed" and "running" in names
    p.close()


def test_loopback_stream_on_an_unknown_job(worker_url):
    p = SshProvider(session=FakeSession(worker_url))
    with pytest.raises(RemoteComputeError, match="unknown job"):
        asyncio.run(collect(p.stream(JobHandle(job_id="ghost", task="benchmark", provider="ssh"))))
    p.close()


@needs_surrogate
def test_loopback_upload_then_remote_failure_surfaces_cleanly(worker_url, tmp_path, monkeypatch):
    """The drift path end-to-end: local weights the worker lacks are uploaded for real, the job runs
    against them, and its failure arrives as RemoteComputeError carrying the remote text — the exact
    signal PR 3's fallback banner consumes."""
    from fusionlab import mast, physics

    fake = tmp_path / "surrogate.pt"
    fake.write_bytes(b"not-a-real-weight-dict")
    monkeypatch.setattr(ssh_mod, "local_weight_paths", lambda: {"surrogate.pt": fake})
    p = SshProvider(session=FakeSession(worker_url))
    shot = mast.load_shot(30166)
    inputs = {"shot": shot, "P_loss_MW": physics.replay(shot)["P_loss_MW"]}
    with pytest.raises(RemoteComputeError, match="remote task failed") as ei:
        p.result(p.submit("run_surrogate_shot", inputs))
    assert ei.value.phase == "job"
    p.close()


@needs_surrogate
def test_loopback_composite_task_matches_the_local_answer(worker_url):
    """The acceptance shape for the surrogate path: the remote run equals the in-process run."""
    import numpy as np

    from fusionlab import mast, physics
    from fusionlab.compute.local import LocalProvider

    shot = mast.load_shot(30166)
    inputs = {"shot": shot, "P_loss_MW": physics.replay(shot)["P_loss_MW"]}
    remote_p = SshProvider(session=FakeSession(worker_url))
    remote = remote_p.result(remote_p.submit("run_surrogate_shot", inputs))
    remote_p.close()
    local_p = LocalProvider()
    local = local_p.result(local_p.submit("run_surrogate_shot", inputs))
    assert remote.shape == local.shape == shot["t_s"].shape
    assert np.array_equal(remote, local)
