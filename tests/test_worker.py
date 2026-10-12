"""The fusionlab worker over an ASGI client, CPU-only: handshake, fixed registry, depth-1 queue,
SSE events, FLF1 result framing, the sha256 artifact cache, and cancellation.

No GPU and no SSH here — this is the worker's own surface. The provider side lives in
test_ssh_provider.py; its loopback tests start this same app under a real uvicorn server.
"""

import hashlib
import time
from dataclasses import replace as dc_replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fusionlab import frames, worker

needs_surrogate = pytest.mark.skipif(not worker.shipped_models().get("surrogate.pt"),
                                     reason="models/surrogate.pt missing: run `make train`")


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """A TestClient over the worker app with a clean registry and an isolated artifact cache."""
    monkeypatch.setenv("FUSIONLAB_WORKER_ARTIFACTS", str(tmp_path / "artifacts"))
    worker._JOBS._jobs.clear()
    worker._HANDSHAKE = None
    with TestClient(worker.app) as c:
        yield c
    worker._JOBS._jobs.clear()
    worker._HANDSHAKE = None


def _wait_result(client, job_id: str, tries: int = 240, headers: dict | None = None):
    """Poll until the job is terminal; returns the 200 (frames) or 500 (error JSON) response."""
    for _ in range(tries):
        r = client.get(f"/v1/jobs/{job_id}/result", headers=headers)
        if r.status_code in (200, 500):
            return r
        assert r.status_code == 409, f"unexpected: {r.status_code} {r.text[:300]}"
        time.sleep(0.25)
    raise AssertionError("job did not finish in time")


def _event_names(client, job_id: str) -> list[str]:
    return [line[len("event: "):] for line in client.get(f"/v1/jobs/{job_id}/events").text.splitlines()
            if line.startswith("event: ")]


def _sse_events(client, job_id: str) -> list[dict]:
    """Parse the worker's SSE stream for one job (TestClient reads it to the terminal event)."""
    import json

    events, current = [], None
    for line in client.get(f"/v1/jobs/{job_id}/events").text.splitlines():
        if line.startswith("event: "):
            current = {"event": line[len("event: "):]}
        elif line.startswith("data: ") and current is not None:
            current.update(json.loads(line[len("data: "):]))
            events.append(current)
    return events


# ---------------------------------------------------------------- handshake
def test_health_is_ok(client):
    assert client.get("/health").json() == {"ok": True}


def test_capabilities_handshake(client):
    caps = client.get("/v1/capabilities").json()
    assert caps["worker_protocol_version"] == worker.WORKER_PROTOCOL_VERSION == 1
    assert caps["tasks"] == sorted(worker.TASKS)
    assert set(worker.TASKS) >= {"trace_fieldlines", "run_surrogate", "run_eq_surrogate", "benchmark"}
    assert isinstance(caps["cuda_available"], bool)
    assert caps["torch"] and caps["warp"]                     # versions of the install that runs the jobs
    assert set(caps["models"]) == {"surrogate.pt", "eq_surrogate.pt", "qlknn.pt"}
    assert caps["artifacts"] == {}                            # fresh cache: nothing fetched yet


def test_capabilities_models_carry_real_hashes(client):
    from fusionlab import eq_surrogate, surrogate

    caps = client.get("/v1/capabilities").json()
    assert caps["models"]["surrogate.pt"]["sha256"] == worker.sha256_file(surrogate.MODEL_FILE)
    assert caps["models"]["eq_surrogate.pt"]["bytes"] == eq_surrogate.MODEL_FILE.stat().st_size


@needs_surrogate
def test_handshake_weight_hashes_are_cached(monkeypatch, client):
    """HARN-3: weights are immutable once trained — the second handshake re-reads no file."""
    worker._SHA256_CACHE.clear()
    reads = {"n": 0}
    real = worker._sha256_file

    def counting(path):
        reads["n"] += 1
        return real(path)

    monkeypatch.setattr(worker, "_sha256_file", counting)
    first = client.get("/v1/capabilities").json()["models"]
    assert reads["n"] == 3                                # all three shipped weights, hashed exactly once
    assert client.get("/v1/capabilities").json()["models"] == first
    assert reads["n"] == 3                                # zero file reads on the second handshake


# ---------------------------------------------------------------- registry and validation
def test_unknown_task_is_rejected_with_clear_error(client):
    r = client.post("/v1/jobs", json={"task": "rm -rf /", "inputs": {}})
    assert r.status_code == 422
    assert "rm -rf /" in r.json()["detail"] and "The registry is the worker's whole surface" in r.json()["detail"]


def test_schema_violations_are_rejected(client):
    missing = client.post("/v1/jobs", json={"task": "run_surrogate_shot", "inputs": {"shot": {}}})
    assert missing.status_code == 422 and "P_loss_MW" in missing.json()["detail"]
    extra = client.post("/v1/jobs", json={"task": "benchmark", "inputs": {"profile": "quick", "cmd": "id"}})
    assert extra.status_code == 422   # extra="forbid": inputs outside the schema never reach a task


def test_malformed_framing_is_rejected(client):
    """Arrays ride as {__fl_frame_b64: ...}; garbage inside the envelope dies before any task runs."""
    r = client.post("/v1/jobs", json={"task": "run_eq_surrogate",
                                      "inputs": {"eq_inputs": {"__fl_frame_b64": "!!!not-base64!!!"}}})
    assert r.status_code == 422 and "malformed job inputs" in r.json()["detail"]


# ---------------------------------------------------------------- job lifecycle
def test_fast_job_roundtrips_through_frames(client):
    r = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}})
    assert r.status_code == 201 and r.json()["state"] == "queued"
    rr = _wait_result(client, r.json()["job_id"])
    assert rr.status_code == 200
    assert rr.headers["content-type"] == worker.RESULT_MEDIA_TYPE
    out = frames.unpack_result(rr.content)
    assert {"best_on_session_split", "block_size", "data", "device"} <= set(out)
    assert _event_names(client, r.json()["job_id"]) == ["queued", "running", "completed"]


def test_completed_job_result_is_consumed_once(client):
    job_id = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}}).json()["job_id"]
    assert _wait_result(client, job_id).status_code == 200
    again = client.get(f"/v1/jobs/{job_id}/result")
    assert again.status_code == 409 and "already consumed" in again.json()["detail"]


def test_depth_one_queue_rejects_while_busy(client):
    first = client.post("/v1/jobs", json={"task": "benchmark", "inputs": {"profile": "quick"}})
    assert first.status_code == 201
    second = client.post("/v1/jobs", json={"task": "benchmark", "inputs": {"profile": "quick"}})
    assert second.status_code == 409
    assert first.json()["job_id"] in second.json()["detail"]
    assert _wait_result(client, first.json()["job_id"]).status_code == 200


def test_benchmark_result_carries_measured_rows(client):
    job_id = client.post("/v1/jobs", json={"task": "benchmark", "inputs": {"profile": "quick"}}).json()["job_id"]
    out = frames.unpack_result(_wait_result(client, job_id).content)
    assert out["profile"] == "quick" and out["n_slices"] > 0
    labels = [row["label"] for row in out["rows"]]
    assert len(labels) == 5 and any("Replay" in label for label in labels)


def test_sse_events_stream_to_completion_in_order(client):
    job_id = client.post("/v1/jobs", json={"task": "benchmark", "inputs": {"profile": "quick"}}).json()["job_id"]
    events = _sse_events(client, job_id)
    names = [e["event"] for e in events]
    assert names[0] == "queued" and names[1] == "running" and names[-1] == "completed"
    assert any(e["event"] == "progress" and "section" in e for e in events[2:-1])


def test_sse_events_are_replayed_from_zero_for_a_late_subscriber(client):
    """PR 3's chip may attach after the job ran: the stream replays the full history, then ends."""
    job_id = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}}).json()["job_id"]
    _wait_result(client, job_id)
    assert _event_names(client, job_id) == ["queued", "running", "completed"]


@needs_surrogate
def test_surrogate_task_roundtrips_real_features_and_weights(client):
    """The composite replay path remotely: a real shot plus its replayed P_loss in, the per-slice
    correction array back — the shot dict rides as FLF1 frames both ways, exactly as the provider
    sends it (frames.encode_inputs before POST)."""
    from fusionlab import mast, physics

    shot = mast.load_shot(30166)
    p_loss = physics.replay(shot)["P_loss_MW"]
    inputs = frames.encode_inputs({"shot": shot, "P_loss_MW": p_loss})
    r = client.post("/v1/jobs", json={"task": "run_surrogate_shot", "inputs": inputs})
    assert r.status_code == 201, r.text
    out = frames.unpack_result(_wait_result(client, r.json()["job_id"]).content)
    assert np.asarray(out).shape == shot["t_s"].shape


# ---------------------------------------------------------------- artifacts
def test_artifact_upload_is_content_addressed(client):
    blob = b"weight-bytes-" + str(time.time()).encode()
    digest = hashlib.sha256(blob).hexdigest()
    first = client.post(f"/v1/artifacts/{digest}", content=blob)
    assert first.status_code == 201
    again = client.post(f"/v1/artifacts/{digest}", content=blob)   # same bytes: idempotent, not a duplicate
    assert again.status_code == 201 and again.json()["sha256"] == first.json()["sha256"]
    assert digest in client.get("/v1/capabilities").json()["artifacts"]   # the cache listing the handshake carries


def test_artifact_upload_rejects_mismatched_and_oversized_blobs(client):
    digest = hashlib.sha256(b"other-bytes").hexdigest()
    assert client.post(f"/v1/artifacts/{digest}", content=b"other-bytes!").status_code == 422
    assert client.post(f"/v1/artifacts/{'0' * 64}", content=b"x" * (worker.MAX_ARTIFACT_BYTES + 1)).status_code == 413
    assert client.post("/v1/artifacts/nothex", content=b"x").status_code == 422


def test_concurrent_uploads_leave_no_temp_litter(client):
    """HARN-3: artifact temps are pid-unique dot-files published with os.replace — concurrent
    uploads of the same and of different hashes leave exactly the hash-named weights, no litter."""
    import concurrent.futures

    blobs = [b"weight-bytes-a", b"weight-bytes-b"]
    digests = [hashlib.sha256(b).hexdigest() for b in blobs]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(client.post, f"/v1/artifacts/{digests[i % 2]}", content=blobs[i % 2])
                for i in range(6)]
        assert all(f.result().status_code == 201 for f in futs)
    root = worker.artifact_dir()
    assert sorted(p.name for p in root.iterdir()) == sorted(digests)   # zero .tmp* litter
    assert (root / digests[0]).read_bytes() == blobs[0]


def test_job_rejects_unsatisfied_artifact_hash(client):
    r = client.post("/v1/jobs", json={"task": "run_surrogate", "inputs": {"features": {}},
                                      "artifact_refs": [{"role": "surrogate.pt", "sha256": "f" * 64}]})
    assert r.status_code == 422 and "artifact ref not satisfied" in r.json()["detail"]


@needs_surrogate
def test_job_resolves_uploaded_artifact_over_shipped_weights(client, tmp_path):
    """A ref whose hash is neither shipped nor cached is uploaded first; the job then resolves against
    the uploaded bytes — the drift contract SshProvider depends on. The fake weight cannot load, so the
    job fails remotely with the loader's error: proof the uploaded bytes were the ones used."""
    from fusionlab import mast, physics

    fake = tmp_path / "surrogate.pt"
    fake.write_bytes(b"not-a-real-weight-dict")           # a valid upload, deliberately unloadable
    digest = hashlib.sha256(fake.read_bytes()).hexdigest()
    assert client.post(f"/v1/artifacts/{digest}", content=fake.read_bytes()).status_code == 201
    shot = mast.load_shot(30166)
    p_loss = physics.replay(shot)["P_loss_MW"]
    inputs = frames.encode_inputs({"shot": shot, "P_loss_MW": p_loss})
    r = client.post("/v1/jobs", json={"task": "run_surrogate_shot", "inputs": inputs,
                                      "artifact_refs": [{"role": "surrogate.pt", "sha256": digest}]})
    assert r.status_code == 201, r.text
    rr = _wait_result(client, r.json()["job_id"])
    assert rr.status_code == 500 and rr.json()["error_type"] is not None


def test_job_status_reports_state_without_payload_and_unknown_ids_404(client):
    assert client.get("/v1/jobs/nope").status_code == 404
    job_id = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}}).json()["job_id"]
    early = client.get(f"/v1/jobs/{job_id}").json()
    assert early["task"] == "model_metrics" and early["state"] in ("queued", "running", "completed")
    assert _wait_result(client, job_id).status_code == 200
    assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "completed"


# ---------------------------------------------------------------- cancellation
def test_cancellation_cancels_a_running_job(client):
    job_id = client.post("/v1/jobs", json={"task": "benchmark", "inputs": {"profile": "quick"}}).json()["job_id"]
    assert client.post(f"/v1/jobs/{job_id}/cancel").json()["cancelling"] is True
    names = _event_names(client, job_id)      # the SSE stream blocks to the terminal event
    assert names[0] == "queued" and names[-1] == "cancelled"
    late = client.post(f"/v1/jobs/{job_id}/cancel")
    assert late.status_code == 409
    got = client.get(f"/v1/jobs/{job_id}/result")
    assert got.status_code == 409 and "cancelled" in got.json()["detail"]


# ---------------------------------------------------------------- runner cancellation (HARN-3)
def test_cancelling_the_runner_task_frees_the_depth_one_slot(client, monkeypatch):
    """A cancelled runner — the task itself, not a cooperative cancel (worker shutdown, a reaped
    stuck job) — used to leave state 'running' forever and 409 every later submit. It must park
    terminal and free the depth-1 slot."""

    def slow(_inputs, _progress):
        time.sleep(2)
        return {}

    monkeypatch.setitem(worker.TASKS, "benchmark", dc_replace(worker.TASKS["benchmark"], run=slow))
    job_id = client.post("/v1/jobs", json={"task": "benchmark", "inputs": {"profile": "quick"}}).json()["job_id"]
    job = worker._JOBS.get(job_id)
    for _ in range(40):                       # wait until the runner is inside the task body
        if job.state == "running":
            break
        time.sleep(0.05)
    assert job.state == "running"

    job.runner.get_loop().call_soon_threadsafe(job.runner.cancel)   # cancel the runner task itself
    for _ in range(40):
        if job.state in worker.TERMINAL_STATES:
            break
        time.sleep(0.05)
    assert job.state == "cancelled"
    assert job.completed_at is not None   # HARN-3: terminal jobs carry the FIFO eviction stamp
    assert _event_names(client, job_id)[-1] == "cancelled"

    nxt = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}})
    assert nxt.status_code == 201, nxt.text   # the slot freed: no 409
    assert _wait_result(client, nxt.json()["job_id"]).status_code == 200


def test_terminal_job_eviction_is_fifo_not_random(monkeypatch):
    """Eviction sorted by random uuid4 hex, so a fresh unconsumed result could be evicted while an
    old consumed one lingered — its later result fetch would 404. Evict oldest completion first."""
    worker._JOBS._jobs.clear()
    clock = iter(range(1, 1000))
    monkeypatch.setattr(worker, "_now", lambda: float(next(clock)))
    jobs = []
    for _ in range(17):                          # depth-1 book: create, finish, then the next
        j = worker._JOBS.create("benchmark")
        j.park("completed")                      # the real stamping path, deterministic clock
        jobs.append(j)
    jobs[-1].result = b"fresh-unconsumed"        # the newest terminal job's result is not yet fetched
    fresh = worker._JOBS.create("benchmark")     # 18th job: eviction makes room (16 kept + 1 incoming)
    evicted = [j.job_id for j in jobs if j.job_id not in worker._JOBS._jobs]
    assert evicted == [jobs[0].job_id, jobs[1].job_id]   # the two oldest completions, in order
    assert jobs[-1].result == b"fresh-unconsumed"        # the fresh unconsumed result survived
    assert fresh.state == "queued"
    worker._JOBS._jobs.clear()


# ---------------------------------------------------------------- runner safety net
def test_task_body_exception_becomes_error_state(client, monkeypatch):
    def boom(_inputs, _progress):
        raise ZeroDivisionError("boom")

    monkeypatch.setitem(worker.TASKS, "model_metrics", dc_replace(worker.TASKS["model_metrics"], run=boom))
    job_id = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}}).json()["job_id"]
    r = client.get(f"/v1/jobs/{job_id}/result")
    assert r.status_code == 500 and r.json()["error_type"] == "ZeroDivisionError" and "boom" in r.json()["detail"]


# ---------------------------------------------------------------- auth (opt-in shared token, #45)
AUTH_ROUTES = [   # (method, path, body): every /v1/* route — the 401 matrix walks all of them
    ("get", "/v1/capabilities", None),
    ("post", "/v1/jobs", {"task": "model_metrics", "inputs": {"model": "surrogate"}}),
    ("get", "/v1/jobs/ghost", None),
    ("get", "/v1/jobs/ghost/events", None),
    ("get", "/v1/jobs/ghost/result", None),
    ("post", "/v1/jobs/ghost/cancel", None),
    ("post", f"/v1/artifacts/{'0' * 64}", b"x"),
]
_AUTH_IDS = [f"{m.upper()} {p}" for m, p, _ in AUTH_ROUTES]


def _call(client, method: str, path: str, body, headers: dict | None = None):
    if isinstance(body, bytes):
        return client.request(method, path, content=body, headers=headers)
    if body is not None:
        return client.request(method, path, json=body, headers=headers)
    return client.request(method, path, headers=headers)


def test_token_unset_is_todays_behavior(client):
    """Opt-in by absence: with no FUSIONLAB_WORKER_TOKEN, /v1 routes answer requests that carry no
    Authorization header at all — the pre-token worker, pinned here (the other tests pin the rest)."""
    assert worker.worker_token() == ""
    assert client.get("/v1/capabilities").status_code == 200
    assert client.post("/v1/jobs", json={"task": "model_metrics",
                                         "inputs": {"model": "surrogate"}}).status_code == 201


def test_handshake_advertises_the_auth_mode(client, monkeypatch):
    assert client.get("/v1/capabilities").json()["auth"] == "open"
    monkeypatch.setenv("FUSIONLAB_WORKER_TOKEN", "tk")
    r = client.get("/v1/capabilities", headers={"Authorization": "Bearer tk"})
    assert r.json()["auth"] == "token"


@pytest.mark.parametrize(("method", "path", "body"), AUTH_ROUTES, ids=_AUTH_IDS)
def test_401_when_the_token_header_is_missing(client, monkeypatch, method, path, body):
    """Auth runs before routing, validation, and queue state: a tokenless request to any /v1 route
    is a 401 with WWW-Authenticate — never the 404/422/201 that route itself would have answered."""
    monkeypatch.setenv("FUSIONLAB_WORKER_TOKEN", "tk")
    r = _call(client, method, path, body)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(("method", "path", "body"), AUTH_ROUTES, ids=_AUTH_IDS)
def test_401_when_the_token_is_wrong(client, monkeypatch, method, path, body):
    """Same matrix, wrong Bearer value: still a 401 before the route ever sees the request."""
    monkeypatch.setenv("FUSIONLAB_WORKER_TOKEN", "tk")
    r = _call(client, method, path, body, headers={"Authorization": "Bearer not-the-token"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_a_401_submission_never_touches_the_queue(client, monkeypatch):
    """The gate sits before body parsing and the depth-1 book: a rejected submit leaves no job state
    behind and the slot is free for the next (correctly authed) one."""
    monkeypatch.setenv("FUSIONLAB_WORKER_TOKEN", "tk")
    r = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert worker._JOBS._jobs == {}
    ok = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}},
                     headers={"Authorization": "Bearer tk"})
    assert ok.status_code == 201


def test_health_stays_open_with_a_token_set(client, monkeypatch):
    """The tunnel/autostart probe sends no token: /health must answer it even when the worker is
    token-gated, or a running worker reads as absent and autostart starts a doomed second one."""
    monkeypatch.setenv("FUSIONLAB_WORKER_TOKEN", "tk")
    assert client.get("/health").json() == {"ok": True}


def test_the_correct_token_passes_end_to_end(client, monkeypatch):
    """The happy path with the gate on: handshake and a full job roundtrip, every call bearing the
    Bearer header."""
    monkeypatch.setenv("FUSIONLAB_WORKER_TOKEN", "tk")
    headers = {"Authorization": "Bearer tk"}
    assert client.get("/v1/capabilities", headers=headers).json()["auth"] == "token"
    job_id = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}},
                         headers=headers).json()["job_id"]
    rr = _wait_result(client, job_id, headers=headers)
    assert rr.status_code == 200
    assert "block_size" in frames.unpack_result(rr.content)


needs_qlknn = pytest.mark.skipif(not worker.shipped_models().get("qlknn.pt"),
                                 reason="models/qlknn.pt missing: run scripts/convert_qlknn_weights.py")


@needs_qlknn
def test_qlknn_task_roundtrips_and_the_handshake_carries_its_weight(client):
    """The QLKNN rails remotely: the shipped qlknn.pt is advertised in the handshake's models and a
    bare job resolves against it — the per-slice (K, alpha) dict comes back through the FLF1 frames
    equal to the in-process module call."""
    from fusionlab import mast, physics, qlknn

    assert "qlknn.pt" in client.get("/v1/capabilities").json()["models"]
    shot = mast.load_shot(30166)
    p_loss = physics.replay(shot)["P_loss_MW"]
    inputs = frames.encode_inputs({"shot": shot, "P_loss_MW": p_loss})
    r = client.post("/v1/jobs", json={"task": "run_qlknn", "inputs": inputs})
    assert r.status_code == 201, r.text
    out = frames.unpack_result(_wait_result(client, r.json()["job_id"]).content)
    direct = qlknn.confinement_coeff(shot, p_loss)
    assert np.array_equal(out["K"], direct["K"]) and np.array_equal(out["alpha"], direct["alpha"])


@needs_qlknn
def test_qlknn_job_resolves_uploaded_artifact_over_shipped_weights(client, tmp_path):
    """The drift contract for the third weight: a ref whose hash matches nothing shipped or cached is
    uploaded first, and the job resolves against the uploaded bytes — the fake weight cannot load, so
    the failure is the loader's, proving those bytes were used."""
    fake = tmp_path / "qlknn.pt"
    fake.write_bytes(b"not-a-real-weight-dict")
    digest = hashlib.sha256(fake.read_bytes()).hexdigest()
    assert client.post(f"/v1/artifacts/{digest}", content=fake.read_bytes()).status_code == 201
    from fusionlab import mast, physics

    shot = mast.load_shot(30166)
    inputs = frames.encode_inputs({"shot": shot, "P_loss_MW": physics.replay(shot)["P_loss_MW"]})
    r = client.post("/v1/jobs", json={"task": "run_qlknn", "inputs": inputs,
                                      "artifact_refs": [{"role": "qlknn.pt", "sha256": digest}]})
    assert r.status_code == 201, r.text
    rr = _wait_result(client, r.json()["job_id"])
    assert rr.status_code == 500 and rr.json()["error_type"] is not None
