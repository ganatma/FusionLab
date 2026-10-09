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


def _wait_result(client, job_id: str, tries: int = 240):
    """Poll until the job is terminal; returns the 200 (frames) or 500 (error JSON) response."""
    for _ in range(tries):
        r = client.get(f"/v1/jobs/{job_id}/result")
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
    assert set(caps["models"]) == {"surrogate.pt", "eq_surrogate.pt"}
    assert caps["artifacts"] == {}                            # fresh cache: nothing fetched yet


def test_capabilities_models_carry_real_hashes(client):
    from fusionlab import eq_surrogate, surrogate

    caps = client.get("/v1/capabilities").json()
    assert caps["models"]["surrogate.pt"]["sha256"] == worker.sha256_file(surrogate.MODEL_FILE)
    assert caps["models"]["eq_surrogate.pt"]["bytes"] == eq_surrogate.MODEL_FILE.stat().st_size


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


# ---------------------------------------------------------------- runner safety net
def test_task_body_exception_becomes_error_state(client, monkeypatch):
    def boom(_inputs, _progress):
        raise ZeroDivisionError("boom")

    monkeypatch.setitem(worker.TASKS, "model_metrics", dc_replace(worker.TASKS["model_metrics"], run=boom))
    job_id = client.post("/v1/jobs", json={"task": "model_metrics", "inputs": {"model": "surrogate"}}).json()["job_id"]
    r = client.get(f"/v1/jobs/{job_id}/result")
    assert r.status_code == 500 and r.json()["error_type"] == "ZeroDivisionError" and "boom" in r.json()["detail"]
