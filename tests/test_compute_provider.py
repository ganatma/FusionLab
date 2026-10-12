"""ComputeProvider seam: selection defaults to LocalProvider, config overrides parse, and LocalProvider's
tasks return exactly what the wrapped functions return (offline, CPU)."""

import ast
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from fusionlab import eq_surrogate, mast, qlknn, surrogate
from fusionlab.compute import JobHandle, LocalProvider, get_provider, provider_name
from fusionlab.physics import replay

needs_surrogate = pytest.mark.skipif(not surrogate.available(), reason="no trained model: run `make train`")
needs_eq = pytest.mark.skipif(not eq_surrogate.MODEL_FILE.exists(),
                              reason="models/eq_surrogate.pt missing: run scripts/train_eq_surrogate.py")


# ---------------------------------------------------------------- selection
def test_no_config_means_local(monkeypatch, tmp_path):
    monkeypatch.setattr("fusionlab.compute._provider", None)
    monkeypatch.setattr("fusionlab.compute._TOML_PATH", tmp_path / "fusionlab.toml")   # absent
    monkeypatch.delenv("FUSIONLAB_COMPUTE_PROVIDER", raising=False)
    assert provider_name() == "local"
    a = get_provider()
    assert isinstance(a, LocalProvider) and get_provider() is a   # one instance per process


def test_env_override_is_parsed(monkeypatch):
    monkeypatch.setattr("fusionlab.compute._provider", None)
    monkeypatch.setenv("FUSIONLAB_COMPUTE_PROVIDER", "local")
    assert provider_name() == "local"


def test_toml_override_is_parsed(monkeypatch, tmp_path):
    monkeypatch.setattr("fusionlab.compute._provider", None)
    monkeypatch.delenv("FUSIONLAB_COMPUTE_PROVIDER", raising=False)
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute]\nprovider = "local"\n')   # the shape the design outline shows for [compute]
    monkeypatch.setattr("fusionlab.compute._TOML_PATH", toml)
    assert provider_name() == "local"


def test_env_beats_toml_and_names_parse_before_they_construct(monkeypatch, tmp_path):
    monkeypatch.setattr("fusionlab.compute._provider", None)
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute]\nprovider = "local"\n')
    monkeypatch.setattr("fusionlab.compute._TOML_PATH", toml)
    monkeypatch.setenv("FUSIONLAB_COMPUTE_PROVIDER", "ssh")   # PR 2's provider: parsed now, constructed later
    assert provider_name() == "ssh"
    with pytest.raises(ValueError, match="ssh"):
        get_provider()


def test_unknown_provider_fails_loudly(monkeypatch):
    monkeypatch.setattr("fusionlab.compute._provider", None)
    monkeypatch.setenv("FUSIONLAB_COMPUTE_PROVIDER", "slurm")
    with pytest.raises(ValueError, match="slurm"):
        get_provider()


# ---------------------------------------------------------------- LocalProvider behavior
def test_capabilities_reports_tasks_and_model_availability():
    caps = LocalProvider().capabilities()
    assert caps["provider"] == "local"
    assert caps["surrogate_available"] == surrogate.available()
    assert caps["eq_surrogate_available"] == eq_surrogate.MODEL_FILE.exists()
    assert set(caps["tasks"]) == {"trace_fieldlines", "run_surrogate", "run_surrogate_shot", "run_eq_surrogate",
                                  "model_metrics", "benchmark", "run_qlknn"}
    assert caps["qlknn_available"] == qlknn.MODEL_FILE.exists()


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown compute task"):
        LocalProvider().submit("rm -rf", {})


def test_result_is_consumed_once():
    provider = LocalProvider()
    job = provider.submit("model_metrics", {"model": "surrogate"}) if surrogate.available() else None
    if job is None:
        pytest.skip("no trained model: run `make train`")
    assert provider.result(job) == json.loads(surrogate.METRICS_FILE.read_text())
    with pytest.raises(KeyError, match="already consumed"):
        provider.result(job)


def test_stream_replays_the_worker_vocabulary_for_a_local_job():
    """HARN-1: a local job's stream is the worker's event sequence (queued → running → terminal), so
    subscribers render local and remote identically."""
    provider = LocalProvider()
    job = provider.submit("model_metrics", {"model": "surrogate"}) if surrogate.available() else None
    if job is None:
        pytest.skip("no trained model: run `make train`")

    async def events():
        return [e async for e in provider.stream(job)]

    assert asyncio.run(events()) == [{"event": "queued", "task": "model_metrics"},
                                     {"event": "running", "task": "model_metrics"},
                                     {"event": "completed", "job_id": job.job_id, "task": "model_metrics"}]
    with pytest.raises(KeyError):
        asyncio.run(_collect(provider.stream(JobHandle(job_id="nope", task="model_metrics", provider="local"))))


def test_benchmark_task_emits_section_progress(monkeypatch):
    """HARN-1: the local benchmark body wires bench.timings' section hook into the job's event list —
    the same progress events the worker's SSE emits between sections."""
    from fusionlab import bench

    def fake_timings(profile: str = "full", progress=None):
        for section in ("replay", "operating_map"):
            if progress is not None:
                progress({"section": section})
        return {"rows": [], "profile": profile}

    monkeypatch.setattr(bench, "timings", fake_timings)
    provider = LocalProvider()
    job = provider.submit("benchmark", {"profile": "quick"})
    assert provider.result(job) == {"rows": [], "profile": "quick"}
    assert asyncio.run(_collect(provider.stream(job))) == [
        {"event": "queued", "task": "benchmark"},
        {"event": "running", "task": "benchmark"},
        {"event": "progress", "section": "replay"},
        {"event": "progress", "section": "operating_map"},
        {"event": "completed", "job_id": job.job_id, "task": "benchmark"},
    ]
    assert provider.active_job("benchmark") is None   # no longer running


def test_stream_ends_quietly_when_a_held_record_is_pruned(monkeypatch):
    """The detached-dict guard: a record pruned while a subscriber polls it — between pulls, no
    terminal event yet — ends the stream after its replayed events instead of spinning forever on a
    detached dict. (A job unknown at stream start still raises KeyError — the worker's own contract.)"""
    from fusionlab.compute import local as local_mod

    monkeypatch.setitem(local_mod.TASKS, "trivial", lambda inputs: 1)
    provider = LocalProvider()
    job = provider.submit("trivial", {})
    provider._jobs[job.job_id]["events"] = provider._jobs[job.job_id]["events"][:1]   # a mid-run record

    async def scenario():
        agen = provider.stream(job)
        first = await agen.__anext__()          # the replayed 'queued' event
        del provider._jobs[job.job_id]          # pruned while this subscriber holds the record
        rest = [first]
        async for event in agen:                # the guard sees the detached record and ends quietly
            rest.append(event)
        return rest

    assert asyncio.run(scenario()) == [{"event": "queued", "task": "trivial"}]


def test_job_records_are_pruned_to_a_bound(monkeypatch):
    """The shared provider instance (run_task's local path) must not accumulate old job records."""
    from fusionlab.compute import local as local_mod

    monkeypatch.setitem(local_mod.TASKS, "trivial", lambda inputs: 1)
    provider = LocalProvider()
    first = provider.submit("trivial", {})
    for _ in range(local_mod.MAX_JOBS_REMEMBERED):
        provider.submit("trivial", {})
    assert first.job_id not in provider._jobs
    assert len(provider._jobs) == local_mod.MAX_JOBS_REMEMBERED


async def _collect(stream):
    return [e async for e in stream]


# ---------------------------------------------------------------- passthrough equivalence
@needs_surrogate
def test_run_surrogate_passthrough_matches_direct_call():
    x = {"Ip_MA": np.array([0.8]), "B_T": np.array([0.45]), "n_e20": np.array([0.4]), "P_loss_MW": np.array([2.0]),
         "R_m": np.array([0.8]), "a_m": np.array([0.55]), "kappa_a": np.array([1.7]), "f_nbi": np.array([0.8])}
    provider = LocalProvider()
    out = provider.result(provider.submit("run_surrogate", {"features": x}))
    assert np.array_equal(out, surrogate.correction(x))


@needs_surrogate
def test_run_surrogate_shot_passthrough_matches_direct_call():
    s = mast.load_shot(30166)
    P_loss = replay(s)["P_loss_MW"]
    provider = LocalProvider()
    out = provider.result(provider.submit("run_surrogate_shot", {"shot": s, "P_loss_MW": P_loss}))
    assert out.shape == s["t_s"].shape
    assert np.array_equal(out, surrogate.correction(surrogate.shot_features(s, P_loss)))


@needs_eq
def test_run_eq_surrogate_passthrough_matches_direct_call():
    s = mast.load_shot(30166)
    provider = LocalProvider()
    out = provider.result(provider.submit("run_eq_surrogate", {"eq_inputs": s["eq_inputs"]}))
    assert out.shape == (s["t_s"].size, eq_surrogate.NZ, eq_surrogate.NR)
    assert np.array_equal(out, eq_surrogate.predict_psi(s["eq_inputs"]))


@needs_surrogate
def test_model_metrics_passthrough_matches_direct_read():
    provider = LocalProvider()
    out = provider.result(provider.submit("model_metrics", {"model": "surrogate"}))
    assert out == json.loads(surrogate.METRICS_FILE.read_text())


def test_trace_fieldlines_passthrough_matches_direct_calls():
    pytest.importorskip("warp")
    from fusionlab import fieldlines

    s = mast.load_shot(30420)   # the same small ohmic shot the field-lines tests use
    provider = LocalProvider()
    out = provider.result(provider.submit("trace_fieldlines", {"shot": s}))
    pts, tr = fieldlines.usd_lines(s)
    qc = fieldlines.q_check(s)

    assert np.array_equal(out["pts"], pts)
    assert np.array_equal(out["q_check"]["q_traced"], qc["q_traced"])
    # launch_s is wall-clock timing (varies run to run); every physics key must match exactly
    task_summary = {k: v for k, v in out["q_check"]["summary"].items() if k != "launch_s"}
    direct_summary = {k: v for k, v in qc["summary"].items() if k != "launch_s"}
    assert task_summary == direct_summary
    assert out["device"] == str(tr.get("device", fieldlines.default_device()))
    assert out["psi_n_start"] == list(fieldlines.USD_PSI_N)


# ---------------------------------------------------------------- the seam's core constraint, pinned statically
_GPU_MODULES = {"torch", "warp", "fieldlines", "surrogate", "eq_surrogate", "qlknn"}


def _enclosing_function(parents: dict, node) -> str | None:
    while node is not None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
        node = parents.get(node)
    return None


def test_routers_reach_gpu_compute_only_through_the_provider():
    """PR 1's core constraint, pinned statically so it cannot regress silently: every function in the two
    router files reaches torch, Warp, or the modules that wrap them only through fusionlab.compute. The one
    exemption is replay_usd, whose Warp-traced export writes a file this process serves back (a remote
    provider would need artifact sync — design §9) and which already falls back to a plain stage."""
    for name in ("api_replay", "api_virtual"):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "fusionlab" / f"{name}.py").read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            offenders += [n for n in names if n.split(".")[-1] in _GPU_MODULES
                          and _enclosing_function(parents, node) != "replay_usd"]
        assert offenders == [], f"{name} imports GPU modules outside the provider seam: {offenders}"


needs_qlknn = pytest.mark.skipif(not qlknn.MODEL_FILE.exists(),
                                 reason="models/qlknn.pt missing: run scripts/convert_qlknn_weights.py")


@needs_qlknn
def test_run_qlknn_passthrough_matches_direct_call():
    """The qlknn rails: the provider passthrough equals the in-process module call, key for key —
    the same equivalence the surrogate paths pin above, now for the vendored-flux closure."""
    s = mast.load_shot(30166)
    p_loss = replay(s)["P_loss_MW"]
    provider = LocalProvider()
    out = provider.result(provider.submit("run_qlknn", {"shot": s, "P_loss_MW": p_loss}))
    direct = qlknn.confinement_coeff(s, p_loss)
    assert set(out) == set(direct)
    assert np.array_equal(out["K"], direct["K"])
    assert np.array_equal(out["alpha"], direct["alpha"])
    assert out["n_unresolved"] == direct["n_unresolved"] and out["n_clipped"] == direct["n_clipped"]
