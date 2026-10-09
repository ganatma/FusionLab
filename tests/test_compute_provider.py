"""ComputeProvider seam: selection defaults to LocalProvider, config overrides parse, and LocalProvider's
tasks return exactly what the wrapped functions return (offline, CPU)."""

import asyncio
import json

import numpy as np
import pytest

from fusionlab import eq_surrogate, mast, surrogate
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
                                  "model_metrics"}


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


def test_stream_yields_one_completed_event_for_a_local_job():
    provider = LocalProvider()
    job = provider.submit("model_metrics", {"model": "surrogate"}) if surrogate.available() else None
    if job is None:
        pytest.skip("no trained model: run `make train`")

    async def events():
        return [e async for e in provider.stream(job)]

    assert asyncio.run(events()) == [{"event": "completed", "job_id": job.job_id, "task": "model_metrics"}]
    with pytest.raises(KeyError):
        asyncio.run(_collect(provider.stream(JobHandle(job_id="nope", task="model_metrics", provider="local"))))


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
