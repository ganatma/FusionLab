"""EFIT surrogate: offline, fast. Skips when the model was never trained (models/*.pt is git-ignored)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
eqs = pytest.importorskip("fusionlab.eq_surrogate")

if not eqs.MODEL_FILE.exists():
    pytest.skip("models/eq_surrogate.pt missing: run scripts/train_eq_surrogate.py", allow_module_level=True)


def test_predict_psi_shape_and_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, eqs.N_RAW)).astype(np.float32)
    x[0, :5] = np.nan                                     # dead sensors must be imputed, not propagated
    x[1, :] = 0.0
    psi = eqs.predict_psi(x, device="cpu")
    assert psi.shape == (7, eqs.NZ, eqs.NR)
    assert np.isfinite(psi).all()


def test_single_slice_and_bad_width():
    assert eqs.predict_psi(np.ones(eqs.N_RAW), device="cpu").shape == (1, eqs.NZ, eqs.NR)
    with pytest.raises(ValueError):
        eqs.predict_psi(np.ones((2, 10)), device="cpu")


def test_checkpoint_matches_metrics_file():
    model = eqs.load("cpu")
    assert len(model.input_names) == model.cfg["in_features"]
    assert model.R_m.shape == (eqs.NR,) and model.Z_m.shape == (eqs.NZ,)
    if eqs.METRICS_FILE.exists():                         # the honesty flag must be present, whichever way it came out
        m = json.loads(eqs.METRICS_FILE.read_text())
        assert isinstance(m["beats_linear"], bool)
        assert m["input_names"] == model.input_names


def test_axis_finder_on_a_known_peak():
    R, Z = torch.linspace(0.06, 1.98, eqs.NR), torch.linspace(-2, 2, eqs.NZ)
    psi = -((R[None, :] - 0.91) ** 2 + 0.5 * (Z[:, None] - 0.07) ** 2)[None]          # paraboloid, maximum at (0.91, 0.07)
    r, z = eqs.magnetic_axis(psi, torch.ones(1), R, Z)
    assert abs(float(r) - 0.91) < 1e-3 and abs(float(z) - 0.07) < 1e-3
    r, z = eqs.magnetic_axis(-psi, -torch.ones(1), R, Z)                               # reversed current: a minimum
    assert abs(float(r) - 0.91) < 1e-3 and abs(float(z) - 0.07) < 1e-3


def test_forward_spike_metrics_record_the_gate_decision_the_readme_reports():
    """The coils-only spike (scripts/train_eq_surrogate.py --inputs coils, scripts/eq_forward_jacobian.py) ships nothing;
    its numbers and its go / no-go are on disk, and the README says what the file says."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    f = root / "models" / "eq_forward_spike_metrics.json"
    if not f.exists():
        pytest.skip("forward spike not run in this checkout")
    m = json.loads(f.read_text())
    assert m["test_metrics"]["physicsnemo"]["rel_l2"]["median"] > m["test_metrics"]["pca_truncation_floor_k64"]["rel_l2"]["median"]
    j = m["jacobian_check"]
    assert set(j["main_circuits"]) <= set(j["circuits"]) and isinstance(j["verdict"]["go"], bool)
    for name in j["main_circuits"]:
        c = j["circuits"][name]
        assert -1 <= c["r_all"]["median"] <= 1 and c["n"] > 20 and c["sign"] in ("same", "opposite")
    assert j["verdict"]["go"] == (j["verdict"]["consistent_sign_on_main_circuits"] and all(j["circuits"][n]["passes"] for n in j["main_circuits"]))
    assert j["verdict"]["main_circuits_passing"] == [n for n in j["main_circuits"] if j["circuits"][n]["passes"]]
    docs = "".join(p.read_text() for p in (root / "README.md", root / "docs" / "RESULTS.md") if p.exists())
    assert ("a spike that did not ship" in docs) == (not j["verdict"]["go"])
