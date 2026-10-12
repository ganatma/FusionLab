"""The QLKNN7_11 module: vendored-checkpoint fidelity, the laziness the app-import rule depends
on, and the documented power-balance reduction's behaviour.

The physics constants (profile shapes, balance surface, Ti = Te, ...) are ASSUMPTIONS pinned in
fusionlab/qlknn.py's docstring — this suite pins structure and numerical sanity, never levels.
The closure's honesty lives in the re-fly evidence (scripts/validate_virtual.py →
models/virtual_metrics.json), not in a hard-coded accuracy assertion here.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fusionlab import qlknn

FIXTURES = Path(__file__).parent / "fixtures"
needs_qlknn = pytest.mark.skipif(not qlknn.available(),
                                 reason="models/qlknn.pt missing: run scripts/convert_qlknn_weights.py")


def test_module_import_never_loads_torch():
    """The app-import rule in its pristine form: importing fusionlab.qlknn must not pull torch.
    The api process imports fusionlab.virtual, whose qlknn branch imports this module lazily."""
    out = subprocess.run(
        [sys.executable, "-c", "import sys, fusionlab.qlknn; print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@needs_qlknn
def test_reference_pairs_match_upstream():
    """The 25 upstream reference pairs, recomputed through the vendored checkpoint: the conversion
    is faithful to the upstream implementation (the JAX nets), not approximately."""
    with np.load(FIXTURES / "qlknn_7_11_test_data.npz") as td:
        x, expected = td["inputs"], td["outputs"]
    assert float(np.max(np.abs(qlknn.predict_targets(x) - expected))) < 1e-6


@needs_qlknn
def test_checkpoint_loads_weights_only_with_consistent_stats():
    m = qlknn.load()
    n_in, n_out = len(m.input_names), len(m.target_names)
    assert m.i_mean.shape == (n_in,) and m.i_sd.shape == (n_in,)
    assert m.t_mean.shape == (n_out,) and m.t_sd.shape == (n_out,)
    for arr in (m.i_mean, m.i_sd, m.t_mean, m.t_sd):
        assert np.all(np.isfinite(arr))


@needs_qlknn
def test_confinement_coeff_resolves_the_raw_shot():
    """Shot 30166's raw bundle: a per-slice (K, alpha) solve with no unresolved slice and alpha in
    the open interval (0, 1) the eliminated power-balance ODE needs."""
    from fusionlab import mast
    from fusionlab.physics import replay

    shot = mast.load_shot(30166)
    out = qlknn.confinement_coeff(shot, replay(shot)["P_loss_MW"])
    assert out["K"].shape == shot["t_s"].shape and out["alpha"].shape == shot["t_s"].shape
    assert out["n_unresolved"] == 0
    alpha = np.asarray(out["alpha"], float)
    assert np.nanmin(alpha) > 0.0 and np.nanmax(alpha) < 1.0
    assert np.nanmedian(np.asarray(out["K"], float)) > 0.0


@needs_qlknn
def test_power_balance_tau_decreases_with_loss_power():
    """tau = K · P^-alpha with alpha > 0: doubling the loss power shortens the confinement time —
    the sign the whole reduction exists to preserve."""
    from fusionlab import mast
    from fusionlab.physics import replay

    shot = mast.load_shot(30166)
    p_loss = replay(shot)["P_loss_MW"]
    lo = qlknn.confinement_coeff(shot, p_loss)
    hi = qlknn.confinement_coeff(shot, 2.0 * p_loss)
    tau_lo = np.asarray(lo["K"], float) * p_loss ** (-np.asarray(lo["alpha"], float))
    tau_hi = np.asarray(hi["K"], float) * (2.0 * p_loss) ** (-np.asarray(hi["alpha"], float))
    assert np.nanmedian(tau_hi) < np.nanmedian(tau_lo)


@needs_qlknn
def test_available_tracks_the_weights(monkeypatch, tmp_path):
    assert qlknn.available() is True
    monkeypatch.setattr(qlknn, "MODEL_FILE", tmp_path / "absent.pt")
    assert qlknn.available() is False


def test_agent_schema_picks_up_the_closure():
    """No hardcoded UI list, agent edition: run_whatif's closure enum is built from
    virtual.CLOSURES at schema-build time, so the new closure is offered without edits."""
    from fusionlab import agent_tools, virtual

    schema = agent_tools.TOOLS["run_whatif"].definition["input_schema"]
    assert schema["properties"]["closure"]["enum"] == list(virtual.CLOSURES)


def test_virtual_availability_excludes_the_closure_without_weights(monkeypatch):
    """A checkout without models/qlknn.pt never advertises or accepts the closure — the gate
    filters, the route 422s, and the default closure still answers (verified at the route level
    in test_api.py)."""
    from fusionlab import virtual

    assert "qlknn" in virtual.available_closures()
    monkeypatch.setattr(qlknn, "MODEL_FILE", Path("/tmp/absent.pt"))
    assert "qlknn" not in virtual.available_closures()
    assert virtual.available_closures() == ("learned", "ipb98", "iter89")
