"""The scripts' cache writes (HARN-3): an interrupted write can never leave a corrupt cache.

pytest's pythonpath covers only the repo root, so the scripts are imported the same way
tests/test_atomicio.py imports eq_forward_jacobian: scripts/ prepended to sys.path.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    return __import__(name)


def test_interrupted_refly_write_never_caches_a_corrupt_npz(monkeypatch, tmp_path):
    """validate_virtual wrote its re-fly cache straight to the final path: a crash left a corrupt
    npz that every later run treated as cached forever. Through atomicio the temp dies with the
    write and the next fetch starts clean."""
    vv = _script("validate_virtual")
    monkeypatch.setattr(vv, "REFLY", tmp_path)
    bundle = {"meta": {"P_nbi_max_MW": 0.0}, "t_s": np.arange(3.0),
              **{k: np.arange(3.0) for k in vv.KEEP if k != "t_s"}}
    monkeypatch.setattr(vv.mast, "fetch_shot", lambda shot_id: bundle)
    real_savez = np.savez_compressed

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(np, "savez_compressed", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        vv.fetch(31415)
    assert not (tmp_path / "31415.npz").exists()   # no corrupt file cached for eternity
    assert not list(tmp_path.glob(".*tmp*"))       # and no temp litter either

    monkeypatch.setattr(np, "savez_compressed", real_savez)   # the next fetch writes cleanly
    assert vv.fetch(31415).endswith("slices")
    with np.load(tmp_path / "31415.npz") as z:
        assert z["W_MJ"].shape == (3,) and str(z["meta"]) == vv.json.dumps(bundle["meta"])
