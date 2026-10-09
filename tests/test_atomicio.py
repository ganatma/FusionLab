"""Helper-level tests for fusionlab.atomicio — written once for the whole atomic-write class."""

import json

import pytest

from fusionlab.atomicio import atomic_write


def test_writes_and_publishes_the_artifact(tmp_path):
    f = tmp_path / "artifact.txt"
    atomic_write(f, lambda tmp: tmp.write_text("new"))
    assert f.read_text() == "new"
    assert not list(tmp_path.glob(".*tmp*"))


def test_partial_write_leaves_the_prior_artifact_byte_identical(tmp_path):
    """A writer that dies mid-write (simulated kill) must not damage the previous artifact."""
    f = tmp_path / "artifact.json"
    good = json.dumps({"rmse": 0.1})
    f.write_text(good)

    def killed_mid_write(tmp):
        tmp.write_text('{"rmse": 0.')            # partial bytes reach the temp file, then the process dies
        raise RuntimeError("simulated kill")

    with pytest.raises(RuntimeError, match="simulated kill"):
        atomic_write(f, killed_mid_write)
    assert f.read_bytes() == good.encode()                     # byte-identical
    assert json.loads(f.read_text()) == {"rmse": 0.1}          # still loadable
    assert not list(tmp_path.glob(".*tmp*"))                   # no .tmp* litter


def test_temp_file_is_invisible_to_suffix_globs(tmp_path):
    """cached_shots() globs *.npz and int()s the stems: the dot-prefixed temp name must never match."""
    f = tmp_path / "30166.npz"
    seen = {}

    def write(tmp):
        seen["glob"] = [p.name for p in tmp_path.glob("*.npz")]   # what a concurrent reader lists right now
        seen["tmp_name"] = tmp.name
        tmp.write_text("x")

    atomic_write(f, write)
    assert seen["glob"] == []                                   # a *.npz glob mid-write never sees the temp file
    assert seen["tmp_name"].startswith(".") and not seen["tmp_name"].endswith(".npz")
    assert f.read_text() == "x"
