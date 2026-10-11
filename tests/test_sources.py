"""Tests for the DataSource seam (DATA-2): one door, validated bundles out, whichever source serves.

The synthetic bundle here is deliberately NOT MAST: a different id shape (letters), a ~3 s time
base (MAST shots run ~0.5 s), reactor-adjacent geometry (R0 1.65 m, B 2.4 T). Nothing MAST-scale
about it — if the seam leaks MAST assumptions, these are the first tests to catch it.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from fusionlab import mast
from fusionlab.data import sources
from fusionlab.data.bundle import BundleValidationError
from fusionlab.data.machine import MAST_CONFIG


def synthetic_bundle(shot_id: str = "AUG-34578", nt: int = 40) -> dict:
    """A valid non-MAST bundle dict — the worked example of docs/DATA_CONTRACT.md."""
    t = np.linspace(3.0, 3.4, nt)
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r0, a, kap = 1.65, 0.5, 1.7
    s: dict = {
        "t_s": t,
        "Ip_MA": 0.8 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, nt)),
        "B_T": np.full(nt, 2.4),
        "n_e20": np.full(nt, 0.6),
        "P_ohm_MW": np.full(nt, 0.5),
        "P_nbi_MW": np.full(nt, 5.0),
        "W_MJ": np.full(nt, 0.35),
        "dWdt_MW": np.zeros(nt),
        "q95": np.full(nt, 4.2),
        "beta_N": np.full(nt, 1.9),
        "kappa": np.full(nt, kap),
        "a_m": np.full(nt, a),
        "R_m": np.full(nt, r0),
        "V_m3": np.full(nt, 13.0),
        "lcfs_R": np.broadcast_to(r0 + a * np.cos(theta), (nt, 24)),
        "lcfs_Z": np.broadcast_to(a * kap * np.sin(theta), (nt, 24)),
        "psi_R": np.linspace(r0 - a - 0.05, r0 + a + 0.05, 16),
        "psi_Z": np.linspace(-a * kap - 0.05, a * kap + 0.05, 20),
        "R_mag_m": np.full(nt, r0),
        "Z_mag_m": np.zeros(nt),
        "wall_R": r0 + (a + 0.08) * np.cos(theta),
        "wall_Z": (a + 0.08) * kap * np.sin(theta),
        "meta": {"shot_id": shot_id, "campaign": "AUG-2026", "heating": "NBI",
                 "attribution": "User data: synthetic AUG-like discharge (test fixture)"},
    }
    r2, z2 = np.meshgrid(s["psi_R"], s["psi_Z"])       # rows = Z
    psi = ((r2 - r0) ** 2 + (z2 / kap) ** 2) / a**2    # 0 at the axis, 1 at the boundary
    s["psi_n"] = np.broadcast_to(psi[None], (nt, 20, 16)).astype(np.float32)
    return s


def npz_bytes(s: dict) -> bytes:
    """The bundle dict as an npz file's bytes, exactly as a physicist would prepare it."""
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=json.dumps(s["meta"]),
                        **{k: v for k, v in s.items() if k != "meta"})
    return buf.getvalue()


# ---------------------------------------------------------------- the factory

def test_get_source_defaults_to_the_fair_mast_facade(monkeypatch):
    monkeypatch.delenv("FUSIONLAB_DATA_SOURCE", raising=False)
    src = sources.get_source()
    assert src.name == "fair-mast" and src.config is MAST_CONFIG


def test_get_source_resolves_the_configured_default(monkeypatch):
    monkeypatch.setenv("FUSIONLAB_DATA_SOURCE", "user-file")
    assert sources.get_source().name == "user-file"


def test_get_source_fails_loudly_on_an_unknown_name(monkeypatch):
    monkeypatch.delenv("FUSIONLAB_DATA_SOURCE", raising=False)
    with pytest.raises(ValueError, match="unknown data source"):
        sources.get_source("not-a-source")


def test_fair_mast_source_delegates_to_the_facade(monkeypatch):
    """The facade stays the monkeypatch surface the tests own — the source delegates, never re-fetches."""
    monkeypatch.setattr(mast, "cached_shots", lambda: [30166])
    seen: list[int] = []

    def fake_load(shot_id: int) -> dict:
        seen.append(shot_id)
        return {"t_s": np.array([0.1]), "meta": {"shot_id": shot_id}}

    monkeypatch.setattr(mast, "load_shot", fake_load)
    src = sources.FairMastSource()
    assert src.cached_shots() == ["30166"]
    s = src.load_shot("30166")
    assert seen == [30166] and s["meta"]["shot_id"] == 30166


# ---------------------------------------------------------------- the bundle-id rule

@pytest.mark.parametrize("ok", ["30166", "AUG-34578", "shot_1", "a.b-c_d"])
def test_valid_bundle_id_accepts_the_documented_shapes(ok):
    assert sources.valid_bundle_id(ok)


@pytest.mark.parametrize("bad", ["", ".hidden", "-lead", "../evil", "a/b", "a b", "x" * 65])
def test_valid_bundle_id_rejects_traversal_and_junk(bad):
    """No traversal, no dotfiles, no separators: the id names a file inside the bundles dir and a URL segment."""
    assert not sources.valid_bundle_id(bad)


# ---------------------------------------------------------------- the user-file ingestor

def write_machine_config(tmp_path, **fields):
    (tmp_path / "machine.json").write_text(json.dumps(fields), encoding="utf-8")


def test_import_publishes_and_serves_a_valid_bundle(tmp_path):
    src = sources.UserFileSource(bundles_dir=tmp_path)
    s = src.import_bundle("AUG-34578", npz_bytes(synthetic_bundle()))
    assert s["meta"]["shot_id"] == "AUG-34578"      # identity stamped from the file name
    assert src.cached_shots() == ["AUG-34578"]
    assert src.config.usd_prim_root == "/USER" and src.config.device == ""   # the user defaults
    assert src.load_shot("AUG-34578")["meta"]["shot_id"] == "AUG-34578"


def test_bundle_without_meta_gets_defaults_and_a_stamped_id(tmp_path):
    src = sources.UserFileSource(bundles_dir=tmp_path)
    s = synthetic_bundle()
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v for k, v in s.items() if k != "meta"})
    out = src.import_bundle("AUG-1", buf.getvalue())
    assert out["meta"] == {"shot_id": "AUG-1"}


def test_import_rejects_a_bad_bundle_naming_the_field_without_a_partial_file(tmp_path):
    """The acceptance case: a field-level 400-class error AND no partial write (validate-then-publish)."""
    src = sources.UserFileSource(bundles_dir=tmp_path)
    broken = npz_bytes(synthetic_bundle()).replace(b"psi_n", b"psi_X")   # rename a contract field
    with pytest.raises(BundleValidationError, match=r"psi_n: incomplete group"):
        src.import_bundle("AUG-34578", broken)
    assert src.cached_shots() == []                 # nothing published
    assert not list(tmp_path.glob("*.npz"))         # no partial file, not even the target name


def test_import_rejects_an_unreadable_archive_with_an_actionable_message(tmp_path):
    src = sources.UserFileSource(bundles_dir=tmp_path)
    with pytest.raises(BundleValidationError, match="not a readable npz shot bundle"):
        src.import_bundle("junk", b"not an npz at all")


def test_import_rejects_bad_ids_and_oversized_uploads(tmp_path):
    src = sources.UserFileSource(bundles_dir=tmp_path)
    with pytest.raises(ValueError, match="not importable"):
        src.import_bundle("../evil", npz_bytes(synthetic_bundle()))
    with pytest.raises(ValueError, match="import cap"):
        src.import_bundle("big", b"x" * (sources.MAX_BUNDLE_BYTES + 1))
    assert src.cached_shots() == []


def test_machine_json_overrides_the_user_config_and_fails_loudly_on_unknown_fields(tmp_path):
    write_machine_config(tmp_path, usd_prim_root="/AUG", attribution="User data: AUG discharge")
    src = sources.UserFileSource(bundles_dir=tmp_path)
    assert src.config.usd_prim_root == "/AUG" and src.config.attribution == "User data: AUG discharge"
    write_machine_config(tmp_path, not_a_field=1)
    with pytest.raises(ValueError, match="not_a_field"):
        sources.UserFileSource(bundles_dir=tmp_path)


def test_user_file_derived_catalog_and_missing_bundle_error(tmp_path):
    src = sources.UserFileSource(bundles_dir=tmp_path)
    src.import_bundle("AUG-34578", npz_bytes(synthetic_bundle()))
    db = src.load_db()
    assert list(db["shot_id"]) == ["AUG-34578"]
    s = synthetic_bundle()
    i = int(np.nanargmax(np.abs(s["Ip_MA"])))       # rows are evaluated at the peak-current slice
    expected_tau = float(s["W_MJ"][i]) / float(s["P_ohm_MW"][i] + s["P_nbi_MW"][i] - s["dWdt_MW"][i])
    assert np.isclose(float(db["tau_E_s"][0]), expected_tau)
    assert int(db["useful"][0]) == 1 and int(db["abort"][0]) == 0
    with pytest.raises(FileNotFoundError, match="no imported bundle"):
        src.load_shot("DIII-D-1")


def test_user_bundles_env_points_the_source_at_a_directory(tmp_path, monkeypatch):
    (tmp_path / "X-1.npz").write_bytes(npz_bytes(synthetic_bundle("X-1")))
    monkeypatch.setenv("FUSIONLAB_USER_BUNDLES", str(tmp_path))
    monkeypatch.setenv("FUSIONLAB_DATA_SOURCE", "user-file")
    assert sources.get_source().cached_shots() == ["X-1"]
