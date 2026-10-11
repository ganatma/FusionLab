"""The derived L/H-mode flag (FEAT-4, #38): rule, unknowns, and the storage paths.

The rule is one comparison — tau_E against the twin's own ITER89-P L-mode law — so the tests
pin exactly that (shared source, no copied constants): synthetic rows above/below the cut, the
unknown contract (missing or non-positive scalars match neither filter), and the two storage
paths (derived during ingest for new caches; derived on read for caches predating #38, without
writing back to the shipped cache).
"""

import hashlib
import json

import numpy as np
import pytest

from fusionlab import mast
from fusionlab.data import lh_mode, reader
from fusionlab.data.ingestors import fair_mast
from fusionlab.physics import tau_coeff_L, volume

# Columns derive_mode reads, at MAST-ish magnitudes (one finite, physical row).
_ROW = {"tau_E_s": 0.05, "Ip_MA": 0.8, "B_T": 0.5, "n_e20": 0.4,
        "P_ohm_MW": 0.3, "P_nbi_MW": 2.0, "dWdt_MW": 0.0, "R_m": 0.85, "a_m": 0.6, "V_m3": 3.0}


def _row(**over):
    return {**_ROW, **over}


def _db(rows: list[dict]) -> dict:
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


# ---- the rule


def test_rows_above_the_cut_flag_h_and_the_rest_l():
    flag = lh_mode.derive_mode(_db([_row(tau_E_s=0.08), _row(tau_E_s=0.02), _row(tau_E_s=0.05)]),
                               isotope_md=2.0, enhancement_min=1.5)
    assert flag[0] == 1.0            # 0.08 s beats the L-mode law by far: H regime
    assert flag[1] == 0.0            # 0.02 s sits below it: L regime
    assert flag[2] in (0.0, 1.0)     # the middle row classifies without ambiguity either way


def test_the_cut_is_the_configured_enhancement_ratio():
    """The boundary sits exactly at the cut: tau_E == enhancement_min * tau_89 is L, just above is H."""
    C, alpha = tau_coeff_L({"R": np.array([0.85]), "a": np.array([0.6]),
                            "kappa": np.array([3.0 / volume(0.85, 0.6, 1.0)])},
                           np.array([0.8]), np.array([0.5]), np.array([0.4]), 1.0, 2.0)
    tau_89 = float(C[0]) * 2.3 ** -float(alpha)   # P_loss = 0.3 + 2.0 - 0.0 = 2.3 MW; alpha is a scalar
    at_cut = lh_mode.derive_mode(_db([_row(tau_E_s=1.5 * tau_89)]), isotope_md=2.0, enhancement_min=1.5)
    just_above = lh_mode.derive_mode(_db([_row(tau_E_s=1.5001 * tau_89)]), isotope_md=2.0, enhancement_min=1.5)
    assert at_cut[0] == 0.0 and just_above[0] == 1.0


def test_enhancement_ratio_matches_the_shared_law():
    """The ratio is tau_E / (ITER89-P with the volume-derived elongation) — physics.replay's convention."""
    db = _db([_row()])
    ratio = lh_mode.enhancement_ratio(db, isotope_md=2.0)
    g = {"R": db["R_m"], "a": db["a_m"], "kappa": db["V_m3"] / volume(db["R_m"], db["a_m"], 1.0)}
    C, alpha = tau_coeff_L(g, db["Ip_MA"], db["B_T"], db["n_e20"], 1.0, 2.0)
    tau_89 = C * (0.3 + 2.0) ** -alpha
    assert ratio[0] == pytest.approx(0.05 / tau_89[0])


# ---- the unknown contract


def test_incomplete_or_garbage_scalars_are_unknown_not_l():
    rows = [_row(),                                  # the finite reference
            _row(tau_E_s=np.nan),                    # the archive's most common gap (cpf_tautot)
            _row(R_m=np.nan), _row(a_m=np.nan), _row(V_m3=np.nan),
            _row(Ip_MA=np.nan), _row(B_T=np.nan), _row(n_e20=np.nan),
            _row(P_ohm_MW=np.nan), _row(P_nbi_MW=np.nan), _row(dWdt_MW=np.nan),
            _row(P_ohm_MW=0.0, P_nbi_MW=0.0),        # no heating: P_loss would be 0 (invalid)
            _row(Ip_MA=-0.8),                        # non-positive garbage: the scalings take fractional powers
            _row(tau_E_s=-0.05), _row(n_e20=-0.4)]
    flag = lh_mode.derive_mode(_db(rows), isotope_md=2.0, enhancement_min=1.5)
    assert flag[0] in (0.0, 1.0)
    assert np.all(np.isnan(flag[1:]))


def test_unknown_rows_match_neither_filter():
    """The route's contract: mode=H and mode=L partition the *flaggable* rows; unknown rows match neither."""
    flag = lh_mode.derive_mode(_db([_row(tau_E_s=0.08), _row(tau_E_s=np.nan)]),
                               isotope_md=2.0, enhancement_min=1.5)
    assert np.isnan(flag[1]) and flag[1] != 1 and flag[1] != 0   # NaN compares false both ways


# ---- the shipped catalog: the rule reproduces the two hand-verified anchors (docs/PHYSICS.md)


def test_hand_verified_anchors_reproduce():
    """30420 is ohmic (L-mode edge) and 30166's logbook says "H-mode from 223 ms" with peak current at
    0.245 s — the catalog's two mode-certain rows. The shipped rule must classify them so, with the
    configured cut genuinely between their ratios (the calibration sandwich, recomputed not copied)."""
    db = mast.load_db()
    ids, flag = db["shot_id"], db["mode_derived"]
    i_l, i_h = int(np.flatnonzero(ids == 30420)[0]), int(np.flatnonzero(ids == 30166)[0])
    assert flag[i_l] == 0.0 and flag[i_h] == 1.0
    ratio = lh_mode.enhancement_ratio(db, isotope_md=mast.CONFIG.isotope_md)
    assert ratio[i_l] < mast.CONFIG.lh_enhancement_min < ratio[i_h]


# ---- storage path 1: derived during catalog build (the ingestor)


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._payload


def _ndjson_row(**over):
    row = {"shot_id": 30420, "campaign": "M9",
           "cpf_ip_max": 800.0, "cpf_bt_ipmax": 0.5, "cpf_ne_bar_ipmax": 4e19,
           "cpf_pohm_ipmax": 0.3, "cpf_pnbi_ipmax": 2.0, "cpf_dwmhd_ipmax": 0.0,
           "cpf_tautot_ipmax": 0.08, "cpf_rgeo_ipmax": 0.85, "cpf_amin_ipmax": 0.6, "cpf_vol_ipmax": 3.0}
    row.update(over)
    return json.dumps(row).encode()


def test_ingestor_derives_the_flag_at_catalog_build(monkeypatch):
    """fetch_db stores mode_derived with the column, so a rebuilt cache carries it on disk."""
    payload = b"\n".join([_ndjson_row(shot_id=30166),                          # clearly above the L-law
                          _ndjson_row(shot_id=30420, cpf_tautot_ipmax=None),   # no tau_E: unknown
                          _ndjson_row(shot_id=29823, cpf_ip_max=-800.0)])      # garbage: unknown
    monkeypatch.setattr(fair_mast.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))
    db = fair_mast.fetch_db()
    expected = lh_mode.derive_mode(db, isotope_md=fair_mast.MAST_CONFIG.isotope_md,
                                   enhancement_min=fair_mast.MAST_CONFIG.lh_enhancement_min)
    np.testing.assert_array_equal(db["mode_derived"], expected)   # NaN (unknown) == NaN: numpy treats them equal
    assert db["mode_derived"][0] == 1.0 and np.all(np.isnan(db["mode_derived"][1:]))


# ---- storage path 2: derived on read for caches predating #38


def _save_catalog(path, rows: list[dict], extra: dict | None = None) -> None:
    cols = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    cols.update(extra or {})
    np.savez(path, **cols)


@pytest.fixture
def old_cache(tmp_path, monkeypatch):
    """A catalog npz in the pre-#38 shape (no mode_derived column), wired in as the facade's DB_FILE."""
    path = tmp_path / "mast_db.npz"
    _save_catalog(path, [_row(), _row(tau_E_s=np.nan)])
    monkeypatch.setattr(mast, "DB_FILE", path)
    return path


def test_facade_backfills_the_flag_on_read(old_cache):
    db = mast.load_db()
    assert "mode_derived" in db
    expected = lh_mode.derive_mode(reader.load_db(False, db_file=old_cache, fetch=mast.fetch_db),
                                   isotope_md=mast.CONFIG.isotope_md,
                                   enhancement_min=mast.CONFIG.lh_enhancement_min)
    np.testing.assert_array_equal(db["mode_derived"], expected)


def test_facade_never_rewrites_an_old_cache(old_cache):
    """The shipped cache stays byte-identical (the derived column lives in memory; the next rebuild stores it)."""
    before = hashlib.sha256(old_cache.read_bytes()).hexdigest()
    mast.load_db()
    assert hashlib.sha256(old_cache.read_bytes()).hexdigest() == before


def test_facade_passes_a_stored_flag_through_untouched(tmp_path, monkeypatch):
    """A cache that already carries mode_derived is served as stored — no recompute, no overwrite."""
    path = tmp_path / "mast_db.npz"
    _save_catalog(path, [_row(), _row()], extra={"mode_derived": np.array([7.0, 7.0])})   # not a real flag value
    monkeypatch.setattr(mast, "DB_FILE", path)
    assert mast.load_db()["mode_derived"].tolist() == [7.0, 7.0]
