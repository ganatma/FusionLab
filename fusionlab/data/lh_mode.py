"""Derived L/H-mode flag for the catalog: transparent, labelled derived — never measured.

The FAIR-MAST shot table has no confinement-mode field (the ndjson columns are logbook
text only), and mode is the first filter every MAST analysis applies (#38). The twin's own
scaling laws give the transparent route: a row is in the H regime when its measured energy
confinement time beats the machine's L-mode law (ITER89-P, `physics.tau_coeff_L`) by more
than the machine-configured factor, at the catalog's peak-current snapshot:

    mode_derived = 1  if tau_E_s > lh_enhancement_min * tau_89
                 = 0  if tau_E_s <= lh_enhancement_min * tau_89
                 NaN  when the scalars the ratio needs are missing — the row is unknown
                      and matches neither the H nor the L filter

Why not the Martin L-H power threshold (`physics.p_lh`)? On MAST it under-predicts the
threshold by 7-9x across the catalog — every heating group, ohmic included, sits far above
it (docs/PHYSICS.md, "known limits") — so a power-threshold flag would mark nearly every
row H. The enhancement rule instead asks the twin's own L-mode law, which the archive's
tau_E is directly comparable to (both are W / (P_ohm + P_nbi - dW/dt), radiation not
subtracted — the `clean_db` / `replay` convention).

Known failure modes (quantified by scripts/mode_flag_report.py, listed in docs/PHYSICS.md):
the flag describes the peak-current snapshot, not the discharge — L-H transitions mid-shot
are invisible to scalars; ELMy and locked-mode phases depress tau_E and can flip H to L;
roughly half the catalog lacks the cpf scalars and stays unknown; the cut is calibrated on
two hand-verified anchors (30420 ohmic -> L, 30166 "H-mode from 223 ms" with peak current
at 0.245 s -> H) and corroborated by the heating-label gradient — not validated against a
labelled mode database, because none exists. It is a filter aid with documented limits,
never a mode measurement; the replay model's per-slice hmode choice (raw Martin comparison)
is a different quantity and may disagree.
"""

from __future__ import annotations

import numpy as np

from fusionlab.physics import tau_coeff_L, volume


def enhancement_ratio(db: dict, *, isotope_md: float) -> np.ndarray:
    """Measured tau_E over the ITER89-P L-mode prediction, per catalog row (NaN where not computable).

    Geometry uses the IPB98 volume-derived elongation (V / (2 pi^2 R a^2)), the same
    convention as `_db_view` and `physics.replay`; stand-in values keep the vectorized
    scaling warning-free on rows that will be NaN'd anyway.
    """
    R, a, V = (np.asarray(db[k], dtype=float) for k in ("R_m", "a_m", "V_m3"))
    tau_E, Ip, B, n = (np.asarray(db[k], dtype=float) for k in ("tau_E_s", "Ip_MA", "B_T", "n_e20"))
    P_in = np.asarray(db["P_ohm_MW"], dtype=float) + np.asarray(db["P_nbi_MW"], dtype=float)
    P_loss = np.maximum(P_in - np.asarray(db["dWdt_MW"], dtype=float), 0.2 * P_in)
    ok = (np.isfinite(R) & np.isfinite(a) & np.isfinite(V) & np.isfinite(tau_E)
          & np.isfinite(Ip) & np.isfinite(B) & np.isfinite(n) & np.isfinite(P_loss) & (P_loss > 0))
    # The scalings take fractional powers: non-positive garbage scalars (old-campaign rows
    # carry some) are unknown, not warned into NaN.
    ok &= (tau_E > 0) & (Ip > 0) & (B > 0) & (n > 0) & (R > 0) & (a > 0) & (V > 0)
    R_, a_ = np.where(ok, R, 1.0), np.where(ok, a, 1.0)
    g = {"R": R_, "a": a_, "kappa": np.where(ok, V / volume(R_, a_, 1.0), 1.0)}
    C, alpha = tau_coeff_L(g, np.where(ok, Ip, 1.0), np.where(ok, B, 1.0), np.where(ok, n, 1.0), 1.0, isotope_md)
    tau_89 = C * np.where(ok, P_loss, 1.0) ** -alpha
    return np.where(ok, tau_E / tau_89, np.nan)


def derive_mode(db: dict, *, isotope_md: float, enhancement_min: float) -> np.ndarray:
    """The catalog's `mode_derived` column: 1 = H regime, 0 = L regime, NaN = unknown."""
    ratio = enhancement_ratio(db, isotope_md=isotope_md)
    return np.where(np.isfinite(ratio), (ratio > enhancement_min).astype(float), np.nan)
