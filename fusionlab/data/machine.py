"""Per-machine configuration: everything MAST-specific that is not data.

MachineConfig carries the values that were previously hard-coded at a dozen sites —
the physics isotope/alpha flags (physics.py), the catalog-filter thresholds
(mast.clean_db), the attribution and USD naming (api_replay, usd_export, fieldlines),
the campaign naming convention (the FAIR-MAST ingestor), the warm-up shot
(api_replay._warm_up) and the unit-sanity ranges the tests assert against — so a second
machine is a new config, not a new fork of the code. It is a plain frozen dataclass with
one shipped instance (MAST_CONFIG): the values are few, stable and reviewed in code;
a config-file loader can be added when a second machine actually lands (#34).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CleanLimits:
    """Catalog-filter thresholds, calibrated per machine (MAST values were hard-coded in
    `mast.clean_db`): a row is kept only if it passes all three."""

    tau_E_max_s: float
    kappa_min: float
    a_min_m: float


@dataclass(frozen=True)
class MachineConfig:
    """Machine-identity values for one data source (bundles record their source's name)."""

    device: str                    # key into physics.DEVICES presets
    isotope_md: float              # isotope mass for the tau_coeff_* scalings (2.0 = deuterium)
    alpha_particles: bool          # False for MAST: it ran deuterium, so replay omits alpha heating
    attribution: str               # provenance string carried on replay/USD payloads
    campaign_pattern: str          # regex the archive's campaign labels follow ("M9")
    usd_prim_root: str             # stage root prim for USD exports ("/MAST/...")
    reconstruction_name: str       # what the archive calls its equilibrium reconstructions ("EFIT")
    te_display_range: tuple[float, float]  # 3D plasma (vessel3d) Te colour range [keV]
    te_usd_ramp_max_kev: float    # USD export Te->colour ramp ceiling [keV] (was usd_export.TE_RAMP_MAX_KEV)
    warmup_shot: str | None        # shot loaded at server start to warm the caches
    clean_limits: CleanLimits      # catalog-filter thresholds
    lh_enhancement_min: float      # mode_derived cut: H above this tau_E / tau_89 (ITER89-P) ratio
                                   # (fusionlab.data.lh_mode; MAST value calibrated on the #38 anchors)
    bench_shot: str | None = None  # the shot the benchmark task replays (None: the caller names one)
    unit_sanity: tuple[tuple[str, float, float], ...] = ()  # (field, lo, hi) mid-shot unit guards


MAST_CONFIG = MachineConfig(
    device="mast",
    isotope_md=2.0,
    alpha_particles=False,
    attribution="MAST data: UKAEA FAIR-MAST (mastapp.site), CC BY-SA 4.0",
    campaign_pattern=r"M\d+",
    usd_prim_root="/MAST",
    reconstruction_name="EFIT",
    te_display_range=(0.2, 1.4),
    te_usd_ramp_max_kev=1.5,
    warmup_shot="30166",
    clean_limits=CleanLimits(tau_E_max_s=0.5, kappa_min=1.2, a_min_m=0.3),
    # Midway between the two hand-verified anchors: 30420 (ohmic, ratio 1.29) and 30166
    # ("H-mode from 223 ms", peak current 0.245 s, ratio 1.81) — scripts/mode_flag_report.py.
    lh_enhancement_min=1.5,
    bench_shot="30166",
    unit_sanity=(   # mid-shot guards against a missed SI conversion (the values the suite asserts)
        ("R_m", 0.6, 1.0),
        ("a_m", 0.4, 0.7),
        ("B_T", 0.3, 0.7),          # sign removed
        ("Ip_MA", 0.3, 1.5),
        ("n_e20", 0.05, 1.0),
        ("W_MJ", 0.005, 0.3),       # plasma_energy, not the diamagnetic "wmhd"
        ("kappa", 1.4, 2.5),
        ("P_ohm_MW", 0.1, 5.0),
        ("Te0_keV", 0.1, 3.0),
    ),
)
