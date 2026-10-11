"""The DataSource seam: every surface consumes validated shot bundles, whichever source served them.

DATA-2's bring-your-own-data path. A `DataSource` is one named source with its machine's
`MachineConfig`; `get_source()` resolves the configured default (FUSIONLAB_DATA_SOURCE,
"fair-mast" unless set). `FairMastSource` delegates to the fusionlab.mast facade — kept
intact because that module is the monkeypatch surface the tests own — and a directory of
imported npz bundles is served by `UserFileSource` below. Shot resolution overlays user
bundles on the archive (api_replay._shot), so an uploaded shot replays, compares and
exports beside the shipped MAST cache without any route knowing which source answered.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from fusionlab import mast
from fusionlab.atomicio import atomic_write
from fusionlab.data import reader
from fusionlab.data.bundle import BundleValidationError, ShotBundle
from fusionlab.data.ingestors.fair_mast import _DB_COLS
from fusionlab.data.machine import CleanLimits, MachineConfig

# A bundle id names its file ({id}.npz) and rides in URLs and export filenames: letters,
# digits, dots, dashes or underscores, no leading separator (no traversal, no dotfiles).
BUNDLE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def valid_bundle_id(shot_id: str) -> bool:
    """True when `shot_id` is a well-formed bundle id (the only ids the routes accept)."""
    return BUNDLE_ID_RE.fullmatch(shot_id) is not None


class DataSource(Protocol):
    """One named data source: its machine's config, and validated bundles out.

    `load_shot` returns the plain dict the whole app passes around — a ShotBundle v1
    contract (fusionlab.data.bundle), validated at this door; `load_db` returns the
    catalog as a dict of equal-length columns.
    """

    name: str
    config: MachineConfig
    bundles_dir: Path   # where this source's bundle files live (an archive: its cache directory)

    def load_shot(self, shot_id: str) -> dict: ...
    def load_db(self) -> dict: ...
    def cached_shots(self) -> list[str]: ...
    def clean_db(self, db: dict) -> dict:
        """Rows usable for confinement analysis — thresholds from this source's machine config."""
        ...

    def import_bundle(self, shot_id: str, raw: bytes, meta: dict | None = None) -> dict:
        """Validate raw npz bytes against the ShotBundle v1 contract and publish atomically
        (no partial file on a rejected bundle). Read-only archive sources refuse."""
        ...


class FairMastSource:
    """The FAIR-MAST archive through the fusionlab.mast facade."""

    name = "fair-mast"
    config: MachineConfig = mast.CONFIG

    bundles_dir = reader.SHOTS   # the archive's own cache; imports never land here

    def import_bundle(self, shot_id: str, raw: bytes, meta: dict | None = None) -> dict:
        """The archive is read-only: bundles enter the app through a user-file source."""
        raise TypeError(f"source {self.name!r} is a read-only archive — "
                        "import bundles through a user-file source (the upload route)")

    def load_shot(self, shot_id: str) -> dict:
        return mast.load_shot(int(shot_id))

    def cached_shots(self) -> list[str]:
        return [str(i) for i in mast.cached_shots()]

    def load_db(self) -> dict:
        return mast.load_db()

    def clean_db(self, db: dict) -> dict:
        return mast.clean_db(db)


# ---------------------------------------------------------------- imported bundles (user-file)

MAX_BUNDLE_BYTES = 64 * 2**20   # shipped MAST bundles are ~1 MB; 64 MiB is generous headroom


def _read_bundle_npz(f: Path) -> dict:
    """One npz bundle file -> the shot dict (+ "meta"). `meta` rides as a JSON string (the reader
    cache's convention); anything unreadable raises a contract error that names the problem —
    never a raw zipfile or JSON traceback at the route."""
    try:
        with np.load(f) as z:
            out = {k: z[k] for k in z.files if k != "meta"}
            out["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    except (ValueError, zipfile.BadZipFile, AttributeError, OSError) as err:
        raise BundleValidationError(
            f"not a readable npz shot bundle ({f.name}): {err} — save the arrays with np.savez and "
            "meta as a JSON string") from err
    if not isinstance(out["meta"], dict):
        raise BundleValidationError(
            "meta: expected a JSON object (save it as np.savez(f, meta=json.dumps(meta), ...))")
    return out


def _default_user_config() -> MachineConfig:
    """Defaults for imported data — deliberately modest: generic attribution and naming, no
    MAST-calibrated thresholds, no unit-sanity ranges. A machine.json beside the bundles
    overrides any of it (docs/DATA_CONTRACT.md)."""
    return MachineConfig(
        device="",                      # no DEVICES preset is assumed for an imported machine
        isotope_md=2.0,                 # deuterium: the common case; set yours in machine.json
        alpha_particles=False,
        attribution="User-provided shot bundle; provenance as supplied with the data",
        campaign_pattern=r".*",         # imported campaigns are free-form strings
        usd_prim_root="/USER",
        reconstruction_name="imported reconstruction",
        te_display_range=(0.2, 10.0),   # foreign machines run hotter than MAST; set yours
        te_usd_ramp_max_kev=10.0,
        warmup_shot=None,
        clean_limits=CleanLimits(tau_E_max_s=float("inf"), kappa_min=0.0, a_min_m=0.0),
    )   # imported catalogs are hand-curated: no machine-calibrated filtering


def _machine_config(bundles_dir: Path) -> MachineConfig:
    """The imported machine's config: the user defaults overlaid with machine.json from the bundles
    directory. Unknown keys and unreadable JSON fail loudly, naming the field."""
    base = _default_user_config()
    f = bundles_dir / "machine.json"
    if not f.is_file():
        return base
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as err:
        raise ValueError(f"{f}: unreadable machine config: {err}") from err
    raw = raw.get("machine", raw) if isinstance(raw, dict) else None   # {"machine": {...}} or the flat dict
    if not isinstance(raw, dict):
        raise ValueError(f"{f}: expected a JSON object of MachineConfig fields")
    unknown = [k for k in raw if k not in MachineConfig.__dataclass_fields__]
    if unknown:
        raise ValueError(f"{f}: unknown MachineConfig field(s) {', '.join(sorted(unknown))}")
    try:
        if "clean_limits" in raw:
            raw["clean_limits"] = CleanLimits(**{**base.clean_limits.__dict__, **raw["clean_limits"]})
        if "te_display_range" in raw:
            raw["te_display_range"] = tuple(raw["te_display_range"])
        if "unit_sanity" in raw:
            raw["unit_sanity"] = tuple((k, float(lo), float(hi)) for k, lo, hi in raw["unit_sanity"])
        return replace(base, **raw)
    except TypeError as err:
        raise ValueError(f"{f}: bad machine config value: {err}") from err


class UserFileSource:
    """Imported shot bundles: one npz file per shot in `bundles_dir` (the reader cache's format),
    validated against ShotBundle v1 on every read. Point FUSIONLAB_USER_BUNDLES at a directory of
    bundles, or upload through the API (import_bundle); machine.json in the same directory
    configures the machine (docs/DATA_CONTRACT.md)."""

    name = "user-file"

    def __init__(self, bundles_dir: Path | None = None):
        env = os.environ.get("FUSIONLAB_USER_BUNDLES", "")
        if bundles_dir is not None:
            self.bundles_dir = Path(bundles_dir)
        elif env:
            self.bundles_dir = Path(env)
        else:
            self.bundles_dir = reader.DATA / "user_bundles"
        self.config: MachineConfig = _machine_config(self.bundles_dir)

    def cached_shots(self) -> list[str]:
        """Sorted ids of the imported bundles (one id per .npz file)."""
        if not self.bundles_dir.is_dir():
            return []
        return sorted(p.stem for p in self.bundles_dir.glob("*.npz") if valid_bundle_id(p.stem))

    def load_shot(self, shot_id: str) -> dict:
        """The bundle's arrays + meta, the file name's id stamped into meta (identity cannot drift
        from the catalog), validated against the contract at this door."""
        f = self.bundles_dir / f"{shot_id}.npz"
        if not f.is_file():
            raise FileNotFoundError(
                f"no imported bundle named {shot_id!r} in {self.bundles_dir} — upload it "
                "(POST /data/bundles/{id}) or point FUSIONLAB_USER_BUNDLES at its directory")
        out = _read_bundle_npz(f)
        out["meta"]["shot_id"] = shot_id
        ShotBundle.from_dict(out, source=self.name)
        return out

    def import_bundle(self, shot_id: str, raw: bytes, meta: dict | None = None) -> dict:
        """Validate-then-publish: raw npz bytes -> one written bundle. The bytes land on a
        same-directory temp (fusionlab.atomicio), are validated from there against the contract
        (the error names the first bad field), and only a valid bundle is renamed into place —
        a rejected upload never leaves a partial file behind. Returns the published bundle."""
        if not valid_bundle_id(shot_id):
            raise ValueError(f"bundle id {shot_id!r} is not importable: use letters, digits, dots, "
                             "dashes or underscores, starting with a letter or digit")
        if shot_id.isdigit():   # bare shot numbers are the archive's - imports never shadow them
            raise ValueError(f"bundle id {shot_id!r} is a bare shot number, which the archive owns - "
                             "name the import something else (e.g. {shot_id}_local)")
        if len(raw) > MAX_BUNDLE_BYTES:
            raise ValueError(f"bundle exceeds the {MAX_BUNDLE_BYTES} byte import cap")
        target = self.bundles_dir / f"{shot_id}.npz"

        def write_and_validate(tmp: Path) -> None:
            tmp.write_bytes(raw)
            out = _read_bundle_npz(tmp)
            out["meta"] = {**out["meta"], **(meta or {}), "shot_id": shot_id}
            with tmp.open("wb") as f:   # the published copy carries the merged meta
                np.savez_compressed(f, meta=json.dumps(out["meta"]),
                                    **{k: v for k, v in out.items() if k != "meta"})
            ShotBundle.from_dict(out, source=self.name)   # before the publish rename

        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(target, write_and_validate)
        return self.load_shot(shot_id)

    def load_db(self) -> dict:
        """A derived catalog: one row per imported bundle, evaluated at its peak-current slice (the
        archives' convention). tau_E and beta_t_pct are computed here by the documented formulas
        and labelled computed; useful=1 (imported on purpose) and abort=0 (the archives' curation
        flags do not exist for imported data) are the imported-catalog defaults."""
        cols: dict[str, list] = {"shot_id": [], "campaign": [], **{k: [] for k in _DB_COLS},
                                 "useful": [], "abort": []}
        for sid in self.cached_shots():
            s = self.load_shot(sid)
            i = int(np.nanargmax(np.abs(s["Ip_MA"])))
            p_loss = float(s["P_ohm_MW"][i] + s["P_nbi_MW"][i] - s["dWdt_MW"][i])
            tau_e = float(s["W_MJ"][i]) / p_loss if p_loss > 0 else float("nan")
            with np.errstate(divide="ignore", invalid="ignore"):
                beta_t = float(s["beta_N"][i]) * float(s["Ip_MA"][i]) / (float(s["a_m"][i]) * float(s["B_T"][i]))
            for k, v in {
                "shot_id": sid, "campaign": str(s["meta"].get("campaign", "")),
                "Ip_MA": float(s["Ip_MA"][i]), "B_T": float(s["B_T"][i]), "n_e20": float(s["n_e20"][i]),
                "P_nbi_MW": float(s["P_nbi_MW"][i]), "P_ohm_MW": float(s["P_ohm_MW"][i]),
                "P_rad_MW": float(s["P_rad_MW"][i]) if "P_rad_MW" in s else float("nan"),
                "W_MJ": float(s["W_MJ"][i]),
                "Te0_keV": float(s["Te0_keV"][i]) if "Te0_keV" in s else float("nan"),
                "q95": float(s["q95"][i]), "kappa": float(s["kappa"][i]),
                "beta_t_pct": beta_t, "a_m": float(s["a_m"][i]), "R_m": float(s["R_m"][i]),
                "V_m3": float(s["V_m3"][i]), "tau_E_s": tau_e, "dWdt_MW": float(s["dWdt_MW"][i]),
                "t_ipmax_s": float(s["t_s"][i]), "useful": 1, "abort": 0,
            }.items():
                cols[k].append(v)
        return {k: np.asarray(v) for k, v in cols.items()}

    def clean_db(self, db: dict) -> dict:
        """The imported catalog is hand-curated; the source's clean_limits (default: no filtering)
        is all that applies - never MAST's machine-calibrated thresholds."""
        return reader.clean_db(db, limits=self.config.clean_limits)


_SOURCES: dict[str, type] = {"fair-mast": FairMastSource, "user-file": UserFileSource}
DEFAULT_SOURCE = "fair-mast"


def get_source(name: str | None = None) -> DataSource:
    """The named source, or the configured default: FUSIONLAB_DATA_SOURCE, "fair-mast" unless set.

    Unknown names fail loudly — the app must never silently replay the wrong machine's data.
    """
    name = name or os.environ.get("FUSIONLAB_DATA_SOURCE", DEFAULT_SOURCE)
    if name not in _SOURCES:
        raise ValueError(f"unknown data source {name!r}; available: {sorted(_SOURCES)}")
    src: DataSource = _SOURCES[name]()
    return src
