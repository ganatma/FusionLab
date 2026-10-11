"""FAIR-MAST source facade: real MAST shots -> arrays in FusionLab units, cached under data/.

This module is the app's data door and keeps the API it has always had (`load_shot`,
`load_db`, `clean_db`, `cached_shots` and the fetch internals tests monkeypatch). The
archive specifics live in `fusionlab.data.ingestors.fair_mast` (REST + anonymous S3/Zarr +
ndjson), the generic npz cache in `fusionlab.data.reader`, the versioned contract in
`fusionlab.data.bundle` (ShotBundle v1 — every load is validated against it), and the
machine-specific values in `fusionlab.data.machine.MAST_CONFIG`. A second machine (issue
#34) is a new ingestor plus a new config — this facade's callers do not change.

Data: UKAEA FAIR-MAST (https://mastapp.site), licence CC BY-SA 4.0. Cite Jackson et al.,
SoftwareX 27 (2024) 101869. Signal -> quantity table and conversions: docs/PHYSICS.md.

Two anonymous sources:
  * shot table (ndjson, one request)   -> load_db():   one row per shot at the time of peak current
  * level-2 Zarr on S3 (one per shot)  -> load_shot(): traces on the EFIT time base + geometry

Everything leaves this module in project units: m, T, MA, 1e20 m^-3, keV, MW (plus MJ, s).
The shipped cache powers the demo with no network; /replay/{id} additionally exposes load_shot()'s
fetch-and-cache path, so any shot in the catalog can be opened on demand (multi-shot compare).
"""

from __future__ import annotations

from fusionlab.data import reader
from fusionlab.data.ingestors import fair_mast
from fusionlab.data.machine import MAST_CONFIG

# Paths (tests monkeypatch SHOTS/DB_FILE by module attribute; wrappers below read them at call time)
DATA = reader.DATA
SHOTS = reader.SHOTS
DB_FILE = reader.DB_FILE

# The machine this facade serves.
CONFIG = MAST_CONFIG

# Archive internals, re-exported for backward compatibility (tests monkeypatch fetch_shot here).
_open = fair_mast._open
_on = fair_mast._on
_pf_coils = fair_mast._pf_coils
_psi_tzr = fair_mast._psi_tzr
_META_KEYS = fair_mast._META_KEYS
_nbi_trace = fair_mast._nbi_trace
_DB_COLS = fair_mast._DB_COLS
REST = fair_mast.REST
S3_ENDPOINT = fair_mast.S3_ENDPOINT
IP_MIN_MA = fair_mast.IP_MIN_MA
fetch_db = fair_mast.fetch_db
fetch_meta = fair_mast.fetch_meta
fetch_shot = fair_mast.fetch_shot
_savez = reader._savez


def load_shot(shot_id: int, refresh: bool = False) -> dict:
    """One shot as a dict of arrays (+ "meta"). Reads data/shots/{id}.npz; fetches from S3 only if missing.

    Validated against ShotBundle v1 on read (fusionlab.data.reader).
    """
    return reader.load_shot(shot_id, refresh, shots_dir=SHOTS, fetch=fetch_shot,
                            source="fair-mast")


def cached_shots() -> list[int]:
    """Sorted ids of the shot bundles currently in the cache."""
    return reader.cached_shots(SHOTS)


def load_db(refresh: bool = False) -> dict:
    """Shot table as a dict of equal-length arrays. Cached in data/mast_db.npz."""
    return reader.load_db(refresh, db_file=DB_FILE, fetch=fetch_db)


def clean_db(db: dict) -> dict:
    """Rows usable for confinement analysis — thresholds from the machine config."""
    return reader.clean_db(db, limits=CONFIG.clean_limits)
