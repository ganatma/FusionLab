"""Generic shot-bundle cache: npz files under data/shots/, catalog in data/mast_db.npz.

Source-agnostic by construction: the caller passes the directory and the fetch callable,
so a second ingestor (#34) plugs in without touching this module — the cache layout, the
atomic writes and the contract validation at the data door are shared. Every load is
validated against the ShotBundle v1 contract (fusionlab.data.bundle) so a malformed bundle
fails with a message naming the field, before any consumer sees it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from fusionlab.atomicio import atomic_write
from fusionlab.data.bundle import ShotBundle
from fusionlab.data.machine import CleanLimits

DATA = Path(__file__).resolve().parents[2] / "data"
SHOTS = DATA / "shots"
DB_FILE = DATA / "mast_db.npz"


def _savez(tmp: Path, **arrays) -> None:
    """np.savez_compressed(Path) appends .npz to names not ending in it; a file object writes exactly where told."""
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)


def load_shot(shot_id: int, refresh: bool = False, *, shots_dir: Path = SHOTS,
              fetch: Callable[[int], dict], source: str = "fair-mast") -> dict:
    """One shot as a dict of arrays (+ "meta"). Reads {shots_dir}/{id}.npz; fetches only if missing.

    The dict is validated against ShotBundle v1 on every read — the contract door: a
    corrupt cache or a misbehaving ingestor surfaces as a field-naming error, not as
    downstream physics breakage.
    """
    f = shots_dir / f"{int(shot_id)}.npz"
    if refresh or not f.exists():
        s = fetch(int(shot_id))
        shots_dir.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(s.pop("meta"))
        atomic_write(f, lambda tmp: _savez(tmp, meta=meta, **s))
    with np.load(f) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    ShotBundle.from_dict(out, source=source)
    return out


def cached_shots(shots_dir: Path = SHOTS) -> list[int]:
    """Sorted ids of the shot bundles currently in the cache directory."""
    return sorted(int(p.stem) for p in shots_dir.glob("*.npz") if p.stem.isdigit())


def load_db(refresh: bool = False, *, db_file: Path = DB_FILE, fetch: Callable[[], dict]) -> dict:
    """Shot table as a dict of equal-length arrays. Cached in db_file."""
    if refresh or not db_file.exists():
        db_file.parent.mkdir(exist_ok=True)
        atomic_write(db_file, lambda tmp: _savez(tmp, **fetch()))
    with np.load(db_file) as z:
        return {k: z[k] for k in z.files}


def clean_db(db: dict, *, limits: CleanLimits) -> dict:
    """Rows usable for confinement analysis: flat-top-ish, not aborted, every input finite and sane.

    The thresholds are machine-calibrated and come from the source's MachineConfig.
    """
    need = ("Ip_MA", "B_T", "n_e20", "P_ohm_MW", "W_MJ", "dWdt_MW", "tau_E_s", "q95", "kappa", "a_m", "R_m", "V_m3")
    ok = np.all([np.isfinite(db[k]) for k in need], axis=0) & (db["abort"] == 0)
    P_loss = db["P_ohm_MW"] + db["P_nbi_MW"] - db["dWdt_MW"]
    ok &= (db["Ip_MA"] > 0.3) & (db["B_T"] > 0.2) & (db["n_e20"] > 0.05) & (db["W_MJ"] > 0.005)
    ok &= (P_loss > 0.1) & (np.abs(db["dWdt_MW"]) < 0.3 * (db["P_ohm_MW"] + db["P_nbi_MW"]))  # near steady state
    ok &= ((db["tau_E_s"] > 1e-3) & (db["tau_E_s"] < limits.tau_E_max_s)
           & (db["kappa"] > limits.kappa_min) & (db["a_m"] > limits.a_min_m))
    out = {k: v[ok] for k, v in db.items()}
    out["P_loss_MW"] = P_loss[ok]
    return out
