"""Virtual-shot routes: a what-if anchored on a cached real shot (fusionlab/virtual.py)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fusionlab import virtual
from fusionlab.api_replay import ATTRIBUTION, _j, _shot

router = APIRouter()

LAW_NAMES = {"ipb98": "IPB98(y,2)", "valovic": "Valovič 2009 (MAST)", "refit": "Refit power law (this project)"}
CLOSURE_NAMES = {"learned": "IPB98(y,2) × PhysicsNeMo correction", "ipb98": "IPB98(y,2)", "iter89": "ITER89-P"}
# which sliders stand on which matched-pairs test (scripts/validate_virtual.py); beam power and timing act through P
SLIDER_GATE = {"Ip": "Ip", "B": "B", "n": "n", "P_nbi": "P", "nbi_shift_s": "P"}
CAVEATS = [
    "Educational what-if, not a prediction. The level of confinement comes from this shot; the response to an edit comes from published scaling-law exponents.",
    "Not modelled: L–H transitions, fast-ion slowing down, the 30 ms after a beam edge (the beam is a logged box), the equilibrium's response to pressure.",
    "Limits are scaled from the measured values (β_N ∝ W/(B·Ip), q95 ∝ B/Ip, n_GW ∝ Ip). 'Would cross a limit' is a distance, never a forecast of a disruption.",
]


class EditIn(BaseModel):
    Ip: float = Field(1.0, ge=0.5, le=1.5)
    n: float = Field(1.0, ge=0.5, le=1.5)
    P_nbi: float = Field(1.0, ge=0.0, le=2.0)
    B: float = Field(1.0, ge=0.5, le=1.5)
    nbi_shift_s: float = Field(0.0, ge=-0.1, le=0.1)


class VirtualIn(BaseModel):
    edit: EditIn = EditIn()
    closure: str | None = None       # None = the closure that won the re-fly of held-out shots


def _validation() -> dict:
    m = virtual._metrics("virtual_metrics.json")
    return {"refly": m.get("refly"), "matched_pairs": m.get("matched_pairs"), "validated": bool(m)}


@router.get("/virtual/gate")
def virtual_gate():
    """What the UI may offer: one entry per slider with the matched-pairs evidence behind it, the closures' re-fly
    errors on held-out shots, and the response laws as the engine will apply them."""
    g, v = virtual.gate(), _validation()
    sliders = {}
    for slider, act in SLIDER_GATE.items():
        row = g.get(act)
        sliders[slider] = {"enabled": bool(row["slider"]) if row else True, "n_pairs": row["n_pairs"] if row else None,
                           "admissible": [LAW_NAMES.get(k, k) for k in (row or {}).get("admissible", [])],
                           "measured_exponent": (row or {}).get("measured_exponent")}
    return {"validated": v["validated"], "sliders": sliders, "refly": v["refly"], "matched_pairs": v["matched_pairs"],
            "closures": CLOSURE_NAMES, "default_closure": (v["refly"] or {}).get("default_closure") or "learned",
            "laws": {LAW_NAMES.get(k, k): dict(zip(("Ip", "B", "n", "alpha"), e)) for k, e in virtual.gated_laws().items()},
            "caveats": CAVEATS}


@router.post("/virtual/{shot_id}")
def virtual_shot(shot_id: int, body: VirtualIn):
    s = _shot(shot_id)
    g = virtual.gate()
    edit = virtual.Edit(**body.edit.model_dump())
    for slider, act in SLIDER_GATE.items():   # the gate is enforced here, not only drawn in the UI
        if getattr(edit, slider) != getattr(virtual.Edit(), slider) and g.get(act) and not g[act]["slider"]:
            raise HTTPException(422, f"'{slider}' cannot be edited: the archive has {g[act]['n_pairs']} matched pairs for it, "
                                     "too few to test any response law (scripts/validate_virtual.py)")
    closure = body.closure or (_validation()["refly"] or {}).get("default_closure") or "learned"
    if closure not in virtual.CLOSURES:
        raise HTTPException(422, f"closure must be one of {virtual.CLOSURES}")
    r = virtual.run(s, edit, closure)

    Wa, Wb, t = r["W_anchored_MJ"][0], r["W_blind_MJ"][0], r["t_s"]          # (laws, slices)
    limits = {"greenwald": r["f_greenwald"][0], "troyon": r["troyon"][0], "kink": r["kink"][0]}
    crossing = {}
    for k, v in limits.items():
        over = np.flatnonzero(np.nan_to_num(v, nan=0.0) >= 1.0)
        crossing[k] = None if over.size == 0 else round(float(t[over[0]]), 3)
    pct = lambda a: [round(100 * float(np.min(a)), 1), round(100 * float(np.max(a)), 1)]
    return {
        "shot_id": shot_id, "attribution": ATTRIBUTION, "closure": closure, "closure_name": CLOSURE_NAMES[closure],
        "edit": body.edit.model_dump(), "is_identity": edit.is_identity(), "timing_edit": edit.nbi_shift_s != 0.0,
        "laws": [LAW_NAMES.get(k, k) for k in r["laws"]],
        "t_s": _j(t), "W_meas_MJ": _j(r["W_meas_MJ"], 5), "W_refly_MJ": _j(r["W_refly_MJ"], 5), "P_nbi_MW": _j(r["P_nbi_MW"][0]),
        "anchored": {"lo": _j(Wa.min(0), 5), "hi": _j(Wa.max(0), 5), "by_law": _j(Wa, 5)},
        "blind": {"lo": _j(Wb.min(0), 5), "hi": _j(Wb.max(0), 5), "by_law": _j(Wb, 5)},
        "dW_flat_pct": {"anchored": pct(r["dW_flat_anchored"][0]), "blind": pct(r["dW_flat_blind"][0])},
        "limits": {k: _j(v) for k, v in limits.items()}, "would_cross_at_s": crossing,
        "in_distribution": _j(r["in_distribution"][0]), "refly_error": r["refly_error"],
        "validated": _validation()["validated"], "caveats": CAVEATS,
    }
