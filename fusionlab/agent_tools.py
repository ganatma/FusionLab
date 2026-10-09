"""Allowlisted tool registry for the in-app natural-language agent (blueprint art_LbvYoBDn, "Tool registry v1").

Every tool wraps the module functions the routers use — never an HTTP self-call — so the agent sees
exactly what the UI sees: the /db/search filter vector (search_shots calls the route function itself),
the replay cache boundary (_shot, the loaders behind /shots and /replay/{id}), the matched-pairs gate
on /virtual/{id} (run_whatif calls the virtual_shot route function, so the gate's 422 text and the
EditIn bounds are the route's own, verbatim), and /simulate's range guard.

Read-only tools execute and return what they found. The two tools that would change anything the user
can see — run_whatif (the what-if sliders in the Replay panel) and export_usd (writes out/ and
downloads) — execute nothing: they return a ToolResult with `proposal` set,

    {"tool": ..., "args": <as given>, "math"?: <echoed arithmetic>, "apply": <payload>}

and the agent loop must hold that proposal for user confirmation before the browser driver applies it.
`apply` for run_whatif names the what-if sliders by their EditIn keys (the web/replay.js panel keys:
P_nbi, n, Ip, B, nbi_shift_s) with the values as strings, since DOM .value is a string; for
export_usd it names the POST the driver should make (the route stays the only writer of out/).

Refusals travel in ToolResult.error verbatim — the gate's 422 detail, the cache-boundary 404 — so the
model can explain a refusal instead of retrying it. One deliberate deviation from the blueprint's
illustrative block: run_whatif's `data` carries the route's summary fields, not its full time arrays;
the model argues from bands and crossings, and the UI draws the curves after confirmation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from fastapi import HTTPException

from fusionlab import mast, virtual
from fusionlab.api import devices, simulate_point
from fusionlab.api_replay import _shot, db_search, replay_shot, shots
from fusionlab.api_virtual import EditIn, VirtualIn, virtual_gate, virtual_shot
from fusionlab.physics import DEVICES

SEARCH_LIMIT = 200   # the tool's own cap; /db/search itself allows 1000 — the model works page-size

_RANGE_KEYS = {
    "ip_MA": ("ip_min", "ip_max"), "bt_T": ("bt_min", "bt_max"), "pnbi_MW": ("pnbi_min", "pnbi_max"),
    "ne_20": ("ne_min", "ne_max"), "W_MJ": ("w_min", "w_max"), "q95": ("q95_min", "q95_max"),
}
_RANGE_UNITS = {"ip_MA": "MA", "bt_T": "T", "pnbi_MW": "MW", "ne_20": "1e20 m^-3", "W_MJ": "MJ", "q95": "dimensionless"}


# ---------------------------------------------------------------- results and refusals
@dataclass(frozen=True)
class ToolResult:
    """One tool execution. `error` set means the tool refused (gate, cache boundary, bad args);
    `proposal` set means a confirm-then-apply tool described a change instead of making it."""

    summary: str
    data: dict[str, Any]
    error: str | None = None
    proposal: dict[str, Any] | None = None


def _err(message: str, data: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(summary="", data=data or {}, error=message)


def _guard(handler: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """A domain refusal is a tool result, never a raised exception reaching the agent loop: the model
    must be able to read the matched-pairs text and explain it (blueprint verification table)."""

    @wraps(handler)
    def wrapped(**kw: Any) -> ToolResult:
        try:
            return handler(**kw)
        except HTTPException as e:   # route-function detail strings, verbatim
            return _err(str(e.detail))

    return wrapped


# ---------------------------------------------------------------- read-only tools
@_guard
def search_shots(q: str | None = None, campaign: str | None = None,
                 ip_MA: list[float | None] | None = None, bt_T: list[float | None] | None = None,
                 pnbi_MW: list[float | None] | None = None, ne_20: list[float | None] | None = None,
                 W_MJ: list[float | None] | None = None, q95: list[float | None] | None = None,
                 useful: bool | None = None, abort: bool | None = None, limit: int = SEARCH_LIMIT) -> ToolResult:
    """Filter the full FAIR-MAST catalog — the /db/search route function itself, so the mapping is 1:1 by construction."""
    range_args = {"ip_MA": ip_MA, "bt_T": bt_T, "pnbi_MW": pnbi_MW, "ne_20": ne_20, "W_MJ": W_MJ, "q95": q95}
    kw: dict[str, Any] = {"q": q, "campaign": campaign, "useful": useful, "abort": abort}
    for arg, pair in range_args.items():
        if pair is None:
            continue
        if len(pair) != 2:
            return _err(f"{arg} must be a [min, max] pair, e.g. [0.8, 1.2]; either end may be null")
        kw[_RANGE_KEYS[arg][0]], kw[_RANGE_KEYS[arg][1]] = pair[0], pair[1]
    r = db_search(**kw, limit=min(int(limit), SEARCH_LIMIT), offset=0)
    n, shown = r["total"], len(r["rows"])
    page = f"showing the first {shown}" if shown < n else "showing all"
    return ToolResult(
        summary=f"{n} shots match the filters ({page}), in ascending shot_id order — the archive has no sort control.",
        data=r,
    )


@_guard
def list_cached_shots() -> ToolResult:
    """The shots replay can actually run — the /shots route function. The app never fetches from the network."""
    r = shots()
    ids = [s["shot_id"] for s in r["shots"]]
    return ToolResult(summary=f"{len(ids)} shots are cached locally (replay works offline on these only): {ids}.", data=r)


@_guard
def load_shot(shot_id: int, H: float = 1.0) -> ToolResult:
    """A cached shot's replay summary — the /replay/{id} internals. Time arrays stay out of the model's
    context: the UI draws the traces after the driver opens the shot."""
    try:
        r = replay_shot(int(shot_id), H=float(H))
    except HTTPException as e:
        cached = mast.cached_shots()
        return _err(f"{e.detail} Replay covers only the shots cached under data/shots ({len(cached)}: {cached}); "
                    "use search_shots to explore the full catalog instead.", {"shot_id": int(shot_id), "cached_shot_ids": cached})
    meta = r["meta"]
    n = len(r["measured"]["t_s"])
    s = meta.get("postshot") or ""
    return ToolResult(
        summary=(f"Shot {shot_id} (campaign {meta.get('campaign')}): {n} slices over "
                 f"{r['measured']['t_s'][0]}–{r['measured']['t_s'][-1]} s, peak beam power "
                 f"{meta.get('P_nbi_max_MW')} MW. {r['summary']} Logbook: {s[:200]}"),
        data={"shot_id": int(shot_id), "meta": meta, "summary": r["summary"], "limit_names": r["limit_names"],
              "n_slices": n, "attribution": r["attribution"]},
    )


@_guard
def get_whatif_gate() -> ToolResult:
    """What the /virtual/{id} route will accept — the /virtual/gate route function."""
    r = virtual_gate()
    on = [k for k, v in r["sliders"].items() if v["enabled"]]
    off = [k for k, v in r["sliders"].items() if not v["enabled"]]
    refused = f" Refused for too few matched pairs (get_whatif_gate details): {', '.join(off)}." if off else ""
    return ToolResult(summary=f"What-if edits the archive can test: {', '.join(on)}.{refused}", data=r)


@_guard
def get_simulate(device: str = "iter", Ip: float | None = None, B: float | None = None, n: float | None = None,
                 P_aux: float | None = None, H: float | None = None, Zeff: float | None = None) -> ToolResult:
    """One 0D operating point — the /simulate route function (same range guard, same 422s)."""
    kw = {k: v for k, v in dict(Ip=Ip, B=B, n=n, P_aux=P_aux, H=H, Zeff=Zeff).items() if v is not None}
    r = simulate_point(device=device, **kw)
    res, c = r["result"], r["controls"]
    return ToolResult(
        summary=(f"{c['device']}: T = {res['T_keV']:.1f} keV, Q = {res['Q']:.2f}, "
                 f"{'H-mode' if res['hmode'] else 'L-mode'}, worst operating limit {res['worst_limit']:.2f} "
                 "(1.0 is the limit's edge — a distance, not a forecast)."),
        data=r,
    )


@_guard
def list_devices() -> ToolResult:
    """The sandbox devices — the /devices route function."""
    r = devices()
    return ToolResult(summary=f"{len(r)} devices: " + ", ".join(f"{v['name']} ({k})" for k, v in r.items()) + ".", data={"devices": r})


# ---------------------------------------------------------------- confirm-then-apply tools
def _slider_bounds(field: str) -> dict[str, Any]:
    """EditIn's enforced bounds, read from the model's own Field constraints: the run_whatif schema can
    never drift from what the route enforces (the gate is enforced here, not only drawn in the UI)."""
    f = EditIn.model_fields[field]
    ge = [m.ge for m in f.metadata if getattr(m, "ge", None) is not None]
    le = [m.le for m in f.metadata if getattr(m, "le", None) is not None]
    if not ge or not le:
        raise RuntimeError(f"EditIn.{field} has no enforced bounds; the run_whatif schema depends on them")
    return {"minimum": ge[0], "maximum": le[0]}


def _fmt(value: float, places: int) -> str:
    return f"{value:.{places}f}"


@_guard
def run_whatif(shot_id: int, p_nbi_MW: float | None = None, p_nbi_scale: float | None = None,
               n_scale: float | None = None, ip_scale: float | None = None, bt_scale: float | None = None,
               nbi_shift_s: float | None = None, closure: str | None = None) -> ToolResult:
    """Compute a bounded what-if on a cached shot and propose it — the /virtual/{id} route function, so
    the matched-pairs gate is re-enforced with its own 422 text. Nothing is applied: the proposal waits
    for confirmation, then the browser driver moves the what-if sliders."""
    if p_nbi_MW is not None and p_nbi_scale is not None:
        return _err("pass one of p_nbi_MW (absolute beam power) or p_nbi_scale (relative to the shot's logged peak), not both")

    edit: dict[str, float] = {}
    math_bits: list[str] = []
    power: float | None = None          # the requested beam power, absolute MW or a scale factor
    power_is_absolute = False
    if p_nbi_MW is not None:
        power, power_is_absolute = float(p_nbi_MW), True
    elif p_nbi_scale is not None:
        power = float(p_nbi_scale)
    if power is not None:
        pmax = float(_shot(int(shot_id))["meta"].get("P_nbi_max_MW") or 0.0)
        if not pmax:
            return _err(f"shot {shot_id} has no NBI beams in the archive (P_nbi_max 0 MW): beam power cannot be "
                        "scaled — try n_scale or a beam-timing edit (nbi_shift_s)")
        scale = power / pmax if power_is_absolute else power   # exact: this is what the engine runs, the slider string is formatted later
        b = _slider_bounds("P_nbi")
        if not b["minimum"] <= scale <= b["maximum"]:
            return _err(f"the requested beam power is {scale:.2f} × P_nbi_max {pmax:g} MW, "
                        f"outside the enforced range {b['minimum']:g}–{b['maximum']:g}")
        edit["P_nbi"] = scale
        math_bits.append(f"P_nbi {power:g} MW = {scale:.2f} × P_nbi_max {pmax:g} MW" if power_is_absolute
                         else f"P_nbi scale {scale:.2f} (× P_nbi_max {pmax:g} MW)")
    for arg, value, field in (("n_scale", n_scale, "n"), ("ip_scale", ip_scale, "Ip"), ("bt_scale", bt_scale, "B")):
        if value is None:
            continue
        b = _slider_bounds(field)
        if not b["minimum"] <= value <= b["maximum"]:
            return _err(f"{arg} {value:g} is outside the enforced range {b['minimum']:g}–{b['maximum']:g} "
                        "(the what-if sliders' bounds)")
        edit[field] = float(value)
    if nbi_shift_s is not None:
        b = _slider_bounds("nbi_shift_s")
        if not b["minimum"] <= nbi_shift_s <= b["maximum"]:
            return _err(f"nbi_shift_s {nbi_shift_s:g} is outside the enforced range {b['minimum']:g}–{b['maximum']:g}")
        edit["nbi_shift_s"] = float(nbi_shift_s)
    if not edit:
        return _err("no edit given: pass at least one of p_nbi_MW/p_nbi_scale, n_scale, ip_scale, bt_scale, nbi_shift_s")

    r = virtual_shot(int(shot_id), VirtualIn(edit=EditIn(**edit), closure=closure))
    apply = {"sliders": {k: _fmt(v, 3 if k == "nbi_shift_s" else 2) for k, v in edit.items()},
             "events": ["fusionlab:tab"]}
    proposal = {"tool": "run_whatif",
                "args": {k: v for k, v in dict(shot_id=int(shot_id), p_nbi_MW=p_nbi_MW, p_nbi_scale=p_nbi_scale,
                                               n_scale=n_scale, ip_scale=ip_scale, bt_scale=bt_scale,
                                               nbi_shift_s=nbi_shift_s, closure=closure).items() if v is not None},
                "math": "; ".join(math_bits),
                "apply": apply}
    data = {k: r[k] for k in ("shot_id", "closure", "closure_name", "edit", "is_identity", "timing_edit", "laws",
                              "dW_flat_pct", "would_cross_at_s", "in_distribution", "refly_error", "validated",
                              "caveats", "attribution")}
    return ToolResult(summary=_whatif_summary(int(shot_id), r), data=data, proposal=proposal)


def _whatif_summary(shot_id: int, r: dict) -> str:
    """Plain language for the bands the route computed. House wording: crossings are distances,
    confinement is compared with MAST data, nothing is a prediction."""
    lo, hi = r["dW_flat_pct"]["anchored"]
    out = [f"What-if on shot {shot_id} with {r['closure_name']}: flat-top stored energy changes "
           f"{lo:+.1f}% to {hi:+.1f}% across the response laws, compared with MAST data."]
    crossings = [f"would cross the {k} limit at t = {t:g} s" for k, t in r["would_cross_at_s"].items() if t is not None]
    out.append("The edited run " + " and ".join(crossings) + " — a distance, never a forecast of a disruption."
               if crossings else "No operating limit is reached across the shot's time base.")
    off = sum(1 for x in r["in_distribution"] if x is False)
    if off:
        out.append(f"{off} of {len(r['in_distribution'])} slices sit outside the range the learned correction was trained on.")
    out.append("Confirming applies it in the what-if panel; nothing is applied yet.")
    return " ".join(out)


@_guard
def export_usd(shot_id: int) -> ToolResult:
    """Propose an OpenUSD export (Omniverse-compatible) of a cached shot. The write happens only after
    confirmation, when the driver POSTs the route — the export path stays the route's own (temp file,
    atomic publish); the registry never writes."""
    _shot(int(shot_id))   # the cache boundary is the route's: a 404 here is the verbatim refusal
    return ToolResult(
        summary=f"Shot {shot_id} is in the local cache; confirming will export it as an OpenUSD stage "
                "(Omniverse-compatible) and download it.",
        data={"shot_id": int(shot_id)},
        proposal={"tool": "export_usd", "args": {"shot_id": int(shot_id)},
                  "apply": {"download": {"method": "POST", "url": f"/replay/{int(shot_id)}/usd"}}},
    )


# ---------------------------------------------------------------- the registry
@dataclass(frozen=True)
class Tool:
    """One allowlisted tool: its Anthropic definition, its handler, and whether it proposes instead of applying."""

    definition: dict[str, Any]
    handler: Callable[..., ToolResult]
    confirm_then_apply: bool = False


def _range_property(unit: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": ["number", "null"]}, "minItems": 2, "maxItems": 2,
            "description": f"[min, max] in {unit}; either end may be null"}


def _range_properties() -> dict[str, Any]:
    return {arg: _range_property(unit) for arg, unit in _RANGE_UNITS.items()}


TOOLS: dict[str, Tool] = {
    "search_shots": Tool(
        definition={
            "name": "search_shots",
            "description": (
                "Filter the full FAIR-MAST shot catalog (~16k rows) by numbers, not names: shot number/range, "
                "campaign, plasma current, toroidal field, beam power, density, stored energy, q95, and the "
                "archive's useful/abort flags. Filters combine (AND). Results are always in ascending shot_id "
                "order — there is no sort control — and the archive has no H-mode flag, so shots cannot be "
                "filtered or found by confinement mode."
            ),
            "input_schema": {
                "type": "object",
                "properties": {**_range_properties(),
                               "q": {"type": "string", "description": "shot number ('30421') or inclusive range ('30400-30500')"},
                               "campaign": {"type": "string", "description": "campaign as the archive names it ('M9'; bare digits work: '9')"},
                               "useful": {"type": "boolean", "description": "true keeps shots the archive marked useful; false the rest"},
                               "abort": {"type": "boolean", "description": "true keeps shots the archive marked aborted; false the rest"},
                               "limit": {"type": "integer", "minimum": 1, "maximum": SEARCH_LIMIT, "default": SEARCH_LIMIT,
                                         "description": f"page size, at most {SEARCH_LIMIT}; narrow the filters instead of paging blindly"}},
            },
        },
        handler=search_shots,
    ),
    "list_cached_shots": Tool(
        definition={
            "name": "list_cached_shots",
            "description": "List the shots cached locally under data/shots — the only shots replay can run offline. The app never fetches from the network.",
            "input_schema": {"type": "object", "properties": {}},
        },
        handler=list_cached_shots,
    ),
    "load_shot": Tool(
        definition={
            "name": "load_shot",
            "description": (
                "Load a cached shot's replay: measured traces, the 0D model beside them, and the operating limits. "
                "An uncached id returns the cache boundary — use search_shots for the full catalog."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"shot_id": {"type": "integer", "description": "a cached shot id (list_cached_shots)"},
                               "H": {"type": "number", "default": 1.0, "description": "confinement multiplier on the model traces"}},
                "required": ["shot_id"],
            },
        },
        handler=load_shot,
    ),
    "get_whatif_gate": Tool(
        definition={
            "name": "get_whatif_gate",
            "description": (
                "Which what-if edits the archive can test: per-slider matched-pairs evidence, the admissible "
                "response laws, the closures, and the honesty caveats. Call before proposing edits — an edit the "
                "evidence cannot test is refused, never guessed."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        handler=get_whatif_gate,
    ),
    "run_whatif": Tool(
        definition={
            "name": "run_whatif",
            "description": (
                "Re-fly a cached shot with a bounded programme edit: beam power (absolute MW or a scale factor — "
                "the conversion and its arithmetic are echoed), density, plasma current, toroidal field, or beam "
                "timing. Confirm-then-apply: the result is a proposal the user must confirm before the what-if "
                "panel applies it. Edits the matched-pairs evidence cannot test are refused (with the reason) — "
                "never guessed. Educational what-if compared with MAST data, not a prediction."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"shot_id": {"type": "integer", "description": "a cached shot id (list_cached_shots)"},
                               "p_nbi_MW": {"type": "number", "description": "absolute beam power in MW; converted to a scale factor server-side, math echoed"},
                               "p_nbi_scale": {**_slider_bounds("P_nbi"), "description": "beam-power scale factor (1.0 = the shot's logged peak)"},
                               "n_scale": {**_slider_bounds("n"), "description": "density scale factor"},
                               "ip_scale": {**_slider_bounds("Ip"), "description": "plasma-current scale factor"},
                               "bt_scale": {**_slider_bounds("B"), "description": "toroidal-field scale factor"},
                               "nbi_shift_s": {**_slider_bounds("nbi_shift_s"), "description": "beam-box time shift in s"},
                               "closure": {"type": "string", "enum": list(virtual.CLOSURES),
                                           "description": "blind baseline for the re-fly; default is the closure that won the held-out re-fly"}},
                "required": ["shot_id"],
            },
        },
        handler=run_whatif,
        confirm_then_apply=True,
    ),
    "get_simulate": Tool(
        definition={
            "name": "get_simulate",
            "description": (
                "One steady-state operating point from the reduced 0D model (the Sandbox tab): temperature, "
                "fusion gain Q, confinement time, mode, and the operating limits. Device-relative floors and "
                "caps refuse overflow-scale inputs with the route's own 422."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"device": {"type": "string", "enum": sorted(DEVICES)},
                               "Ip": {"type": "number", "description": "plasma current, MA (default: the device's max)"},
                               "B": {"type": "number", "description": "toroidal field, T (default: the device's max)"},
                               "n": {"type": "number", "description": "volume-average density, 1e20 m^-3 (default 1.0)"},
                               "P_aux": {"type": "number", "description": "auxiliary heating, MW (default 50)"},
                               "H": {"type": "number", "description": "confinement multiplier (default 1.0)"},
                               "Zeff": {"type": "number", "description": "effective charge (default 1.7)"}},
            },
        },
        handler=get_simulate,
    ),
    "list_devices": Tool(
        definition={
            "name": "list_devices",
            "description": "List the sandbox devices with their geometry, field, current, and installed heating.",
            "input_schema": {"type": "object", "properties": {}},
        },
        handler=list_devices,
    ),
    "export_usd": Tool(
        definition={
            "name": "export_usd",
            "description": (
                "Export a cached shot's stage as OpenUSD (Omniverse-compatible) and download it. "
                "Confirm-then-apply: the export writes out/ only after the user confirms."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"shot_id": {"type": "integer", "description": "a cached shot id (list_cached_shots)"}},
                "required": ["shot_id"],
            },
        },
        handler=export_usd,
        confirm_then_apply=True,
    ),
}


def tool_definitions() -> list[dict[str, Any]]:
    """Anthropic tool definitions for the Messages API, in the blueprint's stable order."""
    return [t.definition for t in TOOLS.values()]


def run_tool(name: str, **args: Any) -> ToolResult:
    """Execute one allowlisted tool. Bad invocations (unknown name, missing or unexpected arguments) are
    tool errors the model can read and correct — internal bugs still raise."""
    tool = TOOLS.get(name)
    if tool is None:
        return _err(f"unknown tool '{name}'; the allowlisted tools are: {', '.join(TOOLS)}")
    properties = tool.definition["input_schema"].get("properties", {})
    missing = [r for r in tool.definition["input_schema"].get("required", []) if r not in args]
    if missing:
        return _err(f"missing required argument(s) for {name}: {', '.join(missing)}")
    unexpected = sorted(set(args) - set(properties))
    if unexpected:
        return _err(f"unexpected argument(s) for {name}: {', '.join(unexpected)}")
    return tool.handler(**args)
