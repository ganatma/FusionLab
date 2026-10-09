"""The agent's tool registry: every tool must see exactly what the routers see.

The four blueprint verification outcomes live here: filters map 1:1 with /db/search, the absolute-MW
conversion is exact with its arithmetic echoed, the cache boundary is honest on uncached ids, and the
matched-pairs gate's 422 text propagates verbatim. The confirm-then-apply tools must propose without
applying: no export write, no applied edit, ever, from the registry itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fusionlab import agent_tools, mast
from fusionlab.api import app
from fusionlab.api_virtual import EditIn

client = TestClient(app)

BLUEPRINT_TOOLS = {"search_shots", "list_cached_shots", "load_shot", "get_whatif_gate",
                   "run_whatif", "get_simulate", "list_devices", "export_usd"}

# (tool kwargs, the /db/search query params they must mean) — the ranges are the tool's [min, max] shape
FILTER_CASES = [
    ({"ip_MA": [0.8, None]}, {"ip_min": 0.8}),
    ({"bt_T": [0.5, 1.0]}, {"bt_min": 0.5, "bt_max": 1.0}),
    ({"pnbi_MW": [2.0, None]}, {"pnbi_min": 2.0}),
    ({"ne_20": [None, 0.6]}, {"ne_max": 0.6}),
    ({"W_MJ": [0.05, None]}, {"w_min": 0.05}),
    ({"q95": [3.0, None]}, {"q95_min": 3.0}),
    ({"campaign": "M9", "useful": True}, {"campaign": "M9", "useful": True}),
    ({"q": "30400-30500"}, {"q": "30400-30500"}),
    ({"abort": False}, {"abort": False}),
]


# ---- the registry itself
def test_registry_has_the_eight_blueprint_tools():
    assert set(agent_tools.TOOLS) == BLUEPRINT_TOOLS
    assert {n for n, t in agent_tools.TOOLS.items() if t.confirm_then_apply} == {"run_whatif", "export_usd"}
    defs = agent_tools.tool_definitions()
    assert [d["name"] for d in defs] == list(agent_tools.TOOLS)   # stable order for the Messages API
    for d in defs:
        assert d["input_schema"]["type"] == "object" and d["description"]
        assert set(d["input_schema"].get("required", [])) <= set(d["input_schema"]["properties"])


def test_run_whatif_schema_bounds_are_the_enforced_ones():
    """The schema advertises exactly what EditIn enforces — checked against pydantic's own generated
    JSON schema, an independent rendering of the same constraints."""
    props = agent_tools.TOOLS["run_whatif"].definition["input_schema"]["properties"]
    enforced = EditIn.model_json_schema()["properties"]
    for arg, field in (("p_nbi_scale", "P_nbi"), ("n_scale", "n"), ("ip_scale", "Ip"),
                       ("bt_scale", "B"), ("nbi_shift_s", "nbi_shift_s")):
        assert props[arg]["minimum"] == enforced[field]["minimum"]
        assert props[arg]["maximum"] == enforced[field]["maximum"], arg


def test_run_tool_rejects_bad_invocations():
    r = agent_tools.run_tool("nope")
    assert r.error is not None and "allowlisted tools" in r.error and "search_shots" in r.error
    r = agent_tools.run_tool("run_whatif")
    assert r.error is not None and "missing required argument" in r.error and "shot_id" in r.error
    r = agent_tools.run_tool("load_shot", shot_id=30166, sort="desc")   # no sorting, not even by typo
    assert r.error is not None and "unexpected argument" in r.error and "sort" in r.error


# ---- search_shots: the filter vector is /db/search's, 1:1
@pytest.mark.parametrize(("tool_args", "route_params"), FILTER_CASES)
def test_search_shots_maps_filters_1to1_with_db_search(tool_args, route_params):
    tool = agent_tools.run_tool("search_shots", **tool_args)
    route = client.get("/db/search", params=route_params).json()
    assert tool.error is None
    assert tool.data["total"] == route["total"]
    assert [r["shot_id"] for r in tool.data["rows"]] == [r["shot_id"] for r in route["rows"]]


def test_search_shots_caps_limit_at_200():
    tool = agent_tools.run_tool("search_shots", limit=1000)
    route = client.get("/db/search", params={"limit": 200}).json()
    assert tool.error is None and len(tool.data["rows"]) == 200
    assert [r["shot_id"] for r in tool.data["rows"]] == [r["shot_id"] for r in route["rows"]]


def test_search_shots_route_errors_are_verbatim():
    route = client.get("/db/search", params={"q": "nope"})
    assert route.status_code == 422
    tool = agent_tools.run_tool("search_shots", q="nope")
    assert tool.error == route.json()["detail"]


def test_search_shots_empty_match_is_a_result_not_an_error():
    r = agent_tools.run_tool("search_shots", ip_MA=[99.0, None])
    assert r.error is None and r.data["total"] == 0 and r.data["rows"] == []


# ---- load_shot: the cache boundary is honest
def test_load_shot_reports_the_cache_boundary_and_suggests_search():
    route = client.get("/replay/1")
    assert route.status_code == 404
    r = agent_tools.run_tool("load_shot", shot_id=1)
    assert r.error is not None and route.json()["detail"] in r.error   # the route's own 404, verbatim
    assert "search_shots" in r.error
    assert r.data["cached_shot_ids"] == mast.cached_shots()


def test_load_shot_returns_a_compact_summary_without_the_time_arrays():
    r = agent_tools.run_tool("load_shot", shot_id=30166)
    assert r.error is None
    assert r.data["shot_id"] == 30166 and r.data["n_slices"] == mast.load_shot(30166)["t_s"].size
    assert "summary" in r.data and "measured" not in r.data   # traces are the UI's job, not the model's context


# ---- run_whatif: conversion, gate, proposal
def test_run_whatif_converts_absolute_mw_and_echoes_the_math():
    pmax = mast.load_shot(30166)["meta"]["P_nbi_max_MW"]
    mw = round(0.6 * pmax, 4)
    r = agent_tools.run_tool("run_whatif", shot_id=30166, p_nbi_MW=mw)
    assert r.error is None and r.proposal is not None
    assert float(r.proposal["apply"]["sliders"]["P_nbi"]) == pytest.approx(0.6)
    assert r.proposal["args"]["p_nbi_MW"] == pytest.approx(mw)
    assert f"= 0.60 × P_nbi_max {pmax:g} MW" in r.proposal["math"]


def test_run_whatif_proposes_instead_of_applying():
    pmax = mast.load_shot(30166)["meta"]["P_nbi_max_MW"]
    r = agent_tools.run_tool("run_whatif", shot_id=30166, p_nbi_MW=1.2, n_scale=1.1)
    assert r.error is None and r.proposal is not None
    p = r.proposal
    assert p["tool"] == "run_whatif" and p["args"]["shot_id"] == 30166 and p["args"]["p_nbi_MW"] == 1.2
    assert set(p["apply"]["sliders"]) == {"P_nbi", "n"} and "× P_nbi_max" in p["math"]
    assert r.data["edit"]["P_nbi"] == pytest.approx(1.2 / pmax) and r.data["is_identity"] is False
    assert "t_s" not in r.data and "anchored" not in r.data       # bands, not arrays, reach the model
    assert "compared with MAST data" in r.summary                  # house wording
    for bad in ("validated", "predictive", "predict"):             # the forbidden words, per CLAUDE.md
        assert bad not in r.summary.lower()


def test_run_whatif_supports_a_timing_only_edit():
    r = agent_tools.run_tool("run_whatif", shot_id=29823, nbi_shift_s=-0.03)
    assert r.error is None and r.data["timing_edit"] is True
    assert r.proposal is not None and r.proposal["apply"]["sliders"] == {"nbi_shift_s": "-0.030"}


def test_run_whatif_gate_refusal_text_is_verbatim():
    """The shipped validation leaves Ip and B without enough matched pairs; the tool must return the
    route's own 422 sentence, character for character."""
    gate = client.get("/virtual/gate").json()
    assert gate["sliders"]["Ip"]["enabled"] is False   # the committed metrics this test relies on
    route = client.post("/virtual/30420", json={"edit": {"Ip": 1.1}})
    assert route.status_code == 422
    tool = agent_tools.run_tool("run_whatif", shot_id=30420, ip_scale=1.1)
    assert tool.error == route.json()["detail"]
    assert tool.proposal is None   # a refused edit never produces an apply payload


def test_run_whatif_rejects_out_of_bounds_scales_by_name():
    r = agent_tools.run_tool("run_whatif", shot_id=30166, n_scale=5.0)
    assert r.error is not None and "n_scale" in r.error and "1.5" in r.error
    r = agent_tools.run_tool("run_whatif", shot_id=30166, nbi_shift_s=-0.5)
    assert r.error is not None and "nbi_shift_s" in r.error and "-0.1" in r.error


def test_run_whatif_rejects_power_above_the_enforced_scale_with_the_math_shown():
    pmax = mast.load_shot(30166)["meta"]["P_nbi_max_MW"]
    r = agent_tools.run_tool("run_whatif", shot_id=30166, p_nbi_MW=2.0 * pmax + 1.0)
    assert r.error is not None and "outside the enforced range" in r.error and "× P_nbi_max" in r.error


def test_run_whatif_refuses_power_on_a_beamless_shot():
    assert mast.load_shot(30420)["meta"]["P_nbi_max_MW"] in (None, 0)   # the real fixture: no beams logged
    r = agent_tools.run_tool("run_whatif", shot_id=30420, p_nbi_MW=3.0)
    assert r.error is not None and "no NBI beams" in r.error


def test_run_whatif_demands_an_edit():
    r = agent_tools.run_tool("run_whatif", shot_id=30166)
    assert r.error is not None and "no edit given" in r.error


# ---- export_usd: a proposal, never a write
def test_export_usd_proposes_and_executes_nothing(monkeypatch):
    def bomb(*a, **k):
        raise AssertionError("the export ran without confirmation")

    monkeypatch.setattr("fusionlab.fieldlines.export_shot_with_field_lines", bomb)
    monkeypatch.setattr("fusionlab.usd_export.export_shot", bomb)
    r = agent_tools.run_tool("export_usd", shot_id=30166)
    assert r.error is None and r.proposal is not None
    assert r.proposal["args"] == {"shot_id": 30166}
    assert r.proposal["apply"]["download"] == {"method": "POST", "url": "/replay/30166/usd"}


def test_export_usd_uncached_is_the_verbatim_404():
    route = client.post("/replay/1/usd")
    assert route.status_code == 404
    r = agent_tools.run_tool("export_usd", shot_id=1)
    assert r.error is not None and route.json()["detail"] in r.error and r.proposal is None


# ---- the plain read-only trio and the 0D point
def test_read_only_tools_execute():
    r = agent_tools.run_tool("list_cached_shots")
    assert r.error is None and len(r.data["shots"]) == len(mast.cached_shots())
    r = agent_tools.run_tool("get_whatif_gate")
    assert r.error is None and set(r.data["sliders"]) == {"Ip", "B", "n", "P_nbi", "nbi_shift_s"}
    r = agent_tools.run_tool("list_devices")
    assert r.error is None and "iter" in r.data["devices"]


def test_get_simulate_matches_the_route_and_refuses_like_it():
    route = client.get("/simulate", params={"device": "iter"}).json()["result"]
    r = agent_tools.run_tool("get_simulate", device="iter")
    assert r.error is None and r.data["result"]["Q"] == route["Q"]
    for params, code in (({"device": "iter", "n": 1e9}, 422), ({"device": "nope"}, 404)):
        resp = client.get("/simulate", params=params)
        tool = agent_tools.run_tool("get_simulate", **params)
        assert resp.status_code == code
        assert tool.error == resp.json()["detail"]
