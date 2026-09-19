import json
import math

from fastapi.testclient import TestClient

from fusionlab.api import app

client = TestClient(app)


def test_health_and_devices():
    assert client.get("/health").json() == {"ok": True}
    assert set(client.get("/devices").json()) >= {"diiid", "jet", "sparc", "iter"}


def test_simulate_defaults_and_unknown_device():
    r = client.get("/simulate", params={"device": "iter"}).json()["result"]
    assert r["Q"] > 1
    assert client.get("/simulate", params={"device": "nope"}).status_code == 404


def test_index_served():
    assert "FusionLab" in client.get("/").text


def _reject_constant(name):
    raise ValueError(f"non-strict JSON constant {name}")


def test_map_grid_shape_and_strict_json():
    resp = client.get("/map", params={"device": "iter", "nx": 12, "ny": 9})
    assert resp.status_code == 200
    m = json.loads(resp.text, parse_constant=_reject_constant)  # NaN / Infinity would raise here
    assert len(m["n"]) == 12 and len(m["P_aux"]) == 9
    for key in ("Q", "hmode", "disruption", "worst_limit", "stability_limit"):
        assert len(m[key]) == 9 and all(len(row) == 12 for row in m[key]), key
    assert any(q is not None and math.isfinite(q) for row in m["Q"] for q in row)
    assert m["n"][-1] > m["n_GW"] > m["n"][0]           # the density limit is on the map
    assert any(v for row in m["disruption"] for v in row)  # ... and it shows up as a disruption region


def test_map_caps_grid_and_rejects_bad_input():
    m = client.get("/map", params={"device": "diiid", "nx": 5000, "ny": 3}).json()
    assert m["nx"] == 100 and len(m["Q"]) == 3 and len(m["Q"][0]) == 100
    assert client.get("/map", params={"device": "nope"}).status_code == 404
    assert client.get("/map", params={"device": "iter", "Ip": 0}).status_code == 422


def test_map_matches_simulate_at_a_grid_point():
    m = client.get("/map", params={"device": "iter", "nx": 8, "ny": 6}).json()
    i, j = 4, 5  # row = P_aux, column = n
    r = client.get("/simulate", params={"device": "iter", "n": m["n"][j], "P_aux": m["P_aux"][i]}).json()["result"]
    assert math.isclose(m["Q"][i][j], r["Q"], rel_tol=1e-9)
    assert m["hmode"][i][j] == r["hmode"]


def test_static_assets_served():
    for path in ("/static/app.js", "/static/style.css", "/static/replay.js", "/static/vessel3d.js",
                 "/static/vendor/three.module.min.js", "/static/vendor/OrbitControls.js"):
        assert client.get(path).status_code == 200, path
    assert "fusionlab:tab" in client.get("/static/app.js").text
    assert client.get("/static/vendor/plotly.min.js").status_code == 200
    html = client.get("/").text
    assert 'id="tab-replay"' in html and 'id="tab-db"' in html and 'id="tab-sandbox"' in html
    assert "cdn.plot.ly" not in html and "cdn.jsdelivr" not in html and "unpkg" not in html   # the demo never needs the network


# ---- replay of real MAST shots (reads the committed cache, no network)
def test_shots_lists_cached_shots_with_logbook():
    j = client.get("/shots").json()
    ids = [s["shot_id"] for s in j["shots"]]
    assert 30420 in ids and "CC BY-SA" in j["attribution"]
    assert "flux" in next(s for s in j["shots"] if s["shot_id"] == 30420)["postshot"]


def test_replay_returns_measured_and_model_on_one_time_base():
    r = client.get("/replay/30420")
    assert "NaN" not in r.text
    j = r.json()
    n = len(j["measured"]["t_s"])
    assert n > 20 and len(j["model"]["W_H_MJ"]) == n and len(j["lcfs"]["R"]) == n
    assert j["summary"]["H89_median"] is not None and j["limit_names"] == ["greenwald", "troyon", "kink"]
    assert client.get("/replay/1").status_code == 404


def test_replay_psi_slice_matches_the_efit_grid():
    j = client.get("/replay/30420").json()
    p = client.get("/replay/30420/psi/10").json()
    assert len(p["psi_n"]) == len(j["psi_grid"]["Z"]) and len(p["psi_n"][0]) == len(j["psi_grid"]["R"])
    assert client.get("/replay/30420/psi/9999").status_code == 404


def test_db_operating_space():
    j = client.get("/db").json()
    assert j["n"] > 1000 and len(j["columns"]["H98"]) == j["n"] and "NaN" not in client.get("/db").text


def test_surrogate_metrics_and_hybrid_trace_are_served_together():
    r = client.get("/surrogate")
    if r.status_code == 404:   # no trained model in this checkout: the replay must still work without it
        assert "W_hybrid_MJ" not in client.get("/replay/30166").json()["model"]
        return
    assert "split_temporal_M9_held_out" in r.json()
    j = client.get("/replay/30166").json()
    assert len(j["model"]["W_hybrid_MJ"]) == len(j["measured"]["t_s"]) and "rmse_ln_tau_hybrid" in j["summary"]


def test_usd_download_is_a_usd_file():
    r = client.get("/replay/30420/usd")
    assert r.status_code == 200 and r.content[:8] == b"PXR-USDC" and len(r.content) > 100_000
    assert client.get("/replay/30420/usd", params={"fmt": "exe"}).status_code == 422


def test_fieldlines_slice_has_traced_q_next_to_efit():
    r = client.get("/replay/30420/fieldlines/20")
    if r.status_code == 503:   # Warp cannot run on this machine: the endpoint must say so, not crash
        return
    j = r.json()
    assert len(j["lines"]) == len(j["psi_n_start"]) and len(j["lines"][0]["x"]) > 50 and "NaN" not in r.text
    assert abs(j["q95_traced"] / j["q95_efit"] - 1) < 0.02


def test_equilibrium_surrogate_overlay_is_labelled_held_out_or_not():
    j = client.get("/replay/27257").json()
    if "eq_surrogate" not in j["summary"]:   # no trained equilibrium model in this checkout
        assert "surrogate" not in client.get("/replay/27257/psi/30").json()
        return
    assert j["summary"]["eq_surrogate"]["held_out"] is True and j["summary"]["eq_surrogate"]["median_rel_l2"] < 0.05
    assert client.get("/replay/30166").json()["summary"]["eq_surrogate"]["held_out"] is False
    p = client.get("/replay/27257/psi/30").json()
    assert len(p["surrogate"]["psi_n"]) == len(p["psi_n"]) and 0 < p["surrogate"]["rel_l2"] < 0.2
