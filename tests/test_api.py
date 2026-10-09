import json
import logging
import math
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fusionlab import api_replay, mast
from fusionlab.api import NoDotfiles, app

client = TestClient(app)
WEB = Path(__file__).resolve().parents[1] / "web"

# Every spelling a query string can carry for a non-finite float; 1e309 parses to inf.
NON_FINITE = ("NaN", "Infinity", "-Infinity", "1e309")


def test_health_and_devices():
    assert client.get("/health").json() == {"ok": True}
    assert set(client.get("/devices").json()) >= {"diiid", "jet", "sparc", "iter"}


def test_simulate_defaults_and_unknown_device():
    r = client.get("/simulate", params={"device": "iter"}).json()["result"]
    assert r["Q"] > 1
    assert client.get("/simulate", params={"device": "nope"}).status_code == 404


def test_simulate_rejects_overflow_scale_inputs():
    """Extreme finite floats once overflowed the engine into inf and rendered as an unhandled 500."""
    for params in ({"device": "iter", "Ip": 1e308, "H": 1e308},   # overflow-scale: the reproduced 500
                   {"device": "iter", "n": 1e-300}):              # degenerate: positivity alone would miss it
        assert client.get("/simulate", params=params).status_code == 422


def test_simulate_every_device_default_is_strictly_finite_json():
    for device in client.get("/devices").json():
        resp = client.get("/simulate", params={"device": device})
        assert resp.status_code == 200, device
        json.loads(resp.text, parse_constant=_reject_constant)  # NaN / Infinity would raise here


def test_simulate_keeps_the_ui_operating_range():
    """Slider corners from web/app.js (0.1-1x Ip_max, 0.2-1x B_max, 0-1x P_aux_max, 0.05-1.3x n_GW,
    H 0.5-2, Zeff 1-4) must still be accepted: the bounds exclude overflow-scale inputs, never
    real operating points."""
    for name, d in client.get("/devices").json().items():
        n_gw = 0.1 * d["Ip_max"] / (math.pi * d["a"] ** 2)  # n slider's low end sits at the Ip floor
        params = {"Ip": 0.1 * d["Ip_max"], "B": 0.2 * d["B_max"], "n": 0.05 * n_gw,
                  "P_aux": 0, "H": 0.5, "Zeff": 1.0}
        resp = client.get("/simulate", params=params)
        assert resp.status_code == 200, (name, params)


def test_simulate_non_finite_result_is_422_not_500(monkeypatch):
    """Second layer: a non-finite result must surface as 422, never as the JSON encoder's 500."""
    monkeypatch.setattr("fusionlab.api.simulate", lambda c: {"Q": float("inf"), "hmode": True})
    assert client.get("/simulate", params={"device": "iter"}).status_code == 422


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize("param", ("Ip", "B", "n", "P_aux", "H", "Zeff"))
def test_simulate_rejects_non_finite_param(param, value):
    """Any non-finite control must 422 from the range guard, never a 5xx (API-01 probed matrix)."""
    r = client.get("/simulate", params={"device": "iter", param: value})
    assert r.status_code == 422
    assert r.json()["detail"]


@pytest.mark.parametrize(
    "params",
    ({"H": -2}, {"Zeff": 0}, {"P_aux": -100}, {"n": -5}, {"Ip": "1e300"}),
)
def test_simulate_rejects_out_of_domain_params(params):
    """Domain parity: negative H/P_aux/n, Zeff < 1, and overflow-scale Ip are all refused."""
    assert client.get("/simulate", params={"device": "iter", **params}).status_code == 422


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


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize("param", ("Ip", "B", "H", "Zeff"))
def test_map_rejects_non_finite_param(param, value):
    """inf once passed the positivity guard and crashed the strict JSON encoder at render (500)."""
    r = client.get("/map", params={"device": "iter", param: value})
    assert r.status_code == 422
    assert r.json()["detail"]


@pytest.mark.parametrize("params", ({"H": -2}, {"Zeff": 0}, {"Ip": -1}, {"B": -1}))
def test_map_rejects_out_of_domain_params(params):
    assert client.get("/map", params={"device": "iter", **params}).status_code == 422


def test_map_finite_extreme_never_5xx():
    """A finite extreme overflows grid cells to inf, which _grid_json nulls: 200 with strict JSON —
    the /map counterpart of /simulate's output-finiteness net."""
    resp = client.get("/map", params={"device": "iter", "Ip": "1e300"})
    assert resp.status_code == 200
    json.loads(resp.text, parse_constant=_reject_constant)  # no NaN/Infinity literal in the body


def test_nodotfiles_lookup_path():
    """Unit: a dotted component anywhere in the path reports Starlette's "not found" shape;
    a normal asset still resolves."""
    static = NoDotfiles(directory=WEB)
    for dotted in (".env", "../.env", "vendor/.hidden/x"):
        assert static.lookup_path(dotted) == ("", None), dotted
    resolved, stat = static.lookup_path("app.js")
    assert Path(resolved).name == "app.js" and stat is not None


def test_static_dotfile_paths_404_not_403():
    """A dotfile — plain, or reached via an encoded traversal — 404s (the house rule forbids
    403 existence oracles), while normal assets keep serving."""
    for path in ("/static/.env", "/static/%2e%2e/.env"):
        assert client.get(path).status_code == 404, path
    assert client.get("/static/app.js").status_code == 200


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
    # the demo never needs the network: no CDN in the page or in any script of ours (vendored libraries aside)
    ours = [html] + [p.read_text() for p in sorted(WEB.glob("*.js")) + sorted(WEB.glob("*.json"))]
    for text in ours:
        assert not any(cdn in text for cdn in ("cdn.plot.ly", "cdn.jsdelivr", "unpkg.com", "cdnjs."))


def test_reactivity_curve_for_the_first_lesson():
    j = client.get("/reactivity", params={"n": 40}).json()
    T, sv = j["T_keV"], j["sigmav_m3_s"]
    assert len(T) == len(sv) == 40 and T[0] == 1.0 and abs(T[-1] - 100.0) < 1e-9
    assert 50 < T[sv.index(max(sv))] < 80 and sv[0] < 1e-3 * max(sv)   # peaks near 65 keV, negligible at 1 keV


def test_guided_study_hooks_are_in_place():
    """The guided study drives the app through these and nothing else (web/guide.js)."""
    app, replay, vessel = (client.get(f"/static/{f}").text for f in ("app.js", "replay.js", "vessel3d.js"))
    assert "fusionlab:sim" in app and "window.FusionLab" in app
    assert "fusionlab:shot" in replay and "fusionlab:slice" in replay and "loadSeq" in replay
    assert "highlight" in vessel.split("window.Vessel3D =")[1]


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


# ---- multi-shot compare (A2), criterion 4: any shot id is replayable. An uncached catalog shot rides
# mast.load_shot()'s existing cache-then-S3 path; the archive is mocked here — no test may touch the network.


def _uncached_catalog_id():
    """A shot the catalog lists but the repo cache does not ship."""
    cached = set(mast.cached_shots())
    ids = sorted(int(i) for i in mast.load_db()["shot_id"])
    return next(i for i in ids if i > 30000 and i not in cached)


def _fake_payload(real, shot_id):
    return {**real, "meta": {**real["meta"], "shot_id": int(shot_id)}}


def test_uncached_catalog_shot_is_fetched_and_cached(monkeypatch, tmp_path):
    """The compare fan-out can open any catalog shot: one archive fetch, then the local cache serves repeats."""
    uncached = _uncached_catalog_id()
    real = mast.load_shot(30420)
    calls = []

    def fake_fetch(shot_id):
        calls.append(int(shot_id))
        return _fake_payload(real, shot_id)

    monkeypatch.setattr(mast, "SHOTS", tmp_path)
    monkeypatch.setattr(mast, "fetch_shot", fake_fetch)
    r = client.get(f"/replay/{uncached}")
    assert r.status_code == 200
    j = r.json()
    assert calls == [uncached]                                    # exactly one archive fetch
    assert uncached in mast.cached_shots() and (tmp_path / f"{uncached}.npz").exists()
    assert j["meta"]["shot_id"] == uncached and "measured" in j and "model" in j and "psi_grid" in j
    assert j["limit_names"] == ["greenwald", "troyon", "kink"]    # fetched shots carry identical provenance structure
    assert client.get(f"/replay/{uncached}/psi/10").status_code == 200   # per-slice routes ride the same path
    assert client.get(f"/replay/{uncached}").status_code == 200 and calls == [uncached]   # second request: cache hit


def test_concurrent_requests_share_one_fetch(monkeypatch, tmp_path):
    """Two simultaneous requests for the same uncached id must not stampede the archive."""
    import threading
    import time as _time

    uncached = _uncached_catalog_id()
    real = mast.load_shot(30420)
    calls = []

    def slow_fetch(shot_id):
        calls.append(int(shot_id))
        _time.sleep(0.2)
        return _fake_payload(real, shot_id)

    monkeypatch.setattr(mast, "SHOTS", tmp_path)
    monkeypatch.setattr(mast, "fetch_shot", slow_fetch)
    out = []

    def hit():
        out.append(api_replay.replay_shot(uncached)["meta"]["shot_id"])

    threads = [threading.Thread(target=hit) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == [uncached, uncached, uncached]
    assert calls == [uncached]   # one archive fetch, three answers


def test_uncached_shot_not_in_catalog_is_404(monkeypatch, tmp_path):
    """A shot outside the catalog must not reach the archive."""
    monkeypatch.setattr(mast, "SHOTS", tmp_path)
    monkeypatch.setattr(mast, "fetch_shot", lambda sid: (_ for _ in ()).throw(AssertionError("must not fetch")))
    assert client.get("/replay/99999999").status_code == 404


def test_failed_archive_fetch_is_502_and_leaves_no_cache(monkeypatch, tmp_path):
    """When the archive is unreachable the client hears 502, and no partial cache file is left behind."""
    uncached = _uncached_catalog_id()

    def boom(shot_id):
        raise RuntimeError("S3 unreachable")

    monkeypatch.setattr(mast, "SHOTS", tmp_path)
    monkeypatch.setattr(mast, "fetch_shot", boom)
    r = client.get(f"/replay/{uncached}")
    assert r.status_code == 502 and "fetch" in r.json()["detail"].lower()
    assert uncached not in mast.cached_shots() and not (tmp_path / f"{uncached}.npz").exists()


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


def test_usd_export_requires_post():
    """Audit F1: the export recomputes the stage and writes out/ on every hit — a GET must not reach it."""
    assert client.get("/replay/30420/usd").status_code == 405
    assert client.post("/replay/1/usd").status_code == 404   # unknown shot: 404 from the cache allowlist, never 403
    assert client.post("/replay/30420/usd", params={"fmt": "exe"}).status_code == 422


def test_usd_download_is_a_usd_file():
    r = client.post("/replay/30420/usd")
    assert r.status_code == 200 and r.content[:8] == b"PXR-USDC" and len(r.content) > 100_000


def test_usd_download_exports_to_temp_then_publishes_atomically(monkeypatch):
    """Overlapping downloads used to share one in-place output path and readers mid-write got truncated
    stages: both export phases must land on a same-directory temp, published with one os.replace."""
    out = Path(api_replay.__file__).resolve().parent.parent / "out" / "mast_30420.usda"
    out.parent.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)   # the assertion is that the endpoint creates the served path
    seen = {}

    def fake_with_lines(shot, path):
        seen["tmp"] = Path(path)
        seen["final_exists_mid_export"] = out.exists()
        Path(path).write_bytes(b"stage-bytes")

    def plain_must_not_run(shot, path):
        raise AssertionError("plain export ran although the field-line export succeeded")

    monkeypatch.setattr("fusionlab.fieldlines.export_shot_with_field_lines", fake_with_lines)
    monkeypatch.setattr("fusionlab.usd_export.export_shot", plain_must_not_run)
    r = client.post("/replay/30420/usd", params={"fmt": "usda"})
    assert r.status_code == 200 and r.content == b"stage-bytes"
    assert seen["tmp"] != out and seen["tmp"].parent == out.parent   # temp, same dir -> atomic replace
    assert seen["final_exists_mid_export"] is False                  # the served path exists only after publish
    assert out.exists() and not seen["tmp"].exists()
    assert not list(out.parent.glob(f".{out.stem}.tmp*"))            # no temp litter on the success path


def test_usd_download_failure_keeps_the_old_stage_and_leaves_no_temp(monkeypatch):
    """A download that fails mid-write must not damage the previous stage nor leave a temp behind."""
    out = Path(api_replay.__file__).resolve().parent.parent / "out" / "mast_30420.usda"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(b"previous-good-stage")

    def dies_mid_write(shot, path):
        Path(path).write_bytes(b"half-written")
        raise RuntimeError("export exploded mid-write")

    monkeypatch.setattr("fusionlab.fieldlines.export_shot_with_field_lines", dies_mid_write)
    monkeypatch.setattr("fusionlab.usd_export.export_shot", dies_mid_write)
    with pytest.raises(RuntimeError, match="export exploded"):
        client.post("/replay/30420/usd", params={"fmt": "usda"})   # a total failure still surfaces as a 500
    assert out.read_bytes() == b"previous-good-stage"             # the old artifact is untouched
    assert not list(out.parent.glob(f".{out.stem}.tmp*"))         # temp unlinked on failure


def test_usd_plain_stage_fallback_logs_a_warning(caplog, monkeypatch):
    """The plain-stage fallback used to be silent; degradation must be visible, traceback included."""

    def broken(shot, path):
        raise RuntimeError("no warp device here")

    monkeypatch.setattr("fusionlab.fieldlines.export_shot_with_field_lines", broken)
    monkeypatch.setattr("fusionlab.usd_export.export_shot", lambda shot, path: Path(path).write_bytes(b"plain-stage"))
    with caplog.at_level(logging.WARNING, logger="fusionlab.api_replay"):
        r = client.post("/replay/30420/usd", params={"fmt": "usda"})
    assert r.status_code == 200 and r.content == b"plain-stage"
    warns = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(warns) >= 1 and warns[0].exc_info is not None


def test_eq_surrogate_fallback_logs_a_warning(caplog, monkeypatch):
    """The eq-surrogate overlay's fail-open path used to swallow everything silently."""

    def broken_model():
        raise RuntimeError("torch exploded")

    monkeypatch.setattr(api_replay, "_eq_model", broken_model)
    api_replay._eq_psi.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger="fusionlab.api_replay"):
            assert api_replay._eq_psi(30420) is None   # the replay itself carries on
        warns = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
        assert len(warns) >= 1 and warns[0].exc_info is not None
    finally:
        api_replay._eq_psi.cache_clear()   # do not leave the fallback result cached for later tests


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


# ---- catalog search (A1): /db/search against direct numpy masks over the same table
DB_SEARCH_FIELDS = {"shot_id", "campaign", "Ip_MA", "B_T", "n_e20", "P_nbi_MW", "P_ohm_MW", "P_rad_MW", "W_MJ",
                    "Te0_keV", "q95", "kappa", "beta_t_pct", "a_m", "R_m", "V_m3", "tau_E_s", "dWdt_MW",
                    "t_ipmax_s", "useful", "abort"}


def test_db_search_rows_match_direct_numpy_masks():
    """Criterion 1: for every filter field the endpoint's total and ids equal a direct numpy mask over the table."""
    db = mast.load_db()
    cases = [
        ({"q": "30421"}, db["shot_id"] == 30421),
        ({"q": "30400-30500"}, (db["shot_id"] >= 30400) & (db["shot_id"] <= 30500)),
        ({"campaign": "M9"}, db["campaign"] == 9),
        ({"campaign": "5"}, db["campaign"] == 5),
        ({"ip_min": 0.8}, db["Ip_MA"] >= 0.8),
        ({"ip_max": 0.6}, db["Ip_MA"] <= 0.6),
        ({"bt_min": 0.5, "bt_max": 1.0}, (db["B_T"] >= 0.5) & (db["B_T"] <= 1.0)),
        ({"pnbi_min": 2.0}, db["P_nbi_MW"] >= 2.0),
        ({"pnbi_max": 0.0}, db["P_nbi_MW"] <= 0.0),            # ohmic shots: the beams were never on
        ({"ne_min": 0.5, "ne_max": 1.0}, (db["n_e20"] >= 0.5) & (db["n_e20"] <= 1.0)),
        ({"w_min": 0.05}, db["W_MJ"] >= 0.05),
        ({"w_max": 0.1}, db["W_MJ"] <= 0.1),
        ({"q95_min": 3.0}, db["q95"] >= 3.0),
        ({"q95_max": 5.0}, db["q95"] <= 5.0),
        ({"useful": True}, db["useful"] == 1),
        ({"useful": False}, db["useful"] != 1),                # shots the archive never marked useful
        ({"abort": True}, db["abort"] == 1),
        ({"abort": False}, db["abort"] != 1),
        ({"ip_min": 0.8, "campaign": "M9", "useful": True},    # filters AND together
         (db["Ip_MA"] >= 0.8) & (db["campaign"] == 9) & (db["useful"] == 1)),
    ]
    for params, mask in cases:
        j = client.get("/db/search", params=params).json()
        assert j["total"] == int(mask.sum()), params
        # the default page (limit 200) is the head of the ascending ids the mask selects
        assert [r["shot_id"] for r in j["rows"]] == sorted(db["shot_id"][mask].tolist())[:200], params


def test_db_search_rows_carry_every_metadata_field_as_strict_json():
    r = client.get("/db/search", params={"limit": 5})
    assert "NaN" not in r.text and "Infinity" not in r.text   # missing fields surface as null, never NaN
    j = r.json()
    assert set(j["rows"][0]) == DB_SEARCH_FIELDS
    assert j["total"] == 15_969 and j["limit"] == 5 and j["offset"] == 0
    assert {row["campaign"] for row in j["rows"]} <= {5, 6, 7, 8, 9}


def test_db_search_pagination_windows():
    """Criterion 2: windows slice the ascending match set; total never moves; overruns are 422, not 500."""
    all_rows = client.get("/db/search", params={"limit": 1000}).json()
    total, ids = all_rows["total"], [r["shot_id"] for r in all_rows["rows"]]
    p1 = client.get("/db/search", params={"limit": 10}).json()
    p2 = client.get("/db/search", params={"limit": 10, "offset": 10}).json()
    assert p1["total"] == p2["total"] == total
    assert [r["shot_id"] for r in p1["rows"]] == ids[:10]
    assert [r["shot_id"] for r in p2["rows"]] == ids[10:20]
    assert not {r["shot_id"] for r in p1["rows"]} & {r["shot_id"] for r in p2["rows"]}
    last = client.get("/db/search", params={"offset": total - 1}).json()
    assert [r["shot_id"] for r in last["rows"]] == [int(mast.load_db()["shot_id"].max())]
    assert client.get("/db/search", params={"offset": total}).status_code == 422
    assert client.get("/db/search", params={"offset": total + 500}).status_code == 422


def test_db_search_rejects_malformed_params():
    for params in ({"offset": -1}, {"limit": 0}, {"limit": 1001},                        # paging bounds
                   {"ip_min": "abc"}, {"w_max": "lots"},                                 # malformed numbers
                   {"q": "30400-30500-99999"}, {"q": "nope"}, {"q": "30500-30400"},      # malformed / inverted q
                   {"campaign": "MX"}, {"useful": "maybe"}):                             # malformed campaign / bool
        assert client.get("/db/search", params=params).status_code == 422, params


def test_db_search_unfiltered_is_fast():
    """Criterion 3: the full-catalog search is a vectorized mask, not per-row Python (catalog cached)."""
    client.get("/db/search")   # warm the one-time npz read: the budget covers the route itself
    t0 = time.perf_counter()
    j = client.get("/db/search").json()
    assert time.perf_counter() - t0 < 0.5
    assert j["total"] == 15_969 and len(j["rows"]) == 200


def _no_nan_floats(node) -> bool:
    """Strict JSON everywhere: no NaN/inf survived to the encoder (mirrors the _j null convention)."""
    if isinstance(node, float):
        return math.isfinite(node)
    if isinstance(node, dict):
        return all(_no_nan_floats(v) for v in node.values())
    if isinstance(node, list):
        return all(_no_nan_floats(v) for v in node)
    return True


def test_replay_answers_null_not_500_when_the_model_has_nothing_to_say(monkeypatch):
    """A shot without line-average density (some M9 catalog rows) replays an all-NaN model. The summary
    must answer null medians — the strict-JSON encoder turns a single NaN into a 500 (observed on 28788)."""
    real = api_replay.replay
    import numpy as _np

    def nan_model(shot, H=1.0):
        r = real(shot, H=H)
        return {k: _np.full_like(v, _np.nan) if isinstance(v, _np.ndarray) and v.dtype.kind == "f" else v
                for k, v in r.items()}

    monkeypatch.setattr(api_replay, "replay", nan_model)
    r = client.get("/replay/30420")
    assert r.status_code == 200
    j = r.json()
    assert j["summary"]["P_LH_median_MW"] is None
    assert j["summary"]["H98_median"] is None and j["summary"]["H89_median"] is None
    assert _no_nan_floats(j)


def test_replay_of_a_catalog_shot_without_published_data_answers_404(monkeypatch):
    """Some catalog rows have no level-2 Zarr in FAIR-MAST (observed: 30004). The fetch path answers a
    404 that names the gap — not a 502 that blames the archive, and never a half-written cache."""
    from zarr.errors import GroupNotFoundError
    uncached = next(int(i) for i in mast.load_db()["shot_id"] if int(i) not in mast.cached_shots())
    monkeypatch.setattr(mast, "load_shot", lambda sid: (_ for _ in ()).throw(GroupNotFoundError(sid)))
    r = client.get(f"/replay/{uncached}")
    assert r.status_code == 404
    assert "not published" in r.json()["detail"]
    assert uncached not in mast.cached_shots()
