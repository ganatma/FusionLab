import csv
import io
import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fusionlab import api_replay, mast, virtual
from fusionlab.api import NoDotfiles, app
from fusionlab.physics import BETA_N_LIMIT

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


# ---- limit-distance panel (FEAT-3): measured limits over time, the archive's abort flag as context

def _arr(xs: list) -> np.ndarray:
    """A bundle's JSON-safe series (None for NaN) back to a float array."""
    return np.array([np.nan if v is None else v for v in xs], dtype=float)


def test_limit_distances_match_the_map_limit_formulas():
    """FEAT-3: the panel's per-slice distances are /map's limit formulas evaluated on the shot's own
    measured geometry — api.operating_map's f_gw = n/n_gw with n_gw = Ip/(pi a^2), troyon =
    beta_N/BETA_N_LIMIT, kink = 2/q95 — and worst_limit is their nan-propagating max (/map's
    stability_limit relation). physics.replay computes the same stack by construction; the test pins
    it against drift. Bundle series are rounded to 4 dp, so compare at rounding tolerance."""
    for sid in mast.cached_shots():
        b = client.get(f"/replay/{sid}").json()
        me, mo = b["measured"], b["model"]
        n, ip, a = (_arr(me[k]) for k in ("n_e20", "Ip_MA", "a_m"))
        want = {"f_greenwald": n / (ip / (np.pi * a**2)),
                "troyon": _arr(me["beta_N"]) / BETA_N_LIMIT,
                "kink": 2.0 / _arr(me["q95"])}
        for name, w in want.items():
            assert np.allclose(_arr(mo[name]), w, rtol=1e-3, atol=1e-3, equal_nan=True), (sid, name)
        worst = np.max(np.stack(list(want.values())), axis=0)
        assert np.allclose(_arr(mo["worst_limit"]), worst, rtol=1e-3, atol=1e-3, equal_nan=True), sid


def test_archive_flags_align_with_the_catalog():
    """FEAT-3: the bundle's archive block is the catalog's own cpf_useful/cpf_abort row — None where
    the archive never graded the shot (NaN), so absent flags degrade silently downstream."""
    c = api_replay._catalog()
    for sid in mast.cached_shots():
        i = int(np.nonzero(c["shot_id"] == sid)[0][0])
        def row_flag(col, i=i):
            v = float(c[col][i])
            return None if not np.isfinite(v) else bool(v)
        b = client.get(f"/replay/{sid}").json()
        assert b["archive"] == {"useful": row_flag("useful"), "abort": row_flag("abort")}, sid


def test_abort_marker_aligns_with_a_flagged_catalog_row(monkeypatch):
    """Where the archive DOES flag a shot, the bundle carries abort=True and the UI marker aligns
    with the catalog by construction. No cached shot ships flagged, so the alignment is proven on a
    synthetic catalog row (the shipped cache exercises only the silent-degradation path)."""
    c = dict(api_replay._catalog())
    i = int(np.nonzero(c["shot_id"] == 30166)[0][0])
    c["abort"] = c["abort"].copy()
    c["abort"][i] = 1.0
    monkeypatch.setattr(api_replay, "_catalog", lambda: c)
    b = client.get("/replay/30166").json()
    assert b["archive"] == {"useful": True, "abort": True}
    assert client.get("/replay/30420").json()["archive"]["abort"] is False   # unflagged rows stay False


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


# ---- bundles without equilibrium (DATA-1): /replay used to emit lcfs/psi_grid unconditionally,
# so a machine without reconstructed equilibria broke the route outright. Now it degrades.


def _no_equilibrium_payload(real: dict) -> dict:
    """A real cached bundle stripped of every equilibrium/tracing array — a ψ-free machine's shot."""
    eq_keys = ("lcfs_R", "lcfs_Z", "psi_R", "psi_Z", "psi_n", "R_mag_m", "Z_mag_m",
               "F_Tm", "psi_axis_Wb", "psi_bnd_Wb", "psi_norm", "q_mid", "q_mid_R")
    return {k: v for k, v in real.items() if k not in eq_keys}


def test_replay_of_a_psi_free_bundle_answers_200_with_a_hint(monkeypatch, tmp_path):
    """A bundle without equilibrium replays its traces: 200, lcfs/psi_grid null, and a
    human-readable `equilibrium_absent` hint — previously a 500-class breakage."""
    uncached = _uncached_catalog_id()
    real = mast.load_shot(30420)
    stripped = _no_equilibrium_payload(real)
    assert "psi_n" not in stripped and "lcfs_R" not in stripped
    monkeypatch.setattr(mast, "SHOTS", tmp_path)
    monkeypatch.setattr(mast, "fetch_shot", lambda sid: _fake_payload(stripped, sid))
    api_replay._shot_cached.cache_clear()   # the process cache is keyed by id alone: earlier
    # tests in this module replayed a full bundle for the same id — do not serve it stale here.

    r = client.get(f"/replay/{uncached}")
    assert r.status_code == 200
    j = r.json()
    assert j["lcfs"] is None and j["psi_grid"] is None
    assert "psi" in j["equilibrium_absent"]
    assert j["measured"]["t_s"] and j["measured"]["Ip_MA"]        # the traces still replay
    assert j["model"] is not None and j["summary"]                # the 0D twin still answers
    assert j["wall"] is not None                                  # vessel geometry is independent
    assert _no_nan_floats(j)                                      # strict JSON: nulls, never NaN


def test_equilibrium_dependent_routes_on_a_psi_free_bundle_name_the_gap(monkeypatch, tmp_path):
    """The per-slice surfaces answer their own honest errors instead of KeyError: ψ 404,
    field-lines 503 with the reason, USD export 422 (no LCFS to revolve)."""
    uncached = _uncached_catalog_id()
    real = mast.load_shot(30420)
    monkeypatch.setattr(mast, "SHOTS", tmp_path)
    monkeypatch.setattr(mast, "fetch_shot", lambda sid: _fake_payload(_no_equilibrium_payload(real), sid))

    assert client.get(f"/replay/{uncached}").status_code == 200       # prime the cache
    r_psi = client.get(f"/replay/{uncached}/psi/0")
    assert r_psi.status_code == 404 and "psi" in r_psi.json()["detail"]
    r_fl = client.get(f"/replay/{uncached}/fieldlines/0")
    assert r_fl.status_code == 503 and "psi" in r_fl.json()["detail"]
    r_usd = client.post(f"/replay/{uncached}/usd?fmt=usda")
    assert r_usd.status_code == 422 and "equilibrium" in r_usd.json()["detail"]


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
    """Criterion 1: for every filter field the endpoint's total and ids equal a direct numpy mask over the table.
    Campaign labels and row totals are derived from the loaded table + the configured campaign
    pattern (formerly the hard-coded "M9" and 15,969 pins) so a second machine's catalog reads the same."""
    db = mast.load_db()
    camps = np.unique(db["campaign"])
    c_first, c_last = int(camps[0]), int(camps[-1])
    prefix = re.match(r"[A-Za-z]*", mast.CONFIG.campaign_pattern).group(0)   # "M\d+" -> "M"; r"\d+" -> ""
    label = f"{prefix}{c_last}"
    cases = [
        ({"q": "30421"}, db["shot_id"] == 30421),
        ({"q": "30400-30500"}, (db["shot_id"] >= 30400) & (db["shot_id"] <= 30500)),
        ({"campaign": label}, db["campaign"] == c_last),                   # archive label form ("M9")
        ({"campaign": str(c_first)}, db["campaign"] == c_first),           # bare-number form
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
        ({"ip_min": 0.8, "campaign": label, "useful": True},    # filters AND together
         (db["Ip_MA"] >= 0.8) & (db["campaign"] == c_last) & (db["useful"] == 1)),
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
    assert j["total"] == int(mast.load_db()["shot_id"].size) and j["limit"] == 5 and j["offset"] == 0
    assert {row["campaign"] for row in j["rows"]} <= set(int(c) for c in np.unique(mast.load_db()["campaign"]))


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
    assert j["total"] == int(mast.load_db()["shot_id"].size) and len(j["rows"]) == 200


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

    def nan_model(shot, H=1.0, **_):   # the route forwards the source's MachineConfig as config=
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


# ---- take-away export (FEAT-1): the replay bundle's numbers, downloadable with their provenance.
# The contract is equality: the export serves the very dict /replay serves, never a recomputation.


def _cell(v):
    """The CSV cell the csv module writes for a JSON bundle value: None empty, everything else str()."""
    return "" if v is None else str(v)


def test_export_json_equals_the_replay_bundle():
    replay = client.get("/replay/30420").json()
    j = client.get("/export/30420", params={"fmt": "json"}).json()
    assert len(j["shots"]) == 1
    b = j["shots"][0]
    assert b["meta"] == replay["meta"] and b["attribution"] == replay["attribution"]
    assert b["measured"] == replay["measured"] and b["model"] == replay["model"]
    assert b["summary"] == replay["summary"]


def test_export_csv_equals_the_replay_bundle_and_carries_provenance():
    replay = client.get("/replay/30420").json()
    rows = list(csv.DictReader(io.StringIO(client.get("/export/30420").text)))
    series = {**replay["measured"], **{k: v for k, v in replay["model"].items() if k != "t_s"}}
    by_q: dict[str, list[str]] = {}
    for row in rows:
        by_q.setdefault(row["quantity"], []).append(row["value"])
    summary_names = {n for n, v in api_replay._flat_summary(replay["summary"]) if isinstance(v, (bool, int, float))}
    assert set(by_q) == set(series) | summary_names
    for name, vals in series.items():
        assert by_q[name] == [_cell(v) for v in vals]   # exact strings: the bundle's own numbers, str()-ed
    flat = dict(api_replay._flat_summary(replay["summary"]))
    for row in rows:
        if row["t_s"] == "":   # a summary row: empty t_s, the summary's own value
            assert row["value"] == _cell(flat[row["quantity"]])
    prov = {row["quantity"]: row["provenance"] for row in rows}
    assert prov["Ip_MA"] == "measured" and prov["W_H_MJ"] == "model"
    assert prov["H_learned"] == "learned" and prov["H98"] == "computed"
    assert set(prov.values()) <= {"measured", "model", "learned", "computed"}


def test_export_provenance_covers_every_series_with_all_four_labels():
    b = client.get("/export/30420", params={"fmt": "json"}).json()["shots"][0]
    prov = b["export"]["provenance"]
    series = set(b["measured"]) | {k for k in b["model"] if k != "t_s"} \
        | {n for n, _ in api_replay._flat_summary(b["summary"])}
    assert set(prov) == series   # every exported series labelled — a new replay() key fails here until labelled
    assert set(prov.values()) == {"measured", "model", "learned", "computed"}


def test_export_csv_downloads_as_an_attachment():
    r = client.get("/export/30420")
    assert r.headers["content-type"].startswith("text/csv")
    assert 'filename="fusionlab_shot_30420.csv"' in r.headers["content-disposition"]
    assert set(csv.DictReader(io.StringIO(r.text)).fieldnames) == set(api_replay._EXPORT_COLUMNS)


def test_compare_export_groups_rows_by_shot_and_validates():
    j = client.get("/export", params={"shots": "30420,30166,30420", "fmt": "json"}).json()
    assert [s["meta"]["shot_id"] for s in j["shots"]] == [30420, 30166]   # deduped, request order kept
    rows = list(csv.DictReader(io.StringIO(client.get("/export", params={"shots": "30420,30166"}).text)))
    assert {r["shot_id"] for r in rows} == {"30420", "30166"}
    assert client.get("/export", params={"shots": "30420,nope"}).status_code == 422
    assert client.get("/export", params={"shots": "1,2,3,4,5,6,7"}).status_code == 422   # cap: 7 distinct ids
    assert client.get("/export", params={"shots": ",".join(["30166"] * 7)}).status_code == 200   # dupes dedupe first
    assert client.get("/export/1").status_code == 404   # not in the catalog
    assert client.get("/export/30420", params={"fmt": "parquet"}).status_code == 422


# ---- domain shading on the measured replay (FEAT-2): the bundle carries the virtual path's own guard
# and the band coordinates are exactly the flags' out-of-domain runs — one shared source, no copied numbers.


def test_replay_domain_flags_are_the_virtual_guards_own_output():
    """The replay's in_distribution is virtual.in_distribution on the measured inputs (the identity edit),
    not a re-derivation with copied ranges: the same function the what-if path is guarded by."""
    j = client.get("/replay/30420").json()
    shot = mast.load_shot(30420)
    ok = virtual.in_distribution(shot, [virtual.Edit()])[0]
    assert j["model"]["in_distribution"] == ok.tolist()


def test_replay_domain_bands_are_the_domain_flags_own_runs():
    """Band presence/absence and both coordinates are a pure function of the served flags — nothing shaded
    that the guard does not mark, nothing marked that the UI does not shade."""
    j = client.get("/replay/30420").json()
    t, flags = j["measured"]["t_s"], j["model"]["in_distribution"]
    expected, start = [], None
    for i, ok in enumerate(flags):
        if not ok and start is None:
            start = i
        if start is not None and (ok or i == len(flags) - 1):
            expected.append([t[start], t[i if not ok else i - 1]])
            start = None
    assert j["domain_bands"] == expected
    assert bool(j["domain_bands"]) == (False in flags)   # an all-in-domain shot shades nothing


def test_domain_bands_is_run_length_encoding_of_the_flags():
    """Unit contract of the shared helper: consecutive False slices close on their last slice; a run at
    either edge, a single slice, and an all-False trace all shade exactly their own slices."""
    t = [0.0, 0.1, 0.2, 0.3, 0.4]
    f = virtual.domain_bands
    assert f(t, [True] * 5) == []
    assert f(t, [False] * 5) == [[0.0, 0.4]]
    assert f(t, [False]) == [[0.0, 0.0]]
    assert f(t, [True, False, False, True, True]) == [[0.1, 0.2]]
    assert f(t, [True, True, False, False, False]) == [[0.2, 0.4]]
    assert f(t, [False, True, False, True, False]) == [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]]


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
