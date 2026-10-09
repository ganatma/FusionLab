"""Served-JS tripwires for the frontend fetch-error discipline (fix-spec Family 4).

Same style as test_api.py::test_guided_study_hooks_are_in_place: assert substrings of the
JavaScript the server actually serves. They prove code shape, not behavior — the behavioral
evidence is the recorded browser dogfood in the fix PR. No JS harness is added on purpose
(CLAUDE.md: no node, no jsdom, no bundler in the no-build frontend).
"""
from fastapi.testclient import TestClient

from fusionlab.api import app

client = TestClient(app)


def _served(name: str) -> str:
    return client.get(f"/static/{name}").text


def test_stale_failures_are_seq_guarded_in_app_js():
    """JS-01: the seq guard extends into the catch blocks — a superseded failure may not
    overwrite the fresh status bar or heatmap."""
    app_js = _served("app.js")
    sim = app_js.split("async function runSimulate")[1]
    map_fn = app_js.split("async function runMap")[1]
    assert sim.count("seq !== simSeq") == 2                     # try path and catch path
    assert map_fn.count("seq !== mapSeq") == 2                  # try path and catch path
    assert "seq !== mapSeq" in map_fn.split("} catch (e) {")[1]


def test_failed_field_line_fetches_are_evicted_and_retried():
    """JS-02: a failed promise leaves lineCache (the next scrub retries) instead of
    negative-caching a null; lines3dOk goes false only behind the persistent-503 check."""
    replay = _served("replay.js")
    assert "lineCache.get(key) === p" in replay                 # evicts its own failed promise
    draw_lines = replay.split("async function drawLines")[1].split("async function drawPsi")[0]
    assert "linesFail503 < 3" in draw_lines.split("lines3dOk = false")[0]


def test_what_if_gate_is_validated_before_memoization():
    """JS-03: the gate is fetched into a local, checked for shape and r.ok, and only then
    memoized — a failed or malformed gate can no longer poison the panel until reload."""
    replay = _served("replay.js")
    wi_fn = replay.split("async function whatIf")[1].split("function labelsWhatIf")[0]
    assert "fetch('/virtual/gate')" in wi_fn
    assert "Array.isArray(j.caveats)" in wi_fn                  # parsed shape is checked
    assert "wi.gate = gate" in wi_fn                            # memoizes the local, post-validation


def test_what_if_post_is_guarded_and_json_safe():
    """JS-04: the debounced what-if POST is caught, and error bodies that are not JSON
    degrade to the status instead of crashing the handler."""
    replay = _served("replay.js")
    ask = replay.split("async function askWhatIf")[1].split("function drawWhatIf")[0]
    assert "r.json().catch(() => null)" in ask
    assert "d?.detail || r.status" in ask


def test_failed_init_is_not_memoized():
    """JS-05: a failed start() resets initP (the next tab entry retries) and renders a
    one-line error in the replay cell."""
    replay = _served("replay.js")
    init_fn = replay.split("function init()")[1].split("async function start")[0]
    assert "initP = null" in init_fn
    assert "rp-attr" in init_fn                                 # the error state lands in the panel


def test_failed_shot_switch_reverts_the_select():
    """JS-06: a failed loadShot puts the <select> back on the loaded shot and surfaces the
    error, instead of desyncing the dropdown from every panel."""
    replay = _served("replay.js")
    load = replay.split("async function loadShot")[1].split("async function insight")[0]
    assert "if (!r.ok) throw" in load
    assert "$('rp-shot').value = shot ? shot.meta.shot_id : ''" in load


def test_fire_and_forget_panels_render_their_errors():
    """JS-07: the /db and /surrogate fetches carry the figures.js one-line .catch."""
    replay = _served("replay.js")
    assert ".catch(() => { $('tab-db').innerHTML" in replay
    assert ".catch(() => { const el = $('rp-sur'); if (el) el.innerHTML" in replay


def test_compute_chip_polls_status_and_escapes_remote_strings():
    """JS-08 (PR 3): the compute chip polls /compute with a timeout and renders remote-host names
    through esc() — the handshake text comes from another machine, so treat it like shot data."""
    replay = _served("replay.js")
    assert "fetch('/compute'" in replay and "setInterval(pollCompute" in replay
    for name in ("f.computed_on", "q.computed_on", "q.learned_computed_on"):
        assert f"esc({name})" in replay


def test_fallback_banner_is_dismissible_and_self_rearming():
    """JS-09 (PR 3): the banner renders from the same /compute poll as the chip — dismissal is
    per-visit, recovery re-arms it, and the server-provided reason is escaped."""
    replay = _served("replay.js")
    banner = replay.split("function renderFallback")[1].split("async function pollCompute")[0]
    assert "fallbackSeen = false" in banner
    assert "esc(s.reason || '')" in banner
    assert "'rp-fallback-x'" in replay

