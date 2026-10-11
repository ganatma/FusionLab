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


def test_replay_guards_absent_equilibrium():
    """DATA-1: a bundle without equilibrium replays its traces — replay.js reads the
    `equilibrium_absent` hint instead of dereferencing lcfs/psi_grid unconditionally."""
    replay = _served("replay.js")
    assert "s.equilibrium_absent" in replay          # the hint surfaces in the insight panel
    assert "grid ? grid.R : []" in replay            # the section view tolerates a missing ψ grid
    assert "shot.psi_grid ?" in replay               # the header omits the EFIT tag without one
    assert "shot.psi_grid ? 'EFIT, real vessel'" in replay   # the panel sub-label keeps its
    assert "'no reconstruction — trace-only replay'" in replay   # provenance honest without one


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
    # one notice per grid row: while the banner is up it takes the row from the onboarding hint
    assert "$('rp-hint').hidden = show ? true : hintSeen()" in banner


def test_agent_panel_drives_only_public_hooks(monkeypatch):
    """JS-08: the agent driver follows the guide.js discipline — it checks /agent/health before
    rendering, reads the SSE frames as data lines, and drives the app only through public hooks and
    input/change dispatch, never another script's internals. (agent.js is served only when enabled,
    so the test turns the flag on for the served fetch.)"""
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    agent_js = _served("agent.js")
    assert "fetch('/agent/health')" in agent_js                       # the documented gate
    assert "l.startsWith('data:')" in agent_js                        # SSE frames parsed as data lines
    assert "window.FusionLab.replay" in agent_js                      # public hooks only
    assert "dispatchEvent(new Event('input', { bubbles: true }))" in agent_js
    assert "EventSource" not in agent_js                              # the chat is a POST; EventSource is GET-only


def test_export_downloads_are_wired_into_the_served_js():
    """FEAT-1: both take-away export affordances exist in the served replay.js — the toolbar's per-shot
    download and the compare bar's multi-shot download — and the CSV helper keeps the fetch-error
    discipline (the .ok check, the busy state) instead of failing silently."""
    replay = _served("replay.js")
    assert 'id="rp-csv"' in replay                                    # the toolbar's export button
    assert "`/export/${shot.meta.shot_id}`" in replay                 # single-shot download URL
    assert "`/export?shots=" in replay                               # compare download URL
    assert 'data-act="export"' in replay                              # the compare bar's export button
    dl = replay.split("async function dlExport", 1)[1].split("CSV export failed", 1)[0]
    assert "if (!r.ok) throw" in dl
    assert "aria-busy" in dl


def test_job_progress_is_wired_into_the_served_js():
    """HARN-1: the replay UI opens the SSE progress relay for a named task, renders the shared event
    vocabulary, and closes the stream on terminal and native-end events — EventSource's auto-reconnect
    must not spin against a route that ends after the terminal event. The benchmark control and the
    slow field-line threshold wire the same watcher. (HARN-2 consciously changed the terminal set:
    'cancelled' keeps the stream open so the UI can follow the local-fallback re-run.)"""
    replay = _served("replay.js")
    watch = replay.split("function watchProgress")[1].split("const psiCache")[0]
    assert "new EventSource(`/compute/progress?task=${encodeURIComponent(task)}`)" in watch
    assert "es.addEventListener('progress'" in watch
    assert "['completed', 'error'].forEach(name => es.addEventListener(name, end))" in watch
    assert "es.addEventListener('cancelled'" in watch
    native = watch.split("es.addEventListener('error'")[1]
    assert "if (e.data) return;" in native.split("es.close()")[0]   # stream end closes; a job error event doesn't
    assert 'id="rp-bench"' in replay
    bench = replay.split("$('rp-bench').addEventListener")[1]
    assert "watchProgress('benchmark')" in bench
    assert "setTimeout(() => watchProgress('trace_fieldlines'), 1500)" in replay


def test_cancel_affordance_is_wired_into_the_served_js():
    """HARN-2: the progress readout carries a cancel control, shown only while the relayed events
    name the ssh provider (local jobs are synchronous — nothing to interrupt), POSTing the app's
    cancel route with the fetch-error discipline (no silent failures)."""
    replay = _served("replay.js")
    assert 'id="rp-progress-cancel"' in replay
    watch = replay.split("function watchProgress")[1].split("const psiCache")[0]
    assert "cancelBtn.hidden = progressProvider !== 'ssh'" in watch
    click = replay.split("$('rp-progress-cancel').addEventListener")[1]
    assert "fetch(`/compute/cancel?task=${encodeURIComponent(progressTask)}`" in click
    assert "{ method: 'POST' }" in click
    assert "if (!r.ok)" in click


def test_replay_domain_shading_is_the_served_bands_with_a_legend_label():
    """FEAT-2: the W strip shades the server's own domain bands (never recomputed client-side) and
    labels them in its legend, present only when the flags mark something out of domain. The what-if
    bands share the same machinery — one shading style, one vocabulary."""
    replay = _served("replay.js")
    strips = replay.split("function drawStrips")[1].split("function insight")[0]
    assert "shot.domain_bands" in strips               # the coordinates are the replay bundle's
    assert "oodShapes(ood, 'meas-ood')" in strips      # the shared band builder, the what-if's style
    assert "'outside the training range'" in strips    # labelled in the plot legend when present
    whatif = replay.split("function drawWhatIf")[1].split("function revolve")[0]
    assert "oodShapes(oodRuns(" in whatif              # the what-if path maps its own flags with the same machinery


def test_abort_overlay_is_conditional_on_the_catalog_flag():
    """FEAT-3: the abort badge and the limit-strip annotation exist only as conditions on the
    bundle's archive.abort — absent or false flags leave the panel untouched (silent degradation) —
    and the panel wording keeps the house rule: computed distances, never forecasts."""
    replay = _served("replay.js")
    assert 'id="rp-abort" hidden' in replay                     # the badge ships hidden, shown per shot
    assert "$('rp-abort').hidden = !shot.archive?.abort" in replay   # toggled only from the bundle flag
    strip_fn = replay.split("function drawStrips")[1].split("function insight(s)")[0]
    assert "shot.archive?.abort" in strip_fn                    # the annotation is conditional, not static
    assert "⚠ archive abort flag" in strip_fn
    assert "a distance, never a forecast" in replay             # the house wording, on the panel itself
    ins_fn = replay.split("function insight(s)")[1].split("const ohmic")[0]
    assert "flags this shot <b>aborted</b>" in ins_fn and "catalog bookkeeping" in ins_fn
