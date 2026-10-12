"""Golden traces: a persisted, reviewed corpus of full agent sessions, replayed offline in CI.

Two halves. Recording (one-time, needs a key, never in CI): agent_fakes.RecordingClient wraps the
real Messages client and writes each round down, serialized by to_trace() into the fixture format
{message, history, script: [{deltas, blocks}], recorded_model, recorded_at}. Replay (every CI run,
offline): this module scripts a FakeClient with each fixture in tests/fixtures/golden/ and replays
the session at both levels — the loop (agent.turn) and the HTTP surface (POST /agent/chat through
parse_sse) — asserting the replayed stream against the corpus:

- ordered tool-call sequences and the fixed event ordering (tool_call → tool_result → done/error);
- argument shapes against the LIVE registry — keys ⊆ the tool's input_schema.properties, required
  keys present — so a schema rename fails here, loudly;
- no name outside agent_tools.TOOLS is ever dispatched, and run_tool's unknown-tool text stays
  pinned against the live name list (names, never counts — a count drifted once already);
- confirm-then-apply invariants: the proposal reaches the sink and the action_proposed event, the
  model context carries "nothing is applied yet" and never the apply payload;
- the 6-round budget, and the explicit truncation marker on tool results that overflow the cap;
- honesty wording in a final text that ends in prose, and full text fidelity: the streamed deltas
  reproduce the recorded ones exactly.

The fixtures are hand-authored reviewed exemplars of the four sessions the spec commits to
(recorded_at is the authoring date): the "model" in a golden trace is the fixture, so this pins
the loop + registry + gates, never model quality. Hermetic throughout: no API key, no anthropic
SDK — the pristine-interpreter check extends test_api_agent's to this module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fusionlab import agent, agent_tools, api_agent
from fusionlab.agent_config import AgentSettings
from fusionlab.api import app
from tests.agent_fakes import FakeClient, RecordingClient, parse_sse, text_block

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "fixtures" / "golden"
TRACES = sorted(p.name for p in GOLDEN.glob("*.json"))

client = TestClient(app)

CONFIRM_THEN_APPLY = frozenset(
    name for name, tool in agent_tools.TOOLS.items() if tool.confirm_then_apply)

# house wording (CLAUDE.md) a reviewed final answer must carry — any one of these phrases
HOUSE_PHRASES = ("compared with MAST data", "a distance, never a forecast", "nothing is applied yet")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """The host environment must not turn the agent on behind the corpus's back, and no pending
    action outlives a trace (the same fixture test_api_agent.py runs)."""
    for var in ("FUSIONLAB_AGENT_ENABLED", "FUSIONLAB_AGENT_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    api_agent.store.clear()
    yield
    api_agent.store.clear()


# ---------------------------------------------------------------- replay helpers
def load_trace(name: str) -> dict[str, Any]:
    return json.loads((GOLDEN / name).read_text())


def scripted(trace: dict[str, Any]) -> list[tuple[list[str], list[dict[str, Any]]]]:
    """The fixture's script in FakeClient's shape: (deltas, blocks) per round, copied so a replay
    can never mutate the loaded corpus."""
    return [(list(r["deltas"]), [dict(b) for b in r["blocks"]]) for r in trace["script"]]


def scripted_calls(trace: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The corpus's own record of what the model did: every tool_use block, in script order."""
    return [(str(b["name"]), dict(b.get("input") or {}))
            for r in trace["script"] for b in r["blocks"] if b.get("type") == "tool_use"]


def expected_terminal(trace: dict[str, Any]) -> str:
    """A golden turn ends in `done` — unless the corpus itself is the budget shape: exactly
    MAX_TOOL_ROUNDS scripted rounds, every one carrying tool calls, which the cap ends in error.
    (A script with an unconsumed round after 6 tool rounds is a corpus bug: the recorded round can
    never play, and the replay's error terminal will say so.)"""
    rounds = trace["script"]
    every_round_calls = all(any(b.get("type") == "tool_use" for b in r["blocks"]) for r in rounds)
    return "error" if (len(rounds) == agent.MAX_TOOL_ROUNDS and every_round_calls) else "done"


def sink_recording(proposals: list[dict[str, Any]]) -> agent.ProposalSink:
    """A proposal sink that keeps what reached it, for the loop-level sink invariants."""
    def sink(proposal: dict[str, Any]) -> dict[str, Any]:
        proposals.append(proposal)
        return {"action_id": "a_golden", "expires_s": 600}
    return sink


def replay_loop(trace: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                sink: agent.ProposalSink) -> tuple[FakeClient, list[dict[str, Any]]]:
    """Replay a trace at the loop level: agent.turn over the scripted client, events collected.
    The turn runs under the corpus's recorded model configuration."""
    settings = AgentSettings(enabled=True, model=trace["recorded_model"], api_key="test-key",
                             max_upload_bytes=10 * 1024 * 1024)
    fake = FakeClient(scripted(trace))
    monkeypatch.setattr(agent, "_client", lambda key: fake)
    events = list(agent.turn([dict(m) for m in trace["history"]], trace["message"], settings,
                             on_proposal=sink))
    return fake, events


def replay_sse(trace: dict[str, Any],
               monkeypatch: pytest.MonkeyPatch) -> tuple[FakeClient, list[dict[str, Any]]]:
    """Replay a trace at the HTTP surface: POST /agent/chat through the real route (enabled mode,
    the real pending-action store as the sink), read back through the browser's parse_sse."""
    fake = FakeClient(scripted(trace))
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", "1")
    monkeypatch.setenv("FUSIONLAB_AGENT_MODEL", trace["recorded_model"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(agent, "_client", lambda key: fake)
    r = client.post("/agent/chat", json={"message": trace["message"], "history": trace["history"]})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return fake, parse_sse(r.text)


# ---------------------------------------------------------------- the shared golden assertions
def assert_golden(trace: dict[str, Any], fake: FakeClient, events: list[dict[str, Any]],
                  proposals: list[dict[str, Any]] | None) -> None:
    """Every replay — loop or SSE — must reproduce the corpus and hold the loop's invariants.
    `proposals` is the recording sink's list at loop level; None at the SSE level, where the real
    store holds the proposal and the run_whatif trace's confirm handshake checks it instead."""
    names = [e["tool"] for e in events if e["type"] == "tool_call"]
    args = [dict(e["args"]) for e in events if e["type"] == "tool_call"]
    expected = scripted_calls(trace)

    # 1. the corpus's tool sequence, replayed in order with the recorded args
    assert names == [n for n, _ in expected], f"tool sequence drifted: {names} != {[n for n, _ in expected]}"
    assert args == [a for _, a in expected], f"tool args drifted from the recording: {args} != {[a for _, a in expected]}"

    # 2. allowlist and argument shapes, against the live registry — names, never counts
    assert set(names) <= set(agent_tools.TOOLS), \
        f"the trace dispatches outside the registry: {sorted(set(names) - set(agent_tools.TOOLS))}"
    for name, call_args in zip(names, args, strict=True):
        schema = agent_tools.TOOLS[name].definition["input_schema"]
        unexpected = sorted(set(call_args) - set(schema.get("properties", {})))
        assert not unexpected, f"{name}: argument key(s) {unexpected} not in the live input_schema"
        missing = [req for req in schema.get("required", []) if req not in call_args]
        assert not missing, f"{name}: required argument(s) {missing} missing"

    # 3. fixed event ordering: a tool_call is answered by its own tool_result; exactly one terminal
    #    event, of the shape the corpus itself implies
    types = [e["type"] for e in events]
    for i, kind in enumerate(types):
        if kind == "tool_call":
            assert types[i + 1] == "tool_result" and events[i + 1]["tool"] == events[i]["tool"], \
                f"event {i}: a tool_call must be answered by its own tool_result"
    terminal = expected_terminal(trace)
    assert types[-1] == terminal and types.count("done") + types.count("error") == 1, \
        f"the replay ended in {types[-1]!r}; the corpus expects {terminal!r} ({types})"

    # 4. the budget: at most MAX_TOOL_ROUNDS rounds of tool results
    assert types.count("tool_result") <= agent.MAX_TOOL_ROUNDS

    # 5. every model call carried the honesty prompt, the corpus's model, the registry definitions
    #    in TOOLS order, and the corpus's own transcript threading
    assert fake.calls, "the replay never reached the model"
    assert all(c["system"] == agent.SYSTEM_PROMPT for c in fake.calls)
    assert all(c["model"] == trace["recorded_model"] for c in fake.calls)
    assert [d["name"] for d in fake.calls[0]["tools"]] == list(agent_tools.TOOLS)
    assert fake.calls[0]["messages"] == [*trace["history"], {"role": "user", "content": trace["message"]}]

    # 6. capped tool results: what re-enters the model context never exceeds the cap (the explicit
    #    marker is asserted where a trace's data actually overflows it)
    for call in fake.calls:
        for m in call["messages"]:
            if m["role"] == "user" and isinstance(m["content"], list):
                for block in m["content"]:
                    if block.get("type") == "tool_result":
                        assert len(block["content"]) <= agent.TOOL_RESULT_MAX_CHARS + 64

    # 7. full text fidelity, and the honesty gate on a turn that ends in prose
    streamed = "".join(e["text"] for e in events if e["type"] == "message_delta")
    recorded = "".join(d for r in trace["script"] for d in r["deltas"])
    assert streamed == recorded, "the streamed text drifted from the recorded corpus"
    if terminal == "done":
        final_text = "".join(trace["script"][-1]["deltas"])
        assert any(phrase in final_text for phrase in HOUSE_PHRASES), \
            f"the golden final text carries none of the house wording {HOUSE_PHRASES}: {final_text!r}"

    # 8. confirm-then-apply invariants for every proposal round: a confirm-then-apply call the
    #    gate allowed (its tool_result ok) must propose — the proposal reaches the event stream
    #    with the canonical args, and the model context carries "nothing is applied yet" but never
    #    the apply payload. A refused run_whatif (the gate-refusal trace) proposes nothing: the
    #    refusal travels in the result's error and the model explains it instead of retrying.
    #    (Tool results come back 1:1 with the scripted tool_use blocks, in order.)
    scripted_all = [(i, b) for i, r in enumerate(trace["script"]) for b in r["blocks"]
                    if b.get("type") == "tool_use"]
    results = [e for e in events if e["type"] == "tool_result"]
    proposal_rounds = [(i, b) for (i, b), result in zip(scripted_all, results, strict=True)
                       if b["name"] in CONFIRM_THEN_APPLY and result["ok"]]
    proposed_events = [e for e in events if e["type"] == "action_proposed"]
    assert len(proposed_events) == len(proposal_rounds), \
        f"expected {len(proposal_rounds)} proposal(s), saw {len(proposed_events)}"
    for (round_index, block), event in zip(proposal_rounds, proposed_events, strict=True):
        assert event["tool"] == block["name"] and event["args"] == dict(block.get("input") or {})
        blocks = fake.calls[round_index + 1]["messages"][-1]["content"]
        context = next(b for b in blocks if b.get("type") == "tool_result")
        assert "nothing is applied yet" in context["content"]
        assert "apply" not in context["content"]
    if proposals is not None:
        assert [p["args"] for p in proposals] == [e["args"] for e in proposed_events]


# ---------------------------------------------------------------- the corpus, replayed at both levels
@pytest.mark.parametrize("name", TRACES, ids=TRACES)
def test_a_golden_trace_replays_at_the_loop_level(name, monkeypatch):
    trace = load_trace(name)
    proposals: list[dict[str, Any]] = []
    fake, events = replay_loop(trace, monkeypatch, sink_recording(proposals))
    assert_golden(trace, fake, events, proposals)


@pytest.mark.parametrize("name", TRACES, ids=TRACES)
def test_a_golden_trace_replays_over_the_sse_surface(name, monkeypatch):
    trace = load_trace(name)
    fake, events = replay_sse(trace, monkeypatch)
    assert_golden(trace, fake, events, proposals=None)


# ---------------------------------------------------------------- per-trace specifics
def test_the_chain_trace_truncates_the_search_page_before_it_reenters_the_model_context(monkeypatch):
    """The unfiltered search page (200 rows) goes to the event whole and to the model context
    capped, with the explicit marker — the cap is load-bearing in the corpus, not just in unit tests."""
    trace = load_trace("01-read-only-chain.json")
    fake, events = replay_loop(trace, monkeypatch, sink_recording([]))
    search = next(e for e in events if e["type"] == "tool_result" and e["tool"] == "search_shots")
    assert len(search["data"]["rows"]) == 200
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert "…[truncated" in block["content"] and len(block["content"]) <= agent.TOOL_RESULT_MAX_CHARS + 64


def test_the_whatif_trace_confirms_through_the_real_gate(monkeypatch):
    """SSE end-to-end: the proposal reaches the store, confirm replays the server-stored args (a
    tampered client cannot supply its own), and a second confirm is 410 — one-shot."""
    trace = load_trace("02-run-whatif-proposal.json")
    _, events = replay_sse(trace, monkeypatch)
    proposed = next(e for e in events if e["type"] == "action_proposed")
    r = client.post(f"/agent/actions/{proposed['action_id']}/confirm")
    assert r.status_code == 200
    applied = r.json()
    assert applied["type"] == "action_applied" and applied["tool"] == "run_whatif"
    assert applied["args"] == scripted_calls(trace)[0][1]
    assert applied["apply"] == {"sliders": {"P_nbi": "1.40"}, "events": ["fusionlab:tab"]}
    assert client.post(f"/agent/actions/{proposed['action_id']}/confirm").status_code == 410


def test_the_refusal_trace_keeps_the_gate_text_verbatim(monkeypatch):
    """The recorded refusal is the route's own 422 text: the replayed error must equal a live
    run_tool of the recorded args, so the trace and the gate cannot drift apart silently. The
    model explains the refusal and never retries it — the refusal round is the trace's only one."""
    trace = load_trace("03-gate-refusal.json")
    _, events = replay_loop(trace, monkeypatch, sink_recording([]))
    refused = [e for e in events if e["type"] == "tool_result" and not e["ok"]]
    assert len(refused) == 1 and refused[0]["error"]
    name, args = scripted_calls(trace)[0]
    live = agent_tools.run_tool(name, **args)
    assert refused[0]["error"] == live.error
    assert [e["type"] for e in events].count("tool_call") == 1


def test_the_budget_trace_ends_at_the_cap(monkeypatch):
    """The budget trace's terminal event is the cap error's own wording — a seventh round never runs."""
    trace = load_trace("04-budget-exhaustion.json")
    fake, events = replay_loop(trace, monkeypatch, sink_recording([]))
    assert len(fake.calls) == agent.MAX_TOOL_ROUNDS
    assert events[-1]["type"] == "error"
    assert f"{agent.MAX_TOOL_ROUNDS}-round tool cap" in events[-1]["message"]
    assert not any(e["type"] == "done" for e in events)


# ---------------------------------------------------------------- corpus hygiene
def test_the_corpus_holds_the_four_reviewed_traces():
    """The minimum corpus the spec commits to — the four session shapes, pinned by name."""
    assert TRACES == ["01-read-only-chain.json", "02-run-whatif-proposal.json",
                      "03-gate-refusal.json", "04-budget-exhaustion.json"]


@pytest.mark.parametrize("name", TRACES, ids=TRACES)
def test_every_trace_is_a_well_formed_fixture(name):
    """The fixture format is exactly the recording shape, and every scripted tool_use names a tool
    in the live registry — the corpus-level allowlist statement."""
    trace = load_trace(name)
    assert set(trace) == {"message", "history", "script", "recorded_model", "recorded_at"}
    assert trace["message"] and isinstance(trace["history"], list)
    assert all(set(r) == {"deltas", "blocks"} for r in trace["script"])
    for r in trace["script"]:
        for b in r["blocks"]:
            assert b.get("type") in ("text", "tool_use")
            if b.get("type") == "tool_use":
                assert b.get("id") and b["name"] in agent_tools.TOOLS


def test_run_tools_unknown_tool_text_names_the_live_registry():
    """No fixture ever dispatches outside the registry; this pins the refusal text such a name
    would hit — against the live name list, never a hardcoded count."""
    r = agent_tools.run_tool("not_a_tool")
    assert r.error == (f"unknown tool 'not_a_tool'; "
                       f"the allowlisted tools are: {', '.join(agent_tools.TOOLS)}")


def test_the_recording_client_serializes_a_session_into_the_fixture_format():
    """The capture half, exercised over a fake: RecordingClient writes down every round it streams
    and serializes to the fixture format, and the serialized script replays through FakeClient."""
    recorder = RecordingClient(FakeClient([(["Hel", "lo."], [text_block("Hello.")])]))
    with recorder.messages.stream(model="test-model", max_tokens=agent.MAX_OUTPUT_TOKENS,
                                  system="s", tools=[], messages=[]) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == "Hello." and [b["text"] for b in final.content] == ["Hello."]
    trace = recorder.to_trace(message="hi", history=[], recorded_at="2026-10-12T00:00:00Z")
    assert set(trace) == {"message", "history", "script", "recorded_model", "recorded_at"}
    assert trace["recorded_model"] == "test-model"   # from the call that actually ran
    assert trace["script"] == [{"deltas": ["Hel", "lo."], "blocks": [text_block("Hello.")]}]
    replay = FakeClient([(list(r["deltas"]), [dict(b) for b in r["blocks"]]) for r in trace["script"]])
    with replay.messages.stream(messages=[]) as stream:
        assert "".join(stream.text_stream) == "Hello."


# ---------------------------------------------------------------- the harness must be able to fail
def test_the_harness_fails_loudly_on_a_mutated_trace(monkeypatch):
    """Verified once by hand during implementation (a fixture corrupted on disk, pytest red, the
    file restored); the same three corruptions are asserted here so the harness cannot silently
    stop failing. A golden harness that cannot fail is decoration."""
    # an out-of-allowlist tool name
    trace = load_trace("01-read-only-chain.json")
    trace["script"][0]["blocks"][1]["name"] = "not_a_tool"
    fake, events = replay_loop(trace, monkeypatch, sink_recording([]))
    with pytest.raises(AssertionError, match="outside the registry"):
        assert_golden(trace, fake, events, [])

    # an argument key the live schema does not know
    trace = load_trace("01-read-only-chain.json")
    trace["script"][1]["blocks"][1]["input"] = {"shot_ids": "30166"}
    fake, events = replay_loop(trace, monkeypatch, sink_recording([]))
    with pytest.raises(AssertionError, match="not in the live input_schema"):
        assert_golden(trace, fake, events, [])

    # a proposal session truncated: the round that would answer the proposal is gone, and the
    # replay ends in the model-call-failed error instead of the corpus's `done`
    trace = load_trace("02-run-whatif-proposal.json")
    trace["script"] = trace["script"][:1]
    fake, events = replay_loop(trace, monkeypatch, sink_recording([]))
    with pytest.raises(AssertionError, match="the corpus expects"):
        assert_golden(trace, fake, events, [])


# ---------------------------------------------------------------- hermetic, like the rest of the agent suite
def test_a_pristine_interpreter_imports_the_golden_harness_without_the_sdk():
    """The pristine-interpreter check, extended to this module: importing the golden harness (and
    the agent surface it pulls in) must not load the SDK — replay never needs it."""
    code = "import sys; import tests.test_agent_golden; assert 'anthropic' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
