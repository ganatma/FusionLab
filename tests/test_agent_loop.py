"""The agent loop over a scripted fake client, standing in at the one seam (fusionlab.agent._client).

The blueprint's loop-level outcomes live here: the model call carries the configured model, the
honesty system prompt, and the eight allowlisted definitions; tool results are fed back with their
data truncated; proposals reach the sink instead of being acted on; the tool budget is 6 rounds;
and an SDK failure becomes an error event rather than a raise.
"""

from __future__ import annotations

from typing import Any

import pytest

from fusionlab import agent, agent_tools
from fusionlab.agent_config import AgentSettings
from tests.agent_fakes import ExplodingClient, FakeClient, text_block, tool_use_block

SETTINGS = AgentSettings(enabled=True, model="test-model", api_key="test-key",
                         max_upload_bytes=10 * 1024 * 1024)


def scripted(monkeypatch: pytest.MonkeyPatch, script: list[tuple[list[str], list[dict[str, Any]]]]) -> FakeClient:
    fake = FakeClient(script)
    monkeypatch.setattr(agent, "_client", lambda key: fake)
    return fake


def run_turn(monkeypatch: pytest.MonkeyPatch, script: list[tuple[list[str], list[dict[str, Any]]]],
             history: list[dict[str, str]] | None = None, message: str = "hi",
             sink: agent.ProposalSink | None = None) -> tuple[FakeClient, list[dict[str, Any]]]:
    """One turn over the scripted client, events collected; the sink records proposals by default."""
    fake = scripted(monkeypatch, script)
    seen: list[dict[str, Any]] = []

    def default_sink(proposal: dict[str, Any]) -> dict[str, Any]:
        seen.append(proposal)
        return {"action_id": "a_test", "expires_s": 600}

    events = list(agent.turn(history or [], message, SETTINGS, on_proposal=sink or default_sink))
    return fake, events


# ---- a plain-text turn
def test_plain_text_turn_streams_deltas_then_done(monkeypatch):
    fake, events = run_turn(monkeypatch, [(["Hel", "lo"], [text_block("Hello")])])
    assert [e["type"] for e in events] == ["message_delta", "message_delta", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "message_delta") == "Hello"
    assert fake.calls[0]["model"] == "test-model" and fake.calls[0]["max_tokens"] == agent.MAX_OUTPUT_TOKENS
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_the_model_call_carries_the_honesty_prompt_and_the_registry_definitions(monkeypatch):
    fake, _ = run_turn(monkeypatch, [(["."], [text_block(".")])])
    assert fake.calls[0]["system"] == agent.SYSTEM_PROMPT
    assert [d["name"] for d in fake.calls[0]["tools"]] == list(agent_tools.TOOLS)


# ---- tool rounds
def test_tool_round_feeds_the_result_back_and_finishes(monkeypatch):
    script = [(["Looking."], [text_block("Looking."), tool_use_block("tu_1", "search_shots", {"q": "30421"})]),
              (["Found ", "one."], [text_block("Found one.")])]
    fake, events = run_turn(monkeypatch, script)
    assert [e["type"] for e in events] == ["message_delta", "tool_call", "tool_result",
                                           "message_delta", "message_delta", "done"]
    assert events[1] == {"type": "tool_call", "tool": "search_shots", "args": {"q": "30421"}}
    assert events[2]["ok"] is True and events[2]["data"]["total"] == 1 and events[2]["error"] is None
    # the model's context grew by exactly the tool-result block, carrying the summary and the data
    followup = fake.calls[1]["messages"]
    assert [m["role"] for m in followup] == ["user", "assistant", "user"]
    block = followup[-1]["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "tu_1"
    assert block["is_error"] is False and "1 shots match" in block["content"]


def test_a_tool_error_reaches_the_model_as_is_error_and_the_stream_continues(monkeypatch):
    script = [(["."], [tool_use_block("tu_1", "nope")]),
              (["Sorry."], [text_block("Sorry.")])]
    _, events = run_turn(monkeypatch, script)
    assert events[2]["ok"] is False and "unknown tool" in events[2]["error"]
    assert events[-1]["type"] == "done"


def test_read_only_results_are_truncated_before_re_entering_the_model_context(monkeypatch):
    """A full 200-row search page goes to the UI whole; the model sees the summary and a capped head
    with an explicit truncation marker."""
    script = [(["."], [tool_use_block("tu_1", "search_shots")]),
              (["Done."], [text_block("Done.")])]
    fake, events = run_turn(monkeypatch, script)
    assert events[2]["data"]["total"] == 15_969 and len(events[2]["data"]["rows"]) == 200  # the UI: whole page
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert "…[truncated" in block["content"] and len(block["content"]) <= agent.TOOL_RESULT_MAX_CHARS + 64


def test_the_tool_budget_is_six_rounds(monkeypatch):
    calls_tool = (["."], [tool_use_block("tu_n", "list_devices")])
    fake, events = run_turn(monkeypatch, [calls_tool] * 7)
    assert len(fake.calls) == 6                                        # a seventh round never happens
    assert [e["type"] for e in events].count("tool_result") == 6
    assert events[-1]["type"] == "error" and "6-round tool cap" in events[-1]["message"]
    assert not any(e["type"] == "done" for e in events)


def test_an_sdk_failure_becomes_an_error_event(monkeypatch):
    exploding = ExplodingClient()
    monkeypatch.setattr(agent, "_client", lambda key: exploding)
    events = list(agent.turn([], "hi", SETTINGS, on_proposal=lambda p: {"action_id": "a", "expires_s": 600}))
    assert [e["type"] for e in events] == ["error"] and "model call failed" in events[0]["message"]


# ---- proposals: the loop holds them, the sink decides what "held" means
def test_a_proposal_goes_to_the_sink_as_an_event_and_never_into_the_model_context(monkeypatch):
    script = [(["Checking."], [tool_use_block("tu_1", "export_usd", {"shot_id": 30420})]),
              (["Proposed."], [text_block("Proposed.")])]
    fake, events = run_turn(monkeypatch, script)
    proposed = next(e for e in events if e["type"] == "action_proposed")
    assert proposed == {"type": "action_proposed", "action_id": "a_test", "tool": "export_usd",
                        "args": {"shot_id": 30420}, "math": "", "expires_s": 600}
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert "nothing is applied yet" in block["content"]
    assert "apply" not in block["content"]   # the driver's payload stays server-side


# ---- the model-context string, unit level
def test_tool_context_passes_a_refusal_verbatim():
    assert agent.tool_context(agent_tools.ToolResult(summary="", data={}, error="boom")) == "error: boom"


def test_tool_context_truncates_large_data():
    big = agent_tools.ToolResult(summary="s", data={"rows": [{"i": i} for i in range(4000)]})
    ctx = agent.tool_context(big)
    assert "…[truncated" in ctx and len(ctx) <= agent.TOOL_RESULT_MAX_CHARS + 64


# ---- the system prompt encodes the house wording, verbatim
def test_system_prompt_encodes_the_house_wording():
    assert "compared with MAST data" in agent.SYSTEM_PROMPT
    assert 'Never say "validated" or "predictive"' in agent.SYSTEM_PROMPT
    assert '"would cross the Troyon limit at t = 0.31 s", never "would disrupt"' in agent.SYSTEM_PROMPT
    assert "a distance, never a forecast" in agent.SYSTEM_PROMPT
    assert '"OpenUSD export (Omniverse-compatible)"' in agent.SYSTEM_PROMPT
    assert 'Never say it "runs in Omniverse"' in agent.SYSTEM_PROMPT
