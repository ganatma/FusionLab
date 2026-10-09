"""The agent loop: one Anthropic Messages tool-use turn over the allowlisted registry (blueprint
art_LbvYoBDn, "Architecture" piece 1).

`turn()` is transport-agnostic: it consumes the client-held text history plus the new user message
and yields the turn's typed events — message_delta, tool_call, tool_result, action_proposed, done,
error — as plain dicts; `api_agent.py` encodes them as server-sent events. The loop never touches
HTTP and never applies anything: tools run through `agent_tools.run_tool`, the same registry
functions the routers use, and a confirm-then-apply result is handed to the `on_proposal` sink
(the confirmation store) instead of being acted on.

The Anthropic SDK lives behind exactly one seam, `_client`, and is imported inside it — nowhere
else — so importing this module, or the app that mounts it, never loads the SDK: the offline core
loop pays nothing for the agent (CLAUDE.md workflow rule 3). Tests monkeypatch
`fusionlab.agent._client` with a scripted fake, exactly as tests/test_api.py monkeypatches
fusionlab.api.simulate.

Honesty wording is house law (CLAUDE.md) and is encoded in the system prompt verbatim: the twin is
"compared with MAST data", never "validated" or "predictive"; a limit is a distance ("would cross
the Troyon limit at t = ...", never "would disrupt"); the export is "OpenUSD (Omniverse-compatible)",
never "runs in Omniverse".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from fusionlab.agent_config import AgentSettings

if TYPE_CHECKING:
    from fusionlab import agent_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6  # model round-trips that may carry tool calls per turn; beyond that the turn ends in an error
MAX_OUTPUT_TOKENS = 2048
TOOL_RESULT_MAX_CHARS = 6000  # cap on the data JSON that re-enters the model's context per tool result

ProposalSink = Callable[[dict[str, Any]], dict[str, Any]]
"""Receives a confirm-then-apply proposal ({tool, args, math?, apply}), stores it, and returns its
confirmation token ({action_id, expires_s}) for the action_proposed event. Implemented by the
pending-action store in api_agent.py."""

SYSTEM_PROMPT = """\
You are the FusionLab assistant: a chat panel inside an educational tokamak digital twin that replays
real MAST open-data shots beside a reduced 0-D physics model. Answer from the tools, never from
general tokamak lore — if a tool did not return it, you do not know it.

House honesty rules (CLAUDE.md), binding in every answer:
- This twin is an educational model compared with MAST data. Never say "validated" or "predictive".
- A limit is a distance: say "would cross the Troyon limit at t = 0.31 s", never "would disrupt".
  A crossing is a distance, never a forecast of a disruption.
- Say "OpenUSD export (Omniverse-compatible)". Never say it "runs in Omniverse".
- Be honest about the archive's edges: search results come back in ascending shot_id order (there is
  no sort control), the archive has no H-mode flag (shots cannot be found or filtered by confinement
  mode), and replay runs offline only on the locally cached shots — an uncached shot cannot be
  opened, though search_shots still covers the full catalog.

Confirm-then-apply: run_whatif and export_usd execute nothing on their own — they return a proposal
that waits for the user's confirmation in the UI. Until the UI shows it applied, say "proposed",
never "applied". Edits the matched-pairs evidence cannot test are refused with the reason: explain
the refusal, never retry it and never guess a replacement.

Keep numbers with their units (MA, T, MW, MJ, 1e20 m^-3, keV, s), and attribute measurements to
FAIR-MAST (CC BY-SA 4.0, UKAEA) when you cite them."""


def _client(api_key: str | None) -> Any:
    """The single patchable seam: construct the Anthropic Messages client. The SDK import lives here
    and nowhere else, so disabled mode never loads it. Tests monkeypatch this symbol with a scripted
    fake client, exactly like tests/test_api.py monkeypatches fusionlab.api.simulate."""
    import anthropic  # lazy: the offline core loop never pays for the SDK

    return anthropic.Anthropic(api_key=api_key)


def _block_dict(block: Any) -> dict[str, Any]:
    """A content block as a plain dict: the SDK's pydantic blocks dump cleanly, test fakes pass
    dicts through — the loop works on one shape."""
    return block if isinstance(block, dict) else dict(block.model_dump())


def tool_context(result: agent_tools.ToolResult) -> str:
    """What a tool execution leaves in the model's context: the plain-language summary, then the data
    JSON — the joined string is capped at TOOL_RESULT_MAX_CHARS, so a 200-row search page must not
    own the window. (The SSE tool_result event carries the full data; the model argues from the
    summary and the head.)"""
    if result.error is not None:
        return f"error: {result.error}"
    parts = [result.summary]
    if result.proposal is not None:
        parts.append("This is a proposal waiting for the user's confirmation in the UI — nothing is applied yet.")
    if result.data:
        parts.append(f"data: {json_dumps(result.data)}")
    blob = "\n".join(parts)
    if len(blob) > TOOL_RESULT_MAX_CHARS:
        blob = blob[:TOOL_RESULT_MAX_CHARS] + f"…[truncated {len(blob) - TOOL_RESULT_MAX_CHARS} chars]"
    return blob


def json_dumps(payload: Any) -> str:
    """Strict JSON: non-finite floats must surface as a tool-loop error, never reach the model or the
    stream as NaN literals."""
    return json.dumps(payload, allow_nan=False)


def turn(history: list[dict[str, str]], message: str, settings: AgentSettings,
         on_proposal: ProposalSink) -> Iterator[dict[str, Any]]:
    """Run one agent turn and yield its typed events.

    `history` is the client-held transcript (plain text turns); this turn's tool rounds live only
    inside the call, so the server stays stateless per request apart from the confirmation store the
    sink owns (blueprint "Flow and states"). At most MAX_TOOL_ROUNDS rounds may carry tool calls — a
    model still asking for tools after that ends the turn with an error event instead of a seventh
    round. Any SDK failure becomes an error event too: the stream never raises past the route.
    """
    # Late by design: the registry wraps this app's route functions (agent_tools imports
    # fusionlab.api), so a module-level import would make agent → registry → app a circular-import
    # landmine. From inside a turn the app is always fully loaded.
    from fusionlab import agent_tools

    messages: list[dict[str, Any]] = [{**m} for m in history]  # the turn owns its copy of the transcript
    messages.append({"role": "user", "content": message})
    client = _client(settings.api_key)

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            with client.messages.stream(model=settings.model, max_tokens=MAX_OUTPUT_TOKENS, system=SYSTEM_PROMPT,
                                        tools=agent_tools.tool_definitions(), messages=messages) as stream:
                for delta in stream.text_stream:
                    yield {"type": "message_delta", "text": delta}
                final = stream.get_final_message()
        except Exception:
            logger.exception("agent model call failed (model %s)", settings.model)
            yield {"type": "error", "message": "the model call failed — nothing ran; see the server log"}
            return

        blocks = [_block_dict(b) for b in final.content]
        messages.append({"role": "assistant", "content": blocks})
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        if not calls:
            yield {"type": "done"}
            return

        tool_results: list[dict[str, Any]] = []
        for call in calls:
            name, args = str(call["name"]), dict(call.get("input") or {})
            yield {"type": "tool_call", "tool": name, "args": args}
            result = agent_tools.run_tool(name, **args)
            yield {"type": "tool_result", "tool": name, "ok": result.error is None,
                   "summary": result.summary, "data": result.data, "error": result.error}
            if result.proposal is not None:   # confirm-then-apply: hold it, never act on it here
                token = on_proposal(result.proposal)
                yield {"type": "action_proposed", "action_id": token["action_id"], "tool": name, "args": args,
                       "math": str(result.proposal.get("math") or ""), "expires_s": token["expires_s"]}
            tool_results.append({"type": "tool_result", "tool_use_id": str(call["id"]),
                                 "content": tool_context(result), "is_error": result.error is not None})
        messages.append({"role": "user", "content": tool_results})

    yield {"type": "error", "message": f"the turn stopped at the {MAX_TOOL_ROUNDS}-round tool cap — "
                                       "ask a narrower question or start a new turn"}
