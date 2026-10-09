"""HTTP surface for the in-app natural-language agent (blueprint art_LbvYoBDn, "HTTP surface").

`POST /agent/chat` streams one agent turn as `text/event-stream`: a sequence of typed events
(message_delta, tool_call, tool_result, action_proposed, done, error) read by the browser's
fetch-stream reader — EventSource is GET-only, so the chat must be a POST. `action_applied` shares
the same event vocabulary but lives on the confirmation response: the chat stream ends with the
proposal, the user decides, and `POST /agent/actions/{id}/confirm` answers with the apply payload
the browser driver executes.

The confirmation store is one in-memory slot with a 10-minute TTL and a one-pending invariant: a
new proposal replaces the old one and a new message expires whatever is pending. That is
deliberately single-user demo scale — the server is stateless per request apart from this slot
(blueprint "Flow and states"); a restart forgets it, and a confirm is one-shot.

Security shape of the confirm endpoint: the client names an action, it never supplies parameters.
The endpoint takes no body — the stored args were computed server-side through the route's own
gates (the /virtual/{id} matched-pairs gate, the cache boundary), so a tampered client can only
confirm or not; it cannot steer the edit.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fusionlab import agent
from fusionlab.agent_config import agent_settings

router = APIRouter()

logger = logging.getLogger(__name__)

ACTION_TTL_S = 600.0  # 10 minutes: a proposal dies at the next message or the TTL, whichever comes first


# ---------------------------------------------------------------- the pending-action store
class PendingActionError(LookupError):
    """Confirming an unknown, already-applied, or expired action — the route answers 410."""


@dataclass(frozen=True)
class PendingAction:
    """A confirm-then-apply proposal held server-side until the user confirms, the TTL passes, or
    the next message replaces it. `apply` is the browser driver's payload, verbatim from the tool."""

    action_id: str
    tool: str
    args: dict[str, Any]
    math: str
    apply: dict[str, Any]
    created_at: float
    expires_at: float


def _new_action_id() -> str:
    return "a_" + uuid.uuid4().hex[:12]


class PendingActionStore:
    """One slot, one TTL. Single-user demo scale on purpose: the two-phase gate exists so a human
    sees the exact parameters before they apply, not to arbitrate concurrency."""

    def __init__(self, ttl_s: float = ACTION_TTL_S, clock: Callable[[], float] = time.monotonic,
                 new_id: Callable[[], str] = _new_action_id) -> None:
        self._ttl = ttl_s
        self._clock = clock
        self._new_id = new_id
        self._lock = threading.Lock()
        self._pending: PendingAction | None = None

    def create(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Store one proposal — replacing any pending one (the one-pending invariant) — and return
        its confirmation token for the action_proposed event."""
        now = self._clock()
        action = PendingAction(action_id=self._new_id(), tool=str(proposal["tool"]),
                               args=dict(proposal.get("args") or {}), math=str(proposal.get("math") or ""),
                               apply=dict(proposal["apply"]), created_at=now, expires_at=now + self._ttl)
        with self._lock:
            self._pending = action
        return {"action_id": action.action_id, "expires_s": int(self._ttl)}

    def confirm(self, action_id: str) -> PendingAction:
        """Pop the pending action — one-shot: a confirmation is spent on first use. Unknown,
        mismatched, and expired ids raise PendingActionError (an expired action is dropped)."""
        with self._lock:
            action = self._pending
            if action is None:
                raise PendingActionError(f"no pending action; '{action_id}' is unknown or already applied")
            if action.action_id != action_id:
                raise PendingActionError(f"'{action_id}' is not the pending action")
            if self._clock() >= action.expires_at:
                self._pending = None
                raise PendingActionError(f"action '{action_id}' expired — propose it again")
            self._pending = None
            return action

    def clear(self) -> None:
        """A new message expires any pending action — the one-pending invariant's other half."""
        with self._lock:
            self._pending = None


store = PendingActionStore()


# ---------------------------------------------------------------- request/response shapes
class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatIn(BaseModel):
    """The client-held transcript plus the current turn. The tool rounds of a turn never leave the
    server — the browser keeps only the text turns it displayed."""

    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessageIn] = Field(default_factory=list, max_length=40)


def _validated_history(history: list[ChatMessageIn]) -> list[dict[str, str]]:
    """The Messages API wants user/assistant turns that alternate, start with the user, and — since
    the new message follows — end with the assistant. Enforced here as a 422, not as a wrapped
    model error later."""
    turns = [{"role": m.role, "content": m.content} for m in history]
    for i, turn_msg in enumerate(turns):
        want = "user" if i % 2 == 0 else "assistant"
        if turn_msg["role"] != want:
            raise ValueError("history must alternate user, assistant, ... starting with the user")
    if turns and turns[-1]["role"] != "assistant":
        raise ValueError("history must end with the assistant's reply (the new message follows as the user's turn)")
    return turns


def _sse(event: dict[str, Any]) -> str:
    """One server-sent event: a single data line whose JSON carries the type. The browser reader
    splits on the blank line and json.parses the payload."""
    return f"data: {json.dumps(event, allow_nan=False)}\n\n"


# ---------------------------------------------------------------- routes
@router.get("/agent/health")
def agent_health() -> dict[str, Any]:
    """The panel's gate: agent.js loads only when this reports enabled (blueprint "HTTP surface")."""
    s = agent_settings()
    return {"enabled": s.enabled, "model": s.model}


@router.post("/agent/chat")
def agent_chat(body: ChatIn) -> StreamingResponse:
    """One agent turn as a stream of typed events. The server is stateless per request: history
    lives in the browser, and this new message expires any pending action before the turn runs."""
    settings = agent_settings()
    if not settings.enabled:
        raise HTTPException(409, "the agent is disabled: set FUSIONLAB_AGENT_ENABLED=1 and ANTHROPIC_API_KEY "
                                 "in .env (the offline demo never needs them)")
    try:
        history = _validated_history(body.history)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    store.clear()

    def gen() -> Iterator[str]:
        try:
            for event in agent.turn(history, body.message, settings, on_proposal=store.create):
                yield _sse(event)
        except Exception:
            logger.exception("agent turn failed")
            yield _sse({"type": "error", "message": "the agent turn failed — see the server log"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/agent/actions/{action_id}/confirm")
def agent_confirm(action_id: str) -> dict[str, Any]:
    """Second phase of the two-phase gate: replay the server-stored parameters and return the apply
    payload the browser driver executes. The endpoint takes no body — a tampered client cannot
    supply action parameters (they were computed server-side, through the route's own gates)."""
    try:
        action = store.confirm(action_id)
    except PendingActionError as e:
        raise HTTPException(410, str(e)) from e
    return {"type": "action_applied", "action_id": action.action_id, "tool": action.tool,
            "args": action.args, "apply": action.apply}
