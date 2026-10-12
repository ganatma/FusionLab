"""Shared scripted fakes for the agent tests: a stub Messages client standing in behind the
fusionlab.agent._client seam, the SSE reader a browser would run, and the one-time recording
client that captures a live session into a golden fixture. No SDK, no network (imported here)."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use_block(call_id: str, name: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input or {}}


class FakeMessage:
    def __init__(self, content: list[dict[str, Any]]):
        self.content = content


class FakeStream:
    """The slice of the Messages streaming API the loop uses: text deltas, then the final message."""

    def __init__(self, deltas: list[str], blocks: list[dict[str, Any]]):
        self._deltas, self._blocks = deltas, blocks

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def text_stream(self) -> Any:
        return iter(self._deltas)

    def get_final_message(self) -> FakeMessage:
        return FakeMessage(self._blocks)


class FakeClient:
    """Scripted LLM: every model call pops the next (deltas, blocks) pair and records the kwargs,
    so tests can assert what the loop sent — system prompt, tool definitions, fed-back results."""

    def __init__(self, script: list[tuple[list[str], list[dict[str, Any]]]]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def stream(self, **kwargs: Any) -> FakeStream:
        # deep-copy the kwargs: the loop keeps mutating its messages list, and a test that reads
        # calls[n] must see what call n saw, not the final transcript
        self.calls.append(copy.deepcopy(kwargs))
        deltas, blocks = self.script.pop(0)
        return FakeStream(deltas, blocks)

    def create(self, **kwargs: Any) -> FakeMessage:
        """The non-streaming call the protocol extractor uses (forced tool_choice cannot stream);
        the scripted deltas are meaningless here, only the blocks are returned."""
        self.calls.append(copy.deepcopy(kwargs))
        _, blocks = self.script.pop(0)
        return FakeMessage(blocks)


class ExplodingClient:
    """A client whose every call fails — the loop must answer with an error event, not a raise."""

    def __init__(self) -> None:
        self.messages = self

    def stream(self, **kwargs: Any) -> FakeStream:
        raise RuntimeError("api down")


class RecordingClient(FakeClient):
    """The capture half of the golden-trace harness (tests/test_agent_golden.py): wraps a real
    Messages client, writes down every round it streams — the deltas and the final blocks — and
    serializes the session into the golden fixture format via to_trace(). Recording is a one-time,
    key-bearing operation done by hand (monkeypatch fusionlab.agent._client with a lambda that
    wraps the real client); CI only replays the committed fixtures and never constructs this class.

    The wrapped client only needs the slice the loop uses: .messages.stream(**kwargs) as a context
    manager exposing text_stream and get_final_message() — the real SDK client, or FakeClient in a
    test. Nothing here imports the SDK; the wrapper is duck-typed.
    """

    def __init__(self, wrapped: Any) -> None:
        super().__init__([])
        self.wrapped = wrapped

    def stream(self, **kwargs: Any) -> FakeStream:
        # override the replay pop: recording appends what the wrapped client actually returned
        with self.wrapped.messages.stream(**kwargs) as stream:
            deltas = list(stream.text_stream)
            final = stream.get_final_message()
        blocks = [b if isinstance(b, dict) else b.model_dump() for b in final.content]
        self.script.append((deltas, blocks))
        self.calls.append(copy.deepcopy(kwargs))
        return FakeStream(deltas, blocks)

    def to_trace(self, message: str, history: list[dict[str, str]],
                 recorded_at: str | None = None) -> dict[str, Any]:
        """The fixture format: {message, history, script: [{deltas, blocks}], recorded_model,
        recorded_at}. recorded_model comes from the first recorded call — the model that actually
        ran. Recording nothing first is an IndexError on purpose: no session, no trace."""
        if not self.calls:
            raise IndexError("nothing was recorded; to_trace needs a streamed session")
        return {
            "message": message,
            "history": [dict(m) for m in history],
            "script": [{"deltas": list(d), "blocks": [dict(b) for b in blocks]} for d, blocks in self.script],
            "recorded_model": str(self.calls[0]["model"]),
            "recorded_at": recorded_at or datetime.now(UTC).isoformat(),
        }


def parse_sse(text: str) -> list[dict[str, Any]]:
    """The browser's reader: data lines only, one JSON payload each."""
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]
