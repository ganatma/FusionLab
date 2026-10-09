"""Shared scripted fakes for the agent tests: a stub Messages client standing in behind the
fusionlab.agent._client seam, and the SSE reader a browser would run. No SDK, no network."""

from __future__ import annotations

import copy
import json
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


def parse_sse(text: str) -> list[dict[str, Any]]:
    """The browser's reader: data lines only, one JSON payload each."""
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]
