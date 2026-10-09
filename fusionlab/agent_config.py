"""Agent settings — the repo's first env reader (every key optional; the offline demo never needs any).

The in-app natural-language agent is off unless FUSIONLAB_AGENT_ENABLED is explicitly 1/true/yes AND an
API key is present. `AgentSettings.enabled` is that effective switch, so `/agent/health` can report
honestly: a flag without a key reads "disabled", never "enabled but broken".

Importing this module imports nothing beyond the stdlib — in particular it never imports `anthropic`.
The key check here is string presence only; the client is constructed later, inside the agent loop, so
the offline core loop pays nothing for the agent (CLAUDE.md workflow rule 3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "claude-opus-5-5"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB cap on protocol uploads (PDF or pasted text)
_TRUTHY = frozenset({"1", "true", "yes"})


@dataclass(frozen=True)
class AgentSettings:
    """A point-in-time read of the agent's configuration.

    All fields are decided together, in `agent_settings()` — `enabled` is never recomputed against a
    changed environment behind the snapshot's back.
    """

    enabled: bool
    model: str
    api_key: str | None
    max_upload_bytes: int


def _flag_on() -> bool:
    """True only when FUSIONLAB_AGENT_ENABLED is explicitly 1, true, or yes (case-insensitive)."""
    return os.getenv("FUSIONLAB_AGENT_ENABLED", "").lower() in _TRUTHY


def agent_settings() -> AgentSettings:
    """Read the environment on every call — no caching, so tests (and later, route handlers) see live values."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or None  # an empty string is no key
    return AgentSettings(
        enabled=_flag_on() and api_key is not None,
        model=os.getenv("FUSIONLAB_AGENT_MODEL") or DEFAULT_MODEL,
        api_key=api_key,
        max_upload_bytes=MAX_UPLOAD_BYTES,
    )
