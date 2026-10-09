"""The agent HTTP surface: SSE chat over a scripted fake client, the two-phase confirm gate, health.

The blueprint's route-level outcomes live here: the full event sequence parses over the stream, a
proposal changes nothing until confirm returns the apply payload, a tampered client cannot supply
action parameters (they replay only from the server store), the pending action expires after ten
minutes or at the next message, and health reports the effective switch. Hermetic throughout: the
host environment must not turn the agent on behind the tests' backs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fusionlab import agent, api_agent, mast, protocol
from fusionlab.api import app
from tests.agent_fakes import FakeClient, parse_sse, text_block, tool_use_block

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[1]
USD_OUT = REPO_ROOT / "out" / "mast_30420.usda"   # the export route's published path — the driver writes it, never the store


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """The agent starts every test off, and no pending action outlives a test."""
    for var in ("FUSIONLAB_AGENT_ENABLED", "FUSIONLAB_AGENT_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    api_agent.store.clear()
    yield
    api_agent.store.clear()


def enable_agent(monkeypatch: pytest.MonkeyPatch, script: list[tuple[list[str], list[dict[str, Any]]]]) -> FakeClient:
    """The enabled mode tests need: the env on, and the model seam scripted."""
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", "1")
    monkeypatch.setenv("FUSIONLAB_AGENT_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake = FakeClient(script)
    monkeypatch.setattr(agent, "_client", lambda key: fake)
    return fake


def chat(message: str, history: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """One POST /agent/chat turn, read back through the browser's eyes."""
    r = client.post("/agent/chat", json={"message": message, "history": history or []})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return parse_sse(r.text)


def propose_export(monkeypatch: pytest.MonkeyPatch) -> str:
    """A scripted turn that proposes an export; returns the pending action's id."""
    enable_agent(monkeypatch, [(["."], [tool_use_block("tu_1", "export_usd", {"shot_id": 30420})]),
                               (["Proposed."], [text_block("Proposed.")])])
    events = chat("export the shot")
    return next(e for e in events if e["type"] == "action_proposed")["action_id"]


# ---- the panel serving gate (blueprint verification row 1)


def test_disabled_page_carries_no_agent_surface():
    """With the agent off, `/` renders without the agent panel — not even a script tag — and
    /static/agent.js is not served. The offline page is the file on disk, byte for byte."""
    page = client.get("/")
    assert page.status_code == 200
    assert "agent" not in page.text.lower()
    assert client.get("/static/agent.js").status_code == 404


def test_enabled_page_injects_the_panel_script_and_serves_it(monkeypatch):
    """The enabled page's only delta is the agent.js tag (the panel itself is built client-side
    after /agent/health confirms), and the script itself is now served."""
    enable_agent(monkeypatch, [])
    page = client.get("/")
    assert page.status_code == 200
    assert '<script src="/static/agent.js" defer></script>' in page.text
    js = client.get("/static/agent.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"].lower()
    assert "/agent/health" in js.text   # the panel checks the documented gate before rendering


# ---- protocol accept: the reviewer's edits, or the draft as extracted


def _seed_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """One draft in a redirected protocols dir (the repo's data/ is never touched)."""
    pid = "p_" + "0" * 12
    monkeypatch.setattr(protocol, "PROTOCOLS_DIR", tmp_path / "protocols")
    protocol.save_draft(pid, protocol.Protocol(
        title="As extracted",
        source=protocol.ProtocolSource(kind="pasted", sha256="ab" * 32, extracted_by="test-model",
                                       extracted_at="2026-01-01T00:00:00Z"),
        steps=[protocol.ProtocolStep(index=0, actuator="p_nbi", nominal="P_nbi_MW 2.5", slider=0.5)],
        caveats=["Compared with MAST data."]))
    return pid


def test_accept_without_body_promotes_the_draft_as_extracted(tmp_path, monkeypatch):
    _seed_draft(tmp_path, monkeypatch)
    r = client.post(f"/agent/protocols/{'p_' + '0' * 12}/accept")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "accepted" and j["protocol"]["title"] == "As extracted"
    assert j["protocol"]["source"]["sha256"] == "ab" * 32
    assert client.post(f"/agent/protocols/{'p_' + '0' * 12}/accept").status_code == 404   # consumed


def test_accept_with_a_reviewed_body_applies_the_edits_and_keeps_provenance(tmp_path, monkeypatch):
    """The panel's review view sends the human's field edits; provenance stays server-built —
    the body has no source field to forge."""
    pid = _seed_draft(tmp_path, monkeypatch)
    r = client.post(f"/agent/protocols/{pid}/accept", json={
        "title": "Reviewed",
        "steps": [{"index": 0, "actuator": "n", "nominal": "n 0.8 1e20 m^-3", "slider": 0.8,
                   "t_start_s": 0.25, "note": "ramp held"}],
        "caveats": [],
    })
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["protocol"]["title"] == "Reviewed"
    assert j["protocol"]["steps"][0]["actuator"] == "n"
    assert j["protocol"]["source"]["sha256"] == "ab" * 32
    accepted = json.loads((tmp_path / "protocols" / f"{pid}.json").read_text())
    assert accepted["title"] == "Reviewed" and accepted["steps"][0]["actuator"] == "n"


def test_a_reviewed_body_needs_a_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "PROTOCOLS_DIR", tmp_path / "protocols")
    r = client.post(f"/agent/protocols/{'p_' + '0' * 12}/accept", json={"title": "X", "steps": [], "caveats": []})
    assert r.status_code == 404


# ---- health
def test_health_reports_the_effective_switch_and_model():
    assert client.get("/agent/health").json() == {"enabled": False, "model": "claude-opus-5-5"}


def test_health_reflects_the_live_config(monkeypatch):
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", "1")
    monkeypatch.setenv("FUSIONLAB_AGENT_MODEL", "claude-test-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert client.get("/agent/health").json() == {"enabled": True, "model": "claude-test-1"}


# ---- disabled mode
def test_disabled_chat_is_refused_before_any_model_call(monkeypatch):
    def must_not_construct(key: str | None) -> None:
        raise AssertionError("the disabled route must not construct a client")

    monkeypatch.setattr(agent, "_client", must_not_construct)
    r = client.post("/agent/chat", json={"message": "hi"})
    assert r.status_code == 409 and "FUSIONLAB_AGENT_ENABLED" in r.json()["detail"]


def test_the_app_import_never_loads_the_anthropic_sdk():
    """This module's imports ran first (api, agent, api_agent, the registry): none of them may have
    pulled the SDK at module level. The route-level proof is the disabled chat above."""
    assert "anthropic" not in sys.modules


def test_a_pristine_interpreter_imports_the_agent_surface_without_the_sdk():
    """Airtight version: a fresh interpreter importing the agent surface must not load the SDK —
    the import lives inside fusionlab.agent._client and nowhere else."""
    code = "import sys; import fusionlab.api_agent; assert 'anthropic' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr


# ---- the stream
def test_the_stream_parses_the_full_event_sequence(monkeypatch):
    script = [(["Let me ", "look."], [text_block("Looking."), tool_use_block("tu_1", "search_shots", {"q": "30421"})]),
              (["Found ", "one."], [text_block("Found one.")])]
    fake = enable_agent(monkeypatch, script)
    events = chat("find shot 30421")
    assert [e["type"] for e in events] == ["message_delta", "message_delta", "tool_call", "tool_result",
                                           "message_delta", "message_delta", "done"]
    assert events[2] == {"type": "tool_call", "tool": "search_shots", "args": {"q": "30421"}}
    assert events[3]["ok"] is True and events[3]["data"]["total"] == 1
    assert "".join(e["text"] for e in events if e["type"] == "message_delta") == "Let me look.Found one."
    assert fake.calls[0]["system"] == agent.SYSTEM_PROMPT   # the honesty rules reached the model


def test_a_model_failure_streams_an_error_event(monkeypatch):
    class Exploding:
        def __init__(self) -> None:
            self.messages = self

        def stream(self, **kwargs: Any) -> None:
            raise RuntimeError("api down")

    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(agent, "_client", lambda key: Exploding())
    events = chat("hi")
    assert events[-1]["type"] == "error" and "model call failed" in events[-1]["message"]


@pytest.mark.parametrize("history", [
    [{"role": "assistant", "content": "hello"}],                           # must start with the user
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],  # must alternate
    [{"role": "user", "content": "a"}],                                    # must end with the assistant
])
def test_malformed_history_is_422(monkeypatch, history):
    enable_agent(monkeypatch, [])
    r = client.post("/agent/chat", json={"message": "hi", "history": history})
    assert r.status_code == 422 and "history" in r.json()["detail"]


def test_an_empty_message_is_422():
    assert client.post("/agent/chat", json={"message": ""}).status_code == 422


# ---- the two-phase gate
def test_a_proposed_whatif_changes_nothing_until_confirmed(monkeypatch):
    """Acceptance: propose → nothing applied → confirm replays the server-stored math → one-shot."""
    pmax = mast.load_shot(30166)["meta"]["P_nbi_max_MW"]
    mw = round(0.6 * pmax, 4)
    enable_agent(monkeypatch, [(["."], [tool_use_block("tu_1", "run_whatif", {"shot_id": 30166, "p_nbi_MW": mw})]),
                               (["Proposed."], [text_block("Proposed.")])])
    events = chat("raise the beams to 3 MW")
    proposed = next(e for e in events if e["type"] == "action_proposed")
    assert proposed["tool"] == "run_whatif" and proposed["args"] == {"shot_id": 30166, "p_nbi_MW": mw}
    assert "× P_nbi_max" in proposed["math"] and proposed["expires_s"] == 600

    # a tampered client names the same action but supplies its own parameters: they are ignored —
    # the endpoint takes no body, the args replay only from the server store
    tampered = client.post(f"/agent/actions/{proposed['action_id']}/confirm",
                           json={"shot_id": 9999, "p_nbi_MW": 99.0, "apply": {"sliders": {"P_nbi": "1.3"}}})
    assert tampered.status_code == 200
    applied = tampered.json()
    assert applied["type"] == "action_applied" and applied["tool"] == "run_whatif"
    assert applied["args"] == {"shot_id": 30166, "p_nbi_MW": mw}
    assert float(applied["apply"]["sliders"]["P_nbi"]) == pytest.approx(0.6)

    assert client.post(f"/agent/actions/{proposed['action_id']}/confirm").status_code == 410  # one-shot


def test_a_proposed_export_writes_nothing_before_or_after_the_confirm_handshake(monkeypatch):
    """The registry never writes; confirm hands the driver the payload, and the write stays the
    route's own — nothing lands in out/ until the driver POSTs it."""
    USD_OUT.unlink(missing_ok=True)
    action_id = propose_export(monkeypatch)
    assert not USD_OUT.exists()
    r = client.post(f"/agent/actions/{action_id}/confirm")
    assert r.status_code == 200
    applied = r.json()
    assert applied["apply"] == {"download": {"method": "POST", "url": "/replay/30420/usd"}}
    assert not USD_OUT.exists()


def test_confirming_an_unknown_action_is_410():
    r = client.post("/agent/actions/a_nope/confirm")
    assert r.status_code == 410 and "unknown" in r.json()["detail"]


def test_a_new_message_expires_the_pending_action(monkeypatch):
    script = [(["."], [tool_use_block("tu_1", "export_usd", {"shot_id": 30420})]),
              (["Proposed."], [text_block("Proposed.")]),
              (["Hi again."], [text_block("Hi again.")])]
    enable_agent(monkeypatch, script)
    events = chat("export the shot")
    action_id = next(e for e in events if e["type"] == "action_proposed")["action_id"]
    chat("actually, never mind")
    r = client.post(f"/agent/actions/{action_id}/confirm")
    assert r.status_code == 410 and "unknown" in r.json()["detail"]


def test_an_expired_pending_action_confirms_410(monkeypatch):
    """Route-level TTL: the store's clock normally runs on the wall; here it runs on the test's hand."""
    now = [0.0]
    monkeypatch.setattr(api_agent, "store", api_agent.PendingActionStore(clock=lambda: now[0]))
    script = [(["."], [tool_use_block("tu_1", "export_usd", {"shot_id": 30420})]),
              (["Proposed."], [text_block("Proposed.")])]
    enable_agent(monkeypatch, script)
    events = chat("export the shot")
    action_id = next(e for e in events if e["type"] == "action_proposed")["action_id"]
    now[0] = 601.0
    r = client.post(f"/agent/actions/{action_id}/confirm")
    assert r.status_code == 410 and "expired" in r.json()["detail"]


# ---- the store, unit level
def test_the_store_holds_one_pending_action():
    s = api_agent.PendingActionStore()
    a = s.create({"tool": "run_whatif", "args": {}, "math": "", "apply": {"sliders": {}}})
    b = s.create({"tool": "export_usd", "args": {}, "math": "", "apply": {"download": {}}})
    with pytest.raises(api_agent.PendingActionError):
        s.confirm(a["action_id"])          # replaced by the newer proposal
    assert s.confirm(b["action_id"]).tool == "export_usd"


def test_pending_actions_expire_after_ten_minutes():
    now = [0.0]
    s = api_agent.PendingActionStore(clock=lambda: now[0])
    token = s.create({"tool": "export_usd", "args": {}, "math": "", "apply": {}})
    now[0] = 600.0                          # the TTL boundary is exclusive
    with pytest.raises(api_agent.PendingActionError, match="expired"):
        s.confirm(token["action_id"])
    now[0] = 0.0
    token = s.create({"tool": "export_usd", "args": {}, "math": "", "apply": {}})
    now[0] = 599.0
    assert s.confirm(token["action_id"]).action_id == token["action_id"]


def test_a_mismatched_id_does_not_destroy_the_pending_action():
    s = api_agent.PendingActionStore()
    token = s.create({"tool": "export_usd", "args": {}, "math": "", "apply": {}})
    with pytest.raises(api_agent.PendingActionError):
        s.confirm("a_other")
    assert s.confirm(token["action_id"]).action_id == token["action_id"]
