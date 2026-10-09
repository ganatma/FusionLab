"""Agent configuration (fusionlab/agent_config.py): defaults, the enabled-requires-key gate, import laziness."""

import subprocess
import sys
from pathlib import Path

import pytest

from fusionlab.agent_config import MAX_UPLOAD_BYTES, agent_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_agent_env(monkeypatch):
    """Start every test from a blank slate: none of the three agent vars set."""
    for var in ("FUSIONLAB_AGENT_ENABLED", "FUSIONLAB_AGENT_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_with_no_configuration():
    s = agent_settings()
    assert s.enabled is False                    # off by default — the offline demo never turns it on
    assert s.model == "claude-opus-5-5"
    assert s.api_key is None
    assert s.max_upload_bytes == 10 * 1024 * 1024 == MAX_UPLOAD_BYTES


@pytest.mark.parametrize("raw", ["1", "true", "yes", "TRUE", "Yes"])
def test_flag_alone_still_leaves_the_agent_off(monkeypatch, raw):
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", raw)
    assert agent_settings().enabled is False     # the effective switch also requires a key


def test_enabled_requires_both_flag_and_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert agent_settings().enabled is False     # key alone: off
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", "1")
    assert agent_settings().enabled is True      # flag + key: on


@pytest.mark.parametrize("raw", ["0", "false", "no", "", "enabled", "2"])
def test_anything_other_than_1_true_yes_is_off(monkeypatch, raw):
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", raw)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert agent_settings().enabled is False


def test_blank_key_is_no_key(monkeypatch):
    monkeypatch.setenv("FUSIONLAB_AGENT_ENABLED", "yes")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    s = agent_settings()
    assert s.api_key is None and s.enabled is False


def test_model_override(monkeypatch):
    monkeypatch.setenv("FUSIONLAB_AGENT_MODEL", "claude-sonnet-4-5")
    assert agent_settings().model == "claude-sonnet-4-5"


def test_importing_agent_config_never_imports_anthropic():
    """Laziness: the module must stay stdlib-only on import, even with anthropic installed.

    Runs in a subprocess because a fresh import is the only honest proof — and `find_spec` guards
    against passing vacuously when anthropic is missing rather than skipped.
    """
    code = (
        "import importlib.util, sys;"
        "import fusionlab.agent_config;"
        "assert importlib.util.find_spec('anthropic') is not None, 'anthropic not installed';"
        "print('anthropic' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    assert out.stdout.strip() == "False"
