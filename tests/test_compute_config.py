"""The [compute] config surface (PR 3): fusionlab.toml at the repo root, FUSIONLAB_* env overrides on
top, and no credential field anywhere in the path (design §7-8)."""

import tomllib
from pathlib import Path

import pytest

from fusionlab import compute

REPO_ROOT = Path(compute.__file__).resolve().parent.parent.parent
SSH_VARS = ("FUSIONLAB_SSH_HOST", "FUSIONLAB_SSH_REMOTE_DIR", "FUSIONLAB_SSH_AUTOSTART",
            "FUSIONLAB_SSH_PORT", "FUSIONLAB_SSH_REMOTE_PORT", "FUSIONLAB_COMPUTE_PROVIDER")


@pytest.fixture()
def no_env(monkeypatch):
    for var in SSH_VARS:
        monkeypatch.delenv(var, raising=False)


def test_toml_path_is_the_repo_root():
    """The user edits one config file beside .env — never a path inside the package."""
    assert compute._TOML_PATH == REPO_ROOT / "fusionlab.toml"


def test_ssh_options_default_empty(no_env, monkeypatch):
    monkeypatch.setattr(compute, "_TOML_PATH", REPO_ROOT / "no-such-fusionlab.toml")
    assert compute.ssh_options() == {}


def test_toml_ssh_table_parsed_with_port_renamed(no_env, monkeypatch, tmp_path):
    """The toml spells the tunnel end `port` (design §7); SshProvider takes it as local_port."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute.ssh]\nhost = "ousaisrvr"\nremote_dir = "~/FusionLab"\n'
                    'autostart = false\nport = 9000\n')
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    assert compute.ssh_options() == {"host": "ousaisrvr", "remote_dir": "~/FusionLab",
                                     "autostart": False, "local_port": 9000}


def test_env_overrides_beat_the_toml(monkeypatch, tmp_path):
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute.ssh]\nhost = "toml-host"\nremote_dir = "~/FusionLab"\n'
                    'autostart = false\nport = 9000\n')
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    monkeypatch.setenv("FUSIONLAB_SSH_HOST", "env-host")
    monkeypatch.setenv("FUSIONLAB_SSH_AUTOSTART", "on")
    monkeypatch.setenv("FUSIONLAB_SSH_PORT", "1234")
    assert compute.ssh_options() == {"host": "env-host", "remote_dir": "~/FusionLab",
                                     "autostart": True, "local_port": 1234}


def test_example_config_parses_and_holds_no_credentials():
    """The shipped example is valid TOML for exactly this surface, with no credential-shaped key."""
    data = tomllib.loads((REPO_ROOT / "fusionlab.toml.example").read_text())
    ssh = data["compute"]["ssh"]
    assert set(ssh) <= {"host", "remote_dir", "autostart", "port", "remote_port"}
    assert not any(word in key for key in ssh for word in ("key", "password", "secret", "token"))
