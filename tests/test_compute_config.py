"""The [compute] config surface (PR 3): fusionlab.toml at the repo root, FUSIONLAB_* env overrides on
top, and no credential field anywhere in the path (design §7-8). HARN-5 adds per-host profiles:
[compute.hosts.<name>] tables selected by FUSIONLAB_COMPUTE_HOST (env) or [compute] host, with the
single [compute.ssh] table kept as the degenerate case and the zero-config default unchanged."""

import tomllib
from pathlib import Path

import pytest

from fusionlab import compute

REPO_ROOT = Path(compute.__file__).resolve().parent.parent.parent
SSH_VARS = ("FUSIONLAB_SSH_HOST", "FUSIONLAB_SSH_REMOTE_DIR", "FUSIONLAB_SSH_AUTOSTART",
            "FUSIONLAB_SSH_PORT", "FUSIONLAB_SSH_REMOTE_PORT", "FUSIONLAB_COMPUTE_PROVIDER")
HOST_VARS = SSH_VARS + ("FUSIONLAB_COMPUTE_HOST",)
SSH_KEYS = {"host", "remote_dir", "autostart", "port", "remote_port"}
TWO_HOSTS_TOML = ('[compute.hosts.lab-a]\nhost = "box-a"\nremote_dir = "~/FusionLab"\n'
                  'autostart = true\nport = 8421\n'
                  '[compute.hosts.lab-b]\nhost = "box-b"\nremote_dir = "~/FusionLab"\n'
                  'autostart = false\nport = 8422\n')


@pytest.fixture()
def no_env(monkeypatch):
    for var in HOST_VARS:
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
    """The shipped example is valid TOML for exactly this surface, with no credential-shaped key —
    including every [compute.hosts.*] profile (HARN-5)."""
    data = tomllib.loads((REPO_ROOT / "fusionlab.toml.example").read_text())
    cfg = data["compute"]
    tables = [cfg["ssh"], *cfg.get("hosts", {}).values()]
    for table in tables:
        assert set(table) <= SSH_KEYS
        assert not any(word in key for key in table for word in ("key", "password", "secret", "token"))


# -- HARN-5: [compute.hosts.<name>] profiles ---------------------------------------------------


def test_two_profiles_resolve_to_distinct_option_sets(no_env, monkeypatch, tmp_path):
    """The acceptance pair: the same config file yields each box's own options, per selection."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text(TWO_HOSTS_TOML)
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    monkeypatch.setenv("FUSIONLAB_COMPUTE_HOST", "lab-a")
    a = compute.ssh_options()
    assert a == {"host": "box-a", "remote_dir": "~/FusionLab", "autostart": True, "local_port": 8421}
    monkeypatch.setenv("FUSIONLAB_COMPUTE_HOST", "lab-b")
    b = compute.ssh_options()
    assert b == {"host": "box-b", "remote_dir": "~/FusionLab", "autostart": False, "local_port": 8422}
    assert a != b


def test_env_overrides_beat_the_selected_profile(no_env, monkeypatch, tmp_path):
    """FUSIONLAB_SSH_* still wins, now over the profile's matching keys too."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text(TWO_HOSTS_TOML)
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    monkeypatch.setenv("FUSIONLAB_COMPUTE_HOST", "lab-a")
    monkeypatch.setenv("FUSIONLAB_SSH_HOST", "env-host")
    monkeypatch.setenv("FUSIONLAB_SSH_PORT", "1234")
    assert compute.ssh_options() == {"host": "env-host", "remote_dir": "~/FusionLab",
                                     "autostart": True, "local_port": 1234}


def test_toml_host_key_selects_a_profile_and_env_beats_it(no_env, monkeypatch, tmp_path):
    """[compute] host names a profile in the file; FUSIONLAB_COMPUTE_HOST overrides it."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute]\nhost = "lab-b"\n' + TWO_HOSTS_TOML)
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    assert compute.ssh_options()["host"] == "box-b"
    monkeypatch.setenv("FUSIONLAB_COMPUTE_HOST", "lab-a")
    assert compute.ssh_options()["host"] == "box-a"


def test_single_profile_needs_no_selector(no_env, monkeypatch, tmp_path):
    """One GPU box, no selection anywhere: the lone profile is unambiguous."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute.hosts.lab-a]\nhost = "box-a"\nport = 8421\n')
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    assert compute.ssh_options() == {"host": "box-a", "local_port": 8421}


def test_unknown_host_fails_loudly(no_env, monkeypatch, tmp_path):
    """A selector naming no profile is a loud ValueError that lists the known hosts — per
    get_provider's unknown-provider contract, never a silent fallback to another host."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text(TWO_HOSTS_TOML)
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    monkeypatch.setenv("FUSIONLAB_COMPUTE_HOST", "lab-c")
    with pytest.raises(ValueError, match=r"unknown compute host 'lab-c'.*lab-a.*lab-b"):
        compute.ssh_options()


def test_unknown_host_with_no_profiles_fails_loudly(no_env, monkeypatch, tmp_path):
    """Selecting a host when the file defines none says so, instead of an empty option set."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute]\nprovider = "ssh"\n')
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    monkeypatch.setenv("FUSIONLAB_COMPUTE_HOST", "lab-c")
    with pytest.raises(ValueError, match="unknown compute host 'lab-c'.*no \\[compute.hosts.\\*\\] profiles"):
        compute.ssh_options()


def test_several_profiles_without_a_selection_fail_loudly(no_env, monkeypatch, tmp_path):
    """Two boxes and no selector is ambiguous — an error, not a quiet pick."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text(TWO_HOSTS_TOML)
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    with pytest.raises(ValueError, match="no host is selected.*FUSIONLAB_COMPUTE_HOST"):
        compute.ssh_options()


def test_legacy_ssh_table_still_resolves_alone_and_wins_without_a_selector(no_env, monkeypatch, tmp_path):
    """The degenerate case is byte-identical to pre-HARN-5 behavior: alone it resolves; with
    profiles defined it still wins when nothing selects a host (existing configs keep working)."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text('[compute.ssh]\nhost = "legacy"\nport = 9000\n')
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    assert compute.ssh_options() == {"host": "legacy", "local_port": 9000}
    toml.write_text('[compute.ssh]\nhost = "legacy"\nport = 9000\n' + TWO_HOSTS_TOML)
    assert compute.ssh_options() == {"host": "legacy", "local_port": 9000}


def test_unknown_host_fails_at_get_provider(no_env, monkeypatch, tmp_path):
    """The loud contract holds end to end: get_provider() raises for an unknown host (asyncssh is
    only imported at connect time, so the ValueError surfaces even without the remote extra)."""
    toml = tmp_path / "fusionlab.toml"
    toml.write_text(TWO_HOSTS_TOML)
    monkeypatch.setattr(compute, "_TOML_PATH", toml)
    monkeypatch.setattr(compute, "_provider", None)   # get_provider caches one per process
    monkeypatch.setenv("FUSIONLAB_COMPUTE_PROVIDER", "ssh")
    monkeypatch.setenv("FUSIONLAB_COMPUTE_HOST", "lab-c")
    with pytest.raises(ValueError, match="unknown compute host 'lab-c'"):
        compute.get_provider()
