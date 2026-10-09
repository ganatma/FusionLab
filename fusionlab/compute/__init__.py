"""Compute selection: the API imports this package and never torch or Warp.

get_provider() returns the backend named by FUSIONLAB_COMPUTE_PROVIDER or by the optional fusionlab.toml
at the repo root, beside .env ([compute] provider = "local"), and defaults to LocalProvider — today's
in-process behavior, so a bare checkout with no config runs exactly as before. SshProvider (PR 2) is
registered behind a lazy import:
the default install never pulls asyncssh; selecting ssh requires `uv sync --extra remote` and a
[compute.ssh] host (or FUSIONLAB_SSH_HOST) — keys and passwords never appear in config, by design.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fusionlab.compute.local import LocalProvider
from fusionlab.compute.provider import ComputeProvider, JobHandle

__all__ = ["ComputeProvider", "JobHandle", "LocalProvider", "get_provider", "provider_name",
           "ssh_options", "warm_up"]

DEFAULT_PROVIDER = "local"
_TOML_PATH = Path(__file__).resolve().parent.parent.parent / "fusionlab.toml"   # repo root, beside .env; the env var wins

_SSH_ENV_VARS: dict[str, str] = {   # FUSIONLAB_SSH_* overrides; keys map to SshProvider/_SshSession kwargs
    "host": "FUSIONLAB_SSH_HOST",
    "remote_dir": "FUSIONLAB_SSH_REMOTE_DIR",
    "autostart": "FUSIONLAB_SSH_AUTOSTART",
    "local_port": "FUSIONLAB_SSH_PORT",          # fusionlab.toml spells it `port` (design §7): the local tunnel end
    "remote_port": "FUSIONLAB_SSH_REMOTE_PORT",
}


def ssh_options() -> dict[str, Any]:
    """The [compute.ssh] table from fusionlab.toml, merged with FUSIONLAB_SSH_* env overrides (env wins).

    Returns kwargs ready for SshProvider(**...). Never contains credentials — the connection uses the
    user's own SSH agent, keys, and ~/.ssh/config (design §8).
    """
    opts: dict[str, Any] = {}
    if _TOML_PATH.exists():
        with _TOML_PATH.open("rb") as f:
            opts.update(tomllib.load(f).get("compute", {}).get("ssh", {}))
    if "port" in opts:   # the design's toml key names the local end of the tunnel
        opts["local_port"] = opts.pop("port")
    for key, var in _SSH_ENV_VARS.items():
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        if key == "autostart":
            opts[key] = raw.lower() in ("1", "true", "yes", "on")
        elif key in ("local_port", "remote_port"):
            opts[key] = int(raw)
        else:
            opts[key] = raw
    return opts


def _ssh_factory() -> ComputeProvider:
    """Build SshProvider on demand — the only place asyncssh is imported from this package (lazy,
    so the default install gains no dependency)."""
    from fusionlab.compute.ssh import SshProvider

    return SshProvider(**ssh_options())


_PROVIDERS: dict[str, Callable[[], ComputeProvider]] = {"local": LocalProvider, "ssh": _ssh_factory}
_provider: ComputeProvider | None = None   # one instance per process: job handles live on it


def provider_name() -> str:
    """The configured provider: FUSIONLAB_COMPUTE_PROVIDER, else [compute] provider in fusionlab.toml, else "local"."""
    env = os.environ.get("FUSIONLAB_COMPUTE_PROVIDER", "").strip()
    if env:
        return env
    if _TOML_PATH.exists():
        with _TOML_PATH.open("rb") as f:
            compute = tomllib.load(f).get("compute", {})
        if compute.get("provider"):
            return str(compute["provider"])
    return DEFAULT_PROVIDER


def get_provider() -> ComputeProvider:
    """The selected provider, constructed on first use and cached. Unknown names fail loudly here, never
    by silently falling back to local."""
    global _provider
    if _provider is None:
        name = provider_name()
        try:
            factory = _PROVIDERS[name]
        except KeyError:
            raise ValueError(f"unknown compute provider {name!r} (FUSIONLAB_COMPUTE_PROVIDER or [compute] provider); "
                             f"available: {sorted(_PROVIDERS)}") from None
        _provider = factory()
    return _provider


def warm_up() -> None:
    """Prime the selected provider at server start (api.py's lifespan). Local: import torch and load the
    trained correction so the first replay request does not pay for it. A remote provider (PR 2) warms on
    its own host, so there is nothing to prime in this process."""
    provider = get_provider()
    if isinstance(provider, LocalProvider):
        provider.warm_up()
