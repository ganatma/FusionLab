"""Compute selection: the API imports this package and never torch or Warp.

get_provider() returns the backend named by FUSIONLAB_COMPUTE_PROVIDER or by the optional fusionlab.toml
([compute] provider = "local"), and defaults to LocalProvider — today's in-process behavior, so a bare
checkout with no config runs exactly as before. PR 2 registers SshProvider in _PROVIDERS.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from pathlib import Path

from fusionlab.compute.local import LocalProvider
from fusionlab.compute.provider import ComputeProvider, JobHandle

__all__ = ["ComputeProvider", "JobHandle", "LocalProvider", "get_provider", "provider_name", "warm_up"]

DEFAULT_PROVIDER = "local"
_TOML_PATH = Path(__file__).resolve().parent.parent / "fusionlab.toml"   # optional; the env var wins

_PROVIDERS: dict[str, Callable[[], ComputeProvider]] = {"local": LocalProvider}
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
