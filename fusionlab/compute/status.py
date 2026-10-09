"""Compute status and the local-fallback runner: what GET /compute reports and the UI chip and banner
render (design §7).

Three pieces:

- fallback state — recorded when the selected remote provider fails and ``run_task`` re-runs the job on
  LocalProvider; cleared when a remote job succeeds again. The banner and the chip read it.
- run_task / capabilities — the routers' one entry point for named tasks with fallback, plus the same
  for the capabilities handshake. Provenance — where each task last ran — is recorded alongside, so
  results can say "computed on <host>".
- snapshot — the GET /compute payload, rebuilt at most once per TTL with a single in-flight probe: an
  unreachable worker is a status ("fallback": true + reason), never a 5xx storm, and a slow probe never
  holds later polls hostage.

``fusionlab.compute.ssh`` is imported lazily everywhere here (it pulls in httpx, a ``remote``-extra
dependency): the default install never pays for it.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ProbeTimeout
from typing import Any

from fusionlab.compute import get_provider, provider_name
from fusionlab.compute.local import LocalProvider

logger = logging.getLogger(__name__)

TTL_S = 5.0              # the UI polls /compute at this cadence; one probe per TTL, not per request
PROBE_TIMEOUT_S = 10.0   # one status build never holds a request longer than this

_lock = threading.Lock()
_fallback_reason: str | None = None
_fallback_since: float | None = None
_last_runs: dict[str, dict[str, Any]] = {}       # task -> where it last ran: {"provider", "host"}
_cache: dict[str, Any] = {"at": 0.0, "payload": None}
_probing = False
# One worker thread serializes probes; a hung SSH connect therefore delays process exit at most one
# connect timeout, never one per poll.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fusionlab-compute-status")


def _remote_errors() -> tuple[type[Exception], ...]:
    """RemoteComputeError without importing ssh.py (and httpx) on the default install: the `remote`
    extra may simply not be here, in which case no remote provider can exist either."""
    try:
        from fusionlab.compute.ssh import RemoteComputeError
    except ImportError:
        return ()
    return (RemoteComputeError,)


# ---------------------------------------------------------------- the fallback record
def record_fallback(reason: str) -> None:
    """A remote run failed and its work ran locally instead — the banner's reason."""
    global _fallback_reason, _fallback_since
    with _lock:
        if _fallback_reason is None:
            _fallback_since = time.time()
        _fallback_reason = reason


def clear_fallback() -> None:
    """A remote run succeeded — the remote path works again."""
    global _fallback_reason, _fallback_since
    with _lock:
        _fallback_reason, _fallback_since = None, None


def fallback_state() -> dict[str, Any]:
    with _lock:
        return {"fallback": _fallback_reason is not None, "reason": _fallback_reason,
                "since": _fallback_since}


# ---------------------------------------------------------------- provenance
def provenance(task: str) -> dict[str, Any]:
    """Where `task` last ran: {"provider": "local"|"ssh", "host": ...}. Results label themselves
    "computed on <host>" from this — read it immediately after the run_task that produced the result."""
    with _lock:
        return dict(_last_runs.get(task, {"provider": "local", "host": None}))


def _record_run(task: str, provider: Any) -> None:
    with _lock:
        _last_runs[task] = {"provider": provider.name, "host": getattr(provider, "host", None)}


# ---------------------------------------------------------------- the routers' entry point
def run_task(task: str, inputs: dict) -> Any:
    """One named task through the selected provider, falling back to Local on a remote failure.

    A remote failure is recorded (the banner reads it) and the task re-runs in-process — the result the
    user asked for arrives either way, and provenance says honestly where it actually ran. The next call
    tries the remote again: a transient failure never pins the session to Local.
    """
    provider = get_provider()
    if provider.name != "local":
        try:
            out = provider.result(provider.submit(task, inputs))
        except _remote_errors() as e:
            record_fallback(str(e))
            logger.warning("remote compute failed; falling back to Local: %s", e)
        else:
            clear_fallback()
            _record_run(task, provider)
            return out
    local = LocalProvider()
    out = local.result(local.submit(task, inputs))
    _record_run(task, local)
    return out


def capabilities() -> dict:
    """The selected provider's capabilities (the design's version handshake), with the same fallback:
    a remote failure is recorded and Local's capabilities answer instead."""
    provider = get_provider()
    try:
        return provider.capabilities()
    except _remote_errors() as e:
        record_fallback(str(e))
        logger.warning("remote capabilities failed; falling back to Local: %s", e)
        return LocalProvider().capabilities()


# ---------------------------------------------------------------- the GET /compute payload
def _unknown_payload() -> dict[str, Any]:
    return {"provider": provider_name(), "host": None, "reachable": None, "worker": None,
            "version_mismatch": None, "fallback": False, "reason": None, "since": None}


def _build_snapshot() -> dict[str, Any]:
    """Fresh status: provider, liveness, the worker's handshake when configured — never raises. Caches
    its own payload (the timeout path in snapshot() needs the late result cached too). The fallback
    record is merged in; a probe that finds the worker healthy does NOT clear it (only a successful
    remote run does — see run_task)."""
    out = _unknown_payload()
    try:
        provider = get_provider()
    except ValueError as e:                      # a bad provider name is a status, not a 5xx
        out.update(reachable=False, fallback=True, reason=str(e))
        return out
    if provider.name == "local":
        out.update(provider="local", reachable=True)
    else:
        try:
            handshake = provider.capabilities()  # cached after the first connect; autostarts the worker
            provider.health()                    # a live probe through the tunnel, right now
        except _remote_errors() as e:
            out.update(provider=provider.name, reachable=False, reason=str(e))
        else:
            out.update(provider=provider.name, reachable=True, host=handshake.get("host"),
                       worker=handshake.get("remote"),
                       version_mismatch=handshake.get("version_mismatch"))
    recorded = fallback_state()
    out["fallback"] = recorded["fallback"] or out["reachable"] is False
    if recorded["fallback"]:
        out["reason"] = out["reason"] or recorded["reason"]
        out["since"] = recorded["since"]
    with _lock:
        _cache["at"], _cache["payload"] = time.monotonic(), out
    return out


def _probe_done(_future: Any) -> None:
    global _probing
    with _lock:
        _probing = False


def snapshot(max_age_s: float = TTL_S) -> dict[str, Any]:
    """The GET /compute payload, rebuilt at most once per `max_age_s` with at most one probe in flight.

    Later polls during a build serve the last known state (or a "probing" placeholder before the first);
    a build that outlives PROBE_TIMEOUT_S answers "unknown" and caches its real result when it lands.
    """
    global _probing
    with _lock:
        cached = _cache["payload"]
        if cached is not None and time.monotonic() - _cache["at"] < max_age_s:
            return dict(cached)
        if _probing:                             # one build at a time; latecomers never queue behind it
            if cached is not None:
                return dict(cached)
            out = _unknown_payload()
            out["reason"] = "status probe in flight"
            return out
        _probing = True
    future = _executor.submit(_build_snapshot)
    future.add_done_callback(_probe_done)
    try:
        return dict(future.result(timeout=PROBE_TIMEOUT_S))
    except ProbeTimeout:
        out = _unknown_payload()
        out["reason"] = f"status probe did not return within {PROBE_TIMEOUT_S:.0f}s"
        return out


def _reset_for_tests() -> None:
    """Clear the cache, the fallback record and the run provenance (pytest runs modules fresh)."""
    global _probing, _fallback_reason, _fallback_since
    with _lock:
        _cache.update({"at": 0.0, "payload": None})
        _probing = False
        _fallback_reason, _fallback_since = None, None
        _last_runs.clear()
