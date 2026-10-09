"""SshProvider: run the same named tasks on the user's own GPU host over SSH (design outline §3, §5-8).

The backend connects with asyncssh — which honors the user's `~/.ssh/config`, agent, and keys, so
FusionLab never sees or stores a credential — opens a local port-forward to the worker, and talks
plain HTTP over the tunnel:

    FastAPI router (sync def, threadpool) ── SshProvider ── SSH (asyncssh) ── fusionlab worker
                                                │                on the user's host (binds 127.0.0.1)
                                                └─ local forward 127.0.0.1:<port> → worker

Wiring, with the SSH part as small as possible:

- ``_SshSession`` owns the only async work (connect, forward, remote autostart) on a private event
  loop in a thread. The routers are sync ``def`` (threadpool), so the blocking protocol methods
  never touch the API's event loop; PR-1's calling convention is unchanged.
- ``SshProvider`` implements the PR-1 protocol: ``submit`` queues the job and returns at once,
  ``result`` waits for the terminal state and decodes the FLF1 payload, ``stream`` relays the
  worker's SSE events to async consumers (PR 3's progress chip). Every failure — unreachable host,
  dead tunnel, busy worker, failed task — raises :class:`RemoteComputeError`, the one typed error
  the caller catches to fall back to Local; PR 3 renders the banner.
- Artifacts: a job references trained weights by sha256 (the worker's resolve_artifacts enforces
  that remotely). Before submitting, the provider compares local ``models/*.pt`` hashes with the
  worker's handshake and uploads only what is missing — weight traffic is a one-time cost per hash
  and version drift is visible instead of silent.

A session stub (anything with ``ensure``/``close``/``base_url``) replaces the SSH layer for tests —
the loopback end-to-end test runs this provider against a real worker on 127.0.0.1 with no SSH.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import httpx

from fusionlab import frames
from fusionlab.compute.provider import JobHandle

logger = logging.getLogger("fusionlab.compute")

DEFAULT_REMOTE_PORT = 8420          # the worker's loopback port on the remote host
DEFAULT_LOCAL_PORT = 8420           # the local end of the tunnel
DEFAULT_START_TIMEOUT_S = 90.0      # remote autostart: uv boot + torch import on a cold box
DEFAULT_RESULT_TIMEOUT_S = 1800.0   # a full benchmark on a slow box; the poll loop itself is cheap
_POLL_S = 0.25
_TERMINAL_EVENTS = ("completed", "error", "cancelled")


class RemoteComputeError(RuntimeError):
    """Any SshProvider failure the caller should handle by falling back to LocalProvider.

    Carries the phase (connect / submit / job / collect) and, when known, the remote task's own
    error text — the router never inspects SSH internals, and PR 3 renders this message as-is.
    """

    def __init__(self, phase: str, message: str):
        super().__init__(f"remote compute unavailable ({phase}): {message}")
        self.phase = phase
        self.message = message


class WorkerSession(Protocol):
    """What SshProvider needs from a transport: a loopback HTTP base_url that answers after ensure()."""

    base_url: str

    def ensure(self) -> None: ...   # blocking; idempotent; raises RemoteComputeError
    def close(self) -> None: ...


# ---------------------------------------------------------------- the real session (asyncssh)
class _LoopThread:
    """A private event loop in a thread — asyncssh calls are marshalled onto it and awaited."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fusionlab-ssh", daemon=True)
        self._thread.start()
        self._started.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro, timeout: float | None = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def close(self) -> None:
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


class _SshSession:
    """One SSH connection, multiplexed; the local end of the tunnel; remote autostart (design §5)."""

    def __init__(self, host: str, *, remote_dir: str = "~/FusionLab", autostart: bool = True,
                 local_port: int = DEFAULT_LOCAL_PORT, remote_port: int = DEFAULT_REMOTE_PORT,
                 connect_timeout_s: float = 30.0, start_timeout_s: float = DEFAULT_START_TIMEOUT_S) -> None:
        self.host = host
        self.remote_dir = remote_dir
        self.autostart = autostart
        self.local_port = local_port
        self.remote_port = remote_port
        self.connect_timeout_s = connect_timeout_s
        self.start_timeout_s = start_timeout_s
        self.base_url = f"http://127.0.0.1:{local_port}"
        self._loop = _LoopThread()
        self._conn: Any = None          # asyncssh.SSHClientConnection once connected
        self._listener: Any = None      # asyncssh.SSHListener once forwarded
        self._lock = threading.Lock()

    # -- the blocking surface SshProvider uses
    def ensure(self) -> None:
        """Connect + tunnel + worker reachability, once. Thread-safe; raises RemoteComputeError."""
        with self._lock:
            if self._listener is not None:
                return
            try:
                self._loop.run(self._ensure_async(),
                               timeout=self.connect_timeout_s + self.start_timeout_s + 30)
            except RemoteComputeError:
                raise
            except Exception as e:   # asyncssh errors, timeouts, DNS — one typed error for the caller
                self._teardown()
                raise RemoteComputeError("connect", f"{type(e).__name__}: {e}") from None

    def close(self) -> None:
        with self._lock:
            self._teardown()

    def _teardown(self) -> None:
        listener, conn, self._listener, self._conn = self._listener, self._conn, None, None
        for closer in (lambda: listener.close(), lambda: conn.close()):
            with contextlib.suppress(Exception):   # a dead tunnel is not a teardown failure worth raising
                closer()
        self._loop.close()

    # -- the asyncssh parts (the only coroutines in this module that touch SSH)
    async def _ensure_async(self) -> None:
        import asyncssh  # optional extra: `uv sync --extra remote`

        try:
            self._conn = await asyncssh.connect(self.host)   # honors ~/.ssh/config, agent, keys
        except Exception as e:
            raise RemoteComputeError("connect", f"ssh to {self.host!r} failed: {e}") from None
        self._listener = await self._conn.forward_local_port("127.0.0.1", self.local_port,
                                                             "127.0.0.1", self.remote_port)
        await self._wait_worker()

    async def _wait_worker(self) -> None:
        """Reach /health through the tunnel, autostarting the worker once if allowed to."""
        deadline = time.monotonic() + self.start_timeout_s
        started = False
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200 and r.json().get("ok"):
                    return
            except httpx.HTTPError:
                pass
            if self.autostart and not started:
                started = True
                await self._autostart()
            elif not self.autostart:
                raise RemoteComputeError(
                    "connect", f"no fusionlab worker on {self.host}:{self.remote_port} and autostart "
                               "is off — start it with: cd <remote_dir> && uv run fusionlab-worker")
            if time.monotonic() > deadline:
                raise RemoteComputeError(
                    "connect", f"worker on {self.host}:{self.remote_port} did not come up within "
                               f"{self.start_timeout_s:.0f}s — see .fusionlab-worker.log in {self.remote_dir}")
            await asyncio.sleep(0.5)

    async def _autostart(self) -> None:
        """`cd <remote_dir> && uv run fusionlab-worker`, detached under nohup; the poll loop waits.

        ~/.local/bin is prepended because a non-interactive SSH session often lacks it and uv
        installs there. The repo clone + `uv sync` are the documented prerequisites (design §1).
        """
        assert self._conn is not None
        cmd = (f'export PATH="$HOME/.local/bin:$PATH"; cd {self.remote_dir} && '
               f"nohup uv run fusionlab-worker --port {self.remote_port} >> .fusionlab-worker.log 2>&1 &")
        result = await self._conn.run(cmd)
        if result.exit_status != 0:
            detail = (result.stderr.strip() or result.stdout.strip())[:300]
            raise RemoteComputeError("connect", f"autostart failed on {self.host}: {detail}")


# ---------------------------------------------------------------- the provider
# Which shipped weight each task reads, by filename in models/ (the worker's artifact roles).
_TASK_WEIGHTS: dict[str, tuple[str, ...]] = {
    "run_surrogate": ("surrogate.pt",),
    "run_surrogate_shot": ("surrogate.pt",),
    "run_eq_surrogate": ("eq_surrogate.pt",),
}


def local_weight_paths() -> dict[str, Path]:
    """The trained weights of THIS install, by filename — the hashes the worker must match or receive."""
    from fusionlab import (  # module attrs; no torch import needed
        eq_surrogate,
        surrogate,
    )

    return {"surrogate.pt": Path(surrogate.MODEL_FILE), "eq_surrogate.pt": Path(eq_surrogate.MODEL_FILE)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_version() -> str:
    try:
        return importlib.metadata.version("fusionlab")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0+unknown"


class SshProvider:
    """ComputeProvider over an SSH tunnel to a fusionlab worker on the user's own GPU host."""

    name = "ssh"

    def __init__(self, session: WorkerSession | None = None, *,
                 result_timeout_s: float = DEFAULT_RESULT_TIMEOUT_S, **session_kw: Any) -> None:
        if session is not None:
            if session_kw:
                raise TypeError(f"session is given, but so are connection options: {sorted(session_kw)}")
            self._session: WorkerSession = session
        else:
            host = session_kw.pop("host", None)
            if not host:
                raise ValueError("SshProvider needs an SSH host (a ~/.ssh/config alias works) — "
                                 "set [compute.ssh] host in fusionlab.toml or FUSIONLAB_SSH_HOST")
            self._session = _SshSession(host, **session_kw)
        self._result_timeout_s = result_timeout_s
        self._remote: dict[str, Any] | None = None   # the worker's last handshake
        self._http = httpx.Client(base_url=self._session.base_url, timeout=30.0)

    # -- ComputeProvider protocol
    def capabilities(self) -> dict:
        """The worker's handshake, merged under the keys the routers read:
        surrogate_available / eq_surrogate_available describe the REMOTE host now. A version drift
        against this install is a logged warning carried in version_mismatch — never silent."""
        remote = self._handshake()
        remote_version, here = remote.get("worker_version", ""), _local_version()
        mismatch = {"local": here, "remote": remote_version} if remote_version != here else None
        if mismatch:
            logger.warning("fusionlab worker on %s runs version %s but this install is %s — "
                           "pull the same commit on both ends", self._session.base_url, remote_version, here)
        models = {info["sha256"]: name for name, info in remote.get("models", {}).items()}
        return {"provider": self.name, "host": getattr(self._session, "host", "127.0.0.1"),
                "remote": remote, "version_mismatch": mismatch, "tasks": remote.get("tasks", []),
                "surrogate_available": "surrogate.pt" in models,
                "eq_surrogate_available": "eq_surrogate.pt" in models}

    def submit(self, task: str, inputs: dict) -> JobHandle:
        """Encode the inputs, satisfy the artifact refs, queue the job remotely. Returns at once;
        result() collects. A busy worker is a clean RemoteComputeError, not a hidden queue."""
        try:
            payload = {"task": task, "inputs": frames.encode_inputs(inputs),
                       "artifact_refs": self._ensure_artifacts(task)}
            r = self._http.post("/v1/jobs", json=payload)
            if r.status_code == 409:
                raise RemoteComputeError("submit", r.json().get("detail", "worker busy"))
            if r.status_code == 422:
                raise RemoteComputeError("submit", f"worker rejected the job: {r.json().get('detail', r.text)[:300]}")
            r.raise_for_status()
            job_id = r.json()["job_id"]
        except httpx.HTTPError as e:
            raise RemoteComputeError("submit", f"{type(e).__name__}: {e}") from None
        except frames.FrameError as e:
            raise RemoteComputeError("submit", f"could not encode job inputs: {e}") from None
        return JobHandle(job_id=job_id, task=task, provider=self.name)

    async def stream(self, job: JobHandle) -> AsyncIterator[dict]:
        """Relay the worker's SSE events for one job onto the caller's loop (PR 3's progress chip)."""
        try:
            async with httpx.AsyncClient(base_url=self._session.base_url, timeout=None) as client, \
                    client.stream("GET", f"/v1/jobs/{job.job_id}/events") as resp:
                if resp.status_code == 404:
                    raise RemoteComputeError("job", f"unknown job {job.job_id} (worker restarted?)")
                resp.raise_for_status()
                event: dict = {}
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        event = {"event": line[len("event: "):]}
                    elif line.startswith("data: ") and event:
                        event.update(json.loads(line[len("data: "):]))
                        yield event
                        if event["event"] in _TERMINAL_EVENTS:
                            return
        except httpx.HTTPError as e:
            raise RemoteComputeError("job", f"event stream broke: {type(e).__name__}: {e}") from None

    def result(self, job: JobHandle) -> Any:
        """Wait for the terminal state, then decode the FLF1 payload once. A remote task failure
        raises RemoteComputeError carrying the worker's own error text — the local-fallback signal."""
        deadline = time.monotonic() + self._result_timeout_s
        while True:
            try:
                r = self._http.get(f"/v1/jobs/{job.job_id}")
            except httpx.HTTPError as e:
                raise RemoteComputeError("collect", f"{type(e).__name__}: {e}") from None
            if r.status_code == 404:
                raise RemoteComputeError("collect", f"unknown job {job.job_id} (worker restarted?)")
            r.raise_for_status()
            state = r.json()["state"]
            if state == "error":
                raise RemoteComputeError("job", f"remote task failed: {r.json().get('error', 'unknown error')}")
            if state == "cancelled":
                raise RemoteComputeError("job", "remote job was cancelled")
            if state == "completed":
                break
            if time.monotonic() > deadline:
                raise RemoteComputeError("collect", f"job {job.job_id} still {state} after "
                                                    f"{self._result_timeout_s:.0f}s")
            time.sleep(_POLL_S)
        try:
            rr = self._http.get(f"/v1/jobs/{job.job_id}/result")
        except httpx.HTTPError as e:
            raise RemoteComputeError("collect", f"{type(e).__name__}: {e}") from None
        if rr.status_code != 200:
            raise RemoteComputeError("collect", f"result fetch failed: HTTP {rr.status_code}")
        try:
            return frames.unpack_result(rr.content)
        except frames.FrameError as e:
            raise RemoteComputeError("collect", f"corrupt result envelope: {e}") from None

    # -- internals
    def _handshake(self) -> dict:
        if self._remote is None:
            try:
                self._session.ensure()
                r = self._http.get("/v1/capabilities")
                r.raise_for_status()
                self._remote = r.json()
            except httpx.HTTPError as e:
                raise RemoteComputeError("connect", f"handshake failed: {type(e).__name__}: {e}") from None
        return self._remote

    def _ensure_artifacts(self, task: str) -> list[dict[str, str]]:
        """Refs for the weights this task reads: reuse the remote cache or shipped copy, upload only
        what is missing. A local weight that is gone entirely is a clean RemoteComputeError — the
        routers' availability guard already keeps most such jobs from being submitted."""
        remote = self._handshake()
        cached: dict[str, int] = remote.get("artifacts", {})
        shipped: dict[str, dict[str, Any]] = remote.get("models", {})
        refs: list[dict[str, str]] = []
        for filename in _TASK_WEIGHTS.get(task, ()):
            path = local_weight_paths().get(filename)
            if path is None or not path.exists():
                raise RemoteComputeError("submit", f"local weight models/{filename} is missing — "
                                                   f"train it (make train) before running {task!r} remotely")
            digest = sha256_file(path)
            if digest not in cached and shipped.get(filename, {}).get("sha256") != digest:
                self._upload(path, digest)
            refs.append({"sha256": digest, "role": filename})
        return refs

    def _upload(self, path: Path, digest: str) -> None:
        try:
            r = self._http.post(f"/v1/artifacts/{digest}", content=path.read_bytes())
            if r.status_code != 201:
                raise RemoteComputeError("submit", f"weight upload failed: HTTP {r.status_code} {r.text[:200]}")
            logger.info("uploaded models/%s (%d bytes) to the worker cache — one-time per hash",
                        path.name, path.stat().st_size)
        except httpx.HTTPError as e:
            raise RemoteComputeError("submit", f"weight upload failed: {type(e).__name__}: {e}") from None

    def close(self) -> None:
        """Drop the HTTP client, the tunnel, and the SSH connection (tests; a server process just exits)."""
        self._http.close()
        self._session.close()
        self._remote = None
