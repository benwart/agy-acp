"""Live contract tests for `AgyAcpClientProvider` against installed `agy acp` subprocess."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import anyio
import anyio.abc
import pytest
from anyio.streams.buffered import BufferedByteReceiveStream

from agy_acp.logic.tool_classifier import FALLBACK_MODELS, resolve_agy_binary
from agy_acp.providers.agy_acp_client import AgyAcpClientProvider, format_acp_mcp_servers
from agy_acp.providers.state_store import SqliteStateStoreProvider

_STDIO_MCP_STUB_SOURCE = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

log_dir = os.environ.get("MCP_STUB_LOG_DIR")
if log_dir:
    out = pathlib.Path(log_dir) / f"stdio-{os.getpid()}.json"
    out.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "token": os.environ.get("KIROCREW_STUB_SESSION_TOKEN", ""),
            }
        ),
        encoding="utf-8",
    )

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    req_id = msg.get("id")
    method = msg.get("method", "")
    if req_id is None:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stdio-mcp-stub", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\\n")
    sys.stdout.flush()
"""


@dataclass(frozen=True)
class _FileDigestSnapshot:
    """Immutable snapshot of file existence, SHA-256 digest, and nanosecond mtime."""

    exists: bool
    sha256: str | None
    mtime_ns: int | None


def _snapshot_file(path: Path) -> _FileDigestSnapshot:
    """Capture existence, SHA-256 digest, and `st_mtime_ns` of `path`."""
    if not path.exists():
        return _FileDigestSnapshot(exists=False, sha256=None, mtime_ns=None)
    stat_res = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _FileDigestSnapshot(exists=True, sha256=digest, mtime_ns=stat_res.st_mtime_ns)


def _find_live_agy_binary() -> str | None:
    """Return absolute path to installed `agy` binary if available on host, else `None`."""
    saved_env = os.environ.get("AGY_BIN")
    try:
        os.environ.pop("AGY_BIN", None)
        candidate = resolve_agy_binary()
    finally:
        if saved_env is not None:
            os.environ["AGY_BIN"] = saved_env

    candidate_path = Path(candidate)
    if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
        return str(candidate_path)
    return shutil.which(candidate)


LIVE_AGY_BIN = _find_live_agy_binary()

pytestmark = pytest.mark.skipif(
    LIVE_AGY_BIN is None,
    reason="Real `agy` CLI executable is not installed on this host.",
)


def _is_pid_alive(pid: int) -> bool:
    """Return True if OS process `pid` still exists and is not reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_process_group_alive(pgid: int) -> bool:
    """Return True if any process in POSIX process group `pgid` still exists."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_pid_exit(pid: int, *, timeout: float = 5.0) -> None:
    """Poll until OS process `pid` raises `ProcessLookupError` or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return
        await anyio.sleep(0.05)
    pytest.fail(f"Process {pid} remained alive after {timeout:.1f}s")


async def _wait_for_process_group_exit(pgid: int, *, timeout: float = 5.0) -> None:
    """Poll until leader `pgid` and every child process in its POSIX group have exited."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_pid_alive(pgid) and not _is_process_group_alive(pgid):
            return
        await anyio.sleep(0.05)
    pytest.fail(
        f"Process group {pgid} still had live processes after {timeout:.1f}s "
        f"(leader_alive={_is_pid_alive(pgid)}, group_alive={_is_process_group_alive(pgid)})"
    )


def _write_stdio_mcp_stub(tmp_path: Path) -> tuple[Path, Path]:
    """Create an executable stdio MCP server stub and spawn-log directory in `tmp_path`."""
    log_dir = tmp_path / "mcp_stub_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_path / "stdio_mcp_stub.py"
    script_path.write_text(_STDIO_MCP_STUB_SOURCE, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script_path, log_dir


def _build_session_mcp_servers(
    *,
    stdio_script: Path,
    log_dir: Path,
    http_url: str,
    bearer_token: str,
) -> list[dict[str, Any]]:
    """Build per-session stdio + HTTP MCP server definitions."""
    return [
        {
            "name": "kirocrew-stdio-stub",
            "command": sys.executable,
            "args": [str(stdio_script)],
            "env": [{"name": "MCP_STUB_LOG_DIR", "value": str(log_dir)}],
        },
        {
            "type": "http",
            "name": "kirocrew-http-stub",
            "url": http_url,
            "headers": [{"name": "Authorization", "value": f"Bearer {bearer_token}"}],
        },
    ]


async def _write_http_response(
    stream: anyio.abc.ByteStream,
    status: str,
    body: bytes = b"",
) -> None:
    """Write a minimal HTTP/1.1 response with optional JSON payload."""
    content_type = "Content-Type: application/json\r\n" if body else ""
    header = (
        f"HTTP/1.1 {status}\r\n"
        f"{content_type}"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")
    await stream.send(header + body)


class _LocalHttpMcpStub:
    """Minimal loopback Streamable HTTP MCP endpoint for zero-network session isolation tests."""

    def __init__(self) -> None:
        self._stack: contextlib.AsyncExitStack | None = None
        self._serve_tg: anyio.abc.TaskGroup | None = None
        self.port: int = 0
        self.requests: list[tuple[str, dict[str, str]]] = []

    async def __aenter__(self) -> Self:
        self._stack = contextlib.AsyncExitStack()
        listener = await self._stack.enter_async_context(
            await anyio.create_tcp_listener(local_host="127.0.0.1", local_port=0)
        )
        self.port = int(listener.extra(anyio.abc.SocketAttribute.local_port))
        self._serve_tg = await self._stack.enter_async_context(anyio.create_task_group())
        self._serve_tg.start_soon(listener.serve, self._handle_client)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._serve_tg is not None:
            self._serve_tg.cancel_scope.cancel()
        if self._stack is not None:
            await self._stack.aclose()

    def url_for(self, slug: str) -> str:
        """Return loopback HTTP MCP endpoint URL for `slug`."""
        return f"http://127.0.0.1:{self.port}/mcp/{slug}"

    async def _handle_client(self, stream: anyio.abc.SocketStream) -> None:
        buffered = BufferedByteReceiveStream(stream)
        try:
            with anyio.fail_after(2.0):
                request_line = await buffered.receive_until(b"\n", 8192)
            if not request_line:
                return
            parts = request_line.decode("latin-1").strip().split(" ")
            http_method = parts[0].upper() if parts else "GET"
            req_path = parts[1] if len(parts) > 1 else "/"

            headers: dict[str, str] = {}
            while True:
                with anyio.fail_after(2.0):
                    hdr_line = await buffered.receive_until(b"\n", 8192)
                decoded = hdr_line.decode("latin-1").strip()
                if not decoded:
                    break
                if ":" in decoded:
                    key, val = decoded.split(":", 1)
                    headers[key.strip().lower()] = val.strip()

            self.requests.append((req_path, headers))
            content_len = int(headers.get("content-length", "0") or "0")
            if content_len > 0:
                with anyio.fail_after(2.0):
                    body = await buffered.receive_exactly(content_len)
            else:
                body = b""

            if http_method != "POST":
                await _write_http_response(stream, "405 Method Not Allowed")
                return

            payload: dict[str, Any] = json.loads(body.decode("utf-8")) if body else {}
            req_id = payload.get("id")
            if req_id is None:
                await _write_http_response(stream, "202 Accepted")
                return

            rpc_method = str(payload.get("method") or "")
            if rpc_method == "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "http-mcp-stub", "version": "1.0.0"},
                }
            elif rpc_method == "tools/list":
                result = {"tools": []}
            else:
                result = {}

            resp_bytes = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}).encode(
                "utf-8"
            )
            await _write_http_response(stream, "200 OK", resp_bytes)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()


async def test_live_agy_acp_initialize_and_discover_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `initialize` + `discover_models` against real `agy acp --stdio` and clean exit."""
    assert LIVE_AGY_BIN is not None
    monkeypatch.delenv("AGY_BIN", raising=False)

    provider = AgyAcpClientProvider(bin_path=LIVE_AGY_BIN, save_dir=tmp_path)
    worker_pid: int | None = None
    try:
        models = await provider.discover_models(cwd=str(tmp_path), refresh=True)
        worker_pid = provider.worker_pid

        assert worker_pid is not None and worker_pid > 0
        assert _is_pid_alive(worker_pid)

        assert len(models) > 0
        fallback_ids = {m["modelId"] for m in FALLBACK_MODELS}
        state_store: SqliteStateStoreProvider = provider.state_store
        assert {m["modelId"] for m in models} != fallback_ids or state_store.load_models() != []
        for entry in models:
            assert isinstance(entry.get("modelId"), str) and entry["modelId"].strip()
            assert isinstance(entry.get("name"), str) and entry["name"].strip()

        cached = state_store.load_models()
        assert [m["modelId"] for m in cached] == [m["modelId"] for m in models]
    finally:
        await provider.close()

    assert provider.worker_pid is None
    assert worker_pid is not None
    await _wait_for_process_group_exit(worker_pid)
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


async def test_live_agy_acp_per_session_mcp_isolation_without_touching_global_mcp_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify concurrent per-session MCP + mandate isolation leaves `mcp_config.json` untouched."""
    assert LIVE_AGY_BIN is not None
    monkeypatch.delenv("AGY_BIN", raising=False)

    global_mcp_config = Path.home() / ".gemini" / "config" / "mcp_config.json"
    before_snapshot = _snapshot_file(global_mcp_config)

    stdio_script, log_dir = _write_stdio_mcp_stub(tmp_path)
    cwd_alpha = tmp_path / "ws_alpha"
    cwd_beta = tmp_path / "ws_beta"
    cwd_alpha.mkdir()
    cwd_beta.mkdir()

    async with _LocalHttpMcpStub() as http_stub:
        mcp_alpha = _build_session_mcp_servers(
            stdio_script=stdio_script,
            log_dir=log_dir,
            http_url=http_stub.url_for("alpha"),
            bearer_token="http-alpha",
        )
        mcp_beta = _build_session_mcp_servers(
            stdio_script=stdio_script,
            log_dir=log_dir,
            http_url=http_stub.url_for("beta"),
            bearer_token="http-beta",
        )

        env_alpha = {"KIROCREW_STUB_SESSION_TOKEN": "tok-alpha"}
        env_beta = {"KIROCREW_STUB_SESSION_TOKEN": "tok-beta"}
        mandate_alpha = "[AGENT MANDATE] Session ALPHA routing via kirocrew-core::spawn_run."
        mandate_beta = "[AGENT MANDATE] Session BETA routing via kirocrew-core::spawn_run."

        formatted_alpha = format_acp_mcp_servers(mcp_alpha, session_env=env_alpha)
        formatted_beta = format_acp_mcp_servers(mcp_beta, session_env=env_beta)
        assert formatted_alpha != formatted_beta

        worker_pid: int | None = None
        async with AgyAcpClientProvider(bin_path=LIVE_AGY_BIN, save_dir=tmp_path) as provider:
            results: dict[str, dict[str, Any]] = {}

            async def _create_one(
                key: str,
                ws: Path,
                mcp: list[dict[str, Any]],
                mandate: str,
                s_env: dict[str, str],
            ) -> None:
                results[key] = await provider.create_backend_session(
                    cwd=str(ws),
                    mcp_servers=mcp,
                    system_prompt_append=mandate,
                    session_env=s_env,
                )

            async with anyio.create_task_group() as tg:
                tg.start_soon(_create_one, "alpha", cwd_alpha, mcp_alpha, mandate_alpha, env_alpha)
                tg.start_soon(_create_one, "beta", cwd_beta, mcp_beta, mandate_beta, env_beta)

            session_alpha = results["alpha"]
            session_beta = results["beta"]
            worker_pid = provider.worker_pid
            assert worker_pid is not None and _is_pid_alive(worker_pid)

            sid_alpha = str(session_alpha["agySessionId"])
            sid_beta = str(session_beta["agySessionId"])
            assert sid_alpha and sid_beta
            assert sid_alpha != sid_beta
            assert len(session_alpha["models"]) > 0
            assert len(session_beta["models"]) > 0

            await provider.set_backend_config_option(sid_alpha, "thought_level", "low")
            await provider.set_backend_config_option(sid_alpha, "mode", "plan")

            reloaded_alpha = await provider.load_backend_session(
                agy_session_id=sid_alpha,
                cwd=str(cwd_alpha),
                mcp_servers=mcp_alpha,
                system_prompt_append=mandate_alpha,
                mode="plan",
                effort="low",
                session_env=env_alpha,
            )
            reloaded_sid = str(reloaded_alpha["agySessionId"])
            assert reloaded_sid

            await provider.close_backend_session(reloaded_sid)
            if reloaded_sid != sid_alpha:
                await provider.close_backend_session(sid_alpha)
            await provider.close_backend_session(sid_beta)

        assert worker_pid is not None
        await _wait_for_process_group_exit(worker_pid)

    spawned_logs = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(log_dir.glob("stdio-*.json"))
    ]
    if spawned_logs:
        observed_tokens = {str(entry.get("token") or "") for entry in spawned_logs}
        assert {"tok-alpha", "tok-beta"}.issubset(observed_tokens)
        for entry in spawned_logs:
            await _wait_for_pid_exit(int(entry["pid"]))

    after_snapshot = _snapshot_file(global_mcp_config)
    assert after_snapshot == before_snapshot


async def test_live_agy_acp_worker_crash_recovery_and_zero_orphan_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `SIGKILL` crash auto-reconnects transparently and `close()` leaves zero orphans."""
    assert LIVE_AGY_BIN is not None
    monkeypatch.delenv("AGY_BIN", raising=False)

    provider = AgyAcpClientProvider(bin_path=LIVE_AGY_BIN, save_dir=tmp_path)
    pid1: int | None = None
    pid2: int | None = None
    try:
        session1 = await provider.create_backend_session(cwd=str(tmp_path))
        sid1 = str(session1["agySessionId"])
        assert sid1

        pid1 = provider.worker_pid
        assert pid1 is not None and pid1 > 0
        assert _is_pid_alive(pid1)

        os.kill(pid1, signal.SIGKILL)
        await _wait_for_pid_exit(pid1)

        session2 = await provider.create_backend_session(cwd=str(tmp_path))
        sid2 = str(session2["agySessionId"])
        assert sid2 and sid2 != sid1

        pid2 = provider.worker_pid
        assert pid2 is not None and pid2 > 0
        assert pid2 != pid1
        assert _is_pid_alive(pid2)

        await provider.close_backend_session(sid2)
    finally:
        await provider.close()

    assert provider.worker_pid is None
    assert pid1 is not None and pid2 is not None
    await _wait_for_process_group_exit(pid1)
    await _wait_for_process_group_exit(pid2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid2, 0)
