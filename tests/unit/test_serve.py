"""Unit tests for `agy_acp.commands.serve` JSON-RPC stdio server resilience."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from agy_acp.commands.serve import normalize_agent_engine, run_stdio_server
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.logic.session_manager import SessionManager
from agy_acp.logic.tool_classifier import FALLBACK_MODELS, PromptOutcome, ToolCallEvent
from agy_acp.logic.wire import create_wire_adapter
from agy_acp.providers.agy_acp_client import AgyAcpClientProvider
from agy_acp.providers.quota_provider import QuotaProvider


class _ServeStubCliProvider(AgyAcpClientProvider):
    """Stub ACP client provider for exercising `serve.run_stdio_server` offline."""

    def __init__(
        self,
        *,
        save_dir: Path,
        permission_requested: anyio.Event | None = None,
        permission_outcomes: list[bool] | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self._permission_requested = permission_requested
        self._permission_outcomes = permission_outcomes

    async def discover_models(
        self,
        *,
        cwd: str = ".",
        refresh: bool = False,
    ) -> list[dict[str, str]]:
        _ = (cwd, refresh)
        await anyio.sleep(0)
        return [dict(m) for m in FALLBACK_MODELS]

    async def run_slash_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        session_id: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        _ = (cwd, session_id, timeout)
        if command.lstrip("/") == "usage":
            return {
                "conversation_id": "",
                "status": "SUCCESS",
                "command": {"name": "usage", "data": {"groups": []}},
            }
        return {
            "conversation_id": "",
            "status": "SUCCESS",
            "command": {"name": "credits", "data": {"remaining_credits": 100}},
        }

    async def stream_prompt(
        self,
        *,
        cwd: str,
        prompt: str = "",
        session_id: str | None = None,
        conversation_id: str | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
        **kwargs: Any,
    ) -> PromptOutcome:
        _ = (cwd, session_id, kwargs)
        if "CRASH_UNEXPECTED" in prompt:
            raise RuntimeError("boom inside provider")
        if "ASK_PERMISSION" in prompt and on_tool_call is not None:
            if self._permission_requested is not None:
                self._permission_requested.set()
            allowed = await on_tool_call(
                ToolCallEvent(
                    tool_call_id="tc_eof_1",
                    title="Run cmd",
                    kind="execute",
                    status="in_progress",
                    kas_builtin_id="run_command",
                    command="echo hi",
                    raw_input={"command": "echo hi"},
                )
            )
            if self._permission_outcomes is not None:
                self._permission_outcomes.append(allowed)
        return PromptOutcome(
            conversation_id=conversation_id or "conv-serve-1",
            stop_reason="end_turn",
            response="ok",
            usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )


def test_normalize_agent_engine_aliases_and_error() -> None:
    """Verify `normalize_agent_engine` maps v2/v3 aliases and rejects invalid values."""
    assert normalize_agent_engine(None) == "v2"
    assert normalize_agent_engine("kiro") == "v2"
    assert normalize_agent_engine("v3") == "v3"
    assert normalize_agent_engine("KAS") == "v3"
    with pytest.raises(ValueError, match="Unsupported --agent-engine"):
        normalize_agent_engine("v99")


async def test_serve_non_dict_frames_unhashable_rpc_responses_and_exception_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-dict JSON (-32600), unhashable response IDs, error containment, & log safety."""
    log_file = tmp_path / "acp.log"
    monkeypatch.setenv("AGY_ACP_LOG_FILE", str(log_file))

    cli = _ServeStubCliProvider(save_dir=tmp_path)
    adapter = create_wire_adapter("v3")
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    qt = QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=cli.state_store)

    send_in, recv_in = anyio.create_memory_object_stream[str](32)
    written_frames: list[dict[str, Any]] = []

    async def _readline() -> str:
        try:
            return await recv_in.receive()
        except anyio.EndOfStream:
            return ""

    async def _write_bytes(data: bytes) -> None:
        written_frames.append(json.loads(data.decode("utf-8")))

    # Queue malformed JSON containing a secret, non-dict JSON values, unhashable/unsolicited
    # client RPC responses, a crashing request, and a valid initialize + session/new + prompt!
    inputs = [
        '{"broken": "Bearer SECRET_RAW_LINE_TOKEN_999"\n',
        "[]\n",
        "null\n",
        "123\n",
        '"str"\n',
        json.dumps({"jsonrpc": "2.0", "id": [1, 2], "result": {}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": {"bad": 1}, "result": {}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": True, "result": {}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": "srv_unsolicited", "result": {}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n",
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
        + "\n",
    ]
    for item in inputs:
        await send_in.send(item)
    await send_in.aclose()

    with anyio.fail_after(5.0):
        rc = await run_stdio_server(
            session_manager=mgr,
            wire_adapter=adapter,
            quota_tracker=qt,
            save_dir=tmp_path,
            stdin_readline=_readline,
            stdout_write=_write_bytes,
        )
    assert rc == 0

    invalid_req_errors = [f for f in written_frames if f.get("error", {}).get("code") == -32600]
    assert len(invalid_req_errors) == 4
    init_frame = next(f for f in written_frames if f.get("id") == 1 and "result" in f)
    assert init_frame["result"]["agentCapabilities"]["_meta"]["kiro"] == {
        "extensionMethods": [
            "_kiro/session/compact",
            "_kiro/session/delete",
            "_kiro/mcp/resetServer",
            "_kiro/session/context",
            "_kiro/knowledge",
            "_kiro/codeIntelligence",
            "_kiro/checkpoint/revertMultiple",
        ],
        "checkpoints": True,
        "mcpServerOAuth": True,
        "mcpRegistryPointers": True,
        "strictSessionLoad": True,
    }
    new_frame = next(f for f in written_frames if f.get("id") == 2)
    sid = new_frame["result"]["sessionId"]

    # Now run a second session batch verifying uncaught RuntimeError maps to -32603 without
    # crashing the server task group, and subsequent requests still succeed!
    send_in2, recv_in2 = anyio.create_memory_object_stream[str](16)
    written_frames2: list[dict[str, Any]] = []

    async def _readline2() -> str:
        try:
            return await recv_in2.receive()
        except anyio.EndOfStream:
            return ""

    async def _write_bytes2(data: bytes) -> None:
        written_frames2.append(json.loads(data.decode("utf-8")))

    await send_in2.send(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "CRASH_UNEXPECTED"}],
                },
            }
        )
        + "\n"
    )
    await send_in2.send(
        json.dumps({"jsonrpc": "2.0", "id": 11, "method": "session/list", "params": {}}) + "\n"
    )
    await send_in2.aclose()

    with anyio.fail_after(5.0):
        rc2 = await run_stdio_server(
            session_manager=mgr,
            wire_adapter=adapter,
            quota_tracker=qt,
            save_dir=tmp_path,
            stdin_readline=_readline2,
            stdout_write=_write_bytes2,
        )
    assert rc2 == 0
    crash_resp = next(f for f in written_frames2 if f.get("id") == 10)
    assert crash_resp["error"]["code"] == -32603
    list_resp = next(f for f in written_frames2 if f.get("id") == 11)
    assert len(list_resp["result"]["sessions"]) == 1

    # Verify raw unredacted malformed line secret was NOT written into acp.log!
    log_text = log_file.read_text(encoding="utf-8")
    assert "SECRET_RAW_LINE_TOKEN_999" not in log_text


async def test_serve_stdin_eof_wakes_inflight_client_rpc_caller_without_hanging(
    tmp_path: Path,
) -> None:
    """Verify stdin EOF wakes pending `session/request_permission` RPCs and fails closed."""
    perm_outcomes: list[bool] = []
    cli = _ServeStubCliProvider(
        save_dir=tmp_path,
        permission_outcomes=perm_outcomes,
    )
    adapter = create_wire_adapter("v3")
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    qt = QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=cli.state_store)
    desc = await mgr.create_session(cwd=str(tmp_path), mcp_servers=[])
    sid = desc["sessionId"]

    send_in, recv_in = anyio.create_memory_object_stream[str](8)
    server_rpc_sent = anyio.Event()

    async def _readline() -> str:
        try:
            return await recv_in.receive()
        except anyio.EndOfStream:
            return ""

    async def _write_bytes(data: bytes) -> None:
        frame = json.loads(data.decode("utf-8"))
        if frame.get("method") == "session/request_permission":
            server_rpc_sent.set()

    async def _feed_and_eof() -> None:
        await send_in.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "ASK_PERMISSION"}],
                    },
                }
            )
            + "\n"
        )
        await server_rpc_sent.wait()
        # Close stdin WITHOUT answering session/request_permission!
        await send_in.aclose()

    with anyio.fail_after(3.0):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_feed_and_eof)
            rc = await run_stdio_server(
                session_manager=mgr,
                wire_adapter=adapter,
                quota_tracker=qt,
                save_dir=tmp_path,
                stdin_readline=_readline,
                stdout_write=_write_bytes,
            )
    assert rc == 0
    assert perm_outcomes == [False]


async def test_serve_default_write_bytes_offloads_to_thread_under_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_default_write_bytes` offloads blocking stdout writes via `to_thread` under lock."""
    import threading

    cli = _ServeStubCliProvider(save_dir=tmp_path)
    adapter = create_wire_adapter("v3")
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    qt = QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=cli.state_store)

    main_tid = threading.get_ident()
    first_write_entered = threading.Event()
    unblock_first_write = threading.Event()
    observed_tids: list[int] = []
    written_chunks: list[bytes] = []
    active_writers = 0
    max_concurrent_writers = 0
    state_guard = threading.Lock()

    class _BlockingBuffer:
        def write(self, data: bytes) -> int:
            nonlocal active_writers, max_concurrent_writers
            with state_guard:
                active_writers += 1
                max_concurrent_writers = max(max_concurrent_writers, active_writers)
                is_first = len(observed_tids) == 0
                observed_tids.append(threading.get_ident())
            try:
                if is_first:
                    first_write_entered.set()
                    # If running on event loop thread, waiting here blocks loop tasks -> timeout!
                    assert unblock_first_write.wait(timeout=2.0), (
                        "Event loop was blocked by synchronous stdout write"
                    )
                written_chunks.append(bytes(data))
                return len(data)
            finally:
                with state_guard:
                    active_writers -= 1

        def flush(self) -> None:
            return None

    class _FakeStdout:
        def __init__(self) -> None:
            self.buffer = _BlockingBuffer()

    monkeypatch.setattr("agy_acp.commands.serve.sys.stdout", _FakeStdout())

    send_in, recv_in = anyio.create_memory_object_stream[str](8)

    async def _readline() -> str:
        try:
            return await recv_in.receive()
        except anyio.EndOfStream:
            return ""

    async def _concurrent_loop_unblocker() -> None:
        while not first_write_entered.is_set():
            await anyio.sleep(0.005)
        unblock_first_write.set()

    await send_in.send(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n"
    )
    await send_in.send(
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "session/list", "params": {}}) + "\n"
    )
    await send_in.aclose()

    with anyio.fail_after(3.0):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_concurrent_loop_unblocker)
            rc = await run_stdio_server(
                session_manager=mgr,
                wire_adapter=adapter,
                quota_tracker=qt,
                save_dir=tmp_path,
                stdin_readline=_readline,
                stdout_write=None,
            )

    assert rc == 0
    assert len(observed_tids) >= 2
    assert all(tid != main_tid for tid in observed_tids)
    assert max_concurrent_writers == 1
    parsed_frames = [
        json.loads(line)
        for line in b"".join(written_chunks).decode("utf-8").splitlines()
        if line.strip()
    ]
    assert {f.get("id") for f in parsed_frames} == {1, 2}
    assert all("result" in f and "error" not in f for f in parsed_frames)

    # Also verify bufferless text stdout fallback runs cleanly off-loop
    import io

    text_out = io.StringIO()
    monkeypatch.setattr("agy_acp.commands.serve.sys.stdout", text_out)
    send_in2, recv_in2 = anyio.create_memory_object_stream[str](4)

    async def _readline2() -> str:
        try:
            return await recv_in2.receive()
        except anyio.EndOfStream:
            return ""

    await send_in2.send(
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "session/list", "params": {}}) + "\n"
    )
    await send_in2.aclose()
    assert (
        await run_stdio_server(
            session_manager=mgr,
            wire_adapter=adapter,
            quota_tracker=qt,
            save_dir=tmp_path,
            stdin_readline=_readline2,
            stdout_write=None,
        )
        == 0
    )
    assert json.loads(text_out.getvalue().strip())["id"] == 3


async def test_run_stdio_server_closes_session_manager_on_exit(tmp_path: Path) -> None:
    """Verify `run_stdio_server` calls `await session_manager.close()` in its `finally:` block."""
    closed_calls: list[bool] = []

    class _ClosingSessionManager(SessionManager):
        async def close(self) -> None:
            closed_calls.append(True)
            await super().close()

    cli = _ServeStubCliProvider(save_dir=tmp_path)
    mgr = _ClosingSessionManager(save_dir=tmp_path, cli_provider=cli)
    qt = QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=cli.state_store)

    async def _eof_readline() -> str:
        await anyio.sleep(0)
        return ""

    async def _noop_write(_frame: bytes) -> None:
        await anyio.sleep(0)

    rc = await run_stdio_server(
        session_manager=mgr,
        wire_adapter=create_wire_adapter("v3"),
        quota_tracker=qt,
        save_dir=tmp_path,
        stdin_readline=_eof_readline,
        stdout_write=_noop_write,
    )
    assert rc == 0
    assert closed_calls == [True]


async def test_run_stdio_server_cancels_slow_background_quota_refresh_on_eof(
    tmp_path: Path,
) -> None:
    """Verify `run_stdio_server` cancels slow background quota refreshes promptly on EOF."""

    refresh_cancelled = False

    class _SlowUsageCli(_ServeStubCliProvider):
        async def run_slash_command(
            self,
            command: str,
            *,
            cwd: str | None = None,
            session_id: str | None = None,
            timeout: float = 30.0,
        ) -> dict[str, Any]:
            nonlocal refresh_cancelled
            _ = (command, cwd, session_id, timeout)
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                refresh_cancelled = True
                raise
            return {"status": "SUCCESS", "command": {"name": "usage", "data": {"groups": []}}}

    adapter = create_wire_adapter("v3")
    cli = _SlowUsageCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    qt = QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=cli.state_store)
    send_in, recv_in = anyio.create_memory_object_stream[str](8)

    async def _readline() -> str:
        try:
            return await recv_in.receive()
        except anyio.EndOfStream:
            return ""

    async def _noop_write(_frame: bytes) -> None:
        await anyio.sleep(0)

    await send_in.send(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
        + "\n"
    )
    await send_in.aclose()

    with anyio.fail_after(3.0):
        rc = await run_stdio_server(
            session_manager=mgr,
            wire_adapter=create_wire_adapter("v3"),
            quota_tracker=qt,
            save_dir=tmp_path,
            stdin_readline=_readline,
            stdout_write=_noop_write,
        )
    await recv_in.aclose()
    assert rc == 0
    assert refresh_cancelled is True
