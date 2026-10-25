"""Unit tests for AcpDispatcher JSON-RPC framing and routing."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from agy_acp.logic.acp_dispatcher import AcpDispatcher
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.logic.session_manager import SessionManager
from agy_acp.providers.agy_cli_provider import AgyCliProvider, PromptOutcome, ToolCallEvent
from agy_acp.providers.quota_provider import QuotaProvider


class _StubAgyCliProvider(AgyCliProvider):
    """Test double for AgyCliProvider used exclusively in unit tests."""

    async def run_slash_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        _ = (cwd, timeout)
        if command.lstrip("/") == "usage":
            return {
                "conversation_id": "",
                "status": "SUCCESS",
                "command": {
                    "name": "usage",
                    "data": {
                        "groups": [
                            {
                                "name": "Gemini Models",
                                "buckets": [
                                    {
                                        "id": "gemini-5h",
                                        "name": "Five Hour Limit Remaining",
                                        "window": "5h",
                                        "remaining_fraction": 0.95,
                                        "reset_time": "2026-10-24T07:52:04Z",
                                    },
                                    {
                                        "id": "gemini-weekly",
                                        "name": "Weekly Limit Remaining",
                                        "window": "weekly",
                                        "remaining_fraction": 0.39,
                                        "reset_time": "2026-10-24T12:34:28Z",
                                    },
                                ],
                            }
                        ]
                    },
                },
            }
        return {
            "conversation_id": "",
            "status": "SUCCESS",
            "command": {
                "name": "credits",
                "data": {"remaining_credits": 1000, "upgrade_uri": "https://one.google.com/ai"},
            },
        }

    async def stream_prompt(
        self,
        *,
        cwd: str,
        prompt: str,
        session_id: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        mode: str = "auto",
        effort: str | None = None,
        skip_permissions: bool | None = None,
        sandbox: bool = False,
        env: dict[str, str] | None = None,
        on_thought: Callable[[str], Awaitable[None]] | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
        on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None = None,
        on_usage: Callable[[int], Awaitable[None]] | None = None,
        on_steer_consumed: Callable[[str], Awaitable[None]] | None = None,
    ) -> PromptOutcome:
        _ = (
            cwd,
            session_id,
            model,
            mode,
            effort,
            skip_permissions,
            sandbox,
            env,
            on_usage,
            on_steer_consumed,
        )
        if on_thought is not None:
            await on_thought("Analyzing prompt...")
        if "tool" in prompt.lower():
            tc_event = ToolCallEvent(
                tool_call_id="tc_unit_001",
                title="Execute command",
                kind="execute",
                status="in_progress",
                raw_input={"command": prompt},
            )
            allowed = True
            if on_tool_call is not None:
                allowed = await on_tool_call(tc_event)
            if on_tool_update is not None:
                await on_tool_update(
                    ToolCallEvent(
                        tool_call_id="tc_unit_001",
                        title="Execute command",
                        kind="execute",
                        status="completed" if allowed else "failed",
                        raw_input={"command": prompt},
                        output="Executed: " + prompt if allowed else "Rejected",
                    )
                )
        if on_text is not None:
            await on_text(f"Response: {prompt}")
        return PromptOutcome(
            conversation_id=conversation_id or "conv-unit-1",
            stop_reason="end_turn",
            response=f"Response: {prompt}",
            usage={
                "input_tokens": 5,
                "output_tokens": 3,
                "thinking_tokens": 1,
                "cache_read_tokens": 0,
                "total_tokens": 9,
            },
        )


def _make_dispatcher(
    *,
    notification_sink: Any = None,
    client_rpc_caller: Any = None,
) -> AcpDispatcher:
    cli = _StubAgyCliProvider()
    return AcpDispatcher(
        notification_sink=notification_sink,
        client_rpc_caller=client_rpc_caller,
        session_manager=SessionManager(cli_provider=cli),
        quota_tracker=QuotaTracker(provider=QuotaProvider(cli_provider=cli)),
    )


def test_jsonrpc_frame_serialization() -> None:
    """Verify AcpDispatcher serializes newline-delimited UTF-8 JSON-RPC frames."""
    dispatcher = _make_dispatcher()
    raw = dispatcher.serialize_frame({"jsonrpc": "2.0", "id": 1, "result": {}})
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8"))["id"] == 1
    with pytest.raises(ValueError, match="exceeds maximum"):
        dispatcher.serialize_frame(
            {"jsonrpc": "2.0", "id": 1, "result": {"data": "x" * (16 * 1024 * 1024)}}
        )


async def test_handle_initialize_capabilities() -> None:
    """Verify AcpDispatcher.handle_message responds to initialize with required capabilities."""
    dispatcher = _make_dispatcher()
    resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": 1, "clientCapabilities": {}},
            }
        )
    )
    assert resp is not None
    assert resp["result"]["agentCapabilities"]["loadSession"] is True


async def test_prompt_streams_chunks_and_usage() -> None:
    """Verify AcpDispatcher routes session/prompt and returns token usage."""
    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    dispatcher = _make_dispatcher(notification_sink=_sink)
    resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": "s" * 36, "prompt": [{"type": "text", "text": "hi"}]},
            }
        )
    )
    assert resp is not None
    assert resp["result"]["stopReason"] == "end_turn"
    assert resp["result"]["usage"]["inputTokens"] >= 1
    assert resp["result"]["usage"]["outputTokens"] >= 1
    assert resp["result"]["usage"]["cachedReadTokens"] == 0
    assert resp["result"]["usage"]["cachedWriteTokens"] == 0
    assert len(emitted) > 0
    update_kinds = [
        n["params"]["update"]["sessionUpdate"]
        for n in emitted
        if n.get("method") == "session/update"
    ]
    assert "agent_thought_chunk" in update_kinds
    assert "agent_message_chunk" in update_kinds
    assert "usage_update" in update_kinds
    assert any(n.get("method") == "_kiro.dev/metadata" for n in emitted)

    emitted.clear()
    quota_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": "s" * 36, "prompt": [{"type": "text", "text": "/quota"}]},
            }
        )
    )
    assert quota_resp is not None
    assert quota_resp["result"]["stopReason"] == "end_turn"
    message_chunks = [
        n["params"]["update"]["content"]["text"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
    ]
    assert any("Antigravity Quota & Usage (`/quota`)" in chunk for chunk in message_chunks)


async def test_tool_permission_request_allow_and_reject() -> None:
    """Verify AcpDispatcher issues session/request_permission via client_rpc_caller."""
    emitted: list[dict[str, Any]] = []
    rpc_calls: list[tuple[str, dict[str, Any]]] = []
    selected_option = "allow_once"

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    async def _caller(method: str, params: dict[str, object]) -> dict[str, object]:
        assert method == "session/request_permission"
        rpc_calls.append((method, dict(params)))
        return {"outcome": {"outcome": "selected", "optionId": selected_option}}

    dispatcher = _make_dispatcher(notification_sink=_sink, client_rpc_caller=_caller)
    resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": "s" * 36, "prompt": [{"type": "text", "text": "tool"}]},
            }
        )
    )
    assert resp is not None
    assert len(rpc_calls) == 1
    assert rpc_calls[0][0] == "session/request_permission"
    option_ids = [opt["optionId"] for opt in rpc_calls[0][1]["options"]]
    assert option_ids == ["allow_once", "allow_always", "reject_once"]

    updates = [n["params"]["update"] for n in emitted if n.get("method") == "session/update"]
    tool_calls = [u for u in updates if u.get("sessionUpdate") == "tool_call"]
    tool_updates = [u for u in updates if u.get("sessionUpdate") == "tool_call_update"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["status"] == "in_progress"
    assert tool_calls[0]["kind"] == "execute"
    assert tool_calls[0]["_meta"]["kiro"] == {
        "toolName": "execute_bash",
        "mcpServerName": "",
    }
    assert rpc_calls[0][1]["_meta"] == {"kiro": {"toolId": "run_command"}}
    assert len(tool_updates) == 1
    assert tool_updates[0]["status"] == "completed"
    assert tool_updates[0]["toolCallId"] == tool_calls[0]["toolCallId"]

    emitted.clear()
    rpc_calls.clear()
    selected_option = "reject_once"
    resp_reject = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/prompt",
                "params": {"sessionId": "s" * 36, "prompt": [{"type": "text", "text": "tool"}]},
            }
        )
    )
    assert resp_reject is not None
    assert len(rpc_calls) == 1
    reject_updates = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "tool_call_update"
    ]
    assert len(reject_updates) == 1
    assert reject_updates[0]["status"] == "failed"


async def test_acp_dispatcher_logs_incoming_outgoing_and_unhandled_messages(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify AcpDispatcher and acp_logger record JSONL trace events to acp.log."""
    from agy_acp.providers.acp_logger import resolve_log_file_path

    log_file = tmp_path / "acp.log"
    monkeypatch.setenv("AGY_ACP_LOG_FILE", str(log_file))
    assert resolve_log_file_path() == log_file

    dispatcher = _make_dispatcher()
    resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "initialize",
                "params": {"protocolVersion": 1, "clientCapabilities": {}},
            }
        )
    )
    assert resp is not None

    unhandled_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "_kiro.dev/unknown_probe",
                "params": {"foo": "bar"},
            }
        )
    )
    assert unhandled_resp is None

    assert log_file.exists()
    entries = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [e["event"] for e in entries]
    assert "rpc.receive" in events
    assert "rpc.response" in events
    assert "rpc.unhandled" in events
    unhandled_entry = next(e for e in entries if e["event"] == "rpc.unhandled")
    assert unhandled_entry["method"] == "_kiro.dev/unknown_probe"
    assert unhandled_entry["params"] == {"foo": "bar"}


async def test_kiro_commands_execute_quota_effort_mcp_and_tools() -> None:
    """Verify _kiro.dev/commands/execute handles /quota, /effort, /mcp, and /tools."""
    dispatcher = _make_dispatcher()
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 20,
                "method": "session/new",
                "params": {"cwd": "/tmp", "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]

    quota_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": {"command": "quota", "args": {}}},
            }
        )
    )
    assert quota_resp is not None
    assert "Antigravity Quota & Usage" in quota_resp["result"]["message"]

    effort_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "_kiro.dev/commands/execute",
                "params": {
                    "sessionId": sid,
                    "command": {"command": "effort", "args": {"level": "high"}},
                },
            }
        )
    )
    assert effort_resp is not None
    assert "high" in effort_resp["result"]["message"]

    mcp_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 23,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": {"command": "mcp", "args": {}}},
            }
        )
    )
    assert mcp_resp is not None
    assert mcp_resp["result"]["data"]["servers"] == []

    tools_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 24,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": {"command": "tools", "args": {}}},
            }
        )
    )
    assert tools_resp is not None
    assert tools_resp["result"]["data"]["tools"] == []

    steer_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 25,
                "method": "_session/steer",
                "params": {"sessionId": sid, "message": "<user_message>steer</user_message>"},
            }
        )
    )
    assert steer_resp == {"jsonrpc": "2.0", "id": 25, "result": {"queued": False}}


async def test_session_prompt_wires_active_context_tokens_to_metadata() -> None:
    """Verify AcpDispatcher passes usage.context_tokens to QuotaTracker wire notifications."""

    class _MultiStepCliProvider(_StubAgyCliProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            on_usage = kwargs.get("on_usage")
            if on_usage is not None:
                await on_usage(18_500)
            return PromptOutcome(
                conversation_id="conv-multi-1",
                stop_reason="end_turn",
                response="Done",
                usage={
                    "input_tokens": 415_816,
                    "output_tokens": 19_955,
                    "thinking_tokens": 15_000,
                    "cache_read_tokens": 5_748_702,
                    "total_tokens": 6_199_473,
                    "context_tokens": 30_180,
                },
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    cli = _MultiStepCliProvider()
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=SessionManager(cli_provider=cli),
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 29,
                "method": "session/new",
                "params": {"cwd": "/tmp", "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    emitted.clear()

    resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 30,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]},
            }
        )
    )
    assert resp is not None
    usage_notifs = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "usage_update"
    ]
    meta_notif = next(n["params"] for n in emitted if n.get("method") == "_kiro.dev/metadata")
    assert [u["used"] for u in usage_notifs] == [18_500, 30_180]
    assert meta_notif["contextUsagePercentage"] == 3.018

    # Verify _kiro.dev/commands/execute (/usage) and session/load re-emit persisted context_tokens
    emitted.clear()
    fresh_dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=SessionManager(cli_provider=cli),
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
    )
    cmd_resp = await fresh_dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 31,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": {"command": "usage", "args": {}}},
            }
        )
    )
    assert cmd_resp is not None
    cmd_usage = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "usage_update"
    ]
    assert len(cmd_usage) == 1
    assert cmd_usage[0]["used"] == 30_180

    emitted.clear()
    load_resp = await fresh_dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 32,
                "method": "session/load",
                "params": {"sessionId": sid, "cwd": "/tmp", "mcpServers": []},
            }
        )
    )
    assert load_resp is not None
    load_usage = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "usage_update"
    ]
    assert len(load_usage) == 1
    assert load_usage[0]["used"] == 30_180


async def test_mid_turn_steer_and_steering_with_image_attachment(tmp_path: Any) -> None:
    """Verify AcpDispatcher handles _session/steer & _session/steering during an active turn."""
    import anyio

    turn_started = anyio.Event()
    received_prompts: list[str] = []

    class _SteerableStubCliProvider(_StubAgyCliProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            sid = str(kwargs.get("session_id") or "default")
            prompt = str(kwargs.get("prompt") or "")
            on_text = kwargs.get("on_text")
            on_steer_consumed = kwargs.get("on_steer_consumed")
            self._active_turns.add(sid)
            try:
                received_prompts.append(prompt)
                if on_text is not None:
                    await on_text("Initial chunk. ")
                turn_started.set()
                while not self._steer_queues.get(sid):
                    await anyio.sleep(0.01)
                steers = self._pop_queued_steers(sid)
                if steers and on_steer_consumed is not None:
                    await on_steer_consumed("\n".join(s.raw_message for s in steers))
                for s in steers:
                    received_prompts.append(s.prompt)
                    if on_text is not None:
                        await on_text(f"Steered: {s.prompt}")
                return PromptOutcome(
                    conversation_id="conv-steered-1",
                    stop_reason="end_turn",
                    response="Initial chunk. Steered.",
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 6,
                        "thinking_tokens": 0,
                        "cache_read_tokens": 0,
                        "total_tokens": 16,
                    },
                )
            finally:
                self._active_turns.discard(sid)

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    cli = _SteerableStubCliProvider(save_dir=tmp_path)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=SessionManager(save_dir=tmp_path, cli_provider=cli),
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 40,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    emitted.clear()

    img_file = tmp_path / "pasted-ide-error.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nide-error")
    raw_steer_message = (
        f"<user_message>\n![image]({img_file})\n\nHere are the errors from the IDE\n</user_message>"
    )

    prompt_resp: dict[str, Any] | None = None

    async def _run_prompt() -> None:
        nonlocal prompt_resp
        prompt_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 41,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "Start working"}],
                    },
                }
            )
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_prompt)
        await turn_started.wait()
        steer_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "_session/steer",
                    "params": {
                        "sessionId": sid,
                        "message": raw_steer_message,
                    },
                }
            )
        )
        assert steer_resp == {"jsonrpc": "2.0", "id": 42, "result": {"queued": True}}

    assert prompt_resp is not None
    assert prompt_resp["result"]["stopReason"] == "end_turn"
    updates = [n["params"]["update"] for n in emitted if n.get("method") == "session/update"]
    queued_updates = [u for u in updates if u.get("sessionUpdate") == "steering_queued"]
    consumed_updates = [u for u in updates if u.get("sessionUpdate") == "steering_consumed"]
    assert len(queued_updates) == 1
    assert queued_updates[0]["content"] == raw_steer_message
    assert len(consumed_updates) == 1
    assert consumed_updates[0]["content"] == raw_steer_message
    assert len(received_prompts) == 2
    assert "Here are the errors from the IDE" in received_prompts[1]
    assert "[User attached an image:" in received_prompts[1]
    assert "inspect it with view_file if needed]" in received_prompts[1]

    # When no turn is active, _session/steering returns outcome: failed
    idle_steering = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 43,
                "method": "_session/steering",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "Too late"}],
                },
            }
        )
    )
    assert idle_steering == {"jsonrpc": "2.0", "id": 43, "result": {"outcome": "failed"}}
