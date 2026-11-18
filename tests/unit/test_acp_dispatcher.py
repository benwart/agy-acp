"""Unit tests for AcpDispatcher JSON-RPC framing and routing."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agy_acp.logic.acp_dispatcher import AcpDispatcher
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.logic.session_manager import SessionManager
from agy_acp.logic.tool_classifier import PromptOutcome, ToolCallEvent
from agy_acp.providers.agy_acp_client import AgyAcpClientProvider
from agy_acp.providers.quota_provider import QuotaProvider


class _StubAgyAcpClientProvider(AgyAcpClientProvider):
    """Test double for AgyAcpClientProvider used exclusively in unit tests."""

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
        prompt: str = "",
        session_id: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        mode: str = "auto",
        effort: str | None = None,
        skip_permissions: bool | None = None,
        sandbox: bool = False,
        env: Mapping[str, str] | None = None,
        known_mcp_servers: Sequence[str] = (),
        on_thought: Callable[[str], Awaitable[None]] | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
        on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None = None,
        on_usage: Callable[[int], Awaitable[None]] | None = None,
        on_steer_consumed: Callable[[str], Awaitable[None]] | None = None,
        on_subagent_list: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None,
        **_extra: Any,
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
            known_mcp_servers,
            on_usage,
            on_steer_consumed,
            on_subagent_list,
            _extra,
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
    cli = _StubAgyAcpClientProvider()
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
    agent_version = resp["result"]["agentInfo"]["version"]
    major, minor, patch = (int(part) for part in agent_version.split(".")[:3])
    assert (major, minor, patch) >= (2, 28, 0)


async def test_prompt_streams_chunks_and_usage() -> None:
    """Verify AcpDispatcher routes session/prompt and returns token usage."""
    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    dispatcher = _make_dispatcher(notification_sink=_sink)
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
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
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]},
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
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "/quota"}]},
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
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
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
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "tool"}]},
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
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "tool"}]},
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
    assert unhandled_resp is not None
    assert unhandled_resp["error"]["code"] == -32601

    unhandled_notif = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "_kiro.dev/unknown_notif",
                "params": {"foo": "baz"},
            }
        )
    )
    assert unhandled_notif is None

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

    class _MultiStepCliProvider(_StubAgyAcpClientProvider):
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
    received_steer_blocks: list[list[dict[str, Any]] | None] = []

    class _SteerableStubCliProvider(_StubAgyAcpClientProvider):
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
                    await on_steer_consumed("\n".join(s.steer.raw_message for s in steers))
                for s in steers:
                    received_prompts.append(s.steer.prompt)
                    received_steer_blocks.append(s.content_blocks)
                    if on_text is not None:
                        await on_text(f"Steered: {s.steer.prompt}")
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
    assert len(received_steer_blocks) == 1
    assert received_steer_blocks[0] is not None
    steer_img_blocks = [b for b in received_steer_blocks[0] if b.get("type") == "image"]
    assert len(steer_img_blocks) == 1
    assert steer_img_blocks[0]["mimeType"] == "image/png"

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


async def test_kirocrew_0111_mcp_init_agent_switched_and_commands_inventory(
    tmp_path: Any,
) -> None:
    """Verify KiroCrew 0.11.1 MCP init frames, agent/switched, and /mcp + /tools inventory."""
    agents_dir = tmp_path / ".kiro_user_agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "kirocrew-skill-view-a.json").write_text(
        json.dumps(
            {
                "name": "kirocrew-skill-view-a",
                "mcpServers": {
                    "kirocrew-core": {"command": "echo", "args": ["core"]},
                },
            }
        ),
        encoding="utf-8",
    )
    (agents_dir / "kirocrew-skill-view-b.json").write_text(
        json.dumps(
            {
                "name": "kirocrew-skill-view-b",
                "model": "gemini-3.8-flash",
                "mcpServers": {
                    "kirocrew-core": {"command": "echo", "args": ["core"]},
                    "kirocrew-mate": {"command": "echo", "args": ["mate"]},
                },
            }
        ),
        encoding="utf-8",
    )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    cli = _StubAgyAcpClientProvider(save_dir=tmp_path)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=SessionManager(
            save_dir=tmp_path,
            cli_provider=cli,
            default_agent="kirocrew-skill-view-a",
        ),
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
    )

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 50,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [{"name": "broker-stub", "command": "echo", "args": ["stub"]}],
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    init_servers = [
        n["params"]["serverName"]
        for n in emitted
        if n.get("method") == "_kiro.dev/mcp/server_initialized"
        and n["params"].get("sessionId") == sid
    ]
    assert init_servers == ["broker-stub", "kirocrew-core"]

    emitted.clear()
    mode_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 51,
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "kirocrew-skill-view-b"},
            }
        )
    )
    assert mode_resp == {"jsonrpc": "2.0", "id": 51, "result": {}}
    switched = [n["params"] for n in emitted if n.get("method") == "_kiro.dev/agent/switched"]
    assert switched == [
        {
            "sessionId": sid,
            "agentName": "kirocrew-skill-view-b",
            "model": "gemini-3.8-flash",
        }
    ]
    mode_init_servers = [
        n["params"]["serverName"]
        for n in emitted
        if n.get("method") == "_kiro.dev/mcp/server_initialized"
        and n["params"].get("sessionId") == sid
    ]
    assert mode_init_servers == ["broker-stub", "kirocrew-core", "kirocrew-mate"]

    mcp_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 52,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": {"command": "mcp", "args": {}}},
            }
        )
    )
    assert mcp_resp is not None
    assert mcp_resp["result"]["data"]["servers"] == [
        {"name": "broker-stub", "status": "running", "toolCount": 1},
        {"name": "kirocrew-core", "status": "running", "toolCount": 1},
        {"name": "kirocrew-mate", "status": "running", "toolCount": 1},
    ]

    tools_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 53,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": {"command": "tools", "args": {}}},
            }
        )
    )
    assert tools_resp is not None
    tool_sources = {t["source"] for t in tools_resp["result"]["data"]["tools"]}
    assert {
        "mcp:broker-stub",
        "mcp:kirocrew-core",
        "mcp:kirocrew-mate",
    }.issubset(tool_sources)


async def test_compact_slash_command_emits_compaction_status_and_resets_context(
    tmp_path: Any,
) -> None:
    """Verify `/compact` emits `_kiro.dev/compaction/status`, resets context, and hands off."""
    from agy_acp.logic.acp_dispatcher import AVAILABLE_SLASH_COMMANDS

    assert any(cmd["name"] == "compact" for cmd in AVAILABLE_SLASH_COMMANDS)

    calls: list[dict[str, Any]] = []

    class _CompactCliProvider(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            calls.append(dict(kwargs))
            prompt_str = str(kwargs.get("prompt") or "")
            if "Summarize our conversation" in prompt_str:
                return PromptOutcome(
                    conversation_id=str(kwargs.get("conversation_id") or "conv-1"),
                    stop_reason="end_turn",
                    response="Summary: auth module refactored; tests green.",
                    usage={
                        "input_tokens": 200,
                        "output_tokens": 12,
                        "thinking_tokens": 0,
                        "cache_read_tokens": 0,
                        "total_tokens": 212,
                        "context_tokens": 30_392,
                    },
                )
            if len(calls) == 1:
                return PromptOutcome(
                    conversation_id="conv-pre-compact-1",
                    stop_reason="end_turn",
                    response="Initial heavy turn.",
                    usage={
                        "input_tokens": 25_000,
                        "output_tokens": 5_180,
                        "thinking_tokens": 0,
                        "cache_read_tokens": 0,
                        "total_tokens": 30_180,
                        "context_tokens": 30_180,
                    },
                )
            return PromptOutcome(
                conversation_id="conv-post-compact-2",
                stop_reason="end_turn",
                response="Continued cleanly from summary.",
                usage={
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "thinking_tokens": 0,
                    "cache_read_tokens": 0,
                    "total_tokens": 150,
                    "context_tokens": 150,
                },
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    cli = _CompactCliProvider(save_dir=tmp_path)
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
                "id": 60,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]

    # Turn 1 builds up 30,180 context tokens on conv-pre-compact-1
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 61,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "Build auth"}]},
            }
        )
    )
    emitted.clear()

    # Turn 2 runs `/compact focus on auth` inside KiroCrew envelope
    compact_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 62,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [
                        {
                            "type": "text",
                            "text": (
                                "[RUNTIME] KiroCrew\n"
                                "[CURRENT USER REQUEST -- respond to this]\n"
                                "/compact focus on auth"
                            ),
                        }
                    ],
                },
            }
        )
    )
    assert compact_resp is not None
    assert compact_resp["result"]["stopReason"] == "end_turn"
    assert "focus on auth" in calls[1]["prompt"]
    assert calls[1]["conversation_id"] == "conv-pre-compact-1"

    compaction_frames = [
        n["params"] for n in emitted if n.get("method") == "_kiro.dev/compaction/status"
    ]
    assert compaction_frames == [
        {
            "sessionId": sid,
            "status": {"type": "completed"},
            "summary": "Summary: auth module refactored; tests green.",
        }
    ]
    meta_frames = [n["params"] for n in emitted if n.get("method") == "_kiro.dev/metadata"]
    assert len(meta_frames) == 1
    assert meta_frames[0]["contextUsagePercentage"] < 0.1

    post_notifs = dispatcher.pop_post_response_notifications()
    assert len(post_notifs) == 1
    assert post_notifs[0]["method"] == "_kiro.dev/metadata"
    assert post_notifs[0]["params"]["sessionId"] == sid
    assert (
        post_notifs[0]["params"]["contextUsagePercentage"]
        == (meta_frames[0]["contextUsagePercentage"])
    )
    assert post_notifs[0]["params"]["meteringUsage"] == [
        {"unit": "credit", "unitPlural": "credits", "value": 0.0}
    ]
    assert dispatcher.pop_post_response_notifications() == []

    # Turn 3 starts a fresh conversation (`conversation_id=None`) with compacted summary prepended
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 63,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "Continue work"}],
                },
            }
        )
    )
    assert calls[2]["conversation_id"] is None
    assert "[Compacted Session Summary]" in calls[2]["prompt"]
    assert "Summary: auth module refactored; tests green." in calls[2]["prompt"]

    # Turn 4 resumes the new conversation without re-prepending the summary
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 64,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "Next step"}],
                },
            }
        )
    )
    assert calls[3]["conversation_id"] == "conv-post-compact-2"
    assert "[Compacted Session Summary]" not in calls[3]["prompt"]


async def test_session_prompt_emits_kiro_subagent_list_update_notifications() -> None:
    """Verify `AcpDispatcher` emits `_kiro.dev/subagent/list_update` with `sessionId`."""

    class _SubagentStubCliProvider(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            on_subagent_list = kwargs.get("on_subagent_list")
            if on_subagent_list is not None:
                await on_subagent_list(
                    [
                        {
                            "sessionId": "sub-conv-1",
                            "sessionName": "Codebase Researcher",
                            "role": "Codebase Researcher",
                            "agentName": "research",
                            "initialQuery": "Audit subagent wire frames",
                            "status": {"type": "running", "message": "read_file"},
                        }
                    ]
                )
                await on_subagent_list(
                    [
                        {
                            "sessionId": "sub-conv-1",
                            "sessionName": "Codebase Researcher",
                            "role": "Codebase Researcher",
                            "agentName": "research",
                            "initialQuery": "Audit subagent wire frames",
                            "status": {"type": "completed", "message": ""},
                        }
                    ]
                )
            return PromptOutcome(
                conversation_id="conv-sub-wire-1",
                stop_reason="end_turn",
                response="Done",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    cli = _SubagentStubCliProvider()
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
                "id": 70,
                "method": "session/new",
                "params": {"cwd": "/tmp", "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    emitted.clear()

    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 71,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "spawn"}]},
            }
        )
    )
    assert prompt_resp is not None
    assert prompt_resp["result"]["stopReason"] == "end_turn"

    subagent_frames = [
        n["params"] for n in emitted if n.get("method") == "_kiro.dev/subagent/list_update"
    ]
    assert subagent_frames == [
        {
            "sessionId": sid,
            "subagents": [
                {
                    "sessionId": "sub-conv-1",
                    "sessionName": "Codebase Researcher",
                    "role": "Codebase Researcher",
                    "agentName": "research",
                    "initialQuery": "Audit subagent wire frames",
                    "status": {"type": "running", "message": "read_file"},
                }
            ],
        },
        {
            "sessionId": sid,
            "subagents": [
                {
                    "sessionId": "sub-conv-1",
                    "sessionName": "Codebase Researcher",
                    "role": "Codebase Researcher",
                    "agentName": "research",
                    "initialQuery": "Audit subagent wire frames",
                    "status": {"type": "completed", "message": ""},
                }
            ],
        },
    ]


async def test_kas_v3_initialize_session_new_set_mode_load_replay_and_reset_server(
    tmp_path: Path,
) -> None:
    """Verify KAS v3 initialize, session/new, set_mode, load replay, and resetServer."""
    import sys

    from agy_acp.logic.wire import V3KasWireAdapter

    mcp_stub = tmp_path / "mcp_v3_stub.py"
    mcp_stub.write_text(
        "import json, sys\n"
        "tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    mid = msg.get('id')\n"
        "    method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {"
        "'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}}, "
        "'serverInfo': {'name': 'stub', 'version': '1.0'}}}), flush=True)\n"
        "    elif method == 'tools/list':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': ["
        "{'name': f'spawn_run_{tag}', 'description': 'Spawn'}]}}), flush=True)\n",
        encoding="utf-8",
    )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _StubAgyAcpClientProvider(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )

    init_resp = await dispatcher.handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    )
    assert init_resp is not None
    assert init_resp["result"]["protocolVersion"] == 1
    kiro_caps = init_resp["result"]["agentCapabilities"]["_meta"]["kiro"]
    assert kiro_caps["extensionMethods"] == [
        "_kiro/session/compact",
        "_kiro/session/delete",
        "_kiro/mcp/resetServer",
        "_kiro/session/context",
        "_kiro/knowledge",
        "_kiro/codeIntelligence",
        "_kiro/checkpoint/revertMultiple",
    ]
    assert kiro_caps["checkpoints"] is True
    assert kiro_caps["mcpServerOAuth"] is True
    assert kiro_caps["mcpRegistryPointers"] is True

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [
                        {
                            "name": "kirocrew-core",
                            "command": sys.executable,
                            "args": [str(mcp_stub), "initial"],
                        }
                    ],
                    "_meta": {
                        "kiro": {
                            "customAgents": [
                                {
                                    "id": "kirocrew",
                                    "name": "KiroCrew",
                                    "prompt": "Crew prompt",
                                    "tools": ["*"],
                                }
                            ]
                        }
                    },
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    post_new = dispatcher.pop_post_response_notifications()
    assert [n["method"] for n in post_new] == ["_kiro/mcp/status", "_kiro/tools/didChange"]
    assert post_new[0]["params"]["servers"][0]["tools"][0]["name"] == "spawn_run_initial"

    emitted.clear()
    mode_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "kirocrew"},
            }
        )
    )
    assert mode_resp is not None and mode_resp["result"] == {}
    assert all(n["method"] != "_kiro.dev/agent/switched" for n in emitted)
    assert any(
        n.get("params", {}).get("update", {}).get("sessionUpdate") == "current_mode_update"
        for n in emitted
    )
    post_mode = dispatcher.pop_post_response_notifications()
    assert [n["method"] for n in post_mode] == ["_kiro/mcp/status", "_kiro/tools/didChange"]

    emitted.clear()
    cfg_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/set_config_option",
                "params": {"sessionId": sid, "configId": "effortLevel", "value": "xhigh"},
            }
        )
    )
    assert cfg_resp is not None
    assert "configOptions" in cfg_resp["result"]
    assert any(
        n.get("params", {}).get("update", {}).get("sessionUpdate") == "config_option_update"
        for n in emitted
    )

    emitted.clear()
    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "Hello v3"}],
                    "_meta": {"kiro": {"ttft": True}},
                },
            }
        )
    )
    assert prompt_resp is not None
    assert prompt_resp["result"]["stopReason"] == "end_turn"
    updates = [n["params"]["update"] for n in emitted if n.get("method") == "session/update"]
    assert updates[0]["sessionUpdate"] == "session_info_update"
    assert updates[0]["_meta"]["kiro"]["kind"] == "user_message_id_assigned"
    displayed_chunks = [
        u
        for u in updates
        if u.get("sessionUpdate") in {"agent_thought_chunk", "agent_message_chunk"}
    ]
    assert "ttftMs" not in displayed_chunks[0]["_meta"]["kiro"]
    assert displayed_chunks[0]["_meta"]["kiro"]["ttft"]["promptReceivedAt"] >= 1_700_000_000_000
    info_kinds = [
        u["_meta"]["kiro"]["kind"]
        for u in updates
        if u.get("sessionUpdate") == "session_info_update"
    ]
    assert "context_usage" in info_kinds
    assert "turn_completion" in info_kinds
    assert len(mgr.state_store.get_session_turns(sid)) == 1

    # session/close preserves state & turns; session/load replays BEFORE response
    close_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "session/close",
                "params": {"sessionId": sid},
            }
        )
    )
    assert close_resp is not None and close_resp["result"] == {}
    emitted.clear()

    load_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "session/load",
                "params": {
                    "sessionId": sid,
                    "cwd": str(tmp_path),
                    "mcpServers": [
                        {
                            "name": "kirocrew-core",
                            "command": sys.executable,
                            "args": [str(mcp_stub), "reloaded"],
                        }
                    ],
                },
            }
        )
    )
    assert load_resp is not None and load_resp["result"]["sessionId"] == sid
    replay_updates = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n.get("params", {}).get("update", {}).get("_meta", {}).get("kiro", {}).get("replay")
        is True
    ]
    assert [u["sessionUpdate"] for u in replay_updates] == [
        "user_message_chunk",
        "agent_message_chunk",
    ]
    post_load = dispatcher.pop_post_response_notifications()
    assert [n["method"] for n in post_load] == ["_kiro/mcp/status", "_kiro/tools/didChange"]
    assert post_load[0]["params"]["servers"][0]["tools"][0]["name"] == "spawn_run_reloaded"

    # _kiro/mcp/resetServer re-probes server
    emitted.clear()
    reset_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "_kiro/mcp/resetServer",
                "params": {"sessionId": sid, "serverName": "kirocrew-core"},
            }
        )
    )
    assert reset_resp is not None and reset_resp["result"] == {}
    reset_frames = [*emitted, *dispatcher.pop_post_response_notifications()]
    assert any(n.get("method") == "_kiro/mcp/status" for n in reset_frames)

    # _kiro.dev/commands/execute is refused with -32603 on v3
    cmd_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": "/quota"},
            }
        )
    )
    assert cmd_resp is not None and cmd_resp["error"]["code"] == -32603

    # _kiro/session/delete removes session & turns
    del_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "_kiro/session/delete",
                "params": {"sessionId": sid},
            }
        )
    )
    assert del_resp is not None and del_resp["result"] == {}
    assert mgr.state_store.get_session(sid) is None
    assert mgr.state_store.get_session_turns(sid) == []


async def test_kas_v3_tool_permissions_allow_deny_ask_subagents_steer_and_compact(
    tmp_path: Path,
) -> None:
    """Verify KAS v3 permission evaluator, diff tool calls, subagents, steer, and compact RPC."""
    from agy_acp.logic.wire import V3KasWireAdapter

    perm_calls: list[dict[str, Any]] = []
    emitted: list[dict[str, Any]] = []

    class _V3TurnStubCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            prompt_str = str(kwargs.get("prompt", ""))
            if "Summarize our conversation" in prompt_str:
                return PromptOutcome(
                    conversation_id="conv-compact",
                    stop_reason="end_turn",
                    response="Compacted v3 summary.",
                    usage={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                )
            on_tool_call = kwargs.get("on_tool_call")
            on_tool_update = kwargs.get("on_tool_update")
            on_steer_consumed = kwargs.get("on_steer_consumed")
            on_subagent_list = kwargs.get("on_subagent_list")
            assert on_tool_call is not None and on_tool_update is not None

            # 1. Auto-allowed MCP tool
            allowed_mcp = await on_tool_call(
                ToolCallEvent(
                    tool_call_id="tc_mcp_allow",
                    title="@kirocrew-core/spawn_run",
                    kind="other",
                    status="in_progress",
                    raw_input={"tasks": []},
                    mcp_server="kirocrew-core",
                    mcp_tool="spawn_run",
                    kiro_tool_id="mcp_kirocrew-core_spawn_run",
                )
            )
            assert allowed_mcp is True

            # 2. Auto-denied shell tool
            allowed_deny = await on_tool_call(
                ToolCallEvent(
                    tool_call_id="tc_shell_deny",
                    title="rm -rf /tmp/nope",
                    kind="execute",
                    status="in_progress",
                    raw_input={"CommandLine": "rm -rf /tmp/nope"},
                    kas_builtin_id="run_command",
                    command="rm -rf /tmp/nope",
                )
            )
            assert allowed_deny is False

            # 3. Ask edit_file tool with diff content
            allowed_edit = await on_tool_call(
                ToolCallEvent(
                    tool_call_id="tc_edit_ask",
                    title="Edit app.py",
                    kind="edit",
                    status="in_progress",
                    raw_input={"Path": "app.py"},
                    kas_builtin_id="str_replace",
                    locations=[{"path": "app.py"}],
                    diff_content=[
                        {"type": "diff", "path": "app.py", "oldText": "a", "newText": "b"}
                    ],
                )
            )
            assert allowed_edit is True
            await on_tool_update(
                ToolCallEvent(
                    tool_call_id="tc_edit_ask",
                    title="Edit app.py",
                    kind="edit",
                    status="completed",
                    output="Updated app.py",
                    kas_builtin_id="str_replace",
                )
            )

            if on_steer_consumed is not None:
                await on_steer_consumed("Focus on tests")
            if on_subagent_list is not None:
                await on_subagent_list(
                    [
                        {
                            "sessionId": "sub-99",
                            "role": "Verifier",
                            "status": {"type": "running", "message": "working"},
                        },
                        {
                            "sessionId": "sub-99",
                            "role": "Verifier",
                            "status": {"type": "completed", "message": ""},
                        },
                    ]
                )

            return PromptOutcome(
                conversation_id="conv-v3-active",
                stop_reason="end_turn",
                response="Completed turn.",
                usage={"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
            )

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    async def _rpc_caller(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "session/request_permission"
        perm_calls.append(params)
        return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}

    adapter = V3KasWireAdapter()
    cli = _V3TurnStubCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        client_rpc_caller=_rpc_caller,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [],
                    "_meta": {
                        "kiro": {
                            "customAgents": [
                                {
                                    "id": "kirocrew",
                                    "name": "KiroCrew",
                                    "prompt": "Mandate body",
                                    "permissions": {
                                        "rules": [
                                            {
                                                "capability": "mcp",
                                                "match": ["kirocrew-core/*"],
                                                "effect": "allow",
                                            },
                                            {
                                                "capability": "shell",
                                                "match": ["rm -rf *"],
                                                "effect": "deny",
                                            },
                                        ]
                                    },
                                }
                            ]
                        }
                    },
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()

    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "kirocrew"},
            }
        )
    )
    dispatcher.pop_post_response_notifications()
    emitted.clear()

    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "Go"}]},
            }
        )
    )
    assert prompt_resp is not None
    assert prompt_resp["result"]["stopReason"] == "end_turn"

    # Only tc_edit_ask prompted client; allow & deny were handled inline!
    assert len(perm_calls) == 1
    assert perm_calls[0]["_meta"]["kiro"]["toolId"] == "str_replace"
    assert "kind" not in perm_calls[0]["toolCall"]
    assert perm_calls[0]["toolCall"]["content"][0]["type"] == "diff"

    updates = [n["params"]["update"] for n in emitted if n.get("method") == "session/update"]
    subtask_updates = [
        u
        for u in updates
        if u.get("sessionUpdate") == "tool_call_update"
        and u.get("_meta", {}).get("kiro", {}).get("kind") == "agent-subtask"
    ]
    assert [u["status"] for u in subtask_updates] == ["in_progress", "completed"]
    assert any(
        u.get("sessionUpdate") == "session_info_update"
        and u.get("_meta", {}).get("kiro", {}).get("kind") == "steering_injected"
        for u in updates
    )

    # _kiro/session/compact RPC
    emitted.clear()
    compact_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "_kiro/session/compact",
                "params": {"sessionId": sid},
            }
        )
    )
    assert compact_resp is not None
    assert compact_resp["result"] == {"success": True}
    compact_kinds = [
        n["params"]["update"]["_meta"]["kiro"]["kind"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "session_info_update"
    ]
    assert "summarization_started" in compact_kinds
    assert "summarization_completed" in compact_kinds
    assert "context_usage" in compact_kinds
    assert mgr.get_session(sid)["reinjectAgentPrompt"] is True


async def test_wave5b_reset_server_emits_full_snapshot_without_reprobing_peers(
    tmp_path: Path,
) -> None:
    """Verify `_kiro/mcp/resetServer` re-probes only target server and emits full roster."""
    import sys

    from agy_acp.logic.wire import V3KasWireAdapter

    counts_dir = tmp_path / "counts"
    counts_dir.mkdir()
    probe_script = tmp_path / "probe_counter_mcp.py"
    probe_script.write_text(
        "import json, pathlib, sys\n"
        "name = sys.argv[1]\n"
        "cfile = pathlib.Path(sys.argv[2]) / f'{name}.count'\n"
        "n = int(cfile.read_text()) + 1 if cfile.exists() else 1\n"
        "cfile.write_text(str(n))\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    mid = msg.get('id')\n"
        "    method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {"
        "'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}}, "
        "'serverInfo': {'name': name, 'version': '1.0'}}}), flush=True)\n"
        "    elif method == 'tools/list':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': ["
        "{'name': f'{name}_tool_v{n}', 'description': 'd'}]}}), flush=True)\n",
        encoding="utf-8",
    )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _StubAgyAcpClientProvider(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )

    server_names = ["srv_alpha", "srv_beta_under", "srv_gamma"]
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [
                        {
                            "name": sname,
                            "command": sys.executable,
                            "args": [str(probe_script), sname, str(counts_dir)],
                        }
                        for sname in server_names
                    ],
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    post_new = dispatcher.pop_post_response_notifications()
    assert [s["name"] for s in post_new[0]["params"]["servers"]] == server_names
    assert [int((counts_dir / f"{s}.count").read_text()) for s in server_names] == [1, 1, 1]

    emitted.clear()
    reset_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "_kiro/mcp/resetServer",
                "params": {"sessionId": sid, "serverName": "srv_beta_under"},
            }
        )
    )
    assert reset_resp == {"jsonrpc": "2.0", "id": 2, "result": {}}
    # Only srv_beta_under was re-probed; peers stayed at count 1!
    assert [int((counts_dir / f"{s}.count").read_text()) for s in server_names] == [1, 2, 1]
    status_frame = next(n for n in emitted if n.get("method") == "_kiro/mcp/status")
    did_change_frame = next(n for n in emitted if n.get("method") == "_kiro/tools/didChange")
    assert [s["name"] for s in status_frame["params"]["servers"]] == server_names
    tools_by_srv = {s["name"]: s["tools"][0]["name"] for s in status_frame["params"]["servers"]}
    assert tools_by_srv == {
        "srv_alpha": "srv_alpha_tool_v1",
        "srv_beta_under": "srv_beta_under_tool_v2",
        "srv_gamma": "srv_gamma_tool_v1",
    }
    emitted_tags = {t["tag"] for t in did_change_frame["params"]["tags"]}
    assert {
        "@srv_alpha/srv_alpha_tool_v1",
        "@srv_beta_under/srv_beta_under_tool_v2",
        "@srv_gamma/srv_gamma_tool_v1",
    }.issubset(emitted_tags)


async def test_wave5b_strict_permission_evaluation_v2_v3_and_mcp_longest_prefix(
    tmp_path: Path,
) -> None:
    """Verify strict fail-closed permission order, path normalization, and MCP prefix on v2/v3."""
    from agy_acp.logic.wire import V2KiroWireAdapter, V3KasWireAdapter

    recorded_results: list[bool] = []

    class _PermStubCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            on_tool_call = kwargs.get("on_tool_call")
            assert on_tool_call is not None
            prompt_tag = str(kwargs.get("prompt") or "")
            if "EMPTY_TOOLS" in prompt_tag:
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_empty_1",
                            title="Read file",
                            kind="read",
                            status="in_progress",
                            kas_builtin_id="read_file",
                            raw_input={"Path": str(tmp_path / "README.md")},
                        )
                    )
                )
            elif "POLICY_MATRIX" in prompt_tag:
                # 1. fs_write allowed via allowedTools: ["fs_write"] inside src/
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_write_ok",
                            title="Write file",
                            kind="edit",
                            status="in_progress",
                            kas_builtin_id="fs_write",
                            v2_tool_name="fs_write",
                            raw_input={"Path": str(tmp_path / "src" / "main.py")},
                        )
                    )
                )
                # 2. delete_file MUST NOT match allowedTools: ["fs_write"] -> headless ask -> False
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_del_block",
                            title="Delete file",
                            kind="delete",
                            status="in_progress",
                            kas_builtin_id="delete_file",
                            v2_tool_name="fs_write",
                            raw_input={"Path": str(tmp_path / "src" / "old.py")},
                        )
                    )
                )
                # 3. permissions deny on src/secret.py beats allowedTools: ["fs_write"]!
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_perm_deny_beats_allow",
                            title="Write secret",
                            kind="edit",
                            status="in_progress",
                            kas_builtin_id="fs_write",
                            v2_tool_name="fs_write",
                            raw_input={"Path": "src/sub/../secret.py"},
                        )
                    )
                )
                # 4. read_file with exclude: ["secrets/**"] on secrets/key.pem -> ask -> False
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_exclude_secret",
                            title="Read secret",
                            kind="read",
                            status="in_progress",
                            kas_builtin_id="read_file",
                            v2_tool_name="fs_read",
                            raw_input={"Path": str(tmp_path / "secrets" / "key.pem")},
                        )
                    )
                )
                # 5. read_file on src/ok.py -> permissions allow -> True
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_read_ok",
                            title="Read ok",
                            kind="read",
                            status="in_progress",
                            kas_builtin_id="read_file",
                            v2_tool_name="fs_read",
                            raw_input={"Path": str(tmp_path / "src" / "ok.py")},
                        )
                    )
                )
                # 6. Bare MCP tool name "spawn_run" MUST NOT match @my_under_srv/spawn_run
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_mcp_bare_block",
                            title="mcp_my_under_srv_spawn_run",
                            kind="other",
                            status="in_progress",
                            kiro_tool_id="mcp_my_under_srv_spawn_run",
                            mcp_server="my",
                            mcp_tool="under_srv_spawn_run",
                        )
                    )
                )
                # 7. Longest MCP server prefix resolves my_under_srv + run_safe
                recorded_results.append(
                    await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_mcp_longest_ok",
                            title="mcp_my_under_srv_run_safe",
                            kind="other",
                            status="in_progress",
                            kiro_tool_id="mcp_my_under_srv_run_safe",
                            mcp_server="my",
                            mcp_tool="under_srv_run_safe",
                        )
                    )
                )
            return PromptOutcome(
                conversation_id="conv-perm-1",
                stop_reason="end_turn",
                response="ok",
                usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            )

    for adapter in (V2KiroWireAdapter(), V3KasWireAdapter()):
        recorded_results.clear()
        sub_dir = tmp_path / adapter.engine_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        cli = _PermStubCli(save_dir=sub_dir)
        mgr = SessionManager(save_dir=sub_dir, cli_provider=cli, wire_adapter=adapter)
        # Note: client_rpc_caller=None verifies headless fail-closed (`ask` -> False)!
        dispatcher = AcpDispatcher(
            session_manager=mgr,
            quota_tracker=QuotaTracker(
                provider=QuotaProvider(cli_provider=cli),
                state_store=cli.state_store,
            ),
            wire_adapter=adapter,
        )
        new_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/new",
                    "params": {
                        "cwd": str(tmp_path),
                        "mcpServers": [
                            {"name": "my", "url": "http://127.0.0.1:9991/mcp"},
                            {"name": "my_under_srv", "url": "http://127.0.0.1:9992/mcp"},
                        ],
                        "_meta": {
                            "kiro": {
                                "customAgents": [
                                    {
                                        "id": "locked-down",
                                        "name": "Locked Down",
                                        "tools": [],
                                        "allowedTools": ["*"],
                                    },
                                    {
                                        "id": "matrix-agent",
                                        "name": "Matrix Agent",
                                        "tools": [
                                            "fs_read",
                                            "fs_write",
                                            "@my_under_srv",
                                        ],
                                        "allowedTools": [
                                            "fs_write",
                                            "spawn_run",
                                            "@my_under_srv/run_safe",
                                        ],
                                        "permissions": {
                                            "rules": [
                                                {
                                                    "capability": "fs_write",
                                                    "match": ["src/secret.py"],
                                                    "effect": "deny",
                                                },
                                                {
                                                    "capability": "fs_read",
                                                    "match": ["**"],
                                                    "exclude": ["secrets/**"],
                                                    "effect": "allow",
                                                },
                                            ]
                                        },
                                    },
                                ]
                            }
                        },
                    },
                }
            )
        )
        assert new_resp is not None
        sid = new_resp["result"]["sessionId"]
        dispatcher.pop_post_response_notifications()

        await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/set_mode",
                    "params": {"sessionId": sid, "modeId": "locked-down"},
                }
            )
        )
        dispatcher.pop_post_response_notifications()
        await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "EMPTY_TOOLS"}],
                    },
                }
            )
        )
        assert recorded_results == [False]

        recorded_results.clear()
        await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "session/set_mode",
                    "params": {"sessionId": sid, "modeId": "matrix-agent"},
                }
            )
        )
        dispatcher.pop_post_response_notifications()
        await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "POLICY_MATRIX"}],
                    },
                }
            )
        )
        assert recorded_results == [True, False, False, False, True, False, True]


async def test_wave5b_turn1_credit_baseline_ttft_gating_context_breakdown_and_eviction(
    tmp_path: Path,
) -> None:
    """Verify Turn-1 delta_credits baseline, epoch-ms ttft gating, contextBreakdown, & eviction."""
    from agy_acp.logic.wire import V3KasWireAdapter

    usage_calls = 0
    current_credits = 500

    class _TelemetryStubCli(_StubAgyAcpClientProvider):
        async def run_slash_command(
            self,
            command: str,
            *,
            cwd: str | None = None,
            session_id: str | None = None,
            timeout: float = 30.0,
        ) -> dict[str, Any]:
            nonlocal usage_calls
            _ = (cwd, session_id, timeout)
            if command.lstrip("/") == "usage":
                usage_calls += 1
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
                                            "id": "g5",
                                            "name": "5h",
                                            "window": "5h",
                                            "remaining_fraction": 1.0,
                                            "reset_time": "2026-12-01T00:00:00Z",
                                        },
                                        {
                                            "id": "g7",
                                            "name": "7d",
                                            "window": "weekly",
                                            "remaining_fraction": 1.0,
                                            "reset_time": "2026-12-05T00:00:00Z",
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
                    "data": {"remaining_credits": current_credits, "upgrade_uri": ""},
                },
            }

        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            nonlocal current_credits
            current_credits -= 2
            on_thought = kwargs.get("on_thought")
            on_text = kwargs.get("on_text")
            if on_thought is not None:
                await on_thought("Thinking first...")
            if on_text is not None:
                await on_text("Answer text.")
            return PromptOutcome(
                conversation_id="conv-tel-1",
                stop_reason="end_turn",
                response="Answer text.",
                usage={"input_tokens": 10, "output_tokens": 4, "context_tokens": 1200},
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _TelemetryStubCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    qt = QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=cli.state_store)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=qt,
        wire_adapter=adapter,
    )

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [],
                    "_meta": {
                        "kiro": {
                            "customAgents": [
                                {
                                    "id": "sys-agent",
                                    "name": "Sys Agent",
                                    "prompt": "System prompt 24 chars!!",
                                }
                            ]
                        }
                    },
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()

    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "sys-agent"},
            }
        )
    )
    dispatcher.pop_post_response_notifications()
    await dispatcher.wait_background_tasks()
    assert usage_calls == 1
    emitted.clear()
    usage_calls = 0

    # Turn 1: zero synchronous /usage calls on prompt path -> delta reported after bg refresh
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "Hello"}],
                },
            }
        )
    )
    assert usage_calls == 0
    updates_t1 = [n["params"]["update"] for n in emitted if n.get("method") == "session/update"]
    for u in updates_t1:
        if u.get("sessionUpdate") in {"agent_thought_chunk", "agent_message_chunk"}:
            assert "ttft" not in u.get("_meta", {}).get("kiro", {})
            assert "ttftMs" not in u.get("_meta", {}).get("kiro", {})
    completion_t1 = next(
        u["_meta"]["kiro"]
        for u in updates_t1
        if u.get("sessionUpdate") == "session_info_update"
        and u.get("_meta", {}).get("kiro", {}).get("kind") == "turn_completion"
    )
    assert completion_t1["promptTurnSummaries"] == [{"unit": "credit", "usage": 0.0}]
    assert "contextBreakdown" not in completion_t1

    await dispatcher.wait_background_tasks()
    assert usage_calls == 1

    # Turn 2: ttft=True + contextBreakdown="summary" -> conserved 2.0 delta & 0 blocking /usage!
    emitted.clear()
    usage_calls = 0
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "12345"}],
                    "_meta": {"kiro": {"ttft": True, "contextBreakdown": "summary"}},
                },
            }
        )
    )
    assert usage_calls == 0
    updates_t2 = [n["params"]["update"] for n in emitted if n.get("method") == "session/update"]
    thought_chunk = next(u for u in updates_t2 if u.get("sessionUpdate") == "agent_thought_chunk")
    msg_chunk = next(u for u in updates_t2 if u.get("sessionUpdate") == "agent_message_chunk")
    assert thought_chunk["_meta"]["kiro"]["ttft"]["promptReceivedAt"] >= 1_700_000_000_000
    assert "ttftMs" not in thought_chunk["_meta"]["kiro"]
    assert "ttft" not in msg_chunk.get("_meta", {}).get("kiro", {})
    completion_t2 = next(
        u["_meta"]["kiro"]
        for u in updates_t2
        if u.get("sessionUpdate") == "session_info_update"
        and u.get("_meta", {}).get("kiro", {}).get("kind") == "turn_completion"
    )
    assert completion_t2["promptTurnSummaries"] == [{"unit": "credit", "usage": 2.0}]
    cb = completion_t2["contextBreakdown"]
    assert cb == {
        "schema": 1,
        "totalChars": len("System prompt 24 chars!!") + len("12345") + len("Answer text."),
        "modelCalls": 1,
        "categories": {
            "system": {"chars": len("System prompt 24 chars!!")},
            "history": {"chars": len("12345") + len("Answer text.")},
        },
    }
    await dispatcher.wait_background_tasks()

    # Delete session evicts QuotaTracker tracking
    assert qt.get_session_context_tokens(sid) == 1200
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "_kiro/session/delete",
                "params": {"sessionId": sid},
            }
        )
    )
    assert sid not in qt._session_context_tokens
    assert sid not in qt._last_session_effective_credits


async def test_wave5b_jsonrpc_input_validation_and_concurrent_post_notifications(
    tmp_path: Path,
) -> None:
    """Verify strict JSON-RPC validation (-32600..-32002) and task-local post-notifications."""
    import anyio

    from agy_acp.logic.wire import V3KasWireAdapter

    adapter = V3KasWireAdapter()
    cli = _StubAgyAcpClientProvider(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )

    for bad_frame in ("[]", "null", "123", '"hello"'):
        r = await dispatcher.handle_message(bad_frame)
        assert r is not None and r["error"]["code"] == -32600

    # Invalid params type
    r_params = await dispatcher.handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": [1, 2]})
    )
    assert r_params is not None and r_params["error"]["code"] == -32602

    # Invalid protocolVersion values
    for bad_pv in (True, False, 0, -1, "2025-01-01"):
        r_pv = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "initialize",
                    "params": {"protocolVersion": bad_pv},
                }
            )
        )
        assert r_pv is not None and r_pv["error"]["code"] == -32602

    # Missing / non-str / whitespace required string params -> -32602
    invalid_param_calls = [
        ("session/new", {"cwd": ""}),
        ("session/new", {"cwd": None}),
        ("session/new", {"cwd": True}),
        ("session/load", {"sessionId": None, "cwd": "/tmp"}),
        ("session/set_mode", {"sessionId": "sid-1", "modeId": "   "}),
        ("session/set_model", {"sessionId": True, "modelId": "gemini-3.8-flash"}),
        ("session/set_config_option", {"sessionId": "sid-1", "configId": "", "value": "high"}),
        ("session/prompt", {"sessionId": "sid-1", "prompt": "not-a-list"}),
        ("_kiro/mcp/resetServer", {"sessionId": "sid-1", "serverName": ""}),
        ("_session/steer", {"sessionId": "sid-1", "message": "   "}),
        ("session/close", {"sessionId": 123}),
        ("_kiro/session/delete", {"sessionId": None}),
    ]
    for idx, (method, bad_p) in enumerate(invalid_param_calls, start=10):
        resp = await dispatcher.handle_message(
            json.dumps({"jsonrpc": "2.0", "id": idx, "method": method, "params": bad_p})
        )
        assert resp is not None, f"Expected error response for {method} {bad_p}"
        assert resp["error"]["code"] == -32602, f"Expected -32602 for {method} {bad_p}, got {resp}"

    # Non-existent session -> -32002 across all session-bound handlers
    missing_sid_calls = [
        ("session/prompt", {"sessionId": "missing-sid", "prompt": [{"type": "text", "text": "x"}]}),
        ("session/set_mode", {"sessionId": "missing-sid", "modeId": "auto"}),
        ("session/set_model", {"sessionId": "missing-sid", "modelId": "gemini-3.8-flash"}),
        (
            "session/set_config_option",
            {"sessionId": "missing-sid", "configId": "effortLevel", "value": "high"},
        ),
        ("_kiro/session/compact", {"sessionId": "missing-sid"}),
        ("_kiro/mcp/resetServer", {"sessionId": "missing-sid", "serverName": "srv"}),
        ("_session/steer", {"sessionId": "missing-sid", "message": "hi"}),
        ("session/close", {"sessionId": "missing-sid"}),
        ("_kiro/session/delete", {"sessionId": "missing-sid"}),
    ]
    for idx, (method, miss_p) in enumerate(missing_sid_calls, start=50):
        resp = await dispatcher.handle_message(
            json.dumps({"jsonrpc": "2.0", "id": idx, "method": method, "params": miss_p})
        )
        assert resp is not None
        assert resp["error"]["code"] == -32002, f"Expected -32002 for {method}, got {resp}"

    # Concurrent session/new calls isolate pop_post_response_notifications per task!
    captured: dict[int, tuple[str, list[dict[str, Any]]]] = {}

    async def _worker(req_id: int) -> None:
        r = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "session/new",
                    "params": {"cwd": str(tmp_path), "mcpServers": []},
                }
            )
        )
        await anyio.sleep(0.01)
        assert r is not None
        sid_local = r["result"]["sessionId"]
        captured[req_id] = (sid_local, dispatcher.pop_post_response_notifications())

    async with anyio.create_task_group() as tg:
        for req_id in (101, 102, 103, 104):
            tg.start_soon(_worker, req_id)

    for req_id in (101, 102, 103, 104):
        sid_local, post_frames = captured[req_id]
        assert [f["method"] for f in post_frames] == ["_kiro/mcp/status", "_kiro/tools/didChange"]
        assert all(f["params"]["sessionId"] == sid_local for f in post_frames)


async def test_kas_mcp_reset_server_start_oauth_and_model_fallback_notifications(
    tmp_path: Path,
) -> None:
    """Verify KAS 0.12.0 _kiro/mcp/resetServer startOAuth -> _kiro/openExternalUrl & fallback."""
    from agy_acp.logic.wire import McpServerStatus, V2KiroWireAdapter, V3KasWireAdapter

    emitted_v3: list[dict[str, Any]] = []
    outbound_rpcs: list[tuple[str, dict[str, Any]]] = []

    async def _sink_v3(msg: dict[str, Any]) -> None:
        emitted_v3.append(msg)

    async def _rpc_caller(method: str, params: dict[str, Any]) -> dict[str, Any]:
        outbound_rpcs.append((method, dict(params)))
        return {"success": True}

    class _FallbackAndOAuthStubCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            requested_model = str(kwargs.get("model") or "")
            on_text = kwargs.get("on_text")
            if on_text is not None:
                await on_text("Fallback answer.")
            if requested_model == "retired-model-v1":
                return PromptOutcome(
                    conversation_id="conv-fb-kas",
                    stop_reason="end_turn",
                    response="Fallback answer.",
                    usage={"input_tokens": 8, "output_tokens": 4, "context_tokens": 200},
                    resolved_model="gemini-3.8-flash",
                    fallback_from_model="retired-model-v1",
                )
            return PromptOutcome(
                conversation_id="conv-normal",
                stop_reason="end_turn",
                response="Fallback answer.",
                usage={"input_tokens": 8, "output_tokens": 4, "context_tokens": 200},
                resolved_model=requested_model,
            )

    adapter_v3 = V3KasWireAdapter()
    cli_v3 = _FallbackAndOAuthStubCli(save_dir=tmp_path / "v3")
    cli_v3.state_store.save_models(
        [
            {"modelId": "retired-model-v1", "name": "Retired v1", "description": "Old"},
            {"modelId": "gemini-3.8-flash", "name": "Gemini 3.8 Flash", "description": "Fast"},
        ]
    )

    class _OAuthProbeSessionManager(SessionManager):
        async def probe_session_mcp_servers(
            self,
            session_id: str,
            server_filter: str | None = None,
        ) -> list[McpServerStatus]:
            _ = (session_id, server_filter)
            return [
                McpServerStatus(
                    name="oauth-srv",
                    display_name="OAuth Server",
                    status="authRequired",
                    auth_url="https://auth.example.com/oauth/authorize?client=1",
                    failed_authorization=True,
                )
            ]

        def get_session_probe_server_names(self, session_id: str) -> list[str]:
            _ = session_id
            return ["oauth-srv"]

    mgr_v3 = _OAuthProbeSessionManager(
        save_dir=tmp_path / "v3",
        cli_provider=cli_v3,
        wire_adapter=adapter_v3,
    )

    dispatcher_v3 = AcpDispatcher(
        notification_sink=_sink_v3,
        client_rpc_caller=_rpc_caller,
        session_manager=mgr_v3,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli_v3),
            state_store=cli_v3.state_store,
        ),
        wire_adapter=adapter_v3,
    )

    init_resp = await dispatcher_v3.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientCapabilities": {"_meta": {"kiro": {"openExternalUrl": True}}},
                },
            }
        )
    )
    assert init_resp is not None and "result" in init_resp

    new_resp = await dispatcher_v3.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [{"name": "oauth-srv", "url": "https://mcp.example.com/sse"}],
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    dispatcher_v3.pop_post_response_notifications()
    emitted_v3.clear()

    # Non-boolean startOAuth -> -32602
    bad_reset = await dispatcher_v3.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "_kiro/mcp/resetServer",
                "params": {"sessionId": sid, "serverName": "oauth-srv", "startOAuth": "true"},
            }
        )
    )
    assert bad_reset is not None and bad_reset["error"]["code"] == -32602

    # startOAuth=False -> re-probes without calling _kiro/openExternalUrl
    reattempt_resp = await dispatcher_v3.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "_kiro/mcp/resetServer",
                "params": {"sessionId": sid, "serverName": "oauth-srv", "startOAuth": False},
            }
        )
    )
    assert reattempt_resp == {"jsonrpc": "2.0", "id": 4, "result": {}}
    assert outbound_rpcs == []

    # startOAuth=True -> calls _kiro/openExternalUrl before returning resetServer response
    signin_resp = await dispatcher_v3.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "_kiro/mcp/resetServer",
                "params": {"sessionId": sid, "serverName": "oauth-srv", "startOAuth": True},
            }
        )
    )
    assert signin_resp == {"jsonrpc": "2.0", "id": 5, "result": {}}
    assert outbound_rpcs == [
        ("_kiro/openExternalUrl", {"url": "https://auth.example.com/oauth/authorize?client=1"})
    ]

    # Pin retired model and run session/prompt -> emits KAS model_fallback & promotes session model
    await dispatcher_v3.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "session/set_model",
                "params": {"sessionId": sid, "modelId": "retired-model-v1"},
            }
        )
    )
    emitted_v3.clear()
    prompt_resp = await dispatcher_v3.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "Hello"}],
                },
            }
        )
    )
    assert prompt_resp is not None and prompt_resp["result"]["stopReason"] == "end_turn"
    assert mgr_v3.get_session(sid)["model"] == "gemini-3.8-flash"
    fb_v3 = [
        n["params"]["update"]["_meta"]["kiro"]
        for n in emitted_v3
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "session_info_update"
        and n["params"]["update"].get("_meta", {}).get("kiro", {}).get("kind") == "model_fallback"
    ]
    assert fb_v3 == [
        {
            "kind": "model_fallback",
            "from": "retired-model-v1",
            "to": "gemini-3.8-flash",
            "trigger": "availability",
            "scope": "session",
        }
    ]

    # V2 engine also emits _kiro.dev/session/update model_fallback (hop + promoted)
    emitted_v2: list[dict[str, Any]] = []

    async def _sink_v2(msg: dict[str, Any]) -> None:
        emitted_v2.append(msg)

    adapter_v2 = V2KiroWireAdapter()
    cli_v2 = _FallbackAndOAuthStubCli(save_dir=tmp_path / "v2")
    cli_v2.state_store.save_models(
        [
            {"modelId": "retired-model-v1", "name": "Retired v1", "description": "Old"},
            {"modelId": "gemini-3.8-flash", "name": "Gemini 3.8 Flash", "description": "Fast"},
        ]
    )
    mgr_v2 = SessionManager(save_dir=tmp_path / "v2", cli_provider=cli_v2, wire_adapter=adapter_v2)
    dispatcher_v2 = AcpDispatcher(
        notification_sink=_sink_v2,
        session_manager=mgr_v2,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli_v2),
            state_store=cli_v2.state_store,
        ),
        wire_adapter=adapter_v2,
    )
    new_v2 = await dispatcher_v2.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_v2 is not None
    sid_v2 = new_v2["result"]["sessionId"]
    await dispatcher_v2.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "session/set_model",
                "params": {"sessionId": sid_v2, "modelId": "retired-model-v1"},
            }
        )
    )
    emitted_v2.clear()
    await dispatcher_v2.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid_v2,
                    "prompt": [{"type": "text", "text": "Hello"}],
                },
            }
        )
    )
    fb_v2 = [
        n["params"]["update"]
        for n in emitted_v2
        if n.get("method") == "_kiro.dev/session/update"
        and n["params"]["update"].get("sessionUpdate") == "model_fallback"
    ]
    assert [u["promoted"] for u in fb_v2] == [False, True]


async def test_custom_agent_execute_bash_allowlist_permits_manage_task(tmp_path: Path) -> None:
    """Custom agent with execute_bash in tools allowlist permits manage_task without policy deny."""
    from agy_acp.logic.tool_classifier import classify_tool_call_step
    from agy_acp.logic.wire import V3KasWireAdapter
    from agy_acp.providers.agy_acp_client import _normalize_tool_arguments

    observed_allowed: list[bool] = []
    permission_requests: list[dict[str, Any]] = []

    class _ManageTaskCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            on_tool_call = kwargs.get("on_tool_call")
            on_text = kwargs.get("on_text")
            raw_args = {"Action": "list"}
            cls = classify_tool_call_step("tool", "manage_task", raw_args)
            norm_args = _normalize_tool_arguments(raw_args, tool_name="manage_task")
            ev = ToolCallEvent(
                tool_call_id="tc_mt_1",
                title=cls.title,
                kind=cls.acp_kind,
                status="in_progress",
                raw_input=norm_args,
                kiro_tool_name=cls.v2_tool_name,
                kiro_tool_id=cls.kiro_tool_id,
                kas_builtin_id=cls.kas_builtin_id,
                v2_tool_name=cls.v2_tool_name,
                acp_kind=cls.acp_kind,
                command=cls.command,
            )
            assert on_tool_call is not None
            allowed = await on_tool_call(ev)
            observed_allowed.append(allowed)
            if on_text is not None:
                await on_text("Checked tasks.")
            return PromptOutcome(
                conversation_id="conv-mt-1",
                stop_reason="end_turn",
                response="Checked tasks.",
                usage={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            )

    async def _rpc_caller(method: str, params: dict[str, Any]) -> dict[str, Any]:
        permission_requests.append({"method": method, "params": params})
        return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _ManageTaskCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
        client_rpc_caller=_rpc_caller,
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [],
                    "_meta": {
                        "kiro": {
                            "customAgents": [
                                {
                                    "id": "kirocrew",
                                    "name": "KiroCrew",
                                    "prompt": "Crew agent.",
                                    "tools": ["execute_bash", "fs_read"],
                                    "allowedTools": ["execute_bash", "fs_read"],
                                }
                            ]
                        }
                    },
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "kirocrew"},
            }
        )
    )
    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "check tasks"}],
                },
            }
        )
    )
    assert prompt_resp is not None
    assert prompt_resp["result"]["stopReason"] == "end_turn"
    assert observed_allowed == [True]
    assert permission_requests == []


async def test_session_prompt_returns_jsonrpc_error_on_unrecovered_cli_error(
    tmp_path: Path,
) -> None:
    """Unrecovered agy CLI ERROR returns JSON-RPC -32603 after emitting telemetry frames."""
    from agy_acp.logic.wire import V3KasWireAdapter

    class _ConnResetCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **_kwargs: Any) -> PromptOutcome:
            return PromptOutcome(
                conversation_id="conv-reset-1",
                stop_reason="end_turn",
                response="",
                usage={"input_tokens": 12, "output_tokens": 0, "total_tokens": 12},
                error="read ECONNRESET",
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _ConnResetCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    emitted.clear()

    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "run task"}],
                },
            }
        )
    )
    assert prompt_resp == {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {
            "code": -32603,
            "message": "read ECONNRESET",
            "data": {"errorType": "ServiceUnavailableError"},
        },
    }
    assert mgr.state_store.get_session_turns(sid) == []
    usage_updates = [
        n
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "usage_update"
    ]
    assert len(usage_updates) == 1


async def test_session_prompt_normalizes_carousel_stream_and_stamps_ttft_on_first_slide(
    tmp_path: Path,
) -> None:
    """Split carousel stream emits unwrapped image chunks, preserves TTFT, and saves clean turn."""
    from agy_acp.logic.wire import V3KasWireAdapter

    raw_chunks = [
        "````carousel\n",
        "![Slide 1](/Users/benwart/.gemini/antigravity-cli/brain/id/s1.png)\n<!-- slide -->\n",
        "![Slide 2](file:///Users/benwart/.gemini/antigravity-cli/brain/id/s2.png)\n````",
    ]

    class _CarouselStubCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            on_text = kwargs.get("on_text")
            if on_text is not None:
                for chunk in raw_chunks:
                    await on_text(chunk)
            return PromptOutcome(
                conversation_id="conv-carousel-v3",
                stop_reason="end_turn",
                response="".join(raw_chunks),
                usage={"input_tokens": 8, "output_tokens": 16, "total_tokens": 24},
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _CarouselStubCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    emitted.clear()

    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "render carousel"}],
                    "_meta": {"kiro": {"ttft": True}},
                },
            }
        )
    )
    assert prompt_resp is not None
    assert prompt_resp["result"]["stopReason"] == "end_turn"

    msg_updates = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
    ]
    assert len(msg_updates) >= 1
    assert all(u["content"]["text"] != "" for u in msg_updates)
    assert "ttft" in msg_updates[0].get("_meta", {}).get("kiro", {})

    expected = (
        "![Slide 1](/Users/benwart/.gemini/antigravity-cli/brain/id/s1.png)\n\n"
        "![Slide 2](/Users/benwart/.gemini/antigravity-cli/brain/id/s2.png)\n"
    )
    assert "".join(u["content"]["text"] for u in msg_updates) == expected
    saved_turns = mgr.state_store.get_session_turns(sid)
    assert len(saved_turns) == 1
    assert saved_turns[0]["assistant_text"] == expected


async def test_session_load_replays_legacy_persisted_carousel_turns_as_normalized_markdown(
    tmp_path: Path,
) -> None:
    """Legacy turns in SQLite containing raw ````carousel fences unwrap during session/load."""
    from agy_acp.logic.wire import V3KasWireAdapter

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _StubAgyAcpClientProvider(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]

    legacy_carousel = (
        "````carousel\n"
        "![Legacy 1](/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2.png)\n"
        "<!-- slide -->\n"
        "![Legacy 2](file:///Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2b.png)\n"
        "````\n"
    )
    mgr.state_store.append_session_turn(
        session_id=sid,
        user_message_id="msg_legacy_1",
        prompt_blocks=[{"type": "text", "text": "show old turn"}],
        assistant_text=legacy_carousel,
    )

    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "session/close",
                "params": {"sessionId": sid},
            }
        )
    )
    emitted.clear()

    load_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "session/load",
                "params": {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert load_resp is not None
    assert load_resp["result"]["sessionId"] == sid

    replay_replies = [
        n["params"]["update"]["content"]["text"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
        and n["params"]["update"].get("_meta", {}).get("kiro", {}).get("replay") is True
    ]
    assert replay_replies == [
        "![Legacy 1](/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2.png)\n\n"
        "![Legacy 2](/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2b.png)\n"
    ]


async def test_kirocrew_lite_consolidation_with_oversized_prompt_succeeds_without_tool_calls(
    tmp_path: Path,
) -> None:
    """Oversized (>64 KiB) consolidation prompts on `kirocrew-lite` (`tools: []`) return JSON."""
    from agy_acp.logic.wire import V3KasWireAdapter

    fake_agy = tmp_path / "agy-consolidate-wire"
    fake_agy.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "if '-p' in args and args[args.index('-p') + 1] in ('/usage', '/credits'):\n"
        "    print(json.dumps({'status': 'SUCCESS', 'command': {'name': 'quota', 'data': []}}))\n"
        "    sys.exit(0)\n"
        "assert args[:2] == ['acp', '--stdio']\n"
        "sessions = {}\n"
        "seq = 0\n"
        "for raw in sys.stdin:\n"
        "    line = raw.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    msg = json.loads(line)\n"
        "    mid = msg.get('id')\n"
        "    method = msg.get('method')\n"
        "    params = msg.get('params') or {}\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {'protocolVersion': 1}}),"
        " flush=True)\n"
        "    elif method == 'session/new':\n"
        "        seq += 1\n"
        "        sid = f'conv-lite-{seq}'\n"
        "        sessions[sid] = {'meta': params.get('_meta', {}), 'configs': {}}\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {\n"
        "            'sessionId': sid,\n"
        "            'models': {'currentModelId': 'gemini-4-argon-medium', 'availableModels': [\n"
        "                {'modelId': 'gemini-4-argon-medium', 'name': 'Gemini 4 Argon'}\n"
        "            ]},\n"
        "            'configOptions': [{'id': 'mode', 'options': [{'value': 'plan'}]}],\n"
        "        }}), flush=True)\n"
        "    elif method == 'session/set_config_option':\n"
        "        s = sessions.get(params.get('sessionId'), {})\n"
        "        s.setdefault('configs', {})[params.get('configId')] = params.get('value')\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {}}), flush=True)\n"
        "    elif method == 'session/set_model':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {}}), flush=True)\n"
        "    elif method == 'session/close':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {}}), flush=True)\n"
        "    elif method == 'session/prompt':\n"
        "        sid = params['sessionId']\n"
        "        s = sessions[sid]\n"
        "        assert s['configs'].get('mode') == 'plan'\n"
        "        sys_app = s['meta'].get('antigravity', {})"
        ".get('systemPrompt', {}).get('append', '')\n"
        "        assert 'lightweight background agent' in sys_app\n"
        "        body = params['prompt'][0]['text']\n"
        "        out = json.dumps("
        "{'history_entry': f'ok:{len(body)}', 'semantic': [], 'lessons': []})\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
        "            'sessionId': sid,\n"
        "            'update': {'sessionUpdate': 'agent_message_chunk',"
        " 'content': {'type': 'text', 'text': out}},\n"
        "        }}), flush=True)\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
        "            'sessionId': sid,\n"
        "            'update': {'sessionUpdate': 'usage_update', 'used': 21020, 'size': 1000000},\n"
        "        }}), flush=True)\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {"
        "'stopReason': 'end_turn'}}), flush=True)\n",
        encoding="utf-8",
    )
    fake_agy.chmod(0o755)

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    state_dir = tmp_path / "state"
    cli = AgyAcpClientProvider(bin_path=fake_agy, save_dir=state_dir)
    mgr = SessionManager(save_dir=state_dir, cli_provider=cli, wire_adapter=adapter)
    try:
        dispatcher = AcpDispatcher(
            notification_sink=_sink,
            session_manager=mgr,
            quota_tracker=QuotaTracker(
                provider=QuotaProvider(cli_provider=cli),
                state_store=cli.state_store,
            ),
            wire_adapter=adapter,
        )

        ws = tmp_path / "_consolidate"
        ws.mkdir()
        new_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/new",
                    "params": {
                        "cwd": str(ws),
                        "mcpServers": [],
                        "_meta": {
                            "kiro": {
                                "customAgents": [
                                    {
                                        "id": "kirocrew-lite",
                                        "name": "kirocrew-lite",
                                        "prompt": (
                                            "You are a Kiro Crew lightweight background agent."
                                        ),
                                        "tools": [],
                                    }
                                ]
                            }
                        },
                    },
                }
            )
        )
        assert new_resp is not None
        sid = new_resp["result"]["sessionId"]

        mode_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/set_mode",
                    "params": {"sessionId": sid, "modeId": "kirocrew-lite"},
                }
            )
        )
        assert mode_resp is not None
        assert "result" in mode_resp
        emitted.clear()

        big_consolidation = "You are a memory consolidation agent.\n" + ("Row\n" * 22000)
        assert len(big_consolidation.encode("utf-8")) > 65_536

        prompt_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": big_consolidation}],
                    },
                }
            )
        )
        assert prompt_resp is not None
        assert prompt_resp["result"]["stopReason"] == "end_turn"

        tool_updates = [
            n
            for n in emitted
            if n.get("method") == "session/update"
            and n["params"]["update"].get("sessionUpdate") in ("tool_call", "tool_call_update")
        ]
        assert tool_updates == []

        chunks = [
            n["params"]["update"]["content"]["text"]
            for n in emitted
            if n.get("method") == "session/update"
            and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
        ]
        parsed = json.loads("".join(chunks))
        assert parsed["history_entry"].startswith("ok:")
        assert parsed["lessons"] == []
    finally:
        await mgr.close()


async def test_acp_dispatcher_0121_fork_steer_clear_kas_rpcs_and_trailing_compact(
    tmp_path: Path,
) -> None:
    """Verify 0.12.1 RPC handlers: fork, steer/clear, KAS RPCs, and trailing /compact."""
    from agy_acp.logic.wire import V2KiroWireAdapter, V3KasWireAdapter

    # Trailing /compact extraction after sentence punctuation
    assert AcpDispatcher._extract_compact_command("Done with task. /compact") == (
        True,
        "Done with task.",
    )
    assert AcpDispatcher._extract_compact_command("Can we wrap up? /compact") == (
        True,
        "Can we wrap up?",
    )
    assert AcpDispatcher._extract_compact_command("Check /Users/foo/compact path") == (False, "")

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _StubAgyAcpClientProvider(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )
    assert dispatcher._wire is adapter

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 200,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()

    # Run 2 turns so session has history
    for idx in (1, 2):
        await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 200 + idx,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": f"Prompt {idx}"}],
                    },
                }
            )
        )
        dispatcher.pop_post_response_notifications()

    turns_before = mgr.state_store.get_session_turns(sid)
    assert len(turns_before) == 2
    first_msg_id = turns_before[0]["user_message_id"]
    second_msg_id = turns_before[1]["user_message_id"]

    # 1. _session/steer/clear on v3 emits steering_cleared
    cli._active_turns.add(sid)
    await mgr.steer(session_id=sid, message="Drop me 1")
    await mgr.steer(session_id=sid, message="Drop me 2")
    emitted.clear()

    clear_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 210,
                "method": "_session/steer/clear",
                "params": {"sessionId": sid},
            }
        )
    )
    assert clear_resp is not None
    assert clear_resp["result"] == {"cleared": True}
    steer_updates = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "session_info_update"
    ]
    assert len(steer_updates) == 1
    assert steer_updates[0]["_meta"]["kiro"]["kind"] == "steering_cleared"
    assert steer_updates[0]["_meta"]["kiro"]["content"] == "Drop me 1\nDrop me 2"
    cli._active_turns.discard(sid)

    # Also verify v2 emits steering_cleared
    emitted_v2: list[dict[str, Any]] = []

    async def _sink_v2(msg: dict[str, Any]) -> None:
        emitted_v2.append(msg)

    disp_v2 = AcpDispatcher(
        notification_sink=_sink_v2,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=V2KiroWireAdapter(),
    )
    cli._active_turns.add(sid)
    await mgr.steer(session_id=sid, message="Drop v2")
    await disp_v2.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 211,
                "method": "_session/steer/clear",
                "params": {"sessionId": sid},
            }
        )
    )
    assert emitted_v2[0]["params"]["update"]["sessionUpdate"] == "steering_cleared"
    cli._active_turns.discard(sid)

    # 2. session/fork via _meta.jetbrains.air.fork.messageId
    fork_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 212,
                "method": "session/fork",
                "params": {
                    "sessionId": sid,
                    "_meta": {"jetbrains": {"air": {"fork": {"messageId": first_msg_id}}}},
                },
            }
        )
    )
    assert fork_resp is not None
    fork_sid = fork_resp["result"]["sessionId"]
    assert fork_sid != sid
    assert len(mgr.state_store.get_session_turns(fork_sid)) == 1
    post_fork = dispatcher.pop_post_response_notifications()
    assert [n["method"] for n in post_fork] == ["_kiro/mcp/status", "_kiro/tools/didChange"]

    # 3. _kiro/session/context
    ctx_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 213,
                "method": "_kiro/session/context",
                "params": {"sessionId": sid, "subcommand": "add", "path": "a.py"},
            }
        )
    )
    assert ctx_resp is not None
    assert ctx_resp["result"]["success"] is True

    # 4. _kiro/knowledge
    know_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 214,
                "method": "_kiro/knowledge",
                "params": {
                    "sessionId": sid,
                    "subcommand": "add",
                    "name": "Doc",
                    "path": "/tmp/doc.md",
                },
            }
        )
    )
    assert know_resp is not None
    assert know_resp["result"]["success"] is True

    # 5. _kiro/codeIntelligence
    ci_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 215,
                "method": "_kiro/codeIntelligence",
                "params": {"sessionId": sid, "subcommand": "status"},
            }
        )
    )
    assert ci_resp is not None
    assert ci_resp["result"]["success"] is True
    assert "languages" in ci_resp["result"]["status"]

    # 6. _kiro/checkpoint/revertMultiple
    rev_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 216,
                "method": "_kiro/checkpoint/revertMultiple",
                "params": {"sessionId": sid, "messageId": second_msg_id},
            }
        )
    )
    assert rev_resp is not None
    assert rev_resp["result"] == {"success": True}
    assert len(mgr.state_store.get_session_turns(sid)) == 1


async def test_w4_kas_steering_queued_persisted_turns_and_tool_call_recheck_dedup(
    tmp_path: Path,
) -> None:
    """Verify `[M-3]` `_session/steering` queued + persisted blocks and `[H-1]` wire dedup."""
    import anyio

    from agy_acp.logic.tool_classifier import PromptOutcome, ToolCallEvent
    from agy_acp.logic.wire.v3_kas import V3KasWireAdapter

    emitted: list[dict[str, Any]] = []
    turn_started = anyio.Event()

    async def emit_cb(frame: dict[str, Any]) -> None:
        emitted.append(frame)

    class _W4V3StubCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            sid = str(kwargs.get("session_id") or "default")
            on_tool_call = kwargs.get("on_tool_call")
            on_tool_update = kwargs.get("on_tool_update")
            on_steer_consumed = kwargs.get("on_steer_consumed")
            on_text = kwargs.get("on_text")

            self._active_turns.add(sid)
            try:
                turn_started.set()
                for _ in range(50):
                    if self._steer_queues.get(sid):
                        break
                    await anyio.sleep(0.01)

                # [H-1]: First tool_call arrives with empty rawInput={}, then re-evaluated with .env
                if on_tool_call is not None:
                    first_allow = await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_env_1",
                            title="edit_file",
                            kind="edit",
                            status="in_progress",
                            kas_builtin_id="fs_write",
                            v2_tool_name="fs_write",
                            raw_input={},
                        )
                    )
                    assert first_allow is True
                    second_allow = await on_tool_call(
                        ToolCallEvent(
                            tool_call_id="tc_env_1",
                            title="edit_file",
                            kind="edit",
                            status="in_progress",
                            kas_builtin_id="fs_write",
                            v2_tool_name="fs_write",
                            raw_input={"Path": ".env", "path": ".env"},
                        )
                    )
                    assert second_allow is False
                    if on_tool_update is not None:
                        await on_tool_update(
                            ToolCallEvent(
                                tool_call_id="tc_env_1",
                                title="edit_file",
                                kind="edit",
                                status="failed",
                                raw_input={"Path": ".env", "path": ".env"},
                                output="Tool execution rejected by policy.",
                            )
                        )

                entries = self._pop_queued_steers(sid)
                if entries and on_steer_consumed is not None:
                    await on_steer_consumed(
                        "\n".join(e.steer.raw_message for e in entries if e.steer.raw_message)
                    )
                if on_text is not None:
                    await on_text("Steered completion.")
                return PromptOutcome(
                    conversation_id="agy-w4",
                    stop_reason="end_turn",
                    response="Steered completion.",
                    usage={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
                )
            finally:
                self._active_turns.discard(sid)

    cli = _W4V3StubCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=V3KasWireAdapter())
    dispatcher = AcpDispatcher(
        session_manager=mgr,
        wire_adapter=V3KasWireAdapter(),
        notification_sink=emit_cb,
    )

    await dispatcher.handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    )
    custom_agents = [
        {
            "id": "env-guard",
            "name": "env-guard",
            "description": "Deny .env edits",
            "allowedTools": ["*"],
            "permissions": {
                "rules": [
                    {
                        "capability": "fs_write",
                        "match": [".env", "**/.env"],
                        "effect": "deny",
                    }
                ]
            },
        }
    ]
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [],
                    "_meta": {"kiro": {"customAgents": custom_agents}},
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "env-guard"},
            }
        )
    )
    emitted.clear()

    prompt_resp: dict[str, Any] | None = None

    async def _drive_prompt() -> None:
        nonlocal prompt_resp
        prompt_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "Initial instruction"}],
                    },
                }
            )
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_drive_prompt)
        await turn_started.wait()
        steer_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "_session/steering",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "Focus on tests instead"}],
                    },
                }
            )
        )
        assert steer_resp == {"jsonrpc": "2.0", "id": 5, "result": {"outcome": "injected"}}

    assert prompt_resp is not None
    updates = [n["params"]["update"] for n in emitted if n.get("method") == "session/update"]

    # [H-1]: Only ONE wire `tool_call` start notification emitted for `tc_env_1`
    tool_starts = [
        u
        for u in updates
        if u.get("sessionUpdate") == "tool_call" and u.get("toolCallId") == "tc_env_1"
    ]
    assert len(tool_starts) == 1

    # [M-3]: Both `steering_queued` and `steering_injected` emitted on KAS v3 `_session/steering`
    info_kinds = [
        u.get("_meta", {}).get("kiro", {}).get("kind")
        for u in updates
        if u.get("sessionUpdate") == "session_info_update"
    ]
    assert "steering_queued" in info_kinds
    assert "steering_injected" in info_kinds

    # [M-3]: Consumed steered user instruction persisted in `session_turns`
    turns = mgr.state_store.get_session_turns(sid)
    assert len(turns) == 1
    saved_texts = [b.get("text", "") for b in turns[0]["prompt_blocks"] if b.get("type") == "text"]
    assert "Initial instruction" in saved_texts
    assert "Focus on tests instead" in saved_texts


async def test_session_prompt_structured_error_type_classification(tmp_path: Path) -> None:
    """Verify session/prompt attaches KiroCrew 0.12.2 structured errorType data on turn errors."""
    from agy_acp.logic.wire import V3KasWireAdapter

    next_error = ""

    class _DynamicErrorCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            return PromptOutcome(
                conversation_id="conv-err-types",
                stop_reason="end_turn",
                response="",
                error=next_error,
                usage={"input_tokens": 2, "output_tokens": 0, "total_tokens": 2},
            )

    adapter = V3KasWireAdapter()
    cli = _DynamicErrorCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]

    cases = [
        ("The monthly usage limit has been reached for this account", "MonthlyLimitError"),
        ("HTTP 429 Too Many Requests: rate limit exceeded", "ThrottlingException"),
        ("503 Service Temporarily Unavailable", "ServiceUnavailableError"),
        ("Unexpected model decoding failure", "ProviderError"),
    ]
    for idx, (err_text, expected_type) in enumerate(cases, start=10):
        next_error = err_text
        resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": idx,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "check"}],
                    },
                }
            )
        )
        assert resp == {
            "jsonrpc": "2.0",
            "id": idx,
            "error": {
                "code": -32603,
                "message": err_text,
                "data": {"errorType": expected_type},
            },
        }


async def test_kas_mcp_toggle_disables_and_reenables_mcp_servers_and_probing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _kiro/mcp/toggle isolates process-local MCP merging, probing, and stream kwargs."""
    from agy_acp.logic.mcp_catalog_prober import McpCatalogProber
    from agy_acp.logic.wire import McpServerStatus, V3KasWireAdapter

    log_file = tmp_path / "acp-toggle.log"
    monkeypatch.setenv("AGY_ACP_LOG_FILE", str(log_file))

    probe_calls: list[list[str]] = []
    prompt_kwargs_seen: list[dict[str, Any]] = []

    class _CountingProber(McpCatalogProber):
        async def probe_servers(
            self,
            servers: Any,
            *,
            session_env: Any = None,
            cwd: str | None = None,
        ) -> list[McpServerStatus]:
            names = [str(d.get("name") or "") for d in servers]
            probe_calls.append(names)
            return [McpServerStatus(name=n, status="connected") for n in names if n]

    class _InspectPromptCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            prompt_kwargs_seen.append(kwargs)
            return PromptOutcome(
                conversation_id="conv-toggle",
                stop_reason="end_turn",
                response="ok",
                usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            )

    # Peer process sharing the same SQLite store MUST NOT have its session wiped
    peer_cli = _InspectPromptCli(save_dir=tmp_path)
    peer_mgr = SessionManager(save_dir=tmp_path, cli_provider=peer_cli)
    peer_desc = await peer_mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[{"name": "peer-mcp", "command": "echo", "args": ["peer"]}],
    )
    peer_sid = peer_desc["sessionId"]
    assert [s["name"] for s in peer_mgr.get_session(peer_sid)["mcp_servers"]] == ["peer-mcp"]

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    adapter = V3KasWireAdapter()
    cli = _InspectPromptCli(save_dir=tmp_path)
    mgr = SessionManager(
        save_dir=tmp_path,
        cli_provider=cli,
        wire_adapter=adapter,
        mcp_prober=_CountingProber(),
    )
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=adapter,
    )

    toggle_off = await dispatcher.handle_message(
        json.dumps({"jsonrpc": "2.0", "method": "_kiro/mcp/toggle", "params": {"enabled": False}})
    )
    assert toggle_off is None
    assert mgr.mcp_enabled is False
    assert [s["name"] for s in peer_mgr.get_session(peer_sid)["mcp_servers"]] == ["peer-mcp"]

    custom_agents = [
        {
            "id": "kirocrew-lite",
            "name": "KiroCrew Lite",
            "prompt": "Background helper.",
            "mcpServers": {"kirocrew-core": {"command": "echo", "args": ["core"]}},
        }
    ]
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [{"name": "wire-srv", "command": "echo", "args": ["wire"]}],
                    "_meta": {"kiro": {"customAgents": custom_agents}},
                },
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    post_frames = dispatcher.pop_post_response_notifications()
    mcp_status_frames = [f for f in post_frames if f.get("method") == "_kiro/mcp/status"]
    assert len(mcp_status_frames) == 1
    assert mcp_status_frames[0]["params"]["servers"] == []
    assert probe_calls == []
    assert mgr.get_session(sid)["mcp_servers"] == []
    assert [s["name"] for s in mgr.get_session(sid)["wire_mcp_servers"]] == ["wire-srv"]

    mode_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "kirocrew-lite"},
            }
        )
    )
    assert mode_resp == {"jsonrpc": "2.0", "id": 2, "result": {}}
    dispatcher.pop_post_response_notifications()
    assert probe_calls == []
    assert mgr.get_session(sid)["mcp_servers"] == []

    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "summarize"}],
                },
            }
        )
    )
    assert prompt_resp is not None and prompt_resp["result"]["stopReason"] == "end_turn"
    assert len(prompt_kwargs_seen) == 1
    assert prompt_kwargs_seen[0]["mcp_servers"] == []
    assert prompt_kwargs_seen[0]["route_spawn_run"] is False

    bad_toggle = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "_kiro/mcp/toggle",
                "params": {"enabled": "invalid"},
            }
        )
    )
    assert bad_toggle is not None and bad_toggle["error"]["code"] == -32602

    toggle_on = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "_kiro/mcp/toggle",
                "params": {"enabled": True},
            }
        )
    )
    assert toggle_on == {"jsonrpc": "2.0", "id": 5, "result": {}}
    assert mgr.mcp_enabled is True
    assert [s["name"] for s in mgr.get_session(sid)["mcp_servers"]] == [
        "wire-srv",
        "kirocrew-core",
    ]
    probed = await mgr.probe_session_mcp_servers(sid)
    assert [p.name for p in probed] == ["wire-srv", "kirocrew-core"]

    log_events = [
        json.loads(line)["event"]
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "rpc.unhandled" not in log_events


async def test_cancelled_session_prompt_emits_turn_telemetry_on_v2_and_v3(
    tmp_path: Path,
) -> None:
    """Verify cancelled turns emit metering frames on both v2 (post-response drain) and v3."""
    from agy_acp.logic.wire import V2KiroWireAdapter, V3KasWireAdapter

    class _CancelUsageCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            return PromptOutcome(
                conversation_id="conv-cancel-usage",
                stop_reason="cancelled",
                response="",
                usage={
                    "input_tokens": 24,
                    "output_tokens": 9,
                    "cache_read_tokens": 4,
                    "context_tokens": 1200,
                },
            )

    for engine_name, adapter in (("v2", V2KiroWireAdapter()), ("v3", V3KasWireAdapter())):
        sub_dir = tmp_path / engine_name
        sub_dir.mkdir(parents=True)
        emitted: list[dict[str, Any]] = []

        async def _sink(msg: dict[str, Any], bucket: list[dict[str, Any]] = emitted) -> None:
            bucket.append(msg)

        cli = _CancelUsageCli(save_dir=sub_dir)
        mgr = SessionManager(save_dir=sub_dir, cli_provider=cli, wire_adapter=adapter)
        dispatcher = AcpDispatcher(
            notification_sink=_sink,
            session_manager=mgr,
            quota_tracker=QuotaTracker(
                provider=QuotaProvider(cli_provider=cli),
                state_store=cli.state_store,
            ),
            wire_adapter=adapter,
        )
        new_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/new",
                    "params": {"cwd": str(sub_dir), "mcpServers": []},
                }
            )
        )
        assert new_resp is not None
        sid = new_resp["result"]["sessionId"]
        dispatcher.pop_post_response_notifications()
        emitted.clear()

        # 1. Pre-cancelled prompt (cancel arrives before session/prompt)
        await dispatcher.handle_message(
            json.dumps({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": sid}})
        )
        pre_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/prompt",
                    "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "abort"}]},
                }
            )
        )
        assert pre_resp is not None
        assert pre_resp["result"]["stopReason"] == "cancelled"
        pre_post = dispatcher.pop_post_response_notifications()
        if engine_name == "v2":
            assert len(pre_post) == 1
            assert pre_post[0]["method"] == "_kiro.dev/metadata"
            assert pre_post[0]["params"].get("contextUsagePercentage") is not None
            assert pre_post[0]["params"]["meteringUsage"][0]["value"] == 0.0
        else:
            assert pre_post == []
            pre_kinds = [
                n["params"]["update"].get("_meta", {}).get("kiro", {}).get("kind")
                for n in emitted
                if n.get("method") == "session/update"
                and n["params"]["update"].get("sessionUpdate") == "session_info_update"
            ]
            assert "context_usage" in pre_kinds
            assert "turn_completion" in pre_kinds

        emitted.clear()

        # 2. Mid-turn cancelled prompt with partial token usage
        mid_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/prompt",
                    "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "work"}]},
                }
            )
        )
        assert mid_resp == {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "stopReason": "cancelled",
                "usage": {
                    "inputTokens": 24,
                    "outputTokens": 9,
                    "cachedReadTokens": 4,
                    "cachedWriteTokens": 0,
                },
            },
        }
        usage_updates = [
            n["params"]["update"]
            for n in emitted
            if n.get("method") == "session/update"
            and n["params"]["update"].get("sessionUpdate") == "usage_update"
        ]
        assert len(usage_updates) >= 1
        assert usage_updates[-1]["used"] == 1200

        mid_post = dispatcher.pop_post_response_notifications()
        if engine_name == "v2":
            assert len(mid_post) == 1
            assert mid_post[0]["method"] == "_kiro.dev/metadata"
            assert mid_post[0]["params"].get("contextUsagePercentage") is not None
            assert mid_post[0]["params"]["meteringUsage"][0]["value"] == 0.0
        else:
            assert mid_post == []
            mid_kinds = [
                n["params"]["update"].get("_meta", {}).get("kiro", {}).get("kind")
                for n in emitted
                if n.get("method") == "session/update"
                and n["params"]["update"].get("sessionUpdate") == "session_info_update"
            ]
            assert "context_usage" in mid_kinds
            assert "turn_completion" in mid_kinds


@pytest.mark.parametrize("engine_name", ["v2", "v3"])
async def test_acp_dispatcher_default_quota_tracker_uses_session_tier_credits(
    tmp_path: Path,
    engine_name: str,
) -> None:
    """Verify `AcpDispatcher` reports Ultra 20x (4000/800) credits on v2 and v3 wires."""
    from agy_acp.logic.wire import create_wire_adapter
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    class _DroppingUltraCli(_StubAgyAcpClientProvider):
        def __init__(self, store: SqliteStateStoreProvider) -> None:
            super().__init__(state_store=store)
            self._fractions = [0.95, 0.94]

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
                frac = self._fractions.pop(0) if self._fractions else 0.94
                return {
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
                                            "remaining_fraction": frac,
                                            "reset_time": "2026-10-24T07:52:04Z",
                                        },
                                        {
                                            "id": "gemini-weekly",
                                            "name": "Weekly Limit Remaining",
                                            "window": "weekly",
                                            "remaining_fraction": 0.90,
                                            "reset_time": "2026-10-24T12:34:28Z",
                                        },
                                    ],
                                }
                            ]
                        },
                    },
                }
            return {
                "status": "SUCCESS",
                "command": {
                    "name": "credits",
                    "data": {"remaining_credits": 0, "upgrade_uri": ""},
                },
            }

    store = SqliteStateStoreProvider(tmp_path / "ultra-disp-state")
    store.save_tier({"id": "g1-ultra-tier", "name": "Google AI Ultra", "upgradeUri": ""})
    cli = _DroppingUltraCli(store)
    adapter = create_wire_adapter(engine_name)
    emitted: list[dict[str, Any]] = []

    async def _sink(frame: dict[str, Any]) -> None:
        emitted.append(frame)

    # Note: omit quota_tracker=... to exercise AcpDispatcher's default wiring!
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=SessionManager(cli_provider=cli, state_store=store, wire_adapter=adapter),
        wire_adapter=adapter,
    )
    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()
    await dispatcher.wait_background_tasks()
    emitted.clear()

    turn1_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "hello"}]},
            }
        )
    )
    assert turn1_resp is not None
    assert turn1_resp["result"]["stopReason"] == "end_turn"
    await dispatcher.wait_background_tasks()
    dispatcher.pop_post_response_notifications()
    emitted.clear()

    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "hello 2"}]},
            }
        )
    )
    assert prompt_resp is not None
    assert prompt_resp["result"]["stopReason"] == "end_turn"

    post = dispatcher.pop_post_response_notifications()
    if engine_name == "v2":
        meta_frames = [n for n in [*emitted, *post] if n.get("method") == "_kiro.dev/metadata"]
        assert len(meta_frames) == 1
        meta = meta_frames[0]["params"]
        assert meta["creditsCapacity"] == 800.0
        assert meta["credits"] == 752.0
        assert meta["meteringUsage"] == [{"unit": "credit", "unitPlural": "credits", "value": 8.0}]
        assert meta["planTierId"] == "g1-ultra-tier"
    else:
        turn_completions = [
            n["params"]["update"]["_meta"]["kiro"]
            for n in emitted
            if n.get("method") == "session/update"
            and n["params"]["update"].get("sessionUpdate") == "session_info_update"
            and n["params"]["update"].get("_meta", {}).get("kiro", {}).get("kind")
            == "turn_completion"
        ]
        assert len(turn_completions) == 1
        assert turn_completions[0]["promptTurnSummaries"] == [{"unit": "credit", "usage": 8.0}]
    await dispatcher.aclose()


async def test_kas_v3_emits_live_mid_turn_usage_updates_without_cost_and_avoids_sum_fallback(
    tmp_path: Path,
) -> None:
    """Verify KAS v3 streams live mid-turn usage_update (no cost) and never sums turn tokens."""
    from agy_acp.logic.wire import V3KasWireAdapter

    turn_idx = 0

    class _LiveUsageCli(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            nonlocal turn_idx
            turn_idx += 1
            on_usage = kwargs.get("on_usage")
            on_text = kwargs.get("on_text")
            if turn_idx == 1:
                if on_text is not None:
                    await on_text("Unmeasured multi-step turn")
                return PromptOutcome(
                    conversation_id="conv-live-1",
                    stop_reason="end_turn",
                    response="Unmeasured multi-step turn",
                    usage={
                        "input_tokens": 391_156,
                        "output_tokens": 22_121,
                        "cache_read_tokens": 2_086_676,
                        "thinking_tokens": 0,
                        "total_tokens": 2_499_953,
                        "context_tokens": 0,
                    },
                )
            if on_usage is not None:
                await on_usage(42_000)
            if on_text is not None:
                await on_text("Step 1 ")
            if on_usage is not None:
                await on_usage(75_000)
            if on_text is not None:
                await on_text("Step 2")
            return PromptOutcome(
                conversation_id="conv-live-1",
                stop_reason="end_turn",
                response="Step 1 Step 2",
                usage={
                    "input_tokens": 200,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "thinking_tokens": 0,
                    "total_tokens": 250,
                    "context_tokens": 75_000,
                },
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(msg: dict[str, Any]) -> None:
        emitted.append(msg)

    cli = _LiveUsageCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli)
    qt = QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=mgr.state_store)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=qt,
        wire_adapter=V3KasWireAdapter(),
    )

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    emitted.clear()

    # Turn 1: no mid-turn usage_update -> reports 0% / 0 used rather than 249.9953% / 2,499,953
    resp1 = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]},
            }
        )
    )
    assert resp1 is not None
    assert resp1["result"]["stopReason"] == "end_turn"
    updates1 = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update" and isinstance(n.get("params"), dict)
    ]
    ctx_info1 = [
        u
        for u in updates1
        if u.get("sessionUpdate") == "session_info_update"
        and u.get("_meta", {}).get("kiro", {}).get("kind") == "context_usage"
    ]
    usage_upds1 = [u for u in updates1 if u.get("sessionUpdate") == "usage_update"]
    assert len(ctx_info1) == 1
    assert ctx_info1[0]["_meta"]["kiro"]["usagePercentage"] == 0.0
    assert len(usage_upds1) == 1
    assert usage_upds1[0]["used"] == 0
    assert "cost" not in usage_upds1[0]

    emitted.clear()

    # Turn 2: two live mid-turn usage_updates stream before chunks, all without `cost`
    resp2 = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "stream"}]},
            }
        )
    )
    assert resp2 is not None
    assert resp2["result"]["stopReason"] == "end_turn"
    updates2 = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update" and isinstance(n.get("params"), dict)
    ]
    usage_indices = [
        idx for idx, u in enumerate(updates2) if u.get("sessionUpdate") == "usage_update"
    ]
    chunk_indices = [
        idx for idx, u in enumerate(updates2) if u.get("sessionUpdate") == "agent_message_chunk"
    ]
    ctx_idx = next(
        idx
        for idx, u in enumerate(updates2)
        if u.get("sessionUpdate") == "session_info_update"
        and u.get("_meta", {}).get("kiro", {}).get("kind") == "context_usage"
    )
    comp_idx = next(
        idx
        for idx, u in enumerate(updates2)
        if u.get("sessionUpdate") == "session_info_update"
        and u.get("_meta", {}).get("kiro", {}).get("kind") == "turn_completion"
    )

    assert [updates2[i]["used"] for i in usage_indices] == [42_000, 75_000, 75_000]
    for i in usage_indices:
        assert "cost" not in updates2[i]
    assert usage_indices[0] < chunk_indices[0] < usage_indices[1] < chunk_indices[1]
    assert ctx_idx < usage_indices[-1] < comp_idx


def test_should_deny_lazy_call_mcp_tool_when_tool_or_server_disabled(tmp_path: Path) -> None:
    """Verify `_evaluate_tool_permission` denies `disabledTools` and `disabled` MCP servers."""
    from agy_acp.logic.acp_dispatcher import _to_normalized_tool_event
    from agy_acp.logic.wire import V3KasWireAdapter
    from agy_acp.providers.agy_acp_client import (
        _build_tool_event_from_record,
        _merge_tool_record,
    )

    async def _sink(_: dict[str, Any]) -> None:
        return None

    cli = _StubAgyAcpClientProvider(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, wire_adapter=V3KasWireAdapter())
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(
            provider=QuotaProvider(cli_provider=cli),
            state_store=cli.state_store,
        ),
        wire_adapter=V3KasWireAdapter(),
    )
    session_state: dict[str, Any] = {
        "agentAllowedTools": ["*"],
        "mcp_servers": [
            {
                "name": "kirocrew-core",
                "command": "kiro-mcp",
                "disabledTools": ["dangerous_op"],
            },
            {
                "name": "toggled-off-srv",
                "command": "off-mcp",
                "disabled": True,
            },
        ],
    }
    known_servers = ("kirocrew-core", "toggled-off-srv")

    def _make_lazy_norm(server_name: str, tool_name: str) -> Any:
        raw_args = {"ServerName": server_name, "ToolName": tool_name, "Arguments": {"x": 1}}
        merged = _merge_tool_record(
            {},
            {"title": "call_mcp_tool", "rawInput": raw_args, "status": "in_progress"},
        )
        _, _, ev = _build_tool_event_from_record(
            "tc_lazy",
            merged,
            known_mcp_servers=known_servers,
            status_override="in_progress",
        )
        return _to_normalized_tool_event(ev, known_mcp_servers=known_servers)

    assert (
        dispatcher._evaluate_tool_permission(
            session=session_state,
            spec=None,
            norm=_make_lazy_norm("kirocrew-core", "dangerous_op"),
            session_mode="auto",
        )
        == "deny"
    )
    assert (
        dispatcher._evaluate_tool_permission(
            session=session_state,
            spec=None,
            norm=_make_lazy_norm("toggled-off-srv", "any_tool"),
            session_mode="auto",
        )
        == "deny"
    )
    assert (
        dispatcher._evaluate_tool_permission(
            session=session_state,
            spec=None,
            norm=_make_lazy_norm("kirocrew-core", "safe_op"),
            session_mode="auto",
        )
        == "allow"
    )


@pytest.mark.parametrize("engine_name", ["v2", "v3"])
async def test_session_prompt_never_blocks_on_slow_quota_provider_and_conserves_kirocrew_credits(
    tmp_path: Path,
    engine_name: str,
) -> None:
    """Verify `session/prompt` completes without awaiting slow `/usage` & `/credits` CLI spawns."""
    import anyio

    from agy_acp.logic.wire import create_wire_adapter
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    class _SlowQuotaCli(_StubAgyAcpClientProvider):
        def __init__(self, store: SqliteStateStoreProvider) -> None:
            super().__init__(state_store=store)
            self._fractions = [0.95, 0.94, 0.93]
            self.slash_gate = anyio.Event()
            self.slash_completed_calls = 0

        async def run_slash_command(
            self,
            command: str,
            *,
            cwd: str | None = None,
            session_id: str | None = None,
            timeout: float = 30.0,
        ) -> dict[str, Any]:
            _ = (cwd, session_id, timeout)
            await self.slash_gate.wait()
            self.slash_completed_calls += 1
            if command.lstrip("/") == "usage":
                frac = self._fractions.pop(0) if self._fractions else 0.93
                return {
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
                                            "remaining_fraction": frac,
                                            "reset_time": "2026-10-24T07:52:04Z",
                                        },
                                        {
                                            "id": "gemini-weekly",
                                            "name": "Weekly Limit Remaining",
                                            "window": "weekly",
                                            "remaining_fraction": 0.90,
                                            "reset_time": "2026-10-24T12:34:28Z",
                                        },
                                    ],
                                }
                            ]
                        },
                    },
                }
            return {
                "status": "SUCCESS",
                "command": {
                    "name": "credits",
                    "data": {"remaining_credits": 0, "upgrade_uri": ""},
                },
            }

    store = SqliteStateStoreProvider(tmp_path / f"slow-quota-{engine_name}")
    store.save_tier({"id": "g1-ultra-tier", "name": "Google AI Ultra", "upgradeUri": ""})
    cli = _SlowQuotaCli(store)
    adapter = create_wire_adapter(engine_name)
    emitted: list[dict[str, Any]] = []

    async def _sink(frame: dict[str, Any]) -> None:
        emitted.append(frame)

    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=SessionManager(cli_provider=cli, state_store=store, wire_adapter=adapter),
        wire_adapter=adapter,
    )

    with anyio.fail_after(5.0):
        new_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/new",
                    "params": {"cwd": str(tmp_path), "mcpServers": []},
                }
            )
        )
    assert new_resp is not None
    assert not cli.slash_gate.is_set() and cli.slash_completed_calls == 0
    sid = new_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()
    cli.slash_gate.set()
    with anyio.fail_after(5.0):
        await dispatcher.wait_background_tasks()
    assert cli.slash_completed_calls == 2
    emitted.clear()

    # Turn 1 must NOT block while `slash_gate` holds `run_slash_command` open
    cli.slash_gate = anyio.Event()
    with anyio.fail_after(5.0):
        p1 = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "fast turn 1"}],
                    },
                }
            )
        )
    assert p1 is not None and p1["result"]["stopReason"] == "end_turn"
    assert not cli.slash_gate.is_set() and cli.slash_completed_calls == 2

    # Release gate and let post-turn background refresh finish (0.95 -> 0.94 => -8.0)
    cli.slash_gate.set()
    with anyio.fail_after(5.0):
        await dispatcher.wait_background_tasks()
    assert cli.slash_completed_calls == 4
    emitted.clear()
    dispatcher.pop_post_response_notifications()

    # Turn 2 also returns without awaiting `slash_gate` AND delivers conserved 8.0 credits!
    cli.slash_gate = anyio.Event()
    with anyio.fail_after(5.0):
        p2 = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "fast turn 2"}],
                    },
                }
            )
        )
    assert p2 is not None and p2["result"]["stopReason"] == "end_turn"
    assert not cli.slash_gate.is_set() and cli.slash_completed_calls == 4

    post = dispatcher.pop_post_response_notifications()
    if engine_name == "v2":
        meta_frames = [n for n in [*emitted, *post] if n.get("method") == "_kiro.dev/metadata"]
        assert len(meta_frames) == 1
        meta = meta_frames[0]["params"]
        assert meta["creditsCapacity"] == 800.0
        assert meta["credits"] == 752.0
        assert meta["meteringUsage"] == [{"unit": "credit", "unitPlural": "credits", "value": 8.0}]
        assert meta["planTierId"] == "g1-ultra-tier"
    else:
        turn_completions = [
            n["params"]["update"]["_meta"]["kiro"]
            for n in emitted
            if n.get("method") == "session/update"
            and n["params"]["update"].get("sessionUpdate") == "session_info_update"
            and n["params"]["update"].get("_meta", {}).get("kiro", {}).get("kind")
            == "turn_completion"
        ]
        assert len(turn_completions) == 1
        assert turn_completions[0]["promptTurnSummaries"] == [{"unit": "credit", "usage": 8.0}]

    # Shutdown cancels any in-flight gated background refresh immediately without awaiting gate
    with anyio.fail_after(5.0):
        await dispatcher.aclose()
    assert not cli.slash_gate.is_set() and cli.slash_completed_calls == 4


async def test_dispatcher_strips_retry_warnings_in_live_stream_persistence_replay_and_fork(
    tmp_path: Path,
) -> None:
    """Retry banners never leak into streamed chunks, persisted turns, replay, or fork summaries."""
    from agy_acp.logic.wire.v3_kas import V3KasWireAdapter
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    class _RetryStubCli(_StubAgyAcpClientProvider):
        async def stream_prompt(
            self,
            *,
            cwd: str,
            prompt: str = "",
            on_text: Callable[[str], Awaitable[None]] | None = None,
            on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
            **_extra: Any,
        ) -> PromptOutcome:
            _ = (cwd, prompt, _extra)
            raw_banner_1 = (
                "\n\u26a0 API error (attempt 1): request failed: Post "
                '"https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent": '
                "read: connection reset by peer\n"
            )
            raw_banner_2 = (
                "\n\u26a0 API error (attempt 2): request failed: Post "
                '"https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent": '
                "read: operation timed out\n"
            )
            if on_text is not None:
                await on_text(raw_banner_1[:14])
                await on_text(raw_banner_1[14:])
                await on_text("\n\u26a0 API ")
            if on_tool_call is not None:
                await on_tool_call(
                    ToolCallEvent(
                        tool_call_id="tc_retry_boundary",
                        title="Read file",
                        kind="read",
                        status="in_progress",
                        raw_input={"path": str(tmp_path / "a.txt")},
                    )
                )
            if on_text is not None:
                await on_text(raw_banner_2 + "<!-- nothing_to_do -->")
            return PromptOutcome(
                conversation_id="conv-retry-1",
                stop_reason="end_turn",
                response=raw_banner_1 + "\n\u26a0 API " + raw_banner_2 + "<!-- nothing_to_do -->",
                usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            )

    emitted: list[dict[str, Any]] = []

    async def _sink(frame: dict[str, Any]) -> None:
        emitted.append(frame)

    store = SqliteStateStoreProvider(tmp_path / "state")
    cli = _RetryStubCli(state_store=store)
    v3 = V3KasWireAdapter()
    mgr = SessionManager(cli_provider=cli, state_store=store, wire_adapter=v3)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        quota_tracker=QuotaTracker(provider=QuotaProvider(cli_provider=cli), state_store=store),
        wire_adapter=v3,
    )

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None
    sid = new_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()
    await dispatcher.wait_background_tasks()
    emitted.clear()

    prompt_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "[Heartbeat]"}],
                    "_meta": {"kiro": {"ttft": True}},
                },
            }
        )
    )
    assert prompt_resp is not None and "result" in prompt_resp

    msg_updates = [
        n["params"]["update"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
    ]
    combined_streamed = "".join(u["content"]["text"] for u in msg_updates)
    assert "API error (attempt" not in combined_streamed
    assert combined_streamed == "\n\u26a0 API <!-- nothing_to_do -->"
    # First displayed chunk (`\n⚠ API `) gets TTFT metadata, not a suppressed retry chunk
    assert "ttft" in msg_updates[0].get("_meta", {}).get("kiro", {})

    stored_turns = store.get_session_turns(sid)
    assert len(stored_turns) == 1
    assert stored_turns[0]["assistant_text"] == "\n\u26a0 API <!-- nothing_to_do -->"

    # Seed legacy polluted turns in SQLite and verify replay + fork sanitize them
    store.append_session_turn(
        session_id=sid,
        user_message_id="msg_legacy_1",
        prompt_blocks=[{"type": "text", "text": "legacy 1"}],
        assistant_text=(
            '\n\u26a0 API error (attempt 1): request failed (code 502): HTML response "Error 502"\n'
            "Saved [check_upgrades.py]..."
        ),
    )
    store.append_session_turn(
        session_id=sid,
        user_message_id="msg_legacy_empty",
        prompt_blocks=[{"type": "text", "text": "legacy banner only"}],
        assistant_text="\n\u26a0 API error (attempt 1): read: operation timed out\n",
    )

    emitted.clear()
    load_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/load",
                "params": {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert load_resp is not None and "result" in load_resp
    replay_msgs = [
        n["params"]["update"]["content"]["text"]
        for n in emitted
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
    ]
    assert replay_msgs == [
        "\n\u26a0 API <!-- nothing_to_do -->",
        "Saved [check_upgrades.py]...",
    ]

    fork_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/fork",
                "params": {"sessionId": sid, "cwd": str(tmp_path)},
            }
        )
    )
    assert fork_resp is not None and "result" in fork_resp
    fork_sid = fork_resp["result"]["sessionId"]
    fork_state = store.get_session(fork_sid)
    assert fork_state is not None
    assert "API error (attempt" not in str(fork_state.get("compactedSummary") or "")
    for ft in store.get_session_turns(fork_sid):
        assert "API error (attempt" not in ft["assistant_text"]

    await dispatcher.aclose()


async def test_kiro_3_4_0_strict_session_load_and_additional_directories_lifecycle(
    tmp_path: Path,
) -> None:
    """Verify kiro-cli 3.4.0 `strictSessionLoad` (`requireExisting`) and `additionalDirectories`."""
    from agy_acp.logic.wire import create_wire_adapter
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    captured_kwargs: list[dict[str, Any]] = []

    class _RecordingCliProvider(_StubAgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            captured_kwargs.append(dict(kwargs))
            return PromptOutcome(
                conversation_id="conv_340",
                stop_reason="end_turn",
                response="ok",
                usage={"input_tokens": 5, "output_tokens": 3},
            )

    store = SqliteStateStoreProvider(tmp_path / "state.db")
    adapter = create_wire_adapter("v3")
    mgr = SessionManager(
        state_store=store,
        cli_provider=_RecordingCliProvider(state_store=store),
        wire_adapter=adapter,
    )
    dispatcher = AcpDispatcher(session_manager=mgr, wire_adapter=adapter)

    async def _rpc(req_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = await dispatcher.handle_message(
            json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        )
        assert resp is not None
        return resp

    init_resp = await _rpc(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    agent_caps = init_resp["result"]["agentCapabilities"]
    assert agent_caps["sessionCapabilities"]["additionalDirectories"] == {}
    assert agent_caps["_meta"]["kiro"]["strictSessionLoad"] is True

    bad_dirs_resp = await _rpc(
        2,
        "session/new",
        {"cwd": str(tmp_path), "additionalDirectories": "not-a-list", "mcpServers": []},
    )
    assert bad_dirs_resp["error"]["code"] == -32602

    bad_elem_resp = await _rpc(
        3,
        "session/new",
        {"cwd": str(tmp_path), "additionalDirectories": [123], "mcpServers": []},
    )
    assert bad_elem_resp["error"]["code"] == -32602

    extra_a = str(tmp_path / "extra-a")
    extra_b = str(tmp_path / "extra-b")
    new_resp = await _rpc(
        4,
        "session/new",
        {
            "cwd": str(tmp_path),
            "additionalDirectories": [extra_a, extra_b],
            "mcpServers": [],
        },
    )
    sid = new_resp["result"]["sessionId"]

    list_resp = await _rpc(5, "session/list", {})
    assert list_resp["result"]["sessions"][0]["additionalDirectories"] == [extra_a, extra_b]

    # Invalid `_meta.kiro.requireExisting` type returns -32602
    bad_strict = await _rpc(
        6,
        "session/load",
        {
            "sessionId": sid,
            "cwd": str(tmp_path),
            "mcpServers": [],
            "_meta": {"kiro": {"requireExisting": "yes"}},
        },
    )
    assert bad_strict["error"]["code"] == -32602

    # Strict missing session load returns -32002
    missing_strict = await _rpc(
        7,
        "session/load",
        {
            "sessionId": "00000000-0000-0000-0000-000000000000",
            "cwd": str(tmp_path),
            "mcpServers": [],
            "_meta": {"kiro": {"requireExisting": True}},
        },
    )
    assert missing_strict["error"]["code"] == -32002

    # Valid `requireExisting: True` load inherits `additionalDirectories` when omitted
    load_resp = await _rpc(
        8,
        "session/load",
        {
            "sessionId": sid,
            "cwd": str(tmp_path),
            "mcpServers": [],
            "_meta": {"kiro": {"requireExisting": True}},
        },
    )
    assert load_resp["result"]["sessionId"] == sid

    await _rpc(
        9,
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "hello"}]},
    )
    assert captured_kwargs[-1]["additional_directories"] == [extra_a, extra_b]

    # Fork inherits parent's `additionalDirectories` by default and allows explicit override
    fork_inherited = await _rpc(10, "session/fork", {"sessionId": sid})
    fork_sid = fork_inherited["result"]["sessionId"]
    fork_s1 = store.get_session(fork_sid)
    assert fork_s1 is not None
    assert fork_s1["additionalDirectories"] == [extra_a, extra_b]

    extra_c = str(tmp_path / "extra-c")
    fork_overridden = await _rpc(
        11,
        "session/fork",
        {"sessionId": sid, "additionalDirectories": [extra_c]},
    )
    fork_sid_2 = fork_overridden["result"]["sessionId"]
    fork_s2 = store.get_session(fork_sid_2)
    assert fork_s2 is not None
    assert fork_s2["additionalDirectories"] == [extra_c]

    await dispatcher.aclose()
    await mgr.close()
