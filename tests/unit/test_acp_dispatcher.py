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
    ) -> PromptOutcome:
        _ = (cwd, session_id, model, mode, effort, skip_permissions, sandbox, env)
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
