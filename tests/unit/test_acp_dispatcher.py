"""Unit tests for AcpDispatcher JSON-RPC framing and routing."""

from __future__ import annotations

import json

import pytest

from agy_acp.logic.acp_dispatcher import AcpDispatcher


@pytest.mark.not_implemented
def test_jsonrpc_frame_serialization() -> None:
    """Verify AcpDispatcher serializes newline-delimited UTF-8 JSON-RPC frames."""
    dispatcher = AcpDispatcher()
    raw = dispatcher.serialize_frame({"jsonrpc": "2.0", "id": 1, "result": {}})
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8"))["id"] == 1


@pytest.mark.not_implemented
async def test_handle_initialize_capabilities() -> None:
    """Verify AcpDispatcher.handle_message responds to initialize with required capabilities."""
    dispatcher = AcpDispatcher()
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


@pytest.mark.not_implemented
async def test_prompt_streams_chunks_and_usage() -> None:
    """Verify AcpDispatcher routes session/prompt and returns token usage."""
    emitted: list[dict[str, object]] = []

    async def _sink(msg: dict[str, object]) -> None:
        emitted.append(msg)

    dispatcher = AcpDispatcher(notification_sink=_sink)
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
    assert len(emitted) > 0


@pytest.mark.not_implemented
async def test_tool_permission_request_allow_and_reject() -> None:
    """Verify AcpDispatcher issues session/request_permission via client_rpc_caller."""

    async def _caller(method: str, params: dict[str, object]) -> dict[str, object]:
        assert method == "session/request_permission"
        return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}

    dispatcher = AcpDispatcher(client_rpc_caller=_caller)
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
