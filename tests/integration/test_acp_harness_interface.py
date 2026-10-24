"""Executable black-box integration verifications for KiroCrew ACP interface (REQ-ACP-001..011)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from tests.integration.conftest import HarnessFactory


async def test_req_acp_001_stdio_framing_and_stdout_purity(
    acp_harness_factory: HarnessFactory,
) -> None:
    """Verify REQ-ACP-001: newline-delimited JSON-RPC 2.0 over stdio with zero non-JSON stdout."""
    harness = await acp_harness_factory.spawn(["acp", "serve"])
    response = await harness.send_request(
        "initialize",
        {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
            "clientInfo": {"name": "kiro-crew", "version": "0.1.0"},
        },
    )
    assert response["jsonrpc"] == "2.0"
    assert "result" in response


async def test_req_acp_002_kirocrew_cli_invocation_variants(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-002: all KiroCrew backend CLI invocation patterns and gate markers."""
    marker = tmp_path / "gate.marker"
    harness = await acp_harness_factory.spawn(
        ["serve", "--profile", "acp", "--with-builtin", "developer", "--agent", "kirocrew"],
        env={
            "KIROCREW_DSH_GATE_SESSION": "nonce-123",
            "KIROCREW_DSH_GATE_MARKER": str(marker),
        },
    )
    await harness.send_request(
        "initialize",
        {"protocolVersion": "2025-08-22", "clientCapabilities": {}},
    )
    session_resp = await harness.send_request(
        "session/new",
        {"cwd": str(tmp_path), "mcpServers": []},
    )
    assert "result" in session_resp
    assert marker.read_text(encoding="utf-8") == "nonce-123\tloaded\n"


async def test_req_acp_003_initialize_handshake_and_capabilities(
    acp_harness_factory: HarnessFactory,
) -> None:
    """Verify REQ-ACP-003: initialize response advertises KiroCrew-required capabilities."""
    harness = await acp_harness_factory.spawn(["acp", "serve"])
    resp = await harness.send_request(
        "initialize",
        {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
            "clientInfo": {"name": "kiro-crew", "version": "0.1.0"},
        },
    )
    assert "result" in resp
    result = resp["result"]
    caps = result["agentCapabilities"]
    assert caps["loadSession"] is True
    assert caps["mcpCapabilities"]["http"] is True
    assert caps["promptCapabilities"]["image"] is True
    assert result["agentInfo"]["name"] == "agy-acp"
    assert result["agentInfo"]["version"]


async def test_req_acp_004_session_new_mcp_and_config_options(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-004: session/new accepts MCP servers and returns configOptions."""
    harness = await acp_harness_factory.spawn(["acp", "serve"])
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [
                {
                    "name": "local_stdio",
                    "command": "echo",
                    "args": ["hello"],
                    "env": [{"name": "FOO", "value": "BAR"}],
                },
                {
                    "type": "http",
                    "name": "kirocrew_mcp_pool",
                    "url": "http://127.0.0.1:9999/mcp",
                    "headers": [{"name": "Authorization", "value": "Bearer secret"}],
                },
            ],
        },
    )
    assert "result" in resp
    result = resp["result"]
    assert len(result["sessionId"]) >= 32
    assert "modes" in result and "currentModeId" in result["modes"]
    assert "models" in result and "availableModels" in result["models"]
    option_ids = {opt["id"] for opt in result["configOptions"]}
    assert {"model", "mode", "reasoning_effort"}.issubset(option_ids)


async def test_req_acp_005_session_load_resumes_existing_session(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-005: session/load restores a persisted sessionId across processes."""
    save_dir = tmp_path / "sessions"
    h1 = await acp_harness_factory.spawn(["acp", "serve", "--save-dir", str(save_dir)])
    await h1.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await h1.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in new_resp
    session_id = new_resp["result"]["sessionId"]
    await h1.close()

    h2 = await acp_harness_factory.spawn(["acp", "serve", "--save-dir", str(save_dir)])
    await h2.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    load_resp = await h2.send_request(
        "session/load",
        {"sessionId": session_id, "cwd": str(tmp_path), "mcpServers": []},
    )
    assert "result" in load_resp
    assert load_resp["result"]["sessionId"] == session_id


@pytest.mark.not_implemented
async def test_req_acp_006_multiplexed_concurrent_sessions(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-006: single ACP process multiplexes multiple concurrent sessions."""
    harness = await acp_harness_factory.spawn(["acp", "serve"])
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    r1 = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    r2 = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in r1 and "result" in r2
    assert r1["result"]["sessionId"] != r2["result"]["sessionId"]


@pytest.mark.not_implemented
async def test_req_acp_007_set_mode_model_and_config_options(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-007: session/set_mode, session/set_model, and session/set_config_option."""
    harness = await acp_harness_factory.spawn(["acp", "serve"])
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in new_resp
    sid = new_resp["result"]["sessionId"]

    mode_resp = await harness.send_request(
        "session/set_mode",
        {"sessionId": sid, "modeId": "read-only"},
    )
    assert "result" in mode_resp

    model_resp = await harness.send_request(
        "session/set_model",
        {"sessionId": sid, "modelId": "gemini-3.8-flash"},
    )
    assert "result" in model_resp

    cfg_resp = await harness.send_request(
        "session/set_config_option",
        {"sessionId": sid, "configId": "reasoning_effort", "value": "high"},
    )
    assert "result" in cfg_resp
    assert "configOptions" in cfg_resp["result"]


@pytest.mark.not_implemented
async def test_req_acp_008_session_prompt_streaming_thoughts_and_text(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-008: session/prompt streams agent_thought_chunk and agent_message_chunk."""
    harness = await acp_harness_factory.spawn(
        ["acp", "serve"],
        env={"AGY_ACP_FAKE_SDK": "1", "GEMINI_API_KEY": "test-key"},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in new_resp
    sid = new_resp["result"]["sessionId"]

    png_1x1 = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
        b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("ascii")

    prompt_resp = await harness.send_request(
        "session/prompt",
        {
            "sessionId": sid,
            "prompt": [
                {"type": "text", "text": "<kiro-skill-catalog></kiro-skill-catalog>\nSay hello"},
                {"type": "image", "data": png_1x1, "mimeType": "image/png"},
            ],
        },
    )
    assert "result" in prompt_resp
    assert prompt_resp["result"]["stopReason"] == "end_turn"
    assert "inputTokens" in prompt_resp["result"]["usage"]
    update_kinds = [
        n["params"]["update"]["sessionUpdate"]
        for n in harness.notifications
        if n.get("method") == "session/update"
    ]
    assert "agent_thought_chunk" in update_kinds
    assert "agent_message_chunk" in update_kinds


@pytest.mark.not_implemented
async def test_req_acp_009_tool_call_and_permission_gating(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-009: tool_call, session/request_permission, and tool_call_update."""
    harness = await acp_harness_factory.spawn(
        ["acp", "serve"],
        env={"AGY_ACP_FAKE_SDK": "1", "GEMINI_API_KEY": "test-key", "GOOSE_MODE": "approve"},
    )
    harness.on_permission_request(
        lambda params: {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in new_resp
    sid = new_resp["result"]["sessionId"]

    prompt_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Run a tool"}]},
    )
    assert "result" in prompt_resp
    assert prompt_resp["result"]["stopReason"] == "end_turn"
    assert any(r["method"] == "session/request_permission" for r in harness.server_requests)
    update_kinds = [
        n["params"]["update"]["sessionUpdate"]
        for n in harness.notifications
        if n.get("method") == "session/update"
    ]
    assert "tool_call" in update_kinds
    assert "tool_call_update" in update_kinds


@pytest.mark.not_implemented
async def test_req_acp_010_session_cancel_and_mid_turn_steer(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-010: session/cancel resolves in-flight turn with stopReason == cancelled."""
    harness = await acp_harness_factory.spawn(
        ["acp", "serve"],
        env={"AGY_ACP_FAKE_SDK": "slow", "GEMINI_API_KEY": "test-key"},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in new_resp
    sid = new_resp["result"]["sessionId"]

    await harness.send_notification("session/cancel", {"sessionId": sid})
    prompt_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Cancel me"}]},
    )
    assert "result" in prompt_resp
    assert prompt_resp["result"]["stopReason"] == "cancelled"


@pytest.mark.not_implemented
async def test_req_acp_011_session_teardown_methods(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-ACP-011: session/close, _kiro.dev/session/terminate, and _kiro/session/delete."""
    harness = await acp_harness_factory.spawn(["acp", "serve"])
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})

    for teardown_method in (
        "session/close",
        "_kiro.dev/session/terminate",
        "_kiro/session/delete",
    ):
        new_resp = await harness.send_request(
            "session/new", {"cwd": str(tmp_path), "mcpServers": []}
        )
        assert "result" in new_resp
        sid = new_resp["result"]["sessionId"]
        resp = await harness.send_request(teardown_method, {"sessionId": sid})
        assert resp.get("result") == {}
