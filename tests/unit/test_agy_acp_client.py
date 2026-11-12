"""Unit tests for `AgyAcpClientProvider` full-duplex ACP stdio client."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import anyio

from agy_acp.logic.tool_classifier import FALLBACK_MODELS, ToolCallEvent
from agy_acp.providers.agy_acp_client import (
    AgyAcpClientProvider,
    format_acp_mcp_servers,
    normalize_acp_content_blocks,
)
from agy_acp.providers.state_store import SqliteStateStoreProvider

MOCK_AGY_ACP_SERVER = '''#!/usr/bin/env python3
"""Deterministic mock `agy acp --stdio --process-per-session` server for unit tests."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

state_file = Path(os.environ["MOCK_AGY_STATE_FILE"])
write_lock = threading.Lock()
cancelled_sessions: set[str] = set()
pending_perm_events: dict[str, threading.Event] = {}
perm_outcomes: dict[str, dict] = {}


def append_record(record: dict) -> None:
    with write_lock:
        with state_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\\n")


def send_frame(payload: dict) -> None:
    with write_lock:
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()


append_record({"type": "spawn", "pid": os.getpid(), "argv": sys.argv[1:]})

if os.environ.get("MOCK_AGY_FAIL_INIT") == "1":
    sys.exit(1)


def handle_prompt(req_id, params: dict) -> None:
    sid = params["sessionId"]
    cancelled_sessions.discard(sid)
    blocks = params.get("prompt") or []
    text = " ".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    )

    crash_marker = Path(os.environ.get("MOCK_AGY_CRASH_ONCE_MARKER", ""))
    if "CRASH_ONCE" in text and crash_marker.name and not crash_marker.exists():
        crash_marker.write_text("crashed", encoding="utf-8")
        os._exit(1)

    # Always emit available_commands_update to verify client ignores it
    send_frame({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "available_commands_update",
                "availableCommands": [{"name": "goal", "description": "agy native"}],
            },
        },
    })

    if "WAIT_FOR_STEER" in text:
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Initial partial. "},
                },
            },
        })
        deadline = time.monotonic() + 5.0
        while sid not in cancelled_sessions and time.monotonic() < deadline:
            time.sleep(0.01)
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "cancelled" if sid in cancelled_sessions else "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
            },
        })
        return

    if "RUN_NATIVE_SUBAGENT" in text:
        tc_id = "call_sub_1"
        perm_id = f"perm_{uuid.uuid4().hex[:8]}"
        ev = threading.Event()
        pending_perm_events[perm_id] = ev
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tc_id,
                    "name": "invoke_subagent",
                    "args": {"TypeName": "research", "Role": "Researcher", "Prompt": "dig"},
                    "status": "in_progress",
                    "kind": "other",
                    "title": "invoke_subagent",
                    "rawInput": {"TypeName": "research", "Role": "Researcher", "Prompt": "dig"},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": perm_id,
            "method": "session/request_permission",
            "params": {
                "sessionId": sid,
                "toolCall": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tc_id,
                    "name": "invoke_subagent",
                    "args": {"TypeName": "research", "Role": "Researcher", "Prompt": "dig"},
                    "status": "pending",
                    "kind": "other",
                    "title": "invoke_subagent",
                    "rawInput": {"TypeName": "research", "Role": "Researcher", "Prompt": "dig"},
                },
                "options": [
                    {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
                ],
            },
        })
        ev.wait(timeout=5.0)
        outcome = perm_outcomes.get(perm_id, {})
        append_record({"type": "perm_outcome", "tool": "invoke_subagent", "outcome": outcome})

        # Now invoke sanctioned kirocrew-core spawn_run via call_mcp_tool
        mcp_tc_id = "call_spawn_run_1"
        mcp_perm_id = f"perm_{uuid.uuid4().hex[:8]}"
        mcp_ev = threading.Event()
        pending_perm_events[mcp_perm_id] = mcp_ev
        mcp_args = {
            "ServerName": "kirocrew-core",
            "ToolName": "spawn_run",
            "Arguments": {"tasks": [{"prompt": "delegated"}]},
        }
        send_frame({
            "jsonrpc": "2.0",
            "id": mcp_perm_id,
            "method": "session/request_permission",
            "params": {
                "sessionId": sid,
                "toolCall": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": mcp_tc_id,
                    "name": "call_mcp_tool",
                    "args": mcp_args,
                    "status": "pending",
                    "kind": "other",
                    "title": "call_mcp_tool",
                    "rawInput": mcp_args,
                },
                "options": [
                    {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
                ],
            },
        })
        mcp_ev.wait(timeout=5.0)
        mcp_outcome = perm_outcomes.get(mcp_perm_id, {})
        append_record({"type": "perm_outcome", "tool": "spawn_run", "outcome": mcp_outcome})
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": mcp_tc_id,
                    "name": "call_mcp_tool",
                    "args": mcp_args,
                    "status": "completed",
                    "kind": "other",
                    "rawOutput": {"output": "dispatched"},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Used spawn_run."},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {"inputTokens": 25, "outputTokens": 10, "totalTokens": 35},
            },
        })
        return

    if "RUN_COMMAND_PERM" in text:
        tc_id = "call_cmd_1"
        perm_id = f"perm_{uuid.uuid4().hex[:8]}"
        ev = threading.Event()
        pending_perm_events[perm_id] = ev
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tc_id,
                    "name": "run_command",
                    "args": {"CommandLine": "echo hello"},
                    "status": "in_progress",
                    "kind": "execute",
                    "title": "echo hello",
                    "rawInput": {"CommandLine": "echo hello"},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": perm_id,
            "method": "session/request_permission",
            "params": {
                "sessionId": sid,
                "toolCall": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tc_id,
                    "name": "run_command",
                    "args": {"CommandLine": "echo hello"},
                    "status": "pending",
                    "kind": "execute",
                    "title": "echo hello",
                    "rawInput": {"CommandLine": "echo hello"},
                },
                "options": [
                    {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
                ],
            },
        })
        ev.wait(timeout=5.0)
        outcome = perm_outcomes.get(perm_id, {}).get("outcome", {})
        append_record({"type": "perm_outcome", "tool": "run_command", "outcome": outcome})
        if outcome.get("outcome") == "cancelled" or sid in cancelled_sessions:
            send_frame({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "stopReason": "cancelled",
                    "usage": {"inputTokens": 12, "outputTokens": 2, "totalTokens": 14},
                },
            })
            return

        allowed = outcome.get("optionId") == "allow"
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tc_id,
                    "name": "run_command",
                    "args": {"CommandLine": "echo hello"},
                    "status": "completed" if allowed else "failed",
                    "kind": "execute",
                    "rawOutput": {"output": "hello\\n" if allowed else "denied"},
                },
            },
        })
        reply_text = "Allowed!" if allowed else "Handled denial gracefully."
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": reply_text},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {"inputTokens": 20, "outputTokens": 6, "totalTokens": 26},
            },
        })
        return

    if "READ_SECRET_NO_PERM" in text:
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_read_1",
                    "name": "read_file",
                    "args": {"Path": "/etc/secret.txt"},
                    "status": "in_progress",
                    "kind": "read",
                    "locations": [{"path": "/etc/secret.txt"}],
                    "rawInput": {"Path": "/etc/secret.txt"},
                },
            },
        })
        deadline = time.monotonic() + 2.0
        while sid not in cancelled_sessions and time.monotonic() < deadline:
            time.sleep(0.01)
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "cancelled" if sid in cancelled_sessions else "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 1, "totalTokens": 11},
            },
        })
        return

    # Standard streaming turn: thought + read tool + usage_update + text
    send_frame({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "Thinking step..."},
            },
        },
    })
    send_frame({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_read_ok",
                "name": "read_file",
                "args": {"Path": "/workspace/readme.md"},
                "status": "in_progress",
                "kind": "read",
                "locations": [{"path": "/workspace/readme.md"}],
                "rawInput": {"Path": "/workspace/readme.md"},
            },
        },
    })
    send_frame({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_read_ok",
                "name": "read_file",
                "args": {"Path": "/workspace/readme.md"},
                "status": "completed",
                "kind": "read",
                "locations": [{"path": "/workspace/readme.md"}],
                "rawOutput": {"output": "# Readme"},
            },
        },
    })
    send_frame({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "usage_update",
                "used": 4321,
                "size": 1000000,
            },
        },
    })
    send_frame({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"Echo: {text}"},
            },
        },
    })
    send_frame({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 25,
                "cachedReadTokens": 10,
                "thoughtTokens": 5,
                "totalTokens": 140,
            },
        },
    })
    if "EXIT_AFTER_TURN" in text:
        time.sleep(0.05)
        os._exit(1)


for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    msg = json.loads(line)
    append_record({"type": "recv", "pid": os.getpid(), "msg": msg})

    if "method" not in msg:
        resp_id = msg.get("id")
        if resp_id in pending_perm_events:
            perm_outcomes[resp_id] = msg.get("result") or {}
            pending_perm_events[resp_id].set()
        continue

    method = msg["method"]
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
                    "mcpCapabilities": {"http": True},
                    "_meta": {"antigravity": {"tier": True, "systemPrompt": {"append": True}}},
                },
                "agentInfo": {"name": "Antigravity ACP Server", "version": "1.5.6"},
            },
        })
    elif method == "session/new":
        new_sid = f"agy-{uuid.uuid4().hex[:8]}"
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": new_sid,
                "update": {
                    "sessionUpdate": "available_commands_update",
                    "availableCommands": [{"name": "goal", "description": "native"}],
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "sessionId": new_sid,
                "models": {
                    "availableModels": [
                        {"modelId": "gemini-4-carbon-medium", "name": "Gemini 4 Carbon (Medium)"},
                        {"modelId": "gemini-4-carbon-high", "name": "Gemini 4 Carbon (High)"},
                        {"modelId": "gemini-4-carbon-low", "name": "Gemini 4 Carbon (Low)"},
                    ],
                    "currentModelId": "gemini-4-carbon-medium",
                },
                "modes": {
                    "currentModeId": "default",
                    "availableModes": [
                        {"id": "default", "name": "Default"},
                        {"id": "plan", "name": "Plan"},
                    ],
                },
                "configOptions": [
                    {
                        "id": "thought_level",
                        "currentValue": "medium",
                        "options": [
                            {"value": "low", "name": "Low"},
                            {"value": "medium", "name": "Medium"},
                            {"value": "high", "name": "High"},
                        ],
                    },
                    {
                        "id": "mode",
                        "currentValue": "default",
                        "options": [
                            {"value": "default", "name": "Default"},
                            {"value": "plan", "name": "Plan"},
                        ],
                    },
                ],
                "_meta": {
                    "antigravity": {
                        "tier": {"id": "g1-ultra-lite-tier", "name": "Google AI Ultra"}
                    }
                },
            },
        })
    elif method == "session/load":
        target_sid = params.get("sessionId", "")
        if target_sid == "missing-agy-sid":
            send_frame({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32002, "message": "Session not found"},
            })
        else:
            send_frame({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "sessionId": target_sid,
                    "models": {
                        "availableModels": [
                            {
                                "modelId": "gemini-4-carbon-medium",
                                "name": "Gemini 4 Carbon (Medium)",
                            },
                            {"modelId": "gemini-4-carbon-high", "name": "Gemini 4 Carbon (High)"},
                        ],
                        "currentModelId": "gemini-4-carbon-medium",
                    },
                    "modes": {"currentModeId": "default", "availableModes": []},
                    "configOptions": [],
                    "_meta": {"antigravity": {"tier": {"id": "g1-ultra-lite-tier"}}},
                },
            })
    elif method in ("session/set_model", "session/set_mode", "session/close"):
        send_frame({"jsonrpc": "2.0", "id": req_id, "result": {}})
    elif method == "session/set_config_option":
        send_frame({"jsonrpc": "2.0", "id": req_id, "result": {"configOptions": []}})
    elif method == "session/cancel":
        cancelled_sessions.add(params.get("sessionId", ""))
        if req_id is not None:
            send_frame({"jsonrpc": "2.0", "id": req_id, "result": {}})
    elif method == "session/prompt":
        threading.Thread(target=handle_prompt, args=(req_id, params), daemon=True).start()
'''


def _make_mock_agy(tmp_path: Path) -> tuple[Path, Path]:
    script = tmp_path / "mock-agy"
    state_file = tmp_path / "mock-agy-state.jsonl"
    state_file.write_text("", encoding="utf-8")
    script.write_text(MOCK_AGY_ACP_SERVER, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script, state_file


def _read_mock_records(state_file: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in state_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_should_format_acp_mcp_servers_for_stdio_and_http_injecting_session_env() -> None:
    """Verify stdio/HTTP MCP servers serialize to ACP list format and skip disabled servers."""
    servers: list[dict[str, Any]] = [
        {
            "name": "disabled-srv",
            "command": "node",
            "args": ["srv.js"],
            "disabled": True,
        },
        {
            "name": "kirocrew-core",
            "command": "/usr/bin/kirocrew-mcp",
            "args": ["--stdio"],
            "env": {"STATIC_FLAG": "1"},
        },
        {
            "type": "http",
            "name": "remote-docs",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer secret"},
        },
    ]
    formatted = format_acp_mcp_servers(
        servers,
        session_env={
            "KIROCREW_SESSION_KEY": "sess-xyz",
            "KIROCREW_PORT": "9100",
            "KIROCREW_HOME": "",
            "UNRELATED_ENV": "ignored",
        },
    )

    assert len(formatted) == 2
    stdio_srv = formatted[0]
    assert stdio_srv["name"] == "kirocrew-core"
    assert stdio_srv["command"] == "/usr/bin/kirocrew-mcp"
    assert stdio_srv["args"] == ["--stdio"]
    env_map = {item["name"]: item["value"] for item in stdio_srv["env"]}
    assert env_map == {
        "STATIC_FLAG": "1",
        "KIROCREW_SESSION_KEY": "sess-xyz",
        "KIROCREW_PORT": "9100",
    }

    http_srv = formatted[1]
    assert http_srv == {
        "type": "http",
        "name": "remote-docs",
        "url": "https://example.com/mcp",
        "headers": [{"name": "Authorization", "value": "Bearer secret"}],
    }


def test_should_normalize_native_text_image_and_resource_content_blocks() -> None:
    """Verify native text, image, and resource ACP blocks pass through intact."""
    blocks = [
        {"type": "text", "text": "Inspect diagram"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {
            "type": "resource",
            "resource": {
                "uri": "file:///workspace/spec.md",
                "mimeType": "text/markdown",
                "text": "# Spec",
            },
        },
    ]
    normalized = normalize_acp_content_blocks(blocks)
    assert normalized == blocks
    assert normalize_acp_content_blocks(None, prompt="Fallback text") == [
        {"type": "text", "text": "Fallback text"}
    ]


async def test_should_initialize_with_empty_client_capabilities_and_discover_models(
    tmp_path: Path,
) -> None:
    """Verify `initialize` sends empty `clientCapabilities` and `discover_models` caches models."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        models = await client.discover_models(cwd=str(tmp_path))
        assert [m["modelId"] for m in models] == [
            "gemini-4-carbon-medium",
            "gemini-4-carbon-high",
            "gemini-4-carbon-low",
        ]
        assert store.load_models() == models

        records = _read_mock_records(state_file)
        spawns = [r for r in records if r["type"] == "spawn"]
        assert len(spawns) == 1
        assert spawns[0]["argv"] == ["acp", "--stdio", "--process-per-session"]
        assert "--dangerously-skip-permissions" not in spawns[0]["argv"]

        recvs = [r["msg"] for r in records if r["type"] == "recv" and "method" in r["msg"]]
        methods = [m["method"] for m in recvs]
        assert methods == ["initialize", "session/new", "session/close"]
        assert recvs[0]["params"] == {"protocolVersion": 1, "clientCapabilities": {}}

        # Cached call should not open another probe session
        cached = await client.discover_models(cwd=str(tmp_path))
        assert cached == models
        recvs_after = [
            r["msg"]
            for r in _read_mock_records(state_file)
            if r["type"] == "recv" and "method" in r["msg"]
        ]
        assert len(recvs_after) == 3

    # Fallback when initialize fails on a fresh store
    fail_store = SqliteStateStoreProvider(tmp_path / "fail-state")
    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=fail_store,
        env={"MOCK_AGY_STATE_FILE": str(state_file), "MOCK_AGY_FAIL_INIT": "1"},
    ) as failing_client:
        fallback = await failing_client.discover_models(cwd=str(tmp_path), refresh=True)
        assert fallback == FALLBACK_MODELS


async def test_should_create_and_load_backend_session_with_mcp_meta_mode_and_fallback(
    tmp_path: Path,
) -> None:
    """Verify `session/new` & `session/load` pass MCP, `_meta`, mode mapping, and model effort."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    mcp_servers: list[dict[str, Any]] = [
        {"name": "kirocrew-core", "command": "kiro-mcp", "args": ["--stdio"]},
        {"type": "http", "name": "remote-api", "url": "https://api.example.com/mcp"},
    ]

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        created = await client.create_backend_session(
            cwd=str(tmp_path),
            mcp_servers=mcp_servers,
            system_prompt_append="Injected Kiro instructions",
            model="gemini-4-carbon-medium",
            mode="auto",
            effort="high",
            session_env={"KIROCREW_SESSION_KEY": "sess-k1"},
        )
        agy_sid = created["agySessionId"]
        assert agy_sid.startswith("agy-")
        assert created["currentModelId"] == "gemini-4-carbon-high"
        assert created["tier"] == {"id": "g1-ultra-lite-tier", "name": "Google AI Ultra"}

        readonly_info = await client.create_backend_session(
            cwd=str(tmp_path),
            mode="read-only",
            effort="low",
        )
        assert readonly_info["agySessionId"].startswith("agy-")

        loaded = await client.load_backend_session(
            agy_session_id="existing-agy-123",
            cwd=str(tmp_path),
            mcp_servers=mcp_servers,
            system_prompt_append="Reloaded prompt",
        )
        assert loaded["agySessionId"] == "existing-agy-123"

        fallback_loaded = await client.load_backend_session(
            agy_session_id="missing-agy-sid",
            cwd=str(tmp_path),
        )
        assert fallback_loaded["agySessionId"].startswith("agy-")
        assert fallback_loaded["agySessionId"] != "missing-agy-sid"

        recvs = [
            r["msg"]
            for r in _read_mock_records(state_file)
            if r["type"] == "recv" and "method" in r["msg"]
        ]
        first_new = next(m for m in recvs if m["method"] == "session/new")
        assert first_new["params"]["_meta"] == {
            "antigravity": {"systemPrompt": {"append": "Injected Kiro instructions"}}
        }
        assert len(first_new["params"]["mcpServers"]) == 2
        assert first_new["params"]["mcpServers"][1]["type"] == "http"

        mode_configs = [
            m["params"]["value"]
            for m in recvs
            if m["method"] == "session/set_config_option" and m["params"].get("configId") == "mode"
        ]
        assert mode_configs[:2] == ["default", "plan"]

        set_models = [m["params"]["modelId"] for m in recvs if m["method"] == "session/set_model"]
        assert "gemini-4-carbon-high" in set_models


async def test_should_stream_thought_text_tool_and_usage_updates_and_pass_native_blocks(
    tmp_path: Path,
) -> None:
    """Verify streaming thought/text/tool/usage callbacks and native ACP block pass-through."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    store.upsert_session({"sessionId": "sess-stream", "cwd": str(tmp_path)})

    thoughts: list[str] = []
    texts: list[str] = []
    tool_calls: list[ToolCallEvent] = []
    tool_updates: list[ToolCallEvent] = []
    usages: list[int] = []

    async def on_thought(delta: str) -> None:
        thoughts.append(delta)

    async def on_text(delta: str) -> None:
        texts.append(delta)

    async def on_tool_call(ev: ToolCallEvent) -> bool:
        tool_calls.append(ev)
        return True

    async def on_tool_update(ev: ToolCallEvent) -> None:
        tool_updates.append(ev)

    async def on_usage(tokens: int) -> None:
        usages.append(tokens)

    native_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "Hello multimodal"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {
            "type": "resource",
            "resource": {
                "uri": "file:///workspace/context.txt",
                "mimeType": "text/plain",
                "text": "ctx",
            },
        },
    ]

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-stream",
            content_blocks=native_blocks,
            on_thought=on_thought,
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_tool_update=on_tool_update,
            on_usage=on_usage,
        )

    assert outcome.stop_reason == "end_turn"
    assert outcome.response == "Echo: Hello multimodal"
    assert "".join(texts) == "Echo: Hello multimodal"
    assert thoughts == ["Thinking step..."]
    assert usages == [4321]

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call_id == "call_read_ok"
    assert tool_calls[0].status == "in_progress"
    assert tool_calls[0].raw_input["path"] == "/workspace/readme.md"

    assert len(tool_updates) == 1
    assert tool_updates[0].tool_call_id == "call_read_ok"
    assert tool_updates[0].status == "completed"
    assert tool_updates[0].output == "# Readme"

    assert outcome.usage == {
        "input_tokens": 100,
        "output_tokens": 25,
        "thinking_tokens": 5,
        "cache_read_tokens": 10,
        "total_tokens": 140,
        "context_tokens": 4321,
    }
    persisted = store.get_session("sess-stream")
    assert persisted is not None
    assert persisted["contextTokens"] == 4321

    prompt_msgs = [
        r["msg"]
        for r in _read_mock_records(state_file)
        if r["type"] == "recv" and r["msg"].get("method") == "session/prompt"
    ]
    assert len(prompt_msgs) == 1
    assert prompt_msgs[0]["params"]["prompt"] == native_blocks


async def test_should_handle_pre_execution_permission_allow_and_deny_without_sigkill(
    tmp_path: Path,
) -> None:
    """Verify `session/request_permission` selects allow/deny with zero SIGKILL + read-only gate."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        allow_calls: list[ToolCallEvent] = []
        allow_updates: list[ToolCallEvent] = []

        async def approve_tool(ev: ToolCallEvent) -> bool:
            allow_calls.append(ev)
            return True

        async def record_allow_update(ev: ToolCallEvent) -> None:
            allow_updates.append(ev)

        res_allow = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-perm",
            prompt="RUN_COMMAND_PERM",
            on_tool_call=approve_tool,
            on_tool_update=record_allow_update,
        )
        pid_after_allow = client.worker_pid
        assert pid_after_allow is not None
        assert res_allow.response == "Allowed!"
        assert len(allow_calls) == 1
        assert allow_calls[0].raw_input["command"] == "echo hello"
        assert len(allow_updates) == 1
        assert allow_updates[0].status == "completed"

        deny_calls: list[ToolCallEvent] = []
        deny_updates: list[ToolCallEvent] = []

        async def deny_tool(ev: ToolCallEvent) -> bool:
            deny_calls.append(ev)
            return False

        async def record_deny_update(ev: ToolCallEvent) -> None:
            deny_updates.append(ev)

        res_deny = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-perm",
            agy_session_id=res_allow.conversation_id,
            prompt="RUN_COMMAND_PERM",
            on_tool_call=deny_tool,
            on_tool_update=record_deny_update,
        )
        assert client.worker_pid == pid_after_allow
        assert res_deny.stop_reason == "end_turn"
        assert res_deny.response == "Handled denial gracefully."
        assert len(deny_calls) == 1
        assert len(deny_updates) == 1
        assert deny_updates[0].status == "failed"

        # Read-only tool without `session/request_permission` denied by `on_tool_call -> False`
        ro_updates: list[ToolCallEvent] = []

        async def record_ro_update(ev: ToolCallEvent) -> None:
            ro_updates.append(ev)

        res_ro = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-perm",
            agy_session_id=res_allow.conversation_id,
            prompt="READ_SECRET_NO_PERM",
            on_tool_call=deny_tool,
            on_tool_update=record_ro_update,
        )
        assert client.worker_pid == pid_after_allow
        assert "not permitted" in res_ro.response
        assert len(ro_updates) == 1
        assert ro_updates[0].status == "failed"

        records = _read_mock_records(state_file)
        spawns = [r for r in records if r["type"] == "spawn"]
        assert len(spawns) == 1
        perm_outcomes = [r for r in records if r["type"] == "perm_outcome"]
        assert perm_outcomes[0]["outcome"] == {"outcome": "selected", "optionId": "allow"}
        assert perm_outcomes[1]["outcome"] == {"outcome": "selected", "optionId": "deny"}


async def test_should_auto_reject_native_subagent_tools_when_route_spawn_run_enabled(
    tmp_path: Path,
) -> None:
    """Verify native `invoke_subagent` is rejected silently while MCP `spawn_run` is allowed."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    seen_calls: list[ToolCallEvent] = []
    seen_updates: list[ToolCallEvent] = []

    async def on_tool_call(ev: ToolCallEvent) -> bool:
        seen_calls.append(ev)
        return True

    async def on_tool_update(ev: ToolCallEvent) -> None:
        seen_updates.append(ev)

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-sub",
            prompt="RUN_NATIVE_SUBAGENT",
            route_spawn_run=True,
            known_mcp_servers=("kirocrew-core",),
            on_tool_call=on_tool_call,
            on_tool_update=on_tool_update,
        )

    assert outcome.response == "Used spawn_run."
    assert len(seen_calls) == 1
    assert seen_calls[0].mcp_server == "kirocrew-core"
    assert seen_calls[0].mcp_tool == "spawn_run"
    assert len(seen_updates) == 1
    assert seen_updates[0].mcp_tool == "spawn_run"
    assert seen_updates[0].status == "completed"

    perm_records = [r for r in _read_mock_records(state_file) if r["type"] == "perm_outcome"]
    assert perm_records == [
        {
            "type": "perm_outcome",
            "tool": "invoke_subagent",
            "outcome": {"outcome": {"outcome": "selected", "optionId": "deny"}},
        },
        {
            "type": "perm_outcome",
            "tool": "spawn_run",
            "outcome": {"outcome": {"outcome": "selected", "optionId": "allow"}},
        },
    ]


async def test_should_not_deadlock_session_cancel_while_on_tool_call_awaits_approval(
    tmp_path: Path,
) -> None:
    """Verify `cancel_session` breaks a pending `on_tool_call` approval wait without deadlock."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    approval_started = anyio.Event()
    never_resolved = anyio.Event()

    async def blocking_approval(_ev: ToolCallEvent) -> bool:
        approval_started.set()
        await never_resolved.wait()
        return True

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        cancel_result: list[bool] = []

        async def trigger_cancel() -> None:
            await approval_started.wait()
            cancel_result.append(await client.cancel_session("sess-cancel"))

        with anyio.fail_after(4.0):
            async with anyio.create_task_group() as tg:
                tg.start_soon(trigger_cancel)
                outcome = await client.stream_prompt(
                    cwd=str(tmp_path),
                    session_id="sess-cancel",
                    prompt="RUN_COMMAND_PERM",
                    on_tool_call=blocking_approval,
                )

    assert cancel_result == [True]
    assert outcome.stop_reason == "cancelled"
    perm_records = [r for r in _read_mock_records(state_file) if r["type"] == "perm_outcome"]
    assert perm_records[0]["outcome"] == {"outcome": "cancelled"}


async def test_should_interrupt_active_turn_consume_steer_and_continue_on_same_agy_session(
    tmp_path: Path,
) -> None:
    """Verify `steer_session` cancels active turn, fires `on_steer_consumed`, and continues."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    first_chunk_seen = anyio.Event()
    consumed_messages: list[str] = []
    staged_file = tmp_path / "staged-steer.png"
    staged_file.write_bytes(b"png")
    cleared_file = tmp_path / "cleared-steer.png"
    cleared_file.write_bytes(b"png")

    async def on_text(chunk: str) -> None:
        if "Initial partial." in chunk:
            first_chunk_seen.set()

    async def on_steer_consumed(msg: str) -> None:
        consumed_messages.append(msg)

    steered_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "Steered follow-up"},
        {"type": "image", "data": "AA==", "mimeType": "image/png"},
    ]

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        steer_ok: list[bool] = []

        async def send_steer() -> None:
            await first_chunk_seen.wait()
            ok = await client.steer_session(
                "sess-steer",
                raw_message="Steered follow-up",
                content_blocks=steered_blocks,
                temp_files=[staged_file],
            )
            steer_ok.append(ok)

        with anyio.fail_after(5.0):
            async with anyio.create_task_group() as tg:
                tg.start_soon(send_steer)
                outcome = await client.stream_prompt(
                    cwd=str(tmp_path),
                    session_id="sess-steer",
                    prompt="WAIT_FOR_STEER",
                    on_text=on_text,
                    on_steer_consumed=on_steer_consumed,
                )

        # Verify `clear_steers` unlinks temp files when queued manually
        client._active_turns.add("sess-clear")
        await client.steer_session(
            "sess-clear",
            raw_message="Discard me",
            prompt="Discard me",
            temp_files=[cleared_file],
        )
        client._active_turns.discard("sess-clear")
        assert client.clear_steers("sess-clear") == ["Discard me"]
        assert not cleared_file.exists()

    assert steer_ok == [True]
    assert consumed_messages == ["Steered follow-up"]
    assert outcome.stop_reason == "end_turn"
    assert "Initial partial." in outcome.response
    assert "Echo: Steered follow-up" in outcome.response
    assert not staged_file.exists()

    prompt_msgs = [
        r["msg"]
        for r in _read_mock_records(state_file)
        if r["type"] == "recv" and r["msg"].get("method") == "session/prompt"
    ]
    assert len(prompt_msgs) == 2
    assert prompt_msgs[0]["params"]["sessionId"] == prompt_msgs[1]["params"]["sessionId"]
    assert prompt_msgs[1]["params"]["prompt"] == steered_blocks


async def test_should_recover_automatically_after_worker_crash_or_eof(
    tmp_path: Path,
) -> None:
    """Verify `AgyAcpClientProvider` respawns `agy acp` and reloads session after crash/EOF."""
    script, state_file = _make_mock_agy(tmp_path)
    crash_marker = tmp_path / "crash-once.marker"
    store = SqliteStateStoreProvider(tmp_path / "state")

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={
            "MOCK_AGY_STATE_FILE": str(state_file),
            "MOCK_AGY_CRASH_ONCE_MARKER": str(crash_marker),
        },
    ) as client:
        first = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-recover",
            prompt="EXIT_AFTER_TURN",
        )
        agy_sid = first.conversation_id
        assert first.response == "Echo: EXIT_AFTER_TURN"

        # Wait briefly for post-turn exit (`os._exit(1)`) to close stdout
        await anyio.sleep(0.15)

        second = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-recover",
            agy_session_id=agy_sid,
            prompt="CRASH_ONCE",
        )
        assert second.error == ""
        assert second.conversation_id == agy_sid
        assert second.response == "Echo: CRASH_ONCE"

    records = _read_mock_records(state_file)
    spawns = [r for r in records if r["type"] == "spawn"]
    assert len(spawns) == 3
    assert len({s["pid"] for s in spawns}) == 3
    loads = [
        r["msg"]
        for r in records
        if r["type"] == "recv" and r["msg"].get("method") == "session/load"
    ]
    assert len(loads) >= 2
    assert all(m["params"]["sessionId"] == agy_sid for m in loads)


async def test_should_emit_raw_carousel_deltas_sync_loaded_prefs_and_drop_legacy_stubs(
    tmp_path: Path,
) -> None:
    """Verify raw text delta streaming (no double carousel normalization) and pref updates."""
    assert not hasattr(AgyAcpClientProvider, "abort_active_process")
    assert not hasattr(AgyAcpClientProvider, "sync_mcp_servers")

    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    raw_carousel = "Intro\n\n````carousel\n![Slide](/tmp/slide.png)\n````\n"
    seen_chunks: list[str] = []

    async def _on_text(chunk: str) -> None:
        seen_chunks.append(chunk)

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        first = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-raw",
            prompt=raw_carousel,
            model="gemini-3.7-pro",
            mode="auto",
            effort="medium",
            on_text=_on_text,
        )
        assert "````carousel" in "".join(seen_chunks)
        assert "````carousel" in first.response

        # Second turn on already-loaded session updates model + mode + effort
        second = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-raw",
            agy_session_id=first.conversation_id,
            prompt="follow up",
            model="gemini-3.8-flash",
            mode="read-only",
            effort="high",
        )
        assert second.resolved_model == "gemini-3.8-flash"

    set_models = [
        r["msg"]["params"]["modelId"]
        for r in _read_mock_records(state_file)
        if r["type"] == "recv" and r["msg"].get("method") == "session/set_model"
    ]
    assert "gemini-3.8-flash" in set_models


def test_normalize_tool_arguments_populates_tool_use_purpose_and_task_commands() -> None:
    """Verify `_normalize_tool_arguments` maps summary/role, path aliases, and task commands."""
    from agy_acp.providers.agy_acp_client import _normalize_tool_arguments

    norm_cmd = _normalize_tool_arguments(
        {"CommandLine": "pytest -q", "toolSummary": "Run unit tests"},
        tool_name="run_command",
    )
    assert norm_cmd["command"] == "pytest -q"
    assert norm_cmd["description"] == "Run unit tests"
    assert norm_cmd["__toolUsePurpose"] == "Run unit tests"

    norm_find = _normalize_tool_arguments(
        {"TargetDirectory": "/repo/target"},
        tool_name="find_by_name",
    )
    assert norm_find["path"] == "/repo/target"

    norm_sub = _normalize_tool_arguments(
        {"TypeName": "research", "Role": "Codebase Auditor"},
        tool_name="invoke_subagent",
    )
    assert norm_sub["description"] == "Codebase Auditor"
    assert norm_sub["__toolUsePurpose"] == "Codebase Auditor"

    norm_list = _normalize_tool_arguments({"Action": "list"}, tool_name="manage_task")
    assert norm_list["command"] == "manage_task list"

    norm_input = _normalize_tool_arguments(
        {"CommandId": "cmd-9", "Input": "yes\n"},
        tool_name="send_command_input",
    )
    assert norm_input["command"] == "send_command_input cmd-9"


async def test_discover_models_refreshes_when_cache_older_than_one_day(tmp_path: Path) -> None:
    """Verify `discover_models` refreshes expired cache and falls back to stale rows on error."""
    from datetime import UTC, datetime, timedelta

    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path)
    stale_models = [
        {
            "modelId": "gemini-4-argon-medium",
            "name": "Gemini 4 Argon (Medium)",
            "description": "Old",
        }
    ]
    store.save_models(stale_models)
    stale_ts = (datetime.now(UTC) - timedelta(days=1, seconds=5)).isoformat()

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as provider:
        # Warm in-memory cache while fresh, then age SQLite rows past 24h TTL
        assert [m["modelId"] for m in await provider.discover_models(refresh=False)] == [
            "gemini-4-argon-medium"
        ]
        with store._connect() as conn:
            conn.execute("UPDATE models_cache SET updated_at = ?", (stale_ts,))

        refreshed = await provider.discover_models(cwd=str(tmp_path), refresh=False)
        assert [m["modelId"] for m in refreshed] == [
            "gemini-4-carbon-medium",
            "gemini-4-carbon-high",
            "gemini-4-carbon-low",
        ]
        assert store.load_models() == refreshed

    # When live discovery fails and cache expired, fall back to stale rows before FALLBACK_MODELS
    store.save_models(stale_models)
    with store._connect() as conn:
        conn.execute("UPDATE models_cache SET updated_at = ?", (stale_ts,))

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file), "MOCK_AGY_FAIL_INIT": "1"},
    ) as failing_provider:
        fallback = await failing_provider.discover_models(cwd=str(tmp_path), refresh=False)
        assert fallback == stale_models
