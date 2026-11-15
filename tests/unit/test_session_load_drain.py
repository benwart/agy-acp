"""Regression tests for `session/load` replay notification drain and isolation."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

from agy_acp.logic.tool_classifier import ToolCallEvent
from agy_acp.providers.agy_acp_client import AgyAcpClientProvider
from agy_acp.providers.state_store import SqliteStateStoreProvider

MOCK_REPLAY_ACP_SERVER = """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

burst_count = int(os.environ.get("MOCK_REPLAY_BURST_COUNT", "80"))


def write_batch(frames: list[dict]) -> None:
    payload = "".join(json.dumps(frame, separators=(",", ":")) + "\\n" for frame in frames)
    sys.stdout.write(payload)
    sys.stdout.flush()


for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "method" not in msg:
        continue

    method = msg["method"]
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        write_batch([
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": 1,
                    "agentCapabilities": {"loadSession": True},
                    "agentInfo": {"name": "Mock Antigravity ACP", "version": "1.5.6"},
                },
            }
        ])
    elif method == "session/load":
        sid = params["sessionId"]
        frames: list[dict] = []
        for idx in range(burst_count):
            tc_id = f"hist_tc_{idx}"
            frames.extend(
                [
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "agent_thought_chunk",
                                "content": {"type": "text", "text": f"HIST_THOUGHT_{idx} "},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": tc_id,
                                "name": "run_command",
                                "args": {"CommandLine": f"git push origin main #{idx}"},
                                "status": "in_progress",
                                "kind": "execute",
                                "rawInput": {"CommandLine": f"git push origin main #{idx}"},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": tc_id,
                                "name": "run_command",
                                "args": {"CommandLine": f"git push origin main #{idx}"},
                                "status": "completed",
                                "kind": "execute",
                                "rawOutput": {"output": f"HIST_OUT_{idx}"},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "usage_update",
                                "used": 90000 + idx,
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": f"HIST_TEXT_{idx} "},
                            },
                        },
                    },
                ]
            )
        frames.append(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "sessionId": sid,
                    "models": {
                        "availableModels": [
                            {
                                "modelId": "gemini-4-carbon-medium",
                                "name": "Gemini 4 Carbon (Medium)",
                            }
                        ],
                        "currentModelId": "gemini-4-carbon-medium",
                    },
                    "modes": {"currentModeId": "default", "availableModes": []},
                    "configOptions": [],
                },
            }
        )
        # Emit all historical session/update notifications + session/load result in one chunk.
        write_batch(frames)
    elif method in ("session/set_config_option", "session/set_model", "session/close"):
        sid = params.get("sessionId", "")
        batch: list[dict] = []
        if os.environ.get("MOCK_PREF_EMIT_UPDATES") == "1" and sid:
            for idx in range(25):
                batch.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": f"PREF_STALE_{idx} "},
                            },
                        },
                    }
                )
        batch.append({"jsonrpc": "2.0", "id": req_id, "result": {"configOptions": []}})
        write_batch(batch)
    elif method == "session/cancel":
        if req_id is not None:
            write_batch([{"jsonrpc": "2.0", "id": req_id, "result": {}}])
    elif method == "session/prompt":
        sid = params["sessionId"]
        blocks = params.get("prompt") or []
        prompt_text = " ".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
        )
        turn_frames: list[dict] = []
        if "MID_TURN_REPLAY" in prompt_text:
            turn_frames.extend(
                [
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "_meta": {"replay": True},
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": "REPLAY_META_TOP "},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "_meta": {"kiro": {"replay": True}},
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "replay_tc_kiro",
                                "name": "read_file",
                                "args": {"Path": "/workspace/old.py"},
                                "status": "in_progress",
                                "kind": "read",
                                "rawInput": {"Path": "/workspace/old.py"},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "agent_thought_chunk",
                                "_meta": {"replay": True},
                                "content": {"type": "text", "text": "REPLAY_UPDATE_META "},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "_meta": {"kiro": {"replay": True}},
                                "content": {"type": "text", "text": "REPLAY_UPDATE_KIRO "},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "usage_update",
                                "replay": True,
                                "used": 77777,
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "_meta": {"replay": True},
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": "REPLAY_ENVELOPE_META "},
                            },
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "_meta": {"replay": False, "kiro": {"replay": False}},
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": "NON_REPLAY_PREFIX:"},
                            },
                        },
                    },
                ]
            )
        turn_frames.extend(
            [
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "agent_thought_chunk",
                            "content": {"type": "text", "text": "LIVE_THOUGHT"},
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCallId": "live_tc_1",
                            "name": "read_file",
                            "args": {"Path": "/workspace/live.py"},
                            "status": "in_progress",
                            "kind": "read",
                            "rawInput": {"Path": "/workspace/live.py"},
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": "live_tc_1",
                            "name": "read_file",
                            "args": {"Path": "/workspace/live.py"},
                            "status": "completed",
                            "kind": "read",
                            "rawOutput": {"output": "print('ok')"},
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "usage_update",
                            "used": 1234,
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "LIVE_RESPONSE"},
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 20, "outputTokens": 5, "totalTokens": 25},
                    },
                },
            ]
        )
        write_batch(turn_frames)
"""


def _write_mock_server(tmp_path: Path) -> Path:
    script = tmp_path / "mock-replay-agy"
    script.write_text(MOCK_REPLAY_ACP_SERVER, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


async def test_session_load_replay_burst_does_not_bleed_into_stream_prompt(
    tmp_path: Path,
) -> None:
    """Historical `session/update` notifications from `session/load` never reach live callbacks."""
    script = _write_mock_server(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    thoughts: list[str] = []
    texts: list[str] = []
    tool_calls: list[str] = []
    tool_updates: list[str] = []
    usages: list[int] = []

    async def on_thought(chunk: str) -> None:
        thoughts.append(chunk)

    async def on_text(chunk: str) -> None:
        texts.append(chunk)

    async def on_tool_call(event: ToolCallEvent) -> bool:
        tool_calls.append(event.tool_call_id)
        return True

    async def on_tool_update(event: ToolCallEvent) -> None:
        tool_updates.append(event.tool_call_id)

    async def on_usage(tokens: int) -> None:
        usages.append(tokens)

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_REPLAY_BURST_COUNT": "80"},
    ) as client:
        outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="kc-sess-reload",
            agy_session_id="agy-existing-session",
            prompt="Run live prompt",
            skip_permissions=False,
            on_thought=on_thought,
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_tool_update=on_tool_update,
            on_usage=on_usage,
        )

    assert outcome.stop_reason == "end_turn"
    assert outcome.response == "LIVE_RESPONSE"
    assert texts == ["LIVE_RESPONSE"]
    assert thoughts == ["LIVE_THOUGHT"]
    assert tool_calls == ["live_tc_1"]
    assert tool_updates == ["live_tc_1"]
    assert usages == [1234]
    assert outcome.usage["context_tokens"] == 1234


async def test_wait_idle_barriers_drain_queued_notifications_without_fast_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if notifications enter `_notif_send`, `wait_idle()` drains them before turn start."""
    script = _write_mock_server(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    thoughts: list[str] = []
    texts: list[str] = []
    tool_calls: list[str] = []

    async def on_thought(chunk: str) -> None:
        thoughts.append(chunk)

    async def on_text(chunk: str) -> None:
        texts.append(chunk)

    async def on_tool_call(event: ToolCallEvent) -> bool:
        tool_calls.append(event.tool_call_id)
        return True

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_REPLAY_BURST_COUNT": "60", "MOCK_PREF_EMIT_UPDATES": "1"},
    ) as client:
        if hasattr(client, "_should_accept_notification"):
            monkeypatch.setattr(client, "_should_accept_notification", lambda _msg: True)

        await client.load_backend_session(
            agy_session_id="agy-direct-load",
            cwd=str(tmp_path),
        )
        assert client._conn is not None
        assert client._conn._inflight_frames == 0

        # Trigger preference update on already-loaded session + second fresh session load
        outcome_pref = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="kc-pref",
            agy_session_id="agy-direct-load",
            mode="plan",
            prompt="After pref update",
            skip_permissions=False,
            on_thought=on_thought,
            on_text=on_text,
            on_tool_call=on_tool_call,
        )
        assert outcome_pref.response == "LIVE_RESPONSE"

        texts.clear()
        thoughts.clear()
        tool_calls.clear()

        outcome_fresh = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="kc-fresh-load",
            agy_session_id="agy-second-load",
            prompt="Fresh reload turn",
            skip_permissions=False,
            on_thought=on_thought,
            on_text=on_text,
            on_tool_call=on_tool_call,
        )
        assert outcome_fresh.response == "LIVE_RESPONSE"
        assert texts == ["LIVE_RESPONSE"]
        assert thoughts == ["LIVE_THOUGHT"]
        assert tool_calls == ["live_tc_1"]


async def test_replay_stamped_notifications_ignored_mid_turn_and_in_handler(
    tmp_path: Path,
) -> None:
    """Replay-tagged `session/update` notifications are ignored even during an active turn."""
    script = _write_mock_server(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    thoughts: list[str] = []
    texts: list[str] = []
    tool_calls: list[str] = []
    usages: list[int] = []

    async def on_thought(chunk: str) -> None:
        thoughts.append(chunk)

    async def on_text(chunk: str) -> None:
        texts.append(chunk)

    async def on_tool_call(event: ToolCallEvent) -> bool:
        tool_calls.append(event.tool_call_id)
        return True

    async def on_usage(tokens: int) -> None:
        usages.append(tokens)

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_REPLAY_BURST_COUNT": "10"},
    ) as client:
        outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="kc-sess-mid-replay",
            agy_session_id="agy-mid-replay",
            prompt="MID_TURN_REPLAY",
            skip_permissions=False,
            on_thought=on_thought,
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_usage=on_usage,
        )

        assert outcome.response == "NON_REPLAY_PREFIX:LIVE_RESPONSE"
        assert texts == ["NON_REPLAY_PREFIX:", "LIVE_RESPONSE"]
        assert thoughts == ["LIVE_THOUGHT"]
        assert tool_calls == ["live_tc_1"]
        assert usages == [1234]

        # Also verify direct `_handle_notification` ignores replay-stamped frames on active turn
        direct_texts: list[str] = []

        async def capture_direct(chunk: str) -> None:
            direct_texts.append(chunk)

        from agy_acp.providers.agy_acp_client import _ActiveTurnState

        conn = client._conn
        assert conn is not None
        active = _ActiveTurnState(
            session_id="kc-direct",
            agy_session_id="agy-direct",
            route_spawn_run=False,
            skip_permissions=True,
            known_mcp_servers=(),
            on_thought=None,
            on_text=capture_direct,
            on_tool_call=None,
            on_tool_update=None,
            on_usage=None,
            collected_texts=[],
            usage={},
        )
        client._turn_states["agy-direct"] = active
        try:
            replay_variants: list[dict[str, Any]] = [
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "agy-direct",
                        "_meta": {"replay": True},
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "IGNORED_1"},
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "agy-direct",
                        "_meta": {"kiro": {"replay": True}},
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "IGNORED_2"},
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "agy-direct",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "_meta": {"replay": True},
                            "content": {"type": "text", "text": "IGNORED_3"},
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "agy-direct",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "replay": True,
                            "content": {"type": "text", "text": "IGNORED_4"},
                        },
                    },
                },
            ]
            for frame in replay_variants:
                await client._handle_notification(conn, frame)
            assert direct_texts == []
            assert active.collected_texts == []
        finally:
            client._turn_states.pop("agy-direct", None)
