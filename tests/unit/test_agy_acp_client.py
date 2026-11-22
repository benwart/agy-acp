"""Unit tests for `AgyAcpClientProvider` full-duplex ACP stdio client."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import anyio
import pytest

from agy_acp.logic.tool_classifier import FALLBACK_MODELS, ToolCallEvent
from agy_acp.providers.agy_acp_client import (
    AgyAcpClientProvider,
    format_acp_mcp_servers,
    format_lazy_mcp_config,
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


def _extract_flag_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


_spawn_argv = sys.argv[1:]
_spawn_gemini_dir = _extract_flag_value(_spawn_argv, "--gemini-dir")
_spawn_app_data_dir = _extract_flag_value(_spawn_argv, "--app-data-dir")
_spawn_lazy_mcp_config = None
_spawn_mcp_config_mode = None
if _spawn_gemini_dir:
    _mcp_cfg_path = Path(_spawn_gemini_dir) / "config" / "mcp_config.json"
    if _mcp_cfg_path.is_file():
        _spawn_lazy_mcp_config = json.loads(_mcp_cfg_path.read_text(encoding="utf-8"))
        _spawn_mcp_config_mode = oct(_mcp_cfg_path.stat().st_mode & 0o777)

append_record({
    "type": "spawn",
    "pid": os.getpid(),
    "argv": _spawn_argv,
    "gemini_dir": _spawn_gemini_dir,
    "app_data_dir": _spawn_app_data_dir,
    "lazy_mcp_config": _spawn_lazy_mcp_config,
    "mcp_config_mode": _spawn_mcp_config_mode,
    "pythoninspect": os.environ.get("PYTHONINSPECT"),
    "pythonstartup": os.environ.get("PYTHONSTARTUP"),
    "stub_token": os.environ.get("KIROCREW_STUB_SESSION_TOKEN"),
})

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

    if "RETRY_THEN_CRASH" in text:
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "\\n\\u26a0 API error (attempt 1): connection reset by peer\\n",
                    },
                },
            },
        })
        time.sleep(0.05)
        os._exit(1)

    if "RETRY_SPLIT_AND_TOOL" in text:
        for piece in (
            "\\n\\u26a0 API error ",
            '(attempt 1): request failed (code 502): HTML response "Error 502"\\n',
            "\\n\\u26a0 API ",
        ):
            send_frame({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": piece},
                    },
                },
            })
        time.sleep(0.03)
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_retry_read",
                    "name": "read_file",
                    "args": {"Path": "/workspace/check.py"},
                    "status": "in_progress",
                    "kind": "read",
                    "title": "read_file",
                    "rawInput": {"Path": "/workspace/check.py"},
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
                    "toolCallId": "call_retry_read",
                    "name": "read_file",
                    "args": {"Path": "/workspace/check.py"},
                    "status": "completed",
                    "kind": "read",
                    "rawOutput": {"output": "ok"},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": (
                            "\\n\\u26a0\\ufe0f API error (attempt 2): read: operation timed out\\n"
                            "<!-- nothing_to_do -->"
                        ),
                    },
                },
            },
            "method": "session/update",
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {"inputTokens": 15, "outputTokens": 5, "totalTokens": 20},
            },
        })
        return

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

    if "EMPTY_THEN_ENV_PERM" in text:
        tc_id = "call_env_1"
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
                    "name": "edit_file",
                    "status": "in_progress",
                    "kind": "edit",
                    "title": "edit_file",
                    "rawInput": {},
                },
            },
        })
        time.sleep(0.04)
        send_frame({
            "jsonrpc": "2.0",
            "id": perm_id,
            "method": "session/request_permission",
            "params": {
                "sessionId": sid,
                "toolCall": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tc_id,
                    "name": "edit_file",
                    "args": {"Path": ".env"},
                    "status": "pending",
                    "kind": "edit",
                    "title": "edit_file",
                    "rawInput": {"Path": ".env"},
                },
                "options": [
                    {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
                ],
            },
        })
        ev.wait(timeout=5.0)
        outcome = perm_outcomes.get(perm_id, {}).get("outcome", {})
        append_record({"type": "perm_outcome", "tool": "edit_file_env", "outcome": outcome})
        if "DANGLING_NO_UPDATE" not in text:
            send_frame({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tc_id,
                        "name": "edit_file",
                        "args": {"Path": ".env"},
                        "status": "completed",
                        "kind": "edit",
                        "rawOutput": {"output": "LEAKED_ENV_SECRET=hunter2"},
                    },
                },
            })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {"inputTokens": 15, "outputTokens": 3, "totalTokens": 18},
            },
        })
        return

    if "READ_DENY_BUFFERED_LEAK" in text:
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_ro_leak",
                    "name": "read_file",
                    "args": {"Path": "/etc/passwd"},
                    "status": "in_progress",
                    "kind": "read",
                    "rawInput": {"Path": "/etc/passwd"},
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
                    "toolCallId": "call_ro_leak",
                    "name": "read_file",
                    "args": {"Path": "/etc/passwd"},
                    "status": "completed",
                    "kind": "read",
                    "rawOutput": {"output": "root:x:0:0"},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "LEAKED_THOUGHT"},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {"sessionUpdate": "usage_update", "used": 9999},
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "LEAKED_TEXT"},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 1, "totalTokens": 11},
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

    if "PARALLEL_DENY_AND_SIBLINGS" in text:
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_search_1",
                    "name": "search_web",
                    "args": {"query": "floci aws emulator"},
                    "status": "in_progress",
                    "kind": "fetch",
                    "rawInput": {"query": "floci aws emulator"},
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
                    "toolCallId": "call_read_sibling",
                    "name": "read_file",
                    "args": {"Path": "/tmp/docs.md"},
                    "status": "in_progress",
                    "kind": "read",
                    "locations": [{"path": "/tmp/docs.md"}],
                    "rawInput": {"Path": "/tmp/docs.md"},
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
                    "toolCallId": "call_mcp_sibling",
                    "name": "call_mcp_tool",
                    "args": {"ServerName": "kirocrew-work", "ToolName": "work_ledger_read"},
                    "status": "in_progress",
                    "kind": "other",
                    "rawInput": {"ServerName": "kirocrew-work", "ToolName": "work_ledger_read"},
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

    if "MULTI_STEP_NO_USAGE_UPDATE" in text:
        send_frame({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Done multi-step."},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 391156,
                    "cachedReadTokens": 2086676,
                    "outputTokens": 22121,
                    "totalTokens": 2499953,
                },
            },
        })
        return

    if "MULTI_STEP_USAGE_SNAPSHOTS" in text:
        for used_ctx in (50000, 80000):
            send_frame({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "usage_update",
                        "used": used_ctx,
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
                    "content": {"type": "text", "text": "Snapshots complete."},
                },
            },
        })
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 1200,
                    "outputTokens": 300,
                    "cachedReadTokens": 4500,
                    "thoughtTokens": 50,
                    "totalTokens": 6050,
                },
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
                        "tier": {
                            "id": os.environ.get("MOCK_AGY_TIER_ID", "g1-ultra-lite-tier"),
                            "name": os.environ.get("MOCK_AGY_TIER_NAME", "Google AI Ultra"),
                        }
                    }
                },
            },
        })
    elif method == "authenticate":
        send_frame({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "_meta": {
                    "antigravity": {
                        "tier": {
                            "id": os.environ.get("MOCK_AGY_TIER_ID", "g1-ultra-lite-tier"),
                            "name": os.environ.get("MOCK_AGY_TIER_NAME", "Google AI Ultra"),
                            "upgradeUri": "",
                        }
                    }
                }
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
        assert first_new["params"]["mcpServers"] == []

        spawns = [r for r in _read_mock_records(state_file) if r["type"] == "spawn"]
        lazy_spawns = [s for s in spawns if s.get("lazy_mcp_config") is not None]
        assert len(lazy_spawns) >= 1
        first_lazy = lazy_spawns[0]["lazy_mcp_config"]["mcpServers"]
        assert first_lazy["kirocrew-core"] == {
            "command": "kiro-mcp",
            "args": ["--stdio"],
            "env": {"KIROCREW_SESSION_KEY": "sess-k1"},
        }
        assert first_lazy["remote-api"] == {"url": "https://api.example.com/mcp"}

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

        # Wait deterministically for post-turn exit (`os._exit(1)`) to close stdout
        with anyio.fail_after(5.0):
            while client.worker_pid is not None:
                await anyio.sleep(0.01)

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


def test_normalize_tool_arguments_canonicalizes_document_body_and_target_file_keys() -> None:
    """Verify non-shell PascalCase body keys and `TargetFile` map to KiroCrew `tool_paths` keys."""
    from agy_acp.providers.agy_acp_client import _normalize_tool_arguments

    norm_write = _normalize_tool_arguments(
        {"Path": "/repo/doc.md", "Content": "# Heading\n"},
        tool_name="write_file",
    )
    assert norm_write["path"] == "/repo/doc.md"
    assert norm_write["content"] == "# Heading\n"
    assert "Content" not in norm_write

    norm_code = _normalize_tool_arguments(
        {"TargetFile": "/repo/mod.py", "CodeContent": "x = 1\n"},
        tool_name="write_to_file",
    )
    assert norm_code["path"] == "/repo/mod.py"
    assert norm_code["targetFile"] == "/repo/mod.py"
    assert norm_code["content"] == "x = 1\n"
    assert "CodeContent" not in norm_code

    norm_edit = _normalize_tool_arguments(
        {"Path": "/repo/mod.py", "Target": "x = 1", "Replacement": ""},
        tool_name="edit_file",
    )
    assert norm_edit["oldStr"] == "x = 1"
    assert norm_edit["newStr"] == ""
    assert "Target" not in norm_edit
    assert "Replacement" not in norm_edit

    norm_replace_content = _normalize_tool_arguments(
        {
            "TargetFile": "/repo/mod.py",
            "TargetContent": "old_block",
            "ReplacementContent": "new_block",
        },
        tool_name="replace_file_content",
    )
    assert norm_replace_content["targetFile"] == "/repo/mod.py"
    assert norm_replace_content["oldStr"] == "old_block"
    assert norm_replace_content["newStr"] == "new_block"
    assert "TargetContent" not in norm_replace_content
    assert "ReplacementContent" not in norm_replace_content

    orig_chunk = {
        "TargetContent": "chunk_old",
        "ReplacementContent": "chunk_new",
        "AllowMultiple": False,
        "StartLine": 10,
    }
    norm_chunks = _normalize_tool_arguments(
        {"TargetFile": "/repo/multi.py", "ReplacementChunks": [orig_chunk]},
        tool_name="multi_replace_file_content",
    )
    assert norm_chunks["targetFile"] == "/repo/multi.py"
    assert norm_chunks["ReplacementChunks"] == [
        {
            "oldText": "chunk_old",
            "newText": "chunk_new",
            "AllowMultiple": False,
            "StartLine": 10,
        }
    ]
    assert "TargetContent" in orig_chunk

    # Shell calls must NOT rewrite PascalCase keys into DOCUMENT_BODY_KEYS
    norm_shell = _normalize_tool_arguments(
        {"CommandLine": "echo ok", "Content": "keep_visible"},
        tool_name="run_command",
    )
    assert norm_shell["command"] == "echo ok"
    assert norm_shell["Content"] == "keep_visible"
    assert "content" not in norm_shell


async def test_discover_models_refreshes_when_cache_older_than_one_day(tmp_path: Path) -> None:
    """Verify `discover_models` refreshes expired cache and falls back to stale rows on error."""
    from datetime import timedelta

    from freezegun import freeze_time

    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path)
    stale_models = [
        {
            "modelId": "gemini-4-argon-medium",
            "name": "Gemini 4 Argon (Medium)",
            "description": "Old",
        }
    ]

    with freeze_time("2026-11-15T12:00:00Z", real_asyncio=True) as frozen_clock:
        store.save_models(stale_models)

        async with AgyAcpClientProvider(
            bin_path=script,
            state_store=store,
            env={"MOCK_AGY_STATE_FILE": str(state_file)},
        ) as provider:
            # Warm in-memory cache while fresh, then advance mock clock past 24h TTL
            assert [m["modelId"] for m in await provider.discover_models(refresh=False)] == [
                "gemini-4-argon-medium"
            ]
            frozen_clock.tick(timedelta(days=1, seconds=5))

            refreshed = await provider.discover_models(cwd=str(tmp_path), refresh=False)
            assert [m["modelId"] for m in refreshed] == [
                "gemini-4-carbon-medium",
                "gemini-4-carbon-high",
                "gemini-4-carbon-low",
            ]
            assert store.load_models() == refreshed

        # On live discovery failure with expired cache, fall back to stale rows
        store.save_models(stale_models)
        frozen_clock.tick(timedelta(days=1, seconds=5))

        async with AgyAcpClientProvider(
            bin_path=script,
            state_store=store,
            env={"MOCK_AGY_STATE_FILE": str(state_file), "MOCK_AGY_FAIL_INIT": "1"},
        ) as failing_provider:
            fallback = await failing_provider.discover_models(cwd=str(tmp_path), refresh=False)
            assert fallback == stale_models


async def test_w4_reevaluate_populated_input_fail_closed_callbacks_and_shielded_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `[H-1]`, `[H-2]`, `[H-4]`, `[M-1]`, and `[L-2]` in `AgyAcpClientProvider`."""
    monkeypatch.setenv("PYTHONINSPECT", "1")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/evil_startup.py")
    monkeypatch.setenv("KIROCREW_STUB_SESSION_TOKEN", "leaked-tok")

    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "w4_state")

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        # 1. [H-1] + [M-1]: Empty rawInput={} -> allow, then permission {"Path": ".env"} -> deny
        seen_tool_calls: list[ToolCallEvent] = []
        seen_tool_updates: list[ToolCallEvent] = []

        async def policy_deny_dotenv(ev: ToolCallEvent) -> bool:
            seen_tool_calls.append(ev)
            path_val = str(ev.raw_input.get("path") or ev.raw_input.get("Path") or "")
            return ".env" not in path_val

        async def collect_updates(ev: ToolCallEvent) -> None:
            seen_tool_updates.append(ev)

        res_env = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="s-w4",
            prompt="EMPTY_THEN_ENV_PERM",
            on_tool_call=policy_deny_dotenv,
            on_tool_update=collect_updates,
        )
        initial_pid = client.worker_pid
        assert initial_pid is not None
        assert len(seen_tool_calls) == 2
        assert len(seen_tool_updates) == 1
        assert seen_tool_updates[0].status == "failed"
        assert "LEAKED_ENV_SECRET" not in seen_tool_updates[0].output

        # Dangling rejected tool card without server `tool_call_update` still closes with "failed"
        seen_tool_updates.clear()
        await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="s-w4",
            agy_session_id=res_env.conversation_id,
            prompt="EMPTY_THEN_ENV_PERM DANGLING_NO_UPDATE",
            on_tool_call=policy_deny_dotenv,
            on_tool_update=collect_updates,
        )
        assert len(seen_tool_updates) == 1
        assert seen_tool_updates[0].status == "failed"

        # 2. [M-1]: Policy denial without `request_permission` drops post-denial chunks & output
        leaked_thoughts: list[str] = []
        leaked_texts: list[str] = []
        leaked_usages: list[int] = []
        ro_updates: list[ToolCallEvent] = []

        async def deny_all(_ev: ToolCallEvent) -> bool:
            return False

        async def rec_thought(t: str) -> None:
            leaked_thoughts.append(t)

        async def rec_text(t: str) -> None:
            leaked_texts.append(t)

        async def rec_usage(u: int) -> None:
            leaked_usages.append(u)

        async def rec_ro_update(ev: ToolCallEvent) -> None:
            ro_updates.append(ev)

        res_ro = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="s-w4-ro",
            prompt="READ_DENY_BUFFERED_LEAK",
            on_tool_call=deny_all,
            on_tool_update=rec_ro_update,
            on_thought=rec_thought,
            on_text=rec_text,
            on_usage=rec_usage,
        )
        assert leaked_thoughts == []
        assert leaked_usages == []
        assert "LEAKED_TEXT" not in "".join(leaked_texts)
        assert "LEAKED_TEXT" not in res_ro.response
        assert "not permitted" in res_ro.response
        assert len(ro_updates) == 1
        assert ro_updates[0].status == "failed"
        assert "root:x:0:0" not in ro_updates[0].output

        # 3. [H-2]: Callback exception fails closed (`reject_once`) without killing shared process
        async def exploding_tool_cb(_ev: ToolCallEvent) -> bool:
            raise RuntimeError("boom in on_tool_call")

        async def exploding_text_cb(_txt: str) -> None:
            raise ValueError("boom in on_text")

        await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="s-w4",
            agy_session_id=res_env.conversation_id,
            prompt="RUN_COMMAND_PERM",
            on_tool_call=exploding_tool_cb,
            on_text=exploding_text_cb,
        )
        assert client.worker_pid == initial_pid

        records = _read_mock_records(state_file)
        spawns = [r for r in records if r["type"] == "spawn"]
        assert len(spawns) == 1
        assert spawns[0]["pythoninspect"] is None
        assert spawns[0]["pythonstartup"] is None
        assert spawns[0]["stub_token"] is None

        perm_outcomes = [r for r in records if r["type"] == "perm_outcome"]
        assert perm_outcomes[0]["outcome"] == {"outcome": "selected", "optionId": "deny"}
        assert perm_outcomes[1]["outcome"] == {"outcome": "selected", "optionId": "deny"}
        assert perm_outcomes[2]["outcome"] == {"outcome": "selected", "optionId": "deny"}

    # 4. [H-4]: Shielded close() reaps subprocess even under active outer task cancellation
    cancel_client = AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    )
    await cancel_client.discover_models(cwd=str(tmp_path), refresh=True)
    spawned_pid = cancel_client.worker_pid
    assert spawned_pid is not None
    with anyio.CancelScope() as cancel_scope:
        try:
            cancel_scope.cancel()
            await anyio.sleep(0)
        finally:
            await cancel_client.close()

    assert cancel_client.worker_pid is None
    for _ in range(100):
        try:
            os.kill(spawned_pid, 0)
        except ProcessLookupError:
            break
        await anyio.sleep(0.02)
    with pytest.raises(ProcessLookupError):
        os.kill(spawned_pid, 0)


async def test_discover_tier_and_session_caching_persists_ultra_tier(tmp_path: Path) -> None:
    """Verify `discover_tier`, `discover_models`, and `create_backend_session` persist tier."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "tier-state")

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={
            "MOCK_AGY_STATE_FILE": str(state_file),
            "MOCK_AGY_TIER_ID": "g1-ultra-tier",
            "MOCK_AGY_TIER_NAME": "Google AI Ultra",
        },
    ) as client:
        tier = await client.discover_tier()
        assert tier == {
            "id": "g1-ultra-tier",
            "name": "Google AI Ultra",
            "upgradeUri": "",
        }
        assert store.load_tier() == tier

        recvs_first = [
            r["msg"]["method"]
            for r in _read_mock_records(state_file)
            if r["type"] == "recv" and "method" in r["msg"]
        ]
        assert recvs_first == ["initialize", "authenticate"]

        # Second call serves from SQLite cache without extra authenticate RPC
        assert await client.discover_tier() == tier
        recvs_second = [
            r["msg"]["method"]
            for r in _read_mock_records(state_file)
            if r["type"] == "recv" and "method" in r["msg"]
        ]
        assert recvs_second == ["initialize", "authenticate"]

    # Also verify discover_models populates tier in a fresh store
    models_store = SqliteStateStoreProvider(tmp_path / "models-tier-state")
    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=models_store,
        env={
            "MOCK_AGY_STATE_FILE": str(state_file),
            "MOCK_AGY_TIER_ID": "g1-ultra-tier",
        },
    ) as client2:
        await client2.discover_models(cwd=str(tmp_path))
        assert models_store.load_tier() == {
            "id": "g1-ultra-tier",
            "name": "Google AI Ultra",
            "upgradeUri": "",
        }


async def test_should_keep_context_tokens_separate_from_cumulative_turn_totals_and_ledger(
    tmp_path: Path,
) -> None:
    """Verify multi-step turn totals never inflate context_tokens or duplicate session ledger."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    store.upsert_session({"sessionId": "sess-no-ctx", "cwd": str(tmp_path)})
    store.upsert_session({"sessionId": "sess-snapshots", "cwd": str(tmp_path)})

    observed_usages: list[int] = []

    async def on_usage(tokens: int) -> None:
        observed_usages.append(tokens)

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        no_ctx_outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-no-ctx",
            prompt="MULTI_STEP_NO_USAGE_UPDATE",
        )
        snap_outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-snapshots",
            prompt="MULTI_STEP_USAGE_SNAPSHOTS",
            on_usage=on_usage,
        )

    assert no_ctx_outcome.usage == {
        "input_tokens": 391156,
        "output_tokens": 22121,
        "thinking_tokens": 0,
        "cache_read_tokens": 2086676,
        "total_tokens": 2499953,
        "context_tokens": 0,
    }
    assert store.get_token_ledger(scope="session:sess-no-ctx") == {
        "input_tokens": 391156,
        "output_tokens": 22121,
        "cached_read_tokens": 2086676,
        "thinking_tokens": 0,
        "total_tokens": 2499953,
    }

    assert observed_usages == [50000, 80000]
    assert snap_outcome.usage == {
        "input_tokens": 1200,
        "output_tokens": 300,
        "thinking_tokens": 50,
        "cache_read_tokens": 4500,
        "total_tokens": 6050,
        "context_tokens": 80000,
    }
    persisted_snap = store.get_session("sess-snapshots")
    assert persisted_snap is not None
    assert persisted_snap["contextTokens"] == 80000
    assert store.get_token_ledger(scope="session:sess-snapshots") == {
        "input_tokens": 1200,
        "output_tokens": 300,
        "cached_read_tokens": 4500,
        "thinking_tokens": 50,
        "total_tokens": 6050,
    }


def test_should_format_lazy_mcp_config_merging_base_and_filtering_legacy_shims(
    tmp_path: Path,
) -> None:
    """Verify `format_lazy_mcp_config` keeps tools lazy, merges base config, and drops shims."""
    base_servers: dict[str, Any] = {
        "rea": {"command": "/usr/local/bin/rea-mcp", "args": []},
        "kirocrew-stale": {
            "command": "python3",
            "args": ["-m", "agy_acp.providers.mcp_session_mux", "kirocrew-stale"],
        },
        "kirocrew-core": {
            "command": "python3",
            "args": ["-m", "agy_acp.providers.mcp_session_mux", "kirocrew-core"],
        },
        "user-toggled-off": {"command": "some-mcp", "args": []},
    }
    session_servers: list[dict[str, Any]] = [
        {
            "name": "kirocrew-core",
            "command": "kiro-mcp",
            "args": ["--stdio"],
            "cwd": str(tmp_path),
            "env": [{"name": "BASE_K", "value": "1"}],
            "disabledTools": ["dangerous_tool"],
            "tools": {"eager": True},
        },
        {
            "type": "http",
            "name": "remote-api",
            "url": "https://api.example.com/mcp",
            "headers": [{"name": "Authorization", "value": "Bearer tok"}],
        },
        {"name": "user-toggled-off", "command": "some-mcp", "disabled": True},
    ]
    lazy_cfg = format_lazy_mcp_config(
        session_servers,
        session_env={"KIROCREW_SESSION_KEY": "sess-xyz"},
        base_mcp_servers=base_servers,
    )
    servers = lazy_cfg["mcpServers"]
    assert set(servers) == {"rea", "kirocrew-core", "remote-api"}
    assert "kirocrew-stale" not in servers
    assert "user-toggled-off" not in servers
    assert servers["kirocrew-core"] == {
        "command": "kiro-mcp",
        "args": ["--stdio"],
        "cwd": str(tmp_path.resolve()),
        "env": {"BASE_K": "1", "KIROCREW_SESSION_KEY": "sess-xyz"},
        "disabledTools": ["dangerous_tool"],
    }
    assert "tools" not in servers["kirocrew-core"]
    assert servers["remote-api"] == {
        "url": "https://api.example.com/mcp",
        "headers": {"Authorization": "Bearer tok"},
    }


async def test_should_isolate_concurrent_lazy_mcp_sessions_preserve_symlinks_and_cleanup(
    tmp_path: Path,
) -> None:
    """Verify isolated `--gemini-dir` per MCP profile, symlink preservation, and cleanup."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    fake_gemini = tmp_path / "fake_gemini"
    (fake_gemini / "antigravity-cli").mkdir(parents=True)
    (fake_gemini / "antigravity-cli" / "installation_id").write_text("inst-1", encoding="utf-8")
    (fake_gemini / "GEMINI.md").write_text("# Global rule\n", encoding="utf-8")
    (fake_gemini / "jetski-standalone-oauth-token").write_text("oauth-secret", encoding="utf-8")
    cfg_dir = fake_gemini / "config"
    (cfg_dir / "skills" / "demo-skill").mkdir(parents=True)
    skill_md = cfg_dir / "skills" / "demo-skill" / "SKILL.md"
    skill_md.write_text("# Demo Skill\n", encoding="utf-8")
    (cfg_dir / ".migrated").write_text("", encoding="utf-8")
    (cfg_dir / "config.json").write_text('{"theme":"dark"}', encoding="utf-8")
    # Include JSONC comments and trailing comma + broken legacy shim
    (cfg_dir / "mcp_config.json").write_text(
        """{
          // User custom MCP server
          "mcpServers": {
            "rea": {"command": "rea-bin", "args": ["--serve"],},
            "kirocrew-core": {
              "command": "python3",
              "args": ["-m", "agy_acp.providers.mcp_session_mux", "kirocrew-core"],
            },
          },
        }""",
        encoding="utf-8",
    )

    mcp_list = [{"name": "kirocrew-core", "command": "kiro-mcp", "args": ["serve"]}]

    gemini_dirs_seen: list[Path] = []
    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        base_gemini_dir=fake_gemini,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        s_a1 = await client.create_backend_session(
            cwd=str(tmp_path),
            mcp_servers=mcp_list,
            session_env={"KIROCREW_SESSION_KEY": "sess-A"},
        )
        # Identical MCP + session_env reuses same connection without spawning a duplicate
        s_a2 = await client.create_backend_session(
            cwd=str(tmp_path),
            mcp_servers=mcp_list,
            session_env={"KIROCREW_SESSION_KEY": "sess-A"},
        )
        # Distinct session_env gets its own isolated `--gemini-dir`
        s_b = await client.create_backend_session(
            cwd=str(tmp_path),
            mcp_servers=mcp_list,
            session_env={"KIROCREW_SESSION_KEY": "sess-B"},
        )
        assert s_a1["agySessionId"] != s_a2["agySessionId"] != s_b["agySessionId"]

        spawns = [r for r in _read_mock_records(state_file) if r["type"] == "spawn"]
        assert len(spawns) == 2
        for sp in spawns:
            gdir = Path(sp["gemini_dir"])
            gemini_dirs_seen.append(gdir)
            assert gdir.is_dir()
            assert sp["app_data_dir"] is None
            assert (gdir / "antigravity-cli").is_symlink()
            assert (gdir / "antigravity-cli").resolve() == (
                fake_gemini / "antigravity-cli"
            ).resolve()
            assert sp["mcp_config_mode"] == "0o600"
            assert (gdir / "GEMINI.md").is_symlink()
            assert (gdir / "jetski-standalone-oauth-token").is_symlink()
            assert (gdir / "config" / ".migrated").is_symlink()
            assert (gdir / "config" / "skills" / "demo-skill" / "SKILL.md").is_file()
            assert sp["lazy_mcp_config"]["mcpServers"]["rea"] == {
                "command": "rea-bin",
                "args": ["--serve"],
            }

        tokens = {
            sp["lazy_mcp_config"]["mcpServers"]["kirocrew-core"]["env"]["KIROCREW_SESSION_KEY"]
            for sp in spawns
        }
        assert tokens == {"sess-A", "sess-B"}

    # All temp `--gemini-dir` folders removed on close, user files untouched
    for gdir in gemini_dirs_seen:
        assert not gdir.exists()
    assert skill_md.read_text(encoding="utf-8") == "# Demo Skill\n"
    assert (fake_gemini / "jetski-standalone-oauth-token").read_text(
        encoding="utf-8"
    ) == "oauth-secret"


async def test_should_strip_retry_banners_log_diagnostics_and_preserve_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry banners never leak into streamed text/response, emit diagnostics, and keep liveness."""
    from agy_acp.providers import telemetry as telemetry_mod

    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("AGY_ACP_LOG_FILE", str(log_path))
    monkeypatch.setenv("AGY_ACP_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("AGY_ACP_OTLP_ENDPOINT", "http://127.0.0.1:1")
    telemetry_mod.reset_telemetry_for_tests()
    cfg = telemetry_mod.resolve_telemetry_config(tmp_path / "state")
    mgr = telemetry_mod.TelemetryManager(cfg, use_simple_processors=True)
    monkeypatch.setattr(telemetry_mod, "_GLOBAL_MANAGER", mgr)

    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    events_order: list[str] = []
    streamed_chunks: list[str] = []

    async def on_text(chunk: str) -> None:
        streamed_chunks.append(chunk)
        events_order.append(f"text:{chunk}")

    async def on_tool_call(ev: ToolCallEvent) -> bool:
        events_order.append(f"tool:{ev.tool_call_id}")
        return True

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        with mgr.start_span("test.prompt.turn"):
            outcome = await client.stream_prompt(
                cwd=str(tmp_path),
                session_id="sess-retry-1",
                prompt="RETRY_SPLIT_AND_TOOL",
                on_text=on_text,
                on_tool_call=on_tool_call,
            )

        assert "API error (attempt" not in "".join(streamed_chunks)
        assert "API error (attempt" not in outcome.response
        assert outcome.response == "\n\u26a0 API <!-- nothing_to_do -->"
        assert events_order == [
            "text:\n\u26a0 API ",
            "tool:call_retry_read",
            "text:<!-- nothing_to_do -->",
        ]

        # Liveness check: a turn that emitted a retry banner sets `emitted_any_signal = True`,
        # so an abrupt subprocess exit does NOT trigger silent `turn_recovery` auto-replay.
        crash_outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-retry-crash",
            prompt="RETRY_THEN_CRASH",
        )
        assert crash_outcome.error != ""
        assert crash_outcome.response == ""
        spawns = [r for r in _read_mock_records(state_file) if r["type"] == "spawn"]
        assert len(spawns) == 1

    log_entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retry_logs = [e for e in log_entries if e.get("event") == "agy_acp.backend_api_retry"]
    assert [e["warning"] for e in retry_logs] == [
        '\u26a0 API error (attempt 1): request failed (code 502): HTML response "Error 502"',
        "\u26a0\ufe0f API error (attempt 2): read: operation timed out",
        "\u26a0 API error (attempt 1): connection reset by peer",
    ]
    assert retry_logs[0]["session_id"] == "sess-retry-1"
    assert retry_logs[2]["session_id"] == "sess-retry-crash"

    spans = [s for s in mgr.peek_buffered_spans() if s.name == "test.prompt.turn"]
    assert len(spans) == 1
    span_retry_events = [ev for ev in spans[0].events if ev.name == "agy_acp.backend_api_retry"]
    assert [(ev.attributes or {})["warning"] for ev in span_retry_events] == [
        '\u26a0 API error (attempt 1): request failed (code 502): HTML response "Error 502"',
        "\u26a0\ufe0f API error (attempt 2): read: operation timed out",
    ]


async def test_agy_1_5_10_dynamic_thought_level_and_additional_directories(
    tmp_path: Path,
) -> None:
    """`agy 1.5.10` preserves `xhigh` after `set_model` and forwards `additionalDirectories`."""
    trace_file = tmp_path / "agy_1510_trace.jsonl"
    fake_agy = tmp_path / "fake_agy_1510.py"
    fake_agy.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, sys

trace_path = pathlib.Path(os.environ["AGY_TRACE"])
sessions = {}

MODELS = [
    {"modelId": "gemini-4-carbon-medium", "name": "Gemini 4 Carbon (Medium)"},
    {"modelId": "gemini-4-carbon-high", "name": "Gemini 4 Carbon (High)"},
    {"modelId": "gemini-4-hydrogen-low", "name": "Gemini 4 Hydrogen (Low)"},
    {"modelId": "gemini-4-hydrogen-medium", "name": "Gemini 4 Hydrogen (Medium)"},
    {"modelId": "gemini-4-hydrogen-high", "name": "Gemini 4 Hydrogen (High)"},
    {"modelId": "gemini-4-hydrogen-xhigh", "name": "Gemini 4 Hydrogen (XHigh)"},
]

def config_options_for(s: dict) -> list[dict]:
    model_id = s["model"]
    is_hydrogen = model_id.startswith("gemini-4-hydrogen-")
    family = "gemini-4-hydrogen" if is_hydrogen else "gemini-4-carbon"
    efforts = ["low", "medium", "high", "xhigh"] if is_hydrogen else ["low", "medium", "high"]
    cur_eff = model_id.rsplit("-", 1)[-1]
    return [
        {"id": "model", "category": "model", "currentValue": family, "options": [
            {"value": "gemini-4-carbon"}, {"value": "gemini-4-hydrogen"}
        ]},
        {"id": "thought_level", "category": "thought_level", "currentValue": cur_eff, "options": [
            {"value": v} for v in efforts
        ]},
        {"id": "mode", "category": "mode", "currentValue": s["mode"], "options": [
            {"value": "default"}, {"value": "auto"}, {"value": "plan"}
        ]},
    ]

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"method": method, "params": params}) + "\\n")

    if method == "initialize":
        resp = {
            "protocolVersion": 1,
            "agentCapabilities": {
                "loadSession": True,
                "sessionCapabilities": {
                    "resume": {}, "close": {}, "load": {}, "list": {}, "additionalDirectories": {}
                },
            },
            "agentInfo": {"name": "Antigravity ACP Server", "version": "1.5.10"},
        }
    elif method == "session/new":
        sid = "sid-1510-1"
        sessions[sid] = {
            "model": "gemini-4-carbon-medium",
            "mode": "default",
            "additionalDirectories": params.get("additionalDirectories", []),
        }
        resp = {
            "sessionId": sid,
            "models": {"currentModelId": "gemini-4-carbon-medium", "availableModels": MODELS},
            "configOptions": config_options_for(sessions[sid]),
        }
    elif method in ("session/resume", "session/load"):
        sid = params["sessionId"]
        s = sessions.setdefault(sid, {"model": "gemini-4-carbon-medium", "mode": "default"})
        s["additionalDirectories"] = params.get("additionalDirectories", [])
        resp = {
            "sessionId": sid,
            "models": {"currentModelId": s["model"], "availableModels": MODELS},
            "configOptions": config_options_for(s),
        }
    elif method == "session/set_model":
        s = sessions[params["sessionId"]]
        s["model"] = params["modelId"]
        resp = {}
    elif method == "session/set_config_option":
        s = sessions[params["sessionId"]]
        cid = params["configId"]
        val = params["value"]
        if cid == "mode":
            s["mode"] = val
        elif cid == "thought_level":
            base = s["model"].rsplit("-", 1)[0]
            s["model"] = f"{base}-{val}"
        resp = {"configOptions": config_options_for(s)}
    elif method == "session/prompt":
        sid = params["sessionId"]
        s = sessions[sid]
        txt = f"model={s['model']} dirs={s.get('additionalDirectories')}"
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": txt},
                },
            },
        }) + "\\n")
        resp = {"stopReason": "end_turn", "usage": {"inputTokens": 4, "outputTokens": 4}}
    else:
        resp = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": resp}) + "\\n")
    sys.stdout.flush()
""",
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)

    store = SqliteStateStoreProvider(tmp_path / "state")
    extra_dir = str(tmp_path / "shared-pkg")
    extra_dir_2 = str(tmp_path / "shared-pkg-2")
    async with AgyAcpClientProvider(
        bin_path=fake_agy,
        state_store=store,
        env={"AGY_TRACE": str(trace_file)},
    ) as client:
        outcome = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-1510",
            prompt="check model",
            model="gemini-4-hydrogen-medium",
            effort="xhigh",
            additional_directories=[extra_dir],
        )
        assert outcome.resolved_model == "gemini-4-hydrogen-xhigh"
        assert outcome.response == f"model=gemini-4-hydrogen-xhigh dirs={[extra_dir]!r}"

        resumed = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-1510",
            agy_session_id=outcome.conversation_id,
            prompt="check updated dirs",
            model="gemini-4-hydrogen-low",
            effort="xhigh",
            additional_directories=[extra_dir, extra_dir_2],
        )
        assert resumed.response == (
            f"model=gemini-4-hydrogen-xhigh dirs={[extra_dir, extra_dir_2]!r}"
        )

    traces = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    new_calls = [t for t in traces if t["method"] == "session/new"]
    assert new_calls[0]["params"]["additionalDirectories"] == [extra_dir]
    resume_calls = [t for t in traces if t["method"] == "session/resume"]
    assert resume_calls[0]["params"]["additionalDirectories"] == [extra_dir, extra_dir_2]
    thought_calls = [
        t["params"]["value"]
        for t in traces
        if t["method"] == "session/set_config_option" and t["params"]["configId"] == "thought_level"
    ]
    assert thought_calls == ["xhigh", "xhigh"]


def test_resolve_supported_thought_level_maps_max_to_xhigh_when_supported() -> None:
    """Verify `max` maps to `xhigh` when `agy 1.5.12` advertises `xhigh`, else `high`."""
    from agy_acp.providers.agy_acp_client import _resolve_supported_thought_level

    opts_with_xhigh = [
        {
            "id": "thought_level",
            "options": [
                {"value": "low"},
                {"value": "medium"},
                {"value": "high"},
                {"value": "xhigh"},
            ],
        }
    ]
    opts_without_xhigh = [
        {
            "id": "thought_level",
            "options": [
                {"value": "low"},
                {"value": "medium"},
                {"value": "high"},
            ],
        }
    ]
    opts_with_max = [
        {
            "id": "thought_level",
            "options": [
                {"value": "low"},
                {"value": "medium"},
                {"value": "high"},
                {"value": "xhigh"},
                {"value": "max"},
            ],
        }
    ]

    assert _resolve_supported_thought_level("max", opts_with_xhigh) == "xhigh"
    assert _resolve_supported_thought_level("xhigh", opts_with_xhigh) == "xhigh"
    assert _resolve_supported_thought_level("max", opts_without_xhigh) == "high"
    assert _resolve_supported_thought_level("xhigh", opts_without_xhigh) == "high"
    assert _resolve_supported_thought_level("max", opts_with_max) == "max"
    assert _resolve_supported_thought_level("max", None) == "high"
    assert _resolve_supported_thought_level("xhigh", None) == "high"
    assert _resolve_supported_thought_level("medium", opts_with_xhigh) == "medium"


async def test_should_only_report_policy_denied_tool_when_sibling_tools_are_interrupted(
    tmp_path: Path,
) -> None:
    """Verify interrupted sibling tools are not misreported in `policy_denied_tools`."""
    script, state_file = _make_mock_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    async def on_tool_call(ev: ToolCallEvent) -> bool:
        if ev.kas_builtin_id == "remote_web_search":
            return False
        await anyio.sleep(0.2)
        return True

    async with AgyAcpClientProvider(
        bin_path=script,
        state_store=store,
        env={"MOCK_AGY_STATE_FILE": str(state_file)},
    ) as client:
        res = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-parallel-deny",
            prompt="PARALLEL_DENY_AND_SIBLINGS",
            on_tool_call=on_tool_call,
        )

    assert res.stop_reason == "end_turn"
    assert res.response == "Tool `search_web` is not permitted in this session."
    assert "read_file" not in res.response
    assert "call_mcp_tool" not in res.response


async def test_stream_prompt_auto_and_none_model_selects_neon_medium_on_backend(
    tmp_path: Path,
) -> None:
    """Verify `model=None` and `model='auto'` switch backend `agy acp` to `gemini-4-neon-medium`."""
    import stat

    fake_agy = tmp_path / "fake-agy-neon"
    fake_agy.write_text(
        """#!/usr/bin/env python3
import json
import sys

MODELS = [
    {"modelId": "gemini-4-neon-high", "name": "Gemini 4 Neon (High)"},
    {"modelId": "gemini-4-neon-xhigh", "name": "Gemini 4 Neon (XHigh)"},
    {"modelId": "gemini-4-neon-medium", "name": "Gemini 4 Neon (Medium)"},
    {"modelId": "gemini-4-neon-low", "name": "Gemini 4 Neon (Low)"},
    {"modelId": "claude-sonnet-5-5-medium", "name": "Claude Sonnet 5.5 (Medium)"},
]
sessions = {}
seq = 0

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params") or {}
    if mid is None:
        continue
    if method == "initialize":
        resp = {"protocolVersion": 1, "agentCapabilities": {"loadSession": True}}
    elif method == "session/new":
        seq += 1
        sid = f"agy-neon-{seq}"
        # Upstream agy acp starts on gemini-4-neon-high by default
        sessions[sid] = {"model": "gemini-4-neon-high", "models_set": []}
        resp = {
            "sessionId": sid,
            "models": {"currentModelId": "gemini-4-neon-high", "availableModels": MODELS},
            "configOptions": [],
        }
    elif method in ("session/resume", "session/load"):
        sid = params["sessionId"]
        s = sessions.setdefault(sid, {"model": "gemini-4-neon-high", "models_set": []})
        resp = {
            "sessionId": sid,
            "models": {"currentModelId": s["model"], "availableModels": MODELS},
            "configOptions": [],
        }
    elif method == "session/set_model":
        s = sessions[params["sessionId"]]
        s["model"] = params["modelId"]
        s["models_set"].append(params["modelId"])
        resp = {}
    elif method == "session/set_config_option":
        resp = {"configOptions": []}
    elif method == "session/prompt":
        sid = params["sessionId"]
        s = sessions[sid]
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": json.dumps(s)},
                },
            },
        }) + "\\n")
        resp = {"stopReason": "end_turn", "usage": {"inputTokens": 2, "outputTokens": 2}}
    else:
        resp = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": resp}) + "\\n")
    sys.stdout.flush()
""",
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)

    store = SqliteStateStoreProvider(tmp_path / "state")
    async with AgyAcpClientProvider(bin_path=fake_agy, state_store=store) as client:
        # 1. model=None (default) -> switches backend from neon-high to neon-medium
        first = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-auto-neon",
            prompt="hello",
            model=None,
        )
        assert first.resolved_model == "gemini-4-neon-medium"
        assert json.loads(first.response)["model"] == "gemini-4-neon-medium"

        # 2. Switch to claude on resident backend session, then back to model="auto"
        second = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-auto-neon",
            agy_session_id=first.conversation_id,
            prompt="use claude",
            model="claude-sonnet-5-5-medium",
        )
        assert second.resolved_model == "claude-sonnet-5-5-medium"
        assert json.loads(second.response)["model"] == "claude-sonnet-5-5-medium"

        third = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-auto-neon",
            agy_session_id=first.conversation_id,
            prompt="back to auto",
            model="auto",
        )
        assert third.resolved_model == "gemini-4-neon-medium"
        third_state = json.loads(third.response)
        assert third_state["model"] == "gemini-4-neon-medium"
        assert "auto" not in third_state["models_set"]


def _write_fake_quiet_and_policy_agy(tmp_path: Path) -> tuple[Path, Path]:
    """Create a deterministic mock `agy acp` binary for quiet-turn and policy-recovery tests."""
    script = tmp_path / "fake-agy-quiet-policy"
    log_file = tmp_path / "fake-agy-quiet-policy.jsonl"
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import sys
import threading
import time
from pathlib import Path

LOG_PATH = Path({str(log_file)!r})
lock = threading.Lock()
cancelled = set()
prompt_count = 0


def record(obj):
    with lock:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\\n")


def send(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()


def handle_prompt(mid, params):
    global prompt_count
    sid = params["sessionId"]
    cancelled.discard(sid)
    prompt_count += 1
    blocks = params.get("prompt") or []
    text = " ".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    )
    record({{"type": "prompt", "count": prompt_count, "text": text}})

    if "QUIET_NOTHING_TO_DO" in text:
        send({{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc_ntd_1",
                    "name": "call_mcp_tool",
                    "status": "in_progress",
                    "rawInput": {{
                        "ServerName": "kirocrew-core",
                        "ToolName": "nothing_to_do",
                        "Arguments": {{}},
                    }},
                }},
            }},
        }})
        time.sleep(0.02)
        directive_out = (
            'Turn ended quietly.\\n[[KIROCREW_SESSION_DIRECTIVE]]'
            '{{"kind":"nothing_to_do","reason":"quiet wake"}}'
        )
        send({{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc_ntd_1",
                    "status": "completed",
                    "rawOutput": directive_out,
                }},
            }},
        }})
        # Wait up to 0.3s to see if client cancels before Gemini runs a post-directive step
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline and sid not in cancelled:
            time.sleep(0.01)
        if sid in cancelled:
            # Simulate a racing trailing "." chunk after cancel to ensure suppression
            send({{
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {{
                    "sessionId": sid,
                    "update": {{
                        "sessionUpdate": "agent_message_chunk",
                        "content": {{"type": "text", "text": "."}},
                    }},
                }},
            }})
            send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "cancelled"}}}})
            return
        send({{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "agent_message_chunk",
                    "content": {{"type": "text", "text": "."}},
                }},
            }},
        }})
        send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
        return

    if "REFUSED_NOTHING_TO_DO" in text:
        send({{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc_ntd_refused",
                    "name": "call_mcp_tool",
                    "status": "in_progress",
                    "rawInput": {{
                        "ServerName": "kirocrew-core",
                        "ToolName": "nothing_to_do",
                        "Arguments": {{}},
                    }},
                }},
            }},
        }})
        time.sleep(0.02)
        send({{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc_ntd_refused",
                    "status": "completed",
                    "rawOutput": "Refused: a human is waiting on your reply.",
                }},
            }},
        }})
        time.sleep(0.05)
        send({{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "agent_message_chunk",
                    "content": {{"type": "text", "text": "Still here"}},
                }},
            }},
        }})
        send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
        return

    if (
        "DENY_SCHEDULE_THEN_RECOVER" in text
        or "DENY_SCHEDULE_ALWAYS" in text
        or "[TOOL POLICY]" in text
    ):
        if "[TOOL POLICY]" in text and not Path({str(tmp_path / "always_deny")!r}).exists():
            send({{
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {{
                    "sessionId": sid,
                    "update": {{
                        "sessionUpdate": "agent_message_chunk",
                        "content": {{"type": "text", "text": "Recovered cleanly"}},
                    }},
                }},
            }})
            send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
            return
        send({{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "tool_call",
                    "toolCallId": f"tc_sched_{{prompt_count}}",
                    "name": "schedule",
                    "kind": "other",
                    "status": "in_progress",
                    "rawInput": {{"DurationSeconds": 60, "Prompt": "check"}},
                }},
            }},
        }})
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and sid not in cancelled:
            time.sleep(0.01)
        send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "cancelled"}}}})
        return

    send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})


for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params") or {{}}
    if method == "session/cancel":
        sid = params.get("sessionId", "")
        cancelled.add(sid)
        record({{"type": "cancel", "sessionId": sid}})
        continue
    if mid is None:
        continue
    if method == "initialize":
        init_res = {{"protocolVersion": 1, "agentCapabilities": {{"loadSession": True}}}}
        send({{"jsonrpc": "2.0", "id": mid, "result": init_res}})
    elif method == "session/new":
        new_res = {{
            "sessionId": "agy-qp-1",
            "models": {{"currentModelId": "gemini-4-neon-medium", "availableModels": []}},
            "configOptions": [],
        }}
        send({{"jsonrpc": "2.0", "id": mid, "result": new_res}})
    elif method == "session/prompt":
        threading.Thread(target=handle_prompt, args=(mid, params), daemon=True).start()
    else:
        send({{"jsonrpc": "2.0", "id": mid, "result": {{}}}})
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, log_file


async def test_nothing_to_do_directive_terminates_turn_cleanly_without_trailing_dot(
    tmp_path: Path,
) -> None:
    """Verify `kirocrew-core` `nothing_to_do` with directive cancels cleanly as `end_turn`."""
    script, log_file = _write_fake_quiet_and_policy_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    text_chunks: list[str] = []
    tool_updates: list[ToolCallEvent] = []

    async def on_text(chunk: str) -> None:
        text_chunks.append(chunk)

    async def on_tool_update(ev: ToolCallEvent) -> None:
        tool_updates.append(ev)

    async with AgyAcpClientProvider(bin_path=script, state_store=store) as client:
        res = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-quiet-end",
            prompt="QUIET_NOTHING_TO_DO",
            known_mcp_servers=("kirocrew-core",),
            on_text=on_text,
            on_tool_update=on_tool_update,
        )

    assert res.stop_reason == "end_turn"
    assert res.response == ""
    assert text_chunks == []
    assert len(tool_updates) == 1
    assert tool_updates[0].status == "completed"
    assert "[[KIROCREW_SESSION_DIRECTIVE]]" in tool_updates[0].output
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert any(r["type"] == "cancel" for r in records)


async def test_nothing_to_do_refusal_does_not_terminate_turn_early(
    tmp_path: Path,
) -> None:
    """Verify refused `nothing_to_do` (no directive payload) lets the model finish normally."""
    script, log_file = _write_fake_quiet_and_policy_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    async with AgyAcpClientProvider(bin_path=script, state_store=store) as client:
        res = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-refused-ntd",
            prompt="REFUSED_NOTHING_TO_DO",
            known_mcp_servers=("kirocrew-core",),
        )

    assert res.stop_reason == "end_turn"
    assert res.response == "Still here"
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert not any(r["type"] == "cancel" for r in records)


async def test_policy_recovery_retries_once_and_recovers_cleanly(
    tmp_path: Path,
) -> None:
    """Verify `allow_policy_recovery=True` steers once in-turn instead of emitting a chat notice."""
    script, log_file = _write_fake_quiet_and_policy_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")
    text_chunks: list[str] = []

    async def on_text(chunk: str) -> None:
        text_chunks.append(chunk)

    async def on_tool_call(ev: ToolCallEvent) -> bool:
        return ev.v2_tool_name != "schedule"

    async with AgyAcpClientProvider(bin_path=script, state_store=store) as client:
        res = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-policy-recover",
            prompt="DENY_SCHEDULE_THEN_RECOVER",
            allow_policy_recovery=True,
            on_text=on_text,
            on_tool_call=on_tool_call,
        )

    assert res.stop_reason == "end_turn"
    assert res.response == "Recovered cleanly"
    assert text_chunks == ["Recovered cleanly"]
    prompts = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "prompt"
    ]
    assert len(prompts) == 2
    assert "[TOOL POLICY]" in prompts[1]["text"]
    assert "`schedule`" in prompts[1]["text"]


async def test_policy_recovery_falls_back_to_notice_when_retry_also_denied(
    tmp_path: Path,
) -> None:
    """Verify policy recovery retries at most once and falls back to the policy notice."""
    (tmp_path / "always_deny").write_text("1", encoding="utf-8")
    script, log_file = _write_fake_quiet_and_policy_agy(tmp_path)
    store = SqliteStateStoreProvider(tmp_path / "state")

    async def on_tool_call(_ev: ToolCallEvent) -> bool:
        return False

    async with AgyAcpClientProvider(bin_path=script, state_store=store) as client:
        res = await client.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-policy-fallback",
            prompt="DENY_SCHEDULE_ALWAYS",
            allow_policy_recovery=True,
            on_tool_call=on_tool_call,
        )

    assert res.stop_reason == "end_turn"
    assert res.response == "Tool `schedule` is not permitted in this session."
    prompts = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "prompt"
    ]
    assert len(prompts) == 2
