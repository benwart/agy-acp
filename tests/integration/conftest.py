"""Black-box integration test fixtures for driving the agy-acp CLI and ACP stdio server."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agy_acp.cli import app

FAKE_AGY_SCRIPT = """#!/usr/bin/env python3
import json
import os
import queue
import sys
import threading
import time

args = sys.argv[1:]

prompt = ""
for i, arg in enumerate(args):
    if arg in ("-p", "--print", "--prompt") and i + 1 < len(args):
        prompt = args[i + 1]

if prompt == "/usage":
    print(json.dumps({
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
                    },
                    {
                        "name": "Claude and GPT models",
                        "buckets": [
                            {
                                "id": "claude-gpt-5h",
                                "name": "Five Hour Limit Remaining",
                                "window": "5h",
                                "remaining_fraction": 0.80,
                                "reset_time": "2026-10-24T07:52:04Z",
                            },
                            {
                                "id": "claude-gpt-weekly",
                                "name": "Weekly Limit Remaining",
                                "window": "weekly",
                                "remaining_fraction": 0.72,
                                "reset_time": "2026-10-24T12:34:28Z",
                            },
                        ],
                    },
                ]
            },
        },
    }))
    sys.exit(0)

if prompt == "/credits":
    print(json.dumps({
        "conversation_id": "",
        "status": "SUCCESS",
        "command": {
            "name": "credits",
            "data": {
                "remaining_credits": 1000,
                "upgrade_uri": "https://one.google.com/ai",
            },
        },
    }))
    sys.exit(0)

write_lock = threading.Lock()
req_queue = queue.Queue()
perm_waiters = {}
cancel_events = {}
sessions = {}
seq = 0


def send(obj):
    with write_lock:
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()


def reader():
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "method" not in msg and "id" in msg:
            q = perm_waiters.get(msg["id"])
            if q is not None:
                q.put(msg)
        elif msg.get("method") == "session/cancel":
            sid = (msg.get("params") or {}).get("sessionId", "")
            ev = cancel_events.get(sid)
            if ev is not None:
                ev.set()
        else:
            req_queue.put(msg)
    req_queue.put(None)


threading.Thread(target=reader, daemon=True).start()

MODELS = [
    {"modelId": "gemini-3.7-pro", "name": "Gemini 3.7 Pro"},
    {"modelId": "gemini-3.8-flash", "name": "Gemini 3.8 Flash"},
]
CONFIG_OPTS = [
    {"id": "mode", "options": [{"value": "default"}, {"value": "plan"}]},
    {"id": "thought_level", "options": [
        {"value": "low"}, {"value": "medium"}, {"value": "high"}, {"value": "max"}
    ]},
]

while True:
    msg = req_queue.get()
    if msg is None:
        break
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": 1,
            "agentCapabilities": {"loadSession": True},
        }})
    elif method == "session/new":
        seq += 1
        sid = f"conv-integration-{seq:03d}"
        sessions[sid] = {}
        cancel_events[sid] = threading.Event()
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "sessionId": sid,
            "models": {"currentModelId": "gemini-3.7-pro", "availableModels": MODELS},
            "configOptions": CONFIG_OPTS,
        }})
    elif method == "session/load":
        sid = params.get("sessionId") or "conv-integration-001"
        sessions.setdefault(sid, {})
        cancel_events.setdefault(sid, threading.Event())
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "sessionId": sid,
            "models": {"currentModelId": "gemini-3.7-pro", "availableModels": MODELS},
            "configOptions": CONFIG_OPTS,
        }})
    elif method in ("session/set_model", "session/set_config_option", "session/close"):
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif method == "session/prompt":
        sid = params["sessionId"]
        ev = cancel_events.setdefault(sid, threading.Event())
        ev.clear()
        blocks = params.get("prompt") or []
        text = "\\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if os.environ.get("TEST_AGY_SLOW") == "1":
            if ev.wait(0.2):
                send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "cancelled"}})
                continue
        send({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "Processing prompt via agy subprocess..."},
            },
        }})
        if "tool" in text.lower() or os.environ.get("GOOSE_MODE") == "approve":
            tc_id = f"tc-int-{seq}"
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tc_id,
                    "title": "run_command",
                    "kind": "execute",
                    "status": "pending",
                    "rawInput": {"CommandLine": text},
                    "_meta": {"antigravity": {"toolName": "run_command"}},
                },
            }})
            pq = queue.Queue()
            pid = f"perm-{tc_id}"
            perm_waiters[pid] = pq
            send({"jsonrpc": "2.0", "id": pid, "method": "session/request_permission", "params": {
                "sessionId": sid,
                "toolCall": {
                    "toolCallId": tc_id,
                    "title": "run_command",
                    "kind": "execute",
                    "status": "pending",
                    "rawInput": {"CommandLine": text},
                    "_meta": {"antigravity": {"toolName": "run_command"}},
                },
                "options": [
                    {"optionId": "allow_once", "kind": "allow_once"},
                    {"optionId": "reject_once", "kind": "reject_once"},
                ],
            }})
            presp = pq.get(timeout=5.0)
            perm_waiters.pop(pid, None)
            opt = ((presp.get("result") or {}).get("outcome") or {}).get("optionId")
            allowed = opt == "allow_once"
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tc_id,
                    "status": "completed" if allowed else "failed",
                    "rawOutput": f"Executed: {text}" if allowed else "Rejected",
                },
            }})
        send({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"Processed: {text}"},
            },
        }})
        send({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": sid,
            "update": {"sessionUpdate": "usage_update", "used": 22, "size": 1000000},
        }})
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
"""


@pytest.fixture
def fake_agy_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a test-only fake `agy` executable in `tmp_path` and set `AGY_BIN`."""
    bin_file = tmp_path / "fake-agy"
    bin_file.write_text(FAKE_AGY_SCRIPT, encoding="utf-8")
    bin_file.chmod(bin_file.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("AGY_BIN", str(bin_file))
    return bin_file


@dataclass
class CliResult:
    """Captured output and exit code from a black-box CLI invocation."""

    exit_code: int
    stdout: str
    stderr: str

    def json(self) -> dict[str, Any]:
        """Parse stdout as a JSON object."""
        assert self.stdout.strip(), "Expected JSON on stdout, got empty output"
        loaded = json.loads(self.stdout)
        assert isinstance(loaded, dict)
        return loaded


class AcpBlackBoxHarness:
    """Black-box ACP client simulator matching KiroCrew's AcpClient / AcpRuntime stdio framing."""

    def __init__(self, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        self._args = args or ["acp", "serve"]
        self._env = env or {}
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self.notifications: list[dict[str, Any]] = []
        self.server_requests: list[dict[str, Any]] = []
        self._permission_responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    async def start(self) -> None:
        """Spawn the agy-acp subprocess with stdio pipes."""
        merged_env = {**os.environ, **self._env}
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "agy_acp",
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )

    def on_permission_request(
        self,
        responder: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register a callback to respond to server-initiated session/request_permission RPCs."""
        self._permission_responder = responder

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request over stdin and read until matching response arrives."""
        assert self._proc is not None and self._proc.stdin is not None
        assert self._proc.stdout is not None

        req_id = self._next_id
        self._next_id += 1
        frame = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        self._proc.stdin.write((frame + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

        async def _read_until_response() -> dict[str, Any]:
            assert self._proc is not None and self._proc.stdout is not None
            assert self._proc.stdin is not None
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    stderr_bytes = (
                        await self._proc.stderr.read() if self._proc.stderr is not None else b""
                    )
                    msg = (
                        f"ACP server closed stdout before responding to {method} (id={req_id}). "
                        f"stderr: {stderr_bytes.decode('utf-8', errors='replace')}"
                    )
                    raise RuntimeError(msg)
                assert len(raw_line) <= 16 * 1024 * 1024, "Frame exceeded KiroCrew 16 MiB limit"
                message = json.loads(raw_line.decode("utf-8"))
                if "method" in message and "id" not in message:
                    self.notifications.append(message)
                elif "method" in message and "id" in message:
                    self.server_requests.append(message)
                    if (
                        message["method"] == "session/request_permission"
                        and self._permission_responder is not None
                    ):
                        reply_result = self._permission_responder(message["params"])
                        reply = json.dumps(
                            {"jsonrpc": "2.0", "id": message["id"], "result": reply_result}
                        )
                        self._proc.stdin.write((reply + "\n").encode("utf-8"))
                        await self._proc.stdin.drain()
                elif message.get("id") == req_id:
                    return message

        return await asyncio.wait_for(_read_until_response(), timeout=timeout)

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC 2.0 notification (no id field) over stdin."""
        assert self._proc is not None and self._proc.stdin is not None
        frame = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self._proc.stdin.write((frame + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def close(self) -> None:
        """Terminate subprocess if still running."""
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()


@dataclass
class HarnessFactory:
    """Factory that tracks spawned black-box ACP harnesses and cleans them up after each test."""

    default_env: dict[str, str] = field(default_factory=dict)
    _instances: list[AcpBlackBoxHarness] = field(default_factory=list)

    async def spawn(
        self,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> AcpBlackBoxHarness:
        """Start and return a black-box ACP harness process."""
        merged_env = {**self.default_env, **(env or {})}
        harness = AcpBlackBoxHarness(args=args, env=merged_env)
        await harness.start()
        self._instances.append(harness)
        return harness

    async def cleanup(self) -> None:
        """Close all spawned harness instances."""
        for harness in self._instances:
            await harness.close()


@pytest.fixture
async def acp_harness_factory(fake_agy_bin: Path) -> Any:
    """Provide a black-box ACP harness factory and clean up processes on teardown."""
    factory = HarnessFactory(default_env={"AGY_BIN": str(fake_agy_bin)})
    yield factory
    await factory.cleanup()


@pytest.fixture
def run_cli(
    capsys: pytest.CaptureFixture[str],
    fake_agy_bin: Path,
) -> Callable[[list[str]], CliResult]:
    """Invoke the public agy-acp CLI with arguments and capture exit code + output."""
    _ = fake_agy_bin

    def _invoke(tokens: list[str]) -> CliResult:
        exit_code = int(app(tokens, result_action="return_int_as_exit_code_else_zero"))
        captured = capsys.readouterr()
        return CliResult(exit_code=exit_code, stdout=captured.out, stderr=captured.err)

    return _invoke
