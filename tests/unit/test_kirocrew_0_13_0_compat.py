"""Unit tests for KiroCrew 0.13.0 KAS/ACP protocol compatibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from agy_acp.logic.acp_dispatcher import AcpDispatcher
from agy_acp.logic.session_manager import SessionManager
from agy_acp.logic.tool_classifier import ToolCallEvent, ToolPermissionDecision
from agy_acp.logic.wire import NormalizedToolEvent, V2KiroWireAdapter, V3KasWireAdapter
from agy_acp.providers.agy_acp_client import AgyAcpClientProvider


def _write_fake_shared_agy_script(script_path: Path) -> None:
    """Write a fake `agy acp` script supporting concurrent turns and logging permission replies."""
    script_path.write_text(
        """\
import json
import sys
import threading

write_lock = threading.Lock()
perm_events: dict[str, threading.Event] = {}
perm_replies: dict[str, dict] = {}
next_session = 0
next_perm_id = 0


def send(obj: dict) -> None:
    with write_lock:
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()


def handle_prompt(req_id, sid: str, prompt_blocks: list) -> None:
    global next_perm_id
    text = " ".join(b.get("text", "") for b in prompt_blocks if isinstance(b, dict))
    if "NEED_PERM" in text:
        next_perm_id += 1
        pid = f"perm_{next_perm_id}"
        ev = threading.Event()
        perm_events[pid] = ev
        tc_id = f"tc_{sid}"
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": tc_id,
                        "name": "run_command",
                        "title": "Run tests",
                        "kind": "execute",
                        "status": "in_progress",
                        "rawInput": {"CommandLine": "pytest"},
                    },
                },
            }
        )
        send(
            {
                "jsonrpc": "2.0",
                "id": pid,
                "method": "session/request_permission",
                "params": {
                    "sessionId": sid,
                    "toolCall": {
                        "toolCallId": tc_id,
                        "name": "run_command",
                        "title": "Run tests",
                        "kind": "execute",
                        "status": "pending",
                        "rawInput": {"CommandLine": "pytest"},
                    },
                    "options": [
                        {"optionId": "allow_once", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
                    ],
                },
            }
        )
        ev.wait(timeout=10.0)
        reply = perm_replies.pop(pid, {})
        reason = (
            ((reply.get("_meta") or {}).get("kiro") or {}).get("rejectionReason", "")
            if isinstance(reply, dict)
            else ""
        )
        outcome = (reply.get("outcome") or {}) if isinstance(reply, dict) else {}
        selected = outcome.get("optionId", "")
        if selected == "allow_once":
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": tc_id,
                            "status": "completed",
                            "rawOutput": {"output": "ok"},
                        },
                    },
                }
            )
            msg_text = f"allowed:{sid}"
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": tc_id,
                            "status": "failed",
                            "rawOutput": {},
                        },
                    },
                }
            )
            msg_text = f"rejected:{sid}:reason={reason}"
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": msg_text},
                    },
                },
            }
        )
        send({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})
        return

    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": f"echo:{sid}:{text}"},
                },
            },
        }
    )
    send({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})


for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "method" not in msg and "id" in msg:
        pid = str(msg["id"])
        perm_replies[pid] = msg.get("result") or {}
        ev = perm_events.pop(pid, None)
        if ev is not None:
            ev.set()
        continue

    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": 1,
                    "agentCapabilities": {"loadSession": True},
                    "agentInfo": {"name": "fake-agy", "version": "1.0.0"},
                },
            }
        )
    elif method == "session/new":
        next_session += 1
        sid = f"agy_sess_{next_session}"
        send(
            {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "sessionId": sid,
                    "models": {
                        "currentModelId": "gemini-3.7-pro",
                        "availableModels": [{"modelId": "gemini-3.7-pro", "name": "Gemini"}],
                    },
                },
            }
        )
    elif method in ("session/set_model", "session/set_config_option", "session/close"):
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif method == "session/prompt":
        threading.Thread(
            target=handle_prompt,
            args=(mid, params["sessionId"], params.get("prompt") or []),
            daemon=True,
        ).start()
""",
        encoding="utf-8",
    )


async def test_kas_permission_deny_reason_extracted_and_sanitized(tmp_path: Path) -> None:
    """Verify `_request_client_permission` extracts and sanitizes `_meta.kiro.rejectionReason`."""
    responses = iter(
        [
            {
                "outcome": {"outcome": "selected", "optionId": "reject_once"},
                "_meta": {"kiro": {"rejectionReason": "  Use pytest -k foo\x07\nplease\tnow  "}},
            },
            {
                "outcome": {"outcome": "selected", "optionId": "reject_once"},
                "_meta": {"kiro": {"rejectionReason": "x" * 1001}},
            },
            {
                "outcome": {"outcome": "selected", "optionId": "allow_once"},
                "_meta": {"kiro": {"rejectionReason": "ignored on allow"}},
            },
        ]
    )

    async def _rpc_caller(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "session/request_permission"
        _ = params
        return next(responses)

    mgr = SessionManager(save_dir=tmp_path, wire_adapter=V3KasWireAdapter())
    dispatcher = AcpDispatcher(
        client_rpc_caller=_rpc_caller,
        session_manager=mgr,
        wire_adapter=V3KasWireAdapter(),
    )

    res1 = await dispatcher._request_client_permission("s1", {"sessionId": "s1"})
    assert isinstance(res1, ToolPermissionDecision)
    assert res1.allowed is False
    assert bool(res1) is False
    assert res1.rejection_reason == "Use pytest -k foo\nplease\tnow"

    res2 = await dispatcher._request_client_permission("s1", {"sessionId": "s1"})
    assert res2.allowed is False
    assert res2.rejection_reason == ""

    res3 = await dispatcher._request_client_permission("s1", {"sessionId": "s1"})
    assert res3.allowed is True
    assert bool(res3) is True
    assert res3.rejection_reason == ""
    await mgr.close()


async def test_kas_permission_deny_reason_forwarded_to_agy_acp_and_tool_update(
    tmp_path: Path,
) -> None:
    """Verify `AgyAcpClientProvider` forwards `_meta.kiro.rejectionReason` and surfaces it."""
    fake_script = tmp_path / "fake_agy.py"
    _write_fake_shared_agy_script(fake_script)
    wrapper = tmp_path / "fake_agy_bin"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {fake_script} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)

    tool_updates: list[ToolCallEvent] = []

    async def _on_tool_call(event: ToolCallEvent) -> ToolPermissionDecision:
        _ = event
        return ToolPermissionDecision(allowed=False, rejection_reason="Use pytest -k foo instead")

    async def _on_tool_update(event: ToolCallEvent) -> None:
        tool_updates.append(event)

    async with AgyAcpClientProvider(bin_path=str(wrapper), save_dir=tmp_path) as cli:
        outcome = await cli.stream_prompt(
            cwd=str(tmp_path),
            session_id="sess-reject-1",
            prompt="NEED_PERM",
            skip_permissions=False,
            on_tool_call=_on_tool_call,
            on_tool_update=_on_tool_update,
        )

    assert "reason=Use pytest -k foo instead" in outcome.response
    assert any(
        "The user rejected this tool call: Use pytest -k foo instead" in ev.output
        for ev in tool_updates
    )


async def test_shared_runtime_concurrent_sessions_do_not_block_on_peer_permission(
    tmp_path: Path,
) -> None:
    """Verify Session B completes on a shared connection while Session A waits on permission."""
    fake_script = tmp_path / "fake_agy.py"
    _write_fake_shared_agy_script(fake_script)
    wrapper = tmp_path / "fake_agy_bin"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {fake_script} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)

    perm_entered = anyio.Event()
    release_perm = anyio.Event()
    sess_a_texts: list[str] = []
    sess_b_texts: list[str] = []

    async def _on_tool_call_a(event: ToolCallEvent) -> ToolPermissionDecision:
        _ = event
        perm_entered.set()
        await release_perm.wait()
        return ToolPermissionDecision(allowed=True)

    async def _on_text_a(chunk: str) -> None:
        sess_a_texts.append(chunk)

    async def _on_text_b(chunk: str) -> None:
        sess_b_texts.append(chunk)

    async with AgyAcpClientProvider(bin_path=str(wrapper), save_dir=tmp_path) as cli:
        outcome_a_holder: dict[str, Any] = {}

        async def _run_a() -> None:
            outcome_a_holder["outcome"] = await cli.stream_prompt(
                cwd=str(tmp_path),
                session_id="sess-a",
                prompt="NEED_PERM",
                skip_permissions=False,
                on_tool_call=_on_tool_call_a,
                on_text=_on_text_a,
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_a)
            with anyio.fail_after(5.0):
                await perm_entered.wait()

            # While Session A is blocked inside its permission prompt, Session B must
            # open and complete its full turn on the SAME shared subprocess without stalling!
            with anyio.fail_after(5.0):
                outcome_b = await cli.stream_prompt(
                    cwd=str(tmp_path),
                    session_id="sess-b",
                    prompt="HELLO_B",
                    skip_permissions=True,
                    on_text=_on_text_b,
                )
            assert "HELLO_B" in outcome_b.response
            assert sess_b_texts and all("HELLO_B" in t for t in sess_b_texts)
            assert not sess_a_texts

            # Closing Session B must not disrupt Session A.
            await cli.close_backend_session(outcome_b.conversation_id)
            release_perm.set()

        outcome_a = outcome_a_holder["outcome"]
        assert outcome_a.response.startswith("allowed:")
        assert all("HELLO_B" not in t for t in sess_a_texts)


async def test_v2_commands_execute_effort_refusal_reports_success_false(tmp_path: Path) -> None:
    """Verify `_kiro.dev/commands/execute` `/effort <invalid>` returns `success: False`."""
    mgr = SessionManager(save_dir=tmp_path, wire_adapter=V2KiroWireAdapter())
    dispatcher = AcpDispatcher(session_manager=mgr, wire_adapter=V2KiroWireAdapter())

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

    bad_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": "/effort turbo"},
            }
        )
    )
    assert bad_resp is not None and "result" in bad_resp
    assert bad_resp["result"]["success"] is False
    assert "turbo" in bad_resp["result"]["message"]

    good_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "_kiro.dev/commands/execute",
                "params": {"sessionId": sid, "command": "/effort high"},
            }
        )
    )
    assert good_resp is not None and "result" in good_resp
    assert good_resp["result"]["success"] is True
    assert mgr.get_session(sid)["reasoning_effort"] == "high"
    await mgr.close()


@pytest.mark.parametrize("kas_id", ["control_bash_process", "control_pwsh_process"])
def test_policy_alias_deny_only_process_control_tools(tmp_path: Path, kas_id: str) -> None:
    """Verify `execute_bash` denies process-control tools but never auto-approves them."""
    mgr = SessionManager(save_dir=tmp_path, wire_adapter=V3KasWireAdapter())
    dispatcher = AcpDispatcher(session_manager=mgr, wire_adapter=V3KasWireAdapter())
    norm = NormalizedToolEvent(
        tool_call_id="tc-proc-1",
        title=kas_id,
        kind="execute",
        status="in_progress",
        kas_tool_id=kas_id,
        kiro_v2_tool_name="execute_bash",
    )

    allow_session = {"cwd": str(tmp_path), "agentAllowedTools": ["execute_bash"]}
    assert (
        dispatcher._evaluate_tool_permission(
            session=allow_session,
            spec=None,
            norm=norm,
            session_mode="auto",
        )
        == "ask"
    )

    deny_session = {"cwd": str(tmp_path), "agentExcludedTools": ["execute_bash"]}
    assert (
        dispatcher._evaluate_tool_permission(
            session=deny_session,
            spec=None,
            norm=norm,
            session_mode="auto",
        )
        == "deny"
    )
