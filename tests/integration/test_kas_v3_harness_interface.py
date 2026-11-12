"""Black-box stdio integration tests for `agy-acp acp --agent-engine v3` (KAS wire)."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.integration.conftest import AcpBlackBoxHarness, CliResult, HarnessFactory

SRC_ROOT = str(Path(__file__).resolve().parents[2] / "src")


async def _read_next_notifications(
    harness: AcpBlackBoxHarness,
    *,
    count: int,
    timeout: float = 3.0,
) -> list[dict[str, Any]]:
    """Read `count` post-response JSON-RPC notification frames from `harness` stdout."""
    proc = harness._proc
    assert proc is not None and proc.stdout is not None
    collected: list[dict[str, Any]] = []

    async def _drain() -> None:
        assert proc.stdout is not None
        while len(collected) < count:
            raw_line = await proc.stdout.readline()
            assert raw_line, "ACP stdout closed before expected notifications arrived"
            msg = json.loads(raw_line.decode("utf-8"))
            assert "method" in msg and "id" not in msg, f"Expected notification, got: {msg}"
            harness.notifications.append(msg)
            collected.append(msg)

    await asyncio.wait_for(_drain(), timeout=timeout)
    return collected


def _write_mcp_stdio_stub(script_path: Path) -> Path:
    """Create a real JSON-RPC 2.0 MCP stdio server stub reading dynamic tool list if present."""
    script_path.write_text(
        "import json, os, pathlib, sys\n"
        "tools_file = sys.argv[1] if len(sys.argv) > 1 else ''\n"
        "port = os.environ.get('KIROCREW_PORT', '')\n"
        "tok = os.environ.get('KIROCREW_STUB_SESSION_TOKEN', '')\n"
        "for raw in sys.stdin:\n"
        "    line = raw.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    msg = json.loads(line)\n"
        "    mid = msg.get('id')\n"
        "    method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({\n"
        "            'jsonrpc': '2.0',\n"
        "            'id': mid,\n"
        "            'result': {\n"
        "                'protocolVersion': '2025-03-26',\n"
        "                'capabilities': {'tools': {}},\n"
        "                'serverInfo': {'name': 'kirocrew-core', 'version': '1.0.0'},\n"
        "            },\n"
        "        }), flush=True)\n"
        "    elif method == 'tools/list':\n"
        "        if tools_file and pathlib.Path(tools_file).is_file():\n"
        "            tools = json.loads(pathlib.Path(tools_file).read_text(encoding='utf-8'))\n"
        "        else:\n"
        "            tools = [\n"
        "                {'name': 'spawn_run', 'description': f'port={port} tok={tok}'},\n"
        "                {'name': 'memory_recall', 'description': 'Recall crew memory'},\n"
        "            ]\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': tools}}), "
        "flush=True)\n",
        encoding="utf-8",
    )
    return script_path


def _write_custom_fake_agy(bin_path: Path, trace_file: Path, marker_file: Path) -> Path:
    """Write a scenario-driven fake `agy` binary testing KAS v3 tools, steer, and compaction."""
    bin_path.write_text(
        f"""#!/usr/bin/env python3
import json, os, pathlib, queue, sys, threading, time

args = sys.argv[1:]

prompt = ""
for i, arg in enumerate(args):
    if arg in ("-p", "--print", "--prompt") and i + 1 < len(args):
        prompt = args[i + 1]

if prompt == "/usage":
    print(json.dumps({{
        "conversation_id": "",
        "status": "SUCCESS",
        "command": {{"name": "usage", "data": {{"groups": [{{
            "name": "Gemini Models",
            "buckets": [
                {{"id": "g5", "name": "5h", "window": "5h",
                  "remaining_fraction": 0.9, "reset_time": "2026-12-01T00:00:00Z"}},
                {{"id": "g7", "name": "7d", "window": "weekly",
                  "remaining_fraction": 0.8, "reset_time": "2026-12-05T00:00:00Z"}},
            ],
        }}]}}}},
    }}))
    sys.exit(0)

if prompt == "/credits":
    print(json.dumps({{
        "conversation_id": "",
        "status": "SUCCESS",
        "command": {{"name": "credits", "data": {{
            "remaining_credits": 500, "upgrade_uri": "https://one.google.com/ai"
        }}}},
    }}))
    sys.exit(0)

write_lock = threading.Lock()
req_queue = queue.Queue()
perm_waiters = {{}}
cancel_events = {{}}
sessions = {{}}
seq = 0
tc_seq = 0


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
            sid = (msg.get("params") or {{}}).get("sessionId", "")
            ev = cancel_events.get(sid)
            if ev is not None:
                ev.set()
        else:
            req_queue.put(msg)
    req_queue.put(None)


threading.Thread(target=reader, daemon=True).start()

MODELS = [
    {{"modelId": "gemini-3.7-pro", "name": "Gemini 3.7 Pro"}},
    {{"modelId": "gemini-3.8-flash", "name": "Gemini 3.8 Flash"}},
    {{"modelId": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"}},
]
CONFIG_OPTS = [
    {{"id": "mode", "options": [{{"value": "default"}}, {{"value": "plan"}}]}},
    {{"id": "thought_level", "options": [
        {{"value": "low"}}, {{"value": "medium"}}, {{"value": "high"}},
        {{"value": "xhigh"}}, {{"value": "max"}}
    ]}},
]


def ask_tool(sid, tool_name, kind, raw_input, output_text):
    global tc_seq
    tc_seq += 1
    tc_id = f"tc-kas-{{tc_seq}}"
    tc_payload = {{
        "toolCallId": tc_id,
        "title": tool_name,
        "kind": kind,
        "status": "pending",
        "rawInput": raw_input,
        "_meta": {{"antigravity": {{"toolName": tool_name}}}},
    }}
    send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
        "sessionId": sid,
        "update": {{"sessionUpdate": "tool_call", **tc_payload}},
    }}}})
    pq = queue.Queue()
    pid = f"perm-{{tc_id}}"
    perm_waiters[pid] = pq
    send({{"jsonrpc": "2.0", "id": pid, "method": "session/request_permission", "params": {{
        "sessionId": sid,
        "toolCall": tc_payload,
        "options": [
            {{"optionId": "allow_once", "kind": "allow_once"}},
            {{"optionId": "reject_once", "kind": "reject_once"}},
        ],
    }}}})
    presp = pq.get(timeout=5.0)
    perm_waiters.pop(pid, None)
    opt = ((presp.get("result") or {{}}).get("outcome") or {{}}).get("optionId")
    allowed = opt == "allow_once"
    send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
        "sessionId": sid,
        "update": {{
            "sessionUpdate": "tool_call_update",
            "toolCallId": tc_id,
            "status": "completed" if allowed else "failed",
            "rawOutput": output_text if allowed else "Rejected",
        }},
    }}}})
    return allowed


while True:
    msg = req_queue.get()
    if msg is None:
        break
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {{}}
    if method == "initialize":
        send({{"jsonrpc": "2.0", "id": mid, "result": {{
            "protocolVersion": 1,
            "agentCapabilities": {{"loadSession": True}},
        }}}})
    elif method == "session/new":
        seq += 1
        sid = f"conv-kas-v3-{{seq:03d}}"
        sys_app = (
            ((params.get("_meta") or {{}}).get("antigravity") or {{}})
            .get("systemPrompt", {{}})
            .get("append", "")
        )
        sessions[sid] = {{
            "mcp_servers": params.get("mcpServers", []),
            "system_prompt": sys_app,
            "model": "gemini-3.7-pro",
            "configs": {{}},
        }}
        cancel_events[sid] = threading.Event()
        send({{"jsonrpc": "2.0", "id": mid, "result": {{
            "sessionId": sid,
            "models": {{"currentModelId": "gemini-3.7-pro", "availableModels": MODELS}},
            "configOptions": CONFIG_OPTS,
        }}}})
    elif method == "session/load":
        sid = params.get("sessionId") or "conv-kas-v3-001"
        sys_app = (
            ((params.get("_meta") or {{}}).get("antigravity") or {{}})
            .get("systemPrompt", {{}})
            .get("append", "")
        )
        s = sessions.setdefault(sid, {{
            "mcp_servers": [],
            "system_prompt": "",
            "model": "gemini-3.7-pro",
            "configs": {{}},
        }})
        if "mcpServers" in params:
            s["mcp_servers"] = params["mcpServers"]
        if sys_app:
            s["system_prompt"] = sys_app
        cancel_events.setdefault(sid, threading.Event())
        send({{"jsonrpc": "2.0", "id": mid, "result": {{
            "sessionId": sid,
            "models": {{"currentModelId": s["model"], "availableModels": MODELS}},
            "configOptions": CONFIG_OPTS,
        }}}})
    elif method == "session/set_model":
        s = sessions.get(params.get("sessionId"), {{}})
        if "modelId" in params:
            s["model"] = params["modelId"]
        send({{"jsonrpc": "2.0", "id": mid, "result": {{}}}})
    elif method == "session/set_config_option":
        s = sessions.get(params.get("sessionId"), {{}})
        s.setdefault("configs", {{}})[params.get("configId")] = params.get("value")
        send({{"jsonrpc": "2.0", "id": mid, "result": {{}}}})
    elif method == "session/close":
        send({{"jsonrpc": "2.0", "id": mid, "result": {{}}}})
    elif method == "session/prompt":
        sid = params["sessionId"]
        s = sessions[sid]
        ev = cancel_events.setdefault(sid, threading.Event())
        ev.clear()
        blocks = params.get("prompt") or []
        text = "\\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        trace_entry = {{
            "args": args,
            "prompt": text,
            "conversation_id": sid,
            "model": s.get("model", ""),
            "configs": dict(s.get("configs", {{}})),
            "system_prompt": s.get("system_prompt", ""),
            "mcp_servers": s.get("mcp_servers", []),
        }}
        with open({str(trace_file)!r}, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace_entry) + "\\n")

        if "Summarize our conversation so far" in text:
            summary = "Summary: KAS v3 session compacted cleanly."
            send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "agent_message_chunk",
                    "content": {{"type": "text", "text": summary}},
                }},
            }}}})
            send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "sessionId": sid,
                "update": {{"sessionUpdate": "usage_update", "used": 27, "size": 1000000}},
            }}}})
            send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
            continue

        if "TRIGGER_SLOW_STEER" in text:
            send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "agent_message_chunk",
                    "content": {{"type": "text", "text": "Working before steer... "}},
                }},
            }}}})
            if ev.wait(0.5):
                send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "cancelled"}}}})
                continue
            send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
            continue

        if "TRIGGER_KAS_TOOLS" in text:
            ask_tool(sid, "run_command", "execute", {{"CommandLine": "pytest -q"}}, "1 passed")
            ask_tool(
                sid,
                "edit_file",
                "edit",
                {{"Path": "/repo/app.py", "Target": "old_val = 1", "Replacement": "new_val = 2"}},
                "Edited /repo/app.py",
            )
            ask_tool(sid, "delete_file", "delete", {{"Path": "/repo/obsolete.py"}}, "Deleted")
            ask_tool(
                sid,
                "call_mcp_tool",
                "other",
                {{
                    "ServerName": "kirocrew-core",
                    "ToolName": "spawn_run",
                    "Arguments": {{"tasks": [{{"goal": "verify"}}]}},
                }},
                "spawned",
            )
            allowed_drop = ask_tool(
                sid,
                "call_mcp_tool",
                "other",
                {{
                    "ServerName": "ext-srv",
                    "ToolName": "drop_table",
                    "Arguments": {{"table": "users"}},
                }},
                "dropped",
            )
            if allowed_drop:
                pathlib.Path({str(marker_file)!r}).write_text(
                    "LEAKED_AFTER_REJECT", encoding="utf-8"
                )

        reply_text = f"ACK_V3: {{text.splitlines()[-1] if text else ''}}"
        send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
            "sessionId": sid,
            "update": {{
                "sessionUpdate": "agent_message_chunk",
                "content": {{"type": "text", "text": reply_text}},
            }},
        }}}})
        send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
            "sessionId": sid,
            "update": {{"sessionUpdate": "usage_update", "used": 4200, "size": 1000000}},
        }}}})
        send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
""",
        encoding="utf-8",
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)
    return bin_path


async def test_kas_v3_cli_help_passes_kirocrew_engine_gap_probe(
    run_cli: Callable[[list[str]], CliResult],
) -> None:
    """Scenario 1: `agy-acp acp --help` advertises `--agent-engine` and `v3`."""
    res = run_cli(["acp", "--help"])
    assert res.exit_code == 0
    assert "--agent-engine" in res.stdout
    assert "v3" in res.stdout

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "agy_acp",
        "acp",
        "--agent-engine",
        "v99",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": SRC_ROOT},
    )
    _, stderr = await proc.communicate()
    assert proc.returncode == 2
    assert b"v2" in stderr and b"v3" in stderr


async def test_kas_v3_initialize_mcp_catalog_custom_agent_and_mandate_injection(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Scenarios 2 & 3: initialize, session/new, MCP prober, customAgents, set_mode, and mandate."""
    mcp_stub = _write_mcp_stdio_stub(tmp_path / "mcp_core_stub.py")
    trace_file = tmp_path / "agy_trace.jsonl"
    marker_file = tmp_path / "reject_leak.txt"
    custom_agy = _write_custom_fake_agy(tmp_path / "custom-agy", trace_file, marker_file)
    save_dir = tmp_path / "store"

    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(save_dir)],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )

    init_resp = await harness.send_request(
        "initialize",
        {"protocolVersion": 1, "clientCapabilities": {}},
    )
    assert init_resp["result"]["protocolVersion"] == 1
    kiro_caps = init_resp["result"]["agentCapabilities"]["_meta"]["kiro"]
    assert set(kiro_caps["extensionMethods"]) >= {
        "_kiro/session/compact",
        "_kiro/session/delete",
        "_kiro/mcp/resetServer",
        "_kiro/session/context",
        "_kiro/knowledge",
        "_kiro/codeIntelligence",
        "_kiro/checkpoint/revertMultiple",
    }
    assert kiro_caps["checkpoints"] is True
    assert kiro_caps["mcpServerOAuth"] is True
    assert kiro_caps["mcpRegistryPointers"] is True

    new_resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [
                {
                    "name": "kirocrew-core",
                    "command": sys.executable,
                    "args": [str(mcp_stub)],
                    "env": [
                        {"name": "KIROCREW_PORT", "value": "9444"},
                        {"name": "KIROCREW_STUB_SESSION_TOKEN", "value": "tok-live-v3"},
                    ],
                }
            ],
            "_meta": {
                "kiro": {
                    "customAgents": [
                        {
                            "id": "kirocrew",
                            "name": "KiroCrew",
                            "description": "Primary KiroCrew agent",
                            "prompt": "You are Kiro v3 custom agent.",
                            "tools": ["*"],
                            "permissions": {
                                "rules": [
                                    {
                                        "capability": "mcp",
                                        "match": ["kirocrew-core/*"],
                                        "effect": "allow",
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
        timeout=5.0,
    )
    sid = new_resp["result"]["sessionId"]
    modes_by_id = {m["id"]: m for m in new_resp["result"]["modes"]["availableModes"]}
    assert modes_by_id["kirocrew"]["_meta"]["kiro"]["resource"]["source"]["origin"] == "client"

    post_new = await _read_next_notifications(harness, count=2)
    assert [n["method"] for n in post_new] == ["_kiro/mcp/status", "_kiro/tools/didChange"]

    harness.notifications.clear()
    mode_resp = await harness.send_request(
        "session/set_mode",
        {"sessionId": sid, "modeId": "kirocrew"},
        timeout=5.0,
    )
    assert mode_resp["result"] == {}
    assert any(
        n.get("params", {}).get("update", {}).get("sessionUpdate") == "current_mode_update"
        for n in harness.notifications
    )
    assert all(n.get("method") != "_kiro.dev/agent/switched" for n in harness.notifications)

    post_mode = await _read_next_notifications(harness, count=2)
    assert [n["method"] for n in post_mode] == ["_kiro/mcp/status", "_kiro/tools/didChange"]
    srv_status = post_mode[0]["params"]["servers"][0]
    assert srv_status["name"] == "kirocrew-core"
    assert srv_status["status"] == "connected"
    tool_names = [t["name"] for t in srv_status["tools"]]
    assert "spawn_run" in tool_names and "memory_recall" in tool_names
    assert "port=9444 tok=tok-live-v3" in srv_status["tools"][0]["description"]
    tags = post_mode[1]["params"]["tags"]
    assert {"source": "mcp", "tag": "@kirocrew-core/spawn_run"} in tags

    harness.notifications.clear()
    prompt_resp = await harness.send_request(
        "session/prompt",
        {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "Verify turn 1 mandate"}],
        },
        timeout=5.0,
    )
    assert prompt_resp["result"]["stopReason"] == "end_turn"

    traces = [
        json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(traces) == 1
    sys_prompt = traces[0]["system_prompt"]
    assert "[AGENT MANDATE]" in sys_prompt
    assert "[KIROCREW SUBAGENT ROUTING]" in sys_prompt
    assert "[AGENT SYSTEM PROMPT]" in sys_prompt
    assert "You are Kiro v3 custom agent." in sys_prompt
    assert traces[0]["prompt"] == "Verify turn 1 mandate"
    mcp_env = {item["name"]: item["value"] for item in traces[0]["mcp_servers"][0]["env"]}
    assert mcp_env["KIROCREW_PORT"] == "9444"
    assert mcp_env["KIROCREW_STUB_SESSION_TOKEN"] == "tok-live-v3"
    assert mcp_env["KIROCREW_ROUTE_SPAWN_RUN"] == "1"


async def test_kas_v3_live_mcp_wire_reinjection_close_load_replay_order(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Scenario 4: prompt -> close -> load replays turns BEFORE result, then emits updated MCP."""
    tools_v1 = tmp_path / "tools_v1.json"
    tools_v1.write_text(
        json.dumps([{"name": "tool_v1", "description": "Initial tool"}]),
        encoding="utf-8",
    )
    tools_v2 = tmp_path / "tools_v2.json"
    tools_v2.write_text(
        json.dumps([{"name": "tool_reinjected", "description": "Reinjected tool"}]),
        encoding="utf-8",
    )
    mcp_stub = _write_mcp_stdio_stub(tmp_path / "mcp_reinject_stub.py")
    trace_file = tmp_path / "agy_trace.jsonl"
    custom_agy = _write_custom_fake_agy(
        tmp_path / "custom-agy",
        trace_file,
        tmp_path / "marker.txt",
    )
    save_dir = tmp_path / "store"

    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(save_dir)],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [
                {
                    "name": "kirocrew-core",
                    "command": sys.executable,
                    "args": [str(mcp_stub), str(tools_v1)],
                }
            ],
        },
        timeout=5.0,
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)

    await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Remember alpha-42"}]},
        timeout=5.0,
    )
    close_resp = await harness.send_request("session/close", {"sessionId": sid})
    assert close_resp["result"] == {}

    harness.notifications.clear()
    load_resp = await harness.send_request(
        "session/load",
        {
            "sessionId": sid,
            "cwd": str(tmp_path),
            "mcpServers": [
                {
                    "name": "kirocrew-core",
                    "command": sys.executable,
                    "args": [str(mcp_stub), str(tools_v2)],
                }
            ],
            "_meta": {
                "kiro": {
                    "customAgents": [
                        {
                            "id": "reinjected-agent",
                            "name": "Reinjected Agent",
                            "prompt": "Updated prompt",
                        }
                    ]
                }
            },
        },
        timeout=5.0,
    )
    assert load_resp["result"]["sessionId"] == sid

    # Verify replay notifications arrived BEFORE session/load response!
    pre_load_replays = [
        n
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n.get("params", {}).get("_meta", {}).get("kiro", {}).get("replay") is True
        and n.get("params", {}).get("update", {}).get("_meta", {}).get("kiro", {}).get("replay")
        is True
    ]
    assert [n["params"]["update"]["sessionUpdate"] for n in pre_load_replays] == [
        "user_message_chunk",
        "agent_message_chunk",
    ]
    assert pre_load_replays[0]["params"]["update"]["content"]["text"] == "Remember alpha-42"

    # Verify updated MCP status arrived AFTER session/load response!
    post_load = await _read_next_notifications(harness, count=2)
    assert [n["method"] for n in post_load] == ["_kiro/mcp/status", "_kiro/tools/didChange"]
    assert post_load[0]["params"]["servers"][0]["tools"][0]["name"] == "tool_reinjected"


async def test_kas_v3_set_config_option_model_and_effort_level_xhigh(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Scenario 5: `session/set_config_option` updates `model` and `effortLevel` (`xhigh`)."""
    trace_file = tmp_path / "agy_trace.jsonl"
    custom_agy = _write_custom_fake_agy(
        tmp_path / "custom-agy",
        trace_file,
        tmp_path / "marker.txt",
    )
    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(tmp_path / "store")],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {"cwd": str(tmp_path), "mcpServers": []},
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)

    model_cfg = await harness.send_request(
        "session/set_config_option",
        {"sessionId": sid, "configId": "model", "value": "claude-sonnet-4-6"},
    )
    effort_cfg = await harness.send_request(
        "session/set_config_option",
        {"sessionId": sid, "configId": "effortLevel", "value": "xhigh"},
    )
    opts = {o["id"]: o["currentValue"] for o in effort_cfg["result"]["configOptions"]}
    assert opts["model"] == "claude-sonnet-4-6"
    assert opts["effortLevel"] == "xhigh"
    _ = model_cfg

    await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Check flags"}]},
        timeout=5.0,
    )
    traces = [
        json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line
    ]
    assert traces[-1]["model"] == "claude-sonnet-4-6"
    assert traces[-1]["configs"]["thought_level"] == "xhigh"


async def test_kas_v3_tool_call_envelopes_permissions_and_reject_abort(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Scenario 6: run_command, str_replace diff, delete_file, MCP allow vs reject abort."""
    trace_file = tmp_path / "agy_trace.jsonl"
    marker_file = tmp_path / "reject_leak.txt"
    custom_agy = _write_custom_fake_agy(tmp_path / "custom-agy", trace_file, marker_file)

    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(tmp_path / "store")],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )

    def _perm_responder(params: dict[str, Any]) -> dict[str, Any]:
        tool_id = params.get("_meta", {}).get("kiro", {}).get("toolId", "")
        if tool_id == "mcp_ext-srv_drop_table":
            return {"outcome": {"outcome": "selected", "optionId": "reject_once"}}
        return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}

    harness.on_permission_request(_perm_responder)

    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [],
            "_meta": {
                "kiro": {
                    "customAgents": [
                        {
                            "id": "kirocrew",
                            "name": "KiroCrew",
                            "prompt": "Guard tools",
                            "permissions": {
                                "rules": [
                                    {
                                        "capability": "mcp",
                                        "match": ["kirocrew-core/*"],
                                        "effect": "allow",
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)
    await harness.send_request("session/set_mode", {"sessionId": sid, "modeId": "kirocrew"})
    await _read_next_notifications(harness, count=2)
    harness.notifications.clear()
    harness.server_requests.clear()

    await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "TRIGGER_KAS_TOOLS"}]},
        timeout=5.0,
    )

    # Verify kirocrew-core/spawn_run did NOT trigger session/request_permission!
    req_metas = [r["params"]["_meta"]["kiro"] for r in harness.server_requests]
    req_tool_ids = [m["toolId"] for m in req_metas]
    assert req_tool_ids == [
        "run_command",
        "str_replace",
        "delete_file",
        "mcp_ext-srv_drop_table",
    ]
    assert req_metas[0]["command"] == "pytest -q"
    assert req_metas[3]["mcpTool"] == {
        "identity": {"serverName": "ext-srv", "toolName": "drop_table"}
    }
    for req in harness.server_requests:
        assert "kind" not in req["params"]["toolCall"]

    tool_calls = [
        n["params"]["update"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "tool_call"
    ]
    by_kind = [tc["kind"] for tc in tool_calls]
    assert by_kind[:3] == ["execute", "edit", "delete"]
    assert tool_calls[1]["content"][0]["type"] == "diff"
    assert tool_calls[3]["_meta"]["kiro"] == {
        "serverName": "kirocrew-core",
        "toolName": "spawn_run",
        "toolOrigin": "client",
    }

    # Subprocess was terminated immediately on reject_once before sleeping & writing marker_file!
    await asyncio.sleep(0.45)
    assert not marker_file.exists()


async def test_kas_v3_compact_steer_reset_server_and_delete_extensions(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Scenario 7: `_session/steer`, `_kiro/session/compact`, `_kiro/mcp/resetServer`, `delete`."""
    tools_json = tmp_path / "dynamic_tools.json"
    tools_json.write_text(
        json.dumps([{"name": "alpha_tool", "description": "v1"}]),
        encoding="utf-8",
    )
    mcp_stub = _write_mcp_stdio_stub(tmp_path / "mcp_ext_stub.py")
    trace_file = tmp_path / "agy_trace.jsonl"
    custom_agy = _write_custom_fake_agy(
        tmp_path / "custom-agy",
        trace_file,
        tmp_path / "marker.txt",
    )
    save_dir = tmp_path / "store"

    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(save_dir)],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [
                {
                    "name": "kirocrew-core",
                    "command": sys.executable,
                    "args": [str(mcp_stub), str(tools_json)],
                }
            ],
            "_meta": {
                "kiro": {
                    "customAgents": [
                        {
                            "id": "kirocrew",
                            "name": "KiroCrew",
                            "prompt": "Persistent contract",
                        }
                    ]
                }
            },
        },
        timeout=5.0,
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)
    await harness.send_request("session/set_mode", {"sessionId": sid, "modeId": "kirocrew"})
    await _read_next_notifications(harness, count=2)

    # 1. Active turn + mid-turn _session/steer -> steering_queued -> steering_injected
    harness.notifications.clear()
    proc = harness._proc
    assert proc is not None and proc.stdin is not None and proc.stdout is not None

    prompt_req = {
        "jsonrpc": "2.0",
        "id": 900,
        "method": "session/prompt",
        "params": {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "TRIGGER_SLOW_STEER"}],
        },
    }
    proc.stdin.write((json.dumps(prompt_req) + "\n").encode("utf-8"))
    await proc.stdin.drain()

    # Wait until first chunk of slow turn arrives so turn is active
    while True:
        raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        msg = json.loads(raw_line.decode("utf-8"))
        if "method" in msg and "id" not in msg:
            harness.notifications.append(msg)
            if msg.get("params", {}).get("update", {}).get("sessionUpdate") == (
                "agent_message_chunk"
            ):
                break

    steer_req = {
        "jsonrpc": "2.0",
        "id": 901,
        "method": "_session/steer",
        "params": {"sessionId": sid, "message": "Switch to concise mode"},
    }
    proc.stdin.write((json.dumps(steer_req) + "\n").encode("utf-8"))
    await proc.stdin.drain()

    got_900 = False
    got_901 = False
    while not (got_900 and got_901):
        raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        msg = json.loads(raw_line.decode("utf-8"))
        if "method" in msg and "id" not in msg:
            harness.notifications.append(msg)
        elif msg.get("id") == 901:
            assert msg["result"] == {"queued": True}
            got_901 = True
        elif msg.get("id") == 900:
            assert msg["result"]["stopReason"] == "end_turn"
            got_900 = True

    steer_kinds = [
        n["params"]["update"]["_meta"]["kiro"]["kind"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "session_info_update"
    ]
    assert "steering_queued" in steer_kinds
    assert "steering_injected" in steer_kinds

    # 2. _kiro/session/compact -> summarization_started -> summarization_completed -> success: True
    harness.notifications.clear()
    compact_resp = await harness.send_request(
        "_kiro/session/compact",
        {"sessionId": sid},
        timeout=5.0,
    )
    assert compact_resp["result"] == {"success": True}
    compact_kinds = [
        n["params"]["update"]["_meta"]["kiro"]["kind"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "session_info_update"
    ]
    assert "summarization_started" in compact_kinds
    assert "summarization_completed" in compact_kinds
    assert "context_usage" in compact_kinds

    # Next prompt after compaction re-injects summary AND customAgent mandate!
    await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Post-compact turn"}]},
        timeout=5.0,
    )
    traces = [
        json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line
    ]
    assert "[Compacted Session Summary]" in traces[-1]["prompt"]
    assert "Persistent contract" in traces[-1]["system_prompt"]

    # 3. _kiro/mcp/resetServer -> re-probes updated tools_json
    tools_json.write_text(
        json.dumps([{"name": "beta_tool_after_reset", "description": "v2"}]),
        encoding="utf-8",
    )
    harness.notifications.clear()
    reset_resp = await harness.send_request(
        "_kiro/mcp/resetServer",
        {"sessionId": sid, "serverName": "kirocrew-core"},
        timeout=5.0,
    )
    assert reset_resp["result"] == {}
    mcp_status_frames = [n for n in harness.notifications if n.get("method") == "_kiro/mcp/status"]
    assert mcp_status_frames[-1]["params"]["servers"][0]["tools"][0]["name"] == (
        "beta_tool_after_reset"
    )

    # 4. _kiro/session/delete -> wipes SQLite session & turns
    del_resp = await harness.send_request("_kiro/session/delete", {"sessionId": sid})
    assert del_resp["result"] == {}
    missing_load = await harness.send_request(
        "session/load",
        {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
    )
    assert missing_load["error"]["code"] == -32002


def _write_named_mcp_stdio_stub(
    script_path: Path,
    server_name: str,
    default_tools: list[dict[str, str]],
) -> Path:
    """Create a JSON-RPC 2.0 MCP stdio server stub for `server_name` with optional file override."""
    default_tools_json = json.dumps(default_tools)
    script_path.write_text(
        "import json, pathlib, sys\n"
        f"DEFAULT_TOOLS = json.loads({default_tools_json!r})\n"
        "tools_file = sys.argv[1] if len(sys.argv) > 1 else ''\n"
        "for raw in sys.stdin:\n"
        "    line = raw.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    msg = json.loads(line)\n"
        "    mid = msg.get('id')\n"
        "    method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({\n"
        "            'jsonrpc': '2.0',\n"
        "            'id': mid,\n"
        "            'result': {\n"
        "                'protocolVersion': '2025-03-26',\n"
        "                'capabilities': {'tools': {}},\n"
        f"                'serverInfo': {{'name': {server_name!r}, 'version': '1.0.0'}},\n"
        "            },\n"
        "        }), flush=True)\n"
        "    elif method == 'tools/list':\n"
        "        if tools_file and pathlib.Path(tools_file).is_file():\n"
        "            tools = json.loads(pathlib.Path(tools_file).read_text(encoding='utf-8'))\n"
        "        else:\n"
        "            tools = DEFAULT_TOOLS\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': tools}}), "
        "flush=True)\n",
        encoding="utf-8",
    )
    return script_path


def _write_wave6_fake_agy(
    bin_path: Path,
    *,
    turn_counter_file: Path,
    leak_dir: Path,
) -> Path:
    """Write fake `agy` binary supporting multi-turn `/usage` deltas and tool mount probes."""
    bin_path.write_text(
        f"""#!/usr/bin/env python3
import json, pathlib, queue, sys, threading, time

TURN_COUNTER = pathlib.Path({str(turn_counter_file)!r})
LEAK_DIR = pathlib.Path({str(leak_dir)!r})

args = sys.argv[1:]

prompt = ""
for i, arg in enumerate(args):
    if arg in ("-p", "--print", "--prompt") and i + 1 < len(args):
        prompt = args[i + 1]

if prompt == "/usage":
    turns = int(TURN_COUNTER.read_text(encoding="utf-8")) if TURN_COUNTER.is_file() else 0
    rem = max(0.10, round(0.96 - 0.01 * turns, 4))
    print(json.dumps({{
        "conversation_id": "",
        "status": "SUCCESS",
        "command": {{"name": "usage", "data": {{"groups": [{{
            "name": "Gemini Models",
            "buckets": [
                {{"id": "g5", "name": "5h", "window": "5h",
                  "remaining_fraction": rem, "reset_time": "2026-12-01T00:00:00Z"}},
                {{"id": "g7", "name": "7d", "window": "weekly",
                  "remaining_fraction": rem, "reset_time": "2026-12-05T00:00:00Z"}},
            ],
        }}]}}}},
    }}))
    sys.exit(0)

if prompt == "/credits":
    print(json.dumps({{
        "conversation_id": "",
        "status": "SUCCESS",
        "command": {{"name": "credits", "data": {{
            "remaining_credits": 0, "upgrade_uri": "https://one.google.com/ai"
        }}}},
    }}))
    sys.exit(0)

write_lock = threading.Lock()
req_queue = queue.Queue()
perm_waiters = {{}}
sessions = {{}}
seq = 0
tc_seq = 0


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
            pass
        else:
            req_queue.put(msg)
    req_queue.put(None)


threading.Thread(target=reader, daemon=True).start()

MODELS = [{{"modelId": "gemini-3.7-pro", "name": "Gemini 3.7 Pro"}}]
CONFIG_OPTS = [
    {{"id": "mode", "options": [{{"value": "default"}}, {{"value": "plan"}}]}},
    {{"id": "thought_level", "options": [
        {{"value": "low"}}, {{"value": "medium"}}, {{"value": "high"}}
    ]}},
]


def ask_mutating_tool(sid, tool_name, kind, raw_input, output_text):
    global tc_seq
    tc_seq += 1
    tc_id = f"tc-w6-{{tc_seq}}"
    tc_payload = {{
        "toolCallId": tc_id,
        "title": tool_name,
        "kind": kind,
        "status": "pending",
        "rawInput": raw_input,
        "_meta": {{"antigravity": {{"toolName": tool_name}}}},
    }}
    send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
        "sessionId": sid,
        "update": {{"sessionUpdate": "tool_call", **tc_payload}},
    }}}})
    pq = queue.Queue()
    pid = f"perm-{{tc_id}}"
    perm_waiters[pid] = pq
    send({{"jsonrpc": "2.0", "id": pid, "method": "session/request_permission", "params": {{
        "sessionId": sid,
        "toolCall": tc_payload,
        "options": [
            {{"optionId": "allow_once", "kind": "allow_once"}},
            {{"optionId": "reject_once", "kind": "reject_once"}},
        ],
    }}}})
    presp = pq.get(timeout=5.0)
    perm_waiters.pop(pid, None)
    opt = ((presp.get("result") or {{}}).get("outcome") or {{}}).get("optionId")
    allowed = opt == "allow_once"
    send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
        "sessionId": sid,
        "update": {{
            "sessionUpdate": "tool_call_update",
            "toolCallId": tc_id,
            "status": "completed" if allowed else "failed",
            "rawOutput": output_text if allowed else "Rejected",
        }},
    }}}})
    return allowed


while True:
    msg = req_queue.get()
    if msg is None:
        break
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {{}}
    if method == "initialize":
        send({{"jsonrpc": "2.0", "id": mid, "result": {{
            "protocolVersion": 1,
            "agentCapabilities": {{"loadSession": True}},
        }}}})
    elif method == "session/new":
        seq += 1
        sid = f"conv-kas-wave6-{{seq:03d}}"
        sessions[sid] = {{}}
        send({{"jsonrpc": "2.0", "id": mid, "result": {{
            "sessionId": sid,
            "models": {{"currentModelId": "gemini-3.7-pro", "availableModels": MODELS}},
            "configOptions": CONFIG_OPTS,
        }}}})
    elif method == "session/load":
        sid = params.get("sessionId") or "conv-kas-wave6-001"
        sessions.setdefault(sid, {{}})
        send({{"jsonrpc": "2.0", "id": mid, "result": {{
            "sessionId": sid,
            "models": {{"currentModelId": "gemini-3.7-pro", "availableModels": MODELS}},
            "configOptions": CONFIG_OPTS,
        }}}})
    elif method in ("session/set_model", "session/set_config_option", "session/close"):
        send({{"jsonrpc": "2.0", "id": mid, "result": {{}}}})
    elif method == "session/prompt":
        sid = params["sessionId"]
        prev_turns = int(TURN_COUNTER.read_text(encoding="utf-8")) if TURN_COUNTER.is_file() else 0
        TURN_COUNTER.write_text(str(prev_turns + 1), encoding="utf-8")
        blocks = params.get("prompt") or []
        text = "\\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")

        last_line = text.splitlines()[-1] if text else ""

        if "SCENARIO_MOUNT_ZERO_TOOLS" in last_line:
            allowed = ask_mutating_tool(
                sid, "run_command", "execute", {{"CommandLine": "rm -rf /tmp/forbidden"}}, "ran"
            )
            if allowed:
                (LEAK_DIR / "zero_tools_leaked.txt").write_text("LEAKED", encoding="utf-8")
            send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
            continue

        if "SCENARIO_MOUNT_READ_ONLY" in last_line:
            tc_seq += 1
            ro_id = f"tc-w6-{{tc_seq}}"
            send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "tool_call",
                    "toolCallId": ro_id,
                    "title": "read_file",
                    "kind": "read",
                    "status": "in_progress",
                    "rawInput": {{"Path": "/repo/allowed.txt"}},
                    "_meta": {{"antigravity": {{"toolName": "read_file"}}}},
                }},
            }}}})
            send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
                "sessionId": sid,
                "update": {{
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": ro_id,
                    "status": "completed",
                    "rawOutput": "allowed contents",
                }},
            }}}})
            allowed = ask_mutating_tool(
                sid, "delete_file", "delete", {{"Path": "/repo/secret.txt"}}, "deleted"
            )
            if allowed:
                (LEAK_DIR / "read_only_leaked.txt").write_text("LEAKED", encoding="utf-8")
            send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
            continue

        if "SCENARIO_ALLOWED_FS_WRITE_ALIAS" in last_line:
            ask_mutating_tool(
                sid,
                "edit_file",
                "edit",
                {{"Path": "/repo/module.py", "Target": "x = 1", "Replacement": "x = 2"}},
                "Edited /repo/module.py",
            )
            allowed = ask_mutating_tool(
                sid, "delete_file", "delete", {{"Path": "/repo/critical.py"}}, "deleted"
            )
            if allowed:
                (LEAK_DIR / "fs_write_delete_leaked.txt").write_text("LEAKED", encoding="utf-8")
            send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
            continue

        reply_text = f"WAVE6_ACK: {{last_line}}"
        send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
            "sessionId": sid,
            "update": {{
                "sessionUpdate": "agent_thought_chunk",
                "content": {{"type": "text", "text": "Reasoning about turn..."}},
            }},
        }}}})
        send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
            "sessionId": sid,
            "update": {{
                "sessionUpdate": "agent_message_chunk",
                "content": {{"type": "text", "text": reply_text}},
            }},
        }}}})
        send({{"jsonrpc": "2.0", "method": "session/update", "params": {{
            "sessionId": sid,
            "update": {{"sessionUpdate": "usage_update", "used": 1600, "size": 1000000}},
        }}}})
        send({{"jsonrpc": "2.0", "id": mid, "result": {{"stopReason": "end_turn"}}}})
""",
        encoding="utf-8",
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)
    return bin_path


def _assert_kas_v3_turn_telemetry_frames(
    notifications: list[dict[str, Any]],
    *,
    expected_credits: float,
    expect_breakdown: bool,
    expect_ttft: bool,
) -> None:
    """Assert KAS v3 turn telemetry ordering, credit delta, breakdown, and TTFT gating."""
    methods = [n.get("method") for n in notifications]
    assert "_kiro.dev/metadata" not in methods
    assert "_kiro.dev/error/rate_limit" not in methods

    updates = [
        n["params"]["update"]
        for n in notifications
        if n.get("method") == "session/update" and isinstance(n.get("params"), dict)
    ]

    usage_indices = [
        idx for idx, upd in enumerate(updates) if upd.get("sessionUpdate") == "usage_update"
    ]
    context_usage_indices = [
        idx
        for idx, upd in enumerate(updates)
        if upd.get("sessionUpdate") == "session_info_update"
        and upd.get("_meta", {}).get("kiro", {}).get("kind") == "context_usage"
    ]
    completion_indices = [
        idx
        for idx, upd in enumerate(updates)
        if upd.get("sessionUpdate") == "session_info_update"
        and upd.get("_meta", {}).get("kiro", {}).get("kind") == "turn_completion"
    ]

    assert len(context_usage_indices) == 1
    assert len(usage_indices) == 1
    assert len(completion_indices) == 1
    assert context_usage_indices[0] < usage_indices[0] < completion_indices[0]

    usage_upd = updates[usage_indices[0]]
    assert "cost" not in usage_upd
    assert usage_upd["used"] == 1600

    completion_kiro = updates[completion_indices[0]]["_meta"]["kiro"]
    assert completion_kiro["promptTurnSummaries"] == [
        {"unit": "credit", "usage": expected_credits},
    ]
    if expect_breakdown:
        breakdown = completion_kiro["contextBreakdown"]
        assert breakdown["schema"] == 1
        assert breakdown["modelCalls"] == 1
        assert breakdown["totalChars"] > 0
        assert breakdown["categories"]["system"]["chars"] > 0
        assert breakdown["categories"]["history"]["chars"] > 0
    else:
        assert "contextBreakdown" not in completion_kiro

    displayed_chunks = [
        upd
        for upd in updates
        if upd.get("sessionUpdate") in {"agent_thought_chunk", "agent_message_chunk"}
    ]
    assert len(displayed_chunks) >= 2
    for idx, chunk in enumerate(displayed_chunks):
        chunk_kiro = chunk.get("_meta", {}).get("kiro", {})
        assert "ttftMs" not in chunk_kiro
        if expect_ttft and idx == 0:
            ttft = chunk_kiro["ttft"]
            assert ttft["promptReceivedAt"] > 1_700_000_000_000
            assert ttft["firstChunkEmittedAt"] >= ttft["promptReceivedAt"]
        else:
            assert "ttft" not in chunk_kiro


async def test_kas_v3_multi_turn_credit_deltas_telemetry_order_breakdown_and_ttft_gating(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Wave 6A Gap 1: Multi-turn credit deltas, v3 telemetry order, breakdown, and ttft gating."""
    leak_dir = tmp_path / "leaks"
    leak_dir.mkdir()
    custom_agy = _write_wave6_fake_agy(
        tmp_path / "wave6-agy",
        turn_counter_file=tmp_path / "turn_count.txt",
        leak_dir=leak_dir,
    )
    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(tmp_path / "store")],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [],
            "_meta": {
                "kiro": {
                    "customAgents": [
                        {
                            "id": "telemetry-agent",
                            "name": "Telemetry Agent",
                            "prompt": "Telemetry system prompt.",
                            "tools": ["*"],
                        }
                    ]
                }
            },
        },
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)
    await harness.send_request("session/set_mode", {"sessionId": sid, "modeId": "telemetry-agent"})
    await _read_next_notifications(harness, count=2)

    # Turn 1: omit _meta.kiro.ttft and _meta.kiro.contextBreakdown -> delta_credits == 2.0
    harness.notifications.clear()
    turn1_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Turn one telemetry"}]},
        timeout=5.0,
    )
    assert turn1_resp["result"]["stopReason"] == "end_turn"
    _assert_kas_v3_turn_telemetry_frames(
        harness.notifications,
        expected_credits=2.0,
        expect_breakdown=False,
        expect_ttft=False,
    )

    # Turn 2: request _meta.kiro.ttft=True and contextBreakdown="summary" -> delta_credits == 2.0
    harness.notifications.clear()
    turn2_resp = await harness.send_request(
        "session/prompt",
        {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "Turn two telemetry"}],
            "_meta": {"kiro": {"ttft": True, "contextBreakdown": "summary"}},
        },
        timeout=5.0,
    )
    assert turn2_resp["result"]["stopReason"] == "end_turn"
    _assert_kas_v3_turn_telemetry_frames(
        harness.notifications,
        expected_credits=2.0,
        expect_breakdown=True,
        expect_ttft=True,
    )


async def test_kas_v3_multi_server_reset_preserves_peer_mcp_snapshot_and_catalog(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Wave 6A Gap 2: `_kiro/mcp/resetServer` on `echo-mcp` keeps `peer-mcp` in full snapshot."""
    echo_tools_json = tmp_path / "echo_tools.json"
    echo_tools_json.write_text(
        json.dumps([{"name": "echo_v1", "description": "Echo tool v1"}]),
        encoding="utf-8",
    )
    echo_stub = _write_named_mcp_stdio_stub(
        tmp_path / "echo_mcp_stub.py",
        "echo-mcp",
        [{"name": "echo_fallback", "description": "Fallback"}],
    )
    peer_stub = _write_named_mcp_stdio_stub(
        tmp_path / "peer_mcp_stub.py",
        "peer-mcp",
        [
            {"name": "peer_ping", "description": "Ping peer"},
            {"name": "peer_status", "description": "Status peer"},
        ],
    )

    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(tmp_path / "store")],
        env={"PYTHONPATH": SRC_ROOT},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [
                {
                    "name": "echo-mcp",
                    "command": sys.executable,
                    "args": [str(echo_stub), str(echo_tools_json)],
                },
                {
                    "name": "peer-mcp",
                    "command": sys.executable,
                    "args": [str(peer_stub)],
                },
            ],
        },
        timeout=5.0,
    )
    sid = new_resp["result"]["sessionId"]
    initial_mcp = await _read_next_notifications(harness, count=2)
    initial_servers = {s["name"]: s for s in initial_mcp[0]["params"]["servers"]}
    assert set(initial_servers) == {"echo-mcp", "peer-mcp"}
    assert initial_servers["echo-mcp"]["tools"][0]["name"] == "echo_v1"

    # Update echo-mcp's tools and reset ONLY echo-mcp
    echo_tools_json.write_text(
        json.dumps([{"name": "echo_v2_refreshed", "description": "Echo tool v2"}]),
        encoding="utf-8",
    )
    harness.notifications.clear()
    reset_resp = await harness.send_request(
        "_kiro/mcp/resetServer",
        {"sessionId": sid, "serverName": "echo-mcp"},
        timeout=5.0,
    )
    assert reset_resp["result"] == {}

    status_frame = next(n for n in harness.notifications if n.get("method") == "_kiro/mcp/status")
    did_change_frame = next(
        n for n in harness.notifications if n.get("method") == "_kiro/tools/didChange"
    )

    servers_by_name = {s["name"]: s for s in status_frame["params"]["servers"]}
    assert set(servers_by_name) == {"echo-mcp", "peer-mcp"}
    assert servers_by_name["echo-mcp"]["status"] == "connected"
    assert [t["name"] for t in servers_by_name["echo-mcp"]["tools"]] == ["echo_v2_refreshed"]
    assert servers_by_name["peer-mcp"]["status"] == "connected"
    assert [t["name"] for t in servers_by_name["peer-mcp"]["tools"]] == ["peer_ping", "peer_status"]

    expected_tool_ids = [
        "mcp_echo-mcp_echo_v2_refreshed",
        "mcp_peer-mcp_peer_ping",
        "mcp_peer-mcp_peer_status",
    ]
    assert did_change_frame["params"]["initial"]["tools"] == expected_tool_ids
    assert did_change_frame["params"]["added"]["tools"] == expected_tool_ids
    tags = did_change_frame["params"]["tags"]
    assert {"source": "mcp", "tag": "@echo-mcp/echo_v2_refreshed"} in tags
    assert {"source": "mcp", "tag": "@peer-mcp/peer_ping"} in tags
    assert {"source": "mcp", "tag": "@peer-mcp/peer_status"} in tags


async def test_kas_v3_custom_agent_tools_mount_allowlist_and_delete_file_alias_guard(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Wave 6A Gap 3: `tools: []`, `tools: ['read_file']`, and `allowedTools: ['fs_write']`."""
    leak_dir = tmp_path / "leaks"
    leak_dir.mkdir()
    custom_agy = _write_wave6_fake_agy(
        tmp_path / "wave6-agy",
        turn_counter_file=tmp_path / "turn_count.txt",
        leak_dir=leak_dir,
    )
    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(tmp_path / "store")],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )
    harness.on_permission_request(
        lambda _params: {"outcome": {"outcome": "selected", "optionId": "reject_once"}}
    )

    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {
            "cwd": str(tmp_path),
            "mcpServers": [],
            "_meta": {
                "kiro": {
                    "customAgents": [
                        {
                            "id": "zero-tools-agent",
                            "name": "Zero Tools Agent",
                            "prompt": "No tools mounted.",
                            "tools": [],
                        },
                        {
                            "id": "read-only-mount-agent",
                            "name": "Read Only Mount Agent",
                            "prompt": "Only read_file mounted.",
                            "tools": ["read_file"],
                            "allowedTools": ["read_file"],
                        },
                        {
                            "id": "fs-write-alias-agent",
                            "name": "FS Write Alias Agent",
                            "prompt": "fs_write allowed, delete_file must still prompt.",
                            "tools": ["*"],
                            "allowedTools": ["fs_write"],
                        },
                    ]
                }
            },
        },
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)

    # Case A: tools: [] denies run_command without prompting client or leaking execution
    await harness.send_request("session/set_mode", {"sessionId": sid, "modeId": "zero-tools-agent"})
    await _read_next_notifications(harness, count=2)
    harness.notifications.clear()
    harness.server_requests.clear()

    await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "SCENARIO_MOUNT_ZERO_TOOLS"}]},
        timeout=5.0,
    )
    assert harness.server_requests == []
    zero_tool_updates = [
        n["params"]["update"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "tool_call_update"
    ]
    assert len(zero_tool_updates) == 1
    assert zero_tool_updates[0]["status"] == "failed"
    assert zero_tool_updates[0]["rawOutput"]["allowed"] is False

    # Case B: tools: ["read_file"] allows read_file, denies delete_file without prompting client
    await harness.send_request(
        "session/set_mode",
        {"sessionId": sid, "modeId": "read-only-mount-agent"},
    )
    await _read_next_notifications(harness, count=2)
    harness.notifications.clear()
    harness.server_requests.clear()

    await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "SCENARIO_MOUNT_READ_ONLY"}]},
        timeout=5.0,
    )
    assert harness.server_requests == []
    ro_tool_updates = [
        n["params"]["update"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "tool_call_update"
    ]
    assert [u["status"] for u in ro_tool_updates] == ["completed", "failed"]
    assert ro_tool_updates[0]["rawOutput"]["allowed"] is True
    assert ro_tool_updates[1]["rawOutput"]["allowed"] is False

    # Case C: allowedTools: ["fs_write"] auto-approves str_replace, NOT delete_file
    await harness.send_request(
        "session/set_mode",
        {"sessionId": sid, "modeId": "fs-write-alias-agent"},
    )
    await _read_next_notifications(harness, count=2)
    harness.notifications.clear()
    harness.server_requests.clear()

    await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "SCENARIO_ALLOWED_FS_WRITE_ALIAS"}]},
        timeout=5.0,
    )
    assert [r["params"]["_meta"]["kiro"]["toolId"] for r in harness.server_requests] == [
        "delete_file"
    ]
    alias_updates = [
        n["params"]["update"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "tool_call_update"
    ]
    assert [u["status"] for u in alias_updates] == ["completed", "failed"]

    await asyncio.sleep(0.4)
    assert list(leak_dir.iterdir()) == []


async def test_kas_v3_close_load_prompt_resumes_cleanly_with_end_turn(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Wave 6A Gap 4: `session/close` -> `session/load` -> `session/prompt` yields `end_turn`."""
    leak_dir = tmp_path / "leaks"
    leak_dir.mkdir()
    custom_agy = _write_wave6_fake_agy(
        tmp_path / "wave6-agy",
        turn_counter_file=tmp_path / "turn_count.txt",
        leak_dir=leak_dir,
    )
    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(tmp_path / "store")],
        env={"AGY_BIN": str(custom_agy), "PYTHONPATH": SRC_ROOT},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {"cwd": str(tmp_path), "mcpServers": []},
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)

    first_prompt = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Initial turn before close"}]},
        timeout=5.0,
    )
    assert first_prompt["result"]["stopReason"] == "end_turn"

    close_resp = await harness.send_request("session/close", {"sessionId": sid})
    assert close_resp["result"] == {}

    harness.notifications.clear()
    load_resp = await harness.send_request(
        "session/load",
        {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
        timeout=5.0,
    )
    assert load_resp["result"]["sessionId"] == sid
    await _read_next_notifications(harness, count=2)

    harness.notifications.clear()
    resumed_prompt = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Resumed turn after reload"}]},
        timeout=5.0,
    )
    assert resumed_prompt["result"]["stopReason"] == "end_turn"

    live_agent_chunks = [
        n["params"]["update"]["content"]["text"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
        and not n["params"]["update"].get("_meta", {}).get("kiro", {}).get("replay")
    ]
    assert "".join(live_agent_chunks) == "WAVE6_ACK: Resumed turn after reload"


async def test_kas_v3_carousel_prompt_and_session_load_render_markdown_images(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """KAS v3 stdio: carousel blocks unwrap in both live turns and session/load replay."""
    carousel_bin = tmp_path / "carousel-agy"
    carousel_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "if '-p' in args:\n"
        "    print(json.dumps({'status': 'SUCCESS', 'command': {'name': 'quota', 'data': {}}}))\n"
        "    sys.exit(0)\n"
        "chunks = [\n"
        "    'Summary:\\n\\n```',\n"
        "    '`carousel\\n![Shot A](/Users/benwart/.gemini/antigravity-cli/brain/id/a.png)\\n',\n"
        "    '<!-- slide -->\\n![Shot B]'\n"
        "    '(file:///Users/benwart/.gemini/antigravity-cli/brain/id/b.png)\\n````\\n',\n"
        "]\n"
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
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {"
        "'protocolVersion': 1, 'agentCapabilities': {'loadSession': True}}}), flush=True)\n"
        "    elif method == 'session/new':\n"
        "        seq += 1\n"
        "        sid = f'conv-car-e2e-{seq}'\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {\n"
        "            'sessionId': sid,\n"
        "            'models': {'currentModelId': 'gemini-3.7-pro', 'availableModels': [\n"
        "                {'modelId': 'gemini-3.7-pro', 'name': 'Gemini 3.7 Pro'}\n"
        "            ]},\n"
        "            'configOptions': [],\n"
        "        }}), flush=True)\n"
        "    elif method == 'session/load':\n"
        "        sid = params.get('sessionId') or 'conv-car-e2e-1'\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {\n"
        "            'sessionId': sid,\n"
        "            'models': {'currentModelId': 'gemini-3.7-pro', 'availableModels': [\n"
        "                {'modelId': 'gemini-3.7-pro', 'name': 'Gemini 3.7 Pro'}\n"
        "            ]},\n"
        "            'configOptions': [],\n"
        "        }}), flush=True)\n"
        "    elif method in ('session/set_model', 'session/set_config_option', 'session/close'):\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {}}), flush=True)\n"
        "    elif method == 'session/prompt':\n"
        "        sid = params['sessionId']\n"
        "        for d in chunks:\n"
        "            print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
        "                'sessionId': sid,\n"
        "                'update': {'sessionUpdate': 'agent_message_chunk',"
        " 'content': {'type': 'text', 'text': d}},\n"
        "            }}), flush=True)\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
        "            'sessionId': sid,\n"
        "            'update': {'sessionUpdate': 'usage_update', 'used': 24, 'size': 1000000},\n"
        "        }}), flush=True)\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {"
        "'stopReason': 'end_turn'}}), flush=True)\n",
        encoding="utf-8",
    )
    carousel_bin.chmod(0o755)

    harness = await acp_harness_factory.spawn(
        args=["acp", "--agent-engine", "v3", "--save-dir", str(tmp_path / "store")],
        env={"AGY_BIN": str(carousel_bin), "PYTHONPATH": SRC_ROOT},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request(
        "session/new",
        {"cwd": str(tmp_path), "mcpServers": []},
    )
    sid = new_resp["result"]["sessionId"]
    await _read_next_notifications(harness, count=2)
    harness.notifications.clear()

    prompt_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "show gallery"}]},
        timeout=5.0,
    )
    assert prompt_resp["result"]["stopReason"] == "end_turn"

    expected = (
        "Summary:\n\n"
        "![Shot A](/Users/benwart/.gemini/antigravity-cli/brain/id/a.png)\n\n"
        "![Shot B](/Users/benwart/.gemini/antigravity-cli/brain/id/b.png)\n"
    )
    live_chunks = [
        n["params"]["update"]["content"]["text"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
    ]
    assert "".join(live_chunks) == expected

    await harness.send_request("session/close", {"sessionId": sid})
    harness.notifications.clear()

    load_resp = await harness.send_request(
        "session/load",
        {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
        timeout=5.0,
    )
    assert load_resp["result"]["sessionId"] == sid

    replay_chunks = [
        n["params"]["update"]["content"]["text"]
        for n in harness.notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
        and n["params"]["update"].get("_meta", {}).get("kiro", {}).get("replay") is True
    ]
    assert replay_chunks == [expected]
