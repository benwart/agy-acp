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
import json, os, pathlib, sys, time

args = sys.argv[1:]
if args and args[0] == "models":
    print("gemini-3.7-pro\\tGemini 3.7 Pro")
    print("gemini-3.8-flash\\tGemini 3.8 Flash")
    print("claude-sonnet-4-6\\tClaude Sonnet 4.6")
    sys.exit(0)

prompt = ""
conv_id = ""
for i, arg in enumerate(args):
    if arg in ("-p", "--print", "--prompt") and i + 1 < len(args):
        prompt = args[i + 1]
    elif arg == "--conversation" and i + 1 < len(args):
        conv_id = args[i + 1]

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

trace_entry = {{
    "args": args,
    "prompt": prompt,
    "conversation_id": conv_id,
    "env": {{
        "KIROCREW_PORT": os.environ.get("KIROCREW_PORT", ""),
        "KIROCREW_STUB_SESSION_TOKEN": os.environ.get("KIROCREW_STUB_SESSION_TOKEN", ""),
        "KIROCREW_ROUTE_SPAWN_RUN": os.environ.get("KIROCREW_ROUTE_SPAWN_RUN", ""),
    }},
}}
with open({str(trace_file)!r}, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(trace_entry) + "\\n")

active_conv = conv_id or "conv-kas-v3-001"
print(json.dumps({{"event": "init", "init": {{"conversation_id": active_conv}}}}), flush=True)

if "Summarize our conversation so far" in prompt:
    summary = "Summary: KAS v3 session compacted cleanly."
    print(json.dumps({{
        "event": "result",
        "result": {{
            "conversation_id": active_conv,
            "status": "SUCCESS",
            "response": summary,
            "usage": {{"input_tokens": 18, "output_tokens": 9, "total_tokens": 27}},
        }},
    }}), flush=True)
    sys.exit(0)

if "TRIGGER_SLOW_STEER" in prompt:
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 1,
            "step_type": "planner_response",
            "status": "DONE",
            "text": "Working before steer... ",
        }},
    }}), flush=True)
    time.sleep(0.35)
    print(json.dumps({{
        "event": "result",
        "result": {{
            "conversation_id": active_conv,
            "status": "SUCCESS",
            "response": "Working before steer...",
            "usage": {{"input_tokens": 10, "output_tokens": 6, "total_tokens": 16}},
        }},
    }}), flush=True)
    sys.exit(0)

if "TRIGGER_KAS_TOOLS" in prompt:
    # 1. run_command
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 1,
            "step_type": "run_command",
            "status": "RUNNING",
            "tool_name": "run_command",
            "tool_info": {{
                "tool_name": "run_command",
                "arguments": {{"CommandLine": "pytest -q"}},
            }},
        }},
    }}), flush=True)
    time.sleep(0.08)
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 1,
            "step_type": "run_command",
            "status": "DONE",
            "tool_name": "run_command",
            "tool_info": {{
                "tool_name": "run_command",
                "arguments": {{"CommandLine": "pytest -q"}},
                "output": "1 passed",
            }},
        }},
    }}), flush=True)

    # 2. edit_file -> str_replace + diff
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 2,
            "step_type": "edit_file",
            "status": "RUNNING",
            "tool_name": "edit_file",
            "tool_info": {{
                "tool_name": "edit_file",
                "arguments": {{
                    "Path": "/repo/app.py",
                    "Target": "old_val = 1",
                    "Replacement": "new_val = 2",
                }},
            }},
        }},
    }}), flush=True)
    time.sleep(0.08)
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 2,
            "step_type": "edit_file",
            "status": "DONE",
            "tool_name": "edit_file",
            "tool_info": {{
                "tool_name": "edit_file",
                "arguments": {{
                    "Path": "/repo/app.py",
                    "Target": "old_val = 1",
                    "Replacement": "new_val = 2",
                }},
                "output": "Edited /repo/app.py",
            }},
        }},
    }}), flush=True)

    # 3. delete_file -> delete
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 3,
            "step_type": "delete_file",
            "status": "RUNNING",
            "tool_name": "delete_file",
            "tool_info": {{
                "tool_name": "delete_file",
                "arguments": {{"Path": "/repo/obsolete.py"}},
            }},
        }},
    }}), flush=True)
    time.sleep(0.08)
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 3,
            "step_type": "delete_file",
            "status": "DONE",
            "tool_name": "delete_file",
            "tool_info": {{
                "tool_name": "delete_file",
                "arguments": {{"Path": "/repo/obsolete.py"}},
                "output": "Deleted",
            }},
        }},
    }}), flush=True)

    # 4. Allowed MCP tool: kirocrew-core/spawn_run (auto-approved by permissions!)
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 4,
            "step_type": "call_mcp_tool",
            "status": "RUNNING",
            "tool_name": "call_mcp_tool",
            "tool_info": {{
                "tool_name": "call_mcp_tool",
                "arguments": {{
                    "ServerName": "kirocrew-core",
                    "ToolName": "spawn_run",
                    "Arguments": {{"tasks": [{{"goal": "verify"}}]}},
                }},
            }},
        }},
    }}), flush=True)
    time.sleep(0.05)
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 4,
            "step_type": "call_mcp_tool",
            "status": "DONE",
            "tool_name": "call_mcp_tool",
            "tool_info": {{
                "tool_name": "call_mcp_tool",
                "arguments": {{
                    "ServerName": "kirocrew-core",
                    "ToolName": "spawn_run",
                    "Arguments": {{"tasks": [{{"goal": "verify"}}]}},
                }},
                "output": "spawned",
            }},
        }},
    }}), flush=True)

    # 5. Unapproved MCP tool: ext-srv/drop_table (rejected -> aborts subprocess!)
    print(json.dumps({{
        "event": "step_update",
        "step_update": {{
            "step_index": 5,
            "step_type": "call_mcp_tool",
            "status": "RUNNING",
            "tool_name": "call_mcp_tool",
            "tool_info": {{
                "tool_name": "call_mcp_tool",
                "arguments": {{
                    "ServerName": "ext-srv",
                    "ToolName": "drop_table",
                    "Arguments": {{"table": "users"}},
                }},
            }},
        }},
    }}), flush=True)
    time.sleep(0.4)
    # Must NEVER execute if subprocess was terminated upon permission rejection!
    pathlib.Path({str(marker_file)!r}).write_text("LEAKED_AFTER_REJECT", encoding="utf-8")

reply_text = f"ACK_V3: {{prompt.splitlines()[-1] if prompt else ''}}"
print(json.dumps({{
    "event": "step_update",
    "step_update": {{
        "step_index": 10,
        "step_type": "planner_response",
        "status": "DONE",
        "text": reply_text,
    }},
}}), flush=True)
print(json.dumps({{
    "event": "result",
    "result": {{
        "conversation_id": active_conv,
        "status": "SUCCESS",
        "response": reply_text,
        "usage": {{
            "input_tokens": 25,
            "output_tokens": 12,
            "total_tokens": 37,
            "context_tokens": 4200,
        }},
    }},
}}), flush=True)
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
    }
    assert kiro_caps["checkpoints"] is False

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
    sent_prompt = traces[0]["prompt"]
    assert "[AGENT MANDATE]" in sent_prompt
    assert "[KIROCREW SUBAGENT ROUTING]" in sent_prompt
    assert "[AGENT SYSTEM PROMPT]" in sent_prompt
    assert "You are Kiro v3 custom agent." in sent_prompt
    assert traces[0]["env"]["KIROCREW_PORT"] == "9444"
    assert traces[0]["env"]["KIROCREW_STUB_SESSION_TOKEN"] == "tok-live-v3"
    assert traces[0]["env"]["KIROCREW_ROUTE_SPAWN_RUN"] == "1"


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
    sent_args = traces[-1]["args"]
    assert sent_args[sent_args.index("--model") + 1] == "claude-sonnet-4-6"
    assert sent_args[sent_args.index("--effort") + 1] == "xhigh"


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
    assert "Persistent contract" in traces[-1]["prompt"]

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
