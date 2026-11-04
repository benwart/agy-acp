"""Unit tests for AgyCliProvider async subprocess management and stream-json parsing."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import anyio
import pytest

from agy_acp.providers.agy_cli_provider import (
    MAX_INLINE_PROMPT_BYTES,
    AgyCliProvider,
    ToolCallEvent,
    build_agy_args,
    prepare_prompt_arg,
    resolve_agy_binary,
)


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_resolve_agy_binary_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_agy_binary checks explicit path, AGY_BIN, ~/.local/bin/agy, and PATH."""
    explicit = _write_executable(tmp_path / "explicit-agy", "#!/bin/sh\n")
    env_bin = _write_executable(tmp_path / "env-agy", "#!/bin/sh\n")
    monkeypatch.setenv("AGY_BIN", str(env_bin))

    assert resolve_agy_binary(explicit) == str(explicit)
    assert resolve_agy_binary() == str(env_bin)

    monkeypatch.delenv("AGY_BIN", raising=False)
    resolved = resolve_agy_binary()
    assert resolved.endswith("agy")


def test_prepare_prompt_arg_inline_and_offload(tmp_path: Path) -> None:
    """Verify prompts <= 64 KiB stay inline and prompts > 64 KiB offload to a 0600 temp file."""
    short_prompt = "Explain how ACP works."
    arg, temp_file = prepare_prompt_arg(short_prompt, session_id="sess-12345678", temp_dir=tmp_path)
    assert arg == short_prompt
    assert temp_file is None

    large_prompt = "A" * (MAX_INLINE_PROMPT_BYTES + 128)
    arg_large, temp_large = prepare_prompt_arg(
        large_prompt,
        session_id="sess-12345678",
        temp_dir=tmp_path,
    )
    assert temp_large is not None
    assert temp_large.is_file()
    assert (temp_large.stat().st_mode & 0o777) == 0o600
    assert temp_large.read_text(encoding="utf-8") == large_prompt
    assert f"Please read the prompt and context in {temp_large} using view_file" in arg_large


def test_build_agy_args_maps_modes_effort_and_flags() -> None:
    """Verify build_agy_args constructs CLI flags matching antigravity-acp."""
    args_auto = build_agy_args(
        cwd="/workspace",
        prompt_arg="Hello world",
        conversation_id="conv-123",
        model="gemini-3.7-pro",
        mode="auto",
        effort="high",
        sandbox=True,
    )
    assert "--add-dir" in args_auto
    assert args_auto[args_auto.index("--add-dir") + 1] == "/workspace"
    assert "--conversation" in args_auto
    assert args_auto[args_auto.index("--conversation") + 1] == "conv-123"
    assert "--model" in args_auto
    assert args_auto[args_auto.index("--model") + 1] == "gemini-3.7-pro"
    assert "--mode" in args_auto
    assert args_auto[args_auto.index("--mode") + 1] == "auto"
    assert "--effort" in args_auto
    assert args_auto[args_auto.index("--effort") + 1] == "high"
    assert "--dangerously-skip-permissions" in args_auto
    assert "--sandbox" in args_auto
    assert "--output-format" in args_auto
    assert args_auto[args_auto.index("--output-format") + 1] == "stream-json"
    assert args_auto[-2:] == ["-p", "Hello world"]

    args_approve = build_agy_args(
        cwd="/workspace",
        prompt_arg="Check file",
        mode="approve",
    )
    assert args_approve[args_approve.index("--mode") + 1] == "accept-edits"
    assert "--dangerously-skip-permissions" not in args_approve

    args_ro = build_agy_args(
        cwd="/workspace",
        prompt_arg="Plan work",
        mode="read-only",
    )
    assert args_ro[args_ro.index("--mode") + 1] == "plan"
    assert "--dangerously-skip-permissions" not in args_ro


async def test_discover_models_parses_tab_separated_and_caches(tmp_path: Path) -> None:
    """Verify discover_models runs `agy models` asynchronously via anyio and caches output."""
    fake_agy = _write_executable(
        tmp_path / "agy",
        "#!/bin/sh\n"
        'if [ "$1" = "models" ]; then\n'
        '  printf "gemini-3.7-pro\\tGemini 3.7 Pro\\ngemini-3.8-flash\\tGemini 3.8 Flash\\n"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
    )
    db_file = tmp_path / "state.db"
    provider = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    models = await provider.discover_models(refresh=True)
    assert len(models) == 2
    assert models[0]["modelId"] == "gemini-3.7-pro"
    assert models[0]["name"] == "Gemini 3.7 Pro"
    assert db_file.is_file()
    assert provider.state_store.load_models() == models


async def test_run_slash_command_executes_async_subprocess(tmp_path: Path) -> None:
    """Verify run_slash_command executes `agy -p <cmd> --output-format json` via anyio."""
    usage_payload = {
        "conversation_id": "",
        "status": "SUCCESS",
        "response": "Gemini Models\tFive Hour Limit Remaining\t95%\t2026-10-24T07:52:04Z\n",
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
                            }
                        ],
                    }
                ]
            },
        },
    }
    fake_agy = _write_executable(
        tmp_path / "agy",
        f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(usage_payload)}'\n",
    )
    provider = AgyCliProvider(bin_path=fake_agy)
    result = await provider.run_slash_command("/usage", cwd=str(tmp_path))
    assert result["status"] == "SUCCESS"
    assert result["command"]["name"] == "usage"
    assert result["command"]["data"]["groups"][0]["name"] == "Gemini Models"


async def test_stream_prompt_parses_ndjson_events_and_cancels(tmp_path: Path) -> None:
    """Verify stream_prompt streams init, step_update (thought/tool/text), result, and cancels."""
    events = [
        {"event": "init", "init": {"conversation_id": "conv-stream-123"}},
        {
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "step_type": "planner_response",
                "status": "RUNNING",
                "thinking": "Analyzing request...",
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "step_type": "run_command",
                "status": "RUNNING",
                "tool_name": "run_command",
                "tool_info": {
                    "tool_name": "run_command",
                    "arguments": {"command": "ls -la"},
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "step_type": "run_command",
                "status": "DONE",
                "tool_name": "run_command",
                "tool_info": {
                    "tool_name": "run_command",
                    "arguments": {"command": "ls -la"},
                    "output": "total 0",
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 3,
                "step_type": "planner_response",
                "status": "DONE",
                "text": "Hello from agy!",
            },
        },
        {
            "event": "result",
            "result": {
                "conversation_id": "conv-stream-123",
                "status": "SUCCESS",
                "response": "Hello from agy!",
                "usage": {
                    "input_tokens": 42,
                    "output_tokens": 15,
                    "thinking_tokens": 8,
                    "cache_read_tokens": 4,
                    "total_tokens": 69,
                },
            },
        },
    ]
    ndjson_lines = "\n".join(json.dumps(ev) for ev in events)
    fake_agy = _write_executable(
        tmp_path / "agy",
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "if 'SLEEP_FOREVER' in ' '.join(sys.argv):\n"
        "    time.sleep(10)\n"
        f"print({ndjson_lines!r}, flush=True)\n",
    )

    thoughts: list[str] = []
    texts: list[str] = []
    tool_starts: list[ToolCallEvent] = []
    tool_updates: list[ToolCallEvent] = []

    async def _on_thought(chunk: str) -> None:
        thoughts.append(chunk)

    async def _on_text(chunk: str) -> None:
        texts.append(chunk)

    async def _on_tool_call(event: ToolCallEvent) -> bool:
        tool_starts.append(event)
        return True

    async def _on_tool_update(event: ToolCallEvent) -> None:
        tool_updates.append(event)

    provider = AgyCliProvider(bin_path=fake_agy)
    outcome = await provider.stream_prompt(
        cwd=str(tmp_path),
        prompt="Say hello",
        session_id="sess-1",
        on_thought=_on_thought,
        on_text=_on_text,
        on_tool_call=_on_tool_call,
        on_tool_update=_on_tool_update,
    )
    assert outcome.stop_reason == "end_turn"
    assert outcome.conversation_id == "conv-stream-123"
    assert thoughts == ["Analyzing request..."]
    assert "".join(texts) == "Hello from agy!"
    assert len(tool_starts) == 1
    assert tool_starts[0].kind == "execute"
    assert len(tool_updates) == 1
    assert tool_updates[0].status == "completed"
    assert tool_updates[0].output == "total 0"
    assert outcome.usage["input_tokens"] == 42
    assert outcome.usage["output_tokens"] == 15
    assert outcome.usage["cache_read_tokens"] == 4
    assert outcome.usage["thinking_tokens"] == 8

    # Verify mid-turn cancellation terminates subprocess and returns stop_reason == 'cancelled'
    started = anyio.Event()
    cancelled_outcome: Any = None

    async def _run_slow() -> None:
        nonlocal cancelled_outcome
        started.set()
        cancelled_outcome = await provider.stream_prompt(
            cwd=str(tmp_path),
            prompt="SLEEP_FOREVER",
            session_id="sess-cancel",
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_slow)
        await started.wait()
        await anyio.sleep(0.05)
        await provider.cancel_session("sess-cancel")

    assert cancelled_outcome is not None
    assert cancelled_outcome.stop_reason == "cancelled"
    _ = os


async def test_stream_prompt_parses_real_agy_deltas_and_recovers_invalid_model(
    tmp_path: Path,
) -> None:
    """Verify stream_prompt handles real agy state/text_delta/thinking_delta and model fallback."""
    events = [
        {
            "event": "init",
            "conversation_id": "conv-real-999",
            "init": {"model": "gemini-4-argon-medium"},
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "state": "ACTIVE",
                "step_type": "agent_response",
                "thinking_delta": "Thinking delta...",
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "state": "ACTIVE",
                "step_type": "tool",
                "tool_name": "run_command",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"CommandLine": "ls /tmp"},
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "run_command",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"CommandLine": "ls /tmp"},
                    "output": "ok-tmp",
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 3,
                "state": "ACTIVE",
                "step_type": "agent_response",
                "text_delta": "Real ",
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 3,
                "state": "DONE",
                "step_type": "agent_response",
                "text_delta": "output!",
            },
        },
        {
            "event": "result",
            "result": {
                "conversation_id": "conv-real-999",
                "status": "SUCCESS",
                "response": "Real output!",
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            },
        },
    ]
    err_event = {
        "event": "result",
        "result": {
            "conversation_id": "",
            "status": "ERROR",
            "response": "",
            "error": 'invalid model selection (--model "gemini-3.7-pro")',
        },
    }
    ndjson_lines = "\n".join(json.dumps(ev) for ev in events)
    err_line = json.dumps(err_event)
    fake_agy = _write_executable(
        tmp_path / "agy",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if 'gemini-3.7-pro' in sys.argv:\n"
        f"    print({err_line!r}, flush=True)\n"
        "    sys.exit(1)\n"
        f"print({ndjson_lines!r}, flush=True)\n",
    )

    thoughts: list[str] = []
    texts: list[str] = []
    tool_starts: list[ToolCallEvent] = []
    tool_updates: list[ToolCallEvent] = []

    async def _on_thought(chunk: str) -> None:
        thoughts.append(chunk)

    async def _on_text(chunk: str) -> None:
        texts.append(chunk)

    async def _on_tool_call(event: ToolCallEvent) -> bool:
        tool_starts.append(event)
        return True

    async def _on_tool_update(event: ToolCallEvent) -> None:
        tool_updates.append(event)

    provider = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    outcome = await provider.stream_prompt(
        cwd=str(tmp_path),
        prompt="Test real deltas",
        session_id="sess-real",
        model="gemini-3.7-pro",
        on_thought=_on_thought,
        on_text=_on_text,
        on_tool_call=_on_tool_call,
        on_tool_update=_on_tool_update,
    )
    assert outcome.conversation_id == "conv-real-999"
    assert thoughts == ["Thinking delta..."]
    assert "".join(texts) == "Real output!"
    assert len(tool_starts) == 1
    assert tool_starts[0].raw_input == {"CommandLine": "ls /tmp", "command": "ls /tmp"}
    assert tool_starts[0].kiro_tool_name == "execute_bash"
    assert tool_starts[0].mcp_server_name == ""
    assert tool_starts[0].kiro_tool_id == "run_command"
    assert len(tool_updates) == 1
    assert tool_updates[0].status == "completed"
    assert tool_updates[0].output == "ok-tmp"


async def test_resolve_kiro_tool_identity_and_sync_mcp_servers(tmp_path: Path) -> None:
    """Verify Kiro tool identity mapping (built-in & MCP) and MCP config synchronization."""
    from agy_acp.providers.agy_cli_provider import (
        _classify_tool_step,
        _normalize_tool_arguments,
        _resolve_kiro_tool_identity,
        load_kiro_agent_mcp_servers,
    )

    assert _resolve_kiro_tool_identity("tool", "run_command", {"command": "ls"}) == (
        "execute_bash",
        "",
        "run_command",
    )
    assert _resolve_kiro_tool_identity("tool", "read_file", {"Path": "/tmp/a"}) == (
        "fs_read",
        "",
        "read_file",
    )
    assert _resolve_kiro_tool_identity("tool", "edit_file", {"Path": "/tmp/a"}) == (
        "fs_write",
        "",
        "fs_write",
    )
    assert _resolve_kiro_tool_identity("tool", "search_web", {"query": "test"}) == (
        "web_search",
        "",
        "remote_web_search",
    )
    assert _resolve_kiro_tool_identity("tool", "invoke_subagent", {"Role": "dev"}) == (
        "use_subagent",
        "",
        "invoke_sub_agent",
    )

    mcp_raw = {
        "ServerName": "kirocrew-core",
        "ToolName": "spawn_run",
        "Arguments": {"agent": "researcher", "task": "audit"},
    }
    assert _resolve_kiro_tool_identity("tool", "call_mcp_tool", mcp_raw) == (
        "spawn_run",
        "kirocrew-core",
        "@kirocrew-core/spawn_run",
    )
    norm_mcp_args = _normalize_tool_arguments(mcp_raw, tool_name="call_mcp_tool")
    assert norm_mcp_args == {"agent": "researcher", "task": "audit"}
    kind, title, _ = _classify_tool_step("tool", "call_mcp_tool", mcp_raw)
    assert kind == "other"
    assert title == "spawn_run"

    cron_raw = {
        "ServerName": "kirocrew-cron",
        "ToolName": "cron_list",
        "Arguments": {},
    }
    _, cron_title, _ = _classify_tool_step("tool", "call_mcp_tool", cron_raw)
    assert cron_title == "@kirocrew-cron/cron_list"

    # Verify loading MCP servers from .kiro/agents/<agent>.json and syncing into mcp_config.json
    agents_dir = tmp_path / ".kiro" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "kirocrew.json").write_text(
        json.dumps(
            {
                "name": "kirocrew",
                "mcpServers": {
                    "kirocrew-core": {
                        "command": "/usr/local/bin/kirocrew",
                        "args": ["mcp-core"],
                    },
                    "kirocrew-cron": {
                        "command": "/usr/local/bin/kirocrew",
                        "args": ["mcp-cron"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    agent_servers = load_kiro_agent_mcp_servers("kirocrew", cwd=str(tmp_path), include_home=False)
    assert len(agent_servers) == 2

    mcp_cfg_file = tmp_path / "mcp_config.json"
    mcp_cfg_file.write_text(
        json.dumps({"mcpServers": {"rea": {"command": "npx", "args": ["rea"]}}}),
        encoding="utf-8",
    )
    provider = AgyCliProvider(save_dir=tmp_path, mcp_config_path=mcp_cfg_file)
    synced = provider.sync_mcp_servers(agent_servers)
    assert set(synced) == {"kirocrew-core", "kirocrew-cron"}
    merged = json.loads(mcp_cfg_file.read_text(encoding="utf-8"))
    assert "rea" in merged["mcpServers"]
    assert merged["mcpServers"]["kirocrew-core"]["command"] == "/usr/local/bin/kirocrew"
    assert merged["mcpServers"]["kirocrew-core"]["args"] == ["mcp-core"]


async def test_stream_prompt_tracks_last_step_context_tokens_and_ignores_system_message_as_tool(
    tmp_path: Path,
) -> None:
    """Verify stream_prompt extracts context_tokens from last step and ignores system_message."""
    events = [
        {"event": "init", "init": {"conversation_id": "conv-ctx-1", "model": "gemini-4-argon"}},
        {
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "step_type": "agent_response",
                "state": "DONE",
                "usage": {
                    "input_tokens": 23917,
                    "output_tokens": 159,
                    "thinking_tokens": 112,
                    "cache_read_tokens": 0,
                    "total_tokens": 24076,
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 3,
                "step_type": "agent_response",
                "state": "DONE",
                "text": "Final answer",
                "usage": {
                    "input_tokens": 3334,
                    "output_tokens": 1009,
                    "thinking_tokens": 989,
                    "cache_read_tokens": 25837,
                    "total_tokens": 4343,
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 4,
                "step_type": "system_message",
                "state": "DONE",
            },
        },
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "conversation_id": "conv-ctx-1",
                "response": "Final answer",
                "usage": {
                    "input_tokens": 415816,
                    "output_tokens": 19955,
                    "thinking_tokens": 15000,
                    "cache_read_tokens": 5748702,
                    "total_tokens": 6199473,
                },
            },
        },
    ]
    script_lines = ["#!/bin/sh"] + [f"printf '%s\\n' '{json.dumps(e)}'" for e in events]
    fake_agy = _write_executable(tmp_path / "agy", "\n".join(script_lines) + "\n")

    provider = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    await provider.start_or_resume_agent(session_id="sess-ctx-1", cwd=str(tmp_path))
    tool_calls: list[ToolCallEvent] = []
    usage_events: list[int] = []

    async def _on_tool_call(ev: ToolCallEvent) -> bool:
        tool_calls.append(ev)
        return True

    async def _on_usage(ctx_tokens: int) -> None:
        usage_events.append(ctx_tokens)

    outcome = await provider.stream_prompt(
        cwd=str(tmp_path),
        prompt="Run multi-step task",
        session_id="sess-ctx-1",
        on_tool_call=_on_tool_call,
        on_usage=_on_usage,
    )
    assert tool_calls == []
    # Active context window occupancy streamed at step 1 (24076) and step 3 (30180)
    assert usage_events == [24076, 30180]
    assert outcome.usage["context_tokens"] == 30180
    persisted = provider.state_store.get_session("sess-ctx-1")
    assert persisted is not None
    assert persisted["contextTokens"] == 30180


async def test_stream_prompt_mid_turn_steer_interrupts_and_continues_conversation(
    tmp_path: Path,
) -> None:
    """Verify mid-turn steer interrupts running agy, emits steering_consumed, and resumes conv."""
    fake_agy = _write_executable(
        tmp_path / "agy",
        "#!/usr/bin/env python3\n"
        "import json, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
        "args = sys.argv[1:]\n"
        "prompt = args[args.index('-p') + 1] if '-p' in args else ''\n"
        "conv = args[args.index('--conversation') + 1] if '--conversation' in args else ''\n"
        "if 'INITIAL_LONG_TASK' in prompt:\n"
        "    init_ev = {'event': 'init', 'init': {'conversation_id': 'conv-steer-777'}}\n"
        "    print(json.dumps(init_ev), flush=True)\n"
        "    print(json.dumps({\n"
        "        'event': 'step_update',\n"
        "        'step_update': {\n"
        "            'step_index': 1,\n"
        "            'state': 'ACTIVE',\n"
        "            'step_type': 'agent_response',\n"
        "            'text_delta': 'Working on initial task... ',\n"
        "        },\n"
        "    }), flush=True)\n"
        "    tool_info = {'name': 'run_command', 'parameters': {'CommandLine': 'sleep 30'}}\n"
        "    print(json.dumps({\n"
        "        'event': 'step_update',\n"
        "        'step_update': {\n"
        "            'step_index': 2,\n"
        "            'state': 'ACTIVE',\n"
        "            'step_type': 'tool',\n"
        "            'tool_name': 'run_command',\n"
        "            'tool_info': tool_info,\n"
        "        },\n"
        "    }), flush=True)\n"
        "    time.sleep(10)\n"
        "else:\n"
        "    img_ok = False\n"
        "    for token in prompt.split():\n"
        "        if token.endswith('.png') and pathlib.Path(token).is_file():\n"
        "            img_ok = True\n"
        "    cid = conv or 'conv-steer-777'\n"
        "    print(json.dumps({'event': 'init', 'init': {'conversation_id': cid}}), flush=True)\n"
        "    first_line = prompt.splitlines()[0]\n"
        "    delta = f'Steered on {conv} (image_exists={img_ok}): {first_line}'\n"
        "    print(json.dumps({\n"
        "        'event': 'step_update',\n"
        "        'step_update': {\n"
        "            'step_index': 3,\n"
        "            'state': 'DONE',\n"
        "            'step_type': 'agent_response',\n"
        "            'text_delta': delta,\n"
        "        },\n"
        "    }), flush=True)\n"
        "    print(json.dumps({\n"
        "        'event': 'result',\n"
        "        'result': {\n"
        "            'conversation_id': cid,\n"
        "            'status': 'SUCCESS',\n"
        "            'response': 'Steered response',\n"
        "            'usage': {'input_tokens': 25, 'output_tokens': 12, 'total_tokens': 37},\n"
        "        },\n"
        "    }), flush=True)\n",
    )

    provider = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    await provider.start_or_resume_agent(session_id="sess-steer-1", cwd=str(tmp_path))

    first_tool_started = anyio.Event()
    texts: list[str] = []
    tool_updates: list[ToolCallEvent] = []
    consumed_steers: list[str] = []

    async def _on_text(chunk: str) -> None:
        texts.append(chunk)

    async def _on_tool_call(ev: ToolCallEvent) -> bool:
        _ = ev
        first_tool_started.set()
        return True

    async def _on_tool_update(ev: ToolCallEvent) -> None:
        tool_updates.append(ev)

    async def _on_steer_consumed(raw_content: str) -> None:
        consumed_steers.append(raw_content)

    temp_img = tmp_path / "staged-steer-img.png"
    temp_img.write_bytes(b"\x89PNG\r\n\x1a\n")
    raw_steer_msg = (
        "<user_message>\n![image](/tmp/upload.png)\n\n"
        "Here are the errors from the IDE\n</user_message>"
    )
    steered_prompt = (
        f"Here are the errors from the IDE\n\n"
        f"[User attached an image: {temp_img} — inspect it with view_file if needed]"
    )

    outcome: Any = None

    async def _run_turn() -> None:
        nonlocal outcome
        outcome = await provider.stream_prompt(
            cwd=str(tmp_path),
            prompt="INITIAL_LONG_TASK",
            session_id="sess-steer-1",
            on_text=_on_text,
            on_tool_call=_on_tool_call,
            on_tool_update=_on_tool_update,
            on_steer_consumed=_on_steer_consumed,
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_turn)
        await first_tool_started.wait()
        assert provider.is_turn_active("sess-steer-1") is True
        queued = await provider.steer_session(
            "sess-steer-1",
            raw_message=raw_steer_msg,
            prompt=steered_prompt,
            temp_files=[temp_img],
        )
        assert queued is True

    assert outcome is not None
    assert outcome.stop_reason == "end_turn"
    assert outcome.conversation_id == "conv-steer-777"
    assert consumed_steers == [raw_steer_msg]
    assert any(u.status in {"completed", "failed"} for u in tool_updates)
    full_text = "".join(texts)
    assert "Working on initial task..." in full_text
    assert (
        "Steered on conv-steer-777 (image_exists=True): Here are the errors from the IDE"
        in full_text
    )
    assert not temp_img.exists()


def test_normalize_tool_arguments_populates_tool_use_purpose() -> None:
    """Verify `toolSummary` populates both `description` and KiroCrew `__toolUsePurpose`."""
    from agy_acp.providers.agy_cli_provider import _normalize_tool_arguments

    normalized = _normalize_tool_arguments(
        {"CommandLine": "uv run pytest", "toolSummary": "Unit test execution"},
        tool_name="run_command",
    )
    assert normalized["command"] == "uv run pytest"
    assert normalized["description"] == "Unit test execution"
    assert normalized["__toolUsePurpose"] == "Unit test execution"


async def test_stream_prompt_ignores_error_message_step_type(tmp_path: Path) -> None:
    """Verify `step_type: 'error_message'` does not emit a fake tool call or permission prompt."""
    fake_agy = _write_executable(
        tmp_path / "agy",
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'event': 'init', 'init': {'conversation_id': 'conv-err-step'}}))\n"
        "print(json.dumps({\n"
        "    'event': 'step_update',\n"
        "    'step_update': {\n"
        "        'step_index': 1,\n"
        "        'state': 'DONE',\n"
        "        'step_type': 'error_message',\n"
        "    },\n"
        "}))\n"
        "print(json.dumps({\n"
        "    'event': 'result',\n"
        "    'result': {\n"
        "        'conversation_id': 'conv-err-step',\n"
        "        'status': 'SUCCESS',\n"
        "        'response': 'ok',\n"
        "    },\n"
        "}))\n",
    )
    provider = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    tool_calls: list[ToolCallEvent] = []
    tool_updates: list[ToolCallEvent] = []

    async def _on_tool_call(ev: ToolCallEvent) -> bool:
        tool_calls.append(ev)
        return True

    async def _on_tool_update(ev: ToolCallEvent) -> None:
        tool_updates.append(ev)

    outcome = await provider.stream_prompt(
        cwd=str(tmp_path),
        prompt="check",
        session_id="sess-err-step",
        on_tool_call=_on_tool_call,
        on_tool_update=_on_tool_update,
    )
    assert outcome.stop_reason == "end_turn"
    assert tool_calls == []
    assert tool_updates == []


async def test_stream_prompt_extracts_subagent_info_and_tracks_roster_lifecycle(
    tmp_path: Path,
) -> None:
    """Verify `subagent_info.subagents` populates tool I/O and drives `on_subagent_list`."""
    list_output = "You have 2 active subagent(s):\n" + json.dumps(
        [
            {
                "role": "Codebase Researcher",
                "type": "research",
                "conversationId": "sub-1",
                "state": "running",
                "stateDetail": "read_file",
            },
            {
                "role": "Orchestrator",
                "type": "self",
                "conversationId": "sub-3",
                "state": "waiting_for_dependents",
                "stateDetail": "",
            },
        ]
    )
    events = [
        {"event": "init", "init": {"conversation_id": "conv-sub-1"}},
        {
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "state": "DONE",
                "step_type": "subagent",
                "tool_name": "invoke_subagent",
                "subagent_info": {
                    "subagents": [
                        {
                            "type_name": "research",
                            "role": "Codebase Researcher",
                            "initial_prompt": "Inspect auth flow",
                            "conversation_id": "sub-1",
                            "log_uri": "file:///tmp/sub-1.jsonl",
                        },
                        {
                            "type_name": "self",
                            "role": "Test Runner",
                            "initial_prompt": "Run pytest",
                            "conversation_id": "sub-2",
                        },
                    ]
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "manage_subagents",
                "tool_info": {
                    "name": "manage_subagents",
                    "parameters": {"Action": "list"},
                    "output": list_output,
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 3,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "manage_subagents",
                "tool_info": {
                    "name": "manage_subagents",
                    "parameters": {"Action": "kill", "ConversationIds": ["sub-1"]},
                    "output": "Terminated sub-1.",
                },
            },
        },
        {
            "event": "result",
            "result": {
                "conversation_id": "conv-sub-1",
                "status": "SUCCESS",
                "response": "All subagents handled.",
            },
        },
    ]
    script_lines = ["#!/usr/bin/env python3", "import json"]
    for ev in events:
        script_lines.append(f"print(json.dumps({ev!r}), flush=True)")
    fake_agy = _write_executable(tmp_path / "agy", "\n".join(script_lines) + "\n")

    provider = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    tool_calls: list[ToolCallEvent] = []
    tool_updates: list[ToolCallEvent] = []
    roster_snapshots: list[list[dict[str, Any]]] = []

    async def _on_tool_call(ev: ToolCallEvent) -> bool:
        tool_calls.append(ev)
        return True

    async def _on_tool_update(ev: ToolCallEvent) -> None:
        tool_updates.append(ev)

    async def _on_subagent_list(subagents: list[dict[str, Any]]) -> None:
        roster_snapshots.append(json.loads(json.dumps(subagents)))

    outcome = await provider.stream_prompt(
        cwd=str(tmp_path),
        prompt="spawn workers",
        session_id="sess-sub-1",
        on_tool_call=_on_tool_call,
        on_tool_update=_on_tool_update,
        on_subagent_list=_on_subagent_list,
    )
    assert outcome.stop_reason == "end_turn"

    invoke_call = tool_calls[0]
    assert invoke_call.title == "invoke_subagent"
    assert invoke_call.raw_input["Role"] == "Codebase Researcher"
    assert invoke_call.raw_input["TypeName"] == "research"
    assert invoke_call.raw_input["Prompt"] == "Inspect auth flow"
    assert invoke_call.raw_input["__toolUsePurpose"] == "Codebase Researcher"

    invoke_update = tool_updates[0]
    assert invoke_update.status == "completed"
    parsed_output = json.loads(invoke_update.output)
    assert len(parsed_output) == 2
    assert parsed_output[0]["conversationId"] == "sub-1"
    assert parsed_output[0]["logUri"] == "file:///tmp/sub-1.jsonl"
    assert parsed_output[1]["conversationId"] == "sub-2"

    # Snapshot 0: invoke_subagent spawns sub-1 and sub-2 as running
    assert len(roster_snapshots) == 4
    snap0 = {entry["sessionId"]: entry for entry in roster_snapshots[0]}
    assert snap0["sub-1"] == {
        "sessionId": "sub-1",
        "sessionName": "Codebase Researcher",
        "role": "Codebase Researcher",
        "agentName": "research",
        "initialQuery": "Inspect auth flow",
        "status": {"type": "running", "message": ""},
    }
    assert snap0["sub-2"]["status"] == {"type": "running", "message": ""}

    # Snapshot 1: manage_subagents list updates sub-1, completes sub-2, and adds sub-3
    snap1 = {entry["sessionId"]: entry for entry in roster_snapshots[1]}
    assert snap1["sub-1"]["status"] == {"type": "running", "message": "read_file"}
    assert snap1["sub-2"]["status"] == {"type": "completed", "message": ""}
    assert snap1["sub-3"]["status"] == {
        "type": "running",
        "message": "waiting for subagents",
    }
    assert snap1["sub-3"]["initialQuery"] == "Orchestrator"

    # Snapshot 2: manage_subagents kill marks sub-1 cancelled
    snap2 = {entry["sessionId"]: entry for entry in roster_snapshots[2]}
    assert snap2["sub-1"]["status"] == {"type": "cancelled", "message": ""}
    assert snap2["sub-3"]["status"]["type"] == "running"

    # Snapshot 3: end_turn closes remaining running sub-3 as completed
    snap3 = {entry["sessionId"]: entry for entry in roster_snapshots[3]}
    assert snap3["sub-1"]["status"] == {"type": "cancelled", "message": ""}
    assert snap3["sub-2"]["status"] == {"type": "completed", "message": ""}
    assert snap3["sub-3"]["status"] == {"type": "completed", "message": ""}


async def test_stream_prompt_subagent_kill_all_no_active_and_cancelled_turn(
    tmp_path: Path,
) -> None:
    """Verify `kill_all`, `'no active subagents'`, and cancelled turns close tracked subagents."""
    events = [
        {"event": "init", "init": {"conversation_id": "conv-sub-cancel"}},
        {
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "state": "DONE",
                "step_type": "subagent",
                "tool_name": "invoke_subagent",
                "subagent_info": {
                    "subagents": [
                        {
                            "type_name": "research",
                            "role": "Worker A",
                            "initial_prompt": "Task A",
                            "conversation_id": "sub-a",
                        }
                    ]
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "manage_subagents",
                "tool_info": {
                    "name": "manage_subagents",
                    "parameters": {"Action": "list"},
                    "output": "You have no active subagents.",
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 3,
                "state": "DONE",
                "step_type": "subagent",
                "tool_name": "invoke_subagent",
                "subagent_info": {
                    "subagents": [
                        {
                            "type_name": "self",
                            "role": "Worker B",
                            "initial_prompt": "Task B",
                            "conversation_id": "sub-b",
                        },
                        {
                            "type_name": "self",
                            "role": "Worker C",
                            "initial_prompt": "Task C",
                            "conversation_id": "sub-c",
                        },
                    ]
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 4,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "manage_subagents",
                "tool_info": {
                    "name": "manage_subagents",
                    "parameters": {"Action": "kill_all"},
                    "output": "All subagents killed.",
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 5,
                "state": "DONE",
                "step_type": "subagent",
                "tool_name": "invoke_subagent",
                "subagent_info": {
                    "subagents": [
                        {
                            "type_name": "self",
                            "role": "Worker D",
                            "initial_prompt": "Task D",
                            "conversation_id": "sub-d",
                        }
                    ]
                },
            },
        },
        {
            "event": "result",
            "result": {
                "conversation_id": "conv-sub-cancel",
                "status": "CANCELLED",
                "response": "",
            },
        },
    ]
    script_lines = ["#!/usr/bin/env python3", "import json"]
    for ev in events:
        script_lines.append(f"print(json.dumps({ev!r}), flush=True)")
    fake_agy = _write_executable(tmp_path / "agy", "\n".join(script_lines) + "\n")

    provider = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    roster_snapshots: list[list[dict[str, Any]]] = []

    async def _on_subagent_list(subagents: list[dict[str, Any]]) -> None:
        roster_snapshots.append(json.loads(json.dumps(subagents)))

    outcome = await provider.stream_prompt(
        cwd=str(tmp_path),
        prompt="spawn and cancel",
        session_id="sess-sub-cancel",
        on_subagent_list=_on_subagent_list,
    )
    assert outcome.stop_reason == "cancelled"
    assert len(roster_snapshots) == 6
    assert roster_snapshots[1][0]["status"] == {"type": "completed", "message": ""}
    snap_kill_all = {e["sessionId"]: e["status"] for e in roster_snapshots[3]}
    assert snap_kill_all["sub-b"] == {"type": "cancelled", "message": ""}
    assert snap_kill_all["sub-c"] == {"type": "cancelled", "message": ""}
    snap_final = {e["sessionId"]: e["status"] for e in roster_snapshots[5]}
    assert snap_final["sub-d"] == {"type": "cancelled", "message": ""}


def test_sync_mcp_servers_strips_per_session_tokens_and_build_args_skips_auto_model(
    tmp_path: Path,
) -> None:
    """Verify sync_mcp_servers omits per-session KiroCrew tokens from global mcp_config.json."""
    mcp_cfg_path = tmp_path / "mcp_config.json"
    provider = AgyCliProvider(save_dir=tmp_path, mcp_config_path=mcp_cfg_path)
    provider.sync_mcp_servers(
        [
            {
                "name": "kirocrew-core",
                "command": "kirocrew",
                "args": ["mcp-core"],
                "env": {
                    "KIROCREW_STUB_SESSION_TOKEN": "tok-parent-123",
                    "KIROCREW_SESSION_KEY": "dashboard:chat-1",
                },
            },
            {
                "name": "rea",
                "command": "npx",
                "args": ["rea-agents"],
                "env": {
                    "JAVA_HOME": "/opt/jdk",
                    "KIROCREW_STUB_SESSION_TOKEN": "tok-parent-123",
                },
            },
        ]
    )
    written = json.loads(mcp_cfg_path.read_text(encoding="utf-8"))
    assert "env" not in written["mcpServers"]["kirocrew-core"]
    assert written["mcpServers"]["rea"]["env"] == {"JAVA_HOME": "/opt/jdk"}

    args_auto_model = build_agy_args(cwd="/workspace", prompt_arg="Hi", model="auto")
    assert "--model" not in args_auto_model


def test_load_kiro_agent_spec_extracts_prompt_readonly_and_ignores_auto_model(
    tmp_path: Path,
) -> None:
    """Verify load_kiro_agent_spec extracts prompt/read_only and ignores model='auto'."""
    from agy_acp.providers.agy_cli_provider import load_kiro_agent_spec

    agents_dir = tmp_path / ".kiro" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "egm--zoom-out.json").write_text(
        json.dumps(
            {
                "name": "zoom-out",
                "model": "auto",
                "prompt": "Follow zoom_out.md strictly.",
                "tools": ["fs_read", "grep", "glob", "code"],
                "mcpServers": {},
            }
        ),
        encoding="utf-8",
    )
    spec = load_kiro_agent_spec("egm--zoom-out", cwd=str(tmp_path), include_home=False)
    assert spec["model"] is None
    assert spec["prompt"] == "Follow zoom_out.md strictly."
    assert spec["read_only"] is True
