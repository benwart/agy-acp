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
