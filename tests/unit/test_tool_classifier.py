"""Unit tests for `agy_acp.logic.tool_classifier` classification, MCP translation, and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agy_acp.logic.tool_classifier import (
    FALLBACK_MODELS,
    PER_SESSION_MCP_ENV_KEYS,
    VALID_EFFORTS,
    PromptOutcome,
    QueuedSteer,
    ToolCallClassification,
    ToolCallEvent,
    _build_builtin_title,
    _dedupe_mcp_servers,
    _extract_diff_content,
    _extract_mcp_identity,
    _pairs_to_dict,
    _resolve_builtin_triple,
    _resolve_kiro_tool_identity,
    classify_tool_call_step,
    resolve_agy_binary,
    resolve_model_for_effort,
    translate_mcp_server,
)


def test_tool_call_event_post_init_normalizes_legacy_and_v3_fields() -> None:
    """Verify `ToolCallEvent` normalizes bidirectional v2/v3 fields and command extraction."""
    # Given a minimal execute ToolCallEvent with legacy v2 fields
    ev = ToolCallEvent(
        tool_call_id="tc_1",
        title="Run pytest",
        kind="execute",
        status="in_progress",
        raw_input={"CommandLine": "uv run pytest"},
        kiro_tool_name="execute_bash",
        mcp_server_name="",
    )

    # Then post-init populates v3 fields and command
    assert ev.acp_kind == "execute"
    assert ev.v2_tool_name == "execute_bash"
    assert ev.command == "uv run pytest"
    assert ev.kas_builtin_id == "run_command"

    # Given an MCP ToolCallEvent with v3 `mcp_server` and `v2_tool_name`
    mcp_ev = ToolCallEvent(
        tool_call_id="tc_2",
        title="@kirocrew-core/spawn_run",
        kind="other",
        status="in_progress",
        mcp_server="kirocrew-core",
        v2_tool_name="spawn_run",
        kiro_tool_id="@kirocrew-core/spawn_run",
    )

    # Then post-init mirrors legacy fields without setting kas_builtin_id
    assert mcp_ev.mcp_server_name == "kirocrew-core"
    assert mcp_ev.kiro_tool_name == "spawn_run"
    assert mcp_ev.mcp_tool == "spawn_run"
    assert mcp_ev.kas_builtin_id == ""


def test_prompt_outcome_and_queued_steer_dataclasses(tmp_path: Path) -> None:
    """Verify frozen `PromptOutcome` and `QueuedSteer` value objects."""
    outcome = PromptOutcome(
        conversation_id="conv-123",
        stop_reason="end_turn",
        response="done",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    assert outcome.conversation_id == "conv-123"
    assert outcome.raw_result == {}
    assert outcome.error == ""

    tmp_file = tmp_path / "img.png"
    steer = QueuedSteer(raw_message="stop", prompt="stop now", temp_files=[tmp_file])
    assert steer.raw_message == "stop"
    assert steer.temp_files == [tmp_file]


def test_extract_mcp_identity_variants_and_longest_known_server_match() -> None:
    """Verify `_extract_mcp_identity` handles `call_mcp_tool`, `mcp__`, and `mcp_` prefixes."""
    assert _extract_mcp_identity(
        "call_mcp_tool",
        {"ServerName": "kirocrew-core", "ToolName": "spawn_run"},
    ) == ("kirocrew-core", "spawn_run")

    assert _extract_mcp_identity("mcp__github__get_issue", {}) == ("github", "get_issue")

    # Longest known server match resolves underscore-containing server names
    assert _extract_mcp_identity(
        "mcp_my_custom_srv_do_work",
        {},
        known_mcp_servers=["my", "my_custom_srv"],
    ) == ("my_custom_srv", "do_work")

    assert _extract_mcp_identity("read_file", {"Path": "/tmp/a.txt"}) == ("", "")


def test_classify_tool_call_step_mcp_and_builtins() -> None:
    """Verify `classify_tool_call_step` and `_resolve_kiro_tool_identity` across tool families."""
    # 1. Special-cased kirocrew-core/spawn_run title
    spawn_cls = classify_tool_call_step(
        "tool_call",
        "call_mcp_tool",
        {"ServerName": "kirocrew-core", "ToolName": "spawn_run", "Arguments": {"tasks": []}},
    )
    assert isinstance(spawn_cls, ToolCallClassification)
    assert spawn_cls.title == "spawn_run"
    assert spawn_cls.kiro_tool_id == "@kirocrew-core/spawn_run"
    assert spawn_cls.mcp_server == "kirocrew-core"
    assert spawn_cls.mcp_tool == "spawn_run"
    assert spawn_cls.kas_builtin_id == ""

    # 2. Other MCP tool uses @server/tool title
    other_mcp = classify_tool_call_step(
        "tool_call",
        "mcp_kirocrew-core_memory_recall",
        {"query": "test"},
        known_mcp_servers=["kirocrew-core"],
    )
    assert other_mcp.title == "@kirocrew-core/memory_recall"
    assert _resolve_kiro_tool_identity(
        "tool_call",
        "mcp_kirocrew-core_memory_recall",
        {},
        known_mcp_servers=["kirocrew-core"],
    ) == ("memory_recall", "kirocrew-core", "@kirocrew-core/memory_recall")

    # 3. Built-in edit_file produces str_replace + diff_content + locations
    edit_cls = classify_tool_call_step(
        "tool_call",
        "edit_file",
        {"Path": "/repo/app.py", "Target": "old()", "Replacement": "new()"},
    )
    assert edit_cls.kas_builtin_id == "str_replace"
    assert edit_cls.v2_tool_name == "fs_write"
    assert edit_cls.acp_kind == "edit"
    assert edit_cls.title == "Editing /repo/app.py"
    assert edit_cls.locations == [{"path": "/repo/app.py"}]
    assert edit_cls.diff_content == [
        {"type": "diff", "path": "/repo/app.py", "oldText": "old()", "newText": "new()"}
    ]

    # 4. Built-in write_file produces fs_write diff with empty oldText
    write_cls = classify_tool_call_step(
        "tool_call",
        "write_file",
        {"Path": "/repo/new.py", "Content": "print(1)\n"},
    )
    assert write_cls.kas_builtin_id == "fs_write"
    assert write_cls.diff_content == [
        {"type": "diff", "path": "/repo/new.py", "oldText": "", "newText": "print(1)\n"}
    ]

    # 5. Built-in invoke_subagent extracts subagent_target
    sub_cls = classify_tool_call_step(
        "tool_call",
        "invoke_subagent",
        {"TypeName": "research", "Prompt": "check docs"},
    )
    assert sub_cls.kas_builtin_id == "invoke_sub_agent"
    assert sub_cls.v2_tool_name == "use_subagent"
    assert sub_cls.subagent_target == "research"


def test_diff_builtin_triple_and_title_helpers() -> None:
    """Verify `_extract_diff_content`, `_resolve_builtin_triple`, and `_build_builtin_title`."""
    chunks_diff = _extract_diff_content(
        "str_replace",
        "/repo/multi.py",
        {
            "ReplacementChunks": [
                {"TargetContent": "a", "ReplacementContent": "b"},
                {"TargetContent": "c", "ReplacementContent": "d"},
            ]
        },
    )
    assert len(chunks_diff) == 2
    assert chunks_diff[0]["oldText"] == "a"
    assert chunks_diff[1]["newText"] == "d"
    assert _extract_diff_content("str_replace", "", {"Target": "a", "Replacement": "b"}) == []

    assert _resolve_builtin_triple("custom_tool", {"CommandLine": "ls"}) == (
        "run_command",
        "execute_bash",
        "execute",
    )
    assert _resolve_builtin_triple("unknown_xyz", {}) == ("unknown_xyz", "unknown_xyz", "other")

    assert _build_builtin_title("read", "read_file", "/a/b.py", "", {}) == "Viewing /a/b.py"
    assert _build_builtin_title("delete", "delete_file", "/a/b.py", "", {}) == "Deleting /a/b.py"
    assert (
        _build_builtin_title("search", "grep_search", "", "", {"Query": "TODO"}) == "Searching TODO"
    )
    assert (
        _build_builtin_title("fetch", "web_fetch", "", "", {" url": "", "Url": "https://ex.com"})
        == "Fetching https://ex.com"
    )


def test_mcp_translation_dedupe_and_pairs_to_dict() -> None:
    """Verify `_pairs_to_dict`, `translate_mcp_server`, and `_dedupe_mcp_servers`."""
    assert _pairs_to_dict([{"name": "K", "value": "V"}, {"bad": 1}]) == {"K": "V"}
    assert _pairs_to_dict({"A": 1}) == {"A": "1"}
    assert _pairs_to_dict("invalid") == {}

    stdio_srv = translate_mcp_server(
        {
            "name": "core",
            "command": "node",
            "args": ["srv.js"],
            "env": [{"name": "PORT", "value": "8080"}],
            "disabledTools": ["dangerous"],
        }
    )
    assert stdio_srv == {
        "type": "stdio",
        "name": "core",
        "command": "node",
        "args": ["srv.js"],
        "env": {"PORT": "8080"},
        "disabledTools": ["dangerous"],
    }

    http_srv = translate_mcp_server(
        {
            "type": "http",
            "name": "remote",
            "url": "https://mcp.example.com",
            "headers": [{"name": "Authorization", "value": "Bearer tok"}],
        }
    )
    assert http_srv["type"] == "http"
    assert http_srv["headers"] == {"Authorization": "Bearer tok"}

    deduped = _dedupe_mcp_servers(
        [
            {"name": "core", "command": "first"},
            {"name": "core", "command": "second"},
            {"name": "", "command": "ignored"},
            {"name": "remote", "url": "https://mcp.example.com"},
        ]
    )
    assert [s["name"] for s in deduped] == ["core", "remote"]
    assert deduped[0]["command"] == "first"


def test_resolve_agy_binary_and_resolve_model_for_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `resolve_agy_binary` precedence and `resolve_model_for_effort` switching."""
    explicit_bin = tmp_path / "explicit-agy"
    explicit_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    assert resolve_agy_binary(explicit_bin) == str(explicit_bin)

    env_bin = tmp_path / "env-agy"
    env_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("AGY_BIN", str(env_bin))
    assert resolve_agy_binary(tmp_path / "missing") == str(env_bin)

    monkeypatch.delenv("AGY_BIN", raising=False)
    fake_home = tmp_path / "home"
    (fake_home / ".local" / "bin").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setattr("agy_acp.logic.tool_classifier.shutil.which", lambda _: "/usr/bin/agy")
    assert resolve_agy_binary() == "/usr/bin/agy"

    monkeypatch.setattr("agy_acp.logic.tool_classifier.shutil.which", lambda _: None)
    assert resolve_agy_binary() == "agy"

    # Model effort switching
    catalog: list[dict[str, Any]] = [
        {"modelId": "gemini-3.1-pro-low"},
        {"modelId": "gemini-3.1-pro-high"},
    ]
    assert resolve_model_for_effort("gemini-3.1-pro-high", "low", catalog) == "gemini-3.1-pro-low"
    assert (
        resolve_model_for_effort("gemini-3.1-pro-high", "medium", catalog) == "gemini-3.1-pro-high"
    )
    assert resolve_model_for_effort("claude-sonnet-4-6", "high", catalog) == "claude-sonnet-4-6"
    assert "high" in VALID_EFFORTS
    assert "KIROCREW_SESSION_KEY" in PER_SESSION_MCP_ENV_KEYS
    assert len(FALLBACK_MODELS) >= 1


def test_kas_tool_classification_table_and_task_commands() -> None:
    """Verify full KAS built-in ID table, subagent targets, MCP prefix matching, and task tools."""
    cases: list[tuple[str, dict[str, Any], str, str, str]] = [
        ("run_command", {"CommandLine": "pytest -q"}, "run_command", "execute_bash", "execute"),
        ("execute_command", {"command": "echo ok"}, "run_command", "execute_bash", "execute"),
        ("read_file", {"Path": "/a.py"}, "read_file", "fs_read", "read"),
        ("view_file", {"AbsolutePath": "/b.py"}, "read_file", "fs_read", "read"),
        ("view_file_outline", {"AbsolutePath": "/c.py"}, "read_file", "fs_read", "read"),
        ("view_code_item", {"File": "/d.py"}, "read_file", "fs_read", "read"),
        ("list_dir", {"DirectoryPath": "/src"}, "list_directory", "fs_read", "read"),
        ("list_directory", {"path": "/src"}, "list_directory", "fs_read", "read"),
        ("write_file", {"Path": "/w.py", "Content": "x = 1\n"}, "fs_write", "fs_write", "edit"),
        ("write_to_file", {"TargetFile": "/w2.py"}, "fs_write", "fs_write", "edit"),
        ("create_file", {"path": "/w3.py"}, "fs_write", "fs_write", "edit"),
        (
            "edit_file",
            {"Path": "/e.py", "Target": "a = 1", "Replacement": "a = 2"},
            "str_replace",
            "fs_write",
            "edit",
        ),
        ("replace_file_content", {"TargetFile": "/e2.py"}, "str_replace", "fs_write", "edit"),
        ("multi_replace_file_content", {"TargetFile": "/e3.py"}, "str_replace", "fs_write", "edit"),
        ("delete_file", {"Path": "/gone.py"}, "delete_file", "fs_write", "delete"),
        ("remove_file", {"path": "/gone2.py"}, "delete_file", "fs_write", "delete"),
        ("grep_search", {"Query": "TODO"}, "grep_search", "grep", "search"),
        ("grep", {"Pattern": "FIXME"}, "grep_search", "grep", "search"),
        ("find_by_name", {"Pattern": "*.py"}, "file_search", "glob", "search"),
        ("glob", {"pattern": "**/*.ts"}, "file_search", "glob", "search"),
        ("file_search", {"query": "main"}, "file_search", "glob", "search"),
        ("search_web", {"query": "acp spec"}, "remote_web_search", "web_search", "fetch"),
        ("read_url_content", {"Url": "https://example.com"}, "web_fetch", "web_fetch", "fetch"),
        ("web_fetch", {"url": "https://example.com"}, "web_fetch", "web_fetch", "fetch"),
        (
            "invoke_subagent",
            {"TypeName": "research", "Role": "Auditor"},
            "invoke_sub_agent",
            "use_subagent",
            "other",
        ),
        ("todo_list", {"todos": []}, "todo_list", "todo_list", "other"),
        ("task_boundary", {"TaskName": "Build"}, "todo_list", "todo_list", "other"),
    ]
    for tool_name, raw_args, expected_kas, expected_v2, expected_kind in cases:
        cls = classify_tool_call_step("tool", tool_name, raw_args)
        assert cls.kas_builtin_id == expected_kas, f"{tool_name} kas_builtin_id"
        assert cls.v2_tool_name == expected_v2, f"{tool_name} v2_tool_name"
        assert cls.acp_kind == expected_kind, f"{tool_name} acp_kind"
        assert cls.mcp_server == ""
        assert cls.mcp_tool == ""

    sub_by_type = classify_tool_call_step(
        "tool", "invoke_subagent", {"TypeName": "research", "Role": "Codebase Researcher"}
    )
    assert sub_by_type.subagent_target == "research"
    sub_by_agent = classify_tool_call_step(
        "tool", "invoke_subagent", {"AgentName": "reviewer", "Role": "Reviewer"}
    )
    assert sub_by_agent.subagent_target == "reviewer"
    sub_by_role = classify_tool_call_step("tool", "invoke_subagent", {"Role": "Debugger"})
    assert sub_by_role.subagent_target == "Debugger"

    grep_cls = classify_tool_call_step(
        "tool",
        "grep_search",
        {"Query": "secret", "SearchDirectory": "/repo/sub"},
    )
    assert grep_cls.locations == [{"path": "/repo/sub"}]

    underscored = classify_tool_call_step(
        "tool",
        "mcp_my_org_db_run_query",
        {"sql": "SELECT 1"},
        known_mcp_servers=["my", "my_org_db"],
    )
    assert underscored.mcp_server == "my_org_db"
    assert underscored.mcp_tool == "run_query"
    assert underscored.kiro_tool_id == "@my_org_db/run_query"

    cls_list = classify_tool_call_step("tool", "manage_task", {"Action": "list"})
    assert cls_list.kas_builtin_id == "run_command"
    assert cls_list.v2_tool_name == "execute_bash"
    assert cls_list.acp_kind == "execute"
    assert cls_list.command == "manage_task list"
    assert cls_list.title == "Running `manage_task list`"

    cls_status = classify_tool_call_step(
        "tool",
        "manage_task",
        {"Action": "status", "TaskId": "task-42"},
    )
    assert cls_status.command == "manage_task status task-42"
    assert cls_status.title == "Running `manage_task status task-42`"

    cls_input = classify_tool_call_step(
        "tool",
        "send_command_input",
        {"CommandId": "cmd-9", "Input": "yes\n"},
    )
    assert cls_input.kas_builtin_id == "run_command"
    assert cls_input.v2_tool_name == "execute_bash"
    assert cls_input.acp_kind == "execute"
    assert cls_input.command == "send_command_input cmd-9"
