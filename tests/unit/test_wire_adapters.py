"""Unit tests for EngineWireAdapter Strategy (V2KiroWireAdapter & V3KasWireAdapter)."""

from __future__ import annotations

import pytest

from agy_acp.logic.custom_agents import CustomAgentSpec
from agy_acp.logic.wire import (
    McpServerStatus,
    McpToolDef,
    NormalizedToolEvent,
    V2KiroWireAdapter,
    V3KasWireAdapter,
    WireTurnTelemetry,
    create_wire_adapter,
    normalize_agy_tool_event,
)


def test_create_wire_adapter_factory_dispatch_and_rejection() -> None:
    """Verify factory selects V2 or V3 strategy and rejects unknown engine identifiers."""
    for v2_key in (None, "", "default", "v2", "kiro", "V2"):
        adapter = create_wire_adapter(v2_key)
        assert isinstance(adapter, V2KiroWireAdapter)
        assert adapter.engine_id == "v2"
        assert adapter.effort_config_id == "reasoning_effort"
        assert adapter.supported_effort_levels == ("low", "medium", "high")

    for v3_key in ("v3", "kas", "KAS", "V3"):
        adapter = create_wire_adapter(v3_key)
        assert isinstance(adapter, V3KasWireAdapter)
        assert adapter.engine_id == "v3"
        assert adapter.effort_config_id == "effortLevel"
        assert adapter.supported_effort_levels == ("low", "medium", "high", "xhigh", "max")

    with pytest.raises(ValueError, match="Unsupported"):
        create_wire_adapter("v99")


def test_initialize_result_protocol_version_and_kas_capabilities() -> None:
    """Verify integer protocolVersion: 1 and v3 _meta.kiro extensionMethods."""
    v2 = V2KiroWireAdapter().build_initialize_result(version="1.3.0")
    v3 = V3KasWireAdapter().build_initialize_result(version="1.3.0")

    assert v2["protocolVersion"] == 1
    assert type(v2["protocolVersion"]) is int
    assert "_meta" not in v2["agentCapabilities"]

    assert v3["protocolVersion"] == 1
    assert type(v3["protocolVersion"]) is int
    kiro_caps = v3["agentCapabilities"]["_meta"]["kiro"]
    assert kiro_caps == {
        "extensionMethods": [
            "_kiro/session/compact",
            "_kiro/session/delete",
            "_kiro/mcp/resetServer",
        ],
        "checkpoints": False,
    }


def test_available_modes_and_config_options_for_v2_and_v3() -> None:
    """Verify v3 resource.source.origin metadata and concrete model + effort configOptions."""
    custom_agents = [
        CustomAgentSpec(
            agent_id="kiro_default-crewmate",
            name="Default",
            description="Client projected default",
            origin="client",
        ),
        CustomAgentSpec(
            agent_id="local-helper",
            name="Local Helper",
            description="User disk agent",
            origin="user",
        ),
    ]
    v2 = V2KiroWireAdapter()
    v3 = V3KasWireAdapter()

    modes_v2 = v2.build_available_modes(custom_agents=custom_agents)
    assert all("_meta" not in m for m in modes_v2)

    modes_v3 = v3.build_available_modes(custom_agents=custom_agents)
    by_id = {m["id"]: m for m in modes_v3}
    assert (
        by_id["kiro_default-crewmate"]["_meta"]["kiro"]["resource"]["source"]["origin"] == "client"
    )
    assert by_id["local-helper"]["_meta"]["kiro"]["resource"]["source"]["origin"] == "user"
    assert by_id["auto"]["_meta"]["kiro"]["resource"]["source"]["origin"] == "bundled"

    models = [
        {"modelId": "gemini-3.7-pro", "name": "Gemini 3.7 Pro", "description": "Pro"},
        {"modelId": "gemini-3.8-flash", "name": "Gemini 3.8 Flash", "description": "Flash"},
    ]
    # Current model "auto" is not in models -> must resolve to concrete first modelId
    opts_v2 = v2.build_config_options(
        models=models,
        current_model="auto",
        current_mode="auto",
        current_effort="high",
    )
    opts_v3 = v3.build_config_options(
        models=models,
        current_model="auto",
        current_mode="auto",
        current_effort="xhigh",
    )

    v2_by_id = {o["id"]: o for o in opts_v2}
    assert v2_by_id["model"]["category"] == "model"
    assert v2_by_id["model"]["currentValue"] == "gemini-3.7-pro"
    assert v2_by_id["reasoning_effort"]["category"] == "thought_level"
    assert [x["value"] for x in v2_by_id["reasoning_effort"]["options"]] == [
        "low",
        "medium",
        "high",
    ]

    v3_by_id = {o["id"]: o for o in opts_v3}
    assert v3_by_id["model"]["category"] == "model"
    assert v3_by_id["model"]["currentValue"] == "gemini-3.7-pro"
    assert v3_by_id["effortLevel"]["category"] == "thought_level"
    assert v3_by_id["effortLevel"]["currentValue"] == "xhigh"
    assert [x["value"] for x in v3_by_id["effortLevel"]["options"]] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


def test_mcp_readiness_notifications_v2_vs_v3() -> None:
    """Verify v2 per-server MCP notifications vs v3 _kiro/mcp/status + _kiro/tools/didChange."""
    servers = [
        McpServerStatus(
            name="kirocrew-core",
            display_name="KiroCrew Core",
            status="connected",
            tools=(
                McpToolDef(name="spawn_run", description="Spawn worker", disabled=False),
                McpToolDef(name="hidden_tool", description="Disabled", disabled=True),
            ),
        ),
        McpServerStatus(
            name="broken-srv",
            display_name="Broken",
            status="failed",
            error="connection refused",
        ),
        McpServerStatus(
            name="oauth-srv",
            display_name="OAuth",
            status="authRequired",
            auth_url="https://auth.example.com/authorize",
            failed_authorization=True,
            error="401 Unauthorized",
        ),
    ]

    v2_frames = V2KiroWireAdapter().build_mcp_readiness_notifications("sess-1", servers)
    methods_v2 = [f["method"] for f in v2_frames]
    assert methods_v2 == [
        "_kiro.dev/mcp/server_initialized",
        "_kiro.dev/mcp/server_init_failure",
        "_kiro.dev/mcp/oauth_request",
    ]
    assert v2_frames[2]["params"]["oauthUrl"] == "https://auth.example.com/authorize"

    v3_frames = V3KasWireAdapter().build_mcp_readiness_notifications("sess-1", servers)
    assert [f["method"] for f in v3_frames] == ["_kiro/mcp/status", "_kiro/tools/didChange"]

    status_entries = {s["name"]: s for s in v3_frames[0]["params"]["servers"]}
    core_entry = status_entries["kirocrew-core"]
    assert core_entry["status"] == "connected"
    assert core_entry["_meta"] == {
        "kiro": {
            "resource": {
                "resourceType": "mcpServer",
                "source": {"origin": "client"},
            }
        }
    }
    assert core_entry["tools"] == [
        {"name": "spawn_run", "description": "Spawn worker", "disabled": False},
        {"name": "hidden_tool", "description": "Disabled", "disabled": True},
    ]

    broken_entry = status_entries["broken-srv"]
    assert broken_entry["status"] == "failed"
    assert broken_entry["errorMessage"] == "connection refused"
    assert broken_entry["error"] == "connection refused"

    oauth_entry = status_entries["oauth-srv"]
    assert oauth_entry["failedAuthorization"] is True
    assert oauth_entry["authUrl"] == "https://auth.example.com/authorize"

    did_change = v3_frames[1]["params"]
    assert did_change["sessionId"] == "sess-1"
    # Enabled tool exposed as @kirocrew-core/spawn_run; disabled hidden_tool excluded from tags
    assert did_change["tags"] == [
        {"source": "mcp", "tag": "@kirocrew-core/spawn_run"},
    ]
    assert did_change["initial"]["tags"] == did_change["tags"]
    assert did_change["added"]["tags"] == did_change["tags"]


def test_mode_switch_notifications_v2_vs_v3() -> None:
    """Verify v2 emits current_mode_update + _kiro.dev/agent/switched while v3 omits switched."""
    v2_frames = V2KiroWireAdapter().build_mode_switched_notifications(
        "sess-1",
        "reviewer",
        mode_name="Reviewer",
    )
    assert [f["method"] for f in v2_frames] == ["session/update", "_kiro.dev/agent/switched"]

    v3_frames = V3KasWireAdapter().build_mode_switched_notifications(
        "sess-1",
        "reviewer",
        mode_name="Reviewer",
    )
    assert len(v3_frames) == 1
    assert v3_frames[0]["method"] == "session/update"
    assert v3_frames[0]["params"]["update"] == {
        "sessionUpdate": "current_mode_update",
        "currentModeId": "reviewer",
    }


def test_tool_call_and_permission_envelopes_across_builtin_and_mcp() -> None:
    """Verify built-in (shell, delete, subagent, todo_list) and MCP envelopes for v2 and v3."""
    v2 = V2KiroWireAdapter()
    v3 = V3KasWireAdapter()

    # 1. Shell built-in
    shell_ev = NormalizedToolEvent(
        tool_call_id="tc-shell",
        title="Run pytest",
        kind="execute",
        kas_tool_id="run_command",
        kiro_v2_tool_name="execute_bash",
        raw_input={"command": "pytest -q"},
        command="pytest -q",
    )
    v3_shell_tc = v3.build_tool_call_update("sess-1", shell_ev)
    assert v3_shell_tc["params"]["update"]["kind"] == "execute"
    assert v3_shell_tc["params"]["update"]["_meta"] == {"kiro": {"toolOrigin": "builtin"}}
    assert "serverName" not in v3_shell_tc["params"]["update"]["_meta"]["kiro"]

    v3_shell_perm = v3.build_permission_request_params("sess-1", shell_ev)
    assert "kind" not in v3_shell_perm["toolCall"]
    assert v3_shell_perm["_meta"] == {"kiro": {"toolId": "run_command", "command": "pytest -q"}}
    assert [o["optionId"] for o in v3_shell_perm["options"]] == [
        "allow_once",
        "allow_always",
        "reject_once",
    ]
    assert [o["kind"] for o in v3_shell_perm["options"]] == [
        "allow_once",
        "allow_always",
        "reject_once",
    ]

    v2_shell_tc = v2.build_tool_call_update("sess-1", shell_ev)
    assert v2_shell_tc["params"]["update"]["_meta"] == {
        "kiro": {"toolName": "execute_bash", "mcpServerName": ""}
    }
    v2_shell_perm = v2.build_permission_request_params("sess-1", shell_ev)
    assert v2_shell_perm["toolCall"]["kind"] == "execute"
    assert v2_shell_perm["_meta"]["kiro"]["toolId"] == "execute_bash"

    # 2. Delete built-in (`delete_file` -> kind="delete", no command)
    del_ev = NormalizedToolEvent(
        tool_call_id="tc-del",
        title="Delete temp file",
        kind="delete",
        kas_tool_id="delete_file",
        kiro_v2_tool_name="fs_write",
        raw_input={"path": "old.txt"},
        locations=({"path": "/repo/old.txt"},),
    )
    v3_del_perm = v3.build_permission_request_params("sess-1", del_ev)
    assert v3_del_perm["_meta"] == {"kiro": {"toolId": "delete_file"}}

    # 3. Subagent built-in (`invoke_sub_agent` -> consent block, NEVER `command`)
    sub_ev = NormalizedToolEvent(
        tool_call_id="tc-sub",
        title="Invoke researcher",
        kind="other",
        kas_tool_id="invoke_sub_agent",
        kiro_v2_tool_name="use_subagent",
        raw_input={"agent": "researcher"},
        subagent_target="researcher",
    )
    v3_sub_perm = v3.build_permission_request_params("sess-1", sub_ev)
    assert v3_sub_perm["_meta"] == {
        "kiro": {
            "toolId": "invoke_sub_agent",
            "consent": {"capability": "subagent", "resource": "researcher"},
        }
    }
    assert "command" not in v3_sub_perm["_meta"]["kiro"]

    # 4. Todo list built-in (`todo_list` -> preserves `toolName: "todo_list"`)
    todo_ev = NormalizedToolEvent(
        tool_call_id="tc-todo",
        title="Update plan",
        kind="other",
        kas_tool_id="todo_list",
        kiro_v2_tool_name="todo_list",
        raw_input={"todos": []},
    )
    v3_todo_tc = v3.build_tool_call_update("sess-1", todo_ev)
    assert v3_todo_tc["params"]["update"]["_meta"] == {
        "kiro": {"toolOrigin": "builtin", "toolName": "todo_list"}
    }

    # 5. MCP tool call
    mcp_ev = NormalizedToolEvent(
        tool_call_id="tc-mcp",
        title="Spawn run",
        kind="other",
        is_mcp=True,
        mcp_server="kirocrew-core",
        mcp_tool="spawn_run",
        raw_input={"tasks": []},
    )
    v3_mcp_tc = v3.build_tool_call_update("sess-1", mcp_ev)
    assert v3_mcp_tc["params"]["update"]["_meta"] == {
        "kiro": {
            "serverName": "kirocrew-core",
            "toolName": "spawn_run",
            "toolOrigin": "client",
        }
    }
    v3_mcp_perm = v3.build_permission_request_params("sess-1", mcp_ev)
    assert "kind" not in v3_mcp_perm["toolCall"]
    assert v3_mcp_perm["_meta"] == {
        "kiro": {
            "toolId": "mcp_kirocrew-core_spawn_run",
            "mcpTool": {
                "identity": {
                    "serverName": "kirocrew-core",
                    "toolName": "spawn_run",
                }
            },
        }
    }

    v2_mcp_perm = v2.build_permission_request_params("sess-1", mcp_ev)
    assert v2_mcp_perm["_meta"] == {"kiro": {"toolId": "@kirocrew-core/spawn_run"}}


def test_turn_start_ttft_telemetry_steering_compaction_and_subagents() -> None:
    """Verify v3 session_info_update & agent-subtask frames vs v2 _kiro.dev/* notifications."""
    v2 = V2KiroWireAdapter()
    v3 = V3KasWireAdapter()

    # Turn start
    assert v2.build_turn_start_notifications("sess-1", "msg-42") == []
    v3_start = v3.build_turn_start_notifications("sess-1", "msg-42")
    assert v3_start == [
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {
                    "sessionUpdate": "session_info_update",
                    "_meta": {
                        "kiro": {
                            "kind": "user_message_id_assigned",
                            "userMessageId": "msg-42",
                        }
                    },
                },
            },
        }
    ]

    # First-chunk TTFT & replay stamping (no non-standard ttftMs sibling key)
    v3_chunk = v3.build_agent_message_chunk(
        "sess-1",
        "Hello",
        prompt_received_at_ms=1_700_000_000_000,
        first_chunk_emitted_at_ms=1_700_000_000_250,
    )
    chunk_kiro = v3_chunk["params"]["update"]["_meta"]["kiro"]
    assert chunk_kiro == {
        "ttft": {
            "promptReceivedAt": 1_700_000_000_000,
            "firstChunkEmittedAt": 1_700_000_000_250,
        }
    }
    assert "ttftMs" not in chunk_kiro

    v3_thought = v3.build_agent_thought_chunk(
        "sess-1",
        "Thinking...",
        prompt_received_at_ms=1_700_000_000_000,
        first_chunk_emitted_at_ms=1_700_000_000_120,
    )
    assert v3_thought["params"]["update"]["_meta"]["kiro"] == {
        "ttft": {
            "promptReceivedAt": 1_700_000_000_000,
            "firstChunkEmittedAt": 1_700_000_000_120,
        }
    }

    # Negative delta (clock skew / invalid order) omits ttft
    skew_chunk = v3.build_agent_message_chunk(
        "sess-1",
        "Skew",
        prompt_received_at_ms=1_700_000_000_500,
        first_chunk_emitted_at_ms=1_700_000_000_250,
    )
    assert "_meta" not in skew_chunk["params"]["update"]

    replay_chunk = v3.build_agent_message_chunk("sess-1", "Old turn", replay=True)
    assert replay_chunk["params"]["update"]["_meta"]["kiro"]["replay"] is True

    # Turn telemetry
    telemetry = WireTurnTelemetry(
        context_used=40_000,
        context_size=200_000,
        context_usage_percentage=20.0,
        delta_credits=1.25,
        metadata_payload={"contextUsagePercentage": 20.0, "tokens": {"total": 40_000}},
        rate_limited_message="Model bucket exhausted",
    )
    v2_tel = v2.build_turn_telemetry_notifications("sess-1", telemetry)
    assert [f["method"] for f in v2_tel] == [
        "session/update",
        "_kiro.dev/metadata",
        "_kiro.dev/error/rate_limit",
    ]

    v3_tel = v3.build_turn_telemetry_notifications("sess-1", telemetry)
    assert [f["method"] for f in v3_tel] == [
        "session/update",
        "session/update",
        "session/update",
    ]
    assert all(not f["method"].startswith("_kiro.dev/") for f in v3_tel)
    assert v3_tel[0]["params"]["update"] == {
        "sessionUpdate": "session_info_update",
        "_meta": {
            "kiro": {
                "kind": "context_usage",
                "usagePercentage": 20.0,
            }
        },
    }
    assert v3_tel[1]["params"]["update"] == {
        "sessionUpdate": "usage_update",
        "used": 40_000,
        "size": 200_000,
    }
    assert "cost" not in v3_tel[1]["params"]["update"]
    assert v3_tel[2]["params"]["update"]["_meta"]["kiro"] == {
        "kind": "turn_completion",
        "promptTurnSummaries": [{"unit": "credit", "usage": 1.25}],
    }

    # Steering
    assert (
        v2.build_steering_notification("sess-1", "consumed", "Focus on tests")["params"]["update"][
            "sessionUpdate"
        ]
        == "steering_consumed"
    )
    assert v3.build_steering_notification("sess-1", "consumed", "Focus on tests")["params"][
        "update"
    ] == {
        "sessionUpdate": "session_info_update",
        "_meta": {
            "kiro": {
                "kind": "steering_injected",
                "content": "Focus on tests",
            }
        },
    }
    assert (
        v3.build_steering_notification("sess-1", "discarded", "Dropped")["params"]["update"][
            "_meta"
        ]["kiro"]["kind"]
        == "steering_cleared"
    )

    # Compaction
    assert (
        v2.build_compaction_notification("sess-1", "completed", summary="Short summary")["method"]
        == "_kiro.dev/compaction/status"
    )
    assert v3.build_compaction_notification("sess-1", "completed", summary="Short summary")[
        "params"
    ]["update"] == {
        "sessionUpdate": "session_info_update",
        "_meta": {
            "kiro": {
                "kind": "summarization_completed",
                "conversationSummary": "Short summary",
            }
        },
    }
    failed_comp = v3.build_compaction_notification("sess-1", "failed", error="No steps")
    assert failed_comp["params"]["update"]["_meta"]["kiro"]["kind"] == "summarization_failed"
    assert failed_comp["params"]["update"]["_meta"]["kiro"]["error"] == "No steps"

    # Native subagents
    subs = [{"sessionId": "child-1", "role": "Researcher", "status": "running"}]
    v2_subs = v2.build_subagent_notifications("sess-1", subs)
    assert len(v2_subs) == 1
    assert v2_subs[0]["method"] == "_kiro.dev/subagent/list_update"

    v3_subs = v3.build_subagent_notifications("sess-1", subs)
    assert len(v3_subs) == 1
    assert v3_subs[0]["method"] == "session/update"
    sub_update = v3_subs[0]["params"]["update"]
    assert sub_update["status"] == "in_progress"
    assert sub_update["_meta"] == {
        "kiro": {
            "kind": "agent-subtask",
            "agentSubtaskId": "child-1",
        }
    }


def test_normalize_agy_tool_event_mappings() -> None:
    """Verify Antigravity CLI step names map accurately to NormalizedToolEvent."""
    append_ev = normalize_agy_tool_event(
        tool_call_id="tc-1",
        step_name="write_file",
        title="Append log",
        raw_input={"Path": "/ws/log.txt", "Append": True},
    )
    assert append_ev.kind == "edit"
    assert append_ev.kas_tool_id == "fs_append"

    mcp_ev = normalize_agy_tool_event(
        tool_call_id="tc-2",
        step_name="call_mcp_tool",
        title="",
        raw_input={
            "ServerName": "kirocrew-core",
            "ToolName": "memory_recall",
            "Arguments": {"query": "kas"},
        },
    )
    assert mcp_ev.is_mcp is True
    assert mcp_ev.mcp_server == "kirocrew-core"
    assert mcp_ev.mcp_tool == "memory_recall"
    assert mcp_ev.raw_input == {"query": "kas"}

    # Longest-prefix known_mcp_servers disambiguation for underscore server names
    underscored_ev = normalize_agy_tool_event(
        tool_call_id="tc-3",
        step_name="mcp_my_sql_db_run_query",
        title="",
        raw_input={"sql": "SELECT 1"},
        known_mcp_servers=("my", "my_sql_db", "my_sql"),
    )
    assert underscored_ev.is_mcp is True
    assert underscored_ev.mcp_server == "my_sql_db"
    assert underscored_ev.mcp_tool == "run_query"
    assert underscored_ev.title == "@my_sql_db/run_query"

    # Fallback when known_mcp_servers is empty or unmatched
    fallback_ev = normalize_agy_tool_event(
        tool_call_id="tc-4",
        step_name="mcp_github_create_issue",
        title="",
        raw_input={"title": "Bug"},
        known_mcp_servers=("gitlab",),
    )
    assert fallback_ev.is_mcp is True
    assert fallback_ev.mcp_server == "github"
    assert fallback_ev.mcp_tool == "create_issue"

    # EngineWireAdapter.normalize_tool_event static helper alias
    via_adapter = V3KasWireAdapter.normalize_tool_event(
        tool_call_id="tc-5",
        step_name="mcp_corp_auth_verify_token",
        title="",
        known_mcp_servers=["corp_auth"],
    )
    assert via_adapter.is_mcp is True
    assert via_adapter.mcp_server == "corp_auth"
    assert via_adapter.mcp_tool == "verify_token"
