"""Unit tests for KiroCrew 0.14.0-rc.1 KAS/ACP compatibility updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agy_acp.logic.acp_dispatcher import (
    AcpDispatcher,
    _matches_agent_tool_list,
    _to_normalized_tool_event,
)
from agy_acp.logic.custom_agents import CustomAgentSpec
from agy_acp.logic.session_manager import (
    SessionManager,
    _extract_session_mcp_env,
    _session_has_kirocrew_spawn_run,
)
from agy_acp.logic.tool_classifier import ToolCallEvent
from agy_acp.logic.wire import V3KasWireAdapter
from agy_acp.providers.state_store import SqliteStateStoreProvider


def _make_mcp_norm(server: str = "kirocrew-core", tool: str = "spawn_run") -> Any:
    return _to_normalized_tool_event(
        ToolCallEvent(
            tool_call_id="tc-mcp-1",
            title=f"{server}/{tool}",
            kind="other",
            status="pending",
            raw_input={"task": "check"},
            kiro_tool_name=tool,
            kiro_tool_id=f"@{server}/{tool}",
            mcp_server_name=server,
        )
    )


def _make_unresolved_mcp_wrapper_norm(tool_id: str) -> Any:
    return _to_normalized_tool_event(
        ToolCallEvent(
            tool_call_id="tc-wrap-1",
            title=tool_id,
            kind="other",
            status="pending",
            raw_input={},
            kiro_tool_name=tool_id,
            kiro_tool_id=tool_id,
            mcp_server_name="",
        )
    )


def _make_builtin_norm(
    kas_id: str = "read_file",
    v2_name: str = "fs_read",
    kind: str = "read",
) -> Any:
    return _to_normalized_tool_event(
        ToolCallEvent(
            tool_call_id="tc-builtin-1",
            title=kas_id,
            kind=kind,
            status="pending",
            raw_input={"Path": "/repo/a.py"},
            kiro_tool_name=v2_name,
            kiro_tool_id=kas_id,
            mcp_server_name="",
        )
    )


@pytest.mark.parametrize("pattern", ["@mcp", "@*"])
def test_mcp_wildcard_exclusion_blocks_resolved_and_unresolved_mcp_but_keeps_builtins(
    tmp_path: Path,
    pattern: str,
) -> None:
    """`excludedTools: ['@mcp', '@powers']` / `['@*']` denies all MCP calls, never built-ins."""
    store = SqliteStateStoreProvider(tmp_path / "state.db")
    adapter = V3KasWireAdapter()
    mgr = SessionManager(state_store=store, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=None,
        session_manager=mgr,
        wire_adapter=adapter,
    )
    spec = CustomAgentSpec(
        agent_id="no-mcp",
        name="no-mcp",
        tools=("*",),
        excluded_tools=(pattern, "@powers"),
        allowed_tools=("fs_read",),
    )
    session: dict[str, Any] = {
        "sessionId": "s-1",
        "cwd": str(tmp_path),
        "agentTools": ["*"],
        "agentExcludedTools": [pattern, "@powers"],
        "agentAllowedTools": ["fs_read"],
    }

    assert (
        dispatcher._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=_make_mcp_norm("kirocrew-core", "spawn_run"),
            session_mode="auto",
        )
        == "deny"
    )
    assert (
        dispatcher._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=_make_mcp_norm("github", "list_prs"),
            session_mode="auto",
        )
        == "deny"
    )
    for wrapper_id in ("call_mcp_tool", "mcp_resource", "list_resources", "read_resource"):
        assert (
            dispatcher._evaluate_tool_permission(
                session=session,
                spec=spec,
                norm=_make_unresolved_mcp_wrapper_norm(wrapper_id),
                session_mode="auto",
            )
            == "deny"
        )

    # Built-in tools remain unaffected by `@mcp` / `@*` exclusions.
    assert (
        dispatcher._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=_make_builtin_norm("read_file", "fs_read", "read"),
            session_mode="auto",
        )
        == "allow"
    )
    assert (
        dispatcher._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=_make_builtin_norm("run_command", "execute_bash", "execute"),
            session_mode="auto",
        )
        == "ask"
    )


@pytest.mark.parametrize("pattern", ["@mcp", "@*"])
def test_mcp_wildcard_allowlist_matches_resolved_mcp_only_never_unresolved_wrappers(
    pattern: str,
) -> None:
    """`@mcp` / `@*` in `tools`/`allowedTools` matches resolved MCP only, not wrappers."""
    assert _matches_agent_tool_list(
        (pattern,), _make_mcp_norm("docs", "search"), for_allowlist=True
    )
    assert not _matches_agent_tool_list(
        (pattern,), _make_builtin_norm("read_file", "fs_read", "read"), for_allowlist=True
    )
    for wrapper_id in ("call_mcp_tool", "mcp_resource", "list_resources", "read_resource"):
        norm = _make_unresolved_mcp_wrapper_norm(wrapper_id)
        assert not _matches_agent_tool_list((pattern,), norm, for_allowlist=True)
        assert not _matches_agent_tool_list((pattern,), norm, for_allowlist=False)
        assert _matches_agent_tool_list((pattern,), norm, for_allowlist=False, for_exclusions=True)


@pytest.mark.parametrize(
    ("excluded_tools", "agent_tools", "expected"),
    [
        (["@mcp", "@powers"], None, False),
        (["@*"], None, False),
        (["@mcp"], ["*"], False),
        ([], ["@mcp"], True),
        ([], ["@*"], True),
        ([], ["read_file"], False),
    ],
)
def test_session_has_kirocrew_spawn_run_honors_kas_mcp_wildcard_tags(
    excluded_tools: list[str],
    agent_tools: list[str] | None,
    expected: bool,
) -> None:
    """`KIROCREW_ROUTE_SPAWN_RUN` respects `@mcp` and `@*` in `excludedTools` and `agentTools`."""
    session: dict[str, Any] = {
        "sessionId": "s-2",
        "mcp_servers": [{"name": "kirocrew-core", "command": "kirocrew"}],
        "agentExcludedTools": excluded_tools,
    }
    if agent_tools is not None:
        session["agentTools"] = agent_tools

    assert _session_has_kirocrew_spawn_run(session) is expected
    env = _extract_session_mcp_env(session)
    assert env["KIROCREW_ROUTE_SPAWN_RUN"] == ("1" if expected else "")


async def test_initialize_accepts_kirocrew_0_14_0_rc1_client_capabilities_and_settings(
    tmp_path: Path,
) -> None:
    """`initialize` accepts `checksAutoApprovals`, `codeIntelligence`, and `userMemoryOptIn`."""
    notifications: list[dict[str, Any]] = []

    async def _sink(frame: dict[str, Any]) -> None:
        notifications.append(frame)

    store = SqliteStateStoreProvider(tmp_path / "init.db")
    adapter = V3KasWireAdapter()
    mgr = SessionManager(state_store=store, wire_adapter=adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        wire_adapter=adapter,
    )

    init_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientInfo": {"name": "kirocrew", "version": "0.14.0rc1"},
                    "clientCapabilities": {
                        "_meta": {
                            "kiro": {
                                "openExternalUrl": True,
                                "secretStorage": True,
                                "checksAutoApprovals": True,
                                "settings": {
                                    "codeIntelligence": {"enabled": False},
                                    "userMemoryOptIn": {"enabled": False},
                                    "toolSearch": {
                                        "enabled": True,
                                        "neverDefer": ["kirocrew-core/spawn_run"],
                                    },
                                },
                            }
                        }
                    },
                },
            }
        )
    )
    assert init_resp is not None and "result" in init_resp

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert new_resp is not None and "result" in new_resp
    sid = new_resp["result"]["sessionId"]
    stored = store.get_session(sid)
    assert stored is not None
    settings = stored["meta"]["kiro"]["settings"]
    assert settings["codeIntelligence"] == {"enabled": False}
    assert settings["userMemoryOptIn"] == {"enabled": False}
    await mgr.close()
