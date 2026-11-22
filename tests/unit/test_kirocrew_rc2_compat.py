"""Unit and contract tests for KiroCrew 0.13.1-rc.2 compatibility and security parity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agy_acp.logic.acp_dispatcher import AcpDispatcher
from agy_acp.logic.mcp_catalog_prober import McpCatalogProber, kas_mcp_tokens_secret_key
from agy_acp.logic.session_manager import (
    SessionManager,
    _build_system_prompt_append,
    _compute_disallowed_native_tools,
    _session_has_kirocrew_spawn_run,
)
from agy_acp.logic.tool_classifier import ToolCallEvent, classify_tool_call_step
from agy_acp.logic.wire import McpServerStatus, V3KasWireAdapter
from agy_acp.providers.agy_acp_client import (
    _build_tool_event_from_record,
    _normalize_tool_arguments,
    _resolve_raw_tool_name,
)
from agy_acp.providers.state_store import SqliteStateStoreProvider


class _StubProber(McpCatalogProber):
    """Captures server declarations passed to `probe_servers` without network I/O."""

    def __init__(self) -> None:
        super().__init__(timeout_secs=1.0)
        self.received_declarations: list[list[dict[str, Any]]] = []

    async def probe_servers(
        self,
        servers: Any,
        *,
        session_env: Any = None,
        cwd: str | None = None,
    ) -> list[McpServerStatus]:
        _ = (session_env, cwd)
        captured = [dict(s) for s in servers]
        self.received_declarations.append(captured)
        return [
            McpServerStatus(
                name=str(s.get("name") or ""),
                display_name=str(s.get("name") or ""),
                status="connected",
                tools=(),
                origin="client",
            )
            for s in captured
        ]


# ---------------------------------------------------------------------------
# 1. Zero-Tool SessionWorker (kirocrew-lite, tools: []) Guard
# ---------------------------------------------------------------------------


def test_disallowed_native_tools_includes_schedule_workflow_swarm_ask_question() -> None:
    disallowed_empty = _compute_disallowed_native_tools({"agentTools": []})
    for name in ("schedule", "run_workflow", "swarm", "ask_question"):
        assert name in disallowed_empty

    disallowed_specific = _compute_disallowed_native_tools(
        {"agentExcludedTools": ["schedule", "swarm"]}
    )
    assert "schedule" in disallowed_specific
    assert "swarm" in disallowed_specific
    assert "run_workflow" not in disallowed_specific
    assert "ask_question" not in disallowed_specific

    disallowed_builtin = _compute_disallowed_native_tools({"agentExcludedTools": ["builtin"]})
    for name in ("schedule", "run_workflow", "swarm", "ask_question"):
        assert name in disallowed_builtin


def test_system_prompt_append_emits_strict_zero_tool_ban_for_empty_tools() -> None:
    prompt = _build_system_prompt_append({"agentTools": []})
    assert prompt is not None
    assert "[DISALLOWED NATIVE TOOLS]" in prompt
    assert "tools: []" in prompt
    assert "call_mcp_tool" in prompt
    assert "list_resources" in prompt
    assert "read_resource" in prompt
    assert "schedule" in prompt
    assert "run_workflow" in prompt
    assert "swarm" in prompt
    assert "ask_question" in prompt
    assert "plain text" in prompt.lower()

    # Does not duplicate when already present in user_text
    assert (
        _build_system_prompt_append(
            {"agentTools": []},
            user_text="already has [DISALLOWED NATIVE TOOLS]",
        )
        is None
    )


def test_system_prompt_append_emits_strict_zero_tool_ban_for_star_excluded() -> None:
    prompt = _build_system_prompt_append({"agentExcludedTools": ["*"]})
    assert prompt is not None
    assert "[DISALLOWED NATIVE TOOLS]" in prompt
    assert "call_mcp_tool" in prompt
    assert "plain text" in prompt.lower()


def test_get_active_custom_agent_reconstructs_persisted_zero_tool_session(tmp_path: Path) -> None:
    store = SqliteStateStoreProvider(tmp_path)
    mgr = SessionManager(state_store=store, wire_adapter=V3KasWireAdapter())
    store.upsert_session(
        {
            "sessionId": "sess-zero-tool",
            "cwd": str(tmp_path),
            "agent": "kirocrew-lite",
            "agentPrompt": None,
            "agentTools": [],
            "agentExcludedTools": [],
            "agentAllowedTools": [],
            "agentPermissions": None,
            "custom_agents": [],
            "mcp_servers": [],
        }
    )

    spec = mgr.get_active_custom_agent("sess-zero-tool")
    assert spec is not None
    assert spec.agent_id == "kirocrew-lite"
    assert spec.tools == ()

    dispatcher = AcpDispatcher(session_manager=mgr, wire_adapter=V3KasWireAdapter())
    session = store.get_session("sess-zero-tool")
    assert session is not None

    from agy_acp.logic.acp_dispatcher import _to_normalized_tool_event

    read_norm = _to_normalized_tool_event(
        ToolCallEvent(
            tool_call_id="tc-1",
            title="Reading foo.py",
            kind="read",
            status="in_progress",
            kas_builtin_id="read_file",
            kiro_tool_id="read_file",
            kiro_tool_name="fs_read",
        )
    )
    mcp_norm = _to_normalized_tool_event(
        ToolCallEvent(
            tool_call_id="tc-2",
            title="@kirocrew-core/spawn_run",
            kind="other",
            status="in_progress",
            mcp_server="kirocrew-core",
            mcp_tool="spawn_run",
            kiro_tool_id="@kirocrew-core/spawn_run",
        )
    )
    assert (
        dispatcher._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=read_norm,
            session_mode="auto",
        )
        == "deny"
    )
    assert (
        dispatcher._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=mcp_norm,
            session_mode="auto",
        )
        == "deny"
    )


# ---------------------------------------------------------------------------
# 2. MCP list_resources / read_resource Classification & call_mcp_tool Args
# ---------------------------------------------------------------------------


def test_mcp_resource_tools_classify_as_mcp_not_fs_read() -> None:
    list_cls = classify_tool_call_step("tool", "list_resources", {"ServerName": "docs"})
    assert list_cls.mcp_server == "docs"
    assert list_cls.mcp_tool == "list_resources"
    assert list_cls.kiro_tool_id == "@docs/list_resources"
    assert list_cls.acp_kind == "other"
    assert list_cls.kas_builtin_id == ""

    read_cls = classify_tool_call_step(
        "tool",
        "read_resource",
        {"ServerName": "docs", "Uri": "docs://guide"},
    )
    assert read_cls.mcp_server == "docs"
    assert read_cls.mcp_tool == "read_resource"
    assert read_cls.kiro_tool_id == "@docs/read_resource"
    assert read_cls.acp_kind == "other"
    assert read_cls.kas_builtin_id == ""

    inferred_cls = classify_tool_call_step("", "", {"ServerName": "docs", "Uri": "docs://guide"})
    assert inferred_cls.mcp_server == "docs"
    assert inferred_cls.mcp_tool == "read_resource"
    assert inferred_cls.kiro_tool_id == "@docs/read_resource"


def test_mcp_resource_tools_without_server_name_never_classify_as_fs_read() -> None:
    for tool_name in ("list_resources", "read_resource"):
        cls = classify_tool_call_step("tool", tool_name, {})
        assert cls.mcp_server == ""
        assert cls.kas_builtin_id == "mcp_resource"
        assert cls.v2_tool_name == "mcp_resource"
        assert cls.acp_kind == "other"

        cls_blank = classify_tool_call_step("tool", tool_name, {"ServerName": "   "})
        assert cls_blank.kas_builtin_id == "mcp_resource"
        assert cls_blank.v2_tool_name == "mcp_resource"
        assert cls_blank.acp_kind == "other"


def test_fs_read_grant_does_not_auto_approve_mcp_resource_reads(tmp_path: Path) -> None:
    store = SqliteStateStoreProvider(tmp_path)
    mgr = SessionManager(state_store=store, wire_adapter=V3KasWireAdapter())
    dispatcher = AcpDispatcher(session_manager=mgr, wire_adapter=V3KasWireAdapter())
    session = {
        "cwd": str(tmp_path),
        "agentTools": None,
        "agentExcludedTools": [],
        "agentAllowedTools": ["fs_read", "read_file"],
        "agentPermissions": None,
    }
    from agy_acp.logic.acp_dispatcher import _to_normalized_tool_event

    _, _, event = _build_tool_event_from_record(
        "tc-res-1",
        {
            "name": "read_resource",
            "kind": "read",
            "raw_args": {"ServerName": "docs", "Uri": "docs://secret"},
        },
        known_mcp_servers=["docs"],
    )
    norm = _to_normalized_tool_event(event, known_mcp_servers=["docs"])
    assert norm.is_mcp is True
    assert norm.mcp_server == "docs"
    assert norm.mcp_tool == "read_resource"
    assert event.raw_input == {"Uri": "docs://secret"}
    assert (
        dispatcher._evaluate_tool_permission(
            session=session,
            spec=None,
            norm=norm,
            session_mode="auto",
        )
        == "ask"
    )

    # Granting @docs allows the MCP resource read
    session_with_mcp = {**session, "agentAllowedTools": ["@docs"]}
    assert (
        dispatcher._evaluate_tool_permission(
            session=session_with_mcp,
            spec=None,
            norm=norm,
            session_mode="auto",
        )
        == "allow"
    )


def test_resolve_raw_tool_name_and_normalize_arguments_for_mcp_wrappers() -> None:
    assert _resolve_raw_tool_name({}, {"ServerName": "docs", "Uri": "docs://a"}) == "read_resource"

    assert (
        _normalize_tool_arguments(
            {"ServerName": "kirocrew-core", "ToolName": "resource_status"},
            tool_name="call_mcp_tool",
        )
        == {}
    )
    assert (
        _normalize_tool_arguments(
            {"ServerName": "kirocrew-core", "ToolName": "resource_status", "Arguments": None},
            tool_name="call_mcp_tool",
        )
        == {}
    )
    assert _normalize_tool_arguments(
        {
            "ServerName": "kirocrew-core",
            "ToolName": "resource_status",
            "Arguments": {"verbose": True},
        },
        tool_name="call_mcp_tool",
    ) == {"verbose": True}

    assert _normalize_tool_arguments({"ServerName": "docs"}, tool_name="list_resources") == {}
    assert _normalize_tool_arguments(
        {"ServerName": "docs", "Uri": "docs://a"},
        tool_name="read_resource",
    ) == {"Uri": "docs://a"}
    assert _normalize_tool_arguments(
        {"server_name": "docs", "Uri": "docs://a"},
        tool_name="read_resource",
    ) == {"Uri": "docs://a"}


# ---------------------------------------------------------------------------
# 3. KAS _kiro/secret/get Grant Lookup & _meta.kiro.settings.toolSearch
# ---------------------------------------------------------------------------


def test_kas_mcp_tokens_secret_key_matches_engine_hash() -> None:
    url = "  \tHTTPS://MCP.Example.com/v1\r\n"
    expected_digest = hashlib.sha256(b"https://mcp.example.com/v1|").hexdigest()
    assert kas_mcp_tokens_secret_key(url) == f"kiro.mcp.{expected_digest}.tokens"

    # Must NOT strip non-JS-ASCII whitespace like \x1f
    url_with_us = "\x1fhttps://mcp.example.com/v1"
    expected_us_digest = hashlib.sha256(b"\x1fhttps://mcp.example.com/v1|").hexdigest()
    assert kas_mcp_tokens_secret_key(url_with_us) == f"kiro.mcp.{expected_us_digest}.tokens"


@pytest.mark.anyio
async def test_kas_secret_storage_injects_ephemeral_bearer_without_mutating_sqlite(
    tmp_path: Path,
) -> None:
    prober = _StubProber()
    store = SqliteStateStoreProvider(tmp_path)
    mgr = SessionManager(state_store=store, wire_adapter=V3KasWireAdapter(), mcp_prober=prober)
    rpc_calls: list[tuple[str, dict[str, Any]]] = []
    mcp_url = "https://mcp.example.com/sse"
    expected_key = kas_mcp_tokens_secret_key(mcp_url)

    async def _fake_client_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
        rpc_calls.append((method, dict(params)))
        if method == "_kiro/secret/get" and params.get("key") == expected_key:
            return {"value": json.dumps({"access_token": "tok-secret-123"})}
        return {}

    dispatcher = AcpDispatcher(
        session_manager=mgr,
        wire_adapter=V3KasWireAdapter(),
        client_rpc_caller=_fake_client_rpc,
    )

    init_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "_meta": {
                            "kiro": {
                                "secretStorage": True,
                                "settings": {
                                    "toolSearch": {
                                        "neverDefer": ["kirocrew-core/spawn_run"],
                                    }
                                },
                            }
                        }
                    },
                },
            }
        )
    )
    assert init_resp is not None and "result" in init_resp
    assert dispatcher._client_secret_storage is True
    assert dispatcher._client_settings == {
        "toolSearch": {"neverDefer": ["kirocrew-core/spawn_run"]}
    }

    new_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [
                        {"type": "http", "name": "remote", "url": mcp_url, "headers": []},
                        {
                            "type": "http",
                            "name": "already_authed",
                            "url": "https://authed.example.com/mcp",
                            "headers": [{"name": "authorization", "value": "Bearer existing"}],
                        },
                    ],
                },
            }
        )
    )
    assert new_resp is not None and "result" in new_resp
    sid = new_resp["result"]["sessionId"]

    # Only the server lacking an Authorization header triggered _kiro/secret/get
    assert rpc_calls == [("_kiro/secret/get", {"key": expected_key})]

    # Prober received ephemeral Authorization: Bearer header for `remote`
    assert len(prober.received_declarations) == 1
    by_name = {d["name"]: d for d in prober.received_declarations[0]}
    assert by_name["remote"]["headers"]["Authorization"] == "Bearer tok-secret-123"

    # Persisted SQLite session state MUST NOT contain the token
    persisted = store.get_session(sid)
    assert persisted is not None
    persisted_json = json.dumps(persisted)
    assert "tok-secret-123" not in persisted_json


@pytest.mark.anyio
async def test_kas_secret_storage_rejects_control_chars_and_handles_errors(
    tmp_path: Path,
) -> None:
    prober = _StubProber()
    store = SqliteStateStoreProvider(tmp_path)
    mgr = SessionManager(state_store=store, wire_adapter=V3KasWireAdapter(), mcp_prober=prober)

    async def _bad_client_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
        _ = method
        key = params.get("key", "")
        if key == kas_mcp_tokens_secret_key("https://crlf.example.com/mcp"):
            return {"value": json.dumps({"accessToken": "bad\r\nX-Injected: 1"})}
        if key == kas_mcp_tokens_secret_key("https://err.example.com/mcp"):
            raise RuntimeError("vault locked")
        return {"value": "not-json"}

    dispatcher = AcpDispatcher(
        session_manager=mgr,
        wire_adapter=V3KasWireAdapter(),
        client_rpc_caller=_bad_client_rpc,
    )
    await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": 1,
                    "clientCapabilities": {"_meta": {"kiro": {"secretStorage": True}}},
                },
            }
        )
    )
    resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": [
                        {"type": "http", "name": "crlf", "url": "https://crlf.example.com/mcp"},
                        {"type": "http", "name": "err", "url": "https://err.example.com/mcp"},
                        {"type": "http", "name": "badjson", "url": "https://bad.example.com/mcp"},
                    ],
                },
            }
        )
    )
    assert resp is not None and "result" in resp
    for decl in prober.received_declarations[0]:
        headers = decl.get("headers") or {}
        if isinstance(headers, dict):
            assert "Authorization" not in headers


def test_tool_search_never_defer_honors_spawn_run_and_respects_agent_tools() -> None:
    base_core = [{"name": "kirocrew-core", "command": "kirocrew"}]

    assert (
        _session_has_kirocrew_spawn_run(
            {
                "mcp_servers": base_core,
                "meta": {
                    "kiro": {
                        "settings": {"toolSearch": {"neverDefer": ["kirocrew-core/spawn_run"]}}
                    }
                },
            }
        )
        is True
    )
    assert (
        _session_has_kirocrew_spawn_run(
            {
                "mcp_servers": base_core,
                "meta": {
                    "kiro": {
                        "settings": {"toolSearch": {"neverDefer": ["@kirocrew-core/spawn_run"]}}
                    }
                },
            }
        )
        is True
    )
    # Explicit agentTools omitting @kirocrew-core and spawn_run still blocks spawn_run
    assert (
        _session_has_kirocrew_spawn_run(
            {
                "mcp_servers": base_core,
                "agentTools": ["read_file"],
                "meta": {
                    "kiro": {
                        "settings": {"toolSearch": {"neverDefer": ["kirocrew-core/spawn_run"]}}
                    }
                },
            }
        )
        is False
    )
    assert (
        _session_has_kirocrew_spawn_run(
            {
                "mcp_servers": base_core,
                "agentTools": [],
                "meta": {
                    "kiro": {
                        "settings": {"toolSearch": {"neverDefer": ["kirocrew-core/spawn_run"]}}
                    }
                },
            }
        )
        is False
    )
    # disabledTools or agentExcludedTools still blocks spawn_run
    assert (
        _session_has_kirocrew_spawn_run(
            {
                "mcp_servers": [
                    {"name": "kirocrew-core", "command": "kirocrew", "disabledTools": ["spawn_run"]}
                ],
                "meta": {
                    "kiro": {
                        "settings": {"toolSearch": {"neverDefer": ["kirocrew-core/spawn_run"]}}
                    }
                },
            }
        )
        is False
    )
    assert (
        _session_has_kirocrew_spawn_run(
            {
                "mcp_servers": base_core,
                "agentExcludedTools": ["spawn_run"],
                "meta": {
                    "kiro": {
                        "settings": {"toolSearch": {"neverDefer": ["kirocrew-core/spawn_run"]}}
                    }
                },
            }
        )
        is False
    )
