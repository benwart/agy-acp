"""Session lifecycle, multi-session multiplexing, steer, and cancellation logic backed by SQLite."""

from __future__ import annotations

import base64
import contextlib
import copy
import os
import re
import uuid
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

from agy_acp.logic.custom_agents import (
    _READ_ONLY_KIRO_TOOLS,
    CustomAgentRegistry,
    CustomAgentSpec,
    load_kiro_agent_spec,
    read_contained_regular_bytes,
)
from agy_acp.logic.mcp_catalog_prober import McpCatalogProber
from agy_acp.logic.retry_warning_filter import strip_agy_retry_warnings
from agy_acp.logic.tool_classifier import (
    FALLBACK_MODELS,
    PER_SESSION_MCP_ENV_KEYS,
    VALID_EFFORTS,
    ToolCallEvent,
    ToolPermissionDecision,
    _dedupe_mcp_servers,
    _pairs_to_dict,
    resolve_model_for_effort,
    translate_mcp_server,
)
from agy_acp.logic.wire import EngineWireAdapter, McpServerStatus, V2KiroWireAdapter
from agy_acp.logic.wire.base import DEFAULT_MODEL_ID
from agy_acp.providers.agy_acp_client import AgyAcpClientProvider
from agy_acp.providers.state_store import SqliteStateStoreProvider
from agy_acp.providers.telemetry import TelemetryCallbacks, get_telemetry

DEFAULT_REASONING_EFFORT = "medium"
_POLICY_PROMPT_VERSION = 2

_KIROCREW_SPAWN_ROUTING_DIRECTIVE = (
    "[KIROCREW SUBAGENT ROUTING]\n"
    "Subagent delegation in this KiroCrew session MUST use kirocrew-core's `spawn_run` tool "
    'via `call_mcp_tool` (`ServerName="kirocrew-core"`, `ToolName="spawn_run"`, '
    "`Arguments={...}`; read `<appDataDir>/mcp/kirocrew-core/spawn_run.json` first if needed). "
    "NEVER call built-in `invoke_subagent`, `define_subagent`, or `manage_subagents`, "
    "because native subagents bypass KiroCrew chat cards and completion events."
)

_KIROCREW_QUIET_TURN_CONTRACT = (
    "[KIROCREW QUIET TURN CONTRACT]\n"
    "- NEVER call native `schedule` (use `kirocrew-cron`, `monitor_start`, or `wait` when "
    "permitted, or `nothing_to_do` on quiet wakes).\n"
    "- On ANY non-user wake (`[Heartbeat]`, `[Worker report]`, `[work-ledger wake]`, "
    "`[monitor wake]`, `[auto-nudge cycle]`) that needs no user-visible chat message, "
    "ALWAYS end the turn by calling `kirocrew-core` `nothing_to_do` -- NEVER output a bare "
    "`.` or filler text to satisfy an empty-output check.\n"
    "[END KIROCREW QUIET TURN CONTRACT]"
)

_READ_ONLY_TOOL_NAMES: frozenset[str] = _READ_ONLY_KIRO_TOOLS | {
    "read",
    "read_file",
    "list_directory",
    "grep_search",
    "file_search",
    "remote_web_search",
}

_SPAWN_RUN_ALLOW_PATTERNS: frozenset[str] = frozenset(
    {
        "*",
        "@kirocrew-core",
        "@kirocrew-core/*",
        "@kirocrew-core/spawn_run",
        "mcp_kirocrew-core_spawn_run",
    }
)
_SPAWN_RUN_DENY_PATTERNS: frozenset[str] = _SPAWN_RUN_ALLOW_PATTERNS | {"spawn_run"}
_SPAWN_RUN_NEVER_DEFER_PATTERNS: frozenset[str] = frozenset(
    {"kirocrew-core/spawn_run", "@kirocrew-core/spawn_run"}
)

AVAILABLE_MODES: list[dict[str, str]] = [
    {"id": "auto", "name": "Auto"},
    {"id": "approve", "name": "Approve"},
    {"id": "read-only", "name": "Read Only"},
]
VALID_MODE_IDS: frozenset[str] = frozenset(m["id"] for m in AVAILABLE_MODES)
REASONING_EFFORT_CONFIG_IDS: frozenset[str] = frozenset(
    {"effortLevel", "reasoning_effort", "thought_level", "thinkingLevel"}
)
AVAILABLE_MODELS: list[dict[str, str]] = list(FALLBACK_MODELS)

_EXT_TO_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}
_IMAGE_EXTS: frozenset[str] = frozenset(_EXT_TO_MIME)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_USER_MESSAGE_BLOCK_RE = re.compile(r"<user_message>\s*(.*?)\s*</user_message>", re.DOTALL)


def _extract_tool_search_never_defer(meta: Mapping[str, Any] | None) -> list[str]:
    """Extract `_meta.kiro.settings.toolSearch.neverDefer` string entries when present."""
    if not isinstance(meta, Mapping):
        return []
    kiro_meta = meta.get("kiro")
    if not isinstance(kiro_meta, Mapping):
        return []
    settings = kiro_meta.get("settings")
    if not isinstance(settings, Mapping):
        return []
    tool_search = settings.get("toolSearch")
    if not isinstance(tool_search, Mapping):
        return []
    raw_never_defer = tool_search.get("neverDefer")
    if not isinstance(raw_never_defer, list):
        return []
    return [entry.strip() for entry in raw_never_defer if isinstance(entry, str) and entry.strip()]


def _session_has_kirocrew_spawn_run(session: Mapping[str, Any]) -> bool:
    """Return whether `session` exposes enabled `kirocrew-core` `spawn_run`."""
    raw_servers = session.get("mcp_servers")
    if not isinstance(raw_servers, list) or not raw_servers:
        raw_servers = session.get("wire_mcp_servers")
    if not isinstance(raw_servers, list):
        return False

    core_srv = next(
        (
            srv
            for srv in raw_servers
            if isinstance(srv, dict)
            and str(srv.get("name") or "").strip() == "kirocrew-core"
            and srv.get("disabled") is not True
        ),
        None,
    )
    if core_srv is None:
        return False

    disabled_tools = core_srv.get("disabledTools") or core_srv.get("disabled_tools")
    if isinstance(disabled_tools, list) and any(
        str(t).strip() == "spawn_run" for t in disabled_tools
    ):
        return False

    excluded_tools = session.get("agentExcludedTools")
    if isinstance(excluded_tools, list) and any(
        str(t).strip() in _SPAWN_RUN_DENY_PATTERNS for t in excluded_tools
    ):
        return False

    agent_tools = session.get("agentTools")
    if isinstance(agent_tools, list):
        allowed = _SPAWN_RUN_ALLOW_PATTERNS | {"spawn_run", "kirocrew-core/spawn_run"}
        return any(str(tool).strip() in allowed for tool in agent_tools)

    never_defer = _extract_tool_search_never_defer(session.get("meta"))
    if any(entry in _SPAWN_RUN_NEVER_DEFER_PATTERNS for entry in never_defer):
        return True
    return True


def _extract_session_mcp_env(
    session: Mapping[str, Any],
    *,
    session_id: str | None = None,
    state_db_path: Path | None = None,
) -> dict[str, str]:
    """Build isolated per-session MCP identity environment variables for an `agy` subprocess."""
    env: dict[str, str] = dict.fromkeys(PER_SESSION_MCP_ENV_KEYS, "")
    env["KIROCREW_ROUTE_SPAWN_RUN"] = "1" if _session_has_kirocrew_spawn_run(session) else ""

    sid = str(session_id or session.get("sessionId") or "").strip()
    if sid:
        env["AGY_ACP_SESSION_ID"] = sid
    if state_db_path is not None:
        env["AGY_ACP_STATE_DB"] = str(state_db_path)

    for source_key in ("mcp_servers", "wire_mcp_servers"):
        sources = session.get(source_key)
        if not isinstance(sources, list):
            continue
        for srv in sources:
            if not isinstance(srv, dict) or srv.get("disabled") is True:
                continue
            for key, val in _pairs_to_dict(srv.get("env")).items():
                if key in PER_SESSION_MCP_ENV_KEYS and val:
                    env[key] = val
    return env


_NATIVE_TOOL_POLICY_PATTERNS: tuple[tuple[str, frozenset[str]], ...] = (
    ("run_command", frozenset({"run_command", "execute_bash", "shell", "bash", "builtin"})),
    ("manage_task", frozenset({"run_command", "execute_bash", "shell", "bash", "builtin"})),
    ("read_file", frozenset({"read_file", "fs_read", "read", "builtin"})),
    ("write_file", frozenset({"fs_write", "write", "builtin"})),
    ("edit_file", frozenset({"str_replace", "fs_write", "write", "edit", "builtin"})),
    ("search_web", frozenset({"remote_web_search", "web_search", "builtin"})),
    (
        "invoke_subagent",
        frozenset({"invoke_sub_agent", "use_subagent", "subagent", "builtin"}),
    ),
    (
        "define_subagent",
        frozenset({"invoke_sub_agent", "use_subagent", "subagent", "builtin"}),
    ),
    (
        "manage_subagents",
        frozenset({"invoke_sub_agent", "use_subagent", "subagent", "builtin"}),
    ),
    (
        "send_message",
        frozenset({"invoke_sub_agent", "use_subagent", "subagent", "builtin"}),
    ),
    ("schedule", frozenset({"schedule", "builtin"})),
    ("run_workflow", frozenset({"run_workflow", "builtin"})),
    ("swarm", frozenset({"swarm", "builtin"})),
    ("ask_question", frozenset({"ask_question", "builtin"})),
)


def _compute_disallowed_native_tools(session: Mapping[str, Any]) -> tuple[str, ...]:
    """Return native `agy` built-in tool names blocked by `agentTools` / `agentExcludedTools`."""
    raw_excluded = session.get("agentExcludedTools")
    excluded = (
        {str(t).strip() for t in raw_excluded if isinstance(t, str) and t.strip()}
        if isinstance(raw_excluded, list)
        else set()
    )
    raw_tools = session.get("agentTools")
    allowlist = (
        {str(t).strip() for t in raw_tools if isinstance(t, str) and t.strip()}
        if isinstance(raw_tools, list)
        else None
    )
    restrict_by_allowlist = allowlist is not None and "*" not in allowlist

    disallowed: list[str] = []
    for native_name, patterns in _NATIVE_TOOL_POLICY_PATTERNS:
        if (
            "*" in excluded
            or bool(excluded & patterns)
            or (restrict_by_allowlist and allowlist is not None and not (allowlist & patterns))
        ):
            disallowed.append(native_name)
    return tuple(disallowed)


def _is_zero_tool_session(session: Mapping[str, Any]) -> bool:
    """Return True when `agentTools` is `[]` or `agentExcludedTools` contains `'*'`."""
    raw_tools = session.get("agentTools")
    if isinstance(raw_tools, list) and len(raw_tools) == 0:
        return True
    raw_excluded = session.get("agentExcludedTools")
    return isinstance(raw_excluded, list) and any(
        isinstance(t, str) and t.strip() == "*" for t in raw_excluded
    )


def _session_has_kirocrew_core(session: Mapping[str, Any]) -> bool:
    """Return whether `session` exposes an enabled `kirocrew-core` MCP server."""
    raw_servers = session.get("mcp_servers")
    if not isinstance(raw_servers, list) or not raw_servers:
        raw_servers = session.get("wire_mcp_servers")
    if not isinstance(raw_servers, list):
        return False
    return any(
        isinstance(srv, Mapping)
        and str(srv.get("name") or "").strip() == "kirocrew-core"
        and srv.get("disabled") is not True
        for srv in raw_servers
    )


def _build_policy_guidance_sections(
    session: Mapping[str, Any],
    *,
    user_text: str = "",
) -> list[str]:
    """Build `[DISALLOWED NATIVE TOOLS]` and `[KIROCREW QUIET TURN CONTRACT]` sections."""
    sections: list[str] = []
    is_zero_tool = _is_zero_tool_session(session)
    if "[DISALLOWED NATIVE TOOLS]" not in user_text:
        if is_zero_tool:
            sections.append(
                "[DISALLOWED NATIVE TOOLS]\n"
                "ALL built-in and MCP tools (including `call_mcp_tool`, `list_resources`, "
                "`read_resource`, `schedule`, `run_workflow`, `swarm`, `ask_question`, and all "
                "file, shell, web, and subagent tools) are disabled (`tools: []`) and will abort "
                "the turn if called.\n"
                "You MUST respond strictly in plain text without calling any tool.\n"
                "[END DISALLOWED NATIVE TOOLS]"
            )
        else:
            disallowed_native = _compute_disallowed_native_tools(session)
            if disallowed_native:
                tool_list = ", ".join(f"`{name}`" for name in disallowed_native)
                sections.append(
                    "[DISALLOWED NATIVE TOOLS]\n"
                    "The following built-in tools are NOT permitted by this agent's policy and "
                    f"will abort the turn if called: {tool_list}.\n"
                    "NEVER call any of these tools. Use only permitted tools and available "
                    "MCP tools.\n"
                    "[END DISALLOWED NATIVE TOOLS]"
                )
    if (
        not is_zero_tool
        and _session_has_kirocrew_core(session)
        and "[KIROCREW QUIET TURN CONTRACT]" not in user_text
    ):
        sections.append(_KIROCREW_QUIET_TURN_CONTRACT)
    return sections


def _build_system_prompt_append(
    session: Mapping[str, Any],
    *,
    user_text: str = "",
) -> str | None:
    """Build `_meta.antigravity.systemPrompt.append` for `session/new` and `session/load`."""
    sections: list[str] = []
    agent_prompt = session.get("agentPrompt")
    if (
        isinstance(agent_prompt, str)
        and agent_prompt.strip()
        and agent_prompt.strip() not in user_text
    ):
        clean_prompt = agent_prompt.strip()
        if "[AGENT SYSTEM PROMPT]" in clean_prompt:
            sections.append(f"[AGENT MANDATE]\n{clean_prompt}")
        else:
            sections.append(
                "[AGENT MANDATE]\n"
                "Treat the [AGENT SYSTEM PROMPT] ... [END AGENT SYSTEM PROMPT] block below "
                "as your authoritative operating contract.\n\n"
                f"[AGENT SYSTEM PROMPT]\n{clean_prompt}\n[END AGENT SYSTEM PROMPT]"
            )
    if _session_has_kirocrew_spawn_run(session) and "[KIROCREW SUBAGENT ROUTING]" not in user_text:
        sections.append(_KIROCREW_SPAWN_ROUTING_DIRECTIVE)
    sections.extend(_build_policy_guidance_sections(session, user_text=user_text))
    return "\n\n".join(sections) if sections else None


def _extract_wire_custom_agents(meta: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Extract `params._meta.kiro.customAgents` when explicitly provided on the wire."""
    if not isinstance(meta, dict):
        return None
    kiro_meta = meta.get("kiro")
    if not isinstance(kiro_meta, dict) or "customAgents" not in kiro_meta:
        return None
    raw_agents = kiro_meta.get("customAgents")
    if not isinstance(raw_agents, list):
        return []
    return [copy.deepcopy(entry) for entry in raw_agents if isinstance(entry, dict)]


def _format_remaining_turns_summary(turns: Sequence[Mapping[str, Any]]) -> str | None:
    """Format persisted `session_turns` into a compacted conversation summary for replay/fork."""
    if not turns:
        return None
    formatted_turns: list[str] = []
    for turn in turns:
        blocks = turn.get("prompt_blocks")
        user_parts = [
            str(block.get("text") or "").strip()
            for block in (blocks if isinstance(blocks, list) else ())
            if isinstance(block, Mapping)
            and block.get("type") == "text"
            and str(block.get("text") or "").strip()
        ]
        user_text = "\n".join(user_parts).strip()
        assistant_text = strip_agy_retry_warnings(str(turn.get("assistant_text") or "")).strip()
        lines = [
            *(f"User: {user_text}" for _ in (1,) if user_text),
            *(f"Assistant: {assistant_text}" for _ in (1,) if assistant_text),
        ]
        if lines:
            formatted_turns.append("\n".join(lines))
    return "\n\n".join(formatted_turns) if formatted_turns else None


_MAX_INLINE_IMAGE_BYTES = 10 * 1024 * 1024


def _read_contained_local_image_block(
    src_path: Path,
    *,
    cwd: str | None = None,
) -> dict[str, str] | None:
    """Read a `cwd`-contained regular image (<= 10 MiB) into an inline base64 ACP `image` block."""
    if not cwd or not cwd.strip():
        return None
    loaded = read_contained_regular_bytes(
        src_path,
        root_dir=Path(cwd),
        max_bytes=_MAX_INLINE_IMAGE_BYTES,
        allowed_suffixes=_IMAGE_EXTS,
    )
    if loaded is None:
        return None
    mime = _EXT_TO_MIME.get(loaded[0].suffix.lower())
    if mime is None:
        return None
    encoded = base64.b64encode(loaded[1]).decode("ascii")
    return {"type": "image", "data": encoded, "mimeType": mime}


def _extract_markdown_local_image_blocks(
    text: str,
    *,
    cwd: str | None = None,
) -> list[dict[str, str]]:
    """Find `cwd`-contained markdown image links (`![...](path.png)`) and encode inline."""
    if not cwd or not cwd.strip():
        return []
    blocks: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        raw_target = match.group(1).strip().strip("<>\"'").removeprefix("file://")
        if not raw_target or raw_target in seen_targets:
            continue
        candidate = Path(raw_target)
        if candidate.suffix.lower() not in _IMAGE_EXTS:
            continue
        seen_targets.add(raw_target)
        img_block = _read_contained_local_image_block(candidate, cwd=cwd)
        if img_block is not None:
            blocks.append(img_block)
    return blocks


def _format_steer_message(
    raw_message: str,
    *,
    cwd: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Unwrap `<user_message>` steer text and build native ACP content blocks."""
    matches = _USER_MESSAGE_BLOCK_RE.findall(raw_message)
    body = "\n\n".join(m.strip() for m in matches if m.strip()) if matches else raw_message.strip()
    if not body:
        return "", []
    img_blocks = _extract_markdown_local_image_blocks(body, cwd=cwd)
    return body, [{"type": "text", "text": body}, *img_blocks]


def _write_dsh_gate_marker_if_configured() -> None:
    """Write `<nonce>\\tloaded\\n` with 0600 permissions when DSH gate env vars are set."""
    nonce = os.environ.get("KIROCREW_DSH_GATE_SESSION")
    marker_raw = os.environ.get("KIROCREW_DSH_GATE_MARKER")
    if not nonce or not marker_raw:
        return
    marker_path = Path(marker_raw)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, f"{nonce}\tloaded\n".encode())
    finally:
        os.close(fd)
    marker_path.chmod(0o600)


def _build_acp_content_blocks(
    prompt_blocks: Sequence[Mapping[str, Any]],
    *,
    cwd: str | None = None,
    prefix_text: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Build native ACP content blocks and combined text summary with zero disk staging."""
    text_parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    first_text_index: int | None = None

    for block in prompt_blocks:
        if not isinstance(block, Mapping):
            continue
        btype = str(block.get("type", ""))
        if btype == "text":
            text = str(block.get("text", ""))
            if text:
                if first_text_index is None:
                    first_text_index = len(blocks)
                text_parts.append(text)
                blocks.append({"type": "text", "text": text})
                blocks.extend(_extract_markdown_local_image_blocks(text, cwd=cwd))
        elif btype == "resource_link":
            uri = str(block.get("uri", ""))
            name = str(block.get("name") or uri)
            text_parts.append(f"[Resource: {name} ({uri})]")
            blocks.append(dict(block))
        elif btype == "resource":
            res = block.get("resource")
            if isinstance(res, Mapping):
                if "text" in res:
                    uri = str(res.get("uri", "embedded"))
                    text_parts.append(f"--- {uri} ---\n{res['text']}\n---")
                blocks.append({"type": "resource", "resource": dict(res)})
            else:
                blocks.append(dict(block))
        elif btype == "image":
            raw_b64 = str(block.get("data", ""))
            if raw_b64:
                img_entry: dict[str, Any] = {
                    "type": "image",
                    "data": raw_b64,
                    "mimeType": str(block.get("mimeType") or "image/png"),
                }
                if block.get("uri") is not None:
                    img_entry["uri"] = str(block["uri"])
                blocks.append(img_entry)
        elif btype:
            blocks.append(dict(block))

    clean_prefix = prefix_text.strip()
    if clean_prefix:
        text_parts.insert(0, clean_prefix)
        if first_text_index is not None:
            orig_txt = str(blocks[first_text_index].get("text") or "")
            blocks[first_text_index] = {
                "type": "text",
                "text": f"{clean_prefix}\n\n{orig_txt}".strip(),
            }
        else:
            blocks.insert(0, {"type": "text", "text": clean_prefix})

    return "\n\n".join(text_parts).strip(), blocks


def _build_config_options(
    *,
    model: str,
    mode: str,
    reasoning_effort: str,
    available_models: list[dict[str, str]] | None = None,
    wire_adapter: EngineWireAdapter | None = None,
) -> list[dict[str, Any]]:
    """Build ACP configOptions array for the given session state."""
    adapter = wire_adapter or V2KiroWireAdapter()
    models = list(available_models or AVAILABLE_MODELS)
    if model and model != "auto" and all(m.get("modelId") != model for m in models):
        models.insert(0, {"modelId": model, "name": model, "description": model})
    return adapter.build_config_options(
        models=models,
        current_model=model,
        current_mode=mode,
        current_effort=reasoning_effort,
    )


def _with_mcp_origin_meta(
    srv_name: str,
    srv_cfg: Mapping[str, Any],
    origin: str,
) -> dict[str, Any]:
    """Attach default Kiro MCP resource origin metadata to an agent MCP declaration."""
    entry = {"name": srv_name, **dict(srv_cfg)}
    entry.setdefault(
        "_meta",
        {"kiro": {"resource": {"resourceType": "mcpServer", "source": {"origin": origin}}}},
    )
    return entry


def _agent_mcp_mapping_to_list(
    raw_mcp: Mapping[str, Any],
    *,
    origin: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split `CustomAgentSpec.mcp_servers` into enabled server dicts and disabled names."""
    enabled: list[dict[str, Any]] = []
    disabled: list[str] = []
    for srv_name, srv_cfg in raw_mcp.items():
        if not isinstance(srv_name, str) or not srv_name or not isinstance(srv_cfg, Mapping):
            continue
        if srv_cfg.get("disabled") is True:
            disabled.append(srv_name)
        else:
            enabled.append(_with_mcp_origin_meta(srv_name, srv_cfg, origin))
    return enabled, disabled


def _failed_compaction(summary: str, prev_tokens: int) -> dict[str, Any]:
    """Build a failed compaction response preserving `prev_tokens`."""
    return {
        "status": "failed",
        "summary": summary,
        "contextTokens": prev_tokens,
        "inputTokens": 1,
        "outputTokens": 1,
    }


def _count_knowledge_items(path_str: str) -> int:
    """Count files under `path_str` when a directory, defaulting to `1`."""
    target_p = Path(path_str)
    if target_p.is_dir():
        with contextlib.suppress(OSError):
            return max(1, sum(1 for f in target_p.rglob("*") if f.is_file()))
    return 1


class SessionManager:
    """Manages concurrent ACP sessions backed by SQLite storage and `agy acp` subprocesses."""

    def __init__(
        self,
        *,
        save_dir: Path | None = None,
        state_store: SqliteStateStoreProvider | None = None,
        cli_provider: AgyAcpClientProvider | None = None,
        default_agent: str | None = None,
        default_model: str | None = None,
        default_effort: str | None = None,
        wire_adapter: EngineWireAdapter | None = None,
        mcp_prober: McpCatalogProber | None = None,
    ) -> None:
        cleaned_effort = default_effort.strip().lower() if default_effort else None
        if cleaned_effort and cleaned_effort not in VALID_EFFORTS:
            raise ValueError(f"Unsupported effort {default_effort!r}.")
        self._store = state_store or (
            cli_provider.state_store
            if cli_provider is not None
            else SqliteStateStoreProvider(save_dir)
        )
        self._cli = cli_provider or AgyAcpClientProvider(state_store=self._store)
        cleaned_model = default_model.strip() if default_model else None
        self._default_agent = default_agent
        self._default_model = (
            cleaned_model if cleaned_model and cleaned_model.lower() != "auto" else None
        )
        self._default_effort = cleaned_effort
        self._wire_adapter: EngineWireAdapter = wire_adapter or V2KiroWireAdapter()
        self._mcp_prober: McpCatalogProber = mcp_prober or McpCatalogProber()
        self._cancel_scopes: dict[str, anyio.CancelScope] = {}
        self._active_cancel_scopes: dict[str, anyio.CancelScope] = self._cancel_scopes
        self._turn_locks: dict[str, anyio.Lock] = {}
        self._mcp_status_cache: dict[str, dict[str, McpServerStatus]] = {}
        self._mcp_enabled: bool = True
        self._mcp_token_resolver: Callable[[str], Awaitable[str | None]] | None = None
        self._process_session_ids: set[str] = set()
        get_telemetry(self._store.db_path).register_callbacks(
            TelemetryCallbacks(
                backend_up=lambda: 1 if self._cli.worker_pid is not None else 0,
                session_counts=self._observe_session_counts,
                ledger_tokens=self._observe_ledger_tokens,
                mcp_tools_discovered=self._observe_mcp_tools,
            )
        )

    def _observe_session_counts(self) -> dict[str, int]:
        return {
            "active": len(self._process_session_ids),
            "persisted": self._store.count_sessions_silent(),
        }

    def _observe_ledger_tokens(self) -> dict[str, int]:
        ledger = self._store.get_token_ledger(record_metric=False)
        return {
            "input": ledger.get("input_tokens", 0),
            "output": ledger.get("output_tokens", 0),
            "cached": ledger.get("cached_read_tokens", 0),
            "thought": ledger.get("thinking_tokens", 0),
        }

    def _observe_mcp_tools(self) -> dict[str, int]:
        per_server: dict[str, int] = {}
        for status_map in self._mcp_status_cache.values():
            for srv_name, status in status_map.items():
                per_server[srv_name] = max(per_server.get(srv_name, 0), len(status.tools))
        return per_server

    async def close(self) -> None:
        """Close the underlying `AgyAcpClientProvider` and any active `agy acp` process."""
        await self._cli.close()

    @property
    def mcp_enabled(self) -> bool:
        """Return whether MCP server merging and probing are enabled for this process."""
        return self._mcp_enabled

    async def set_mcp_enabled(self, enabled: bool) -> None:
        """Toggle process-local MCP state (`_kiro/mcp/toggle`) without touching peer sessions."""
        next_enabled = bool(enabled)
        if self._mcp_enabled == next_enabled:
            return
        self._mcp_enabled = next_enabled
        self._mcp_status_cache.clear()
        for sid in list(self._process_session_ids):
            session = self._store.get_session(sid)
            if session is None:
                self._process_session_ids.discard(sid)
                continue
            prev_mcp = copy.deepcopy(session.get("mcp_servers", []))
            cwd = str(session.get("cwd") or "")
            registry = self.get_custom_agent_registry(
                cwd=cwd,
                custom_agents=session.get("custom_agents") or [],
            )
            raw_wire = session.get("wire_mcp_servers", [])
            active_agent = session.get("agent")
            session["mcp_servers"] = self._merge_session_mcp(
                cwd=cwd,
                raw_wire_mcp=raw_wire if isinstance(raw_wire, list) else [],
                active_agent=str(active_agent) if active_agent else None,
                registry=registry,
            )
            await self._maybe_reset_backend_on_profile_change(
                session,
                prev_mcp=prev_mcp,
                prev_prompt=session.get("agentPrompt"),
                old_agy_id=str(session.get("agyConversationId") or "").strip(),
            )
            self._store.upsert_session(session)

    @property
    def state_store(self) -> SqliteStateStoreProvider:
        """Return the underlying `SqliteStateStoreProvider`."""
        return self._store

    @property
    def cli_provider(self) -> AgyAcpClientProvider:
        """Return the underlying `AgyAcpClientProvider` instance."""
        return self._cli

    @property
    def wire_adapter(self) -> EngineWireAdapter:
        """Return the active `EngineWireAdapter` strategy."""
        return self._wire_adapter

    @property
    def mcp_prober(self) -> McpCatalogProber:
        """Return the active `McpCatalogProber` instance."""
        return self._mcp_prober

    async def _resolve_available_models(
        self,
        *,
        cwd: str = ".",
        refresh: bool = False,
    ) -> list[dict[str, str]]:
        """Resolve available models from SQLite cache or `AgyAcpClientProvider.discover_models`."""
        if not refresh:
            cached = self._store.load_models()
            if cached:
                return cached
            if (
                self._cli.is_custom_store
                and not self._cli.has_explicit_bin
                and type(self._cli).discover_models is AgyAcpClientProvider.discover_models
            ):
                return list(AVAILABLE_MODELS)
        discovered = await self._cli.discover_models(cwd=cwd, refresh=True)
        return discovered or list(AVAILABLE_MODELS)

    def _build_config_options_for_state(
        self,
        *,
        model: str,
        mode: str,
        reasoning_effort: str,
        available_models: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        models = available_models or self._store.load_models() or AVAILABLE_MODELS
        return _build_config_options(
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
            available_models=models,
            wire_adapter=self._wire_adapter,
        )

    @staticmethod
    def _build_config_options(
        *,
        model: str,
        mode: str,
        reasoning_effort: str,
    ) -> list[dict[str, Any]]:
        """Build ACP configOptions array for the given session state."""
        return _build_config_options(
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
        )

    def _user_agents_dir(self) -> Path | None:
        return (
            self._store.db_path.parent / ".kiro_user_agents" if self._cli.is_custom_store else None
        )

    def _build_auth_config(
        self,
        base: Mapping[str, Any] | None,
        agent: Any,
    ) -> dict[str, Any]:
        auth_cfg: dict[str, Any] = copy.deepcopy(dict(base)) if base else {}
        if agent:
            auth_cfg["kiro_agent"] = str(agent)
        if self._wire_adapter.engine_id == "v3":
            auth_cfg["_engine_id"] = "v3"
        return auth_cfg

    def get_custom_agent_registry(
        self,
        *,
        cwd: str | None,
        custom_agents: Sequence[Mapping[str, Any]] | None = None,
    ) -> CustomAgentRegistry:
        """Load `CustomAgentRegistry` across project, user, and client wire scopes."""
        return CustomAgentRegistry.load(
            cwd=cwd or None,
            wire_custom_agents=custom_agents,
            user_agents_dir=self._user_agents_dir(),
        )

    def get_active_custom_agent(self, session_id: str) -> CustomAgentSpec | None:
        """Return the active `CustomAgentSpec` for `session_id`, if any."""
        session = self._store.get_session(session_id)
        if session is None:
            return None
        agent_id = session.get("agent")
        if not isinstance(agent_id, str) or not agent_id:
            return None
        registry = self.get_custom_agent_registry(
            cwd=str(session.get("cwd") or ""),
            custom_agents=session.get("custom_agents") or [],
        )
        spec = registry.get(agent_id)
        if spec is not None:
            return spec
        if (
            session.get("agentPermissions") is not None
            or session.get("agentPrompt")
            or session.get("agentTools") is not None
        ):
            raw_tools = session.get("agentTools")
            return CustomAgentSpec(
                agent_id=agent_id,
                name=agent_id,
                prompt=str(session.get("agentPrompt") or ""),
                tools=tuple(raw_tools) if isinstance(raw_tools, list) else None,
                excluded_tools=tuple(session.get("agentExcludedTools") or ()),
                allowed_tools=tuple(session.get("agentAllowedTools") or ()),
                permissions=session.get("agentPermissions"),
            )
        return None

    def _build_descriptor(
        self,
        state: dict[str, Any],
        *,
        available_models: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build an ACP `session/new` or `session/load` descriptor via `self._wire_adapter`."""
        raw_model = str(state.get("model", DEFAULT_MODEL_ID))
        mode = str(state.get("mode", "auto"))
        active_agent = state.get("agent") or self._default_agent
        current_mode_id = str(active_agent) if active_agent else mode
        reasoning_effort = str(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        models = list(available_models or AVAILABLE_MODELS)
        if raw_model and raw_model != "auto" and all(m.get("modelId") != raw_model for m in models):
            models.insert(0, {"modelId": raw_model, "name": raw_model, "description": raw_model})
        concrete_model = self._wire_adapter.resolve_concrete_model(models, raw_model)

        registry = self.get_custom_agent_registry(
            cwd=str(state.get("cwd") or ""),
            custom_agents=state.get("custom_agents") or [],
        )
        custom_specs = registry.list_agents()
        existing_ids = {s.agent_id for s in custom_specs} | set(VALID_MODE_IDS)
        virtual_ids: tuple[str | None, ...] = (
            (
                self._default_agent,
                str(active_agent) if active_agent else None,
                "kirocrew",
                "kirocrew-lite",
            )
            if self._wire_adapter.engine_id == "v2"
            else (str(active_agent) if active_agent else None,)
        )
        for virtual_id in virtual_ids:
            if virtual_id and virtual_id not in existing_ids:
                existing_ids.add(virtual_id)
                custom_specs.append(
                    CustomAgentSpec(agent_id=virtual_id, name=virtual_id, origin="bundled")
                )

        return {
            "sessionId": str(state["sessionId"]),
            "modes": {
                "currentModeId": current_mode_id,
                "availableModes": self._wire_adapter.build_available_modes(
                    custom_agents=custom_specs
                ),
            },
            "models": {
                "currentModelId": concrete_model,
                "availableModels": models,
            },
            "configOptions": self._wire_adapter.build_config_options(
                models=models,
                current_model=concrete_model,
                current_mode=mode,
                current_effort=reasoning_effort,
            ),
        }

    def _merge_session_mcp(
        self,
        *,
        cwd: str,
        raw_wire_mcp: list[dict[str, Any]],
        active_agent: str | None,
        registry: CustomAgentRegistry,
    ) -> list[dict[str, Any]]:
        """Merge session wire MCP servers with active customAgent MCP servers (session wins)."""
        if not self._mcp_enabled:
            return []
        wire_translated = _dedupe_mcp_servers(
            [translate_mcp_server(s) for s in raw_wire_mcp if isinstance(s, dict)]
        )
        spec = registry.get(active_agent) if active_agent else None
        agent_raw_servers: list[dict[str, Any]] = []
        disabled_names: list[str] = []

        if spec is not None:
            if spec.mcp_servers:
                agent_raw_servers, disabled_names = _agent_mcp_mapping_to_list(
                    spec.mcp_servers,
                    origin=spec.origin,
                )
        elif self._wire_adapter.engine_id == "v2":
            legacy_spec = load_kiro_agent_spec(
                active_agent,
                cwd=cwd,
                include_home=not self._cli.is_custom_store,
                user_agents_dir=self._user_agents_dir(),
            )
            agent_raw_servers = list(legacy_spec["mcp_servers"])
            disabled_names = list(legacy_spec["disabled_mcp_servers"])

        agent_translated = [translate_mcp_server(s) for s in agent_raw_servers]
        return _dedupe_mcp_servers(
            [
                *[s for s in wire_translated if s.get("disabled") is not True],
                *[
                    s
                    for s in agent_translated
                    if s.get("disabled") is not True
                    and str(s.get("name") or "") not in disabled_names
                ],
            ]
        )

    def _apply_v2_legacy_agent_fallback(
        self,
        target_state: dict[str, Any],
        *,
        agent_name: str | None,
        cwd: str,
        allow_model_override: bool,
    ) -> None:
        """Populate v2 `.kiro/agents/<agent>.json` fallback fields when no v3 spec matches."""
        if self._wire_adapter.engine_id != "v2":
            return
        legacy_spec = load_kiro_agent_spec(
            agent_name,
            cwd=cwd,
            include_home=not self._cli.is_custom_store,
            user_agents_dir=self._user_agents_dir(),
        )
        target_state["agentPrompt"] = legacy_spec.get("prompt")
        target_state["agentTools"] = legacy_spec.get("tools")
        target_state["agentReadOnly"] = bool(legacy_spec.get("read_only"))
        pinned_model = legacy_spec.get("model")
        if isinstance(pinned_model, str) and pinned_model and allow_model_override:
            target_state["model"] = pinned_model
            target_state["switchedModel"] = pinned_model

    def _apply_agent_profile(
        self,
        target_state: dict[str, Any],
        *,
        agent_id: str | None,
        cwd: str,
        registry: CustomAgentRegistry,
        allow_model_override: bool,
        fallback_when_none: bool = False,
    ) -> None:
        """Reset and populate custom-agent or v2 legacy agent fields on `target_state`."""
        self._reset_custom_agent_fields(target_state)
        if agent_id:
            spec = registry.get(agent_id)
            if spec is not None:
                self._apply_custom_agent_fields(
                    target_state,
                    spec,
                    allow_model_override=allow_model_override,
                )
            else:
                self._apply_v2_legacy_agent_fallback(
                    target_state,
                    agent_name=agent_id,
                    cwd=cwd,
                    allow_model_override=allow_model_override,
                )
        elif fallback_when_none:
            self._apply_v2_legacy_agent_fallback(
                target_state,
                agent_name=None,
                cwd=cwd,
                allow_model_override=False,
            )

    async def _maybe_reset_backend_on_profile_change(
        self,
        session: dict[str, Any],
        *,
        prev_mcp: Any,
        prev_prompt: Any,
        old_agy_id: str,
    ) -> None:
        """Close and reset active backend session when agent MCP or system prompt changes."""
        if session.get("mcp_servers") == prev_mcp and session.get("agentPrompt") == prev_prompt:
            return
        if old_agy_id:
            await self._cli.close_backend_session(old_agy_id)
            session["agyConversationId"] = None
            session["reinjectAgentPrompt"] = True
            if not session.get("compactedSummary"):
                prior_turns = self._store.get_session_turns(str(session["sessionId"]))
                summary = _format_remaining_turns_summary(prior_turns)
                if summary:
                    session["compactedSummary"] = summary

    @staticmethod
    def _reset_custom_agent_fields(target_state: dict[str, Any]) -> None:
        """Clear all custom-agent prompt, tool, permission, and model switch fields."""
        target_state["agentPrompt"] = None
        target_state["agentTools"] = None
        target_state["agentExcludedTools"] = []
        target_state["agentAllowedTools"] = []
        target_state["agentPermissions"] = None
        target_state["agentReadOnly"] = False
        target_state["switchedModel"] = None

    @staticmethod
    def _apply_custom_agent_fields(
        target_state: dict[str, Any],
        spec: CustomAgentSpec | None,
        *,
        allow_model_override: bool,
    ) -> None:
        """Populate session agent prompt/tools/permissions/model fields from `spec`."""
        SessionManager._reset_custom_agent_fields(target_state)
        if spec is None:
            return
        target_state["agentPrompt"] = spec.prompt or None
        target_state["agentTools"] = list(spec.tools) if spec.tools is not None else None
        target_state["agentExcludedTools"] = list(spec.excluded_tools)
        target_state["agentAllowedTools"] = list(spec.allowed_tools)
        target_state["agentPermissions"] = copy.deepcopy(spec.permissions)
        target_state["agentReadOnly"] = spec.tools is not None and all(
            t in _READ_ONLY_TOOL_NAMES for t in spec.tools
        )
        candidate_model = (
            spec.model if spec.model and spec.model.strip().lower() != "auto" else None
        )
        if candidate_model and allow_model_override:
            target_state["model"] = candidate_model
            target_state["switchedModel"] = candidate_model
        else:
            target_state["switchedModel"] = None

    def _require_session(self, session_id: str) -> dict[str, Any]:
        """Fetch session state for `session_id` directly from SQLite or raise KeyError."""
        state = self._store.get_session(session_id)
        if state is None:
            raise KeyError(f"Session '{session_id}' not found")
        return state

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return an isolated copy of the SQLite-persisted state for `session_id`."""
        return copy.deepcopy(self._require_session(session_id))

    def list_sessions(self, *, cwd: str | None = None) -> list[dict[str, Any]]:
        """Return all persisted sessions from SQLite, optionally filtered by `cwd`."""
        return self._store.list_sessions(cwd=cwd)

    def get_session_mcp_server_names(self, session_id: str) -> list[str]:
        """Return ordered unique MCP server names configured for `session_id`."""
        if not self._mcp_enabled:
            return []
        session = self._store.get_session(session_id)
        if session is None:
            return []
        raw_servers = session.get("mcp_servers")
        if not isinstance(raw_servers, list):
            return []
        names: list[str] = []
        for entry in raw_servers:
            if isinstance(entry, dict) and entry.get("disabled") is not True:
                name = str(entry.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    def _resolve_session_probe_declarations(
        self,
        session: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build the ordered MCP server declarations to probe for `session`."""
        if not self._mcp_enabled:
            return []
        cwd = str(session.get("cwd") or "")
        registry = self.get_custom_agent_registry(
            cwd=cwd,
            custom_agents=session.get("custom_agents") or [],
        )
        active_agent = session.get("agent")
        spec = registry.get(str(active_agent)) if active_agent else None

        raw_wire = session.get("wire_mcp_servers")
        wire_entries = (
            [dict(s) for s in raw_wire if isinstance(s, dict) and s.get("name")]
            if isinstance(raw_wire, list)
            else []
        )
        agent_entries = (
            [
                _with_mcp_origin_meta(srv_name, srv_cfg, spec.origin)
                for srv_name, srv_cfg in spec.mcp_servers.items()
                if isinstance(srv_name, str) and isinstance(srv_cfg, Mapping)
            ]
            if spec is not None and spec.mcp_servers
            else []
        )

        merged_declarations = _dedupe_mcp_servers([*wire_entries, *agent_entries])
        if not merged_declarations:
            fallback = session.get("mcp_servers")
            if isinstance(fallback, list):
                merged_declarations = [
                    dict(s) for s in fallback if isinstance(s, dict) and s.get("name")
                ]
        return merged_declarations

    def get_session_probe_server_names(self, session_id: str) -> list[str]:
        """Return ordered MCP server names in the full probe roster for `session_id`."""
        session = self._store.get_session(session_id)
        if session is None:
            return []
        return [
            str(s.get("name") or "").strip()
            for s in self._resolve_session_probe_declarations(session)
            if str(s.get("name") or "").strip()
        ]

    def set_mcp_token_resolver(
        self,
        resolver: Callable[[str], Awaitable[str | None]] | None,
    ) -> None:
        """Configure an optional ephemeral Bearer token resolver for HTTP MCP probing."""
        self._mcp_token_resolver = resolver

    async def probe_session_mcp_servers(
        self,
        session_id: str,
        server_filter: str | None = None,
    ) -> list[McpServerStatus]:
        """Probe the effective MCP server roster for `session_id` via `McpCatalogProber`."""
        session = self._store.get_session(session_id)
        if session is None:
            return []

        cwd = str(session.get("cwd") or "")
        merged_declarations = self._resolve_session_probe_declarations(session)
        if server_filter:
            merged_declarations = [
                s for s in merged_declarations if str(s.get("name") or "") == server_filter
            ]
        if not merged_declarations:
            return []

        probe_declarations = (
            await self._attach_ephemeral_bearer_tokens(
                merged_declarations,
                self._mcp_token_resolver,
            )
            if self._mcp_token_resolver is not None
            else merged_declarations
        )
        session_env = _extract_session_mcp_env(
            session,
            session_id=session_id,
            state_db_path=self._store.db_path,
        )
        return await self._mcp_prober.probe_servers(
            probe_declarations,
            session_env=session_env,
            cwd=cwd or None,
        )

    @staticmethod
    async def _attach_ephemeral_bearer_tokens(
        declarations: Sequence[Mapping[str, Any]],
        token_resolver: Callable[[str], Awaitable[str | None]],
    ) -> list[dict[str, Any]]:
        """Attach ephemeral Bearer headers to enabled HTTP servers lacking `Authorization`."""
        enriched: list[dict[str, Any]] = []
        for decl in declarations:
            copy_decl = dict(decl)
            raw_url = copy_decl.get("url")
            if (
                copy_decl.get("disabled") is not True
                and isinstance(raw_url, str)
                and raw_url.strip().lower().startswith(("http://", "https://"))
            ):
                headers_map = _pairs_to_dict(copy_decl.get("headers"))
                if not any(k.lower() == "authorization" for k in headers_map):
                    token = await token_resolver(raw_url.strip())
                    if token:
                        copy_decl["headers"] = {**headers_map, "Authorization": f"Bearer {token}"}
            enriched.append(copy_decl)
        return enriched

    async def _prepare_derived_session(
        self,
        source: dict[str, Any],
        *,
        target_sid: str,
        cwd: str | None,
        mcp_servers: list[dict[str, Any]] | None,
        meta: dict[str, Any] | None,
        additional_directories: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Build merged draft state and available models for `load_session` or `fork_session`."""
        effective_cwd = (
            cwd if (isinstance(cwd, str) and cwd.strip()) else str(source.get("cwd") or "")
        )
        available_models = await self._resolve_available_models(cwd=effective_cwd or ".")
        effective_wire_mcp = (
            copy.deepcopy(mcp_servers)
            if mcp_servers is not None
            else copy.deepcopy(source.get("wire_mcp_servers", source.get("mcp_servers", [])))
        )
        wire_custom_agents = _extract_wire_custom_agents(meta)
        effective_custom_agents = (
            wire_custom_agents
            if wire_custom_agents is not None
            else copy.deepcopy(source.get("custom_agents", []))
        )
        effective_meta = (
            copy.deepcopy(meta) if meta is not None else copy.deepcopy(source.get("meta", {}))
        )
        effective_dirs = (
            list(additional_directories)
            if additional_directories is not None
            else [str(d) for d in (source.get("additionalDirectories") or []) if str(d).strip()]
        )

        agent = source.get("agent") or self._default_agent
        self._cli.discard_cancelled(target_sid)
        registry = self.get_custom_agent_registry(
            cwd=effective_cwd,
            custom_agents=effective_custom_agents,
        )
        combined_mcp = self._merge_session_mcp(
            cwd=effective_cwd,
            raw_wire_mcp=effective_wire_mcp,
            active_agent=str(agent) if agent else None,
            registry=registry,
        )

        draft_state: dict[str, Any] = {
            **copy.deepcopy(source),
            "sessionId": target_sid,
            "cwd": effective_cwd,
            "additionalDirectories": effective_dirs,
            "mcp_servers": combined_mcp,
            "wire_mcp_servers": effective_wire_mcp,
            "custom_agents": effective_custom_agents,
            "meta": effective_meta,
            "auth_config": self._build_auth_config(source.get("auth_config"), agent),
            "mode": str(source.get("mode", "auto")),
            "agent": agent,
            "model": str(source.get("model", self._default_model or DEFAULT_MODEL_ID)),
            "reasoning_effort": str(
                source.get("reasoning_effort", self._default_effort or DEFAULT_REASONING_EFFORT)
            ),
            "cancelled": bool(source.get("cancelled", False)),
        }
        if agent and wire_custom_agents is not None:
            self._apply_custom_agent_fields(
                draft_state,
                registry.get(str(agent)),
                allow_model_override=not bool(source.get("modelPinnedByUser", False)),
            )
        return draft_state, available_models

    async def create_session(
        self,
        *,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        additional_directories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new ACP session (`session/new`) in SQLite and return its descriptor."""
        initial_mode = "approve" if os.environ.get("GOOSE_MODE") else "auto"
        available_models = await self._resolve_available_models(cwd=cwd)
        if self._default_model:
            initial_model = self._default_model
        elif available_models:
            initial_model = self._wire_adapter.resolve_concrete_model(
                available_models,
                DEFAULT_MODEL_ID,
            )
        else:
            initial_model = DEFAULT_MODEL_ID

        isolated_mcp = copy.deepcopy(mcp_servers) if mcp_servers is not None else []
        isolated_meta = copy.deepcopy(meta) if meta is not None else {}
        isolated_dirs = list(additional_directories) if additional_directories is not None else []
        wire_custom_agents = _extract_wire_custom_agents(isolated_meta) or []

        sid = str(uuid.uuid4())
        self._cli.discard_cancelled(sid)
        registry = self.get_custom_agent_registry(cwd=cwd, custom_agents=wire_custom_agents)
        combined_mcp = self._merge_session_mcp(
            cwd=cwd,
            raw_wire_mcp=isolated_mcp,
            active_agent=self._default_agent,
            registry=registry,
        )

        draft_state: dict[str, Any] = {
            "sessionId": sid,
            "agyConversationId": None,
            "cwd": cwd,
            "additionalDirectories": isolated_dirs,
            "mcp_servers": combined_mcp,
            "wire_mcp_servers": isolated_mcp,
            "custom_agents": wire_custom_agents,
            "meta": isolated_meta,
            "auth_config": self._build_auth_config(None, self._default_agent),
            "mode": initial_mode,
            "agent": self._default_agent,
            "model": initial_model,
            "modelPinnedByUser": bool(self._default_model),
            "policyPromptVersion": _POLICY_PROMPT_VERSION,
            "reasoning_effort": self._default_effort or DEFAULT_REASONING_EFFORT,
            "cancelled": False,
        }
        self._apply_agent_profile(
            draft_state,
            agent_id=self._default_agent,
            cwd=cwd,
            registry=registry,
            allow_model_override=not bool(self._default_model),
            fallback_when_none=True,
        )

        state = self._store.upsert_session(draft_state)
        self._process_session_ids.add(sid)
        _write_dsh_gate_marker_if_configured()
        return self._build_descriptor(state, available_models=available_models)

    async def load_session(
        self,
        *,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        additional_directories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Resume an existing ACP session (`session/load`) from SQLite storage."""
        await anyio.sleep(0)
        state = self._require_session(session_id)
        draft_state, available_models = await self._prepare_derived_session(
            state,
            target_sid=session_id,
            cwd=cwd,
            mcp_servers=mcp_servers,
            meta=meta,
            additional_directories=additional_directories,
        )
        old_agy_id = str(state.get("agyConversationId") or "").strip()
        if old_agy_id and (
            draft_state["mcp_servers"] != state.get("mcp_servers")
            or draft_state.get("agentPrompt") != state.get("agentPrompt")
            or draft_state.get("additionalDirectories", [])
            != state.get("additionalDirectories", [])
        ):
            await self._cli.close_backend_session(old_agy_id)
            self._cli.evict_backend_session(old_agy_id)

        restored_state = self._store.upsert_session(draft_state)
        self._process_session_ids.add(session_id)
        _write_dsh_gate_marker_if_configured()
        return self._build_descriptor(restored_state, available_models=available_models)

    async def fork_session(
        self,
        source_session_id: str = "",
        *,
        cwd: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        message_id: str | None = None,
        additional_directories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fork an existing ACP session (`session/fork`) into a new session ID."""
        await anyio.sleep(0)
        source = self._require_session(source_session_id)
        new_sid = str(uuid.uuid4())
        draft_state, available_models = await self._prepare_derived_session(
            source,
            target_sid=new_sid,
            cwd=cwd,
            mcp_servers=mcp_servers,
            meta=meta,
            additional_directories=additional_directories,
        )

        source_turns = self._store.get_session_turns(source_session_id)
        turns_to_copy = list(source_turns)
        if message_id:
            matched_idx = next(
                (
                    idx
                    for idx, turn in enumerate(source_turns)
                    if str(turn.get("user_message_id") or "") == message_id
                ),
                None,
            )
            if matched_idx is not None:
                turns_to_copy = source_turns[: matched_idx + 1]

        draft_state.update(
            {
                "agyConversationId": None,
                "cancelled": False,
                "contextPaths": copy.deepcopy(source.get("contextPaths", [])),
                "knowledgeEntries": copy.deepcopy(source.get("knowledgeEntries", [])),
                "compactedSummary": (
                    _format_remaining_turns_summary(turns_to_copy)
                    if turns_to_copy
                    else source.get("compactedSummary")
                ),
                "reinjectAgentPrompt": (
                    True if turns_to_copy else bool(source.get("reinjectAgentPrompt", False))
                ),
            }
        )

        forked_state = self._store.upsert_session(draft_state)
        self._process_session_ids.add(new_sid)
        for idx, turn in enumerate(turns_to_copy):
            self._store.append_session_turn(
                session_id=new_sid,
                user_message_id=str(turn.get("user_message_id") or ""),
                prompt_blocks=copy.deepcopy(turn.get("prompt_blocks") or []),
                assistant_text=strip_agy_retry_warnings(str(turn.get("assistant_text") or "")),
                turn_idx=idx,
            )
        _write_dsh_gate_marker_if_configured()
        return self._build_descriptor(forked_state, available_models=available_models)

    def _is_valid_agent_mode(
        self,
        mode_id: str,
        *,
        registry: CustomAgentRegistry,
    ) -> bool:
        """Return whether `mode_id` resolves to a registered or virtual custom agent."""
        if not mode_id:
            return False
        if registry.get(mode_id) is not None or mode_id == self._default_agent:
            return True
        return self._wire_adapter.engine_id == "v2" and mode_id.startswith("kirocrew")

    async def set_mode(self, *, session_id: str, mode_id: str) -> dict[str, Any]:
        """Update session approval mode or agent profile (`session/set_mode`) in SQLite."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        prev_mcp = copy.deepcopy(session.get("mcp_servers", []))
        prev_prompt = session.get("agentPrompt")
        prev_switched = session.get("switchedModel")
        old_agy_id = str(session.get("agyConversationId") or "").strip()

        cwd = str(session.get("cwd", ""))
        registry = self.get_custom_agent_registry(
            cwd=cwd,
            custom_agents=session.get("custom_agents") or [],
        )

        spec = registry.get(mode_id)
        if spec is None and mode_id in VALID_MODE_IDS:
            session["mode"] = mode_id
            session["agent"] = None
            self._reset_custom_agent_fields(session)
            active_agent: str | None = None
        else:
            if not self._is_valid_agent_mode(mode_id, registry=registry):
                raise ValueError(f"Mode '{mode_id}' not found")
            session["agent"] = mode_id
            active_agent = mode_id
            self._apply_agent_profile(
                session,
                agent_id=mode_id,
                cwd=cwd,
                registry=registry,
                allow_model_override=not bool(session.get("modelPinnedByUser", False)),
            )

        if (
            prev_switched
            and not session.get("switchedModel")
            and not bool(session.get("modelPinnedByUser", False))
        ):
            session["model"] = self._default_model or self._wire_adapter.resolve_concrete_model(
                self._store.load_models() or AVAILABLE_MODELS,
                DEFAULT_MODEL_ID,
            )

        raw_wire = session.get("wire_mcp_servers", [])
        session["mcp_servers"] = self._merge_session_mcp(
            cwd=cwd,
            raw_wire_mcp=raw_wire if isinstance(raw_wire, list) else [],
            active_agent=active_agent,
            registry=registry,
        )
        await self._maybe_reset_backend_on_profile_change(
            session,
            prev_mcp=prev_mcp,
            prev_prompt=prev_prompt,
            old_agy_id=old_agy_id,
        )
        self._store.upsert_session(session)
        return {}

    def _session_stream_kwargs(
        self,
        session_id: str,
        session: dict[str, Any],
        *,
        user_text: str,
    ) -> dict[str, Any]:
        """Build common `stream_prompt` keyword arguments for `session`."""
        effective_session = (
            session if self._mcp_enabled else {**session, "mcp_servers": [], "wire_mcp_servers": []}
        )
        raw_dirs = session.get("additionalDirectories")
        clean_dirs = (
            [str(d).strip() for d in raw_dirs if isinstance(d, str) and d.strip()]
            if isinstance(raw_dirs, list)
            else []
        )
        return {
            "session_id": session_id,
            "mcp_servers": effective_session.get("mcp_servers"),
            "system_prompt_append": _build_system_prompt_append(
                effective_session,
                user_text=user_text,
            ),
            "route_spawn_run": _session_has_kirocrew_spawn_run(effective_session),
            "known_mcp_servers": self.get_session_mcp_server_names(session_id),
            "env": _extract_session_mcp_env(
                effective_session,
                session_id=session_id,
                state_db_path=self._store.db_path,
            ),
            **({"additional_directories": clean_dirs} if clean_dirs else {}),
        }

    async def compact_session(
        self,
        *,
        session_id: str,
        instructions: str = "",
    ) -> dict[str, Any]:
        """Summarize and reset `session_id` conversation history for compaction."""
        session = self._store.get_session(session_id)
        if session is None:
            return {
                "status": "completed",
                "summary": "Context is already empty.",
                "contextTokens": 0,
                "inputTokens": 1,
                "outputTokens": 1,
            }

        prev_tokens = int(session.get("contextTokens", 0))
        agy_conv_id = session.get("agyConversationId")
        if not agy_conv_id:
            existing_summary = session.get("compactedSummary")
            if isinstance(existing_summary, str) and existing_summary.strip():
                summary = existing_summary.strip()
                compacted_tokens = max(1, len(summary.split()) * 2)
            else:
                summary = "Context is already empty."
                compacted_tokens = 0
            session["contextTokens"] = compacted_tokens
            session["reinjectAgentPrompt"] = True
            self._store.upsert_session(session)
            return {
                "status": "completed",
                "summary": summary,
                "contextTokens": compacted_tokens,
                "inputTokens": 1,
                "outputTokens": max(1, len(summary.split())),
            }

        compact_prompt = (
            "Summarize our conversation so far into a concise handoff summary "
            "preserving goals, key decisions, modified files, and next steps."
        )
        if instructions.strip():
            compact_prompt = f"{compact_prompt}\n\nAdditional instructions: {instructions.strip()}"

        raw_model = str(session.get("model", DEFAULT_MODEL_ID))
        model = resolve_model_for_effort(raw_model, "low", self._store.load_models()) or raw_model
        try:
            outcome = await self._cli.stream_prompt(
                cwd=str(session.get("cwd", "")),
                prompt=compact_prompt,
                content_blocks=[{"type": "text", "text": compact_prompt}],
                conversation_id=str(agy_conv_id),
                model=model,
                mode="read-only",
                effort="low",
                **self._session_stream_kwargs(session_id, session, user_text=compact_prompt),
            )
        except Exception as exc:
            return _failed_compaction(str(exc) or "Compaction failed.", prev_tokens)

        if outcome.stop_reason == "cancelled":
            return _failed_compaction("Compaction cancelled.", prev_tokens)
        if outcome.error:
            return _failed_compaction(outcome.error, prev_tokens)

        summary = (outcome.response or "").strip() or "Conversation context compacted."
        compacted_tokens = max(1, len(summary.split()) * 2)
        await self._cli.close_backend_session(str(agy_conv_id))
        session["agyConversationId"] = None
        session["compactedSummary"] = summary
        session["contextTokens"] = compacted_tokens
        session["reinjectAgentPrompt"] = True
        self._store.upsert_session(session)
        return {
            "status": "completed",
            "summary": summary,
            "contextTokens": compacted_tokens,
            "inputTokens": max(1, int(outcome.usage.get("input_tokens", 1))),
            "outputTokens": max(1, int(outcome.usage.get("output_tokens", len(summary.split())))),
        }

    def _apply_model_selection(self, session: dict[str, Any], value: str) -> None:
        """Apply user model choice, treating `'auto'` as unpinned default resolution."""
        cleaned = value.strip()
        if cleaned.lower() == "auto":
            available_models = self._store.load_models() or AVAILABLE_MODELS
            session["model"] = self._default_model or self._wire_adapter.resolve_concrete_model(
                available_models,
                "auto",
            )
            session["modelPinnedByUser"] = False
        else:
            session["model"] = value
            session["modelPinnedByUser"] = True

    async def set_model(self, *, session_id: str, model_id: str) -> dict[str, Any]:
        """Update active model (`session/set_model`) in SQLite and pin user override."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        self._apply_model_selection(session, model_id)
        self._store.upsert_session(session)
        return {}

    async def set_config_option(
        self,
        *,
        session_id: str,
        config_id: str,
        value: str,
    ) -> dict[str, Any]:
        """Update session configuration option (`session/set_config_option`) in SQLite."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        if config_id == "model":
            self._apply_model_selection(session, value)
        elif config_id == "mode":
            if value not in VALID_MODE_IDS:
                raise ValueError(f"Mode '{value}' not found")
            session["mode"] = value
        elif config_id in REASONING_EFFORT_CONFIG_IDS:
            if value not in self._wire_adapter.supported_effort_levels:
                supported = ", ".join(self._wire_adapter.supported_effort_levels)
                raise ValueError(
                    f"Unsupported effort level '{value}' (expected one of: {supported})"
                )
            session["reasoning_effort"] = value
            resolved_sibling = resolve_model_for_effort(
                str(session.get("model", DEFAULT_MODEL_ID)),
                value,
                self._store.load_models(),
            )
            if resolved_sibling:
                session["model"] = resolved_sibling
        else:
            raise ValueError(f"Config option '{config_id}' not found")
        updated = self._store.upsert_session(session)
        return {
            "configOptions": self._build_config_options_for_state(
                model=str(updated.get("model", DEFAULT_MODEL_ID)),
                mode=str(updated.get("mode", "auto")),
                reasoning_effort=str(updated.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
            )
        }

    async def prompt(
        self,
        *,
        session_id: str,
        prompt_blocks: list[dict[str, Any]],
        on_thought: Callable[[str], Awaitable[None]] | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool | ToolPermissionDecision]]
        | None = None,
        on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None = None,
        on_usage: Callable[[int], Awaitable[None]] | None = None,
        on_steer_consumed: Callable[[str], Awaitable[None]] | None = None,
        on_subagent_list: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Execute a streaming turn (`session/prompt`) via the `agy acp` subprocess."""
        self._require_session(session_id)
        lock = self._turn_locks.setdefault(session_id, anyio.Lock())
        async with lock:
            if self.consume_cancellation(session_id):
                return {"stopReason": "cancelled"}

            session = self._require_session(session_id)
            cwd = str(session.get("cwd", ""))
            model = str(session.get("model", DEFAULT_MODEL_ID))
            agent_read_only = bool(session.get("agentReadOnly"))
            mode = str(session.get("mode", "auto"))
            if agent_read_only:
                mode = "read-only"
            elif os.environ.get("GOOSE_MODE") == "approve":
                mode = "approve"
            skip_permissions = mode == "auto" and not agent_read_only
            effort = str(session.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
            agy_conv_id = session.get("agyConversationId")
            conversation_id = str(agy_conv_id) if agy_conv_id else None
            compacted_summary = session.get("compactedSummary")
            reinjected_prompt = bool(session.get("reinjectAgentPrompt"))

            prefix_sections: list[str] = []
            context_paths = session.get("contextPaths")
            if isinstance(context_paths, list) and context_paths:
                clean_paths = [str(p).strip() for p in context_paths if str(p).strip()]
                if clean_paths:
                    prefix_sections.append(
                        "<session_context_paths>\n"
                        + "\n".join(clean_paths)
                        + "\n</session_context_paths>"
                    )
            consumed_compacted_summary = False
            if (
                conversation_id is None
                and isinstance(compacted_summary, str)
                and compacted_summary.strip()
            ):
                prefix_sections.append(f"[Compacted Session Summary]\n{compacted_summary.strip()}")
                consumed_compacted_summary = True

            effective_session = (
                session
                if self._mcp_enabled
                else {**session, "mcp_servers": [], "wire_mcp_servers": []}
            )
            needs_policy_stamp = (
                int(session.get("policyPromptVersion") or 0) < _POLICY_PROMPT_VERSION
            )
            if conversation_id is not None and needs_policy_stamp:
                incoming_text = "\n".join(
                    str(b.get("text") or "")
                    for b in prompt_blocks
                    if isinstance(b, Mapping) and b.get("type") == "text"
                )
                policy_sections = _build_policy_guidance_sections(
                    effective_session,
                    user_text=incoming_text,
                )
                if policy_sections:
                    prefix_sections.extend(policy_sections)

            prefix_str = "\n\n".join(prefix_sections).strip()
            formatted_prompt, content_blocks = _build_acp_content_blocks(
                prompt_blocks,
                cwd=cwd,
                prefix_text=prefix_str,
            )
            raw_user_text = (
                formatted_prompt.removeprefix(prefix_str).lstrip()
                if prefix_str
                else formatted_prompt
            )
            dedup_user_text = "" if reinjected_prompt else raw_user_text
            allow_policy_recovery = not _is_zero_tool_session(session) and bool(
                _compute_disallowed_native_tools(session)
            )

            outcome = None
            with self.turn_cancel_scope(session_id) as scope:
                outcome = await self._cli.stream_prompt(
                    cwd=cwd,
                    prompt=formatted_prompt,
                    content_blocks=content_blocks,
                    conversation_id=conversation_id,
                    model=model,
                    mode=mode,
                    effort=effort,
                    skip_permissions=skip_permissions,
                    allow_policy_recovery=allow_policy_recovery,
                    on_thought=on_thought,
                    on_text=on_text,
                    on_tool_call=on_tool_call,
                    on_tool_update=on_tool_update,
                    on_usage=on_usage,
                    on_steer_consumed=on_steer_consumed,
                    on_subagent_list=on_subagent_list,
                    **self._session_stream_kwargs(session_id, session, user_text=dedup_user_text),
                )

            if (
                scope.cancel_called
                or self.consume_cancellation(session_id)
                or outcome is None
                or outcome.stop_reason == "cancelled"
            ):
                cancelled_payload: dict[str, Any] = {"stopReason": "cancelled"}
                if outcome is not None:
                    cancel_ctx = int(outcome.usage.get("context_tokens", 0))
                    latest_cancel = self._store.get_session(session_id)
                    if latest_cancel is not None and (outcome.conversation_id or cancel_ctx > 0):
                        if outcome.conversation_id:
                            latest_cancel["agyConversationId"] = outcome.conversation_id
                        if cancel_ctx > 0:
                            latest_cancel["contextTokens"] = cancel_ctx
                        self._store.upsert_session(latest_cancel)
                    if outcome.usage:
                        cancelled_payload["usage"] = outcome.usage
                return cancelled_payload

            ctx_tokens = int(outcome.usage.get("context_tokens", 0))
            fallback_from = outcome.fallback_from_model
            effective_resolved_model = outcome.resolved_model
            if fallback_from and not effective_resolved_model:
                effective_resolved_model = self._wire_adapter.resolve_concrete_model(
                    self._store.load_models() or [],
                    "",
                )

            has_fallback = bool(
                fallback_from
                and effective_resolved_model
                and fallback_from != effective_resolved_model
            )
            latest = self._store.get_session(session_id)
            if latest is not None and (
                outcome.conversation_id
                or effective_resolved_model
                or ctx_tokens > 0
                or consumed_compacted_summary
                or reinjected_prompt
                or needs_policy_stamp
            ):
                if outcome.conversation_id:
                    latest["agyConversationId"] = outcome.conversation_id
                if consumed_compacted_summary:
                    latest["compactedSummary"] = None
                if reinjected_prompt:
                    latest["reinjectAgentPrompt"] = False
                if needs_policy_stamp:
                    latest["policyPromptVersion"] = _POLICY_PROMPT_VERSION
                if has_fallback:
                    latest["model"] = effective_resolved_model
                elif (
                    effective_resolved_model
                    and not latest.get("modelPinnedByUser")
                    and latest.get("model") in {DEFAULT_MODEL_ID, "auto", ""}
                ):
                    latest["model"] = effective_resolved_model
                if ctx_tokens > 0:
                    latest["contextTokens"] = ctx_tokens
                self._store.upsert_session(latest)

            turn_payload: dict[str, Any] = {
                "stopReason": outcome.stop_reason,
                "response": outcome.response,
                "usage": outcome.usage,
                "conversationId": outcome.conversation_id,
            }
            if has_fallback:
                turn_payload["fallbackFromModel"] = fallback_from
                turn_payload["resolvedModel"] = effective_resolved_model
            if outcome.error:
                turn_payload["error"] = outcome.error
            return turn_payload

    def is_cancelled(self, session_id: str) -> bool:
        """Return whether `session_id` has a pending cancellation in SQLite."""
        session = self._store.get_session(session_id)
        if session is not None:
            return bool(session.get("cancelled", False))
        return bool(self._store.get_kv(f"cancelled:{session_id}"))

    def consume_cancellation(self, session_id: str) -> bool:
        """Clear and return whether `session_id` was marked cancelled in SQLite."""
        if not self.is_cancelled(session_id):
            return False
        session = self._store.get_session(session_id)
        if session is not None:
            session["cancelled"] = False
            self._store.upsert_session(session)
        self._store.set_kv(f"cancelled:{session_id}", False)
        return True

    def register_cancel_scope(self, session_id: str, scope: anyio.CancelScope) -> None:
        """Register an active `anyio.CancelScope` for `session_id`."""
        self._cancel_scopes[session_id] = scope

    def unregister_cancel_scope(
        self,
        session_id: str,
        scope: anyio.CancelScope | None = None,
    ) -> None:
        """Unregister the active `anyio.CancelScope` for `session_id`."""
        current = self._cancel_scopes.get(session_id)
        if current is not None and (scope is None or current is scope):
            self._cancel_scopes.pop(session_id, None)

    @contextlib.contextmanager
    def turn_cancel_scope(self, session_id: str) -> Generator[anyio.CancelScope]:
        """Context manager registering an `anyio.CancelScope` for an active turn."""
        with anyio.CancelScope() as scope:
            self.register_cancel_scope(session_id, scope)
            try:
                yield scope
            finally:
                self.unregister_cancel_scope(session_id, scope)

    async def steer(
        self,
        *,
        session_id: str,
        message: str | None = None,
        prompt_blocks: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Queue a mid-turn steer (`_session/steer` or `_session/steering`) into an active turn."""
        if not self._cli.is_turn_active(session_id):
            return False

        session = self._store.get_session(session_id)
        cwd = str(session.get("cwd") or "") if session is not None else None

        if message is not None:
            formatted_prompt, content_blocks = _format_steer_message(message, cwd=cwd)
            raw_message = message
            if not formatted_prompt.strip():
                return False
        else:
            formatted_prompt, content_blocks = _build_acp_content_blocks(
                prompt_blocks or [],
                cwd=cwd,
            )
            raw_message = formatted_prompt
            if not any(
                block.get("type") != "text" or bool(str(block.get("text") or "").strip())
                for block in content_blocks
            ):
                return False

        return await self._cli.steer_session(
            session_id,
            raw_message=raw_message,
            prompt=formatted_prompt,
            content_blocks=content_blocks,
        )

    def clear_session_steers(self, session_id: str) -> list[str]:
        """Clear and return queued mid-turn steers for `session_id`."""
        return self._cli.clear_steers(session_id)

    def handle_session_context(
        self,
        session_id: str,
        *,
        subcommand: str = "",
        path: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Handle `_kiro/session/context` (`add`, `remove`, `clear`) on `contextPaths`."""
        _ = force
        state = self._require_session(session_id)
        context_paths = [
            str(p).strip() for p in (state.get("contextPaths") or []) if str(p).strip()
        ]
        sub = (subcommand or "").strip().lower()
        clean_path = path.strip() if isinstance(path, str) else ""

        if sub == "add":
            if not clean_path:
                return {"success": False, "message": "Path is required to add context."}
            if clean_path not in context_paths:
                context_paths.append(clean_path)
            state["contextPaths"] = context_paths
            self._store.upsert_session(state)
            return {"success": True, "message": f"Added context path: {clean_path}"}

        if sub == "remove":
            if not clean_path:
                return {"success": False, "message": "Path is required to remove context."}
            if clean_path not in context_paths:
                return {"success": False, "message": f"Context path not found: {clean_path}"}
            state["contextPaths"] = [p for p in context_paths if p != clean_path]
            self._store.upsert_session(state)
            return {"success": True, "message": f"Removed context path: {clean_path}"}

        if sub == "clear":
            state["contextPaths"] = []
            self._store.upsert_session(state)
            return {"success": True, "message": "Cleared all session context paths."}

        return {"success": False, "message": f"Unsupported context subcommand: {subcommand}"}

    def handle_knowledge(
        self,
        session_id: str,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Handle `_kiro/knowledge` (`show`, `add`, `remove`, `update`, `clear`, `cancel`)."""
        state = self._require_session(session_id)
        merged: dict[str, Any] = {
            **(dict(params) if isinstance(params, Mapping) else {}),
            **kwargs,
        }
        sub = (
            str(merged.get("subcommand") or merged.get("action") or merged.get("command") or "show")
            .strip()
            .lower()
        )
        entries: list[dict[str, Any]] = [
            dict(e) for e in (state.get("knowledgeEntries") or []) if isinstance(e, dict)
        ]

        if sub == "show":
            return {
                "success": True,
                "entries": [
                    {
                        "name": str(
                            e.get("name") or Path(str(e.get("path") or "")).name or "knowledge"
                        ),
                        "path": str(e.get("path") or ""),
                        "item_count": int(e.get("item_count", 1)),
                    }
                    for e in entries
                ],
            }

        if sub == "add":
            raw_path, raw_name = merged.get("path"), merged.get("name")
            path_str = str(raw_path).strip() if isinstance(raw_path, str) else ""
            if not path_str:
                return {"success": False, "message": "Path is required to add knowledge entry."}
            name_str = (
                str(raw_name).strip()
                if isinstance(raw_name, str) and str(raw_name).strip()
                else (Path(path_str).name or path_str)
            )
            item_count = (
                int(merged["item_count"])
                if merged.get("item_count") is not None
                else _count_knowledge_items(path_str)
            )
            new_entry = {"name": name_str, "path": path_str, "item_count": item_count}
            match_idx = next(
                (
                    idx
                    for idx, e in enumerate(entries)
                    if e.get("name") == name_str or e.get("path") == path_str
                ),
                None,
            )
            if match_idx is not None:
                entries[match_idx] = new_entry
            else:
                entries.append(new_entry)
            state["knowledgeEntries"] = entries
            self._store.upsert_session(state)
            return {"success": True, "message": f"Added knowledge entry '{name_str}'."}

        target = str(merged.get("name") or merged.get("path") or merged.get("id") or "").strip()
        if sub == "remove":
            if not target:
                return {
                    "success": False,
                    "message": "Name or path is required to remove knowledge entry.",
                }
            remaining = [e for e in entries if e.get("name") != target and e.get("path") != target]
            if len(remaining) == len(entries):
                return {"success": False, "message": f"Knowledge entry not found: {target}"}
            state["knowledgeEntries"] = remaining
            self._store.upsert_session(state)
            return {"success": True, "message": f"Removed knowledge entry '{target}'."}

        if sub == "update":
            if not target:
                if entries:
                    return {"success": True, "message": "Updated knowledge entries."}
                return {
                    "success": False,
                    "message": "Name or path is required to update knowledge entry.",
                }
            match_idx = next(
                (
                    idx
                    for idx, e in enumerate(entries)
                    if e.get("name") == target or e.get("path") == target
                ),
                None,
            )
            if match_idx is None:
                return {"success": False, "message": f"Knowledge entry not found: {target}"}
            existing = entries[match_idx]
            updated_name = (
                str(merged["name"]).strip()
                if isinstance(merged.get("name"), str) and str(merged["name"]).strip()
                else str(existing.get("name") or target)
            )
            updated_path = (
                str(merged["path"]).strip()
                if isinstance(merged.get("path"), str) and str(merged["path"]).strip()
                else str(existing.get("path") or "")
            )
            entries[match_idx] = {
                "name": updated_name,
                "path": updated_path,
                "item_count": int(merged.get("item_count", existing.get("item_count", 1))),
            }
            state["knowledgeEntries"] = entries
            self._store.upsert_session(state)
            return {"success": True, "message": f"Updated knowledge entry '{updated_name}'."}

        if sub == "clear":
            state["knowledgeEntries"] = []
            self._store.upsert_session(state)
            return {"success": True, "message": "Cleared all knowledge entries."}

        if sub == "cancel":
            return {"success": True, "message": "Cancelled knowledge indexing operation."}

        return {"success": False, "message": f"Unsupported knowledge subcommand: {sub}"}

    def handle_code_intelligence(
        self,
        session_id: str,
        *,
        subcommand: str = "status",
    ) -> dict[str, Any]:
        """Handle `_kiro/codeIntelligence` (`init`, `status`)."""
        state = self._require_session(session_id)
        meta = dict(state.get("meta") or {})
        sub = (subcommand or "").strip().lower()

        if sub == "init":
            meta["codeIntelligenceInitialized"] = True
            state["meta"] = meta
            self._store.upsert_session(state)
            return {"success": True, "message": "Code intelligence initialized."}

        if sub == "status":
            cwd_str = str(state.get("cwd") or "").strip()
            languages: list[str] = []
            if cwd_str:
                cwd_path = Path(cwd_str)
                for marker, lang in (
                    ("pyproject.toml", "python"),
                    ("tsconfig.json", "typescript"),
                    ("package.json", "typescript"),
                    ("Cargo.toml", "rust"),
                ):
                    if lang not in languages and (cwd_path / marker).exists():
                        languages.append(lang)
            languages = languages or ["python"]
            return {
                "success": True,
                "status": {
                    "initialized": bool(meta.get("codeIntelligenceInitialized", False)),
                    "languages": languages,
                    "lspServers": [{"name": lang, "status": "ready"} for lang in languages],
                },
            }

        return {
            "success": False,
            "message": f"Unsupported codeIntelligence subcommand: {subcommand}",
        }

    def revert_checkpoint_by_message_id(
        self,
        session_id: str,
        *,
        message_id: str,
    ) -> dict[str, Any]:
        """Handle `_kiro/checkpoint/revertMultiple` by truncating turns from `message_id`."""
        state = self._require_session(session_id)
        clean_msg_id = (message_id or "").strip()
        if not clean_msg_id:
            raise ValueError("messageId must be a non-empty string")
        old_agy_id = str(state.get("agyConversationId") or "").strip()
        if old_agy_id:
            self._cli.evict_backend_session(old_agy_id)
        self._store.revert_turns_from_message_id(session_id, clean_msg_id)
        remaining_turns = self._store.get_session_turns(session_id)
        state["compactedSummary"] = _format_remaining_turns_summary(remaining_turns)
        state["agyConversationId"] = None
        state["reinjectAgentPrompt"] = True
        self._store.upsert_session(state)
        return {"success": True}

    prompt_session = prompt

    async def cancel(self, *, session_id: str) -> bool:
        """Cancel an in-flight prompt turn (`session/cancel`) and persist in SQLite."""
        await self._cli.cancel_session(session_id)
        session = self._store.get_session(session_id)
        if session is not None:
            session["cancelled"] = True
            self._store.upsert_session(session)
        else:
            self._store.set_kv(f"cancelled:{session_id}", True)
        scope = self._cancel_scopes.get(session_id)
        if scope is not None:
            scope.cancel()
        return True

    async def evict_session(self, *, session_id: str) -> dict[str, Any]:
        """Evict resident runtime handles for `session_id` while preserving SQLite state & turns."""
        return await self.close_session(session_id=session_id, delete_persisted=False)

    async def terminate_session(
        self,
        *,
        session_id: str,
        delete_persisted: bool = False,
    ) -> dict[str, Any]:
        """Terminate `session_id` and optionally delete persisted SQLite session + turns."""
        return await self.close_session(
            session_id=session_id,
            delete_persisted=delete_persisted,
        )

    async def close_session(
        self,
        *,
        session_id: str,
        delete_persisted: bool = False,
    ) -> dict[str, Any]:
        """Close and clean up an active session (`session/close`, `terminate`, `delete`)."""
        await self._cli.cancel_session(session_id)
        self._cli.discard_cancelled(session_id)
        scope = self._active_cancel_scopes.pop(session_id, None) or self._cancel_scopes.pop(
            session_id, None
        )
        if scope is not None:
            scope.cancel()
        self._mcp_status_cache.pop(session_id, None)
        self._process_session_ids.discard(session_id)
        self._store.set_kv(f"cancelled:{session_id}", False)
        existing = self._store.get_session(session_id)
        old_agy_id = (
            str(existing.get("agyConversationId") or "").strip() if existing is not None else ""
        )
        if old_agy_id:
            await self._cli.close_backend_session(old_agy_id)
        if delete_persisted:
            self._store.delete_session(session_id)
            self._turn_locks.pop(session_id, None)
        elif existing is not None and existing.get("cancelled", False):
            existing["cancelled"] = False
            self._store.upsert_session(existing)
        return {}
