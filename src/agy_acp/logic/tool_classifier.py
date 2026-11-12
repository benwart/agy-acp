"""Engine-neutral tool classification, MCP server translation, and CLI model/binary helpers."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agy_acp.logic.wire.base import synthesize_task_command

VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})

FALLBACK_MODELS: list[dict[str, str]] = [
    {
        "modelId": "gemini-3.7-pro",
        "name": "Gemini 3.7 Pro",
        "description": "Google Antigravity Pro model",
    },
    {
        "modelId": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "description": "Google Antigravity Flash model",
    },
    {
        "modelId": "gemini-3.1-pro-high",
        "name": "Gemini 3.1 Pro (High)",
        "description": "Gemini 3.1 Pro (High)",
    },
    {
        "modelId": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "description": "Claude Sonnet 4.6",
    },
]

PER_SESSION_MCP_ENV_KEYS: frozenset[str] = frozenset(
    {
        "KIROCREW_STUB_SESSION_TOKEN",
        "KIROCREW_SESSION_KEY",
        "KIROCREW_HOST_PID",
        "KIROCREW_CHANNEL_ID",
        "KIROCREW_PORT",
        "KIROCREW_HOME",
    }
)

_PATH_ARGUMENT_KEYS: tuple[str, ...] = (
    "path",
    "Path",
    "AbsolutePath",
    "TargetFile",
    "DirectoryPath",
    "SearchPath",
    "SearchDirectory",
    "TargetDirectory",
    "File",
    "file_path",
    "filePath",
)

_EXACT_BUILTIN_TABLE: dict[str, tuple[str, str, str]] = {
    "run_command": ("run_command", "execute_bash", "execute"),
    "execute_command": ("run_command", "execute_bash", "execute"),
    "execute_bash": ("run_command", "execute_bash", "execute"),
    "command_execution": ("run_command", "execute_bash", "execute"),
    "manage_task": ("run_command", "execute_bash", "execute"),
    "send_command_input": ("run_command", "execute_bash", "execute"),
    "read_file": ("read_file", "fs_read", "read"),
    "view_file": ("read_file", "fs_read", "read"),
    "view_file_outline": ("read_file", "fs_read", "read"),
    "view_code_item": ("read_file", "fs_read", "read"),
    "fs_read": ("read_file", "fs_read", "read"),
    "list_dir": ("list_directory", "fs_read", "read"),
    "list_directory": ("list_directory", "fs_read", "read"),
    "write_file": ("fs_write", "fs_write", "edit"),
    "write_to_file": ("fs_write", "fs_write", "edit"),
    "create_file": ("fs_write", "fs_write", "edit"),
    "fs_write": ("fs_write", "fs_write", "edit"),
    "edit_file": ("str_replace", "fs_write", "edit"),
    "replace_file_content": ("str_replace", "fs_write", "edit"),
    "multi_replace_file_content": ("str_replace", "fs_write", "edit"),
    "str_replace": ("str_replace", "fs_write", "edit"),
    "delete_file": ("delete_file", "fs_write", "delete"),
    "remove_file": ("delete_file", "fs_write", "delete"),
    "grep_search": ("grep_search", "grep", "search"),
    "grep": ("grep_search", "grep", "search"),
    "find_by_name": ("file_search", "glob", "search"),
    "glob": ("file_search", "glob", "search"),
    "file_search": ("file_search", "glob", "search"),
    "search_web": ("remote_web_search", "web_search", "fetch"),
    "web_search": ("remote_web_search", "web_search", "fetch"),
    "remote_web_search": ("remote_web_search", "web_search", "fetch"),
    "read_url_content": ("web_fetch", "web_fetch", "fetch"),
    "read_url": ("web_fetch", "web_fetch", "fetch"),
    "web_fetch": ("web_fetch", "web_fetch", "fetch"),
    "invoke_subagent": ("invoke_sub_agent", "use_subagent", "other"),
    "invoke_sub_agent": ("invoke_sub_agent", "use_subagent", "other"),
    "use_subagent": ("invoke_sub_agent", "use_subagent", "other"),
    "define_subagent": ("invoke_sub_agent", "use_subagent", "other"),
    "manage_subagents": ("invoke_sub_agent", "use_subagent", "other"),
    "send_message": ("invoke_sub_agent", "use_subagent", "other"),
    "todo_list": ("todo_list", "todo_list", "other"),
    "task_boundary": ("todo_list", "todo_list", "other"),
}


@dataclass(frozen=True)
class ToolCallClassification:
    """Engine-neutral KAS v3 and Kiro v2 classification for an `agy` tool step."""

    kas_builtin_id: str
    v2_tool_name: str
    kiro_tool_id: str
    acp_kind: str
    title: str
    locations: list[dict[str, Any]] = field(default_factory=list)
    mcp_server: str = ""
    mcp_tool: str = ""
    command: str = ""
    subagent_target: str = ""
    diff_content: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallEvent:
    """Normalized tool call or tool call update emitted during an `agy` turn."""

    tool_call_id: str
    title: str
    kind: str
    status: str
    raw_input: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    locations: list[dict[str, Any]] = field(default_factory=list)
    kiro_tool_name: str = ""
    mcp_server_name: str = ""
    kiro_tool_id: str = ""
    kas_builtin_id: str = ""
    v2_tool_name: str = ""
    acp_kind: str = ""
    mcp_server: str = ""
    mcp_tool: str = ""
    command: str = ""
    subagent_target: str = ""
    diff_content: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.acp_kind and self.kind:
            object.__setattr__(self, "acp_kind", self.kind)
        if not self.mcp_server and self.mcp_server_name:
            object.__setattr__(self, "mcp_server", self.mcp_server_name)
        elif not self.mcp_server_name and self.mcp_server:
            object.__setattr__(self, "mcp_server_name", self.mcp_server)
        if not self.v2_tool_name and self.kiro_tool_name:
            object.__setattr__(self, "v2_tool_name", self.kiro_tool_name)
        elif not self.kiro_tool_name and self.v2_tool_name:
            object.__setattr__(self, "kiro_tool_name", self.v2_tool_name)
        if self.mcp_server and not self.mcp_tool and self.kiro_tool_name:
            object.__setattr__(self, "mcp_tool", self.kiro_tool_name)
        if not self.command and self.kind == "execute":
            raw_cmd = self.raw_input.get("CommandLine") or self.raw_input.get("command") or ""
            if isinstance(raw_cmd, str) and raw_cmd:
                object.__setattr__(self, "command", raw_cmd)
        if not self.kas_builtin_id and not self.mcp_server:
            if self.kiro_tool_id and not self.kiro_tool_id.startswith("@"):
                object.__setattr__(self, "kas_builtin_id", self.kiro_tool_id)
            elif self.kind == "execute":
                object.__setattr__(self, "kas_builtin_id", "run_command")


@dataclass(frozen=True)
class PromptOutcome:
    """Result of executing a streaming prompt turn via the `agy` CLI."""

    conversation_id: str
    stop_reason: str
    response: str
    usage: dict[str, int]
    raw_result: dict[str, Any] = field(default_factory=dict)
    resolved_model: str = ""
    fallback_from_model: str = ""
    error: str = ""


@dataclass(frozen=True)
class QueuedSteer:
    """Mid-turn steering message queued for an active `agy` turn."""

    raw_message: str
    prompt: str
    temp_files: list[Path] = field(default_factory=list)


def _pairs_to_dict(items: Any) -> dict[str, str]:
    """Convert a list of `{"name": ..., "value": ...}` dicts or a mapping into `dict[str, str]`."""
    if isinstance(items, dict):
        return {str(k): str(v) for k, v in items.items()}
    if not isinstance(items, list):
        return {}
    result: dict[str, str] = {}
    for entry in items:
        if isinstance(entry, dict) and "name" in entry and "value" in entry:
            result[str(entry["name"])] = str(entry["value"])
    return result


def translate_mcp_server(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate an ACP MCP server definition into normalized stdio or HTTP config."""
    server_type = str(raw.get("type", "stdio")).lower()
    if server_type == "http" or "url" in raw:
        translated: dict[str, Any] = {
            "type": "http",
            "name": str(raw.get("name", "")),
            "url": str(raw.get("url", "")),
            "headers": _pairs_to_dict(raw.get("headers", [])),
        }
    else:
        translated = {
            "type": "stdio",
            "name": str(raw.get("name", "")),
            "command": str(raw.get("command", "")),
            "args": [str(arg) for arg in raw.get("args", [])],
            "env": _pairs_to_dict(raw.get("env", [])),
        }
    for extra_key in ("disabled", "disabledTools", "displayName", "cwd", "_meta"):
        if extra_key in raw:
            translated[extra_key] = raw[extra_key]
    return translated


def _dedupe_mcp_servers(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate MCP server entries by non-empty `name`, preserving first-seen precedence."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for srv in servers:
        name = str(srv.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(srv)
    return unique


def resolve_agy_binary(explicit_bin: str | Path | None = None) -> str:
    """Resolve the path to the `agy` CLI executable."""
    if explicit_bin is not None:
        explicit_path = Path(explicit_bin)
        if explicit_path.exists():
            return str(explicit_path)

    env_bin = os.environ.get("AGY_BIN")
    if env_bin:
        env_path = Path(env_bin)
        if env_path.exists():
            return str(env_path)

    local_bin = Path.home() / ".local" / "bin" / "agy"
    if local_bin.exists():
        return str(local_bin)

    which_bin = shutil.which("agy")
    if which_bin:
        return which_bin

    return "agy"


def _split_model_effort_suffix(model: str | None) -> tuple[str, str] | None:
    """Return `(base_family, effort_suffix)` if `model` ends in a recognized `-<effort>` suffix."""
    if not model:
        return None
    cleaned = model.strip()
    base, sep, suffix = cleaned.rpartition("-")
    if sep and base and suffix.lower() in VALID_EFFORTS:
        return base, suffix.lower()
    return None


def resolve_model_for_effort(
    model: str | None,
    effort: str | None,
    available_models: Sequence[Mapping[str, Any]] = (),
) -> str | None:
    """Switch an effort-suffixed `agy` model ID to its matching `-<effort>` sibling when cached."""
    if not model or not effort:
        return model
    normalized_effort = effort.strip().lower()
    if normalized_effort not in VALID_EFFORTS:
        return model
    parsed = _split_model_effort_suffix(model)
    if parsed is None or parsed[1] == normalized_effort:
        return model
    candidate_id = f"{parsed[0]}-{normalized_effort}"
    known_ids = {
        str(entry.get("modelId") or "").strip()
        for entry in available_models
        if isinstance(entry, Mapping) and entry.get("modelId")
    }
    return candidate_id if candidate_id in known_ids else model


def _extract_mcp_identity(
    raw_name: str,
    arguments: dict[str, Any],
    *,
    known_mcp_servers: Sequence[str] = (),
) -> tuple[str, str]:
    """Return `(server_name, tool_name)` when the step invokes an MCP tool, else `('', '')`."""
    lower_name = raw_name.lower()
    if lower_name == "call_mcp_tool" or ("ServerName" in arguments and "ToolName" in arguments):
        server = str(
            arguments.get("ServerName")
            or arguments.get("server_name")
            or arguments.get("server")
            or ""
        ).strip()
        tool = str(
            arguments.get("ToolName") or arguments.get("tool_name") or arguments.get("tool") or ""
        ).strip()
        if server and tool:
            return server, tool

    if raw_name.startswith("mcp__"):
        rest = raw_name[len("mcp__") :]
        server, sep, tool = rest.rpartition("__")
        if sep and server and tool:
            return server, tool
    elif raw_name.startswith("mcp_"):
        rest = raw_name[len("mcp_") :]
        if known_mcp_servers:
            ordered = sorted(
                {s.strip() for s in known_mcp_servers if isinstance(s, str) and s.strip()},
                key=len,
                reverse=True,
            )
            for srv in ordered:
                prefix = f"{srv}_"
                if rest.startswith(prefix) and len(rest) > len(prefix):
                    return srv, rest[len(prefix) :]
        server, sep, tool = rest.partition("_")
        if sep and server and tool:
            return server, tool

    return "", ""


def _extract_first_str(arguments: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty string value from `arguments` matching `keys`."""
    for key in keys:
        val = arguments.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _extract_diff_content(
    kas_builtin_id: str,
    path_val: str,
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build ACP `diff` content blocks for `str_replace` and `fs_write` tool calls."""
    if not path_val:
        return []
    if kas_builtin_id == "str_replace":
        target_keys = ("Target", "TargetContent", "old_str", "oldText")
        replacement_keys = ("Replacement", "ReplacementContent", "new_str", "newText")
        has_target = any(isinstance(arguments.get(k), str) for k in target_keys)
        has_replacement = any(isinstance(arguments.get(k), str) for k in replacement_keys)
        if has_target and has_replacement:
            old_text = next(
                (str(arguments[k]) for k in target_keys if isinstance(arguments.get(k), str)),
                "",
            )
            new_text = next(
                (str(arguments[k]) for k in replacement_keys if isinstance(arguments.get(k), str)),
                "",
            )
            return [{"type": "diff", "path": path_val, "oldText": old_text, "newText": new_text}]
        chunks = arguments.get("ReplacementChunks")
        if isinstance(chunks, list):
            return [
                {
                    "type": "diff",
                    "path": path_val,
                    "oldText": str(chunk.get("TargetContent", "")),
                    "newText": str(chunk.get("ReplacementContent", "")),
                }
                for chunk in chunks
                if isinstance(chunk, dict)
                and ("TargetContent" in chunk or "ReplacementContent" in chunk)
            ]
    elif kas_builtin_id == "fs_write":
        for content_key in ("Content", "CodeContent", "content"):
            val = arguments.get(content_key)
            if isinstance(val, str):
                return [{"type": "diff", "path": path_val, "oldText": "", "newText": val}]
    return []


def _resolve_builtin_triple(
    lower_name: str,
    arguments: dict[str, Any],
) -> tuple[str, str, str]:
    """Return `(kas_builtin_id, v2_tool_name, acp_kind)` for a built-in tool step."""
    exact = _EXACT_BUILTIN_TABLE.get(lower_name)
    if exact is not None:
        return exact
    if any(k in lower_name for k in ("delete_file", "remove_file")):
        return "delete_file", "fs_write", "delete"
    if any(k in lower_name for k in ("list_dir", "list_directory")):
        return "list_directory", "fs_read", "read"
    if any(k in lower_name for k in ("search_web", "web_search", "remote_web_search")):
        return "remote_web_search", "web_search", "fetch"
    if any(k in lower_name for k in ("read_url", "web_fetch")):
        return "web_fetch", "web_fetch", "fetch"
    if any(k in lower_name for k in ("write_file", "write_to_file", "create_file")):
        return "fs_write", "fs_write", "edit"
    if any(
        k in lower_name
        for k in ("code_action", "edit_file", "replace", "sed_file", "notebook_edit")
    ):
        return "str_replace", "fs_write", "edit"
    if any(
        k in lower_name
        for k in (
            "view_file",
            "read_file",
            "view_code",
            "list_resources",
            "read_resource",
            "read_browser",
        )
    ):
        return "read_file", "fs_read", "read"
    if "grep" in lower_name:
        return "grep_search", "grep", "search"
    if any(k in lower_name for k in ("find_by_name", "file_search", "glob", "find")):
        return "file_search", "glob", "search"
    if any(k in lower_name for k in ("run_command", "bash", "exec", "shell")):
        return "run_command", "execute_bash", "execute"
    if any(k in lower_name for k in ("subagent", "send_message")):
        return "invoke_sub_agent", "use_subagent", "other"
    if any(k in lower_name for k in ("todo_list", "task_boundary")):
        return "todo_list", "todo_list", "other"
    if "command" in arguments or "CommandLine" in arguments:
        return "run_command", "execute_bash", "execute"
    fallback = lower_name or "tool"
    return fallback, fallback, "other"


def _build_builtin_title(
    acp_kind: str,
    raw_name: str,
    path_val: str,
    cmd_val: str,
    arguments: dict[str, Any],
) -> str:
    """Format a human-readable ACP tool call title for a built-in step."""
    label = raw_name or "Tool call"
    if acp_kind == "execute":
        return f"Running `{cmd_val[:80]}`" if cmd_val else "Execute command"
    if acp_kind == "read":
        return f"Viewing {path_val}" if path_val else f"Reading ({label})"
    if acp_kind == "edit":
        return f"Editing {path_val}" if path_val else f"Editing ({label})"
    if acp_kind == "delete":
        return f"Deleting {path_val}" if path_val else f"Deleting ({label})"
    if acp_kind == "search":
        query = _extract_first_str(arguments, ("query", "Query", "Pattern", "pattern")) or path_val
        return f"Searching {query}" if query else f"Searching ({label})"
    if acp_kind == "fetch":
        target = _extract_first_str(arguments, ("Url", "url", "query", "Query")) or path_val
        return f"Fetching {target}" if target else f"Fetching ({label})"
    return label


def classify_tool_call_step(
    step_type: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    known_mcp_servers: Sequence[str] = (),
) -> ToolCallClassification:
    """Classify an `agy` step into normalized KAS v3 and Kiro v2 tool metadata."""
    raw_name = (tool_name or step_type).strip()
    mcp_server, mcp_tool = _extract_mcp_identity(
        raw_name,
        arguments,
        known_mcp_servers=known_mcp_servers,
    )
    if mcp_server and mcp_tool:
        title = (
            "spawn_run"
            if (mcp_server == "kirocrew-core" and mcp_tool == "spawn_run")
            else f"@{mcp_server}/{mcp_tool}"
        )
        return ToolCallClassification(
            kas_builtin_id="",
            v2_tool_name=mcp_tool,
            kiro_tool_id=f"@{mcp_server}/{mcp_tool}",
            acp_kind="other",
            title=title,
            locations=[],
            mcp_server=mcp_server,
            mcp_tool=mcp_tool,
        )

    kas_builtin_id, v2_tool_name, acp_kind = _resolve_builtin_triple(raw_name.lower(), arguments)
    path_val = _extract_first_str(arguments, _PATH_ARGUMENT_KEYS)
    locations = [{"path": path_val}] if path_val else []
    cmd_val = (
        (
            _extract_first_str(arguments, ("CommandLine", "command", "cmd", "Command"))
            or synthesize_task_command(raw_name, arguments)
        )
        if acp_kind == "execute"
        else ""
    )
    subagent_target = (
        _extract_first_str(
            arguments,
            ("TypeName", "AgentName", "Role", "type_name", "agent_name", "role"),
        ).strip()
        if kas_builtin_id == "invoke_sub_agent"
        else ""
    )
    diff_content = _extract_diff_content(kas_builtin_id, path_val, arguments)
    title = _build_builtin_title(acp_kind, tool_name or step_type, path_val, cmd_val, arguments)
    return ToolCallClassification(
        kas_builtin_id=kas_builtin_id,
        v2_tool_name=v2_tool_name,
        kiro_tool_id=kas_builtin_id,
        acp_kind=acp_kind,
        title=title,
        locations=locations,
        command=cmd_val,
        subagent_target=subagent_target,
        diff_content=diff_content,
    )


def _resolve_kiro_tool_identity(
    step_type: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    known_mcp_servers: Sequence[str] = (),
) -> tuple[str, str, str]:
    """Resolve `(kiro_tool_name, mcp_server_name, kiro_tool_id)` for ACP `_meta.kiro`."""
    cls = classify_tool_call_step(
        step_type,
        tool_name,
        arguments,
        known_mcp_servers=known_mcp_servers,
    )
    return cls.v2_tool_name, cls.mcp_server, cls.kiro_tool_id
