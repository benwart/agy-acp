"""Async subprocess provider for driving the Antigravity CLI (`agy`) via anyio."""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import sys
import threading
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import anyio
import anyio.abc
from anyio.streams.text import TextReceiveStream

from agy_acp.logic.custom_agents import (
    CustomAgentSecurityError,
    resolve_contained_prompt_file,
    sanitize_project_agent_spec,
)
from agy_acp.providers.acp_logger import log_acp_event
from agy_acp.providers.state_store import SqliteStateStoreProvider

MAX_INLINE_PROMPT_BYTES: int = 64 * 1024
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

_MODE_TO_AGY_CLI: dict[str, str] = {
    "auto": "auto",
    "approve": "accept-edits",
    "read-only": "plan",
    "accept-edits": "accept-edits",
    "plan": "plan",
}

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

_READ_ONLY_KIRO_TOOLS: frozenset[str] = frozenset(
    {
        "fs_read",
        "grep",
        "glob",
        "code",
        "web_search",
        "web_fetch",
        "tool_search",
        "todo_list",
    }
)


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


def prepare_prompt_arg(
    prompt: str,
    *,
    session_id: str | None = None,
    temp_dir: Path | None = None,
) -> tuple[str, Path | None]:
    """Return `(prompt_arg, temp_file)` offloading prompts > 64 KiB to a 0600 temporary file."""
    encoded = prompt.encode("utf-8")
    if len(encoded) <= MAX_INLINE_PROMPT_BYTES:
        return prompt, None

    target_dir = (
        Path(temp_dir) if temp_dir is not None else (Path.home() / ".config" / "agy-acp" / "tmp")
    )
    target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        target_dir.chmod(0o700)
    tag = re.sub(r"[^A-Za-z0-9_-]", "", (session_id or "default"))[:8] or "default"
    filename = f"agy-acp-prompt-{tag}-{secrets.token_hex(8)}.md"
    file_path = target_dir / filename
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(file_path, flags, 0o600)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)
    file_path.chmod(0o600)
    instruction = (
        f"Please read the prompt and context in {file_path} using view_file "
        "and follow the instructions in it."
    )
    return instruction, file_path


def build_agy_args(
    *,
    cwd: str,
    prompt_arg: str,
    conversation_id: str | None = None,
    model: str | None = None,
    mode: str = "auto",
    effort: str | None = None,
    skip_permissions: bool | None = None,
    sandbox: bool = False,
    output_format: str = "stream-json",
) -> list[str]:
    """Construct `agy` CLI flags for non-interactive print mode (`-p`)."""
    args: list[str] = []
    if cwd:
        args.extend(["--add-dir", cwd])
    if conversation_id:
        args.extend(["--conversation", conversation_id])
    if model and model.strip().lower() != "auto":
        args.extend(["--model", model])

    cli_mode = _MODE_TO_AGY_CLI.get(mode, "auto")
    args.extend(["--mode", cli_mode])

    if effort:
        normalized_effort = effort.strip().lower()
        if normalized_effort in VALID_EFFORTS:
            args.extend(["--effort", normalized_effort])

    effective_skip = skip_permissions if skip_permissions is not None else (mode == "auto")
    if effective_skip:
        args.append("--dangerously-skip-permissions")
    if sandbox:
        args.append("--sandbox")

    args.extend(["--print-timeout", "0"])
    if output_format:
        args.extend(["--output-format", output_format])
    args.extend(["-p", prompt_arg])
    return args


build_command_args = build_agy_args


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


def load_kiro_agent_spec(
    agent_name: str | None = None,
    *,
    cwd: str | None = None,
    include_home: bool = True,
    user_agents_dir: Path | None = None,
) -> dict[str, Any]:
    """Load sanitized project agent fields and trusted user-scope `mcpServers`."""
    candidates: list[str] = []
    if agent_name and agent_name not in {"auto", "approve", "read-only"}:
        candidates.append(agent_name)
    if "kirocrew" not in candidates:
        candidates.append("kirocrew")

    # Each entry is (agents_dir, is_project_scope, containment_root)
    search_dirs: list[tuple[Path, bool, Path]] = []
    if cwd:
        with contextlib.suppress(OSError, RuntimeError):
            cwd_path = Path(cwd)
            resolved_cwd = cwd_path.resolve()
            kiro_dir = cwd_path / ".kiro"
            proj_agents_dir = kiro_dir / "agents"
            if (
                not kiro_dir.is_symlink()
                and not proj_agents_dir.is_symlink()
                and proj_agents_dir.is_dir()
                and proj_agents_dir.resolve().is_relative_to(resolved_cwd)
            ):
                search_dirs.append((proj_agents_dir, True, resolved_cwd))

    trusted_user_dir: Path | None = None
    if user_agents_dir is not None:
        trusted_user_dir = Path(user_agents_dir)
    elif include_home:
        with contextlib.suppress(RuntimeError, OSError):
            trusted_user_dir = Path.home() / ".kiro" / "agents"

    if trusted_user_dir is not None:
        with contextlib.suppress(OSError, RuntimeError):
            if not trusted_user_dir.is_symlink() and trusted_user_dir.is_dir():
                search_dirs.append((trusted_user_dir, False, trusted_user_dir.resolve()))

    pinned_model: str | None = None
    agent_prompt: str | None = None
    agent_tools: list[str] | None = None
    servers_by_name: dict[str, dict[str, Any]] = {}
    disabled_names: list[str] = []

    for cand in candidates:
        found_candidate = False
        for agents_dir, is_project_scope, containment_root in search_dirs:
            spec_path = agents_dir / f"{cand}.json"
            if spec_path.is_symlink() or not spec_path.is_file():
                continue
            with contextlib.suppress(
                OSError,
                json.JSONDecodeError,
                CustomAgentSecurityError,
                ValueError,
            ):
                resolved_spec = spec_path.resolve(strict=True)
                if not resolved_spec.is_relative_to(containment_root):
                    continue
                loaded_spec = json.loads(resolved_spec.read_text(encoding="utf-8"))
                if not isinstance(loaded_spec, dict):
                    continue
                if is_project_scope:
                    raw_spec = sanitize_project_agent_spec(loaded_spec, agents_dir=agents_dir)
                else:
                    raw_spec = dict(loaded_spec)
                    prompt_ref = raw_spec.get("prompt")
                    if isinstance(prompt_ref, str) and prompt_ref.startswith("file://"):
                        raw_spec["prompt"] = resolve_contained_prompt_file(
                            prompt_ref,
                            base_dir=agents_dir,
                        )
                found_candidate = True
                raw_model = raw_spec.get("model")
                if (
                    pinned_model is None
                    and isinstance(raw_model, str)
                    and raw_model.strip()
                    and raw_model.strip().lower() != "auto"
                ):
                    pinned_model = raw_model.strip()
                raw_prompt = raw_spec.get("prompt")
                if agent_prompt is None and isinstance(raw_prompt, str) and raw_prompt.strip():
                    agent_prompt = raw_prompt.strip()
                raw_tools = raw_spec.get("tools")
                if agent_tools is None and isinstance(raw_tools, list):
                    agent_tools = [str(t).strip() for t in raw_tools if str(t).strip()]
                if not is_project_scope:
                    raw_mcp = raw_spec.get("mcpServers")
                    if isinstance(raw_mcp, dict):
                        for srv_name, srv_cfg in raw_mcp.items():
                            if not isinstance(srv_name, str) or not srv_name:
                                continue
                            if not isinstance(srv_cfg, dict):
                                continue
                            if srv_cfg.get("disabled"):
                                if srv_name not in disabled_names:
                                    disabled_names.append(srv_name)
                            elif srv_name not in servers_by_name and srv_name not in disabled_names:
                                servers_by_name[srv_name] = {"name": srv_name, **srv_cfg}
                    elif isinstance(raw_mcp, list):
                        for item in raw_mcp:
                            if isinstance(item, dict) and item.get("name"):
                                sname = str(item["name"])
                                if item.get("disabled"):
                                    if sname not in disabled_names:
                                        disabled_names.append(sname)
                                elif sname not in servers_by_name and sname not in disabled_names:
                                    servers_by_name[sname] = dict(item)
        if found_candidate:
            break

    is_read_only = agent_tools is not None and all(
        tool in _READ_ONLY_KIRO_TOOLS for tool in agent_tools
    )
    return {
        "model": pinned_model,
        "prompt": agent_prompt,
        "tools": agent_tools,
        "read_only": is_read_only,
        "mcp_servers": list(servers_by_name.values()),
        "disabled_mcp_servers": disabled_names,
    }


def load_kiro_agent_mcp_servers(
    agent_name: str | None = None,
    *,
    cwd: str | None = None,
    include_home: bool = True,
    user_agents_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load `mcpServers` definitions strictly from trusted user-scope `.kiro/agents`."""
    spec = load_kiro_agent_spec(
        agent_name,
        cwd=cwd,
        include_home=include_home,
        user_agents_dir=user_agents_dir,
    )
    return list(spec["mcp_servers"])


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
        _extract_first_str(arguments, ("CommandLine", "command", "cmd", "Command"))
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


def _normalize_tool_arguments(
    arguments: dict[str, Any],
    *,
    tool_name: str = "",
) -> dict[str, Any]:
    """Normalize `agy` tool argument keys to ACP/KiroCrew standard keys (`command`, `path`)."""
    if tool_name.lower() == "call_mcp_tool" or (
        "ServerName" in arguments and "ToolName" in arguments
    ):
        inner = arguments.get("Arguments")
        if isinstance(inner, dict):
            return dict(inner)

    normalized = dict(arguments)
    if "command" not in normalized:
        cmd_val = _extract_first_str(normalized, ("CommandLine", "cmd", "Command"))
        if cmd_val:
            normalized["command"] = cmd_val
    if "path" not in normalized:
        path_val = _extract_first_str(normalized, _PATH_ARGUMENT_KEYS)
        if path_val:
            normalized["path"] = path_val
    summary = normalized.get("toolSummary")
    if isinstance(summary, str) and summary.strip():
        clean_summary = summary.strip()
        normalized.setdefault("description", clean_summary)
        normalized.setdefault("__toolUsePurpose", clean_summary)
    elif tool_name.lower() == "invoke_subagent":
        role = normalized.get("Role")
        if isinstance(role, str) and role.strip():
            clean_role = role.strip()
            normalized.setdefault("description", clean_role)
            normalized.setdefault("__toolUsePurpose", clean_role)
    return normalized


_NON_TOOL_STEP_TYPES: frozenset[str] = frozenset(
    {
        "agent_response",
        "planner_response",
        "user_input",
        "thinking",
        "text",
        "checkpoint",
        "ephemeral_message",
        "system_message",
        "system",
        "error",
        "error_message",
        "summary",
        "compaction",
    }
)

_ACTIVE_SUBAGENT_STATES: frozenset[str] = frozenset(
    {
        "running",
        "waiting_for_dependents",
        "pending",
        "queued",
        "starting",
        "in_progress",
        "working",
    }
)
_COMPLETED_SUBAGENT_STATES: frozenset[str] = frozenset({"idle", "completed", "done", "succeeded"})
_FAILED_SUBAGENT_STATES: frozenset[str] = frozenset({"failed", "error"})
_CANCELLED_SUBAGENT_STATES: frozenset[str] = frozenset(
    {"cancelled", "canceled", "killed", "stopped"}
)


def _extract_subagent_entries(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract `subagent_info.subagents` entries from an `agy` step update."""
    subagent_info = step.get("subagent_info")
    if not isinstance(subagent_info, dict):
        return []
    raw_list = subagent_info.get("subagents")
    if not isinstance(raw_list, list):
        return []
    return [item for item in raw_list if isinstance(item, dict)]


def _synthesize_subagent_arguments(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Synthesize `invoke_subagent` input fields from `subagent_info.subagents`."""
    if not entries:
        return {}
    first = entries[0]
    synthesized: dict[str, Any] = {}
    type_name = str(first.get("type_name") or first.get("type") or "").strip()
    role = str(first.get("role") or "").strip()
    initial_prompt = str(first.get("initial_prompt") or "").strip()
    if type_name:
        synthesized["TypeName"] = type_name
    if role:
        synthesized["Role"] = role
    if initial_prompt:
        synthesized["Prompt"] = initial_prompt
    return synthesized


def _format_subagent_tool_output(entries: list[dict[str, Any]]) -> str:
    """Format `subagent_info.subagents` into a JSON tool output string."""
    summary: list[dict[str, str]] = []
    for item in entries:
        entry: dict[str, str] = {
            "conversationId": str(item.get("conversation_id") or item.get("conversationId") or ""),
            "role": str(item.get("role") or ""),
            "type": str(item.get("type_name") or item.get("type") or ""),
        }
        log_uri = str(item.get("log_uri") or item.get("transcript") or "")
        if log_uri:
            entry["logUri"] = log_uri
        summary.append(entry)
    return json.dumps(summary[0] if len(summary) == 1 else summary)


def _parse_manage_subagents_list(output: str) -> list[dict[str, Any]] | None:
    """Parse the active subagent JSON array from `manage_subagents(Action='list')` output."""
    start = output.find("[")
    end = output.rfind("]")
    if start != -1 and end > start:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(output[start : end + 1])
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
    if "no active subagent" in output.lower():
        return []
    return None


class _SubagentRosterTracker:
    """Translate `agy` subagent steps into KiroCrew `_kiro.dev/subagent/list_update` snapshots."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._last_snapshot: list[dict[str, Any]] | None = None

    def _emit_if_changed(self) -> list[dict[str, Any]] | None:
        if not self._entries:
            return None
        snapshot = [
            {
                **entry,
                "status": dict(entry["status"]),
            }
            for entry in self._entries.values()
        ]
        if snapshot == self._last_snapshot:
            return None
        self._last_snapshot = [
            {
                **entry,
                "status": dict(entry["status"]),
            }
            for entry in snapshot
        ]
        return snapshot

    def record_spawn(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Register or refresh spawned subagents as `running`."""
        for item in entries:
            conv_id = str(item.get("conversation_id") or item.get("conversationId") or "").strip()
            if not conv_id:
                continue
            role = str(item.get("role") or "").strip()
            agent_name = str(item.get("type_name") or item.get("type") or "").strip()
            initial_query = str(item.get("initial_prompt") or "").strip()
            existing = self._entries.get(conv_id)
            if existing is not None:
                role = role or str(existing.get("role") or "")
                agent_name = agent_name or str(existing.get("agentName") or "")
                initial_query = initial_query or str(existing.get("initialQuery") or "")
            session_name = role or agent_name or conv_id
            self._entries[conv_id] = {
                "sessionId": conv_id,
                "sessionName": session_name,
                "role": role,
                "agentName": agent_name,
                "initialQuery": initial_query or session_name,
                "status": {"type": "running", "message": ""},
            }
        return self._emit_if_changed()

    def record_manage_step(
        self,
        raw_args: dict[str, Any],
        output: str,
    ) -> list[dict[str, Any]] | None:
        """Apply a completed `manage_subagents` step (`list`, `kill`, `kill_all`) to the roster."""
        action = str(raw_args.get("Action") or raw_args.get("action") or "").strip().lower()
        if action == "list":
            return self._record_manage_list(output)
        if action == "kill_all":
            for entry in self._entries.values():
                if entry["status"].get("type") == "running":
                    entry["status"] = {"type": "cancelled", "message": ""}
            return self._emit_if_changed()
        if action == "kill":
            raw_ids = raw_args.get("ConversationIds") or raw_args.get("conversationIds") or []
            target_ids = (
                {str(cid).strip() for cid in raw_ids if str(cid).strip()}
                if isinstance(raw_ids, list)
                else set()
            )
            for cid in target_ids:
                if cid in self._entries:
                    self._entries[cid]["status"] = {"type": "cancelled", "message": ""}
            return self._emit_if_changed()
        return None

    def _record_manage_list(self, output: str) -> list[dict[str, Any]] | None:
        listed = _parse_manage_subagents_list(output)
        if listed is None:
            return None
        seen_ids: set[str] = set()
        for item in listed:
            conv_id = str(item.get("conversationId") or item.get("conversation_id") or "").strip()
            if not conv_id:
                continue
            seen_ids.add(conv_id)
            raw_state = str(item.get("state") or "running").strip().lower()
            detail = str(item.get("stateDetail") or "").strip()
            if raw_state in _ACTIVE_SUBAGENT_STATES:
                msg = detail or (
                    "waiting for subagents" if raw_state == "waiting_for_dependents" else ""
                )
                status = {"type": "running", "message": msg}
            elif raw_state in _COMPLETED_SUBAGENT_STATES:
                status = {"type": "completed", "message": ""}
            elif raw_state in _FAILED_SUBAGENT_STATES:
                status = {"type": "failed", "message": detail}
            elif raw_state in _CANCELLED_SUBAGENT_STATES:
                status = {"type": "cancelled", "message": ""}
            else:
                status = {"type": "running", "message": detail}

            role = str(item.get("role") or "").strip()
            agent_name = str(item.get("type") or item.get("type_name") or "").strip()
            existing = self._entries.get(conv_id)
            if existing is not None:
                role = role or str(existing.get("role") or "")
                agent_name = agent_name or str(existing.get("agentName") or "")
                initial_query = str(existing.get("initialQuery") or "")
            else:
                initial_query = ""
            session_name = role or agent_name or conv_id
            self._entries[conv_id] = {
                "sessionId": conv_id,
                "sessionName": session_name,
                "role": role,
                "agentName": agent_name,
                "initialQuery": initial_query or session_name,
                "status": status,
            }

        for conv_id, entry in self._entries.items():
            if conv_id not in seen_ids and entry["status"].get("type") == "running":
                entry["status"] = {"type": "completed", "message": ""}
        return self._emit_if_changed()

    def finalize_turn(self, *, cancelled: bool) -> list[dict[str, Any]] | None:
        """Close any remaining `running` subagents at the end of a turn."""
        terminal_type = "cancelled" if cancelled else "completed"
        for entry in self._entries.values():
            if entry["status"].get("type") == "running":
                entry["status"] = {"type": terminal_type, "message": ""}
        return self._emit_if_changed()


def _classify_tool_step(
    step_type: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    known_mcp_servers: Sequence[str] = (),
) -> tuple[str, str, list[dict[str, Any]]]:
    """Map an `agy` step/tool name and arguments to ACP `(kind, title, locations)`."""
    cls = classify_tool_call_step(
        step_type,
        tool_name,
        arguments,
        known_mcp_servers=known_mcp_servers,
    )
    return cls.acp_kind, cls.title, cls.locations


def _write_atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via a 0600 temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp_path, flags, 0o600)
    try:
        try:
            os.write(fd, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp_path.chmod(0o600)
        os.replace(tmp_path, path)
        path.chmod(0o600)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def _signal_process_group(proc: anyio.abc.Process, sig: signal.Signals) -> None:
    """Send `sig` to the process group of `proc`, falling back to `proc` directly."""
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, sig)
            return
        except ProcessLookupError, PermissionError, OSError:
            pass
    with contextlib.suppress(ProcessLookupError, OSError):
        if sig == signal.SIGKILL:
            proc.kill()
        elif sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.send_signal(sig)


class AgyCliProvider:
    """Async subprocess driver for the `agy` CLI binary using `anyio`."""

    def __init__(
        self,
        *,
        bin_path: str | Path | None = None,
        save_dir: Path | None = None,
        state_store: SqliteStateStoreProvider | None = None,
        mcp_config_path: Path | None = None,
    ) -> None:
        self._explicit_bin = bin_path
        self._store = state_store or SqliteStateStoreProvider(save_dir)
        default_state_dir = Path.home() / ".config" / "agy-acp"
        self._is_custom_store = self._store.db_path.parent != default_state_dir
        if mcp_config_path is not None:
            self._mcp_config_path = Path(mcp_config_path)
        elif save_dir is not None:
            self._mcp_config_path = Path(save_dir) / "mcp_config.json"
        elif self._is_custom_store:
            self._mcp_config_path = self._store.db_path.parent / "mcp_config.json"
        else:
            env_mcp = os.environ.get("AGY_MCP_CONFIG_PATH")
            self._mcp_config_path = (
                Path(env_mcp)
                if env_mcp
                else (Path.home() / ".gemini" / "config" / "mcp_config.json")
            )
        self._active_processes: dict[str, anyio.abc.Process] = {}
        self._abort_timers: dict[str, threading.Timer] = {}
        self._cancelled_sessions: set[str] = set()
        self._active_turns: set[str] = set()
        self._steer_queues: dict[str, list[QueuedSteer]] = {}
        self._in_permission_request: dict[str, int] = {}
        self._turn_conv_ids: dict[str, str] = {}

    @property
    def is_custom_store(self) -> bool:
        """Return whether this provider uses a non-default isolated state directory."""
        return self._is_custom_store

    def _has_conflicting_stdio_server(
        self,
        server_name: str,
        candidate_entry: dict[str, Any],
        *,
        exclude_session_id: str | None = None,
    ) -> bool:
        """Return True if another persisted session defines `server_name` with different args."""
        for other in self._store.list_sessions():
            other_sid = str(other.get("sessionId") or "")
            if exclude_session_id and other_sid == exclude_session_id:
                continue
            for srv in other.get("mcp_servers") or []:
                if not isinstance(srv, dict) or str(srv.get("name") or "").strip() != server_name:
                    continue
                translated = translate_mcp_server(srv)
                if translated.get("type") == "http" or translated.get("url"):
                    continue
                other_entry: dict[str, Any] = {
                    "command": str(translated.get("command") or ""),
                    "args": [str(a) for a in translated.get("args", [])],
                }
                if isinstance(translated.get("env"), dict) and translated["env"]:
                    static_env = {
                        str(k): str(v)
                        for k, v in translated["env"].items()
                        if str(k) not in PER_SESSION_MCP_ENV_KEYS
                    }
                    if static_env:
                        other_entry["env"] = static_env
                if other_entry != candidate_entry:
                    return True
        return False

    def _build_stdio_mux_entry(self, server_name: str) -> dict[str, Any]:
        """Build a per-session stdio mux entry dispatching via `$AGY_ACP_SESSION_ID`."""
        src_root = str(Path(__file__).resolve().parents[2])
        mux_env: dict[str, str] = {}
        existing_py = os.environ.get("PYTHONPATH", "")
        if src_root not in existing_py.split(os.pathsep):
            mux_env["PYTHONPATH"] = (
                f"{src_root}{os.pathsep}{existing_py}" if existing_py else src_root
            )
        entry: dict[str, Any] = {
            "command": sys.executable,
            "args": [
                "-m",
                "agy_acp.providers.mcp_session_mux",
                "--db",
                str(self._store.db_path),
                "--server",
                server_name,
            ],
        }
        if mux_env:
            entry["env"] = mux_env
        return entry

    def sync_mcp_servers(
        self,
        mcp_servers: list[dict[str, Any]],
        *,
        remove_servers: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[str]:
        """Synchronize MCP server configurations into `agy`'s `mcp_config.json`."""
        if not mcp_servers and not remove_servers:
            return []
        existing_data: dict[str, Any] = {}
        if self._mcp_config_path.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                loaded = json.loads(self._mcp_config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing_data = loaded
        servers_map = existing_data.get("mcpServers")
        if not isinstance(servers_map, dict):
            servers_map = {}
            existing_data["mcpServers"] = servers_map

        changed = False
        for removed in remove_servers or []:
            if removed in servers_map:
                servers_map.pop(removed, None)
                changed = True

        synced: list[str] = []
        for srv in mcp_servers:
            translated = translate_mcp_server(srv)
            if translated.get("disabled") is True:
                continue
            name = str(translated.get("name") or "").strip()
            if not name:
                continue
            entry: dict[str, Any] = {}
            if translated.get("transport") == "http" or translated.get("url"):
                entry["serverUrl"] = str(translated.get("url") or "")
                if isinstance(translated.get("headers"), dict) and translated["headers"]:
                    entry["headers"] = dict(translated["headers"])
            else:
                cmd = str(translated.get("command") or "")
                if not cmd:
                    continue
                entry["command"] = cmd
                entry["args"] = [str(a) for a in translated.get("args", [])]
                has_any_env = bool(isinstance(translated.get("env"), dict) and translated["env"])
                if has_any_env:
                    static_env = {
                        str(k): str(v)
                        for k, v in translated["env"].items()
                        if str(k) not in PER_SESSION_MCP_ENV_KEYS
                    }
                    if static_env:
                        entry["env"] = static_env
                if session_id is not None or self._has_conflicting_stdio_server(
                    name,
                    entry,
                    exclude_session_id=session_id,
                ):
                    entry = self._build_stdio_mux_entry(name)
            if servers_map.get(name) != entry:
                servers_map[name] = entry
                changed = True
            synced.append(name)

        if changed:
            with contextlib.suppress(OSError):
                _write_atomic_private_json(self._mcp_config_path, existing_data)
        return synced

    @property
    def state_store(self) -> SqliteStateStoreProvider:
        """Return the underlying `SqliteStateStoreProvider`."""
        return self._store

    @property
    def bin_path(self) -> str:
        """Resolve the `agy` binary path dynamically."""
        return resolve_agy_binary(self._explicit_bin)

    async def discover_models(self, *, refresh: bool = False) -> list[dict[str, str]]:
        """Discover models via `agy models`, caching to SQLite `models_cache` table."""
        if not refresh:
            cached = self._store.load_models()
            if cached:
                return cached

        try:
            with anyio.fail_after(15.0):
                result = await anyio.run_process([self.bin_path, "models"], check=False)
            if result.returncode == 0:
                stdout = result.stdout.decode("utf-8", errors="replace")
                discovered: list[dict[str, str]] = []
                for raw_line in stdout.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    model_id = parts[0].strip()
                    display_name = parts[1].strip() if len(parts) > 1 else model_id
                    if model_id:
                        discovered.append(
                            {
                                "modelId": model_id,
                                "name": display_name,
                                "description": display_name,
                            }
                        )
                if discovered:
                    self._store.save_models(discovered)
                    return discovered
        except TimeoutError, OSError:
            pass

        return list(FALLBACK_MODELS)

    async def run_slash_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 30.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a non-interactive slash command (`agy -p <cmd> --output-format json`) via anyio."""
        normalized = command.strip()
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        args: list[str] = [self.bin_path]
        if cwd:
            args.extend(["--add-dir", cwd])
        args.extend(["--output-format", "json", "-p", normalized])

        child_env = dict(os.environ)
        if session_id:
            child_env["AGY_ACP_SESSION_ID"] = session_id
            child_env.setdefault("AGY_ACP_STATE_DB", str(self._store.db_path))

        with anyio.fail_after(timeout):
            result = await anyio.run_process(args, check=False, env=child_env)

        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        stderr = result.stderr.decode("utf-8", errors="replace").strip()

        if stdout:
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(stdout)
                if isinstance(parsed, dict):
                    return parsed
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    parsed_line = json.loads(line)
                    if isinstance(parsed_line, dict):
                        if "result" in parsed_line and isinstance(parsed_line["result"], dict):
                            return parsed_line["result"]
                        return parsed_line

        if result.returncode != 0:
            raise RuntimeError(
                f"agy command '{normalized}' exited with code {result.returncode}: "
                f"{stderr or stdout}"
            )
        return {
            "status": "SUCCESS",
            "response": stdout,
            "command": {"name": normalized.lstrip("/")},
        }

    async def start_or_resume_agent(
        self,
        *,
        session_id: str | None,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        model: str | None = None,
        mode: str = "auto",
        reasoning_effort: str = "medium",
        auth_config: dict[str, Any] | None = None,
    ) -> str:
        """Initialize or resume a session backed by SQLite state storage and the `agy` CLI."""
        await anyio.sleep(0)
        sid = session_id or str(uuid.uuid4())
        self._cancelled_sessions.discard(sid)
        existing = (self._store.get_session(sid) if session_id is not None else None) or {}

        raw_mcp = (
            list(mcp_servers)
            if mcp_servers is not None
            else list(existing.get("wire_mcp_servers", existing.get("mcp_servers", [])))
        )
        agent_name = str(
            (auth_config or {}).get("kiro_agent")
            or existing.get("agent")
            or os.environ.get("AGY_KIRO_AGENT")
            or "kirocrew"
        )
        spec: dict[str, Any]
        if (auth_config or {}).get("_engine_id") == "v3":
            spec = {
                "model": None,
                "prompt": None,
                "tools": None,
                "read_only": False,
                "mcp_servers": [],
                "disabled_mcp_servers": [],
            }
        else:
            spec = load_kiro_agent_spec(
                agent_name,
                cwd=cwd,
                include_home=not self._is_custom_store,
                user_agents_dir=(
                    (self._store.db_path.parent / ".kiro_user_agents")
                    if self._is_custom_store
                    else None
                ),
            )
        wire_translated = _dedupe_mcp_servers([translate_mcp_server(srv) for srv in raw_mcp])
        agent_translated = [translate_mcp_server(srv) for srv in spec["mcp_servers"]]
        combined_mcp = _dedupe_mcp_servers([*wire_translated, *agent_translated])
        wire_names = {str(s.get("name") or "") for s in wire_translated}
        other_session_names = {
            str(srv.get("name") or "")
            for other in self._store.list_sessions()
            if str(other.get("sessionId") or "") != sid
            for srv in (other.get("mcp_servers") or [])
            if isinstance(srv, dict) and srv.get("name")
        }
        removable_disabled = [
            name
            for name in spec["disabled_mcp_servers"]
            if name not in wire_names and name not in other_session_names
        ]
        if combined_mcp or removable_disabled:
            self.sync_mcp_servers(
                combined_mcp,
                remove_servers=removable_disabled,
                session_id=sid,
            )
        state: dict[str, Any] = {
            "sessionId": sid,
            "agyConversationId": existing.get("agyConversationId"),
            "cwd": cwd,
            "mcp_servers": combined_mcp,
            "wire_mcp_servers": wire_translated,
            "agentPrompt": spec.get("prompt"),
            "agentTools": spec.get("tools"),
            "agentReadOnly": bool(spec.get("read_only")),
            "model": model or str(existing.get("model", "gemini-3.7-pro")),
            "mode": mode,
            "reasoning_effort": reasoning_effort,
            "auth_config": auth_config or dict(existing.get("auth_config", {})),
        }
        self._store.upsert_session(state)
        return sid

    def is_turn_active(self, session_id: str) -> bool:
        """Return whether `session_id` currently has an active `stream_prompt` turn."""
        return session_id in self._active_turns or session_id in self._active_processes

    def _pop_queued_steers(self, session_id: str) -> list[QueuedSteer]:
        """Pop and return all queued mid-turn steers for `session_id`."""
        queue = self._steer_queues.pop(session_id, [])
        return list(queue)

    def _interrupt_process_for_steer(self, session_id: str) -> None:
        """Interrupt the active `agy` subprocess for `session_id` when safe to steer."""
        if self._in_permission_request.get(session_id, 0) > 0:
            return
        proc = self._active_processes.get(session_id)
        if proc is not None and proc.returncode is None:
            _signal_process_group(proc, signal.SIGINT)
            _signal_process_group(proc, signal.SIGTERM)

    async def steer_session(
        self,
        session_id: str,
        *,
        raw_message: str,
        prompt: str,
        temp_files: list[Path] | None = None,
    ) -> bool:
        """Queue a mid-turn steer and interrupt the active `agy` subprocess to continue."""
        await anyio.sleep(0)
        if not self.is_turn_active(session_id) or session_id in self._cancelled_sessions:
            return False
        self._steer_queues.setdefault(session_id, []).append(
            QueuedSteer(
                raw_message=raw_message,
                prompt=prompt,
                temp_files=list(temp_files or []),
            )
        )
        if self._turn_conv_ids.get(session_id):
            self._interrupt_process_for_steer(session_id)
        return True

    def _cancel_abort_timer(self, session_id: str) -> None:
        """Cancel and remove any pending SIGKILL escalation timer for `session_id`."""
        timer = self._abort_timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()

    def abort_active_process(self, session_id: str) -> bool:
        """Terminate an active `agy` subprocess group and escalate to SIGKILL after 0.25s."""
        proc = self._active_processes.get(session_id)
        if proc is not None and proc.returncode is None:
            _signal_process_group(proc, signal.SIGINT)
            _signal_process_group(proc, signal.SIGTERM)
            self._cancel_abort_timer(session_id)

            def _escalate() -> None:
                self._abort_timers.pop(session_id, None)
                if proc.returncode is None:
                    _signal_process_group(proc, signal.SIGKILL)

            timer = threading.Timer(0.25, _escalate)
            timer.daemon = True
            self._abort_timers[session_id] = timer
            timer.start()
            return True
        return False

    def discard_cancelled(self, session_id: str) -> None:
        """Clear any stale cancellation marker for `session_id`."""
        self._cancelled_sessions.discard(session_id)

    async def cancel_session(self, session_id: str) -> bool:
        """Send SIGINT/SIGTERM to an active `agy` subprocess group for `session_id`."""
        await anyio.sleep(0)
        active = self.is_turn_active(session_id)
        if active:
            self._cancelled_sessions.add(session_id)
        for steer in self._pop_queued_steers(session_id):
            for tmp in steer.temp_files:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
        proc = self._active_processes.get(session_id)
        if proc is not None and proc.returncode is None:
            _signal_process_group(proc, signal.SIGINT)
            _signal_process_group(proc, signal.SIGTERM)
            return True
        return active

    async def stream_prompt(
        self,
        *,
        cwd: str,
        prompt: str,
        session_id: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        mode: str = "auto",
        effort: str | None = None,
        skip_permissions: bool | None = None,
        sandbox: bool = False,
        env: dict[str, str] | None = None,
        known_mcp_servers: Sequence[str] = (),
        on_thought: Callable[[str], Awaitable[None]] | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
        on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None = None,
        on_usage: Callable[[int], Awaitable[None]] | None = None,
        on_steer_consumed: Callable[[str], Awaitable[None]] | None = None,
        on_subagent_list: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None,
    ) -> PromptOutcome:
        """Spawn `agy --output-format stream-json` via `anyio.open_process` and stream events."""
        sid = session_id or "default"
        if sid in self._cancelled_sessions:
            self._cancelled_sessions.discard(sid)
            return PromptOutcome(
                conversation_id=conversation_id or "",
                stop_reason="cancelled",
                response="",
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "cache_read_tokens": 0,
                    "total_tokens": 0,
                },
            )

        merged_env = {**os.environ, **(env or {})}
        if session_id:
            merged_env["AGY_ACP_SESSION_ID"] = session_id
            merged_env.setdefault("AGY_ACP_STATE_DB", str(self._store.db_path))

        effective_known_mcp: tuple[str, ...] = tuple(known_mcp_servers)
        if not effective_known_mcp and session_id:
            persisted_sess = self._store.get_session(session_id)
            if persisted_sess is not None:
                effective_known_mcp = tuple(
                    str(srv.get("name") or "").strip()
                    for srv in (persisted_sess.get("mcp_servers") or [])
                    if isinstance(srv, dict) and str(srv.get("name") or "").strip()
                )
        effective_cwd = cwd if cwd and Path(cwd).is_dir() else None

        resolved_conv_id = conversation_id or ""
        resolved_model = model or ""
        current_prompt = prompt
        current_model = model
        collected_texts: list[str] = []
        final_result_obj: dict[str, Any] = {}
        last_step_context_tokens: int = 0
        usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 0,
            "context_tokens": 0,
        }
        steer_temp_files: list[Path] = []
        active_temp_file: Path | None = None
        subagent_tracker = _SubagentRosterTracker()
        self._active_turns.add(sid)
        self._turn_conv_ids[sid] = resolved_conv_id

        try:
            while True:
                prompt_arg, temp_file = prepare_prompt_arg(current_prompt, session_id=sid)
                active_temp_file = temp_file
                args = build_agy_args(
                    cwd=cwd,
                    prompt_arg=prompt_arg,
                    conversation_id=resolved_conv_id or None,
                    model=current_model,
                    mode=mode,
                    effort=effort,
                    skip_permissions=skip_permissions,
                    sandbox=sandbox,
                    output_format="stream-json",
                )
                emitted_thoughts_by_step: dict[int, int] = {}
                emitted_texts_by_step: dict[int, int] = {}
                active_tool_calls: dict[int, ToolCallEvent] = {}
                completed_tool_steps: set[int] = set()
                rejected_tool_steps: set[int] = set()
                streamed_any_text = False
                plain_stdout_lines: list[str] = []
                stderr_chunks: list[str] = []
                run_result_obj: dict[str, Any] = {}

                async def _handle_step_update(
                    step: dict[str, Any],
                    *,
                    _emitted_thoughts: dict[int, int] = emitted_thoughts_by_step,
                    _emitted_texts: dict[int, int] = emitted_texts_by_step,
                    _active_tools: dict[int, ToolCallEvent] = active_tool_calls,
                    _completed_tools: set[int] = completed_tool_steps,
                    _rejected_tools: set[int] = rejected_tool_steps,
                    _run_prompt: str = current_prompt,
                ) -> None:
                    nonlocal streamed_any_text, last_step_context_tokens
                    step_idx = int(step.get("step_index", step.get("idx", 0)))
                    step_type = str(step.get("step_type", step.get("type_name", "")))
                    status_raw = str(step.get("status", step.get("state", "RUNNING"))).upper()

                    step_usage = step.get("usage")
                    step_ctx = 0
                    if isinstance(step_usage, dict):
                        step_ctx = (
                            int(step_usage.get("input_tokens", 0))
                            + int(step_usage.get("cache_read_tokens", 0))
                            + int(step_usage.get("output_tokens", 0))
                        )
                        if step_ctx > 0:
                            last_step_context_tokens = step_ctx

                    if self._store.get_session(sid) is not None:
                        self._store.upsert_step(
                            session_id=sid,
                            idx=step_idx,
                            step_type=step_type or "step",
                            status=status_raw,
                            data=step,
                        )
                    if step_ctx > 0 and on_usage is not None:
                        await on_usage(step_ctx)

                    thinking_delta = step.get("thinking_delta")
                    thinking_full = step.get("thinking")
                    if isinstance(thinking_delta, str) and thinking_delta:
                        if on_thought is not None:
                            await on_thought(thinking_delta)
                    elif isinstance(thinking_full, str) and thinking_full:
                        prev_len = _emitted_thoughts.get(step_idx, 0)
                        if len(thinking_full) > prev_len:
                            delta = thinking_full[prev_len:]
                            _emitted_thoughts[step_idx] = len(thinking_full)
                            if on_thought is not None:
                                await on_thought(delta)

                    tool_info = step.get("tool_info")
                    tool_name = str(
                        step.get("tool_name")
                        or (tool_info.get("tool_name") if isinstance(tool_info, dict) else "")
                        or (tool_info.get("name") if isinstance(tool_info, dict) else "")
                        or ""
                    )
                    is_tool_step = bool(
                        tool_name
                        or isinstance(tool_info, dict)
                        or (step_type and step_type.lower() not in _NON_TOOL_STEP_TYPES)
                    )

                    if is_tool_step:
                        if _rejected_tools and step_idx not in _active_tools:
                            return
                        subagent_entries = _extract_subagent_entries(step)
                        raw_args: dict[str, Any] = {}
                        if isinstance(tool_info, dict):
                            params = tool_info.get("arguments") or tool_info.get("parameters")
                            if isinstance(params, dict):
                                raw_args = params
                        elif isinstance(step.get("arguments"), dict):
                            raw_args = dict(step["arguments"])
                        if not raw_args and subagent_entries:
                            raw_args = _synthesize_subagent_arguments(subagent_entries)

                        cls = classify_tool_call_step(
                            step_type,
                            tool_name,
                            raw_args,
                            known_mcp_servers=effective_known_mcp,
                        )
                        raw_args = _normalize_tool_arguments(raw_args, tool_name=tool_name)
                        is_done = any(
                            tok in status_raw
                            for tok in ("DONE", "COMPLETED", "SUCCESS", "ERROR", "FAILED")
                        )
                        is_failed = any(
                            tok in status_raw for tok in ("ERROR", "FAILED", "CANCELED")
                        ) or (isinstance(tool_info, dict) and bool(tool_info.get("call_error")))

                        if step_idx not in _active_tools:
                            tc_id = f"tc_{uuid.uuid4().hex[:12]}"
                            effective_input = raw_args or (
                                {"command": _run_prompt} if cls.acp_kind == "execute" else {}
                            )
                            effective_cmd = cls.command or (
                                _run_prompt if cls.acp_kind == "execute" and not raw_args else ""
                            )
                            start_event = ToolCallEvent(
                                tool_call_id=tc_id,
                                title=cls.title,
                                kind=cls.acp_kind,
                                status="in_progress",
                                raw_input=effective_input,
                                locations=cls.locations,
                                kiro_tool_name=cls.v2_tool_name,
                                mcp_server_name=cls.mcp_server,
                                kiro_tool_id=cls.kiro_tool_id,
                                kas_builtin_id=cls.kas_builtin_id,
                                v2_tool_name=cls.v2_tool_name,
                                acp_kind=cls.acp_kind,
                                mcp_server=cls.mcp_server,
                                mcp_tool=cls.mcp_tool,
                                command=effective_cmd,
                                subagent_target=cls.subagent_target,
                                diff_content=cls.diff_content,
                            )
                            _active_tools[step_idx] = start_event

                            allowed = True
                            if on_tool_call is not None:
                                self._in_permission_request[sid] = (
                                    self._in_permission_request.get(sid, 0) + 1
                                )
                                try:
                                    allowed = await on_tool_call(start_event)
                                finally:
                                    remaining = max(0, self._in_permission_request.get(sid, 1) - 1)
                                    self._in_permission_request[sid] = remaining
                                    if (
                                        remaining == 0
                                        and self._steer_queues.get(sid)
                                        and self._turn_conv_ids.get(sid)
                                    ):
                                        self._interrupt_process_for_steer(sid)
                            if not allowed:
                                _rejected_tools.add(step_idx)
                                self.abort_active_process(sid)
                                if on_tool_update is not None:
                                    await on_tool_update(
                                        replace(
                                            start_event,
                                            status="failed",
                                            raw_input=raw_args,
                                            output="Tool execution rejected by user.",
                                        )
                                    )
                                return

                        if subagent_entries and not is_failed and on_subagent_list is not None:
                            spawn_snapshot = subagent_tracker.record_spawn(subagent_entries)
                            if spawn_snapshot is not None:
                                await on_subagent_list(spawn_snapshot)

                        if is_done and step_idx not in _rejected_tools:
                            _completed_tools.add(step_idx)
                            base_event = _active_tools[step_idx]
                            tool_output = ""
                            if isinstance(tool_info, dict):
                                tool_output = str(
                                    tool_info.get("output")
                                    or tool_info.get("call_error")
                                    or step.get("output", "")
                                )
                            else:
                                tool_output = str(step.get("output", ""))
                            if not tool_output and subagent_entries:
                                tool_output = _format_subagent_tool_output(subagent_entries)
                            if on_tool_update is not None:
                                await on_tool_update(
                                    replace(
                                        base_event,
                                        title=cls.title,
                                        kind=cls.acp_kind,
                                        acp_kind=cls.acp_kind,
                                        status="failed" if is_failed else "completed",
                                        raw_input=raw_args,
                                        output=tool_output,
                                        locations=cls.locations or base_event.locations,
                                        diff_content=cls.diff_content or base_event.diff_content,
                                    )
                                )
                            if (
                                tool_name.lower() == "manage_subagents"
                                and not is_failed
                                and on_subagent_list is not None
                            ):
                                manage_snapshot = subagent_tracker.record_manage_step(
                                    raw_args,
                                    tool_output,
                                )
                                if manage_snapshot is not None:
                                    await on_subagent_list(manage_snapshot)

                    text_delta = step.get("text_delta")
                    text_full = step.get("text") or step.get("response") or step.get("content")
                    if isinstance(text_delta, str) and text_delta:
                        streamed_any_text = True
                        collected_texts.append(text_delta)
                        if on_text is not None:
                            await on_text(text_delta)
                    elif isinstance(text_full, str) and text_full and not is_tool_step:
                        prev_len = _emitted_texts.get(step_idx, 0)
                        if len(text_full) > prev_len:
                            delta = text_full[prev_len:]
                            _emitted_texts[step_idx] = len(text_full)
                            streamed_any_text = True
                            collected_texts.append(delta)
                            if on_text is not None:
                                await on_text(delta)

                async def _consume_stdout(
                    proc: anyio.abc.Process,
                    *,
                    _plain_lines: list[str] = plain_stdout_lines,
                ) -> None:
                    nonlocal resolved_conv_id, resolved_model, run_result_obj, streamed_any_text
                    if proc.stdout is None:
                        return
                    buffer = ""
                    async for chunk in TextReceiveStream(proc.stdout):
                        buffer += chunk
                        while "\n" in buffer:
                            raw_line, buffer = buffer.split("\n", 1)
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                payload = json.loads(line)
                            except json.JSONDecodeError:
                                _plain_lines.append(line)
                                continue
                            if not isinstance(payload, dict):
                                continue

                            event_type = str(payload.get("event", ""))
                            if event_type == "init" or "init" in payload:
                                init_obj = payload.get("init")
                                if isinstance(init_obj, dict):
                                    if init_obj.get("conversation_id"):
                                        resolved_conv_id = str(init_obj["conversation_id"])
                                    if init_obj.get("model"):
                                        resolved_model = str(init_obj["model"])
                                if not resolved_conv_id and payload.get("conversation_id"):
                                    resolved_conv_id = str(payload["conversation_id"])
                                if resolved_conv_id:
                                    self._turn_conv_ids[sid] = resolved_conv_id
                                    if self._steer_queues.get(sid):
                                        self._interrupt_process_for_steer(sid)

                            elif event_type == "step_update" or "step_update" in payload:
                                step_obj = payload.get("step_update")
                                await _handle_step_update(
                                    step_obj if isinstance(step_obj, dict) else payload
                                )
                            elif (
                                event_type == "result" or "result" in payload or "status" in payload
                            ):
                                res_obj = payload.get("result")
                                run_result_obj = res_obj if isinstance(res_obj, dict) else payload
                                if run_result_obj.get("conversation_id"):
                                    resolved_conv_id = str(run_result_obj["conversation_id"])
                                    self._turn_conv_ids[sid] = resolved_conv_id
                                raw_usage = run_result_obj.get("usage")
                                if isinstance(raw_usage, dict):
                                    in_tok = int(raw_usage.get("input_tokens", 0))
                                    out_tok = int(raw_usage.get("output_tokens", 0))
                                    think_tok = int(raw_usage.get("thinking_tokens", 0))
                                    cache_tok = int(raw_usage.get("cache_read_tokens", 0))
                                    tot_tok = int(
                                        raw_usage.get(
                                            "total_tokens",
                                            in_tok + out_tok + think_tok + cache_tok,
                                        )
                                    )
                                    usage["input_tokens"] += in_tok
                                    usage["output_tokens"] += out_tok
                                    usage["thinking_tokens"] += think_tok
                                    usage["cache_read_tokens"] += cache_tok
                                    usage["total_tokens"] += tot_tok
                                resp_text = run_result_obj.get("response")
                                if (
                                    isinstance(resp_text, str)
                                    and resp_text
                                    and not streamed_any_text
                                ):
                                    streamed_any_text = True
                                    collected_texts.append(resp_text)
                                    if on_text is not None:
                                        await on_text(resp_text)

                    trailing = buffer.strip()
                    if trailing:
                        with contextlib.suppress(json.JSONDecodeError):
                            payload = json.loads(trailing)
                            if isinstance(payload, dict):
                                res_obj = payload.get("result")
                                run_result_obj = res_obj if isinstance(res_obj, dict) else payload

                async def _consume_stderr(
                    proc: anyio.abc.Process,
                    *,
                    _stderr_chunks: list[str] = stderr_chunks,
                ) -> None:
                    if proc.stderr is None:
                        return
                    async for chunk in TextReceiveStream(proc.stderr):
                        _stderr_chunks.append(chunk)

                proc: anyio.abc.Process | None = None
                try:
                    log_acp_event(
                        "agy.spawn",
                        save_dir_or_db=self._store.db_path,
                        session_id=sid,
                        bin_path=self.bin_path,
                        args=args,
                        cwd=effective_cwd,
                    )
                    proc = await anyio.open_process(
                        [self.bin_path, *args],
                        cwd=effective_cwd,
                        env=merged_env,
                        start_new_session=True,
                    )
                    self._active_processes[sid] = proc
                    if self._steer_queues.get(sid) and self._turn_conv_ids.get(sid):
                        self._interrupt_process_for_steer(sid)
                    async with proc:
                        async with anyio.create_task_group() as tg:
                            tg.start_soon(_consume_stdout, proc)
                            tg.start_soon(_consume_stderr, proc)
                        await proc.wait()
                    log_acp_event(
                        "agy.exit",
                        save_dir_or_db=self._store.db_path,
                        session_id=sid,
                        returncode=proc.returncode,
                        stderr="".join(stderr_chunks).strip(),
                        result_status=run_result_obj.get("status"),
                        result_error=run_result_obj.get("error"),
                    )
                except anyio.get_cancelled_exc_class():
                    if proc is not None and proc.returncode is None:
                        _signal_process_group(proc, signal.SIGINT)
                        _signal_process_group(proc, signal.SIGTERM)
                    if on_subagent_list is not None:
                        cancelled_snap = subagent_tracker.finalize_turn(cancelled=True)
                        if cancelled_snap is not None:
                            with anyio.CancelScope(shield=True), contextlib.suppress(Exception):
                                await on_subagent_list(cancelled_snap)
                    raise
                finally:
                    self._cancel_abort_timer(sid)
                    self._active_processes.pop(sid, None)
                    if temp_file is not None:
                        with contextlib.suppress(OSError):
                            temp_file.unlink(missing_ok=True)
                    active_temp_file = None

                final_result_obj = run_result_obj
                was_cancelled = sid in self._cancelled_sessions or (
                    not self._steer_queues.get(sid)
                    and str(final_result_obj.get("status", "")).upper() in {"CANCELLED", "CANCELED"}
                )
                self._cancelled_sessions.discard(sid)
                if was_cancelled:
                    if on_subagent_list is not None:
                        cancelled_snap = subagent_tracker.finalize_turn(cancelled=True)
                        if cancelled_snap is not None:
                            await on_subagent_list(cancelled_snap)
                    return PromptOutcome(
                        conversation_id=resolved_conv_id,
                        stop_reason="cancelled",
                        response="".join(collected_texts),
                        usage=usage,
                        raw_result=final_result_obj,
                    )

                err_msg = str(final_result_obj.get("error", ""))
                if (
                    current_model
                    and str(final_result_obj.get("status", "")).upper() == "ERROR"
                    and "invalid model" in err_msg.lower()
                ):
                    log_acp_event(
                        "agy.model_fallback",
                        save_dir_or_db=self._store.db_path,
                        session_id=sid,
                        rejected_model=current_model,
                        error=err_msg,
                    )
                    current_model = None
                    continue

                if not streamed_any_text and plain_stdout_lines:
                    fallback_text = "\n".join(plain_stdout_lines)
                    collected_texts.append(fallback_text)
                    if on_text is not None:
                        await on_text(fallback_text)

                if self._steer_queues.get(sid):
                    if on_tool_update is not None:
                        for step_idx, tc_start in active_tool_calls.items():
                            if (
                                step_idx not in completed_tool_steps
                                and step_idx not in rejected_tool_steps
                            ):
                                await on_tool_update(
                                    replace(
                                        tc_start,
                                        status="completed",
                                        output="Interrupted by user steering.",
                                    )
                                )
                    steers = self._pop_queued_steers(sid)
                    for s in steers:
                        steer_temp_files.extend(s.temp_files)
                    if resolved_conv_id:
                        persisted = self._store.get_session(sid)
                        if persisted is not None:
                            persisted["agyConversationId"] = resolved_conv_id
                            self._store.upsert_session(persisted)
                    if on_steer_consumed is not None:
                        await on_steer_consumed("\n".join(s.raw_message for s in steers))
                    current_prompt = "\n\n".join(s.prompt for s in steers if s.prompt)
                    continue

                break
        finally:
            self._cancel_abort_timer(sid)
            self._active_turns.discard(sid)
            self._turn_conv_ids.pop(sid, None)
            self._in_permission_request.pop(sid, None)
            if active_temp_file is not None:
                with contextlib.suppress(OSError):
                    active_temp_file.unlink(missing_ok=True)
            for leftover in self._pop_queued_steers(sid):
                steer_temp_files.extend(leftover.temp_files)
            for tmp_path in steer_temp_files:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)

        full_response = "".join(collected_texts)
        if usage["input_tokens"] == 0 and prompt:
            usage["input_tokens"] = max(1, len(prompt.split()))
        if usage["output_tokens"] == 0 and full_response:
            usage["output_tokens"] = max(1, len(full_response.split()))
        if usage["total_tokens"] == 0:
            usage["total_tokens"] = (
                usage["input_tokens"]
                + usage["output_tokens"]
                + usage["thinking_tokens"]
                + usage["cache_read_tokens"]
            )
        usage["context_tokens"] = (
            last_step_context_tokens
            if last_step_context_tokens > 0
            else (
                usage["input_tokens"]
                + usage["output_tokens"]
                + usage["thinking_tokens"]
                + usage["cache_read_tokens"]
            )
        )

        if on_subagent_list is not None:
            completed_snap = subagent_tracker.finalize_turn(cancelled=False)
            if completed_snap is not None:
                await on_subagent_list(completed_snap)

        return PromptOutcome(
            conversation_id=resolved_conv_id,
            stop_reason="end_turn",
            response=full_response,
            usage=usage,
            raw_result=final_result_obj,
            resolved_model=resolved_model,
        )
