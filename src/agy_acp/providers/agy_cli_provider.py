"""Async subprocess provider for driving the Antigravity CLI (`agy`) via anyio."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import anyio.abc
from anyio.streams.text import TextReceiveStream

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
        return {
            "type": "http",
            "name": str(raw.get("name", "")),
            "url": str(raw.get("url", "")),
            "headers": _pairs_to_dict(raw.get("headers", [])),
        }
    return {
        "type": "stdio",
        "name": str(raw.get("name", "")),
        "command": str(raw.get("command", "")),
        "args": [str(arg) for arg in raw.get("args", [])],
        "env": _pairs_to_dict(raw.get("env", [])),
    }


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

    target_dir = temp_dir or Path(tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)
    tag = (session_id or "default")[:8]
    filename = f"agy-acp-prompt-{tag}-{int(time.time() * 1000)}.md"
    file_path = target_dir / filename
    fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
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
    if model:
        args.extend(["--model", model])

    cli_mode = _MODE_TO_AGY_CLI.get(mode, "auto")
    args.extend(["--mode", cli_mode])

    if effort and effort in VALID_EFFORTS:
        args.extend(["--effort", effort])

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


def load_kiro_agent_mcp_servers(
    agent_name: str | None = None,
    *,
    cwd: str | None = None,
    include_home: bool = True,
) -> list[dict[str, Any]]:
    """Load `mcpServers` definitions from `<cwd>/.kiro/agents/<agent>.json` or `~/.kiro/agents`."""
    candidates: list[str] = []
    if agent_name and agent_name not in {"auto", "approve", "read-only"}:
        candidates.append(agent_name)
    if "kirocrew" not in candidates:
        candidates.append("kirocrew")

    search_dirs: list[Path] = []
    if cwd:
        search_dirs.append(Path(cwd) / ".kiro" / "agents")
    if include_home:
        with contextlib.suppress(RuntimeError, OSError):
            search_dirs.append(Path.home() / ".kiro" / "agents")

    servers_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        for agents_dir in search_dirs:
            spec_path = agents_dir / f"{cand}.json"
            if not spec_path.is_file():
                continue
            with contextlib.suppress(OSError, json.JSONDecodeError):
                raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
                if not isinstance(raw_spec, dict):
                    continue
                raw_mcp = raw_spec.get("mcpServers")
                if isinstance(raw_mcp, dict):
                    for srv_name, srv_cfg in raw_mcp.items():
                        if (
                            isinstance(srv_name, str)
                            and srv_name
                            and isinstance(srv_cfg, dict)
                            and not srv_cfg.get("disabled")
                            and srv_name not in servers_by_name
                        ):
                            servers_by_name[srv_name] = {"name": srv_name, **srv_cfg}
                elif isinstance(raw_mcp, list):
                    for item in raw_mcp:
                        if isinstance(item, dict) and item.get("name"):
                            sname = str(item["name"])
                            if sname not in servers_by_name:
                                servers_by_name[sname] = dict(item)
    return list(servers_by_name.values())


def _resolve_kiro_tool_identity(
    step_type: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, str, str]:
    """Resolve `(kiro_tool_name, mcp_server_name, kiro_tool_id)` for ACP `_meta.kiro`."""
    raw_name = (tool_name or step_type).strip()
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
            return tool, server, f"@{server}/{tool}"

    if raw_name.startswith("mcp__"):
        rest = raw_name[len("mcp__") :]
        server, sep, tool = rest.rpartition("__")
        if sep and server and tool:
            return tool, server, f"@{server}/{tool}"
    elif raw_name.startswith("mcp_"):
        rest = raw_name[len("mcp_") :]
        server, sep, tool = rest.partition("_")
        if sep and server and tool:
            return tool, server, f"@{server}/{tool}"

    if any(
        k in lower_name
        for k in ("run_command", "command_execution", "send_command_input", "bash", "exec", "shell")
    ):
        return "execute_bash", "", "run_command"
    if any(
        k in lower_name
        for k in (
            "view_file",
            "read_file",
            "view_code",
            "list_dir",
            "list_resources",
            "read_resource",
            "read_browser",
        )
    ):
        return "fs_read", "", "read_file"
    if any(
        k in lower_name
        for k in (
            "code_action",
            "edit_file",
            "write_file",
            "write_to_file",
            "replace",
            "sed_file",
            "notebook_edit",
        )
    ):
        return "fs_write", "", "fs_write"
    if "grep" in lower_name:
        return "grep", "", "grep_search"
    if any(k in lower_name for k in ("find_by_name", "file_search", "glob", "find")):
        return "glob", "", "file_search"
    if any(k in lower_name for k in ("search_web", "web_search", "remote_web_search")):
        return "web_search", "", "remote_web_search"
    if any(k in lower_name for k in ("read_url", "web_fetch")):
        return "web_fetch", "", "web_fetch"
    if any(
        k in lower_name
        for k in (
            "invoke_subagent",
            "define_subagent",
            "manage_subagents",
            "send_message",
            "use_subagent",
        )
    ):
        return "use_subagent", "", "invoke_sub_agent"
    if any(k in lower_name for k in ("todo_list", "task_boundary")):
        return "todo_list", "", "todo_list"
    if "command" in arguments or "CommandLine" in arguments:
        return "execute_bash", "", "run_command"

    fallback = lower_name or "tool"
    return fallback, "", fallback


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
        for cmd_key in ("CommandLine", "cmd", "Command"):
            val = normalized.get(cmd_key)
            if isinstance(val, str) and val:
                normalized["command"] = val
                break
    if "path" not in normalized:
        for path_key in (
            "Path",
            "AbsolutePath",
            "TargetFile",
            "DirectoryPath",
            "SearchPath",
            "file_path",
            "filePath",
        ):
            val = normalized.get(path_key)
            if isinstance(val, str) and val:
                normalized["path"] = val
                break
    if "description" not in normalized:
        summary = normalized.get("toolSummary")
        if isinstance(summary, str) and summary.strip():
            normalized["description"] = summary.strip()
    return normalized


def _classify_tool_step(
    step_type: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    """Map an `agy` step/tool name and arguments to ACP `(kind, title, locations)`."""
    kiro_tool, mcp_server, _ = _resolve_kiro_tool_identity(step_type, tool_name, arguments)
    if mcp_server and kiro_tool:
        title = (
            "spawn_run"
            if (mcp_server == "kirocrew-core" and kiro_tool == "spawn_run")
            else f"@{mcp_server}/{kiro_tool}"
        )
        return "other", title, []

    normalized = (tool_name or step_type).lower()
    path_val = str(
        arguments.get("path")
        or arguments.get("Path")
        or arguments.get("AbsolutePath")
        or arguments.get("file_path")
        or arguments.get("TargetFile")
        or arguments.get("DirectoryPath")
        or arguments.get("SearchPath")
        or ""
    )
    locations = [{"path": path_val}] if path_val else []

    if any(
        k in normalized
        for k in (
            "view_file",
            "read_file",
            "view_code",
            "read_resource",
            "read_url",
            "read_browser",
        )
    ):
        title = f"Viewing {path_val}" if path_val else f"Reading ({tool_name or step_type})"
        return "read", title, locations
    if any(
        k in normalized
        for k in (
            "code_action",
            "edit_file",
            "write_file",
            "write_to_file",
            "replace",
            "sed_file",
            "notebook_edit",
        )
    ):
        title = f"Editing {path_val}" if path_val else f"Editing ({tool_name or step_type})"
        return "edit", title, locations
    if any(k in normalized for k in ("grep", "find", "list_dir", "search")):
        query = str(
            arguments.get("query")
            or arguments.get("Query")
            or arguments.get("Pattern")
            or path_val
            or ""
        )
        title = f"Searching {query}" if query else f"Searching ({tool_name or step_type})"
        return "search", title, locations
    if any(
        k in normalized
        for k in ("run_command", "command_execution", "send_command_input", "bash", "exec", "shell")
    ):
        cmd = str(arguments.get("command") or arguments.get("CommandLine") or "")
        short_cmd = cmd[:80] if cmd else (tool_name or step_type or "Execute command")
        title = f"Running `{short_cmd}`" if cmd else "Execute command"
        return "execute", title, locations
    if "command" in arguments or "CommandLine" in arguments:
        cmd = str(arguments.get("command") or arguments.get("CommandLine") or "")
        short_cmd = cmd[:80] if cmd else (tool_name or step_type or "Execute command")
        title = f"Running `{short_cmd}`" if cmd else "Execute command"
        return "execute", title, locations
    return "other", tool_name or step_type or "Tool call", locations


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
        self._cancelled_sessions: set[str] = set()
        self._active_turns: set[str] = set()
        self._steer_queues: dict[str, list[QueuedSteer]] = {}
        self._in_permission_request: dict[str, int] = {}
        self._turn_conv_ids: dict[str, str] = {}

    def sync_mcp_servers(self, mcp_servers: list[dict[str, Any]]) -> list[str]:
        """Synchronize MCP server configurations into `agy`'s `mcp_config.json`."""
        if not mcp_servers:
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

        synced: list[str] = []
        changed = False
        for srv in mcp_servers:
            translated = translate_mcp_server(srv)
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
                if isinstance(translated.get("env"), dict) and translated["env"]:
                    entry["env"] = {str(k): str(v) for k, v in translated["env"].items()}
            if servers_map.get(name) != entry:
                servers_map[name] = entry
                changed = True
            synced.append(name)

        if changed:
            with contextlib.suppress(OSError):
                self._mcp_config_path.parent.mkdir(parents=True, exist_ok=True)
                self._mcp_config_path.write_text(
                    json.dumps(existing_data, indent=2) + "\n",
                    encoding="utf-8",
                )
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
    ) -> dict[str, Any]:
        """Run a non-interactive slash command (`agy -p <cmd> --output-format json`) via anyio."""
        normalized = command.strip()
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        args: list[str] = [self.bin_path]
        if cwd:
            args.extend(["--add-dir", cwd])
        args.extend(["--output-format", "json", "-p", normalized])

        with anyio.fail_after(timeout):
            result = await anyio.run_process(args, check=False)

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
        existing = (self._store.get_session(sid) if session_id is not None else None) or {}

        raw_mcp = (
            list(mcp_servers) if mcp_servers is not None else list(existing.get("mcp_servers", []))
        )
        agent_name = str(
            (auth_config or {}).get("kiro_agent") or os.environ.get("AGY_KIRO_AGENT") or "kirocrew"
        )
        agent_mcp = load_kiro_agent_mcp_servers(
            agent_name,
            cwd=cwd,
            include_home=not self._is_custom_store,
        )
        combined_mcp = [*agent_mcp, *raw_mcp]
        translated_mcp = [translate_mcp_server(srv) for srv in combined_mcp]
        if translated_mcp:
            self.sync_mcp_servers(translated_mcp)
        state: dict[str, Any] = {
            "sessionId": sid,
            "agyConversationId": existing.get("agyConversationId"),
            "cwd": cwd,
            "mcp_servers": translated_mcp,
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
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.send_signal(signal.SIGINT)
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()

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

    async def cancel_session(self, session_id: str) -> bool:
        """Send SIGINT/terminate to an active `agy` subprocess for `session_id`."""
        await anyio.sleep(0)
        self._cancelled_sessions.add(session_id)
        for steer in self._pop_queued_steers(session_id):
            for tmp in steer.temp_files:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
        proc = self._active_processes.get(session_id)
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.send_signal(signal.SIGINT)
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
            return True
        return False

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
        on_thought: Callable[[str], Awaitable[None]] | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
        on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None = None,
        on_usage: Callable[[int], Awaitable[None]] | None = None,
        on_steer_consumed: Callable[[str], Awaitable[None]] | None = None,
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
        self._active_turns.add(sid)
        self._turn_conv_ids[sid] = resolved_conv_id

        try:
            while True:
                prompt_arg, temp_file = prepare_prompt_arg(current_prompt, session_id=sid)
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
                        or (
                            step_type
                            and step_type.lower()
                            not in {
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
                                "summary",
                                "compaction",
                            }
                        )
                    )

                    if is_tool_step:
                        raw_args: dict[str, Any] = {}
                        if isinstance(tool_info, dict):
                            params = tool_info.get("arguments") or tool_info.get("parameters")
                            if isinstance(params, dict):
                                raw_args = params
                        elif isinstance(step.get("arguments"), dict):
                            raw_args = dict(step["arguments"])

                        kiro_tool_name, mcp_server_name, kiro_tool_id = _resolve_kiro_tool_identity(
                            step_type, tool_name, raw_args
                        )
                        kind, title, locations = _classify_tool_step(step_type, tool_name, raw_args)
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
                            start_event = ToolCallEvent(
                                tool_call_id=tc_id,
                                title=title,
                                kind=kind,
                                status="in_progress",
                                raw_input=raw_args
                                or ({"command": _run_prompt} if kind == "execute" else {}),
                                locations=locations,
                                kiro_tool_name=kiro_tool_name,
                                mcp_server_name=mcp_server_name,
                                kiro_tool_id=kiro_tool_id,
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
                                if on_tool_update is not None:
                                    await on_tool_update(
                                        ToolCallEvent(
                                            tool_call_id=tc_id,
                                            title=title,
                                            kind=kind,
                                            status="failed",
                                            raw_input=raw_args,
                                            output="Tool execution rejected by user.",
                                            locations=locations,
                                            kiro_tool_name=kiro_tool_name,
                                            mcp_server_name=mcp_server_name,
                                            kiro_tool_id=kiro_tool_id,
                                        )
                                    )
                                return

                        if is_done and step_idx not in _rejected_tools:
                            _completed_tools.add(step_idx)
                            tc_id = _active_tools[step_idx].tool_call_id
                            tool_output = ""
                            if isinstance(tool_info, dict):
                                tool_output = str(
                                    tool_info.get("output")
                                    or tool_info.get("call_error")
                                    or step.get("output", "")
                                )
                            else:
                                tool_output = str(step.get("output", ""))
                            if on_tool_update is not None:
                                await on_tool_update(
                                    ToolCallEvent(
                                        tool_call_id=tc_id,
                                        title=title,
                                        kind=kind,
                                        status="failed" if is_failed else "completed",
                                        raw_input=raw_args,
                                        output=tool_output,
                                        locations=locations,
                                        kiro_tool_name=kiro_tool_name,
                                        mcp_server_name=mcp_server_name,
                                        kiro_tool_id=kiro_tool_id,
                                    )
                                )

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
                        with contextlib.suppress(ProcessLookupError, OSError):
                            proc.send_signal(signal.SIGINT)
                        with contextlib.suppress(ProcessLookupError, OSError):
                            proc.terminate()
                    raise
                finally:
                    self._active_processes.pop(sid, None)
                    if temp_file is not None:
                        with contextlib.suppress(OSError):
                            temp_file.unlink(missing_ok=True)

                final_result_obj = run_result_obj
                was_cancelled = sid in self._cancelled_sessions or (
                    not self._steer_queues.get(sid)
                    and str(final_result_obj.get("status", "")).upper() in {"CANCELLED", "CANCELED"}
                )
                self._cancelled_sessions.discard(sid)
                if was_cancelled:
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
                                    ToolCallEvent(
                                        tool_call_id=tc_start.tool_call_id,
                                        title=tc_start.title,
                                        kind=tc_start.kind,
                                        status="completed",
                                        raw_input=tc_start.raw_input,
                                        output="Interrupted by user steering.",
                                        locations=tc_start.locations,
                                        kiro_tool_name=tc_start.kiro_tool_name,
                                        mcp_server_name=tc_start.mcp_server_name,
                                        kiro_tool_id=tc_start.kiro_tool_id,
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
            self._active_turns.discard(sid)
            self._turn_conv_ids.pop(sid, None)
            self._in_permission_request.pop(sid, None)
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

        return PromptOutcome(
            conversation_id=resolved_conv_id,
            stop_reason="end_turn",
            response=full_response,
            usage=usage,
            raw_result=final_result_obj,
            resolved_model=resolved_model,
        )
