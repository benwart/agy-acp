"""Async subprocess provider for driving the Antigravity CLI (`agy`) via anyio."""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import anyio
import anyio.abc
from anyio.streams.text import TextReceiveStream

from agy_acp.logic.carousel_normalizer import CarouselStreamNormalizer
from agy_acp.logic.custom_agents import (
    _READ_ONLY_KIRO_TOOLS as _READ_ONLY_KIRO_TOOLS,
)
from agy_acp.logic.custom_agents import (
    load_kiro_agent_mcp_servers as load_kiro_agent_mcp_servers,
)
from agy_acp.logic.custom_agents import (
    load_kiro_agent_spec as load_kiro_agent_spec,
)
from agy_acp.logic.tool_classifier import (
    _PATH_ARGUMENT_KEYS,
    _extract_first_str,
    _split_model_effort_suffix,
)
from agy_acp.logic.tool_classifier import (
    FALLBACK_MODELS as FALLBACK_MODELS,
)
from agy_acp.logic.tool_classifier import (
    PER_SESSION_MCP_ENV_KEYS as PER_SESSION_MCP_ENV_KEYS,
)
from agy_acp.logic.tool_classifier import (
    VALID_EFFORTS as VALID_EFFORTS,
)
from agy_acp.logic.tool_classifier import (
    PromptOutcome as PromptOutcome,
)
from agy_acp.logic.tool_classifier import (
    QueuedSteer as QueuedSteer,
)
from agy_acp.logic.tool_classifier import (
    ToolCallClassification as ToolCallClassification,
)
from agy_acp.logic.tool_classifier import (
    ToolCallEvent as ToolCallEvent,
)
from agy_acp.logic.tool_classifier import (
    _build_builtin_title as _build_builtin_title,
)
from agy_acp.logic.tool_classifier import (
    _dedupe_mcp_servers as _dedupe_mcp_servers,
)
from agy_acp.logic.tool_classifier import (
    _extract_diff_content as _extract_diff_content,
)
from agy_acp.logic.tool_classifier import (
    _extract_mcp_identity as _extract_mcp_identity,
)
from agy_acp.logic.tool_classifier import (
    _pairs_to_dict as _pairs_to_dict,
)
from agy_acp.logic.tool_classifier import (
    _resolve_builtin_triple as _resolve_builtin_triple,
)
from agy_acp.logic.tool_classifier import (
    _resolve_kiro_tool_identity as _resolve_kiro_tool_identity,
)
from agy_acp.logic.tool_classifier import (
    classify_tool_call_step as classify_tool_call_step,
)
from agy_acp.logic.tool_classifier import (
    resolve_agy_binary as resolve_agy_binary,
)
from agy_acp.logic.tool_classifier import (
    resolve_model_for_effort as resolve_model_for_effort,
)
from agy_acp.logic.tool_classifier import (
    translate_mcp_server as translate_mcp_server,
)
from agy_acp.logic.wire.base import synthesize_task_command
from agy_acp.providers.acp_logger import log_acp_event
from agy_acp.providers.state_store import SqliteStateStoreProvider

MAX_INLINE_PROMPT_BYTES: int = 64 * 1024

_MODE_TO_AGY_CLI: dict[str, str] = {
    "auto": "auto",
    "approve": "accept-edits",
    "read-only": "plan",
    "accept-edits": "accept-edits",
    "plan": "plan",
}


@dataclass(frozen=True)
class PromptDelivery:
    """CLI prompt argument and optional `stream-json` stdin payload for an `agy` turn."""

    prompt_arg: str
    stdin_payload: bytes | None = None
    input_format: str | None = None


def prepare_prompt_delivery(prompt: str) -> PromptDelivery:
    """Deliver prompts <= 64 KiB inline on `-p` and larger prompts via `stream-json` stdin."""
    if len(prompt.encode("utf-8")) <= MAX_INLINE_PROMPT_BYTES:
        return PromptDelivery(prompt_arg=prompt)

    ndjson_line = (
        json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n"
    )
    return PromptDelivery(
        prompt_arg="",
        stdin_payload=ndjson_line.encode("utf-8"),
        input_format="stream-json",
    )


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
    input_format: str | None = None,
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
    if cli_mode == "plan" and not prompt_arg.lstrip().startswith("/"):
        args.append("--disable-slash-commands")

    if effort:
        normalized_effort = effort.strip().lower()
        model_effort = _split_model_effort_suffix(model)
        if normalized_effort in VALID_EFFORTS and (
            model_effort is None or model_effort[1] == normalized_effort
        ):
            args.extend(["--effort", normalized_effort])

    effective_skip = skip_permissions if skip_permissions is not None else (mode == "auto")
    if effective_skip:
        args.append("--dangerously-skip-permissions")
    if sandbox:
        args.append("--sandbox")

    args.extend(["--print-timeout", "0"])
    if input_format:
        args.extend(["--input-format", input_format])
    if output_format:
        args.extend(["--output-format", output_format])
    args.extend(["-p", prompt_arg])
    return args


build_command_args = build_agy_args


def _format_denied_tools_label(denied_tools: Sequence[str]) -> str:
    """Format unique denied tool identifiers into a compact label."""
    unique = [name.strip() for name in dict.fromkeys(denied_tools) if name and name.strip()]
    return ", ".join(unique) if unique else "requested tool"


def _build_tool_denial_prompt(denied_tools: Sequence[str]) -> str:
    """Build a single-shot continuation prompt after a mid-turn tool policy denial."""
    label = _format_denied_tools_label(denied_tools)
    return (
        f"[Tool policy] Tool `{label}` was denied in this session and was not executed. "
        "Do NOT retry denied tools; continue or provide your closing reply using only "
        "permitted tools and information already available."
    )


def _build_tool_denial_closing_notice(denied_tools: Sequence[str]) -> str:
    """Build a fallback closing message when a tool-denied turn yielded no assistant text."""
    label = _format_denied_tools_label(denied_tools)
    return f"Tool `{label}` is not permitted in this session."


_ERROR_STEP_TYPES: frozenset[str] = frozenset({"error", "error_message"})
_PASSIVE_BOOKKEEPING_STEP_TYPES: frozenset[str] = frozenset(
    {
        "user_input",
        "system_message",
        "system",
        "checkpoint",
        "ephemeral_message",
        "error",
        "error_message",
    }
)


@dataclass
class _RunErrorTracker:
    """Tracks per-invocation step progression to distinguish sticky vs terminal `agy` errors."""

    last_error_step_idx: int = -1
    last_completed_step_idx: int = -1

    def record_step(self, step_idx: int, step_type: str, status_raw: str) -> None:
        lower_type = step_type.strip().lower()
        if lower_type in _ERROR_STEP_TYPES:
            self.last_error_step_idx = max(self.last_error_step_idx, step_idx)
            return
        if lower_type in _PASSIVE_BOOKKEEPING_STEP_TYPES:
            return
        is_completed = any(tok in status_raw for tok in ("DONE", "COMPLETED", "SUCCESS"))
        is_failed = any(tok in status_raw for tok in ("ERROR", "FAILED", "CANCELED", "CANCELLED"))
        if is_completed and not is_failed:
            self.last_completed_step_idx = max(self.last_completed_step_idx, step_idx)

    def has_recovered_from_error(self, *, has_output: bool) -> bool:
        if self.last_error_step_idx >= 0:
            return self.last_completed_step_idx > self.last_error_step_idx
        return self.last_completed_step_idx >= 0 or has_output


def _extract_unrecovered_cli_error(
    final_result_obj: Mapping[str, Any],
    returncode: int | None,
    stderr_text: str,
    *,
    has_response: bool,
    error_tracker: _RunErrorTracker | None = None,
) -> str:
    """Return an error message when `agy` exited with an unrecovered error, else `''`."""
    status = str(final_result_obj.get("status", "")).upper()
    if status == "ERROR":
        has_output = has_response or bool(str(final_result_obj.get("response") or "").strip())
        if (
            returncode == 0
            and error_tracker is not None
            and error_tracker.has_recovered_from_error(has_output=has_output)
        ):
            return ""
        return (
            str(final_result_obj.get("error") or "").strip()
            or stderr_text
            or f"agy exited with status {returncode or 1}"
        )
    if returncode not in (0, None) and not has_response and not final_result_obj:
        return stderr_text or f"agy exited with status {returncode}"
    return ""


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
        cmd_val = _extract_first_str(
            normalized,
            ("CommandLine", "cmd", "Command"),
        ) or synthesize_task_command(tool_name, normalized)
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

    def clear_steers(self, session_id: str) -> list[str]:
        """Pop and discard all queued mid-turn steers for `session_id`, unlinking temp files."""
        steers = self._pop_queued_steers(session_id)
        for steer in steers:
            for tmp in steer.temp_files:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
        return [s.raw_message for s in steers if s.raw_message]

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

    def _persist_resolved_conversation_id(self, sid: str, conv_id: str) -> None:
        """Persist `conv_id` onto `sid` in SQLite when the session row exists."""
        if not conv_id:
            return
        persisted = self._store.get_session(sid)
        if persisted is not None:
            persisted["agyConversationId"] = conv_id
            self._store.upsert_session(persisted)

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
        fallback_from_model = ""
        current_prompt = prompt
        current_model = model
        collected_texts: list[str] = []
        final_result_obj: dict[str, Any] = {}
        outcome_error = ""
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
        did_denial_recovery = False
        subagent_tracker = _SubagentRosterTracker()
        self._active_turns.add(sid)
        self._turn_conv_ids[sid] = resolved_conv_id

        try:
            while True:
                delivery = prepare_prompt_delivery(current_prompt)
                args = build_agy_args(
                    cwd=cwd,
                    prompt_arg=delivery.prompt_arg,
                    conversation_id=resolved_conv_id or None,
                    model=current_model,
                    mode=mode,
                    effort=effort,
                    skip_permissions=skip_permissions,
                    sandbox=sandbox,
                    input_format=delivery.input_format,
                    output_format="stream-json",
                )
                emitted_thoughts_by_step: dict[int, int] = {}
                emitted_texts_by_step: dict[int, int] = {}
                active_tool_calls: dict[int, ToolCallEvent] = {}
                completed_tool_steps: set[int] = set()
                rejected_tool_steps: set[int] = set()
                rejected_tool_names: list[str] = []
                streamed_any_text = False
                plain_stdout_lines: list[str] = []
                stderr_chunks: list[str] = []
                run_result_obj: dict[str, Any] = {}
                run_error_tracker = _RunErrorTracker()
                carousel_normalizer = CarouselStreamNormalizer()

                async def _emit_normalized_text(
                    raw_chunk: str,
                    *,
                    _norm: CarouselStreamNormalizer = carousel_normalizer,
                ) -> None:
                    piece = _norm.feed(raw_chunk)
                    if piece:
                        collected_texts.append(piece)
                        if on_text is not None:
                            await on_text(piece)

                async def _flush_normalized_text(
                    *,
                    boundary_only: bool = False,
                    emit_callback: bool = True,
                    _norm: CarouselStreamNormalizer = carousel_normalizer,
                ) -> None:
                    piece = _norm.flush_prose_before_boundary() if boundary_only else _norm.flush()
                    if piece:
                        collected_texts.append(piece)
                        if emit_callback and on_text is not None:
                            await on_text(piece)

                async def _handle_step_update(
                    step: dict[str, Any],
                    *,
                    _emitted_thoughts: dict[int, int] = emitted_thoughts_by_step,
                    _emitted_texts: dict[int, int] = emitted_texts_by_step,
                    _active_tools: dict[int, ToolCallEvent] = active_tool_calls,
                    _completed_tools: set[int] = completed_tool_steps,
                    _rejected_tools: set[int] = rejected_tool_steps,
                    _rejected_names: list[str] = rejected_tool_names,
                    _error_tracker: _RunErrorTracker = run_error_tracker,
                    _run_prompt: str = current_prompt,
                ) -> None:
                    nonlocal streamed_any_text, last_step_context_tokens
                    step_idx = int(step.get("step_index", step.get("idx", 0)))
                    step_type = str(step.get("step_type", step.get("type_name", "")))
                    status_raw = str(step.get("status", step.get("state", "RUNNING"))).upper()
                    _error_tracker.record_step(step_idx, step_type, status_raw)

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
                            await _flush_normalized_text(boundary_only=True)
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
                                _rejected_names.append(
                                    tool_name or cls.kiro_tool_id or cls.v2_tool_name or cls.title
                                )
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
                        await _emit_normalized_text(text_delta)
                    elif isinstance(text_full, str) and text_full and not is_tool_step:
                        prev_len = _emitted_texts.get(step_idx, 0)
                        if len(text_full) > prev_len:
                            delta = text_full[prev_len:]
                            _emitted_texts[step_idx] = len(text_full)
                            streamed_any_text = True
                            await _emit_normalized_text(delta)

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
                                    await _emit_normalized_text(resp_text)

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

                async def _feed_stdin(
                    proc: anyio.abc.Process,
                    *,
                    _payload: bytes | None = delivery.stdin_payload,
                ) -> None:
                    if proc.stdin is None:
                        return
                    try:
                        if _payload is not None:
                            with contextlib.suppress(
                                BrokenPipeError,
                                anyio.BrokenResourceError,
                                anyio.ClosedResourceError,
                                OSError,
                            ):
                                await proc.stdin.send(_payload)
                    finally:
                        with contextlib.suppress(
                            BrokenPipeError,
                            anyio.BrokenResourceError,
                            anyio.ClosedResourceError,
                            OSError,
                        ):
                            await proc.stdin.aclose()

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
                        stdin=(
                            subprocess.PIPE
                            if delivery.stdin_payload is not None
                            else subprocess.DEVNULL
                        ),
                        cwd=effective_cwd,
                        env=merged_env,
                        start_new_session=True,
                    )
                    self._active_processes[sid] = proc
                    if self._steer_queues.get(sid) and self._turn_conv_ids.get(sid):
                        self._interrupt_process_for_steer(sid)
                    async with proc:
                        async with anyio.create_task_group() as tg:
                            tg.start_soon(_feed_stdin, proc)
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

                final_result_obj = run_result_obj
                was_cancelled = sid in self._cancelled_sessions or (
                    not self._steer_queues.get(sid)
                    and str(final_result_obj.get("status", "")).upper() in {"CANCELLED", "CANCELED"}
                )
                self._cancelled_sessions.discard(sid)
                if was_cancelled:
                    await _flush_normalized_text(emit_callback=False)
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
                    if not fallback_from_model:
                        fallback_from_model = current_model
                    resolved_model = ""
                    current_model = None
                    continue

                if not streamed_any_text and plain_stdout_lines:
                    fallback_text = "\n".join(plain_stdout_lines)
                    await _emit_normalized_text(fallback_text)

                await _flush_normalized_text()

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
                    self._persist_resolved_conversation_id(sid, resolved_conv_id)
                    if on_steer_consumed is not None:
                        await on_steer_consumed("\n".join(s.raw_message for s in steers))
                    current_prompt = "\n\n".join(s.prompt for s in steers if s.prompt)
                    continue

                if rejected_tool_steps:
                    if not did_denial_recovery and resolved_conv_id:
                        did_denial_recovery = True
                        self._persist_resolved_conversation_id(sid, resolved_conv_id)
                        current_prompt = _build_tool_denial_prompt(rejected_tool_names)
                        log_acp_event(
                            "agy.tool_denial_recovery",
                            save_dir_or_db=self._store.db_path,
                            session_id=sid,
                            conversation_id=resolved_conv_id,
                            denied_tools=list(dict.fromkeys(rejected_tool_names)),
                        )
                        continue
                    if not collected_texts:
                        closing_notice = _build_tool_denial_closing_notice(rejected_tool_names)
                        collected_texts.append(closing_notice)
                        if on_text is not None:
                            await on_text(closing_notice)
                else:
                    outcome_error = _extract_unrecovered_cli_error(
                        final_result_obj,
                        proc.returncode if proc is not None else None,
                        "".join(stderr_chunks).strip(),
                        has_response=bool(collected_texts),
                        error_tracker=run_error_tracker,
                    )

                break
        finally:
            self._cancel_abort_timer(sid)
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
            fallback_from_model=fallback_from_model,
            error=outcome_error,
        )
