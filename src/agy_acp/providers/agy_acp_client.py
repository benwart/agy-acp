"""Full-duplex ACP stdio client provider for driving `agy acp --stdio --process-per-session`."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import subprocess
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

import anyio
import anyio.abc
from anyio.streams.text import TextReceiveStream

from agy_acp.logic.tool_classifier import (
    FALLBACK_MODELS,
    PER_SESSION_MCP_ENV_KEYS,
    PromptOutcome,
    QueuedSteer,
    ToolCallClassification,
    ToolCallEvent,
    classify_tool_call_step,
    resolve_agy_binary,
    resolve_model_for_effort,
    translate_mcp_server,
)
from agy_acp.logic.wire.base import synthesize_task_command
from agy_acp.providers.acp_logger import log_acp_event
from agy_acp.providers.state_store import SqliteStateStoreProvider

_NATIVE_SUBAGENT_TOOLS: frozenset[str] = frozenset(
    {
        "invoke_subagent",
        "define_subagent",
        "manage_subagents",
        "invoke_sub_agent",
    }
)

_READ_ONLY_ACP_KINDS: frozenset[str] = frozenset({"read", "search", "fetch"})

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

_KIND_TO_FALLBACK_TOOL: dict[str, str] = {
    "execute": "run_command",
    "read": "read_file",
    "edit": "edit_file",
    "delete": "delete_file",
    "search": "grep_search",
    "fetch": "web_fetch",
}


class AgyAcpError(RuntimeError):
    """Base error raised by `AgyAcpClientProvider`."""


class AgyAcpTransportError(AgyAcpError):
    """Raised when the `agy acp` subprocess exits unexpectedly or closes stdio."""


class AgyAcpRpcError(AgyAcpError):
    """Raised when `agy acp` returns a JSON-RPC 2.0 error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.rpc_message = message
        self.data = data


def format_acp_mcp_servers(
    servers: Sequence[Mapping[str, Any]] | None,
    *,
    session_env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Serialize MCP server definitions into ACP `session/new` & `session/load` list format."""
    if not servers:
        return []

    seen_names: set[str] = set()
    formatted: list[dict[str, Any]] = []
    for raw_srv in servers:
        if not isinstance(raw_srv, Mapping) or raw_srv.get("disabled") is True:
            continue
        translated = translate_mcp_server(dict(raw_srv))
        if translated.get("disabled") is True:
            continue
        name = str(translated.get("name") or "").strip()
        if not name or name in seen_names:
            continue

        if translated.get("type") == "http" or "url" in raw_srv or "serverUrl" in raw_srv:
            url = str(translated.get("url") or raw_srv.get("serverUrl") or "").strip()
            if not url:
                continue
            raw_headers = translated.get("headers")
            headers_map = raw_headers if isinstance(raw_headers, Mapping) else {}
            seen_names.add(name)
            formatted.append(
                {
                    "type": "http",
                    "name": name,
                    "url": url,
                    "headers": [
                        {"name": str(k), "value": str(v)}
                        for k, v in headers_map.items()
                        if v is not None
                    ],
                }
            )
            continue

        cmd = str(translated.get("command") or "").strip()
        if not cmd:
            continue
        raw_env = translated.get("env")
        env_map: dict[str, str] = (
            {str(k): str(v) for k, v in raw_env.items() if v is not None}
            if isinstance(raw_env, Mapping)
            else {}
        )
        if session_env is not None:
            for env_key in (*PER_SESSION_MCP_ENV_KEYS, "KIROCREW_ROUTE_SPAWN_RUN"):
                val = session_env.get(env_key)
                if isinstance(val, str) and val:
                    env_map[env_key] = val

        seen_names.add(name)
        formatted.append(
            {
                "name": name,
                "command": cmd,
                "args": [str(arg) for arg in translated.get("args", [])],
                "env": [{"name": k, "value": v} for k, v in env_map.items()],
            }
        )

    return formatted


def normalize_acp_content_blocks(
    content_blocks: Sequence[Mapping[str, Any]] | None,
    *,
    prompt: str = "",
) -> list[dict[str, Any]]:
    """Normalize prompt inputs into native ACP `text`, `image`, and `resource` content blocks."""
    if not content_blocks:
        return [{"type": "text", "text": prompt}]

    normalized: list[dict[str, Any]] = []
    for block in content_blocks:
        if not isinstance(block, Mapping):
            continue
        btype = str(block.get("type") or "").strip()
        if btype == "text":
            normalized.append({"type": "text", "text": str(block.get("text") or "")})
        elif btype == "image":
            img_block: dict[str, Any] = {
                "type": "image",
                "data": str(block.get("data") or ""),
                "mimeType": str(block.get("mimeType") or "image/png"),
            }
            if "uri" in block and block["uri"] is not None:
                img_block["uri"] = str(block["uri"])
            normalized.append(img_block)
        elif btype == "resource":
            res_inner = block.get("resource")
            if isinstance(res_inner, Mapping):
                normalized.append({"type": "resource", "resource": dict(res_inner)})
            else:
                normalized.append(dict(block))
        elif btype == "resource_link":
            uri = str(block.get("uri") or "")
            name = str(block.get("name") or uri)
            normalized.append({"type": "text", "text": f"[Resource: {name} ({uri})]"})
        else:
            normalized.append(dict(block))

    return normalized or [{"type": "text", "text": prompt}]


def _build_session_meta(system_prompt_append: str | None) -> dict[str, Any] | None:
    """Build `_meta.antigravity.systemPrompt.append` when `system_prompt_append` is non-empty."""
    if isinstance(system_prompt_append, str) and system_prompt_append.strip():
        return {"antigravity": {"systemPrompt": {"append": system_prompt_append}}}
    return None


def _normalize_discovered_models(raw_models: Any) -> list[dict[str, str]]:
    """Normalize `models.availableModels` entries into `{modelId, name, description}` dicts."""
    if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_models:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("modelId") or item.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        name = str(item.get("name") or model_id).strip()
        description = str(item.get("description") or name).strip()
        out.append({"modelId": model_id, "name": name, "description": description})
    return out


def _resolve_supported_thought_level(
    effort: str | None,
    config_options: Any,
) -> str | None:
    """Map `effort` onto `thought_level` options supported by `agy acp`."""
    if not effort or not effort.strip():
        return None
    cleaned = effort.strip().lower()
    mapped = "high" if cleaned in {"xhigh", "max"} else cleaned
    if isinstance(config_options, Sequence) and not isinstance(config_options, (str, bytes)):
        for opt in config_options:
            if not isinstance(opt, Mapping) or opt.get("id") != "thought_level":
                continue
            raw_choices = opt.get("options")
            if isinstance(raw_choices, Sequence) and not isinstance(raw_choices, (str, bytes)):
                valid_values = {
                    str(c.get("value") or "").strip().lower()
                    for c in raw_choices
                    if isinstance(c, Mapping) and c.get("value")
                }
                if cleaned in valid_values:
                    return cleaned
                if mapped in valid_values:
                    return mapped
                return None
    return mapped if mapped in {"low", "medium", "high"} else None


def _select_permission_option_id(options: Any, *, allow: bool) -> str:
    """Select the `optionId` matching `allow_once` or `reject_once` from `params.options`."""
    target_exact_kind = "allow_once" if allow else "reject_once"
    target_keyword = "allow" if allow else "reject"
    fallback_ids = ("allow_once", "allow") if allow else ("reject_once", "deny", "reject")

    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for item in options:
            if isinstance(item, Mapping) and str(item.get("kind") or "") == target_exact_kind:
                opt_id = str(item.get("optionId") or "").strip()
                if opt_id:
                    return opt_id
        for item in options:
            if isinstance(item, Mapping) and target_keyword in str(item.get("kind") or "").lower():
                opt_id = str(item.get("optionId") or "").strip()
                if opt_id:
                    return opt_id
        for item in options:
            if isinstance(item, Mapping):
                opt_id = str(item.get("optionId") or "").strip()
                if opt_id.lower() in fallback_ids:
                    return opt_id
    return "allow_once" if allow else "reject_once"


def _extract_first_str(arguments: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        val = arguments.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _normalize_tool_arguments(
    arguments: Mapping[str, Any],
    *,
    tool_name: str = "",
) -> dict[str, Any]:
    """Normalize `agy` tool argument keys to ACP/KiroCrew standard keys (`command`, `path`)."""
    if tool_name.lower() == "call_mcp_tool" or (
        "ServerName" in arguments and "ToolName" in arguments
    ):
        inner = arguments.get("Arguments")
        if isinstance(inner, Mapping):
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


def _extract_tool_output(record: Mapping[str, Any]) -> str:
    """Extract text output from a merged `tool_call` / `tool_call_update` state dict."""
    raw_output = record.get("rawOutput")
    if isinstance(raw_output, Mapping):
        for key in ("output", "text", "error", "message"):
            val = raw_output.get(key)
            if isinstance(val, str) and val:
                return val
    elif isinstance(raw_output, str) and raw_output:
        return raw_output

    direct_out = record.get("output")
    if isinstance(direct_out, str) and direct_out:
        return direct_out

    content = record.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item.get("type") == "content" and isinstance(item.get("content"), Mapping):
                inner_txt = item["content"].get("text")
                if isinstance(inner_txt, str) and inner_txt:
                    parts.append(inner_txt)
        if parts:
            return "\n".join(parts)
    return ""


def _merge_tool_record(existing: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Merge incremental ACP `tool_call` / `tool_call_update` / `toolCall` fields."""
    merged = dict(existing)
    for key in ("toolCallId", "name", "toolName", "title", "kind", "status", "_meta"):
        val = incoming.get(key)
        if val is not None and val != "":
            merged[key] = val

    merged_args: dict[str, Any] = dict(merged.get("raw_args") or {})
    for arg_key in ("rawInput", "args"):
        candidate = incoming.get(arg_key)
        if isinstance(candidate, Mapping) and candidate:
            merged_args.update(candidate)
    merged["raw_args"] = merged_args

    locs = incoming.get("locations")
    if isinstance(locs, list) and locs:
        merged["locations"] = locs

    content = incoming.get("content")
    if isinstance(content, list) and content:
        merged["content"] = content

    if "rawOutput" in incoming and incoming["rawOutput"] is not None:
        merged["rawOutput"] = incoming["rawOutput"]
    if "output" in incoming and incoming["output"] is not None:
        merged["output"] = incoming["output"]

    return merged


def _resolve_raw_tool_name(record: Mapping[str, Any], raw_args: Mapping[str, Any]) -> str:
    """Resolve raw `agy` tool identifier from merged ACP tool metadata."""
    for key in ("name", "toolName"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    meta = record.get("_meta")
    if isinstance(meta, Mapping):
        kiro_meta = meta.get("kiro")
        if isinstance(kiro_meta, Mapping) and isinstance(kiro_meta.get("toolName"), str):
            k_name = kiro_meta["toolName"].strip()
            if k_name:
                return k_name
        if isinstance(meta.get("toolName"), str) and meta["toolName"].strip():
            return meta["toolName"].strip()

    if "ServerName" in raw_args and "ToolName" in raw_args:
        return "call_mcp_tool"

    title = str(record.get("title") or "").strip()
    if title and (" " not in title) and ("/" not in title or title.startswith("@")):
        return title.lstrip("@").replace("/", "_") if title.startswith("@") else title

    kind = str(record.get("kind") or "").strip().lower()
    return _KIND_TO_FALLBACK_TOOL.get(kind, title or "tool")


def _is_native_subagent_tool(raw_tool_name: str, cls: ToolCallClassification) -> bool:
    """Return True when the tool call is a built-in `agy` subagent tool (never MCP `spawn_run`)."""
    if cls.mcp_server:
        return False
    return (
        raw_tool_name.strip().lower() in _NATIVE_SUBAGENT_TOOLS
        or cls.kas_builtin_id == "invoke_sub_agent"
    )


def _build_tool_event_from_record(
    tc_id: str,
    record: Mapping[str, Any],
    *,
    known_mcp_servers: Sequence[str],
    status_override: str | None = None,
) -> tuple[str, ToolCallClassification, ToolCallEvent]:
    """Classify merged `record` into `(raw_tool_name, ToolCallClassification, ToolCallEvent)`."""
    raw_args_obj = record.get("raw_args")
    raw_args: dict[str, Any] = dict(raw_args_obj) if isinstance(raw_args_obj, Mapping) else {}
    raw_tool_name = _resolve_raw_tool_name(record, raw_args)
    raw_kind = str(record.get("kind") or "").strip().lower()

    cls = classify_tool_call_step(
        raw_kind or raw_tool_name,
        raw_tool_name,
        raw_args,
        known_mcp_servers=known_mcp_servers,
    )
    normalized_input = _normalize_tool_arguments(raw_args, tool_name=raw_tool_name)
    record_locs = record.get("locations")
    locations = cls.locations or (
        [dict(loc) for loc in record_locs if isinstance(loc, Mapping)]
        if isinstance(record_locs, list)
        else []
    )

    diff_blocks = list(cls.diff_content)
    if not diff_blocks and isinstance(record.get("content"), list):
        diff_blocks = [
            dict(b) for b in record["content"] if isinstance(b, Mapping) and b.get("type") == "diff"
        ]

    acp_kind = cls.acp_kind
    if acp_kind == "other" and not cls.mcp_server and raw_kind in _KIND_TO_FALLBACK_TOOL:
        acp_kind = raw_kind

    status = status_override or str(record.get("status") or "in_progress")
    event = ToolCallEvent(
        tool_call_id=tc_id,
        title=cls.title,
        kind=acp_kind,
        status=status,
        raw_input=normalized_input,
        output=_extract_tool_output(record),
        locations=locations,
        kiro_tool_name=cls.v2_tool_name,
        mcp_server_name=cls.mcp_server,
        kiro_tool_id=cls.kiro_tool_id,
        kas_builtin_id=cls.kas_builtin_id,
        v2_tool_name=cls.v2_tool_name,
        acp_kind=acp_kind,
        mcp_server=cls.mcp_server,
        mcp_tool=cls.mcp_tool,
        command=cls.command,
        subagent_target=cls.subagent_target,
        diff_content=diff_blocks,
    )
    return raw_tool_name, cls, event


@dataclass
class _QueuedAcpSteer:
    """Internal steer entry pairing `QueuedSteer` metadata with native ACP `content_blocks`."""

    steer: QueuedSteer
    content_blocks: list[dict[str, Any]]


@dataclass
class _ActiveTurnState:
    """Mutable per-`session/prompt` turn state for routing inbound notifications and permissions."""

    session_id: str
    agy_session_id: str
    route_spawn_run: bool
    skip_permissions: bool
    known_mcp_servers: tuple[str, ...]
    on_thought: Callable[[str], Awaitable[None]] | None
    on_text: Callable[[str], Awaitable[None]] | None
    on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None
    on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None
    on_usage: Callable[[int], Awaitable[None]] | None
    collected_texts: list[str]
    usage: dict[str, int]
    stream_lock: anyio.Lock = field(default_factory=anyio.Lock)
    interrupt_event: anyio.Event = field(default_factory=anyio.Event)
    tool_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    announced_tools: set[str] = field(default_factory=set)
    completed_tools: set[str] = field(default_factory=set)
    blocked_subagent_tools: set[str] = field(default_factory=set)
    perm_arrived_events: dict[str, anyio.Event] = field(default_factory=dict)
    decision_events: dict[str, anyio.Event] = field(default_factory=dict)
    decisions: dict[str, bool] = field(default_factory=dict)
    policy_denied_tools: list[str] = field(default_factory=list)
    last_context_tokens: int = 0
    emitted_any_signal: bool = False
    in_flight_handlers: int = 0
    drained_event: anyio.Event = field(default_factory=anyio.Event)

    def __post_init__(self) -> None:
        self.drained_event.set()

    def handler_enter(self) -> None:
        self.in_flight_handlers += 1
        if self.in_flight_handlers == 1:
            self.drained_event = anyio.Event()

    def handler_exit(self) -> None:
        self.in_flight_handlers = max(0, self.in_flight_handlers - 1)
        if self.in_flight_handlers == 0:
            self.drained_event.set()


@dataclass
class _PendingRpc:
    event: anyio.Event = field(default_factory=anyio.Event)
    response: dict[str, Any] | None = None
    error: BaseException | None = None


class _AcpSubprocessConnection:
    """Single full-duplex `agy acp --stdio --process-per-session` child connection."""

    def __init__(
        self,
        *,
        bin_path: str,
        env: dict[str, str],
        save_dir_or_db: Path,
        on_server_request: Callable[
            [_AcpSubprocessConnection, dict[str, Any]], Coroutine[Any, Any, None]
        ],
        on_notification: Callable[
            [_AcpSubprocessConnection, dict[str, Any]], Coroutine[Any, Any, None]
        ],
        on_disconnect: Callable[[_AcpSubprocessConnection], None],
    ) -> None:
        self._bin_path = bin_path
        self._env = env
        self._save_dir_or_db = save_dir_or_db
        self._on_server_request = on_server_request
        self._on_notification = on_notification
        self._on_disconnect = on_disconnect

        self._write_lock = anyio.Lock()
        self._next_id = 0
        self._pending: dict[int | str, _PendingRpc] = {}
        self._proc: anyio.abc.Process | None = None
        self._ready_event = anyio.Event()
        self._shutdown_event = anyio.Event()
        self._closed_event = anyio.Event()
        self._spawn_error: BaseException | None = None
        self._closed = False
        self._stderr_tail: list[str] = []
        self._notif_send, self._notif_recv = anyio.create_memory_object_stream[dict[str, Any]](
            math.inf
        )
        self._inflight_frames = 0
        self._idle_event = anyio.Event()
        self._idle_event.set()
        self.loaded_sessions: set[str] = set()
        self.session_results: dict[str, dict[str, Any]] = {}
        self.session_prefs: dict[str, tuple[str, str, str]] = {}
        self.initialized = False

    @property
    def is_closed(self) -> bool:
        if self._closed:
            return True
        if self._proc is not None and self._proc.returncode is not None:
            return True
        return False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    async def start(self) -> None:
        """Spawn the supervisor actor task and await subprocess readiness."""
        loop = asyncio.get_running_loop()
        loop.create_task(self._actor_main())
        await self._ready_event.wait()
        if self._spawn_error is not None:
            raise self._spawn_error

    async def close(self) -> None:
        """Signal the supervisor actor task to terminate the child process and wait."""
        self._shutdown_event.set()
        self._fail_pending(AgyAcpTransportError("ACP connection closed"))
        await self._closed_event.wait()

    def _fail_pending(self, exc: BaseException) -> None:
        self._closed = True
        for pending in list(self._pending.values()):
            if pending.response is None and pending.error is None:
                pending.error = exc
            pending.event.set()
        self._pending.clear()

    async def send_payload(self, payload: dict[str, Any]) -> None:
        """Serialize and write a single JSON-RPC line to child `stdin` under `anyio.Lock`."""
        if self.is_closed or self._proc is None or self._proc.stdin is None:
            raise AgyAcpTransportError("agy acp stdin is closed")
        encoded = (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        try:
            async with self._write_lock:
                await self._proc.stdin.send(encoded)
        except (
            BrokenPipeError,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            OSError,
        ) as exc:
            self._fail_pending(AgyAcpTransportError(f"agy acp stdin write failed: {exc}"))
            self._shutdown_event.set()
            raise AgyAcpTransportError(f"agy acp stdin write failed: {exc}") from exc

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        await self.send_payload({"jsonrpc": "2.0", "method": method, "params": params})

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and await its correlated response."""
        if self.is_closed:
            raise AgyAcpTransportError("agy acp connection is closed")
        self._next_id += 1
        req_id = self._next_id
        slot = _PendingRpc()
        self._pending[req_id] = slot
        try:
            await self.send_payload(
                {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            )
            if timeout is not None and timeout > 0:
                with anyio.fail_after(timeout):
                    await slot.event.wait()
            else:
                await slot.event.wait()
        except TimeoutError as exc:
            raise TimeoutError(f"Timed out waiting for agy acp '{method}' response") from exc
        finally:
            self._pending.pop(req_id, None)

        if slot.error is not None:
            raise slot.error
        resp = slot.response or {}
        if "error" in resp and isinstance(resp["error"], Mapping):
            err_obj = resp["error"]
            raise AgyAcpRpcError(
                int(err_obj.get("code", -32603)),
                str(err_obj.get("message") or "RPC error"),
                err_obj.get("data"),
            )
        result = resp.get("result")
        return dict(result) if isinstance(result, Mapping) else {}

    async def _actor_main(self) -> None:
        args = [self._bin_path, "acp", "--stdio", "--process-per-session"]
        try:
            log_acp_event(
                "agy_acp.spawn",
                save_dir_or_db=self._save_dir_or_db,
                bin_path=self._bin_path,
                args=args[1:],
            )
            proc = await anyio.open_process(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                start_new_session=True,
            )
            self._proc = proc
        except BaseException as exc:
            self._spawn_error = AgyAcpTransportError(f"Failed to spawn agy acp: {exc}")
            self._closed = True
            self._ready_event.set()
            self._closed_event.set()
            return

        try:
            async with proc:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._notification_worker)
                    tg.start_soon(self._stdout_reader, proc, tg)
                    tg.start_soon(self._stderr_reader, proc)
                    tg.start_soon(self._shutdown_watcher, proc, tg)
                    self._ready_event.set()
                with contextlib.suppress(Exception):
                    await proc.wait()
        except BaseException as exc:
            self._fail_pending(AgyAcpTransportError(f"agy acp transport crashed: {exc}"))
        finally:
            stderr_summary = "".join(self._stderr_tail[-20:]).strip()
            self._fail_pending(
                AgyAcpTransportError(
                    f"agy acp exited (code={proc.returncode})"
                    + (f": {stderr_summary}" if stderr_summary else "")
                )
            )
            log_acp_event(
                "agy_acp.exit",
                save_dir_or_db=self._save_dir_or_db,
                returncode=proc.returncode,
                stderr=stderr_summary,
            )
            self._on_disconnect(self)
            self._ready_event.set()
            self._closed_event.set()

    async def _shutdown_watcher(
        self,
        proc: anyio.abc.Process,
        tg: anyio.abc.TaskGroup,
    ) -> None:
        await self._shutdown_event.wait()
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                await proc.stdin.aclose()
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
            with anyio.move_on_after(0.5):
                await proc.wait()
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
        tg.cancel_scope.cancel()

    async def _stderr_reader(self, proc: anyio.abc.Process) -> None:
        if proc.stderr is None:
            return
        with contextlib.suppress(Exception):
            async for chunk in TextReceiveStream(proc.stderr):
                self._stderr_tail.append(chunk)
                if len(self._stderr_tail) > 50:
                    del self._stderr_tail[:25]

    async def _stdout_reader(
        self,
        proc: anyio.abc.Process,
        tg: anyio.abc.TaskGroup,
    ) -> None:
        if proc.stdout is None:
            self._shutdown_event.set()
            return
        buffer = ""
        try:
            async for chunk in TextReceiveStream(proc.stdout):
                buffer += chunk
                while "\n" in buffer:
                    raw_line, buffer = buffer.split("\n", 1)
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    self._route_frame(msg, tg)
        except anyio.BrokenResourceError, anyio.ClosedResourceError, OSError:
            pass
        finally:
            stderr_summary = "".join(self._stderr_tail[-20:]).strip()
            with contextlib.suppress(Exception):
                self._notif_send.close()
            self._fail_pending(
                AgyAcpTransportError(
                    "agy acp stdout closed unexpectedly"
                    + (f": {stderr_summary}" if stderr_summary else "")
                )
            )
            self._shutdown_event.set()

    def _begin_frame(self) -> None:
        if self._inflight_frames == 0:
            self._idle_event = anyio.Event()
        self._inflight_frames += 1

    def _finish_frame(self) -> None:
        self._inflight_frames = max(0, self._inflight_frames - 1)
        if self._inflight_frames == 0:
            self._idle_event.set()

    async def wait_idle(self) -> None:
        """Wait until all inbound notifications and server requests received so far finish."""
        while self._inflight_frames > 0:
            await self._idle_event.wait()

    async def _notification_worker(self) -> None:
        with self._notif_recv:
            async for msg in self._notif_recv:
                try:
                    await self._on_notification(self, msg)
                finally:
                    self._finish_frame()

    async def _run_server_request(self, msg: dict[str, Any]) -> None:
        try:
            await self._on_server_request(self, msg)
        finally:
            self._finish_frame()

    def _route_frame(self, msg: dict[str, Any], tg: anyio.abc.TaskGroup) -> None:
        """Dispatch JSON-RPC responses immediately, FIFO notifications, and async server RPCs."""
        has_method = isinstance(msg.get("method"), str) and bool(msg["method"])
        if not has_method and "id" in msg and ("result" in msg or "error" in msg):
            resp_id = msg["id"]
            slot = self._pending.get(resp_id)
            if slot is not None:
                slot.response = msg
                slot.event.set()
            return

        if has_method and "id" in msg:
            self._begin_frame()
            tg.start_soon(self._run_server_request, msg)
            return

        if has_method:
            self._begin_frame()
            try:
                self._notif_send.send_nowait(msg)
            except anyio.BrokenResourceError, anyio.ClosedResourceError, anyio.WouldBlock:
                self._finish_frame()


class AgyAcpClientProvider:
    """Full-duplex ACP client driving `agy acp --stdio --process-per-session`."""

    def __init__(
        self,
        *,
        bin_path: str | Path | None = None,
        save_dir: Path | None = None,
        state_store: SqliteStateStoreProvider | None = None,
        env: Mapping[str, str] | None = None,
        rpc_timeout: float = 60.0,
    ) -> None:
        self._explicit_bin = bin_path
        self._store = state_store or SqliteStateStoreProvider(save_dir)
        default_state_dir = Path.home() / ".config" / "agy-acp"
        self._is_custom_store = self._store.db_path.parent != default_state_dir
        self._base_env: dict[str, str] = dict(env) if env is not None else {}
        self._rpc_timeout = rpc_timeout

        self._conn_lock = anyio.Lock()
        self._conn: _AcpSubprocessConnection | None = None
        self._cached_models: list[dict[str, str]] = []

        self._active_turns: set[str] = set()
        self._cancelled_sessions: set[str] = set()
        self._session_to_agy_sid: dict[str, str] = {}
        self._turn_states: dict[str, _ActiveTurnState] = {}
        self._steer_queues: dict[str, list[_QueuedAcpSteer]] = {}

    @property
    def state_store(self) -> SqliteStateStoreProvider:
        """Return the underlying `SqliteStateStoreProvider`."""
        return self._store

    @property
    def is_custom_store(self) -> bool:
        """Return whether this provider uses a non-default isolated state directory."""
        return self._is_custom_store

    @property
    def has_explicit_bin(self) -> bool:
        """Return whether a binary path was explicitly configured via arg or `AGY_BIN`."""
        return bool(self._explicit_bin or os.environ.get("AGY_BIN"))

    @property
    def bin_path(self) -> str:
        """Resolve the `agy` CLI executable path dynamically."""
        return resolve_agy_binary(self._explicit_bin)

    @property
    def worker_pid(self) -> int | None:
        """Return the active `agy acp` process PID if connected, else `None`."""
        return self._conn.pid if self._conn is not None and not self._conn.is_closed else None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Shut down the active `agy acp` subprocess cleanly."""
        async with self._conn_lock:
            conn = self._conn
            self._conn = None
        if conn is not None:
            await conn.close()

    aclose = close

    async def run_slash_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        session_id: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Run a non-interactive slash command (`agy -p <cmd> --output-format json`)."""
        from agy_acp.providers.quota_provider import run_agy_slash_json

        _ = session_id
        return await run_agy_slash_json(
            command,
            agy_bin=self.bin_path,
            cwd=cwd,
            timeout=timeout,
        )

    def _on_connection_disconnect(self, conn: _AcpSubprocessConnection) -> None:
        if self._conn is conn:
            self._conn = None

    async def _ensure_connection(self) -> _AcpSubprocessConnection:
        """Lazily spawn and `initialize` `agy acp`, auto-restarting after exit or EOF."""
        async with self._conn_lock:
            if self._conn is not None and not self._conn.is_closed and self._conn.initialized:
                return self._conn

            stale = self._conn
            self._conn = None
            if stale is not None:
                with contextlib.suppress(Exception):
                    await stale.close()

            scrubbed_keys = {*PER_SESSION_MCP_ENV_KEYS, "KIROCREW_ROUTE_SPAWN_RUN"}
            child_env = {k: v for k, v in os.environ.items() if k not in scrubbed_keys}
            child_env.update(self._base_env)
            child_env.setdefault("AGY_ACP_STATE_DB", str(self._store.db_path))
            conn = _AcpSubprocessConnection(
                bin_path=self.bin_path,
                env=child_env,
                save_dir_or_db=self._store.db_path,
                on_server_request=self._handle_server_request,
                on_notification=self._handle_notification,
                on_disconnect=self._on_connection_disconnect,
            )
            await conn.start()
            try:
                await conn.request(
                    "initialize",
                    {"protocolVersion": 1, "clientCapabilities": {}},
                    timeout=self._rpc_timeout,
                )
            except BaseException:
                await conn.close()
                raise
            conn.initialized = True
            self._conn = conn
            return conn

    async def _rpc_with_reconnect(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
        retry_on_disconnect: bool = True,
    ) -> tuple[_AcpSubprocessConnection, dict[str, Any]]:
        """Execute a JSON-RPC call on `agy acp`, reconnecting once if transport broke."""
        conn = await self._ensure_connection()
        try:
            res = await conn.request(method, params, timeout=timeout)
            return conn, res
        except AgyAcpTransportError:
            if not retry_on_disconnect:
                raise
            conn = await self._ensure_connection()
            res = await conn.request(method, params, timeout=timeout)
            return conn, res

    def _cache_models(self, raw_models: Any) -> list[dict[str, str]]:
        models = _normalize_discovered_models(raw_models)
        if models:
            self._cached_models = models
            self._store.save_models(models)
        return models

    async def discover_models(
        self,
        *,
        cwd: str = ".",
        refresh: bool = False,
    ) -> list[dict[str, str]]:
        """Return cached `availableModels` or probe via `session/new` + `session/close`."""
        if not refresh:
            if self._cached_models:
                return [dict(m) for m in self._cached_models]
            stored = self._store.load_models()
            if stored:
                self._cached_models = stored
                return [dict(m) for m in stored]

        resolved_cwd = str(Path(cwd).resolve()) if cwd else str(Path.cwd())
        temp_sid = ""
        try:
            conn, res = await self._rpc_with_reconnect(
                "session/new",
                {"cwd": resolved_cwd, "mcpServers": []},
                timeout=self._rpc_timeout,
            )
            temp_sid = str(res.get("sessionId") or "").strip()
            models_block = res.get("models")
            raw_avail = (
                models_block.get("availableModels") if isinstance(models_block, Mapping) else None
            )
            discovered = self._cache_models(raw_avail)
            if temp_sid:
                with contextlib.suppress(Exception):
                    await conn.request(
                        "session/close",
                        {"sessionId": temp_sid},
                        timeout=10.0,
                    )
            if discovered:
                return [dict(m) for m in discovered]
        except AgyAcpError, OSError, TimeoutError:
            if temp_sid and self._conn is not None and not self._conn.is_closed:
                with contextlib.suppress(Exception):
                    await self._conn.request(
                        "session/close",
                        {"sessionId": temp_sid},
                        timeout=5.0,
                    )

        return [dict(m) for m in FALLBACK_MODELS]

    async def _apply_session_preferences(
        self,
        conn: _AcpSubprocessConnection,
        agy_sid: str,
        *,
        session_result: Mapping[str, Any],
        model: str | None,
        mode: str,
        effort: str | None,
        available_models: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        """Configure backend `mode` (`default`/`plan`), `model`, and `thought_level`."""
        target_agy_mode = "plan" if mode in {"read-only", "plan"} else "default"
        with contextlib.suppress(AgyAcpRpcError):
            await conn.request(
                "session/set_config_option",
                {
                    "sessionId": agy_sid,
                    "configId": "mode",
                    "value": target_agy_mode,
                },
                timeout=self._rpc_timeout,
            )

        models_block = session_result.get("models")
        current_model_id = (
            str(models_block.get("currentModelId") or "").strip()
            if isinstance(models_block, Mapping)
            else ""
        )
        resolved_model = current_model_id
        fallback_from_model = ""

        if model and model.strip() and model.strip().lower() != "auto":
            clean_model = model.strip()
            candidate_model = (
                resolve_model_for_effort(clean_model, effort, available_models) or clean_model
            )
            try:
                await conn.request(
                    "session/set_model",
                    {"sessionId": agy_sid, "modelId": candidate_model},
                    timeout=self._rpc_timeout,
                )
                resolved_model = candidate_model
            except AgyAcpRpcError:
                fallback_from_model = clean_model

        config_options = session_result.get("configOptions")
        target_thought = _resolve_supported_thought_level(effort, config_options)
        if target_thought is not None:
            with contextlib.suppress(AgyAcpRpcError):
                await conn.request(
                    "session/set_config_option",
                    {
                        "sessionId": agy_sid,
                        "configId": "thought_level",
                        "value": target_thought,
                    },
                    timeout=self._rpc_timeout,
                )

        return resolved_model, fallback_from_model

    @staticmethod
    def _extract_tier(session_result: Mapping[str, Any]) -> dict[str, Any] | None:
        meta = session_result.get("_meta")
        if not isinstance(meta, Mapping):
            return None
        agy_meta = meta.get("antigravity")
        if not isinstance(agy_meta, Mapping):
            return None
        tier = agy_meta.get("tier")
        return dict(tier) if isinstance(tier, Mapping) else None

    async def create_backend_session(
        self,
        *,
        cwd: str,
        mcp_servers: Sequence[Mapping[str, Any]] | None = None,
        system_prompt_append: str | None = None,
        model: str | None = None,
        mode: str = "auto",
        effort: str | None = "medium",
        session_env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a new `agy acp` session (`session/new`) and apply mode/model/thought settings."""
        params: dict[str, Any] = {
            "cwd": str(cwd),
            "mcpServers": format_acp_mcp_servers(mcp_servers, session_env=session_env),
        }
        meta = _build_session_meta(system_prompt_append)
        if meta is not None:
            params["_meta"] = meta

        conn, res = await self._rpc_with_reconnect(
            "session/new",
            params,
            timeout=self._rpc_timeout,
        )
        agy_sid = str(res.get("sessionId") or "").strip()
        if not agy_sid:
            raise AgyAcpRpcError(-32603, "session/new returned empty sessionId")

        conn.loaded_sessions.add(agy_sid)
        conn.session_results[agy_sid] = dict(res)
        models_block = res.get("models")
        raw_avail = (
            models_block.get("availableModels") if isinstance(models_block, Mapping) else None
        )
        models = self._cache_models(raw_avail) or list(self._cached_models)
        resolved_model, fallback_from = await self._apply_session_preferences(
            conn,
            agy_sid,
            session_result=res,
            model=model,
            mode=mode,
            effort=effort,
            available_models=models,
        )
        conn.session_prefs[agy_sid] = (model or "", mode, effort or "")
        return {
            "agySessionId": agy_sid,
            "models": models,
            "currentModelId": resolved_model,
            "fallbackFromModel": fallback_from,
            "tier": self._extract_tier(res),
            "modes": dict(res["modes"]) if isinstance(res.get("modes"), Mapping) else {},
            "configOptions": (
                list(res["configOptions"]) if isinstance(res.get("configOptions"), list) else []
            ),
        }

    async def load_backend_session(
        self,
        *,
        agy_session_id: str,
        cwd: str,
        mcp_servers: Sequence[Mapping[str, Any]] | None = None,
        system_prompt_append: str | None = None,
        model: str | None = None,
        mode: str = "auto",
        effort: str | None = "medium",
        session_env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Load an existing `agy acp` session (`session/load`), falling back to `session/new`."""
        params: dict[str, Any] = {
            "sessionId": agy_session_id,
            "cwd": str(cwd),
            "mcpServers": format_acp_mcp_servers(mcp_servers, session_env=session_env),
        }
        meta = _build_session_meta(system_prompt_append)
        if meta is not None:
            params["_meta"] = meta

        try:
            conn, res = await self._rpc_with_reconnect(
                "session/load",
                params,
                timeout=self._rpc_timeout,
            )
            loaded_sid = str(res.get("sessionId") or agy_session_id).strip()
            if not loaded_sid:
                raise AgyAcpRpcError(-32002, "session/load returned empty sessionId")
        except AgyAcpRpcError:
            return await self.create_backend_session(
                cwd=cwd,
                mcp_servers=mcp_servers,
                system_prompt_append=system_prompt_append,
                model=model,
                mode=mode,
                effort=effort,
                session_env=session_env,
            )

        conn.loaded_sessions.add(loaded_sid)
        conn.session_results[loaded_sid] = dict(res)
        models_block = res.get("models")
        raw_avail = (
            models_block.get("availableModels") if isinstance(models_block, Mapping) else None
        )
        models = self._cache_models(raw_avail) or list(self._cached_models)
        resolved_model, fallback_from = await self._apply_session_preferences(
            conn,
            loaded_sid,
            session_result=res,
            model=model,
            mode=mode,
            effort=effort,
            available_models=models,
        )
        conn.session_prefs[loaded_sid] = (model or "", mode, effort or "")
        return {
            "agySessionId": loaded_sid,
            "models": models,
            "currentModelId": resolved_model,
            "fallbackFromModel": fallback_from,
            "tier": self._extract_tier(res),
            "modes": dict(res["modes"]) if isinstance(res.get("modes"), Mapping) else {},
            "configOptions": (
                list(res["configOptions"]) if isinstance(res.get("configOptions"), list) else []
            ),
        }

    async def set_backend_model(self, agy_session_id: str, model_id: str) -> dict[str, Any]:
        """Call `session/set_model` on `agy acp`."""
        _, res = await self._rpc_with_reconnect(
            "session/set_model",
            {"sessionId": agy_session_id, "modelId": model_id},
            timeout=self._rpc_timeout,
        )
        return res

    async def set_backend_config_option(
        self,
        agy_session_id: str,
        config_id: str,
        value: str,
    ) -> dict[str, Any]:
        """Call `session/set_config_option` (`model`, `thought_level`, `mode`) on `agy acp`."""
        _, res = await self._rpc_with_reconnect(
            "session/set_config_option",
            {"sessionId": agy_session_id, "configId": config_id, "value": value},
            timeout=self._rpc_timeout,
        )
        return res

    def evict_backend_session(self, agy_session_id: str) -> None:
        """Synchronously drop `agy_session_id` from loaded session tracking."""
        if not agy_session_id or self._conn is None:
            return
        self._conn.loaded_sessions.discard(agy_session_id)
        self._conn.session_results.pop(agy_session_id, None)
        self._conn.session_prefs.pop(agy_session_id, None)

    async def close_backend_session(self, agy_session_id: str) -> dict[str, Any]:
        """Close `agy_session_id` via `session/close` and evict from loaded session set."""
        await anyio.sleep(0)
        if not agy_session_id or self._conn is None or self._conn.is_closed:
            return {}
        self.evict_backend_session(agy_session_id)
        with contextlib.suppress(AgyAcpError, TimeoutError, OSError):
            return await self._conn.request(
                "session/close",
                {"sessionId": agy_session_id},
                timeout=10.0,
            )
        return {}

    def is_turn_active(self, session_id: str) -> bool:
        """Return whether `session_id` currently has an active `stream_prompt` turn."""
        return session_id in self._active_turns

    def _pop_queued_steers(self, session_id: str) -> list[_QueuedAcpSteer]:
        return list(self._steer_queues.pop(session_id, []))

    def clear_steers(self, session_id: str) -> list[str]:
        """Discard queued mid-turn steers for `session_id` and unlink any staged temp files."""
        entries = self._pop_queued_steers(session_id)
        for entry in entries:
            for tmp in entry.steer.temp_files:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
        return [e.steer.raw_message for e in entries if e.steer.raw_message]

    def discard_cancelled(self, session_id: str) -> None:
        """Clear any stale cancellation marker for `session_id`."""
        self._cancelled_sessions.discard(session_id)

    async def _send_agy_cancel(self, agy_sid: str) -> None:
        conn = self._conn
        if agy_sid and conn is not None and not conn.is_closed:
            with contextlib.suppress(AgyAcpError, OSError):
                await conn.send_notification("session/cancel", {"sessionId": agy_sid})

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an in-flight turn for `session_id` via non-blocking `session/cancel`."""
        await anyio.sleep(0)
        active = self.is_turn_active(session_id)
        if active:
            self._cancelled_sessions.add(session_id)
        for entry in self._pop_queued_steers(session_id):
            for tmp in entry.steer.temp_files:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
        agy_sid = self._session_to_agy_sid.get(session_id, "")
        turn_state = self._turn_states.get(agy_sid) if agy_sid else None
        if turn_state is not None:
            turn_state.interrupt_event.set()
        if agy_sid:
            await self._send_agy_cancel(agy_sid)
        return active

    async def steer_session(
        self,
        session_id: str,
        *,
        raw_message: str = "",
        prompt: str = "",
        content_blocks: Sequence[Mapping[str, Any]] | None = None,
        temp_files: list[Path] | None = None,
    ) -> bool:
        """Queue a mid-turn steer and send `session/cancel` to transition into the steered turn."""
        await anyio.sleep(0)
        if not self.is_turn_active(session_id) or session_id in self._cancelled_sessions:
            return False

        blocks = normalize_acp_content_blocks(content_blocks, prompt=prompt)
        effective_raw = (
            raw_message
            or prompt
            or " ".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text").strip()
        )
        queued = QueuedSteer(
            raw_message=effective_raw,
            prompt=prompt,
            temp_files=list(temp_files or []),
        )
        self._steer_queues.setdefault(session_id, []).append(
            _QueuedAcpSteer(steer=queued, content_blocks=blocks)
        )
        agy_sid = self._session_to_agy_sid.get(session_id, "")
        turn_state = self._turn_states.get(agy_sid) if agy_sid else None
        if turn_state is not None:
            turn_state.interrupt_event.set()
        if agy_sid:
            await self._send_agy_cancel(agy_sid)
        return True

    async def _evaluate_tool_call_once(
        self,
        turn: _ActiveTurnState,
        tc_id: str,
        event: ToolCallEvent,
    ) -> tuple[bool, bool]:
        """Evaluate `on_tool_call(event)` at most once per `tc_id`.

        Returns `(allowed, interrupted_by_cancel_or_steer)`.
        """
        if tc_id in turn.decision_events:
            await turn.decision_events[tc_id].wait()
            interrupted = turn.interrupt_event.is_set()
            return turn.decisions.get(tc_id, False), interrupted

        dec_event = anyio.Event()
        turn.decision_events[tc_id] = dec_event
        turn.announced_tools.add(tc_id)

        tool_cb = turn.on_tool_call
        if tool_cb is None:
            allowed = turn.skip_permissions
            turn.decisions[tc_id] = allowed
            dec_event.set()
            return allowed, turn.interrupt_event.is_set()

        result_holder: dict[str, bool] = {"allowed": False, "interrupted": False}

        async def _run_callback(tg: anyio.abc.TaskGroup) -> None:
            result_holder["allowed"] = await tool_cb(event)
            tg.cancel_scope.cancel()

        async def _watch_interrupt(tg: anyio.abc.TaskGroup) -> None:
            await turn.interrupt_event.wait()
            result_holder["interrupted"] = True
            tg.cancel_scope.cancel()

        try:
            async with anyio.create_task_group() as race_tg:
                race_tg.start_soon(_run_callback, race_tg)
                race_tg.start_soon(_watch_interrupt, race_tg)
        finally:
            turn.decisions[tc_id] = result_holder["allowed"]
            dec_event.set()

        return result_holder["allowed"], result_holder["interrupted"]

    async def _handle_server_request(
        self,
        conn: _AcpSubprocessConnection,
        msg: dict[str, Any],
    ) -> None:
        """Handle inbound server-to-client RPCs (`session/request_permission`) via `tg`."""
        req_id = msg.get("id")
        method = str(msg.get("method") or "")
        params = msg.get("params")
        params_dict = params if isinstance(params, Mapping) else {}

        if method != "session/request_permission":
            with contextlib.suppress(AgyAcpError):
                await conn.send_payload(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Unsupported method: {method}"},
                    }
                )
            return

        agy_sid = str(params_dict.get("sessionId") or "").strip()
        turn = self._turn_states.get(agy_sid)
        options = params_dict.get("options")
        if turn is None:
            with contextlib.suppress(AgyAcpError):
                await conn.send_payload(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"outcome": {"outcome": "cancelled"}},
                    }
                )
            return

        turn.handler_enter()
        try:
            turn.emitted_any_signal = True
            raw_tc = params_dict.get("toolCall")
            tc_dict = raw_tc if isinstance(raw_tc, Mapping) else {}
            tc_id = str(tc_dict.get("toolCallId") or f"perm_{req_id}").strip()

            perm_ev = turn.perm_arrived_events.setdefault(tc_id, anyio.Event())
            perm_ev.set()

            merged = _merge_tool_record(turn.tool_records.get(tc_id, {}), tc_dict)
            turn.tool_records[tc_id] = merged
            raw_name, cls, event = _build_tool_event_from_record(
                tc_id,
                merged,
                known_mcp_servers=turn.known_mcp_servers,
                status_override="in_progress",
            )

            if turn.route_spawn_run and _is_native_subagent_tool(raw_name, cls):
                turn.blocked_subagent_tools.add(tc_id)
                reject_opt = _select_permission_option_id(options, allow=False)
                await conn.send_payload(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "outcome": {"outcome": "selected", "optionId": reject_opt},
                        },
                    }
                )
                return

            if turn.interrupt_event.is_set():
                await conn.send_payload(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"outcome": {"outcome": "cancelled"}},
                    }
                )
                return

            allowed, interrupted = await self._evaluate_tool_call_once(turn, tc_id, event)
            if interrupted or turn.interrupt_event.is_set():
                outcome_payload: dict[str, Any] = {"outcome": {"outcome": "cancelled"}}
            else:
                chosen_opt = _select_permission_option_id(options, allow=allowed)
                outcome_payload = {
                    "outcome": {"outcome": "selected", "optionId": chosen_opt},
                }
            await conn.send_payload({"jsonrpc": "2.0", "id": req_id, "result": outcome_payload})
        except AgyAcpError:
            pass
        finally:
            turn.handler_exit()

    async def _handle_notification(
        self,
        conn: _AcpSubprocessConnection,
        msg: dict[str, Any],
    ) -> None:
        """Handle inbound `session/update` notifications spawned via `tg.start_soon`."""
        _ = conn
        if msg.get("method") != "session/update":
            return
        params = msg.get("params")
        if not isinstance(params, Mapping):
            return
        update = params.get("update")
        if not isinstance(update, Mapping):
            return

        kind = str(update.get("sessionUpdate") or "")
        if kind == "available_commands_update":
            return

        agy_sid = str(params.get("sessionId") or "").strip()
        turn = self._turn_states.get(agy_sid)
        if turn is None:
            return

        turn.handler_enter()
        try:
            await self._dispatch_turn_update(turn, kind, update)
        finally:
            turn.handler_exit()

    async def _dispatch_turn_update(
        self,
        turn: _ActiveTurnState,
        kind: str,
        update: Mapping[str, Any],
    ) -> None:
        if kind == "agent_thought_chunk":
            content = update.get("content")
            delta = (
                str(content.get("text") or "")
                if isinstance(content, Mapping)
                else str(update.get("text") or "")
            )
            if delta:
                turn.emitted_any_signal = True
                async with turn.stream_lock:
                    if turn.on_thought is not None:
                        await turn.on_thought(delta)
            return

        if kind == "agent_message_chunk":
            content = update.get("content")
            delta = (
                str(content.get("text") or "")
                if isinstance(content, Mapping)
                else str(update.get("text") or "")
            )
            if delta:
                turn.emitted_any_signal = True
                async with turn.stream_lock:
                    turn.collected_texts.append(delta)
                    if turn.on_text is not None:
                        await turn.on_text(delta)
            return

        if kind == "usage_update":
            turn.emitted_any_signal = True
            ctx_tokens = int(
                update.get("used")
                or update.get("contextTokens")
                or update.get("context_tokens")
                or 0
            )
            if ctx_tokens > 0:
                turn.last_context_tokens = ctx_tokens
                turn.usage["context_tokens"] = ctx_tokens
                persisted = self._store.get_session(turn.session_id)
                if persisted is not None:
                    persisted["contextTokens"] = ctx_tokens
                    self._store.upsert_session(persisted)
                self._store.record_tokens(
                    input_tokens=ctx_tokens,
                    scope=f"session:{turn.session_id}",
                )
                async with turn.stream_lock:
                    if turn.on_usage is not None:
                        await turn.on_usage(ctx_tokens)
            return

        if kind in {"tool_call", "tool_call_update"}:
            turn.emitted_any_signal = True
            tc_id = str(update.get("toolCallId") or "").strip()
            if not tc_id:
                return
            merged = _merge_tool_record(turn.tool_records.get(tc_id, {}), update)
            turn.tool_records[tc_id] = merged

            raw_status = str(update.get("status") or merged.get("status") or "in_progress").lower()
            failed_statuses = {"failed", "error", "cancelled", "canceled"}
            done_statuses = {"completed", "done", "succeeded"}
            raw_name, cls, event = _build_tool_event_from_record(
                tc_id,
                merged,
                known_mcp_servers=turn.known_mcp_servers,
                status_override=(
                    "failed"
                    if raw_status in failed_statuses
                    else ("completed" if raw_status in done_statuses else "in_progress")
                ),
            )

            if turn.route_spawn_run and _is_native_subagent_tool(raw_name, cls):
                turn.blocked_subagent_tools.add(tc_id)
                return

            if tc_id in turn.blocked_subagent_tools:
                return

            perm_ev = turn.perm_arrived_events.setdefault(tc_id, anyio.Event())
            if raw_status == "pending":
                perm_ev.set()

            allowed, interrupted = await self._evaluate_tool_call_once(
                turn,
                tc_id,
                replace(event, status="in_progress"),
            )
            if interrupted:
                return

            if not allowed and not perm_ev.is_set():
                if cls.acp_kind not in _READ_ONLY_ACP_KINDS:
                    with anyio.move_on_after(0.02):
                        await perm_ev.wait()
                if not perm_ev.is_set():
                    label = raw_name or cls.kiro_tool_id or cls.v2_tool_name or cls.title
                    turn.policy_denied_tools.append(label)
                    await self._send_agy_cancel(turn.agy_session_id)
                    if tc_id not in turn.completed_tools:
                        turn.completed_tools.add(tc_id)
                        async with turn.stream_lock:
                            if turn.on_tool_update is not None:
                                await turn.on_tool_update(
                                    replace(
                                        event,
                                        status="failed",
                                        output="Tool execution rejected by policy.",
                                    )
                                )
                    return

            if raw_status in (done_statuses | failed_statuses):
                if tc_id not in turn.completed_tools:
                    turn.completed_tools.add(tc_id)
                    terminal_status = "failed" if raw_status in failed_statuses else "completed"
                    async with turn.stream_lock:
                        if turn.on_tool_update is not None:
                            await turn.on_tool_update(replace(event, status=terminal_status))

    async def stream_prompt(
        self,
        *,
        cwd: str,
        session_id: str | None = None,
        agy_session_id: str | None = None,
        conversation_id: str | None = None,
        content_blocks: Sequence[Mapping[str, Any]] | None = None,
        prompt: str = "",
        mcp_servers: Sequence[Mapping[str, Any]] | None = None,
        system_prompt_append: str | None = None,
        model: str | None = None,
        mode: str = "auto",
        skip_permissions: bool | None = True,
        sandbox: bool = False,
        effort: str | None = "medium",
        route_spawn_run: bool = False,
        known_mcp_servers: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        on_thought: Callable[[str], Awaitable[None]] | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
        on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None = None,
        on_usage: Callable[[int], Awaitable[None]] | None = None,
        on_steer_consumed: Callable[[str], Awaitable[None]] | None = None,
        on_subagent_list: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None,
    ) -> PromptOutcome:
        """Stream a `session/prompt` turn over `agy acp` with pre-exec gating and steering."""
        _ = (sandbox, on_subagent_list)
        sid = session_id or "default"
        initial_agy_sid = (agy_session_id or conversation_id or "").strip()

        if sid in self._cancelled_sessions:
            self._cancelled_sessions.discard(sid)
            return PromptOutcome(
                conversation_id=initial_agy_sid,
                stop_reason="cancelled",
                response="",
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "cache_read_tokens": 0,
                    "total_tokens": 0,
                    "context_tokens": 0,
                },
            )

        effective_mcp = list(mcp_servers) if mcp_servers is not None else None
        if effective_mcp is None and session_id:
            persisted = self._store.get_session(session_id)
            if persisted is not None and isinstance(persisted.get("mcp_servers"), list):
                effective_mcp = list(persisted["mcp_servers"])

        effective_known_mcp: tuple[str, ...] = tuple(known_mcp_servers)
        if not effective_known_mcp and effective_mcp:
            effective_known_mcp = tuple(
                str(s.get("name") or "").strip()
                for s in effective_mcp
                if isinstance(s, Mapping) and str(s.get("name") or "").strip()
            )

        effective_route_spawn = route_spawn_run or bool(
            env and str(env.get("KIROCREW_ROUTE_SPAWN_RUN") or "") == "1"
        )
        effective_skip_perms = (
            skip_permissions if skip_permissions is not None else (mode == "auto")
        )

        collected_texts: list[str] = []
        usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 0,
            "context_tokens": 0,
        }
        steer_temp_files: list[Path] = []
        resolved_agy_sid = initial_agy_sid
        resolved_model = model or ""
        fallback_from_model = ""
        final_result_obj: dict[str, Any] = {}
        outcome_error = ""
        current_blocks = normalize_acp_content_blocks(content_blocks, prompt=prompt)
        recovered_once = False

        self._active_turns.add(sid)
        try:
            while True:
                try:
                    conn = await self._ensure_connection()
                    if resolved_agy_sid:
                        if resolved_agy_sid not in conn.loaded_sessions:
                            backend_info = await self.load_backend_session(
                                agy_session_id=resolved_agy_sid,
                                cwd=cwd,
                                mcp_servers=effective_mcp,
                                system_prompt_append=system_prompt_append,
                                model=model,
                                mode=mode,
                                effort=effort,
                                session_env=env,
                            )
                            resolved_agy_sid = str(backend_info["agySessionId"])
                            resolved_model = str(
                                backend_info.get("currentModelId") or resolved_model
                            )
                            fallback_from_model = str(
                                backend_info.get("fallbackFromModel") or fallback_from_model
                            )
                            conn = await self._ensure_connection()
                        else:
                            desired_prefs = (model or "", mode, effort or "")
                            if conn.session_prefs.get(resolved_agy_sid) != desired_prefs:
                                avail = list(self._cached_models) or self._store.load_models()
                                pref_model, pref_fb = await self._apply_session_preferences(
                                    conn,
                                    resolved_agy_sid,
                                    session_result=conn.session_results.get(resolved_agy_sid, {}),
                                    model=model,
                                    mode=mode,
                                    effort=effort,
                                    available_models=avail,
                                )
                                conn.session_prefs[resolved_agy_sid] = desired_prefs
                                if pref_model:
                                    resolved_model = pref_model
                                if pref_fb:
                                    fallback_from_model = pref_fb
                    else:
                        backend_info = await self.create_backend_session(
                            cwd=cwd,
                            mcp_servers=effective_mcp,
                            system_prompt_append=system_prompt_append,
                            model=model,
                            mode=mode,
                            effort=effort,
                            session_env=env,
                        )
                        resolved_agy_sid = str(backend_info["agySessionId"])
                        resolved_model = str(backend_info.get("currentModelId") or resolved_model)
                        fallback_from_model = str(
                            backend_info.get("fallbackFromModel") or fallback_from_model
                        )
                        conn = await self._ensure_connection()
                except AgyAcpError as exc:
                    outcome_error = str(exc)
                    break

                self._session_to_agy_sid[sid] = resolved_agy_sid
                turn_state = _ActiveTurnState(
                    session_id=sid,
                    agy_session_id=resolved_agy_sid,
                    route_spawn_run=effective_route_spawn,
                    skip_permissions=effective_skip_perms,
                    known_mcp_servers=effective_known_mcp,
                    on_thought=on_thought,
                    on_text=on_text,
                    on_tool_call=on_tool_call,
                    on_tool_update=on_tool_update,
                    on_usage=on_usage,
                    collected_texts=collected_texts,
                    usage=usage,
                )
                self._turn_states[resolved_agy_sid] = turn_state

                if sid in self._cancelled_sessions or self._steer_queues.get(sid):
                    turn_state.interrupt_event.set()

                prompt_res: dict[str, Any] = {}
                transport_failed: AgyAcpTransportError | None = None
                try:
                    if sid in self._cancelled_sessions:
                        prompt_res = {"stopReason": "cancelled"}
                    else:
                        prompt_res = await conn.request(
                            "session/prompt",
                            {"sessionId": resolved_agy_sid, "prompt": current_blocks},
                            timeout=None,
                        )
                except AgyAcpTransportError as exc:
                    transport_failed = exc
                except AgyAcpRpcError as exc:
                    outcome_error = exc.rpc_message
                finally:
                    with anyio.move_on_after(1.0):
                        await conn.wait_idle()
                        await turn_state.drained_event.wait()
                    self._turn_states.pop(resolved_agy_sid, None)

                if transport_failed is not None:
                    if not recovered_once and not turn_state.emitted_any_signal:
                        recovered_once = True
                        continue
                    outcome_error = str(transport_failed)
                    break

                final_result_obj = prompt_res
                self._accumulate_prompt_usage(usage, prompt_res, turn_state.last_context_tokens)

                was_user_cancelled = sid in self._cancelled_sessions or (
                    not self._steer_queues.get(sid)
                    and not turn_state.policy_denied_tools
                    and str(prompt_res.get("stopReason") or "").lower() in {"cancelled", "canceled"}
                )
                self._cancelled_sessions.discard(sid)

                if was_user_cancelled:
                    return PromptOutcome(
                        conversation_id=resolved_agy_sid,
                        stop_reason="cancelled",
                        response="".join(collected_texts),
                        usage=usage,
                        raw_result=final_result_obj,
                        resolved_model=resolved_model,
                        fallback_from_model=fallback_from_model,
                    )

                if self._steer_queues.get(sid):
                    entries = self._pop_queued_steers(sid)
                    for entry in entries:
                        steer_temp_files.extend(entry.steer.temp_files)
                    raw_summary = "\n".join(
                        e.steer.raw_message for e in entries if e.steer.raw_message
                    )
                    if on_steer_consumed is not None:
                        await on_steer_consumed(raw_summary)
                    current_blocks = [block for entry in entries for block in entry.content_blocks]
                    continue

                if turn_state.policy_denied_tools and not collected_texts:
                    unique_denied = list(dict.fromkeys(turn_state.policy_denied_tools))
                    notice = f"Tool `{', '.join(unique_denied)}` is not permitted in this session."
                    collected_texts.append(notice)
                    if on_text is not None:
                        await on_text(notice)

                break
        finally:
            self._active_turns.discard(sid)
            self._session_to_agy_sid.pop(sid, None)
            for leftover in self._pop_queued_steers(sid):
                steer_temp_files.extend(leftover.steer.temp_files)
            for tmp_path in steer_temp_files:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)

        full_response = "".join(collected_texts)
        if usage["input_tokens"] == 0 and (prompt or current_blocks):
            usage["input_tokens"] = 1
        if usage["output_tokens"] == 0 and full_response:
            usage["output_tokens"] = max(1, len(full_response.split()))
        if usage["total_tokens"] == 0:
            usage["total_tokens"] = (
                usage["input_tokens"]
                + usage["output_tokens"]
                + usage["thinking_tokens"]
                + usage["cache_read_tokens"]
            )
        if usage["context_tokens"] == 0:
            usage["context_tokens"] = usage["total_tokens"]

        return PromptOutcome(
            conversation_id=resolved_agy_sid,
            stop_reason="end_turn",
            response=full_response,
            usage=usage,
            raw_result=final_result_obj,
            resolved_model=resolved_model,
            fallback_from_model=fallback_from_model,
            error=outcome_error,
        )

    @staticmethod
    def _accumulate_prompt_usage(
        usage: dict[str, int],
        prompt_res: Mapping[str, Any],
        last_context_tokens: int,
    ) -> None:
        raw_usage = prompt_res.get("usage")
        if isinstance(raw_usage, Mapping):
            in_tok = int(raw_usage.get("inputTokens", raw_usage.get("input_tokens", 0)) or 0)
            out_tok = int(raw_usage.get("outputTokens", raw_usage.get("output_tokens", 0)) or 0)
            cache_tok = int(
                raw_usage.get(
                    "cachedReadTokens",
                    raw_usage.get("cache_read_tokens", raw_usage.get("cached_read_tokens", 0)),
                )
                or 0
            )
            think_tok = int(
                raw_usage.get(
                    "thoughtTokens",
                    raw_usage.get("thinking_tokens", raw_usage.get("thought_tokens", 0)),
                )
                or 0
            )
            tot_tok = int(
                raw_usage.get(
                    "totalTokens",
                    raw_usage.get("total_tokens", in_tok + out_tok + cache_tok + think_tok),
                )
                or 0
            )
            usage["input_tokens"] += in_tok
            usage["output_tokens"] += out_tok
            usage["cache_read_tokens"] += cache_tok
            usage["thinking_tokens"] += think_tok
            usage["total_tokens"] += tot_tok
        if last_context_tokens > 0:
            usage["context_tokens"] = last_context_tokens
        elif usage["total_tokens"] > 0:
            usage["context_tokens"] = usage["total_tokens"]
