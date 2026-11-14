"""ACP JSON-RPC 2.0 request/notification dispatcher and framing engine."""

from __future__ import annotations

import contextvars
import json
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from agy_acp import KIRO_CLI_COMPAT_VERSION, __version__
from agy_acp.logic.auth_manager import AuthManager
from agy_acp.logic.carousel_normalizer import (
    CarouselStreamNormalizer,
    normalize_carousel_markdown,
)
from agy_acp.logic.custom_agents import (
    CustomAgentSpec,
    PermissionDecision,
    evaluate_tool_call_permission,
)
from agy_acp.logic.quota_tracker import QuotaTracker, TurnTelemetry
from agy_acp.logic.session_manager import DEFAULT_MODEL_ID, SessionManager
from agy_acp.logic.tool_classifier import (
    PATH_ARGUMENT_KEYS,
    ToolCallEvent,
    _extract_first_str,
    match_mcp_server_prefix,
)
from agy_acp.logic.wire import (
    EngineWireAdapter,
    McpServerStatus,
    NormalizedToolEvent,
    ToolCallKind,
    V2KiroWireAdapter,
    WireTurnTelemetry,
)
from agy_acp.providers.acp_logger import log_acp_event
from agy_acp.providers.quota_provider import QuotaProvider
from agy_acp.providers.telemetry import get_telemetry

MAX_FRAME_SIZE_BYTES = 16 * 1024 * 1024
SLASH_QUOTA_COMMANDS = frozenset({"/quota", "/usage", "/credits"})
SLASH_RESET_COMMANDS = frozenset({"/clear", "/reset"})
AVAILABLE_SLASH_COMMANDS: list[dict[str, str]] = [
    {
        "name": "quota",
        "description": "Display 5h and 7d rolling quota remaining percentages and reset times",
    },
    {
        "name": "usage",
        "description": "Display session token usage and quota summary",
    },
    {
        "name": "credits",
        "description": "Display remaining Antigravity G1 credits",
    },
    {
        "name": "compact",
        "description": "Compact conversation history to free context window space",
    },
]

_FS_TOOL_IDS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_directory",
        "grep_search",
        "file_search",
        "fs_write",
        "str_replace",
        "fs_append",
        "delete_file",
        "fs_read",
        "read",
        "grep",
        "glob",
        "write",
        "edit",
    }
)
_TOOL_CAPABILITY_ALIASES: dict[str, tuple[str, ...]] = {
    "read_file": ("fs_read", "read"),
    "list_directory": ("fs_read", "read"),
    "grep_search": ("fs_read", "grep"),
    "file_search": ("fs_read", "glob"),
    "fs_write": ("fs_write", "write"),
    "str_replace": ("fs_write", "write", "edit"),
    "fs_append": ("fs_write", "write"),
    "run_command": ("execute_bash", "shell", "bash"),
    "remote_web_search": ("web_search",),
    "invoke_sub_agent": ("use_subagent", "subagent"),
}

NotificationSink = Callable[[dict[str, Any]], Awaitable[None]]
ClientRpcCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _coerce_tool_kind(raw_kind: str) -> ToolCallKind:
    match raw_kind:
        case "execute" | "read" | "edit" | "delete" | "search" | "fetch" | "other":
            return raw_kind
        case _:
            return "other"


def _resolve_event_mcp_identity(
    event: ToolCallEvent,
    known_mcp_servers: Sequence[str],
) -> tuple[str, str]:
    """Resolve `(mcp_server, mcp_tool)` for `event` using longest configured server prefix."""
    if event.mcp_server and event.mcp_tool:
        matched = match_mcp_server_prefix(
            f"{event.mcp_server}_{event.mcp_tool}",
            known_mcp_servers,
        )
        return matched if matched != ("", "") else (event.mcp_server, event.mcp_tool)

    for raw_candidate in (event.kiro_tool_id, event.kiro_tool_name, event.title):
        if not isinstance(raw_candidate, str):
            continue
        if raw_candidate.startswith("mcp_") and len(raw_candidate) > 4:
            after_prefix = raw_candidate[4:]
            matched = match_mcp_server_prefix(after_prefix, known_mcp_servers)
            if matched != ("", ""):
                return matched
            srv, sep, tool = after_prefix.partition("_")
            return srv, (tool if sep else after_prefix)
        if raw_candidate.startswith("@") and "/" in raw_candidate:
            srv, _, tool = raw_candidate[1:].partition("/")
            if srv and tool:
                return srv, tool
    return event.mcp_server, event.mcp_tool


def _to_normalized_tool_event(
    event: ToolCallEvent,
    *,
    known_mcp_servers: Sequence[str] = (),
) -> NormalizedToolEvent:
    """Convert an `AgyAcpClientProvider` `ToolCallEvent` into a wire `NormalizedToolEvent`."""
    mcp_server, mcp_tool = _resolve_event_mcp_identity(event, known_mcp_servers)
    is_mcp = bool(mcp_server and mcp_tool)
    kas_id = event.kas_builtin_id or (
        "run_command"
        if event.kind == "execute"
        else (f"mcp_{mcp_server}_{mcp_tool}" if is_mcp else (event.kiro_tool_id or "other"))
    )
    fallback_v2 = (
        "execute_bash" if event.kind == "execute" else (event.kiro_tool_name or event.title)
    )
    v2_name = mcp_tool if is_mcp else (event.v2_tool_name or fallback_v2)
    return NormalizedToolEvent(
        tool_call_id=event.tool_call_id,
        title=event.title,
        kind=_coerce_tool_kind(event.acp_kind or event.kind),
        status=event.status or "in_progress",
        kas_tool_id=kas_id,
        kiro_v2_tool_name=v2_name,
        raw_input=dict(event.raw_input),
        locations=tuple(dict(loc) for loc in event.locations),
        content=tuple(dict(block) for block in event.diff_content),
        command=event.command,
        subagent_target=event.subagent_target,
        is_mcp=is_mcp,
        mcp_server=mcp_server,
        mcp_tool=mcp_tool,
    )


def _resolve_v2_tool_meta(
    event: ToolCallEvent,
    norm: NormalizedToolEvent,
) -> tuple[str, str, str]:
    """Resolve `(kiro_tool_name, mcp_server_name, kiro_tool_id)` for v2 wire payloads."""
    if norm.is_mcp:
        return norm.mcp_tool, norm.mcp_server, f"@{norm.mcp_server}/{norm.mcp_tool}"
    kiro_tool_name = event.kiro_tool_name or (
        "execute_bash" if event.kind == "execute" else event.title
    )
    kiro_tool_id = event.kiro_tool_id or (
        "run_command" if event.kind == "execute" else kiro_tool_name
    )
    return kiro_tool_name, event.mcp_server_name, kiro_tool_id


def _is_mcp_tool_disabled(session: Mapping[str, Any], norm: NormalizedToolEvent) -> bool:
    """Return True if `norm` targets a disabled MCP server or a tool in `disabledTools`."""
    if not norm.is_mcp:
        return False
    raw_servers = session.get("mcp_servers")
    if not isinstance(raw_servers, Sequence) or isinstance(raw_servers, (str, bytes)):
        return False
    for srv in raw_servers:
        if not isinstance(srv, Mapping):
            continue
        if str(srv.get("name") or "").strip() != norm.mcp_server:
            continue
        if srv.get("disabled") is True:
            return True
        raw_dt = (
            srv.get("disabledTools")
            if srv.get("disabledTools") is not None
            else srv.get("disabled_tools")
        )
        if isinstance(raw_dt, Sequence) and not isinstance(raw_dt, (str, bytes)):
            if any(str(t).strip() == norm.mcp_tool for t in raw_dt):
                return True
    return False


def _matches_agent_tool_list(
    patterns: Sequence[str],
    norm: NormalizedToolEvent,
    *,
    for_allowlist: bool = False,
) -> bool:
    """Return True if `norm` matches an entry in `tools`, `excludedTools`, or `allowedTools`."""
    if not patterns:
        return False
    candidates: set[str] = {"*"}
    if norm.is_mcp:
        candidates.update(
            {
                f"@{norm.mcp_server}",
                f"@{norm.mcp_server}/*",
                f"@{norm.mcp_server}/{norm.mcp_tool}",
                f"mcp_{norm.mcp_server}_{norm.mcp_tool}",
            }
        )
    elif for_allowlist and (norm.kas_tool_id == "delete_file" or norm.kind == "delete"):
        # POLICY_ALIAS_DENY_ONLY_IDS: `fs_write` / `write` MUST NEVER auto-approve `delete_file`.
        candidates.update({"delete_file", "builtin"})
    else:
        candidates.update({norm.kas_tool_id, norm.kiro_v2_tool_name, "builtin"})
        candidates.update(_TOOL_CAPABILITY_ALIASES.get(norm.kas_tool_id, ()))
    return any(p in candidates for p in patterns)


def _extract_primary_path(norm: NormalizedToolEvent) -> str:
    if norm.locations:
        loc_path = norm.locations[0].get("path")
        if isinstance(loc_path, str) and loc_path:
            return loc_path
    return _extract_first_str(norm.raw_input, PATH_ARGUMENT_KEYS)


def _adapt_permissions_for_cwd(
    permissions: Any,
    *,
    cwd_posix: str,
    for_absolute: bool,
) -> Any:
    """Adapt relative/absolute glob patterns in `permissions` for `abs_posix` or `rel_posix`."""
    if not cwd_posix or cwd_posix == "/":
        return permissions
    prefix = f"{cwd_posix.rstrip('/')}/"

    def _adapt_pattern_list(raw_list: Any, *, is_exclude: bool) -> Any:
        if not isinstance(raw_list, Sequence) or isinstance(raw_list, (str, bytes)):
            return raw_list
        out: list[Any] = []
        for item in raw_list:
            if not isinstance(item, str) or not item.strip():
                out.append(item)
                continue
            pat = item.strip()
            out.append(pat)
            if for_absolute and not pat.startswith("/"):
                if is_exclude or pat != "**":
                    out.append(f"{prefix}{pat}")
            elif not for_absolute and pat.startswith(prefix):
                rel_pat = pat[len(prefix) :]
                if rel_pat:
                    out.append(rel_pat)
        return out

    def _adapt_rule(entry: Any) -> Any:
        if not isinstance(entry, Mapping):
            return entry
        updated = dict(entry)
        if "match" in updated:
            updated["match"] = _adapt_pattern_list(updated["match"], is_exclude=False)
        if "exclude" in updated:
            updated["exclude"] = _adapt_pattern_list(updated["exclude"], is_exclude=True)
        return updated

    if isinstance(permissions, Mapping):
        raw_rules = permissions.get("rules")
        if isinstance(raw_rules, Sequence) and not isinstance(raw_rules, (str, bytes)):
            return {**dict(permissions), "rules": [_adapt_rule(r) for r in raw_rules]}
        return permissions
    if isinstance(permissions, Sequence) and not isinstance(permissions, (str, bytes)):
        return [_adapt_rule(r) for r in permissions]
    return permissions


def _evaluate_permissions_with_cwd(
    permissions: Any,
    norm: NormalizedToolEvent,
    *,
    cwd: str,
) -> PermissionDecision:
    """Evaluate `permissions` normalizing target paths against `cwd` (`deny > ask > allow`)."""
    url = str(norm.raw_input.get("url") or norm.raw_input.get("Url") or "")
    query = str(norm.raw_input.get("query") or norm.raw_input.get("Query") or "")
    raw_path = _extract_primary_path(norm)

    if not norm.is_mcp and norm.kas_tool_id in _FS_TOOL_IDS and raw_path:
        raw_p = Path(raw_path).expanduser()
        cwd_clean = cwd.strip()
        cwd_resolved = Path(cwd_clean).expanduser().resolve(strict=False) if cwd_clean else None
        target = (
            raw_p
            if raw_p.is_absolute()
            else ((cwd_resolved / raw_p) if cwd_resolved is not None else raw_p)
        )
        resolved = target.resolve(strict=False)
        cwd_posix = cwd_resolved.as_posix() if cwd_resolved is not None else ""
        abs_dec = evaluate_tool_call_permission(
            _adapt_permissions_for_cwd(permissions, cwd_posix=cwd_posix, for_absolute=True),
            tool_id=norm.kas_tool_id,
            path=resolved.as_posix(),
            url=url,
            query=query,
            subagent_target=norm.subagent_target,
        )
        if cwd_resolved is not None and resolved.is_relative_to(cwd_resolved):
            rel_posix = (
                "." if resolved == cwd_resolved else resolved.relative_to(cwd_resolved).as_posix()
            )
            rel_dec = evaluate_tool_call_permission(
                _adapt_permissions_for_cwd(permissions, cwd_posix=cwd_posix, for_absolute=False),
                tool_id=norm.kas_tool_id,
                path=rel_posix,
                url=url,
                query=query,
                subagent_target=norm.subagent_target,
            )
            if "deny" in (abs_dec, rel_dec):
                return "deny"
            if "ask" in (abs_dec, rel_dec):
                return "ask"
            return "allow"
        return abs_dec

    return evaluate_tool_call_permission(
        permissions,
        tool_id=norm.kas_tool_id,
        is_mcp=norm.is_mcp,
        server_name=norm.mcp_server,
        tool_name=norm.mcp_tool,
        command=norm.command,
        path=raw_path,
        url=url,
        query=query,
        subagent_target=norm.subagent_target,
    )


def _stamp_replay_meta(frame: dict[str, Any]) -> dict[str, Any]:
    """Ensure `params._meta.kiro.replay` is True on a replay notification frame."""
    params = dict(frame.get("params") or {})
    params_meta = dict(params.get("_meta") or {})
    params["_meta"] = {
        **params_meta,
        "kiro": {**dict(params_meta.get("kiro") or {}), "replay": True},
    }
    return {**frame, "params": params}


def _stamp_ttft_meta(
    frame: dict[str, Any],
    *,
    prompt_received_ms: int,
    emitted_ms: int,
) -> dict[str, Any]:
    """Attach `_meta.kiro.ttft` onto a v3 thought or message chunk frame."""
    params = dict(frame.get("params") or {})
    update = dict(params.get("update") or {})
    update_meta = dict(update.get("_meta") or {})
    ttft = {"promptReceivedAt": prompt_received_ms, "firstChunkEmittedAt": emitted_ms}
    update["_meta"] = {**update_meta, "kiro": {**dict(update_meta.get("kiro") or {}), "ttft": ttft}}
    return {**frame, "params": {**params, "update": update}}


def _to_wire_turn_telemetry(
    telemetry: TurnTelemetry,
    *,
    turn_id: str = "",
    context_breakdown: dict[str, Any] | None = None,
) -> WireTurnTelemetry:
    return WireTurnTelemetry(
        context_used=telemetry.context_tokens,
        context_size=telemetry.context_window,
        context_usage_percentage=telemetry.context_usage_pct,
        delta_credits=telemetry.delta_credits,
        metadata_payload={k: v for k, v in telemetry.metadata_params.items() if k != "sessionId"},
        rate_limited_message=telemetry.rate_limit_message,
        context_breakdown=context_breakdown,
        turn_id=turn_id,
    )


def _require_non_empty_str(params: dict[str, Any], key: str) -> str | None:
    val = params.get(key)
    return val if isinstance(val, str) and bool(val.strip()) else None


def _jsonrpc_result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _jsonrpc_error(
    msg_id: Any,
    code: int,
    message: str,
    *,
    data: Any = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


_RE_MONTHLY_LIMIT_ERROR = re.compile(
    r"monthly\s+usage\s+limit|monthly.*limit|usage\s+limit\s+has\s+been\s+reached"
    r"|quota\s+(?:limit|exhausted|exceeded)|freetierlimitexceeded|monthlylimiterror",
    re.IGNORECASE,
)
_RE_THROTTLE_ERROR = re.compile(
    r"\b429\b|too\s+many\s+requests|rate[\s_-]?limit|throttl",
    re.IGNORECASE,
)
_RE_SERVICE_UNAVAILABLE_ERROR = re.compile(
    r"econnreset|econnrefused|etimedout|epipe"
    r"|connection\s+(?:reset|refused|closed|error|aborted)"
    r"|temporarily\s+unavailable|service\s+unavailable"
    r"|\b50[0234]\b|bad\s+gateway|gateway\s+timeout|overloaded",
    re.IGNORECASE,
)


def _classify_turn_error_type(turn_error: str) -> str:
    """Classify a turn-level error message into KiroCrew 0.12.2 `data.errorType`."""
    if _RE_MONTHLY_LIMIT_ERROR.search(turn_error):
        return "MonthlyLimitError"
    if _RE_THROTTLE_ERROR.search(turn_error):
        return "ThrottlingException"
    if _RE_SERVICE_UNAVAILABLE_ERROR.search(turn_error):
        return "ServiceUnavailableError"
    return "ProviderError"


def _prompt_turn_result(
    msg_id: Any,
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_read_tokens: int = 0,
) -> dict[str, Any]:
    return _jsonrpc_result(
        msg_id,
        {
            "stopReason": stop_reason,
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "cachedReadTokens": cached_read_tokens,
                "cachedWriteTokens": 0,
            },
        },
    )


def _session_update_frame(session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": session_id, "update": update},
    }


def _parse_session_wire_options(
    msg_id: Any,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Validate and extract `(mcp_servers, meta, error_response)` from session open params."""
    raw_mcp = params.get("mcpServers")
    if raw_mcp is not None and not isinstance(raw_mcp, list):
        return (
            None,
            None,
            _jsonrpc_error(msg_id, -32602, "Invalid params: 'mcpServers' must be a list"),
        )
    mcp_servers = [s for s in raw_mcp if isinstance(s, dict)] if isinstance(raw_mcp, list) else None
    raw_meta = params.get("_meta")
    if raw_meta is not None and not isinstance(raw_meta, dict):
        return (
            None,
            None,
            _jsonrpc_error(msg_id, -32602, "Invalid params: '_meta' must be an object"),
        )
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else None
    return mcp_servers, meta, None


def _extract_fork_message_id(
    meta: dict[str, Any] | None,
    raw_top_msg_id: str | None,
) -> str | None:
    """Resolve optional fork `messageId` across JetBrains Air, Kiro `_meta`, or top-level params."""
    if isinstance(meta, dict):
        for path in (("jetbrains", "air", "fork"), ("kiro", "fork")):
            node: Any = meta
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, dict) and isinstance(node.get("messageId"), str):
                cleaned = node["messageId"].strip()
                if cleaned:
                    return cleaned
    return raw_top_msg_id.strip() or None if isinstance(raw_top_msg_id, str) else None


class AcpDispatcher:
    """Routes ACP JSON-RPC 2.0 messages between KiroCrew stdio and SessionManager."""

    def __init__(
        self,
        *,
        notification_sink: NotificationSink | None = None,
        client_rpc_caller: ClientRpcCaller | None = None,
        session_manager: SessionManager | None = None,
        auth_manager: AuthManager | None = None,
        quota_tracker: QuotaTracker | None = None,
        wire_adapter: EngineWireAdapter | None = None,
    ) -> None:
        self._notification_sink = notification_sink
        self._client_rpc_caller = client_rpc_caller
        resolved_adapter = wire_adapter or (
            session_manager.wire_adapter if session_manager is not None else V2KiroWireAdapter()
        )
        self._wire_adapter: EngineWireAdapter = resolved_adapter
        self._wire: EngineWireAdapter = resolved_adapter
        self._session_manager = session_manager or SessionManager(wire_adapter=resolved_adapter)
        self._auth_manager = auth_manager
        self._quota_tracker = quota_tracker or QuotaTracker(
            provider=QuotaProvider(cli_provider=self._session_manager.cli_provider),
            state_store=self._session_manager.state_store,
        )
        self._post_response_ctx: contextvars.ContextVar[list[dict[str, Any]] | None] = (
            contextvars.ContextVar(f"agy_acp_post_notifs_{id(self)}", default=None)
        )
        self._mcp_status_cache: dict[str, dict[str, McpServerStatus]] = {}
        self._client_opens_external_url: bool = False

    @property
    def wire_adapter(self) -> EngineWireAdapter:
        """Return the active `EngineWireAdapter` strategy."""
        return self._wire_adapter

    def _queue_post_response_notifications(self, frames: Sequence[dict[str, Any]]) -> None:
        bucket = self._post_response_ctx.get()
        if bucket is None:
            bucket = []
            self._post_response_ctx.set(bucket)
        bucket.extend(frames)

    def pop_post_response_notifications(self) -> list[dict[str, Any]]:
        """Drain notifications queued by the current task to flush after its response frame."""
        bucket = self._post_response_ctx.get()
        if not bucket:
            return []
        queued = list(bucket)
        bucket.clear()
        return queued

    async def _emit_notification(self, notification: dict[str, Any]) -> None:
        """Emit a JSON-RPC 2.0 notification to the configured sink if present."""
        log_acp_event(
            "rpc.notification",
            save_dir_or_db=self._session_manager.state_store.db_path,
            method=notification.get("method"),
            params=notification.get("params"),
        )
        if self._notification_sink is not None:
            await self._notification_sink(notification)

    def _require_existing_session(
        self,
        msg_id: Any,
        params: dict[str, Any],
        *,
        is_request: bool = True,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Validate `sessionId` and verify session existence in SQLite."""
        session_id = _require_non_empty_str(params, "sessionId")
        if session_id is None:
            return None, (_jsonrpc_error(msg_id, -32602, "Invalid params") if is_request else None)
        try:
            self._session_manager.get_session(session_id)
        except KeyError as exc:
            return None, (_jsonrpc_error(msg_id, -32002, str(exc)) if is_request else None)
        return session_id, None

    def _resolve_session_model(self, session_id: str) -> str:
        """Return the active model ID for `session_id`, falling back to the default model."""
        try:
            return str(self._session_manager.get_session(session_id).get("model", DEFAULT_MODEL_ID))
        except KeyError:
            return DEFAULT_MODEL_ID

    def _resolve_session_mode(self, session_id: str) -> str:
        """Return the active mode ID for `session_id`, falling back to GOOSE_MODE or `auto`."""
        try:
            return str(self._session_manager.get_session(session_id).get("mode", "auto"))
        except KeyError:
            return "approve" if os.environ.get("GOOSE_MODE") == "approve" else "auto"

    @staticmethod
    def _extract_prompt_text(prompt_blocks: list[dict[str, Any]]) -> str:
        """Extract concatenated text from ACP prompt content blocks."""
        return "\n".join(
            str(block["text"])
            for block in prompt_blocks
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
        ).strip()

    @staticmethod
    def _extract_user_tail(prompt_text: str) -> str:
        if not prompt_text:
            return ""
        idx = prompt_text.rfind("[CURRENT USER REQUEST")
        if idx != -1:
            close_bracket = prompt_text.find("]", idx)
            if close_bracket != -1:
                return prompt_text[close_bracket + 1 :].strip()
        return prompt_text.strip()

    @classmethod
    def _extract_compact_command(cls, prompt_text: str) -> tuple[bool, str]:
        """Return `(is_compact, instructions)` if `prompt_text` invokes `/compact`."""
        candidate = cls._extract_user_tail(prompt_text)
        lower = candidate.lower()
        if lower == "/compact":
            return True, ""
        if lower.startswith(("/compact ", "/compact\n", "/compact\t")):
            return True, candidate[len("/compact") :].strip()
        if lower.endswith("/compact"):
            prefix = candidate[: -len("/compact")]
            if prefix and (prefix[-1].isspace() or prefix[-1] in {".", "?", "!", ":", ";"}):
                return True, prefix.strip()
        return False, ""

    @classmethod
    def _extract_slash_command(cls, prompt_text: str) -> str:
        """Extract a slash command (`/quota`, `/usage`, `/credits`, `/clear`, `/reset`)."""
        if not prompt_text:
            return ""
        all_slash = SLASH_QUOTA_COMMANDS | SLASH_RESET_COMMANDS
        tail = cls._extract_user_tail(prompt_text)
        first_line = tail.splitlines()[0].strip().lower() if tail else ""
        if first_line in all_slash:
            return first_line
        last_line = prompt_text.splitlines()[-1].strip().lower()
        return last_line if last_line in all_slash else ""

    @staticmethod
    def _cancelled_prompt_response(message_id: Any) -> dict[str, Any]:
        """Build a JSON-RPC 2.0 `session/prompt` response with `stopReason == 'cancelled'`."""
        return _prompt_turn_result(message_id, stop_reason="cancelled")

    async def _finish_cancelled_prompt(
        self,
        message_id: Any,
        *,
        session_id: str,
        usage: Mapping[str, Any] | None = None,
        turn_id: str = "",
    ) -> dict[str, Any]:
        """Emit cancelled-turn telemetry (and post-response v2 drain metadata) and build result."""
        usage_map = usage if isinstance(usage, Mapping) else {}
        input_tokens = max(0, int(usage_map.get("input_tokens", 0)))
        output_tokens = max(0, int(usage_map.get("output_tokens", 0)))
        cached_read_tokens = max(0, int(usage_map.get("cache_read_tokens", 0)))
        thinking_tokens = max(0, int(usage_map.get("thinking_tokens", 0)))
        raw_ctx = usage_map.get("context_tokens")
        effective_ctx = (
            int(raw_ctx)
            if raw_ctx is not None and int(raw_ctx) > 0
            else self._quota_tracker.get_session_context_tokens(session_id)
        )
        telemetry = await self._emit_turn_telemetry_frames(
            session_id=session_id,
            model_id=self._resolve_session_model(session_id),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
            context_tokens=effective_ctx,
            turn_id=turn_id,
        )
        if self._wire_adapter.engine_id == "v2":
            post_params = {
                **telemetry.metadata_params,
                "meteringUsage": [{"unit": "credit", "unitPlural": "credits", "value": 0.0}],
            }
            self._queue_post_response_notifications(
                [{"jsonrpc": "2.0", "method": "_kiro.dev/metadata", "params": post_params}]
            )
        return _prompt_turn_result(
            message_id,
            stop_reason="cancelled",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
        )

    def _evaluate_tool_permission(
        self,
        *,
        session: Mapping[str, Any],
        spec: CustomAgentSpec | None,
        norm: NormalizedToolEvent,
        session_mode: str,
    ) -> PermissionDecision:
        """Evaluate tool permissions in strict fail-closed order across both v2 and v3."""
        if (session_mode == "read-only" or bool(session.get("agentReadOnly"))) and norm.kind in {
            "execute",
            "edit",
            "delete",
        }:
            return "deny"

        if _is_mcp_tool_disabled(session, norm):
            return "deny"

        excluded_tools: tuple[str, ...] = (
            spec.excluded_tools
            if spec is not None
            else tuple(
                str(x)
                for x in (session.get("agentExcludedTools") or [])
                if isinstance(x, str) and x
            )
        )
        if _matches_agent_tool_list(excluded_tools, norm, for_allowlist=False):
            return "deny"

        raw_session_tools = session.get("agentTools")
        agent_tools: tuple[str, ...] | None = (
            spec.tools
            if spec is not None
            else (
                tuple(str(x) for x in raw_session_tools if isinstance(x, str))
                if isinstance(raw_session_tools, list)
                else None
            )
        )
        if (
            agent_tools is not None
            and "*" not in agent_tools
            and not _matches_agent_tool_list(agent_tools, norm, for_allowlist=False)
        ):
            return "deny"

        perms = (
            spec.permissions
            if (spec is not None and spec.permissions is not None)
            else session.get("agentPermissions")
        )
        perm_decision: PermissionDecision | None = None
        if perms is not None:
            perm_decision = _evaluate_permissions_with_cwd(
                perms,
                norm,
                cwd=str(session.get("cwd") or ""),
            )
            if perm_decision == "deny":
                return "deny"

        allowed_tools: tuple[str, ...] = (
            spec.allowed_tools
            if spec is not None
            else tuple(
                str(x) for x in (session.get("agentAllowedTools") or []) if isinstance(x, str) and x
            )
        )
        if _matches_agent_tool_list(allowed_tools, norm, for_allowlist=True):
            return "allow"
        return "allow" if perm_decision == "allow" else "ask"

    async def _request_client_permission(
        self,
        session_id: str,
        perm_params: dict[str, Any],
    ) -> bool:
        """Prompt the ACP client for tool approval, failing closed when headless or rejected."""
        if self._client_rpc_caller is None:
            return False

        perm_response = await self._client_rpc_caller("session/request_permission", perm_params)
        outcome = perm_response.get("outcome") if isinstance(perm_response, dict) else None
        if isinstance(outcome, dict) and outcome.get("outcome") == "cancelled":
            await self._session_manager.cancel(session_id=session_id)
            return False
        selected_option = outcome.get("optionId") if isinstance(outcome, dict) else None
        return selected_option in {"allow_once", "allow_always"}

    async def _authorize_tool_call(
        self,
        session_id: str,
        session: Mapping[str, Any],
        norm: NormalizedToolEvent,
        perm_params_factory: Callable[[], dict[str, Any]],
    ) -> bool:
        """Evaluate customAgent/session permissions and prompt client only on `ask`."""
        telemetry = get_telemetry(self._session_manager.state_store.db_path)
        decision = self._evaluate_tool_permission(
            session=session,
            spec=self._session_manager.get_active_custom_agent(session_id),
            norm=norm,
            session_mode=self._resolve_session_mode(session_id),
        )
        if decision == "allow":
            telemetry.record_tool_permission(norm.kind, outcome="auto_allow")
            return True
        if decision == "deny":
            telemetry.record_tool_permission(norm.kind, outcome="policy_deny")
            return False
        client_allowed = await self._request_client_permission(session_id, perm_params_factory())
        telemetry.record_tool_permission(
            norm.kind,
            outcome="user_allow" if client_allowed else "user_deny",
        )
        return client_allowed

    async def _emit_turn_telemetry_frames(
        self,
        *,
        session_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int = 0,
        thinking_tokens: int = 0,
        context_tokens: int | None = None,
        turn_id: str = "",
        context_breakdown: dict[str, Any] | None = None,
    ) -> TurnTelemetry:
        telemetry = await self._quota_tracker.build_turn_telemetry(
            session_id=session_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
            context_tokens=context_tokens,
        )
        if self._wire_adapter.engine_id == "v3":
            wire_tel = _to_wire_turn_telemetry(
                telemetry,
                turn_id=turn_id,
                context_breakdown=context_breakdown,
            )
            for frame in self._wire_adapter.build_turn_telemetry_notifications(
                session_id,
                wire_tel,
            ):
                await self._emit_notification(frame)
            return telemetry

        await self._emit_notification(telemetry.usage_update_notification)
        await self._emit_notification(
            {
                "jsonrpc": "2.0",
                "method": "_kiro.dev/metadata",
                "params": telemetry.metadata_params,
            }
        )
        if telemetry.rate_limit_params is not None:
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/error/rate_limit",
                    "params": telemetry.rate_limit_params,
                }
            )
        return telemetry

    async def _handle_v3_tool_call(
        self,
        session_id: str,
        event: ToolCallEvent,
        *,
        emit_start: bool = True,
    ) -> bool:
        session = self._session_manager.state_store.get_session(session_id) or {}
        known_mcp = self._session_manager.get_session_mcp_server_names(session_id)
        norm = _to_normalized_tool_event(event, known_mcp_servers=known_mcp)
        if emit_start:
            await self._emit_notification(
                self._wire_adapter.build_tool_call_update(session_id, norm)
            )
        return await self._authorize_tool_call(
            session_id,
            session,
            norm,
            lambda: self._wire_adapter.build_permission_request_params(session_id, norm),
        )

    async def _handle_v2_tool_call(
        self,
        session_id: str,
        event: ToolCallEvent,
        *,
        emit_start: bool = True,
    ) -> bool:
        session = self._session_manager.state_store.get_session(session_id) or {}
        known_mcp = self._session_manager.get_session_mcp_server_names(session_id)
        norm = _to_normalized_tool_event(event, known_mcp_servers=known_mcp)
        kiro_tool_name, mcp_server_name, kiro_tool_id = _resolve_v2_tool_meta(event, norm)

        tool_descriptor: dict[str, Any] = {
            "toolCallId": event.tool_call_id,
            "title": event.title,
            "kind": event.kind,
            "status": "in_progress",
            "rawInput": event.raw_input,
            "_meta": {"kiro": {"toolName": kiro_tool_name, "mcpServerName": mcp_server_name}},
            **({"locations": event.locations} if event.locations else {}),
            **({"content": event.diff_content} if event.diff_content else {}),
        }
        if emit_start:
            await self._emit_notification(
                _session_update_frame(session_id, {"sessionUpdate": "tool_call", **tool_descriptor})
            )
        return await self._authorize_tool_call(
            session_id,
            session,
            norm,
            lambda: {
                "sessionId": session_id,
                "toolCall": tool_descriptor,
                "_meta": {"kiro": {"toolId": kiro_tool_id}},
                "options": self._wire_adapter.build_permission_options(),
            },
        )

    async def _handle_inline_compact_slash(
        self,
        message_id: Any,
        session_id: str,
        instructions: str,
    ) -> dict[str, Any]:
        if self._wire_adapter.engine_id == "v3":
            await self._emit_notification(
                self._wire_adapter.build_compaction_notification(session_id, "started")
            )
        compact_res = await self._session_manager.compact_session(
            session_id=session_id,
            instructions=instructions,
        )
        status_type = str(compact_res.get("status", "completed"))
        summary_text = str(compact_res.get("summary", ""))
        compacted_ctx = int(compact_res.get("contextTokens", 0))
        input_tokens = max(1, int(compact_res.get("inputTokens", 1)))
        output_tokens = max(1, int(compact_res.get("outputTokens", 1)))

        phase = "failed" if status_type == "failed" else "completed"
        get_telemetry(self._session_manager.state_store.db_path).record_compaction(
            trigger="slash",
            status=phase,
        )
        await self._emit_notification(
            self._wire_adapter.build_compaction_notification(
                session_id,
                phase,
                summary=summary_text if phase == "completed" else "",
                error=summary_text if phase == "failed" else "",
            )
        )
        if summary_text:
            await self._emit_notification(
                self._wire_adapter.build_agent_message_chunk(session_id, summary_text)
            )
        telemetry = await self._emit_turn_telemetry_frames(
            session_id=session_id,
            model_id=self._resolve_session_model(session_id),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=0,
            context_tokens=compacted_ctx,
        )
        if self._wire_adapter.engine_id == "v2":
            post_params = {
                **telemetry.metadata_params,
                "meteringUsage": [{"unit": "credit", "unitPlural": "credits", "value": 0.0}],
            }
            self._queue_post_response_notifications(
                [{"jsonrpc": "2.0", "method": "_kiro.dev/metadata", "params": post_params}]
            )
        return _prompt_turn_result(
            message_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _handle_session_prompt(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/prompt`, streaming thought/message/tool chunks from `agy` CLI."""
        session_id = _require_non_empty_str(params, "sessionId")
        raw_prompt = params.get("prompt")
        if session_id is None or not isinstance(raw_prompt, list):
            return _jsonrpc_error(message_id, -32602, "Invalid params")

        if self._auth_manager is not None and not self._auth_manager.status().authenticated:
            return _jsonrpc_error(
                message_id,
                -32001,
                "Not logged in. Run `agy-acp auth login` to authenticate.",
            )

        try:
            pre_session = self._session_manager.get_session(session_id)
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))

        if self._session_manager.consume_cancellation(session_id):
            await self._quota_tracker.seed_session_baseline(
                session_id,
                model_id=self._resolve_session_model(session_id),
            )
            return await self._finish_cancelled_prompt(message_id, session_id=session_id)

        prompt_blocks: list[dict[str, Any]] = [b for b in raw_prompt if isinstance(b, dict)]
        prompt_text = self._extract_prompt_text(prompt_blocks)
        is_compact, compact_instructions = self._extract_compact_command(prompt_text)
        if is_compact:
            return await self._handle_inline_compact_slash(
                message_id,
                session_id,
                compact_instructions,
            )

        slash_cmd = self._extract_slash_command(prompt_text)
        if slash_cmd in SLASH_RESET_COMMANDS:
            self._session_manager.state_store.clear_session_turns(session_id)
            existing = self._session_manager.state_store.get_session(session_id)
            if existing is not None:
                old_agy_id = str(existing.get("agyConversationId") or "").strip()
                if old_agy_id:
                    await self._session_manager.cli_provider.close_backend_session(old_agy_id)
                existing["agyConversationId"] = None
                existing["compactedSummary"] = None
                existing["contextTokens"] = 0
                existing["reinjectAgentPrompt"] = True
                self._session_manager.state_store.upsert_session(existing)
            await self._emit_notification(
                self._wire_adapter.build_agent_message_chunk(session_id, "Conversation cleared.")
            )
            return _prompt_turn_result(message_id, input_tokens=1, output_tokens=1)

        if slash_cmd in SLASH_QUOTA_COMMANDS:
            response_text = await self._quota_tracker.format_slash_command_response(slash_cmd)
            await self._emit_notification(
                self._wire_adapter.build_agent_thought_chunk(
                    session_id,
                    f"Querying quota report for {slash_cmd}...",
                )
            )
            await self._emit_notification(
                self._wire_adapter.build_agent_message_chunk(session_id, response_text)
            )
            input_tokens = max(1, len(prompt_text.split()))
            output_tokens = max(1, len(response_text.split()))
            await self._emit_turn_telemetry_frames(
                session_id=session_id,
                model_id=self._resolve_session_model(session_id),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_read_tokens=0,
                context_tokens=self._quota_tracker.get_session_context_tokens(session_id),
            )
            return _prompt_turn_result(
                message_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        raw_meta = params.get("_meta")
        kiro_prompt_meta = (
            raw_meta.get("kiro")
            if isinstance(raw_meta, dict) and isinstance(raw_meta.get("kiro"), dict)
            else {}
        )
        want_ttft = bool(kiro_prompt_meta.get("ttft"))
        raw_cb_req = kiro_prompt_meta.get("contextBreakdown")
        want_context_breakdown = raw_cb_req is True or raw_cb_req in ("summary", "detailed")
        was_compacted = bool(pre_session.get("compactedSummary"))
        agent_prompt_str = str(pre_session.get("agentPrompt") or "")

        user_message_id = f"msg_{uuid.uuid4().hex[:12]}"
        prompt_received_ms = int(time.time() * 1000)
        turn_start = time.perf_counter()
        ttft_s: float | None = None
        first_displayed_chunk = True
        collected_chunks: list[str] = []
        telemetry = get_telemetry(self._session_manager.state_store.db_path)
        turn_otel_ctx = telemetry.current_context()
        tool_start_times: dict[str, float] = {}
        tool_spans: dict[str, Span] = {}

        await self._quota_tracker.seed_session_baseline(
            session_id,
            model_id=self._resolve_session_model(session_id),
        )

        for start_frame in self._wire_adapter.build_turn_start_notifications(
            session_id,
            user_message_id,
        ):
            await self._emit_notification(start_frame)

        def _maybe_attach_ttft(frame: dict[str, Any], chunk_text: str) -> dict[str, Any]:
            nonlocal first_displayed_chunk, ttft_s
            if chunk_text and ttft_s is None:
                ttft_s = max(0.0, time.perf_counter() - turn_start)
            if (
                self._wire_adapter.engine_id == "v3"
                and want_ttft
                and first_displayed_chunk
                and chunk_text
            ):
                first_displayed_chunk = False
                return _stamp_ttft_meta(
                    frame,
                    prompt_received_ms=prompt_received_ms,
                    emitted_ms=max(prompt_received_ms, int(time.time() * 1000)),
                )
            return frame

        carousel_normalizer = CarouselStreamNormalizer()

        async def _emit_message_piece(piece: str) -> None:
            if piece:
                collected_chunks.append(piece)
                frame = self._wire_adapter.build_agent_message_chunk(session_id, piece)
                await self._emit_notification(_maybe_attach_ttft(frame, piece))

        async def _on_thought(chunk: str) -> None:
            frame = self._wire_adapter.build_agent_thought_chunk(session_id, chunk)
            await self._emit_notification(_maybe_attach_ttft(frame, chunk))

        async def _on_text(chunk: str) -> None:
            await _emit_message_piece(carousel_normalizer.feed(chunk))

        announced_tool_ids: set[str] = set()
        persisted_prompt_blocks: list[dict[str, Any]] = list(prompt_blocks)

        async def _on_tool_call(event: ToolCallEvent) -> bool:
            first_announce = event.tool_call_id not in announced_tool_ids
            announced_tool_ids.add(event.tool_call_id)
            if first_announce:
                tool_start_times[event.tool_call_id] = time.perf_counter()
                tool_spans[event.tool_call_id] = telemetry.start_manual_span(
                    "agy.tool_call",
                    attributes={
                        "tool.name": event.title or event.kiro_tool_name or "tool",
                        "tool.kind": event.kind,
                        "tool.id": event.tool_call_id,
                    },
                    parent_context=turn_otel_ctx,
                )
                await _emit_message_piece(carousel_normalizer.flush_prose_before_boundary())
            if self._wire_adapter.engine_id == "v3":
                return await self._handle_v3_tool_call(
                    session_id,
                    event,
                    emit_start=first_announce,
                )
            return await self._handle_v2_tool_call(
                session_id,
                event,
                emit_start=first_announce,
            )

        async def _on_tool_update(event: ToolCallEvent) -> None:
            allowed = event.status != "failed"
            t_start = tool_start_times.pop(event.tool_call_id, None)
            t_dur = max(0.0, time.perf_counter() - t_start) if t_start is not None else 0.0
            telemetry.record_tool_call(
                event.title or event.kiro_tool_name or "tool",
                kind=event.kind,
                status=event.status,
                duration_s=t_dur,
            )
            t_span = tool_spans.pop(event.tool_call_id, None)
            if t_span is not None:
                if not allowed:
                    t_span.set_status(Status(StatusCode.ERROR, event.status))
                t_span.end()
            content_blocks = list(event.diff_content) or [
                {"type": "content", "content": {"type": "text", "text": event.output}}
            ]
            if self._wire_adapter.engine_id == "v3":
                await self._emit_notification(
                    self._wire_adapter.build_tool_call_completion(
                        session_id,
                        tool_call_id=event.tool_call_id,
                        status=event.status,
                        raw_output={"output": event.output, "allowed": allowed},
                        content=content_blocks,
                    )
                )
                return

            known_mcp = self._session_manager.get_session_mcp_server_names(session_id)
            norm = _to_normalized_tool_event(event, known_mcp_servers=known_mcp)
            kiro_tool_name, mcp_server_name, _ = _resolve_v2_tool_meta(event, norm)
            await self._emit_notification(
                _session_update_frame(
                    session_id,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": event.tool_call_id,
                        "status": event.status,
                        "kind": event.kind,
                        "content": content_blocks,
                        "rawOutput": {"output": event.output, "allowed": allowed},
                        "_meta": {
                            "kiro": {
                                "toolName": kiro_tool_name,
                                "mcpServerName": mcp_server_name,
                            }
                        },
                    },
                )
            )

        async def _on_usage(ctx_tokens: int) -> None:
            await self._emit_persisted_usage_update(session_id, ctx_tokens)

        async def _on_steer_consumed(raw_content: str) -> None:
            if raw_content.strip():
                persisted_prompt_blocks.append({"type": "text", "text": raw_content})
            await _emit_message_piece(carousel_normalizer.flush_prose_before_boundary())
            await self._emit_notification(
                self._wire_adapter.build_steering_notification(
                    session_id,
                    "consumed",
                    raw_content,
                )
            )

        async def _on_subagent_list(subagents: list[dict[str, Any]]) -> None:
            for frame in self._wire_adapter.build_subagent_notifications(session_id, subagents):
                await self._emit_notification(frame)

        telemetry.turn_started()
        try:
            turn_result = await self._session_manager.prompt(
                session_id=session_id,
                prompt_blocks=prompt_blocks,
                on_thought=_on_thought,
                on_text=_on_text,
                on_tool_call=_on_tool_call,
                on_tool_update=_on_tool_update,
                on_usage=_on_usage,
                on_steer_consumed=_on_steer_consumed,
                on_subagent_list=_on_subagent_list,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        finally:
            telemetry.turn_finished()
            for dangling_span in tool_spans.values():
                dangling_span.end()
            tool_spans.clear()

        active_model = self._resolve_session_model(session_id)
        stop_reason = str(turn_result.get("stopReason", "end_turn"))
        if stop_reason == "cancelled" or self._session_manager.consume_cancellation(session_id):
            await _emit_message_piece(carousel_normalizer.flush())
            telemetry.record_turn(
                stop_reason="cancelled",
                model=active_model,
                engine=self._wire_adapter.engine_id,
                duration_s=time.perf_counter() - turn_start,
                ttft_s=ttft_s,
            )
            raw_cancel_usage = turn_result.get("usage")
            return await self._finish_cancelled_prompt(
                message_id,
                session_id=session_id,
                usage=raw_cancel_usage if isinstance(raw_cancel_usage, Mapping) else None,
                turn_id=user_message_id,
            )

        await _emit_message_piece(carousel_normalizer.flush())

        fallback_from = str(turn_result.get("fallbackFromModel") or "")
        resolved_to = str(turn_result.get("resolvedModel") or "")
        if fallback_from and resolved_to and fallback_from != resolved_to:
            telemetry.record_model_fallback(fallback_from, resolved_to, reason="availability")
            for fb_frame in self._wire_adapter.build_model_fallback_notifications(
                session_id,
                from_model=fallback_from,
                to_model=resolved_to,
                trigger="availability",
                scope="session",
            ):
                await self._emit_notification(fb_frame)

        turn_error = str(turn_result.get("error") or "").strip()
        raw_assistant_text = str(turn_result.get("response") or "".join(collected_chunks))
        assistant_text = normalize_carousel_markdown(raw_assistant_text)
        if (not turn_error or assistant_text) and self._session_manager.state_store.get_session(
            session_id
        ) is not None:
            self._session_manager.state_store.append_session_turn(
                session_id=session_id,
                user_message_id=user_message_id,
                prompt_blocks=persisted_prompt_blocks,
                assistant_text=assistant_text,
            )

        raw_usage = turn_result.get("usage")
        usage_dict = raw_usage if isinstance(raw_usage, dict) else {}
        input_tokens = max(1, int(usage_dict.get("input_tokens", 0)))
        output_tokens = max(1, int(usage_dict.get("output_tokens", 0)))
        cached_read_tokens = int(usage_dict.get("cache_read_tokens", 0))
        thinking_tokens = int(usage_dict.get("thinking_tokens", 0))
        raw_ctx = usage_dict.get("context_tokens")
        context_tokens = int(raw_ctx) if raw_ctx is not None and int(raw_ctx) > 0 else None

        context_breakdown: dict[str, Any] | None = None
        if want_context_breakdown and self._wire_adapter.engine_id == "v3":
            system_chars = len(agent_prompt_str)
            history_chars = len(prompt_text) + len(assistant_text)
            context_breakdown = {
                "schema": 1,
                "totalChars": system_chars + history_chars,
                "modelCalls": 1,
                "categories": {
                    "system": {"chars": system_chars},
                    "history": {"chars": history_chars},
                },
                **({"compacted": True} if was_compacted else {}),
            }

        await self._emit_turn_telemetry_frames(
            session_id=session_id,
            model_id=self._resolve_session_model(session_id),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
            context_tokens=context_tokens,
            turn_id=user_message_id,
            context_breakdown=context_breakdown,
        )

        telemetry.record_turn(
            stop_reason="error" if turn_error else stop_reason,
            model=self._resolve_session_model(session_id),
            engine=self._wire_adapter.engine_id,
            duration_s=time.perf_counter() - turn_start,
            ttft_s=ttft_s,
        )
        if turn_error:
            return _jsonrpc_error(
                message_id,
                -32603,
                turn_error,
                data={"errorType": _classify_turn_error_type(turn_error)},
            )

        return _prompt_turn_result(
            message_id,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
        )

    async def _emit_persisted_usage_update(
        self,
        session_id: str,
        context_tokens: int,
    ) -> None:
        """Emit persisted `usage_update` (omitting hardcoded `cost` on KAS v3)."""
        notif = self._quota_tracker.build_usage_update_notification(
            session_id=session_id,
            context_tokens=context_tokens,
        )
        if self._wire_adapter.engine_id == "v3":
            upd = dict(notif["params"]["update"])
            upd.pop("cost", None)
            notif = {**notif, "params": {**notif["params"], "update": upd}}
        await self._emit_notification(notif)

    async def _emit_available_commands_update(self, session_id: str) -> None:
        """Emit the `available_commands_update` notification for a session."""
        await self._emit_notification(
            _session_update_frame(
                session_id,
                {
                    "sessionUpdate": "available_commands_update",
                    "availableCommands": list(AVAILABLE_SLASH_COMMANDS),
                },
            )
        )

    async def _emit_mcp_servers_initialized(self, session_id: str) -> None:
        """Emit `_kiro.dev/mcp/server_initialized` for every configured session MCP server."""
        for server_name in self._session_manager.get_session_mcp_server_names(session_id):
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/mcp/server_initialized",
                    "params": {"sessionId": session_id, "serverName": server_name},
                }
            )

    async def _emit_or_queue_mcp_readiness(
        self,
        session_id: str,
        *,
        post_response: bool = False,
        server_filter: str | None = None,
    ) -> None:
        """Emit full-snapshot MCP readiness frames (`_kiro/mcp/status` + `didChange` on v3)."""
        if self._wire_adapter.engine_id != "v3":
            await self._emit_mcp_servers_initialized(session_id)
            return

        if server_filter is None:
            statuses = await self._session_manager.probe_session_mcp_servers(
                session_id,
                server_filter=None,
            )
            self._mcp_status_cache[session_id] = {s.name: s for s in statuses}
        else:
            ordered_names = self._session_manager.get_session_probe_server_names(session_id)
            cache = self._mcp_status_cache.setdefault(session_id, {})
            for stale_name in list(cache):
                if stale_name not in ordered_names:
                    cache.pop(stale_name, None)
            targets = [
                *([server_filter] if server_filter in ordered_names else []),
                *[name for name in ordered_names if name not in cache and name != server_filter],
            ]
            for target_name in targets:
                for item in await self._session_manager.probe_session_mcp_servers(
                    session_id,
                    server_filter=target_name,
                ):
                    cache[item.name] = item
            statuses = [cache[name] for name in ordered_names if name in cache]

        frames = self._wire_adapter.build_mcp_readiness_notifications(session_id, statuses)
        if post_response:
            self._queue_post_response_notifications(frames)
        else:
            for frame in frames:
                await self._emit_notification(frame)

    async def _emit_post_session_open(
        self,
        session_id: str,
        *,
        include_persisted_usage: bool = True,
    ) -> None:
        """Emit available commands, MCP readiness, and optional persisted usage on session open."""
        if self._wire_adapter.engine_id == "v2":
            await self._emit_available_commands_update(session_id)
        await self._emit_or_queue_mcp_readiness(
            session_id,
            post_response=(self._wire_adapter.engine_id == "v3"),
        )
        if include_persisted_usage:
            persisted_ctx = self._quota_tracker.get_session_context_tokens(session_id)
            if persisted_ctx is not None and persisted_ctx > 0:
                await self._emit_persisted_usage_update(session_id, persisted_ctx)

    async def _emit_session_replay(self, session_id: str) -> None:
        """Replay stored turns (`_meta.kiro.replay == True`) before `session/load` returns."""
        for turn in self._session_manager.state_store.get_session_turns(session_id):
            msg_id = str(turn.get("user_message_id") or turn.get("userMessageId") or "")
            blocks = turn.get("prompt_blocks")
            if blocks is None:
                blocks = turn.get("promptBlocks")
            for block in blocks if isinstance(blocks, list) else ():
                if isinstance(block, dict):
                    frame = self._wire_adapter.build_user_message_chunk(
                        session_id,
                        block,
                        user_message_id=msg_id,
                        replay=True,
                    )
                    await self._emit_notification(_stamp_replay_meta(frame))
            assistant_text = normalize_carousel_markdown(
                str(turn.get("assistant_text") or turn.get("assistantText") or "")
            )
            if assistant_text:
                reply_frame = self._wire_adapter.build_agent_message_chunk(
                    session_id,
                    assistant_text,
                    replay=True,
                )
                await self._emit_notification(_stamp_replay_meta(reply_frame))

    async def _handle_session_new(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/new`, creating a session and emitting MCP readiness notifications."""
        cwd = _require_non_empty_str(params, "cwd")
        if cwd is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params: 'cwd' must be non-empty")
        mcp_servers, meta, err = _parse_session_wire_options(message_id, params)
        if err is not None:
            return err
        desc = await self._session_manager.create_session(
            cwd=cwd,
            mcp_servers=mcp_servers,
            meta=meta,
        )
        await self._emit_post_session_open(desc["sessionId"], include_persisted_usage=True)
        return _jsonrpc_result(message_id, desc)

    async def _handle_session_load(
        self,
        message_id: Any,
        params: dict[str, Any],
        *,
        require_cwd: bool = True,
    ) -> dict[str, Any]:
        """Handle `session/load`, replaying turns BEFORE result and queueing MCP readiness after."""
        session_id = _require_non_empty_str(params, "sessionId")
        if session_id is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params: 'sessionId' is required")
        if require_cwd:
            cwd = _require_non_empty_str(params, "cwd")
            if cwd is None:
                return _jsonrpc_error(message_id, -32602, "Invalid params: 'cwd' is required")
        elif params.get("cwd") is None:
            cwd = ""
        else:
            cwd = _require_non_empty_str(params, "cwd") or ""
            if not cwd:
                return _jsonrpc_error(message_id, -32602, "Invalid params: 'cwd' must be non-empty")

        mcp_servers, meta, err = _parse_session_wire_options(message_id, params)
        if err is not None:
            return err
        try:
            desc = await self._session_manager.load_session(
                session_id=session_id,
                cwd=cwd,
                mcp_servers=mcp_servers,
                meta=meta,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))

        sid = desc["sessionId"]
        await self._emit_session_replay(sid)
        await self._emit_post_session_open(sid, include_persisted_usage=True)
        return _jsonrpc_result(message_id, desc)

    async def _handle_session_fork(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/fork`, copying session state and turns into a new session."""
        source_session_id = _require_non_empty_str(params, "sessionId")
        if source_session_id is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params: 'sessionId' is required")
        raw_cwd = params.get("cwd")
        cwd: str | None = None
        if raw_cwd is not None:
            if not isinstance(raw_cwd, str) or not raw_cwd.strip():
                return _jsonrpc_error(message_id, -32602, "Invalid params: 'cwd' must be non-empty")
            cwd = raw_cwd.strip()
        mcp_servers, meta, err = _parse_session_wire_options(message_id, params)
        if err is not None:
            return err

        raw_top_msg_id = params.get("messageId")
        if raw_top_msg_id is not None and not isinstance(raw_top_msg_id, str):
            return _jsonrpc_error(
                message_id,
                -32602,
                "Invalid params: 'messageId' must be a string",
            )
        fork_msg_id = _extract_fork_message_id(meta, raw_top_msg_id)
        try:
            desc = await self._session_manager.fork_session(
                source_session_id=source_session_id,
                cwd=cwd,
                mcp_servers=mcp_servers,
                meta=meta,
                message_id=fork_msg_id,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))

        await self._emit_post_session_open(desc["sessionId"], include_persisted_usage=False)
        return _jsonrpc_result(message_id, desc)

    async def _handle_session_set_mode(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/set_mode`, emitting mode switch and post-switch MCP readiness."""
        session_id = _require_non_empty_str(params, "sessionId")
        mode_id = _require_non_empty_str(params, "modeId")
        if session_id is None or mode_id is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        try:
            res = await self._session_manager.set_mode(
                session_id=session_id,
                mode_id=mode_id,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        except ValueError as exc:
            return _jsonrpc_error(message_id, -32602, "Invalid params", data=str(exc))

        if self._wire_adapter.engine_id == "v3":
            for frame in self._wire_adapter.build_mode_switched_notifications(session_id, mode_id):
                await self._emit_notification(frame)
            await self._emit_or_queue_mcp_readiness(session_id, post_response=True)
            return _jsonrpc_result(message_id, res)

        session_state = self._session_manager.get_session(session_id)
        is_custom_agent = bool(session_state.get("agent"))
        if is_custom_agent:
            switched_model = session_state.get("switchedModel")
            switched_params: dict[str, Any] = {
                "sessionId": session_id,
                "agentName": mode_id,
                **(
                    {"model": switched_model}
                    if isinstance(switched_model, str) and switched_model
                    else {}
                ),
            }
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/agent/switched",
                    "params": switched_params,
                }
            )
        await self._emit_notification(
            _session_update_frame(
                session_id,
                {"sessionUpdate": "current_mode_update", "currentModeId": mode_id},
            )
        )
        if is_custom_agent:
            await self._emit_mcp_servers_initialized(session_id)
        return _jsonrpc_result(message_id, res)

    async def _handle_session_set_model(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/set_model`, updating the active session model."""
        session_id = _require_non_empty_str(params, "sessionId")
        model_id = _require_non_empty_str(params, "modelId")
        if session_id is None or model_id is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        try:
            res = await self._session_manager.set_model(
                session_id=session_id,
                model_id=model_id,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        except ValueError as exc:
            return _jsonrpc_error(message_id, -32602, "Invalid params", data=str(exc))
        return _jsonrpc_result(message_id, res)

    async def _handle_kas_session_compact(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/session/compact` extension RPC."""
        raw_instructions = params.get("instructions")
        if raw_instructions is not None and not isinstance(raw_instructions, str):
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        session_id, err = self._require_existing_session(message_id, params)
        if err is not None or session_id is None:
            return err or _jsonrpc_error(message_id, -32602, "Invalid params")

        await self._emit_notification(
            self._wire_adapter.build_compaction_notification(session_id, "started")
        )
        try:
            compact_res = await self._session_manager.compact_session(
                session_id=session_id,
                instructions=raw_instructions or "",
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))

        status_type = str(compact_res.get("status", "completed"))
        summary_text = str(compact_res.get("summary", ""))
        get_telemetry(self._session_manager.state_store.db_path).record_compaction(
            trigger="rpc",
            status="failed" if status_type == "failed" else "completed",
        )
        if status_type == "failed":
            await self._emit_notification(
                self._wire_adapter.build_compaction_notification(
                    session_id,
                    "failed",
                    error=summary_text or "Compaction failed",
                )
            )
            return _jsonrpc_result(message_id, {"success": False})

        await self._emit_notification(
            self._wire_adapter.build_compaction_notification(
                session_id,
                "completed",
                summary=summary_text,
            )
        )
        telemetry = await self._quota_tracker.build_turn_telemetry(
            session_id=session_id,
            model_id=self._resolve_session_model(session_id),
            input_tokens=max(1, int(compact_res.get("inputTokens", 1))),
            output_tokens=max(1, int(compact_res.get("outputTokens", 1))),
            cached_read_tokens=0,
            context_tokens=int(compact_res.get("contextTokens", 0)),
        )
        await self._emit_persisted_usage_update(session_id, telemetry.context_tokens)
        if self._wire_adapter.engine_id == "v3":
            await self._emit_notification(
                _session_update_frame(
                    session_id,
                    {
                        "sessionUpdate": "session_info_update",
                        "_meta": {
                            "kiro": {
                                "kind": "context_usage",
                                "usagePercentage": telemetry.context_usage_pct,
                            }
                        },
                    },
                )
            )
        return _jsonrpc_result(message_id, {"success": True})

    async def _handle_kas_mcp_toggle(
        self,
        message_id: Any,
        params: dict[str, Any],
        *,
        is_request: bool = False,
    ) -> dict[str, Any] | None:
        """Handle `_kiro/mcp/toggle` notification or request to toggle process MCP subsystem."""
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            return _jsonrpc_error(message_id, -32602, "Invalid params") if is_request else None
        self._mcp_status_cache.clear()
        await self._session_manager.set_mcp_enabled(enabled)
        return _jsonrpc_result(message_id, {}) if is_request else None

    async def _handle_kas_mcp_reset_server(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/mcp/resetServer` by re-probing target server and emitting full roster."""
        server_name = _require_non_empty_str(params, "serverName")
        raw_start_oauth = params.get("startOAuth")
        if server_name is None or (
            raw_start_oauth is not None and not isinstance(raw_start_oauth, bool)
        ):
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        session_id, err = self._require_existing_session(message_id, params)
        if err is not None or session_id is None:
            return err or _jsonrpc_error(message_id, -32602, "Invalid params")

        clean_server_name = server_name.strip()
        await self._emit_or_queue_mcp_readiness(
            session_id,
            post_response=False,
            server_filter=clean_server_name,
        )
        if (
            raw_start_oauth is True
            and self._client_opens_external_url
            and self._client_rpc_caller is not None
        ):
            target_status = self._mcp_status_cache.get(session_id, {}).get(clean_server_name)
            auth_url = target_status.auth_url if target_status is not None else None
            if isinstance(auth_url, str) and auth_url.startswith(("http://", "https://")):
                await self._client_rpc_caller("_kiro/openExternalUrl", {"url": auth_url})
        return _jsonrpc_result(message_id, {})

    async def _handle_commands_execute(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro.dev/commands/execute` (v2 only; refused with `-32603` on KAS v3)."""
        if self._wire_adapter.engine_id == "v3":
            return _jsonrpc_error(
                message_id,
                -32603,
                "Method '_kiro.dev/commands/execute' is not supported on KAS v3",
            )

        session_id = ""
        if params.get("sessionId") is not None:
            validated_sid, err = self._require_existing_session(message_id, params)
            if err is not None or validated_sid is None:
                return err or _jsonrpc_error(message_id, -32602, "Invalid params")
            session_id = validated_sid

        raw_cmd = params.get("command")
        cmd_args: dict[str, Any] = {}
        if isinstance(raw_cmd, dict):
            cmd_name = str(raw_cmd.get("command", "")).strip().lstrip("/").lower()
            if isinstance(raw_cmd.get("args"), dict):
                cmd_args = dict(raw_cmd["args"])
        elif isinstance(raw_cmd, str):
            parts = raw_cmd.strip().split(None, 1)
            cmd_name = parts[0].lstrip("/").lower() if parts else ""
            if len(parts) > 1:
                cmd_args = {"value": parts[1]}
        else:
            return _jsonrpc_error(message_id, -32602, "Invalid params")

        if cmd_name in {"quota", "usage", "credits"}:
            md = await self._quota_tracker.format_slash_command_response(f"/{cmd_name}")
            if session_id:
                persisted_ctx = self._quota_tracker.get_session_context_tokens(session_id)
                if persisted_ctx is not None and persisted_ctx > 0:
                    await self._emit_persisted_usage_update(session_id, persisted_ctx)
            return _jsonrpc_result(
                message_id, {"message": md, "text": md, "stopReason": "end_turn"}
            )
        if cmd_name == "effort":
            level = str(cmd_args.get("level") or cmd_args.get("value") or "medium").strip().lower()
            if session_id:
                try:
                    await self._session_manager.set_config_option(
                        session_id=session_id,
                        config_id=self._wire_adapter.effort_config_id,
                        value=level,
                    )
                except KeyError, ValueError:
                    pass
            msg = f"Reasoning effort set to {level}"
            return _jsonrpc_result(
                message_id, {"message": msg, "text": msg, "stopReason": "end_turn"}
            )
        if cmd_name in {"mcp", "tools"}:
            server_names = (
                self._session_manager.get_session_mcp_server_names(session_id) if session_id else []
            )
            if cmd_name == "mcp":
                data: dict[str, Any] = {
                    "servers": [
                        {"name": name, "status": "running", "toolCount": 1} for name in server_names
                    ]
                }
                label = "MCP servers"
            else:
                data = {
                    "tools": [
                        {"name": f"@{name}", "source": f"mcp:{name}"} for name in server_names
                    ]
                }
                label = "Tools"
            return _jsonrpc_result(
                message_id,
                {"message": label, "data": data, "stopReason": "end_turn"},
            )
        return _jsonrpc_result(message_id, {"message": "", "text": "", "stopReason": "end_turn"})

    async def _handle_session_steer(
        self,
        method: str,
        message_id: Any,
        params: dict[str, Any],
        *,
        is_request: bool = True,
    ) -> dict[str, Any] | None:
        """Handle `_session/steer` and `_session/steering` during an active turn."""
        if method == "_session/steering":
            raw_prompt = params.get("prompt")
            if (
                not isinstance(raw_prompt, list)
                or not raw_prompt
                or _require_non_empty_str(params, "sessionId") is None
            ):
                return _jsonrpc_error(message_id, -32602, "Invalid params") if is_request else None
            session_id, err = self._require_existing_session(
                message_id, params, is_request=is_request
            )
            if session_id is None:
                return err
            steer_blocks = [b for b in raw_prompt if isinstance(b, dict)]
            queued = await self._session_manager.steer(
                session_id=session_id,
                prompt_blocks=steer_blocks,
            )
            get_telemetry(self._session_manager.state_store.db_path).record_steer(
                outcome="queued" if queued else "failed"
            )
            if queued:
                await self._emit_notification(
                    self._wire_adapter.build_steering_notification(
                        session_id,
                        "queued",
                        self._extract_prompt_text(steer_blocks),
                    )
                )
            return (
                _jsonrpc_result(message_id, {"outcome": "injected" if queued else "failed"})
                if is_request
                else None
            )

        raw_message = _require_non_empty_str(params, "message")
        if raw_message is None or _require_non_empty_str(params, "sessionId") is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params") if is_request else None
        session_id, err = self._require_existing_session(message_id, params, is_request=is_request)
        if session_id is None:
            return err

        queued = await self._session_manager.steer(session_id=session_id, message=raw_message)
        get_telemetry(self._session_manager.state_store.db_path).record_steer(
            outcome="queued" if queued else "failed"
        )
        if queued:
            await self._emit_notification(
                self._wire_adapter.build_steering_notification(session_id, "queued", raw_message)
            )
        return _jsonrpc_result(message_id, {"queued": queued}) if is_request else None

    async def _handle_session_steer_clear(
        self,
        message_id: Any,
        params: dict[str, Any],
        *,
        is_request: bool = True,
    ) -> dict[str, Any] | None:
        """Handle `_session/steer/clear` to discard queued steers and emit notification."""
        session_id, err = self._require_existing_session(message_id, params, is_request=is_request)
        if session_id is None:
            return err

        dropped = self._session_manager.clear_session_steers(session_id)
        await self._emit_notification(
            self._wire_adapter.build_steering_notification(
                session_id,
                "discarded",
                "\n".join(dropped),
            )
        )
        return _jsonrpc_result(message_id, {"cleared": True}) if is_request else None

    async def _handle_kas_session_context(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/session/context` extension RPC."""
        session_id = _require_non_empty_str(params, "sessionId")
        subcommand = (
            _require_non_empty_str(params, "subcommand")
            or _require_non_empty_str(params, "action")
            or _require_non_empty_str(params, "command")
        )
        raw_path = params.get("path")
        raw_force = params.get("force", False)
        if (
            session_id is None
            or subcommand is None
            or (raw_path is not None and not isinstance(raw_path, str))
            or not isinstance(raw_force, bool)
        ):
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        try:
            res = self._session_manager.handle_session_context(
                session_id,
                subcommand=subcommand,
                path=raw_path,
                force=raw_force,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        return _jsonrpc_result(message_id, res)

    async def _handle_kas_knowledge(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/knowledge` extension RPC."""
        session_id = _require_non_empty_str(params, "sessionId")
        raw_sub = params.get("subcommand")
        if session_id is None or (
            raw_sub is not None and (not isinstance(raw_sub, str) or not raw_sub.strip())
        ):
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        try:
            res = self._session_manager.handle_knowledge(session_id, params)
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        return _jsonrpc_result(message_id, res)

    async def _handle_kas_code_intelligence(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/codeIntelligence` extension RPC."""
        session_id = _require_non_empty_str(params, "sessionId")
        raw_sub = (
            params.get("subcommand")
            if "subcommand" in params
            else (params.get("action") if "action" in params else params.get("command"))
        )
        if session_id is None or (
            raw_sub is not None and (not isinstance(raw_sub, str) or not raw_sub.strip())
        ):
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        subcommand = raw_sub.strip() if isinstance(raw_sub, str) else "status"
        try:
            res = self._session_manager.handle_code_intelligence(
                session_id,
                subcommand=subcommand,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        return _jsonrpc_result(message_id, res)

    async def _handle_kas_checkpoint_revert_multiple(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/checkpoint/revertMultiple` extension RPC."""
        session_id = _require_non_empty_str(params, "sessionId")
        target_msg_id = _require_non_empty_str(params, "messageId")
        if session_id is None or target_msg_id is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        existing = self._session_manager.state_store.get_session(session_id)
        old_agy_id = (
            str(existing.get("agyConversationId") or "").strip() if existing is not None else ""
        )
        if old_agy_id:
            await self._session_manager.cli_provider.close_backend_session(old_agy_id)
        try:
            res = self._session_manager.revert_checkpoint_by_message_id(
                session_id,
                message_id=target_msg_id,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        except ValueError as exc:
            return _jsonrpc_error(message_id, -32602, "Invalid params", data=str(exc))
        return _jsonrpc_result(message_id, res)

    async def _handle_session_set_config_option(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/set_config_option` and emit `config_option_update`."""
        session_id = _require_non_empty_str(params, "sessionId")
        config_id = _require_non_empty_str(params, "configId")
        value = _require_non_empty_str(params, "value")
        if session_id is None or config_id is None or value is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        try:
            res = await self._session_manager.set_config_option(
                session_id=session_id,
                config_id=config_id,
                value=value,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))
        except ValueError as exc:
            return _jsonrpc_error(message_id, -32602, "Invalid params", data=str(exc))
        await self._emit_notification(
            _session_update_frame(
                session_id,
                {
                    "sessionUpdate": "config_option_update",
                    "configOptions": res["configOptions"],
                },
            )
        )
        return _jsonrpc_result(message_id, res)

    def serialize_frame(self, payload: dict[str, Any]) -> bytes:
        """Serialize a JSON-RPC 2.0 payload into a newline-delimited UTF-8 frame <= 16 MiB."""
        serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        encoded = f"{serialized}\n".encode()
        if len(encoded) > MAX_FRAME_SIZE_BYTES:
            raise ValueError(
                f"Serialized JSON-RPC frame size ({len(encoded)} bytes) "
                f"exceeds maximum of {MAX_FRAME_SIZE_BYTES} bytes"
            )
        return encoded

    async def handle_message(self, raw_line: str) -> dict[str, Any] | None:
        """Parse and dispatch a single JSON-RPC 2.0 request or notification line."""
        self._post_response_ctx.set([])
        await anyio.sleep(0)
        try:
            parsed: Any = json.loads(raw_line)
        except json.JSONDecodeError:
            return _jsonrpc_error(None, -32700, "Parse error")

        if not isinstance(parsed, dict):
            return _jsonrpc_error(None, -32600, "Invalid Request")

        has_id = "id" in parsed
        msg_id = parsed.get("id")
        if has_id and not (
            msg_id is None or (isinstance(msg_id, (str, int)) and not isinstance(msg_id, bool))
        ):
            return _jsonrpc_error(None, -32600, "Invalid Request")

        method = parsed.get("method")
        raw_params = parsed.get("params")
        db_path = self._session_manager.state_store.db_path
        telemetry = get_telemetry(db_path)
        span_method = method.strip() if isinstance(method, str) and method.strip() else "invalid"
        rpc_start = time.perf_counter()

        with telemetry.start_span(
            f"acp.rpc.{span_method}",
            kind=SpanKind.SERVER,
            attributes={
                "rpc.system": "jsonrpc",
                "rpc.method": span_method,
                "agy_acp.engine": self._wire_adapter.engine_id,
            },
        ) as span:
            log_acp_event(
                "rpc.receive",
                save_dir_or_db=db_path,
                id=msg_id,
                method=method,
                params=raw_params,
                raw_message=parsed,
            )
            if not isinstance(method, str) or not method.strip():
                err_resp = _jsonrpc_error(msg_id if has_id else None, -32600, "Invalid Request")
                telemetry.record_rpc_error(
                    span_method,
                    code=-32600,
                    engine=self._wire_adapter.engine_id,
                )
                span.set_status(Status(StatusCode.ERROR, "Invalid Request"))
                return err_resp

            if raw_params is not None and not isinstance(raw_params, dict):
                telemetry.record_rpc_error(
                    span_method,
                    code=-32602,
                    engine=self._wire_adapter.engine_id,
                )
                span.set_status(Status(StatusCode.ERROR, "Invalid params"))
                return _jsonrpc_error(msg_id, -32602, "Invalid params") if has_id else None

            response = await self._dispatch_method(
                method,
                msg_id,
                raw_params,
                parsed,
                is_request=has_id,
            )
            duration_s = time.perf_counter() - rpc_start
            telemetry.record_rpc_request(
                method,
                duration_s=duration_s,
                engine=self._wire_adapter.engine_id,
            )
            if response is not None:
                if isinstance(response.get("error"), Mapping):
                    err_code = response["error"].get("code", -32603)
                    telemetry.record_rpc_error(
                        method,
                        code=err_code,
                        engine=self._wire_adapter.engine_id,
                    )
                    span.set_status(
                        Status(StatusCode.ERROR, str(response["error"].get("message", "")))
                    )
                log_acp_event(
                    "rpc.response",
                    save_dir_or_db=db_path,
                    id=msg_id,
                    method=method,
                    response=response,
                )
            return response

    def _build_initialize_response(
        self,
        msg_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if "protocolVersion" in params and params["protocolVersion"] is not None:
            pv = params["protocolVersion"]
            valid_int_pv = isinstance(pv, int) and not isinstance(pv, bool) and pv >= 1
            if not (valid_int_pv or pv == "2025-08-22"):
                return _jsonrpc_error(
                    msg_id,
                    -32602,
                    "Invalid params: 'protocolVersion' must be an integer >= 1",
                )
        raw_client_caps = params.get("clientCapabilities")
        if isinstance(raw_client_caps, Mapping):
            raw_client_meta = raw_client_caps.get("_meta")
            if isinstance(raw_client_meta, Mapping):
                raw_client_kiro = raw_client_meta.get("kiro")
                if isinstance(raw_client_kiro, Mapping):
                    self._client_opens_external_url = raw_client_kiro.get("openExternalUrl") is True

        init_res = self._wire_adapter.build_initialize_result(
            version=KIRO_CLI_COMPAT_VERSION,
            name="agy-acp",
            title="Google Antigravity ACP Harness",
        )
        caps = {
            **dict(init_res["agentCapabilities"]),
            "sessionCapabilities": {"close": {}, "fork": {}, "list": {}, "resume": {}},
        }
        info = {**dict(init_res["agentInfo"]), "_meta": {"agyAcpVersion": __version__}}
        return _jsonrpc_result(
            msg_id,
            {
                "protocolVersion": init_res["protocolVersion"],
                "agentCapabilities": caps,
                "agentInfo": info,
                "authMethods": [],
            },
        )

    async def _dispatch_method(
        self,
        method: str,
        msg_id: Any,
        raw_params: dict[str, Any] | None,
        message: dict[str, Any],
        *,
        is_request: bool = True,
    ) -> dict[str, Any] | None:
        """Route a parsed JSON-RPC 2.0 method to its handler."""
        params = raw_params if isinstance(raw_params, dict) else {}
        if method == "initialize":
            return self._build_initialize_response(msg_id, params)
        if method == "session/new":
            return await self._handle_session_new(msg_id, params)
        if method == "session/fork":
            return await self._handle_session_fork(msg_id, params)
        if method in {"session/load", "session/resume"}:
            return await self._handle_session_load(
                msg_id,
                params,
                require_cwd=(method == "session/load"),
            )
        if method == "session/list":
            cwd_filter = params.get("cwd")
            if cwd_filter is not None and not isinstance(cwd_filter, str):
                return _jsonrpc_error(msg_id, -32602, "Invalid params")
            sessions = self._session_manager.list_sessions(
                cwd=cwd_filter if isinstance(cwd_filter, str) and cwd_filter.strip() else None
            )
            return _jsonrpc_result(
                msg_id,
                {
                    "sessions": [
                        {
                            "sessionId": s["sessionId"],
                            "cwd": s["cwd"],
                            "title": s.get("title"),
                            "updatedAt": s.get("updatedAt"),
                        }
                        for s in sessions
                    ]
                },
            )
        if method == "session/set_mode":
            return await self._handle_session_set_mode(msg_id, params)
        if method == "session/set_model":
            return await self._handle_session_set_model(msg_id, params)
        if method == "session/set_config_option":
            return await self._handle_session_set_config_option(msg_id, params)
        if method == "_kiro/session/compact":
            return await self._handle_kas_session_compact(msg_id, params)
        if method == "_kiro/session/context":
            return await self._handle_kas_session_context(msg_id, params)
        if method == "_kiro/knowledge":
            return await self._handle_kas_knowledge(msg_id, params)
        if method == "_kiro/codeIntelligence":
            return await self._handle_kas_code_intelligence(msg_id, params)
        if method == "_kiro/checkpoint/revertMultiple":
            return await self._handle_kas_checkpoint_revert_multiple(msg_id, params)
        if method == "_kiro/mcp/toggle":
            return await self._handle_kas_mcp_toggle(msg_id, params, is_request=is_request)
        if method == "_kiro/mcp/resetServer":
            return await self._handle_kas_mcp_reset_server(msg_id, params)
        if method == "_kiro.dev/commands/execute":
            return await self._handle_commands_execute(msg_id, params)
        if method == "session/prompt":
            return await self._handle_session_prompt(msg_id, params)
        if method == "session/cancel":
            session_id = _require_non_empty_str(params, "sessionId")
            if session_id is None:
                return _jsonrpc_error(msg_id, -32602, "Invalid params") if is_request else None
            if is_request:
                _, err = self._require_existing_session(msg_id, params)
                if err is not None:
                    return err
            await self._session_manager.cancel(session_id=session_id)
            return _jsonrpc_result(msg_id, {}) if is_request else None
        if method == "_session/steer/clear":
            return await self._handle_session_steer_clear(msg_id, params, is_request=is_request)
        if method in {"_session/steer", "_session/steering"}:
            return await self._handle_session_steer(
                method,
                msg_id,
                params,
                is_request=is_request,
            )
        if method in {"session/close", "_kiro.dev/session/terminate", "_kiro/session/delete"}:
            session_id, err = self._require_existing_session(msg_id, params)
            if err is not None or session_id is None:
                return err or _jsonrpc_error(msg_id, -32602, "Invalid params")
            is_delete = method == "_kiro/session/delete"
            res = (
                await self._session_manager.evict_session(session_id=session_id)
                if method == "session/close"
                else await self._session_manager.terminate_session(
                    session_id=session_id,
                    delete_persisted=is_delete,
                )
            )
            self._mcp_status_cache.pop(session_id, None)
            if is_delete:
                self._quota_tracker.evict_session(session_id)
            return _jsonrpc_result(msg_id, res)

        log_acp_event(
            "rpc.unhandled",
            save_dir_or_db=self._session_manager.state_store.db_path,
            id=msg_id,
            method=method,
            params=raw_params,
            raw_message=message,
        )
        return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}") if is_request else None
