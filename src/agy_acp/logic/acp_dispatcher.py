"""ACP JSON-RPC 2.0 request/notification dispatcher and framing engine."""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

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
from agy_acp.logic.wire import (
    EngineWireAdapter,
    McpServerStatus,
    NormalizedToolEvent,
    ToolCallKind,
    V2KiroWireAdapter,
    WireTurnTelemetry,
)
from agy_acp.providers.acp_logger import log_acp_event
from agy_acp.providers.agy_cli_provider import ToolCallEvent

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


def _resolve_mcp_server_and_tool(
    after_prefix: str,
    known_mcp_servers: Sequence[str],
) -> tuple[str, str]:
    """Match `<server>_<tool>` against longest known session MCP server name first."""
    ordered = sorted(
        (s for s in known_mcp_servers if isinstance(s, str) and s),
        key=len,
        reverse=True,
    )
    for srv in ordered:
        prefix = f"{srv}_"
        if after_prefix.startswith(prefix) and len(after_prefix) > len(prefix):
            return srv, after_prefix[len(prefix) :]
    srv, sep, tool = after_prefix.partition("_")
    return srv, (tool if sep else after_prefix)


def _resolve_event_mcp_identity(
    event: ToolCallEvent,
    known_mcp_servers: Sequence[str],
) -> tuple[str, str]:
    """Resolve `(mcp_server, mcp_tool)` for `event` using longest configured server prefix."""
    if event.mcp_server and event.mcp_tool:
        combined = f"{event.mcp_server}_{event.mcp_tool}"
        ordered = sorted(
            (s for s in known_mcp_servers if isinstance(s, str) and s),
            key=len,
            reverse=True,
        )
        for srv in ordered:
            prefix = f"{srv}_"
            if combined.startswith(prefix) and len(combined) > len(prefix):
                return srv, combined[len(prefix) :]
        return event.mcp_server, event.mcp_tool

    for raw_candidate in (event.kiro_tool_id, event.kiro_tool_name, event.title):
        if not isinstance(raw_candidate, str):
            continue
        if raw_candidate.startswith("mcp_") and len(raw_candidate) > 4:
            return _resolve_mcp_server_and_tool(raw_candidate[4:], known_mcp_servers)
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
    """Convert an `AgyCliProvider` `ToolCallEvent` into a wire `NormalizedToolEvent`."""
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
    for key in (
        "Path",
        "path",
        "TargetFile",
        "AbsolutePath",
        "DirectoryPath",
        "SearchDirectory",
        "TargetDirectory",
        "file_path",
    ):
        val = norm.raw_input.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


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
        abs_posix = resolved.as_posix()
        cwd_posix = cwd_resolved.as_posix() if cwd_resolved is not None else ""
        abs_perms = _adapt_permissions_for_cwd(
            permissions,
            cwd_posix=cwd_posix,
            for_absolute=True,
        )
        abs_dec = evaluate_tool_call_permission(
            abs_perms,
            tool_id=norm.kas_tool_id,
            path=abs_posix,
            url=url,
            query=query,
            subagent_target=norm.subagent_target,
        )
        if cwd_resolved is not None and resolved.is_relative_to(cwd_resolved):
            rel_posix = (
                "." if resolved == cwd_resolved else resolved.relative_to(cwd_resolved).as_posix()
            )
            rel_perms = _adapt_permissions_for_cwd(
                permissions,
                cwd_posix=cwd_posix,
                for_absolute=False,
            )
            rel_dec = evaluate_tool_call_permission(
                rel_perms,
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
    """Ensure both `params._meta.kiro.replay` and `params.update._meta.kiro.replay` are True."""
    params = dict(frame.get("params") or {})
    params_meta = dict(params.get("_meta") or {})
    params_kiro = dict(params_meta.get("kiro") or {})
    params_kiro["replay"] = True
    params_meta["kiro"] = params_kiro
    params["_meta"] = params_meta
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
    update_kiro = dict(update_meta.get("kiro") or {})
    update_kiro["ttft"] = {
        "promptReceivedAt": prompt_received_ms,
        "firstChunkEmittedAt": emitted_ms,
    }
    update_meta["kiro"] = update_kiro
    update["_meta"] = update_meta
    params["update"] = update
    return {**frame, "params": params}


def _to_wire_turn_telemetry(
    telemetry: TurnTelemetry,
    *,
    turn_id: str = "",
    context_breakdown: dict[str, Any] | None = None,
) -> WireTurnTelemetry:
    meta_without_sid = {k: v for k, v in telemetry.metadata_params.items() if k != "sessionId"}
    return WireTurnTelemetry(
        context_used=telemetry.context_tokens,
        context_size=telemetry.context_window,
        context_usage_percentage=telemetry.context_usage_pct,
        delta_credits=telemetry.delta_credits,
        metadata_payload=meta_without_sid,
        rate_limited_message=telemetry.rate_limit_message,
        context_breakdown=context_breakdown,
        turn_id=turn_id,
    )


def _require_non_empty_str(params: dict[str, Any], key: str) -> str | None:
    val = params.get(key)
    if isinstance(val, str) and bool(val.strip()):
        return val
    return None


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
        self._session_manager = session_manager or SessionManager(wire_adapter=resolved_adapter)
        self._auth_manager = auth_manager
        self._quota_tracker = quota_tracker or QuotaTracker(
            state_store=self._session_manager.state_store
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

    def _resolve_session_model(self, session_id: str) -> str:
        """Return the active model ID for `session_id`, falling back to the default model."""
        try:
            session_state = self._session_manager.get_session(session_id)
            return str(session_state.get("model", DEFAULT_MODEL_ID))
        except KeyError:
            return DEFAULT_MODEL_ID

    def _resolve_session_mode(self, session_id: str) -> str:
        """Return the active mode ID for `session_id`, falling back to GOOSE_MODE or `auto`."""
        try:
            session_state = self._session_manager.get_session(session_id)
            return str(session_state.get("mode", "auto"))
        except KeyError:
            return "approve" if os.environ.get("GOOSE_MODE") == "approve" else "auto"

    @staticmethod
    def _extract_prompt_text(prompt_blocks: list[dict[str, Any]]) -> str:
        """Extract concatenated text from ACP prompt content blocks."""
        parts = [
            str(block.get("text", ""))
            for block in prompt_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _extract_user_tail(prompt_text: str) -> str:
        if not prompt_text:
            return ""
        marker = "[CURRENT USER REQUEST"
        idx = prompt_text.rfind(marker)
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
        return False, ""

    @classmethod
    def _extract_slash_command(cls, prompt_text: str) -> str:
        """Extract a slash command (`/quota`, `/usage`, `/credits`, `/clear`, `/reset`)."""
        if not prompt_text:
            return ""
        tail = cls._extract_user_tail(prompt_text)
        first_line = tail.splitlines()[0].strip().lower() if tail else ""
        if first_line in SLASH_QUOTA_COMMANDS or first_line in SLASH_RESET_COMMANDS:
            return first_line
        last_line = prompt_text.splitlines()[-1].strip().lower()
        if last_line in SLASH_QUOTA_COMMANDS or last_line in SLASH_RESET_COMMANDS:
            return last_line
        return ""

    @staticmethod
    def _cancelled_prompt_response(message_id: Any) -> dict[str, Any]:
        """Build a JSON-RPC 2.0 `session/prompt` response with `stopReason == 'cancelled'`."""
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "stopReason": "cancelled",
                "usage": {
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "cachedReadTokens": 0,
                    "cachedWriteTokens": 0,
                },
            },
        }

    def _evaluate_tool_permission(
        self,
        *,
        session: Mapping[str, Any],
        spec: CustomAgentSpec | None,
        norm: NormalizedToolEvent,
        session_mode: str,
    ) -> PermissionDecision:
        """Evaluate tool permissions in strict fail-closed order across both v2 and v3."""
        agent_read_only = bool(session.get("agentReadOnly"))
        if (session_mode == "read-only" or agent_read_only) and norm.kind in {
            "execute",
            "edit",
            "delete",
        }:
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

        if perm_decision == "allow":
            return "allow"

        return "ask"

    async def _request_client_permission(
        self,
        session_id: str,
        perm_params: dict[str, Any],
    ) -> bool:
        """Prompt the ACP client for tool approval, failing closed when headless or rejected."""
        if self._client_rpc_caller is None:
            self._session_manager.cli_provider.abort_active_process(session_id)
            return False

        perm_response = await self._client_rpc_caller("session/request_permission", perm_params)
        outcome = perm_response.get("outcome") if isinstance(perm_response, dict) else None
        if isinstance(outcome, dict) and outcome.get("outcome") == "cancelled":
            await self._session_manager.cancel(session_id=session_id)
            return False
        selected_option = outcome.get("optionId") if isinstance(outcome, dict) else None
        allowed = selected_option in {"allow_once", "allow_always"}
        if not allowed:
            self._session_manager.cli_provider.abort_active_process(session_id)
        return allowed

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
    ) -> bool:
        session = self._session_manager.state_store.get_session(session_id) or {}
        known_mcp = self._session_manager.get_session_mcp_server_names(session_id)
        norm = _to_normalized_tool_event(event, known_mcp_servers=known_mcp)
        await self._emit_notification(self._wire_adapter.build_tool_call_update(session_id, norm))

        spec = self._session_manager.get_active_custom_agent(session_id)
        decision = self._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=norm,
            session_mode=self._resolve_session_mode(session_id),
        )
        if decision == "allow":
            return True
        if decision == "deny":
            self._session_manager.cli_provider.abort_active_process(session_id)
            return False

        perm_params = self._wire_adapter.build_permission_request_params(session_id, norm)
        return await self._request_client_permission(session_id, perm_params)

    async def _handle_v2_tool_call(
        self,
        session_id: str,
        event: ToolCallEvent,
    ) -> bool:
        session = self._session_manager.state_store.get_session(session_id) or {}
        known_mcp = self._session_manager.get_session_mcp_server_names(session_id)
        norm = _to_normalized_tool_event(event, known_mcp_servers=known_mcp)

        if norm.is_mcp:
            kiro_tool_name = norm.mcp_tool
            mcp_server_name = norm.mcp_server
            kiro_tool_id = f"@{norm.mcp_server}/{norm.mcp_tool}"
        else:
            kiro_tool_name = event.kiro_tool_name or (
                "execute_bash" if event.kind == "execute" else event.title
            )
            mcp_server_name = event.mcp_server_name
            kiro_tool_id = event.kiro_tool_id or (
                "run_command" if event.kind == "execute" else kiro_tool_name
            )
        tool_descriptor: dict[str, Any] = {
            "toolCallId": event.tool_call_id,
            "title": event.title,
            "kind": event.kind,
            "status": "in_progress",
            "rawInput": event.raw_input,
            "_meta": {
                "kiro": {
                    "toolName": kiro_tool_name,
                    "mcpServerName": mcp_server_name,
                }
            },
        }
        if event.locations:
            tool_descriptor["locations"] = event.locations
        if event.diff_content:
            tool_descriptor["content"] = event.diff_content

        await self._emit_notification(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "tool_call",
                        **tool_descriptor,
                    },
                },
            }
        )

        spec = self._session_manager.get_active_custom_agent(session_id)
        decision = self._evaluate_tool_permission(
            session=session,
            spec=spec,
            norm=norm,
            session_mode=self._resolve_session_mode(session_id),
        )
        if decision == "allow":
            return True
        if decision == "deny":
            self._session_manager.cli_provider.abort_active_process(session_id)
            return False

        perm_params = {
            "sessionId": session_id,
            "toolCall": tool_descriptor,
            "_meta": {"kiro": {"toolId": kiro_tool_id}},
            "options": self._wire_adapter.build_permission_options(),
        }
        return await self._request_client_permission(session_id, perm_params)

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
        model_id = self._resolve_session_model(session_id)
        telemetry = await self._emit_turn_telemetry_frames(
            session_id=session_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=0,
            context_tokens=compacted_ctx,
        )
        if self._wire_adapter.engine_id == "v2":
            post_params = dict(telemetry.metadata_params)
            post_params["meteringUsage"] = [
                {"unit": "credit", "unitPlural": "credits", "value": 0.0}
            ]
            self._queue_post_response_notifications(
                [
                    {
                        "jsonrpc": "2.0",
                        "method": "_kiro.dev/metadata",
                        "params": post_params,
                    }
                ]
            )
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "cachedReadTokens": 0,
                    "cachedWriteTokens": 0,
                },
            },
        }

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
            return self._cancelled_prompt_response(message_id)

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
                existing["agyConversationId"] = None
                existing["compactedSummary"] = None
                existing["contextTokens"] = 0
                existing["reinjectAgentPrompt"] = True
                self._session_manager.state_store.upsert_session(existing)
            await self._emit_notification(
                self._wire_adapter.build_agent_message_chunk(session_id, "Conversation cleared.")
            )
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "cachedReadTokens": 0,
                        "cachedWriteTokens": 0,
                    },
                },
            }

        if slash_cmd in SLASH_QUOTA_COMMANDS:
            thought_text = f"Querying quota report for {slash_cmd}..."
            response_text = await self._quota_tracker.format_slash_command_response(slash_cmd)
            await self._emit_notification(
                self._wire_adapter.build_agent_thought_chunk(session_id, thought_text)
            )
            await self._emit_notification(
                self._wire_adapter.build_agent_message_chunk(session_id, response_text)
            )
            input_tokens = max(1, len(prompt_text.split()))
            output_tokens = max(1, len(response_text.split()))
            model_id = self._resolve_session_model(session_id)
            await self._emit_turn_telemetry_frames(
                session_id=session_id,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_read_tokens=0,
                context_tokens=self._quota_tracker.get_session_context_tokens(session_id),
            )
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                        "cachedReadTokens": 0,
                        "cachedWriteTokens": 0,
                    },
                },
            }

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
        first_displayed_chunk = True
        collected_chunks: list[str] = []

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
            nonlocal first_displayed_chunk
            if (
                self._wire_adapter.engine_id == "v3"
                and want_ttft
                and first_displayed_chunk
                and chunk_text
            ):
                first_displayed_chunk = False
                emit_ms = max(prompt_received_ms, int(time.time() * 1000))
                return _stamp_ttft_meta(
                    frame,
                    prompt_received_ms=prompt_received_ms,
                    emitted_ms=emit_ms,
                )
            return frame

        carousel_normalizer = CarouselStreamNormalizer()

        async def _emit_message_piece(piece: str) -> None:
            if not piece:
                return
            collected_chunks.append(piece)
            frame = self._wire_adapter.build_agent_message_chunk(session_id, piece)
            await self._emit_notification(_maybe_attach_ttft(frame, piece))

        async def _on_thought(chunk: str) -> None:
            frame = self._wire_adapter.build_agent_thought_chunk(session_id, chunk)
            await self._emit_notification(_maybe_attach_ttft(frame, chunk))

        async def _on_text(chunk: str) -> None:
            await _emit_message_piece(carousel_normalizer.feed(chunk))

        async def _on_tool_call(event: ToolCallEvent) -> bool:
            await _emit_message_piece(carousel_normalizer.flush_prose_before_boundary())
            if self._wire_adapter.engine_id == "v3":
                return await self._handle_v3_tool_call(session_id, event)
            return await self._handle_v2_tool_call(session_id, event)

        async def _on_tool_update(event: ToolCallEvent) -> None:
            allowed = event.status != "failed"
            content_blocks = list(event.diff_content) or [
                {
                    "type": "content",
                    "content": {"type": "text", "text": event.output},
                }
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
            kiro_tool_name = (
                norm.mcp_tool
                if norm.is_mcp
                else (
                    event.kiro_tool_name
                    or ("execute_bash" if event.kind == "execute" else event.title)
                )
            )
            mcp_server_name = norm.mcp_server if norm.is_mcp else event.mcp_server_name
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
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
                    },
                }
            )

        async def _on_usage(ctx_tokens: int) -> None:
            notif = self._quota_tracker.build_usage_update_notification(
                session_id=session_id,
                context_tokens=ctx_tokens,
            )
            if self._wire_adapter.engine_id != "v3":
                await self._emit_notification(notif)

        async def _on_steer_consumed(raw_content: str) -> None:
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

        stop_reason = str(turn_result.get("stopReason", "end_turn"))
        if stop_reason == "cancelled" or self._session_manager.consume_cancellation(session_id):
            return self._cancelled_prompt_response(message_id)

        await _emit_message_piece(carousel_normalizer.flush())

        fallback_from = str(turn_result.get("fallbackFromModel") or "")
        resolved_to = str(turn_result.get("resolvedModel") or "")
        if fallback_from and resolved_to and fallback_from != resolved_to:
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
                prompt_blocks=prompt_blocks,
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
            total_chars = system_chars + history_chars
            context_breakdown = {
                "schema": 1,
                "totalChars": total_chars,
                "modelCalls": 1,
                "categories": {
                    "system": {"chars": system_chars},
                    "history": {"chars": history_chars},
                },
                **({"compacted": True} if was_compacted else {}),
            }

        model_id = self._resolve_session_model(session_id)
        await self._emit_turn_telemetry_frames(
            session_id=session_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
            context_tokens=context_tokens,
            turn_id=user_message_id,
            context_breakdown=context_breakdown,
        )

        if turn_error:
            return _jsonrpc_error(message_id, -32603, turn_error)

        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "stopReason": stop_reason,
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "cachedReadTokens": cached_read_tokens,
                    "cachedWriteTokens": 0,
                },
            },
        }

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
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "available_commands_update",
                        "availableCommands": list(AVAILABLE_SLASH_COMMANDS),
                    },
                },
            }
        )

    async def _emit_mcp_servers_initialized(self, session_id: str) -> None:
        """Emit `_kiro.dev/mcp/server_initialized` for every configured session MCP server."""
        for server_name in self._session_manager.get_session_mcp_server_names(session_id):
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/mcp/server_initialized",
                    "params": {
                        "sessionId": session_id,
                        "serverName": server_name,
                    },
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
        if self._wire_adapter.engine_id == "v3":
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
                if server_filter in ordered_names:
                    refreshed = await self._session_manager.probe_session_mcp_servers(
                        session_id,
                        server_filter=server_filter,
                    )
                    for item in refreshed:
                        cache[item.name] = item
                for peer_name in ordered_names:
                    if peer_name not in cache:
                        peer_statuses = await self._session_manager.probe_session_mcp_servers(
                            session_id,
                            server_filter=peer_name,
                        )
                        for item in peer_statuses:
                            cache[item.name] = item
                statuses = [cache[name] for name in ordered_names if name in cache]

            frames = self._wire_adapter.build_mcp_readiness_notifications(session_id, statuses)
            if post_response:
                self._queue_post_response_notifications(frames)
            else:
                for frame in frames:
                    await self._emit_notification(frame)
            return

        await self._emit_mcp_servers_initialized(session_id)

    async def _emit_session_replay(self, session_id: str) -> None:
        """Replay stored turns (`_meta.kiro.replay == True`) before `session/load` returns."""
        turns = self._session_manager.state_store.get_session_turns(session_id)
        for turn in turns:
            msg_id = str(turn.get("user_message_id") or turn.get("userMessageId") or "")
            blocks = turn.get("prompt_blocks")
            if blocks is None:
                blocks = turn.get("promptBlocks")
            block_list = blocks if isinstance(blocks, list) and blocks else []
            for block in block_list:
                if not isinstance(block, dict):
                    continue
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
        raw_mcp = params.get("mcpServers")
        if raw_mcp is not None and not isinstance(raw_mcp, list):
            return _jsonrpc_error(message_id, -32602, "Invalid params: 'mcpServers' must be a list")
        mcp_servers: list[dict[str, Any]] | None = (
            [s for s in raw_mcp if isinstance(s, dict)] if isinstance(raw_mcp, list) else None
        )
        raw_meta = params.get("_meta")
        if raw_meta is not None and not isinstance(raw_meta, dict):
            return _jsonrpc_error(message_id, -32602, "Invalid params: '_meta' must be an object")
        meta: dict[str, Any] | None = dict(raw_meta) if isinstance(raw_meta, dict) else None
        desc = await self._session_manager.create_session(
            cwd=cwd,
            mcp_servers=mcp_servers,
            meta=meta,
        )
        sid = desc["sessionId"]
        if self._wire_adapter.engine_id == "v2":
            await self._emit_available_commands_update(sid)
        await self._emit_or_queue_mcp_readiness(
            sid,
            post_response=(self._wire_adapter.engine_id == "v3"),
        )
        persisted_ctx = self._quota_tracker.get_session_context_tokens(sid)
        if persisted_ctx is not None and persisted_ctx > 0:
            await self._emit_persisted_usage_update(sid, persisted_ctx)
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": desc,
        }

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
        raw_cwd = params.get("cwd")
        if require_cwd:
            cwd = _require_non_empty_str(params, "cwd")
            if cwd is None:
                return _jsonrpc_error(message_id, -32602, "Invalid params: 'cwd' is required")
        elif raw_cwd is None:
            cwd = ""
        else:
            cwd = _require_non_empty_str(params, "cwd") or ""
            if not cwd:
                return _jsonrpc_error(message_id, -32602, "Invalid params: 'cwd' must be non-empty")

        raw_mcp = params.get("mcpServers")
        if raw_mcp is not None and not isinstance(raw_mcp, list):
            return _jsonrpc_error(message_id, -32602, "Invalid params: 'mcpServers' must be a list")
        mcp_servers: list[dict[str, Any]] | None = (
            [s for s in raw_mcp if isinstance(s, dict)] if isinstance(raw_mcp, list) else None
        )
        raw_meta = params.get("_meta")
        if raw_meta is not None and not isinstance(raw_meta, dict):
            return _jsonrpc_error(message_id, -32602, "Invalid params: '_meta' must be an object")
        meta: dict[str, Any] | None = dict(raw_meta) if isinstance(raw_meta, dict) else None
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
        if self._wire_adapter.engine_id == "v2":
            await self._emit_available_commands_update(sid)
        await self._emit_or_queue_mcp_readiness(
            sid,
            post_response=(self._wire_adapter.engine_id == "v3"),
        )
        persisted_ctx = self._quota_tracker.get_session_context_tokens(sid)
        if persisted_ctx is not None and persisted_ctx > 0:
            await self._emit_persisted_usage_update(sid, persisted_ctx)
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": desc,
        }

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
            for frame in self._wire_adapter.build_mode_switched_notifications(
                session_id,
                mode_id,
            ):
                await self._emit_notification(frame)
            await self._emit_or_queue_mcp_readiness(session_id, post_response=True)
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": res,
            }

        session_state = self._session_manager.get_session(session_id)
        is_custom_agent = bool(session_state.get("agent"))
        if is_custom_agent:
            switched_params: dict[str, Any] = {
                "sessionId": session_id,
                "agentName": mode_id,
            }
            switched_model = session_state.get("switchedModel")
            if isinstance(switched_model, str) and switched_model:
                switched_params["model"] = switched_model
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/agent/switched",
                    "params": switched_params,
                }
            )
        await self._emit_notification(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "current_mode_update",
                        "currentModeId": mode_id,
                    },
                },
            }
        )
        if is_custom_agent:
            await self._emit_mcp_servers_initialized(session_id)
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": res,
        }

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
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": res,
        }

    async def _handle_kas_session_compact(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/session/compact` extension RPC."""
        session_id = _require_non_empty_str(params, "sessionId")
        if session_id is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        raw_instructions = params.get("instructions")
        if raw_instructions is not None and not isinstance(raw_instructions, str):
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        instructions = raw_instructions or ""
        try:
            self._session_manager.get_session(session_id)
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))

        await self._emit_notification(
            self._wire_adapter.build_compaction_notification(session_id, "started")
        )
        try:
            compact_res = await self._session_manager.compact_session(
                session_id=session_id,
                instructions=instructions,
            )
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))

        status_type = str(compact_res.get("status", "completed"))
        summary_text = str(compact_res.get("summary", ""))
        if status_type == "failed":
            await self._emit_notification(
                self._wire_adapter.build_compaction_notification(
                    session_id,
                    "failed",
                    error=summary_text or "Compaction failed",
                )
            )
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {"success": False},
            }

        await self._emit_notification(
            self._wire_adapter.build_compaction_notification(
                session_id,
                "completed",
                summary=summary_text,
            )
        )
        compacted_ctx = int(compact_res.get("contextTokens", 0))
        input_tokens = max(1, int(compact_res.get("inputTokens", 1)))
        output_tokens = max(1, int(compact_res.get("outputTokens", 1)))
        model_id = self._resolve_session_model(session_id)
        telemetry = await self._quota_tracker.build_turn_telemetry(
            session_id=session_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=0,
            context_tokens=compacted_ctx,
        )
        await self._emit_notification(telemetry.usage_update_notification)
        if self._wire_adapter.engine_id == "v3":
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "session_info_update",
                            "_meta": {
                                "kiro": {
                                    "kind": "context_usage",
                                    "usagePercentage": telemetry.context_usage_pct,
                                }
                            },
                        },
                    },
                }
            )
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {"success": True},
        }

    async def _handle_kas_mcp_reset_server(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `_kiro/mcp/resetServer` by re-probing target server and emitting full roster."""
        session_id = _require_non_empty_str(params, "sessionId")
        server_name = _require_non_empty_str(params, "serverName")
        raw_start_oauth = params.get("startOAuth")
        if session_id is None or server_name is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        if raw_start_oauth is not None and not isinstance(raw_start_oauth, bool):
            return _jsonrpc_error(message_id, -32602, "Invalid params")
        try:
            self._session_manager.get_session(session_id)
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc))

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
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {},
        }

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

        raw_sid = params.get("sessionId")
        session_id = ""
        if raw_sid is not None:
            validated_sid = _require_non_empty_str(params, "sessionId")
            if validated_sid is None:
                return _jsonrpc_error(message_id, -32602, "Invalid params")
            try:
                self._session_manager.get_session(validated_sid)
            except KeyError as exc:
                return _jsonrpc_error(message_id, -32002, str(exc))
            session_id = validated_sid

        raw_cmd = params.get("command")
        cmd_name = ""
        cmd_args: dict[str, Any] = {}
        if isinstance(raw_cmd, dict):
            cmd_name = str(raw_cmd.get("command", "")).strip().lstrip("/").lower()
            raw_args = raw_cmd.get("args")
            if isinstance(raw_args, dict):
                cmd_args = dict(raw_args)
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
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "message": md,
                    "text": md,
                    "stopReason": "end_turn",
                },
            }
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
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "message": msg,
                    "text": msg,
                    "stopReason": "end_turn",
                },
            }
        if cmd_name == "mcp":
            server_names = (
                self._session_manager.get_session_mcp_server_names(session_id) if session_id else []
            )
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "message": "MCP servers",
                    "data": {
                        "servers": [
                            {"name": name, "status": "running", "toolCount": 1}
                            for name in server_names
                        ]
                    },
                    "stopReason": "end_turn",
                },
            }
        if cmd_name == "tools":
            server_names = (
                self._session_manager.get_session_mcp_server_names(session_id) if session_id else []
            )
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "message": "Tools",
                    "data": {
                        "tools": [
                            {"name": f"@{name}", "source": f"mcp:{name}"} for name in server_names
                        ]
                    },
                    "stopReason": "end_turn",
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "message": "",
                "text": "",
                "stopReason": "end_turn",
            },
        }

    async def _handle_session_steer(
        self,
        method: str,
        message_id: Any,
        params: dict[str, Any],
        *,
        is_request: bool = True,
    ) -> dict[str, Any] | None:
        """Handle `_session/steer` and `_session/steering` during an active turn."""
        session_id = _require_non_empty_str(params, "sessionId")
        if session_id is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params") if is_request else None

        if method == "_session/steering":
            raw_prompt = params.get("prompt")
            if not isinstance(raw_prompt, list) or not raw_prompt:
                return _jsonrpc_error(message_id, -32602, "Invalid params") if is_request else None
            try:
                self._session_manager.get_session(session_id)
            except KeyError as exc:
                return _jsonrpc_error(message_id, -32002, str(exc)) if is_request else None
            prompt_blocks: list[dict[str, Any]] = [b for b in raw_prompt if isinstance(b, dict)]
            queued = await self._session_manager.steer(
                session_id=session_id,
                prompt_blocks=prompt_blocks,
            )
            if not is_request:
                return None
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {"outcome": "injected" if queued else "failed"},
            }

        raw_message = _require_non_empty_str(params, "message")
        if raw_message is None:
            return _jsonrpc_error(message_id, -32602, "Invalid params") if is_request else None
        try:
            self._session_manager.get_session(session_id)
        except KeyError as exc:
            return _jsonrpc_error(message_id, -32002, str(exc)) if is_request else None

        queued = await self._session_manager.steer(
            session_id=session_id,
            message=raw_message,
        )
        if queued:
            await self._emit_notification(
                self._wire_adapter.build_steering_notification(
                    session_id,
                    "queued",
                    raw_message,
                )
            )
        if not is_request:
            return None
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {"queued": queued},
        }

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
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "config_option_update",
                        "configOptions": res["configOptions"],
                    },
                },
            }
        )
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": res,
        }

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
        log_acp_event(
            "rpc.receive",
            save_dir_or_db=db_path,
            id=msg_id,
            method=method,
            params=raw_params,
            raw_message=parsed,
        )
        if not isinstance(method, str) or not method.strip():
            return _jsonrpc_error(msg_id if has_id else None, -32600, "Invalid Request")

        if raw_params is not None and not isinstance(raw_params, dict):
            return _jsonrpc_error(msg_id, -32602, "Invalid params") if has_id else None

        response = await self._dispatch_method(
            method,
            msg_id,
            raw_params,
            parsed,
            is_request=has_id,
        )
        if response is not None:
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
            if not (isinstance(pv, int) and not isinstance(pv, bool) and pv >= 1):
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
        caps = dict(init_res["agentCapabilities"])
        caps["sessionCapabilities"] = {
            "close": {},
            "fork": {},
            "list": {},
            "resume": {},
        }
        info = dict(init_res["agentInfo"])
        info["_meta"] = {"agyAcpVersion": __version__}
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": init_res["protocolVersion"],
                "agentCapabilities": caps,
                "agentInfo": info,
                "authMethods": [],
            },
        }

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
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
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
            }
        if method == "session/set_mode":
            return await self._handle_session_set_mode(msg_id, params)
        if method == "session/set_model":
            return await self._handle_session_set_model(msg_id, params)
        if method == "session/set_config_option":
            return await self._handle_session_set_config_option(msg_id, params)
        if method == "_kiro/session/compact":
            return await self._handle_kas_session_compact(msg_id, params)
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
                try:
                    self._session_manager.get_session(session_id)
                except KeyError as exc:
                    return _jsonrpc_error(msg_id, -32002, str(exc))
            await self._session_manager.cancel(session_id=session_id)
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}} if is_request else None
        if method in {"_session/steer", "_session/steering"}:
            return await self._handle_session_steer(
                method,
                msg_id,
                params,
                is_request=is_request,
            )
        if method == "session/close":
            session_id = _require_non_empty_str(params, "sessionId")
            if session_id is None:
                return _jsonrpc_error(msg_id, -32602, "Invalid params")
            try:
                self._session_manager.get_session(session_id)
            except KeyError as exc:
                return _jsonrpc_error(msg_id, -32002, str(exc))
            res = await self._session_manager.evict_session(session_id=session_id)
            self._mcp_status_cache.pop(session_id, None)
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}
        if method in {"_kiro.dev/session/terminate", "_kiro/session/delete"}:
            session_id = _require_non_empty_str(params, "sessionId")
            if session_id is None:
                return _jsonrpc_error(msg_id, -32602, "Invalid params")
            try:
                self._session_manager.get_session(session_id)
            except KeyError as exc:
                return _jsonrpc_error(msg_id, -32002, str(exc))
            is_delete = method == "_kiro/session/delete"
            res = await self._session_manager.terminate_session(
                session_id=session_id,
                delete_persisted=is_delete,
            )
            self._mcp_status_cache.pop(session_id, None)
            if is_delete:
                self._quota_tracker.evict_session(session_id)
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}

        log_acp_event(
            "rpc.unhandled",
            save_dir_or_db=self._session_manager.state_store.db_path,
            id=msg_id,
            method=method,
            params=raw_params,
            raw_message=message,
        )
        if is_request:
            return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")
        return None
