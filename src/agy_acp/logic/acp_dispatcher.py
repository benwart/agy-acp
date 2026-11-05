"""ACP JSON-RPC 2.0 request/notification dispatcher and framing engine."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from agy_acp import KIRO_CLI_COMPAT_VERSION, __version__
from agy_acp.logic.auth_manager import AuthManager
from agy_acp.logic.custom_agents import CustomAgentSpec, PermissionDecision
from agy_acp.logic.quota_tracker import QuotaTracker, TurnTelemetry
from agy_acp.logic.session_manager import DEFAULT_MODEL_ID, VALID_MODE_IDS, SessionManager
from agy_acp.logic.wire import (
    EngineWireAdapter,
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

NotificationSink = Callable[[dict[str, Any]], Awaitable[None]]
ClientRpcCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _coerce_tool_kind(raw_kind: str) -> ToolCallKind:
    match raw_kind:
        case "execute" | "read" | "edit" | "delete" | "search" | "fetch" | "other":
            return raw_kind
        case _:
            return "other"


def _to_normalized_tool_event(event: ToolCallEvent) -> NormalizedToolEvent:
    """Convert an `AgyCliProvider` `ToolCallEvent` into a wire `NormalizedToolEvent`."""
    is_mcp = bool(event.mcp_server)
    kas_id = event.kas_builtin_id or (
        "run_command" if event.kind == "execute" else (event.kiro_tool_id or "other")
    )
    v2_name = event.v2_tool_name or (
        "execute_bash" if event.kind == "execute" else (event.kiro_tool_name or event.title)
    )
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
        mcp_server=event.mcp_server,
        mcp_tool=event.mcp_tool,
    )


def _matches_agent_tool_list(patterns: tuple[str, ...], norm: NormalizedToolEvent) -> bool:
    """Return True if `norm` matches an entry in `allowedTools` or `excludedTools`."""
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
                norm.mcp_tool,
            }
        )
    else:
        candidates.update({norm.kas_tool_id, norm.kiro_v2_tool_name, "builtin"})
    return any(p in candidates for p in patterns)


def _extract_primary_path(norm: NormalizedToolEvent) -> str:
    if norm.locations:
        loc_path = norm.locations[0].get("path")
        if isinstance(loc_path, str) and loc_path:
            return loc_path
    for key in ("Path", "path", "TargetFile", "AbsolutePath", "DirectoryPath", "file_path"):
        val = norm.raw_input.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _stamp_replay_meta(frame: dict[str, Any]) -> dict[str, Any]:
    """Ensure both `params._meta.kiro.replay` and `params.update._meta.kiro.replay` are True."""
    params = dict(frame.get("params") or {})
    params_meta = dict(params.get("_meta") or {})
    params_kiro = dict(params_meta.get("kiro") or {})
    params_kiro["replay"] = True
    params_meta["kiro"] = params_kiro
    params["_meta"] = params_meta
    return {**frame, "params": params}


def _to_wire_turn_telemetry(telemetry: TurnTelemetry, *, turn_id: str = "") -> WireTurnTelemetry:
    meta_without_sid = {k: v for k, v in telemetry.metadata_params.items() if k != "sessionId"}
    return WireTurnTelemetry(
        context_used=telemetry.context_tokens,
        context_size=telemetry.context_window,
        context_usage_percentage=telemetry.context_usage_pct,
        delta_credits=telemetry.delta_credits,
        metadata_payload=meta_without_sid,
        rate_limited_message=telemetry.rate_limit_message,
        turn_id=turn_id,
    )


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
        self._post_response_notifications: list[dict[str, Any]] = []

    @property
    def wire_adapter(self) -> EngineWireAdapter:
        """Return the active `EngineWireAdapter` strategy."""
        return self._wire_adapter

    def pop_post_response_notifications(self) -> list[dict[str, Any]]:
        """Drain notifications queued to flush immediately after a JSON-RPC response frame."""
        queued = list(self._post_response_notifications)
        self._post_response_notifications.clear()
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

    def _evaluate_custom_agent_permission(
        self,
        spec: CustomAgentSpec | None,
        norm: NormalizedToolEvent,
        *,
        session_mode: str,
    ) -> PermissionDecision:
        """Evaluate customAgent tool allow/exclude/KAS rules before prompting client."""
        if spec is not None:
            if _matches_agent_tool_list(spec.excluded_tools, norm):
                return "deny"
            if _matches_agent_tool_list(spec.allowed_tools, norm):
                return "allow"
            if spec.permissions is not None:
                url = str(norm.raw_input.get("url") or norm.raw_input.get("Url") or "")
                query = str(norm.raw_input.get("query") or norm.raw_input.get("Query") or "")
                return spec.evaluate_tool_call(
                    tool_id=norm.kas_tool_id,
                    is_mcp=norm.is_mcp,
                    server_name=norm.mcp_server,
                    tool_name=norm.mcp_tool,
                    command=norm.command,
                    path=_extract_primary_path(norm),
                    url=url,
                    query=query,
                    subagent_target=norm.subagent_target,
                )
        if session_mode == "read-only" and norm.kind in {"execute", "edit", "delete"}:
            return "deny"
        return "ask"

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
            wire_tel = _to_wire_turn_telemetry(telemetry, turn_id=turn_id)
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
        norm = _to_normalized_tool_event(event)
        await self._emit_notification(self._wire_adapter.build_tool_call_update(session_id, norm))

        spec = self._session_manager.get_active_custom_agent(session_id)
        decision = self._evaluate_custom_agent_permission(
            spec,
            norm,
            session_mode=self._resolve_session_mode(session_id),
        )
        if decision == "allow":
            return True
        if decision == "deny":
            self._session_manager.cli_provider.abort_active_process(session_id)
            return False
        if self._client_rpc_caller is None:
            return True

        perm_params = self._wire_adapter.build_permission_request_params(session_id, norm)
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

    async def _handle_v2_tool_call(
        self,
        session_id: str,
        event: ToolCallEvent,
    ) -> bool:
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
        if self._client_rpc_caller is None:
            return True

        perm_response = await self._client_rpc_caller(
            "session/request_permission",
            {
                "sessionId": session_id,
                "toolCall": tool_descriptor,
                "_meta": {"kiro": {"toolId": kiro_tool_id}},
                "options": self._wire_adapter.build_permission_options(),
            },
        )
        outcome = perm_response.get("outcome") if isinstance(perm_response, dict) else None
        if isinstance(outcome, dict) and outcome.get("outcome") == "cancelled":
            await self._session_manager.cancel(session_id=session_id)
            return False
        selected_option = outcome.get("optionId") if isinstance(outcome, dict) else None
        allowed = selected_option in {"allow_once", "allow_always"}
        if not allowed:
            self._session_manager.cli_provider.abort_active_process(session_id)
        return allowed

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
            self._post_response_notifications.append(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/metadata",
                    "params": post_params,
                }
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
        if self._auth_manager is not None and not self._auth_manager.status().authenticated:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32001,
                    "message": "Not logged in. Run `agy-acp auth login` to authenticate.",
                },
            }

        session_id = str(params.get("sessionId", ""))
        if self._session_manager.consume_cancellation(session_id):
            return self._cancelled_prompt_response(message_id)

        raw_prompt = params.get("prompt")
        prompt_blocks: list[dict[str, Any]] = (
            [b for b in raw_prompt if isinstance(b, dict)] if isinstance(raw_prompt, list) else []
        )
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

        user_message_id = f"msg_{uuid.uuid4().hex[:12]}"
        prompt_received_ms = int(time.monotonic() * 1000)
        first_text_chunk = True
        collected_chunks: list[str] = []

        for start_frame in self._wire_adapter.build_turn_start_notifications(
            session_id,
            user_message_id,
        ):
            await self._emit_notification(start_frame)

        async def _on_thought(chunk: str) -> None:
            await self._emit_notification(
                self._wire_adapter.build_agent_thought_chunk(session_id, chunk)
            )

        async def _on_text(chunk: str) -> None:
            nonlocal first_text_chunk
            collected_chunks.append(chunk)
            rx_ms: int | None = None
            emit_ms: int | None = None
            if first_text_chunk:
                first_text_chunk = False
                rx_ms = prompt_received_ms
                emit_ms = max(prompt_received_ms, int(time.monotonic() * 1000))
            await self._emit_notification(
                self._wire_adapter.build_agent_message_chunk(
                    session_id,
                    chunk,
                    prompt_received_at_ms=rx_ms,
                    first_chunk_emitted_at_ms=emit_ms,
                )
            )

        async def _on_tool_call(event: ToolCallEvent) -> bool:
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

            kiro_tool_name = event.kiro_tool_name or (
                "execute_bash" if event.kind == "execute" else event.title
            )
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
                                    "mcpServerName": event.mcp_server_name,
                                }
                            },
                        },
                    },
                }
            )

        async def _on_usage(ctx_tokens: int) -> None:
            await self._emit_notification(
                self._quota_tracker.build_usage_update_notification(
                    session_id=session_id,
                    context_tokens=ctx_tokens,
                )
            )

        async def _on_steer_consumed(raw_content: str) -> None:
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

        stop_reason = str(turn_result.get("stopReason", "end_turn"))
        if stop_reason == "cancelled" or self._session_manager.consume_cancellation(session_id):
            return self._cancelled_prompt_response(message_id)

        assistant_text = str(turn_result.get("response") or "".join(collected_chunks))
        if self._session_manager.state_store.get_session(session_id) is not None:
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
        )

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
        """Emit MCP readiness frames (`_kiro/mcp/status` + `didChange` on v3; `v2` init on v2)."""
        if self._wire_adapter.engine_id == "v3":
            statuses = await self._session_manager.probe_session_mcp_servers(
                session_id,
                server_filter=server_filter,
            )
            frames = self._wire_adapter.build_mcp_readiness_notifications(session_id, statuses)
            if post_response:
                self._post_response_notifications.extend(frames)
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
            assistant_text = str(turn.get("assistant_text") or turn.get("assistantText") or "")
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
        cwd = str(params.get("cwd", ""))
        raw_mcp = params.get("mcpServers")
        mcp_servers: list[dict[str, Any]] | None = (
            [s for s in raw_mcp if isinstance(s, dict)] if isinstance(raw_mcp, list) else None
        )
        raw_meta = params.get("_meta")
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
            await self._emit_notification(
                self._quota_tracker.build_usage_update_notification(
                    session_id=sid,
                    context_tokens=persisted_ctx,
                )
            )
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": desc,
        }

    async def _handle_session_load(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/load`, replaying turns BEFORE result and queueing MCP readiness after."""
        session_id = str(params.get("sessionId", ""))
        cwd = str(params.get("cwd", ""))
        raw_mcp = params.get("mcpServers")
        mcp_servers: list[dict[str, Any]] | None = (
            [s for s in raw_mcp if isinstance(s, dict)] if isinstance(raw_mcp, list) else None
        )
        raw_meta = params.get("_meta")
        meta: dict[str, Any] | None = dict(raw_meta) if isinstance(raw_meta, dict) else None
        try:
            desc = await self._session_manager.load_session(
                session_id=session_id,
                cwd=cwd,
                mcp_servers=mcp_servers,
                meta=meta,
            )
        except KeyError as exc:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32002,
                    "message": str(exc),
                },
            }

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
            await self._emit_notification(
                self._quota_tracker.build_usage_update_notification(
                    session_id=sid,
                    context_tokens=persisted_ctx,
                )
            )
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
        session_id = str(params.get("sessionId", ""))
        mode_id = str(params.get("modeId", ""))
        try:
            res = await self._session_manager.set_mode(
                session_id=session_id,
                mode_id=mode_id,
            )
        except ValueError as exc:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(exc),
                },
            }

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

        if mode_id not in VALID_MODE_IDS:
            session_state = self._session_manager.get_session(session_id)
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
        if mode_id not in VALID_MODE_IDS:
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
        session_id = str(params.get("sessionId", ""))
        model_id = str(params.get("modelId", ""))
        res = await self._session_manager.set_model(
            session_id=session_id,
            model_id=model_id,
        )
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
        session_id = str(params.get("sessionId", ""))
        instructions = str(params.get("instructions") or "")
        await self._emit_notification(
            self._wire_adapter.build_compaction_notification(session_id, "started")
        )
        compact_res = await self._session_manager.compact_session(
            session_id=session_id,
            instructions=instructions,
        )
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
        """Handle `_kiro/mcp/resetServer` extension RPC by re-probing the target MCP server."""
        session_id = str(params.get("sessionId", ""))
        server_name = str(params.get("serverName") or "").strip() or None
        await self._emit_or_queue_mcp_readiness(
            session_id,
            post_response=False,
            server_filter=server_name,
        )
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
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32603,
                    "message": "Method '_kiro.dev/commands/execute' is not supported on KAS v3",
                },
            }

        session_id = str(params.get("sessionId", ""))
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

        if cmd_name in {"quota", "usage", "credits"}:
            md = await self._quota_tracker.format_slash_command_response(f"/{cmd_name}")
            if session_id:
                persisted_ctx = self._quota_tracker.get_session_context_tokens(session_id)
                if persisted_ctx is not None and persisted_ctx > 0:
                    await self._emit_notification(
                        self._quota_tracker.build_usage_update_notification(
                            session_id=session_id,
                            context_tokens=persisted_ctx,
                        )
                    )
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
    ) -> dict[str, Any] | None:
        """Handle `_session/steer` and `_session/steering` during an active turn."""
        session_id = str(params.get("sessionId", ""))
        if method == "_session/steering":
            raw_prompt = params.get("prompt")
            prompt_blocks: list[dict[str, Any]] = (
                [b for b in raw_prompt if isinstance(b, dict)]
                if isinstance(raw_prompt, list)
                else []
            )
            queued = await self._session_manager.steer(
                session_id=session_id,
                prompt_blocks=prompt_blocks,
            )
            if message_id is None:
                return None
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {"outcome": "injected" if queued else "failed"},
            }

        raw_message = str(params.get("message", ""))
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
        if message_id is None:
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
        session_id = str(params.get("sessionId", ""))
        config_id = str(params.get("configId", ""))
        value = str(params.get("value", ""))
        try:
            res = await self._session_manager.set_config_option(
                session_id=session_id,
                config_id=config_id,
                value=value,
            )
        except ValueError as exc:
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(exc),
                },
            }
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
        await anyio.sleep(0)
        message: dict[str, Any] = json.loads(raw_line)
        method = message.get("method")
        msg_id = message.get("id")
        raw_params = message.get("params")
        db_path = self._session_manager.state_store.db_path
        log_acp_event(
            "rpc.receive",
            save_dir_or_db=db_path,
            id=msg_id,
            method=method,
            params=raw_params,
            raw_message=message,
        )
        response = await self._dispatch_method(method, msg_id, raw_params, message)
        if response is not None:
            log_acp_event(
                "rpc.response",
                save_dir_or_db=db_path,
                id=msg_id,
                method=method,
                response=response,
            )
        return response

    def _build_initialize_response(self, msg_id: Any) -> dict[str, Any]:
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
        method: Any,
        msg_id: Any,
        raw_params: Any,
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Route a parsed JSON-RPC 2.0 method to its handler."""
        params = raw_params if isinstance(raw_params, dict) else {}
        if method == "initialize":
            return self._build_initialize_response(msg_id)
        if method == "session/new":
            return await self._handle_session_new(msg_id, params)
        if method in {"session/load", "session/resume"}:
            return await self._handle_session_load(msg_id, params)
        if method == "session/list":
            cwd_filter = params.get("cwd")
            sessions = self._session_manager.list_sessions(
                cwd=str(cwd_filter) if cwd_filter else None
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
            await self._session_manager.cancel(session_id=str(params.get("sessionId", "")))
            return None
        if method in {"_session/steer", "_session/steering"}:
            return await self._handle_session_steer(method, msg_id, params)
        if method == "session/close":
            res = await self._session_manager.evict_session(
                session_id=str(params.get("sessionId", ""))
            )
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}
        if method in {"_kiro.dev/session/terminate", "_kiro/session/delete"}:
            res = await self._session_manager.terminate_session(
                session_id=str(params.get("sessionId", "")),
                delete_persisted=(method == "_kiro/session/delete"),
            )
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}
        log_acp_event(
            "rpc.unhandled",
            save_dir_or_db=self._session_manager.state_store.db_path,
            id=msg_id,
            method=method,
            params=raw_params,
            raw_message=message,
        )
        return None
