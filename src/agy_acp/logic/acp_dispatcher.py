"""ACP JSON-RPC 2.0 request/notification dispatcher and framing engine."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from agy_acp import __version__
from agy_acp.logic.auth_manager import AuthManager
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.logic.session_manager import DEFAULT_MODEL_ID, SessionManager
from agy_acp.providers.agy_cli_provider import ToolCallEvent

MAX_FRAME_SIZE_BYTES = 16 * 1024 * 1024
SLASH_QUOTA_COMMANDS = frozenset({"/quota", "/usage", "/credits"})
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
]

NotificationSink = Callable[[dict[str, Any]], Awaitable[None]]
ClientRpcCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


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
    ) -> None:
        self._notification_sink = notification_sink
        self._client_rpc_caller = client_rpc_caller
        self._session_manager = session_manager or SessionManager()
        self._auth_manager = auth_manager
        self._quota_tracker = quota_tracker or QuotaTracker(
            state_store=self._session_manager.state_store
        )

    async def _emit_notification(self, notification: dict[str, Any]) -> None:
        """Emit a JSON-RPC 2.0 notification to the configured sink if present."""
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
        slash_cmd = prompt_text.splitlines()[-1].strip().lower() if prompt_text else ""

        if slash_cmd in SLASH_QUOTA_COMMANDS:
            thought_text = f"Querying quota report for {slash_cmd}..."
            response_text = await self._quota_tracker.format_slash_command_response(slash_cmd)
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_thought_chunk",
                            "content": {"type": "text", "text": thought_text},
                        },
                    },
                }
            )
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": response_text},
                        },
                    },
                }
            )
            input_tokens = max(1, len(prompt_text.split()))
            output_tokens = max(1, len(response_text.split()))
            model_id = self._resolve_session_model(session_id)
            for wire_update in await self._quota_tracker.build_kirocrew_wire_updates(
                session_id=session_id,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_read_tokens=0,
            ):
                await self._emit_notification(wire_update)
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

        async def _on_thought(chunk: str) -> None:
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_thought_chunk",
                            "content": {"type": "text", "text": chunk},
                        },
                    },
                }
            )

        async def _on_text(chunk: str) -> None:
            await self._emit_notification(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": chunk},
                        },
                    },
                }
            )

        async def _on_tool_call(event: ToolCallEvent) -> bool:
            tool_descriptor: dict[str, Any] = {
                "toolCallId": event.tool_call_id,
                "title": event.title,
                "kind": event.kind,
                "status": "in_progress",
                "rawInput": event.raw_input,
            }
            if event.locations:
                tool_descriptor["locations"] = event.locations

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
                    "options": [
                        {
                            "optionId": "allow_once",
                            "name": "Allow once",
                            "kind": "allow_once",
                        },
                        {
                            "optionId": "allow_always",
                            "name": "Always allow",
                            "kind": "allow_always",
                        },
                        {
                            "optionId": "reject_once",
                            "name": "Reject",
                            "kind": "reject_once",
                        },
                    ],
                },
            )
            outcome = perm_response.get("outcome") if isinstance(perm_response, dict) else None
            selected_option = outcome.get("optionId") if isinstance(outcome, dict) else None
            return selected_option in {"allow_once", "allow_always"}

        async def _on_tool_update(event: ToolCallEvent) -> None:
            allowed = event.status != "failed"
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
                            "content": [
                                {
                                    "type": "content",
                                    "content": {"type": "text", "text": event.output},
                                }
                            ],
                            "rawOutput": {"output": event.output, "allowed": allowed},
                        },
                    },
                }
            )

        turn_result = await self._session_manager.prompt(
            session_id=session_id,
            prompt_blocks=prompt_blocks,
            on_thought=_on_thought,
            on_text=_on_text,
            on_tool_call=_on_tool_call,
            on_tool_update=_on_tool_update,
        )

        stop_reason = str(turn_result.get("stopReason", "end_turn"))
        if stop_reason == "cancelled" or self._session_manager.consume_cancellation(session_id):
            return self._cancelled_prompt_response(message_id)

        raw_usage = turn_result.get("usage")
        usage_dict = raw_usage if isinstance(raw_usage, dict) else {}
        input_tokens = max(1, int(usage_dict.get("input_tokens", 0)))
        output_tokens = max(1, int(usage_dict.get("output_tokens", 0)))
        cached_read_tokens = int(usage_dict.get("cache_read_tokens", 0))
        thinking_tokens = int(usage_dict.get("thinking_tokens", 0))

        model_id = self._resolve_session_model(session_id)
        for wire_update in await self._quota_tracker.build_kirocrew_wire_updates(
            session_id=session_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
        ):
            await self._emit_notification(wire_update)

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

    async def _handle_session_new(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/new`, creating a session and emitting available slash commands."""
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
        await self._emit_available_commands_update(desc["sessionId"])
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
        """Handle `session/load`, restoring a session and emitting available slash commands."""
        session_id = str(params.get("sessionId", ""))
        cwd = str(params.get("cwd", ""))
        raw_mcp = params.get("mcpServers")
        mcp_servers: list[dict[str, Any]] | None = (
            [s for s in raw_mcp if isinstance(s, dict)] if isinstance(raw_mcp, list) else None
        )
        try:
            desc = await self._session_manager.load_session(
                session_id=session_id,
                cwd=cwd,
                mcp_servers=mcp_servers,
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
        await self._emit_available_commands_update(desc["sessionId"])
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
        """Handle `session/set_mode`, updating session mode and emitting `current_mode_update`."""
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

    async def _handle_session_set_config_option(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/set_config_option`, updating a session configuration option."""
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
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": True,
                        "mcpCapabilities": {"http": True, "sse": False},
                        "promptCapabilities": {"image": True, "embeddedContext": True},
                        "sessionCapabilities": {
                            "close": {},
                            "fork": {},
                            "list": {},
                            "resume": {},
                        },
                    },
                    "agentInfo": {
                        "name": "agy-acp",
                        "title": "Google Antigravity ACP Harness",
                        "version": __version__,
                    },
                    "authMethods": [],
                },
            }
        if method == "session/new":
            params = message.get("params")
            return await self._handle_session_new(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        if method in {"session/load", "session/resume"}:
            params = message.get("params")
            return await self._handle_session_load(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        if method == "session/list":
            raw_params = message.get("params")
            params = raw_params if isinstance(raw_params, dict) else {}
            cwd_filter = params.get("cwd")
            sessions = self._session_manager.list_sessions(
                cwd=str(cwd_filter) if cwd_filter else None
            )
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
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
            params = message.get("params")
            return await self._handle_session_set_mode(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        if method == "session/set_model":
            params = message.get("params")
            return await self._handle_session_set_model(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        if method == "session/set_config_option":
            params = message.get("params")
            return await self._handle_session_set_config_option(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        if method == "session/prompt":
            params = message.get("params")
            return await self._handle_session_prompt(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        if method == "session/cancel":
            raw_params = message.get("params")
            params = raw_params if isinstance(raw_params, dict) else {}
            session_id = str(params.get("sessionId", ""))
            await self._session_manager.cancel(session_id=session_id)
            return None
        if method in {"session/close", "_kiro.dev/session/terminate", "_kiro/session/delete"}:
            raw_params = message.get("params")
            params = raw_params if isinstance(raw_params, dict) else {}
            session_id = str(params.get("sessionId", ""))
            res = await self._session_manager.close_session(
                session_id=session_id,
                delete_persisted=(method == "_kiro/session/delete"),
            )
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": res,
            }
        return None
