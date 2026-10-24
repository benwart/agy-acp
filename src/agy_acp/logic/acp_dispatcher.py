"""ACP JSON-RPC 2.0 request/notification dispatcher and framing engine."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from agy_acp import __version__
from agy_acp.logic.auth_manager import AuthManager
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.logic.session_manager import DEFAULT_MODEL_ID, SessionManager

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
        self._quota_tracker = quota_tracker or QuotaTracker()

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

    def _should_execute_tool(self, session_id: str, prompt_text: str) -> bool:
        """Return True when the prompt or session approval mode triggers a tool execution."""
        if "tool" in prompt_text.lower():
            return True
        if os.environ.get("GOOSE_MODE") == "approve":
            return True
        return self._resolve_session_mode(session_id) == "approve"

    async def _execute_tool_with_permission(
        self,
        *,
        session_id: str,
        prompt_text: str,
    ) -> None:
        """Emit `tool_call`, gate via `session/request_permission`, and emit `tool_call_update`."""
        tool_call_id = f"tc_{uuid.uuid4().hex[:12]}"
        tool_descriptor: dict[str, Any] = {
            "toolCallId": tool_call_id,
            "title": "Execute command",
            "kind": "execute",
            "status": "in_progress",
            "rawInput": {"command": prompt_text},
        }
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

        allowed = True
        if self._client_rpc_caller is not None:
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
            allowed = selected_option in {"allow_once", "allow_always"}

        tool_status = "completed" if allowed else "failed"
        output_text = f"Executed: {prompt_text}" if allowed else "Tool execution rejected by user."
        await self._emit_notification(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tool_call_id,
                        "status": tool_status,
                        "kind": "execute",
                        "content": [
                            {
                                "type": "content",
                                "content": {"type": "text", "text": output_text},
                            }
                        ],
                        "rawOutput": {"output": output_text, "allowed": allowed},
                    },
                },
            }
        )

    @staticmethod
    def _extract_prompt_text(prompt_blocks: list[dict[str, Any]]) -> str:
        """Extract concatenated text from ACP prompt content blocks."""
        parts = [
            str(block.get("text", ""))
            for block in prompt_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()

    async def _handle_session_prompt(
        self,
        message_id: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle `session/prompt`, streaming thought/message chunks and quota updates."""
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
        raw_prompt = params.get("prompt")
        prompt_blocks: list[dict[str, Any]] = (
            [b for b in raw_prompt if isinstance(b, dict)] if isinstance(raw_prompt, list) else []
        )

        stop_reason = "end_turn"
        try:
            turn_result = await self._session_manager.prompt(
                session_id=session_id,
                prompt_blocks=prompt_blocks,
            )
            if isinstance(turn_result, dict) and turn_result.get("stopReason"):
                stop_reason = str(turn_result["stopReason"])
        except KeyError:
            pass

        prompt_text = self._extract_prompt_text(prompt_blocks)
        slash_cmd = prompt_text.splitlines()[-1].strip().lower() if prompt_text else ""
        if slash_cmd in SLASH_QUOTA_COMMANDS:
            thought_text = f"Formatting quota report for {slash_cmd}"
            response_text = self._quota_tracker.format_slash_command_response(slash_cmd)
        else:
            thought_text = "Processing prompt..."
            response_text = f"Processed: {prompt_text}" if prompt_text else "Ready."

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
        if slash_cmd not in SLASH_QUOTA_COMMANDS and self._should_execute_tool(
            session_id, prompt_text
        ):
            await self._execute_tool_with_permission(
                session_id=session_id,
                prompt_text=prompt_text,
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

        input_tokens = max(1, len(prompt_text.split())) if prompt_text else 1
        output_tokens = max(1, len(response_text.split()))
        model_id = self._resolve_session_model(session_id)
        for wire_update in self._quota_tracker.build_kirocrew_wire_updates(
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
                "stopReason": stop_reason,
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "cachedReadTokens": 0,
                    "cachedWriteTokens": 0,
                },
            },
        }

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
        await self._emit_notification(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": desc["sessionId"],
                    "update": {
                        "sessionUpdate": "available_commands_update",
                        "availableCommands": list(AVAILABLE_SLASH_COMMANDS),
                    },
                },
            }
        )
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": desc,
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
        if method == "session/prompt":
            params = message.get("params")
            return await self._handle_session_prompt(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        return None
