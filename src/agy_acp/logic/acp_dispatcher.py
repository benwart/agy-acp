"""ACP JSON-RPC 2.0 request/notification dispatcher and framing engine."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from agy_acp import __version__
from agy_acp.logic.auth_manager import AuthManager
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.logic.session_manager import DEFAULT_MODEL_ID, SessionManager

MAX_FRAME_SIZE_BYTES = 16 * 1024 * 1024
SLASH_QUOTA_COMMANDS = frozenset({"/quota", "/usage", "/credits"})

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
        if method == "session/prompt":
            params = message.get("params")
            return await self._handle_session_prompt(
                message.get("id"),
                params if isinstance(params, dict) else {},
            )
        return None
