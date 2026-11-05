"""Kiro v2 (`_kiro.dev/*` + `reasoning_effort`) ACP wire dialect Strategy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from agy_acp.logic.custom_agents import CustomAgentSpec
from agy_acp.logic.wire.base import (
    DEFAULT_BASE_MODES,
    CompactionPhase,
    EngineWireAdapter,
    McpServerStatus,
    NormalizedToolEvent,
    SteeringPhase,
    WireTurnTelemetry,
)


class V2KiroWireAdapter(EngineWireAdapter):
    """Concrete Strategy for Kiro CLI v2 (`agent_engine='v2'` / `'default'`)."""

    @property
    def engine_id(self) -> Literal["v2", "v3"]:
        return "v2"

    @property
    def effort_config_id(self) -> str:
        return "reasoning_effort"

    @property
    def supported_effort_levels(self) -> tuple[str, ...]:
        return ("low", "medium", "high")

    def _augment_initialize_capabilities(self, capabilities: dict[str, Any]) -> None:
        _ = capabilities

    def build_available_modes(
        self,
        *,
        custom_agents: Sequence[CustomAgentSpec] = (),
        base_modes: Sequence[Mapping[str, str]] = DEFAULT_BASE_MODES,
    ) -> list[dict[str, Any]]:
        modes: list[dict[str, Any]] = []
        seen: set[str] = set()
        for spec in custom_agents:
            if spec.agent_id in seen:
                continue
            seen.add(spec.agent_id)
            modes.append(spec.to_available_mode(v3_meta=False))
        for base in base_modes:
            mode_id = str(base["id"])
            if mode_id in seen:
                continue
            seen.add(mode_id)
            modes.append(
                {
                    "id": mode_id,
                    "name": str(base["name"]),
                    "description": str(base.get("description", base["name"])),
                }
            )
        return modes

    def build_config_options(
        self,
        *,
        models: Sequence[Mapping[str, str]],
        current_model: str,
        current_mode: str,
        current_effort: str,
    ) -> list[dict[str, Any]]:
        concrete_model = self.resolve_concrete_model(models, current_model)
        effort = current_effort if current_effort in self.supported_effort_levels else "medium"
        return [
            {
                "id": "model",
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": concrete_model,
                "options": [
                    {
                        "value": str(m["modelId"]),
                        "name": str(m.get("name", m["modelId"])),
                        "description": str(m.get("description", m.get("name", m["modelId"]))),
                    }
                    for m in models
                    if m.get("modelId")
                ],
            },
            {
                "id": "mode",
                "name": "Approval Mode",
                "category": "mode",
                "type": "select",
                "currentValue": current_mode or "auto",
                "options": [
                    {
                        "value": "read-only",
                        "name": "Read-Only",
                        "description": "No file writes or shell commands",
                    },
                    {
                        "value": "approve",
                        "name": "Supervised",
                        "description": "Ask permission before mutations",
                    },
                    {
                        "value": "auto",
                        "name": "Autonomous",
                        "description": "Auto-approve safe operations",
                    },
                ],
            },
            {
                "id": "reasoning_effort",
                "name": "Reasoning Effort",
                "category": "thought_level",
                "type": "select",
                "currentValue": effort,
                "options": [
                    {"value": "low", "name": "Low", "description": "Fast thinking"},
                    {
                        "value": "medium",
                        "name": "Medium",
                        "description": "Balanced thinking",
                    },
                    {"value": "high", "name": "High", "description": "Deep thinking"},
                ],
            },
        ]

    def build_mcp_readiness_notifications(
        self,
        session_id: str,
        servers: Sequence[McpServerStatus],
    ) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for srv in servers:
            if srv.status == "authRequired" or srv.failed_authorization:
                frames.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "_kiro.dev/mcp/oauth_request",
                        "params": {
                            "sessionId": session_id,
                            "serverName": srv.name,
                            "oauthUrl": srv.auth_url or "",
                        },
                    }
                )
            elif srv.status == "connected":
                frames.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "_kiro.dev/mcp/server_initialized",
                        "params": {
                            "sessionId": session_id,
                            "serverName": srv.name,
                        },
                    }
                )
            elif srv.status == "failed":
                frames.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "_kiro.dev/mcp/server_init_failure",
                        "params": {
                            "sessionId": session_id,
                            "serverName": srv.name,
                            "error": srv.error or "MCP server initialization failed",
                        },
                    }
                )
        return frames

    def build_mode_switched_notifications(
        self,
        session_id: str,
        mode_id: str,
        *,
        mode_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
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
            },
            {
                "jsonrpc": "2.0",
                "method": "_kiro.dev/agent/switched",
                "params": {
                    "sessionId": session_id,
                    "agentName": mode_name or mode_id,
                },
            },
        ]

    def build_tool_call_update(
        self,
        session_id: str,
        event: NormalizedToolEvent,
        *,
        replay: bool = False,
    ) -> dict[str, Any]:
        if event.is_mcp:
            tool_name = f"@{event.mcp_server}/{event.mcp_tool}"
            mcp_server_name = event.mcp_server
        else:
            tool_name = event.kiro_v2_tool_name or event.kas_tool_id
            mcp_server_name = ""

        kiro_meta: dict[str, Any] = {
            "toolName": tool_name,
            "mcpServerName": mcp_server_name,
        }
        if replay:
            kiro_meta["replay"] = True

        update: dict[str, Any] = {
            "sessionUpdate": "tool_call",
            "toolCallId": event.tool_call_id,
            "title": event.title,
            "kind": event.kind,
            "status": event.status,
            "rawInput": dict(event.raw_input),
            "_meta": {"kiro": kiro_meta},
        }
        if event.locations:
            update["locations"] = [dict(loc) for loc in event.locations]
        if event.content:
            update["content"] = [dict(block) for block in event.content]

        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": update,
            },
        }

    def build_permission_request_params(
        self,
        session_id: str,
        event: NormalizedToolEvent,
    ) -> dict[str, Any]:
        if event.is_mcp:
            tool_id = f"@{event.mcp_server}/{event.mcp_tool}"
        else:
            tool_id = event.kiro_v2_tool_name or event.kas_tool_id

        kiro_meta: dict[str, Any] = {"toolId": tool_id}
        if not event.is_mcp and event.command:
            kiro_meta["command"] = event.command

        tool_call: dict[str, Any] = {
            "toolCallId": event.tool_call_id,
            "title": event.title,
            "kind": event.kind,
            "status": "pending",
            "rawInput": dict(event.raw_input),
        }
        if event.locations:
            tool_call["locations"] = [dict(loc) for loc in event.locations]
        if event.content:
            tool_call["content"] = [dict(block) for block in event.content]

        return {
            "sessionId": session_id,
            "toolCall": tool_call,
            "options": self.build_permission_options(),
            "_meta": {"kiro": kiro_meta},
        }

    def build_turn_start_notifications(
        self,
        session_id: str,
        user_message_id: str,
    ) -> list[dict[str, Any]]:
        _ = (session_id, user_message_id)
        return []

    def build_agent_message_chunk(
        self,
        session_id: str,
        text: str,
        *,
        prompt_received_at_ms: int | None = None,
        first_chunk_emitted_at_ms: int | None = None,
        agent_subtask_id: str = "",
        replay: bool = False,
    ) -> dict[str, Any]:
        _ = (prompt_received_at_ms, first_chunk_emitted_at_ms, agent_subtask_id)
        update: dict[str, Any] = {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        }
        if replay:
            update["_meta"] = {"kiro": {"replay": True}}
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": update,
            },
        }

    def build_turn_telemetry_notifications(
        self,
        session_id: str,
        telemetry: WireTurnTelemetry,
    ) -> list[dict[str, Any]]:
        meta_params = {"sessionId": session_id, **telemetry.metadata_payload}
        frames: list[dict[str, Any]] = [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "usage_update",
                        "used": telemetry.context_used,
                        "size": telemetry.context_size,
                        "cost": {"amount": 0.0, "currency": "USD"},
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "_kiro.dev/metadata",
                "params": meta_params,
            },
        ]
        if telemetry.rate_limited_message:
            frames.append(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/error/rate_limit",
                    "params": {
                        "sessionId": session_id,
                        "message": telemetry.rate_limited_message,
                    },
                }
            )
        return frames

    def build_steering_notification(
        self,
        session_id: str,
        phase: SteeringPhase,
        content: str,
    ) -> dict[str, Any]:
        phase_map: dict[str, str] = {
            "queued": "steering_queued",
            "consumed": "steering_consumed",
            "injected": "steering_consumed",
            "discarded": "steering_discarded",
            "cleared": "steering_discarded",
        }
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": phase_map.get(phase, "steering_queued"),
                    "content": content,
                },
            },
        }

    def build_compaction_notification(
        self,
        session_id: str,
        phase: CompactionPhase,
        *,
        summary: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "sessionId": session_id,
            "status": {"type": phase},
            "summary": summary,
        }
        if error:
            params["error"] = error
        return {
            "jsonrpc": "2.0",
            "method": "_kiro.dev/compaction/status",
            "params": params,
        }

    def build_subagent_notifications(
        self,
        session_id: str,
        subagents: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not subagents:
            return []
        return [
            {
                "jsonrpc": "2.0",
                "method": "_kiro.dev/subagent/list_update",
                "params": {
                    "sessionId": session_id,
                    "subagents": [dict(entry) for entry in subagents],
                },
            }
        ]
