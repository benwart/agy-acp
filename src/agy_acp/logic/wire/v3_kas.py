"""KAS v3 (`_kiro/*` + `session_info_update` + `effortLevel`) ACP wire dialect Strategy."""

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

KAS_EXTENSION_METHODS: tuple[str, ...] = (
    "_kiro/session/compact",
    "_kiro/session/delete",
    "_kiro/mcp/resetServer",
)


class V3KasWireAdapter(EngineWireAdapter):
    """Concrete Strategy for Kiro Agent Server v3 (`agent_engine='v3'` / `'kas'`)."""

    @property
    def engine_id(self) -> Literal["v2", "v3"]:
        return "v3"

    @property
    def effort_config_id(self) -> str:
        return "effortLevel"

    @property
    def supported_effort_levels(self) -> tuple[str, ...]:
        return ("low", "medium", "high", "xhigh", "max")

    def _augment_initialize_capabilities(self, capabilities: dict[str, Any]) -> None:
        capabilities["_meta"] = {
            "kiro": {
                "extensionMethods": list(KAS_EXTENSION_METHODS),
                "checkpoints": False,
            }
        }

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
            modes.append(spec.to_available_mode(v3_meta=True))
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
                    "_meta": {
                        "kiro": {
                            "resource": {
                                "source": {"origin": "bundled"},
                            }
                        }
                    },
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
                "id": "effortLevel",
                "name": "Effort",
                "category": "thought_level",
                "type": "select",
                "currentValue": effort,
                "options": [
                    {"value": "low", "name": "Low", "description": "Minimal effort"},
                    {
                        "value": "medium",
                        "name": "Medium",
                        "description": "Balanced effort",
                    },
                    {"value": "high", "name": "High", "description": "High effort"},
                    {
                        "value": "xhigh",
                        "name": "Extra High",
                        "description": "Extended reasoning effort",
                    },
                    {
                        "value": "max",
                        "name": "Max",
                        "description": "Maximum reasoning effort",
                    },
                ],
            },
        ]

    def build_mcp_readiness_notifications(
        self,
        session_id: str,
        servers: Sequence[McpServerStatus],
    ) -> list[dict[str, Any]]:
        status_servers: list[dict[str, Any]] = []
        exposed_tags: list[dict[str, str]] = []
        exposed_tool_ids: list[str] = []

        for srv in servers:
            tool_entries: list[dict[str, Any]] = []
            for tool in srv.tools:
                tool_entries.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "disabled": tool.disabled,
                    }
                )
                if srv.status == "connected" and not tool.disabled:
                    exposed_tags.append({"source": "mcp", "tag": f"@{srv.name}/{tool.name}"})
                    exposed_tool_ids.append(f"mcp_{srv.name}_{tool.name}")

            entry: dict[str, Any] = {
                "name": srv.name,
                "displayName": srv.display_name or srv.name,
                "status": srv.status,
                "tools": tool_entries,
                "_meta": {
                    "kiro": {
                        "resource": {
                            "resourceType": "mcpServer",
                            "source": {"origin": srv.origin},
                        }
                    }
                },
            }
            if srv.error:
                entry["error"] = srv.error
                entry["errorMessage"] = srv.error
            if srv.auth_url:
                entry["authUrl"] = srv.auth_url
            if srv.failed_authorization or srv.status == "authRequired":
                entry["failedAuthorization"] = True
            status_servers.append(entry)

        return [
            {
                "jsonrpc": "2.0",
                "method": "_kiro/mcp/status",
                "params": {
                    "sessionId": session_id,
                    "servers": status_servers,
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "_kiro/tools/didChange",
                "params": {
                    "sessionId": session_id,
                    "tags": exposed_tags,
                    "initial": {
                        "tools": exposed_tool_ids,
                        "tags": exposed_tags,
                    },
                    "added": {
                        "tools": exposed_tool_ids,
                        "tags": exposed_tags,
                    },
                    "removed": {
                        "tools": [],
                        "tags": [],
                    },
                },
            },
        ]

    def build_mode_switched_notifications(
        self,
        session_id: str,
        mode_id: str,
        *,
        mode_name: str | None = None,
    ) -> list[dict[str, Any]]:
        _ = mode_name
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
            }
        ]

    def build_tool_call_update(
        self,
        session_id: str,
        event: NormalizedToolEvent,
        *,
        replay: bool = False,
    ) -> dict[str, Any]:
        kiro_meta: dict[str, Any]
        if event.is_mcp:
            kiro_meta = {
                "serverName": event.mcp_server,
                "toolName": event.mcp_tool,
                "toolOrigin": "client",
            }
        else:
            kiro_meta = {"toolOrigin": "builtin"}
            if event.kas_tool_id == "todo_list":
                kiro_meta["toolName"] = "todo_list"

        if event.agent_subtask_id:
            kiro_meta["agentSubtaskId"] = event.agent_subtask_id
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
        kiro_meta: dict[str, Any]
        if event.is_mcp:
            kiro_meta = {
                "toolId": f"mcp_{event.mcp_server}_{event.mcp_tool}",
                "mcpTool": {
                    "identity": {
                        "serverName": event.mcp_server,
                        "toolName": event.mcp_tool,
                    }
                },
            }
        else:
            kiro_meta = {"toolId": event.kas_tool_id}
            if event.kas_tool_id == "run_command" and event.command:
                kiro_meta["command"] = event.command
            elif event.kas_tool_id == "invoke_sub_agent" and event.subagent_target:
                kiro_meta["consent"] = {
                    "capability": "subagent",
                    "resource": event.subagent_target,
                }

        # KAS v3 intentionally omits `kind` on `request_permission.toolCall`;
        # KiroCrew resolves write-plane `kind` from the preceding `tool_call` cache.
        tool_call: dict[str, Any] = {
            "toolCallId": event.tool_call_id,
            "title": event.title,
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
        if not user_message_id:
            return []
        return [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "session_info_update",
                        "_meta": {
                            "kiro": {
                                "kind": "user_message_id_assigned",
                                "userMessageId": user_message_id,
                            }
                        },
                    },
                },
            }
        ]

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
        update: dict[str, Any] = {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        }
        kiro_meta: dict[str, Any] = {}
        if (
            prompt_received_at_ms is not None
            and first_chunk_emitted_at_ms is not None
            and first_chunk_emitted_at_ms >= prompt_received_at_ms
        ):
            kiro_meta["ttft"] = {
                "promptReceivedAt": int(prompt_received_at_ms),
                "firstChunkEmittedAt": int(first_chunk_emitted_at_ms),
            }
            kiro_meta["ttftMs"] = int(first_chunk_emitted_at_ms - prompt_received_at_ms)
        if agent_subtask_id:
            kiro_meta["agentSubtaskId"] = agent_subtask_id
        if replay:
            kiro_meta["replay"] = True
        if kiro_meta:
            update["_meta"] = {"kiro": kiro_meta}
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
        completion_kiro: dict[str, Any] = {
            "kind": "turn_completion",
            "promptTurnSummaries": [
                {"unit": "credit", "usage": telemetry.delta_credits},
            ],
        }
        if telemetry.context_breakdown is not None:
            completion_kiro["contextBreakdown"] = dict(telemetry.context_breakdown)
        if telemetry.turn_id:
            completion_kiro["turnId"] = telemetry.turn_id

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
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "session_info_update",
                        "_meta": {
                            "kiro": {
                                "kind": "context_usage",
                                "usagePercentage": telemetry.context_usage_percentage,
                            }
                        },
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "session_info_update",
                        "_meta": {"kiro": completion_kiro},
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
        kind_map: dict[str, str] = {
            "queued": "steering_queued",
            "consumed": "steering_injected",
            "injected": "steering_injected",
            "discarded": "steering_cleared",
            "cleared": "steering_cleared",
        }
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "session_info_update",
                    "_meta": {
                        "kiro": {
                            "kind": kind_map.get(phase, "steering_queued"),
                            "content": content,
                        }
                    },
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
        kiro_meta: dict[str, Any]
        if phase == "started":
            kiro_meta = {"kind": "summarization_started"}
        elif phase == "completed":
            kiro_meta = {
                "kind": "summarization_completed",
                "conversationSummary": summary,
            }
        else:
            kiro_meta = {
                "kind": "summarization_failed",
                "error": error or "Compaction failed",
                "message": error or "Compaction failed",
            }
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "session_info_update",
                    "_meta": {"kiro": kiro_meta},
                },
            },
        }

    def build_subagent_notifications(
        self,
        session_id: str,
        subagents: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        status_map = {
            "running": "in_progress",
            "in_progress": "in_progress",
            "pending": "pending",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        frames: list[dict[str, Any]] = []
        for entry in subagents:
            sub_id = str(
                entry.get("sessionId") or entry.get("conversationId") or entry.get("id") or ""
            )
            if not sub_id:
                continue
            label = str(
                entry.get("role") or entry.get("sessionName") or entry.get("name") or sub_id
            )
            raw_status = str(entry.get("status") or "in_progress").lower()
            mapped_status = status_map.get(raw_status, "in_progress")
            frames.append(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": f"subtask_{sub_id}",
                            "title": f"Sub-agent: {label}",
                            "status": mapped_status,
                            "_meta": {
                                "kiro": {
                                    "kind": "agent-subtask",
                                    "agentSubtaskId": sub_id,
                                }
                            },
                        },
                    },
                }
            )
        return frames
