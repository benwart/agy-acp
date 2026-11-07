"""EngineWireAdapter Strategy package and factory for Kiro v2 and KAS v3 dialects."""

from __future__ import annotations

from agy_acp.logic.wire.base import (
    DEFAULT_BASE_MODES,
    PERMISSION_OPTIONS,
    CompactionPhase,
    EngineWireAdapter,
    McpConnectionState,
    McpServerStatus,
    McpToolDef,
    ModelFallbackScope,
    ModelFallbackTrigger,
    NormalizedToolEvent,
    SteeringPhase,
    ToolCallKind,
    WireTurnTelemetry,
    normalize_agy_tool_event,
)
from agy_acp.logic.wire.v2_kiro import V2KiroWireAdapter
from agy_acp.logic.wire.v3_kas import KAS_EXTENSION_METHODS, V3KasWireAdapter

_V2_KEYS: frozenset[str] = frozenset({"", "default", "v2", "kiro"})
_V3_KEYS: frozenset[str] = frozenset({"v3", "kas"})


def create_wire_adapter(agent_engine: str | None = None) -> EngineWireAdapter:
    """Instantiate the `EngineWireAdapter` Strategy matching `agent_engine`."""
    normalized = (agent_engine or "").strip().lower()
    if normalized in _V2_KEYS:
        return V2KiroWireAdapter()
    if normalized in _V3_KEYS:
        return V3KasWireAdapter()
    raise ValueError(
        f"Unsupported agent engine {agent_engine!r}; expected one of: default, v2, kiro, v3, kas"
    )


__all__ = [
    "CompactionPhase",
    "DEFAULT_BASE_MODES",
    "EngineWireAdapter",
    "KAS_EXTENSION_METHODS",
    "McpConnectionState",
    "McpServerStatus",
    "McpToolDef",
    "ModelFallbackScope",
    "ModelFallbackTrigger",
    "NormalizedToolEvent",
    "PERMISSION_OPTIONS",
    "SteeringPhase",
    "ToolCallKind",
    "V2KiroWireAdapter",
    "V3KasWireAdapter",
    "WireTurnTelemetry",
    "create_wire_adapter",
    "normalize_agy_tool_event",
]
