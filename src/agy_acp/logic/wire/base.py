"""Base Strategy ABC and value types for Kiro v2 and KAS v3 ACP wire dialects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agy_acp.logic.custom_agents import AgentOrigin, CustomAgentSpec

ToolCallKind = Literal["execute", "read", "edit", "delete", "search", "fetch", "other"]
McpConnectionState = Literal["connecting", "connected", "failed", "disabled", "authRequired"]
SteeringPhase = Literal["queued", "consumed", "injected", "discarded", "cleared"]
CompactionPhase = Literal["started", "completed", "failed"]
ModelFallbackTrigger = Literal["availability", "refusal"]
ModelFallbackScope = Literal["turn", "session"]

DEFAULT_BASE_MODES: tuple[dict[str, str], ...] = (
    {
        "id": "read-only",
        "name": "Read-Only",
        "description": "Deny file modifications and command execution.",
    },
    {
        "id": "approve",
        "name": "Supervised",
        "description": "Request permission via ACP before running commands or modifying files.",
    },
    {
        "id": "auto",
        "name": "Autonomous",
        "description": "Auto-approve safe operations while guarding destructive actions.",
    },
)

PERMISSION_OPTIONS: tuple[dict[str, str], ...] = (
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "allow_always", "name": "Always allow", "kind": "allow_always"},
    {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
)

_AGY_STEP_TOOL_MAP: dict[str, tuple[ToolCallKind, str, str]] = {
    "run_command": ("execute", "run_command", "execute_bash"),
    "command_action": ("execute", "run_command", "execute_bash"),
    "execute_bash": ("execute", "run_command", "execute_bash"),
    "manage_task": ("execute", "run_command", "execute_bash"),
    "send_command_input": ("execute", "run_command", "execute_bash"),
    "read_file": ("read", "read_file", "fs_read"),
    "view_file": ("read", "read_file", "fs_read"),
    "list_dir": ("read", "list_directory", "fs_read"),
    "list_directory": ("read", "list_directory", "fs_read"),
    "write_file": ("edit", "fs_write", "fs_write"),
    "write_to_file": ("edit", "fs_write", "fs_write"),
    "edit_file": ("edit", "str_replace", "fs_write"),
    "replace_file_content": ("edit", "str_replace", "fs_write"),
    "multi_replace_file_content": ("edit", "str_replace", "fs_write"),
    "append_file": ("edit", "fs_append", "fs_write"),
    "delete_file": ("delete", "delete_file", "fs_write"),
    "remove_file": ("delete", "delete_file", "fs_write"),
    "grep_search": ("search", "grep_search", "grep"),
    "find_by_name": ("search", "file_search", "glob"),
    "file_search": ("search", "file_search", "glob"),
    "search_web": ("search", "remote_web_search", "web_search"),
    "remote_web_search": ("search", "remote_web_search", "web_search"),
    "read_url_content": ("fetch", "web_fetch", "web_fetch"),
    "web_fetch": ("fetch", "web_fetch", "web_fetch"),
    "invoke_subagent": ("other", "invoke_sub_agent", "use_subagent"),
    "invoke_sub_agent": ("other", "invoke_sub_agent", "use_subagent"),
    "todo_list": ("other", "todo_list", "todo_list"),
}


@dataclass(frozen=True, slots=True)
class McpToolDef:
    """Catalog definition for one tool discovered from an MCP server."""

    name: str
    description: str = ""
    disabled: bool = False


@dataclass(frozen=True, slots=True)
class McpServerStatus:
    """Readiness snapshot for one session-declared MCP server."""

    name: str
    display_name: str = ""
    status: McpConnectionState = "connected"
    tools: tuple[McpToolDef, ...] = ()
    error: str | None = None
    auth_url: str | None = None
    failed_authorization: bool = False
    origin: AgentOrigin = "client"


@dataclass(frozen=True, slots=True)
class NormalizedToolEvent:
    """Dialect-neutral representation of a built-in or MCP tool invocation."""

    tool_call_id: str
    title: str
    kind: ToolCallKind = "other"
    status: str = "in_progress"
    kas_tool_id: str = "other"
    kiro_v2_tool_name: str = ""
    raw_input: dict[str, Any] = field(default_factory=dict)
    raw_output: dict[str, Any] | None = None
    locations: tuple[dict[str, Any], ...] = ()
    content: tuple[dict[str, Any], ...] = ()
    command: str = ""
    subagent_target: str = ""
    is_mcp: bool = False
    mcp_server: str = ""
    mcp_tool: str = ""
    agent_subtask_id: str = ""


@dataclass(frozen=True, slots=True)
class WireTurnTelemetry:
    """Per-turn context window, credit usage, metadata, and rate-limit telemetry."""

    context_used: int
    context_size: int
    context_usage_percentage: float
    delta_credits: float = 0.0
    metadata_payload: dict[str, Any] = field(default_factory=dict)
    rate_limited_message: str | None = None
    context_breakdown: dict[str, Any] | None = None
    turn_id: str = ""


def _split_mcp_step_name(
    remainder: str,
    known_mcp_servers: Collection[str],
) -> tuple[str, str] | None:
    """Split `<server>_<tool>` matching the longest configured server prefix first."""
    for candidate in sorted((str(s) for s in known_mcp_servers if s), key=len, reverse=True):
        prefix = f"{candidate}_"
        if remainder.startswith(prefix):
            tool_part = remainder[len(prefix) :]
            if tool_part:
                return candidate, tool_part
    srv, sep, tool = remainder.partition("_")
    if sep and srv and tool:
        return srv, tool
    return None


def synthesize_task_command(step_name: str, arguments: Mapping[str, Any]) -> str:
    """Synthesize a concise command summary for task-management built-ins."""
    lower = step_name.strip().lower()
    if lower == "manage_task":
        action = str(arguments.get("Action") or arguments.get("action") or "").strip()
        task_id = str(
            arguments.get("TaskId") or arguments.get("task_id") or arguments.get("taskId") or ""
        ).strip()
        return " ".join(part for part in ("manage_task", action, task_id) if part)
    if lower == "send_command_input":
        target = str(
            arguments.get("CommandId")
            or arguments.get("TaskId")
            or arguments.get("command_id")
            or arguments.get("task_id")
            or ""
        ).strip()
        return " ".join(part for part in ("send_command_input", target) if part)
    return ""


def normalize_agy_tool_event(
    *,
    tool_call_id: str,
    step_name: str,
    title: str,
    raw_input: Mapping[str, Any] | None = None,
    status: str = "in_progress",
    locations: Sequence[dict[str, Any]] = (),
    content: Sequence[dict[str, Any]] = (),
    agent_subtask_id: str = "",
    known_mcp_servers: Collection[str] = (),
) -> NormalizedToolEvent:
    """Normalize an Antigravity CLI tool step into a `NormalizedToolEvent`."""
    args = dict(raw_input) if isinstance(raw_input, Mapping) else {}
    lower_step = step_name.strip()

    if lower_step == "call_mcp_tool":
        srv = str(args.get("ServerName") or args.get("server_name") or "")
        tool = str(args.get("ToolName") or args.get("tool_name") or "")
        inner_args = args.get("Arguments")
        payload = dict(inner_args) if isinstance(inner_args, Mapping) else args
        return NormalizedToolEvent(
            tool_call_id=tool_call_id,
            title=title or f"@{srv}/{tool}",
            kind="other",
            status=status,
            raw_input=payload,
            locations=tuple(locations),
            content=tuple(content),
            is_mcp=True,
            mcp_server=srv,
            mcp_tool=tool,
            agent_subtask_id=agent_subtask_id,
        )

    if lower_step.startswith("mcp_"):
        split_pair = _split_mcp_step_name(lower_step[len("mcp_") :], known_mcp_servers)
        if split_pair is not None:
            srv, tool = split_pair
            return NormalizedToolEvent(
                tool_call_id=tool_call_id,
                title=title or f"@{srv}/{tool}",
                kind="other",
                status=status,
                raw_input=args,
                locations=tuple(locations),
                content=tuple(content),
                is_mcp=True,
                mcp_server=srv,
                mcp_tool=tool,
                agent_subtask_id=agent_subtask_id,
            )

    kind, kas_id, v2_name = _AGY_STEP_TOOL_MAP.get(
        lower_step,
        ("other", lower_step or "other", lower_step or "other"),
    )
    if kas_id == "fs_write" and args.get("Append") is True:
        kas_id = "fs_append"

    cmd = ""
    if kas_id == "run_command":
        cmd = str(args.get("CommandLine") or args.get("command") or "") or synthesize_task_command(
            lower_step,
            args,
        )

    sub_target = ""
    if kas_id == "invoke_sub_agent":
        sub_target = str(args.get("TypeName") or args.get("agent") or args.get("Role") or "")

    return NormalizedToolEvent(
        tool_call_id=tool_call_id,
        title=title or lower_step,
        kind=kind,
        status=status,
        kas_tool_id=kas_id,
        kiro_v2_tool_name=v2_name,
        raw_input=args,
        locations=tuple(locations),
        content=tuple(content),
        command=cmd,
        subagent_target=sub_target,
        agent_subtask_id=agent_subtask_id,
    )


class EngineWireAdapter(ABC):
    """Strategy interface encapsulating Kiro v2 vs KAS v3 JSON-RPC wire shapes."""

    @property
    @abstractmethod
    def engine_id(self) -> Literal["v2", "v3"]:
        """Return canonical engine identifier (`'v2'` or `'v3'`)."""

    @property
    @abstractmethod
    def effort_config_id(self) -> str:
        """Return config option ID for thought/reasoning effort."""

    @property
    @abstractmethod
    def supported_effort_levels(self) -> tuple[str, ...]:
        """Return ordered effort level values supported on the wire."""

    @staticmethod
    def resolve_concrete_model(
        models: Sequence[Mapping[str, str]],
        current_model: str,
    ) -> str:
        """Resolve `"auto"` or unknown model IDs to the first concrete option when needed."""
        option_ids = [
            str(m.get("modelId") or m.get("id") or "")
            for m in models
            if m.get("modelId") or m.get("id")
        ]
        if current_model and current_model in option_ids:
            return current_model
        if option_ids:
            return option_ids[0]
        return current_model or "gemini-3.7-pro"

    @staticmethod
    def build_permission_options() -> list[dict[str, str]]:
        """Return deep-copied explicit ACP permission options (`allow_once`, etc.)."""
        return [dict(opt) for opt in PERMISSION_OPTIONS]

    @staticmethod
    def normalize_tool_event(
        *,
        tool_call_id: str,
        step_name: str,
        title: str,
        raw_input: Mapping[str, Any] | None = None,
        status: str = "in_progress",
        locations: Sequence[dict[str, Any]] = (),
        content: Sequence[dict[str, Any]] = (),
        agent_subtask_id: str = "",
        known_mcp_servers: Collection[str] = (),
    ) -> NormalizedToolEvent:
        """Normalize an Antigravity CLI tool step into a `NormalizedToolEvent`."""
        return normalize_agy_tool_event(
            tool_call_id=tool_call_id,
            step_name=step_name,
            title=title,
            raw_input=raw_input,
            status=status,
            locations=locations,
            content=content,
            agent_subtask_id=agent_subtask_id,
            known_mcp_servers=known_mcp_servers,
        )

    def build_initialize_result(
        self,
        *,
        version: str,
        name: str = "agy-acp",
        title: str = "Antigravity ACP Bridge",
    ) -> dict[str, Any]:
        """Build the ACP `initialize` result envelope with integer `protocolVersion: 1`."""
        capabilities: dict[str, Any] = {
            "loadSession": True,
            "promptCapabilities": {
                "image": True,
                "audio": False,
                "embeddedContext": True,
            },
            "mcpCapabilities": {
                "http": True,
                "sse": True,
            },
        }
        self._augment_initialize_capabilities(capabilities)
        return {
            "protocolVersion": 1,
            "agentCapabilities": capabilities,
            "agentInfo": {
                "name": name,
                "title": title,
                "version": version,
            },
        }

    @abstractmethod
    def _augment_initialize_capabilities(self, capabilities: dict[str, Any]) -> None:
        """Mutate `agentCapabilities` with dialect-specific `_meta` extensions if any."""

    @abstractmethod
    def build_available_modes(
        self,
        *,
        custom_agents: Sequence[CustomAgentSpec] = (),
        base_modes: Sequence[Mapping[str, str]] = DEFAULT_BASE_MODES,
    ) -> list[dict[str, Any]]:
        """Build ACP `modes.availableModes` entries for session/new and session/load."""

    @abstractmethod
    def build_config_options(
        self,
        *,
        models: Sequence[Mapping[str, str]],
        current_model: str,
        current_mode: str,
        current_effort: str,
    ) -> list[dict[str, Any]]:
        """Build ACP `configOptions` array for model, mode, and effort selection."""

    @abstractmethod
    def build_mcp_readiness_notifications(
        self,
        session_id: str,
        servers: Sequence[McpServerStatus],
    ) -> list[dict[str, Any]]:
        """Build post-session-start MCP server readiness notifications."""

    @abstractmethod
    def build_mode_switched_notifications(
        self,
        session_id: str,
        mode_id: str,
        *,
        mode_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build notifications emitted when switching mode / agent."""

    @abstractmethod
    def build_tool_call_update(
        self,
        session_id: str,
        event: NormalizedToolEvent,
        *,
        replay: bool = False,
    ) -> dict[str, Any]:
        """Build an ACP `session/update` (`tool_call`) notification."""

    def build_tool_call_completion(
        self,
        session_id: str,
        *,
        tool_call_id: str,
        status: str = "completed",
        raw_output: Mapping[str, Any] | None = None,
        content: Sequence[dict[str, Any]] = (),
        agent_subtask_id: str = "",
        replay: bool = False,
    ) -> dict[str, Any]:
        """Build an ACP `session/update` (`tool_call_update`) completion notification."""
        update: dict[str, Any] = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": tool_call_id,
            "status": status,
        }
        if raw_output is not None:
            update["rawOutput"] = dict(raw_output)
        if content:
            update["content"] = list(content)
        kiro_meta: dict[str, Any] = {}
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

    @abstractmethod
    def build_permission_request_params(
        self,
        session_id: str,
        event: NormalizedToolEvent,
    ) -> dict[str, Any]:
        """Build `params` for a bidirectional `session/request_permission` call."""

    @abstractmethod
    def build_turn_start_notifications(
        self,
        session_id: str,
        user_message_id: str,
    ) -> list[dict[str, Any]]:
        """Build notifications emitted immediately when a prompt turn begins."""

    @abstractmethod
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
        """Build a `session/update` (`agent_message_chunk`) notification."""

    def build_user_message_chunk(
        self,
        session_id: str,
        content_block: Mapping[str, Any],
        *,
        user_message_id: str = "",
        replay: bool = False,
    ) -> dict[str, Any]:
        """Build a `session/update` (`user_message_chunk`) notification (used on replay)."""
        update: dict[str, Any] = {
            "sessionUpdate": "user_message_chunk",
            "content": dict(content_block),
        }
        kiro_meta: dict[str, Any] = {}
        if user_message_id:
            kiro_meta["userMessageId"] = user_message_id
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

    def build_agent_thought_chunk(
        self,
        session_id: str,
        thought: str,
        *,
        prompt_received_at_ms: int | None = None,
        first_chunk_emitted_at_ms: int | None = None,
        agent_subtask_id: str = "",
        replay: bool = False,
    ) -> dict[str, Any]:
        """Build a `session/update` (`agent_thought_chunk`) notification."""
        update: dict[str, Any] = {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": thought},
        }
        kiro_meta: dict[str, Any] = {}
        if (
            self.engine_id == "v3"
            and prompt_received_at_ms is not None
            and first_chunk_emitted_at_ms is not None
            and first_chunk_emitted_at_ms >= prompt_received_at_ms
        ):
            kiro_meta["ttft"] = {
                "promptReceivedAt": int(prompt_received_at_ms),
                "firstChunkEmittedAt": int(first_chunk_emitted_at_ms),
            }
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

    @abstractmethod
    def build_turn_telemetry_notifications(
        self,
        session_id: str,
        telemetry: WireTurnTelemetry,
    ) -> list[dict[str, Any]]:
        """Build end-of-turn usage, context, credit, metadata, and rate-limit frames."""

    @abstractmethod
    def build_steering_notification(
        self,
        session_id: str,
        phase: SteeringPhase,
        content: str,
    ) -> dict[str, Any]:
        """Build mid-turn steering lifecycle notification."""

    @abstractmethod
    def build_compaction_notification(
        self,
        session_id: str,
        phase: CompactionPhase,
        *,
        summary: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        """Build conversation compaction / summarization lifecycle notification."""

    @abstractmethod
    def build_subagent_notifications(
        self,
        session_id: str,
        subagents: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build native child subagent roster/progress notifications."""

    @abstractmethod
    def build_model_fallback_notifications(
        self,
        session_id: str,
        *,
        from_model: str,
        to_model: str | None,
        trigger: ModelFallbackTrigger = "availability",
        scope: ModelFallbackScope = "session",
        reason: str = "",
    ) -> list[dict[str, Any]]:
        """Build engine-initiated model fallback notifications."""
