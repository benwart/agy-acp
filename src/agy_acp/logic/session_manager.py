"""Session lifecycle, multi-session multiplexing, steer, and cancellation logic backed by SQLite."""

from __future__ import annotations

import base64
import contextlib
import copy
import os
import re
import secrets
import shutil
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

from agy_acp.logic.custom_agents import CustomAgentRegistry, CustomAgentSpec
from agy_acp.logic.mcp_catalog_prober import McpCatalogProber
from agy_acp.logic.wire import EngineWireAdapter, McpServerStatus, V2KiroWireAdapter
from agy_acp.providers.agy_cli_provider import (
    FALLBACK_MODELS,
    PER_SESSION_MCP_ENV_KEYS,
    AgyCliProvider,
    ToolCallEvent,
    _dedupe_mcp_servers,
    load_kiro_agent_spec,
    translate_mcp_server,
)
from agy_acp.providers.state_store import SqliteStateStoreProvider

DEFAULT_MODEL_ID = "gemini-3.7-pro"
DEFAULT_REASONING_EFFORT = "medium"

_KIROCREW_SPAWN_ROUTING_DIRECTIVE = (
    "[KIROCREW SUBAGENT ROUTING]\n"
    "Subagent delegation in this KiroCrew session MUST use kirocrew-core's `spawn_run` tool "
    'via `call_mcp_tool` (`ServerName="kirocrew-core"`, `ToolName="spawn_run"`, '
    "`Arguments={...}`; read `<appDataDir>/mcp/kirocrew-core/spawn_run.json` first if needed). "
    "NEVER call built-in `invoke_subagent`, `define_subagent`, or `manage_subagents`, "
    "because native subagents bypass KiroCrew chat cards and completion events."
)

_READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "fs_read",
        "read",
        "read_file",
        "list_directory",
        "grep",
        "grep_search",
        "glob",
        "file_search",
        "code",
        "web_search",
        "remote_web_search",
        "web_fetch",
        "tool_search",
        "todo_list",
    }
)

_SPAWN_RUN_ALLOW_PATTERNS: frozenset[str] = frozenset(
    {
        "*",
        "@kirocrew-core",
        "@kirocrew-core/*",
        "@kirocrew-core/spawn_run",
        "mcp_kirocrew-core_spawn_run",
    }
)

_SPAWN_RUN_DENY_PATTERNS: frozenset[str] = _SPAWN_RUN_ALLOW_PATTERNS | {"spawn_run"}


def _normalize_env_mapping(raw_env: Any) -> dict[str, str]:
    """Normalize MCP `env` (`dict` or `[{name, value}]`) into `dict[str, str]`."""
    if isinstance(raw_env, Mapping):
        return {str(k): str(v) for k, v in raw_env.items() if v is not None}
    if isinstance(raw_env, Sequence) and not isinstance(raw_env, (str, bytes)):
        out: dict[str, str] = {}
        for item in raw_env:
            if isinstance(item, Mapping) and item.get("name") and item.get("value") is not None:
                out[str(item["name"])] = str(item["value"])
        return out
    return {}


def _session_has_kirocrew_spawn_run(session: dict[str, Any]) -> bool:
    """Return whether `session` exposes enabled `kirocrew-core` `spawn_run`."""
    raw_servers = session.get("mcp_servers")
    if not isinstance(raw_servers, list) or not raw_servers:
        raw_servers = session.get("wire_mcp_servers")
    if not isinstance(raw_servers, list):
        return False

    core_srv = next(
        (
            srv
            for srv in raw_servers
            if isinstance(srv, dict)
            and str(srv.get("name") or "").strip() == "kirocrew-core"
            and srv.get("disabled") is not True
        ),
        None,
    )
    if core_srv is None:
        return False

    disabled_tools = core_srv.get("disabledTools") or core_srv.get("disabled_tools")
    if isinstance(disabled_tools, list) and any(
        str(t).strip() == "spawn_run" for t in disabled_tools
    ):
        return False

    excluded_tools = session.get("agentExcludedTools")
    if isinstance(excluded_tools, list) and any(
        str(t).strip() in _SPAWN_RUN_DENY_PATTERNS for t in excluded_tools
    ):
        return False

    agent_tools = session.get("agentTools")
    if isinstance(agent_tools, list):
        return any(str(tool).strip() in _SPAWN_RUN_ALLOW_PATTERNS for tool in agent_tools)
    return True


def _extract_session_mcp_env(
    session: dict[str, Any],
    *,
    session_id: str | None = None,
    state_db_path: Path | None = None,
) -> dict[str, str]:
    """Build isolated per-session MCP identity environment variables for an `agy` subprocess."""
    env: dict[str, str] = dict.fromkeys(PER_SESSION_MCP_ENV_KEYS, "")
    env["KIROCREW_ROUTE_SPAWN_RUN"] = "1" if _session_has_kirocrew_spawn_run(session) else ""

    sid = str(session_id or session.get("sessionId") or "").strip()
    if sid:
        env["AGY_ACP_SESSION_ID"] = sid
    if state_db_path is not None:
        env["AGY_ACP_STATE_DB"] = str(state_db_path)

    for source_key in ("mcp_servers", "wire_mcp_servers"):
        sources = session.get(source_key)
        if not isinstance(sources, list):
            continue
        for srv in sources:
            if not isinstance(srv, dict) or srv.get("disabled") is True:
                continue
            for key, val in _normalize_env_mapping(srv.get("env")).items():
                if key in PER_SESSION_MCP_ENV_KEYS and val:
                    env[key] = val
    return env


def _apply_session_prompt_directives(
    formatted_prompt: str,
    session: dict[str, Any],
    *,
    is_new_conversation: bool,
) -> str:
    """Prepend Kiro mandate, system prompt, and `spawn_run` routing when starting/reinjecting."""
    should_inject = is_new_conversation or bool(session.get("reinjectAgentPrompt"))
    if not should_inject:
        return formatted_prompt

    prefixes: list[str] = []
    agent_prompt = session.get("agentPrompt")
    if (
        isinstance(agent_prompt, str)
        and agent_prompt.strip()
        and agent_prompt.strip() not in formatted_prompt
    ):
        clean_prompt = agent_prompt.strip()
        if "[AGENT SYSTEM PROMPT]" in clean_prompt:
            prefixes.append(f"[AGENT MANDATE]\n{clean_prompt}")
        else:
            prefixes.append(
                "[AGENT MANDATE]\n"
                "Treat the [AGENT SYSTEM PROMPT] ... [END AGENT SYSTEM PROMPT] block below "
                "as your authoritative operating contract.\n\n"
                f"[AGENT SYSTEM PROMPT]\n{clean_prompt}\n[END AGENT SYSTEM PROMPT]"
            )
    if _session_has_kirocrew_spawn_run(session) and "[KIROCREW SUBAGENT ROUTING]" not in (
        formatted_prompt
    ):
        prefixes.append(_KIROCREW_SPAWN_ROUTING_DIRECTIVE)
    if not prefixes:
        return formatted_prompt
    return "\n\n".join([*prefixes, formatted_prompt]).strip()


def _extract_wire_custom_agents(meta: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Extract `params._meta.kiro.customAgents` when explicitly provided on the wire."""
    if not isinstance(meta, dict):
        return None
    kiro_meta = meta.get("kiro")
    if not isinstance(kiro_meta, dict) or "customAgents" not in kiro_meta:
        return None
    raw_agents = kiro_meta.get("customAgents")
    if not isinstance(raw_agents, list):
        return []
    return [copy.deepcopy(entry) for entry in raw_agents if isinstance(entry, dict)]


AVAILABLE_MODES: list[dict[str, str]] = [
    {"id": "auto", "name": "Auto"},
    {"id": "approve", "name": "Approve"},
    {"id": "read-only", "name": "Read Only"},
]
VALID_MODE_IDS: frozenset[str] = frozenset(m["id"] for m in AVAILABLE_MODES)
REASONING_EFFORT_CONFIG_IDS: frozenset[str] = frozenset(
    {"effortLevel", "reasoning_effort", "thought_level", "thinkingLevel"}
)

AVAILABLE_MODELS: list[dict[str, str]] = list(FALLBACK_MODELS)

_MIME_TO_EXT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg",
}

_IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff"}
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_USER_MESSAGE_BLOCK_RE = re.compile(r"<user_message>\s*(.*?)\s*</user_message>", re.DOTALL)


def _ensure_private_images_dir(image_dir: Path | None = None) -> Path:
    """Create and return a 0700 private directory for staged prompt/steer images."""
    target = (
        Path(image_dir)
        if image_dir is not None
        else (Path.home() / ".config" / "agy-acp" / "images")
    )
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        target.chmod(0o700)
    return target


def _stage_local_image_file(
    src_path: Path,
    *,
    cwd: str | None = None,
    image_dir: Path | None = None,
) -> Path | None:
    """Copy a `cwd`-contained local image file into a 0700 private directory with mode 0600."""
    if not cwd or not cwd.strip():
        return None
    with contextlib.suppress(OSError, RuntimeError):
        cwd_root = Path(cwd).resolve()
        candidate = src_path if src_path.is_absolute() else (Path(cwd) / src_path)
        resolved_src = candidate.resolve(strict=True)
        if not resolved_src.is_relative_to(cwd_root) or not resolved_src.is_file():
            return None
        ext = resolved_src.suffix.lower().lstrip(".") or "png"
        img_dir = _ensure_private_images_dir(image_dir)
        dest_path = img_dir / f"img-{secrets.token_hex(8)}.{ext}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(dest_path, flags, 0o600)
        try:
            with (
                resolved_src.open("rb") as src_file,
                os.fdopen(fd, "wb", closefd=False) as dst_file,
            ):
                shutil.copyfileobj(src_file, dst_file)
        finally:
            os.close(fd)
        dest_path.chmod(0o600)
        return dest_path
    return None


def _extract_markdown_local_images(
    text: str,
    *,
    cwd: str | None = None,
    image_dir: Path | None = None,
) -> tuple[list[str], list[Path]]:
    """Find `cwd`-contained markdown image links (`![...](path.png)`) and stage them privately."""
    if not cwd or not cwd.strip():
        return [], []
    hints: list[str] = []
    staged: list[Path] = []
    seen_targets: set[str] = set()
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        raw_target = match.group(1).strip().strip("<>\"'")
        if raw_target.startswith("file://"):
            raw_target = raw_target[len("file://") :]
        if not raw_target or raw_target in seen_targets:
            continue
        candidate = Path(raw_target)
        if candidate.suffix.lower() not in _IMAGE_EXTS:
            continue
        seen_targets.add(raw_target)
        staged_path = _stage_local_image_file(candidate, cwd=cwd, image_dir=image_dir)
        if staged_path is not None:
            staged.append(staged_path)
            hints.append(
                f"[User attached an image: {staged_path} — inspect it with view_file if needed]"
            )
    return hints, staged


def _format_steer_message(
    raw_message: str,
    *,
    cwd: str | None = None,
    image_dir: Path | None = None,
) -> tuple[str, list[Path]]:
    """Unwrap `<user_message>` steer text and stage any `cwd`-contained markdown images."""
    matches = _USER_MESSAGE_BLOCK_RE.findall(raw_message)
    body = "\n\n".join(m.strip() for m in matches if m.strip()) if matches else raw_message.strip()
    hints, staged_images = _extract_markdown_local_images(body, cwd=cwd, image_dir=image_dir)
    parts = [body] if body else []
    parts.extend(hints)
    return "\n\n".join(parts).strip(), staged_images


def _write_dsh_gate_marker_if_configured() -> None:
    """Write `<nonce>\\tloaded\\n` with 0600 permissions when DSH gate env vars are set."""
    nonce = os.environ.get("KIROCREW_DSH_GATE_SESSION")
    marker_raw = os.environ.get("KIROCREW_DSH_GATE_MARKER")
    if not nonce or not marker_raw:
        return
    marker_path = Path(marker_raw)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, f"{nonce}\tloaded\n".encode())
    finally:
        os.close(fd)
    marker_path.chmod(0o600)


def _format_prompt_blocks(
    prompt_blocks: list[dict[str, Any]],
    *,
    cwd: str | None = None,
    image_dir: Path | None = None,
) -> tuple[str, list[Path]]:
    """Convert ACP prompt content blocks into a text prompt and temporary image files."""
    parts: list[str] = []
    temp_images: list[Path] = []

    for block in prompt_blocks:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type", ""))
        if btype == "text":
            text = str(block.get("text", ""))
            if text:
                parts.append(text)
                md_hints, md_images = _extract_markdown_local_images(
                    text,
                    cwd=cwd,
                    image_dir=image_dir,
                )
                parts.extend(md_hints)
                temp_images.extend(md_images)
        elif btype == "resource_link":
            uri = str(block.get("uri", ""))
            name = str(block.get("name") or uri)
            parts.append(f"[Resource: {name} ({uri})]")
        elif btype == "resource":
            res = block.get("resource")
            if isinstance(res, dict) and "text" in res:
                uri = str(res.get("uri", "embedded"))
                parts.append(f"--- {uri} ---\n{res['text']}\n---")
        elif btype == "image":
            raw_b64 = str(block.get("data", ""))
            mime_type = str(block.get("mimeType", "image/png")).lower()
            ext = _MIME_TO_EXT.get(mime_type, "png")
            if raw_b64:
                with contextlib.suppress(ValueError, OSError):
                    decoded = base64.b64decode(raw_b64)
                    img_dir = _ensure_private_images_dir(image_dir)
                    img_path = img_dir / f"img-{secrets.token_hex(8)}.{ext}"
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                    fd = os.open(img_path, flags, 0o600)
                    try:
                        os.write(fd, decoded)
                    finally:
                        os.close(fd)
                    img_path.chmod(0o600)
                    temp_images.append(img_path)
                    parts.append(
                        f"[User attached an image: {img_path} — "
                        "inspect it with view_file if needed]"
                    )

    return "\n\n".join(parts).strip(), temp_images


def _build_config_options(
    *,
    model: str,
    mode: str,
    reasoning_effort: str,
    available_models: list[dict[str, str]] | None = None,
    wire_adapter: EngineWireAdapter | None = None,
) -> list[dict[str, Any]]:
    """Build ACP configOptions array for the given session state."""
    adapter = wire_adapter or V2KiroWireAdapter()
    models = list(available_models or AVAILABLE_MODELS)
    if model and model != "auto" and all(m.get("modelId") != model for m in models):
        models.insert(0, {"modelId": model, "name": model, "description": model})
    return adapter.build_config_options(
        models=models,
        current_model=model,
        current_mode=mode,
        current_effort=reasoning_effort,
    )


def _agent_mcp_mapping_to_list(
    raw_mcp: Mapping[str, Any],
    *,
    origin: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split `CustomAgentSpec.mcp_servers` into enabled server dicts and disabled names."""
    enabled: list[dict[str, Any]] = []
    disabled: list[str] = []
    for srv_name, srv_cfg in raw_mcp.items():
        if not isinstance(srv_name, str) or not srv_name or not isinstance(srv_cfg, Mapping):
            continue
        if srv_cfg.get("disabled") is True:
            disabled.append(srv_name)
            continue
        entry = {"name": srv_name, **dict(srv_cfg)}
        entry.setdefault(
            "_meta",
            {"kiro": {"resource": {"resourceType": "mcpServer", "source": {"origin": origin}}}},
        )
        enabled.append(entry)
    return enabled, disabled


class SessionManager:
    """Manages concurrent ACP sessions backed by SQLite storage and `agy` CLI subprocesses."""

    def __init__(
        self,
        *,
        save_dir: Path | None = None,
        state_store: SqliteStateStoreProvider | None = None,
        cli_provider: AgyCliProvider | None = None,
        default_agent: str | None = None,
        default_model: str | None = None,
        wire_adapter: EngineWireAdapter | None = None,
        mcp_prober: McpCatalogProber | None = None,
    ) -> None:
        self._store = state_store or (
            cli_provider.state_store
            if cli_provider is not None
            else SqliteStateStoreProvider(save_dir)
        )
        self._cli = cli_provider or AgyCliProvider(state_store=self._store)
        self._default_agent = default_agent
        self._default_model = default_model
        self._wire_adapter: EngineWireAdapter = wire_adapter or V2KiroWireAdapter()
        self._mcp_prober: McpCatalogProber = mcp_prober or McpCatalogProber()
        self._cancel_scopes: dict[str, anyio.CancelScope] = {}
        self._active_cancel_scopes: dict[str, anyio.CancelScope] = self._cancel_scopes
        self._turn_locks: dict[str, anyio.Lock] = {}
        self._mcp_status_cache: dict[str, dict[str, McpServerStatus]] = {}

    def _private_images_dir(self) -> Path:
        """Return a 0700 private image staging directory colocated with the state store."""
        return _ensure_private_images_dir(self._store.db_path.parent / "images")

    @property
    def state_store(self) -> SqliteStateStoreProvider:
        """Return the underlying `SqliteStateStoreProvider`."""
        return self._store

    @property
    def cli_provider(self) -> AgyCliProvider:
        """Return the underlying `AgyCliProvider` instance."""
        return self._cli

    @property
    def wire_adapter(self) -> EngineWireAdapter:
        """Return the active `EngineWireAdapter` strategy."""
        return self._wire_adapter

    @property
    def mcp_prober(self) -> McpCatalogProber:
        """Return the active `McpCatalogProber` instance."""
        return self._mcp_prober

    def _build_config_options_for_state(
        self,
        *,
        model: str,
        mode: str,
        reasoning_effort: str,
        available_models: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        models = available_models or self._store.load_models() or AVAILABLE_MODELS
        return _build_config_options(
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
            available_models=models,
            wire_adapter=self._wire_adapter,
        )

    @staticmethod
    def _build_config_options(
        *,
        model: str,
        mode: str,
        reasoning_effort: str,
    ) -> list[dict[str, Any]]:
        """Build ACP configOptions array for the given session state."""
        return _build_config_options(
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
        )

    def _user_agents_dir(self) -> Path | None:
        if self._cli.is_custom_store:
            return self._store.db_path.parent / ".kiro_user_agents"
        return None

    def get_custom_agent_registry(
        self,
        *,
        cwd: str | None,
        custom_agents: Sequence[Mapping[str, Any]] | None = None,
    ) -> CustomAgentRegistry:
        """Load `CustomAgentRegistry` across project, user, and client wire scopes."""
        return CustomAgentRegistry.load(
            cwd=cwd or None,
            wire_custom_agents=custom_agents,
            user_agents_dir=self._user_agents_dir(),
        )

    def get_active_custom_agent(self, session_id: str) -> CustomAgentSpec | None:
        """Return the active `CustomAgentSpec` for `session_id`, if any."""
        session = self._store.get_session(session_id)
        if session is None:
            return None
        agent_id = session.get("agent")
        if not isinstance(agent_id, str) or not agent_id:
            return None
        registry = self.get_custom_agent_registry(
            cwd=str(session.get("cwd") or ""),
            custom_agents=session.get("custom_agents") or [],
        )
        spec = registry.get(agent_id)
        if spec is not None:
            return spec
        if session.get("agentPermissions") is not None or session.get("agentPrompt"):
            raw_tools = session.get("agentTools")
            return CustomAgentSpec(
                agent_id=agent_id,
                name=agent_id,
                prompt=str(session.get("agentPrompt") or ""),
                tools=tuple(raw_tools) if isinstance(raw_tools, list) else None,
                excluded_tools=tuple(session.get("agentExcludedTools") or ()),
                allowed_tools=tuple(session.get("agentAllowedTools") or ()),
                permissions=session.get("agentPermissions"),
            )
        return None

    def _build_descriptor(
        self,
        state: dict[str, Any],
        *,
        available_models: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build an ACP `session/new` or `session/load` descriptor via `self._wire_adapter`."""
        raw_model = str(state.get("model", DEFAULT_MODEL_ID))
        mode = str(state.get("mode", "auto"))
        active_agent = state.get("agent") or self._default_agent
        current_mode_id = str(active_agent) if active_agent else mode
        reasoning_effort = str(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        models = list(available_models or AVAILABLE_MODELS)
        if raw_model and raw_model != "auto" and all(m.get("modelId") != raw_model for m in models):
            models.insert(0, {"modelId": raw_model, "name": raw_model, "description": raw_model})
        concrete_model = self._wire_adapter.resolve_concrete_model(models, raw_model)

        registry = self.get_custom_agent_registry(
            cwd=str(state.get("cwd") or ""),
            custom_agents=state.get("custom_agents") or [],
        )
        custom_specs = registry.list_agents()
        if self._wire_adapter.engine_id == "v2":
            existing_ids = {s.agent_id for s in custom_specs} | set(VALID_MODE_IDS)
            for virtual_id in (
                self._default_agent,
                str(active_agent) if active_agent else None,
                "kirocrew",
                "kirocrew-lite",
            ):
                if virtual_id and virtual_id not in existing_ids:
                    existing_ids.add(virtual_id)
                    custom_specs.append(
                        CustomAgentSpec(
                            agent_id=virtual_id,
                            name=virtual_id,
                            origin="bundled",
                        )
                    )
        elif active_agent and str(active_agent) not in (
            {s.agent_id for s in custom_specs} | set(VALID_MODE_IDS)
        ):
            custom_specs.append(
                CustomAgentSpec(
                    agent_id=str(active_agent),
                    name=str(active_agent),
                    origin="bundled",
                )
            )

        modes_list = self._wire_adapter.build_available_modes(custom_agents=custom_specs)
        return {
            "sessionId": str(state["sessionId"]),
            "modes": {
                "currentModeId": current_mode_id,
                "availableModes": modes_list,
            },
            "models": {
                "currentModelId": concrete_model,
                "availableModels": models,
            },
            "configOptions": self._wire_adapter.build_config_options(
                models=models,
                current_model=concrete_model,
                current_mode=mode,
                current_effort=reasoning_effort,
            ),
        }

    def _merge_and_sync_session_mcp(
        self,
        *,
        session_id: str,
        cwd: str,
        raw_wire_mcp: list[dict[str, Any]],
        active_agent: str | None,
        registry: CustomAgentRegistry,
        prev_mcp_servers: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """Merge session wire MCP servers with active customAgent MCP servers (session wins)."""
        wire_translated = _dedupe_mcp_servers(
            [translate_mcp_server(s) for s in raw_wire_mcp if isinstance(s, dict)]
        )
        spec = registry.get(active_agent) if active_agent else None
        agent_raw_servers: list[dict[str, Any]] = []
        disabled_names: list[str] = []

        if spec is not None:
            if spec.mcp_servers:
                agent_raw_servers, disabled_names = _agent_mcp_mapping_to_list(
                    spec.mcp_servers,
                    origin=spec.origin,
                )
        elif self._wire_adapter.engine_id == "v2":
            legacy_spec = load_kiro_agent_spec(
                active_agent,
                cwd=cwd,
                include_home=not self._cli.is_custom_store,
                user_agents_dir=self._user_agents_dir(),
            )
            agent_raw_servers = list(legacy_spec["mcp_servers"])
            disabled_names = list(legacy_spec["disabled_mcp_servers"])

        agent_translated = [translate_mcp_server(s) for s in agent_raw_servers]
        combined = _dedupe_mcp_servers(
            [
                *[s for s in wire_translated if s.get("disabled") is not True],
                *[
                    s
                    for s in agent_translated
                    if s.get("disabled") is not True
                    and str(s.get("name") or "") not in disabled_names
                ],
            ]
        )
        wire_names = {str(s.get("name") or "") for s in wire_translated if s.get("name")}
        prev_names = [
            str(s.get("name") or "")
            for s in (prev_mcp_servers or [])
            if isinstance(s, dict) and s.get("name")
        ]
        other_session_names = {
            str(srv.get("name") or "")
            for other in self._store.list_sessions()
            if str(other.get("sessionId") or "") != session_id
            for srv in (other.get("mcp_servers") or [])
            if isinstance(srv, dict) and srv.get("name")
        }
        combined_names = {str(s.get("name") or "") for s in combined}
        removable = [
            name
            for name in [*disabled_names, *prev_names]
            if name
            and name not in wire_names
            and name not in combined_names
            and name not in other_session_names
        ]
        return combined, wire_translated, removable

    @staticmethod
    def _reset_custom_agent_fields(target_state: dict[str, Any]) -> None:
        """Clear all custom-agent prompt, tool, permission, and model switch fields."""
        target_state["agentPrompt"] = None
        target_state["agentTools"] = None
        target_state["agentExcludedTools"] = []
        target_state["agentAllowedTools"] = []
        target_state["agentPermissions"] = None
        target_state["agentReadOnly"] = False
        target_state["switchedModel"] = None

    @staticmethod
    def _apply_custom_agent_fields(
        target_state: dict[str, Any],
        spec: CustomAgentSpec | None,
        *,
        allow_model_override: bool,
    ) -> None:
        """Populate session agent prompt/tools/permissions/model fields from `spec`."""
        SessionManager._reset_custom_agent_fields(target_state)
        if spec is None:
            return
        target_state["agentPrompt"] = spec.prompt or None
        target_state["agentTools"] = list(spec.tools) if spec.tools is not None else None
        target_state["agentExcludedTools"] = list(spec.excluded_tools)
        target_state["agentAllowedTools"] = list(spec.allowed_tools)
        target_state["agentPermissions"] = copy.deepcopy(spec.permissions)
        target_state["agentReadOnly"] = spec.tools is not None and all(
            t in _READ_ONLY_TOOL_NAMES for t in spec.tools
        )
        candidate_model = (
            spec.model if spec.model and spec.model.strip().lower() != "auto" else None
        )
        if candidate_model and allow_model_override:
            target_state["model"] = candidate_model
            target_state["switchedModel"] = candidate_model
        else:
            target_state["switchedModel"] = None

    def _require_session(self, session_id: str) -> dict[str, Any]:
        """Fetch session state for `session_id` directly from SQLite or raise KeyError."""
        state = self._store.get_session(session_id)
        if state is None:
            raise KeyError(f"Session '{session_id}' not found")
        return state

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return an isolated copy of the SQLite-persisted state for `session_id`."""
        return copy.deepcopy(self._require_session(session_id))

    def list_sessions(self, *, cwd: str | None = None) -> list[dict[str, Any]]:
        """Return all persisted sessions from SQLite, optionally filtered by `cwd`."""
        return self._store.list_sessions(cwd=cwd)

    def get_session_mcp_server_names(self, session_id: str) -> list[str]:
        """Return ordered unique MCP server names configured for `session_id`."""
        session = self._store.get_session(session_id)
        if session is None:
            return []
        raw_servers = session.get("mcp_servers")
        if not isinstance(raw_servers, list):
            return []
        names: list[str] = []
        for entry in raw_servers:
            if isinstance(entry, dict) and entry.get("disabled") is not True:
                name = str(entry.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    def _resolve_session_probe_declarations(
        self,
        session: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build the ordered MCP server declarations to probe for `session`."""
        cwd = str(session.get("cwd") or "")
        registry = self.get_custom_agent_registry(
            cwd=cwd,
            custom_agents=session.get("custom_agents") or [],
        )
        active_agent = session.get("agent")
        spec = registry.get(str(active_agent)) if active_agent else None

        raw_wire = session.get("wire_mcp_servers")
        wire_entries = (
            [dict(s) for s in raw_wire if isinstance(s, dict) and s.get("name")]
            if isinstance(raw_wire, list)
            else []
        )
        agent_entries: list[dict[str, Any]] = []
        if spec is not None and spec.mcp_servers:
            for srv_name, srv_cfg in spec.mcp_servers.items():
                if not isinstance(srv_name, str) or not isinstance(srv_cfg, Mapping):
                    continue
                entry = {"name": srv_name, **dict(srv_cfg)}
                entry.setdefault(
                    "_meta",
                    {
                        "kiro": {
                            "resource": {
                                "resourceType": "mcpServer",
                                "source": {"origin": spec.origin},
                            }
                        }
                    },
                )
                agent_entries.append(entry)

        merged_declarations = _dedupe_mcp_servers([*wire_entries, *agent_entries])
        if not merged_declarations:
            fallback = session.get("mcp_servers")
            if isinstance(fallback, list):
                merged_declarations = [
                    dict(s) for s in fallback if isinstance(s, dict) and s.get("name")
                ]
        return merged_declarations

    def get_session_probe_server_names(self, session_id: str) -> list[str]:
        """Return ordered MCP server names in the full probe roster for `session_id`."""
        session = self._store.get_session(session_id)
        if session is None:
            return []
        return [
            str(s.get("name") or "").strip()
            for s in self._resolve_session_probe_declarations(session)
            if str(s.get("name") or "").strip()
        ]

    async def probe_session_mcp_servers(
        self,
        session_id: str,
        server_filter: str | None = None,
    ) -> list[McpServerStatus]:
        """Probe the effective MCP server roster for `session_id` via `McpCatalogProber`."""
        session = self._store.get_session(session_id)
        if session is None:
            return []

        cwd = str(session.get("cwd") or "")
        merged_declarations = self._resolve_session_probe_declarations(session)
        if server_filter:
            merged_declarations = [
                s for s in merged_declarations if str(s.get("name") or "") == server_filter
            ]
        if not merged_declarations:
            return []

        session_env = _extract_session_mcp_env(
            session,
            session_id=session_id,
            state_db_path=self._store.db_path,
        )
        return await self._mcp_prober.probe_servers(
            merged_declarations,
            session_env=session_env,
            cwd=cwd or None,
        )

    async def create_session(
        self,
        *,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new ACP session (`session/new`) in SQLite and return its descriptor."""
        initial_mode = "approve" if os.environ.get("GOOSE_MODE") else "auto"
        cached_models = self._store.load_models() or None
        if self._default_model:
            initial_model = self._default_model
        elif cached_models and DEFAULT_MODEL_ID not in {m.get("modelId") for m in cached_models}:
            initial_model = str(cached_models[0].get("modelId") or DEFAULT_MODEL_ID)
        else:
            initial_model = DEFAULT_MODEL_ID

        isolated_mcp = copy.deepcopy(mcp_servers) if mcp_servers is not None else []
        isolated_meta = copy.deepcopy(meta) if meta is not None else {}
        wire_custom_agents = _extract_wire_custom_agents(isolated_meta) or []

        auth_cfg: dict[str, Any] = {}
        if self._default_agent:
            auth_cfg["kiro_agent"] = self._default_agent
        if self._wire_adapter.engine_id == "v3":
            auth_cfg["_engine_id"] = "v3"

        sid = await self._cli.start_or_resume_agent(
            session_id=None,
            cwd=cwd,
            mcp_servers=isolated_mcp,
            model=initial_model,
            mode=initial_mode,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            auth_config=auth_cfg or None,
        )
        synced_state = self._store.get_session(sid) or {}
        registry = self.get_custom_agent_registry(cwd=cwd, custom_agents=wire_custom_agents)
        combined_mcp, _, removable = self._merge_and_sync_session_mcp(
            session_id=sid,
            cwd=cwd,
            raw_wire_mcp=isolated_mcp,
            active_agent=self._default_agent,
            registry=registry,
        )

        draft_state: dict[str, Any] = {
            **synced_state,
            "sessionId": sid,
            "agyConversationId": None,
            "cwd": cwd,
            "mcp_servers": combined_mcp,
            "wire_mcp_servers": isolated_mcp,
            "custom_agents": wire_custom_agents,
            "meta": isolated_meta,
            "mode": initial_mode,
            "agent": self._default_agent,
            "model": initial_model,
            "modelPinnedByUser": bool(self._default_model),
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "cancelled": False,
        }
        if self._default_agent:
            self._apply_custom_agent_fields(
                draft_state,
                registry.get(self._default_agent),
                allow_model_override=not bool(self._default_model),
            )

        state = self._store.upsert_session(draft_state)
        if combined_mcp or removable:
            self._cli.sync_mcp_servers(combined_mcp, remove_servers=removable, session_id=sid)
        _write_dsh_gate_marker_if_configured()
        return self._build_descriptor(state, available_models=cached_models)

    async def load_session(
        self,
        *,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resume an existing ACP session (`session/load`) from SQLite storage."""
        await anyio.sleep(0)
        state = self._require_session(session_id)

        effective_wire_mcp = (
            copy.deepcopy(mcp_servers)
            if mcp_servers is not None
            else copy.deepcopy(state.get("wire_mcp_servers", state.get("mcp_servers", [])))
        )
        wire_custom_agents = _extract_wire_custom_agents(meta)
        effective_custom_agents = (
            wire_custom_agents
            if wire_custom_agents is not None
            else copy.deepcopy(state.get("custom_agents", []))
        )
        effective_meta = (
            copy.deepcopy(meta) if meta is not None else copy.deepcopy(state.get("meta", {}))
        )

        model = str(state.get("model", self._default_model or DEFAULT_MODEL_ID))
        mode = str(state.get("mode", "auto"))
        agent = state.get("agent") or self._default_agent
        reasoning_effort = str(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        auth_cfg: dict[str, Any] = dict(state.get("auth_config") or {})
        if agent:
            auth_cfg["kiro_agent"] = str(agent)
        if self._wire_adapter.engine_id == "v3":
            auth_cfg["_engine_id"] = "v3"

        await self._cli.start_or_resume_agent(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=effective_wire_mcp,
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
            auth_config=auth_cfg or None,
        )
        registry = self.get_custom_agent_registry(cwd=cwd, custom_agents=effective_custom_agents)
        combined_mcp, _, removable = self._merge_and_sync_session_mcp(
            session_id=session_id,
            cwd=cwd,
            raw_wire_mcp=effective_wire_mcp,
            active_agent=str(agent) if agent else None,
            registry=registry,
            prev_mcp_servers=state.get("mcp_servers", []),
        )

        draft_state: dict[str, Any] = {
            **copy.deepcopy(state),
            "sessionId": session_id,
            "cwd": cwd,
            "mcp_servers": combined_mcp,
            "wire_mcp_servers": effective_wire_mcp,
            "custom_agents": effective_custom_agents,
            "meta": effective_meta,
            "mode": mode,
            "agent": agent,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "cancelled": bool(state.get("cancelled", False)),
        }
        if agent and wire_custom_agents is not None:
            self._apply_custom_agent_fields(
                draft_state,
                registry.get(str(agent)),
                allow_model_override=not bool(state.get("modelPinnedByUser", False)),
            )

        restored_state = self._store.upsert_session(draft_state)
        if combined_mcp or removable:
            self._cli.sync_mcp_servers(
                combined_mcp,
                remove_servers=removable,
                session_id=session_id,
            )
        _write_dsh_gate_marker_if_configured()
        cached_models = self._store.load_models() or None
        return self._build_descriptor(restored_state, available_models=cached_models)

    def _is_valid_agent_mode(
        self,
        mode_id: str,
        *,
        registry: CustomAgentRegistry,
    ) -> bool:
        """Return whether `mode_id` resolves to a registered or virtual custom agent."""
        if not mode_id:
            return False
        if registry.get(mode_id) is not None:
            return True
        if mode_id == self._default_agent:
            return True
        return self._wire_adapter.engine_id == "v2" and mode_id.startswith("kirocrew")

    async def set_mode(self, *, session_id: str, mode_id: str) -> dict[str, Any]:
        """Update session approval mode or agent profile (`session/set_mode`) in SQLite."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        cwd = str(session.get("cwd", ""))
        registry = self.get_custom_agent_registry(
            cwd=cwd,
            custom_agents=session.get("custom_agents") or [],
        )

        spec = registry.get(mode_id)
        if spec is None and mode_id in VALID_MODE_IDS:
            session["mode"] = mode_id
            session["agent"] = None
            self._reset_custom_agent_fields(session)
            raw_wire = session.get("wire_mcp_servers", [])
            combined, _, removable = self._merge_and_sync_session_mcp(
                session_id=session_id,
                cwd=cwd,
                raw_wire_mcp=raw_wire if isinstance(raw_wire, list) else [],
                active_agent=None,
                registry=registry,
                prev_mcp_servers=session.get("mcp_servers", []),
            )
            session["mcp_servers"] = combined
            self._store.upsert_session(session)
            if combined or removable:
                self._cli.sync_mcp_servers(
                    combined, remove_servers=removable, session_id=session_id
                )
            return {}

        if not self._is_valid_agent_mode(mode_id, registry=registry):
            raise ValueError(f"Mode '{mode_id}' not found")

        session["agent"] = mode_id
        self._reset_custom_agent_fields(session)
        raw_wire = session.get("wire_mcp_servers", [])
        combined, _, removable = self._merge_and_sync_session_mcp(
            session_id=session_id,
            cwd=cwd,
            raw_wire_mcp=raw_wire if isinstance(raw_wire, list) else [],
            active_agent=mode_id,
            registry=registry,
            prev_mcp_servers=session.get("mcp_servers", []),
        )
        session["mcp_servers"] = combined

        model_pinned = bool(session.get("modelPinnedByUser", False))
        if spec is not None:
            self._apply_custom_agent_fields(
                session,
                spec,
                allow_model_override=not model_pinned,
            )
        elif self._wire_adapter.engine_id == "v2":
            legacy_spec = load_kiro_agent_spec(
                mode_id,
                cwd=cwd,
                include_home=not self._cli.is_custom_store,
                user_agents_dir=self._user_agents_dir(),
            )
            session["agentPrompt"] = legacy_spec.get("prompt")
            session["agentTools"] = legacy_spec.get("tools")
            session["agentReadOnly"] = bool(legacy_spec.get("read_only"))
            pinned_model = legacy_spec.get("model")
            if isinstance(pinned_model, str) and pinned_model and not model_pinned:
                session["model"] = pinned_model
                session["switchedModel"] = pinned_model

        self._store.upsert_session(session)
        if combined or removable:
            self._cli.sync_mcp_servers(combined, remove_servers=removable, session_id=session_id)
        return {}

    async def compact_session(
        self,
        *,
        session_id: str,
        instructions: str = "",
    ) -> dict[str, Any]:
        """Summarize and reset `session_id` conversation history for compaction."""
        session = self._store.get_session(session_id)
        if session is None:
            return {
                "status": "completed",
                "summary": "Context is already empty.",
                "contextTokens": 0,
                "inputTokens": 1,
                "outputTokens": 1,
            }

        prev_tokens = int(session.get("contextTokens", 0))
        agy_conv_id = session.get("agyConversationId")
        if not agy_conv_id:
            existing_summary = session.get("compactedSummary")
            if isinstance(existing_summary, str) and existing_summary.strip():
                summary = existing_summary.strip()
                compacted_tokens = max(1, len(summary.split()) * 2)
            else:
                summary = "Context is already empty."
                compacted_tokens = 0
            session["contextTokens"] = compacted_tokens
            session["reinjectAgentPrompt"] = True
            self._store.upsert_session(session)
            return {
                "status": "completed",
                "summary": summary,
                "contextTokens": compacted_tokens,
                "inputTokens": 1,
                "outputTokens": max(1, len(summary.split())),
            }

        compact_prompt = (
            "Summarize our conversation so far into a concise handoff summary "
            "preserving goals, key decisions, modified files, and next steps."
        )
        if instructions.strip():
            compact_prompt = f"{compact_prompt}\n\nAdditional instructions: {instructions.strip()}"

        cwd = str(session.get("cwd", ""))
        model = str(session.get("model", DEFAULT_MODEL_ID))
        try:
            outcome = await self._cli.stream_prompt(
                cwd=cwd,
                prompt=compact_prompt,
                session_id=session_id,
                conversation_id=str(agy_conv_id),
                model=model,
                mode="read-only",
                effort="low",
                env=_extract_session_mcp_env(session, state_db_path=self._store.db_path),
            )
        except Exception as exc:
            return {
                "status": "failed",
                "summary": str(exc) or "Compaction failed.",
                "contextTokens": prev_tokens,
                "inputTokens": 1,
                "outputTokens": 1,
            }

        if outcome.stop_reason == "cancelled":
            return {
                "status": "failed",
                "summary": "Compaction cancelled.",
                "contextTokens": prev_tokens,
                "inputTokens": 1,
                "outputTokens": 1,
            }

        summary = (outcome.response or "").strip() or "Conversation context compacted."
        compacted_tokens = max(1, len(summary.split()) * 2)
        session["agyConversationId"] = None
        session["compactedSummary"] = summary
        session["contextTokens"] = compacted_tokens
        session["reinjectAgentPrompt"] = True
        self._store.upsert_session(session)
        return {
            "status": "completed",
            "summary": summary,
            "contextTokens": compacted_tokens,
            "inputTokens": max(1, int(outcome.usage.get("input_tokens", 1))),
            "outputTokens": max(1, int(outcome.usage.get("output_tokens", len(summary.split())))),
        }

    async def set_model(self, *, session_id: str, model_id: str) -> dict[str, Any]:
        """Update active model (`session/set_model`) in SQLite and pin user override."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        session["model"] = model_id
        session["modelPinnedByUser"] = True
        self._store.upsert_session(session)
        return {}

    async def set_config_option(
        self,
        *,
        session_id: str,
        config_id: str,
        value: str,
    ) -> dict[str, Any]:
        """Update session configuration option (`session/set_config_option`) in SQLite."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        if config_id == "model":
            session["model"] = value
            session["modelPinnedByUser"] = True
        elif config_id == "mode":
            if value not in VALID_MODE_IDS:
                raise ValueError(f"Mode '{value}' not found")
            session["mode"] = value
        elif config_id in REASONING_EFFORT_CONFIG_IDS:
            if value not in self._wire_adapter.supported_effort_levels:
                supported = ", ".join(self._wire_adapter.supported_effort_levels)
                raise ValueError(
                    f"Unsupported effort level '{value}' (expected one of: {supported})"
                )
            session["reasoning_effort"] = value
        else:
            raise ValueError(f"Config option '{config_id}' not found")
        updated = self._store.upsert_session(session)
        return {
            "configOptions": self._build_config_options_for_state(
                model=str(updated.get("model", DEFAULT_MODEL_ID)),
                mode=str(updated.get("mode", "auto")),
                reasoning_effort=str(updated.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
            )
        }

    async def prompt(
        self,
        *,
        session_id: str,
        prompt_blocks: list[dict[str, Any]],
        on_thought: Callable[[str], Awaitable[None]] | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[ToolCallEvent], Awaitable[bool]] | None = None,
        on_tool_update: Callable[[ToolCallEvent], Awaitable[None]] | None = None,
        on_usage: Callable[[int], Awaitable[None]] | None = None,
        on_steer_consumed: Callable[[str], Awaitable[None]] | None = None,
        on_subagent_list: Callable[[list[dict[str, Any]]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Execute a streaming turn (`session/prompt`) via the `agy` CLI subprocess."""
        self._require_session(session_id)
        lock = self._turn_locks.setdefault(session_id, anyio.Lock())
        async with lock:
            if self.consume_cancellation(session_id):
                return {"stopReason": "cancelled"}

            session = self._require_session(session_id)
            cwd = str(session.get("cwd", ""))
            model = str(session.get("model", DEFAULT_MODEL_ID))
            agent_read_only = bool(session.get("agentReadOnly"))
            mode = str(session.get("mode", "auto"))
            if agent_read_only:
                mode = "read-only"
            elif os.environ.get("GOOSE_MODE") == "approve":
                mode = "approve"
            skip_permissions = mode == "auto" and not agent_read_only
            effort = str(session.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
            agy_conv_id = session.get("agyConversationId")
            conversation_id = str(agy_conv_id) if agy_conv_id else None
            compacted_summary = session.get("compactedSummary")
            reinjected_prompt = bool(session.get("reinjectAgentPrompt"))

            formatted_prompt, temp_images = _format_prompt_blocks(
                prompt_blocks,
                cwd=cwd,
                image_dir=self._private_images_dir(),
            )
            consumed_compacted_summary = False
            if (
                conversation_id is None
                and isinstance(compacted_summary, str)
                and compacted_summary.strip()
            ):
                summary_clean = compacted_summary.strip()
                formatted_prompt = (
                    f"[Compacted Session Summary]\n{summary_clean}\n\n{formatted_prompt}"
                ).strip()
                consumed_compacted_summary = True
            formatted_prompt = _apply_session_prompt_directives(
                formatted_prompt,
                session,
                is_new_conversation=conversation_id is None,
            )
            session_env = _extract_session_mcp_env(
                session,
                session_id=session_id,
                state_db_path=self._store.db_path,
            )

            outcome = None
            try:
                with self.turn_cancel_scope(session_id) as scope:
                    outcome = await self._cli.stream_prompt(
                        cwd=cwd,
                        prompt=formatted_prompt,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        model=model,
                        mode=mode,
                        effort=effort,
                        skip_permissions=skip_permissions,
                        env=session_env,
                        on_thought=on_thought,
                        on_text=on_text,
                        on_tool_call=on_tool_call,
                        on_tool_update=on_tool_update,
                        on_usage=on_usage,
                        on_steer_consumed=on_steer_consumed,
                        on_subagent_list=on_subagent_list,
                    )
            finally:
                for img_path in temp_images:
                    with contextlib.suppress(OSError):
                        img_path.unlink(missing_ok=True)

            if (
                scope.cancel_called
                or self.consume_cancellation(session_id)
                or outcome is None
                or outcome.stop_reason == "cancelled"
            ):
                return {"stopReason": "cancelled"}

            ctx_tokens = int(outcome.usage.get("context_tokens", 0))
            latest = self._store.get_session(session_id)
            if latest is not None and (
                outcome.conversation_id
                or outcome.resolved_model
                or ctx_tokens > 0
                or consumed_compacted_summary
                or reinjected_prompt
            ):
                if outcome.conversation_id:
                    latest["agyConversationId"] = outcome.conversation_id
                if consumed_compacted_summary:
                    latest["compactedSummary"] = None
                if reinjected_prompt:
                    latest["reinjectAgentPrompt"] = False
                if (
                    outcome.resolved_model
                    and not latest.get("modelPinnedByUser")
                    and latest.get("model") in {DEFAULT_MODEL_ID, "auto", ""}
                ):
                    latest["model"] = outcome.resolved_model
                if ctx_tokens > 0:
                    latest["contextTokens"] = ctx_tokens
                self._store.upsert_session(latest)

            return {
                "stopReason": outcome.stop_reason,
                "response": outcome.response,
                "usage": outcome.usage,
                "conversationId": outcome.conversation_id,
            }

    def is_cancelled(self, session_id: str) -> bool:
        """Return whether `session_id` has a pending cancellation in SQLite."""
        session = self._store.get_session(session_id)
        if session is not None:
            return bool(session.get("cancelled", False))
        return bool(self._store.get_kv(f"cancelled:{session_id}"))

    def consume_cancellation(self, session_id: str) -> bool:
        """Clear and return whether `session_id` was marked cancelled in SQLite."""
        was_cancelled = self.is_cancelled(session_id)
        if not was_cancelled:
            return False
        session = self._store.get_session(session_id)
        if session is not None:
            session["cancelled"] = False
            self._store.upsert_session(session)
        self._store.set_kv(f"cancelled:{session_id}", False)
        return True

    def register_cancel_scope(self, session_id: str, scope: anyio.CancelScope) -> None:
        """Register an active `anyio.CancelScope` for `session_id`."""
        self._cancel_scopes[session_id] = scope

    def unregister_cancel_scope(
        self,
        session_id: str,
        scope: anyio.CancelScope | None = None,
    ) -> None:
        """Unregister the active `anyio.CancelScope` for `session_id`."""
        current = self._cancel_scopes.get(session_id)
        if current is not None and (scope is None or current is scope):
            self._cancel_scopes.pop(session_id, None)

    @contextlib.contextmanager
    def turn_cancel_scope(self, session_id: str) -> Generator[anyio.CancelScope]:
        """Context manager registering an `anyio.CancelScope` for an active turn."""
        with anyio.CancelScope() as scope:
            self.register_cancel_scope(session_id, scope)
            try:
                yield scope
            finally:
                self.unregister_cancel_scope(session_id, scope)

    async def steer(
        self,
        *,
        session_id: str,
        message: str | None = None,
        prompt_blocks: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Queue a mid-turn steer (`_session/steer` or `_session/steering`) into an active turn."""
        if not self._cli.is_turn_active(session_id):
            return False

        session = self._store.get_session(session_id)
        cwd = str(session.get("cwd") or "") if session is not None else None
        img_dir = self._private_images_dir()

        if message is not None:
            formatted_prompt, temp_files = _format_steer_message(
                message,
                cwd=cwd,
                image_dir=img_dir,
            )
            raw_message = message
        else:
            formatted_prompt, temp_files = _format_prompt_blocks(
                prompt_blocks or [],
                cwd=cwd,
                image_dir=img_dir,
            )
            raw_message = formatted_prompt

        if not formatted_prompt or not formatted_prompt.strip():
            for img_path in temp_files:
                with contextlib.suppress(OSError):
                    img_path.unlink(missing_ok=True)
            return False

        queued = await self._cli.steer_session(
            session_id,
            raw_message=raw_message,
            prompt=formatted_prompt,
            temp_files=temp_files,
        )
        if not queued:
            for img_path in temp_files:
                with contextlib.suppress(OSError):
                    img_path.unlink(missing_ok=True)
        return queued

    async def cancel(self, *, session_id: str) -> bool:
        """Cancel an in-flight prompt turn (`session/cancel`) and persist in SQLite."""
        await self._cli.cancel_session(session_id)
        session = self._store.get_session(session_id)
        if session is not None:
            session["cancelled"] = True
            self._store.upsert_session(session)
        else:
            self._store.set_kv(f"cancelled:{session_id}", True)
        scope = self._cancel_scopes.get(session_id)
        if scope is not None:
            scope.cancel()
        return True

    async def evict_session(self, *, session_id: str) -> dict[str, Any]:
        """Evict resident runtime handles for `session_id` while preserving SQLite state & turns."""
        return await self.close_session(session_id=session_id, delete_persisted=False)

    async def terminate_session(
        self,
        *,
        session_id: str,
        delete_persisted: bool = False,
    ) -> dict[str, Any]:
        """Terminate `session_id` and optionally delete persisted SQLite session + turns."""
        return await self.close_session(
            session_id=session_id,
            delete_persisted=delete_persisted,
        )

    async def close_session(
        self,
        *,
        session_id: str,
        delete_persisted: bool = False,
    ) -> dict[str, Any]:
        """Close and clean up an active session (`session/close`, `terminate`, `delete`)."""
        await self._cli.cancel_session(session_id)
        self._cli.discard_cancelled(session_id)
        scope = self._active_cancel_scopes.pop(session_id, None) or self._cancel_scopes.pop(
            session_id, None
        )
        if scope is not None:
            scope.cancel()
        self._mcp_status_cache.pop(session_id, None)
        self._store.set_kv(f"cancelled:{session_id}", False)
        if delete_persisted:
            target = self._store.get_session(session_id)
            target_mcp_names: list[str] = []
            if target is not None:
                for key in ("mcp_servers", "wire_mcp_servers"):
                    for srv in target.get(key) or []:
                        if isinstance(srv, dict):
                            sname = str(srv.get("name") or "").strip()
                            if sname and sname not in target_mcp_names:
                                target_mcp_names.append(sname)
            other_names = {
                str(srv.get("name") or "").strip()
                for other in self._store.list_sessions()
                if str(other.get("sessionId") or "") != session_id
                for srv in [
                    *(other.get("mcp_servers") or []),
                    *(other.get("wire_mcp_servers") or []),
                ]
                if isinstance(srv, dict) and str(srv.get("name") or "").strip()
            }
            removable = [name for name in target_mcp_names if name not in other_names]
            self._store.delete_session(session_id)
            self._turn_locks.pop(session_id, None)
            if removable:
                self._cli.sync_mcp_servers([], remove_servers=removable, session_id=session_id)
        else:
            session = self._store.get_session(session_id)
            if session is not None and session.get("cancelled", False):
                session["cancelled"] = False
                self._store.upsert_session(session)
        return {}
