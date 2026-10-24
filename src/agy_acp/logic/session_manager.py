"""Session lifecycle, multi-session multiplexing, steer, and cancellation logic backed by SQLite."""

from __future__ import annotations

import base64
import contextlib
import copy
import os
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path
from typing import Any

import anyio

from agy_acp.providers.agy_cli_provider import (
    FALLBACK_MODELS,
    AgyCliProvider,
    ToolCallEvent,
)
from agy_acp.providers.state_store import SqliteStateStoreProvider

DEFAULT_MODEL_ID = "gemini-3.7-pro"
DEFAULT_REASONING_EFFORT = "medium"

AVAILABLE_MODES: list[dict[str, str]] = [
    {"id": "auto", "name": "Auto"},
    {"id": "approve", "name": "Approve"},
    {"id": "read-only", "name": "Read Only"},
]
VALID_MODE_IDS: frozenset[str] = frozenset(m["id"] for m in AVAILABLE_MODES)
REASONING_EFFORT_CONFIG_IDS: frozenset[str] = frozenset(
    {"reasoning_effort", "thought_level", "thinkingLevel"}
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
                    img_dir = Path(tempfile.gettempdir()) / "agy-acp-images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img_path = img_dir / f"img-{uuid.uuid4().hex[:12]}.{ext}"
                    fd = os.open(img_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
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
) -> list[dict[str, Any]]:
    """Build ACP configOptions array for the given session state."""
    models = available_models or AVAILABLE_MODELS
    return [
        {
            "id": "model",
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": model,
            "options": [{"value": m["modelId"], "name": m["name"]} for m in models],
        },
        {
            "id": "mode",
            "name": "Approval Mode",
            "category": "mode",
            "type": "select",
            "currentValue": mode,
            "options": [
                {"value": "read-only", "name": "Read Only"},
                {"value": "approve", "name": "Approve"},
                {"value": "auto", "name": "Default"},
            ],
        },
        {
            "id": "reasoning_effort",
            "name": "Reasoning Effort",
            "category": "thought_level",
            "type": "select",
            "currentValue": reasoning_effort,
            "options": [
                {"value": "low", "name": "Low"},
                {"value": "medium", "name": "Medium"},
                {"value": "high", "name": "High"},
            ],
        },
    ]


def _discover_agent_mode_ids(
    *,
    cwd: str | None = None,
    default_agent: str | None = None,
    active_agent: str | None = None,
) -> list[str]:
    """Discover KiroCrew agent mode IDs from spawn flags and `.kiro/agents/*.json`."""
    found: list[str] = []
    seen: set[str] = set(VALID_MODE_IDS)

    def _add(agent_id: str | None) -> None:
        if agent_id and agent_id not in seen:
            seen.add(agent_id)
            found.append(agent_id)

    _add(default_agent)
    _add(active_agent)
    _add("kirocrew")
    _add("kirocrew-lite")

    search_dirs: list[Path] = []
    if cwd:
        search_dirs.append(Path(cwd) / ".kiro" / "agents")
    with contextlib.suppress(RuntimeError, OSError):
        search_dirs.append(Path.home() / ".kiro" / "agents")

    for agents_dir in search_dirs:
        if not agents_dir.is_dir():
            continue
        with contextlib.suppress(OSError):
            for spec_file in sorted(agents_dir.glob("*.json")):
                _add(spec_file.stem)
    return found


def _build_session_descriptor(
    state: dict[str, Any],
    *,
    available_models: list[dict[str, str]] | None = None,
    default_agent: str | None = None,
) -> dict[str, Any]:
    """Build an ACP session response payload from persisted session state."""
    model = str(state.get("model", DEFAULT_MODEL_ID))
    mode = str(state.get("mode", "auto"))
    active_agent = state.get("agent") or default_agent
    current_mode_id = str(active_agent) if active_agent else mode
    reasoning_effort = str(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
    models = list(available_models or AVAILABLE_MODELS)
    if model and all(m.get("modelId") != model for m in models):
        models.insert(0, {"modelId": model, "name": model, "description": model})

    extra_modes = _discover_agent_mode_ids(
        cwd=str(state.get("cwd", "")),
        default_agent=default_agent,
        active_agent=str(active_agent) if active_agent else None,
    )
    modes_list = list(AVAILABLE_MODES) + [{"id": mid, "name": mid} for mid in extra_modes]
    return {
        "sessionId": str(state["sessionId"]),
        "modes": {
            "currentModeId": current_mode_id,
            "availableModes": modes_list,
        },
        "models": {
            "currentModelId": model,
            "availableModels": models,
        },
        "configOptions": _build_config_options(
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
            available_models=models,
        ),
    }


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
    ) -> None:
        self._store = state_store or (
            cli_provider.state_store
            if cli_provider is not None
            else SqliteStateStoreProvider(save_dir)
        )
        self._cli = cli_provider or AgyCliProvider(state_store=self._store)
        self._default_agent = default_agent
        self._default_model = default_model
        self._cancel_scopes: dict[str, anyio.CancelScope] = {}
        self._active_cancel_scopes: dict[str, anyio.CancelScope] = self._cancel_scopes

    @property
    def state_store(self) -> SqliteStateStoreProvider:
        """Return the underlying `SqliteStateStoreProvider`."""
        return self._store

    @property
    def cli_provider(self) -> AgyCliProvider:
        """Return the underlying `AgyCliProvider` instance."""
        return self._cli

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
        sid = await self._cli.start_or_resume_agent(
            session_id=None,
            cwd=cwd,
            mcp_servers=isolated_mcp,
            model=initial_model,
            mode=initial_mode,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        state = self._store.upsert_session(
            {
                "sessionId": sid,
                "agyConversationId": None,
                "cwd": cwd,
                "mcp_servers": isolated_mcp,
                "meta": isolated_meta,
                "mode": initial_mode,
                "agent": self._default_agent,
                "model": initial_model,
                "reasoning_effort": DEFAULT_REASONING_EFFORT,
                "cancelled": False,
            }
        )
        _write_dsh_gate_marker_if_configured()
        return _build_session_descriptor(
            state,
            available_models=cached_models,
            default_agent=self._default_agent,
        )

    async def load_session(
        self,
        *,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resume an existing ACP session (`session/load`) from SQLite storage."""
        await anyio.sleep(0)
        state = self._require_session(session_id)

        effective_mcp = (
            copy.deepcopy(mcp_servers)
            if mcp_servers is not None
            else copy.deepcopy(state.get("mcp_servers", []))
        )
        model = str(state.get("model", self._default_model or DEFAULT_MODEL_ID))
        mode = str(state.get("mode", "auto"))
        agent = state.get("agent") or self._default_agent
        reasoning_effort = str(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))

        await self._cli.start_or_resume_agent(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=effective_mcp,
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
        )

        restored_state = self._store.upsert_session(
            {
                **copy.deepcopy(state),
                "sessionId": session_id,
                "cwd": cwd,
                "mcp_servers": effective_mcp,
                "mode": mode,
                "agent": agent,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "cancelled": bool(state.get("cancelled", False)),
            }
        )
        _write_dsh_gate_marker_if_configured()
        cached_models = self._store.load_models() or None
        return _build_session_descriptor(
            restored_state,
            available_models=cached_models,
            default_agent=self._default_agent,
        )

    def _is_valid_agent_mode(self, mode_id: str, *, cwd: str | None = None) -> bool:
        """Return whether `mode_id` is a valid KiroCrew agent mode ID."""
        if not mode_id:
            return False
        if mode_id.startswith("kirocrew"):
            return True
        known = _discover_agent_mode_ids(
            cwd=cwd,
            default_agent=self._default_agent,
        )
        return mode_id in known

    async def set_mode(self, *, session_id: str, mode_id: str) -> dict[str, Any]:
        """Update session approval mode or agent profile (`session/set_mode`) in SQLite."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        if mode_id in VALID_MODE_IDS:
            session["mode"] = mode_id
            session["agent"] = None
        elif self._is_valid_agent_mode(mode_id, cwd=str(session.get("cwd", ""))):
            session["agent"] = mode_id
        else:
            raise ValueError(f"Mode '{mode_id}' not found")
        self._store.upsert_session(session)
        return {}

    async def set_model(self, *, session_id: str, model_id: str) -> dict[str, Any]:
        """Update active model (`session/set_model`) in SQLite."""
        await anyio.sleep(0)
        session = self._require_session(session_id)
        session["model"] = model_id
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
        elif config_id == "mode":
            if value not in VALID_MODE_IDS:
                raise ValueError(f"Mode '{value}' not found")
            session["mode"] = value
        elif config_id in REASONING_EFFORT_CONFIG_IDS:
            session["reasoning_effort"] = value
        else:
            raise ValueError(f"Config option '{config_id}' not found")
        updated = self._store.upsert_session(session)
        return {
            "configOptions": self._build_config_options(
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
    ) -> dict[str, Any]:
        """Execute a streaming turn (`session/prompt`) via the `agy` CLI subprocess."""
        if self.consume_cancellation(session_id):
            return {"stopReason": "cancelled"}

        session = self._store.get_session(session_id) or {}
        cwd = str(session.get("cwd", ""))
        model = str(session.get("model", DEFAULT_MODEL_ID))
        mode = str(session.get("mode", "auto"))
        if os.environ.get("GOOSE_MODE") == "approve":
            mode = "approve"
        effort = str(session.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        agy_conv_id = session.get("agyConversationId")
        conversation_id = str(agy_conv_id) if agy_conv_id else None

        formatted_prompt, temp_images = _format_prompt_blocks(prompt_blocks)
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
                    on_thought=on_thought,
                    on_text=on_text,
                    on_tool_call=on_tool_call,
                    on_tool_update=on_tool_update,
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

        if session and (outcome.conversation_id or outcome.resolved_model):
            if outcome.conversation_id:
                session["agyConversationId"] = outcome.conversation_id
            if outcome.resolved_model and session.get("model") in {DEFAULT_MODEL_ID, "auto", ""}:
                session["model"] = outcome.resolved_model
            self._store.upsert_session(session)

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

    async def close_session(
        self,
        *,
        session_id: str,
        delete_persisted: bool = False,
    ) -> dict[str, Any]:
        """Close and clean up an active session (`session/close`, `terminate`, `delete`)."""
        await self._cli.cancel_session(session_id)
        scope = self._active_cancel_scopes.pop(session_id, None) or self._cancel_scopes.pop(
            session_id, None
        )
        if scope is not None:
            scope.cancel()
        self._store.set_kv(f"cancelled:{session_id}", False)
        if delete_persisted:
            self._store.delete_session(session_id)
        else:
            session = self._store.get_session(session_id)
            if session is not None and session.get("cancelled", False):
                session["cancelled"] = False
                self._store.upsert_session(session)
        return {}
