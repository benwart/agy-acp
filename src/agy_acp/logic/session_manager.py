"""Session lifecycle, multi-session multiplexing, steer, and cancellation logic."""

from __future__ import annotations

import contextlib
import copy
import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import anyio

from agy_acp.providers.antigravity_sdk import AntigravitySdkProvider

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

AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "modelId": "gemini-3.7-pro",
        "name": "Gemini 3.7 Pro",
        "description": "Google Antigravity Pro model",
    },
    {
        "modelId": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "description": "Google Antigravity Flash model",
    },
]


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


def _build_config_options(*, model: str, mode: str, reasoning_effort: str) -> list[dict[str, Any]]:
    """Build ACP configOptions array for the given session state."""
    return [
        {
            "id": "model",
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": model,
            "options": [{"value": m["modelId"], "name": m["name"]} for m in AVAILABLE_MODELS],
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


def _build_session_descriptor(state: dict[str, Any]) -> dict[str, Any]:
    """Build an ACP session response payload from internal session state."""
    model = str(state.get("model", DEFAULT_MODEL_ID))
    mode = str(state.get("mode", "auto"))
    reasoning_effort = str(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
    return {
        "sessionId": str(state["sessionId"]),
        "modes": {
            "currentModeId": mode,
            "availableModes": list(AVAILABLE_MODES),
        },
        "models": {
            "currentModelId": model,
            "availableModels": list(AVAILABLE_MODELS),
        },
        "configOptions": _build_config_options(
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
        ),
    }


class SessionManager:
    """Manages concurrent ACP sessions backed by google.antigravity Agent trajectories."""

    def __init__(
        self,
        *,
        save_dir: Path | None = None,
        sdk_provider: AntigravitySdkProvider | None = None,
    ) -> None:
        self._save_dir = save_dir
        self._sdk = sdk_provider or AntigravitySdkProvider(save_dir=save_dir)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._cancelled_sessions: set[str] = set()
        self._cancel_scopes: dict[str, anyio.CancelScope] = {}
        self._active_cancel_scopes: dict[str, anyio.CancelScope] = self._cancel_scopes

    def _persist_session(self, sid: str, state: dict[str, Any]) -> None:
        """Persist session state to `self._save_dir` when configured."""
        if self._save_dir is None:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        session_file = self._save_dir / f"{sid}.json"
        session_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_persisted_session(self, session_id: str) -> dict[str, Any] | None:
        """Load persisted session state from `self._save_dir` if available."""
        if self._save_dir is None:
            return None
        session_file = self._save_dir / f"{session_id}.json"
        if not session_file.is_file():
            return None
        loaded = json.loads(session_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return None
        return loaded

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

    def _require_session_mut(self, session_id: str) -> dict[str, Any]:
        """Return mutable session state for `session_id`, loading from disk if needed."""
        state = self._sessions.get(session_id)
        if state is None:
            state = self._load_persisted_session(session_id)
            if state is not None:
                self._sessions[session_id] = state
        if state is None:
            raise KeyError(f"Session '{session_id}' not found")
        return state

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return an isolated copy of the state for `session_id`."""
        return copy.deepcopy(self._require_session_mut(session_id))

    async def create_session(
        self,
        *,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new ACP session (`session/new`) and return its descriptor."""
        initial_mode = "approve" if os.environ.get("GOOSE_MODE") else "auto"
        isolated_mcp = copy.deepcopy(mcp_servers) if mcp_servers is not None else []
        isolated_meta = copy.deepcopy(meta) if meta is not None else {}
        sid = await self._sdk.start_or_resume_agent(
            session_id=None,
            cwd=cwd,
            mcp_servers=isolated_mcp,
            model=DEFAULT_MODEL_ID,
            mode=initial_mode,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        state: dict[str, Any] = {
            "sessionId": sid,
            "cwd": cwd,
            "mcp_servers": isolated_mcp,
            "meta": isolated_meta,
            "mode": initial_mode,
            "model": DEFAULT_MODEL_ID,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "cancelled": False,
        }
        self._sessions[sid] = state
        self._persist_session(sid, state)
        _write_dsh_gate_marker_if_configured()
        return _build_session_descriptor(state)

    async def load_session(
        self,
        *,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resume an existing ACP session (`session/load`) from persistent trajectory storage."""
        await anyio.sleep(0)
        state = self._sessions.get(session_id)
        if state is None:
            state = self._load_persisted_session(session_id)
        if state is None:
            raise KeyError(f"Session '{session_id}' not found")

        effective_mcp = (
            copy.deepcopy(mcp_servers)
            if mcp_servers is not None
            else copy.deepcopy(state.get("mcp_servers", []))
        )
        model = str(state.get("model", DEFAULT_MODEL_ID))
        mode = str(state.get("mode", "auto"))
        reasoning_effort = str(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))

        await self._sdk.start_or_resume_agent(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=effective_mcp,
            model=model,
            mode=mode,
            reasoning_effort=reasoning_effort,
        )

        restored_state: dict[str, Any] = {
            **copy.deepcopy(state),
            "sessionId": session_id,
            "cwd": cwd,
            "mcp_servers": effective_mcp,
            "mode": mode,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "cancelled": bool(state.get("cancelled", False)),
        }
        self._sessions[session_id] = restored_state
        self._persist_session(session_id, restored_state)
        _write_dsh_gate_marker_if_configured()
        return _build_session_descriptor(restored_state)

    async def set_mode(self, *, session_id: str, mode_id: str) -> dict[str, Any]:
        """Update session approval mode or agent profile (`session/set_mode`)."""
        await anyio.sleep(0)
        if mode_id not in VALID_MODE_IDS:
            raise ValueError(f"Mode '{mode_id}' not found")
        session = self._require_session_mut(session_id)
        session["mode"] = mode_id
        self._persist_session(session_id, session)
        return {}

    async def set_model(self, *, session_id: str, model_id: str) -> dict[str, Any]:
        """Update active model (`session/set_model`)."""
        await anyio.sleep(0)
        session = self._require_session_mut(session_id)
        session["model"] = model_id
        self._persist_session(session_id, session)
        return {}

    async def set_config_option(
        self,
        *,
        session_id: str,
        config_id: str,
        value: str,
    ) -> dict[str, Any]:
        """Update session configuration option (`session/set_config_option`)."""
        await anyio.sleep(0)
        session = self._require_session_mut(session_id)
        if config_id == "model":
            session["model"] = value
        elif config_id == "mode":
            await self.set_mode(session_id=session_id, mode_id=value)
        elif config_id in REASONING_EFFORT_CONFIG_IDS:
            session["reasoning_effort"] = value
        else:
            raise ValueError(f"Config option '{config_id}' not found")
        self._persist_session(session_id, session)
        return {
            "configOptions": self._build_config_options(
                model=str(session.get("model", DEFAULT_MODEL_ID)),
                mode=str(session.get("mode", "auto")),
                reasoning_effort=str(session.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
            )
        }

    async def prompt(
        self,
        *,
        session_id: str,
        prompt_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute or steer a streaming turn (`session/prompt`)."""
        _ = prompt_blocks
        if self.consume_cancellation(session_id):
            return {"stopReason": "cancelled"}
        with self.turn_cancel_scope(session_id) as scope:
            if os.environ.get("AGY_ACP_FAKE_SDK") == "slow":
                await anyio.sleep(0.2)
            else:
                await anyio.sleep(0)
        if scope.cancel_called or self.consume_cancellation(session_id):
            return {"stopReason": "cancelled"}
        return {"stopReason": "end_turn"}

    def is_cancelled(self, session_id: str) -> bool:
        """Return whether `session_id` has a pending cancellation."""
        if session_id in self._cancelled_sessions:
            return True
        session = self._sessions.get(session_id)
        return bool(session and session.get("cancelled", False))

    def consume_cancellation(self, session_id: str) -> bool:
        """Clear and return whether `session_id` was marked cancelled."""
        was_cancelled = self.is_cancelled(session_id)
        self._cancelled_sessions.discard(session_id)
        session = self._sessions.get(session_id)
        if session is not None and session.get("cancelled", False):
            session["cancelled"] = False
            self._persist_session(session_id, session)
        return was_cancelled

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
        """Cancel an in-flight prompt turn (`session/cancel`)."""
        await anyio.sleep(0)
        self._cancelled_sessions.add(session_id)
        if session_id in self._sessions:
            self._sessions[session_id]["cancelled"] = True
            self._persist_session(session_id, self._sessions[session_id])
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
        await anyio.sleep(0)
        scope = self._active_cancel_scopes.pop(session_id, None) or self._cancel_scopes.pop(
            session_id, None
        )
        if scope is not None:
            scope.cancel()
        self._sessions.pop(session_id, None)
        self._cancelled_sessions.discard(session_id)
        if delete_persisted and self._save_dir is not None:
            session_file = self._save_dir / f"{session_id}.json"
            if session_file.is_file():
                session_file.unlink()
        return {}
