"""Session lifecycle, multi-session multiplexing, steer, and cancellation logic."""

from __future__ import annotations

import copy
import json
import os
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

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return an isolated copy of the state for `session_id`."""
        state = self._sessions.get(session_id)
        if state is None:
            state = self._load_persisted_session(session_id)
        if state is None:
            raise KeyError(f"Session '{session_id}' not found")
        return copy.deepcopy(state)

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
        return {"unimplemented": True}

    async def set_model(self, *, session_id: str, model_id: str) -> dict[str, Any]:
        """Update active model (`session/set_model`)."""
        return {"unimplemented": True}

    async def set_config_option(
        self,
        *,
        session_id: str,
        config_id: str,
        value: str,
    ) -> dict[str, Any]:
        """Update session configuration option (`session/set_config_option`)."""
        return {}

    async def prompt(
        self,
        *,
        session_id: str,
        prompt_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute or steer a streaming turn (`session/prompt`)."""
        return {}

    async def cancel(self, *, session_id: str) -> bool:
        """Cancel an in-flight prompt turn (`session/cancel`)."""
        return False

    async def close_session(self, *, session_id: str) -> dict[str, Any]:
        """Close and clean up an active session (`session/close`, `terminate`, `delete`)."""
        return {"closed": False}
