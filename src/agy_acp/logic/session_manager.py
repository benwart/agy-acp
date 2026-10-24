"""Session lifecycle, multi-session multiplexing, steer, and cancellation logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SessionManager:
    """Manages concurrent ACP sessions backed by google.antigravity Agent trajectories."""

    def __init__(self, *, save_dir: Path | None = None) -> None:
        self._save_dir = save_dir

    async def create_session(
        self,
        *,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new ACP session (`session/new`) and return its descriptor."""
        return {"sessionId": ""}

    async def load_session(
        self,
        *,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resume an existing ACP session (`session/load`) from persistent trajectory storage."""
        return {"sessionId": ""}

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
