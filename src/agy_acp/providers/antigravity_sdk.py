"""Adapter wrapping google.antigravity Agent, LocalAgentConfig, CapabilitiesConfig, and hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class AntigravitySdkProvider:
    """Wraps `google.antigravity.Agent` lifecycle, streaming, tool hooks, and trajectory resume."""

    def __init__(self, *, save_dir: Path | None = None) -> None:
        self._save_dir = save_dir

    async def start_or_resume_agent(
        self,
        *,
        session_id: str | None,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        model: str | None = None,
        mode: str = "auto",
        reasoning_effort: str = "medium",
        auth_config: dict[str, Any] | None = None,
    ) -> str:
        """Spawn or resume a `google.antigravity.Agent` and return its conversation_id."""
        return ""
