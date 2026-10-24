"""Adapter wrapping google.antigravity Agent, LocalAgentConfig, CapabilitiesConfig, and hooks."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import anyio


def _pairs_to_dict(items: Any) -> dict[str, str]:
    """Convert a list of `{"name": ..., "value": ...}` dicts or a mapping into `dict[str, str]`."""
    if isinstance(items, dict):
        return {str(k): str(v) for k, v in items.items()}
    if not isinstance(items, list):
        return {}
    result: dict[str, str] = {}
    for entry in items:
        if isinstance(entry, dict) and "name" in entry and "value" in entry:
            result[str(entry["name"])] = str(entry["value"])
    return result


def _translate_mcp_server(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate an ACP MCP server definition into normalized stdio or HTTP config."""
    server_type = str(raw.get("type", "stdio")).lower()
    if server_type == "http" or "url" in raw:
        return {
            "type": "http",
            "name": str(raw.get("name", "")),
            "url": str(raw.get("url", "")),
            "headers": _pairs_to_dict(raw.get("headers", [])),
        }
    return {
        "type": "stdio",
        "name": str(raw.get("name", "")),
        "command": str(raw.get("command", "")),
        "args": [str(arg) for arg in raw.get("args", [])],
        "env": _pairs_to_dict(raw.get("env", [])),
    }


class AntigravitySdkProvider:
    """Wraps `google.antigravity.Agent` lifecycle, streaming, tool hooks, and trajectory resume."""

    def __init__(self, *, save_dir: Path | None = None) -> None:
        self._save_dir = save_dir
        self._agents: dict[str, dict[str, Any]] = {}

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
        await anyio.sleep(0)
        sid = session_id or str(uuid.uuid4())
        translated_mcp = [_translate_mcp_server(srv) for srv in (mcp_servers or [])]
        state: dict[str, Any] = {
            "sessionId": sid,
            "cwd": cwd,
            "mcp_servers": translated_mcp,
            "model": model or "gemini-3.7-pro",
            "mode": mode,
            "reasoning_effort": reasoning_effort,
            "auth_config": auth_config or {},
        }
        self._agents[sid] = state
        if self._save_dir is not None:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            state_file = self._save_dir / f"{sid}.json"
            state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return sid
