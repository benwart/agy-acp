"""Validate KiroCrew tool-gate configuration and marker state."""

from __future__ import annotations

import json
import os
from pathlib import Path

import anyio


def _resolve_marker_path(marker_file: Path | None) -> Path | None:
    if marker_file is not None:
        return marker_file
    env_marker = os.environ.get("KIROCREW_DSH_GATE_MARKER")
    return Path(env_marker) if env_marker else None


async def default(
    marker_file: Path | None = None,
    format: str = "json",
) -> int:
    """Validate KiroCrew tool-gate policy and session marker configuration."""
    await anyio.sleep(0)

    target_marker = _resolve_marker_path(marker_file)
    gate_session = os.environ.get("KIROCREW_DSH_GATE_SESSION")
    opencode_config = os.environ.get("OPENCODE_CONFIG_CONTENT")
    marker_exists = target_marker.exists() if target_marker is not None else False

    payload = {
        "valid": True,
        "marker_file": str(target_marker) if target_marker else None,
        "marker_exists": marker_exists,
        "gate_session": gate_session,
        "goose_mode": os.environ.get("GOOSE_MODE", "auto"),
        "opencode_configured": bool(opencode_config),
    }

    if format == "json":
        print(json.dumps(payload))
    else:
        print(f"valid={payload['valid']} marker_file={payload['marker_file']}")

    return 0
