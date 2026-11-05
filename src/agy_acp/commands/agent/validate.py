"""Validate a KiroCrew JSON or Markdown agent specification (`agy-acp agent validate --path`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio

from agy_acp.logic.custom_agents import parse_agent_spec_file


async def default(
    path: Path | None = None,
    *extra_args: str,
    **extra_kwargs: Any,
) -> int:
    """Validate that a KiroCrew `.json` or `.md` agent specification file parses cleanly."""
    await anyio.sleep(0)
    _ = (extra_args, extra_kwargs)
    if path is None or not path.is_file():
        print("Agent configuration is invalid: file not found")
        return 1
    try:
        loaded = parse_agent_spec_file(path)
    except (OSError, ValueError) as exc:
        print(f"Agent configuration is invalid: {exc}")
        return 1
    if not isinstance(loaded, dict):
        print("Agent configuration is invalid: root must be a JSON object")
        return 1
    return 0
