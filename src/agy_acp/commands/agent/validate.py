"""Validate a KiroCrew JSON or Markdown agent specification (`agy-acp agent validate --path`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio

from agy_acp.logic.custom_agents import validate_agent_spec_file


async def default(
    path: Path | None = None,
    *extra_args: str,
    **extra_kwargs: Any,
) -> int:
    """Validate syntax, containment, and schema of a KiroCrew `.json` or `.md` agent spec."""
    await anyio.sleep(0)
    _ = (extra_args, extra_kwargs)
    if path is None or not path.is_file():
        print("Agent configuration is invalid: file not found")
        return 1
    try:
        validate_agent_spec_file(path)
    except (OSError, ValueError) as exc:
        print(f"Agent configuration is invalid: {exc}")
        return 1
    return 0
