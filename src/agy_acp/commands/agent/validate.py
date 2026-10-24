"""Validate a KiroCrew JSON agent specification file (`agy-acp agent validate --path <path>`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio


async def default(
    path: Path | None = None,
    *extra_args: str,
    **extra_kwargs: Any,
) -> int:
    """Validate that a KiroCrew agent specification file exists and contains a JSON object."""
    await anyio.sleep(0)
    _ = (extra_args, extra_kwargs)
    if path is None or not path.is_file():
        print("Agent configuration is invalid: file not found")
        return 1
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Agent configuration is invalid: {exc}")
        return 1
    if not isinstance(loaded, dict):
        print("Agent configuration is invalid: root must be a JSON object")
        return 1
    return 0
