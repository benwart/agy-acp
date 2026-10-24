"""Validate KiroCrew tool-gate configuration and marker state."""

from __future__ import annotations

from pathlib import Path


def default(
    marker_file: Path | None = None,
    format: str = "json",
) -> int:
    """Validate KiroCrew tool-gate policy and session marker configuration."""
    return 1
