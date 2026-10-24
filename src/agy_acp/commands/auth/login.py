"""Authenticate with Google Antigravity SDK and persist hot-reloadable credentials."""

from __future__ import annotations

from pathlib import Path


def default(
    api_key: str | None = None,
    vertex: bool = False,
    project: str | None = None,
    location: str | None = None,
    oauth: bool = False,
    auth_file: Path | None = None,
    format: str = "text",
) -> int:
    """Log in to Google Antigravity SDK and atomically update credentials for hot-reload."""
    return 1
