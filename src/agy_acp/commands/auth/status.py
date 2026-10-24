"""Inspect current Google Antigravity SDK authentication status and fingerprint."""

from __future__ import annotations

from pathlib import Path


def default(
    auth_file: Path | None = None,
    format: str = "text",
) -> int:
    """Show current authentication status, active method, and credential fingerprint."""
    return 1
