"""Clear stored Google Antigravity SDK credentials without restarting ACP servers."""

from __future__ import annotations

from pathlib import Path


def default(
    auth_file: Path | None = None,
    format: str = "text",
) -> int:
    """Log out of Google Antigravity SDK and signal connected ACP servers without restart."""
    return 1
