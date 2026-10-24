"""Inspect 7-day (weekly) rolling quota remaining percentage and reset time."""

from __future__ import annotations


def default(
    group: str | None = None,
    format: str = "text",
) -> int:
    """Display 7-day (weekly) rolling quota remaining percentage, usage, and reset time."""
    return 1
