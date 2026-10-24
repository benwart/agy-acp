"""Inspect combined 5-hour quota, 7-day quota, credits, and token usage."""

from __future__ import annotations


def default(
    group: str | None = None,
    format: str = "text",
) -> int:
    """Display unified 5-hour quota, 7-day quota, G1 credit balance, and token usage."""
    return 1
