"""Top-level status command displaying colored Rich quota progress bars and over/under trends."""

from __future__ import annotations

import anyio

from agy_acp.commands.quota.status import default as quota_status_default


async def default(
    group: str | None = None,
    format: str = "text",
) -> int:
    """Display unified 7-day and 5-hour quota progress bars, trends, credits, and token usage."""
    await anyio.sleep(0)
    return await quota_status_default(group=group, format=format)
