"""Inspect 7-day (weekly) rolling quota remaining percentage and reset time."""

from __future__ import annotations

import json

import anyio

from agy_acp.logic.quota_tracker import QuotaTracker


async def default(
    group: str | None = None,
    format: str = "text",
) -> int:
    """Display 7-day (weekly) rolling quota remaining percentage, usage, and reset time."""
    await anyio.sleep(0)
    tracker = QuotaTracker()
    data = await tracker.get_seven_day_quota(group=group)
    if format == "json":
        print(json.dumps(data))
    else:
        for bucket in data["buckets"]:
            print(
                f"[{bucket['group']}] {bucket['name']} (7d): "
                f"{bucket['remaining_percent']:.1f}% remaining "
                f"(resets {bucket['reset_time']})"
            )
    return 0
