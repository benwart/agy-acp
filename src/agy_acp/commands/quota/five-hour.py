"""Inspect 5-hour rolling quota remaining percentage and reset time."""

from __future__ import annotations

import json

import anyio

from agy_acp.commands.quota.status import format_reset_time
from agy_acp.logic.quota_tracker import QuotaTracker


async def default(
    group: str | None = None,
    format: str = "text",
) -> int:
    """Display 5-hour rolling quota remaining percentage, usage, and reset time."""
    await anyio.sleep(0)
    tracker = QuotaTracker()
    data = await tracker.get_five_hour_quota(group=group)
    if format == "json":
        print(json.dumps(data))
    else:
        for bucket in data["buckets"]:
            reset_label = format_reset_time(str(bucket.get("reset_time", "")))
            print(
                f"[{bucket['group']}] {bucket['name']} (5h): "
                f"{bucket['remaining_percent']:.1f}% remaining "
                f"(resets {reset_label})"
            )
    return 0
