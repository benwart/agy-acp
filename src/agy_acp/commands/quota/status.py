"""Inspect combined 5-hour quota, 7-day quota, credits, and token usage."""

from __future__ import annotations

import json

import anyio

from agy_acp.logic.quota_tracker import QuotaTracker


async def default(
    group: str | None = None,
    format: str = "text",
) -> int:
    """Display unified 5-hour quota, 7-day quota, G1 credit balance, and token usage."""
    await anyio.sleep(0)
    data = await QuotaTracker().get_unified_status(group=group)
    if format == "json":
        print(json.dumps(data))
    else:
        for bucket in [*data["five_hour"]["buckets"], *data["seven_day"]["buckets"]]:
            print(
                f"[{bucket['group']}] {bucket['name']} ({bucket['window']}): "
                f"{bucket['remaining_percent']:.1f}% remaining "
                f"(resets {bucket['reset_time']})"
            )
        credits = data["credits"]
        tokens = data["token_usage"]
        print(
            f"Remaining Credits: {credits.get('remaining_credits', 0)} "
            f"(Total Tokens: {tokens['total_tokens']})"
        )
    return 0
