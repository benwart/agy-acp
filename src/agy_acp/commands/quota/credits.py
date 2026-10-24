"""Inspect remaining Antigravity credits and cumulative session token usage."""

from __future__ import annotations

import json

import anyio

from agy_acp.logic.quota_tracker import QuotaTracker


async def default(
    format: str = "text",
) -> int:
    """Display remaining Antigravity G1 credits and cumulative session token usage."""
    await anyio.sleep(0)
    data = QuotaTracker().get_credits_and_tokens()
    if format == "json":
        print(json.dumps(data))
    else:
        tokens = data["token_usage"]
        print(
            f"Remaining Credits: {data['remaining_credits']} "
            f"(Total Tokens: {tokens['total_tokens']})"
        )
    return 0
