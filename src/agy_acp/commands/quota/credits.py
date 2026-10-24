"""Inspect remaining Antigravity credits and cumulative session token usage."""

from __future__ import annotations

import json

import anyio

from agy_acp.commands.quota.status import format_reset_time
from agy_acp.logic.quota_tracker import QuotaTracker


async def default(
    format: str = "text",
) -> int:
    """Display remaining Antigravity G1 credits, Kiro quota credits, and session token usage."""
    await anyio.sleep(0)
    data = await QuotaTracker().get_credits_and_tokens()
    if format == "json":
        print(json.dumps(data))
    else:
        tokens = data["token_usage"]
        kiro = data["kiro_credits"]
        reset_label = format_reset_time(str(kiro.get("reset_time", "")))
        print(
            f"Effective Kiro Credits: {kiro['effective_credits']:.1f} / "
            f"{kiro['credit_capacity']:.0f} "
            f"({kiro['bottleneck_window']} bottleneck, resets {reset_label}) | "
            f"G1 Overage Credits: {data['remaining_credits']} "
            f"(Total Tokens: {tokens['total_tokens']})"
        )
    return 0
