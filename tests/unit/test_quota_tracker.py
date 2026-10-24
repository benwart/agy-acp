"""Unit tests for QuotaProvider and QuotaTracker 5h/7d buckets, credits, and KiroCrew formatting."""

from __future__ import annotations

from typing import Any

from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.providers.agy_cli_provider import AgyCliProvider
from agy_acp.providers.quota_provider import QuotaProvider


class _StubAgyCliProvider(AgyCliProvider):
    """Test double for AgyCliProvider returning `/usage` and `/credits` JSON payloads."""

    async def run_slash_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        _ = (cwd, timeout)
        if command.lstrip("/") == "usage":
            return {
                "conversation_id": "",
                "status": "SUCCESS",
                "command": {
                    "name": "usage",
                    "data": {
                        "groups": [
                            {
                                "name": "Gemini Models",
                                "buckets": [
                                    {
                                        "id": "gemini-5h",
                                        "name": "Five Hour Limit Remaining",
                                        "window": "5h",
                                        "remaining_fraction": 0.95,
                                        "reset_time": "2026-10-24T07:52:04Z",
                                    },
                                    {
                                        "id": "gemini-weekly",
                                        "name": "Weekly Limit Remaining",
                                        "window": "weekly",
                                        "remaining_fraction": 0.39,
                                        "reset_time": "2026-10-24T12:34:28Z",
                                    },
                                ],
                            },
                            {
                                "name": "Claude and GPT models",
                                "buckets": [
                                    {
                                        "id": "claude-gpt-5h",
                                        "name": "Five Hour Limit Remaining",
                                        "window": "5h",
                                        "remaining_fraction": 0.80,
                                        "reset_time": "2026-10-24T07:52:04Z",
                                    },
                                    {
                                        "id": "claude-gpt-weekly",
                                        "name": "Weekly Limit Remaining",
                                        "window": "weekly",
                                        "remaining_fraction": 0.72,
                                        "reset_time": "2026-10-24T12:34:28Z",
                                    },
                                ],
                            },
                        ]
                    },
                },
            }
        if command.lstrip("/") == "credits":
            return {
                "conversation_id": "",
                "status": "SUCCESS",
                "command": {
                    "name": "credits",
                    "data": {
                        "remaining_credits": 1000,
                        "upgrade_uri": "https://one.google.com/ai",
                    },
                },
            }
        raise ValueError(f"Unexpected command: {command}")


def _make_tracker() -> QuotaTracker:
    return QuotaTracker(provider=QuotaProvider(cli_provider=_StubAgyCliProvider()))


async def test_five_hour_bucket_calculation() -> None:
    """Verify QuotaTracker.get_five_hour_quota returns 5h bucket structure."""
    tracker = _make_tracker()
    res = await tracker.get_five_hour_quota()
    assert res["window"] == "5h"
    assert len(res["buckets"]) >= 2
    first = res["buckets"][0]
    assert first["id"] == "gemini-5h"
    assert first["group"] == "Gemini Models"
    assert first["window"] == "5h"
    assert first["remaining_fraction"] == 0.95
    assert first["remaining_percent"] == 95.0
    assert first["used_percent"] == 5.0
    assert "reset_time" in first

    filtered = await tracker.get_five_hour_quota(group="Gemini Models")
    assert len(filtered["buckets"]) == 1
    assert filtered["buckets"][0]["group"] == "Gemini Models"


async def test_seven_day_bucket_calculation() -> None:
    """Verify QuotaTracker.get_seven_day_quota returns 7d bucket structure."""
    tracker = _make_tracker()
    res = await tracker.get_seven_day_quota()
    assert res["window"] == "7d"
    assert len(res["buckets"]) >= 2
    first = res["buckets"][0]
    assert first["id"] == "gemini-weekly"
    assert first["group"] == "Gemini Models"
    assert first["window"] == "7d"
    assert first["remaining_fraction"] == 0.39
    assert first["remaining_percent"] == 39.0
    assert first["used_percent"] == 61.0
    assert "reset_time" in first

    filtered = await tracker.get_seven_day_quota(group="Gemini Models")
    assert len(filtered["buckets"]) == 1
    assert filtered["buckets"][0]["group"] == "Gemini Models"


async def test_combined_quota_and_token_ledger() -> None:
    """Verify QuotaTracker.get_unified_status and get_credits_and_tokens."""
    tracker = _make_tracker()
    tracker.record_token_usage(
        input_tokens=120,
        output_tokens=30,
        cached_read_tokens=40,
        thinking_tokens=10,
    )
    credits_and_tokens = await tracker.get_credits_and_tokens()
    assert credits_and_tokens["remaining_credits"] == 1000
    assert credits_and_tokens["upgrade_uri"] == "https://one.google.com/ai"
    assert credits_and_tokens["token_usage"] == {
        "input_tokens": 120,
        "output_tokens": 30,
        "cached_read_tokens": 40,
        "thinking_tokens": 10,
        "total_tokens": 200,
    }

    status = await tracker.get_unified_status()
    assert "five_hour" in status
    assert "seven_day" in status
    assert "credits" in status
    assert status["token_usage"]["total_tokens"] == 200


async def test_format_kirocrew_usage_and_slash_response() -> None:
    """Verify QuotaTracker builds KiroCrew wire notifications and slash command Markdown."""
    tracker = _make_tracker()
    updates = await tracker.build_kirocrew_wire_updates(
        session_id="s" * 36,
        model_id="gemini-3.8-flash",
        input_tokens=1000,
        output_tokens=250,
    )
    assert len(updates) == 2
    assert updates[0]["method"] == "session/update"
    assert updates[0]["params"]["update"]["sessionUpdate"] == "usage_update"
    assert updates[0]["params"]["update"]["used"] == 1250
    assert updates[0]["params"]["update"]["size"] == 1000000
    assert updates[1]["method"] == "_kiro.dev/metadata"
    assert updates[1]["params"]["modelId"] == "gemini-3.8-flash"
    assert "five_hour" in updates[1]["params"]["_meta"]["quota"]
    assert "seven_day" in updates[1]["params"]["_meta"]["quota"]

    md = await tracker.format_slash_command_response("/quota")
    assert "5h" in md and "Five Hour" in md
    assert "7d" in md and "Weekly" in md
    assert "1250" in md

    class _ExhaustedQuotaProvider(QuotaProvider):
        async def fetch_quota_groups(self) -> list[dict[str, Any]]:
            return [
                {
                    "group": "Gemini Models",
                    "buckets": [
                        {
                            "id": "gemini-5h",
                            "group": "Gemini Models",
                            "name": "Five Hour Limit Remaining",
                            "window": "5h",
                            "remaining_fraction": 0.0,
                            "reset_time": "2026-10-24T07:52:04Z",
                        }
                    ],
                }
            ]

        async def fetch_credits(self) -> dict[str, Any]:
            return {"remaining_credits": 0, "upgrade_uri": "https://antigravity.google/g1-upgrade"}

    exhausted_tracker = QuotaTracker(provider=_ExhaustedQuotaProvider())
    exhausted_updates = await exhausted_tracker.build_kirocrew_wire_updates(
        session_id="s" * 36,
        model_id="gemini-3.8-flash",
        input_tokens=10,
        output_tokens=5,
    )
    assert len(exhausted_updates) == 3
    assert exhausted_updates[2]["method"] == "_kiro.dev/error/rate_limit"


async def test_rich_status_progress_bars_colors_grouping_and_trend() -> None:
    """Verify Rich quota status renders thin colored bars, 7d->5h grouping, and fixed trend."""
    from datetime import UTC, datetime
    from io import StringIO

    from rich.console import Console

    from agy_acp.commands.quota.status import (
        TREND_FIXED_WIDTH,
        build_quota_status_renderables,
        format_over_under_trend,
        render_rich_quota_status,
        resolve_quota_bar_style,
    )

    # 1. Color rules: Gemini blue (red < 5%), other green (orange1 < 5%)
    assert resolve_quota_bar_style("Gemini Models", 95.0) == "blue"
    assert resolve_quota_bar_style("Gemini Models", 5.0) == "blue"
    assert resolve_quota_bar_style("Gemini Models", 4.9) == "red"
    assert resolve_quota_bar_style("Claude and GPT models", 80.0) == "green"
    assert resolve_quota_bar_style("Claude and GPT models", 5.0) == "green"
    assert resolve_quota_bar_style("Claude and GPT models", 4.9) == "orange1"

    # 2. Fixed-width over/under trend calculation relative to remaining time % of window
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)
    # 2.5 hours left in a 5h window => 50% time remaining
    reset_5h_half = "2026-10-24T14:30:00Z"
    over_str, over_style = format_over_under_trend(
        remaining_percent=75.0,
        window="5h",
        reset_time=reset_5h_half,
        now=now,
    )
    under_str, under_style = format_over_under_trend(
        remaining_percent=30.0,
        window="5h",
        reset_time=reset_5h_half,
        now=now,
    )
    even_str, _ = format_over_under_trend(
        remaining_percent=50.0,
        window="5h",
        reset_time=reset_5h_half,
        now=now,
    )
    assert len(over_str) == TREND_FIXED_WIDTH
    assert len(under_str) == TREND_FIXED_WIDTH
    assert len(even_str) == TREND_FIXED_WIDTH
    assert "OVER" in over_str and "+25.0%" in over_str
    assert "UNDER" in under_str and "-20.0%" in under_str
    assert over_style != under_style

    # 3. Grouping (Gemini then Claude/GPT), ordering (7d then 5h), and per-bucket low-quota color
    status_data = {
        "five_hour": {
            "window": "5h",
            "buckets": [
                {
                    "id": "gemini-5h",
                    "group": "Gemini Models",
                    "name": "Five Hour Limit Remaining",
                    "window": "5h",
                    "remaining_fraction": 0.03,
                    "remaining_percent": 3.0,
                    "used_percent": 97.0,
                    "reset_time": reset_5h_half,
                },
                {
                    "id": "3p-5h",
                    "group": "Claude and GPT models",
                    "name": "Five Hour Limit Remaining",
                    "window": "5h",
                    "remaining_fraction": 0.80,
                    "remaining_percent": 80.0,
                    "used_percent": 20.0,
                    "reset_time": reset_5h_half,
                },
            ],
        },
        "seven_day": {
            "window": "7d",
            "buckets": [
                {
                    "id": "gemini-weekly",
                    "group": "Gemini Models",
                    "name": "Weekly Limit Remaining",
                    "window": "7d",
                    "remaining_fraction": 0.90,
                    "remaining_percent": 90.0,
                    "used_percent": 10.0,
                    "reset_time": "2026-10-27T12:00:00Z",
                },
                {
                    "id": "3p-weekly",
                    "group": "Claude and GPT models",
                    "name": "Weekly Limit Remaining",
                    "window": "7d",
                    "remaining_fraction": 0.04,
                    "remaining_percent": 4.0,
                    "used_percent": 96.0,
                    "reset_time": "2026-10-27T12:00:00Z",
                },
            ],
        },
        "credits": {"remaining_credits": 500, "upgrade_uri": "https://one.google.com/ai"},
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_read_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 15,
        },
    }

    rows_meta = build_quota_status_renderables(status_data, now=now)
    assert [r["id"] for r in rows_meta] == [
        "gemini-weekly",
        "gemini-5h",
        "3p-weekly",
        "3p-5h",
    ]
    assert [r["bar"].complete_style for r in rows_meta] == [
        "blue",
        "red",
        "orange1",
        "green",
    ]

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True, color_system="truecolor")
    render_rich_quota_status(status_data, console=console, now=now)
    rendered = buf.getvalue()

    assert "━" in rendered
    gemini_pos = rendered.index("Gemini Models")
    claude_pos = rendered.index("Claude and GPT models")
    assert gemini_pos < claude_pos

    # Within Gemini group, Weekly (7d) appears before Five Hour (5h)
    gemini_weekly_pos = rendered.index("Weekly Limit Remaining", gemini_pos)
    gemini_5h_pos = rendered.index("Five Hour Limit Remaining", gemini_pos)
    assert gemini_weekly_pos < gemini_5h_pos < claude_pos

    # On each quota line, thin bar '━' precedes trend ('OVER'/'UNDER'), which precedes name
    for line in rendered.splitlines():
        if "Weekly Limit Remaining" in line or "Five Hour Limit Remaining" in line:
            bar_idx = line.index("━")
            trend_idx = line.index("OVER") if "OVER" in line else line.index("UNDER")
            name_idx = (
                line.index("Weekly Limit Remaining")
                if "Weekly Limit Remaining" in line
                else line.index("Five Hour Limit Remaining")
            )
            assert bar_idx < trend_idx < name_idx
