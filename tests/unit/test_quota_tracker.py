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
        format_reset_time,
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
    assert format_reset_time("2026-10-28T21:41:00Z", now=now) == "in 4d 9h 41m"
    assert format_reset_time("2026-10-27T12:00:00Z", now=now) == "in 3d"
    assert format_reset_time("2026-10-24T14:30:00Z", now=now) == "in 2h 30m"
    assert format_reset_time("2026-10-24T12:00:45Z", now=now) == "in 45s"
    assert format_reset_time("2026-10-24T11:59:00Z", now=now) == "now"
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

    # Reset strings must be human-readable durations rather than raw ISO timestamps
    assert "resets in 3d" in rendered
    assert "resets in 2h 30m" in rendered
    assert "2026-10-27T12:00:00Z" not in rendered
    assert "2026-10-24T14:30:00Z" not in rendered


async def test_kiro_credit_conversion_picks_min_of_7d_and_5h_with_matching_reset() -> None:
    """Verify Option 1 credit conversion (7d=1000, 5h=200) picks min credits and its reset_time."""
    tracker = _make_tracker()

    # In _StubAgyCliProvider for Gemini:
    # 5h: 0.95 * 200 = 190.0 credits (resets 2026-10-24T07:52:04Z)
    # 7d: 0.39 * 1000 = 390.0 credits (resets 2026-10-24T12:34:28Z)
    # G1 overage: 1000 credits
    # Bottleneck is 5h (190.0 < 390.0) -> quota_credits=190.0, reset_time=5h reset_time
    five_hour = await tracker.get_five_hour_quota()
    seven_day = await tracker.get_seven_day_quota()
    eff_5h = tracker.calculate_effective_kiro_credits(
        five_hour["buckets"],
        seven_day["buckets"],
        model_id="gemini-3.7-pro",
        g1_overage_credits=0,
    )
    assert eff_5h["five_hour_credits"] == 190.0
    assert eff_5h["seven_day_credits"] == 390.0
    assert eff_5h["bottleneck_window"] == "5h"
    assert eff_5h["quota_credits"] == 190.0
    assert eff_5h["effective_credits"] == 190.0
    assert eff_5h["reset_time"] == "2026-10-24T07:52:04Z"

    # Now test when 7d is the bottleneck: 7d = 0.05 * 1000 = 50.0 vs 5h = 0.80 * 200 = 160.0
    eff_7d = tracker.calculate_effective_kiro_credits(
        [
            {
                "id": "gemini-5h",
                "group": "Gemini Models",
                "name": "Five Hour Limit Remaining",
                "window": "5h",
                "remaining_fraction": 0.80,
                "remaining_credits": 160.0,
                "credit_capacity": 200.0,
                "reset_time": "2026-10-24T14:30:00Z",
            }
        ],
        [
            {
                "id": "gemini-weekly",
                "group": "Gemini Models",
                "name": "Weekly Limit Remaining",
                "window": "7d",
                "remaining_fraction": 0.05,
                "remaining_credits": 50.0,
                "credit_capacity": 1000.0,
                "reset_time": "2026-10-27T12:00:00Z",
            }
        ],
        model_id="gemini-3.7-pro",
        g1_overage_credits=25,
    )
    assert eff_7d["five_hour_credits"] == 160.0
    assert eff_7d["seven_day_credits"] == 50.0
    assert eff_7d["bottleneck_window"] == "7d"
    assert eff_7d["quota_credits"] == 50.0
    assert eff_7d["effective_credits"] == 75.0
    assert eff_7d["reset_time"] == "2026-10-27T12:00:00Z"

    # Verify _kiro.dev/metadata wire payload includes creditsResetTime and converted credits
    updates = await tracker.build_kirocrew_wire_updates(
        session_id="s" * 36,
        model_id="gemini-3.7-pro",
        input_tokens=100,
        output_tokens=50,
    )
    meta_params = updates[1]["params"]
    assert meta_params["credits"] == 1190.0  # 190.0 quota_credits + 1000 G1 overage in stub
    assert meta_params["quotaCredits"] == 190.0
    assert meta_params["creditsBottleneckWindow"] == "5h"
    assert meta_params["creditsResetTime"] == "2026-10-24T07:52:04Z"


async def test_context_usage_reflects_active_session_context_not_global_ledger_and_compacts(
    tmp_path: Any,
) -> None:
    """Verify wire context usage uses current session context_tokens, not global lifetime total."""
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    store = SqliteStateStoreProvider(tmp_path / "state.db")
    # Seed 8.13M global historical tokens from prior sessions/tool loops
    store.record_tokens(
        input_tokens=911_566,
        output_tokens=36_221,
        cached_read_tokens=7_167_559,
        thinking_tokens=15_532,
    )
    tracker = QuotaTracker(
        provider=QuotaProvider(cli_provider=_StubAgyCliProvider(save_dir=tmp_path)),
        state_store=store,
    )

    # Short session turn with multi-step billed tokens vs actual active context window (24,311)
    updates = await tracker.build_kirocrew_wire_updates(
        session_id="session-short-1",
        model_id="gemini-4-argon-medium",
        input_tokens=148_431,
        output_tokens=6_399,
        cached_read_tokens=1_277_177,
        thinking_tokens=500,
        context_tokens=24_311,
    )
    usage_update = updates[0]["params"]["update"]
    meta_params = updates[1]["params"]
    assert usage_update["used"] == 24_311
    assert meta_params["contextUsagePercentage"] == 2.4311

    # Subsequent /compact turn shrinks active context window to 3,200 tokens
    compact_updates = await tracker.build_kirocrew_wire_updates(
        session_id="session-short-1",
        model_id="gemini-4-argon-medium",
        input_tokens=10,
        output_tokens=50,
        context_tokens=3_200,
    )
    assert compact_updates[0]["params"]["update"]["used"] == 3_200
    assert compact_updates[1]["params"]["contextUsagePercentage"] == 0.32
