"""Unit tests for QuotaTracker 5h/7d buckets, credits, and KiroCrew wire formatting."""

from __future__ import annotations

from typing import Any

from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.providers.quota_provider import QuotaProvider


def test_five_hour_bucket_calculation() -> None:
    """Verify QuotaTracker.get_five_hour_quota returns 5h bucket structure."""
    tracker = QuotaTracker()
    res = tracker.get_five_hour_quota()
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

    filtered = tracker.get_five_hour_quota(group="Gemini Models")
    assert len(filtered["buckets"]) == 1
    assert filtered["buckets"][0]["group"] == "Gemini Models"


def test_seven_day_bucket_calculation() -> None:
    """Verify QuotaTracker.get_seven_day_quota returns 7d bucket structure."""
    tracker = QuotaTracker()
    res = tracker.get_seven_day_quota()
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

    filtered = tracker.get_seven_day_quota(group="Gemini Models")
    assert len(filtered["buckets"]) == 1
    assert filtered["buckets"][0]["group"] == "Gemini Models"


def test_combined_quota_and_token_ledger() -> None:
    """Verify QuotaTracker.get_unified_status and get_credits_and_tokens."""
    tracker = QuotaTracker()
    tracker.record_token_usage(
        input_tokens=120,
        output_tokens=30,
        cached_read_tokens=40,
        thinking_tokens=10,
    )
    credits_and_tokens = tracker.get_credits_and_tokens()
    assert credits_and_tokens["remaining_credits"] == 1000
    assert credits_and_tokens["upgrade_uri"] == "https://one.google.com/ai"
    assert credits_and_tokens["token_usage"] == {
        "input_tokens": 120,
        "output_tokens": 30,
        "cached_read_tokens": 40,
        "thinking_tokens": 10,
        "total_tokens": 200,
    }

    status = tracker.get_unified_status()
    assert "five_hour" in status
    assert "seven_day" in status
    assert "credits" in status
    assert status["token_usage"]["total_tokens"] == 200


def test_format_kirocrew_usage_and_slash_response() -> None:
    """Verify QuotaTracker builds KiroCrew wire notifications and slash command Markdown."""
    tracker = QuotaTracker()
    updates = tracker.build_kirocrew_wire_updates(
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

    md = tracker.format_slash_command_response("/quota")
    assert "5h" in md and "Five Hour" in md
    assert "7d" in md and "Weekly" in md
    assert "1250" in md

    class _ExhaustedQuotaProvider(QuotaProvider):
        def fetch_quota_groups(self) -> list[dict[str, Any]]:
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

    exhausted_tracker = QuotaTracker(provider=_ExhaustedQuotaProvider())
    exhausted_updates = exhausted_tracker.build_kirocrew_wire_updates(
        session_id="s" * 36,
        model_id="gemini-3.8-flash",
        input_tokens=10,
        output_tokens=5,
    )
    assert len(exhausted_updates) == 3
    assert exhausted_updates[2]["method"] == "_kiro.dev/error/rate_limit"
