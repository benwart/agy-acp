"""Unit tests for QuotaTracker 5h/7d buckets, credits, and KiroCrew wire formatting."""

from __future__ import annotations

import pytest

from agy_acp.logic.quota_tracker import QuotaTracker


@pytest.mark.not_implemented
def test_five_hour_bucket_calculation() -> None:
    """Verify QuotaTracker.get_five_hour_quota returns 5h bucket structure."""
    tracker = QuotaTracker()
    res = tracker.get_five_hour_quota()
    assert res["window"] == "5h"
    assert len(res["buckets"]) >= 1


@pytest.mark.not_implemented
def test_seven_day_bucket_calculation() -> None:
    """Verify QuotaTracker.get_seven_day_quota returns 7d bucket structure."""
    tracker = QuotaTracker()
    res = tracker.get_seven_day_quota()
    assert res["window"] == "7d"
    assert len(res["buckets"]) >= 1


@pytest.mark.not_implemented
def test_combined_quota_and_token_ledger() -> None:
    """Verify QuotaTracker.get_unified_status and get_credits_and_tokens."""
    tracker = QuotaTracker()
    status = tracker.get_unified_status()
    assert "five_hour" in status
    assert "seven_day" in status
    assert "credits" in status


@pytest.mark.not_implemented
def test_format_kirocrew_usage_and_slash_response() -> None:
    """Verify QuotaTracker builds KiroCrew wire notifications and slash command Markdown."""
    tracker = QuotaTracker()
    updates = tracker.build_kirocrew_wire_updates(
        session_id="s" * 36,
        model_id="gemini-3.8-flash",
        input_tokens=1000,
        output_tokens=250,
    )
    assert len(updates) >= 2
    md = tracker.format_slash_command_response("/quota")
    assert "5h" in md or "Five Hour" in md
