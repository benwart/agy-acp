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
