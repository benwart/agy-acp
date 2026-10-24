"""Provider for querying Antigravity 5-hour / 7-day rolling quotas and G1 credits."""

from __future__ import annotations

from typing import Any


class QuotaProvider:
    """Fetches 5-hour (`5h`) and 7-day (`weekly`) quota buckets and G1 credit balances."""

    def fetch_quota_groups(self) -> list[dict[str, Any]]:
        """Return raw quota groups containing 5h and weekly buckets."""
        return [
            {
                "group": "Gemini Models",
                "buckets": [
                    {
                        "id": "gemini-5h",
                        "group": "Gemini Models",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 0.95,
                        "reset_time": "2026-10-24T07:52:04Z",
                    },
                    {
                        "id": "gemini-weekly",
                        "group": "Gemini Models",
                        "name": "Weekly Limit Remaining",
                        "window": "weekly",
                        "remaining_fraction": 0.39,
                        "reset_time": "2026-10-24T12:34:28Z",
                    },
                ],
            },
            {
                "group": "Claude and GPT models",
                "buckets": [
                    {
                        "id": "claude-gpt-5h",
                        "group": "Claude and GPT models",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 0.80,
                        "reset_time": "2026-10-24T07:52:04Z",
                    },
                    {
                        "id": "claude-gpt-weekly",
                        "group": "Claude and GPT models",
                        "name": "Weekly Limit Remaining",
                        "window": "weekly",
                        "remaining_fraction": 0.72,
                        "reset_time": "2026-10-24T12:34:28Z",
                    },
                ],
            },
        ]

    def fetch_credits(self) -> dict[str, Any]:
        """Return G1 remaining credits and upgrade URI."""
        return {
            "remaining_credits": 1000,
            "upgrade_uri": "https://one.google.com/ai",
        }
