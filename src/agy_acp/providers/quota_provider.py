"""Provider for querying Antigravity 5-hour / 7-day rolling quotas and G1 credits."""

from __future__ import annotations

from typing import Any


class QuotaProvider:
    """Fetches 5-hour (`5h`) and 7-day (`weekly`) quota buckets and G1 credit balances."""

    def fetch_quota_groups(self) -> list[dict[str, Any]]:
        """Return raw quota groups containing 5h and weekly buckets."""
        return []

    def fetch_credits(self) -> dict[str, Any]:
        """Return G1 remaining credits and upgrade URI."""
        return {}
