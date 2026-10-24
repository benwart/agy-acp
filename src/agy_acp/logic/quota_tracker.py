"""5-hour and 7-day rolling quota and session token/credit usage tracker."""

from __future__ import annotations

from typing import Any

from agy_acp.providers.quota_provider import QuotaProvider


class QuotaTracker:
    """Aggregates 5-hour (`5h`) and 7-day (`7d`) quota buckets, credits, and token usage."""

    def __init__(self, *, provider: QuotaProvider | None = None) -> None:
        self._provider = provider or QuotaProvider()

    def _extract_raw_buckets(self, *, group: str | None = None) -> list[dict[str, Any]]:
        """Flatten quota groups into raw bucket dicts, optionally filtered by group."""
        raw_buckets: list[dict[str, Any]] = []
        for entry in self._provider.fetch_quota_groups():
            nested = entry.get("buckets")
            candidates = nested if isinstance(nested, list) else [entry]
            for candidate in candidates:
                bucket_group = str(candidate.get("group") or entry.get("group") or "")
                if group is not None and bucket_group.lower() != group.lower():
                    continue
                raw_buckets.append({**candidate, "group": bucket_group})
        return raw_buckets

    @staticmethod
    def _format_bucket(raw_bucket: dict[str, Any], *, window: str) -> dict[str, Any]:
        """Enrich a raw quota bucket with remaining and used percentages."""
        remaining_fraction = float(raw_bucket["remaining_fraction"])
        remaining_percent = round(remaining_fraction * 100.0, 2)
        used_percent = round((1.0 - remaining_fraction) * 100.0, 2)
        return {
            "id": raw_bucket["id"],
            "group": raw_bucket["group"],
            "name": raw_bucket["name"],
            "window": window,
            "remaining_fraction": remaining_fraction,
            "remaining_percent": remaining_percent,
            "used_percent": used_percent,
            "reset_time": raw_bucket["reset_time"],
        }

    def get_five_hour_quota(self, *, group: str | None = None) -> dict[str, Any]:
        """Return 5-hour rolling quota buckets and reset times."""
        buckets = [
            self._format_bucket(bucket, window="5h")
            for bucket in self._extract_raw_buckets(group=group)
            if bucket.get("window") == "5h"
        ]
        return {"window": "5h", "buckets": buckets}

    def get_seven_day_quota(self, *, group: str | None = None) -> dict[str, Any]:
        """Return 7-day (weekly) rolling quota buckets and reset times."""
        return {"window": "", "buckets": []}

    def get_credits_and_tokens(self) -> dict[str, Any]:
        """Return G1 credit balance and cumulative session token ledger."""
        return {}

    def get_unified_status(self, *, group: str | None = None) -> dict[str, Any]:
        """Return combined 5-hour, 7-day, credit, and token usage snapshot."""
        return {}

    def build_kirocrew_wire_updates(
        self,
        *,
        session_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        """Build `usage_update` and `_kiro.dev/metadata` notifications for KiroCrew."""
        return []

    def format_slash_command_response(self, command: str) -> str:
        """Render `/quota`, `/usage`, or `/credits` Markdown table for KiroCrew chat."""
        return ""
