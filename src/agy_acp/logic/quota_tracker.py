"""5-hour and 7-day rolling quota and session token/credit usage tracker."""

from __future__ import annotations

from typing import Any


class QuotaTracker:
    """Aggregates 5-hour (`5h`) and 7-day (`7d`) quota buckets, credits, and token usage."""

    def get_five_hour_quota(self, *, group: str | None = None) -> dict[str, Any]:
        """Return 5-hour rolling quota buckets and reset times."""
        return {"window": "", "buckets": []}

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
