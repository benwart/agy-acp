"""5-hour and 7-day rolling quota and session token/credit usage tracker."""

from __future__ import annotations

from typing import Any

from agy_acp.providers.quota_provider import QuotaProvider


class QuotaTracker:
    """Aggregates 5-hour (`5h`) and 7-day (`7d`) quota buckets, credits, and token usage."""

    CONTEXT_WINDOW_SIZE: int = 1_000_000

    def __init__(self, *, provider: QuotaProvider | None = None) -> None:
        self._provider = provider or QuotaProvider()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_read_tokens: int = 0
        self.thinking_tokens: int = 0
        self.total_tokens: int = 0
        self._total_tokens: int = 0

    def record_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_read_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> None:
        """Accumulate session token usage into the ledger."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_read_tokens += cached_read_tokens
        self.thinking_tokens += thinking_tokens
        self.total_tokens = (
            self.input_tokens + self.output_tokens + self.cached_read_tokens + self.thinking_tokens
        )
        self._total_tokens = self.total_tokens

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
        buckets = [
            self._format_bucket(bucket, window="7d")
            for bucket in self._extract_raw_buckets(group=group)
            if bucket.get("window") in {"weekly", "7d"}
        ]
        return {"window": "7d", "buckets": buckets}

    def get_credits_and_tokens(self) -> dict[str, Any]:
        """Return G1 credit balance and cumulative session token ledger."""
        credits = self._provider.fetch_credits()
        return {
            "remaining_credits": credits.get("remaining_credits", 0),
            "upgrade_uri": credits.get("upgrade_uri"),
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_read_tokens": self.cached_read_tokens,
                "thinking_tokens": self.thinking_tokens,
                "total_tokens": self.total_tokens,
            },
        }

    def get_unified_status(self, *, group: str | None = None) -> dict[str, Any]:
        """Return combined 5-hour, 7-day, credit, and token usage snapshot."""
        credits_and_tokens = self.get_credits_and_tokens()
        return {
            "five_hour": self.get_five_hour_quota(group=group),
            "seven_day": self.get_seven_day_quota(group=group),
            "credits": self._provider.fetch_credits(),
            "token_usage": credits_and_tokens["token_usage"],
        }

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
        self.record_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
        )
        five_hour = self.get_five_hour_quota()
        seven_day = self.get_seven_day_quota()
        credits_info = self._provider.fetch_credits()
        remaining_credits = credits_info.get("remaining_credits", 0)
        context_usage_pct = round((self._total_tokens / self.CONTEXT_WINDOW_SIZE) * 100.0, 4)

        notifications: list[dict[str, Any]] = [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "usage_update",
                        "used": self._total_tokens,
                        "size": self.CONTEXT_WINDOW_SIZE,
                        "cost": {"amount": 0.0, "currency": "USD"},
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "_kiro.dev/metadata",
                "params": {
                    "sessionId": session_id,
                    "contextUsagePercentage": context_usage_pct,
                    "contextWindowSize": self.CONTEXT_WINDOW_SIZE,
                    "credits": remaining_credits,
                    "modelId": model_id,
                    "_meta": {
                        "quota": {
                            "five_hour": five_hour,
                            "seven_day": seven_day,
                        }
                    },
                },
            },
        ]

        all_buckets = [*five_hour["buckets"], *seven_day["buckets"]]
        exhausted = [bucket for bucket in all_buckets if float(bucket["remaining_fraction"]) <= 0.0]
        if exhausted:
            notifications.append(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/error/rate_limit",
                    "params": {
                        "sessionId": session_id,
                        "modelId": model_id,
                        "message": "Quota limit reached for one or more rolling windows.",
                        "exhaustedBuckets": exhausted,
                        "resetTime": exhausted[0]["reset_time"],
                    },
                }
            )

        return notifications

    def format_slash_command_response(self, command: str) -> str:
        """Render `/quota`, `/usage`, or `/credits` Markdown table for KiroCrew chat."""
        normalized = command.strip().lower() or "/quota"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        five_hour = self.get_five_hour_quota()["buckets"]
        seven_day = self.get_seven_day_quota()["buckets"]
        credits_and_tokens = self.get_credits_and_tokens()
        tokens = credits_and_tokens["token_usage"]

        lines = [
            f"### Antigravity Quota & Usage (`{normalized}`)",
            "",
            "| Group | Window | Label | Remaining | Used | Reset Time |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for bucket in [*five_hour, *seven_day]:
            window_label = "Five Hour (5h)" if bucket["window"] == "5h" else "Weekly (7d)"
            lines.append(
                f"| {bucket['group']} | {window_label} | {bucket['name']} | "
                f"{bucket['remaining_percent']:.1f}% | {bucket['used_percent']:.1f}% | "
                f"{bucket['reset_time']} |"
            )

        lines.extend(
            [
                "",
                "**Credits & Session Tokens**",
                f"- Remaining Credits: `{credits_and_tokens['remaining_credits']}` "
                f"({credits_and_tokens['upgrade_uri']})",
                f"- Input Tokens: `{tokens['input_tokens']}`",
                f"- Output Tokens: `{tokens['output_tokens']}`",
                f"- Cached Read Tokens: `{tokens['cached_read_tokens']}`",
                f"- Thinking Tokens: `{tokens['thinking_tokens']}`",
                f"- Total Tokens: `{tokens['total_tokens']}`",
            ]
        )
        return "\n".join(lines)
