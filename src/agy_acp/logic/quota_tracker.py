"""5-hour and 7-day rolling quota and session token/credit usage tracker."""

from __future__ import annotations

from typing import Any

from agy_acp.providers.quota_provider import QuotaProvider
from agy_acp.providers.state_store import SqliteStateStoreProvider


class QuotaTracker:
    """Aggregates 5-hour (`5h`) and 7-day (`7d`) quota buckets, credits, and token usage."""

    CONTEXT_WINDOW_SIZE: int = 1_000_000
    WEEKLY_CREDIT_CAPACITY: float = 1_000.0
    FIVE_HOUR_CREDIT_CAPACITY: float = 200.0

    def __init__(
        self,
        *,
        provider: QuotaProvider | None = None,
        state_store: SqliteStateStoreProvider | None = None,
    ) -> None:
        self._provider = provider or QuotaProvider()
        self._store = state_store
        initial = (
            self._store.get_token_ledger()
            if self._store is not None
            else {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_read_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
            }
        )
        self.input_tokens: int = initial["input_tokens"]
        self.output_tokens: int = initial["output_tokens"]
        self.cached_read_tokens: int = initial["cached_read_tokens"]
        self.thinking_tokens: int = initial["thinking_tokens"]
        self.total_tokens: int = initial["total_tokens"]
        self._total_tokens: int = self.total_tokens

    def record_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_read_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> None:
        """Accumulate session token usage into the ledger and persist to SQLite if configured."""
        if self._store is not None:
            ledger = self._store.record_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_read_tokens=cached_read_tokens,
                thinking_tokens=thinking_tokens,
            )
            self.input_tokens = ledger["input_tokens"]
            self.output_tokens = ledger["output_tokens"]
            self.cached_read_tokens = ledger["cached_read_tokens"]
            self.thinking_tokens = ledger["thinking_tokens"]
            self.total_tokens = ledger["total_tokens"]
            self._total_tokens = self.total_tokens
            return

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_read_tokens += cached_read_tokens
        self.thinking_tokens += thinking_tokens
        self.total_tokens = (
            self.input_tokens + self.output_tokens + self.cached_read_tokens + self.thinking_tokens
        )
        self._total_tokens = self.total_tokens

    async def _extract_raw_buckets(self, *, group: str | None = None) -> list[dict[str, Any]]:
        """Flatten quota groups into raw bucket dicts, optionally filtered by group."""
        raw_buckets: list[dict[str, Any]] = []
        for entry in await self._provider.fetch_quota_groups():
            nested = entry.get("buckets")
            candidates = nested if isinstance(nested, list) else [entry]
            for candidate in candidates:
                bucket_group = str(
                    candidate.get("group") or entry.get("group") or entry.get("name") or ""
                )
                if group is not None and bucket_group.lower() != group.lower():
                    continue
                raw_buckets.append({**candidate, "group": bucket_group})
        return raw_buckets

    @classmethod
    def _format_bucket(cls, raw_bucket: dict[str, Any], *, window: str) -> dict[str, Any]:
        """Enrich a raw quota bucket with remaining/used percentages and Kiro credit equivalents."""
        remaining_fraction = float(raw_bucket["remaining_fraction"])
        remaining_percent = round(remaining_fraction * 100.0, 2)
        used_percent = round((1.0 - remaining_fraction) * 100.0, 2)
        credit_capacity = (
            cls.WEEKLY_CREDIT_CAPACITY
            if window in {"7d", "weekly"}
            else cls.FIVE_HOUR_CREDIT_CAPACITY
        )
        remaining_credits = round(remaining_fraction * credit_capacity, 2)
        return {
            "id": raw_bucket["id"],
            "group": raw_bucket["group"],
            "name": raw_bucket["name"],
            "window": window,
            "remaining_fraction": remaining_fraction,
            "remaining_percent": remaining_percent,
            "used_percent": used_percent,
            "credit_capacity": credit_capacity,
            "remaining_credits": remaining_credits,
            "reset_time": raw_bucket["reset_time"],
        }

    @classmethod
    def calculate_effective_kiro_credits(
        cls,
        five_hour_buckets: list[dict[str, Any]],
        seven_day_buckets: list[dict[str, Any]],
        *,
        model_id: str | None = None,
        g1_overage_credits: int = 0,
    ) -> dict[str, Any]:
        """Convert 7d (1000) and 5h (200) quotas to Kiro credits, picking min and its reset."""
        prefer_gemini = model_id is None or "gemini" in model_id.lower()

        def _pick_group_bucket(buckets: list[dict[str, Any]]) -> dict[str, Any] | None:
            for bucket in buckets:
                is_gemini = "gemini" in str(bucket.get("group", "")).lower()
                if is_gemini == prefer_gemini:
                    return bucket
            return buckets[0] if buckets else None

        bucket_5h = _pick_group_bucket(five_hour_buckets)
        bucket_7d = _pick_group_bucket(seven_day_buckets)

        credits_5h = (
            float(
                bucket_5h.get(
                    "remaining_credits",
                    float(bucket_5h.get("remaining_fraction", 0.0)) * cls.FIVE_HOUR_CREDIT_CAPACITY,
                )
            )
            if bucket_5h is not None
            else None
        )
        credits_7d = (
            float(
                bucket_7d.get(
                    "remaining_credits",
                    float(bucket_7d.get("remaining_fraction", 0.0)) * cls.WEEKLY_CREDIT_CAPACITY,
                )
            )
            if bucket_7d is not None
            else None
        )

        if (
            bucket_7d is not None
            and credits_7d is not None
            and (credits_5h is None or credits_7d <= credits_5h)
        ):
            bottleneck_window = "7d"
            quota_credits = round(credits_7d, 2)
            base_capacity = cls.WEEKLY_CREDIT_CAPACITY
            reset_time = str(bucket_7d.get("reset_time", ""))
            active_group = str(bucket_7d.get("group", ""))
        elif bucket_5h is not None and credits_5h is not None:
            bottleneck_window = "5h"
            quota_credits = round(credits_5h, 2)
            base_capacity = cls.FIVE_HOUR_CREDIT_CAPACITY
            reset_time = str(bucket_5h.get("reset_time", ""))
            active_group = str(bucket_5h.get("group", ""))
        else:
            bottleneck_window = "5h"
            quota_credits = 0.0
            base_capacity = cls.FIVE_HOUR_CREDIT_CAPACITY
            reset_time = ""
            active_group = "Gemini Models" if prefer_gemini else "Claude and GPT models"

        effective_credits = round(quota_credits + float(g1_overage_credits), 2)
        return {
            "effective_credits": effective_credits,
            "quota_credits": quota_credits,
            "g1_overage_credits": g1_overage_credits,
            "credit_capacity": round(base_capacity + float(g1_overage_credits), 2),
            "bottleneck_window": bottleneck_window,
            "group": active_group,
            "reset_time": reset_time,
            "five_hour_credits": round(credits_5h, 2) if credits_5h is not None else 0.0,
            "seven_day_credits": round(credits_7d, 2) if credits_7d is not None else 0.0,
        }

    async def get_five_hour_quota(self, *, group: str | None = None) -> dict[str, Any]:
        """Return 5-hour rolling quota buckets and reset times."""
        buckets = [
            self._format_bucket(bucket, window="5h")
            for bucket in await self._extract_raw_buckets(group=group)
            if bucket.get("window") == "5h"
        ]
        return {"window": "5h", "buckets": buckets}

    async def get_seven_day_quota(self, *, group: str | None = None) -> dict[str, Any]:
        """Return 7-day (weekly) rolling quota buckets and reset times."""
        buckets = [
            self._format_bucket(bucket, window="7d")
            for bucket in await self._extract_raw_buckets(group=group)
            if bucket.get("window") in {"weekly", "7d"}
        ]
        return {"window": "7d", "buckets": buckets}

    async def get_credits_and_tokens(self, *, group: str | None = None) -> dict[str, Any]:
        """Return G1 credit balance, converted Kiro quota credits, and session token ledger."""
        credits = await self._provider.fetch_credits()
        five_hour = await self.get_five_hour_quota(group=group)
        seven_day = await self.get_seven_day_quota(group=group)
        g1_overage = int(credits.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour["buckets"],
            seven_day["buckets"],
            model_id=group,
            g1_overage_credits=g1_overage,
        )
        return {
            "remaining_credits": g1_overage,
            "upgrade_uri": credits.get("upgrade_uri"),
            "kiro_credits": kiro_credits,
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_read_tokens": self.cached_read_tokens,
                "thinking_tokens": self.thinking_tokens,
                "total_tokens": self.total_tokens,
            },
        }

    async def get_unified_status(self, *, group: str | None = None) -> dict[str, Any]:
        """Return combined 5-hour, 7-day, credit, and token usage snapshot."""
        five_hour = await self.get_five_hour_quota(group=group)
        seven_day = await self.get_seven_day_quota(group=group)
        raw_credits = await self._provider.fetch_credits()
        g1_overage = int(raw_credits.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour["buckets"],
            seven_day["buckets"],
            model_id=group,
            g1_overage_credits=g1_overage,
        )
        return {
            "five_hour": five_hour,
            "seven_day": seven_day,
            "credits": {
                **raw_credits,
                "kiro_credits": kiro_credits,
            },
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_read_tokens": self.cached_read_tokens,
                "thinking_tokens": self.thinking_tokens,
                "total_tokens": self.total_tokens,
            },
        }

    async def build_kirocrew_wire_updates(
        self,
        *,
        session_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        """Build `usage_update` and `_kiro.dev/metadata` notifications for KiroCrew."""
        self.record_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
        )
        five_hour = await self.get_five_hour_quota()
        seven_day = await self.get_seven_day_quota()
        credits_info = await self._provider.fetch_credits()
        g1_overage = int(credits_info.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour["buckets"],
            seven_day["buckets"],
            model_id=model_id,
            g1_overage_credits=g1_overage,
        )
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
                    "credits": kiro_credits["effective_credits"],
                    "quotaCredits": kiro_credits["quota_credits"],
                    "creditsCapacity": kiro_credits["credit_capacity"],
                    "creditsBottleneckWindow": kiro_credits["bottleneck_window"],
                    "creditsResetTime": kiro_credits["reset_time"],
                    "g1OverageCredits": g1_overage,
                    "modelId": model_id,
                    "_meta": {
                        "quota": {
                            "five_hour": five_hour,
                            "seven_day": seven_day,
                            "kiro_credits": kiro_credits,
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

    async def format_slash_command_response(self, command: str) -> str:
        """Render `/quota`, `/usage`, or `/credits` Markdown table for KiroCrew chat."""
        normalized = command.strip().lower() or "/quota"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        five_hour = (await self.get_five_hour_quota())["buckets"]
        seven_day = (await self.get_seven_day_quota())["buckets"]
        credits_and_tokens = await self.get_credits_and_tokens()
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
