"""5-hour and 7-day rolling quota and session token/credit usage tracker."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Any

from agy_acp.providers.quota_provider import QuotaProvider
from agy_acp.providers.state_store import SqliteStateStoreProvider

RATE_LIMIT_MESSAGE = "Quota limit reached for one or more rolling windows."


@dataclass(frozen=True)
class TurnTelemetry:
    """Engine-neutral per-turn token, context window, credit, and rate-limit snapshot."""

    session_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int
    thinking_tokens: int
    context_tokens: int
    context_window: int
    context_usage_pct: float
    delta_credits: float
    metering_usage: list[dict[str, Any]]
    metadata_params: dict[str, Any]
    usage_update_notification: dict[str, Any]
    rate_limit_message: str | None = None
    rate_limit_params: dict[str, Any] | None = None


def _parse_reset_datetime(reset_time: str) -> datetime | None:
    """Parse an ISO-8601 reset timestamp into a timezone-aware UTC-compatible datetime."""
    if not reset_time:
        return None
    try:
        parsed = datetime.fromisoformat(reset_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def format_reset_countdown(reset_time: str, *, now: datetime | None = None) -> str:
    """Convert an ISO-8601 reset timestamp into a human-readable relative countdown."""
    if not reset_time:
        return "unknown"
    reset_dt = _parse_reset_datetime(reset_time)
    if reset_dt is None:
        return reset_time

    active_now = now or datetime.now(UTC)
    remaining_sec = int((reset_dt - active_now).total_seconds())
    if remaining_sec <= 0:
        return "now"

    days, rem = divmod(remaining_sec, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{seconds}s")
    return f"in {' '.join(parts)}"


def format_local_reset_time(
    reset_time: str,
    *,
    now: datetime | None = None,
    local_tz: tzinfo | None = None,
) -> str:
    """Format an ISO-8601 reset timestamp in local system time with a relative countdown."""
    if not reset_time:
        return "unknown"
    reset_dt = _parse_reset_datetime(reset_time)
    if reset_dt is None:
        return reset_time

    local_dt = reset_dt.astimezone(local_tz)
    hour_12 = local_dt.hour % 12 or 12
    am_pm = "AM" if local_dt.hour < 12 else "PM"
    tz_label = local_dt.tzname() or local_dt.strftime("%z")
    countdown = format_reset_countdown(reset_time, now=now)
    return f"{local_dt:%Y-%m-%d} {hour_12}:{local_dt:%M} {am_pm} {tz_label} ({countdown})"


def format_local_reset_date(
    reset_time: str,
    *,
    local_tz: tzinfo | None = None,
) -> str:
    """Format an ISO-8601 reset timestamp as a local `YYYY-MM-DD` date string."""
    if not reset_time:
        return ""
    reset_dt = _parse_reset_datetime(reset_time)
    if reset_dt is None:
        return reset_time.split("T", 1)[0]
    return f"{reset_dt.astimezone(local_tz):%Y-%m-%d}"


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
        self._session_context_tokens: dict[str, int] = {}
        self._last_session_effective_credits: dict[str, float] = {}

    def evict_session(self, session_id: str) -> None:
        """Evict per-session token and credit baseline tracking for `session_id`."""
        if not session_id:
            return
        self._session_context_tokens.pop(session_id, None)
        self._last_session_effective_credits.pop(session_id, None)

    async def seed_session_baseline(
        self,
        session_id: str,
        *,
        model_id: str | None = None,
    ) -> None:
        """Seed pre-turn effective Kiro credits for `session_id` if not yet recorded."""
        if not session_id or session_id in self._last_session_effective_credits:
            return
        with contextlib.suppress(Exception):
            five_hour, seven_day = await self._fetch_both_quotas()
            credits_info = await self._provider.fetch_credits()
            g1_overage = int(credits_info.get("remaining_credits", 0))
            kiro_credits = self.calculate_effective_kiro_credits(
                five_hour["buckets"],
                seven_day["buckets"],
                model_id=model_id,
                g1_overage_credits=g1_overage,
            )
            self._last_session_effective_credits[session_id] = float(
                kiro_credits["effective_credits"]
            )

    def get_session_context_tokens(self, session_id: str) -> int | None:
        """Return the active context window token count for `session_id`, if recorded."""
        cached = self._session_context_tokens.get(session_id)
        if cached is not None:
            return cached
        if self._store is not None and session_id:
            session = self._store.get_session(session_id)
            if session is not None:
                persisted = int(session.get("contextTokens", 0))
                if persisted > 0:
                    self._session_context_tokens[session_id] = persisted
                    return persisted
        return None

    def build_usage_update_notification(
        self,
        *,
        session_id: str,
        context_tokens: int,
    ) -> dict[str, Any]:
        """Record `context_tokens` for `session_id` and build an ACP `usage_update` frame."""
        active_context_tokens = max(0, int(context_tokens))
        self._session_context_tokens[session_id] = active_context_tokens
        if self._store is not None and session_id:
            session = self._store.get_session(session_id)
            if (
                session is not None
                and int(session.get("contextTokens", 0)) != active_context_tokens
            ):
                session["contextTokens"] = active_context_tokens
                self._store.upsert_session(session)
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "usage_update",
                    "used": active_context_tokens,
                    "size": self.CONTEXT_WINDOW_SIZE,
                    "cost": {"amount": 0.0, "currency": "USD"},
                },
            },
        }

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

    async def _fetch_both_quotas(
        self,
        *,
        group: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch raw quota groups once and partition into formatted 5h and 7d snapshots."""
        raw = await self._extract_raw_buckets(group=group)
        five_hour_buckets = [
            self._format_bucket(bucket, window="5h")
            for bucket in raw
            if bucket.get("window") == "5h"
        ]
        seven_day_buckets = [
            self._format_bucket(bucket, window="7d")
            for bucket in raw
            if bucket.get("window") in {"weekly", "7d"}
        ]
        return (
            {"window": "5h", "buckets": five_hour_buckets},
            {"window": "7d", "buckets": seven_day_buckets},
        )

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
        five_hour, seven_day = await self._fetch_both_quotas(group=group)
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
        five_hour, seven_day = await self._fetch_both_quotas(group=group)
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

    async def build_turn_telemetry(
        self,
        *,
        session_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int = 0,
        thinking_tokens: int = 0,
        context_tokens: int | None = None,
    ) -> TurnTelemetry:
        """Record turn tokens and return structured engine-neutral `TurnTelemetry`."""
        self.record_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
        )
        active_context_tokens = (
            max(0, int(context_tokens))
            if context_tokens is not None
            else (input_tokens + output_tokens + cached_read_tokens + thinking_tokens)
        )
        usage_notification = self.build_usage_update_notification(
            session_id=session_id,
            context_tokens=active_context_tokens,
        )

        five_hour, seven_day = await self._fetch_both_quotas()
        credits_info = await self._provider.fetch_credits()
        g1_overage = int(credits_info.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour["buckets"],
            seven_day["buckets"],
            model_id=model_id,
            g1_overage_credits=g1_overage,
        )
        context_usage_pct = round((active_context_tokens / self.CONTEXT_WINDOW_SIZE) * 100.0, 4)
        effective_credits = float(kiro_credits["effective_credits"])
        prev_credits = self._last_session_effective_credits.get(session_id)
        self._last_session_effective_credits[session_id] = effective_credits
        delta_credits = (
            round(prev_credits - effective_credits, 4)
            if prev_credits is not None and prev_credits > effective_credits
            else 0.0
        )
        metering_usage: list[dict[str, Any]] = [
            {
                "unit": "credit",
                "unitPlural": "credits",
                "value": delta_credits,
            }
        ]
        metadata_params: dict[str, Any] = {
            "sessionId": session_id,
            "contextUsagePercentage": context_usage_pct,
            "contextWindowSize": self.CONTEXT_WINDOW_SIZE,
            "meteringUsage": metering_usage,
            "credits": effective_credits,
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
        }

        all_buckets = [*five_hour["buckets"], *seven_day["buckets"]]
        exhausted = [bucket for bucket in all_buckets if float(bucket["remaining_fraction"]) <= 0.0]
        rate_limit_message: str | None = None
        rate_limit_params: dict[str, Any] | None = None
        if exhausted:
            rate_limit_message = RATE_LIMIT_MESSAGE
            rate_limit_params = {
                "sessionId": session_id,
                "modelId": model_id,
                "message": rate_limit_message,
                "exhaustedBuckets": exhausted,
                "resetTime": exhausted[0]["reset_time"],
            }

        return TurnTelemetry(
            session_id=session_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
            context_tokens=active_context_tokens,
            context_window=self.CONTEXT_WINDOW_SIZE,
            context_usage_pct=context_usage_pct,
            delta_credits=delta_credits,
            metering_usage=metering_usage,
            metadata_params=metadata_params,
            usage_update_notification=usage_notification,
            rate_limit_message=rate_limit_message,
            rate_limit_params=rate_limit_params,
        )

    compute_turn_telemetry = build_turn_telemetry

    async def build_kirocrew_wire_updates(
        self,
        *,
        session_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int = 0,
        thinking_tokens: int = 0,
        context_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build `usage_update` and `_kiro.dev/metadata` notifications for KiroCrew."""
        telemetry = await self.build_turn_telemetry(
            session_id=session_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_read_tokens=cached_read_tokens,
            thinking_tokens=thinking_tokens,
            context_tokens=context_tokens,
        )
        notifications: list[dict[str, Any]] = [
            telemetry.usage_update_notification,
            {
                "jsonrpc": "2.0",
                "method": "_kiro.dev/metadata",
                "params": telemetry.metadata_params,
            },
        ]
        if telemetry.rate_limit_params is not None:
            notifications.append(
                {
                    "jsonrpc": "2.0",
                    "method": "_kiro.dev/error/rate_limit",
                    "params": telemetry.rate_limit_params,
                }
            )
        return notifications

    async def format_slash_command_response(
        self,
        command: str,
        *,
        now: datetime | None = None,
        local_tz: tzinfo | None = None,
    ) -> str:
        """Render `/quota`, `/usage`, or `/credits` Markdown table for KiroCrew chat."""
        normalized = command.strip().lower() or "/quota"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        five_hour_snap, seven_day_snap = await self._fetch_both_quotas()
        five_hour = five_hour_snap["buckets"]
        seven_day = seven_day_snap["buckets"]
        credits = await self._provider.fetch_credits()
        tokens = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_read_tokens": self.cached_read_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
        }
        credits_and_tokens = {
            "remaining_credits": int(credits.get("remaining_credits", 0)),
            "upgrade_uri": credits.get("upgrade_uri"),
        }

        lines = [
            f"### Antigravity Quota & Usage (`{normalized}`)",
            "",
            "| Group | Window | Label | Remaining | Used | Reset Time |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for bucket in [*five_hour, *seven_day]:
            window_label = "Five Hour (5h)" if bucket["window"] == "5h" else "Weekly (7d)"
            reset_display = format_local_reset_time(
                str(bucket["reset_time"]),
                now=now,
                local_tz=local_tz,
            )
            lines.append(
                f"| {bucket['group']} | {window_label} | {bucket['name']} | "
                f"{bucket['remaining_percent']:.1f}% | {bucket['used_percent']:.1f}% | "
                f"{reset_display} |"
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
