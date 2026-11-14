"""5-hour and 7-day rolling quota and session token/credit usage tracker."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Any

from agy_acp.providers.quota_provider import QuotaProvider
from agy_acp.providers.state_store import SqliteStateStoreProvider
from agy_acp.providers.telemetry import TelemetryCallbacks, get_telemetry

RATE_LIMIT_MESSAGE = "Quota limit reached for one or more rolling windows."


@dataclass(frozen=True)
class PlanCreditSpec:
    """Credit capacities mapped from an Antigravity Google AI plan tier."""

    tier_id: str
    label: str
    weekly_capacity: float
    five_hour_capacity: float


PLAN_CREDIT_SPECS: dict[str, PlanCreditSpec] = {
    "g1-ultra-tier": PlanCreditSpec(
        tier_id="g1-ultra-tier",
        label="Google AI Ultra (20x)",
        weekly_capacity=4_000.0,
        five_hour_capacity=800.0,
    ),
    "g1-ultra-lite-tier": PlanCreditSpec(
        tier_id="g1-ultra-lite-tier",
        label="Google AI Ultra (5x)",
        weekly_capacity=1_000.0,
        five_hour_capacity=200.0,
    ),
    "g1-pro-tier": PlanCreditSpec(
        tier_id="g1-pro-tier",
        label="Google AI Pro (1x)",
        weekly_capacity=200.0,
        five_hour_capacity=40.0,
    ),
    "g1-plus-tier": PlanCreditSpec(
        tier_id="g1-plus-tier",
        label="Google AI Plus",
        weekly_capacity=100.0,
        five_hour_capacity=20.0,
    ),
    "free-tier": PlanCreditSpec(
        tier_id="free-tier",
        label="Free",
        weekly_capacity=40.0,
        five_hour_capacity=8.0,
    ),
}

DEFAULT_PLAN_CREDIT_SPEC: PlanCreditSpec = PLAN_CREDIT_SPECS["g1-ultra-lite-tier"]


def resolve_plan_credit_spec(tier: Mapping[str, Any] | str | None = None) -> PlanCreditSpec:
    """Resolve weekly (`7d`) and 5-hour (`5h`) credit capacities for `tier`."""
    raw_id = ""
    if isinstance(tier, Mapping):
        raw_id = str(tier.get("id") or "")
    elif isinstance(tier, str):
        raw_id = tier
    normalized = raw_id.strip().lower().replace("_", "-")
    if not normalized:
        return DEFAULT_PLAN_CREDIT_SPEC
    return PLAN_CREDIT_SPECS.get(normalized, DEFAULT_PLAN_CREDIT_SPEC)


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
        self._store = (
            state_store
            if state_store is not None
            else (SqliteStateStoreProvider() if provider is None else None)
        )
        self._provider = provider or QuotaProvider()
        self._cached_tier: dict[str, str] | None = None
        self._tier_probed: bool = False
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
        self._last_otel_buckets: list[dict[str, Any]] = []
        self._last_otel_remaining_credits: float | None = None
        self._last_otel_capacity_credits: float | None = None
        db_ref = self._store.db_path if self._store is not None else None
        get_telemetry(db_ref).register_callbacks(
            TelemetryCallbacks(quota_metrics=self._get_otel_quota_snapshot)
        )

    def _update_otel_quota_snapshot(
        self,
        five_hour_buckets: list[dict[str, Any]],
        seven_day_buckets: list[dict[str, Any]],
        kiro_credits: Mapping[str, Any],
    ) -> None:
        buckets: list[dict[str, Any]] = []
        for win_label, items in (("5h", five_hour_buckets), ("7d", seven_day_buckets)):
            for b in items:
                frac = float(b.get("remaining_fraction", 0.0))
                buckets.append(
                    {
                        "group": str(b.get("group") or b.get("name") or "default"),
                        "window": win_label,
                        "percent": round(max(0.0, min(100.0, frac * 100.0)), 2),
                    }
                )
        self._last_otel_buckets = buckets
        self._last_otel_remaining_credits = float(kiro_credits.get("effective_credits", 0.0))
        self._last_otel_capacity_credits = float(kiro_credits.get("credit_capacity", 0.0))

    def _get_otel_quota_snapshot(
        self,
    ) -> tuple[list[dict[str, Any]], float | None, float | None]:
        if self._last_otel_buckets:
            return (
                list(self._last_otel_buckets),
                self._last_otel_remaining_credits,
                self._last_otel_capacity_credits,
            )
        loader = getattr(self, "_load_cached_quota_groups", None)
        cached_groups = loader() if callable(loader) else None
        if not cached_groups:
            return [], self._last_otel_remaining_credits, self._last_otel_capacity_credits
        derived: list[dict[str, Any]] = []
        for grp in cached_groups:
            grp_name = str(grp.get("group") or grp.get("name") or "default")
            for b in grp.get("buckets", []) if isinstance(grp.get("buckets"), list) else []:
                if isinstance(b, Mapping) and "remaining_fraction" in b:
                    raw_win = str(b.get("window") or "5h").lower()
                    win = "7d" if "week" in raw_win or "7" in raw_win else "5h"
                    derived.append(
                        {
                            "group": grp_name,
                            "window": win,
                            "percent": round(float(b["remaining_fraction"]) * 100.0, 2),
                        }
                    )
        return derived, self._last_otel_remaining_credits, self._last_otel_capacity_credits

    async def _resolve_tier(self) -> dict[str, str] | None:
        """Resolve active Antigravity plan tier from SQLite cache, provider probe, or fallback."""
        if self._store is not None:
            stored = self._store.load_tier()
            if stored is not None:
                self._cached_tier = stored
                return stored
        if self._cached_tier is not None:
            return self._cached_tier
        if not self._tier_probed:
            self._tier_probed = True
            fetch_tier = getattr(self._provider, "fetch_tier", None)
            if callable(fetch_tier):
                with contextlib.suppress(Exception):
                    fetched = await fetch_tier()
                    if isinstance(fetched, Mapping):
                        tier_id = str(fetched.get("id") or "").strip()
                        if tier_id:
                            normalized = {
                                "id": tier_id,
                                "name": str(fetched.get("name") or "").strip(),
                                "upgradeUri": str(fetched.get("upgradeUri") or "").strip(),
                            }
                            if self._store is not None:
                                self._store.save_tier(normalized)
                            self._cached_tier = normalized
                            return normalized
        if self._store is not None:
            stale = self._store.load_tier(max_age_seconds=None)
            if stale is not None:
                self._cached_tier = stale
                return stale
        return None

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
            active_tier = await self._resolve_tier()
            five_hour, seven_day = await self._fetch_both_quotas(tier=active_tier)
            credits_info = await self._provider.fetch_credits()
            g1_overage = int(credits_info.get("remaining_credits", 0))
            kiro_credits = self.calculate_effective_kiro_credits(
                five_hour["buckets"],
                seven_day["buckets"],
                model_id=model_id,
                g1_overage_credits=g1_overage,
                tier=active_tier,
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
    def _format_bucket(
        cls,
        raw_bucket: dict[str, Any],
        *,
        window: str,
        tier: Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        """Enrich a raw quota bucket with remaining/used percentages and Kiro credit equivalents."""
        remaining_fraction = float(raw_bucket["remaining_fraction"])
        remaining_percent = round(remaining_fraction * 100.0, 2)
        used_percent = round((1.0 - remaining_fraction) * 100.0, 2)
        if tier is not None:
            spec = resolve_plan_credit_spec(tier)
            credit_capacity = (
                spec.weekly_capacity if window in {"7d", "weekly"} else spec.five_hour_capacity
            )
        else:
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
        tier: Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        """Convert tier-scaled 7d and 5h quotas to Kiro credits, picking min and its reset."""
        prefer_gemini = model_id is None or "gemini" in model_id.lower()
        spec = resolve_plan_credit_spec(tier) if tier is not None else None
        default_5h_capacity = (
            spec.five_hour_capacity if spec is not None else cls.FIVE_HOUR_CREDIT_CAPACITY
        )
        default_7d_capacity = (
            spec.weekly_capacity if spec is not None else cls.WEEKLY_CREDIT_CAPACITY
        )

        def _pick_group_bucket(buckets: list[dict[str, Any]]) -> dict[str, Any] | None:
            for bucket in buckets:
                is_gemini = "gemini" in str(bucket.get("group", "")).lower()
                if is_gemini == prefer_gemini:
                    return bucket
            return buckets[0] if buckets else None

        bucket_5h = _pick_group_bucket(five_hour_buckets)
        bucket_7d = _pick_group_bucket(seven_day_buckets)

        cap_5h = (
            float(bucket_5h.get("credit_capacity", default_5h_capacity))
            if bucket_5h is not None
            else default_5h_capacity
        )
        cap_7d = (
            float(bucket_7d.get("credit_capacity", default_7d_capacity))
            if bucket_7d is not None
            else default_7d_capacity
        )

        credits_5h = (
            float(
                bucket_5h.get(
                    "remaining_credits",
                    float(bucket_5h.get("remaining_fraction", 0.0)) * cap_5h,
                )
            )
            if bucket_5h is not None
            else None
        )
        credits_7d = (
            float(
                bucket_7d.get(
                    "remaining_credits",
                    float(bucket_7d.get("remaining_fraction", 0.0)) * cap_7d,
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
            base_capacity = cap_7d
            reset_time = str(bucket_7d.get("reset_time", ""))
            active_group = str(bucket_7d.get("group", ""))
        elif bucket_5h is not None and credits_5h is not None:
            bottleneck_window = "5h"
            quota_credits = round(credits_5h, 2)
            base_capacity = cap_5h
            reset_time = str(bucket_5h.get("reset_time", ""))
            active_group = str(bucket_5h.get("group", ""))
        else:
            bottleneck_window = "5h"
            quota_credits = 0.0
            base_capacity = default_5h_capacity
            reset_time = ""
            active_group = "Gemini Models" if prefer_gemini else "Claude and GPT models"

        effective_credits = round(quota_credits + float(g1_overage_credits), 2)
        tier_id = (
            str(tier.get("id") or "").strip()
            if isinstance(tier, Mapping)
            else (tier.strip() if isinstance(tier, str) else "")
        )
        tier_name = (
            str(tier.get("name") or "").strip()
            if isinstance(tier, Mapping)
            else (spec.label if spec is not None and tier_id else "")
        )
        result: dict[str, Any] = {
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
        if tier_id:
            result["tier_id"] = tier_id
            result["tier_name"] = tier_name or resolve_plan_credit_spec(tier_id).label
        return result

    async def _fetch_both_quotas(
        self,
        *,
        group: str | None = None,
        tier: Mapping[str, Any] | str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch raw quota groups once and partition into formatted 5h and 7d snapshots."""
        active_tier = tier if tier is not None else await self._resolve_tier()
        raw = await self._extract_raw_buckets(group=group)
        five_hour_buckets = [
            self._format_bucket(bucket, window="5h", tier=active_tier)
            for bucket in raw
            if bucket.get("window") == "5h"
        ]
        seven_day_buckets = [
            self._format_bucket(bucket, window="7d", tier=active_tier)
            for bucket in raw
            if bucket.get("window") in {"weekly", "7d"}
        ]
        return (
            {"window": "5h", "buckets": five_hour_buckets},
            {"window": "7d", "buckets": seven_day_buckets},
        )

    async def get_five_hour_quota(self, *, group: str | None = None) -> dict[str, Any]:
        """Return 5-hour rolling quota buckets and reset times."""
        active_tier = await self._resolve_tier()
        buckets = [
            self._format_bucket(bucket, window="5h", tier=active_tier)
            for bucket in await self._extract_raw_buckets(group=group)
            if bucket.get("window") == "5h"
        ]
        return {"window": "5h", "buckets": buckets}

    async def get_seven_day_quota(self, *, group: str | None = None) -> dict[str, Any]:
        """Return 7-day (weekly) rolling quota buckets and reset times."""
        active_tier = await self._resolve_tier()
        buckets = [
            self._format_bucket(bucket, window="7d", tier=active_tier)
            for bucket in await self._extract_raw_buckets(group=group)
            if bucket.get("window") in {"weekly", "7d"}
        ]
        return {"window": "7d", "buckets": buckets}

    async def get_credits_and_tokens(self, *, group: str | None = None) -> dict[str, Any]:
        """Return G1 credit balance, converted Kiro quota credits, and session token ledger."""
        active_tier = await self._resolve_tier()
        credits = await self._provider.fetch_credits()
        five_hour, seven_day = await self._fetch_both_quotas(group=group, tier=active_tier)
        g1_overage = int(credits.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour["buckets"],
            seven_day["buckets"],
            model_id=group,
            g1_overage_credits=g1_overage,
            tier=active_tier,
        )
        self._update_otel_quota_snapshot(
            five_hour["buckets"],
            seven_day["buckets"],
            kiro_credits,
        )
        return {
            "remaining_credits": g1_overage,
            "upgrade_uri": credits.get("upgrade_uri"),
            "tier": active_tier,
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
        active_tier = await self._resolve_tier()
        five_hour, seven_day = await self._fetch_both_quotas(group=group, tier=active_tier)
        raw_credits = await self._provider.fetch_credits()
        g1_overage = int(raw_credits.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour["buckets"],
            seven_day["buckets"],
            model_id=group,
            g1_overage_credits=g1_overage,
            tier=active_tier,
        )
        self._update_otel_quota_snapshot(
            five_hour["buckets"],
            seven_day["buckets"],
            kiro_credits,
        )
        return {
            "five_hour": five_hour,
            "seven_day": seven_day,
            "tier": active_tier,
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
            else (self.get_session_context_tokens(session_id) or 0)
        )
        usage_notification = self.build_usage_update_notification(
            session_id=session_id,
            context_tokens=active_context_tokens,
        )

        active_tier = await self._resolve_tier()
        five_hour, seven_day = await self._fetch_both_quotas(tier=active_tier)
        credits_info = await self._provider.fetch_credits()
        g1_overage = int(credits_info.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour["buckets"],
            seven_day["buckets"],
            model_id=model_id,
            g1_overage_credits=g1_overage,
            tier=active_tier,
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
        self._update_otel_quota_snapshot(
            five_hour["buckets"],
            seven_day["buckets"],
            kiro_credits,
        )
        otel = get_telemetry(self._store.db_path if self._store is not None else None)
        otel.record_tokens(
            model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_read_tokens,
            thought_tokens=thinking_tokens,
        )
        otel.record_credits_consumed(model_id, delta_credits)
        otel.record_context_usage(
            model_id,
            used_tokens=active_context_tokens,
            usage_percent=context_usage_pct,
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
        if active_tier is not None and active_tier.get("id"):
            metadata_params["planTierId"] = active_tier["id"]
            metadata_params["planTierName"] = (
                active_tier.get("name") or resolve_plan_credit_spec(active_tier).label
            )

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

        active_tier = await self._resolve_tier()
        five_hour_snap, seven_day_snap = await self._fetch_both_quotas(tier=active_tier)
        five_hour = five_hour_snap["buckets"]
        seven_day = seven_day_snap["buckets"]
        credits = await self._provider.fetch_credits()
        g1_overage = int(credits.get("remaining_credits", 0))
        kiro_credits = self.calculate_effective_kiro_credits(
            five_hour,
            seven_day,
            g1_overage_credits=g1_overage,
            tier=active_tier,
        )
        tokens = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_read_tokens": self.cached_read_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
        }
        credits_and_tokens = {
            "remaining_credits": g1_overage,
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

        lines.extend(["", "**Credits & Session Tokens**"])
        if active_tier is not None and active_tier.get("id"):
            spec = resolve_plan_credit_spec(active_tier)
            tier_label = active_tier.get("name") or spec.label
            lines.append(f"- Plan Tier: `{tier_label}` (`{active_tier['id']}`)")
        lines.extend(
            [
                f"- Effective Kiro Credits: `{kiro_credits['effective_credits']:.1f} / "
                f"{kiro_credits['credit_capacity']:.0f}` "
                f"(`{kiro_credits['bottleneck_window']}` bottleneck)",
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
