"""Reset-immune quota burn-rate deltas and empirical 5h-to-7d capacity ratio estimation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

DEFAULT_5H_PER_7D_RATIO: float = 5.0
_RESET_JUMP_TOLERANCE_SECONDS: float = 60.0
_FRACTION_EPSILON: float = 1e-6
_ROLLING_SUM_CAP_7D: float = 2.0


def normalize_quota_group(raw_group: str | None) -> str:
    """Normalize an Antigravity quota group label into `google_ai` or `anthropic_ai`."""
    cleaned = (raw_group or "").strip().lower()
    if not cleaned:
        return "google_ai"
    if any(tok in cleaned for tok in ("claude", "anthropic", "gpt")):
        return "anthropic_ai"
    if any(tok in cleaned for tok in ("gemini", "google")):
        return "google_ai"
    return "google_ai"


def classify_model_quota_group(model_id: str | None) -> str:
    """Map a concrete `model_id` onto its underlying provider quota pool."""
    cleaned = (model_id or "").strip().lower()
    if not cleaned or "gemini" in cleaned or "google" in cleaned:
        return "google_ai"
    return "anthropic_ai"


def parse_reset_epoch_seconds(reset_time: str | None) -> float | None:
    """Parse an ISO-8601 `reset_time` string into Unix epoch seconds, or `None` if invalid."""
    if not reset_time:
        return None
    try:
        parsed = datetime.fromisoformat(reset_time.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


@dataclass(frozen=True, slots=True)
class BucketGaugePoint:
    """Normalized live quota gauge observation for a single `(quota_group, window)` bucket."""

    quota_group: str
    group: str
    window: str
    remaining_fraction: float
    used_fraction: float
    remaining_percent: float
    remaining_credits: float
    reset_timestamp_seconds: float | None


@dataclass(frozen=True, slots=True)
class BucketBurnEvent:
    """Monotonic non-resetting quota fraction and credit delta attributed to a model."""

    quota_group: str
    window: str
    model_id: str
    fraction_burnt: float
    credits_burnt: float


@dataclass(frozen=True, slots=True)
class QuotaBurnObservation:
    """Combined gauge snapshot, monotonic burn deltas, and empirical 5h/7d ratios."""

    bucket_gauges: list[BucketGaugePoint]
    burnt_deltas: list[BucketBurnEvent]
    ratios_by_group: dict[str, float]
    state_changed: bool


@dataclass(slots=True)
class _WindowBaseline:
    remaining_fraction: float
    reset_time: str


@dataclass(slots=True)
class _RatioAccumulator:
    sum_delta_5h: float = 0.0
    sum_delta_7d: float = 0.0
    samples: int = 0


class QuotaBurnLedger:
    """Tracks per-window quota baselines to emit reset-immune burn deltas and 5h/7d ratio `R`."""

    def __init__(self, initial_state: Mapping[str, Any] | None = None) -> None:
        self._baselines: dict[tuple[str, str], _WindowBaseline] = {}
        self._ratios: dict[str, _RatioAccumulator] = {
            "google_ai": _RatioAccumulator(),
            "anthropic_ai": _RatioAccumulator(),
        }
        self._last_model_by_group: dict[str, str] = {}
        if isinstance(initial_state, Mapping):
            self._load_state(initial_state)

    def _load_state(self, state: Mapping[str, Any]) -> None:
        raw_baselines = state.get("baselines")
        if isinstance(raw_baselines, Mapping):
            for key, entry in raw_baselines.items():
                if not isinstance(key, str) or ":" not in key or not isinstance(entry, Mapping):
                    continue
                q_group, win = key.split(":", 1)
                if "remaining_fraction" not in entry:
                    continue
                self._baselines[(q_group, win)] = _WindowBaseline(
                    remaining_fraction=float(entry["remaining_fraction"]),
                    reset_time=str(entry.get("reset_time") or ""),
                )

        raw_ratios = state.get("ratios")
        if isinstance(raw_ratios, Mapping):
            for q_group, entry in raw_ratios.items():
                if not isinstance(entry, Mapping):
                    continue
                self._ratios[str(q_group)] = _RatioAccumulator(
                    sum_delta_5h=max(0.0, float(entry.get("sum_delta_5h", 0.0))),
                    sum_delta_7d=max(0.0, float(entry.get("sum_delta_7d", 0.0))),
                    samples=max(0, int(entry.get("samples", 0))),
                )

        raw_models = state.get("last_models")
        if isinstance(raw_models, Mapping):
            for q_group, mod in raw_models.items():
                if isinstance(mod, str) and mod.strip():
                    self._last_model_by_group[str(q_group)] = mod.strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialize baselines, paired burn accumulators, and last pool models for SQLite."""
        return {
            "baselines": {
                f"{q_group}:{win}": {
                    "remaining_fraction": bl.remaining_fraction,
                    "reset_time": bl.reset_time,
                }
                for (q_group, win), bl in sorted(self._baselines.items())
            },
            "ratios": {
                q_group: {
                    "sum_delta_5h": round(acc.sum_delta_5h, 6),
                    "sum_delta_7d": round(acc.sum_delta_7d, 6),
                    "samples": acc.samples,
                }
                for q_group, acc in sorted(self._ratios.items())
            },
            "last_models": dict(sorted(self._last_model_by_group.items())),
        }

    def note_active_model(self, model_id: str | None) -> None:
        """Record the latest active `model_id` on its corresponding provider quota pool."""
        if not model_id or not model_id.strip():
            return
        cleaned = model_id.strip()
        self._last_model_by_group[classify_model_quota_group(cleaned)] = cleaned

    def get_ratios(self, *, default_ratio: float = DEFAULT_5H_PER_7D_RATIO) -> dict[str, float]:
        """Return empirical `5h`-per-`7d` capacity ratio `R` per quota pool."""
        fallback = max(0.1, float(default_ratio))
        resolved: dict[str, float] = {}
        for q_group in ("google_ai", "anthropic_ai", *self._ratios.keys()):
            acc = self._ratios.get(q_group)
            if acc is not None and acc.sum_delta_7d > _FRACTION_EPSILON:
                resolved[q_group] = round(acc.sum_delta_5h / acc.sum_delta_7d, 4)
            else:
                resolved[q_group] = round(fallback, 4)
        return resolved

    def _resolve_attributed_model(self, quota_group: str, active_model_id: str | None) -> str:
        if active_model_id and classify_model_quota_group(active_model_id) == quota_group:
            return active_model_id.strip()
        remembered = self._last_model_by_group.get(quota_group)
        if remembered:
            return remembered
        return "gemini-default" if quota_group == "google_ai" else "claude-default"

    def observe_snapshot(
        self,
        five_hour_buckets: Sequence[Mapping[str, Any]],
        seven_day_buckets: Sequence[Mapping[str, Any]],
        *,
        active_model_id: str | None = None,
        default_ratio: float = DEFAULT_5H_PER_7D_RATIO,
    ) -> QuotaBurnObservation:
        """Process a 5h/7d quota snapshot into gauge points, burn deltas, and updated `R`."""
        self.note_active_model(active_model_id)
        bucket_gauges: list[BucketGaugePoint] = []
        burnt_deltas: list[BucketBurnEvent] = []
        state_changed = False

        # Track per-group step behavior to update paired 5h/7d ratio cleanly
        step_info: dict[tuple[str, str], tuple[float, float, float, bool]] = {}

        for win_label, items in (("5h", five_hour_buckets), ("7d", seven_day_buckets)):
            default_cap = 1_000.0 if win_label == "7d" else 200.0
            for raw in items:
                raw_group = str(raw.get("group") or raw.get("name") or "Gemini Models")
                q_group = normalize_quota_group(raw_group)
                rem_frac = max(0.0, min(1.0, float(raw.get("remaining_fraction", 0.0))))
                used_frac = round(max(0.0, min(1.0, 1.0 - rem_frac)), 6)
                rem_pct = round(rem_frac * 100.0, 2)
                cap = float(raw.get("credit_capacity", default_cap))
                rem_credits = float(raw.get("remaining_credits", round(rem_frac * cap, 2)))
                reset_str = str(raw.get("reset_time") or "")
                reset_epoch = parse_reset_epoch_seconds(reset_str)

                bucket_gauges.append(
                    BucketGaugePoint(
                        quota_group=q_group,
                        group=raw_group,
                        window=win_label,
                        remaining_fraction=rem_frac,
                        used_fraction=used_frac,
                        remaining_percent=rem_pct,
                        remaining_credits=round(rem_credits, 2),
                        reset_timestamp_seconds=reset_epoch,
                    )
                )

                key = (q_group, win_label)
                prev = self._baselines.get(key)
                if prev is None:
                    self._baselines[key] = _WindowBaseline(
                        remaining_fraction=rem_frac,
                        reset_time=reset_str,
                    )
                    state_changed = True
                    continue

                prev_frac = prev.remaining_fraction
                prev_epoch = parse_reset_epoch_seconds(prev.reset_time)
                reset_advanced = (
                    prev_epoch is not None
                    and reset_epoch is not None
                    and reset_epoch > prev_epoch + _RESET_JUMP_TOLERANCE_SECONDS
                )
                did_reset = reset_advanced or (rem_frac > prev_frac + _FRACTION_EPSILON)

                delta_u = 0.0
                if did_reset:
                    self._baselines[key] = _WindowBaseline(
                        remaining_fraction=rem_frac,
                        reset_time=reset_str,
                    )
                    state_changed = True
                elif prev_frac > rem_frac + _FRACTION_EPSILON:
                    delta_u = round(prev_frac - rem_frac, 6)
                    self._baselines[key] = _WindowBaseline(
                        remaining_fraction=rem_frac,
                        reset_time=reset_str,
                    )
                    state_changed = True
                    model_attr = self._resolve_attributed_model(q_group, active_model_id)
                    burnt_deltas.append(
                        BucketBurnEvent(
                            quota_group=q_group,
                            window=win_label,
                            model_id=model_attr,
                            fraction_burnt=delta_u,
                            credits_burnt=round(delta_u * cap, 4),
                        )
                    )
                elif reset_str and reset_str != prev.reset_time:
                    self._baselines[key] = _WindowBaseline(
                        remaining_fraction=rem_frac,
                        reset_time=reset_str,
                    )
                    state_changed = True

                step_info[key] = (prev_frac, rem_frac, delta_u, did_reset)

        for q_group in ("google_ai", "anthropic_ai"):
            info_5h = step_info.get((q_group, "5h"))
            info_7d = step_info.get((q_group, "7d"))
            if info_5h is None or info_7d is None:
                continue
            prev_5h, new_5h, delta_5h, reset_5h = info_5h
            prev_7d, new_7d, delta_7d, reset_7d = info_7d
            unclamped = (
                0.0 < new_5h < 1.0
                and 0.0 < new_7d < 1.0
                and 0.0 < prev_5h <= 1.0
                and 0.0 < prev_7d <= 1.0
            )
            if (
                not reset_5h
                and not reset_7d
                and unclamped
                and delta_5h > _FRACTION_EPSILON
                and delta_7d > _FRACTION_EPSILON
            ):
                acc = self._ratios.setdefault(q_group, _RatioAccumulator())
                if acc.sum_delta_7d >= _ROLLING_SUM_CAP_7D:
                    acc.sum_delta_5h *= 0.5
                    acc.sum_delta_7d *= 0.5
                acc.sum_delta_5h += delta_5h
                acc.sum_delta_7d += delta_7d
                acc.samples += 1
                state_changed = True

        return QuotaBurnObservation(
            bucket_gauges=bucket_gauges,
            burnt_deltas=burnt_deltas,
            ratios_by_group=self.get_ratios(default_ratio=default_ratio),
            state_changed=state_changed,
        )
