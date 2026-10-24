"""Inspect combined 5-hour quota, 7-day quota, credits, and token usage with Rich progress bars."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import anyio
from rich.console import Console
from rich.progress_bar import ProgressBar
from rich.text import Text

from agy_acp.logic.quota_tracker import QuotaTracker

LOW_QUOTA_THRESHOLD_PERCENT: float = 5.0
BAR_WIDTH: int = 28
TREND_FIXED_WIDTH: int = 13

_WINDOW_DURATION_SECONDS: dict[str, float] = {
    "7d": 7.0 * 24.0 * 3600.0,
    "weekly": 7.0 * 24.0 * 3600.0,
    "5h": 5.0 * 3600.0,
}


def resolve_quota_bar_style(group_name: str, remaining_percent: float) -> str:
    """Return the Rich bar color style for a quota row based on model group and 5% threshold."""
    is_gemini = "gemini" in group_name.lower()
    is_below_limit = remaining_percent < LOW_QUOTA_THRESHOLD_PERCENT
    if is_gemini:
        return "red" if is_below_limit else "blue"
    return "orange1" if is_below_limit else "green"


def format_reset_time(reset_time: str, *, now: datetime | None = None) -> str:
    """Convert an ISO-8601 reset timestamp into a human-readable relative countdown."""
    if not reset_time:
        return "unknown"
    try:
        reset_dt = datetime.fromisoformat(reset_time.replace("Z", "+00:00"))
    except ValueError:
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


def format_over_under_trend(
    *,
    remaining_percent: float,
    window: str,
    reset_time: str,
    now: datetime | None = None,
) -> tuple[str, str]:
    """Compute fixed-width over/under trend comparing remaining quota % to remaining time %."""
    active_now = now or datetime.now(UTC)
    duration = _WINDOW_DURATION_SECONDS.get(window.lower(), 5.0 * 3600.0)
    time_remaining_pct = 0.0
    if reset_time:
        try:
            reset_dt = datetime.fromisoformat(reset_time.replace("Z", "+00:00"))
            remaining_sec = max(0.0, min(duration, (reset_dt - active_now).total_seconds()))
            time_remaining_pct = (remaining_sec / duration) * 100.0
        except ValueError:
            time_remaining_pct = 0.0

    delta = round(remaining_percent - time_remaining_pct, 1)
    if delta > 0.05:
        direction = "OVER"
        style = "green"
    elif delta < -0.05:
        direction = "UNDER"
        style = "red" if remaining_percent < LOW_QUOTA_THRESHOLD_PERCENT else "yellow"
    else:
        direction = "EVEN"
        delta = 0.0
        style = "dim"

    formatted = f"{direction:<5} {delta:+6.1f}%"
    return formatted, style


def build_quota_status_renderables(
    data: dict[str, Any],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Order quota buckets by group (Gemini -> Claude/GPT) and window (7d -> 5h) with bars."""
    seven_day_buckets = data.get("seven_day", {}).get("buckets", [])
    five_hour_buckets = data.get("five_hour", {}).get("buckets", [])
    all_buckets: list[dict[str, Any]] = [
        *(b for b in seven_day_buckets if isinstance(b, dict)),
        *(b for b in five_hour_buckets if isinstance(b, dict)),
    ]

    def _sort_key(bucket: dict[str, Any]) -> tuple[int, str, int]:
        group_name = str(bucket.get("group", ""))
        group_rank = 0 if "gemini" in group_name.lower() else 1
        win = str(bucket.get("window", "")).lower()
        window_rank = 0 if win in {"7d", "weekly"} else 1
        return (group_rank, group_name.lower(), window_rank)

    ordered = sorted(all_buckets, key=_sort_key)
    rows: list[dict[str, Any]] = []
    for bucket in ordered:
        group_name = str(bucket.get("group", ""))
        remaining_pct = float(bucket.get("remaining_percent", 0.0))
        window = str(bucket.get("window", "5h"))
        reset_time = str(bucket.get("reset_time", ""))
        bar_style = resolve_quota_bar_style(group_name, remaining_pct)
        bar = ProgressBar(
            total=100.0,
            completed=max(0.0, min(100.0, remaining_pct)),
            width=BAR_WIDTH,
            style="bright_black",
            complete_style=bar_style,
            finished_style=bar_style,
        )
        trend_text, trend_style = format_over_under_trend(
            remaining_percent=remaining_pct,
            window=window,
            reset_time=reset_time,
            now=now,
        )
        rows.append(
            {
                "id": str(bucket.get("id", "")),
                "group": group_name,
                "name": str(bucket.get("name", "")),
                "window": window,
                "remaining_percent": remaining_pct,
                "reset_time": reset_time,
                "reset_display": format_reset_time(reset_time, now=now),
                "bar": bar,
                "bar_style": bar_style,
                "trend_text": trend_text,
                "trend_style": trend_style,
            }
        )
    return rows


def render_rich_quota_status(
    data: dict[str, Any],
    *,
    console: Console | None = None,
    now: datetime | None = None,
) -> None:
    """Render grouped thin colored progress bars, fixed-width trend, and quota labels via Rich."""
    active_console = console or Console()
    rows = build_quota_status_renderables(data, now=now)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)

    for idx, (group_name, group_rows) in enumerate(grouped.items()):
        if idx > 0:
            active_console.print()
        header_color = "blue" if "gemini" in group_name.lower() else "green"
        active_console.print(Text(group_name, style=f"bold {header_color}"))

        for row in group_rows:
            bar_segments = list(
                active_console.render(
                    row["bar"],
                    active_console.options.update_width(BAR_WIDTH),
                )
            )
            line = Text()
            bar_chars = 0
            for seg in bar_segments:
                if seg.text != "\n":
                    line.append(seg.text, style=seg.style)
                    bar_chars += len(seg.text)
            if bar_chars < BAR_WIDTH:
                line.append("─" * (BAR_WIDTH - bar_chars), style="bright_black")
            line.append(f"  {row['remaining_percent']:5.1f}%  ", style=f"bold {row['bar_style']}")
            line.append(f"{row['trend_text']}  ", style=row["trend_style"])
            line.append(f"{row['name']} ({row['window']})", style="bold")
            line.append(f"  resets {row['reset_display']}", style="dim")
            active_console.print(line, overflow="ignore", crop=False, soft_wrap=True)

    credits = data.get("credits", {})
    tokens = data.get("token_usage", {})
    kiro_credits = credits.get("kiro_credits")
    if not isinstance(kiro_credits, dict):
        kiro_credits = QuotaTracker.calculate_effective_kiro_credits(
            data.get("five_hour", {}).get("buckets", []),
            data.get("seven_day", {}).get("buckets", []),
            g1_overage_credits=int(credits.get("remaining_credits", 0)),
        )
    bottleneck_reset = format_reset_time(str(kiro_credits.get("reset_time", "")), now=now)
    active_console.print()
    active_console.print(
        Text.assemble(
            ("Effective Kiro Credits: ", "bold"),
            (
                f"{kiro_credits['effective_credits']:.1f} / {kiro_credits['credit_capacity']:.0f}",
                "cyan",
            ),
            (
                f" ({kiro_credits['bottleneck_window']} bottleneck, resets {bottleneck_reset})",
                "dim",
            ),
            (
                f" | G1 Overage: {credits.get('remaining_credits', 0)}"
                f" | Total Tokens: {tokens.get('total_tokens', 0)}",
                "dim",
            ),
        )
    )


async def default(
    group: str | None = None,
    format: str = "text",
) -> int:
    """Display unified 5-hour quota, 7-day quota, G1 credit balance, and token usage."""
    await anyio.sleep(0)
    data = await QuotaTracker().get_unified_status(group=group)
    if format == "json":
        print(json.dumps(data))
    else:
        render_rich_quota_status(data)
    return 0
