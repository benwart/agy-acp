"""Inspect or configure OpenTelemetry traces, metrics, and structured logs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agy_acp.commands.telemetry.status import default as status_default


def default(
    *,
    format: Literal["text", "json"] = "text",  # noqa: A002
    save_dir: Path | None = None,
) -> int:
    """Show active OpenTelemetry status, OTLP endpoint reachability, and disk log state."""
    return status_default(format=format, save_dir=save_dir)
