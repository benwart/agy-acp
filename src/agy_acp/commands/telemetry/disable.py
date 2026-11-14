"""Persistently disable OpenTelemetry export for agy-acp."""

from __future__ import annotations

from pathlib import Path

from agy_acp.providers.telemetry import (
    resolve_telemetry_config,
    save_persisted_telemetry_settings,
)


def default(
    *,
    disk_logs: bool | None = None,
    save_dir: Path | None = None,
) -> int:
    """Persistently disable OpenTelemetry export in `telemetry.json`."""
    updates: dict[str, object] = {"enabled": False}
    if disk_logs is not None:
        updates["disk_logs"] = disk_logs

    cfg_path = save_persisted_telemetry_settings(updates, save_dir_or_db=save_dir)
    resolved = resolve_telemetry_config(save_dir)
    print(
        f"Disabled OpenTelemetry (disk_logs={'on' if resolved.disk_logs else 'off'}, "
        f"saved to {cfg_path})"
    )
    return 0
