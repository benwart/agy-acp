"""Persistently enable OpenTelemetry export for all agy-acp CLI and ACP subprocess runs."""

from __future__ import annotations

from pathlib import Path

from agy_acp.providers.telemetry import (
    DEFAULT_OTLP_ENDPOINT,
    resolve_telemetry_config,
    save_persisted_telemetry_settings,
)


def default(
    *,
    endpoint: str | None = None,
    service_name: str | None = None,
    disk_logs: bool | None = None,
    save_dir: Path | None = None,
) -> int:
    """Persistently enable OpenTelemetry export across all future `agy-acp` invocations."""
    updates: dict[str, object] = {"enabled": True}
    if endpoint is not None:
        updates["endpoint"] = endpoint
    elif not resolve_telemetry_config(save_dir).config_path.exists():
        updates["endpoint"] = DEFAULT_OTLP_ENDPOINT
    if service_name is not None:
        updates["service_name"] = service_name
    if disk_logs is not None:
        updates["disk_logs"] = disk_logs

    cfg_path = save_persisted_telemetry_settings(updates, save_dir_or_db=save_dir)
    resolved = resolve_telemetry_config(save_dir)
    print(
        f"Enabled OpenTelemetry -> {resolved.endpoint} "
        f"(disk_logs={'on' if resolved.disk_logs else 'off'}, saved to {cfg_path})"
    )
    return 0
