"""Display current OpenTelemetry configuration and OTLP collector reachability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from agy_acp.commands.telemetry._compose import probe_stack_services, resolve_stack_service_urls
from agy_acp.providers.telemetry import EndpointReachabilityProbe, resolve_telemetry_config


def default(
    *,
    format: Literal["text", "json"] = "text",  # noqa: A002
    save_dir: Path | None = None,
) -> int:
    """Show active OpenTelemetry status, OTLP endpoint reachability, and disk log state."""
    cfg = resolve_telemetry_config(save_dir)
    reachable = EndpointReachabilityProbe(cfg.endpoint).is_reachable()
    services = probe_stack_services()
    urls = resolve_stack_service_urls()
    payload = {
        "enabled": cfg.enabled,
        "endpoint": cfg.endpoint,
        "service_name": cfg.service_name,
        "disk_logs": cfg.disk_logs,
        "collector_reachable": reachable,
        "grafana_url": urls["grafana"],
        "services": services,
        "metric_export_interval_ms": cfg.metric_export_interval_ms,
        "log_retention_seconds": cfg.log_retention_seconds,
        "config_path": str(cfg.config_path),
    }
    if format == "json":
        print(json.dumps(payload, indent=2))
        return 0

    collector_label = "reachable" if reachable else "offline (auto-buffering until up)"
    grafana_up = bool(services["grafana"]["reachable"])
    grafana_label = "reachable" if grafana_up else "offline"
    print(f"Telemetry:   {'enabled' if cfg.enabled else 'disabled'}")
    print(f"Endpoint:    {cfg.endpoint} ({collector_label})")
    print(f"Grafana:     {urls['grafana']} ({grafana_label})")
    print(f"Service:     {cfg.service_name}")
    print(f"Disk logs:   {'enabled' if cfg.disk_logs else 'disabled'}")
    print(f"Log buffer:  {cfg.log_retention_seconds / 60:.0f}m rolling window")
    print(f"Config file: {cfg.config_path}")
    return 0
