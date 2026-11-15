"""Start the local 5-service OpenTelemetry & Grafana Docker Compose stack."""

from __future__ import annotations

from pathlib import Path

from agy_acp.commands.telemetry._compose import resolve_stack_service_urls, run_docker_compose


def default(
    *,
    compose_file: Path | None = None,
    wait: bool = False,
) -> int:
    """Start local OpenTelemetry Collector, Prometheus, Tempo, Loki, and Grafana containers."""
    args = ["up", "-d"]
    if wait:
        args.append("--wait")
    rc = run_docker_compose(args, compose_file=compose_file)
    if rc == 0:
        urls = resolve_stack_service_urls()
        print("Started agy-acp local telemetry stack on 127.0.0.1:")
        print(f"  Grafana:        {urls['grafana']}")
        print(f"  OTLP Collector: {urls['otel_collector']}")
        print(f"  Prometheus:     {urls['prometheus']}")
    return rc
