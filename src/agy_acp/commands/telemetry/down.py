"""Stop the local OpenTelemetry & Grafana Docker Compose stack."""

from __future__ import annotations

from pathlib import Path

from agy_acp.commands.telemetry._compose import run_docker_compose


def default(
    *,
    compose_file: Path | None = None,
    volumes: bool = False,
) -> int:
    """Stop and remove local telemetry stack containers (preserving volumes by default)."""
    args = ["down"]
    if volumes:
        args.append("--volumes")
    return run_docker_compose(args, compose_file=compose_file)
