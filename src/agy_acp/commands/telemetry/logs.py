"""Inspect or tail container logs from the local OpenTelemetry Docker Compose stack."""

from __future__ import annotations

from pathlib import Path

from agy_acp.commands.telemetry._compose import run_docker_compose


def default(
    *,
    service: str | None = None,
    tail: int = 100,
    follow: bool = False,
    compose_file: Path | None = None,
) -> int:
    """Print or follow Docker Compose logs for all or one telemetry stack service."""
    args = ["logs", "--tail", str(max(1, tail))]
    if follow:
        args.append("--follow")
    if service:
        args.append(service)
    return run_docker_compose(args, compose_file=compose_file)
