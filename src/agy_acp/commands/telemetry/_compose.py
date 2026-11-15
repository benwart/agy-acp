"""Shared Docker Compose discovery and localhost endpoint resolution for `agy-acp telemetry`."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from agy_acp.providers.telemetry import EndpointReachabilityProbe

_DEFAULT_PORTS: dict[str, tuple[str, int]] = {
    "otel_collector": ("AGY_ACP_OTEL_HTTP_PORT", 4318),
    "prometheus": ("AGY_ACP_PROMETHEUS_PORT", 9090),
    "tempo": ("AGY_ACP_TEMPO_PORT", 3200),
    "loki": ("AGY_ACP_LOKI_PORT", 3100),
    "grafana": ("AGY_ACP_GRAFANA_PORT", 3000),
}


def resolve_compose_file(compose_file: Path | None = None) -> Path:
    """Locate `docker-compose.yml` from `--compose-file`, `$AGY_ACP_COMPOSE_FILE`, cwd, or repo."""
    candidates: list[Path] = []
    if compose_file is not None:
        candidates.append(compose_file.expanduser())
    env_path = os.environ.get("AGY_ACP_COMPOSE_FILE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "docker-compose.yml")
    candidates.append(Path(__file__).resolve().parents[4] / "docker-compose.yml")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    msg = f"Unable to locate docker-compose.yml (searched: {[str(c) for c in candidates]})"
    raise FileNotFoundError(msg)


def resolve_stack_service_urls() -> dict[str, str]:
    """Return `127.0.0.1` HTTP base URLs for all 5 local observability stack services."""
    urls: dict[str, str] = {}
    for service_key, (env_var, default_port) in _DEFAULT_PORTS.items():
        raw_port = os.environ.get(env_var, "").strip()
        port = int(raw_port) if raw_port.isdigit() else default_port
        urls[service_key] = f"http://127.0.0.1:{port}"
    return urls


def probe_stack_services() -> dict[str, dict[str, object]]:
    """Probe TCP reachability of all 5 localhost observability services."""
    urls = resolve_stack_service_urls()
    return {
        name: {
            "url": url,
            "reachable": EndpointReachabilityProbe(url).is_reachable(),
        }
        for name, url in urls.items()
    }


def run_docker_compose(
    subargs: Sequence[str],
    *,
    compose_file: Path | None = None,
) -> int:
    """Invoke `docker compose -f <compose_file> ...` without shell interpolation."""
    try:
        resolved = resolve_compose_file(compose_file)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cmd = ["docker", "compose", "-f", str(resolved), *subargs]
    try:
        completed = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("Error: `docker` CLI not found on PATH.", file=sys.stderr)
        return 1
    return int(completed.returncode)
