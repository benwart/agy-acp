# agy-acp

Google Antigravity SDK wrapper exposing an **Agent Client Protocol (ACP)** JSON-RPC 2.0 server and Cyclopts CLI for **KiroCrew** (`kiro_crew`).

## Tooling (`asdf` + `uv` + `pre-commit` + Python 3.15)

```bash
asdf install
uv sync --all-groups
pre-commit install
```

## CLI Structure

- `agy-acp serve` / `agy-acp acp serve`: Start ACP JSON-RPC 2.0 server over stdio.
- `agy-acp acp validate-gate`: Validate KiroCrew tool-gate policy and session marker.
- `agy-acp auth login | logout | status`: Out-of-band authentication with zero-downtime hot-reload in connected ACP servers.
- `agy-acp quota status | five-hour | seven-day | credits`: Inspect 5-hour and 7-day rolling quotas, G1 credits, and session token usage.
- `agy-acp docs build | serve`: Build and serve the Zensical documentation site (`zensical.toml` / `docs/`).
