# agy-acp

Full-duplex **Agent Client Protocol (ACP)** JSON-RPC 2.0 server and Cyclopts CLI wrapping `agy acp --stdio --process-per-session` for **KiroCrew** (`kiro_crew`).

## Architecture Highlights

- **Persistent `agy acp` Subprocess Bridge (`AgyAcpClientProvider`)**: Multiplexes sessions over a persistent `agy acp --stdio --process-per-session` child process group using `anyio` structured concurrency, automatic crash recovery, and clean POSIX process-group teardown.
- **Zero Global Config or Temp-File Mutation**: Injects per-session stdio/HTTP `mcpServers` (`env` and `headers`), system mandates (`_meta.antigravity.systemPrompt.append`), and `cwd`-contained inline base64 `image` blocks (`<= 10 MiB`, `O_NOFOLLOW | O_NONBLOCK`) directly in memory—never touching `~/.gemini/config/mcp_config.json` or staging temp files.
- **Real-Time Permission & Policy Gate**: Handles bidirectional `session/request_permission` (`allow_once` / `reject_once`) in flight without `SIGKILL`, blocks native `agy` subagent tools when KiroCrew `spawn_run` routing is active, and enforces `.kiro/agents` custom-agent `allowedTools` and path/command rules.
- **Dual Wire Engines (`v2` & `v3`) + Async Stdio Framing**: Supports both Kiro ACP (`--agent-engine v2`) and KAS (`--agent-engine v3`, `_kiro/*` extensions, `effortLevel`, `_session/steering`, checkpoints, and compaction) with off-loop `stdout` writes (`anyio.to_thread.run_sync`) and synchronous SQLite WAL state (`agy_acp.db`).

## Tooling (`asdf` + `uv` + `pre-commit` + Python 3.15)

```bash
asdf install
uv sync --all-groups
pre-commit install
```

## CLI Structure

- `agy-acp serve` / `agy-acp acp serve`: Start ACP JSON-RPC 2.0 server over stdio (`--agent-engine v2` default Kiro ACP wire; `--agent-engine v3` KAS wire).
- `agy-acp acp validate-gate | logs`: Validate KiroCrew tool-gate markers or inspect/tail/clear ACP diagnostic logs.
- `agy-acp chat`: Non-interactive model listing (`--list-models`), `/usage` inspection, and one-shot prompt execution over `AgyAcpClientProvider`.
- `agy-acp agent validate`: Validate `.kiro/agents` JSON and Markdown custom agent specifications.
- `agy-acp auth login | logout | status` & `agy-acp whoami | status`: Out-of-band authentication with zero-downtime hot-reload in connected ACP servers.
- `agy-acp quota status | five-hour | seven-day | credits`: Inspect 5-hour and 7-day rolling quotas, G1 credits, and SQLite session token usage.
- `agy-acp docs build | serve`: Build and serve the Zensical documentation site (`zensical.toml` / `docs/`).
