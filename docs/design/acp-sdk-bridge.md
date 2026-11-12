# ACP Protocol & `agy acp` Subprocess Bridge Design

## Persistent Full-Duplex Process Model

`AgyAcpClientProvider` (`src/agy_acp/providers/agy_acp_client.py`) manages a long-lived `agy acp --stdio --process-per-session` child process rather than spawning per-turn CLI processes:

- **Capability Negotiation (`initialize`)**: Sends `{"protocolVersion": 1, "clientCapabilities": {}}` so `agy acp` executes its own built-in filesystem and shell tools internally instead of delegating `fs/*` or `terminal/*` RPCs back to KiroCrew. Never passes `--dangerously-skip-permissions`.
- **Non-Blocking Full-Duplex Routing (`_AcpSubprocessConnection`)**: `_stdout_reader` resolves pending client RPCs immediately, dispatches inbound `session/request_permission` server requests onto concurrent `anyio` tasks, and forwards `session/update` notifications through a memory object stream so `session/cancel` and permission responses can never deadlock behind a blocked reader.
- **Zero Global Config or Temp-File Side Effects**:
  - Per-session stdio (`command`, `args`, `env: [{name, value}]`) and HTTP (`type: "http"`, `url`, `headers: [{name, value}]`) MCP servers are passed directly in `session/new` and `session/load`—never mutating `~/.gemini/config/mcp_config.json` or leaking `KIROCREW_STUB_SESSION_TOKEN` across sessions.
  - System prompts, custom-agent mandates, and KiroCrew `spawn_run` routing instructions are passed via `_meta.antigravity.systemPrompt.append`.
  - Markdown local image references contained within `cwd` are opened atomically (`O_RDONLY | O_NOFOLLOW | O_NONBLOCK`, `S_ISREG`, `<= 10 MiB`) and sent inline as base64 ACP `image` content blocks without staging temporary files.
- **Process-Group Lifecycle & Crash Recovery**: Spawns `agy acp` in a dedicated POSIX session (`start_new_session=True`) with scrubbed subprocess environment variables (`PER_SESSION_MCP_ENV_KEYS`, `KIROCREW_ROUTE_SPAWN_RUN`, `PYTHONINSPECT`, `PYTHONSTARTUP`), reaps the entire process group (`SIGTERM` -> `SIGKILL` via `os.killpg`) inside `anyio.CancelScope(shield=True)` on shutdown, and transparently reconnects if the worker crashes between turns.

## Mapping KiroCrew ACP Concepts to `agy acp`

| KiroCrew ACP (`v2` / `v3` KAS) | `agy acp` JSON-RPC 2.0 Construct |
|---|---|
| `session/new` (`cwd`, `mcpServers`, `_meta`) | `session/new` (`cwd`, normalized `mcpServers`, `_meta.antigravity.systemPrompt.append`) + initial `session/set_model` & `session/set_config_option` |
| `session/load` (`sessionId`, `cwd`, `mcpServers`) | `session/load` (`sessionId=agyConversationId`, `cwd`, `mcpServers`, `_meta`); falls back to fresh `session/new` + `[Compacted Session Summary]` if upstream trajectory is absent |
| `session/fork` / `_kiro/checkpoint/revertMultiple` / `_kiro/session/compact` | Persists forked/reverted/compacted turn summary in SQLite `agy_acp.db`, closes stale backend session via `session/close`, and seeds next turn's `session/new` while preserving `_meta.antigravity.systemPrompt.append` |
| `session/set_mode` (`"read-only"`, `"approve"`, `"auto"`, or custom agent) | `session/set_config_option` (`configId="mode"`, `value="ask"` or `"default"`) + custom-agent policy enforcement |
| `session/set_model` / `configId="model"` | `session/set_model` (`sessionId`, `modelId`) |
| `configId="reasoning_effort"` / `"effortLevel"` | `session/set_config_option` (`configId="thought_level"`, `value="low" \| "medium" \| "high"`, snapped to supported options) |
| `session/prompt` (`text`, `image`, `resource`) | `session/prompt` (`prompt` content blocks with inline base64 `image` blocks and embedded context) |
| `session/request_permission` from `agy acp` | Re-evaluates populated `rawInput` against native-subagent block (`invoke_subagent` / `define_subagent` / `manage_subagents`), custom-agent `allowedTools`/globs, and KiroCrew's upstream `session/request_permission`, replying fail-closed with `allow_once` or `reject_once` |
| `session/update` (`available_commands_update`) | Ignored from upstream `agy acp` so `agy-acp`'s advertised KiroCrew slash commands (`/quota`, `/usage`, `/credits`, `/compact`) are never overwritten |
| `session/cancel` & `_session/steer` / `_session/steering` | Sends `session/cancel` notification to `agy acp`, drains interrupted leg, emits `steering_queued` / `steering_applied` / `steering_cleared`, replays steered prompt blocks, and persists steered user text in `session_turns` |
| `session/close` / `_kiro.dev/session/terminate` / `_kiro/session/delete` | Sends `session/close` to `agy acp --process-per-session` and updates/purges session state in `SqliteStateStoreProvider` |
