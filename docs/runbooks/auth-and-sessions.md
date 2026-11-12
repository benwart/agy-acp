# Auth & Session Troubleshooting Runbook

## Symptom: KiroCrew Displays "Authentication Required / Not Logged In"

1. Verify current credential status from a terminal:
   ```bash
   agy-acp auth status --format json
   ```
2. Authenticate using `agy-acp auth login`:
   ```bash
   agy-acp auth login --api-key "$GEMINI_API_KEY"
   # or OAuth:
   agy-acp auth login --oauth
   ```
3. Retry the prompt in KiroCrew. Because `agy-acp` checks the credential store generation before each turn (`REQ-AUTH-004`), you do **not** need to restart KiroCrew or the `agy-acp` process.

## Symptom: Resuming a Session Fails (`session/load`) or Backend Subprocess Disconnects

1. Verify that `AGY_ACP_SAVE_DIR` (or default `~/.local/share/agy-acp/sessions`) is writable and contains `agy_acp.db` (SQLite WAL state store).
2. Inspect structured ACP diagnostic events and `agy acp` subprocess lifecycle logs:
   ```bash
   agy-acp acp logs --path
   agy-acp acp logs --tail 50
   ```
3. Verify that `agy` (`AGY_BIN` or `agy` on `PATH`) supports `agy acp --help`. Note that `AgyAcpClientProvider` automatically restarts crashed `agy acp` subprocesses on the next RPC and falls back to `[Compacted Session Summary]` when an upstream `agy` conversation ID is no longer on disk.
