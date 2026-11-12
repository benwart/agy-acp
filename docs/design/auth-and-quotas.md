# Credential Hot-Reload, SQLite State & Quota Design

## Atomic Credential Store (`CredentialStoreProvider`)

- Stored by default at `~/.config/agy-acp/credentials.json` (overridable via `AGY_ACP_AUTH_FILE` or `--auth-file`).
- Written atomically using `tempfile.NamedTemporaryFile` in the same directory, `os.chmod(..., 0o600)`, `os.fsync`, and `os.replace`.
- Supports API keys, Vertex AI ADC (`--vertex --project ... --location ...`), and Antigravity OAuth (`OAuthProvider`).
- Schema:
  ```json
  {
    "authenticated": true,
    "method": "api_key",
    "identity": "gemini-api-key",
    "api_key": "...",
    "vertex": false,
    "project": null,
    "location": null,
    "generation": 1,
    "fingerprint": "sha256:...",
    "updated_at": "2026-10-23T20:00:00Z"
  }
  ```

## Synchronous SQLite WAL State Store (`SqliteStateStoreProvider`)

- Persists local harness metadata in `<save_dir>/agy_acp.db` using synchronous Python `sqlite3` in `WAL` journal mode (`synchronous=NORMAL`, `busy_timeout=5000`).
- Tables:
  - `sessions` & `session_turns`: Session descriptors, `agyConversationId` mapping, custom-agent state, compacted summaries, and turn history for `session/load`, `session/fork`, and `_kiro/checkpoint/revertMultiple`.
  - `models_cache`: Discovered `agy acp` model catalog with a **24-hour TTL** (`MODELS_CACHE_TTL_SECONDS = 86,400s`); expired caches automatically refresh via `AgyAcpClientProvider.discover_models(refresh=True)` and fall back to stale rows if offline.
  - `token_ledger` & `kv_state`: Cumulative token counters across sessions and auxiliary harness metadata.

## Zero-Downtime Session Continuation on Credential Change

1. Each active session records `authFingerprint` alongside its SQLite session metadata and mapped `agyConversationId`.
2. Before `SessionManager.prompt(session_id, ...)` dispatches a turn, `AuthManager` inspects the credential store on disk.
3. If `authenticated == False` (after an explicit `agy-acp auth logout`), `SessionManager` immediately raises an auth error (`"Authentication required: not logged in"`) while keeping the ACP server alive and preserving the SQLite session record.
4. When `agy-acp auth login` writes rotated credentials (`store.fingerprint != session["authFingerprint"]`), the next `session/prompt` updates the session fingerprint and resumes or re-opens the `agy acp` backend session seamlessly without restarting `agy-acp`.

## 5-Hour & 7-Day Quota Aggregation (`QuotaProvider` + `QuotaTracker`)

- `QuotaProvider` executes standalone `agy -p /usage --output-format json` and `agy -p /credits --output-format json` subprocesses (`run_agy_slash_json`) with scrubbed environment variables to query rolling `5h` and `7d` (`weekly`) buckets across model groups (*Gemini Models*, *Claude and GPT models*) and G1 credit balances.
- `QuotaTracker` combines upstream bucket fractions with cumulative session token counts stored in `SqliteStateStoreProvider` (`token_ledger`).
- Exposes both CLI formatters (`text` and `json`, plus `agy-acp chat /usage`) and ACP in-band handlers (`/quota`, `/usage`, `/credits` slash commands, `usage_update`, `_kiro.dev/metadata`, and `_kiro.dev/error/rate_limit`).
