# Credential Hot-Reload & Quota Design

## Atomic Credential Store (`CredentialStoreProvider`)

- Stored by default at `~/.config/agy-acp/credentials.json` (overridable via `AGY_ACP_AUTH_FILE` or `--auth-file`).
- Written atomically using `tempfile.NamedTemporaryFile` in the same directory, `os.chmod(..., 0o600)`, `os.fsync`, and `os.replace`.
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

## Zero-Downtime Session Rebinding on Credential Change

1. Each active ACP session maintains `active_auth_fingerprint` and stores its conversation state in `<save_dir>/<sessionId>`.
2. Before `SessionManager.prompt(session_id, ...)` dispatches a turn, `AuthManager.ensure_current()` reads the credential store metadata.
3. If `authenticated == False`, `SessionManager` immediately raises an auth error (`"Authentication required: not logged in"`) without destroying the session record.
4. If `authenticated == True` and `store.fingerprint != session.active_auth_fingerprint`, `SessionManager` closes the stale `Agent` context and opens a new `Agent` context bound to the same `conversation_id=session_id` with `SessionContinuationMode.RESUME` and the updated credentials.

## 5-Hour & 7-Day Quota Aggregation (`QuotaTracker`)

- Queries `QuotaProvider` for rolling `5h` and `7d` (`weekly`) buckets across model groups (*Gemini Models*, *Claude and GPT models*) and G1 credit balances.
- Combines upstream bucket fractions with cumulative session token counts recorded from `google.antigravity.UsageMetadata`.
- Exposes both CLI formatters (`text` and `json`) and ACP in-band handlers (`/quota`, `/usage`, `/credits` slash commands, `usage_update`, `_kiro.dev/metadata`, and `_kiro.dev/error/rate_limit`).
