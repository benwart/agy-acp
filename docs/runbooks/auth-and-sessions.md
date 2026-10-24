# Auth & Session Troubleshooting Runbook

## Symptom: KiroCrew Displays "Authentication Required / Not Logged In"

1. Verify current credential status from a terminal:
   ```bash
   agy-acp auth status --format json
   ```
2. Authenticate using `agy-acp auth login`:
   ```bash
   agy-acp auth login --api-key "$GEMINI_API_KEY"
   ```
3. Retry the prompt in KiroCrew. Because `agy-acp` checks the credential store generation before each turn (`REQ-AUTH-004`), you do **not** need to restart KiroCrew or the `agy-acp` process.

## Symptom: Resuming a Session Fails (`session/load`)

1. Verify that `AGY_ACP_SAVE_DIR` (or default `~/.local/share/agy-acp/sessions`) is writable and contains the trajectory folder for the target `sessionId`.
2. Check `stderr` logs in KiroCrew (`~/Library/Logs/kirocrew-desktop/`) for any trajectory corruption warnings.
