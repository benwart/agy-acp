# 5-Hour / 7-Day Quotas & Token Usage Requirements

Google Antigravity enforces rolling **5-hour (`5h`)** and **7-day (`weekly`)** quota windows per model group (e.g., *Gemini Models* and *Claude and GPT models*) alongside G1 credits and per-turn token usage (`UsageMetadata`). `agy-acp` exposes these quotas both via standalone CLI commands and in-band to KiroCrew.

---

## 1. CLI Quota & Usage Commands

### REQ-QUOTA-001: 5-Hour Rolling Quota Command (`agy-acp quota five-hour`)
- **Command**: `agy-acp quota five-hour [--group <model_group>] [--format json|text]`
- **Behavior**:
  - Queries the active Antigravity quota provider for all `window == "5h"` buckets.
  - Outputs remaining fraction (`0.0`–`1.0`), remaining percentage (`0`–`100%`), used percentage, ISO-8601 `reset_time`, and human-readable time until reset for each model group.
  - Expected JSON structure (`--format json`):
    ```json
    {
      "window": "5h",
      "buckets": [
        {
          "id": "gemini-5h",
          "group": "Gemini Models",
          "name": "Five Hour Limit Remaining",
          "window": "5h",
          "remaining_fraction": 0.95,
          "remaining_percent": 95.0,
          "used_percent": 5.0,
          "reset_time": "2026-10-24T07:52:04Z"
        }
      ]
    }
    ```

### REQ-QUOTA-002: 7-Day Rolling Quota Command (`agy-acp quota seven-day`)
- **Command**: `agy-acp quota seven-day [--group <model_group>] [--format json|text]`
- **Behavior**:
  - Queries the active Antigravity quota provider for all 7-day (`window == "weekly"` / `"7d"`) buckets.
  - Expected JSON structure (`--format json`):
    ```json
    {
      "window": "7d",
      "buckets": [
        {
          "id": "gemini-weekly",
          "group": "Gemini Models",
          "name": "Weekly Limit Remaining",
          "window": "7d",
          "remaining_fraction": 0.39,
          "remaining_percent": 39.0,
          "used_percent": 61.0,
          "reset_time": "2026-10-24T12:34:28Z"
        }
      ]
    }
    ```

### REQ-QUOTA-003: Credits & Unified Quota Status Commands (`agy-acp quota credits` & `agy-acp quota status`)
- **Commands**:
  - `agy-acp quota credits [--format json|text]`
  - `agy-acp quota status [--format json|text]`
- **Behavior**:
  - `agy-acp quota credits` reports `remaining_credits`, `upgrade_uri`, and cumulative local session token usage (`input_tokens`, `output_tokens`, `cached_read_tokens`, `thinking_tokens`, `total_tokens`).
  - `agy-acp quota status` returns a unified snapshot combining `five_hour` buckets, `seven_day` buckets, `credits`, and `token_usage`.

---

## 2. KiroCrew Quota & Usage Visibility (Stretch Goal)

### REQ-QUOTA-004: ACP Wire Usage Telemetry & In-Chat Quota Slash Commands
Because KiroCrew's built-in dashboard tracks context window usage, token counts, and credits via ACP notifications, and surfaces agent slash commands in the chat UI, `agy-acp` bridges Antigravity usage and 5h/7d quotas into KiroCrew through four mechanisms:

1. **Turn Response Token Usage (`session/prompt` result)**:
   - Maps `google.antigravity.UsageMetadata` (`prompt_token_count`, `candidates_token_count`, `cached_content_token_count`, `thoughts_token_count`) into `result.usage`:
     ```json
     {
       "stopReason": "end_turn",
       "usage": {
         "inputTokens": 1500,
         "outputTokens": 420,
         "cachedReadTokens": 1000,
         "cachedWriteTokens": 0
       }
     }
     ```
2. **Context Window & Cost Updates (`session/update` -> `usage_update` & `_kiro.dev/metadata`)**:
   - After each turn, emits `session/update` with `sessionUpdate: "usage_update"` (`used`, `size`, `cost`) AND `_kiro.dev/metadata` (`sessionId`, `contextUsagePercentage`, `contextWindowSize`, `credits`, `modelId`, plus `_meta.quota` containing the 5h and 7d remaining percentages).
3. **Advertised KiroCrew Slash Commands (`available_commands_update`)**:
   - On `session/new` and `session/load`, `agy-acp` emits `session/update` with `sessionUpdate: "available_commands_update"` advertising:
     - `/quota`: Display 5-hour and 7-day rolling quota remaining percentages and reset times.
     - `/usage`: Display session token usage, context window utilization, and 5h/7d quota summary.
     - `/credits`: Display remaining Antigravity G1 credits and token ledger.
   - When a user sends `/quota`, `/usage`, or `/credits` as a `session/prompt` in KiroCrew, `agy-acp` intercepts the command, queries the live 5h/7d quota & token store, streams a formatted Markdown summary table as an `agent_message_chunk`, and completes the turn with `stopReason: "end_turn"` without burning model tokens.
4. **Proactive Rate-Limit / Quota Exhaustion Notice (`_kiro.dev/error/rate_limit`)**:
   - When `response.stop_reason == StopReason.QUOTA_EXHAUSTED` or either the 5-hour or 7-day bucket reaches `0%` remaining, `agy-acp` emits a `_kiro.dev/error/rate_limit` notification with the exact bucket name and `reset_time` so KiroCrew displays a rate-limit banner in the UI.
