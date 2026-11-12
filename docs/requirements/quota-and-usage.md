# 5-Hour / 7-Day Quotas & Token Usage Requirements

Google Antigravity enforces rolling **5-hour (`5h`)** and **7-day (`weekly`)** quota windows per model group (e.g., *Gemini Models* and *Claude and GPT models*) alongside G1 credits and per-turn token usage. `agy-acp` exposes these quotas both via standalone CLI commands (`QuotaProvider` invoking `agy -p /usage --output-format json` and `/credits`) and in-band to KiroCrew.

---

## 1. CLI Quota & Usage Commands

### REQ-QUOTA-001: 5-Hour Rolling Quota Command (`agy-acp quota five-hour`)
- **Command**: `agy-acp quota five-hour [--group <model_group>] [--format json|text]`
- **Behavior**:
  - Queries `QuotaProvider` for all `window == "5h"` buckets.
  - Outputs remaining fraction (`0.0`–`1.0`), remaining percentage (`0`–`100%`), used percentage, ISO-8601 `reset_time`, and human-readable time until reset for each model group.

### REQ-QUOTA-002: 7-Day Rolling Quota Command (`agy-acp quota seven-day`)
- **Command**: `agy-acp quota seven-day [--group <model_group>] [--format json|text]`
- **Behavior**:
  - Queries `QuotaProvider` for all 7-day (`window == "weekly"` / `"7d"`) buckets and reports remaining/used percentages and `reset_time`.

### REQ-QUOTA-003: Credits & Unified Quota Status Commands (`agy-acp quota credits` & `agy-acp quota status`)
- **Commands**:
  - `agy-acp quota credits [--format json|text]`
  - `agy-acp quota status [--format json|text]`
- **Behavior**:
  - `agy-acp quota credits` reports `remaining_credits`, `upgrade_uri`, and cumulative SQLite session token usage (`input_tokens`, `output_tokens`, `cached_read_tokens`, `thinking_tokens`, `total_tokens`).
  - `agy-acp quota status` returns a unified snapshot combining `five_hour` buckets, `seven_day` buckets, `credits`, and `token_usage`.

---

## 2. KiroCrew Quota & Usage Visibility

### REQ-QUOTA-004: ACP Wire Usage Telemetry & In-Chat Quota Slash Commands
1. **Turn Response Token Usage (`session/prompt` result)**:
   - Maps `agy acp` turn `usage` and `usage_update` counters into `result.usage` (`inputTokens`, `outputTokens`, `cachedReadTokens`, `cachedWriteTokens`) and accumulates totals into SQLite `token_ledger`.
2. **Context Window & Cost Updates (`session/update` -> `usage_update` & `_kiro.dev/metadata`)**:
   - After each turn, emits `session/update` with `sessionUpdate: "usage_update"` (`used`, `size`, `cost`) AND `_kiro.dev/metadata` (`sessionId`, `contextUsagePercentage`, `contextWindowSize`, `credits`, `modelId`, plus `_meta.quota` containing the 5h and 7d remaining percentages).
3. **Advertised KiroCrew Slash Commands (`available_commands_update`)**:
   - On `session/new` and `session/load`, `agy-acp` advertises `/quota`, `/usage`, `/credits`, and `/compact` (ignoring upstream `agy acp` `available_commands_update` frames so KiroCrew's command palette is preserved).
   - Intercepts `/quota`, `/usage`, and `/credits` locally without burning model tokens.
4. **Proactive Rate-Limit / Quota Exhaustion Notice (`_kiro.dev/error/rate_limit`)**:
   - Emits `_kiro.dev/error/rate_limit` with the exact bucket name and `reset_time` when upstream rate limits occur or either rolling bucket reaches `0%` remaining.
