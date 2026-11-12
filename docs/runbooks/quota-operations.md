# Quota & Rate Limit Operations Runbook

## Inspecting 5-Hour (`5h`) and 7-Day (`7d`) Rolling Quotas

Run the quota inspection commands (`QuotaProvider` querying `agy -p /usage --output-format json` and `/credits`) to see remaining percentages and exact UTC/local reset timestamps:

```bash
agy-acp quota status --format json
agy-acp quota five-hour
agy-acp quota seven-day
agy-acp chat --no-interactive /usage
```

## Handling Quota Exhaustion (`_kiro.dev/error/rate_limit`)

1. Check which bucket (`gemini-5h`, `gemini-weekly`, `3p-5h`, or `3p-weekly`) is at `0%` remaining using `agy-acp quota status`.
2. Switch the active model group in KiroCrew (via the model selector or `session/set_model`) or wait until the bucket's `reset_time`.
3. Check available G1 fallback credits and SQLite `token_ledger` totals via `agy-acp quota credits`.
