# KiroCrew Integration Guide

## Configuring KiroCrew to Use `agy-acp`

KiroCrew resolves ACP backend binaries via environment variable overrides (`kiro_crew/acp/launch.py`). You can point any KiroCrew backend slot at `agy-acp`:

```bash
export OPENCODE_BIN="/Users/benwart/source/benwart/agy-acp/.venv/bin/agy-acp"
# or:
export CLAUDE_AGENT_ACP_BIN="/Users/benwart/source/benwart/agy-acp/.venv/bin/agy-acp"
```

## Authenticating Without Restarting KiroCrew

You can log in, rotate credentials, or log out from any terminal while KiroCrew's `agy-acp` subprocess remains connected:

```bash
# Check current auth status
agy-acp auth status

# Log in with a Gemini API key or Vertex AI ADC
agy-acp auth login --api-key "$GEMINI_API_KEY"
# or:
agy-acp auth login --vertex --project my-gcp-project --location global

# Log out (running ACP server stays alive and prompts for re-auth on next turn)
agy-acp auth logout
```

## Checking 5-Hour & 7-Day Quotas

From the terminal:

```bash
agy-acp quota status
agy-acp quota five-hour --format json
agy-acp quota seven-day --format json
agy-acp quota credits
```

From inside KiroCrew chat:
- Type `/quota`, `/usage`, or `/credits` to view a live Markdown table of your 5-hour and 7-day rolling limits and session token usage.
