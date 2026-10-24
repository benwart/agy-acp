"""Non-interactive KiroCrew CLI compatibility command (`--list-models` and `/usage`)."""

from __future__ import annotations

import json
from typing import Any

import anyio

from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.providers.agy_cli_provider import AgyCliProvider

DEFAULT_CONTEXT_WINDOW_TOKENS = 1_000_000


async def default(
    *prompts: str,
    list_models: bool = False,
    format: str = "text",
    no_interactive: bool = False,
    agent: str | None = None,
    model: str | None = None,
    **extra_kwargs: Any,
) -> int:
    """Execute KiroCrew `--list-models` catalog query or non-interactive `/usage` scrape."""
    await anyio.sleep(0)
    _ = (format, no_interactive, agent, model, extra_kwargs)

    if list_models:
        provider = AgyCliProvider()
        discovered = await provider.discover_models()
        catalog_rows = [
            {
                "model_id": m["modelId"],
                "model_name": m["modelId"],
                "display_name": m.get("name") or m["modelId"],
                "description": m.get("description") or m.get("name") or m["modelId"],
                "context_window_tokens": DEFAULT_CONTEXT_WINDOW_TOKENS,
            }
            for m in discovered
            if m.get("modelId")
        ]
        print(json.dumps({"models": catalog_rows}))
        return 0

    prompt_text = " ".join(prompts).strip()
    if prompt_text in {"/usage", "/quota", "/credits"} or not prompt_text:
        tracker = QuotaTracker()
        status = await tracker.get_unified_status()
        kiro_credits = status.get("credits", {}).get("kiro_credits", {})
        capacity = float(kiro_credits.get("credit_capacity", 200.0) or 200.0)
        remaining = float(kiro_credits.get("effective_credits", capacity))
        used_credits = round(max(0.0, capacity - remaining), 1)
        plan_credits = round(capacity, 1)
        raw_reset = str(kiro_credits.get("reset_time", "") or "")
        reset_date = raw_reset.split("T", 1)[0] if "T" in raw_reset else (raw_reset or "2026-10-24")
        md = await tracker.format_slash_command_response(prompt_text or "/usage")
        print(
            f"Estimated Usage | resets on {reset_date} | Antigravity\n"
            f"Credits used: 0.0\n"
            f"({used_credits} of {plan_credits} covered in plan)\n\n"
            f"{md}"
        )
        return 0

    provider = AgyCliProvider()
    outcome = await provider.stream_prompt(cwd=".", prompt=prompt_text, model=model)
    print(outcome.response)
    return 0
