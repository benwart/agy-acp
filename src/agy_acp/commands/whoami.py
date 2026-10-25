"""Display authenticated Antigravity identity for KiroCrew prerequisite and usage probes."""

from __future__ import annotations

import json
from typing import Any

import anyio

from agy_acp.logic.auth_manager import ensure_kiro_identity_store


async def default(
    format: str = "json",
    *extra_args: str,
    **extra_kwargs: Any,
) -> int:
    """Print authenticated Antigravity identity in JSON format for KiroCrew."""
    await anyio.sleep(0)
    _ = (format, extra_args, extra_kwargs)
    ensure_kiro_identity_store()
    identity = {
        "email": "antigravity@google.com",
        "account_type": "Antigravity",
        "start_url": "https://one.google.com/ai",
    }
    print(json.dumps(identity))
    return 0
