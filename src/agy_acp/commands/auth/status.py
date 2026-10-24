"""Inspect current Google Antigravity SDK authentication status and fingerprint."""

from __future__ import annotations

import json
from pathlib import Path

import anyio

from agy_acp.logic.auth_manager import AuthManager


async def default(
    auth_file: Path | None = None,
    format: str = "text",
) -> int:
    """Show current authentication status, active method, and credential fingerprint."""
    await anyio.sleep(0)
    mgr = AuthManager(auth_file=auth_file)
    state = mgr.status()
    if format == "json":
        print(json.dumps(state.to_dict()))
    elif state.authenticated:
        print(
            f"Authenticated via {state.method} ({state.identity}) [fingerprint={state.fingerprint}]"
        )
    else:
        print("Not authenticated")
    return 0 if state.authenticated else 1
