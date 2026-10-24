"""Clear stored Google Antigravity SDK credentials without restarting ACP servers."""

from __future__ import annotations

import json
from pathlib import Path

import anyio

from agy_acp.logic.auth_manager import AuthManager


async def default(
    auth_file: Path | None = None,
    format: str = "text",
) -> int:
    """Log out of Google Antigravity SDK and signal connected ACP servers without restart."""
    await anyio.sleep(0)
    mgr = AuthManager(auth_file=auth_file)
    state = mgr.logout()
    if format == "json":
        print(json.dumps(state.to_dict()))
    else:
        print(f"Logged out [generation={state.generation}]")
    return 0
