"""Authenticate with Google Antigravity SDK and persist hot-reloadable credentials."""

from __future__ import annotations

import json
from pathlib import Path

import anyio

from agy_acp.logic.auth_manager import AuthManager


async def default(
    api_key: str | None = None,
    vertex: bool = False,
    project: str | None = None,
    location: str | None = None,
    oauth: bool = False,
    auth_file: Path | None = None,
    format: str = "text",
) -> int:
    """Log in to Google Antigravity SDK and atomically update credentials for hot-reload."""
    await anyio.sleep(0)
    mgr = AuthManager(auth_file=auth_file)
    state = mgr.login(
        api_key=api_key,
        vertex=vertex,
        project=project,
        location=location,
        oauth=oauth,
    )
    if format == "json":
        print(json.dumps(state.to_dict()))
    else:
        print(
            f"Authenticated via {state.method} ({state.identity}) [generation={state.generation}]"
        )
    return 0
