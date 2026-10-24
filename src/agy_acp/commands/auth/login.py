"""Authenticate with Google Antigravity SDK and persist hot-reloadable credentials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio

from agy_acp.logic.auth_manager import AuthManager


async def default(
    api_key: str | None = None,
    vertex: bool = False,
    project: str | None = None,
    location: str | None = None,
    oauth: bool = False,
    no_browser: bool = False,
    timeout: float = 180.0,
    auth_file: Path | None = None,
    format: str = "text",
) -> int:
    """Log in to Google Antigravity SDK via browser OAuth, API key, or Vertex ADC."""
    await anyio.sleep(0)
    mgr = AuthManager(auth_file=auth_file)
    use_oauth_browser = oauth or (api_key is None and not vertex)
    if use_oauth_browser:
        state = await mgr.login_oauth_browser(
            open_browser=not no_browser,
            timeout_seconds=timeout,
            on_auth_url=lambda url: print(
                "Opening browser for Google Antigravity OAuth login...\n"
                f"If your browser does not open automatically, visit:\n  {url}",
                file=sys.stderr,
                flush=True,
            ),
        )
    else:
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
