"""ACP stdio server subcommand."""

from __future__ import annotations

from pathlib import Path

import anyio

from agy_acp.commands.serve import default as serve_default


async def default(
    profile: str = "acp",
    patch: Path | None = None,
    agent: str | None = None,
    model: str | None = None,
    with_builtin: str | None = None,
    agent_engine: str | None = None,
    auth_method: str | None = None,
    auth_file: Path | None = None,
    save_dir: Path | None = None,
) -> int:
    """Run the ACP JSON-RPC 2.0 stdio harness for KiroCrew."""
    await anyio.sleep(0)
    return await serve_default(
        profile=profile,
        patch=patch,
        agent=agent,
        model=model,
        with_builtin=with_builtin,
        agent_engine=agent_engine,
        auth_method=auth_method,
        auth_file=auth_file,
        save_dir=save_dir,
    )
