"""ACP stdio server subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    stdio: bool = False,
    experimental_acp: bool = False,
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    *extra_args: str,
    **extra_kwargs: Any,
) -> int:
    """Run the ACP JSON-RPC 2.0 stdio harness for KiroCrew."""
    await anyio.sleep(0)
    return await serve_default(
        profile,
        patch,
        agent,
        model,
        with_builtin,
        agent_engine,
        auth_method,
        auth_file,
        save_dir,
        stdio,
        experimental_acp,
        verbose,
        debug,
        quiet,
        *extra_args,
        **extra_kwargs,
    )
