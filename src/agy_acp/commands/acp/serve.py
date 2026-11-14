"""ACP stdio server subcommand."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import anyio
from cyclopts import Parameter

from agy_acp.commands.serve import AGENT_ENGINE_HELP, normalize_agent_engine, normalize_effort
from agy_acp.commands.serve import default as serve_default


async def default(
    profile: str = "acp",
    patch: Path | None = None,
    agent: str | None = None,
    model: str | None = None,
    with_builtin: str | None = None,
    agent_engine: Annotated[str | None, Parameter(help=AGENT_ENGINE_HELP)] = None,
    auth_method: Annotated[
        str | None,
        Parameter(name=["--auth-method", "--authMethod"]),
    ] = None,
    auth_file: Path | None = None,
    save_dir: Path | None = None,
    stdio: bool = False,
    experimental_acp: bool = False,
    verbose: Annotated[bool, Parameter(name=["--verbose", "-v"], negative=())] = False,
    debug: bool = False,
    quiet: bool = False,
    effort: str | None = None,
    trust_all_tools: Annotated[
        bool,
        Parameter(name=["--trust-all-tools", "-a"], negative=()),
    ] = False,
    trust_tools: str | None = None,
    login: bool = False,
    *extra_args: str,
    **extra_kwargs: Any,
) -> int:
    """Run the ACP JSON-RPC 2.0 stdio harness for KiroCrew."""
    try:
        normalize_agent_engine(agent_engine)
        normalize_effort(effort)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

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
        effort,
        trust_all_tools,
        trust_tools,
        login,
        *extra_args,
        **extra_kwargs,
    )
