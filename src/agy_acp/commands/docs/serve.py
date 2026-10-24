"""Serve Zensical documentation site locally."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import anyio
from cyclopts import Parameter


async def default(
    host: str = "127.0.0.1",
    port: int = 8000,
    config_file: Path = Path("zensical.toml"),
    open_browser: Annotated[bool, Parameter(name=["--open", "--open-browser"])] = False,
    strict: bool = False,
    dry_run: bool = False,
) -> int:
    """Serve the Zensical documentation site locally with live reload."""
    await anyio.sleep(0)
    if not config_file.is_file():
        return 1
    if dry_run:
        return 0

    cmd = [
        sys.executable,
        "-m",
        "zensical",
        "serve",
        "--config-file",
        str(config_file),
        "--dev-addr",
        f"{host}:{port}",
    ]
    if open_browser:
        cmd.append("--open")
    if strict:
        cmd.append("--strict")

    result = await anyio.run_process(cmd, check=False)
    return result.returncode
