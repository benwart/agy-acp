"""Serve Zensical documentation site locally."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter


def default(
    host: str = "127.0.0.1",
    port: int = 8000,
    config_file: Path = Path("zensical.toml"),
    open_browser: Annotated[bool, Parameter(name=["--open", "--open-browser"])] = False,
    strict: bool = False,
    dry_run: bool = False,
) -> int:
    """Serve the Zensical documentation site locally with live reload."""
    return 1
