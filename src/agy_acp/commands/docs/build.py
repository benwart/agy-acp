"""Build static Zensical documentation site."""

from __future__ import annotations

from pathlib import Path


def default(
    site_dir: Path = Path("site"),
    config_file: Path = Path("zensical.toml"),
    clean: bool = False,
    strict: bool = False,
    dry_run: bool = False,
) -> int:
    """Build the static Zensical documentation site."""
    return 1
