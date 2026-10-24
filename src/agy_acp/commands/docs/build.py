"""Build static Zensical documentation site."""

from __future__ import annotations

from pathlib import Path

import anyio


async def default(
    site_dir: Path = Path("site"),
    config_file: Path = Path("zensical.toml"),
    clean: bool = False,
    strict: bool = False,
    dry_run: bool = False,
) -> int:
    """Build the static Zensical documentation site."""
    await anyio.sleep(0)
    if not config_file.is_file():
        return 1
    if dry_run:
        site_dir.mkdir(parents=True, exist_ok=True)
        return 0

    cmd = ["zensical", "build", "--config-file", str(config_file)]
    if clean:
        cmd.append("--clean")
    if strict:
        cmd.append("--strict")

    result = await anyio.run_process(cmd, check=False)
    return result.returncode
