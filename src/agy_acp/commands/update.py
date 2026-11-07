"""Compatibility `kiro-cli update` subcommand that invalidates mtime-keyed CLI caches."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import anyio

from agy_acp import KIRO_CLI_COMPAT_VERSION, __version__


def _touch_executable_paths() -> None:
    """Refresh mtimes on invoked and PATH-resolved wrappers so KiroCrew re-probes `--version`."""
    candidates: list[Path] = []
    if sys.argv and sys.argv[0]:
        candidates.append(Path(sys.argv[0]))
    for bin_name in ("agy-acp", "kiro-cli"):
        found = shutil.which(bin_name)
        if found:
            candidates.append(Path(found))

    seen: set[Path] = set()
    for candidate in candidates:
        for target in (candidate, candidate.resolve(strict=False)):
            if target in seen:
                continue
            seen.add(target)
            with contextlib.suppress(OSError):
                if target.is_file():
                    os.utime(target, None)


async def default(*extra_args: str, **extra_kwargs: Any) -> int:
    """Verify the installed agy-acp shim and refresh wrapper mtimes for KiroCrew caches."""
    del extra_args, extra_kwargs
    await anyio.sleep(0)
    _touch_executable_paths()
    print(f"agy-acp {__version__} (kiro-cli {KIRO_CLI_COMPAT_VERSION}) is up to date.")
    return 0
