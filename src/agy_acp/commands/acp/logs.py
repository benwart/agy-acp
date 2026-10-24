"""Inspect or clear ACP server diagnostic JSONL logs."""

from __future__ import annotations

from pathlib import Path

from agy_acp.providers.acp_logger import resolve_log_file_path


def default(
    *,
    tail: int = 50,
    path: bool = False,
    clear: bool = False,
    save_dir: Path | None = None,
) -> int:
    """Display, locate, or clear the ACP server JSONL diagnostic log (`acp.log`)."""
    log_path = resolve_log_file_path(save_dir)
    if path:
        print(str(log_path))
        return 0

    if clear:
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")
        print(f"Cleared {log_path}")
        return 0

    if not log_path.exists():
        print(f"No log file found at {log_path}")
        return 0

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[-tail:] if tail > 0 else lines
    for line in selected:
        print(line)
    return 0
