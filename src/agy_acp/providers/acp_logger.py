"""Structured JSONL diagnostic logger for ACP stdio server traffic and CLI invocations."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agy_acp.providers.state_store import resolve_state_db_path

_RELEVANT_ENV_PREFIXES: tuple[str, ...] = (
    "KIRO",
    "GOOSE",
    "ACP",
    "AGY",
    "GEMINI",
    "GOOGLE",
)


def resolve_log_file_path(save_dir_or_db: Path | str | None = None) -> Path:
    """Resolve the diagnostic JSONL log path from `$AGY_ACP_LOG_FILE` or state directory."""
    env_log = os.environ.get("AGY_ACP_LOG_FILE")
    if env_log:
        return Path(env_log)

    db_path = resolve_state_db_path(save_dir_or_db)
    return db_path.with_name("acp.log")


def snapshot_relevant_env() -> dict[str, str]:
    """Capture environment variables matching Kiro/Goose/ACP/Antigravity prefixes."""
    captured: dict[str, str] = {}
    for key, val in sorted(os.environ.items()):
        upper = key.upper()
        if upper.startswith(_RELEVANT_ENV_PREFIXES):
            if any(secret in upper for secret in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
                captured[key] = "<redacted>"
            else:
                captured[key] = val
    return captured


def log_acp_event(
    event: str,
    *,
    save_dir_or_db: Path | str | None = None,
    **fields: Any,
) -> None:
    """Append a single JSONL diagnostic record to `acp.log` (and stderr if enabled)."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    try:
        serialized = json.dumps(record, ensure_ascii=False, default=str)
    except TypeError, ValueError:
        serialized = json.dumps(
            {
                "timestamp": record["timestamp"],
                "pid": record["pid"],
                "event": event,
                "raw": repr(fields),
            },
            ensure_ascii=False,
        )

    line_bytes = f"{serialized}\n".encode()
    log_path = resolve_log_file_path(save_dir_or_db)
    with contextlib.suppress(OSError):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line_bytes)
        finally:
            os.close(fd)

    if os.environ.get("AGY_ACP_LOG_STDERR") == "1":
        with contextlib.suppress(OSError):
            sys.stderr.write(f"[agy-acp] {serialized}\n")
            sys.stderr.flush()
