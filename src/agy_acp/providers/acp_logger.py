"""Structured JSONL diagnostic logger for ACP stdio server traffic and CLI invocations."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agy_acp.providers.state_store import resolve_state_db_path

REDACTED_PLACEHOLDER = "[REDACTED]"

_RELEVANT_ENV_PREFIXES: tuple[str, ...] = (
    "KIRO",
    "GOOSE",
    "ACP",
    "AGY",
    "GEMINI",
    "GOOGLE",
)

_SENSITIVE_CONTAINERS: frozenset[str] = frozenset({"env", "headers"})

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(api[-_]?key|secret|token|password|passwd|authorization|"
    r"auth[-_]?header|bearer|credential|cookie|private[-_]?key)"
)
_INLINE_BEARER_RE = re.compile(r"(?i)(bearer\s+)(?!\[REDACTED\])[^\s\"',;}]+")
_INLINE_HEADER_RE = re.compile(
    r"(?i)((?:authorization|x-api-key|cookie)\s*:\s*)(?!\[REDACTED\])[^\r\n\"',;}]+"
)
_INLINE_KV_RE = re.compile(
    r"(?i)(\b[a-z0-9_-]*(?:api[-_]?key|secret|token|password|passwd|authorization|"
    r"auth[-_]?header|bearer|credential|cookie|private[-_]?key)[a-z0-9_-]*\s*=\s*)"
    r"(?!\[REDACTED\])[^\s&\"',;}]+"
)
_INLINE_JSON_KEY_RE = re.compile(
    r'(?i)("(?:[^"]*(?:api[-_]?key|secret|token|password|passwd|authorization|'
    r'auth[-_]?header|bearer|credential|cookie|private[-_]?key)[^"]*)"\s*:\s*)"(?!\[REDACTED\])[^"]*"'
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
            if _SENSITIVE_KEY_RE.search(key):
                captured[key] = REDACTED_PLACEHOLDER
            else:
                captured[key] = _redact_string(val, depth=0)
    return captured


def _scrub_string_patterns(text: str) -> str:
    scrubbed = _INLINE_BEARER_RE.sub(rf"\1{REDACTED_PLACEHOLDER}", text)
    scrubbed = _INLINE_HEADER_RE.sub(rf"\1{REDACTED_PLACEHOLDER}", scrubbed)
    scrubbed = _INLINE_KV_RE.sub(rf"\1{REDACTED_PLACEHOLDER}", scrubbed)
    scrubbed = _INLINE_JSON_KEY_RE.sub(rf'\1"{REDACTED_PLACEHOLDER}"', scrubbed)
    return scrubbed


def _redact_string(text: str, *, depth: int) -> str:
    if not text:
        return text
    if depth < 4:
        stripped = text.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            with contextlib.suppress(json.JSONDecodeError, ValueError, RecursionError):
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    trailing = (
                        "\r\n" if text.endswith("\r\n") else ("\n" if text.endswith("\n") else "")
                    )
                    redacted_parsed = _redact_value(parsed, depth=depth + 1)
                    return json.dumps(redacted_parsed, ensure_ascii=False) + trailing
    return _scrub_string_patterns(text)


def _redact_container_value(container: Any, *, depth: int) -> Any:
    """Redact the value of an `env` or `headers` field while preserving variable/header names."""
    if container is None:
        return None
    if isinstance(container, Mapping):
        return {str(k): REDACTED_PLACEHOLDER for k in container}
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
        redacted_items: list[Any] = []
        for item in container:
            if isinstance(item, Mapping):
                new_item: dict[str, Any] = {}
                for k, v in item.items():
                    key_str = str(k)
                    if key_str.lower() == "value":
                        new_item[key_str] = REDACTED_PLACEHOLDER
                    else:
                        new_item[key_str] = _redact_value(v, depth=depth + 1)
                redacted_items.append(new_item)
            else:
                redacted_items.append(REDACTED_PLACEHOLDER)
        return redacted_items
    return REDACTED_PLACEHOLDER


def _redact_value(value: Any, *, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value, depth=depth)
    if isinstance(value, Mapping):
        name_field = value.get("name")
        is_sensitive_named_pair = (
            isinstance(name_field, str)
            and "value" in value
            and bool(_SENSITIVE_KEY_RE.search(name_field))
        )
        out: dict[str, Any] = {}
        for raw_k, raw_v in value.items():
            key_str = str(raw_k)
            lower_k = key_str.lower()
            if lower_k in _SENSITIVE_CONTAINERS:
                out[key_str] = _redact_container_value(raw_v, depth=depth + 1)
            elif is_sensitive_named_pair and lower_k == "value":
                out[key_str] = REDACTED_PLACEHOLDER
            elif _SENSITIVE_KEY_RE.search(key_str):
                out[key_str] = REDACTED_PLACEHOLDER
            else:
                out[key_str] = _redact_value(raw_v, depth=depth + 1)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_value(item, depth=depth + 1) for item in value]
    return value


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact credentials, tokens, `env`, `headers`, and raw JSON frames."""
    return _redact_value(value, depth=0)


def log_acp_event(
    event: str,
    *,
    save_dir_or_db: Path | str | None = None,
    **fields: Any,
) -> None:
    """Append a single redacted JSONL diagnostic record to `acp.log` (and stderr if enabled)."""
    redacted_fields = redact_sensitive_data(fields)
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "event": event,
        **(redacted_fields if isinstance(redacted_fields, dict) else {}),
    }
    try:
        serialized = json.dumps(record, ensure_ascii=False, default=str)
        serialized = _scrub_string_patterns(serialized)
    except TypeError, ValueError:
        serialized = json.dumps(
            {
                "timestamp": record["timestamp"],
                "pid": record["pid"],
                "event": event,
                "raw": _scrub_string_patterns(repr(redacted_fields)),
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
