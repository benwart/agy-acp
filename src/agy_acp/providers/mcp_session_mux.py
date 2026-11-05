"""Per-session stdio MCP exec multiplexer for concurrent multiplexed `agy` sessions."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agy_acp.providers.state_store import SqliteStateStoreProvider


def _find_server_config(
    store: SqliteStateStoreProvider,
    server_name: str,
    session_id: str,
) -> dict[str, Any] | None:
    normalized_sid = session_id.strip()
    if not normalized_sid:
        return None
    session = store.get_session(normalized_sid)
    if session is None:
        return None
    for srv in session.get("mcp_servers") or []:
        if isinstance(srv, dict) and str(srv.get("name") or "") == server_name:
            if bool(srv.get("disabled")) is True:
                return None
            return srv
    return None


def _merge_server_env(base_env: Mapping[str, str], raw_env: Any) -> dict[str, str]:
    child_env = dict(base_env)
    if isinstance(raw_env, Mapping):
        for k, v in raw_env.items():
            if v is not None:
                child_env[str(k)] = str(v)
    elif isinstance(raw_env, Sequence) and not isinstance(raw_env, (str, bytes)):
        for item in raw_env:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            value = item.get("value")
            if isinstance(name, str) and name and value is not None:
                child_env[name] = str(value)
    return child_env


def main(argv: list[str] | None = None) -> int:
    """Resolve the active session's MCP server definition and `execvpe` it in-place."""
    parser = argparse.ArgumentParser(description="Per-session stdio MCP router")
    parser.add_argument("--db", required=True, help="Path to SQLite state.db")
    parser.add_argument("--server", required=True, help="MCP server name")
    parsed = parser.parse_args(argv)

    session_id = os.environ.get("AGY_ACP_SESSION_ID", "").strip()
    if not session_id:
        print("MCP router error: missing active session identifier", file=sys.stderr)
        return 1

    store = SqliteStateStoreProvider(Path(parsed.db))
    cfg = _find_server_config(store, parsed.server, session_id)
    if cfg is None:
        print(
            f"MCP router error: server {parsed.server!r} is not configured or enabled for session",
            file=sys.stderr,
        )
        return 1

    command = str(cfg.get("command") or "").strip()
    if not command:
        print(f"MCP router error: server {parsed.server!r} has no stdio command", file=sys.stderr)
        return 1

    raw_cwd = cfg.get("cwd")
    if isinstance(raw_cwd, str) and raw_cwd.strip():
        cwd_path = Path(raw_cwd.strip())
        if not cwd_path.is_dir():
            print(
                "MCP router error: configured server working directory is invalid", file=sys.stderr
            )
            return 1
        os.chdir(cwd_path)

    raw_args = cfg.get("args")
    cmd_args = (
        [str(a) for a in raw_args]
        if isinstance(raw_args, Sequence) and not isinstance(raw_args, (str, bytes))
        else []
    )
    child_env = _merge_server_env(os.environ, cfg.get("env"))

    os.execvpe(command, [command, *cmd_args], child_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
