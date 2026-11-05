"""Per-session stdio MCP exec multiplexer for concurrent multiplexed `agy` sessions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from agy_acp.providers.state_store import SqliteStateStoreProvider


def _find_server_config(
    store: SqliteStateStoreProvider,
    server_name: str,
    session_id: str,
) -> dict[str, Any] | None:
    if session_id:
        session = store.get_session(session_id)
        if session is not None:
            for srv in session.get("mcp_servers") or []:
                if isinstance(srv, dict) and str(srv.get("name") or "") == server_name:
                    return srv
    for session in store.list_sessions():
        for srv in session.get("mcp_servers") or []:
            if isinstance(srv, dict) and str(srv.get("name") or "") == server_name:
                return srv
    return None


def main(argv: list[str] | None = None) -> int:
    """Resolve the active session's MCP server definition and `execvpe` it in-place."""
    parser = argparse.ArgumentParser(description="Per-session stdio MCP router")
    parser.add_argument("--db", required=True, help="Path to SQLite state.db")
    parser.add_argument("--server", required=True, help="MCP server name")
    parsed = parser.parse_args(argv)

    session_id = os.environ.get("AGY_ACP_SESSION_ID", "").strip()
    store = SqliteStateStoreProvider(Path(parsed.db))
    cfg = _find_server_config(store, parsed.server, session_id)
    if cfg is None:
        print(
            f"MCP server {parsed.server!r} not configured for session {session_id!r}",
            file=sys.stderr,
        )
        return 1

    command = str(cfg.get("command") or "").strip()
    if not command:
        print(f"MCP server {parsed.server!r} has no stdio command", file=sys.stderr)
        return 1

    raw_args = cfg.get("args")
    args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
    merged_env = dict(os.environ)
    raw_env = cfg.get("env")
    if isinstance(raw_env, dict):
        for k, v in raw_env.items():
            if v is not None:
                merged_env[str(k)] = str(v)

    os.execvpe(command, [command, *args], merged_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
