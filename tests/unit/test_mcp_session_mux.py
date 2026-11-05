"""Unit tests for per-session stdio MCP exec multiplexer (`mcp_session_mux`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agy_acp.providers import mcp_session_mux
from agy_acp.providers.state_store import SqliteStateStoreProvider


def _seed_store(tmp_path: Path) -> SqliteStateStoreProvider:
    store = SqliteStateStoreProvider(tmp_path / "state.db")
    store.upsert_session(
        {
            "sessionId": "sess-a",
            "cwd": str(tmp_path),
            "mcp_servers": [
                {
                    "name": "priv-db",
                    "command": "/usr/bin/env",
                    "args": ["python3", "-m", "priv_db"],
                    "env": {"DB_PASSWORD": "TOP_SECRET_A_PASSWORD"},
                },
                {
                    "name": "disabled-srv",
                    "command": "/usr/bin/env",
                    "args": ["echo", "disabled"],
                    "disabled": True,
                },
            ],
        }
    )
    store.upsert_session(
        {
            "sessionId": "sess-b",
            "cwd": str(tmp_path),
            "mcp_servers": [
                {
                    "name": "public-docs",
                    "command": "/usr/bin/env",
                    "args": ["echo", "docs"],
                }
            ],
        }
    )
    return store


def test_find_server_config_enforces_strict_session_isolation(tmp_path: Path) -> None:
    """Verify missing session, unknown session, cross-session lookup, and disabled return None."""
    store = _seed_store(tmp_path)

    # Missing / whitespace session_id -> None (never falls back to list_sessions)
    assert mcp_session_mux._find_server_config(store, "priv-db", "") is None
    assert mcp_session_mux._find_server_config(store, "priv-db", "   ") is None

    # Unknown session_id -> None
    assert mcp_session_mux._find_server_config(store, "priv-db", "sess-unknown") is None

    # Server only in Session A is strictly denied to Session B
    assert mcp_session_mux._find_server_config(store, "priv-db", "sess-b") is None

    # Disabled server in Session A -> None
    assert mcp_session_mux._find_server_config(store, "disabled-srv", "sess-a") is None

    # Valid server in owning Session A -> returns config
    cfg = mcp_session_mux._find_server_config(store, "priv-db", "sess-a")
    assert cfg is not None
    assert cfg["name"] == "priv-db"
    assert cfg["env"]["DB_PASSWORD"] == "TOP_SECRET_A_PASSWORD"


def test_main_rejects_unconfigured_or_cross_session_without_leaking_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify main() exits 1 cleanly without leaking credentials when denied."""
    store = _seed_store(tmp_path)

    # Empty AGY_ACP_SESSION_ID
    monkeypatch.delenv("AGY_ACP_SESSION_ID", raising=False)
    rc = mcp_session_mux.main(["--db", str(store.db_path), "--server", "priv-db"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "TOP_SECRET_A_PASSWORD" not in captured.err
    assert "TOP_SECRET_A_PASSWORD" not in captured.out

    # Cross-session hijack attempt from sess-b
    monkeypatch.setenv("AGY_ACP_SESSION_ID", "sess-b")
    rc2 = mcp_session_mux.main(["--db", str(store.db_path), "--server", "priv-db"])
    captured2 = capsys.readouterr()
    assert rc2 == 1
    assert "TOP_SECRET_A_PASSWORD" not in captured2.err


def test_main_validates_cwd_and_execs_configured_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify main() chdirs to valid cwd, rejects non-directory cwd, and execs server."""
    srv_cwd = tmp_path / "srv_workdir"
    srv_cwd.mkdir()
    store = SqliteStateStoreProvider(tmp_path / "state.db")
    store.upsert_session(
        {
            "sessionId": "sess-exec",
            "cwd": str(tmp_path),
            "mcp_servers": [
                {
                    "name": "worker-mcp",
                    "command": "/bin/echo",
                    "args": ["hello"],
                    "cwd": str(srv_cwd),
                    "env": [{"name": "MCP_TOKEN", "value": "tok-xyz"}],
                },
                {
                    "name": "bad-cwd-mcp",
                    "command": "/bin/echo",
                    "args": ["bad"],
                    "cwd": str(tmp_path / "nonexistent_dir"),
                },
            ],
        }
    )

    monkeypatch.setenv("AGY_ACP_SESSION_ID", "sess-exec")

    # Invalid cwd must exit 1 without calling execvpe
    exec_calls: list[tuple[str, list[str], dict[str, str], Path]] = []

    def _fake_execvpe(file: str, args: list[str], env: dict[str, Any]) -> None:
        exec_calls.append((file, list(args), dict(env), Path.cwd()))

    monkeypatch.setattr("os.execvpe", _fake_execvpe)
    rc_bad = mcp_session_mux.main(["--db", str(store.db_path), "--server", "bad-cwd-mcp"])
    assert rc_bad == 1
    assert exec_calls == []

    orig_cwd = Path.cwd()
    try:
        rc_ok = mcp_session_mux.main(["--db", str(store.db_path), "--server", "worker-mcp"])
        assert rc_ok == 0
        assert len(exec_calls) == 1
        cmd, argv, child_env, observed_cwd = exec_calls[0]
        assert cmd == "/bin/echo"
        assert argv == ["/bin/echo", "hello"]
        assert child_env["MCP_TOKEN"] == "tok-xyz"
        assert observed_cwd.resolve() == srv_cwd.resolve()
    finally:
        monkeypatch.chdir(orig_cwd)
