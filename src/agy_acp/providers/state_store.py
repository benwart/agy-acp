"""SQLite persistent state store for sessions, conversation steps, models cache, and tokens."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    agy_conversation_id TEXT,
    cwd TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'auto',
    model TEXT NOT NULL DEFAULT 'gemini-3.7-pro',
    reasoning_effort TEXT NOT NULL DEFAULT 'medium',
    cancelled INTEGER NOT NULL DEFAULT 0,
    last_step_idx INTEGER NOT NULL DEFAULT -1,
    title TEXT,
    mcp_servers_json TEXT NOT NULL DEFAULT '[]',
    meta_json TEXT NOT NULL DEFAULT '{}',
    auth_config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_cwd ON sessions(cwd);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS steps (
    session_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, idx),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_steps_session_idx ON steps(session_id, idx ASC);

CREATE TABLE IF NOT EXISTS models_cache (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS token_ledger (
    scope TEXT PRIMARY KEY,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_read_tokens INTEGER NOT NULL DEFAULT 0,
    thinking_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def resolve_state_db_path(save_dir_or_db: Path | str | None = None) -> Path:
    """Resolve the SQLite database path from `save_dir_or_db`, `$AGY_ACP_STATE_DB`, or default."""
    if save_dir_or_db is not None:
        candidate = Path(save_dir_or_db)
        if candidate.suffix in {".db", ".sqlite", ".sqlite3"}:
            return candidate
        return candidate / "state.db"

    env_db = os.environ.get("AGY_ACP_STATE_DB")
    if env_db:
        return Path(env_db)

    return Path.home() / ".config" / "agy-acp" / "state.db"


class SqliteStateStoreProvider:
    """Manages SQLite persistence for sessions, steps, discovered models, and token usage."""

    def __init__(self, save_dir_or_db: Path | str | None = None) -> None:
        self._db_path = resolve_state_db_path(save_dir_or_db)
        self._ensure_initialized()

    @property
    def db_path(self) -> Path:
        """Return the resolved SQLite database file path."""
        return self._db_path

    def _ensure_initialized(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._db_path.exists():
            fd = os.open(self._db_path, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(fd)
        self._db_path.chmod(0o600)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @contextlib.contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sessionId": str(row["session_id"]),
            "agyConversationId": row["agy_conversation_id"],
            "cwd": str(row["cwd"]),
            "mode": str(row["mode"]),
            "model": str(row["model"]),
            "reasoning_effort": str(row["reasoning_effort"]),
            "cancelled": bool(row["cancelled"]),
            "lastStepIdx": int(row["last_step_idx"]),
            "title": row["title"],
            "mcp_servers": json.loads(row["mcp_servers_json"] or "[]"),
            "meta": json.loads(row["meta_json"] or "{}"),
            "auth_config": json.loads(row["auth_config_json"] or "{}"),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def upsert_session(self, state: dict[str, Any]) -> dict[str, Any]:
        """Insert or update a session row in SQLite and return the normalized session dict."""
        now = datetime.now(UTC).isoformat()
        sid = str(state["sessionId"])
        existing = self.get_session(sid) or {}

        agy_conv_id = state.get("agyConversationId", existing.get("agyConversationId"))
        cwd = str(state.get("cwd", existing.get("cwd", "")))
        mode = str(state.get("mode", existing.get("mode", "auto")))
        model = str(state.get("model", existing.get("model", "gemini-3.7-pro")))
        effort = str(state.get("reasoning_effort", existing.get("reasoning_effort", "medium")))
        cancelled = int(bool(state.get("cancelled", existing.get("cancelled", False))))
        last_step_idx = int(state.get("lastStepIdx", existing.get("lastStepIdx", -1)))
        title = state.get("title", existing.get("title"))
        mcp_servers = state.get("mcp_servers", existing.get("mcp_servers", []))
        meta = state.get("meta", existing.get("meta", {}))
        auth_config = state.get("auth_config", existing.get("auth_config", {}))
        created_at = str(existing.get("createdAt") or now)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, agy_conversation_id, cwd, mode, model, reasoning_effort,
                    cancelled, last_step_idx, title, mcp_servers_json, meta_json,
                    auth_config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    agy_conversation_id = excluded.agy_conversation_id,
                    cwd = excluded.cwd,
                    mode = excluded.mode,
                    model = excluded.model,
                    reasoning_effort = excluded.reasoning_effort,
                    cancelled = excluded.cancelled,
                    last_step_idx = excluded.last_step_idx,
                    title = excluded.title,
                    mcp_servers_json = excluded.mcp_servers_json,
                    meta_json = excluded.meta_json,
                    auth_config_json = excluded.auth_config_json,
                    updated_at = excluded.updated_at
                """,
                (
                    sid,
                    agy_conv_id,
                    cwd,
                    mode,
                    model,
                    effort,
                    cancelled,
                    last_step_idx,
                    title,
                    json.dumps(mcp_servers),
                    json.dumps(meta),
                    json.dumps(auth_config),
                    created_at,
                    now,
                ),
            )
        loaded = self.get_session(sid)
        assert loaded is not None
        return loaded

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a single session by `session_id` from SQLite."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_sessions(self, *, cwd: str | None = None) -> list[dict[str, Any]]:
        """Return persisted sessions ordered by `updated_at DESC`, optionally filtered by `cwd`."""
        with self._connect() as conn:
            if cwd is not None:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE cwd = ? ORDER BY updated_at DESC",
                    (cwd,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY updated_at DESC",
                ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete `session_id` (and cascading steps) from SQLite. Returns True if a row existed."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            return cur.rowcount > 0

    def upsert_step(
        self,
        *,
        session_id: str,
        idx: int,
        step_type: str,
        status: str,
        data: dict[str, Any] | str,
    ) -> None:
        """Persist or update a conversation step row in SQLite."""
        now = datetime.now(UTC).isoformat()
        serialized = data if isinstance(data, str) else json.dumps(data)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO steps (session_id, idx, step_type, status, data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, idx) DO UPDATE SET
                    step_type = excluded.step_type,
                    status = excluded.status,
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (session_id, idx, step_type, status, serialized, now),
            )
            conn.execute(
                """
                UPDATE sessions
                SET last_step_idx = MAX(last_step_idx, ?), updated_at = ?
                WHERE session_id = ?
                """,
                (idx, now, session_id),
            )

    def get_steps_since(self, session_id: str, *, after_idx: int = -1) -> list[dict[str, Any]]:
        """Query `steps` for `session_id` with `idx > after_idx` ordered by `idx ASC`."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT idx, step_type, status, data, updated_at
                FROM steps
                WHERE session_id = ? AND idx > ?
                ORDER BY idx ASC
                """,
                (session_id, after_idx),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            raw_data = row["data"]
            try:
                parsed_data = json.loads(raw_data)
            except json.JSONDecodeError:
                parsed_data = {"raw": raw_data}
            result.append(
                {
                    "idx": int(row["idx"]),
                    "step_type": str(row["step_type"]),
                    "status": str(row["status"]),
                    "data": parsed_data,
                    "updated_at": str(row["updated_at"]),
                }
            )
        return result

    def save_models(self, models: list[dict[str, str]]) -> None:
        """Replace cached models in the SQLite `models_cache` table."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM models_cache")
            for idx, m in enumerate(models):
                model_id = str(m.get("modelId") or m.get("id", ""))
                if not model_id:
                    continue
                name = str(m.get("name", model_id))
                desc = str(m.get("description", name))
                conn.execute(
                    """
                    INSERT INTO models_cache (model_id, name, description, sort_order, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (model_id, name, desc, idx, now),
                )

    def load_models(self) -> list[dict[str, str]]:
        """Load cached models from the SQLite `models_cache` table ordered by `sort_order ASC`."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_id, name, description FROM models_cache ORDER BY sort_order ASC"
            ).fetchall()
        return [
            {
                "modelId": str(r["model_id"]),
                "name": str(r["name"]),
                "description": str(r["description"]),
            }
            for r in rows
        ]

    def record_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_read_tokens: int = 0,
        thinking_tokens: int = 0,
        scope: str = "global",
    ) -> dict[str, int]:
        """Atomically increment cumulative token counts in SQLite `token_ledger`."""
        now = datetime.now(UTC).isoformat()
        added_total = input_tokens + output_tokens + cached_read_tokens + thinking_tokens
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO token_ledger (
                    scope, input_tokens, output_tokens, cached_read_tokens,
                    thinking_tokens, total_tokens, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    input_tokens = token_ledger.input_tokens + excluded.input_tokens,
                    output_tokens = token_ledger.output_tokens + excluded.output_tokens,
                    cached_read_tokens = (
                        token_ledger.cached_read_tokens + excluded.cached_read_tokens
                    ),
                    thinking_tokens = token_ledger.thinking_tokens + excluded.thinking_tokens,
                    total_tokens = token_ledger.total_tokens + excluded.total_tokens,
                    updated_at = excluded.updated_at
                """,
                (
                    scope,
                    input_tokens,
                    output_tokens,
                    cached_read_tokens,
                    thinking_tokens,
                    added_total,
                    now,
                ),
            )
        return self.get_token_ledger(scope=scope)

    def get_token_ledger(self, *, scope: str = "global") -> dict[str, int]:
        """Return cumulative token ledger for `scope` from SQLite."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT input_tokens, output_tokens, cached_read_tokens,
                       thinking_tokens, total_tokens
                FROM token_ledger
                WHERE scope = ?
                """,
                (scope,),
            ).fetchone()
        if row is None:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_read_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
            }
        return {
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "cached_read_tokens": int(row["cached_read_tokens"]),
            "thinking_tokens": int(row["thinking_tokens"]),
            "total_tokens": int(row["total_tokens"]),
        }

    def set_kv(self, key: str, value: Any) -> None:
        """Persist a JSON-serializable value in `kv_state`."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kv_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), now),
            )

    def get_kv(self, key: str) -> Any:
        """Retrieve a JSON value from `kv_state`, or `None` if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM kv_state WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["value_json"])
