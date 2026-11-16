"""Unit tests for SqliteStateStoreProvider persistence of sessions, steps, models, and tokens."""

from __future__ import annotations

from pathlib import Path

from agy_acp.providers.state_store import SqliteStateStoreProvider


def test_sqlite_session_crud_and_query_by_cwd(tmp_path: Path) -> None:
    """Verify sessions persist in SQLite and query by session_id or cwd without file scans."""
    store = SqliteStateStoreProvider(tmp_path)
    assert store.db_path == tmp_path / "state.db"
    assert store.db_path.is_file()
    assert (store.db_path.stat().st_mode & 0o777) == 0o600

    store.upsert_session(
        {
            "sessionId": "sess-alpha",
            "agyConversationId": "conv-alpha",
            "cwd": "/workspace/alpha",
            "mode": "auto",
            "model": "gemini-3.7-pro",
            "reasoning_effort": "medium",
            "cancelled": False,
            "mcp_servers": [{"name": "srv1", "command": "echo"}],
            "meta": {"team": "core"},
        }
    )
    store.upsert_session(
        {
            "sessionId": "sess-beta",
            "agyConversationId": None,
            "cwd": "/workspace/beta",
            "mode": "read-only",
            "model": "gemini-3.8-flash",
            "reasoning_effort": "high",
            "cancelled": True,
            "mcp_servers": [],
            "meta": {},
        }
    )

    # Verify a fresh store instance reads directly from SQLite on disk
    store_reloaded = SqliteStateStoreProvider(tmp_path)
    alpha = store_reloaded.get_session("sess-alpha")
    assert alpha is not None
    assert alpha["sessionId"] == "sess-alpha"
    assert alpha["agyConversationId"] == "conv-alpha"
    assert alpha["cwd"] == "/workspace/alpha"
    assert alpha["mcp_servers"] == [{"name": "srv1", "command": "echo"}]
    assert alpha["meta"] == {"team": "core"}
    assert alpha["cancelled"] is False

    beta = store_reloaded.get_session("sess-beta")
    assert beta is not None
    assert beta["mode"] == "read-only"
    assert beta["cancelled"] is True

    # Verify indexed query by cwd
    alpha_list = store_reloaded.list_sessions(cwd="/workspace/alpha")
    assert len(alpha_list) == 1
    assert alpha_list[0]["sessionId"] == "sess-alpha"

    all_sessions = store_reloaded.list_sessions()
    assert len(all_sessions) == 2

    # Verify deletion
    assert store_reloaded.delete_session("sess-alpha") is True
    assert store_reloaded.get_session("sess-alpha") is None
    assert store_reloaded.delete_session("sess-alpha") is False


def test_sqlite_steps_recording_and_retrieval(tmp_path: Path) -> None:
    """Verify conversation step rows are stored and queried incrementally by idx."""
    store = SqliteStateStoreProvider(tmp_path)
    store.upsert_session({"sessionId": "sess-1", "cwd": "/ws"})

    store.upsert_step(
        session_id="sess-1",
        idx=1,
        step_type="planner_response",
        status="RUNNING",
        data={"thinking": "Step 1 thought"},
    )
    store.upsert_step(
        session_id="sess-1",
        idx=2,
        step_type="run_command",
        status="DONE",
        data={"tool_name": "run_command", "output": "ok"},
    )

    all_steps = store.get_steps_since("sess-1", after_idx=-1)
    assert len(all_steps) == 2
    assert all_steps[0]["idx"] == 1
    assert all_steps[0]["step_type"] == "planner_response"
    assert all_steps[0]["data"]["thinking"] == "Step 1 thought"

    newer_steps = store.get_steps_since("sess-1", after_idx=1)
    assert len(newer_steps) == 1
    assert newer_steps[0]["idx"] == 2
    assert newer_steps[0]["status"] == "DONE"

    # Deleting session cascades to steps
    store.delete_session("sess-1")
    assert store.get_steps_since("sess-1", after_idx=-1) == []


def test_sqlite_models_cache_and_kv_state(tmp_path: Path) -> None:
    """Verify models cache and key-value state persist in SQLite."""
    store = SqliteStateStoreProvider(tmp_path)
    assert store.load_models() == []

    models = [
        {"modelId": "gemini-3.7-pro", "name": "Gemini 3.7 Pro", "description": "Pro"},
        {"modelId": "gemini-3.8-flash", "name": "Gemini 3.8 Flash", "description": "Flash"},
    ]
    store.save_models(models)

    store2 = SqliteStateStoreProvider(tmp_path)
    loaded = store2.load_models()
    assert loaded == models

    store.set_kv("last_sync", {"ts": 12345})
    assert store2.get_kv("last_sync") == {"ts": 12345}
    assert store2.get_kv("missing") is None


def test_sqlite_models_cache_expires_after_one_day(tmp_path: Path) -> None:
    """Verify load_models returns empty list when cached rows are older than 24 hours."""
    from datetime import timedelta

    from freezegun import freeze_time

    store = SqliteStateStoreProvider(tmp_path)
    models = [
        {
            "modelId": "gemini-4-argon-medium",
            "name": "Gemini 4 Argon (Medium)",
            "description": "Argon",
        },
    ]
    with freeze_time("2026-11-15T12:00:00Z") as frozen_clock:
        store.save_models(models)
        assert store.load_models() == models

        frozen_clock.tick(timedelta(days=1, seconds=1))
        assert store.load_models() == []
        assert store.load_models(max_age_seconds=None) == models


def test_sqlite_token_ledger_persistence_across_instances(tmp_path: Path) -> None:
    """Verify cumulative token usage persists in SQLite across instances."""
    store1 = SqliteStateStoreProvider(tmp_path)
    store1.record_tokens(
        input_tokens=100,
        output_tokens=40,
        cached_read_tokens=20,
        thinking_tokens=10,
    )

    store2 = SqliteStateStoreProvider(tmp_path)
    store2.record_tokens(
        input_tokens=50,
        output_tokens=10,
        cached_read_tokens=5,
        thinking_tokens=5,
    )
    ledger = store2.get_token_ledger()
    assert ledger == {
        "input_tokens": 150,
        "output_tokens": 50,
        "cached_read_tokens": 25,
        "thinking_tokens": 15,
        "total_tokens": 240,
    }


def test_sqlite_custom_agents_column_and_legacy_db_migration(tmp_path: Path) -> None:
    """Verify custom_agents_json persists on new DBs and migrates cleanly on legacy DBs."""
    import sqlite3

    legacy_db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy_db))
    conn.executescript(
        """
        CREATE TABLE sessions (
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
        INSERT INTO sessions (
            session_id, cwd, created_at, updated_at
        ) VALUES ('legacy-sess', '/legacy', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
        """
    )
    conn.commit()
    conn.close()

    store = SqliteStateStoreProvider(legacy_db)
    migrated = store.get_session("legacy-sess")
    assert migrated is not None
    assert migrated["custom_agents"] == []

    agents = [
        {"id": "reviewer", "name": "Code Reviewer", "prompt": "Review diffs carefully."},
    ]
    updated = store.upsert_session(
        {
            "sessionId": "legacy-sess",
            "custom_agents": agents,
        }
    )
    assert updated["custom_agents"] == agents

    reloaded = SqliteStateStoreProvider(legacy_db).get_session("legacy-sess")
    assert reloaded is not None
    assert reloaded["custom_agents"] == agents


def test_sqlite_session_turns_lifecycle_and_cascade(tmp_path: Path) -> None:
    """Verify session_turns append, ordering, clear, reload preservation, and delete cascade."""
    store = SqliteStateStoreProvider(tmp_path)
    store.upsert_session({"sessionId": "sess-turns", "cwd": "/ws"})

    idx0 = store.append_session_turn(
        session_id="sess-turns",
        user_message_id="msg-1",
        prompt_blocks=[{"type": "text", "text": "Hello"}],
        assistant_text="Hi there!",
    )
    idx1 = store.append_session_turn(
        session_id="sess-turns",
        user_message_id="msg-2",
        prompt_blocks=[{"type": "text", "text": "Follow up"}],
        assistant_text="Second reply.",
    )
    assert (idx0, idx1) == (0, 1)

    # Non-destructive session metadata update and store reload keep turns intact
    store.upsert_session({"sessionId": "sess-turns", "mode": "approve"})
    store_reloaded = SqliteStateStoreProvider(tmp_path)
    turns = store_reloaded.get_session_turns("sess-turns")
    assert len(turns) == 2
    assert turns[0]["turn_idx"] == 0
    assert turns[0]["user_message_id"] == "msg-1"
    assert turns[0]["prompt_blocks"] == [{"type": "text", "text": "Hello"}]
    assert turns[0]["assistant_text"] == "Hi there!"
    assert turns[1]["turn_idx"] == 1
    assert turns[1]["user_message_id"] == "msg-2"
    assert turns[1]["assistant_text"] == "Second reply."

    # Explicit clear removes turns without deleting the session
    removed = store_reloaded.clear_session_turns("sess-turns")
    assert removed == 2
    assert store_reloaded.get_session_turns("sess-turns") == []
    assert store_reloaded.get_session("sess-turns") is not None

    # Appending again restarts at turn_idx 0, and delete_session cascades
    assert (
        store_reloaded.append_session_turn(
            session_id="sess-turns",
            user_message_id="msg-3",
            prompt_blocks=[{"type": "text", "text": "After compact"}],
            assistant_text="Fresh start.",
        )
        == 0
    )
    assert store_reloaded.delete_session("sess-turns") is True
    assert store_reloaded.get_session_turns("sess-turns") == []


def test_sqlite_context_paths_knowledge_entries_and_revert_turns_from_message_id(
    tmp_path: Path,
) -> None:
    """Verify contextPaths, knowledgeEntries persistence and revert_turns_from_message_id."""
    store = SqliteStateStoreProvider(tmp_path)
    store.upsert_session(
        {
            "sessionId": "sess-0121",
            "cwd": "/ws",
            "contextPaths": ["/ws/a.py", "/ws/b.py"],
            "knowledgeEntries": [
                {"name": "Arch", "path": "/ws/docs/arch.md", "item_count": 1},
            ],
        }
    )

    reloaded = SqliteStateStoreProvider(tmp_path).get_session("sess-0121")
    assert reloaded is not None
    assert reloaded["contextPaths"] == ["/ws/a.py", "/ws/b.py"]
    assert reloaded["knowledgeEntries"] == [
        {"name": "Arch", "path": "/ws/docs/arch.md", "item_count": 1}
    ]

    store.append_session_turn(
        session_id="sess-0121",
        user_message_id="m-0",
        prompt_blocks=[{"type": "text", "text": "t0"}],
        assistant_text="a0",
    )
    store.append_session_turn(
        session_id="sess-0121",
        user_message_id="m-1",
        prompt_blocks=[{"type": "text", "text": "t1"}],
        assistant_text="a1",
    )
    store.append_session_turn(
        session_id="sess-0121",
        user_message_id="m-2",
        prompt_blocks=[{"type": "text", "text": "t2"}],
        assistant_text="a2",
    )

    assert store.revert_turns_from_message_id("sess-0121", "missing-id") == 0
    assert len(store.get_session_turns("sess-0121")) == 3

    reverted = store.revert_turns_from_message_id("sess-0121", "m-1")
    assert reverted == 2
    remaining = store.get_session_turns("sess-0121")
    assert len(remaining) == 1
    assert remaining[0]["user_message_id"] == "m-0"


def test_save_and_load_quota_groups_and_credits_with_ttl(tmp_path: Path) -> None:
    """Verify `SqliteStateStoreProvider` persists quota groups and G1 credits with TTL bounds."""
    from datetime import timedelta

    from agy_acp.providers.state_store import (
        QUOTA_SNAPSHOT_TTL_SECONDS,
        SqliteStateStoreProvider,
    )

    db_path = tmp_path / "state.db"
    store = SqliteStateStoreProvider(db_path)
    assert store.load_quota_groups() is None
    assert store.load_credits() is None

    sample_groups = [
        {
            "group": "Gemini Models",
            "buckets": [
                {
                    "id": "gemini-5h",
                    "group": "Gemini Models",
                    "name": "Five Hour Limit Remaining",
                    "window": "5h",
                    "remaining_fraction": 0.96,
                    "reset_time": "2026-11-14T08:00:00Z",
                }
            ],
        }
    ]
    sample_credits = {"remaining_credits": 250, "upgrade_uri": "https://one.google.com/ai"}

    saved_groups = store.save_quota_groups(sample_groups)
    saved_credits = store.save_credits(sample_credits)
    assert saved_groups == sample_groups
    assert saved_credits == sample_credits

    reloaded_store = SqliteStateStoreProvider(db_path)
    assert reloaded_store.load_quota_groups() == sample_groups
    assert reloaded_store.load_credits() == sample_credits

    # Empty/degraded writes must not clobber valid cached snapshots
    assert reloaded_store.save_quota_groups([]) is None
    assert reloaded_store.save_quota_groups([{"group": "Gemini Models", "buckets": []}]) is None
    assert reloaded_store.save_credits({}) is None
    assert reloaded_store.load_quota_groups() == sample_groups
    assert reloaded_store.load_credits() == sample_credits

    # Expire timestamps beyond QUOTA_SNAPSHOT_TTL_SECONDS using deterministic clock
    from freezegun import freeze_time

    with freeze_time("2026-11-15T12:00:00Z") as frozen_clock:
        reloaded_store.save_quota_groups(sample_groups)
        reloaded_store.save_credits(sample_credits)
        frozen_clock.tick(timedelta(seconds=QUOTA_SNAPSHOT_TTL_SECONDS + 60))
        assert reloaded_store.load_quota_groups() is None
        assert reloaded_store.load_credits() is None
        assert reloaded_store.load_quota_groups(max_age_seconds=None) == sample_groups
        assert reloaded_store.load_credits(max_age_seconds=None) == sample_credits
