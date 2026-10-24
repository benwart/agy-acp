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
