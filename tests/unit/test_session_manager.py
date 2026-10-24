"""Unit tests for SessionManager lifecycle, multiplexing, mode/model updates, and cancellation."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from agy_acp.logic.session_manager import SessionManager


async def test_create_session_maps_mcp_and_cwd(tmp_path: Path) -> None:
    """Verify SessionManager.create_session returns valid sessionId and configOptions."""
    mgr = SessionManager(save_dir=tmp_path)
    desc = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[{"name": "srv", "command": "echo", "args": [], "env": []}],
    )
    assert len(desc["sessionId"]) >= 32
    assert "configOptions" in desc


async def test_load_session_resumes_trajectory(tmp_path: Path) -> None:
    """Verify SessionManager.load_session restores an existing sessionId."""
    mgr = SessionManager(save_dir=tmp_path)
    created = await mgr.create_session(cwd=str(tmp_path))
    loaded = await mgr.load_session(session_id=created["sessionId"], cwd=str(tmp_path))
    assert len(loaded["sessionId"]) >= 32
    assert loaded["sessionId"] == created["sessionId"]


async def test_multiplexed_session_isolation(tmp_path: Path) -> None:
    """Verify SessionManager isolates state between concurrent sessionIds."""
    mgr = SessionManager(save_dir=tmp_path)
    ws1 = tmp_path / "ws1"
    ws2 = tmp_path / "ws2"
    mcp1 = [{"name": "srv1", "command": "echo", "args": ["1"], "env": []}]
    mcp2 = [{"name": "srv2", "command": "echo", "args": ["2"], "env": []}]
    s1 = await mgr.create_session(cwd=str(ws1), mcp_servers=mcp1, meta={"role": "a"})
    s2 = await mgr.create_session(cwd=str(ws2), mcp_servers=mcp2, meta={"role": "b"})
    assert s1["sessionId"] != s2["sessionId"]

    mcp1[0]["name"] = "mutated"
    state1 = mgr.get_session(s1["sessionId"])
    state2 = mgr.get_session(s2["sessionId"])
    assert state1["cwd"] == str(ws1)
    assert state2["cwd"] == str(ws2)
    assert state1["mcp_servers"][0]["name"] == "srv1"
    assert state2["mcp_servers"][0]["name"] == "srv2"
    assert state1["meta"] == {"role": "a"}
    assert state2["meta"] == {"role": "b"}

    state1["meta"]["role"] = "tampered"
    assert mgr.get_session(s1["sessionId"])["meta"]["role"] == "a"


async def test_update_session_mode_and_model(tmp_path: Path) -> None:
    """Verify SessionManager.set_mode, set_model, and set_config_option."""
    mgr = SessionManager(save_dir=tmp_path)
    s1 = await mgr.create_session(cwd=str(tmp_path))
    sid = s1["sessionId"]
    assert await mgr.set_mode(session_id=sid, mode_id="read-only") == {}
    assert await mgr.set_model(session_id=sid, model_id="gemini-3.8-flash") == {}
    cfg = await mgr.set_config_option(session_id=sid, config_id="reasoning_effort", value="high")
    assert "configOptions" in cfg

    by_id = {opt["id"]: opt["currentValue"] for opt in cfg["configOptions"]}
    assert by_id == {
        "model": "gemini-3.8-flash",
        "mode": "read-only",
        "reasoning_effort": "high",
    }

    await mgr.set_config_option(session_id=sid, config_id="mode", value="approve")
    await mgr.set_config_option(session_id=sid, config_id="model", value="gemini-3.7-pro")
    cfg2 = await mgr.set_config_option(session_id=sid, config_id="thinkingLevel", value="low")
    by_id2 = {opt["id"]: opt["currentValue"] for opt in cfg2["configOptions"]}
    assert by_id2 == {
        "model": "gemini-3.7-pro",
        "mode": "approve",
        "reasoning_effort": "low",
    }

    with pytest.raises(ValueError, match="Mode 'invalid' not found"):
        await mgr.set_mode(session_id=sid, mode_id="invalid")


async def test_cancel_and_steer_active_turn(tmp_path: Path) -> None:
    """Verify SessionManager.cancel stops an active turn and tracks cancellation."""
    mgr = SessionManager(save_dir=tmp_path)
    s1 = await mgr.create_session(cwd=str(tmp_path))
    sid = s1["sessionId"]

    assert mgr.is_cancelled(sid) is False
    cancelled = await mgr.cancel(session_id=sid)
    assert cancelled is True
    assert mgr.is_cancelled(sid) is True
    assert mgr.get_session(sid)["cancelled"] is True

    assert mgr.consume_cancellation(sid) is True
    assert mgr.is_cancelled(sid) is False
    assert mgr.get_session(sid)["cancelled"] is False
    assert mgr.consume_cancellation(sid) is False

    scope_cancelled = False
    turn_started = anyio.Event()

    async def _run_active_turn() -> None:
        nonlocal scope_cancelled
        with mgr.turn_cancel_scope(sid) as scope:
            turn_started.set()
            await anyio.sleep(10)
        scope_cancelled = scope.cancel_called

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_active_turn)
        await turn_started.wait()
        assert await mgr.cancel(session_id=sid) is True

    assert scope_cancelled is True
    assert mgr.is_cancelled(sid) is True


async def test_terminate_and_delete_session(tmp_path: Path) -> None:
    """Verify SessionManager.close_session cleans up session state in SQLite without JSON files."""
    mgr = SessionManager(save_dir=tmp_path)
    s1 = await mgr.create_session(cwd=str(tmp_path))
    sid = s1["sessionId"]
    db_file = tmp_path / "state.db"
    assert db_file.is_file()
    assert not (tmp_path / f"{sid}.json").exists()
    assert len(mgr.list_sessions(cwd=str(tmp_path))) == 1

    await mgr.cancel(session_id=sid)
    assert mgr.is_cancelled(sid) is True

    scope_cancelled = False
    turn_started = anyio.Event()

    async def _run_active_turn() -> None:
        nonlocal scope_cancelled
        with mgr.turn_cancel_scope(sid) as scope:
            turn_started.set()
            await anyio.sleep(10)
        scope_cancelled = scope.cancel_called

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_active_turn)
        await turn_started.wait()
        res = await mgr.close_session(session_id=sid)
        assert res == {}

    assert scope_cancelled is True
    assert mgr.is_cancelled(sid) is False
    assert sid not in mgr._active_cancel_scopes
    assert mgr.get_session(sid)["sessionId"] == sid

    del_res = await mgr.close_session(session_id=sid, delete_persisted=True)
    assert del_res == {}
    assert mgr.list_sessions(cwd=str(tmp_path)) == []
    with pytest.raises(KeyError, match=sid):
        mgr.get_session(sid)


async def test_session_manager_kirocrew_spawn_agent_and_agent_modes(tmp_path: Path) -> None:
    """Verify SessionManager advertises spawn --agent and kirocrew agent modes for Guard A/A2."""
    agents_dir = tmp_path / ".kiro" / "agents"
    agents_dir.mkdir(parents=True)
    agent_path = agents_dir / "custom-reviewer.json"
    agent_path.write_text('{"name": "custom-reviewer"}', encoding="utf-8")

    mgr = SessionManager(
        save_dir=tmp_path,
        default_agent="kirocrew-skill-view-9608729a",
        default_model="gemini-4-argon-medium",
    )
    desc = await mgr.create_session(cwd=str(tmp_path))
    assert desc["modes"]["currentModeId"] == "kirocrew-skill-view-9608729a"
    assert desc["models"]["currentModelId"] == "gemini-4-argon-medium"
    mode_ids = {m["id"] for m in desc["modes"]["availableModes"]}
    assert "kirocrew-skill-view-9608729a" in mode_ids
    assert "kirocrew" in mode_ids
    assert "kirocrew-lite" in mode_ids
    assert "custom-reviewer" in mode_ids

    sid = desc["sessionId"]
    assert await mgr.set_mode(session_id=sid, mode_id="kirocrew-lite") == {}
    loaded = await mgr.load_session(session_id=sid, cwd=str(tmp_path))
    assert loaded["modes"]["currentModeId"] == "kirocrew-lite"

    assert await mgr.set_mode(session_id=sid, mode_id="custom-reviewer") == {}
    assert mgr.get_session(sid)["agent"] == "custom-reviewer"
