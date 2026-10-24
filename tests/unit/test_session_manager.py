"""Unit tests for SessionManager lifecycle, multiplexing, mode/model updates, and cancellation."""

from __future__ import annotations

from pathlib import Path

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


@pytest.mark.not_implemented
async def test_load_session_resumes_trajectory(tmp_path: Path) -> None:
    """Verify SessionManager.load_session restores an existing sessionId."""
    mgr = SessionManager(save_dir=tmp_path)
    created = await mgr.create_session(cwd=str(tmp_path))
    loaded = await mgr.load_session(session_id=created["sessionId"], cwd=str(tmp_path))
    assert len(loaded["sessionId"]) >= 32
    assert loaded["sessionId"] == created["sessionId"]


@pytest.mark.not_implemented
async def test_multiplexed_session_isolation(tmp_path: Path) -> None:
    """Verify SessionManager isolates state between concurrent sessionIds."""
    mgr = SessionManager(save_dir=tmp_path)
    s1 = await mgr.create_session(cwd=str(tmp_path))
    s2 = await mgr.create_session(cwd=str(tmp_path))
    assert s1["sessionId"] != s2["sessionId"]


@pytest.mark.not_implemented
async def test_update_session_mode_and_model(tmp_path: Path) -> None:
    """Verify SessionManager.set_mode, set_model, and set_config_option."""
    mgr = SessionManager(save_dir=tmp_path)
    s1 = await mgr.create_session(cwd=str(tmp_path))
    sid = s1["sessionId"]
    await mgr.set_mode(session_id=sid, mode_id="read-only")
    await mgr.set_model(session_id=sid, model_id="gemini-3.8-flash")
    cfg = await mgr.set_config_option(session_id=sid, config_id="reasoning_effort", value="high")
    assert "configOptions" in cfg


@pytest.mark.not_implemented
async def test_cancel_and_steer_active_turn(tmp_path: Path) -> None:
    """Verify SessionManager.cancel stops an active turn."""
    mgr = SessionManager(save_dir=tmp_path)
    s1 = await mgr.create_session(cwd=str(tmp_path))
    cancelled = await mgr.cancel(session_id=s1["sessionId"])
    assert cancelled is True


@pytest.mark.not_implemented
async def test_terminate_and_delete_session(tmp_path: Path) -> None:
    """Verify SessionManager.close_session cleans up session state."""
    mgr = SessionManager(save_dir=tmp_path)
    s1 = await mgr.create_session(cwd=str(tmp_path))
    res = await mgr.close_session(session_id=s1["sessionId"])
    assert res == {}
