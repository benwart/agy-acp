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


async def test_skill_view_agent_mcp_sync_and_pinned_model_on_set_mode(tmp_path: Path) -> None:
    """Verify default_agent and set_mode sync agent-spec MCP servers and pinned model."""
    import json

    agents_dir = tmp_path / ".kiro" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "kirocrew-skill-view-111.json").write_text(
        json.dumps(
            {
                "name": "kirocrew-skill-view-111",
                "mcpServers": {
                    "crew-core": {"command": "echo", "args": ["core"]},
                    "legacy-srv": {"command": "echo", "args": ["legacy"]},
                },
            }
        ),
        encoding="utf-8",
    )
    (agents_dir / "kirocrew-skill-view-222.json").write_text(
        json.dumps(
            {
                "name": "kirocrew-skill-view-222",
                "model": "gemini-3.8-flash",
                "mcpServers": {
                    "crew-core": {"command": "echo", "args": ["core"]},
                    "skill-extra": {"command": "echo", "args": ["extra"]},
                    "legacy-srv": {"command": "echo", "args": ["legacy"], "disabled": True},
                },
            }
        ),
        encoding="utf-8",
    )

    mgr = SessionManager(
        save_dir=tmp_path,
        default_agent="kirocrew-skill-view-111",
    )
    desc = await mgr.create_session(cwd=str(tmp_path), mcp_servers=[])
    sid = desc["sessionId"]

    mcp_cfg = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    assert "crew-core" in mcp_cfg["mcpServers"]
    assert "legacy-srv" in mcp_cfg["mcpServers"]
    assert [s["name"] for s in mgr.get_session(sid)["mcp_servers"]] == ["crew-core", "legacy-srv"]

    await mgr.set_mode(session_id=sid, mode_id="kirocrew-skill-view-222")
    session_after = mgr.get_session(sid)
    assert session_after["agent"] == "kirocrew-skill-view-222"
    assert session_after["model"] == "gemini-3.8-flash"
    assert session_after.get("switchedModel") == "gemini-3.8-flash"
    assert [s["name"] for s in session_after["mcp_servers"]] == ["crew-core", "skill-extra"]

    mcp_cfg_after = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    assert "crew-core" in mcp_cfg_after["mcpServers"]
    assert "skill-extra" in mcp_cfg_after["mcpServers"]
    assert "legacy-srv" not in mcp_cfg_after["mcpServers"]


async def test_format_steer_message_and_markdown_local_images(tmp_path: Path) -> None:
    """Verify markdown image links in prompts/steers are staged to temp dir with view_file hint."""
    from agy_acp.logic.session_manager import _format_prompt_blocks, _format_steer_message

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True)
    img_file = uploads_dir / "4bd8ebef_pasted-image.png"
    png_bytes = b"\x89PNG\r\n\x1a\nfake-png-payload"
    img_file.write_bytes(png_bytes)

    raw_steer = (
        f"<user_message>\n![image]({img_file})\n\nHere are the errors from the IDE\n</user_message>"
    )
    formatted, staged_files = _format_steer_message(raw_steer)
    try:
        assert "<user_message>" not in formatted
        assert "</user_message>" not in formatted
        assert "Here are the errors from the IDE" in formatted
        assert len(staged_files) == 1
        staged_path = staged_files[0]
        assert staged_path.is_file()
        assert staged_path.read_bytes() == png_bytes
        assert (staged_path.stat().st_mode & 0o777) == 0o600
        assert (
            f"[User attached an image: {staged_path} — inspect it with view_file if needed]"
            in formatted
        )
    finally:
        for f in staged_files:
            f.unlink(missing_ok=True)

    # Also verify _format_prompt_blocks extracts markdown local images from text blocks
    prompt_text, prompt_files = _format_prompt_blocks(
        [{"type": "text", "text": f"Check this screenshot: ![screenshot]({img_file})"}]
    )
    try:
        assert len(prompt_files) == 1
        assert prompt_files[0].read_bytes() == png_bytes
        assert (
            f"[User attached an image: {prompt_files[0]} — inspect it with view_file if needed]"
            in prompt_text
        )
    finally:
        for f in prompt_files:
            f.unlink(missing_ok=True)


async def test_prompt_isolates_per_session_mcp_token_and_injects_spawn_run_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SessionManager.prompt isolates KIROCREW_STUB_SESSION_TOKEN and steers to spawn_run."""
    import json
    import stat

    monkeypatch.setenv("KIROCREW_STUB_SESSION_TOKEN", "leaked-parent-token")
    monkeypatch.setenv("KIROCREW_ROUTE_SPAWN_RUN", "1")

    fake_agy = tmp_path / "agy"
    fake_agy.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "prompt = sys.argv[sys.argv.index('-p') + 1]\n"
        "resp = json.dumps({\n"
        "    'tok': os.environ.get('KIROCREW_STUB_SESSION_TOKEN'),\n"
        "    'route': os.environ.get('KIROCREW_ROUTE_SPAWN_RUN'),\n"
        "    'prompt': prompt,\n"
        "    'args': sys.argv[1:],\n"
        "})\n"
        "print(json.dumps({'event': 'init', 'init': {'conversation_id': 'conv-kc-1'}}))\n"
        "print(json.dumps({'event': 'result', 'result': {"
        "'conversation_id': 'conv-kc-1', 'status': 'SUCCESS', 'response': resp}}))\n",
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)

    agents_dir = tmp_path / ".kiro" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "egm--zoom-out.json").write_text(
        json.dumps(
            {
                "name": "zoom-out",
                "model": "auto",
                "prompt": "Read zoom_out.md strictly.",
                "tools": ["fs_read", "grep", "glob", "code"],
                "mcpServers": {},
            }
        ),
        encoding="utf-8",
    )

    from agy_acp.providers.agy_cli_provider import AgyCliProvider

    cli = AgyCliProvider(bin_path=fake_agy, save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, default_model="gemini-4-argon-medium")

    parent = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[
            {
                "name": "kirocrew-core",
                "command": "kirocrew",
                "args": ["mcp-core"],
                "env": [{"name": "KIROCREW_STUB_SESSION_TOKEN", "value": "tok-parent-aaa"}],
            }
        ],
    )
    child = await mgr.create_session(cwd=str(tmp_path), mcp_servers=[])
    await mgr.set_mode(session_id=child["sessionId"], mode_id="egm--zoom-out")

    parent_res = await mgr.prompt(
        session_id=parent["sessionId"],
        prompt_blocks=[{"type": "text", "text": "Run drift review"}],
    )
    parent_payload = json.loads(parent_res["response"])
    assert parent_payload["tok"] == "tok-parent-aaa"
    assert parent_payload["route"] == "1"
    assert "spawn_run" in parent_payload["prompt"]
    assert "invoke_subagent" in parent_payload["prompt"]

    child_res = await mgr.prompt(
        session_id=child["sessionId"],
        prompt_blocks=[{"type": "text", "text": "Audit architecture"}],
    )
    child_payload = json.loads(child_res["response"])
    assert child_payload["tok"] == ""
    assert child_payload["route"] == ""
    assert "Read zoom_out.md strictly." in child_payload["prompt"]
    assert "--mode" in child_payload["args"]
    assert child_payload["args"][child_payload["args"].index("--mode") + 1] == "plan"
    assert "auto" not in [
        child_payload["args"][i + 1]
        for i, flag in enumerate(child_payload["args"][:-1])
        if flag == "--model"
    ]
