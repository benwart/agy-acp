"""Unit tests for SessionManager lifecycle, multiplexing, mode/model updates, and cancellation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    """Verify default_agent and set_mode sync user-scope agent MCP servers and pinned model."""
    import json

    # Untrusted project-scope .kiro/agents MUST NOT inject mcpServers
    proj_agents_dir = tmp_path / ".kiro" / "agents"
    proj_agents_dir.mkdir(parents=True)
    (proj_agents_dir / "untrusted-proj.json").write_text(
        json.dumps(
            {
                "name": "untrusted-proj",
                "mcpServers": {"evil-proj-mcp": {"command": "echo", "args": ["pwn"]}},
            }
        ),
        encoding="utf-8",
    )

    # Trusted user-scope .kiro_user_agents supplies mcpServers cleanly
    agents_dir = tmp_path / ".kiro_user_agents"
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
    assert "evil-proj-mcp" not in mcp_cfg["mcpServers"]
    assert [s["name"] for s in mgr.get_session(sid)["mcp_servers"]] == ["crew-core", "legacy-srv"]

    await mgr.set_mode(session_id=sid, mode_id="untrusted-proj")
    assert mgr.get_session(sid)["mcp_servers"] == []

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
    """Verify markdown image links inside cwd are staged to 0700 dir while escapes are blocked."""
    from agy_acp.logic.session_manager import _format_prompt_blocks, _format_steer_message

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True)
    img_file = uploads_dir / "4bd8ebef_pasted-image.png"
    png_bytes = b"\x89PNG\r\n\x1a\nfake-png-payload"
    img_file.write_bytes(png_bytes)

    raw_steer = (
        f"<user_message>\n![image]({img_file})\n\nHere are the errors from the IDE\n</user_message>"
    )
    priv_img_dir = tmp_path / "priv_images"
    formatted, staged_files = _format_steer_message(
        raw_steer,
        cwd=str(tmp_path),
        image_dir=priv_img_dir,
    )
    try:
        assert "<user_message>" not in formatted
        assert "</user_message>" not in formatted
        assert "Here are the errors from the IDE" in formatted
        assert len(staged_files) == 1
        staged_path = staged_files[0]
        assert staged_path.is_file()
        assert staged_path.read_bytes() == png_bytes
        assert (priv_img_dir.stat().st_mode & 0o777) == 0o700
        assert (staged_path.stat().st_mode & 0o777) == 0o600
        assert (
            f"[User attached an image: {staged_path} — inspect it with view_file if needed]"
            in formatted
        )
    finally:
        for f in staged_files:
            f.unlink(missing_ok=True)

    # Also verify _format_prompt_blocks extracts markdown local images inside cwd
    prompt_text, prompt_files = _format_prompt_blocks(
        [{"type": "text", "text": f"Check this screenshot: ![screenshot]({img_file})"}],
        cwd=str(tmp_path),
        image_dir=priv_img_dir,
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

    # Markdown image outside cwd or without cwd MUST NOT be staged
    outside_img = tmp_path.parent / f"outside_{tmp_path.name}.png"
    outside_img.write_bytes(png_bytes)
    try:
        _, no_cwd_files = _format_prompt_blocks(
            [{"type": "text", "text": f"![x]({img_file})"}],
            image_dir=priv_img_dir,
        )
        assert no_cwd_files == []
        _, escaped_files = _format_prompt_blocks(
            [{"type": "text", "text": f"![x]({outside_img})"}],
            cwd=str(tmp_path),
            image_dir=priv_img_dir,
        )
        assert escaped_files == []
    finally:
        outside_img.unlink(missing_ok=True)


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


async def test_kas_v3_custom_agents_modes_effort_and_spawn_run_routing(tmp_path: Path) -> None:
    """Verify V3KasWireAdapter customAgents origin, MCP merge, effortLevel, and spawn_run."""
    from agy_acp.logic.session_manager import (
        _extract_session_mcp_env,
        _session_has_kirocrew_spawn_run,
    )
    from agy_acp.logic.wire import V3KasWireAdapter
    from agy_acp.providers.agy_cli_provider import PER_SESSION_MCP_ENV_KEYS

    assert "KIROCREW_PORT" in PER_SESSION_MCP_ENV_KEYS
    assert "KIROCREW_HOME" in PER_SESSION_MCP_ENV_KEYS

    agents_dir = tmp_path / ".kiro" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "md-helper.md").write_text(
        "---\nname: MD Helper\ndescription: Markdown disk spec\ntools: read_file\n---\n"
        "Follow markdown spec instructions.\n",
        encoding="utf-8",
    )

    mgr = SessionManager(save_dir=tmp_path, wire_adapter=V3KasWireAdapter())
    wire_meta = {
        "kiro": {
            "customAgents": [
                {
                    "id": "crew-kiro_default",
                    "name": "kiro_default",
                    "description": "Client projected default agent",
                    "prompt": "You are Crew Default.",
                    "model": "gemini-3.8-flash",
                    "tools": ["*"],
                    "excludedTools": [],
                    "permissions": {
                        "rules": [
                            {
                                "capability": "mcp",
                                "match": ["kirocrew-core/*"],
                                "effect": "allow",
                            }
                        ]
                    },
                    "mcpServers": {
                        "kirocrew-core": {
                            "command": "agent-overridden-cmd",
                            "args": ["--wrong"],
                        },
                        "agent-stdio": {
                            "command": "python3",
                            "args": ["-m", "kas_env_shim", "--session", "s1"],
                            "env": {"KIROCREW_HOME": "/crew/home/s1"},
                        },
                    },
                }
            ]
        }
    }
    hoisted_mcp = [
        {
            "name": "kirocrew-core",
            "command": "python3",
            "args": ["-m", "kiro_crew.mcp_core"],
            "env": [
                {"name": "KIROCREW_PORT", "value": "9123"},
                {"name": "KIROCREW_SESSION_KEY", "value": "skey-1"},
                {"name": "KIROCREW_STUB_SESSION_TOKEN", "value": "tok-1"},
            ],
        }
    ]

    desc = await mgr.create_session(cwd=str(tmp_path), mcp_servers=hoisted_mcp, meta=wire_meta)
    sid = desc["sessionId"]

    modes_by_id = {m["id"]: m for m in desc["modes"]["availableModes"]}
    assert (
        modes_by_id["crew-kiro_default"]["_meta"]["kiro"]["resource"]["source"]["origin"]
        == "client"
    )
    assert modes_by_id["md-helper"]["_meta"]["kiro"]["resource"]["source"]["origin"] == "project"

    cfg_by_id = {o["id"]: o for o in desc["configOptions"]}
    assert "effortLevel" in cfg_by_id
    assert "reasoning_effort" not in cfg_by_id
    assert [x["value"] for x in cfg_by_id["effortLevel"]["options"]] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]

    # Switch mode to client customAgent -> merges MCP (session overrides agent), sets model/prompt
    await mgr.set_mode(session_id=sid, mode_id="crew-kiro_default")
    state = mgr.get_session(sid)
    assert state["agent"] == "crew-kiro_default"
    assert state["agentPrompt"] == "You are Crew Default."
    assert state["agentTools"] == ["*"]
    assert state["agentPermissions"] == {
        "rules": [{"capability": "mcp", "match": ["kirocrew-core/*"], "effect": "allow"}]
    }
    assert state["model"] == "gemini-3.8-flash"

    mcp_by_name = {s["name"]: s for s in state["mcp_servers"]}
    assert mcp_by_name["kirocrew-core"]["command"] == "python3"
    assert mcp_by_name["kirocrew-core"]["args"] == ["-m", "kiro_crew.mcp_core"]
    assert "agent-stdio" in mcp_by_name

    # tools=["*"] with hoisted kirocrew-core must enable spawn_run routing!
    assert _session_has_kirocrew_spawn_run(state) is True
    extracted_env = _extract_session_mcp_env(state)
    assert extracted_env["KIROCREW_PORT"] == "9123"
    assert extracted_env["KIROCREW_SESSION_KEY"] == "skey-1"
    assert extracted_env["KIROCREW_STUB_SESSION_TOKEN"] == "tok-1"
    assert extracted_env["KIROCREW_HOME"] == "/crew/home/s1"
    assert extracted_env["KIROCREW_ROUTE_SPAWN_RUN"] == "1"

    # Excluding spawn_run disables spawn_run routing
    excluded_state = {**state, "agentExcludedTools": ["@kirocrew-core/spawn_run"]}
    assert _session_has_kirocrew_spawn_run(excluded_state) is False

    # Pinning model explicitly prevents subsequent set_mode from clobbering user's choice
    cfg_res = await mgr.set_config_option(
        session_id=sid,
        config_id="model",
        value="claude-sonnet-4-6",
    )
    assert {o["id"]: o["currentValue"] for o in cfg_res["configOptions"]}["model"] == (
        "claude-sonnet-4-6"
    )
    effort_res = await mgr.set_config_option(
        session_id=sid,
        config_id="effortLevel",
        value="xhigh",
    )
    assert {o["id"]: o["currentValue"] for o in effort_res["configOptions"]}["effortLevel"] == (
        "xhigh"
    )
    with pytest.raises(ValueError, match="effort"):
        await mgr.set_config_option(session_id=sid, config_id="effortLevel", value="turbo")

    await mgr.set_mode(session_id=sid, mode_id="md-helper")
    await mgr.set_mode(session_id=sid, mode_id="crew-kiro_default")
    assert mgr.get_session(sid)["model"] == "claude-sonnet-4-6"


async def test_evict_vs_delete_session_and_load_with_updated_mcp_and_turns(tmp_path: Path) -> None:
    """Verify evict_session preserves turns & state for session/load, while delete removes both."""
    from agy_acp.logic.wire import V3KasWireAdapter

    mgr = SessionManager(save_dir=tmp_path, wire_adapter=V3KasWireAdapter())
    created = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[{"name": "srv-initial", "command": "echo", "args": ["1"]}],
    )
    sid = created["sessionId"]
    mgr.state_store.append_session_turn(
        session_id=sid,
        user_message_id="msg_001",
        prompt_blocks=[{"type": "text", "text": "Hello"}],
        assistant_text="World",
    )

    await mgr.evict_session(session_id=sid)
    assert len(mgr.state_store.get_session_turns(sid)) == 1

    updated_meta = {
        "kiro": {
            "customAgents": [
                {
                    "id": "kirocrew",
                    "name": "KiroCrew",
                    "prompt": "Updated prompt",
                    "tools": ["*"],
                }
            ]
        }
    }
    reloaded = await mgr.load_session(
        session_id=sid,
        cwd=str(tmp_path / "new-cwd"),
        mcp_servers=[{"name": "srv-reinjected", "command": "echo", "args": ["2"]}],
        meta=updated_meta,
    )
    assert reloaded["sessionId"] == sid
    reloaded_state = mgr.get_session(sid)
    assert reloaded_state["cwd"] == str(tmp_path / "new-cwd")
    assert [s["name"] for s in reloaded_state["mcp_servers"]] == ["srv-reinjected"]
    reloaded_modes = {m["id"]: m for m in reloaded["modes"]["availableModes"]}
    assert reloaded_modes["kirocrew"]["_meta"]["kiro"]["resource"]["source"]["origin"] == "client"
    assert len(mgr.state_store.get_session_turns(sid)) == 1

    await mgr.terminate_session(session_id=sid, delete_persisted=True)
    assert mgr.state_store.get_session(sid) is None
    assert mgr.state_store.get_session_turns(sid) == []


async def test_concurrent_sessions_mcp_shim_isolation_and_prober(tmp_path: Path) -> None:
    """Verify concurrent sessions isolate conflicting MCP server args and probe session MCPs."""
    import json
    import sys

    from agy_acp.logic.wire import V3KasWireAdapter

    mcp_stub = tmp_path / "mcp_echo_stub.py"
    mcp_stub.write_text(
        "import json, os, sys\n"
        "session_arg = sys.argv[sys.argv.index('--session') + 1]\n"
        "port = os.environ.get('KIROCREW_PORT', '')\n"
        "home = os.environ.get('KIROCREW_HOME', '')\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    mid = msg.get('id')\n"
        "    method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {"
        "'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}}, "
        "'serverInfo': {'name': 'stub', 'version': '1.0'}}}), flush=True)\n"
        "    elif method == 'tools/list':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {'tools': ["
        "{'name': f'tool_{session_arg}_{port}', 'description': home}]}}), flush=True)\n",
        encoding="utf-8",
    )

    mgr = SessionManager(save_dir=tmp_path, wire_adapter=V3KasWireAdapter())
    s_a = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[
            {
                "name": "shared-shim",
                "command": sys.executable,
                "args": [str(mcp_stub), "--session", "sessA"],
                "env": [
                    {"name": "KIROCREW_PORT", "value": "7001"},
                    {"name": "KIROCREW_HOME", "value": "/home/a"},
                ],
            }
        ],
    )
    s_b = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[
            {
                "name": "shared-shim",
                "command": sys.executable,
                "args": [str(mcp_stub), "--session", "sessB"],
                "env": [
                    {"name": "KIROCREW_PORT", "value": "7002"},
                    {"name": "KIROCREW_HOME", "value": "/home/b"},
                ],
            }
        ],
    )

    statuses_a = await mgr.probe_session_mcp_servers(s_a["sessionId"])
    statuses_b = await mgr.probe_session_mcp_servers(s_b["sessionId"])
    assert len(statuses_a) == 1 and statuses_a[0].status == "connected"
    assert [t.name for t in statuses_a[0].tools] == ["tool_sessA_7001"]
    assert statuses_a[0].tools[0].description == "/home/a"

    assert len(statuses_b) == 1 and statuses_b[0].status == "connected"
    assert [t.name for t in statuses_b[0].tools] == ["tool_sessB_7002"]
    assert statuses_b[0].tools[0].description == "/home/b"

    # Also verify mcp_config.json uses mux wrapper when conflicting args exist across sessions
    mcp_cfg = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    mux_entry = mcp_cfg["mcpServers"]["shared-shim"]
    assert "mcp_session_mux" in " ".join(mux_entry["args"])


async def test_wave5b_session_manager_security_and_lifecycle(tmp_path: Path) -> None:
    """Verify Wave 5B SessionManager fixes: tools=[], unshadowing, locks, mid-turn, delete MCP."""
    import json

    from agy_acp.logic.wire import V3KasWireAdapter
    from agy_acp.providers.agy_cli_provider import AgyCliProvider, PromptOutcome

    turn_started = anyio.Event()
    allow_finish = anyio.Event()
    observed_skip_perms: list[bool | None] = []
    active_count = 0
    max_concurrent = 0

    class _LockStubCli(AgyCliProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            nonlocal active_count, max_concurrent
            observed_skip_perms.append(kwargs.get("skip_permissions"))
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
            turn_started.set()
            await allow_finish.wait()
            active_count -= 1
            return PromptOutcome(
                conversation_id="conv-locked-1",
                stop_reason="end_turn",
                response="ok",
                usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )

    cli = _LockStubCli(save_dir=tmp_path)
    mgr = SessionManager(
        save_dir=tmp_path,
        cli_provider=cli,
        wire_adapter=V3KasWireAdapter(),
        default_agent="fallback-bundled",
    )

    # 1. CRITICAL-02: prompt() on unknown sessionId raises KeyError
    with pytest.raises(KeyError):
        await mgr.prompt(
            session_id="non-existent-session",
            prompt_blocks=[{"type": "text", "text": "hi"}],
        )

    # 2. Custom agent named "auto" is NOT shadowed; tools=[] sets agentReadOnly=True;
    #    switching to virtual agent with spec=None clears stale agentPermissions/agentAllowedTools
    meta = {
        "kiro": {
            "customAgents": [
                {
                    "id": "auto",
                    "name": "Custom Auto Agent",
                    "prompt": "You are custom auto.",
                    "tools": [],
                    "allowedTools": ["@srv/t"],
                    "permissions": {"rules": [{"capability": "fs_read", "effect": "allow"}]},
                }
            ]
        }
    }
    s1 = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[
            {"name": "exclusive-s1", "command": "echo", "args": ["1"]},
            {"name": "shared-srv", "command": "echo", "args": ["shared"]},
        ],
        meta=meta,
    )
    s2 = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[{"name": "shared-srv", "command": "echo", "args": ["shared"]}],
    )
    sid1 = s1["sessionId"]
    sid2 = s2["sessionId"]

    await mgr.set_mode(session_id=sid1, mode_id="auto")
    st_auto = mgr.get_session(sid1)
    assert st_auto["agent"] == "auto"
    assert st_auto["agentPrompt"] == "You are custom auto."
    assert st_auto["agentTools"] == []
    assert st_auto["agentReadOnly"] is True
    assert st_auto["agentPermissions"] is not None

    # Switching to virtual default_agent (spec is None) clears stale permissions/tools/prompt
    await mgr.set_mode(session_id=sid1, mode_id="fallback-bundled")
    st_cleared = mgr.get_session(sid1)
    assert st_cleared["agent"] == "fallback-bundled"
    assert st_cleared["agentPrompt"] is None
    assert st_cleared["agentTools"] is None
    assert st_cleared["agentAllowedTools"] == []
    assert st_cleared["agentPermissions"] is None
    assert st_cleared["agentReadOnly"] is False

    # Re-activate tools=[] custom agent and verify skip_permissions=False, empty steer rejected,
    # per-session lock serialization, and mid-turn set_model preservation
    await mgr.set_mode(session_id=sid1, mode_id="auto")

    async def _turn_a() -> None:
        await mgr.prompt(session_id=sid1, prompt_blocks=[{"type": "text", "text": "t1"}])

    async def _turn_b() -> None:
        await turn_started.wait()
        await mgr.prompt(session_id=sid1, prompt_blocks=[{"type": "text", "text": "t2"}])

    cli._active_turns.add(sid1)
    assert await mgr.steer(session_id=sid1, message="<user_message>   \n  </user_message>") is False
    cli._active_turns.discard(sid1)

    with anyio.fail_after(5.0):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_turn_a)
            tg.start_soon(_turn_b)
            await turn_started.wait()
            await anyio.sleep(0.02)
            # Mutate model mid-turn; prompt completion must NOT clobber it!
            await mgr.set_model(session_id=sid1, model_id="claude-sonnet-4-6")
            allow_finish.set()

    assert max_concurrent == 1
    assert observed_skip_perms == [False, False]
    after_turns = mgr.get_session(sid1)
    assert after_turns["model"] == "claude-sonnet-4-6"
    assert after_turns["agyConversationId"] == "conv-locked-1"

    # 3. Mid-turn session deletion MUST NOT resurrect the deleted session row
    turn_started = anyio.Event()
    allow_finish = anyio.Event()

    async def _turn_deleted() -> None:
        await mgr.prompt(session_id=sid2, prompt_blocks=[{"type": "text", "text": "bye"}])

    async with anyio.create_task_group() as tg:
        tg.start_soon(_turn_deleted)
        await turn_started.wait()
        mgr.state_store.delete_session(sid2)
        allow_finish.set()

    assert mgr.state_store.get_session(sid2) is None

    # 4. close_session(delete_persisted=True) removes exclusive MCP servers from mcp_config.json
    # Re-create s2 holding shared-srv, then delete s1: exclusive-s1 removed, shared-srv retained
    s2_new = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[{"name": "shared-srv", "command": "echo", "args": ["shared"]}],
    )
    _ = s2_new
    await mgr.close_session(session_id=sid1, delete_persisted=True)
    mcp_cfg = json.loads((tmp_path / "mcp_config.json").read_text(encoding="utf-8"))
    assert "exclusive-s1" not in mcp_cfg["mcpServers"]
    assert "shared-srv" in mcp_cfg["mcpServers"]


async def test_effort_option_and_compact_sync_effort_suffixed_models(tmp_path: Path) -> None:
    """Verify effortLevel updates and compaction resolve sibling effort-suffixed models."""
    from agy_acp.providers.agy_cli_provider import AgyCliProvider, PromptOutcome

    seen_calls: list[tuple[str | None, str | None]] = []

    class _RecordingCli(AgyCliProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            seen_calls.append((kwargs.get("model"), kwargs.get("effort")))
            return PromptOutcome(
                conversation_id="conv-eff",
                stop_reason="end_turn",
                response="Compacted ok.",
                usage={"input_tokens": 10, "output_tokens": 5},
            )

    cli = _RecordingCli(save_dir=tmp_path)
    cli.state_store.save_models(
        [
            {"modelId": "gemini-4-argon-medium", "name": "Gemini 4 Argon (Medium)"},
            {"modelId": "gemini-4-argon-high", "name": "Gemini 4 Argon (High)"},
            {"modelId": "gemini-4-argon-low", "name": "Gemini 4 Argon (Low)"},
        ]
    )
    mgr = SessionManager(cli_provider=cli)
    created = await mgr.create_session(cwd=str(tmp_path))
    sid = created["sessionId"]
    assert mgr.get_session(sid)["model"] == "gemini-4-argon-medium"

    cfg_res = await mgr.set_config_option(
        session_id=sid,
        config_id="effortLevel",
        value="high",
    )
    assert mgr.get_session(sid)["model"] == "gemini-4-argon-high"
    model_opt = next(opt for opt in cfg_res["configOptions"] if opt["id"] == "model")
    assert model_opt["currentValue"] == "gemini-4-argon-high"

    await mgr.prompt(session_id=sid, prompt_blocks=[{"type": "text", "text": "hello"}])
    await mgr.compact_session(session_id=sid)
    assert seen_calls == [
        ("gemini-4-argon-high", "high"),
        ("gemini-4-argon-low", "low"),
    ]


async def test_compact_session_and_prompt_propagate_outcome_error(tmp_path: Path) -> None:
    """Verify SessionManager propagates PromptOutcome.error for prompt() and compact_session()."""
    from agy_acp.providers.agy_cli_provider import AgyCliProvider, PromptOutcome

    fail_next = False

    class _ErrStubCli(AgyCliProvider):
        async def stream_prompt(self, **_kwargs: Any) -> PromptOutcome:
            if fail_next:
                return PromptOutcome(
                    conversation_id="conv-keep-1",
                    stop_reason="end_turn",
                    response="",
                    usage={"input_tokens": 4, "output_tokens": 0, "total_tokens": 4},
                    error="You have exhausted your capacity on this model.",
                )
            return PromptOutcome(
                conversation_id="conv-keep-1",
                stop_reason="end_turn",
                response="Initial reply.",
                usage={"input_tokens": 5, "output_tokens": 3, "context_tokens": 120},
            )

    cli = _ErrStubCli(save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli)
    created = await mgr.create_session(cwd=str(tmp_path))
    sid = created["sessionId"]

    ok_res = await mgr.prompt(session_id=sid, prompt_blocks=[{"type": "text", "text": "hi"}])
    assert "error" not in ok_res
    assert mgr.get_session(sid)["agyConversationId"] == "conv-keep-1"

    fail_next = True
    err_res = await mgr.prompt(session_id=sid, prompt_blocks=[{"type": "text", "text": "again"}])
    assert err_res["error"] == "You have exhausted your capacity on this model."
    assert mgr.get_session(sid)["agyConversationId"] == "conv-keep-1"

    compact_res = await mgr.compact_session(session_id=sid)
    assert compact_res["status"] == "failed"
    assert compact_res["summary"] == "You have exhausted your capacity on this model."
    assert mgr.get_session(sid)["agyConversationId"] == "conv-keep-1"
