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

    closed_backend_ids: list[str] = []

    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider

    class _CloseTrackingCli(AgyAcpClientProvider):
        async def close_backend_session(self, agy_session_id: str) -> dict[str, Any]:
            closed_backend_ids.append(agy_session_id)
            return {}

    cli = _CloseTrackingCli(save_dir=tmp_path)
    mgr = SessionManager(
        save_dir=tmp_path,
        cli_provider=cli,
        default_agent="kirocrew-skill-view-111",
    )
    desc = await mgr.create_session(cwd=str(tmp_path), mcp_servers=[])
    sid = desc["sessionId"]

    assert not (tmp_path / "mcp_config.json").exists()
    assert [s["name"] for s in mgr.get_session(sid)["mcp_servers"]] == ["crew-core", "legacy-srv"]

    # Simulate active backend session + persisted turn before switching custom agent
    seeded = mgr.get_session(sid)
    seeded["agyConversationId"] = "agy-sess-111"
    mgr.state_store.upsert_session(seeded)
    mgr.state_store.append_session_turn(
        session_id=sid,
        user_message_id="msg-before-switch",
        prompt_blocks=[{"type": "text", "text": "Prior turn question"}],
        assistant_text="Prior turn answer",
    )

    await mgr.set_mode(session_id=sid, mode_id="untrusted-proj")
    after_untrusted = mgr.get_session(sid)
    assert after_untrusted["mcp_servers"] == []
    assert after_untrusted["agyConversationId"] is None
    assert closed_backend_ids == ["agy-sess-111"]
    assert "Prior turn question" in (after_untrusted.get("compactedSummary") or "")

    await mgr.set_mode(session_id=sid, mode_id="kirocrew-skill-view-222")
    session_after = mgr.get_session(sid)
    assert session_after["agent"] == "kirocrew-skill-view-222"
    assert session_after["model"] == "gemini-3.8-flash"
    assert session_after.get("switchedModel") == "gemini-3.8-flash"
    assert [s["name"] for s in session_after["mcp_servers"]] == ["crew-core", "skill-extra"]
    assert not (tmp_path / "mcp_config.json").exists()


async def test_format_steer_message_and_markdown_local_images(tmp_path: Path) -> None:
    """Verify markdown image links inside cwd become inline base64 blocks with zero temp files."""
    import base64

    from agy_acp.logic.session_manager import _build_acp_content_blocks, _format_steer_message

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True)
    img_file = uploads_dir / "4bd8ebef_pasted-image.png"
    png_bytes = b"\x89PNG\r\n\x1a\nfake-png-payload"
    img_file.write_bytes(png_bytes)
    expected_b64 = base64.b64encode(png_bytes).decode("ascii")

    raw_steer = (
        f"<user_message>\n![image]({img_file})\n\nHere are the errors from the IDE\n</user_message>"
    )
    formatted, steer_blocks = _format_steer_message(
        raw_steer,
        cwd=str(tmp_path),
    )
    assert "<user_message>" not in formatted
    assert "</user_message>" not in formatted
    assert "Here are the errors from the IDE" in formatted
    assert steer_blocks == [
        {"type": "text", "text": formatted},
        {"type": "image", "data": expected_b64, "mimeType": "image/png"},
    ]
    assert not (tmp_path / "images").exists()

    # Verify _build_acp_content_blocks passes native image/resource blocks and extracts md images
    direct_b64 = base64.b64encode(b"direct-webp").decode("ascii")
    prompt_text, acp_blocks = _build_acp_content_blocks(
        [
            {"type": "text", "text": f"Check this screenshot: ![screenshot]({img_file})"},
            {"type": "image", "data": direct_b64, "mimeType": "image/webp"},
            {"type": "resource", "resource": {"uri": "file:///a.txt", "text": "hello"}},
        ],
        cwd=str(tmp_path),
        prefix_text="<session_context_paths>\nREADME.md\n</session_context_paths>",
    )
    assert prompt_text.startswith("<session_context_paths>")
    assert acp_blocks[0]["type"] == "text"
    assert acp_blocks[0]["text"].startswith("<session_context_paths>")
    assert acp_blocks[1] == {"type": "image", "data": expected_b64, "mimeType": "image/png"}
    assert acp_blocks[2] == {"type": "image", "data": direct_b64, "mimeType": "image/webp"}
    assert acp_blocks[3] == {
        "type": "resource",
        "resource": {"uri": "file:///a.txt", "text": "hello"},
    }
    assert not (tmp_path / "images").exists()

    # Markdown image outside cwd, symlink escaping cwd, or missing cwd MUST NOT be read
    outside_img = tmp_path.parent / f"outside_{tmp_path.name}.png"
    outside_img.write_bytes(png_bytes)
    escaped_symlink = uploads_dir / "escaped_link.png"
    escaped_symlink.symlink_to(outside_img)
    try:
        _, no_cwd_blocks = _build_acp_content_blocks(
            [{"type": "text", "text": f"![x]({img_file})"}],
        )
        assert [b["type"] for b in no_cwd_blocks] == ["text"]
        _, escaped_blocks = _build_acp_content_blocks(
            [{"type": "text", "text": f"![x]({outside_img}) ![y]({escaped_symlink})"}],
            cwd=str(tmp_path),
        )
        assert [b["type"] for b in escaped_blocks] == ["text"]
    finally:
        escaped_symlink.unlink(missing_ok=True)
        outside_img.unlink(missing_ok=True)


async def test_prompt_isolates_per_session_mcp_token_and_injects_spawn_run_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SessionManager.prompt isolates KIROCREW_STUB_SESSION_TOKEN and steers to spawn_run."""
    import json
    import stat

    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider

    monkeypatch.setenv("KIROCREW_STUB_SESSION_TOKEN", "leaked-parent-token")
    monkeypatch.setenv("KIROCREW_ROUTE_SPAWN_RUN", "1")

    fake_agy = tmp_path / "agy"
    fake_agy.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "sessions = {}\n"
        "seq = 0\n"
        "for raw in sys.stdin:\n"
        "    line = raw.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    msg = json.loads(line)\n"
        "    mid = msg.get('id')\n"
        "    method = msg.get('method')\n"
        "    params = msg.get('params') or {}\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {'protocolVersion': 1}}),"
        " flush=True)\n"
        "    elif method == 'session/new':\n"
        "        seq += 1\n"
        "        sid = f'agy-kc-{seq}'\n"
        "        sessions[sid] = {\n"
        "            'mcpServers': params.get('mcpServers', []),\n"
        "            'meta': params.get('_meta', {}),\n"
        "            'configs': {},\n"
        "            'models_set': [],\n"
        "        }\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {\n"
        "            'sessionId': sid,\n"
        "            'models': {'currentModelId': 'gemini-4-argon-medium', 'availableModels': [\n"
        "                {'modelId': 'gemini-4-argon-medium', 'name': 'Gemini 4 Argon'}\n"
        "            ]},\n"
        "            'configOptions': [{'id': 'thought_level', 'options': [\n"
        "                {'value': 'low'}, {'value': 'medium'}, {'value': 'high'}\n"
        "            ]}],\n"
        "        }}), flush=True)\n"
        "    elif method == 'session/set_config_option':\n"
        "        s = sessions.get(params.get('sessionId'), {})\n"
        "        s.setdefault('configs', {})[params.get('configId')] = params.get('value')\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {}}), flush=True)\n"
        "    elif method == 'session/set_model':\n"
        "        s = sessions.get(params.get('sessionId'), {})\n"
        "        s.setdefault('models_set', []).append(params.get('modelId'))\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {}}), flush=True)\n"
        "    elif method == 'session/close':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {}}), flush=True)\n"
        "    elif method == 'session/prompt':\n"
        "        sid = params['sessionId']\n"
        "        s = sessions[sid]\n"
        "        resp = json.dumps({\n"
        "            'procStubToken': os.environ.get('KIROCREW_STUB_SESSION_TOKEN'),\n"
        "            'mcpServers': s['mcpServers'],\n"
        "            'systemPromptAppend': s['meta'].get('antigravity', {})"
        ".get('systemPrompt', {}).get('append', ''),\n"
        "            'configs': s['configs'],\n"
        "            'modelsSet': s['models_set'],\n"
        "            'promptBlocks': params.get('prompt', []),\n"
        "        })\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
        "            'sessionId': sid,\n"
        "            'update': {'sessionUpdate': 'agent_message_chunk',"
        " 'content': {'type': 'text', 'text': resp}},\n"
        "        }}), flush=True)\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': {"
        "'stopReason': 'end_turn'}}), flush=True)\n",
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

    cli = AgyAcpClientProvider(bin_path=fake_agy, save_dir=tmp_path)
    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli, default_model="gemini-4-argon-medium")
    try:
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
        assert parent_payload["procStubToken"] is None
        parent_mcp = parent_payload["mcpServers"][0]
        parent_env = {item["name"]: item["value"] for item in parent_mcp["env"]}
        assert parent_env["KIROCREW_STUB_SESSION_TOKEN"] == "tok-parent-aaa"
        assert "spawn_run" in parent_payload["systemPromptAppend"]
        assert "invoke_subagent" in parent_payload["systemPromptAppend"]
        assert parent_payload["promptBlocks"] == [{"type": "text", "text": "Run drift review"}]

        child_res = await mgr.prompt(
            session_id=child["sessionId"],
            prompt_blocks=[{"type": "text", "text": "Audit architecture"}],
        )
        child_payload = json.loads(child_res["response"])
        assert child_payload["mcpServers"] == []
        assert "Read zoom_out.md strictly." in child_payload["systemPromptAppend"]
        assert "spawn_run" not in child_payload["systemPromptAppend"]
        assert child_payload["configs"]["mode"] == "plan"
        assert "auto" not in child_payload["modelsSet"]
        assert not (tmp_path / "mcp_config.json").exists()
    finally:
        await mgr.close()


async def test_kas_v3_custom_agents_modes_effort_and_spawn_run_routing(tmp_path: Path) -> None:
    """Verify V3KasWireAdapter customAgents origin, MCP merge, effortLevel, and spawn_run."""
    from agy_acp.logic.session_manager import (
        _extract_session_mcp_env,
        _session_has_kirocrew_spawn_run,
    )
    from agy_acp.logic.tool_classifier import PER_SESSION_MCP_ENV_KEYS
    from agy_acp.logic.wire import V3KasWireAdapter

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
    import sys

    from agy_acp.logic.session_manager import _extract_session_mcp_env
    from agy_acp.logic.wire import V3KasWireAdapter
    from agy_acp.providers.agy_acp_client import format_acp_mcp_servers

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

    # Zero global mcp_config.json writes; each session passes isolated ACP mcpServers
    assert not (tmp_path / "mcp_config.json").exists()
    state_a = mgr.get_session(s_a["sessionId"])
    state_b = mgr.get_session(s_b["sessionId"])
    acp_a = format_acp_mcp_servers(
        state_a["mcp_servers"], session_env=_extract_session_mcp_env(state_a)
    )
    acp_b = format_acp_mcp_servers(
        state_b["mcp_servers"], session_env=_extract_session_mcp_env(state_b)
    )
    assert acp_a[0]["args"] == [str(mcp_stub), "--session", "sessA"]
    assert acp_b[0]["args"] == [str(mcp_stub), "--session", "sessB"]


async def test_wave5b_session_manager_security_and_lifecycle(tmp_path: Path) -> None:
    """Verify Wave 5B SessionManager fixes: tools=[], unshadowing, locks, mid-turn, delete MCP."""
    from agy_acp.logic.tool_classifier import PromptOutcome
    from agy_acp.logic.wire import V3KasWireAdapter
    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider

    turn_started = anyio.Event()
    allow_finish = anyio.Event()
    observed_skip_perms: list[bool | None] = []
    closed_agy_ids: list[str] = []
    active_count = 0
    max_concurrent = 0

    class _LockStubCli(AgyAcpClientProvider):
        async def close_backend_session(self, agy_session_id: str) -> dict[str, Any]:
            closed_agy_ids.append(agy_session_id)
            return {}

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

    # 4. close_session(delete_persisted=True) closes backend session while leaving peer untouched
    s2_new = await mgr.create_session(
        cwd=str(tmp_path),
        mcp_servers=[{"name": "shared-srv", "command": "echo", "args": ["shared"]}],
    )
    await mgr.close_session(session_id=sid1, delete_persisted=True)
    assert "conv-locked-1" in closed_agy_ids
    assert not (tmp_path / "mcp_config.json").exists()
    assert [s["name"] for s in mgr.get_session(s2_new["sessionId"])["mcp_servers"]] == [
        "shared-srv"
    ]


async def test_effort_option_and_compact_sync_effort_suffixed_models(tmp_path: Path) -> None:
    """Verify effortLevel updates and compaction resolve sibling effort-suffixed models."""
    from agy_acp.logic.tool_classifier import PromptOutcome
    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider

    seen_calls: list[tuple[str | None, str | None]] = []

    class _RecordingCli(AgyAcpClientProvider):
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
    from agy_acp.logic.tool_classifier import PromptOutcome
    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider

    fail_next = False

    class _ErrStubCli(AgyAcpClientProvider):
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


async def test_session_manager_fork_steer_clear_context_knowledge_code_intel_and_revert(
    tmp_path: Path,
) -> None:
    """Verify 0.12.1 SessionManager fork, steer clear, context, knowledge, code intel, revert."""
    from agy_acp.logic.tool_classifier import PromptOutcome
    from agy_acp.logic.wire import V3KasWireAdapter
    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider

    seen_prompts: list[dict[str, Any]] = []

    class _RecordCli(AgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:
            seen_prompts.append(dict(kwargs))
            return PromptOutcome(
                conversation_id=f"conv-{len(seen_prompts)}",
                stop_reason="end_turn",
                response=f"Reply {len(seen_prompts)}",
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'demo'\n", encoding="utf-8")

    cli = _RecordCli(save_dir=tmp_path / "state")
    mgr = SessionManager(
        save_dir=tmp_path / "state",
        cli_provider=cli,
        wire_adapter=V3KasWireAdapter(),
    )
    created = await mgr.create_session(
        cwd=str(tmp_path),
        meta={
            "kiro": {
                "customAgents": [
                    {
                        "id": "kirocrew",
                        "name": "KiroCrew",
                        "prompt": "Agent instructions",
                        "tools": ["*"],
                    }
                ]
            }
        },
    )
    sid = created["sessionId"]
    await mgr.set_mode(session_id=sid, mode_id="kirocrew")

    # 1. Context paths add/remove/clear and injection in prompt()
    ctx_add1 = mgr.handle_session_context(
        sid,
        subcommand="add",
        path="src/main.py",
    )
    assert ctx_add1["success"] is True
    ctx_add2 = mgr.handle_session_context(
        sid,
        subcommand="add",
        path="README.md",
    )
    assert ctx_add2["success"] is True
    ctx_rem = mgr.handle_session_context(
        sid,
        subcommand="remove",
        path="src/main.py",
    )
    assert ctx_rem["success"] is True
    assert mgr.get_session(sid)["contextPaths"] == ["README.md"]

    await mgr.prompt(
        session_id=sid,
        prompt_blocks=[{"type": "text", "text": "Turn 1"}],
    )
    assert (
        "<session_context_paths>\nREADME.md\n</session_context_paths>" in seen_prompts[0]["prompt"]
    )
    mgr.state_store.append_session_turn(
        session_id=sid,
        user_message_id="msg-1",
        prompt_blocks=[{"type": "text", "text": "Turn 1"}],
        assistant_text="Reply 1",
    )
    mgr.state_store.append_session_turn(
        session_id=sid,
        user_message_id="msg-2",
        prompt_blocks=[{"type": "text", "text": "Turn 2"}],
        assistant_text="Reply 2",
    )
    mgr.state_store.append_session_turn(
        session_id=sid,
        user_message_id="msg-3",
        prompt_blocks=[{"type": "text", "text": "Turn 3"}],
        assistant_text="Reply 3",
    )

    # 2. Fork session bounded by message_id="msg-1" (copies turns up to msg-1 inclusive)
    fork_desc = await mgr.fork_session(source_session_id=sid, message_id="msg-1")
    fork_sid = fork_desc["sessionId"]
    assert fork_sid != sid
    fork_turns = mgr.state_store.get_session_turns(fork_sid)
    assert len(fork_turns) == 1
    assert fork_turns[0]["user_message_id"] == "msg-1"
    fork_state = mgr.get_session(fork_sid)
    assert fork_state["agyConversationId"] is None
    assert fork_state["reinjectAgentPrompt"] is True
    assert "Turn 1" in (fork_state["compactedSummary"] or "")
    assert "Turn 2" not in (fork_state["compactedSummary"] or "")
    assert fork_state["contextPaths"] == ["README.md"]

    # Prompting the forked session consumes compactedSummary and supplies system_prompt_append
    await mgr.prompt(
        session_id=fork_sid,
        prompt_blocks=[{"type": "text", "text": "Forked turn"}],
    )
    assert "[Compacted Session Summary]" in seen_prompts[-1]["prompt"]
    assert "[Compacted Session Summary]" in seen_prompts[-1]["content_blocks"][0]["text"]
    assert "Agent instructions" in (seen_prompts[-1]["system_prompt_append"] or "")

    # 3. Revert checkpoint by message_id on original session
    rev_res = mgr.revert_checkpoint_by_message_id(sid, message_id="msg-2")
    assert rev_res == {"success": True}
    assert len(mgr.state_store.get_session_turns(sid)) == 1
    rev_state = mgr.get_session(sid)
    assert rev_state["agyConversationId"] is None
    assert rev_state["reinjectAgentPrompt"] is True
    assert "Turn 1" in (rev_state["compactedSummary"] or "")

    # 4. Steer clear
    cli._active_turns.add(sid)
    await mgr.steer(session_id=sid, message="Pending steer 1")
    assert mgr.clear_session_steers(sid) == ["Pending steer 1"]
    cli._active_turns.discard(sid)

    # 5. Knowledge CRUD
    assert mgr.handle_knowledge(sid, subcommand="show") == {"success": True, "entries": []}
    k_added = mgr.handle_knowledge(
        sid,
        subcommand="add",
        name="Spec",
        path="/docs/spec.md",
        item_count=3,
    )
    assert k_added["success"] is True
    k_show = mgr.handle_knowledge(sid, subcommand="show")
    assert k_show["entries"] == [{"name": "Spec", "path": "/docs/spec.md", "item_count": 3}]
    k_upd = mgr.handle_knowledge(
        sid,
        subcommand="update",
        name="Spec",
        path="/docs/spec-v2.md",
        item_count=4,
    )
    assert k_upd["success"] is True
    assert mgr.handle_knowledge(sid, subcommand="show")["entries"][0]["path"] == "/docs/spec-v2.md"
    k_rem = mgr.handle_knowledge(sid, subcommand="remove", name="Spec")
    assert k_rem["success"] is True
    assert mgr.handle_knowledge(sid, subcommand="show") == {"success": True, "entries": []}
    mgr.handle_knowledge(sid, subcommand="add", name="Tmp", path="/tmp/a.md")
    assert mgr.handle_knowledge(sid, subcommand="clear")["success"] is True
    assert mgr.handle_knowledge(sid, subcommand="cancel")["success"] is True

    # 6. Code intelligence
    ci_init = mgr.handle_code_intelligence(sid, subcommand="init")
    assert ci_init["success"] is True
    ci_status = mgr.handle_code_intelligence(sid, subcommand="status")
    assert ci_status["success"] is True
    assert ci_status["status"]["initialized"] is True
    assert "python" in ci_status["status"]["languages"]
    assert "rust" in ci_status["status"]["languages"]


async def test_create_session_refreshes_expired_models_cache(tmp_path: Path) -> None:
    """Verify create_session re-discovers models when `models_cache` is older than 24h."""
    from datetime import UTC, datetime, timedelta

    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider

    class _DiscoveryCli(AgyAcpClientProvider):
        async def discover_models(
            self,
            *,
            cwd: str = ".",
            refresh: bool = False,
        ) -> list[dict[str, str]]:
            _ = (cwd, refresh)
            fresh = [
                {
                    "modelId": "gemini-4-carbon-medium",
                    "name": "Gemini 4 Carbon (Medium)",
                    "description": "Gemini 4 Carbon (Medium)",
                }
            ]
            self.state_store.save_models(fresh)
            return fresh

    cli = _DiscoveryCli(save_dir=tmp_path)
    cli.state_store.save_models(
        [
            {
                "modelId": "gemini-4-argon-medium",
                "name": "Gemini 4 Argon (Medium)",
                "description": "Old",
            }
        ]
    )
    stale_ts = (datetime.now(UTC) - timedelta(days=1, seconds=10)).isoformat()
    with cli.state_store._connect() as conn:
        conn.execute("UPDATE models_cache SET updated_at = ?", (stale_ts,))

    mgr = SessionManager(save_dir=tmp_path, cli_provider=cli)
    desc = await mgr.create_session(cwd=str(tmp_path))
    available = desc["models"]["availableModels"]
    assert [m["modelId"] for m in available] == ["gemini-4-carbon-medium"]
