"""Unit tests for Cyclopts command hierarchy, docstrings, and Zensical docs commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from agy_acp import KIRO_CLI_COMPAT_VERSION, __version__
from agy_acp.cli import app
from agy_acp.commands.acp import serve as acp_serve_cmd
from agy_acp.commands.docs import build as docs_build_cmd


def test_cli_scaffold_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify Cyclopts scaffold registers subcommands, group docstrings, and version."""
    assert __version__ == "1.3.2"
    assert KIRO_CLI_COMPAT_VERSION == "3.3.0"
    app(["--version"], result_action="return_value")
    version_out = capsys.readouterr().out.strip()
    assert "1.3.2 kiro-cli 3.3.0" in version_out
    last_token = version_out.split()[-1]
    major, minor, patch = (int(part) for part in last_token.split(".")[:3])
    assert (major, minor, patch) >= (3, 3, 0)

    app(["--help"], result_action="return_value")
    root_help = capsys.readouterr().out
    for group_name in ("acp", "auth", "quota", "docs", "serve", "update"):
        assert group_name in root_help

    app(["quota", "--help"], result_action="return_value")
    quota_help = capsys.readouterr().out
    assert "Inspect token, credit, 5-hour, and 7-day rolling quota usage." in quota_help
    assert "five-hour" in quota_help
    assert "seven-day" in quota_help


def test_update_command_refreshes_wrapper_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `kiro-cli update` compatibility command exits 0 and bumps wrapper mtime."""
    import os
    import sys

    from agy_acp.cli import main

    fake_bin = tmp_path / "agy-acp"
    fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    os.utime(fake_bin, (1_700_000_000, 1_700_000_000))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [str(fake_bin), "update"])

    rc = main(["update"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1.3.2" in out
    assert "3.3.0" in out
    assert fake_bin.stat().st_mtime > 1_700_000_000


def test_agent_engine_help_satisfies_kirocrew_kas_floor_probe(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--help` output advertises `--agent-engine` and `[possible values: v2, v3]`."""
    import re

    possible_values_re = re.compile(r"\[possible values:\s*([^\]]*)\]")
    monkeypatch.setenv("COLUMNS", "80")
    for help_tokens in (
        ["--help"],
        ["serve", "--help"],
        ["acp", "--help"],
        ["acp", "serve", "--help"],
        ["chat", "--help"],
    ):
        app(help_tokens, result_action="return_value")
        help_text = capsys.readouterr().out
        assert "--agent-engine" in help_text, f"Missing --agent-engine in {help_tokens}"
        assert "v3" in help_text, f"Missing v3 in {help_tokens}"
        at = help_text.find("--agent-engine")
        end = help_text.find("\n      --", at + 1)
        block = help_text[at : end if end > 0 else len(help_text)]
        match = possible_values_re.search(block)
        assert match is not None, f"Missing [possible values: ...] in {help_tokens}:\n{block}"
        parsed = tuple(v.strip() for v in match.group(1).split(",") if v.strip())
        assert parsed == ("v2", "v3")


async def test_normalize_agent_engine_and_cli_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify normalize_agent_engine maps v2/v3 aliases and CLI exits 2 on invalid engine."""
    from agy_acp.commands.serve import default as serve_default
    from agy_acp.commands.serve import normalize_agent_engine

    for raw_v2 in (None, "", "  ", "default", "v1", "V1", "v2", "V2", "kiro", "KIRO"):
        assert normalize_agent_engine(raw_v2) == "v2"
    for raw_v3 in ("v3", "V3", "kas", "KAS", " v3 "):
        assert normalize_agent_engine(raw_v3) == "v3"

    with pytest.raises(ValueError, match="v99"):
        normalize_agent_engine("v99")

    assert await serve_default(agent_engine="v2", save_dir=tmp_path) == 0
    assert await serve_default(agent_engine="v3", save_dir=tmp_path) == 0
    assert await acp_serve_cmd.default(agent_engine="kas", save_dir=tmp_path) == 0

    with pytest.raises(SystemExit) as serve_exc:
        await serve_default(agent_engine="v99", save_dir=tmp_path)
    assert serve_exc.value.code == 2
    assert "v99" in capsys.readouterr().err

    with pytest.raises(SystemExit) as acp_exc:
        await acp_serve_cmd.default(agent_engine="v99", save_dir=tmp_path)
    assert acp_exc.value.code == 2
    assert "v99" in capsys.readouterr().err

    with pytest.raises(SystemExit) as cli_exc:
        app(
            ["acp", "--agent-engine", "v99", "--save-dir", str(tmp_path)],
            result_action="return_int_as_exit_code_else_zero",
        )
    assert cli_exc.value.code == 2
    assert "v99" in capsys.readouterr().err


async def test_acp_serve_accepts_kirocrew_flags(tmp_path: Path) -> None:
    """Verify acp serve handler executes and returns 0 when invoked in dry/validation mode."""
    rc = await acp_serve_cmd.default(
        profile="acp",
        with_builtin="developer",
        agent="kirocrew",
        save_dir=tmp_path,
    )
    assert rc == 0

    # Verify direct KiroCrew invocation patterns without explicit 'serve' subcommand
    for tokens in (
        [],
        ["--profile", "acp"],
        ["--stdio", "--unknown-kiro-flag", "val"],
        ["acp"],
        ["acp", "--with-builtin", "developer"],
        ["acp", "--agent", "kirocrew", "--model", "gemini-3.7-pro"],
        ["acp", "--agent-engine", "v3", "--authMethod", "client"],
        ["acp", "--effort", "high", "-a", "--trust-tools", "fs_read"],
        ["acp", "serve", "--custom-flag", "123"],
    ):
        cmd_rc = int(app(tokens, result_action="return_int_as_exit_code_else_zero"))
        assert cmd_rc == 0


async def test_default_effort_and_asbx_kiro_env_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `--effort` seeds new session effort, rejects invalid effort, and logs ASBX_KIRO_*."""
    from agy_acp.commands.serve import normalize_effort
    from agy_acp.logic.session_manager import SessionManager
    from agy_acp.logic.wire import create_wire_adapter
    from agy_acp.providers.acp_logger import snapshot_relevant_env

    for valid in ("low", "medium", "high", "xhigh", "max", " HIGH "):
        assert normalize_effort(valid) == valid.strip().lower()
    assert normalize_effort(None) is None
    assert normalize_effort("  ") is None
    with pytest.raises(ValueError, match="turbo"):
        normalize_effort("turbo")

    v2_mgr = SessionManager(save_dir=tmp_path / "v2", default_effort="high")
    v2_desc = await v2_mgr.create_session(cwd=str(tmp_path), mcp_servers=[])
    v2_opt = next(opt for opt in v2_desc["configOptions"] if opt["category"] == "thought_level")
    assert v2_opt["currentValue"] == "high"
    await v2_mgr.close()

    v3_mgr = SessionManager(
        save_dir=tmp_path / "v3",
        default_effort="xhigh",
        wire_adapter=create_wire_adapter("v3"),
    )
    v3_desc = await v3_mgr.create_session(cwd=str(tmp_path), mcp_servers=[])
    v3_opt = next(opt for opt in v3_desc["configOptions"] if opt["category"] == "thought_level")
    assert v3_opt["currentValue"] == "xhigh"
    await v3_mgr.close()

    with pytest.raises(SystemExit) as cli_exc:
        app(
            ["acp", "--effort", "turbo", "--save-dir", str(tmp_path)],
            result_action="return_int_as_exit_code_else_zero",
        )
    assert cli_exc.value.code == 2
    assert "turbo" in capsys.readouterr().err

    monkeypatch.setenv("ASBX_KIRO_MANDATORY_MCPS", "")
    snap = snapshot_relevant_env()
    assert "ASBX_KIRO_MANDATORY_MCPS" in snap


def test_commands_hierarchy_and_docstrings(tmp_path: Path) -> None:
    """Verify hierarchical command registration and execution of docs build dry-run."""
    rc = int(
        app(
            ["docs", "build", "--site-dir", str(tmp_path), "--dry-run"],
            result_action="return_int_as_exit_code_else_zero",
        )
    )
    assert rc == 0


async def test_zensical_config_and_pages(tmp_path: Path) -> None:
    """Verify docs build handler invokes Zensical build cleanly."""
    rc = await docs_build_cmd.default(site_dir=tmp_path, dry_run=True)
    assert rc == 0


def test_acp_logs_command_displays_and_clears_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify agp-acp acp logs prints log path, tails entries, and clears log."""
    from agy_acp.cli import main

    log_file = tmp_path / "acp.log"
    monkeypatch.setenv("AGY_ACP_LOG_FILE", str(log_file))

    rc_path = main(["acp", "logs", "--path"])
    assert rc_path == 0
    assert str(log_file) in capsys.readouterr().out

    rc_tail = main(["acp", "logs", "--tail", "10"])
    assert rc_tail == 0
    out_tail = capsys.readouterr().out
    assert "cli.invoke" in out_tail

    rc_clear = main(["acp", "logs", "--clear"])
    assert rc_clear == 0
    capsys.readouterr()
    assert not log_file.exists() or log_file.read_text(encoding="utf-8") == ""


def test_kirocrew_prerequisite_and_dashboard_cli_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify whoami, agent validate, and chat (--list-models and /usage) CLI commands."""
    import json

    from agy_acp.cli import main
    from agy_acp.commands import chat as chat_cmd
    from agy_acp.logic.quota_tracker import QuotaTracker
    from agy_acp.providers.quota_provider import QuotaProvider
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    state_db = tmp_path / "state.db"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("AGY_ACP_STATE_DB", str(state_db))
    monkeypatch.setenv("AGY_BIN", str(tmp_path / "missing-agy"))

    store = SqliteStateStoreProvider(state_db)
    store.save_models(
        [
            {
                "modelId": "gemini-3.7-pro",
                "name": "Gemini 3.7 Pro",
                "description": "Google Antigravity Pro model",
            }
        ]
    )
    store.save_tier({"id": "g1-ultra-tier", "name": "Google AI Ultra", "upgradeUri": ""})

    class _IsolatedQuotaProvider(QuotaProvider):
        async def fetch_quota_groups(self) -> list[dict[str, object]]:
            return [
                {
                    "group": "Gemini Models",
                    "buckets": [
                        {
                            "id": "gemini-5h",
                            "group": "Gemini Models",
                            "name": "Five Hour Limit Remaining",
                            "window": "5h",
                            "remaining_fraction": 0.95,
                            "reset_time": "2026-11-14T08:00:00Z",
                        }
                    ],
                }
            ]

        async def fetch_credits(self) -> dict[str, object]:
            return {"remaining_credits": 0, "upgrade_uri": "https://one.google.com/ai"}

    monkeypatch.setattr(
        chat_cmd,
        "QuotaTracker",
        lambda: QuotaTracker(provider=_IsolatedQuotaProvider(), state_store=store),
    )

    rc_whoami = main(["whoami", "--format", "json"])
    assert rc_whoami == 0
    whoami_data = json.loads(capsys.readouterr().out)
    assert "email" in whoami_data
    assert "start_url" in whoami_data

    spec_file = tmp_path / "kirocrew.json"
    spec_file.write_text('{"name": "kirocrew", "description": "Valid spec"}', encoding="utf-8")
    rc_val = main(["agent", "validate", "--path", str(spec_file)])
    val_out = capsys.readouterr()
    assert rc_val == 0
    assert "is invalid" not in val_out.out.lower()
    assert "is invalid" not in val_out.err.lower()

    rc_models = main(["chat", "--list-models", "--format", "json", "--no-interactive"])
    assert rc_models == 0
    models_data = json.loads(capsys.readouterr().out)
    assert isinstance(models_data.get("models"), list)
    assert len(models_data["models"]) >= 1
    assert "model_id" in models_data["models"][0]
    assert "model_name" in models_data["models"][0]
    assert "context_window_tokens" in models_data["models"][0]

    rc_usage = main(
        ["chat", "--no-interactive", "--agent-engine", "v2", "--agent", "kirocrew-lite", "/usage"]
    )
    assert rc_usage == 0
    usage_out = capsys.readouterr().out
    assert "Estimated Usage" in usage_out
    assert "covered in plan" in usage_out


async def test_missing_agy_binary_falls_back_cleanly_in_quota_provider_and_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify QuotaProvider and chat CLI gracefully fall back when `agy` is absent on PATH."""
    import json

    from agy_acp.commands import chat as chat_cmd
    from agy_acp.providers.quota_provider import QuotaProvider
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    fake_home = tmp_path / "empty-home"
    fake_home.mkdir()
    cold_db = tmp_path / "cold-state.db"
    missing_bin = tmp_path / "nonexistent-agy"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("AGY_ACP_STATE_DB", str(cold_db))
    monkeypatch.setenv("AGY_BIN", str(missing_bin))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    provider = QuotaProvider(agy_bin=missing_bin)
    assert await provider.fetch_quota_groups() == []
    assert await provider.fetch_credits() == {}
    assert await provider.fetch_tier() is None

    rc_models = await chat_cmd.default(list_models=True, format="json", no_interactive=True)
    assert rc_models == 0
    models_payload = json.loads(capsys.readouterr().out)
    assert len(models_payload["models"]) >= 1

    rc_cold_usage = await chat_cmd.default("/usage", no_interactive=True, agent_engine="v2")
    assert rc_cold_usage == 0
    cold_out = capsys.readouterr().out
    assert "Estimated Usage" in cold_out
    assert "covered in plan" in cold_out

    warm_store = SqliteStateStoreProvider(cold_db)
    warm_store.save_tier({"id": "g1-ultra-tier", "name": "Google AI Ultra", "upgradeUri": ""})
    warm_store.save_quota_groups(
        [
            {
                "group": "Gemini Models",
                "buckets": [
                    {
                        "id": "gemini-5h",
                        "group": "Gemini Models",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 0.95,
                        "reset_time": "2026-11-14T08:00:00Z",
                    }
                ],
            }
        ]
    )
    warm_store.save_credits({"remaining_credits": 0, "upgrade_uri": "https://one.google.com/ai"})

    rc_warm_usage = await chat_cmd.default("/usage", no_interactive=True, agent_engine="v2")
    assert rc_warm_usage == 0
    warm_out = capsys.readouterr().out
    assert "(40.0 of 800.0 covered in plan)" in warm_out


async def test_chat_usage_scrape_uses_local_reset_date(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `agy-acp chat /usage` converts UTC reset timestamps to the local calendar date."""
    import os
    import time

    from agy_acp.commands import chat as chat_cmd
    from agy_acp.logic.quota_tracker import QuotaTracker
    from agy_acp.providers.quota_provider import QuotaProvider

    class _LateEveningUtcProvider(QuotaProvider):
        async def fetch_quota_groups(self) -> list[dict[str, object]]:
            return [
                {
                    "group": "Gemini Models",
                    "buckets": [
                        {
                            "id": "gemini-5h",
                            "group": "Gemini Models",
                            "name": "Five Hour Limit Remaining",
                            "window": "5h",
                            "remaining_fraction": 0.90,
                            "reset_time": "2026-11-04T06:21:14Z",
                        }
                    ],
                }
            ]

        async def fetch_credits(self) -> dict[str, object]:
            return {"remaining_credits": 0, "upgrade_uri": "https://one.google.com/ai"}

    prev_tz = os.environ.get("TZ")
    monkeypatch.setattr(
        chat_cmd,
        "QuotaTracker",
        lambda: QuotaTracker(provider=_LateEveningUtcProvider()),
    )
    try:
        os.environ["TZ"] = "America/Los_Angeles"
        time.tzset()
        rc = await chat_cmd.default("/usage", no_interactive=True)
    finally:
        if prev_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = prev_tz
        time.tzset()

    assert rc == 0
    out = capsys.readouterr().out
    assert "Estimated Usage | resets on 2026-11-03 | Antigravity" in out
    assert "2026-11-03 10:21 PM PST (" in out


async def test_run_stdio_server_batches_response_and_post_notifications_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify KAS v3 session/new response and post-response MCP frames flush in one OS write."""
    import json
    import sys

    import anyio

    from agy_acp.commands.serve import run_stdio_server
    from agy_acp.logic.session_manager import SessionManager
    from agy_acp.logic.wire import create_wire_adapter

    adapter = create_wire_adapter("v3")
    mgr = SessionManager(save_dir=tmp_path, wire_adapter=adapter)
    writes: list[bytes] = []
    flushes = 0
    ready_for_eof = anyio.Event()
    inputs = iter(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/new",
                    "params": {"cwd": str(tmp_path), "mcpServers": []},
                }
            )
            + "\n"
        ]
    )

    class _FakeBuffer:
        def write(self, data: bytes) -> int:
            writes.append(data)
            return len(data)

        def flush(self) -> None:
            nonlocal flushes
            flushes += 1
            ready_for_eof.set()

    class _FakeStdout:
        buffer = _FakeBuffer()

    monkeypatch.setattr(sys, "stdout", _FakeStdout())

    async def _readline() -> str:
        try:
            return next(inputs)
        except StopIteration:
            await ready_for_eof.wait()
            return ""

    rc = await run_stdio_server(
        session_manager=mgr,
        wire_adapter=adapter,
        save_dir=tmp_path,
        stdin_readline=_readline,
    )
    assert rc == 0
    assert len(writes) == 1
    assert flushes == 1
    lines = [json.loads(line) for line in writes[0].decode("utf-8").splitlines() if line]
    assert len(lines) == 3
    assert lines[0]["id"] == 1
    assert [lines[1]["method"], lines[2]["method"]] == [
        "_kiro/mcp/status",
        "_kiro/tools/didChange",
    ]


def test_whoami_emits_kirocrew_camelcase_and_snakecase_identity_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `whoami` emits camelCase keys for KiroCrew's `parse_whoami_output` plus snake_case."""
    import json

    from agy_acp.cli import main

    rc = main(["whoami", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["email"] == "antigravity@google.com"
    assert payload["accountType"] == "Antigravity"
    assert payload["startUrl"] == "https://one.google.com/ai"
    assert payload["account_type"] == "Antigravity"
    assert payload["start_url"] == "https://one.google.com/ai"


async def test_chat_usage_scrape_offline_wake_fallback_and_single_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `chat /usage` fetches once when online and falls back to SQLite after sleep."""
    from typing import Any

    from agy_acp.commands import chat as chat_cmd
    from agy_acp.logic.quota_tracker import QuotaTracker
    from agy_acp.providers.quota_provider import QuotaProvider
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    store = SqliteStateStoreProvider(tmp_path / "wake_state.db")
    store.save_tier({"id": "g1-ultra-tier", "name": "Google AI Ultra", "upgradeUri": ""})
    calls = {"usage": 0, "credits": 0, "offline": False}

    class _ToggleWakeProvider(QuotaProvider):
        async def fetch_quota_groups(self) -> list[dict[str, Any]]:
            calls["usage"] += 1
            if calls["offline"]:
                raise RuntimeError("Wi-Fi reconnecting after sleep")
            return [
                {
                    "group": "Gemini Models",
                    "buckets": [
                        {
                            "id": "gemini-5h",
                            "group": "Gemini Models",
                            "name": "Five Hour Limit Remaining",
                            "window": "5h",
                            "remaining_fraction": 0.95,
                            "reset_time": "2026-11-14T08:00:00Z",
                        }
                    ],
                }
            ]

        async def fetch_credits(self) -> dict[str, Any]:
            calls["credits"] += 1
            if calls["offline"]:
                raise RuntimeError("Wi-Fi reconnecting after sleep")
            return {"remaining_credits": 0, "upgrade_uri": "https://one.google.com/ai"}

    monkeypatch.setattr(
        chat_cmd,
        "QuotaTracker",
        lambda: QuotaTracker(provider=_ToggleWakeProvider(), state_store=store),
    )

    rc_online = await chat_cmd.default("/usage", no_interactive=True)
    assert rc_online == 0
    online_out = capsys.readouterr().out
    assert "(40.0 of 800.0 covered in plan)" in online_out
    assert calls["usage"] == 1
    assert calls["credits"] == 1

    # Simulate KiroCrew running `kiro-cli chat /usage` immediately on wake before Wi-Fi is up
    calls["offline"] = True
    rc_wake = await chat_cmd.default("/usage", no_interactive=True)
    assert rc_wake == 0
    wake_out = capsys.readouterr().out
    assert "Estimated Usage | resets on " in wake_out
    assert "(40.0 of 800.0 covered in plan)" in wake_out
