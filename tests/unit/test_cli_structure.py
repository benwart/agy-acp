"""Unit tests for Cyclopts command hierarchy, docstrings, and Zensical docs commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from agy_acp import __version__
from agy_acp.cli import app
from agy_acp.commands.acp import serve as acp_serve_cmd
from agy_acp.commands.docs import build as docs_build_cmd


def test_cli_scaffold_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify Cyclopts scaffold registers subcommands, group docstrings, and version."""
    app(["--version"], result_action="return_value")
    assert __version__ in capsys.readouterr().out

    app(["--help"], result_action="return_value")
    root_help = capsys.readouterr().out
    for group_name in ("acp", "auth", "quota", "docs", "serve"):
        assert group_name in root_help

    app(["quota", "--help"], result_action="return_value")
    quota_help = capsys.readouterr().out
    assert "Inspect token, credit, 5-hour, and 7-day rolling quota usage." in quota_help
    assert "five-hour" in quota_help
    assert "seven-day" in quota_help


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
        ["acp", "serve", "--custom-flag", "123"],
    ):
        cmd_rc = int(app(tokens, result_action="return_int_as_exit_code_else_zero"))
        assert cmd_rc == 0


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
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify whoami, agent validate, and chat (--list-models and /usage) CLI commands."""
    import json

    from agy_acp.cli import main

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

    rc_usage = main(["chat", "--no-interactive", "--agent", "kirocrew-lite", "/usage"])
    assert rc_usage == 0
    usage_out = capsys.readouterr().out
    assert "Estimated Usage" in usage_out
    assert "covered in plan" in usage_out
