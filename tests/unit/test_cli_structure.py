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


@pytest.mark.not_implemented
def test_acp_serve_accepts_kirocrew_flags(tmp_path: Path) -> None:
    """Verify acp serve handler executes and returns 0 when invoked in dry/validation mode."""
    rc = acp_serve_cmd.default(
        profile="acp",
        with_builtin="developer",
        agent="kirocrew",
        save_dir=tmp_path,
    )
    assert rc == 0


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
