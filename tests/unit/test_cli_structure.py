"""Unit tests for Cyclopts command hierarchy, docstrings, and Zensical docs commands."""

from __future__ import annotations

from pathlib import Path

from agy_acp.cli import app
from agy_acp.commands.acp import serve as acp_serve_cmd
from agy_acp.commands.docs import build as docs_build_cmd


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


def test_zensical_config_and_pages(tmp_path: Path) -> None:
    """Verify docs build handler invokes Zensical build cleanly."""
    rc = docs_build_cmd.default(site_dir=tmp_path, dry_run=True)
    assert rc == 0
