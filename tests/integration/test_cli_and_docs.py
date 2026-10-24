"""Black-box integration verifications for CLI hierarchy and Zensical docs (REQ-CLI-001..002)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.integration.conftest import CliResult


@pytest.mark.not_implemented
def test_req_cli_001_hierarchical_cli_execution(
    run_cli: Callable[[list[str]], CliResult],
    tmp_path: Path,
) -> None:
    """Verify REQ-CLI-001: hierarchical Cyclopts commands execute and return exit code 0."""
    marker = tmp_path / "gate.marker"
    res = run_cli(["acp", "validate-gate", "--marker-file", str(marker), "--format", "json"])
    assert res.exit_code == 0
    assert "valid" in res.json()


def test_req_cli_002_zensical_docs_build(
    run_cli: Callable[[list[str]], CliResult],
    tmp_path: Path,
) -> None:
    """Verify REQ-CLI-002: agy-acp docs build & serve dry-run execute cleanly."""
    build_res = run_cli(["docs", "build", "--site-dir", str(tmp_path), "--dry-run"])
    assert build_res.exit_code == 0

    serve_res = run_cli(["docs", "serve", "--port", "8099", "--dry-run"])
    assert serve_res.exit_code == 0
