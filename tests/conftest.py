"""Global pytest fixtures isolating SQLite state storage per test."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_state_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point AGY_ACP_STATE_DB at an isolated per-test SQLite file by default."""
    monkeypatch.setenv("AGY_ACP_STATE_DB", str(tmp_path / "isolated_state.db"))
