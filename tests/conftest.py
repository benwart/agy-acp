"""Global pytest fixtures isolating SQLite state storage and telemetry per test."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_state_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point AGY_ACP_STATE_DB and telemetry config at isolated per-test paths by default."""
    monkeypatch.setenv("AGY_ACP_STATE_DB", str(tmp_path / "isolated_state.db"))
    monkeypatch.setenv("AGY_ACP_TELEMETRY_CONFIG", str(tmp_path / "isolated_telemetry.json"))
    monkeypatch.setenv("AGY_ACP_TELEMETRY_ENABLED", "0")
    yield
    telemetry_mod = sys.modules.get("agy_acp.providers.telemetry")
    if telemetry_mod is not None and hasattr(telemetry_mod, "reset_telemetry_for_tests"):
        telemetry_mod.reset_telemetry_for_tests()
