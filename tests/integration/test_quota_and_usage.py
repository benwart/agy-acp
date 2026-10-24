"""Black-box integration tests for 5h/7d quotas & KiroCrew visibility (REQ-QUOTA-001..004)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.integration.conftest import CliResult, HarnessFactory


def test_req_quota_001_cli_five_hour_quota(
    run_cli: Callable[[list[str]], CliResult],
) -> None:
    """Verify REQ-QUOTA-001: agy-acp quota five-hour returns 5h bucket fractions and reset times."""
    res = run_cli(["quota", "five-hour", "--format", "json"])
    assert res.exit_code == 0
    payload = res.json()
    assert payload["window"] == "5h"
    assert len(payload["buckets"]) >= 1
    first = payload["buckets"][0]
    assert first["window"] == "5h"
    assert 0.0 <= first["remaining_fraction"] <= 1.0
    assert "reset_time" in first


@pytest.mark.not_implemented
def test_req_quota_002_cli_seven_day_quota(
    run_cli: Callable[[list[str]], CliResult],
) -> None:
    """Verify REQ-QUOTA-002: agy-acp quota seven-day returns 7d bucket fractions and reset times."""
    res = run_cli(["quota", "seven-day", "--format", "json"])
    assert res.exit_code == 0
    payload = res.json()
    assert payload["window"] == "7d"
    assert len(payload["buckets"]) >= 1
    first = payload["buckets"][0]
    assert first["window"] == "7d"
    assert 0.0 <= first["remaining_fraction"] <= 1.0
    assert "reset_time" in first


@pytest.mark.not_implemented
def test_req_quota_003_cli_credits_and_combined_status(
    run_cli: Callable[[list[str]], CliResult],
) -> None:
    """Verify REQ-QUOTA-003: agy-acp quota credits and agy-acp quota status commands."""
    credits_res = run_cli(["quota", "credits", "--format", "json"])
    assert credits_res.exit_code == 0
    credits_data = credits_res.json()
    assert "remaining_credits" in credits_data
    assert "token_usage" in credits_data

    status_res = run_cli(["quota", "status", "--format", "json"])
    assert status_res.exit_code == 0
    status_data = status_res.json()
    assert "five_hour" in status_data
    assert "seven_day" in status_data
    assert "credits" in status_data
    assert "token_usage" in status_data


@pytest.mark.not_implemented
async def test_req_quota_004_kirocrew_wire_usage_and_slash_commands(
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify REQ-QUOTA-004: ACP wire emits usage_update, metadata, and /quota slash command."""
    harness = await acp_harness_factory.spawn(
        ["acp", "serve"],
        env={"AGY_ACP_FAKE_SDK": "1", "GEMINI_API_KEY": "test-key"},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in new_resp
    sid = new_resp["result"]["sessionId"]

    prompt_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "/quota"}]},
    )
    assert prompt_resp["result"]["stopReason"] == "end_turn"
    methods = [n["method"] for n in harness.notifications]
    assert "_kiro.dev/metadata" in methods
    update_kinds = [
        n["params"]["update"]["sessionUpdate"]
        for n in harness.notifications
        if n.get("method") == "session/update"
    ]
    assert "usage_update" in update_kinds
    assert "available_commands_update" in update_kinds
