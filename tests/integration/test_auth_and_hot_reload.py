"""Black-box integration tests for CLI auth & zero-downtime hot-reload (REQ-AUTH-001..004)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.integration.conftest import CliResult, HarnessFactory


def test_req_auth_001_cli_login_persists_credentials(
    run_cli: Callable[[list[str]], CliResult],
    tmp_path: Path,
) -> None:
    """Verify REQ-AUTH-001: agy-acp auth login writes 0600 credentials atomically."""
    auth_file = tmp_path / "credentials.json"
    res = run_cli(
        [
            "auth",
            "login",
            "--api-key",
            "test-gemini-key",
            "--auth-file",
            str(auth_file),
            "--format",
            "json",
        ]
    )
    assert res.exit_code == 0
    payload = res.json()
    assert payload["authenticated"] is True
    assert payload["method"] == "api_key"
    assert auth_file.is_file()
    assert (auth_file.stat().st_mode & 0o777) == 0o600


@pytest.mark.not_implemented
def test_req_auth_002_cli_logout_clears_credentials(
    run_cli: Callable[[list[str]], CliResult],
    tmp_path: Path,
) -> None:
    """Verify REQ-AUTH-002: agy-acp auth logout clears credentials and increments generation."""
    auth_file = tmp_path / "credentials.json"
    run_cli(["auth", "login", "--api-key", "test-key", "--auth-file", str(auth_file)])
    res = run_cli(["auth", "logout", "--auth-file", str(auth_file), "--format", "json"])
    assert res.exit_code == 0
    payload = res.json()
    assert payload["authenticated"] is False
    assert payload["generation"] >= 2


@pytest.mark.not_implemented
def test_req_auth_003_cli_auth_status_reporting(
    run_cli: Callable[[list[str]], CliResult],
    tmp_path: Path,
) -> None:
    """Verify REQ-AUTH-003: agy-acp auth status reports auth state and fingerprint."""
    auth_file = tmp_path / "credentials.json"
    unauth = run_cli(["auth", "status", "--auth-file", str(auth_file), "--format", "json"])
    assert unauth.exit_code == 1
    assert unauth.json()["authenticated"] is False

    run_cli(["auth", "login", "--api-key", "test-key", "--auth-file", str(auth_file)])
    authed = run_cli(["auth", "status", "--auth-file", str(auth_file), "--format", "json"])
    assert authed.exit_code == 0
    assert authed.json()["authenticated"] is True
    assert authed.json()["fingerprint"].startswith("sha256:")


@pytest.mark.not_implemented
async def test_req_auth_004_running_acp_server_hot_reloads_auth_without_restart(
    acp_harness_factory: HarnessFactory,
    run_cli: Callable[[list[str]], CliResult],
    tmp_path: Path,
) -> None:
    """Verify REQ-AUTH-004: running ACP server picks up external CLI auth without restart."""
    auth_file = tmp_path / "credentials.json"
    save_dir = tmp_path / "sessions"

    harness = await acp_harness_factory.spawn(
        ["acp", "serve", "--auth-file", str(auth_file), "--save-dir", str(save_dir)],
        env={"AGY_ACP_FAKE_SDK": "1"},
    )
    await harness.send_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    new_resp = await harness.send_request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
    assert "result" in new_resp
    sid = new_resp["result"]["sessionId"]

    err_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Hello"}]},
    )
    assert "error" in err_resp
    assert "not logged in" in err_resp["error"]["message"].lower()

    login_res = run_cli(["auth", "login", "--api-key", "live-key", "--auth-file", str(auth_file)])
    assert login_res.exit_code == 0

    ok_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Hello after login"}]},
    )
    assert ok_resp["result"]["stopReason"] == "end_turn"

    logout_res = run_cli(["auth", "logout", "--auth-file", str(auth_file)])
    assert logout_res.exit_code == 0
    post_logout_resp = await harness.send_request(
        "session/prompt",
        {"sessionId": sid, "prompt": [{"type": "text", "text": "Hello after logout"}]},
    )
    assert "error" in post_logout_resp
    assert "not logged in" in post_logout_resp["error"]["message"].lower()
