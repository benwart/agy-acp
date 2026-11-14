"""Unit tests for recursive secret redaction in `acp_logger` (`acp.log` and `stderr`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agy_acp.providers import acp_logger
from agy_acp.providers.acp_logger import log_acp_event, snapshot_relevant_env


def test_redact_sensitive_data_dicts_lists_env_and_headers() -> None:
    """Verify recursive redaction of env, headers, and sensitive keys across nested structures."""
    payload = {
        "sessionId": "sess-123",
        "cwd": "/workspace/project",
        "env": {
            "KIROCREW_STUB_SESSION_TOKEN": "stub_secret_token_abc",
            "PLAIN_VAR": "plain_val_in_env_map",
        },
        "Headers": {
            "Authorization": "Bearer super_secret_jwt_999",
            "X-Custom-Token": "hdr_tok_777",
        },
        "mcpServers": [
            {
                "name": "kirocrew-core",
                "env": [
                    {"name": "KIROCREW_STUB_SESSION_TOKEN", "value": "list_secret_token_xyz"},
                    {"name": "LOG_LEVEL", "value": "debug"},
                ],
                "headers": [
                    {"name": "Authorization", "value": "Bearer header_list_jwt_555"},
                ],
            }
        ],
        "nested": {
            "apiKey": "ak_live_111",
            "api_key": "ak_live_222",
            "client_secret": "sec_333",
            "password": "pw_444",
            "private_key": "-----BEGIN PRIVATE KEY-----",
            "cookie": "sid=cookie_555",
            "safeField": "keep-me",
        },
    }

    redacted = acp_logger.redact_sensitive_data(payload)
    assert redacted["sessionId"] == "sess-123"
    assert redacted["cwd"] == "/workspace/project"
    assert redacted["env"] == {
        "KIROCREW_STUB_SESSION_TOKEN": "[REDACTED]",
        "PLAIN_VAR": "[REDACTED]",
    }
    assert redacted["Headers"] == {
        "Authorization": "[REDACTED]",
        "X-Custom-Token": "[REDACTED]",
    }
    assert redacted["mcpServers"][0]["name"] == "kirocrew-core"
    assert redacted["mcpServers"][0]["env"] == [
        {"name": "KIROCREW_STUB_SESSION_TOKEN", "value": "[REDACTED]"},
        {"name": "LOG_LEVEL", "value": "[REDACTED]"},
    ]
    assert redacted["mcpServers"][0]["headers"] == [
        {"name": "Authorization", "value": "[REDACTED]"},
    ]
    assert redacted["nested"] == {
        "apiKey": "[REDACTED]",
        "api_key": "[REDACTED]",
        "client_secret": "[REDACTED]",
        "password": "[REDACTED]",
        "private_key": "[REDACTED]",
        "cookie": "[REDACTED]",
        "safeField": "keep-me",
    }


def test_log_acp_event_redacts_raw_json_frames_and_strings_in_file_and_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `acp.log` and `stderr` never contain plaintext tokens from raw JSON frames."""
    log_path = tmp_path / "acp.log"
    monkeypatch.setenv("AGY_ACP_LOG_FILE", str(log_path))
    monkeypatch.setenv("AGY_ACP_LOG_STDERR", "1")
    monkeypatch.setenv("KIROCREW_STUB_SESSION_TOKEN", "ENV_LEAK_98765")

    raw_rpc_frame = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": "/repo",
                    "mcpServers": [
                        {
                            "name": "kirocrew-core",
                            "env": [
                                {
                                    "name": "KIROCREW_STUB_SESSION_TOKEN",
                                    "value": "FRAME_STUB_SECRET_111",
                                }
                            ],
                            "headers": [
                                {
                                    "name": "Authorization",
                                    "value": "Bearer FRAME_BEARER_SECRET_222",
                                }
                            ],
                        }
                    ],
                    "_meta": {"apiKey": "FRAME_META_KEY_333"},
                },
            }
        )
        + "\n"
    )

    log_acp_event(
        "stdin.raw",
        raw=raw_rpc_frame,
        auth_note=(
            "Header Authorization: Bearer INLINE_BEARER_444 "
            "and KIROCREW_STUB_SESSION_TOKEN=INLINE_TOK_555"
        ),
        env_snap=snapshot_relevant_env(),
    )

    captured = capsys.readouterr()
    file_text = log_path.read_text(encoding="utf-8")

    for leaked in (
        "FRAME_STUB_SECRET_111",
        "FRAME_BEARER_SECRET_222",
        "FRAME_META_KEY_333",
        "INLINE_BEARER_444",
        "INLINE_TOK_555",
        "ENV_LEAK_98765",
    ):
        assert leaked not in file_text
        assert leaked not in captured.err

    record = json.loads(file_text.strip())
    assert record["event"] == "stdin.raw"
    parsed_raw = json.loads(record["raw"])
    assert parsed_raw["params"]["cwd"] == "/repo"
    assert parsed_raw["params"]["mcpServers"][0]["env"][0]["value"] == "[REDACTED]"
    assert parsed_raw["params"]["mcpServers"][0]["headers"][0]["value"] == "[REDACTED]"
    assert parsed_raw["params"]["_meta"]["apiKey"] == "[REDACTED]"


def test_log_acp_event_skips_disk_by_default_and_streams_to_otel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """By default `log_acp_event` does not touch `acp.log` on disk and emits redacted OTel logs."""
    from agy_acp.providers import telemetry as tel_mod

    monkeypatch.delenv("AGY_ACP_LOG_FILE", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_DISK", raising=False)
    monkeypatch.setenv("AGY_ACP_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("AGY_ACP_OTLP_ENDPOINT", "http://127.0.0.1:59999")
    tel_mod.reset_telemetry_for_tests()

    mgr = tel_mod.TelemetryManager(
        tel_mod.resolve_telemetry_config(tmp_path),
        use_simple_processors=True,
    )
    monkeypatch.setattr(tel_mod, "_GLOBAL_MANAGER", mgr)
    try:
        with mgr.start_span("parent-span") as span:
            log_acp_event(
                "rpc.receive",
                save_dir_or_db=tmp_path,
                method="session/new",
                headers={"Authorization": "Bearer TOP_SECRET_OTEL_999"},
            )
            expected_trace_id = span.get_span_context().trace_id

        default_log = tmp_path / "acp.log"
        assert not default_log.exists()

        buffered = mgr.peek_buffered_logs()
        assert len(buffered) == 1
        rec = buffered[0].log_record
        assert rec.trace_id == expected_trace_id
        assert rec.attributes is not None
        assert rec.attributes["event"] == "rpc.receive"
        assert "TOP_SECRET_OTEL_999" not in str(rec.body)
        assert "TOP_SECRET_OTEL_999" not in str(rec.attributes)
    finally:
        tel_mod.reset_telemetry_for_tests()


def test_log_acp_event_writes_disk_when_opt_in_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting `AGY_ACP_LOG_DISK=1` opts back into writing redacted JSONL to `acp.log`."""
    monkeypatch.delenv("AGY_ACP_LOG_FILE", raising=False)
    monkeypatch.setenv("AGY_ACP_LOG_DISK", "1")

    log_acp_event("opt_in.event", save_dir_or_db=tmp_path, detail="hello")
    default_log = tmp_path / "acp.log"
    assert default_log.exists()
    parsed = json.loads(default_log.read_text(encoding="utf-8").strip())
    assert parsed["event"] == "opt_in.event"
    assert parsed["detail"] == "hello"
