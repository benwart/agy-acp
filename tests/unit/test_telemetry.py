"""Unit and integration tests for OpenTelemetry configuration, resilience, and export."""

from __future__ import annotations

import socket
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agy_acp.providers.telemetry import (
    DEFAULT_OTLP_ENDPOINT,
    TelemetryCallbacks,
    TelemetryConfig,
    TelemetryManager,
    load_persisted_telemetry_settings,
    resolve_telemetry_config,
    save_persisted_telemetry_settings,
)


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LocalOtlpCollector:
    """Ephemeral HTTP OTLP collector that can go up and down on a fixed localhost port."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._lock = threading.Lock()
        self.requests_by_path: dict[str, list[bytes]] = defaultdict(list)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        outer = self

        class _ReusableServer(ThreadingHTTPServer):
            allow_reuse_address = True

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                with outer._lock:
                    outer.requests_by_path[self.path].append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

        self._server = _ReusableServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def count(self, path: str) -> int:
        with self._lock:
            return len(self.requests_by_path[path])

    def payloads(self, path: str) -> list[bytes]:
        with self._lock:
            return list(self.requests_by_path[path])


def test_telemetry_config_defaults_to_enabled_and_disk_logs_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry is enabled by default for subprocess launches while disk logs default to off."""
    monkeypatch.delenv("AGY_ACP_TELEMETRY_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("AGY_ACP_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_FILE", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_DISK", raising=False)

    cfg = resolve_telemetry_config()
    assert cfg.enabled is True
    assert cfg.endpoint == DEFAULT_OTLP_ENDPOINT == "http://127.0.0.1:4318"
    assert cfg.disk_logs is False
    assert cfg.service_name == "agy-acp"


def test_telemetry_config_persistence_and_env_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted `telemetry.json` applies across runs and environment variables take precedence."""
    monkeypatch.delenv("AGY_ACP_TELEMETRY_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_FILE", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_DISK", raising=False)

    saved_path = save_persisted_telemetry_settings(
        {
            "enabled": False,
            "endpoint": "http://127.0.0.1:14318/",
            "disk_logs": True,
            "service_name": "agy-acp-custom",
        },
        save_dir_or_db=tmp_path,
    )
    assert saved_path.exists()
    loaded = load_persisted_telemetry_settings(tmp_path)
    assert loaded["enabled"] is False

    cfg_from_file = resolve_telemetry_config(tmp_path)
    assert cfg_from_file.enabled is False
    assert cfg_from_file.endpoint == "http://127.0.0.1:14318"
    assert cfg_from_file.disk_logs is True
    assert cfg_from_file.service_name == "agy-acp-custom"

    # Environment overrides persisted settings
    monkeypatch.setenv("AGY_ACP_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:24318")
    monkeypatch.setenv("AGY_ACP_LOG_DISK", "0")
    cfg_env = resolve_telemetry_config(tmp_path)
    assert cfg_env.enabled is True
    assert cfg_env.endpoint == "http://127.0.0.1:24318"
    assert cfg_env.disk_logs is False

    # Explicit AGY_ACP_LOG_FILE always enables disk logging
    monkeypatch.setenv("AGY_ACP_LOG_FILE", str(tmp_path / "explicit.log"))
    assert resolve_telemetry_config(tmp_path).disk_logs is True

    # Standard OTEL_SDK_DISABLED=true disables telemetry
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert resolve_telemetry_config(tmp_path).enabled is False


def test_resilient_otlp_exporter_survives_collector_down_up_down_up_without_restart(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify zero stderr noise while offline and automatic flush when collector comes up/down."""
    port = _reserve_free_port()
    collector = _LocalOtlpCollector(port)
    cfg = TelemetryConfig(
        enabled=True,
        endpoint=collector.endpoint,
        service_name="agy-acp-test",
        disk_logs=False,
        metric_export_interval_ms=60_000,
        export_timeout_seconds=0.5,
        max_buffered_spans=32,
        max_buffered_logs=64,
        config_path=tmp_path / "telemetry.json",
    )
    mgr = TelemetryManager(cfg, use_simple_processors=True)
    mgr.register_callbacks(
        TelemetryCallbacks(
            backend_up=lambda: 1,
            session_counts=lambda: {"active": 2, "persisted": 5},
            quota_metrics=lambda: (
                [{"group": "Gemini Models", "window": "5h", "percent": 87.5}],
                700.0,
                800.0,
            ),
            ledger_tokens=lambda: {"input": 1200, "output": 340},
            mcp_tools_discovered=lambda: {"kirocrew-core": 14},
        )
    )

    try:
        # Phase 1: Collector is DOWN -> emit traces, logs, and metrics with zero stderr noise
        with mgr.start_span("acp.rpc.session/prompt", attributes={"rpc.method": "session/prompt"}):
            mgr.record_rpc_request("session/prompt", duration_s=0.12, engine="v3")
            mgr.record_turn(
                stop_reason="end_turn",
                model="gemini-3.1-pro-high",
                engine="v3",
                duration_s=1.45,
                ttft_s=0.22,
            )
            mgr.emit_log("turn.complete", {"session_id": "sess-1", "status": "ok"})

        mgr.collect_metrics()
        assert capsys.readouterr().err == ""
        assert mgr.buffered_spans_count >= 1
        assert mgr.buffered_logs_count >= 1
        assert collector.count("/v1/traces") == 0
        assert collector.count("/v1/logs") == 0
        assert collector.count("/v1/metrics") == 0

        # Phase 2: Collector comes UP while agy-acp is idle -> next metric tick flushes everything!
        collector.start()
        mgr.collect_metrics()

        assert collector.count("/v1/metrics") >= 1
        assert collector.count("/v1/traces") >= 1
        assert collector.count("/v1/logs") >= 1
        assert mgr.buffered_spans_count == 0
        assert mgr.buffered_logs_count == 0
        assert b"session/prompt" in b"".join(collector.payloads("/v1/traces"))
        assert b"turn.complete" in b"".join(collector.payloads("/v1/logs"))
        assert b"agy_acp.turns" in b"".join(collector.payloads("/v1/metrics"))

        # Phase 3: Collector goes DOWN again -> emit more data -> zero stderr, buffered cleanly
        collector.stop()
        traces_before = collector.count("/v1/traces")
        logs_before = collector.count("/v1/logs")
        metrics_before = collector.count("/v1/metrics")

        with mgr.start_span("agy.tool_call", attributes={"tool.name": "read_file"}):
            mgr.record_tool_call("read_file", kind="read", status="completed", duration_s=0.03)
            mgr.emit_log("tool.done", {"tool": "read_file"})

        mgr.collect_metrics()
        assert capsys.readouterr().err == ""
        assert mgr.buffered_spans_count >= 1
        assert mgr.buffered_logs_count >= 1

        # Phase 4: Collector comes UP again (fresh TCP listener) -> reconnects without stale socket!
        collector.start()
        mgr.collect_metrics()

        assert collector.count("/v1/metrics") > metrics_before
        assert collector.count("/v1/traces") > traces_before
        assert collector.count("/v1/logs") > logs_before
        assert mgr.buffered_spans_count == 0
        assert mgr.buffered_logs_count == 0
        assert b"read_file" in b"".join(collector.payloads("/v1/traces"))
        assert capsys.readouterr().err == ""
    finally:
        mgr.shutdown()
        collector.stop()


def test_offline_ring_buffer_is_strictly_bounded_and_correlates_trace_ids(
    tmp_path: Path,
) -> None:
    """Offline span/log buffers enforce maxlen bounds and preserve trace_id/span_id correlation."""
    port = _reserve_free_port()
    cfg = TelemetryConfig(
        enabled=True,
        endpoint=f"http://127.0.0.1:{port}",
        service_name="agy-acp-bounded",
        disk_logs=False,
        metric_export_interval_ms=60_000,
        export_timeout_seconds=0.3,
        max_buffered_spans=4,
        max_buffered_logs=5,
        config_path=tmp_path / "telemetry.json",
    )
    mgr = TelemetryManager(cfg, use_simple_processors=True)
    try:
        for idx in range(12):
            with mgr.start_span(f"span-{idx}") as span:
                ctx = span.get_span_context()
                mgr.emit_log(f"event-{idx}", {"idx": idx})
                assert ctx.is_valid

        assert mgr.buffered_spans_count == 4
        assert mgr.buffered_logs_count == 5

        latest_span = mgr.peek_buffered_spans()[-1]
        latest_log = mgr.peek_buffered_logs()[-1]
        assert latest_span.name == "span-11"
        assert latest_log.log_record.trace_id == latest_span.context.trace_id
        assert latest_log.log_record.span_id == latest_span.context.span_id
        assert latest_log.log_record.attributes is not None
        assert latest_log.log_record.attributes["event"] == "event-11"
    finally:
        mgr.shutdown()


def test_all_telemetry_metric_instruments_record_and_export(tmp_path: Path) -> None:
    """Verify every supported metric instrument records and serializes without error."""
    port = _reserve_free_port()
    collector = _LocalOtlpCollector(port)
    collector.start()
    cfg = TelemetryConfig(
        enabled=True,
        endpoint=collector.endpoint,
        service_name="agy-acp-metrics",
        disk_logs=False,
        metric_export_interval_ms=60_000,
        export_timeout_seconds=0.5,
        max_buffered_spans=32,
        max_buffered_logs=64,
        config_path=tmp_path / "telemetry.json",
    )
    mgr = TelemetryManager(cfg, use_simple_processors=True)
    mgr.register_callbacks(
        TelemetryCallbacks(
            backend_up=lambda: 1,
            session_counts=lambda: {"active": 1, "persisted": 3},
            quota_metrics=lambda: (
                [{"group": "Gemini Models", "window": "5h", "percent": 92.0}],
                736.0,
                800.0,
            ),
            ledger_tokens=lambda: {"input": 500, "output": 125},
            mcp_tools_discovered=lambda: {"kirocrew-core": 20},
        )
    )
    try:
        with mgr.start_span("turn"):
            mgr.record_rpc_request("session/prompt", duration_s=0.5, engine="v3")
            mgr.record_rpc_error("session/prompt", code=-32603, engine="v3")
            mgr.record_frame("in", 256)
            mgr.record_frame("out", 512)
            mgr.turn_started()
            mgr.turn_finished()
            mgr.record_turn(
                stop_reason="end_turn",
                model="gemini-3.1-pro-high",
                engine="v3",
                duration_s=1.2,
                ttft_s=0.15,
            )
            mgr.record_steer(outcome="applied")
            mgr.record_compaction(trigger="auto", status="completed")
            mgr.record_model_fallback("gemini-3.1-pro-high", "gemini-3-flash", reason="exhausted")
            mgr.record_tokens(
                "gemini-3.1-pro-high",
                input_tokens=100,
                output_tokens=40,
                cached_tokens=20,
                thought_tokens=10,
            )
            mgr.record_credits_consumed("gemini-3.1-pro-high", 1.5)
            mgr.record_context_usage("gemini-3.1-pro-high", used_tokens=50_000, usage_percent=5.0)
            mgr.record_tool_call("run_command", kind="execute", status="completed", duration_s=0.25)
            mgr.record_tool_permission("execute", outcome="selected")
            mgr.record_backend_spawn()
            mgr.record_backend_exit(returncode=0)
            mgr.record_backend_reconnect(reason="eof")
            mgr.record_backend_rpc("conversation/turn", status="ok", duration_s=0.4)
            mgr.record_mcp_probe("kirocrew-core", status="ok", duration_s=0.08)
            mgr.record_quota_fetch("usage", status="ok", duration_s=0.11)
            mgr.record_sqlite_operation("save_session", duration_s=0.002)

        mgr.collect_metrics()
        blob = b"".join(collector.payloads("/v1/metrics"))
        for expected_metric in (
            b"agy_acp.rpc.requests",
            b"agy_acp.rpc.errors",
            b"agy_acp.rpc.duration",
            b"agy_acp.frame.size",
            b"agy_acp.turns",
            b"agy_acp.turns.active",
            b"agy_acp.turn.duration",
            b"agy_acp.turn.ttft",
            b"agy_acp.steers",
            b"agy_acp.compactions",
            b"agy_acp.model.fallbacks",
            b"agy_acp.tokens",
            b"agy_acp.credits.consumed",
            b"agy_acp.context.tokens",
            b"agy_acp.context.usage_percent",
            b"agy_acp.tool.calls",
            b"agy_acp.tool.duration",
            b"agy_acp.tool.permissions",
            b"agy_acp.backend.up",
            b"agy_acp.backend.spawns",
            b"agy_acp.backend.exits",
            b"agy_acp.backend.reconnects",
            b"agy_acp.backend.rpc.duration",
            b"agy_acp.mcp.probes",
            b"agy_acp.mcp.probe.duration",
            b"agy_acp.mcp.tools_discovered",
            b"agy_acp.quota.fetch.duration",
            b"agy_acp.quota.remaining_percent",
            b"agy_acp.quota.remaining_credits",
            b"agy_acp.quota.capacity_credits",
            b"agy_acp.sessions.count",
            b"agy_acp.ledger.tokens",
            b"agy_acp.sqlite.duration",
            b"agy_acp.telemetry.buffered_items",
        ):
            assert expected_metric in blob, f"Missing {expected_metric!r} in exported OTLP metrics"
    finally:
        mgr.shutdown()
        collector.stop()


def test_telemetry_cli_status_enable_and_disable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `agy-acp telemetry status | enable | disable` persists config without env vars."""
    import json

    from agy_acp.cli import main

    monkeypatch.delenv("AGY_ACP_TELEMETRY_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_FILE", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_DISK", raising=False)

    rc_enable = main(
        [
            "telemetry",
            "enable",
            "--endpoint",
            "http://127.0.0.1:14318",
            "--disk-logs",
            "--save-dir",
            str(tmp_path),
        ]
    )
    assert rc_enable == 0
    capsys.readouterr()

    rc_status = main(["telemetry", "status", "--format", "json", "--save-dir", str(tmp_path)])
    assert rc_status == 0
    status_json = json.loads(capsys.readouterr().out)
    assert status_json["enabled"] is True
    assert status_json["endpoint"] == "http://127.0.0.1:14318"
    assert status_json["disk_logs"] is True
    assert status_json["collector_reachable"] is False

    rc_disable = main(["telemetry", "disable", "--no-disk-logs", "--save-dir", str(tmp_path)])
    assert rc_disable == 0
    capsys.readouterr()

    rc_status2 = main(["telemetry", "status", "--format", "json", "--save-dir", str(tmp_path)])
    assert rc_status2 == 0
    status_json2 = json.loads(capsys.readouterr().out)
    assert status_json2["enabled"] is False
    assert status_json2["disk_logs"] is False


async def test_dispatcher_and_session_instrumentation_exports_traces_metrics_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `AcpDispatcher` + `SessionManager` emit correlated OTel spans, metrics, and logs."""
    import json
    from typing import Any

    from agy_acp.logic.acp_dispatcher import AcpDispatcher
    from agy_acp.logic.session_manager import SessionManager
    from agy_acp.logic.tool_classifier import PromptOutcome, ToolCallEvent
    from agy_acp.logic.wire import create_wire_adapter
    from agy_acp.providers import telemetry as tel_mod
    from agy_acp.providers.agy_acp_client import AgyAcpClientProvider
    from agy_acp.providers.state_store import SqliteStateStoreProvider

    port = _reserve_free_port()
    collector = _LocalOtlpCollector(port)
    collector.start()

    monkeypatch.setenv("AGY_ACP_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("AGY_ACP_OTLP_ENDPOINT", collector.endpoint)
    monkeypatch.delenv("AGY_ACP_LOG_FILE", raising=False)
    monkeypatch.delenv("AGY_ACP_LOG_DISK", raising=False)
    tel_mod.reset_telemetry_for_tests()

    mgr = TelemetryManager(
        resolve_telemetry_config(tmp_path),
        use_simple_processors=True,
    )
    monkeypatch.setattr(tel_mod, "_GLOBAL_MANAGER", mgr)

    class _FakeCliProvider(AgyAcpClientProvider):
        async def stream_prompt(self, **kwargs: Any) -> PromptOutcome:  # type: ignore[override]
            on_text = kwargs.get("on_text")
            on_tool_call = kwargs.get("on_tool_call")
            on_tool_update = kwargs.get("on_tool_update")
            if callable(on_tool_call):
                ev = ToolCallEvent(
                    tool_call_id="tc-1",
                    title="read_file",
                    kind="read",
                    status="in_progress",
                    raw_input={"path": "README.md"},
                    output="",
                    locations=[{"path": "README.md"}],
                    kiro_tool_name="fs_read",
                    mcp_server_name="",
                    kiro_tool_id="read_file",
                )
                await on_tool_call(ev)
                if callable(on_tool_update):
                    await on_tool_update(
                        ToolCallEvent(
                            tool_call_id="tc-1",
                            title="read_file",
                            kind="read",
                            status="completed",
                            raw_input={"path": "README.md"},
                            output="ok",
                            locations=[{"path": "README.md"}],
                            kiro_tool_name="fs_read",
                            mcp_server_name="",
                            kiro_tool_id="read_file",
                        )
                    )
            if callable(on_text):
                await on_text("Hello telemetry!")
            return PromptOutcome(
                conversation_id="agy-conv-1",
                stop_reason="end_turn",
                response="Hello telemetry!",
                usage={
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "cache_read_tokens": 10,
                    "thinking_tokens": 5,
                    "total_tokens": 165,
                    "context_tokens": 150,
                },
                resolved_model="gemini-3.1-pro-high",
            )

    try:
        store = SqliteStateStoreProvider(tmp_path)
        cli = _FakeCliProvider(state_store=store)
        adapter = create_wire_adapter("v3")
        sm = SessionManager(state_store=store, cli_provider=cli, wire_adapter=adapter)
        notifs: list[dict[str, object]] = []

        async def _sink(frame: dict[str, object]) -> None:
            notifs.append(frame)

        dispatcher = AcpDispatcher(
            notification_sink=_sink,
            session_manager=sm,
            wire_adapter=adapter,
        )

        init_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": 1, "clientCapabilities": {}},
                }
            )
        )
        assert init_resp is not None and "result" in init_resp

        new_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {"cwd": str(tmp_path), "mcpServers": []},
                }
            )
        )
        assert new_resp is not None
        sid = new_resp["result"]["sessionId"]

        prompt_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": [{"type": "text", "text": "Hi!"}],
                    },
                }
            )
        )
        assert prompt_resp is not None
        assert prompt_resp["result"]["stopReason"] == "end_turn"

        mgr.collect_metrics()
        assert not (tmp_path / "acp.log").exists()

        traces_blob = b"".join(collector.payloads("/v1/traces"))
        logs_blob = b"".join(collector.payloads("/v1/logs"))
        metrics_blob = b"".join(collector.payloads("/v1/metrics"))

        assert b"acp.rpc.initialize" in traces_blob
        assert b"acp.rpc.session/new" in traces_blob
        assert b"acp.rpc.session/prompt" in traces_blob
        assert b"rpc.receive" in logs_blob
        assert b"agy_acp.rpc.requests" in metrics_blob
        assert b"agy_acp.turns" in metrics_blob
        assert b"agy_acp.tokens" in metrics_blob
        assert b"agy_acp.tool.calls" in metrics_blob
        assert b"agy_acp.sqlite.duration" in metrics_blob
    finally:
        tel_mod.reset_telemetry_for_tests()
        collector.stop()


def test_offline_log_buffer_evicts_logs_older_than_twenty_minutes_and_flushes_recent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline log buffer keeps a rolling 20-minute window while spans stay count-capped."""
    monkeypatch.delenv("AGY_ACP_TELEMETRY_ENABLED", raising=False)
    default_cfg = resolve_telemetry_config(tmp_path)
    assert default_cfg.log_retention_seconds == 1200.0
    assert default_cfg.max_buffered_spans == 512

    port = _reserve_free_port()
    collector = _LocalOtlpCollector(port)
    now = [1000.0]

    cfg = TelemetryConfig(
        enabled=True,
        endpoint=collector.endpoint,
        service_name="agy-acp-20m-logs",
        disk_logs=False,
        metric_export_interval_ms=60_000,
        export_timeout_seconds=0.3,
        max_buffered_spans=512,
        max_buffered_logs=16_384,
        log_retention_seconds=1200.0,
        config_path=tmp_path / "telemetry.json",
    )
    mgr = TelemetryManager(cfg, use_simple_processors=True, clock=lambda: now[0])
    try:
        mgr.emit_log("log.t0", {"marker": "expired-at-zero"})
        now[0] += 600.0  # +10m
        mgr.emit_log("log.t10m", {"marker": "expired-at-ten"})
        now[0] += 605.0  # +20m 5s total -> log.t0 is >20m old
        mgr.emit_log("log.t20m", {"marker": "fresh-twenty"})

        assert mgr.buffered_logs_count == 2
        buffered_events = [
            (r.log_record.attributes or {})["event"] for r in mgr.peek_buffered_logs()
        ]
        assert buffered_events == ["log.t10m", "log.t20m"]

        # Advance another 10m 5s without new logs -> log.t10m also ages past 20m
        now[0] += 605.0
        assert mgr.buffered_logs_count == 1

        collector.start()
        mgr.collect_metrics()

        assert mgr.buffered_logs_count == 0
        logs_blob = b"".join(collector.payloads("/v1/logs"))
        assert b"fresh-twenty" in logs_blob
        assert b"expired-at-zero" not in logs_blob
        assert b"expired-at-ten" not in logs_blob
    finally:
        mgr.shutdown()
        collector.stop()
