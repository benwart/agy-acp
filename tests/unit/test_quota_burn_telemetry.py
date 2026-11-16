"""Unit tests for quota pool burn-rate deltas, 5h/7d capacity ratio R, and OTel exemplars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from agy_acp.logic.quota_burn import (
    QuotaBurnLedger,
    classify_model_quota_group,
    normalize_quota_group,
    parse_reset_epoch_seconds,
)
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.providers.quota_provider import QuotaProvider
from agy_acp.providers.state_store import SqliteStateStoreProvider
from agy_acp.providers.telemetry import (
    TelemetryConfig,
    TelemetryManager,
)


def _bucket(
    *,
    group: str,
    window: str,
    remaining_fraction: float,
    reset_time: str,
    credit_capacity: float,
) -> dict[str, Any]:
    return {
        "id": f"{group}-{window}",
        "group": group,
        "name": f"{group} {window}",
        "window": window,
        "remaining_fraction": remaining_fraction,
        "remaining_percent": round(remaining_fraction * 100.0, 2),
        "used_percent": round((1.0 - remaining_fraction) * 100.0, 2),
        "credit_capacity": credit_capacity,
        "remaining_credits": round(remaining_fraction * credit_capacity, 2),
        "reset_time": reset_time,
    }


def test_pool_normalization_and_reset_epoch_parsing() -> None:
    assert normalize_quota_group("Gemini Models") == "google_ai"
    assert normalize_quota_group("google_ai") == "google_ai"
    assert normalize_quota_group("Claude and GPT models") == "anthropic_ai"
    assert normalize_quota_group("anthropic_ai") == "anthropic_ai"

    assert classify_model_quota_group("gemini-4-carbon-medium") == "google_ai"
    assert classify_model_quota_group("gemini-3.1-pro-high") == "google_ai"
    assert classify_model_quota_group("claude-sonnet-4-6") == "anthropic_ai"
    assert classify_model_quota_group("claude-opus-4-6") == "anthropic_ai"
    assert classify_model_quota_group("gpt-oss-120b") == "anthropic_ai"

    assert parse_reset_epoch_seconds("2026-11-15T20:00:00Z") == pytest.approx(1794772800.0)
    assert parse_reset_epoch_seconds("") is None
    assert parse_reset_epoch_seconds("not-a-timestamp") is None


def test_quota_burn_ledger_paired_ratio_reset_immunity_and_floor_clamping() -> None:
    ledger = QuotaBurnLedger()

    # Step 1: Cold seed -> populates gauges & default R=5.0, emits zero burn deltas
    obs0 = ledger.observe_snapshot(
        [
            _bucket(
                group="Gemini Models",
                window="5h",
                remaining_fraction=0.90,
                reset_time="2026-11-15T20:00:00Z",
                credit_capacity=200.0,
            ),
            _bucket(
                group="Claude and GPT models",
                window="5h",
                remaining_fraction=0.80,
                reset_time="2026-11-15T20:00:00Z",
                credit_capacity=200.0,
            ),
        ],
        [
            _bucket(
                group="Gemini Models",
                window="7d",
                remaining_fraction=0.98,
                reset_time="2026-11-22T00:00:00Z",
                credit_capacity=1000.0,
            ),
            _bucket(
                group="Claude and GPT models",
                window="7d",
                remaining_fraction=0.95,
                reset_time="2026-11-22T00:00:00Z",
                credit_capacity=1000.0,
            ),
        ],
        active_model_id="gemini-4-carbon-medium",
    )
    assert obs0.burnt_deltas == []
    assert len(obs0.bucket_gauges) == 4
    assert obs0.ratios_by_group["google_ai"] == pytest.approx(5.0)
    assert obs0.ratios_by_group["anthropic_ai"] == pytest.approx(5.0)

    # Step 2: Anthropic turn burns 16% of 5h (0.80 -> 0.64) and 2% of 7d (0.95 -> 0.93) => R = 8.0
    obs1 = ledger.observe_snapshot(
        [
            _bucket(
                group="Gemini Models",
                window="5h",
                remaining_fraction=0.90,
                reset_time="2026-11-15T20:00:00Z",
                credit_capacity=200.0,
            ),
            _bucket(
                group="Claude and GPT models",
                window="5h",
                remaining_fraction=0.64,
                reset_time="2026-11-15T20:00:00Z",
                credit_capacity=200.0,
            ),
        ],
        [
            _bucket(
                group="Gemini Models",
                window="7d",
                remaining_fraction=0.98,
                reset_time="2026-11-22T00:00:00Z",
                credit_capacity=1000.0,
            ),
            _bucket(
                group="Claude and GPT models",
                window="7d",
                remaining_fraction=0.93,
                reset_time="2026-11-22T00:00:00Z",
                credit_capacity=1000.0,
            ),
        ],
        active_model_id="claude-sonnet-4-6",
    )
    assert len(obs1.burnt_deltas) == 2
    by_win = {d.window: d for d in obs1.burnt_deltas}
    assert by_win["5h"].quota_group == "anthropic_ai"
    assert by_win["5h"].model_id == "claude-sonnet-4-6"
    assert by_win["5h"].fraction_burnt == pytest.approx(0.16)
    assert by_win["5h"].credits_burnt == pytest.approx(32.0)
    assert by_win["7d"].fraction_burnt == pytest.approx(0.02)
    assert by_win["7d"].credits_burnt == pytest.approx(20.0)
    assert obs1.ratios_by_group["anthropic_ai"] == pytest.approx(8.0)

    # Step 3: 5h window resets (0.64 -> 1.00, reset_time +5h) while 7d drops 0.93 -> 0.92
    # Must NOT emit negative 5h delta and must NOT corrupt R=8.0
    obs2 = ledger.observe_snapshot(
        [
            _bucket(
                group="Claude and GPT models",
                window="5h",
                remaining_fraction=1.0,
                reset_time="2026-11-16T01:00:00Z",
                credit_capacity=200.0,
            ),
        ],
        [
            _bucket(
                group="Claude and GPT models",
                window="7d",
                remaining_fraction=0.92,
                reset_time="2026-11-22T00:00:00Z",
                credit_capacity=1000.0,
            ),
        ],
        active_model_id="claude-sonnet-4-6",
    )
    assert [d.window for d in obs2.burnt_deltas] == ["7d"]
    assert obs2.burnt_deltas[0].fraction_burnt == pytest.approx(0.01)
    assert obs2.ratios_by_group["anthropic_ai"] == pytest.approx(8.0)

    # Step 4: 5h clamps at 0.0 (1.00 -> 0.00) while 7d drops -> skipped for R
    obs3 = ledger.observe_snapshot(
        [
            _bucket(
                group="Claude and GPT models",
                window="5h",
                remaining_fraction=0.0,
                reset_time="2026-11-16T01:00:00Z",
                credit_capacity=200.0,
            ),
        ],
        [
            _bucket(
                group="Claude and GPT models",
                window="7d",
                remaining_fraction=0.72,
                reset_time="2026-11-22T00:00:00Z",
                credit_capacity=1000.0,
            ),
        ],
        active_model_id="claude-sonnet-4-6",
    )
    assert len(obs3.burnt_deltas) == 2
    assert obs3.ratios_by_group["anthropic_ai"] == pytest.approx(8.0)


async def test_quota_tracker_persists_burn_ratio_and_attaches_turn_trace_exemplars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agy_acp.providers import telemetry as tel_mod

    tel_mod.reset_telemetry_for_tests()
    reader = InMemoryMetricReader()
    tp = TracerProvider()
    mp = MeterProvider(metric_readers=[reader])

    cfg = TelemetryConfig(
        enabled=True,
        endpoint="http://127.0.0.1:4318",
        service_name="agy-acp-exemplar-test",
        disk_logs=False,
        metric_export_interval_ms=60_000,
        export_timeout_seconds=0.2,
        max_buffered_spans=16,
        max_buffered_logs=16,
        config_path=tmp_path / "telemetry.json",
    )
    mgr = TelemetryManager(cfg, use_simple_processors=True)
    mgr._tracer = tp.get_tracer("agy_acp")
    mgr._init_instruments(mp)
    monkeypatch.setattr(tel_mod, "_GLOBAL_MANAGER", mgr)

    snapshots = [
        [
            {
                "group": "Gemini Models",
                "buckets": [
                    {
                        "id": "g-5h",
                        "group": "Gemini Models",
                        "name": "Gemini 5h",
                        "window": "5h",
                        "remaining_fraction": 0.90,
                        "reset_time": "2026-11-15T20:00:00Z",
                    },
                    {
                        "id": "g-7d",
                        "group": "Gemini Models",
                        "name": "Gemini 7d",
                        "window": "weekly",
                        "remaining_fraction": 0.98,
                        "reset_time": "2026-11-22T00:00:00Z",
                    },
                ],
            }
        ],
        [
            {
                "group": "Gemini Models",
                "buckets": [
                    {
                        "id": "g-5h",
                        "group": "Gemini Models",
                        "name": "Gemini 5h",
                        "window": "5h",
                        "remaining_fraction": 0.80,
                        "reset_time": "2026-11-15T20:00:00Z",
                    },
                    {
                        "id": "g-7d",
                        "group": "Gemini Models",
                        "name": "Gemini 7d",
                        "window": "weekly",
                        "remaining_fraction": 0.96,
                        "reset_time": "2026-11-22T00:00:00Z",
                    },
                ],
            }
        ],
    ]

    class _FakeQuotaProvider(QuotaProvider):
        def __init__(self) -> None:
            super().__init__()
            self.idx = 0

        async def fetch_quota_groups(self) -> list[dict[str, Any]]:
            val = snapshots[min(self.idx, len(snapshots) - 1)]
            self.idx += 1
            return val

        async def fetch_credits(self) -> dict[str, Any]:
            return {"remaining_credits": 25, "upgrade_uri": None}

        async def fetch_tier(self) -> dict[str, str]:
            return {"id": "g1-ultra-lite-tier", "name": "Google AI Ultra (5x)", "upgradeUri": ""}

    try:
        store = SqliteStateStoreProvider(tmp_path)
        provider = _FakeQuotaProvider()
        tracker = QuotaTracker(provider=provider, state_store=store)

        # Seed initial snapshot
        await tracker.refresh_snapshot(seed_session_ids=("s1",), model_id="gemini-4-carbon-medium")

        # Simulate prompt turn span and async refresh capturing span context for exemplars
        with mgr.start_span("acp.rpc.session/prompt") as turn_span:
            expected_trace_id = turn_span.get_span_context().trace_id
            turn_ctx = mgr.current_context()
            tracker.build_cached_turn_telemetry(
                session_id="s1",
                model_id="gemini-4-carbon-medium",
                input_tokens=500,
                output_tokens=150,
                context_tokens=10_000,
            )

        # Even after turn_span exits, refresh_snapshot carrying turn_ctx attaches exemplars!
        await tracker.refresh_snapshot(
            seed_session_ids=("s1",),
            model_id="gemini-4-carbon-medium",
            otel_context=turn_ctx,
        )

        md = reader.get_metrics_data()
        assert md is not None
        by_name = {
            metric.name: metric
            for rm in md.resource_metrics
            for sm in rm.scope_metrics
            for metric in sm.metrics
        }

        for counter_name in ("agy_acp.quota.fraction_burnt", "agy_acp.credits.burnt"):
            assert counter_name in by_name
            dps = list(by_name[counter_name].data.data_points)
            assert len(dps) >= 2
            exemplar_trace_ids = {ex.trace_id for dp in dps for ex in getattr(dp, "exemplars", ())}
            assert expected_trace_id in exemplar_trace_ids

        # Verify persisted R=5.0 (0.10 / 0.02 = 5.0) survives new QuotaTracker instance
        reloaded = QuotaTracker(provider=provider, state_store=store)
        assert reloaded.get_empirical_5h_per_7d_ratios()["google_ai"] == pytest.approx(5.0)
    finally:
        tel_mod.reset_telemetry_for_tests()
        mp.shutdown()
        tp.shutdown()


def test_telemetry_cli_up_down_logs_and_extended_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json
    import subprocess

    from agy_acp.cli import main

    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")

    recorded_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        recorded_cmds.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("AGY_ACP_GRAFANA_PORT", "13000")

    assert main(["telemetry", "up", "--compose-file", str(compose_file)]) == 0
    out_up = capsys.readouterr().out
    assert "http://127.0.0.1:13000" in out_up
    assert recorded_cmds[-1] == ["docker", "compose", "-f", str(compose_file), "up", "-d"]

    assert (
        main(
            [
                "telemetry",
                "logs",
                "--service",
                "grafana",
                "--tail",
                "25",
                "--compose-file",
                str(compose_file),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert recorded_cmds[-1] == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "logs",
        "--tail",
        "25",
        "grafana",
    ]

    assert main(["telemetry", "down", "--volumes", "--compose-file", str(compose_file)]) == 0
    capsys.readouterr()
    assert recorded_cmds[-1] == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "down",
        "--volumes",
    ]

    assert main(["telemetry", "status", "--format", "json", "--save-dir", str(tmp_path)]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["grafana_url"] == "http://127.0.0.1:13000"
    assert set(status_payload["services"].keys()) == {
        "otel_collector",
        "prometheus",
        "tempo",
        "loki",
        "grafana",
    }
