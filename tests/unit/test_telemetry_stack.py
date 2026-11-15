"""Contract tests for the Docker Compose OTel observability stack and Grafana dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader, Sum

from agy_acp.providers.telemetry import (
    SUPPORTED_METRIC_NAMES,
    TelemetryCallbacks,
    TelemetryConfig,
    TelemetryManager,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
TELEMETRY_DIR = REPO_ROOT / "telemetry"
OTEL_COLLECTOR_CFG = TELEMETRY_DIR / "otel-collector-config.yaml"
PROMETHEUS_CFG = TELEMETRY_DIR / "prometheus.yml"
TEMPO_CFG = TELEMETRY_DIR / "tempo.yaml"
LOKI_CFG = TELEMETRY_DIR / "loki-config.yaml"
GRAFANA_DATASOURCES_CFG = (
    TELEMETRY_DIR / "grafana" / "provisioning" / "datasources" / "datasources.yml"
)
GRAFANA_DASHBOARDS_CFG = (
    TELEMETRY_DIR / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
)
GRAFANA_DASHBOARD_JSON = TELEMETRY_DIR / "grafana" / "dashboards" / "agy-acp-overview.json"

_UCUM_UNIT_MAP: dict[str, str] = {
    "d": "days",
    "h": "hours",
    "min": "minutes",
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
    "ns": "nanoseconds",
    "By": "bytes",
    "1": "ratio",
    "%": "percent",
}


def _translate_otel_metric_to_prom_series(
    name: str,
    unit: str | None,
    data: Any,
) -> tuple[str, ...]:
    """Replicate OpenTelemetry Collector `pkg/translator/prometheus` metric naming."""
    tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", name) if tok]
    clean_unit = re.sub(r"\{[^}]*\}", "", unit or "").strip()
    unit_suffix = _UCUM_UNIT_MAP.get(clean_unit, clean_unit)
    if unit_suffix and tokens[-1] != unit_suffix:
        tokens.append(unit_suffix)
    base = "_".join(tokens)

    if isinstance(data, Histogram):
        return (f"{base}_bucket", f"{base}_sum", f"{base}_count")
    if isinstance(data, Sum) and data.is_monotonic:
        if not base.endswith("_total"):
            return (f"{base}_total",)
        return (base,)
    return (base,)


def _collect_all_34_sdk_metrics(
    tmp_path: Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]]]:
    """Record one sample on all 34 `TelemetryManager` instruments and map to PromQL names."""
    from opentelemetry.sdk.metrics import MeterProvider

    cfg = TelemetryConfig(
        enabled=True,
        endpoint="http://127.0.0.1:4318",
        service_name="agy-acp",
        disk_logs=False,
        metric_export_interval_ms=60_000,
        export_timeout_seconds=0.2,
        max_buffered_spans=16,
        max_buffered_logs=16,
        config_path=tmp_path / "telemetry.json",
    )
    tm = TelemetryManager(cfg, use_simple_processors=True)
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])
    tm._init_instruments(mp)
    tm.register_callbacks(
        TelemetryCallbacks(
            backend_up=lambda: True,
            session_counts=lambda: {"active": 1, "persisted": 2},
            quota_metrics=lambda: (
                [{"group": "claude", "window": "5h", "percent": 80.0}],
                450.0,
                500.0,
            ),
            ledger_tokens=lambda: {"input": 100, "output": 50},
            mcp_tools_discovered=lambda: {"kirocrew-core": 12},
        )
    )

    tm.record_rpc_request("initialize", duration_s=0.01, engine="v2")
    tm.record_rpc_error("initialize", code=-32600, engine="v2")
    tm.record_frame("inbound", 128)
    tm.turn_started()
    tm.turn_finished()
    tm.record_turn(
        stop_reason="end_turn",
        model="claude-sonnet-4-6",
        engine="v2",
        duration_s=0.5,
        ttft_s=0.12,
    )
    tm.record_steer(outcome="accepted")
    tm.record_compaction(trigger="auto", status="completed")
    tm.record_model_fallback(
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        reason="quota_exhausted",
    )
    tm.record_tokens("claude-sonnet-4-6", input_tokens=100, output_tokens=50)
    tm.record_credits_consumed("claude-sonnet-4-6", 1.5)
    tm.record_credits_burnt("anthropic_ai", window="5h", model_id="claude-sonnet-4-6", credits=1.5)
    tm.record_quota_fraction_burnt(
        "anthropic_ai",
        window="5h",
        model_id="claude-sonnet-4-6",
        fraction=0.01,
    )
    tm.record_context_usage("claude-sonnet-4-6", used_tokens=4000, usage_percent=2.0)
    tm.record_tool_call("read_file", kind="read", status="completed", duration_s=0.02)
    tm.record_tool_permission("read", outcome="allow_once")
    tm.record_backend_spawn()
    tm.record_backend_exit(returncode=0)
    tm.record_backend_reconnect(reason="crash_recovery")
    tm.record_backend_rpc("session/new", status="ok", duration_s=0.03)
    tm.record_mcp_probe("kirocrew-core", status="ok", duration_s=0.04)
    tm.record_quota_fetch("usage", status="ok", duration_s=0.15)
    tm.record_sqlite_operation("upsert_session", duration_s=0.002)

    metrics_data = reader.get_metrics_data()
    tm.shutdown()
    mp.shutdown()

    mapped: dict[str, tuple[str, ...]] = {}
    series_labels: dict[str, set[str]] = {}
    assert metrics_data is not None
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                prom_series = _translate_otel_metric_to_prom_series(
                    metric.name,
                    metric.unit,
                    metric.data,
                )
                mapped[metric.name] = prom_series
                attrs: set[str] = {"le"} if isinstance(metric.data, Histogram) else set()
                for dp in metric.data.data_points:
                    if dp.attributes:
                        attrs.update(str(k) for k in dp.attributes.keys())
                for s_name in prom_series:
                    series_labels[s_name] = attrs
    return mapped, series_labels


def test_docker_compose_binds_localhost_only_and_defines_full_stack() -> None:
    assert DOCKER_COMPOSE_PATH.is_file(), f"Missing {DOCKER_COMPOSE_PATH}"
    compose = yaml.safe_load(DOCKER_COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    expected_services = {"otel-collector", "prometheus", "tempo", "loki", "grafana"}
    assert expected_services.issubset(services.keys())

    published_ports: list[str] = []
    for svc_name, svc in services.items():
        for port in svc.get("ports", []):
            port_str = str(port)
            assert port_str.startswith("127.0.0.1:"), (
                f"Service {svc_name} exposes non-loopback port mapping: {port_str}"
            )
            published_ports.append(port_str)

    assert any(":4318" in p for p in published_ports)
    assert any(":3000" in p for p in published_ports)


def test_telemetry_configs_valid_and_cross_linked() -> None:
    for cfg_path in (
        OTEL_COLLECTOR_CFG,
        PROMETHEUS_CFG,
        TEMPO_CFG,
        LOKI_CFG,
        GRAFANA_DATASOURCES_CFG,
        GRAFANA_DASHBOARDS_CFG,
    ):
        assert cfg_path.is_file(), f"Missing config file {cfg_path}"
        parsed = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), f"Invalid YAML root in {cfg_path}"

    collector = yaml.safe_load(OTEL_COLLECTOR_CFG.read_text(encoding="utf-8"))
    pipelines = collector["service"]["pipelines"]
    assert "traces" in pipelines
    assert "metrics" in pipelines
    assert "logs" in pipelines
    prom_exp = collector["exporters"]["prometheus"]
    assert prom_exp.get("enable_open_metrics") is True
    assert prom_exp.get("resource_to_telemetry_conversion", {}).get("enabled") is True

    loki = yaml.safe_load(LOKI_CFG.read_text(encoding="utf-8"))
    assert loki.get("limits_config", {}).get("allow_structured_metadata") is True

    datasources_doc = yaml.safe_load(GRAFANA_DATASOURCES_CFG.read_text(encoding="utf-8"))
    by_uid = {ds["uid"]: ds for ds in datasources_doc["datasources"]}
    assert {"prometheus", "tempo", "loki"} == set(by_uid.keys())

    prom_ds = by_uid["prometheus"]
    exemplar_dests = prom_ds["jsonData"]["exemplarTraceIdDestinations"]
    assert any(d["datasourceUid"] == "tempo" for d in exemplar_dests)

    loki_ds = by_uid["loki"]
    derived = loki_ds["jsonData"]["derivedFields"]
    assert any(d["datasourceUid"] == "tempo" for d in derived)

    tempo_ds = by_uid["tempo"]
    assert tempo_ds["jsonData"]["tracesToLogsV2"]["datasourceUid"] == "loki"
    assert tempo_ds["jsonData"]["tracesToMetrics"]["datasourceUid"] == "prometheus"


def _iter_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for panel in panels:
        flat.append(panel)
        if "panels" in panel and isinstance(panel["panels"], list):
            flat.extend(_iter_panels(panel["panels"]))
    return flat


def test_grafana_dashboard_covers_all_34_metrics_exemplars_and_deep_links(
    tmp_path: Path,
) -> None:
    assert GRAFANA_DASHBOARD_JSON.is_file(), f"Missing {GRAFANA_DASHBOARD_JSON}"
    dashboard = json.loads(GRAFANA_DASHBOARD_JSON.read_text(encoding="utf-8"))
    assert dashboard["uid"] == "agy-acp-overview"

    panels = _iter_panels(dashboard.get("panels", []))
    assert len(panels) >= 20

    panel_ids = [p["id"] for p in panels if "id" in p]
    assert len(panel_ids) == len(set(panel_ids)), "Duplicate Grafana panel IDs found"

    # Dashboard-level and panel-level links to correlated Loki logs and Tempo traces
    dash_links = json.dumps(dashboard.get("links", []))
    assert "loki" in dash_links.lower()
    assert "tempo" in dash_links.lower()

    all_panel_links = [
        link
        for p in panels
        for link in (
            *p.get("links", []),
            *p.get("fieldConfig", {}).get("defaults", {}).get("links", []),
        )
    ]
    links_blob = json.dumps(all_panel_links)
    assert "/explore" in links_blob
    assert "loki" in links_blob
    assert "tempo" in links_blob

    # Embedded Loki logs panel and Tempo trace query panel
    assert any(p.get("type") == "logs" for p in panels), "Dashboard must embed a Loki logs panel"
    assert any(p.get("datasource", {}).get("uid") == "tempo" for p in panels), (
        "Dashboard must embed a Tempo traces panel"
    )

    # Collect all PromQL expressions and verify exemplars on latency histograms
    prom_targets = [
        t
        for p in panels
        if p.get("datasource", {}).get("uid") == "prometheus"
        for t in p.get("targets", [])
        if isinstance(t, dict) and "expr" in t
    ]
    all_exprs = "\n".join(str(t["expr"]) for t in prom_targets)
    exemplar_exprs = [str(t["expr"]) for t in prom_targets if t.get("exemplar") is True]
    assert len(exemplar_exprs) >= 10, (
        "Counter and latency histogram panels should enable trace exemplars"
    )
    assert any("agy_acp_credits_burnt_total" in e for e in exemplar_exprs)
    assert any("agy_acp_quota_fraction_burnt_total" in e for e in exemplar_exprs)
    assert any("agy_acp_tokens_total" in e for e in exemplar_exprs)

    # Verify quota burn pace, 5h/7d capacity ratio R, and predict_linear forecast panels exist
    assert "predict_linear(" in all_exprs
    assert "agy_acp_quota_5h_per_7d_ratio" in all_exprs
    assert "agy_acp_quota_reset_timestamp_seconds" in all_exprs
    assert "agy_acp_g1_overage_credits" in all_exprs

    # Verify all 43 SDK instruments map cleanly and appear in dashboard PromQL
    sdk_prom_map, series_labels = _collect_all_34_sdk_metrics(tmp_path)
    assert set(sdk_prom_map.keys()) == set(SUPPORTED_METRIC_NAMES)
    assert len(SUPPORTED_METRIC_NAMES) == 43

    missing_metrics: list[str] = []
    for otel_name in SUPPORTED_METRIC_NAMES:
        candidate_series = sdk_prom_map[otel_name]
        if not any(series in all_exprs for series in candidate_series):
            missing_metrics.append(f"{otel_name} -> {candidate_series}")

    assert not missing_metrics, (
        f"Grafana dashboard is missing PromQL queries for metrics: {missing_metrics}"
    )

    # Verify every `by (...)` label clause references real emitted OTel metric attributes
    for target in prom_targets:
        expr = str(target["expr"])
        matched_series = [s for s in series_labels if s in expr]
        assert matched_series, f"Expression {expr!r} did not match any known SDK metric series"
        allowed_labels = set().union(*(series_labels[s] for s in matched_series))
        for by_match in re.finditer(r"\bby\s*\(([^)]+)\)", expr):
            requested_labels = {lbl.strip() for lbl in by_match.group(1).split(",") if lbl.strip()}
            unknown = requested_labels - allowed_labels
            assert not unknown, (
                f"PromQL {expr!r} groups by non-existent labels {unknown}; "
                f"emitted labels are {allowed_labels}"
            )
