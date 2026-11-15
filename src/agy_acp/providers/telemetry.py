"""OpenTelemetry SDK facade and offline-resilient OTLP HTTP exporters for agy-acp."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from agy_acp import KIRO_CLI_COMPAT_VERSION, __version__
from agy_acp.logic.quota_burn import (
    DEFAULT_5H_PER_7D_RATIO,
    classify_model_quota_group,
    normalize_quota_group,
    parse_reset_epoch_seconds,
)
from agy_acp.providers.state_store import resolve_state_db_path

DEFAULT_OTLP_ENDPOINT = "http://127.0.0.1:4318"
DEFAULT_SERVICE_NAME = "agy-acp"
DEFAULT_METRIC_EXPORT_INTERVAL_MS = 5_000
DEFAULT_EXPORT_TIMEOUT_SECONDS = 0.75
DEFAULT_MAX_BUFFERED_SPANS = 512
DEFAULT_MAX_BUFFERED_LOGS = 16_384
DEFAULT_LOG_RETENTION_SECONDS = 1200.0

_TRUTHY_STRINGS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSY_STRINGS: frozenset[str] = frozenset({"0", "false", "no", "off"})

DURATION_SECONDS_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)
FRAME_SIZE_BYTES_BUCKETS: tuple[float, ...] = (
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
    262144.0,
    1048576.0,
    4194304.0,
)
CONTEXT_TOKENS_BUCKETS: tuple[float, ...] = (
    1_000.0,
    5_000.0,
    10_000.0,
    25_000.0,
    50_000.0,
    100_000.0,
    200_000.0,
    400_000.0,
    800_000.0,
    1_000_000.0,
)
PERCENT_BUCKETS: tuple[float, ...] = (
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    95.0,
    100.0,
)
RATIO_BUCKETS: tuple[float, ...] = (
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    1.0,
)

SUPPORTED_METRIC_NAMES: tuple[str, ...] = (
    "agy_acp.rpc.requests",
    "agy_acp.rpc.errors",
    "agy_acp.rpc.duration",
    "agy_acp.frame.size",
    "agy_acp.turns",
    "agy_acp.turns.active",
    "agy_acp.turn.duration",
    "agy_acp.turn.ttft",
    "agy_acp.steers",
    "agy_acp.compactions",
    "agy_acp.model.fallbacks",
    "agy_acp.tokens",
    "agy_acp.credits.consumed",
    "agy_acp.credits.burnt",
    "agy_acp.context.tokens",
    "agy_acp.context.usage_percent",
    "agy_acp.context.window_utilization",
    "agy_acp.tool.calls",
    "agy_acp.tool.duration",
    "agy_acp.tool.permissions",
    "agy_acp.backend.up",
    "agy_acp.backend.spawns",
    "agy_acp.backend.exits",
    "agy_acp.backend.reconnects",
    "agy_acp.backend.rpc.duration",
    "agy_acp.mcp.probes",
    "agy_acp.mcp.probe.duration",
    "agy_acp.mcp.tools_discovered",
    "agy_acp.quota.fetch.duration",
    "agy_acp.quota.remaining_percent",
    "agy_acp.quota.remaining_fraction",
    "agy_acp.quota.used_fraction",
    "agy_acp.quota.remaining_credits",
    "agy_acp.quota.bucket_remaining_credits",
    "agy_acp.quota.capacity_credits",
    "agy_acp.quota.reset_timestamp",
    "agy_acp.quota.5h_per_7d_ratio",
    "agy_acp.quota.fraction_burnt",
    "agy_acp.g1.overage_credits",
    "agy_acp.sessions.count",
    "agy_acp.ledger.tokens",
    "agy_acp.sqlite.duration",
    "agy_acp.telemetry.buffered_items",
)


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Resolved runtime configuration for OpenTelemetry export and diagnostic disk logging."""

    enabled: bool
    endpoint: str
    service_name: str
    disk_logs: bool
    metric_export_interval_ms: int
    export_timeout_seconds: float
    max_buffered_spans: int
    max_buffered_logs: int
    config_path: Path
    log_retention_seconds: float = DEFAULT_LOG_RETENTION_SECONDS


@dataclass(slots=True)
class TelemetryCallbacks:
    """Live gauge callbacks registered by `SessionManager` and `QuotaTracker`."""

    backend_up: Callable[[], int] | None = None
    session_counts: Callable[[], Mapping[str, int]] | None = None
    quota_metrics: (
        Callable[[], tuple[Sequence[Mapping[str, Any]], float | None, float | None]] | None
    ) = None
    quota_extended_metrics: Callable[[], Mapping[str, Any]] | None = None
    ledger_tokens: Callable[[], Mapping[str, int]] | None = None
    mcp_tools_discovered: Callable[[], Mapping[str, int]] | None = None


@contextmanager
def _bind_optional_context(ctx: Context | None) -> Generator[None]:
    if ctx is None:
        yield
        return
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def _parse_bool_env(raw: str | None) -> bool | None:
    if raw is None:
        return None
    cleaned = raw.strip().lower()
    if cleaned in _TRUTHY_STRINGS:
        return True
    if cleaned in _FALSY_STRINGS:
        return False
    return None


def _normalize_endpoint(raw_endpoint: str) -> str:
    cleaned = raw_endpoint.strip().rstrip("/")
    return cleaned or DEFAULT_OTLP_ENDPOINT


def resolve_telemetry_config_path(save_dir_or_db: Path | str | None = None) -> Path:
    """Resolve `telemetry.json` path from `$AGY_ACP_TELEMETRY_CONFIG` or state directory."""
    env_cfg = os.environ.get("AGY_ACP_TELEMETRY_CONFIG")
    if env_cfg:
        return Path(env_cfg).expanduser()
    return resolve_state_db_path(save_dir_or_db).with_name("telemetry.json")


def load_persisted_telemetry_settings(
    save_dir_or_db: Path | str | None = None,
) -> dict[str, Any]:
    """Load persisted telemetry preferences from `telemetry.json`, returning `{}` if absent."""
    cfg_path = resolve_telemetry_config_path(save_dir_or_db)
    if not cfg_path.is_file():
        return {}
    with contextlib.suppress(OSError, json.JSONDecodeError, ValueError):
        parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            return parsed
    return {}


def save_persisted_telemetry_settings(
    settings: Mapping[str, Any],
    *,
    save_dir_or_db: Path | str | None = None,
) -> Path:
    """Persist telemetry settings into `telemetry.json` with `0o600` permissions."""
    cfg_path = resolve_telemetry_config_path(save_dir_or_db)
    existing = load_persisted_telemetry_settings(save_dir_or_db)
    merged: dict[str, Any] = {**existing}
    for key, val in settings.items():
        if val is not None:
            if key == "endpoint" and isinstance(val, str):
                merged[key] = _normalize_endpoint(val)
            else:
                merged[key] = val
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(merged, indent=2, sort_keys=True) + "\n"
    fd = os.open(cfg_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, serialized.encode("utf-8"))
    finally:
        os.close(fd)
    return cfg_path


def resolve_telemetry_config(save_dir_or_db: Path | str | None = None) -> TelemetryConfig:
    """Resolve effective `TelemetryConfig` from env overrides, `telemetry.json`, and defaults."""
    cfg_path = resolve_telemetry_config_path(save_dir_or_db)
    persisted = load_persisted_telemetry_settings(save_dir_or_db)

    enabled = bool(persisted.get("enabled", True))
    env_enabled = _parse_bool_env(os.environ.get("AGY_ACP_TELEMETRY_ENABLED"))
    if env_enabled is not None:
        enabled = env_enabled
    otel_disabled = _parse_bool_env(os.environ.get("OTEL_SDK_DISABLED"))
    if otel_disabled is True:
        enabled = False

    raw_endpoint = (
        os.environ.get("AGY_ACP_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or str(persisted.get("endpoint") or DEFAULT_OTLP_ENDPOINT)
    )
    endpoint = _normalize_endpoint(raw_endpoint)

    service_name = (
        os.environ.get("OTEL_SERVICE_NAME")
        or str(persisted.get("service_name") or DEFAULT_SERVICE_NAME)
    ).strip() or DEFAULT_SERVICE_NAME

    disk_logs = bool(persisted.get("disk_logs", False))
    env_disk_logs = _parse_bool_env(os.environ.get("AGY_ACP_LOG_DISK"))
    if env_disk_logs is not None:
        disk_logs = env_disk_logs
    if os.environ.get("AGY_ACP_LOG_FILE"):
        disk_logs = True

    interval_ms = DEFAULT_METRIC_EXPORT_INTERVAL_MS
    raw_interval = os.environ.get("AGY_ACP_METRIC_EXPORT_INTERVAL_MS")
    if raw_interval:
        with contextlib.suppress(ValueError):
            interval_ms = max(250, int(raw_interval))

    log_retention_s = DEFAULT_LOG_RETENTION_SECONDS
    raw_retention = os.environ.get("AGY_ACP_LOG_RETENTION_SECONDS") or persisted.get(
        "log_retention_seconds"
    )
    if raw_retention is not None:
        with contextlib.suppress(TypeError, ValueError):
            log_retention_s = max(1.0, float(raw_retention))

    return TelemetryConfig(
        enabled=enabled,
        endpoint=endpoint,
        service_name=service_name,
        disk_logs=disk_logs,
        metric_export_interval_ms=interval_ms,
        export_timeout_seconds=DEFAULT_EXPORT_TIMEOUT_SECONDS,
        max_buffered_spans=DEFAULT_MAX_BUFFERED_SPANS,
        max_buffered_logs=DEFAULT_MAX_BUFFERED_LOGS,
        config_path=cfg_path,
        log_retention_seconds=log_retention_s,
    )


def _silence_otel_internal_loggers() -> None:
    """Prevent OpenTelemetry and urllib3 retry warnings from leaking onto ACP stdio `stderr`."""
    if os.environ.get("AGY_ACP_OTEL_DEBUG") == "1":
        return
    for logger_name in ("opentelemetry", "urllib3"):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
            logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)


class EndpointReachabilityProbe:
    """Fast non-blocking TCP probe checking if the OTLP HTTP collector port is accepting sockets."""

    def __init__(self, endpoint: str, *, connect_timeout: float = 0.08) -> None:
        parsed = urlparse(endpoint)
        self.host = parsed.hostname or "127.0.0.1"
        default_port = 443 if parsed.scheme == "https" else 4318
        self.port = parsed.port or default_port
        self.connect_timeout = connect_timeout

    def is_reachable(self) -> bool:
        """Return True if `(host, port)` accepts a TCP connection within `connect_timeout`."""
        try:
            with socket.create_connection((self.host, self.port), timeout=self.connect_timeout):
                return True
        except OSError:
            return False


def _new_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Connection": "close"})
    return session


class ResilientSpanExporter(SpanExporter):
    """OTLP HTTP span exporter with bounded offline ring buffer and seamless reconnect."""

    def __init__(
        self,
        *,
        endpoint: str,
        probe: EndpointReachabilityProbe,
        timeout: float,
        max_buffered_spans: int,
    ) -> None:
        self._traces_url = f"{_normalize_endpoint(endpoint)}/v1/traces"
        self._probe = probe
        self._timeout = timeout
        self._buffer: deque[ReadableSpan] = deque(maxlen=max_buffered_spans)
        self._lock = threading.Lock()
        self._session: requests.Session | None = None
        self._delegate: OTLPSpanExporter | None = None
        self._shutdown = False

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def peek_buffered(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._buffer)

    def _ensure_delegate(self) -> OTLPSpanExporter:
        if self._delegate is None:
            self._session = _new_http_session()
            self._delegate = OTLPSpanExporter(
                endpoint=self._traces_url,
                timeout=self._timeout,
                session=self._session,
            )
        return self._delegate

    def _reset_delegate(self) -> None:
        if self._delegate is not None:
            with contextlib.suppress(Exception):
                self._delegate.shutdown()
            self._delegate = None
        if self._session is not None:
            with contextlib.suppress(Exception):
                self._session.close()
            self._session = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            if self._shutdown:
                return SpanExportResult.SUCCESS
            if not self._probe.is_reachable():
                self._reset_delegate()
                self._buffer.extend(spans)
                return SpanExportResult.SUCCESS
            batch = [*self._buffer, *spans]
            self._buffer.clear()
            return self._send_locked(batch)

    def flush_backlog(self) -> bool:
        """Export any buffered spans when the collector becomes reachable."""
        with self._lock:
            if self._shutdown or not self._buffer:
                return True
            if not self._probe.is_reachable():
                self._reset_delegate()
                return False
            batch = list(self._buffer)
            self._buffer.clear()
            self._send_locked(batch)
            return len(self._buffer) == 0

    def _send_locked(self, batch: list[ReadableSpan]) -> SpanExportResult:
        if not batch:
            return SpanExportResult.SUCCESS
        delegate = self._ensure_delegate()
        try:
            res = delegate.export(batch)
        except Exception:
            res = SpanExportResult.FAILURE
        if res is not SpanExportResult.SUCCESS:
            self._reset_delegate()
            self._buffer.extend(batch)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return self.flush_backlog()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._reset_delegate()


class ResilientLogExporter(LogRecordExporter):
    """OTLP HTTP log exporter with rolling time-window offline buffer and seamless reconnect."""

    def __init__(
        self,
        *,
        endpoint: str,
        probe: EndpointReachabilityProbe,
        timeout: float,
        max_buffered_logs: int,
        retention_seconds: float = DEFAULT_LOG_RETENTION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logs_url = f"{_normalize_endpoint(endpoint)}/v1/logs"
        self._probe = probe
        self._timeout = timeout
        self._retention_seconds = max(1.0, float(retention_seconds))
        self._clock = clock
        self._buffer: deque[tuple[float, ReadableLogRecord]] = deque(maxlen=max_buffered_logs)
        self._lock = threading.Lock()
        self._session: requests.Session | None = None
        self._delegate: OTLPLogExporter | None = None
        self._shutdown = False

    def _prune_expired_locked(self) -> None:
        cutoff = self._clock() - self._retention_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    @property
    def buffered_count(self) -> int:
        with self._lock:
            self._prune_expired_locked()
            return len(self._buffer)

    def peek_buffered(self) -> list[ReadableLogRecord]:
        with self._lock:
            self._prune_expired_locked()
            return [record for _, record in self._buffer]

    def _ensure_delegate(self) -> OTLPLogExporter:
        if self._delegate is None:
            self._session = _new_http_session()
            self._delegate = OTLPLogExporter(
                endpoint=self._logs_url,
                timeout=self._timeout,
                session=self._session,
            )
        return self._delegate

    def _reset_delegate(self) -> None:
        if self._delegate is not None:
            with contextlib.suppress(Exception):
                self._delegate.shutdown()
            self._delegate = None
        if self._session is not None:
            with contextlib.suppress(Exception):
                self._session.close()
            self._session = None

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        with self._lock:
            if self._shutdown:
                return LogRecordExportResult.SUCCESS
            self._prune_expired_locked()
            now = self._clock()
            stamped_batch = [(now, rec) for rec in batch]
            if not self._probe.is_reachable():
                self._reset_delegate()
                self._buffer.extend(stamped_batch)
                return LogRecordExportResult.SUCCESS
            combined = [*self._buffer, *stamped_batch]
            self._buffer.clear()
            return self._send_locked(combined)

    def flush_backlog(self) -> bool:
        """Export any non-expired buffered logs when the collector becomes reachable."""
        with self._lock:
            self._prune_expired_locked()
            if self._shutdown or not self._buffer:
                return True
            if not self._probe.is_reachable():
                self._reset_delegate()
                return False
            combined = list(self._buffer)
            self._buffer.clear()
            self._send_locked(combined)
            return len(self._buffer) == 0

    def _send_locked(
        self,
        stamped_batch: list[tuple[float, ReadableLogRecord]],
    ) -> LogRecordExportResult:
        if not stamped_batch:
            return LogRecordExportResult.SUCCESS
        delegate = self._ensure_delegate()
        records = [rec for _, rec in stamped_batch]
        try:
            res = delegate.export(records)
        except Exception:
            res = LogRecordExportResult.FAILURE
        if res is not LogRecordExportResult.SUCCESS:
            self._reset_delegate()
            self._buffer.extend(stamped_batch)
            self._prune_expired_locked()
        return LogRecordExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return self.flush_backlog()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._reset_delegate()


class ResilientMetricExporter(MetricExporter):
    """OTLP HTTP metric exporter that retains cumulative totals offline and flushes backlogs."""

    def __init__(
        self,
        *,
        endpoint: str,
        probe: EndpointReachabilityProbe,
        timeout: float,
        on_collector_online: Callable[[], None] | None = None,
    ) -> None:
        template = OTLPMetricExporter(endpoint=f"{_normalize_endpoint(endpoint)}/v1/metrics")
        super().__init__(
            preferred_temporality=template._preferred_temporality,
            preferred_aggregation=template._preferred_aggregation,
        )
        with contextlib.suppress(Exception):
            template.shutdown()
        self._metrics_url = f"{_normalize_endpoint(endpoint)}/v1/metrics"
        self._probe = probe
        self._timeout = timeout
        self._on_collector_online = on_collector_online
        self._lock = threading.Lock()
        self._session: requests.Session | None = None
        self._delegate: OTLPMetricExporter | None = None
        self._shutdown = False

    def _ensure_delegate(self) -> OTLPMetricExporter:
        if self._delegate is None:
            self._session = _new_http_session()
            self._delegate = OTLPMetricExporter(
                endpoint=self._metrics_url,
                timeout=self._timeout,
                session=self._session,
            )
        return self._delegate

    def _reset_delegate(self) -> None:
        if self._delegate is not None:
            with contextlib.suppress(Exception):
                self._delegate.shutdown()
            self._delegate = None
        if self._session is not None:
            with contextlib.suppress(Exception):
                self._session.close()
            self._session = None

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        del timeout_millis, kwargs
        with self._lock:
            if self._shutdown:
                return MetricExportResult.SUCCESS
            if not self._probe.is_reachable():
                self._reset_delegate()
                return MetricExportResult.SUCCESS
            delegate = self._ensure_delegate()
            try:
                res = delegate.export(metrics_data, timeout_millis=self._timeout * 1000.0)
            except Exception:
                res = MetricExportResult.FAILURE
            if res is not MetricExportResult.SUCCESS:
                self._reset_delegate()
                return MetricExportResult.SUCCESS
        if self._on_collector_online is not None:
            self._on_collector_online()
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        del timeout_millis
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: Any) -> None:
        del timeout_millis, kwargs
        with self._lock:
            self._shutdown = True
            self._reset_delegate()


def _sanitize_otel_attribute(value: Any) -> str | bool | int | float:
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError, ValueError:
        return str(value)


class TelemetryManager:
    """Facade owning OpenTelemetry traces, metrics, and structured logs for `agy-acp`."""

    def __init__(
        self,
        config: TelemetryConfig,
        *,
        use_simple_processors: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _silence_otel_internal_loggers()
        self.config = config
        self._callbacks = TelemetryCallbacks()
        self._callbacks_lock = threading.Lock()
        self._shutdown_done = False

        self._probe = EndpointReachabilityProbe(config.endpoint)
        self._span_exporter: ResilientSpanExporter | None = None
        self._log_exporter: ResilientLogExporter | None = None
        self._metric_exporter: ResilientMetricExporter | None = None
        self._metric_reader: PeriodicExportingMetricReader | None = None
        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self._logger_provider: LoggerProvider | None = None

        if not config.enabled:
            self._tracer = trace.NoOpTracer()
            self._logger = None
            self._init_noop_instruments()
            return

        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": __version__,
                "agy_acp.kiro_compat_version": KIRO_CLI_COMPAT_VERSION,
                "process.pid": os.getpid(),
            }
        )

        self._span_exporter = ResilientSpanExporter(
            endpoint=config.endpoint,
            probe=self._probe,
            timeout=config.export_timeout_seconds,
            max_buffered_spans=config.max_buffered_spans,
        )
        self._log_exporter = ResilientLogExporter(
            endpoint=config.endpoint,
            probe=self._probe,
            timeout=config.export_timeout_seconds,
            max_buffered_logs=config.max_buffered_logs,
            retention_seconds=config.log_retention_seconds,
            clock=clock,
        )
        self._metric_exporter = ResilientMetricExporter(
            endpoint=config.endpoint,
            probe=self._probe,
            timeout=config.export_timeout_seconds,
            on_collector_online=self._flush_backlogs,
        )

        self._tracer_provider = TracerProvider(resource=resource)
        if use_simple_processors:
            self._tracer_provider.add_span_processor(SimpleSpanProcessor(self._span_exporter))
        else:
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    self._span_exporter,
                    schedule_delay_millis=1000,
                    max_queue_size=2048,
                    export_timeout_millis=int(config.export_timeout_seconds * 1000),
                )
            )
        self._tracer = self._tracer_provider.get_tracer("agy_acp", __version__)

        self._logger_provider = LoggerProvider(resource=resource)
        if use_simple_processors:
            self._logger_provider.add_log_record_processor(
                SimpleLogRecordProcessor(self._log_exporter)
            )
        else:
            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(
                    self._log_exporter,
                    schedule_delay_millis=1000,
                    max_queue_size=4096,
                    export_timeout_millis=int(config.export_timeout_seconds * 1000),
                )
            )
        self._logger = self._logger_provider.get_logger("agy_acp.acp_logger", __version__)

        self._metric_reader = PeriodicExportingMetricReader(
            self._metric_exporter,
            export_interval_millis=config.metric_export_interval_ms,
            export_timeout_millis=max(1000, int(config.export_timeout_seconds * 2000)),
        )
        views = [
            View(
                instrument_name="*.duration",
                aggregation=ExplicitBucketHistogramAggregation(DURATION_SECONDS_BUCKETS),
            ),
            View(
                instrument_name="agy_acp.turn.ttft",
                aggregation=ExplicitBucketHistogramAggregation(DURATION_SECONDS_BUCKETS),
            ),
            View(
                instrument_name="agy_acp.frame.size",
                aggregation=ExplicitBucketHistogramAggregation(FRAME_SIZE_BYTES_BUCKETS),
            ),
            View(
                instrument_name="agy_acp.context.tokens",
                aggregation=ExplicitBucketHistogramAggregation(CONTEXT_TOKENS_BUCKETS),
            ),
            View(
                instrument_name="agy_acp.context.usage_percent",
                aggregation=ExplicitBucketHistogramAggregation(PERCENT_BUCKETS),
            ),
            View(
                instrument_name="agy_acp.context.window_utilization",
                aggregation=ExplicitBucketHistogramAggregation(RATIO_BUCKETS),
            ),
        ]
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[self._metric_reader],
            views=views,
        )
        self._init_instruments(self._meter_provider)

    def _init_noop_instruments(self) -> None:
        noop_mp = MeterProvider()
        self._init_instruments(noop_mp)

    def _init_instruments(self, meter_provider: MeterProvider) -> None:
        meter = meter_provider.get_meter("agy_acp", __version__)
        self._rpc_requests = meter.create_counter(
            "agy_acp.rpc.requests",
            unit="{request}",
            description="Total ACP JSON-RPC requests handled.",
        )
        self._rpc_errors = meter.create_counter(
            "agy_acp.rpc.errors",
            unit="{error}",
            description="Total ACP JSON-RPC error responses.",
        )
        self._rpc_duration = meter.create_histogram(
            "agy_acp.rpc.duration",
            unit="s",
            description="ACP JSON-RPC handler latency in seconds.",
        )
        self._frame_size = meter.create_histogram(
            "agy_acp.frame.size",
            unit="By",
            description="Raw stdio JSON-RPC frame size in bytes.",
        )
        self._turns = meter.create_counter(
            "agy_acp.turns",
            unit="{turn}",
            description="Completed prompt turns by stop reason and model.",
        )
        self._turns_active = meter.create_up_down_counter(
            "agy_acp.turns.active",
            unit="{turn}",
            description="Currently streaming prompt turns.",
        )
        self._turn_duration = meter.create_histogram(
            "agy_acp.turn.duration",
            unit="s",
            description="End-to-end prompt turn duration in seconds.",
        )
        self._turn_ttft = meter.create_histogram(
            "agy_acp.turn.ttft",
            unit="s",
            description="Time to first streamed token or thought chunk in seconds.",
        )
        self._steers = meter.create_counter(
            "agy_acp.steers",
            unit="{steer}",
            description="Mid-turn steer (`_kiro.dev/session/steer`) attempts.",
        )
        self._compactions = meter.create_counter(
            "agy_acp.compactions",
            unit="{compaction}",
            description="Context compaction executions.",
        )
        self._model_fallbacks = meter.create_counter(
            "agy_acp.model.fallbacks",
            unit="{fallback}",
            description="Automatic model fallback switches.",
        )
        self._tokens = meter.create_counter(
            "agy_acp.tokens",
            unit="{token}",
            description="Tokens processed by model and token type.",
        )
        self._credits_consumed = meter.create_counter(
            "agy_acp.credits.consumed",
            unit="{credit}",
            description="Estimated Google AI plan credits consumed.",
        )
        self._credits_burnt = meter.create_counter(
            "agy_acp.credits.burnt",
            unit="{credit}",
            description="Monotonic quota credits burnt by quota_group, window, and model_id.",
        )
        self._quota_fraction_burnt = meter.create_counter(
            "agy_acp.quota.fraction_burnt",
            unit="{fraction}",
            description="Monotonic non-resetting quota fraction burnt by pool, window, and model.",
        )
        self._context_tokens = meter.create_histogram(
            "agy_acp.context.tokens",
            unit="{token}",
            description="Active conversation context token count per turn.",
        )
        self._context_usage_percent = meter.create_histogram(
            "agy_acp.context.usage_percent",
            description="Active conversation context window utilization percentage.",
        )
        self._context_window_utilization = meter.create_histogram(
            "agy_acp.context.window_utilization",
            unit="1",
            description="Active conversation context window fill ratio (0.0 to 1.0).",
        )
        self._tool_calls = meter.create_counter(
            "agy_acp.tool.calls",
            unit="{call}",
            description="Agent tool calls observed during prompt turns.",
        )
        self._tool_duration = meter.create_histogram(
            "agy_acp.tool.duration",
            unit="s",
            description="Agent tool execution duration in seconds.",
        )
        self._tool_permissions = meter.create_counter(
            "agy_acp.tool.permissions",
            unit="{request}",
            description="ACP `session/request_permission` outcomes.",
        )
        self._backend_spawns = meter.create_counter(
            "agy_acp.backend.spawns",
            unit="{spawn}",
            description="Persistent `agy acp` subprocess spawns.",
        )
        self._backend_exits = meter.create_counter(
            "agy_acp.backend.exits",
            unit="{exit}",
            description="Persistent `agy acp` subprocess exits.",
        )
        self._backend_reconnects = meter.create_counter(
            "agy_acp.backend.reconnects",
            unit="{reconnect}",
            description="Persistent `agy acp` subprocess reconnections.",
        )
        self._backend_rpc_duration = meter.create_histogram(
            "agy_acp.backend.rpc.duration",
            unit="s",
            description="Upstream `agy acp` subprocess RPC latency in seconds.",
        )
        self._mcp_probes = meter.create_counter(
            "agy_acp.mcp.probes",
            unit="{probe}",
            description="MCP server `tools/list` catalog probes.",
        )
        self._mcp_probe_duration = meter.create_histogram(
            "agy_acp.mcp.probe.duration",
            unit="s",
            description="MCP server catalog probe latency in seconds.",
        )
        self._quota_fetch_duration = meter.create_histogram(
            "agy_acp.quota.fetch.duration",
            unit="s",
            description="Antigravity quota/credits fetch latency in seconds.",
        )
        self._sqlite_duration = meter.create_histogram(
            "agy_acp.sqlite.duration",
            unit="s",
            description="SQLite state store query/write latency in seconds.",
        )

        if not self.config.enabled:
            return

        meter.create_observable_gauge(
            "agy_acp.backend.up",
            callbacks=[self._observe_backend_up],
            description="1 if persistent `agy acp` backend subprocess is alive, else 0.",
        )
        meter.create_observable_gauge(
            "agy_acp.sessions.count",
            callbacks=[self._observe_sessions_count],
            unit="{session}",
            description="Session count by state (active or persisted).",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.remaining_percent",
            callbacks=[self._observe_quota_percent],
            description="Remaining Antigravity rolling quota percentage by group and window.",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.remaining_fraction",
            callbacks=[self._observe_quota_remaining_fraction],
            description="Live remaining quota fraction (0.0 to 1.0) per provider pool and window.",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.used_fraction",
            callbacks=[self._observe_quota_used_fraction],
            description="Live consumed quota fraction (0.0 to 1.0) per provider pool and window.",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.remaining_credits",
            callbacks=[self._observe_quota_remaining_credits],
            unit="{credit}",
            description="Estimated remaining Google AI plan credits.",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.bucket_remaining_credits",
            callbacks=[self._observe_quota_bucket_remaining_credits],
            unit="{credit}",
            description="Remaining quota credits per provider pool and rolling window.",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.capacity_credits",
            callbacks=[self._observe_quota_capacity_credits],
            unit="{credit}",
            description="Estimated total Google AI plan credit capacity.",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.reset_timestamp",
            callbacks=[self._observe_quota_reset_timestamp],
            unit="s",
            description="Scheduled quota reset Unix timestamp in seconds by pool and window.",
        )
        meter.create_observable_gauge(
            "agy_acp.quota.5h_per_7d_ratio",
            callbacks=[self._observe_quota_5h_per_7d_ratio],
            description="Empirical number of 5h quotas equal to 1 full 7d quota per pool.",
        )
        meter.create_observable_gauge(
            "agy_acp.g1.overage_credits",
            callbacks=[self._observe_g1_overage_credits],
            unit="{credit}",
            description="Remaining Google One / G1 overage credit balance.",
        )
        meter.create_observable_gauge(
            "agy_acp.ledger.tokens",
            callbacks=[self._observe_ledger_tokens],
            unit="{token}",
            description="Persisted SQLite cumulative token ledger totals.",
        )
        meter.create_observable_gauge(
            "agy_acp.mcp.tools_discovered",
            callbacks=[self._observe_mcp_tools_discovered],
            unit="{tool}",
            description="Discovered MCP tools count per server.",
        )
        meter.create_observable_gauge(
            "agy_acp.telemetry.buffered_items",
            callbacks=[self._observe_buffered_items],
            unit="{item}",
            description="Spans and log records currently buffered in RAM while collector is down.",
        )

    def _flush_backlogs(self) -> None:
        if self._span_exporter is not None:
            self._span_exporter.flush_backlog()
        if self._log_exporter is not None:
            self._log_exporter.flush_backlog()

    def register_callbacks(self, callbacks: TelemetryCallbacks) -> None:
        """Merge live observable gauge callbacks into this manager."""
        with self._callbacks_lock:
            if callbacks.backend_up is not None:
                self._callbacks.backend_up = callbacks.backend_up
            if callbacks.session_counts is not None:
                self._callbacks.session_counts = callbacks.session_counts
            if callbacks.quota_metrics is not None:
                self._callbacks.quota_metrics = callbacks.quota_metrics
            if callbacks.quota_extended_metrics is not None:
                self._callbacks.quota_extended_metrics = callbacks.quota_extended_metrics
            if callbacks.ledger_tokens is not None:
                self._callbacks.ledger_tokens = callbacks.ledger_tokens
            if callbacks.mcp_tools_discovered is not None:
                self._callbacks.mcp_tools_discovered = callbacks.mcp_tools_discovered

    def _observe_backend_up(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        with self._callbacks_lock:
            fn = self._callbacks.backend_up
        if fn is None:
            return ()
        with contextlib.suppress(Exception):
            return (Observation(int(fn())),)
        return ()

    def _observe_sessions_count(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        with self._callbacks_lock:
            fn = self._callbacks.session_counts
        if fn is None:
            return ()
        with contextlib.suppress(Exception):
            counts = fn()
            return [Observation(int(val), {"state": str(k)}) for k, val in counts.items()]
        return ()

    def _observe_quota_percent(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        with self._callbacks_lock:
            fn = self._callbacks.quota_metrics
        if fn is None:
            return ()
        with contextlib.suppress(Exception):
            buckets, _, _ = fn()
            return [
                Observation(
                    float(item["percent"]),
                    {
                        "group": str(item.get("group", "unknown")),
                        "window": str(item.get("window", "5h")),
                    },
                )
                for item in buckets
                if "percent" in item
            ]
        return ()

    def _observe_quota_remaining_credits(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        with self._callbacks_lock:
            fn = self._callbacks.quota_metrics
        if fn is None:
            return ()
        with contextlib.suppress(Exception):
            _, remaining, _ = fn()
            if remaining is not None:
                return (Observation(float(remaining)),)
        return ()

    def _observe_quota_capacity_credits(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        with self._callbacks_lock:
            fn = self._callbacks.quota_metrics
        if fn is None:
            return ()
        with contextlib.suppress(Exception):
            _, _, capacity = fn()
            if capacity is not None:
                return (Observation(float(capacity)),)
        return ()

    def _resolve_extended_quota_payload(self) -> dict[str, Any] | None:
        with self._callbacks_lock:
            ext_fn = self._callbacks.quota_extended_metrics
            base_fn = self._callbacks.quota_metrics
        if ext_fn is not None:
            with contextlib.suppress(Exception):
                data = ext_fn()
                if isinstance(data, Mapping):
                    return dict(data)
        if base_fn is not None:
            with contextlib.suppress(Exception):
                buckets, rem, _ = base_fn()
                derived: list[dict[str, Any]] = []
                now_ts = time.time()
                for item in buckets:
                    grp = str(item.get("group", "Gemini Models"))
                    q_grp = str(item.get("quota_group") or normalize_quota_group(grp))
                    win = str(item.get("window", "5h"))
                    pct = float(item.get("percent", 0.0))
                    frac = float(item.get("remaining_fraction", max(0.0, min(1.0, pct / 100.0))))
                    reset_ts = item.get("reset_timestamp_seconds")
                    if reset_ts is None and item.get("reset_time"):
                        reset_ts = parse_reset_epoch_seconds(str(item.get("reset_time")))
                    if reset_ts is None:
                        reset_ts = now_ts + (604_800.0 if win == "7d" else 18_000.0)
                    derived.append(
                        {
                            "quota_group": q_grp,
                            "group": grp,
                            "window": win,
                            "remaining_fraction": frac,
                            "used_fraction": round(max(0.0, min(1.0, 1.0 - frac)), 6),
                            "remaining_credits": float(
                                item.get("remaining_credits", rem if rem is not None else 0.0)
                            ),
                            "reset_timestamp_seconds": float(reset_ts),
                        }
                    )
                return {
                    "buckets": derived,
                    "g1_overage_credits": 0.0,
                    "ratios": {
                        "google_ai": DEFAULT_5H_PER_7D_RATIO,
                        "anthropic_ai": DEFAULT_5H_PER_7D_RATIO,
                    },
                }
        return None

    def _observe_quota_remaining_fraction(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        payload = self._resolve_extended_quota_payload()
        if payload is None:
            return ()
        return [
            Observation(
                float(b["remaining_fraction"]),
                {
                    "quota_group": str(b["quota_group"]),
                    "group": str(b["group"]),
                    "window": str(b["window"]),
                },
            )
            for b in payload.get("buckets", ())
            if "remaining_fraction" in b
        ]

    def _observe_quota_used_fraction(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        payload = self._resolve_extended_quota_payload()
        if payload is None:
            return ()
        return [
            Observation(
                float(b["used_fraction"]),
                {
                    "quota_group": str(b["quota_group"]),
                    "group": str(b["group"]),
                    "window": str(b["window"]),
                },
            )
            for b in payload.get("buckets", ())
            if "used_fraction" in b
        ]

    def _observe_quota_bucket_remaining_credits(
        self, options: CallbackOptions
    ) -> Iterable[Observation]:
        del options
        payload = self._resolve_extended_quota_payload()
        if payload is None:
            return ()
        return [
            Observation(
                float(b["remaining_credits"]),
                {
                    "quota_group": str(b["quota_group"]),
                    "group": str(b["group"]),
                    "window": str(b["window"]),
                },
            )
            for b in payload.get("buckets", ())
            if "remaining_credits" in b
        ]

    def _observe_quota_reset_timestamp(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        payload = self._resolve_extended_quota_payload()
        if payload is None:
            return ()
        return [
            Observation(
                float(b["reset_timestamp_seconds"]),
                {
                    "quota_group": str(b["quota_group"]),
                    "group": str(b["group"]),
                    "window": str(b["window"]),
                },
            )
            for b in payload.get("buckets", ())
            if b.get("reset_timestamp_seconds") is not None
        ]

    def _observe_quota_5h_per_7d_ratio(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        payload = self._resolve_extended_quota_payload()
        if payload is None:
            return ()
        ratios = payload.get("ratios")
        if not isinstance(ratios, Mapping):
            return ()
        return [
            Observation(float(val), {"quota_group": str(q_grp)})
            for q_grp, val in ratios.items()
            if val is not None
        ]

    def _observe_g1_overage_credits(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        payload = self._resolve_extended_quota_payload()
        if payload is None or payload.get("g1_overage_credits") is None:
            return ()
        return (Observation(float(payload["g1_overage_credits"])),)

    def _observe_ledger_tokens(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        with self._callbacks_lock:
            fn = self._callbacks.ledger_tokens
        if fn is None:
            return ()
        with contextlib.suppress(Exception):
            totals = fn()
            return [Observation(int(val), {"token_type": str(k)}) for k, val in totals.items()]
        return ()

    def _observe_mcp_tools_discovered(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        with self._callbacks_lock:
            fn = self._callbacks.mcp_tools_discovered
        if fn is None:
            return ()
        with contextlib.suppress(Exception):
            per_server = fn()
            return [Observation(int(val), {"server": str(k)}) for k, val in per_server.items()]
        return ()

    def _observe_buffered_items(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        return (
            Observation(self.buffered_spans_count, {"signal": "traces"}),
            Observation(self.buffered_logs_count, {"signal": "logs"}),
        )

    @property
    def buffered_spans_count(self) -> int:
        return self._span_exporter.buffered_count if self._span_exporter is not None else 0

    @property
    def buffered_logs_count(self) -> int:
        return self._log_exporter.buffered_count if self._log_exporter is not None else 0

    def peek_buffered_spans(self) -> list[ReadableSpan]:
        return self._span_exporter.peek_buffered() if self._span_exporter is not None else []

    def peek_buffered_logs(self) -> list[ReadableLogRecord]:
        return self._log_exporter.peek_buffered() if self._log_exporter is not None else []

    def is_collector_reachable(self) -> bool:
        """Check whether the configured OTLP HTTP collector port is currently reachable."""
        return self._probe.is_reachable()

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        parent_context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Generator[Span]:
        """Start an OpenTelemetry span as the current context span."""
        clean_attrs = (
            {str(k): _sanitize_otel_attribute(v) for k, v in attributes.items() if v is not None}
            if attributes
            else None
        )
        with self._tracer.start_as_current_span(
            name,
            context=parent_context,
            kind=kind,
            attributes=clean_attrs,
        ) as span:
            try:
                yield span
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def start_manual_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        parent_context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Span:
        """Start a span without binding to context (for async multi-notification tool spans)."""
        clean_attrs = (
            {str(k): _sanitize_otel_attribute(v) for k, v in attributes.items() if v is not None}
            if attributes
            else None
        )
        return self._tracer.start_span(
            name,
            context=parent_context,
            kind=kind,
            attributes=clean_attrs,
        )

    def current_context(self) -> Context:
        """Capture the active OpenTelemetry `Context` for cross-task propagation."""
        return otel_context.get_current()

    def current_trace_identifiers(self) -> tuple[str | None, str | None]:
        """Return `(trace_id_hex, span_id_hex)` for the currently active span if valid."""
        span_ctx = trace.get_current_span().get_span_context()
        if not span_ctx.is_valid:
            return None, None
        return f"{span_ctx.trace_id:032x}", f"{span_ctx.span_id:016x}"

    def emit_log(
        self,
        event: str,
        fields: Mapping[str, Any],
        *,
        body: str | None = None,
        context: Context | None = None,
    ) -> None:
        """Emit a redacted structured OpenTelemetry `LogRecord` correlated to active trace."""
        if not self.config.enabled or self._logger is None:
            return
        lower_ev = event.lower()
        if "error" in lower_ev or "exception" in lower_ev:
            sev_num = SeverityNumber.ERROR
            sev_txt = "ERROR"
        elif (
            "warn" in lower_ev
            or "fallback" in lower_ev
            or "ignored" in lower_ev
            or "retry" in lower_ev
        ):
            sev_num = SeverityNumber.WARN
            sev_txt = "WARN"
        else:
            sev_num = SeverityNumber.INFO
            sev_txt = "INFO"

        attrs: dict[str, str | bool | int | float] = {"event": event}
        for key, val in fields.items():
            if val is not None:
                attrs[str(key)] = _sanitize_otel_attribute(val)

        active_ctx = context if context is not None else otel_context.get_current()
        now_ns = time.time_ns()
        log_body = body if body is not None else json.dumps({"event": event, **fields}, default=str)
        with contextlib.suppress(Exception):
            self._logger.emit(
                timestamp=now_ns,
                observed_timestamp=now_ns,
                context=active_ctx,
                severity_number=sev_num,
                severity_text=sev_txt,
                body=log_body,
                attributes=attrs,
                event_name=event,
            )

    def record_rpc_request(self, method: str, *, duration_s: float, engine: str = "v2") -> None:
        attrs = {"method": method, "engine": engine}
        self._rpc_requests.add(1, attrs)
        self._rpc_duration.record(max(0.0, duration_s), attrs)

    def record_rpc_error(self, method: str, *, code: int | str, engine: str = "v2") -> None:
        self._rpc_errors.add(1, {"method": method, "code": str(code), "engine": engine})

    def record_frame(self, direction: str, size_bytes: int) -> None:
        self._frame_size.record(max(0, size_bytes), {"direction": direction})

    def turn_started(self) -> None:
        self._turns_active.add(1)

    def turn_finished(self) -> None:
        self._turns_active.add(-1)

    def record_turn(
        self,
        *,
        stop_reason: str,
        model: str,
        engine: str,
        duration_s: float,
        ttft_s: float | None = None,
    ) -> None:
        quota_group = classify_model_quota_group(model)
        status = "ok" if stop_reason in {"end_turn", "max_tokens", "stop_sequence"} else stop_reason
        attrs = {
            "stop_reason": stop_reason,
            "status": status,
            "model": model,
            "model_id": model,
            "quota_group": quota_group,
            "engine": engine,
        }
        self._turns.add(1, attrs)
        self._turn_duration.record(max(0.0, duration_s), attrs)
        if ttft_s is not None:
            self._turn_ttft.record(
                max(0.0, ttft_s),
                {
                    "model": model,
                    "model_id": model,
                    "quota_group": quota_group,
                    "engine": engine,
                },
            )

    def record_steer(self, *, outcome: str) -> None:
        self._steers.add(1, {"outcome": outcome})

    def record_compaction(self, *, trigger: str, status: str) -> None:
        self._compactions.add(1, {"trigger": trigger, "status": status})

    def record_model_fallback(self, from_model: str, to_model: str, *, reason: str) -> None:
        self._model_fallbacks.add(
            1,
            {"from_model": from_model, "to_model": to_model, "reason": reason},
        )

    def record_tokens(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        thought_tokens: int = 0,
    ) -> None:
        quota_group = classify_model_quota_group(model)
        for token_type, count in (
            ("input", input_tokens),
            ("output", output_tokens),
            ("cached", cached_tokens),
            ("thought", thought_tokens),
        ):
            if count > 0:
                self._tokens.add(
                    count,
                    {
                        "model": model,
                        "model_id": model,
                        "quota_group": quota_group,
                        "token_type": token_type,
                    },
                )

    def record_credits_consumed(self, model: str, credits: float) -> None:
        if credits > 0.0:
            self._credits_consumed.add(
                credits,
                {
                    "model": model,
                    "model_id": model,
                    "quota_group": classify_model_quota_group(model),
                },
            )

    def record_quota_fraction_burnt(
        self,
        quota_group: str,
        *,
        window: str,
        model_id: str,
        fraction: float,
        context: Context | None = None,
    ) -> None:
        if fraction <= 0.0:
            return
        with _bind_optional_context(context):
            self._quota_fraction_burnt.add(
                fraction,
                {
                    "quota_group": quota_group,
                    "window": window,
                    "model_id": model_id,
                },
            )

    def record_credits_burnt(
        self,
        quota_group: str,
        *,
        window: str,
        model_id: str,
        credits: float,
        context: Context | None = None,
    ) -> None:
        if credits <= 0.0:
            return
        with _bind_optional_context(context):
            self._credits_burnt.add(
                credits,
                {
                    "quota_group": quota_group,
                    "window": window,
                    "model_id": model_id,
                },
            )

    def record_context_usage(
        self,
        model: str,
        *,
        used_tokens: int,
        usage_percent: float,
    ) -> None:
        attrs = {
            "model": model,
            "model_id": model,
            "quota_group": classify_model_quota_group(model),
        }
        clamped_pct = max(0.0, usage_percent)
        self._context_tokens.record(max(0, used_tokens), attrs)
        self._context_usage_percent.record(clamped_pct, attrs)
        self._context_window_utilization.record(min(1.0, clamped_pct / 100.0), attrs)

    def record_tool_call(
        self,
        tool_name: str,
        *,
        kind: str,
        status: str,
        duration_s: float | None = None,
    ) -> None:
        attrs = {"tool_name": tool_name, "kind": kind, "status": status}
        self._tool_calls.add(1, attrs)
        if duration_s is not None:
            self._tool_duration.record(max(0.0, duration_s), attrs)

    def record_tool_permission(self, kind: str, *, outcome: str) -> None:
        self._tool_permissions.add(1, {"kind": kind, "outcome": outcome})

    def record_backend_spawn(self) -> None:
        self._backend_spawns.add(1)

    def record_backend_exit(self, *, returncode: int | str) -> None:
        self._backend_exits.add(1, {"returncode": str(returncode)})

    def record_backend_reconnect(self, *, reason: str) -> None:
        self._backend_reconnects.add(1, {"reason": reason})

    def record_backend_rpc(self, method: str, *, status: str, duration_s: float) -> None:
        attrs = {"method": method, "status": status}
        self._backend_rpc_duration.record(max(0.0, duration_s), attrs)

    def record_mcp_probe(self, server: str, *, status: str, duration_s: float) -> None:
        attrs = {"server": server, "status": status}
        self._mcp_probes.add(1, attrs)
        self._mcp_probe_duration.record(max(0.0, duration_s), attrs)

    def record_quota_fetch(self, endpoint_type: str, *, status: str, duration_s: float) -> None:
        self._quota_fetch_duration.record(
            max(0.0, duration_s),
            {"endpoint": endpoint_type, "status": status},
        )

    def record_sqlite_operation(self, operation: str, *, duration_s: float) -> None:
        self._sqlite_duration.record(max(0.0, duration_s), {"operation": operation})

    def collect_metrics(self) -> None:
        """Trigger an immediate metric collection and export cycle if enabled."""
        if self._metric_reader is not None:
            with contextlib.suppress(Exception):
                self._metric_reader.collect()

    def shutdown(self) -> None:
        """Flush (when collector is reachable) and shut down all SDK providers cleanly."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._tracer_provider is not None:
            with contextlib.suppress(Exception):
                self._tracer_provider.shutdown()
        if self._logger_provider is not None:
            with contextlib.suppress(Exception):
                self._logger_provider.shutdown()
        if self._meter_provider is not None:
            with contextlib.suppress(Exception):
                self._meter_provider.shutdown()


_GLOBAL_LOCK = threading.Lock()
_GLOBAL_MANAGER: TelemetryManager | None = None
_ATEXIT_REGISTERED = False


def get_telemetry(save_dir_or_db: Path | str | None = None) -> TelemetryManager:
    """Return the process-wide `TelemetryManager` singleton, initializing lazily on first use."""
    global _GLOBAL_MANAGER, _ATEXIT_REGISTERED
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            cfg = resolve_telemetry_config(save_dir_or_db)
            _GLOBAL_MANAGER = TelemetryManager(cfg)
            if not _ATEXIT_REGISTERED:
                atexit.register(_shutdown_global_telemetry)
                _ATEXIT_REGISTERED = True
        return _GLOBAL_MANAGER


def _shutdown_global_telemetry() -> None:
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        mgr = _GLOBAL_MANAGER
        _GLOBAL_MANAGER = None
    if mgr is not None:
        mgr.shutdown()


def reset_telemetry_for_tests() -> None:
    """Shut down and clear the global `TelemetryManager` singleton between tests."""
    _shutdown_global_telemetry()
