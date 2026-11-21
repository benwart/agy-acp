"""Async bounded MCP server catalog prober (`initialize` + `tools/list`) for stdio and HTTP/SSE."""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

from agy_acp import __version__ as _AGY_ACP_VERSION
from agy_acp.logic.custom_agents import AgentOrigin
from agy_acp.logic.tool_classifier import build_scrubbed_subprocess_env
from agy_acp.logic.wire.base import McpServerStatus, McpToolDef
from agy_acp.providers.telemetry import get_telemetry

MCP_PROTOCOL_LATEST = "2025-03-26"
MCP_PROTOCOL_LEGACY = "2024-11-05"
INIT_DEADLINE_STRETCH_CAP_SECS = 120.0
INIT_DEADLINE_REAP_KEYS_MAX = 256
_MAX_PAGINATION_PAGES = 20
_MAX_OAUTH_METADATA_BYTES = 16 * 1024
MAX_HTTP_BYTES = 4 * 1024 * 1024
MAX_STDIO_CONTENT_LENGTH = 4 * 1024 * 1024
MAX_STDIO_FRAMES = 256
MAX_TOOLS_PER_SERVER = 512
MAX_CONCURRENT_PROBES = 8
_MAX_HEADER_LINES = 32
_MAX_ERROR_CHARS = 500
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

_AUTH_URI_PARAM_RE = re.compile(
    r'(?:authorization_uri|auth_url|oauth_url|resource_metadata)\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)
_URL_FALLBACK_RE = re.compile(r"(https?://[^\s\",;>]+)")
_URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@\"']+@")
_URL_QUERY_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^\s?#\"']+)[?#][^\s\"')>\]]*")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)\S+")
_SECRET_KV_RE = re.compile(
    r"(?i)(\b[a-z0-9_-]*(?:token|key|secret|password|passwd|auth)[a-z0-9_-]*\s*=\s*)[^\s&\"',;]+"
)


def _is_safe_http_url(url: str) -> bool:
    """Return True iff `url` uses `http` or `https` and has a non-empty hostname."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def sanitize_mcp_error(raw_error: str) -> str:
    """Scrub URL userinfo, URL query strings, Bearer tokens, and secret assignments from errors."""
    scrubbed = _URL_USERINFO_RE.sub(r"\1", raw_error)
    scrubbed = _URL_QUERY_RE.sub(r"\1", scrubbed)
    scrubbed = _BEARER_RE.sub(r"\1[REDACTED]", scrubbed)
    scrubbed = _SECRET_KV_RE.sub(r"\1[REDACTED]", scrubbed)
    if len(scrubbed) > _MAX_ERROR_CHARS:
        scrubbed = scrubbed[:_MAX_ERROR_CHARS]
    return scrubbed


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler restricting targets strictly to `http://` and `https://` with hostnames."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        if not _is_safe_http_url(resolved):
            raise urllib.error.URLError("Unsafe HTTP redirect target")
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _build_safe_http_opener() -> urllib.request.OpenerDirector:
    """Build an HTTP/HTTPS-only opener excluding `FileHandler`, `FTPHandler`, and `DataHandler`."""
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    opener.add_handler(_SafeRedirectHandler())
    return opener


def _build_no_redirect_http_opener() -> urllib.request.OpenerDirector:
    """Build an HTTP/HTTPS-only opener that never follows redirects (for RFC 9728 metadata)."""
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    return opener


def normalized_mcp_endpoint(value: object) -> str | None:
    """Normalize an HTTP/HTTPS MCP endpoint URL per KiroCrew `tool_aliases.normalized_endpoint`."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parts = urllib.parse.urlsplit(text)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        return None
    host = parts.hostname.lower()
    formatted_host = f"[{host}]" if ":" in host else host
    default_port = 80 if scheme == "http" else 443
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    userinfo = f"{parts.netloc.rsplit('@', 1)[0]}@" if "@" in parts.netloc else ""
    netloc = f"{userinfo}{formatted_host}{port_suffix}"
    trimmed_path = parts.path.rstrip("/")
    return urllib.parse.urlunsplit((scheme, netloc, trimmed_path, parts.query, ""))


def _protected_resource_metadata_urls(mcp_url: str) -> list[str]:
    """Return RFC 9728 §3.1 metadata candidate URLs (path-inserted first, then origin root)."""
    try:
        parts = urllib.parse.urlsplit(mcp_url.strip())
        port = parts.port
    except ValueError:
        return []
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        return []
    host = parts.hostname.lower()
    formatted_host = f"[{host}]" if ":" in host else host
    default_port = 80 if scheme == "http" else 443
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    origin = f"{scheme}://{formatted_host}{port_suffix}"
    trimmed_path = parts.path.rstrip("/")
    root_url = f"{origin}/.well-known/oauth-protected-resource"
    if trimmed_path:
        return [f"{root_url}{trimmed_path}", root_url]
    return [root_url]


def _is_loopback_hostname(hostname: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    return host in _LOOPBACK_HOSTS


def _is_valid_authorization_server_url(candidate: object, *, mcp_is_loopback: bool) -> bool:
    """Validate RFC 9728 `authorization_servers[0]` without dereferencing it."""
    if not isinstance(candidate, str) or not candidate:
        return False
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in candidate):
        return False
    if "?" in candidate or "#" in candidate:
        return False
    try:
        parts = urllib.parse.urlsplit(candidate)
        _ = parts.port
    except ValueError:
        return False
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        return False
    if parts.query or parts.fragment:
        return False
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower().rstrip(".")
    if not hostname:
        return False
    try:
        ip_addr = ipaddress.ip_address(hostname)
    except ValueError:
        ip_addr = None
    server_is_loopback = hostname in _LOOPBACK_HOSTS or (
        ip_addr is not None and ip_addr.is_loopback
    )
    if mcp_is_loopback:
        return scheme == "https" or (scheme == "http" and server_is_loopback)
    server_is_local = (
        server_is_loopback
        or hostname.endswith(".local")
        or hostname.endswith(".localhost")
        or ip_addr is not None
    )
    return scheme == "https" and not server_is_local


def _discover_protected_resource_auth_url(mcp_url: str, timeout_secs: float) -> str | None:
    """Probe RFC 9728 OAuth Protected Resource Metadata on a bare HTTP 401 response."""
    expected_resource = normalized_mcp_endpoint(mcp_url)
    if expected_resource is None:
        return None
    try:
        mcp_host = urllib.parse.urlsplit(mcp_url.strip()).hostname or ""
    except ValueError:
        return None
    mcp_is_loopback = _is_loopback_hostname(mcp_host)
    meta_timeout = min(max(float(timeout_secs), 0.1), 5.0)
    opener = _build_no_redirect_http_opener()

    for meta_url in _protected_resource_metadata_urls(mcp_url):
        request = urllib.request.Request(
            meta_url,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            method="GET",
        )
        try:
            with opener.open(request, timeout=meta_timeout) as resp:
                if int(resp.status) != 200:
                    continue
                content_encoding = (resp.headers.get("Content-Encoding") or "").strip().lower()
                if content_encoding not in {"", "identity"}:
                    continue
                raw = resp.read(_MAX_OAUTH_METADATA_BYTES + 1)
                if len(raw) > _MAX_OAUTH_METADATA_BYTES:
                    continue
                document = json.loads(raw.decode("utf-8"))
        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ):
            continue

        if not isinstance(document, dict):
            continue
        raw_resource = document.get("resource")
        if isinstance(raw_resource, str):
            resource_entries = [raw_resource]
        elif isinstance(raw_resource, list) and all(isinstance(v, str) for v in raw_resource):
            resource_entries = raw_resource
        else:
            continue
        if not any(normalized_mcp_endpoint(v) == expected_resource for v in resource_entries):
            continue

        servers = document.get("authorization_servers")
        if not isinstance(servers, list) or not servers:
            continue
        candidate = servers[0]
        if _is_valid_authorization_server_url(candidate, mcp_is_loopback=mcp_is_loopback):
            return candidate
    return None


def _probe_reap_slot(*identity_parts: object) -> str:
    """Compute a stable SHA-256 hex digest slot for MCP initialize deadline stretch tracking."""
    payload = json.dumps(identity_parts, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_version_negotiation(server_cfg: Mapping[str, Any]) -> str:
    meta = server_cfg.get("_meta")
    kiro = meta.get("kiro") if isinstance(meta, Mapping) else None
    mode = kiro.get("versionNegotiation") if isinstance(kiro, Mapping) else None
    if mode == "legacy":
        return MCP_PROTOCOL_LEGACY
    return MCP_PROTOCOL_LATEST


def _normalize_kv_pairs(raw: Any) -> dict[str, str]:
    """Normalize either `{k: v}` mapping or `[{name, value}]` list into `dict[str, str]`."""
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items() if v is not None}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            value = item.get("value")
            if isinstance(name, str) and name and value is not None:
                out[name] = str(value)
        return out
    return {}


def _build_stdio_probe_env(
    session_env: Mapping[str, str] | None,
    server_env: Any,
) -> dict[str, str]:
    """Build scrubbed child environment omitting empty `session_env` sentinel values."""
    merged_env = build_scrubbed_subprocess_env()
    if session_env:
        for raw_key, raw_val in session_env.items():
            key = str(raw_key)
            val = "" if raw_val is None else str(raw_val)
            if val:
                merged_env[key] = val
            else:
                merged_env.pop(key, None)
    merged_env.update(_normalize_kv_pairs(server_env))
    return merged_env


def _extract_disabled_tools(server_cfg: Mapping[str, Any]) -> frozenset[str]:
    raw = server_cfg.get("disabledTools")
    if raw is None:
        raw = server_cfg.get("disabled_tools")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return frozenset(str(t) for t in raw if isinstance(t, str) and t)
    return frozenset()


def _extract_origin(server_cfg: Mapping[str, Any]) -> AgentOrigin:
    meta = server_cfg.get("_meta")
    kiro = meta.get("kiro") if isinstance(meta, Mapping) else None
    resource = kiro.get("resource") if isinstance(kiro, Mapping) else None
    source = resource.get("source") if isinstance(resource, Mapping) else None
    origin = source.get("origin") if isinstance(source, Mapping) else None
    if origin in ("client", "user", "project", "bundled"):
        return origin  # type: ignore[return-value]
    return "client"


def _extract_auth_url(www_authenticate: str, body_text: str) -> str | None:
    if www_authenticate:
        match = _AUTH_URI_PARAM_RE.search(www_authenticate)
        if match and _is_safe_http_url(match.group(1)):
            return match.group(1).strip()
    if body_text:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(body_text)
            if isinstance(parsed, Mapping):
                for key in (
                    "authorization_url",
                    "authorization_uri",
                    "authUrl",
                    "oauthUrl",
                    "url",
                ):
                    val = parsed.get(key)
                    if isinstance(val, str) and _is_safe_http_url(val):
                        return val.strip()
    if www_authenticate:
        fallback = _URL_FALLBACK_RE.search(www_authenticate)
        if fallback and _is_safe_http_url(fallback.group(1)):
            return fallback.group(1).strip()
    return None


def _parse_http_rpc_body(content_type: str, raw_bytes: bytes) -> dict[str, Any] | None:
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    if "text/event-stream" in content_type.lower():
        data_lines: list[str] = []
        last_json: dict[str, Any] | None = None
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if payload:
                    data_lines.append(payload)
            elif not line.strip() and data_lines:
                joined = "\n".join(data_lines)
                data_lines.clear()
                with contextlib.suppress(json.JSONDecodeError):
                    candidate = json.loads(joined)
                    if isinstance(candidate, dict):
                        last_json = candidate
        if data_lines:
            joined = "\n".join(data_lines)
            with contextlib.suppress(json.JSONDecodeError):
                candidate = json.loads(joined)
                if isinstance(candidate, dict):
                    last_json = candidate
        return last_json

    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else None


class _HttpAuthRequiredError(Exception):
    """Internal signal raised when an HTTP MCP endpoint responds with 401/403."""

    def __init__(self, status_code: int, auth_url: str | None, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.auth_url = auth_url
        self.detail = detail


class _ProbeTimeoutError(TimeoutError):
    """Internal timeout carrying the effective timeout duration used for the timed-out phase."""

    def __init__(self, timeout_secs: float) -> None:
        super().__init__(f"MCP server probe timed out after {timeout_secs:.2f}s")
        self.timeout_secs = timeout_secs


def _post_mcp_http_sync(
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout_secs: float,
) -> tuple[int, dict[str, str], dict[str, Any] | None]:
    if not _is_safe_http_url(url):
        raise ValueError("Unsupported or invalid MCP HTTP URL: expected http:// or https:// host")

    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **headers,
    }
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, headers=req_headers, method="POST")
    opener = _build_safe_http_opener()
    try:
        with opener.open(request, timeout=timeout_secs) as resp:
            status_code = int(resp.status)
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            raw_body = resp.read(MAX_HTTP_BYTES + 1)
            if len(raw_body) > MAX_HTTP_BYTES:
                raise ValueError("MCP HTTP response body exceeds maximum allowed size")
            ctype = resp_headers.get("content-type", "application/json")
            parsed = _parse_http_rpc_body(ctype, raw_body)
            return status_code, resp_headers, parsed
    except urllib.error.HTTPError as exc:
        body_bytes = b""
        try:
            body_bytes = exc.read(MAX_HTTP_BYTES + 1)
        except OSError:
            body_bytes = b""
        finally:
            with contextlib.suppress(Exception):
                exc.close()
        if len(body_bytes) > MAX_HTTP_BYTES:
            raise ValueError("MCP HTTP error response body exceeds maximum allowed size") from exc
        if exc.code in (401, 403):
            body_text = body_bytes.decode("utf-8", errors="replace")
            www_auth = exc.headers.get("WWW-Authenticate", "") if exc.headers else ""
            auth_url = _extract_auth_url(www_auth, body_text)
            if exc.code == 401 and auth_url is None:
                auth_url = _discover_protected_resource_auth_url(url, timeout_secs)
            raise _HttpAuthRequiredError(
                exc.code,
                auth_url,
                f"HTTP {exc.code} {exc.reason}".strip(),
            ) from exc
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}".strip()) from exc


async def _read_stdio_jsonrpc_response(
    stream: BufferedByteReceiveStream,
    expected_id: int,
) -> dict[str, Any]:
    """Read bounded JSON-RPC frames from stdio until the response matching `expected_id` arrives."""
    for _ in range(MAX_STDIO_FRAMES):
        raw_line = await stream.receive_until(b"\n", 1_048_576)
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise ValueError("Invalid or oversized Content-Length") from exc
            if length <= 0 or length > MAX_STDIO_CONTENT_LENGTH:
                raise ValueError("Invalid or oversized Content-Length")
            # Consume blank header separator line (`\r\n` or `\n`)
            for _ in range(_MAX_HEADER_LINES):
                sep = await stream.receive_until(b"\n", 4096)
                if not sep.strip():
                    break
            else:
                raise ValueError("Too many header lines in stdio MCP frame")
            body = await stream.receive_exactly(length)
            msg = json.loads(body.decode("utf-8", errors="replace"))
        else:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
        if isinstance(msg, dict) and msg.get("id") == expected_id:
            return msg
    raise RuntimeError(f"Exceeded {MAX_STDIO_FRAMES} stdio frames waiting for JSON-RPC response")


class McpCatalogProber:
    """Bounded async MCP `initialize` + `tools/list` prober for session startup & resetServer."""

    def __init__(self, *, timeout_secs: float = 10.0) -> None:
        self._timeout_secs = float(timeout_secs)
        self._deadline_reaps: dict[str, int] = {}

    def initialize_timeout_for(self, slot: str) -> float:
        """Return effective `initialize` timeout for `slot`, stretching 2x per consecutive reap."""
        reaps = self._deadline_reaps.get(slot, 0)
        if reaps <= 0:
            return self._timeout_secs
        stretched = min(self._timeout_secs * (2**reaps), INIT_DEADLINE_STRETCH_CAP_SECS)
        return max(self._timeout_secs, stretched)

    def _record_initialize_timeout(self, slot: str) -> None:
        reaps = self._deadline_reaps.pop(slot, 0)
        self._deadline_reaps[slot] = min(reaps + 1, 8)
        while len(self._deadline_reaps) > INIT_DEADLINE_REAP_KEYS_MAX:
            oldest_slot = next(iter(self._deadline_reaps))
            self._deadline_reaps.pop(oldest_slot, None)

    def _clear_initialize_timeout(self, slot: str) -> None:
        self._deadline_reaps.pop(slot, None)

    async def probe_servers(
        self,
        servers: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
        *,
        session_env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[McpServerStatus]:
        """Probe all declared MCP servers concurrently (capped) and return ordered statuses."""
        normalized_list = self._normalize_server_declarations(servers)
        if not normalized_list:
            return []

        results: list[McpServerStatus | None] = [None] * len(normalized_list)
        limiter = anyio.CapacityLimiter(MAX_CONCURRENT_PROBES)

        async def _run_one(idx: int, cfg: dict[str, Any]) -> None:
            async with limiter:
                results[idx] = await self.probe_server(
                    cfg,
                    session_env=session_env,
                    cwd=cwd,
                )

        async with anyio.create_task_group() as tg:
            for index, server_cfg in enumerate(normalized_list):
                tg.start_soon(_run_one, index, server_cfg)

        return [res for res in results if res is not None]

    @staticmethod
    def _normalize_server_declarations(
        servers: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if isinstance(servers, Mapping):
            for name, raw_cfg in servers.items():
                if not isinstance(raw_cfg, Mapping):
                    continue
                merged = dict(raw_cfg)
                merged.setdefault("name", str(name))
                out.append(merged)
        else:
            for item in servers:
                if isinstance(item, Mapping) and item.get("name"):
                    out.append(dict(item))
        return out

    async def probe_server(
        self,
        server_cfg: Mapping[str, Any],
        *,
        session_env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> McpServerStatus:
        """Probe a single stdio or HTTP/SSE MCP server without raising uncaught errors."""
        name = str(server_cfg.get("name") or "")
        display_name = str(server_cfg.get("displayName") or name)
        origin = _extract_origin(server_cfg)

        if server_cfg.get("disabled") is True:
            return McpServerStatus(
                name=name,
                display_name=display_name,
                status="disabled",
                tools=(),
                origin=origin,
            )

        url = server_cfg.get("url")
        command = server_cfg.get("command")
        telemetry = get_telemetry()
        start_t = time.perf_counter()
        result_status = "failed"
        with telemetry.start_span(
            "mcp.probe",
            attributes={"mcp.server": name, "mcp.origin": origin},
        ):
            try:
                if isinstance(url, str) and url.strip():
                    res = await self._probe_http_server(
                        name=name,
                        display_name=display_name,
                        url=url.strip(),
                        server_cfg=server_cfg,
                        origin=origin,
                    )
                    result_status = res.status
                    return res
                if isinstance(command, str) and command.strip():
                    res = await self._probe_stdio_server(
                        name=name,
                        display_name=display_name,
                        command=command.strip(),
                        server_cfg=server_cfg,
                        session_env=session_env,
                        cwd=cwd,
                        origin=origin,
                    )
                    result_status = res.status
                    return res
                return McpServerStatus(
                    name=name,
                    display_name=display_name,
                    status="failed",
                    error="MCP server config specifies neither 'command' nor 'url'",
                    origin=origin,
                )
            except TimeoutError as exc:
                result_status = "timeout"
                timeout_used = (
                    exc.timeout_secs if isinstance(exc, _ProbeTimeoutError) else self._timeout_secs
                )
                return McpServerStatus(
                    name=name,
                    display_name=display_name,
                    status="failed",
                    error=f"MCP server probe timed out after {timeout_used:.2f}s",
                    origin=origin,
                )
            except _HttpAuthRequiredError as auth_err:
                result_status = "authRequired"
                return McpServerStatus(
                    name=name,
                    display_name=display_name,
                    status="authRequired",
                    auth_url=auth_err.auth_url,
                    failed_authorization=True,
                    error=sanitize_mcp_error(auth_err.detail),
                    origin=origin,
                )
            except Exception as exc:  # noqa: BLE001 - prober must contain all transport errors
                result_status = "failed"
                return McpServerStatus(
                    name=name,
                    display_name=display_name,
                    status="failed",
                    error=sanitize_mcp_error(str(exc) or type(exc).__name__),
                    origin=origin,
                )
            finally:
                telemetry.record_mcp_probe(
                    name or "unknown",
                    status=result_status,
                    duration_s=time.perf_counter() - start_t,
                )

    async def _probe_stdio_server(
        self,
        *,
        name: str,
        display_name: str,
        command: str,
        server_cfg: Mapping[str, Any],
        session_env: Mapping[str, str] | None,
        cwd: str | None,
        origin: AgentOrigin,
    ) -> McpServerStatus:
        raw_args = server_cfg.get("args")
        args = (
            [str(a) for a in raw_args]
            if isinstance(raw_args, Sequence) and not isinstance(raw_args, (str, bytes))
            else []
        )
        merged_env = _build_stdio_probe_env(session_env, server_cfg.get("env"))

        server_cwd = server_cfg.get("cwd")
        effective_cwd = str(server_cwd) if isinstance(server_cwd, str) and server_cwd else cwd
        proto_version = _extract_version_negotiation(server_cfg)
        disabled_tools = _extract_disabled_tools(server_cfg)

        slot = _probe_reap_slot(name, command, tuple(args), effective_cwd or "")
        effective_timeout = self.initialize_timeout_for(slot)
        init_completed = False
        proc: anyio.abc.Process | None = None
        try:
            try:
                with anyio.fail_after(effective_timeout):
                    proc = await anyio.open_process(
                        [command, *args],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        env=merged_env,
                        cwd=effective_cwd,
                        start_new_session=True,
                    )
                    assert proc.stdin is not None
                    assert proc.stdout is not None
                    buffered_out = BufferedByteReceiveStream(proc.stdout)

                    req_id = 1
                    init_req = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": proto_version,
                            "capabilities": {},
                            "clientInfo": {"name": "agy-acp", "version": _AGY_ACP_VERSION},
                        },
                    }
                    await proc.stdin.send((json.dumps(init_req) + "\n").encode("utf-8"))
                    init_resp = await _read_stdio_jsonrpc_response(buffered_out, req_id)
            except TimeoutError as exc:
                if not init_completed:
                    self._record_initialize_timeout(slot)
                raise _ProbeTimeoutError(effective_timeout) from exc

            if "error" in init_resp:
                raise RuntimeError(f"MCP initialize error: {init_resp['error']}")

            init_completed = True
            self._clear_initialize_timeout(slot)

            try:
                with anyio.fail_after(self._timeout_secs):
                    initialized_notif = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    }
                    await proc.stdin.send((json.dumps(initialized_notif) + "\n").encode("utf-8"))

                    discovered: list[McpToolDef] = []
                    cursor: str | None = None
                    for _ in range(_MAX_PAGINATION_PAGES):
                        req_id += 1
                        list_params: dict[str, Any] = {"cursor": cursor} if cursor else {}
                        list_req = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "method": "tools/list",
                            "params": list_params,
                        }
                        await proc.stdin.send((json.dumps(list_req) + "\n").encode("utf-8"))
                        list_resp = await _read_stdio_jsonrpc_response(buffered_out, req_id)
                        if "error" in list_resp:
                            raise RuntimeError(f"MCP tools/list error: {list_resp['error']}")
                        result = list_resp.get("result")
                        if not isinstance(result, Mapping):
                            break
                        discovered.extend(
                            self._parse_tool_items(result.get("tools"), disabled_tools)
                        )
                        if len(discovered) >= MAX_TOOLS_PER_SERVER:
                            discovered = discovered[:MAX_TOOLS_PER_SERVER]
                            break
                        next_cursor = result.get("nextCursor")
                        if not isinstance(next_cursor, str) or not next_cursor:
                            break
                        cursor = next_cursor

                    return McpServerStatus(
                        name=name,
                        display_name=display_name,
                        status="connected",
                        tools=tuple(discovered),
                        origin=origin,
                    )
            except TimeoutError as exc:
                raise _ProbeTimeoutError(self._timeout_secs) from exc
        finally:
            if proc is not None:
                with anyio.CancelScope(shield=True):
                    if hasattr(os, "killpg") and proc.pid:
                        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                            os.killpg(proc.pid, signal.SIGKILL)
                    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        await proc.wait()

    async def _probe_http_server(
        self,
        *,
        name: str,
        display_name: str,
        url: str,
        server_cfg: Mapping[str, Any],
        origin: AgentOrigin,
    ) -> McpServerStatus:
        if not _is_safe_http_url(url):
            raise ValueError("Unsupported or invalid MCP HTTP URL: expected http:// or https://")

        headers = _normalize_kv_pairs(server_cfg.get("headers"))
        proto_version = _extract_version_negotiation(server_cfg)
        disabled_tools = _extract_disabled_tools(server_cfg)

        slot = _probe_reap_slot(name, url)
        effective_timeout = self.initialize_timeout_for(slot)
        init_completed = False
        req_id = 1
        init_req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": proto_version,
                "capabilities": {},
                "clientInfo": {"name": "agy-acp", "version": _AGY_ACP_VERSION},
            },
        }
        try:
            with anyio.fail_after(effective_timeout):
                _, resp_headers, init_resp = await anyio.to_thread.run_sync(
                    lambda: _post_mcp_http_sync(url, init_req, headers, effective_timeout),
                    abandon_on_cancel=True,
                )
        except TimeoutError as exc:
            if not init_completed:
                self._record_initialize_timeout(slot)
            raise _ProbeTimeoutError(effective_timeout) from exc

        if init_resp is not None and "error" in init_resp:
            raise RuntimeError(f"MCP initialize error: {init_resp['error']}")

        init_completed = True
        self._clear_initialize_timeout(slot)

        try:
            with anyio.fail_after(self._timeout_secs):
                mcp_session_id = resp_headers.get("mcp-session-id")
                if mcp_session_id:
                    headers = {**headers, "Mcp-Session-Id": mcp_session_id}

                initialized_notif = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
                await anyio.to_thread.run_sync(
                    lambda: _post_mcp_http_sync(
                        url,
                        initialized_notif,
                        headers,
                        self._timeout_secs,
                    ),
                    abandon_on_cancel=True,
                )

                discovered: list[McpToolDef] = []
                cursor: str | None = None
                for _ in range(_MAX_PAGINATION_PAGES):
                    req_id += 1
                    list_params: dict[str, Any] = {"cursor": cursor} if cursor else {}
                    list_req = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "method": "tools/list",
                        "params": list_params,
                    }
                    _, _, list_resp = await anyio.to_thread.run_sync(
                        lambda req=list_req: _post_mcp_http_sync(
                            url,
                            req,
                            headers,
                            self._timeout_secs,
                        ),
                        abandon_on_cancel=True,
                    )
                    if list_resp is None:
                        break
                    if "error" in list_resp:
                        raise RuntimeError(f"MCP tools/list error: {list_resp['error']}")
                    result = list_resp.get("result")
                    if not isinstance(result, Mapping):
                        break
                    discovered.extend(self._parse_tool_items(result.get("tools"), disabled_tools))
                    if len(discovered) >= MAX_TOOLS_PER_SERVER:
                        discovered = discovered[:MAX_TOOLS_PER_SERVER]
                        break
                    next_cursor = result.get("nextCursor")
                    if not isinstance(next_cursor, str) or not next_cursor:
                        break
                    cursor = next_cursor

                return McpServerStatus(
                    name=name,
                    display_name=display_name,
                    status="connected",
                    tools=tuple(discovered),
                    origin=origin,
                )
        except TimeoutError as exc:
            raise _ProbeTimeoutError(self._timeout_secs) from exc

    @staticmethod
    def _parse_tool_items(raw_tools: Any, disabled_tools: frozenset[str]) -> list[McpToolDef]:
        if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
            return []
        parsed: list[McpToolDef] = []
        for entry in raw_tools:
            if not isinstance(entry, Mapping):
                continue
            t_name = entry.get("name")
            if not isinstance(t_name, str) or not t_name.strip():
                continue
            desc = entry.get("description")
            parsed.append(
                McpToolDef(
                    name=t_name.strip(),
                    description=desc if isinstance(desc, str) else "",
                    disabled=t_name.strip() in disabled_tools or entry.get("disabled") is True,
                )
            )
        return parsed
