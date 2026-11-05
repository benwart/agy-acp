"""Async bounded MCP server catalog prober (`initialize` + `tools/list`) for stdio and HTTP/SSE."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

from agy_acp.logic.custom_agents import AgentOrigin
from agy_acp.logic.wire.base import McpServerStatus, McpToolDef

MCP_PROTOCOL_LATEST = "2025-03-26"
MCP_PROTOCOL_LEGACY = "2024-11-05"
_MAX_PAGINATION_PAGES = 20

_AUTH_URI_PARAM_RE = re.compile(
    r'(?:authorization_uri|auth_url|oauth_url|resource_metadata)\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)
_URL_FALLBACK_RE = re.compile(r"(https?://[^\s\",;>]+)")


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
        if match:
            return match.group(1)
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
                    if isinstance(val, str) and val.startswith(("http://", "https://")):
                        return val
    if www_authenticate:
        fallback = _URL_FALLBACK_RE.search(www_authenticate)
        if fallback:
            return fallback.group(1)
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


def _post_mcp_http_sync(
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout_secs: float,
) -> tuple[int, dict[str, str], dict[str, Any] | None]:
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **headers,
    }
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_secs) as resp:
            status_code = int(resp.status)
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            raw_body = resp.read()
            ctype = resp_headers.get("content-type", "application/json")
            parsed = _parse_http_rpc_body(ctype, raw_body)
            return status_code, resp_headers, parsed
    except urllib.error.HTTPError as exc:
        body_bytes = b""
        with contextlib.suppress(Exception):
            body_bytes = exc.read()
        body_text = body_bytes.decode("utf-8", errors="replace")
        if exc.code in (401, 403):
            www_auth = exc.headers.get("WWW-Authenticate", "") if exc.headers else ""
            auth_url = _extract_auth_url(www_auth, body_text)
            raise _HttpAuthRequiredError(
                exc.code,
                auth_url,
                f"HTTP {exc.code} {exc.reason}".strip(),
            ) from exc
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body_text[:200]}".strip()) from exc


async def _read_stdio_jsonrpc_response(
    stream: BufferedByteReceiveStream,
    expected_id: int,
) -> dict[str, Any]:
    """Read JSON-RPC frames from stdio until the response matching `expected_id` arrives."""
    while True:
        raw_line = await stream.receive_until(b"\n", 1_048_576)
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
            # Consume blank header separator line (`\r\n` or `\n`)
            while True:
                sep = await stream.receive_until(b"\n", 4096)
                if not sep.strip():
                    break
            body = await stream.receive_exactly(length)
            msg = json.loads(body.decode("utf-8", errors="replace"))
        else:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
        if isinstance(msg, dict) and msg.get("id") == expected_id:
            return msg


class McpCatalogProber:
    """Bounded async MCP `initialize` + `tools/list` prober for session startup & resetServer."""

    def __init__(self, *, timeout_secs: float = 10.0) -> None:
        self._timeout_secs = float(timeout_secs)

    async def probe_servers(
        self,
        servers: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
        *,
        session_env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[McpServerStatus]:
        """Probe all declared MCP servers concurrently and return ordered `McpServerStatus`."""
        normalized_list = self._normalize_server_declarations(servers)
        if not normalized_list:
            return []

        results: list[McpServerStatus | None] = [None] * len(normalized_list)

        async def _run_one(idx: int, cfg: dict[str, Any]) -> None:
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
        try:
            if isinstance(url, str) and url.strip():
                return await self._probe_http_server(
                    name=name,
                    display_name=display_name,
                    url=url.strip(),
                    server_cfg=server_cfg,
                    origin=origin,
                )
            if isinstance(command, str) and command.strip():
                return await self._probe_stdio_server(
                    name=name,
                    display_name=display_name,
                    command=command.strip(),
                    server_cfg=server_cfg,
                    session_env=session_env,
                    cwd=cwd,
                    origin=origin,
                )
            return McpServerStatus(
                name=name,
                display_name=display_name,
                status="failed",
                error="MCP server config specifies neither 'command' nor 'url'",
                origin=origin,
            )
        except TimeoutError:
            return McpServerStatus(
                name=name,
                display_name=display_name,
                status="failed",
                error=f"MCP server probe timed out after {self._timeout_secs:.2f}s",
                origin=origin,
            )
        except _HttpAuthRequiredError as auth_err:
            return McpServerStatus(
                name=name,
                display_name=display_name,
                status="authRequired",
                auth_url=auth_err.auth_url,
                failed_authorization=True,
                error=auth_err.detail,
                origin=origin,
            )
        except Exception as exc:  # noqa: BLE001 - prober must contain all transport errors
            return McpServerStatus(
                name=name,
                display_name=display_name,
                status="failed",
                error=str(exc) or type(exc).__name__,
                origin=origin,
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
        merged_env = dict(os.environ)
        if session_env:
            merged_env.update({str(k): str(v) for k, v in session_env.items()})
        merged_env.update(_normalize_kv_pairs(server_cfg.get("env")))

        server_cwd = server_cfg.get("cwd")
        effective_cwd = str(server_cwd) if isinstance(server_cwd, str) and server_cwd else cwd
        proto_version = _extract_version_negotiation(server_cfg)
        disabled_tools = _extract_disabled_tools(server_cfg)

        proc: anyio.abc.Process | None = None
        try:
            with anyio.fail_after(self._timeout_secs):
                proc = await anyio.open_process(
                    [command, *args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=merged_env,
                    cwd=effective_cwd,
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
                        "clientInfo": {"name": "agy-acp", "version": "1.2.0"},
                    },
                }
                await proc.stdin.send((json.dumps(init_req) + "\n").encode("utf-8"))
                init_resp = await _read_stdio_jsonrpc_response(buffered_out, req_id)
                if "error" in init_resp:
                    raise RuntimeError(f"MCP initialize error: {init_resp['error']}")

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
                    discovered.extend(self._parse_tool_items(result.get("tools"), disabled_tools))
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
        finally:
            if proc is not None:
                with anyio.CancelScope(shield=True):
                    with contextlib.suppress(ProcessLookupError, OSError):
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
        headers = _normalize_kv_pairs(server_cfg.get("headers"))
        proto_version = _extract_version_negotiation(server_cfg)
        disabled_tools = _extract_disabled_tools(server_cfg)

        with anyio.fail_after(self._timeout_secs):
            req_id = 1
            init_req = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": proto_version,
                    "capabilities": {},
                    "clientInfo": {"name": "agy-acp", "version": "1.2.0"},
                },
            }
            _, resp_headers, init_resp = await anyio.to_thread.run_sync(
                lambda: _post_mcp_http_sync(url, init_req, headers, self._timeout_secs),
                abandon_on_cancel=True,
            )
            if init_resp is not None and "error" in init_resp:
                raise RuntimeError(f"MCP initialize error: {init_resp['error']}")

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
