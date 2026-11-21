"""Unit tests for async bounded McpCatalogProber (stdio, HTTP/SSE, legacy version, OAuth 401)."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from agy_acp.logic.mcp_catalog_prober import McpCatalogProber


async def test_stdio_probe_tools_pagination_env_and_disabled_tools(tmp_path: Path) -> None:
    """Verify stdio MCP handshake, env list injection, cursor pagination, and disabledTools."""
    script = tmp_path / "mock_mcp_stdio.py"
    script.write_text(
        "import json, os, sys\n"
        "seen_init = False\n"
        "proto = ''\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        proto = msg['params']['protocolVersion']\n"
        "        resp = {\n"
        "            'jsonrpc': '2.0',\n"
        "            'id': msg['id'],\n"
        "            'result': {\n"
        "                'protocolVersion': proto,\n"
        "                'capabilities': {'tools': {}},\n"
        "                'serverInfo': {'name': 'mock-stdio', 'version': '1.0'},\n"
        "            },\n"
        "        }\n"
        "        sys.stdout.write(json.dumps(resp) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif method == 'notifications/initialized':\n"
        "        seen_init = True\n"
        "    elif method == 'tools/list':\n"
        "        cursor = (msg.get('params') or {}).get('cursor')\n"
        "        env_tag = os.environ.get('PROBE_TOKEN', 'missing')\n"
        "        if not cursor:\n"
        "            result = {\n"
        "                'tools': [\n"
        "                    {\n"
        "                        'name': 'spawn_run',\n"
        "                        'description': f'proto={proto} env={env_tag}',\n"
        "                    },\n"
        "                ],\n"
        "                'nextCursor': 'page2',\n"
        "            }\n"
        "        else:\n"
        "            result = {\n"
        "                'tools': [\n"
        "                    {'name': 'muted_tool', 'description': f'init={seen_init}'},\n"
        "                ],\n"
        "            }\n"
        "        sys.stdout.write(\n"
        "            json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}) + '\\n'\n"
        "        )\n"
        "        sys.stdout.flush()\n"
        "        if cursor:\n"
        "            break\n",
        encoding="utf-8",
    )

    prober = McpCatalogProber(timeout_secs=5.0)
    statuses = await prober.probe_servers(
        [
            {
                "name": "kirocrew-core",
                "displayName": "KiroCrew Core",
                "command": sys.executable,
                "args": ["-u", str(script)],
                "env": [{"name": "PROBE_TOKEN", "value": "ok123"}],
                "disabledTools": ["muted_tool"],
            }
        ]
    )
    assert len(statuses) == 1
    srv = statuses[0]
    assert srv.name == "kirocrew-core"
    assert srv.display_name == "KiroCrew Core"
    assert srv.status == "connected"
    assert len(srv.tools) == 2
    assert srv.tools[0].name == "spawn_run"
    assert srv.tools[0].description == "proto=2025-03-26 env=ok123"
    assert srv.tools[0].disabled is False
    assert srv.tools[1].name == "muted_tool"
    assert srv.tools[1].description == "init=True"
    assert srv.tools[1].disabled is True


async def test_stdio_legacy_version_negotiation_timeout_and_failures(tmp_path: Path) -> None:
    """Verify legacy 2024-11-05 negotiation, bounded timeout, missing binary, and disabled flag."""
    legacy_script = tmp_path / "legacy_mcp.py"
    legacy_script.write_text(
        "import json, sys\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    m = msg.get('method')\n"
        "    if m == 'initialize':\n"
        "        pv = msg['params']['protocolVersion']\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': {'protocolVersion': pv, 'capabilities': {}}}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif m == 'tools/list':\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': {'tools': [{'name': f'ver_{pv}', 'description': 'legacy'}]}}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "        break\n",
        encoding="utf-8",
    )

    hang_script = tmp_path / "hang_mcp.py"
    hang_script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    prober = McpCatalogProber(timeout_secs=0.25)
    results = await prober.probe_servers(
        {
            "legacy-srv": {
                "command": sys.executable,
                "args": ["-u", str(legacy_script)],
                "_meta": {"kiro": {"versionNegotiation": "legacy"}},
            },
            "hanging-srv": {
                "command": sys.executable,
                "args": ["-u", str(hang_script)],
            },
            "missing-bin": {
                "command": "/nonexistent/mcp-binary-xyz",
                "args": [],
            },
            "muted-srv": {
                "command": sys.executable,
                "args": [],
                "disabled": True,
            },
        }
    )
    by_name = {r.name: r for r in results}
    assert by_name["legacy-srv"].status == "connected"
    assert by_name["legacy-srv"].tools[0].name == "ver_2024-11-05"

    assert by_name["hanging-srv"].status == "failed"
    assert by_name["hanging-srv"].error is not None
    assert "timed out" in by_name["hanging-srv"].error.lower()

    assert by_name["missing-bin"].status == "failed"
    assert by_name["missing-bin"].error is not None

    assert by_name["muted-srv"].status == "disabled"
    assert by_name["muted-srv"].tools == ()


async def test_http_sse_and_oauth_401_probing() -> None:
    """Verify Streamable-HTTP SSE response parsing and 401 WWW-Authenticate authRequired status."""
    seen_headers: dict[str, str] = {}

    class _MockMcpHttpHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/oauth-protected":
                self.send_response(401)
                self.send_header(
                    "WWW-Authenticate",
                    'Bearer realm="mcp", authorization_uri="https://idp.example.com/oauth/authorize"',
                )
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "unauthorized"}')
                return

            seen_headers["x-custom"] = self.headers.get("X-Custom", "")
            method = body.get("method")
            if method == "initialize":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Mcp-Session-Id", "mcp-sess-99")
                self.end_headers()
                resp = {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {}},
                    },
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            elif method == "notifications/initialized":
                seen_headers["mcp-session-id"] = self.headers.get("Mcp-Session-Id", "")
                self.send_response(202)
                self.end_headers()
            elif method == "tools/list":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                sse_payload = {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "tools": [
                            {"name": "remote_lookup", "description": "SSE tool"},
                        ]
                    },
                }
                frame = f"event: message\ndata: {json.dumps(sse_payload)}\n\n"
                self.wfile.write(frame.encode("utf-8"))

    httpd = HTTPServer(("127.0.0.1", 0), _MockMcpHttpHandler)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        prober = McpCatalogProber(timeout_secs=5.0)
        statuses = await prober.probe_servers(
            [
                {
                    "name": "http-sse-srv",
                    "url": f"http://127.0.0.1:{port}/mcp",
                    "headers": [{"name": "X-Custom", "value": "header-ok"}],
                },
                {
                    "name": "oauth-http-srv",
                    "url": f"http://127.0.0.1:{port}/oauth-protected",
                },
            ]
        )
        by_name = {s.name: s for s in statuses}
        sse_srv = by_name["http-sse-srv"]
        assert sse_srv.status == "connected"
        assert len(sse_srv.tools) == 1
        assert sse_srv.tools[0].name == "remote_lookup"
        assert seen_headers["x-custom"] == "header-ok"
        assert seen_headers["mcp-session-id"] == "mcp-sess-99"

        oauth_srv = by_name["oauth-http-srv"]
        assert oauth_srv.status == "authRequired"
        assert oauth_srv.failed_authorization is True
        assert oauth_srv.auth_url == "https://idp.example.com/oauth/authorize"
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_http_scheme_validation_redirect_ssrf_auth_url_and_error_scrubbing(
    tmp_path: Path,
) -> None:
    """Verify HIGH-02 scheme/redirect/authUrl defenses and MEDIUM-01 error secret scrubbing."""
    secret_file = tmp_path / "secret.json"
    secret_file.write_text('{"secret": "DO_NOT_READ"}', encoding="utf-8")

    class _EvilHttpHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                _ = self.rfile.read(length)
            if self.path.startswith("/redirect-file"):
                self.send_response(302)
                self.send_header("Location", secret_file.as_uri())
                self.end_headers()
                return
            if self.path.startswith("/bad-oauth-js"):
                self.send_response(401)
                self.send_header(
                    "WWW-Authenticate",
                    'Bearer authorization_uri="javascript:alert(document.cookie)"',
                )
                self.end_headers()
                self.wfile.write(b'{"authorization_url": "file:///etc/passwd"}')
                return
            if self.path.startswith("/error-500"):
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Internal leak: Bearer SUPER_SECRET_BEARER_999 token=RAW_TOK_1")
                return

    httpd = HTTPServer(("127.0.0.1", 0), _EvilHttpHandler)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        prober = McpCatalogProber(timeout_secs=5.0)
        statuses = await prober.probe_servers(
            [
                {"name": "file-srv", "url": secret_file.as_uri()},
                {"name": "ftp-srv", "url": "ftp://example.com/mcp"},
                {"name": "redirect-srv", "url": f"http://127.0.0.1:{port}/redirect-file"},
                {"name": "bad-oauth", "url": f"http://127.0.0.1:{port}/bad-oauth-js"},
                {
                    "name": "err-500",
                    "url": f"http://user:TOP_PASS@127.0.0.1:{port}/error-500?api_key=Q_SECRET_77",
                },
            ]
        )
        by_name = {s.name: s for s in statuses}

        assert by_name["file-srv"].status == "failed"
        assert "DO_NOT_READ" not in (by_name["file-srv"].error or "")

        assert by_name["ftp-srv"].status == "failed"

        assert by_name["redirect-srv"].status == "failed"
        assert "DO_NOT_READ" not in (by_name["redirect-srv"].error or "")

        assert by_name["bad-oauth"].status == "authRequired"
        assert by_name["bad-oauth"].auth_url is None

        err_srv = by_name["err-500"]
        assert err_srv.status == "failed"
        assert err_srv.error is not None
        assert "SUPER_SECRET_BEARER_999" not in err_srv.error
        assert "RAW_TOK_1" not in err_srv.error
        assert "TOP_PASS" not in err_srv.error
        assert "Q_SECRET_77" not in err_srv.error
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_stdio_oversized_content_length_process_group_cleanup_and_error_scrubbing(
    tmp_path: Path,
) -> None:
    """Verify HIGH-03 oversized Content-Length rejection, pgid kill, and query secret scrubbing."""
    oversized_script = tmp_path / "oversized_mcp.py"
    oversized_script.write_text(
        "import sys\n"
        "_ = sys.stdin.readline()\n"
        "sys.stdout.write('Content-Length: 999999999\\r\\n\\r\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )

    grandchild_pid_file = tmp_path / "grandchild.pid"
    wrapper_script = tmp_path / "wrapper_mcp.py"
    wrapper_script.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(grandchild_pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    noisy_err_script = tmp_path / "noisy_err.py"
    noisy_err_script.write_text(
        "import json, sys\n"
        "msg = json.loads(sys.stdin.readline())\n"
        "err = {\n"
        "    'jsonrpc': '2.0',\n"
        "    'id': msg['id'],\n"
        "    'error': {\n"
        "        'code': -32000,\n"
        "        'message': 'Upstream https://admin:PW123@api.internal/v1?token=TOK999 '\n"
        "                   'Bearer SECRET_JWT_ABC secret=MY_SEC_42',\n"
        "    },\n"
        "}\n"
        "sys.stdout.write(json.dumps(err) + '\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )

    prober = McpCatalogProber(timeout_secs=0.4)
    statuses = await prober.probe_servers(
        [
            {
                "name": "oversized-srv",
                "command": sys.executable,
                "args": ["-u", str(oversized_script)],
            },
            {
                "name": "wrapper-srv",
                "command": sys.executable,
                "args": ["-u", str(wrapper_script)],
            },
            {
                "name": "noisy-err-srv",
                "command": sys.executable,
                "args": ["-u", str(noisy_err_script)],
            },
        ]
    )
    by_name = {s.name: s for s in statuses}

    assert by_name["oversized-srv"].status == "failed"
    assert "content-length" in (by_name["oversized-srv"].error or "").lower()

    assert by_name["wrapper-srv"].status == "failed"
    assert grandchild_pid_file.is_file()
    grandchild_pid = int(grandchild_pid_file.read_text(encoding="utf-8").strip())
    # Give OS process table brief moment to reap/clear killed grandchild
    deadline = time.monotonic() + 2.0
    alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.05)
    if alive:
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, signal.SIGKILL)
        pytest.fail(f"Orphan grandchild process {grandchild_pid} was not killed with process group")

    noisy_err = by_name["noisy-err-srv"].error or ""
    assert by_name["noisy-err-srv"].status == "failed"
    assert "PW123" not in noisy_err
    assert "TOK999" not in noisy_err
    assert "SECRET_JWT_ABC" not in noisy_err
    assert "MY_SEC_42" not in noisy_err


async def test_stdio_probe_scrubs_ambient_and_empty_session_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify stdio probing omits empty session_env placeholders and scrubs ambient KiroCrew env."""
    from agy_acp.logic.session_manager import _extract_session_mcp_env
    from agy_acp.logic.tool_classifier import PER_SESSION_MCP_ENV_KEYS

    assert "KIROCREW_BOUND_PORT" in PER_SESSION_MCP_ENV_KEYS

    monkeypatch.setenv("KIROCREW_PORT", "9999")
    monkeypatch.setenv("KIROCREW_SESSION_KEY", "ambient:leaked-session")

    strict_port_script = tmp_path / "strict_port_mcp.py"
    strict_port_script.write_text(
        "import json, os, sys\n"
        "# Mirrors loader_credentials: int(os.environ.get('KIROCREW_PORT', 5476))\n"
        "port = int(os.environ.get('KIROCREW_PORT', 5476))\n"
        "bound = os.environ.get('KIROCREW_BOUND_PORT', 'unset')\n"
        "sess_key = os.environ.get('KIROCREW_SESSION_KEY', 'unset')\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    m = msg.get('method')\n"
        "    if m == 'initialize':\n"
        "        pv = msg['params']['protocolVersion']\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': {'protocolVersion': pv, 'capabilities': {'tools': {}}}}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif m == 'tools/list':\n"
        "        desc = f'port={port} bound={bound} key={sess_key}'\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': {'tools': [{'name': 'chat_folder_tree', 'description': desc}]}}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "        break\n",
        encoding="utf-8",
    )

    server_decl = {
        "name": "kirocrew-dashboard",
        "command": sys.executable,
        "args": ["-u", str(strict_port_script)],
        "env": [
            {"name": "KIROCREW_BOUND_PORT", "value": "5476"},
            {"name": "KIROCREW_SESSION_KEY", "value": "dashboard:slot-1"},
        ],
        "_meta": {"kiro": {"versionNegotiation": "legacy"}},
    }
    session_env = _extract_session_mcp_env(
        {"mcp_servers": [server_decl]},
        session_id="sess-1",
        state_db_path=tmp_path / "state.db",
    )

    prober = McpCatalogProber(timeout_secs=5.0)
    statuses = await prober.probe_servers([server_decl], session_env=session_env, cwd=str(tmp_path))
    assert len(statuses) == 1
    assert statuses[0].status == "connected", f"Unexpected probe error: {statuses[0].error}"
    assert len(statuses[0].tools) == 1
    assert statuses[0].tools[0].description == "port=5476 bound=5476 key=dashboard:slot-1"

    # Also verify a server declaration with NO env at all does not inherit ambient KIROCREW_* vars
    bare_decl = {
        "name": "bare-mcp",
        "command": sys.executable,
        "args": ["-u", str(strict_port_script)],
    }
    bare_env = _extract_session_mcp_env({"mcp_servers": [bare_decl]}, session_id="sess-2")
    bare_statuses = await prober.probe_servers([bare_decl], session_env=bare_env, cwd=str(tmp_path))
    assert bare_statuses[0].status == "connected", f"Unexpected error: {bare_statuses[0].error}"
    assert bare_statuses[0].tools[0].description == "port=5476 bound=unset key=unset"


async def test_bare_401_rfc9728_discovery_and_security_invariants() -> None:
    """Verify RFC 9728 OAuth metadata discovery on bare 401 and all security negative cases."""
    from agy_acp import __version__ as agy_version
    from agy_acp.logic.mcp_catalog_prober import (
        _is_valid_authorization_server_url,
        normalized_mcp_endpoint,
    )

    assert (
        normalized_mcp_endpoint("HTTPS://Example.COM:443/mcp/?b=2#frag")
        == "https://example.com/mcp?b=2"
    )
    assert normalized_mcp_endpoint("http://Example.COM:80/mcp/") == "http://example.com/mcp"
    assert (
        normalized_mcp_endpoint("http://user:pw@Example.COM:8080/mcp/")
        == "http://user:pw@example.com:8080/mcp"
    )
    assert normalized_mcp_endpoint("ftp://example.com/mcp") is None
    assert normalized_mcp_endpoint("https://example.com:99999/mcp") is None
    assert normalized_mcp_endpoint(123) is None

    # Non-loopback MCP server strictly requires non-local, non-IP-literal https:// auth server
    assert _is_valid_authorization_server_url("https://auth.stripe.com", mcp_is_loopback=False)
    assert not _is_valid_authorization_server_url("http://auth.stripe.com", mcp_is_loopback=False)
    assert not _is_valid_authorization_server_url("https://127.0.0.1/oauth", mcp_is_loopback=False)
    assert not _is_valid_authorization_server_url(
        "https://192.168.1.10/oauth", mcp_is_loopback=False
    )
    assert not _is_valid_authorization_server_url(
        "https://broker.local/oauth", mcp_is_loopback=False
    )
    assert _is_valid_authorization_server_url("http://127.0.0.1:9000/oauth", mcp_is_loopback=True)
    assert not _is_valid_authorization_server_url(
        "http://evil.example.com/oauth", mcp_is_loopback=True
    )

    seen_well_known_headers: list[dict[str, str]] = []
    seen_client_versions: list[str] = []

    class _Rfc9728Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if body.get("method") == "initialize":
                seen_client_versions.append(
                    str((body.get("params") or {}).get("clientInfo", {}).get("version", ""))
                )
            # All endpoints respond with bare 401 (no WWW-Authenticate, no auth URL in body)
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "unauthorized"}')

        def do_GET(self) -> None:
            seen_well_known_headers.append({k.lower(): v for k, v in self.headers.items()})
            assert isinstance(self.server, HTTPServer)
            srv_port = self.server.server_port
            origin = f"http://127.0.0.1:{srv_port}"
            if self.path == "/.well-known/oauth-protected-resource/mcp":
                self._send_json(
                    {
                        "resource": f"{origin}/mcp/",
                        "authorization_servers": ["https://auth.example.com/path-inserted"],
                    }
                )
                return
            if self.path == "/.well-known/oauth-protected-resource/root-fallback":
                self.send_response(404)
                self.end_headers()
                return
            if self.path == "/.well-known/oauth-protected-resource":
                # Resource list contains root-fallback (matching) and /other (sibling)
                self._send_json(
                    {
                        "resource": [f"{origin}/other", f"{origin}/root-fallback"],
                        "authorization_servers": [f"http://127.0.0.1:{srv_port}/as"],
                    }
                )
                return
            if self.path == "/.well-known/oauth-protected-resource/redirect-case":
                self.send_response(302)
                self.send_header("Location", "/.well-known/oauth-protected-resource/mcp")
                self.end_headers()
                return
            if self.path == "/.well-known/oauth-protected-resource/oversized-case":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                padding = "x" * (17 * 1024)
                payload = json.dumps(
                    {
                        "resource": f"{origin}/oversized-case",
                        "authorization_servers": ["https://auth.example.com/oversized"],
                        "pad": padding,
                    }
                ).encode("utf-8")
                self.wfile.write(payload)
                return
            if self.path == "/.well-known/oauth-protected-resource/bad-query":
                self._send_json(
                    {
                        "resource": f"{origin}/bad-query",
                        "authorization_servers": ["https://auth.example.com/as?evil=1"],
                    }
                )
                return
            if self.path == "/.well-known/oauth-protected-resource/bad-frag":
                self._send_json(
                    {
                        "resource": f"{origin}/bad-frag",
                        "authorization_servers": ["https://auth.example.com/as#frag"],
                    }
                )
                return
            if self.path == "/.well-known/oauth-protected-resource/bad-userinfo":
                self._send_json(
                    {
                        "resource": f"{origin}/bad-userinfo",
                        "authorization_servers": ["https://user:pass@auth.example.com/as"],
                    }
                )
                return
            self.send_response(404)
            self.end_headers()

        def _send_json(self, doc: dict[str, Any]) -> None:
            raw = json.dumps(doc).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    httpd = HTTPServer(("127.0.0.1", 0), _Rfc9728Handler)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        prober = McpCatalogProber(timeout_secs=5.0)
        secret_headers = [
            {"name": "Authorization", "value": "Bearer TOP_SECRET_TOKEN"},
            {"name": "X-Custom", "value": "SECRET_HEADER"},
        ]
        statuses = await prober.probe_servers(
            [
                {
                    "name": "path-inserted",
                    "url": f"http://127.0.0.1:{port}/mcp",
                    "headers": secret_headers,
                },
                {
                    "name": "root-fallback",
                    "url": f"http://127.0.0.1:{port}/root-fallback",
                    "headers": secret_headers,
                },
                {
                    "name": "sibling-mismatch",
                    "url": f"http://127.0.0.1:{port}/sibling-unmatched",
                },
                {
                    "name": "redirect-case",
                    "url": f"http://127.0.0.1:{port}/redirect-case",
                },
                {
                    "name": "oversized-case",
                    "url": f"http://127.0.0.1:{port}/oversized-case",
                },
                {"name": "bad-query", "url": f"http://127.0.0.1:{port}/bad-query"},
                {"name": "bad-frag", "url": f"http://127.0.0.1:{port}/bad-frag"},
                {"name": "bad-userinfo", "url": f"http://127.0.0.1:{port}/bad-userinfo"},
            ]
        )
        by_name = {s.name: s for s in statuses}

        assert by_name["path-inserted"].status == "authRequired"
        assert by_name["path-inserted"].auth_url == "https://auth.example.com/path-inserted"

        assert by_name["root-fallback"].status == "authRequired"
        assert by_name["root-fallback"].auth_url == f"http://127.0.0.1:{port}/as"

        for rejected in (
            "sibling-mismatch",
            "redirect-case",
            "oversized-case",
            "bad-query",
            "bad-frag",
            "bad-userinfo",
        ):
            assert by_name[rejected].status == "authRequired"
            assert by_name[rejected].auth_url is None, f"Expected None for {rejected}"

        # Custom/secret headers MUST NEVER be forwarded to well-known metadata GETs
        assert seen_well_known_headers
        for hdrs in seen_well_known_headers:
            assert "authorization" not in hdrs
            assert "x-custom" not in hdrs
            assert hdrs.get("accept") == "application/json"
            assert hdrs.get("accept-encoding") == "identity"

        # Verify clientInfo.version matches agy_acp.__version__
        assert seen_client_versions
        assert all(v == agy_version for v in seen_client_versions)
    finally:
        httpd.shutdown()
        httpd.server_close()


async def test_adaptive_initialize_deadline_stretch_and_eviction(tmp_path: Path) -> None:
    """Verify 2x initialize deadline stretch, reset on success, non-timeout isolation, and cap."""
    from agy_acp.logic.mcp_catalog_prober import (
        INIT_DEADLINE_REAP_KEYS_MAX,
        INIT_DEADLINE_STRETCH_CAP_SECS,
        _probe_reap_slot,
    )

    # 1. Pure formula & eviction cap checks
    cap_prober = McpCatalogProber(timeout_secs=30.0)
    s0 = _probe_reap_slot("slow", "cmd", (), "")
    assert cap_prober.initialize_timeout_for(s0) == 30.0
    cap_prober._record_initialize_timeout(s0)
    assert cap_prober.initialize_timeout_for(s0) == 60.0
    cap_prober._record_initialize_timeout(s0)
    assert cap_prober.initialize_timeout_for(s0) == INIT_DEADLINE_STRETCH_CAP_SECS
    cap_prober._record_initialize_timeout(s0)
    assert cap_prober.initialize_timeout_for(s0) == INIT_DEADLINE_STRETCH_CAP_SECS

    # High base timeout (> 120s) is never lowered
    high_prober = McpCatalogProber(timeout_secs=180.0)
    high_prober._record_initialize_timeout(s0)
    assert high_prober.initialize_timeout_for(s0) == 180.0

    # Evicts oldest slot once exceeding INIT_DEADLINE_REAP_KEYS_MAX (256)
    evict_prober = McpCatalogProber(timeout_secs=10.0)
    first_slot = _probe_reap_slot("srv-0", "cmd", (), "")
    for idx in range(INIT_DEADLINE_REAP_KEYS_MAX + 5):
        evict_prober._record_initialize_timeout(_probe_reap_slot(f"srv-{idx}", "cmd", (), ""))
    assert len(evict_prober._deadline_reaps) == INIT_DEADLINE_REAP_KEYS_MAX
    assert first_slot not in evict_prober._deadline_reaps

    # 2. Live stdio behavioral verification:
    #    - 1st init hangs -> times out at 0.15s, stretches to 0.30s
    #    - non-timeout failure (JSON-RPC error) -> leaves reap count at 1 (0.30s)
    #    - 2nd init succeeds (sleeps 0.20s, which only passes because stretched to 0.30s!) -> resets
    #    - tools/list timeout after init succeeds -> does NOT increment reap count
    state_file = tmp_path / "mode.txt"
    adaptive_script = tmp_path / "adaptive_mcp.py"
    adaptive_script.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        f"mode = Path({str(state_file)!r}).read_text(encoding='utf-8').strip()\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw)\n"
        "    m = msg.get('method')\n"
        "    if m == 'initialize':\n"
        "        if mode == 'hang_init':\n"
        "            time.sleep(5)\n"
        "        elif mode == 'error_init':\n"
        "            sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'error': {'code': -32000, 'message': 'init failed'}}) + '\\n')\n"
        "            sys.stdout.flush()\n"
        "            break\n"
        "        elif mode == 'slow_init_ok':\n"
        "            time.sleep(0.20)\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': {'protocolVersion': '2025-03-26', 'capabilities': {'tools': {}}}}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "    elif m == 'tools/list':\n"
        "        if mode == 'hang_list':\n"
        "            time.sleep(5)\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': {'tools': [{'name': 'ok_tool', 'description': 'ok'}]}}) + '\\n')\n"
        "        sys.stdout.flush()\n"
        "        break\n",
        encoding="utf-8",
    )

    cfg = {
        "name": "adaptive-srv",
        "command": sys.executable,
        "args": ["-u", str(adaptive_script)],
    }
    slot = _probe_reap_slot("adaptive-srv", sys.executable, ("-u", str(adaptive_script)), "")
    prober = McpCatalogProber(timeout_secs=0.15)

    state_file.write_text("hang_init", encoding="utf-8")
    r1 = await prober.probe_server(cfg)
    assert r1.status == "failed"
    assert "0.15s" in (r1.error or "")
    assert prober.initialize_timeout_for(slot) == pytest.approx(0.30)

    # Non-timeout failure must leave reap count unchanged
    state_file.write_text("error_init", encoding="utf-8")
    r_err = await prober.probe_server(cfg)
    assert r_err.status == "failed"
    assert prober.initialize_timeout_for(slot) == pytest.approx(0.30)

    # Slow init (0.20s) now succeeds under 0.30s stretched deadline and resets reap count
    state_file.write_text("slow_init_ok", encoding="utf-8")
    r2 = await prober.probe_server(cfg)
    assert r2.status == "connected", f"Expected connected under stretched deadline: {r2.error}"
    assert prober.initialize_timeout_for(slot) == pytest.approx(0.15)

    # tools/list timeout after initialize succeeded does NOT increment reap count
    state_file.write_text("hang_list", encoding="utf-8")
    r3 = await prober.probe_server(cfg)
    assert r3.status == "failed"
    assert "0.15s" in (r3.error or "")
    assert prober.initialize_timeout_for(slot) == pytest.approx(0.15)
