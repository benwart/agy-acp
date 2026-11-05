"""Unit tests for async bounded McpCatalogProber (stdio, HTTP/SSE, legacy version, OAuth 401)."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

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
