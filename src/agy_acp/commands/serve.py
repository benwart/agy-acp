"""Top-level ACP server command."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import anyio


async def default(
    profile: str = "acp",
    patch: Path | None = None,
    agent: str | None = None,
    model: str | None = None,
    with_builtin: str | None = None,
    agent_engine: str | None = None,
    auth_method: str | None = None,
    auth_file: Path | None = None,
    save_dir: Path | None = None,
) -> int:
    """Start the Agent Client Protocol (ACP) JSON-RPC 2.0 server over stdio."""
    _ = (
        profile,
        patch,
        agent,
        model,
        with_builtin,
        agent_engine,
        auth_method,
        auth_file,
        save_dir,
    )
    await anyio.sleep(0)
    while True:
        try:
            raw_line = sys.stdin.readline()
        except io.UnsupportedOperation, OSError:
            break
        if not raw_line:
            break
        line = raw_line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "id" in msg:
            reply = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {"code": -32601, "message": "Not implemented"},
            }
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0
