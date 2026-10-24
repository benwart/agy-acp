"""Top-level ACP server command."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import anyio

from agy_acp.logic.acp_dispatcher import AcpDispatcher
from agy_acp.logic.auth_manager import AuthManager
from agy_acp.logic.session_manager import SessionManager


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
    )
    write_lock = anyio.Lock()
    pending_events: dict[Any, anyio.Event] = {}
    pending_results: dict[Any, dict[str, Any]] = {}
    next_rpc_seq = 0

    session_manager = SessionManager(save_dir=save_dir)
    auth_manager = AuthManager(auth_file=auth_file) if auth_file is not None else None
    dispatcher: AcpDispatcher

    async def _write_frame(frame: bytes) -> None:
        async with write_lock:
            buffer = getattr(sys.stdout, "buffer", None)
            if buffer is not None:
                buffer.write(frame)
                buffer.flush()
            else:
                sys.stdout.write(frame.decode("utf-8"))
                sys.stdout.flush()

    async def _notification_sink(notification: dict[str, Any]) -> None:
        await _write_frame(dispatcher.serialize_frame(notification))

    async def _client_rpc_caller(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_rpc_seq
        next_rpc_seq += 1
        rpc_id = f"srv_{next_rpc_seq}"
        event = anyio.Event()
        pending_events[rpc_id] = event
        request_payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
            "params": params,
        }
        await _write_frame(dispatcher.serialize_frame(request_payload))
        await event.wait()
        pending_events.pop(rpc_id, None)
        return pending_results.pop(rpc_id, {})

    dispatcher = AcpDispatcher(
        notification_sink=_notification_sink,
        client_rpc_caller=_client_rpc_caller,
        session_manager=session_manager,
        auth_manager=auth_manager,
    )

    async def _dispatch_line(raw_line: str) -> None:
        response = await dispatcher.handle_message(raw_line)
        if response is not None:
            await _write_frame(dispatcher.serialize_frame(response))

    async with anyio.create_task_group() as tg:
        while True:
            try:
                raw_line = await anyio.to_thread.run_sync(sys.stdin.readline)
            except io.UnsupportedOperation, OSError:
                break
            if not raw_line:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(parsed, dict)
                and "method" not in parsed
                and ("result" in parsed or "error" in parsed)
            ):
                resp_id = parsed.get("id")
                result_obj = parsed.get("result")
                pending_results[resp_id] = result_obj if isinstance(result_obj, dict) else {}
                pending_event = pending_events.get(resp_id)
                if pending_event is not None:
                    pending_event.set()
                continue
            tg.start_soon(_dispatch_line, line)

    return 0
