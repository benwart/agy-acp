"""Top-level ACP server command."""

from __future__ import annotations

import io
import json
import sys
import traceback
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from cyclopts import Parameter

from agy_acp.logic.acp_dispatcher import AcpDispatcher
from agy_acp.logic.auth_manager import AuthManager, ensure_kiro_identity_store
from agy_acp.logic.session_manager import SessionManager
from agy_acp.providers.acp_logger import log_acp_event, snapshot_relevant_env

AGENT_ENGINE_HELP = "Agent engine wire protocol (v2, v3)."
_V2_ENGINE_ALIASES: frozenset[str] = frozenset({"", "default", "v2", "kiro"})
_V3_ENGINE_ALIASES: frozenset[str] = frozenset({"v3", "kas"})


def normalize_agent_engine(agent_engine: str | None) -> Literal["v2", "v3"]:
    """Normalize `--agent-engine` flag values into canonical `'v2'` or `'v3'`."""
    cleaned = (agent_engine or "").strip().lower()
    if cleaned in _V2_ENGINE_ALIASES:
        return "v2"
    if cleaned in _V3_ENGINE_ALIASES:
        return "v3"
    raise ValueError(f"Unsupported --agent-engine {agent_engine!r}. Expected 'v2' or 'v3'.")


async def default(
    profile: str = "acp",
    patch: Path | None = None,
    agent: str | None = None,
    model: str | None = None,
    with_builtin: str | None = None,
    agent_engine: Annotated[str | None, Parameter(help=AGENT_ENGINE_HELP)] = None,
    auth_method: str | None = None,
    auth_file: Path | None = None,
    save_dir: Path | None = None,
    stdio: bool = False,
    experimental_acp: bool = False,
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    *extra_args: str,
    **extra_kwargs: Any,
) -> int:
    """Start the Agent Client Protocol (ACP) JSON-RPC 2.0 server over stdio."""
    try:
        normalized_engine = normalize_agent_engine(agent_engine)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if save_dir is None:
        ensure_kiro_identity_store()
    log_acp_event(
        "serve.start",
        save_dir_or_db=save_dir,
        argv=list(sys.argv),
        cwd=str(Path.cwd()),
        params={
            "profile": profile,
            "patch": str(patch) if patch else None,
            "agent": agent,
            "model": model,
            "with_builtin": with_builtin,
            "agent_engine": agent_engine,
            "normalized_agent_engine": normalized_engine,
            "auth_method": auth_method,
            "auth_file": str(auth_file) if auth_file else None,
            "save_dir": str(save_dir) if save_dir else None,
            "stdio": stdio,
            "experimental_acp": experimental_acp,
            "verbose": verbose,
            "debug": debug,
            "quiet": quiet,
            "extra_args": list(extra_args),
            "extra_kwargs": extra_kwargs,
        },
        env=snapshot_relevant_env(),
    )
    write_lock = anyio.Lock()
    pending_events: dict[Any, anyio.Event] = {}
    pending_results: dict[Any, dict[str, Any]] = {}
    next_rpc_seq = 0

    session_manager = SessionManager(
        save_dir=save_dir,
        default_agent=agent,
        default_model=model,
    )
    auth_manager = AuthManager(auth_file=auth_file) if auth_file is not None else None
    dispatcher: AcpDispatcher

    async def _write_frame(frame: bytes) -> None:
        log_acp_event(
            "stdout.frame",
            save_dir_or_db=save_dir,
            raw=frame.decode("utf-8", errors="replace").rstrip("\n"),
        )
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
        log_acp_event(
            "rpc.server_request",
            save_dir_or_db=save_dir,
            id=rpc_id,
            method=method,
            params=params,
        )
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
        try:
            response = await dispatcher.handle_message(raw_line)
            if response is not None:
                await _write_frame(dispatcher.serialize_frame(response))
            for post_notif in dispatcher.pop_post_response_notifications():
                await _write_frame(dispatcher.serialize_frame(post_notif))
        except Exception as exc:
            log_acp_event(
                "serve.dispatch_error",
                save_dir_or_db=save_dir,
                raw=raw_line,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            raise

    async with anyio.create_task_group() as tg:
        while True:
            try:
                raw_line = await anyio.to_thread.run_sync(sys.stdin.readline)
            except io.UnsupportedOperation, OSError:
                log_acp_event("stdin.io_error", save_dir_or_db=save_dir)
                break
            if not raw_line:
                log_acp_event("stdin.eof", save_dir_or_db=save_dir)
                break
            line = raw_line.strip()
            log_acp_event("stdin.raw", save_dir_or_db=save_dir, raw=line)
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                log_acp_event(
                    "stdin.json_decode_error",
                    save_dir_or_db=save_dir,
                    raw=line,
                    error=str(exc),
                )
                continue
            if (
                isinstance(parsed, dict)
                and "method" not in parsed
                and ("result" in parsed or "error" in parsed)
            ):
                resp_id = parsed.get("id")
                result_obj = parsed.get("result")
                log_acp_event(
                    "stdin.client_rpc_response",
                    save_dir_or_db=save_dir,
                    id=resp_id,
                    payload=parsed,
                )
                pending_results[resp_id] = result_obj if isinstance(result_obj, dict) else {}
                pending_event = pending_events.get(resp_id)
                if pending_event is not None:
                    pending_event.set()
                continue
            tg.start_soon(_dispatch_line, line)

    log_acp_event("serve.stop", save_dir_or_db=save_dir)
    return 0
