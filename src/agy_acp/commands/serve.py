"""Top-level ACP server command."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from cyclopts import Parameter

from agy_acp.logic.acp_dispatcher import AcpDispatcher
from agy_acp.logic.auth_manager import AuthManager, ensure_kiro_identity_store
from agy_acp.logic.quota_tracker import QuotaTracker
from agy_acp.logic.session_manager import SessionManager
from agy_acp.logic.wire import EngineWireAdapter, create_wire_adapter
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


def _is_valid_rpc_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool)


async def run_stdio_server(
    *,
    session_manager: SessionManager,
    wire_adapter: EngineWireAdapter,
    auth_manager: AuthManager | None = None,
    quota_tracker: QuotaTracker | None = None,
    save_dir: Path | None = None,
    stdin_readline: Callable[[], Awaitable[str]] | None = None,
    stdout_write: Callable[[bytes], Awaitable[None]] | None = None,
) -> int:
    """Run the JSON-RPC 2.0 ACP event loop over `stdin_readline` and `stdout_write`."""
    write_lock = anyio.Lock()
    pending_events: dict[str | int, anyio.Event] = {}
    pending_results: dict[str | int, dict[str, Any]] = {}
    next_rpc_seq = 0
    connection_closed = False
    dispatcher: AcpDispatcher

    async def _default_readline() -> str:
        return await anyio.to_thread.run_sync(sys.stdin.readline)

    async def _default_write_bytes(frame: bytes) -> None:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(frame)
            buffer.flush()
        else:
            sys.stdout.write(frame.decode("utf-8"))
            sys.stdout.flush()

    readline_fn = stdin_readline or _default_readline

    async def _write_payloads(payloads: Sequence[dict[str, Any]]) -> None:
        if not payloads:
            return
        frames = [dispatcher.serialize_frame(payload) for payload in payloads]
        async with write_lock:
            if stdout_write is not None:
                for frame in frames:
                    await stdout_write(frame)
            else:
                await _default_write_bytes(b"".join(frames))
        for payload in payloads:
            log_acp_event(
                "stdout.frame",
                save_dir_or_db=save_dir,
                payload=payload,
            )

    async def _write_payload(payload: dict[str, Any]) -> None:
        await _write_payloads([payload])

    async def _notification_sink(notification: dict[str, Any]) -> None:
        await _write_payload(notification)

    async def _client_rpc_caller(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_rpc_seq
        if connection_closed:
            return {}
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
        await _write_payload(request_payload)
        if connection_closed:
            pending_events.pop(rpc_id, None)
            pending_results.pop(rpc_id, None)
            return {}
        await event.wait()
        pending_events.pop(rpc_id, None)
        return pending_results.pop(rpc_id, {})

    dispatcher = AcpDispatcher(
        notification_sink=_notification_sink,
        client_rpc_caller=_client_rpc_caller,
        session_manager=session_manager,
        auth_manager=auth_manager,
        quota_tracker=quota_tracker,
        wire_adapter=wire_adapter,
    )

    def _wake_all_pending_rpcs() -> None:
        nonlocal connection_closed
        connection_closed = True
        for pending_ev in list(pending_events.values()):
            pending_ev.set()

    async def _dispatch_line(raw_line: str, msg_meta: tuple[bool, Any, Any]) -> None:
        has_id, msg_id, method_name = msg_meta
        try:
            response = await dispatcher.handle_message(raw_line)
            outgoing: list[dict[str, Any]] = []
            if response is not None:
                outgoing.append(response)
            outgoing.extend(dispatcher.pop_post_response_notifications())
            await _write_payloads(outgoing)
        except KeyError as exc:
            log_acp_event(
                "serve.dispatch_error",
                save_dir_or_db=save_dir,
                id=msg_id,
                method=method_name,
                error_type="KeyError",
                error=str(exc),
            )
            if has_id:
                await _write_payload(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id if (msg_id is None or _is_valid_rpc_id(msg_id)) else None,
                        "error": {"code": -32002, "message": str(exc)},
                    }
                )
        except Exception as exc:
            log_acp_event(
                "serve.dispatch_error",
                save_dir_or_db=save_dir,
                id=msg_id,
                method=method_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if has_id:
                await _write_payload(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id if (msg_id is None or _is_valid_rpc_id(msg_id)) else None,
                        "error": {"code": -32603, "message": "Internal error"},
                    }
                )

    async with anyio.create_task_group() as tg:
        try:
            while True:
                try:
                    raw_line = await readline_fn()
                except io.UnsupportedOperation, OSError:
                    log_acp_event("stdin.io_error", save_dir_or_db=save_dir)
                    break
                if not raw_line:
                    log_acp_event("stdin.eof", save_dir_or_db=save_dir)
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    log_acp_event(
                        "stdin.json_decode_error",
                        save_dir_or_db=save_dir,
                        line_length=len(line),
                        error=str(exc),
                    )
                    continue
                if not isinstance(parsed, dict):
                    log_acp_event(
                        "stdin.invalid_request",
                        save_dir_or_db=save_dir,
                        payload_type=type(parsed).__name__,
                    )
                    await _write_payload(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Invalid Request"},
                        }
                    )
                    continue

                log_acp_event("stdin.frame", save_dir_or_db=save_dir, payload=parsed)
                if "method" not in parsed and ("result" in parsed or "error" in parsed):
                    resp_id = parsed.get("id")
                    if not _is_valid_rpc_id(resp_id) or resp_id not in pending_events:
                        log_acp_event(
                            "stdin.ignored_client_rpc_response",
                            save_dir_or_db=save_dir,
                            has_valid_id=_is_valid_rpc_id(resp_id),
                        )
                        continue
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

                msg_meta = ("id" in parsed, parsed.get("id"), parsed.get("method"))
                tg.start_soon(_dispatch_line, line, msg_meta)
        finally:
            _wake_all_pending_rpcs()

    log_acp_event("serve.stop", save_dir_or_db=save_dir)
    return 0


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

    wire_adapter = create_wire_adapter(normalized_engine)
    session_manager = SessionManager(
        save_dir=save_dir,
        default_agent=agent,
        default_model=model,
        wire_adapter=wire_adapter,
    )
    auth_manager = AuthManager(auth_file=auth_file) if auth_file is not None else None
    return await run_stdio_server(
        session_manager=session_manager,
        wire_adapter=wire_adapter,
        auth_manager=auth_manager,
        save_dir=save_dir,
    )
