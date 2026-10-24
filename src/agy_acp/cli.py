"""Cyclopts CLI entrypoint for the agy-acp agent harness."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from agy_acp import __version__
from agy_acp.commands import AnyioApp, register_commands
from agy_acp.commands.serve import default as serve_default
from agy_acp.providers.acp_logger import log_acp_event, snapshot_relevant_env

__all__ = ["app", "main"]

app = AnyioApp(
    name="agy-acp",
    help="Google Antigravity SDK wrapper exposing an ACP server for KiroCrew.",
    version=__version__,
)
app.default(serve_default)
register_commands(app)


def main(tokens: Sequence[str] | None = None) -> int:
    """Run the agy-acp CLI application."""
    argv = list(tokens) if tokens is not None else sys.argv[1:]
    log_acp_event(
        "cli.invoke",
        argv=argv,
        sys_argv=list(sys.argv),
        cwd=str(Path.cwd()),
        env=snapshot_relevant_env(),
    )
    try:
        if tokens is not None:
            result = app(list(tokens), result_action="return_int_as_exit_code_else_zero")
            return int(result)
        result = app(result_action="return_int_as_exit_code_else_zero")
        return int(result)
    except BaseException as exc:
        log_acp_event(
            "cli.exception",
            argv=argv,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
