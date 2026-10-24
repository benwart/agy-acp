"""Cyclopts CLI entrypoint for the agy-acp agent harness."""

from __future__ import annotations

from collections.abc import Sequence

from cyclopts import App

from agy_acp import __version__
from agy_acp.commands import register_commands

__all__ = ["app", "main"]

app = App(
    name="agy-acp",
    help="Google Antigravity SDK wrapper exposing an ACP server for KiroCrew.",
    version=__version__,
)
register_commands(app)


def main(tokens: Sequence[str] | None = None) -> int:
    """Run the agy-acp CLI application."""
    if tokens is not None:
        result = app(list(tokens), result_action="return_int_as_exit_code_else_zero")
        return int(result)
    result = app(result_action="return_int_as_exit_code_else_zero")
    return int(result)
