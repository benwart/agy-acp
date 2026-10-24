"""Black-box integration test fixtures for driving the agy-acp CLI and ACP stdio server."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from agy_acp.cli import app


@dataclass
class CliResult:
    """Captured output and exit code from a black-box CLI invocation."""

    exit_code: int
    stdout: str
    stderr: str

    def json(self) -> dict[str, Any]:
        """Parse stdout as a JSON object."""
        assert self.stdout.strip(), "Expected JSON on stdout, got empty output"
        loaded = json.loads(self.stdout)
        assert isinstance(loaded, dict)
        return loaded


class AcpBlackBoxHarness:
    """Black-box ACP client simulator matching KiroCrew's AcpClient / AcpRuntime stdio framing."""

    def __init__(self, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        self._args = args or ["acp", "serve"]
        self._env = env or {}
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self.notifications: list[dict[str, Any]] = []
        self.server_requests: list[dict[str, Any]] = []
        self._permission_responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    async def start(self) -> None:
        """Spawn the agy-acp subprocess with stdio pipes."""
        import os

        merged_env = {**os.environ, **self._env}
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "agy_acp",
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )

    def on_permission_request(
        self,
        responder: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register a callback to respond to server-initiated session/request_permission RPCs."""
        self._permission_responder = responder

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request over stdin and read until matching response arrives."""
        assert self._proc is not None and self._proc.stdin is not None
        assert self._proc.stdout is not None

        req_id = self._next_id
        self._next_id += 1
        frame = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        self._proc.stdin.write((frame + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

        async def _read_until_response() -> dict[str, Any]:
            assert self._proc is not None and self._proc.stdout is not None
            assert self._proc.stdin is not None
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    stderr_bytes = (
                        await self._proc.stderr.read() if self._proc.stderr is not None else b""
                    )
                    msg = (
                        f"ACP server closed stdout before responding to {method} (id={req_id}). "
                        f"stderr: {stderr_bytes.decode('utf-8', errors='replace')}"
                    )
                    raise RuntimeError(msg)
                assert len(raw_line) <= 16 * 1024 * 1024, "Frame exceeded KiroCrew 16 MiB limit"
                message = json.loads(raw_line.decode("utf-8"))
                if "method" in message and "id" not in message:
                    self.notifications.append(message)
                elif "method" in message and "id" in message:
                    self.server_requests.append(message)
                    if (
                        message["method"] == "session/request_permission"
                        and self._permission_responder is not None
                    ):
                        reply_result = self._permission_responder(message["params"])
                        reply = json.dumps(
                            {"jsonrpc": "2.0", "id": message["id"], "result": reply_result}
                        )
                        self._proc.stdin.write((reply + "\n").encode("utf-8"))
                        await self._proc.stdin.drain()
                elif message.get("id") == req_id:
                    return message

        return await asyncio.wait_for(_read_until_response(), timeout=timeout)

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC 2.0 notification (no id field) over stdin."""
        assert self._proc is not None and self._proc.stdin is not None
        frame = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self._proc.stdin.write((frame + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def close(self) -> None:
        """Terminate subprocess if still running."""
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()


@dataclass
class HarnessFactory:
    """Factory that tracks spawned black-box ACP harnesses and cleans them up after each test."""

    _instances: list[AcpBlackBoxHarness] = field(default_factory=list)

    async def spawn(
        self,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> AcpBlackBoxHarness:
        """Start and return a black-box ACP harness process."""
        harness = AcpBlackBoxHarness(args=args, env=env)
        await harness.start()
        self._instances.append(harness)
        return harness

    async def cleanup(self) -> None:
        """Close all spawned harness instances."""
        for harness in self._instances:
            await harness.close()


@pytest.fixture
async def acp_harness_factory() -> Any:
    """Provide a black-box ACP harness factory and clean up processes on teardown."""
    factory = HarnessFactory()
    yield factory
    await factory.cleanup()


@pytest.fixture
def run_cli(capsys: pytest.CaptureFixture[str]) -> Callable[[list[str]], CliResult]:
    """Invoke the public agy-acp CLI with arguments and capture exit code + output."""

    def _invoke(tokens: list[str]) -> CliResult:
        exit_code = int(app(tokens, result_action="return_int_as_exit_code_else_zero"))
        captured = capsys.readouterr()
        return CliResult(exit_code=exit_code, stdout=captured.out, stderr=captured.err)

    return _invoke
