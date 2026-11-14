"""Provider for querying Antigravity 5-hour / 7-day rolling quotas and G1 credits via `agy` CLI."""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

from agy_acp.logic.tool_classifier import build_scrubbed_subprocess_env, resolve_agy_binary
from agy_acp.providers.telemetry import get_telemetry


async def run_agy_slash_json(
    command: str,
    *,
    agy_bin: str | Path | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run a non-interactive slash command (`agy -p <cmd> --output-format json`) via anyio."""
    normalized = command.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    bin_path = resolve_agy_binary(agy_bin)
    args: list[str] = [bin_path]
    if cwd:
        args.extend(["--add-dir", cwd])
    args.extend(["--output-format", "json", "-p", normalized])

    with anyio.fail_after(timeout):
        result = await anyio.run_process(args, check=False, env=build_scrubbed_subprocess_env())

    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()

    if stdout:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                return parsed
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                parsed_line = json.loads(line)
                if isinstance(parsed_line, dict):
                    if "result" in parsed_line and isinstance(parsed_line["result"], dict):
                        return parsed_line["result"]
                    return parsed_line

    if result.returncode != 0:
        raise RuntimeError(
            f"agy command '{normalized}' exited with code {result.returncode}: {stderr or stdout}"
        )
    return {
        "status": "SUCCESS",
        "response": stdout,
        "command": {"name": normalized.lstrip("/")},
    }


def _supports_tier_capability(init_res: Mapping[str, Any]) -> bool:
    """Return whether `initialize` advertised `_meta.antigravity.tier`."""
    caps = init_res.get("agentCapabilities")
    caps_meta = caps.get("_meta") if isinstance(caps, Mapping) else None
    caps_agy = caps_meta.get("antigravity") if isinstance(caps_meta, Mapping) else None
    if isinstance(caps_agy, Mapping) and bool(caps_agy.get("tier")):
        return True
    top_meta = init_res.get("_meta")
    top_agy = top_meta.get("antigravity") if isinstance(top_meta, Mapping) else None
    return isinstance(top_agy, Mapping) and bool(top_agy.get("tier"))


async def run_agy_acp_tier_json(
    *,
    agy_bin: str | Path | None = None,
    timeout: float = 15.0,
) -> dict[str, str] | None:
    """Probe `agy acp --stdio` via `initialize` + `authenticate` for `_meta.antigravity.tier`."""
    bin_path = resolve_agy_binary(agy_bin)
    with anyio.fail_after(timeout):
        proc = await anyio.open_process(
            [bin_path, "acp", "--stdio"],
            env=build_scrubbed_subprocess_env(),
            stderr=subprocess.DEVNULL,
        )
        async with proc:
            try:
                if proc.stdin is None or proc.stdout is None:
                    return None
                reader = BufferedByteReceiveStream(proc.stdout)

                async def _rpc(req_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
                    assert proc.stdin is not None
                    payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
                    frame = (json.dumps(payload) + "\n").encode("utf-8")
                    await proc.stdin.send(frame)
                    while True:
                        raw_line = await reader.receive_until(b"\n", 1024 * 1024)
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        with contextlib.suppress(json.JSONDecodeError):
                            msg = json.loads(line)
                            if isinstance(msg, dict) and msg.get("id") == req_id:
                                res = msg.get("result")
                                return dict(res) if isinstance(res, Mapping) else {}

                init_res = await _rpc(
                    1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}}
                )
                if not _supports_tier_capability(init_res):
                    return None
                auth_res = await _rpc(2, "authenticate", {"methodId": "oauth-personal"})
                meta = auth_res.get("_meta")
                agy_meta = meta.get("antigravity") if isinstance(meta, Mapping) else None
                tier = agy_meta.get("tier") if isinstance(agy_meta, Mapping) else None
                if not isinstance(tier, Mapping):
                    return None
                tier_id = str(tier.get("id") or "").strip()
                if not tier_id:
                    return None
                return {
                    "id": tier_id,
                    "name": str(tier.get("name") or "").strip(),
                    "upgradeUri": str(tier.get("upgradeUri") or "").strip(),
                }
            finally:
                if proc.stdin is not None:
                    with contextlib.suppress(Exception):
                        await proc.stdin.aclose()
                if proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.terminate()


class QuotaProvider:
    """Fetches 5-hour (`5h`) and 7-day (`weekly`) quota buckets and G1 credits from `agy` CLI."""

    def __init__(
        self,
        *,
        cli_provider: Any | None = None,
        agy_bin: str | Path | None = None,
    ) -> None:
        self._cli = cli_provider
        self._agy_bin = agy_bin

    async def _run_slash(self, command: str) -> dict[str, Any]:
        if self._cli is not None:
            return await self._cli.run_slash_command(command)
        return await run_agy_slash_json(command, agy_bin=self._agy_bin)

    async def fetch_quota_groups(self) -> list[dict[str, Any]]:
        """Query `agy -p /usage --output-format json` and return normalized quota groups."""
        telemetry = get_telemetry()
        start_t = time.perf_counter()
        status = "ok"
        try:
            with telemetry.start_span("quota.fetch.usage"):
                payload = await self._run_slash("/usage")
        except Exception:
            status = "error"
            raise
        finally:
            telemetry.record_quota_fetch(
                "usage",
                status=status,
                duration_s=time.perf_counter() - start_t,
            )
        cmd = payload.get("command")
        cmd_data = cmd.get("data") if isinstance(cmd, dict) else None
        raw_groups = cmd_data.get("groups") if isinstance(cmd_data, dict) else None

        groups: list[dict[str, Any]] = []
        if isinstance(raw_groups, list):
            for entry in raw_groups:
                if not isinstance(entry, dict):
                    continue
                group_name = str(entry.get("name") or entry.get("group") or "")
                raw_buckets = entry.get("buckets")
                buckets: list[dict[str, Any]] = []
                if isinstance(raw_buckets, list):
                    for bucket in raw_buckets:
                        if not isinstance(bucket, dict):
                            continue
                        buckets.append(
                            {
                                "id": str(bucket.get("id", "")),
                                "group": group_name,
                                "name": str(bucket.get("name", "")),
                                "window": str(bucket.get("window", "")),
                                "remaining_fraction": float(bucket.get("remaining_fraction", 0.0)),
                                "reset_time": str(bucket.get("reset_time", "")),
                            }
                        )
                groups.append({"group": group_name, "buckets": buckets})
        return groups

    async def fetch_credits(self) -> dict[str, Any]:
        """Query `agy -p /credits --output-format json` and return G1 remaining credits."""
        telemetry = get_telemetry()
        start_t = time.perf_counter()
        status = "ok"
        try:
            with telemetry.start_span("quota.fetch.credits"):
                payload = await self._run_slash("/credits")
        except Exception:
            status = "error"
            raise
        finally:
            telemetry.record_quota_fetch(
                "credits",
                status=status,
                duration_s=time.perf_counter() - start_t,
            )
        cmd = payload.get("command")
        cmd_data = cmd.get("data") if isinstance(cmd, dict) else None
        if isinstance(cmd_data, dict) and "remaining_credits" in cmd_data:
            return {
                "remaining_credits": int(cmd_data.get("remaining_credits", 0)),
                "upgrade_uri": cmd_data.get("upgrade_uri"),
            }

        remaining_credits = 0
        upgrade_uri: str | None = None
        response_text = str(payload.get("response", ""))
        for line in response_text.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                key = parts[0].strip().lower()
                val = parts[1].strip()
                if "credit" in key:
                    try:
                        remaining_credits = int(val)
                    except ValueError:
                        remaining_credits = 0
                elif "upgrade" in key:
                    upgrade_uri = val
        return {
            "remaining_credits": remaining_credits,
            "upgrade_uri": upgrade_uri,
        }

    async def fetch_tier(self) -> dict[str, str] | None:
        """Query the active Antigravity Google AI plan tier via `cli_provider` or `agy acp`."""
        if self._cli is not None:
            discover_tier = getattr(self._cli, "discover_tier", None)
            if callable(discover_tier):
                res = await discover_tier()
                if isinstance(res, Mapping) and str(res.get("id") or "").strip():
                    return {
                        "id": str(res["id"]).strip(),
                        "name": str(res.get("name") or "").strip(),
                        "upgradeUri": str(res.get("upgradeUri") or "").strip(),
                    }
            return None
        if type(self) is QuotaProvider:
            return await run_agy_acp_tier_json(agy_bin=self._agy_bin)
        return None
