"""Provider for querying Antigravity 5-hour / 7-day rolling quotas and G1 credits via `agy` CLI."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import anyio

from agy_acp.logic.tool_classifier import build_scrubbed_subprocess_env, resolve_agy_binary


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
        payload = await self._run_slash("/usage")
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
        payload = await self._run_slash("/credits")
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
