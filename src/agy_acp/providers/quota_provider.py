"""Provider for querying Antigravity 5-hour / 7-day rolling quotas and G1 credits via `agy` CLI."""

from __future__ import annotations

from typing import Any

from agy_acp.providers.agy_cli_provider import AgyCliProvider


class QuotaProvider:
    """Fetches 5-hour (`5h`) and 7-day (`weekly`) quota buckets and G1 credits from `agy` CLI."""

    def __init__(self, *, cli_provider: AgyCliProvider | None = None) -> None:
        self._cli = cli_provider or AgyCliProvider()

    async def fetch_quota_groups(self) -> list[dict[str, Any]]:
        """Query `agy -p /usage --output-format json` and return normalized quota groups."""
        payload = await self._cli.run_slash_command("/usage")
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
        payload = await self._cli.run_slash_command("/credits")
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
