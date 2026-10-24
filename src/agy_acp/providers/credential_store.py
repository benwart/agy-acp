"""Atomic 0600 JSON credential file storage provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CredentialStoreProvider:
    """Reads and atomically writes credentials on disk with generation tracking."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        """Read the current credential record from disk."""
        return {}

    def write(self, data: dict[str, Any]) -> dict[str, Any]:
        """Atomically write credential record with 0600 permissions."""
        return {}
