"""Atomic 0600 JSON credential file storage provider."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CredentialStoreProvider:
    """Reads and atomically writes credentials on disk with generation tracking."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        """Read the current credential record from disk."""
        if not self.path.is_file():
            return {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def write(self, data: dict[str, Any]) -> dict[str, Any]:
        """Atomically write credential record with 0600 permissions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stored = dict(data)
        serialized = json.dumps(stored, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            os.chmod(tmp_path, 0o600)
            tmp_file.write(serialized)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, self.path)
        return stored
