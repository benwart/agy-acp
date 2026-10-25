"""Zero-downtime authentication state and credential hot-reload manager."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agy_acp.providers.credential_store import CredentialStoreProvider
from agy_acp.providers.oauth_provider import OAuthBrowserFlowProvider

_KIRO_IDENTITY_CLAIM_FIELDS: frozenset[str] = frozenset(
    {"client_id", "oauth_flow", "region", "scopes", "start_url"}
)
_KIRO_IDENTITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS migrations (
    id INTEGER PRIMARY KEY,
    version INTEGER NOT NULL,
    migration_time INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value BLOB
);
"""


def resolve_kiro_identity_db_path() -> Path:
    """Return the platform-specific Kiro CLI SQLite identity store path."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "kiro-cli" / "data.sqlite3"
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        base = Path(local_appdata) if local_appdata else (home / "AppData" / "Local")
        return base / "kiro-cli" / "data.sqlite3"
    return home / ".local" / "share" / "kiro-cli" / "data.sqlite3"


def ensure_kiro_identity_store(db_path: Path | None = None) -> bool:
    """Seed stable Kiro identity claims into `data.sqlite3` if `auth_kv` has no claim rows."""
    target = Path(db_path) if db_path is not None else resolve_kiro_identity_db_path()
    try:
        if target.is_symlink():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd = os.open(target, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(fd)
        with contextlib.suppress(OSError):
            target.chmod(0o600)

        with contextlib.closing(sqlite3.connect(str(target), timeout=5.0)) as conn:
            conn.executescript(_KIRO_IDENTITY_SCHEMA_SQL)
            rows = conn.execute("SELECT key, value FROM auth_kv").fetchall()
            for _, raw_val in rows:
                with contextlib.suppress(ValueError, TypeError):
                    parsed = json.loads(raw_val if isinstance(raw_val, str) else str(raw_val))
                    if isinstance(parsed, dict) and any(
                        field in parsed for field in _KIRO_IDENTITY_CLAIM_FIELDS
                    ):
                        return False

            identity_blob = json.dumps(
                {
                    "client_id": "agy-acp",
                    "oauth_flow": "PKCE",
                    "region": "us-east-1",
                    "start_url": "https://one.google.com/ai",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO migrations (id, version, migration_time) "
                    "VALUES (1, 1, 1700000000)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO auth_kv (key, value) VALUES (?, ?)",
                    ("agy-acp:identity", identity_blob),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
                    ("auth.idc.start-url", '"https://one.google.com/ai"'),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
                    ("auth.idc.region", '"us-east-1"'),
                )
            with contextlib.suppress(sqlite3.Error):
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        return True
    except OSError, sqlite3.Error:
        return False


@dataclass(frozen=True)
class AuthState:
    """Snapshot of current authentication state and fingerprint."""

    authenticated: bool
    method: str | None
    identity: str | None
    project: str | None
    location: str | None
    generation: int
    fingerprint: str
    auth_file: Path

    def to_dict(self) -> dict[str, Any]:
        """Serialize authentication state to a JSON-compatible dictionary."""
        return {
            "authenticated": self.authenticated,
            "method": self.method,
            "identity": self.identity,
            "project": self.project,
            "location": self.location,
            "generation": self.generation,
            "fingerprint": self.fingerprint,
            "auth_file": str(self.auth_file),
        }


def _default_auth_file() -> Path:
    env_path = os.environ.get("AGY_ACP_AUTH_FILE")
    if env_path:
        return Path(env_path)
    return Path.home() / ".config" / "agy-acp" / "credentials.json"


def _compute_fingerprint(record: dict[str, Any] | str) -> str:
    if isinstance(record, str):
        digest = hashlib.sha256(record.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
    canonical = {k: v for k, v in record.items() if k != "fingerprint"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


class AuthManager:
    """Manages CLI login/logout/status and detects external credential updates for ACP sessions."""

    def __init__(
        self,
        *,
        auth_file: Path | None = None,
        oauth_provider: OAuthBrowserFlowProvider | None = None,
    ) -> None:
        self._explicit_auth_file = auth_file is not None
        self._auth_file = Path(auth_file) if auth_file is not None else _default_auth_file()
        self._store = CredentialStoreProvider(self._auth_file)
        self._oauth_provider = oauth_provider or OAuthBrowserFlowProvider()

    def login(
        self,
        *,
        api_key: str | None = None,
        vertex: bool = False,
        project: str | None = None,
        location: str | None = None,
        oauth: bool = False,
        oauth_token: dict[str, Any] | None = None,
        identity: str | None = None,
    ) -> AuthState:
        """Persist credentials atomically and increment generation counter."""
        prev = self._store.read()
        generation = int(prev.get("generation", 0)) + 1
        if vertex:
            method = "vertex"
            resolved_identity = identity or "vertex-adc"
        elif oauth:
            method = "oauth"
            resolved_identity = identity or "antigravity-oauth"
        else:
            method = "api_key"
            resolved_identity = identity or "gemini-api-key"

        record: dict[str, Any] = {
            "authenticated": True,
            "method": method,
            "identity": resolved_identity,
            "api_key": api_key,
            "project": project,
            "location": location,
            "oauth": oauth,
            "oauth_token": oauth_token,
            "generation": generation,
        }
        fingerprint = _compute_fingerprint(record)
        record["fingerprint"] = fingerprint
        self._store.write(record)

        return AuthState(
            authenticated=True,
            method=method,
            identity=resolved_identity,
            project=project,
            location=location,
            generation=generation,
            fingerprint=fingerprint,
            auth_file=self._auth_file,
        )

    async def login_oauth_browser(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: float = 180.0,
        on_auth_url: Callable[[str], None] | None = None,
    ) -> AuthState:
        """Authenticate interactively via browser OAuth 2.0 + PKCE and persist credentials."""
        token_result = await self._oauth_provider.authenticate(
            open_browser=open_browser,
            timeout_seconds=timeout_seconds,
            on_auth_url=on_auth_url,
        )
        return self.login(
            oauth=True,
            oauth_token=token_result.to_dict(),
            identity=token_result.identity,
        )

    def logout(self) -> AuthState:
        """Clear stored credentials atomically without killing running ACP servers."""
        existing = self._store.read()
        generation = int(existing.get("generation", 0)) + 1
        record: dict[str, Any] = {
            "authenticated": False,
            "method": None,
            "identity": None,
            "api_key": None,
            "vertex": False,
            "project": None,
            "location": None,
            "oauth": False,
            "generation": generation,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        fingerprint = _compute_fingerprint(record)
        record["fingerprint"] = fingerprint
        self._store.write(record)

        return AuthState(
            authenticated=False,
            method=None,
            identity=None,
            project=None,
            location=None,
            generation=generation,
            fingerprint=fingerprint,
            auth_file=self._auth_file,
        )

    def status(self) -> AuthState:
        """Return current authentication state and credential fingerprint."""
        record = self._store.read()
        if "authenticated" in record:
            return AuthState(
                authenticated=bool(record["authenticated"]),
                method=record.get("method"),
                identity=record.get("identity"),
                project=record.get("project"),
                location=record.get("location"),
                generation=int(record.get("generation", 0)),
                fingerprint=str(record.get("fingerprint", "")),
                auth_file=self._auth_file,
            )

        env_key = os.environ.get("GEMINI_API_KEY")
        if not self._explicit_auth_file and "AGY_ACP_AUTH_FILE" not in os.environ and env_key:
            return AuthState(
                authenticated=True,
                method="api_key",
                identity="gemini-api-key",
                project=None,
                location=None,
                generation=1,
                fingerprint=_compute_fingerprint(f"env:{env_key}"),
                auth_file=self._auth_file,
            )

        return AuthState(
            authenticated=False,
            method=None,
            identity=None,
            project=None,
            location=None,
            generation=0,
            fingerprint="",
            auth_file=self._auth_file,
        )

    def has_changed_since(self, previous_fingerprint: str | None) -> bool:
        """Return True if the on-disk credential store changed since previous_fingerprint."""
        current_fp = self.status().fingerprint
        return current_fp != (previous_fingerprint or "")
