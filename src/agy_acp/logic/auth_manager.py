"""Zero-downtime authentication state and credential hot-reload manager."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agy_acp.providers.credential_store import CredentialStoreProvider
from agy_acp.providers.oauth_provider import OAuthBrowserFlowProvider


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
