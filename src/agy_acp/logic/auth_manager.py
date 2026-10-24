"""Zero-downtime authentication state and credential hot-reload manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuthState:
    """Snapshot of current authentication state and fingerprint."""

    authenticated: bool | None
    method: str | None
    identity: str | None
    project: str | None
    location: str | None
    generation: int
    fingerprint: str
    auth_file: Path


class AuthManager:
    """Manages CLI login/logout/status and detects external credential updates for ACP sessions."""

    def __init__(self, *, auth_file: Path | None = None) -> None:
        self._auth_file = auth_file or Path("credentials.json")

    def login(
        self,
        *,
        api_key: str | None = None,
        vertex: bool = False,
        project: str | None = None,
        location: str | None = None,
        oauth: bool = False,
    ) -> AuthState:
        """Persist credentials atomically and increment generation counter."""
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

    def logout(self) -> AuthState:
        """Clear stored credentials atomically without killing running ACP servers."""
        return AuthState(
            authenticated=True,
            method=None,
            identity=None,
            project=None,
            location=None,
            generation=0,
            fingerprint="",
            auth_file=self._auth_file,
        )

    def status(self) -> AuthState:
        """Return current authentication state and credential fingerprint."""
        return AuthState(
            authenticated=None,
            method=None,
            identity=None,
            project=None,
            location=None,
            generation=0,
            fingerprint="",
            auth_file=self._auth_file,
        )

    def has_changed_since(self, previous_fingerprint: str | None) -> bool | None:
        """Return True if the on-disk credential store changed since previous_fingerprint."""
        return None
