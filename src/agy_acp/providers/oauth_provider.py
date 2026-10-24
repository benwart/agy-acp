"""Browser-based OAuth 2.0 + PKCE loopback provider for Google Antigravity."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
import anyio.abc

GOOGLE_OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_OAUTH_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)


@dataclass(frozen=True)
class OAuthTokenResult:
    """Normalized OAuth 2.0 token bundle and resolved identity."""

    access_token: str
    refresh_token: str | None
    id_token: str | None
    token_type: str
    expiry: str | None
    identity: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize token result for atomic credential storage."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "token_type": self.token_type,
            "expiry": self.expiry,
        }


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a cryptographic PKCE (code_verifier, S256 code_challenge) pair."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _extract_identity_from_id_token(id_token: str | None) -> str:
    """Decode JWT payload without signature check to read the user's email claim."""
    if not id_token:
        return "antigravity-oauth"
    parts = id_token.split(".")
    if len(parts) < 2:
        return "antigravity-oauth"
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        if isinstance(claims, dict):
            email = claims.get("email")
            if isinstance(email, str) and email.strip():
                return email.strip()
    except ValueError, KeyError, json.JSONDecodeError:
        pass
    return "antigravity-oauth"


def _default_token_exchanger(form_data: dict[str, str]) -> dict[str, Any]:
    """Exchange an authorization code for OAuth tokens over HTTPS."""
    encoded = urllib.parse.urlencode(form_data).encode("utf-8")
    req = urllib.request.Request(
        GOOGLE_OAUTH_TOKEN_URL,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Invalid OAuth token response payload")
    return parsed


def _parse_callback_request(raw_http: bytes) -> tuple[str, dict[str, list[str]]]:
    """Extract request path and query parameters from an HTTP/1.1 GET request."""
    text = raw_http.decode("utf-8", errors="replace")
    request_line = text.splitlines()[0] if text else ""
    parts = request_line.split(" ")
    target = parts[1] if len(parts) >= 2 else "/"
    parsed = urllib.parse.urlparse(target)
    return parsed.path, urllib.parse.parse_qs(parsed.query)


def _build_callback_http_response(*, success: bool, detail: str = "") -> bytes:
    """Construct the HTTP response sent to the user's browser on OAuth callback."""
    if success:
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Authentication Successful</title></head>"
            "<body><h2>Authentication successful! You can close this window.</h2></body></html>"
        )
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(html.encode('utf-8'))}\r\n"
            "Connection: close\r\n\r\n"
        )
        return (headers + html).encode("utf-8")

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Authentication Failed</title></head>"
        f"<body><h2>Authentication failed</h2><p>{detail}</p></body></html>"
    )
    headers = (
        "HTTP/1.1 400 Bad Request\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(html.encode('utf-8'))}\r\n"
        "Connection: close\r\n\r\n"
    )
    return (headers + html).encode("utf-8")


class OAuthBrowserFlowProvider:
    """Executes browser OAuth 2.0 + PKCE login with a local loopback callback listener."""

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        scopes: tuple[str, ...] = DEFAULT_OAUTH_SCOPES,
        browser_opener: Callable[[str], bool] | None = None,
        token_exchanger: Callable[[dict[str, str]], dict[str, Any]] | None = None,
    ) -> None:
        self._client_id = client_id or os.environ.get("AGY_ACP_OAUTH_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("AGY_ACP_OAUTH_CLIENT_SECRET", "")
        self._scopes = scopes
        self._browser_opener = browser_opener or webbrowser.open
        self._token_exchanger = token_exchanger or _default_token_exchanger

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        code_challenge: str,
    ) -> str:
        """Build the Google OAuth 2.0 authorization URL with PKCE parameters."""
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self._scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{GOOGLE_OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def authenticate(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: float = 180.0,
        on_auth_url: Callable[[str], None] | None = None,
    ) -> OAuthTokenResult:
        """Run loopback HTTP callback listener, open browser, and exchange code for tokens."""
        if os.environ.get("AGY_ACP_FAKE_OAUTH"):
            await anyio.sleep(0)
            fake_identity = os.environ.get("AGY_ACP_FAKE_OAUTH_EMAIL", "antigravity-oauth")
            return OAuthTokenResult(
                access_token="fake-oauth-access-token",
                refresh_token="fake-oauth-refresh-token",
                id_token=None,
                token_type="Bearer",
                expiry=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                identity=fake_identity,
            )

        if not self._client_id:
            raise ValueError(
                "OAuth client_id is required. Set AGY_ACP_OAUTH_CLIENT_ID or pass client_id."
            )

        code_verifier, code_challenge = generate_pkce_pair()
        expected_state = secrets.token_urlsafe(24)
        received_code: str | None = None
        callback_error: str | None = None
        code_ready = anyio.Event()

        listener = await anyio.create_tcp_listener(local_host="127.0.0.1", local_port=0)
        async with listener:
            port = listener.extra(anyio.abc.SocketAttribute.local_port)
            redirect_uri = f"http://localhost:{port}/auth/callback"
            auth_url = self.build_authorization_url(
                redirect_uri=redirect_uri,
                state=expected_state,
                code_challenge=code_challenge,
            )
            if on_auth_url is not None:
                on_auth_url(auth_url)
            if open_browser:
                self._browser_opener(auth_url)

            async def _handle_client(stream: anyio.abc.SocketStream) -> None:
                nonlocal received_code, callback_error
                async with stream:
                    raw = await stream.receive(8192)
                    path, qs = _parse_callback_request(raw)
                    if path != "/auth/callback":
                        await stream.send(
                            _build_callback_http_response(success=False, detail="Not found")
                        )
                        return

                    if "error" in qs:
                        callback_error = qs["error"][0]
                        await stream.send(
                            _build_callback_http_response(success=False, detail=callback_error)
                        )
                        code_ready.set()
                        return

                    state_vals = qs.get("state", [])
                    code_vals = qs.get("code", [])
                    if not state_vals or state_vals[0] != expected_state:
                        callback_error = "OAuth state mismatch"
                        await stream.send(
                            _build_callback_http_response(success=False, detail=callback_error)
                        )
                        code_ready.set()
                        return

                    if not code_vals or not code_vals[0]:
                        callback_error = "Missing authorization code in callback"
                        await stream.send(
                            _build_callback_http_response(success=False, detail=callback_error)
                        )
                        code_ready.set()
                        return

                    received_code = code_vals[0]
                    await stream.send(_build_callback_http_response(success=True))
                    code_ready.set()

            with anyio.fail_after(timeout_seconds):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(listener.serve, _handle_client)
                    await code_ready.wait()
                    tg.cancel_scope.cancel()

        if callback_error is not None or not received_code:
            raise RuntimeError(callback_error or "OAuth login cancelled or timed out")

        form_data = {
            "code": received_code,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        }
        token_payload = await anyio.to_thread.run_sync(self._token_exchanger, form_data)
        access_token = str(token_payload.get("access_token", ""))
        if not access_token:
            raise RuntimeError("OAuth token response missing access_token")

        refresh_token_raw = token_payload.get("refresh_token")
        refresh_token = str(refresh_token_raw) if refresh_token_raw else None
        id_token_raw = token_payload.get("id_token")
        id_token = str(id_token_raw) if id_token_raw else None
        token_type = str(token_payload.get("token_type", "Bearer"))
        expires_in = int(token_payload.get("expires_in", 3600))
        expiry = (datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat()
        identity = _extract_identity_from_id_token(id_token)

        return OAuthTokenResult(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            token_type=token_type,
            expiry=expiry,
            identity=identity,
        )
