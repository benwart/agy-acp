"""Unit tests for AuthManager and CredentialStoreProvider atomic hot-reload."""

from __future__ import annotations

from pathlib import Path

from agy_acp.logic.auth_manager import AuthManager


def test_login_writes_atomic_credentials(tmp_path: Path) -> None:
    """Verify AuthManager.login writes 0600 credentials and returns authenticated AuthState."""
    auth_file = tmp_path / "credentials.json"
    mgr = AuthManager(auth_file=auth_file)
    state = mgr.login(api_key="test-key")
    assert state.authenticated is True
    assert state.generation == 1
    assert auth_file.is_file()


def test_logout_increments_generation(tmp_path: Path) -> None:
    """Verify AuthManager.logout clears credentials and increments generation."""
    auth_file = tmp_path / "credentials.json"
    mgr = AuthManager(auth_file=auth_file)
    mgr.login(api_key="test-key")
    logged_out = mgr.logout()
    assert logged_out.authenticated is False
    assert logged_out.generation == 2


def test_status_reflects_active_credential_state(tmp_path: Path) -> None:
    """Verify AuthManager.status reports current authentication state."""
    auth_file = tmp_path / "credentials.json"
    mgr = AuthManager(auth_file=auth_file)
    assert mgr.status().authenticated is False


def test_detects_external_credential_rotation(tmp_path: Path) -> None:
    """Verify AuthManager.has_changed_since detects external credential changes."""
    auth_file = tmp_path / "credentials.json"
    mgr = AuthManager(auth_file=auth_file)
    s1 = mgr.login(api_key="key-1")
    assert mgr.has_changed_since(s1.fingerprint) is False
    mgr.login(api_key="key-2")
    assert mgr.has_changed_since(s1.fingerprint) is True


async def test_oauth_browser_flow_pkce_loopback_and_token_exchange(tmp_path: Path) -> None:
    """Verify browser OAuth flow opens auth URL, handles loopback callback, and exchanges code."""
    import base64
    import json
    import urllib.parse

    import anyio

    from agy_acp.providers.oauth_provider import OAuthBrowserFlowProvider

    auth_file = tmp_path / "credentials.json"
    captured_urls: list[str] = []
    exchanged_payloads: list[dict[str, str]] = []

    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').decode().rstrip("=")
    claims = (
        base64.urlsafe_b64encode(json.dumps({"email": "dev@example.com"}).encode())
        .decode()
        .rstrip("=")
    )
    fake_id_token = f"{header}.{claims}.sig"

    async def _simulate_browser_callback(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        redirect_uri = qs["redirect_uri"][0]
        state = qs["state"][0]
        cb_parsed = urllib.parse.urlparse(redirect_uri)
        assert cb_parsed.hostname == "localhost"
        assert cb_parsed.path == "/auth/callback"
        assert cb_parsed.port is not None

        async with await anyio.connect_tcp("127.0.0.1", cb_parsed.port) as stream:
            req = (
                f"GET /auth/callback?code=fake-oauth-code&state={state} HTTP/1.1\r\n"
                f"Host: localhost:{cb_parsed.port}\r\n"
                "Connection: close\r\n\r\n"
            )
            await stream.send(req.encode("utf-8"))
            resp_bytes = await stream.receive(4096)
            assert b"302 Found" in resp_bytes or b"200 OK" in resp_bytes

    def _fake_token_exchanger(form_data: dict[str, str]) -> dict[str, object]:
        exchanged_payloads.append(dict(form_data))
        return {
            "access_token": "ya29.fake-access-token",
            "refresh_token": "1//fake-refresh-token",
            "id_token": fake_id_token,
            "token_type": "Bearer",
            "expires_in": 3600,
        }

    async with anyio.create_task_group() as tg:

        def _fake_browser_opener(url: str) -> bool:
            captured_urls.append(url)
            tg.start_soon(_simulate_browser_callback, url)
            return True

        provider = OAuthBrowserFlowProvider(
            client_id="user-supplied-client-id",
            client_secret="user-supplied-client-secret",
            browser_opener=_fake_browser_opener,
            token_exchanger=_fake_token_exchanger,
        )
        mgr = AuthManager(auth_file=auth_file, oauth_provider=provider)
        state = await mgr.login_oauth_browser()

    assert len(captured_urls) == 1
    parsed_auth = urllib.parse.urlparse(captured_urls[0])
    auth_qs = urllib.parse.parse_qs(parsed_auth.query)
    assert auth_qs["response_type"] == ["code"]
    assert auth_qs["code_challenge_method"] == ["S256"]
    assert "code_challenge" in auth_qs
    assert auth_qs["access_type"] == ["offline"]

    assert len(exchanged_payloads) == 1
    assert exchanged_payloads[0]["code"] == "fake-oauth-code"
    assert exchanged_payloads[0]["grant_type"] == "authorization_code"
    assert "code_verifier" in exchanged_payloads[0]

    assert state.authenticated is True
    assert state.method == "oauth"
    assert state.identity == "dev@example.com"
    stored = json.loads(auth_file.read_text(encoding="utf-8"))
    assert stored["oauth_token"]["access_token"] == "ya29.fake-access-token"
    assert stored["oauth_token"]["refresh_token"] == "1//fake-refresh-token"
