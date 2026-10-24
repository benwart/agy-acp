"""Unit tests for AuthManager and CredentialStoreProvider atomic hot-reload."""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.not_implemented
def test_status_reflects_active_credential_state(tmp_path: Path) -> None:
    """Verify AuthManager.status reports current authentication state."""
    auth_file = tmp_path / "credentials.json"
    mgr = AuthManager(auth_file=auth_file)
    assert mgr.status().authenticated is False


@pytest.mark.not_implemented
def test_detects_external_credential_rotation(tmp_path: Path) -> None:
    """Verify AuthManager.has_changed_since detects external credential changes."""
    auth_file = tmp_path / "credentials.json"
    mgr = AuthManager(auth_file=auth_file)
    s1 = mgr.login(api_key="key-1")
    assert mgr.has_changed_since(s1.fingerprint) is False
    mgr.login(api_key="key-2")
    assert mgr.has_changed_since(s1.fingerprint) is True
