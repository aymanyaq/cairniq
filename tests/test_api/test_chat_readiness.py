"""Backend readiness for the chat-only iOS client.

Covers the contract the SwiftUI app codes against: a public health probe to
validate the (runtime-editable) server URL, auth-gated chat endpoints, and the
guarantee that a native client — which sends no Origin header — is not blocked
by the browser-oriented local-origin gate.
"""

import pytest
from fastapi.testclient import TestClient

_TEST_PASSWORD = "supersecret123"  # noqa: S105 — test fixture credential


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRNIQ_AUTH_DB", str(tmp_path / "auth.json"))
    monkeypatch.setenv("CAIRNIQ_JWT_SECRET", "test-secret-please-ignore-0123456789abcdef")
    monkeypatch.delenv("CAIRNIQ_AUTH_REQUIRED", raising=False)
    import tools.auth as auth

    return auth


@pytest.fixture()
def client():
    from server import app

    return TestClient(app)


def _login(auth, client, username="alice"):
    auth.create_user(username, _TEST_PASSWORD, profile=username)
    resp = client.post(
        "/api/auth/login", json={"username": username, "password": _TEST_PASSWORD}
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


# --- health probe -----------------------------------------------------------
def test_health_is_public_and_reports_version(auth_env, client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]
    # Auth is on by default
    assert body["auth_required"] is True


def test_health_reports_auth_not_required_when_disabled(auth_env, client, monkeypatch):
    monkeypatch.setenv("CAIRNIQ_AUTH_REQUIRED", "0")
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["auth_required"] is False


def test_health_stays_public_when_auth_required(auth_env, client, monkeypatch):
    # Explicitly enabling auth (already the default) keeps /api/health public.
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["auth_required"] is True


# --- chat endpoints are auth-gated -----------------------------------------
def test_chat_post_requires_token_when_auth_on(auth_env, client, monkeypatch):
    # Rejected by the profile middleware before the chat handler / agent runs.
    monkeypatch.setenv("CAIRNIQ_AUTH_REQUIRED", "1")
    resp = client.post("/api/chat", json={"message": "ping"})
    assert resp.status_code == 401


def test_chats_list_gated_then_accessible_with_token(auth_env, client, monkeypatch):
    # GET /api/chats is the resume-on-drop read the app falls back to.
    monkeypatch.setenv("CAIRNIQ_AUTH_REQUIRED", "1")
    assert client.get("/api/chats").status_code == 401

    token = _login(auth_env, client)
    resp = client.get("/api/chats", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


# --- native client (no Origin header) is not blocked -----------------------
def test_bad_origin_blocked_but_native_client_allowed(auth_env, client):
    # A browser POST from a disallowed origin is rejected by the local-origin gate.
    blocked = client.post(
        "/api/auth/login",
        json={"username": "x", "password": "y"},
        headers={"Origin": "http://evil.example"},
    )
    assert blocked.status_code == 403

    # A native client sends no Origin header -> it passes the gate and reaches
    # the handler (401 invalid creds, NOT 403 origin-blocked).
    native = client.post("/api/auth/login", json={"username": "x", "password": "y"})
    assert native.status_code == 401


# --- network-exposure safety guard (VPN hosting) ---------------------------
def test_loopback_bind_never_warns():
    from server import network_exposure_warning

    assert network_exposure_warning("127.0.0.1", auth_on=False) is None
    assert network_exposure_warning("localhost", auth_on=False) is None


def test_exposed_bind_without_auth_warns():
    from server import network_exposure_warning

    for host in ("0.0.0.0", "", "192.168.1.50"):  # noqa: S104 — test data, not a bind
        warning = network_exposure_warning(host, auth_on=False)
        assert warning and "CAIRNIQ_AUTH_REQUIRED" in warning


def test_exposed_bind_with_auth_is_safe():
    from server import network_exposure_warning

    assert network_exposure_warning("0.0.0.0", auth_on=True) is None  # noqa: S104 — test data
