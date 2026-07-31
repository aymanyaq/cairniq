"""Tests for the multi-user auth layer (login -> token -> profile)."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    """Isolate the user store + signing secret; auth enforcement off by default."""
    monkeypatch.setenv("CAIRNIQ_AUTH_DB", str(tmp_path / "auth.json"))
    monkeypatch.setenv("CAIRNIQ_JWT_SECRET", "test-secret-please-ignore-0123456789abcdef")
    monkeypatch.delenv("CAIRNIQ_AUTH_REQUIRED", raising=False)
    import tools.auth as auth

    return auth


@pytest.fixture()
def client():
    from server import app

    return TestClient(app)


# --- password hashing -------------------------------------------------------
def test_password_roundtrip(auth_env):
    encoded = auth_env.hash_password("hunter2hunter")
    assert auth_env.verify_password("hunter2hunter", encoded)
    assert not auth_env.verify_password("wrong-password", encoded)


def test_short_password_rejected(auth_env):
    with pytest.raises(ValueError):
        auth_env.create_user("alice", "short")


# --- user store -------------------------------------------------------------
def test_create_and_verify_credentials(auth_env):
    auth_env.create_user("alice", "supersecret", profile="alice", role="admin")
    assert auth_env.verify_credentials("alice", "supersecret")["profile"] == "alice"
    assert auth_env.verify_credentials("alice", "nope") is None
    assert auth_env.verify_credentials("ghost", "whatever") is None


def test_username_is_case_insensitive(auth_env):
    auth_env.create_user("Alice", "supersecret")
    assert auth_env.verify_credentials("alice", "supersecret") is not None


def test_duplicate_user_rejected(auth_env):
    auth_env.create_user("alice", "supersecret")
    with pytest.raises(ValueError):
        auth_env.create_user("alice", "anothersecret")


# --- tokens -----------------------------------------------------------------
def test_token_roundtrip(auth_env):
    user = auth_env.create_user("alice", "supersecret", profile="alice")
    token, ttl = auth_env.issue_token(user)
    assert ttl > 0
    claims = auth_env.verify_token(token)
    assert claims["sub"] == "alice"
    assert claims["profile"] == "alice"


def test_token_version_invalidates_on_password_change(auth_env):
    user = auth_env.create_user("alice", "supersecret")
    token, _ = auth_env.issue_token(user)
    assert auth_env.verify_token(token) is not None
    auth_env.set_password("alice", "brandnewpassword")
    assert auth_env.verify_token(token) is None


def test_garbage_token_rejected(auth_env):
    assert auth_env.verify_token("not.a.jwt") is None
    assert auth_env.verify_token("") is None


# --- API: login / me --------------------------------------------------------
def test_login_and_me(auth_env, client):
    auth_env.create_user("alice", "supersecret", profile="alice", role="admin")
    resp = client.post(
        "/api/auth/login", json={"username": "alice", "password": "supersecret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["profile"] == "alice"
    assert client.cookies.get("cairniq_token")

    me = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert me.status_code == 200
    assert me.json()["username"] == "alice"


def test_login_bad_password(auth_env, client):
    auth_env.create_user("alice", "supersecret")
    resp = client.post(
        "/api/auth/login", json={"username": "alice", "password": "WRONG"}
    )
    assert resp.status_code == 401


def test_me_requires_token(auth_env, client):
    assert client.get("/api/auth/me").status_code == 401


# --- enforcement flag -------------------------------------------------------
def test_protected_route_open_when_auth_not_required(auth_env, client, monkeypatch):
    # Explicitly disable auth (opt-out): existing single-user behaviour, no token needed.
    monkeypatch.setenv("CAIRNIQ_AUTH_REQUIRED", "0")
    assert client.get("/api/profiles").status_code == 200


def test_enforcement_blocks_without_token(auth_env, client, monkeypatch):
    auth_env.create_user("alice", "supersecret", profile="alice")
    # Auth is on by default; verify protected routes require a token.

    # Protected route without a token is rejected.
    assert client.get("/api/profiles").status_code == 401

    # The login endpoint stays public.
    login = client.post(
        "/api/auth/login", json={"username": "alice", "password": "supersecret"}
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    # With a valid token the route is reachable again.
    ok = client.get("/api/profiles", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200


def test_register_creates_user_and_logs_in(auth_env, client):
    resp = client.post(
        "/api/auth/register", json={"username": "newuser", "password": "securepass"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "newuser"
    assert body["token_type"] == "bearer"
    # Cookie should be set
    assert client.cookies.get("cairniq_token")


def test_register_first_user_is_admin(auth_env, client):
    resp = client.post(
        "/api/auth/register", json={"username": "boss", "password": "securepass"}
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"


def test_register_second_user_is_user(auth_env, client):
    client.post("/api/auth/register", json={"username": "first", "password": "securepass"})
    resp = client.post(
        "/api/auth/register", json={"username": "second", "password": "securepass"}
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "user"


def test_register_duplicate_rejected(auth_env, client):
    client.post("/api/auth/register", json={"username": "alice", "password": "securepass"})
    resp = client.post("/api/auth/register", json={"username": "alice", "password": "another"})
    assert resp.status_code == 400


def test_register_short_password_rejected(auth_env, client):
    resp = client.post("/api/auth/register", json={"username": "alice", "password": "short"})
    assert resp.status_code == 400
