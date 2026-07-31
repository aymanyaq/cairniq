import shutil
from pathlib import Path

from fastapi.testclient import TestClient

import server
from tools.user_profile import get_active_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_profile_middleware_uses_cookie_and_resets_context():
    profile_dir = PROJECT_ROOT / "user_data" / "profiles" / "pytest_apiprofile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(server.app)
    client.cookies.set("profile", "pytest_apiprofile")

    try:
        response = client.get("/api/profiles")
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)

    assert response.status_code == 200
    assert response.json()["active"] == "pytest_apiprofile"
    assert get_active_profile() != "pytest_apiprofile"


def test_profile_middleware_rejects_unknown_or_unsafe_cookie():
    client = TestClient(server.app)
    client.cookies.set("profile", "../../../tmp/pwn")

    response = client.get("/api/profiles")

    assert response.status_code == 200
    assert response.json()["active"] == "default"


def test_profile_middleware_ignores_cookie_in_demo_mode(monkeypatch):
    profile_dir = PROJECT_ROOT / "user_data" / "profiles" / "pytest_demo_api"
    client = TestClient(server.app)
    client.cookies.set("profile", "default")
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_PROFILE", "pytest_demo_api")

    try:
        response = client.get("/api/profiles")
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)

    assert response.status_code == 200
    assert response.json()["active"] == "pytest_demo_api"
    assert response.json()["profiles"] == [
        {
            "name": "pytest_demo_api",
            "display_name": "Demo",
            "data_path": str(profile_dir),
            "active": True,
            "is_demo": True,
        }
    ]


def test_reserved_demo_profile_cookie_is_ignored_outside_demo_mode(monkeypatch):
    profile_dir = PROJECT_ROOT / "user_data" / "profiles" / "demo"
    existed = profile_dir.exists()
    profile_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(server.app)
    client.cookies.set("profile", "demo")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("CAIRNIQ_FORCE_DEMO", raising=False)

    try:
        response = client.get("/api/profiles")
    finally:
        if not existed:
            shutil.rmtree(profile_dir, ignore_errors=True)

    assert response.status_code == 200
    assert response.json()["active"] == "default"


def test_mutating_requests_reject_untrusted_origin():
    client = TestClient(server.app)

    response = client.post(
        "/api/settings/save",
        headers={"Origin": "https://evil.example"},
        json={"settings": {}},
    )

    assert response.status_code == 403


def test_cors_does_not_allow_wildcard_with_credentials():
    assert "*" not in server.ALLOWED_ORIGINS


def test_profile_switch_validates_profiles_and_sets_cookie():
    profile_dir = PROJECT_ROOT / "user_data" / "profiles" / "pytest_switch_target"
    profile_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(server.app)

    try:
        missing = client.post("/api/profile/switch", json={"profile": "does_not_exist"})
        switched = client.post(
            "/api/profile/switch", json={"profile": "pytest_switch_target"}
        )
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)

    assert missing.status_code == 404
    assert switched.status_code == 200
    assert switched.json()["active_profile"] == "pytest_switch_target"
    assert "profile=pytest_switch_target" in switched.headers.get("set-cookie", "")


def test_empty_secret_field_does_not_wipe_existing_keychain_entry(monkeypatch):
    """Regression: a Settings-page Save with an empty value for a configured
    secret must NOT delete it from the keychain. The form can briefly render
    fields as empty (e.g. if `load_secrets_into_env` failed during a transient
    startup hiccup); a save of an unrelated setting must not collateral-damage
    the user's API keys.
    """
    import tools.secrets_store as secrets_store

    set_calls: list[tuple[str, str]] = []
    fake_store: dict[str, str] = {
        "FRED_API_KEY": "existing-real-secret",
        "POLYGON_API_KEY": "another-real-secret",
    }

    def fake_get(name: str) -> str:
        return fake_store.get(name, "")

    def fake_set(name: str, value: str) -> bool:
        set_calls.append((name, value))
        if value:
            fake_store[name] = value
        else:
            fake_store.pop(name, None)
        return True

    monkeypatch.setattr(secrets_store, "get_secret", fake_get)
    monkeypatch.setattr(secrets_store, "set_secret", fake_set)

    client = TestClient(server.app)
    response = client.post(
        "/api/settings/save",
        json={
            "settings": {
                # Empty values for secrets that ARE configured in the keychain.
                # These must NOT trigger a delete — they should be skipped.
                "FRED_API_KEY": "",
                "POLYGON_API_KEY": "",
                # A non-secret unrelated setting being saved at the same time.
                "BASE_CURRENCY": "USD",
                # A genuinely new secret value that SHOULD be written through.
                "TAVILY_API_KEY": "newly-entered-value",
            }
        },
    )

    assert response.status_code == 200, response.text
    # The empty-field secrets must still be present untouched
    assert fake_store["FRED_API_KEY"] == "existing-real-secret"
    assert fake_store["POLYGON_API_KEY"] == "another-real-secret"
    # The newly-entered one must have been written
    assert fake_store["TAVILY_API_KEY"] == "newly-entered-value"
    # set_secret should have been called exactly once — only for the new value
    assert set_calls == [("TAVILY_API_KEY", "newly-entered-value")]


def test_settings_and_profile_switch_are_locked_in_demo_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_PROFILE", "pytest_demo_lock")
    profile_dir = PROJECT_ROOT / "user_data" / "profiles" / "pytest_demo_lock"
    client = TestClient(server.app)

    try:
        settings_response = client.post("/api/settings/save", json={"settings": {"QUESTRADE_ENABLED": "true"}})
        switch_response = client.post("/api/profile/switch", json={"profile": "default"})
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)

    assert settings_response.status_code == 403
    assert switch_response.status_code == 403
    assert switch_response.json()["active_profile"] == "pytest_demo_lock"


def test_dashboard_data_groups_allocation_tail_as_others(monkeypatch):
    holdings = [
        {"symbol": "AAA", "value_cad": 100.0},
        {"symbol": "BBB", "value_cad": 90.0},
        {"symbol": "CCC", "value_cad": 80.0},
        {"symbol": "DDD", "value_cad": 70.0},
        {"symbol": "EEE", "value_cad": 60.0},
        {"symbol": "FFF", "value_cad": 50.0},
        {"symbol": "GGG", "value_cad": 10.0},
        {"symbol": "HHH", "value_cad": 5.0},
    ]

    monkeypatch.setattr(
        "api.routers.dashboard.get_portfolio_summary",
        lambda: {
            "holdings": holdings,
            "summary": {"current_value": "$465"},
            "liquidity": {},
            "accounts": [],
            "top_winners": [],
            "top_losers": [],
            "sync_errors": [],
        },
    )
    client = TestClient(server.app)

    response = client.get("/api/dashboard-data")
    payload = response.json()

    assert response.status_code == 200
    assert payload["allocation"]["labels"] == [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
        "EEE",
        "FFF",
        "Others",
    ]
    assert payload["allocation"]["values"] == [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 15.0]


def test_session_cost_api():
    from api.routers.chat import _thread_costs, _thread_tokens
    _thread_costs["test-thread-id-1"] = 1.25
    _thread_tokens["test-thread-id-1"] = 5000
    _thread_costs["test-thread-id-2"] = 3.50

    client = TestClient(server.app)

    # 1. cost (raw CAD) + tokens for a known thread. `currency` reflects the base
    #    currency, so assert the deterministic fields, not the converted value.
    r1 = client.get("/api/session/cost?thread_id=test-thread-id-1")
    assert r1.status_code == 200
    j1 = r1.json()
    assert j1["cost_cad"] == 1.25
    assert j1["tokens"] == 5000
    assert "currency" in j1

    # 2. another thread (no tokens recorded → 0)
    r2 = client.get("/api/session/cost?thread_id=test-thread-id-2")
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["cost_cad"] == 3.50
    assert j2["tokens"] == 0

    # 3. no thread_id or unknown thread_id → zero
    r3 = client.get("/api/session/cost")
    assert r3.status_code == 200
    assert r3.json()["cost_cad"] == 0.0

    r4 = client.get("/api/session/cost?thread_id=nonexistent-id")
    assert r4.status_code == 200
    assert r4.json()["cost_cad"] == 0.0
