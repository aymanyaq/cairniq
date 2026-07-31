"""Per-profile broker credential isolation (Phase 1)."""

import os
import shutil

import pytest
from fastapi.testclient import TestClient

from tools.user_profile import profile_context


@pytest.fixture()
def bc():
    """The broker_credentials module, with cleanup of any test artefacts.

    The conftest already forces the keychain off, so secrets mirror to
    os.environ (namespaced for named profiles).
    """
    import tools.broker_credentials as module

    yield module

    # Remove test profile dirs and any namespaced env keys we created.
    for name in ("pytest_alice", "pytest_bob"):
        shutil.rmtree(
            os.path.join(module._project_root(), "user_data", "profiles", name),
            ignore_errors=True,
        )
    for key in list(os.environ):
        if "::" in key and any(b in key for b in ("ALPACA", "QUESTRADE")):
            os.environ.pop(key, None)


@pytest.fixture()
def client():
    """TestClient with NO profile cookie, so the middleware resolves to default."""
    from server import app

    return TestClient(app)


# --- classification ---------------------------------------------------------
def test_classification(bc):
    assert bc.is_broker_secret("ALPACA_API_KEY")
    assert bc.is_broker_secret("ALPACA_SECRET_KEY")
    assert bc.is_broker_secret("QUESTRADE_REFRESH_TOKEN")
    assert bc.is_broker_secret("QUESTRADE_REFRESH_TOKEN_2")  # failover variant
    assert bc.is_broker_setting("ALPACA_PAPER_MODE")
    assert bc.is_broker_setting("QUESTRADE_ENABLED")
    assert bc.is_broker_setting("QUESTRADE_ACCESS_TOKEN")
    # Operator-global keys are NOT broker credentials.
    assert not bc.is_broker_credential("OPENAI_API_KEY")
    assert not bc.is_broker_credential("FMP_API_KEY")


# --- default profile keeps legacy global behaviour --------------------------
def test_default_profile_uses_global_namespace(bc):
    with profile_context("default"):
        bc.set_broker_secret("ALPACA_API_KEY", "global_key")
        assert bc.get_broker_secret("ALPACA_API_KEY") == "global_key"
        # stored under the bare (global) name, not namespaced
        assert os.environ.get("ALPACA_API_KEY") == "global_key"
        bc.set_broker_secret("ALPACA_API_KEY", "")  # clean up global env


# --- named profiles are isolated -------------------------------------------
def test_named_profile_secret_is_isolated(bc):
    with profile_context("pytest_alice"):
        bc.set_broker_secret("ALPACA_API_KEY", "alice_key")
        assert bc.get_broker_secret("ALPACA_API_KEY") == "alice_key"
        # stored under a per-profile namespace
        assert os.environ.get("pytest_alice::ALPACA_API_KEY") == "alice_key"

    # A different named profile sees nothing (no fallback to another account).
    with profile_context("pytest_bob"):
        assert bc.get_broker_secret("ALPACA_API_KEY") == ""

    # The default profile never sees a named profile's secret.
    with profile_context("default"):
        assert bc.get_broker_secret("ALPACA_API_KEY") != "alice_key"


def test_named_profile_setting_in_profile_json(bc):
    with profile_context("pytest_alice"):
        bc.set_broker_setting("ALPACA_PAPER_MODE", "false")
        assert bc.get_broker_setting("ALPACA_PAPER_MODE", "true") == "false"
        cfg = os.path.join(
            bc._project_root(), "user_data", "profiles", "pytest_alice", "broker_config.json"
        )
        assert os.path.exists(cfg)

    # Another profile gets the default, not Alice's setting.
    with profile_context("pytest_bob"):
        assert bc.get_broker_setting("ALPACA_PAPER_MODE", "true") == "true"


# --- endpoint wiring --------------------------------------------------------
def test_settings_save_routes_broker_secret(bc, client):
    resp = client.post(
        "/api/settings/save", json={"settings": {"QUESTRADE_REFRESH_TOKEN": "rt_test_123"}}
    )
    assert resp.status_code == 200
    with profile_context("default"):
        assert bc.get_broker_secret("QUESTRADE_REFRESH_TOKEN") == "rt_test_123"
        bc.set_broker_secret("QUESTRADE_REFRESH_TOKEN", "")  # clean up
