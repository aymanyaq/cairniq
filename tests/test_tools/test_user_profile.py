import json
import os
from pathlib import Path

import pytest

from tools.user_profile import (
    _profile_ctx,
    ensure_demo_profile,
    get_active_profile,
    get_data_path,
    get_demo_profile_name,
    is_demo_mode,
    list_available_profiles,
    profile_context,
    reset_profile,
    set_active_profile,
)


@pytest.fixture(autouse=True)
def clean_profile_env():
    # Store original env
    orig_env = os.environ.get("ACTIVE_PROFILE")

    # Reset context var
    token = _profile_ctx.set(None)

    # Store and disable multiuser guard
    from tools.user_profile import enable_multiuser_guard, is_multiuser_guard_enabled
    orig_guard = is_multiuser_guard_enabled()
    enable_multiuser_guard(False)

    yield

    # Restore
    enable_multiuser_guard(orig_guard)
    _profile_ctx.reset(token)
    if orig_env is not None:
        os.environ["ACTIVE_PROFILE"] = orig_env
    elif "ACTIVE_PROFILE" in os.environ:
        del os.environ["ACTIVE_PROFILE"]

def test_get_active_profile_default():
    if "ACTIVE_PROFILE" in os.environ:
        del os.environ["ACTIVE_PROFILE"]
    assert get_active_profile() == "default"

def test_get_active_profile_env():
    os.environ["ACTIVE_PROFILE"] = "env_profile"
    assert get_active_profile() == "env_profile"

def test_reserved_demo_profile_env_ignored_outside_demo_mode():
    os.environ["ACTIVE_PROFILE"] = "demo"
    assert get_active_profile() == "default"

def test_set_and_reset_profile():
    token = set_active_profile("context_profile")
    assert get_active_profile() == "context_profile"

    reset_profile(token)
    assert get_active_profile() == "default"

def test_profile_context():
    assert get_active_profile() == "default"
    with profile_context("temp_profile"):
        assert get_active_profile() == "temp_profile"
    assert get_active_profile() == "default"

def test_list_available_profiles(tmp_path, monkeypatch):
    # Mock the directory where profiles are found
    base_dir = tmp_path
    user_data_dir = base_dir / "user_data"
    profiles_dir = user_data_dir / "profiles"

    # Create some mock profiles
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "alice").mkdir()
    (profiles_dir / "bob_test").mkdir()
    (profiles_dir / "demo").mkdir()
    (profiles_dir / ".hidden").mkdir()

    monkeypatch.setattr("tools.user_profile.os.path.dirname", lambda x: str(base_dir))

    profiles = list_available_profiles()
    names = [p["name"] for p in profiles]

    assert "default" in names
    assert "alice" in names
    assert "bob_test" in names
    assert "demo" not in names
    assert ".hidden" not in names

    # Check display names
    bob_prof = next(p for p in profiles if p["name"] == "bob_test")
    assert bob_prof["display_name"] == "Bob Test"

def test_get_data_path(tmp_path, monkeypatch):
    base_dir = tmp_path
    monkeypatch.setattr("tools.user_profile.os.path.dirname", lambda x: str(base_dir))

    # default profile
    default_path = get_data_path("test.csv")
    assert default_path == str(base_dir / "user_data" / "test.csv")

    # custom profile
    with profile_context("custom_profile"):
        custom_path = get_data_path("test.csv")
        assert custom_path == str(base_dir / "user_data" / "profiles" / "custom_profile" / "test.csv")

def test_demo_mode_forces_isolated_profile(tmp_path, monkeypatch):
    base_dir = tmp_path
    monkeypatch.setattr("tools.user_profile.os.path.dirname", lambda x: str(base_dir))
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_PROFILE", "demo")

    token = set_active_profile("default")
    try:
        assert is_demo_mode() is True
        assert get_demo_profile_name() == "demo"
        assert get_active_profile() == "demo"
        assert get_data_path("my_portfolio.csv") == str(
            base_dir / "user_data" / "profiles" / "demo" / "my_portfolio.csv"
        )
    finally:
        reset_profile(token)

def test_demo_profile_seed_and_reset(tmp_path, monkeypatch):
    base_dir = tmp_path
    monkeypatch.setattr("tools.user_profile.os.path.dirname", lambda x: str(base_dir))
    monkeypatch.setenv("DEMO_MODE", "true")

    profile_dir = Path(ensure_demo_profile(reset=True))
    memory_path = profile_dir / "user_memory.json"
    portfolio_path = profile_dir / "my_portfolio.csv"
    history_path = profile_dir / "demo_portfolio_history.csv"

    assert profile_dir.name == "demo"
    assert portfolio_path.exists()
    assert "Demo Brokerage" in portfolio_path.read_text()
    assert "Current Price" in portfolio_path.read_text()
    assert history_path.exists()
    assert len(history_path.read_text().splitlines()) > 30
    assert json.loads(memory_path.read_text())["user_profile"]["name"] == "Demo User"

    memory_path.write_text('{"user_profile": {"name": "Real Name"}}')
    (profile_dir / "knowledge_graph.json").write_text("{}")
    ensure_demo_profile(reset=True)

    assert json.loads(memory_path.read_text())["user_profile"]["name"] == "Demo User"
    assert not (profile_dir / "knowledge_graph.json").exists()
    assert history_path.exists()
