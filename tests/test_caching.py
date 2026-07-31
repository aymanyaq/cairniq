import os
import time

import pytest

from tools.daily_cache import (
    CACHE_DIR,
    _cache_path,
    _is_error_result,
    get_cached,
    set_cached,
)
from tools.user_profile import reset_profile, set_active_profile


@pytest.fixture
def setup_cache_env():
    """Ensure the cache tests use a safe test profile."""
    set_active_profile("pytest_cache_user")
    os.makedirs(CACHE_DIR, exist_ok=True)
    yield
    # Cleanup test caches
    for f in os.listdir(CACHE_DIR):
        if "pytest_cache_user" in f:
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except Exception:
                pass

def test_is_error_result():
    """Verify that error responses are correctly identified to prevent cache poisoning."""
    assert _is_error_result({"error": "Rate limit exceeded"})
    assert _is_error_result({"Information": "Thank you for using Alpha Vantage! Please visit... for more API limits."})
    assert _is_error_result("Error fetching data")

    assert not _is_error_result({"symbol": "AAPL", "price": 150})
    assert not _is_error_result([1, 2, 3])

def test_an_unavailable_payload_is_not_persisted_by_the_decorator(setup_cache_env):
    """A degraded source must not be pinned for the full TTL.

    `unavailable()` is a statement about the source RIGHT NOW — a missing key, an
    exhausted quota, an outage — not a value. Caching it keeps reporting a dead
    feed after the quota resets, which is the same "stale state presented as
    current" failure the fetch-time stamp exists to prevent. It also blocks the
    retry that would have recovered.
    """
    import uuid

    from tools.cache import cached
    from tools.tool_errors import unavailable

    calls = {"n": 0}
    # Unique per run: the SECOND call's real payload is legitimately cached to
    # disk, so a fixed key would make this test pass once and then read its own
    # previous run's success on the next.
    key = f"pytest_unavailable_probe_{uuid.uuid4().hex}"

    @cached(key_func=lambda: key)
    def flaky():
        calls["n"] += 1
        # Degraded on the first call, real data afterwards.
        if calls["n"] == 1:
            return unavailable("FMP", "quota exhausted")
        return {"value": 42}

    first = flaky()
    assert first["status"] == "unavailable"

    second = flaky()
    assert calls["n"] == 2, "the degraded result was cached and the retry never ran"
    assert second["value"] == 42


def test_daily_cache_persistence(setup_cache_env):
    """Verify that the daily cache saves and retrieves valid JSON data."""
    test_key = "test_persistence_key"
    test_data = {"status": "success", "value": 42}

    set_cached(test_key, test_data)

    retrieved = get_cached(test_key)
    assert retrieved is not None
    assert retrieved["value"] == 42

def test_daily_cache_error_prevention(setup_cache_env):
    """Verify that error results are NOT cached."""
    test_key = "test_error_key"
    error_data = {"error": "API Timeout"}

    set_cached(test_key, error_data)

    retrieved = get_cached(test_key)
    assert retrieved is None, "Error data was cached when it shouldn't have been!"


def test_daily_cache_respects_ttl(setup_cache_env):
    """Verify stale cache files are ignored when a TTL is supplied."""
    test_key = "test_ttl_key"
    set_cached(test_key, {"value": "fresh"})
    path = _cache_path(test_key)
    old_timestamp = time.time() - 3600
    os.utime(path, (old_timestamp, old_timestamp))

    assert get_cached(test_key, ttl_seconds=1) is None
    assert get_cached(test_key) == {"value": "fresh"}


def test_daily_cache_is_scoped_by_profile():
    """Verify one profile cannot read another profile's cached value."""
    import shutil

    from tools.user_profile import get_data_path

    # Clean up any residual data before starting the test
    for profile in ("pytest_cache_profile_a", "pytest_cache_profile_b"):
        token = set_active_profile(profile)
        try:
            profile_dir = os.path.dirname(get_data_path("__cleanup__"))
            if os.path.exists(profile_dir):
                shutil.rmtree(profile_dir, ignore_errors=True)
        finally:
            reset_profile(token)

    token_a = set_active_profile("pytest_cache_profile_a")
    try:
        set_cached("shared_key", {"profile": "a"})
    finally:
        reset_profile(token_a)

    token_b = set_active_profile("pytest_cache_profile_b")
    try:
        assert get_cached("shared_key") is None
        set_cached("shared_key", {"profile": "b"})
        assert get_cached("shared_key") == {"profile": "b"}
    finally:
        reset_profile(token_b)

    token_a = set_active_profile("pytest_cache_profile_a")
    try:
        assert get_cached("shared_key") == {"profile": "a"}
    finally:
        reset_profile(token_a)

    # Clean up at the end of the test
    for profile in ("pytest_cache_profile_a", "pytest_cache_profile_b"):
        # Clear legacy shared cache locations
        for filename in os.listdir(CACHE_DIR):
            if filename.startswith(f"{profile}_shared_key"):
                try: os.remove(os.path.join(CACHE_DIR, filename))
                except: pass
        # Clear isolated test directories
        token = set_active_profile(profile)
        try:
            profile_dir = os.path.dirname(get_data_path("__cleanup__"))
            if os.path.exists(profile_dir):
                shutil.rmtree(profile_dir, ignore_errors=True)
        finally:
            reset_profile(token)
