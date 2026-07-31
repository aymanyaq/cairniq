import os
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from tools.daily_cache import (
    _cache_path,
    _is_error_result,
    cleanup_old,
    get_cached,
    get_or_compute,
    set_cached,
)


def test_is_error_result():
    assert _is_error_result({"error": "Something went wrong"})
    assert _is_error_result({"Information": "Thank you for using Alpha Vantage! Our standard API call frequency..."})
    assert _is_error_result({"Note": "Thank you for using Alpha Vantage! Our standard API call frequency..."})
    assert _is_error_result("Error: Failed to fetch")

    assert not _is_error_result({"data": "valid"})
    assert not _is_error_result("Valid string")
    assert not _is_error_result(["list", "of", "items"])

@patch("tools.daily_cache.get_active_profile", return_value="pytest_profile")
def test_cache_path(mock_profile):
    path = _cache_path("test_key", for_date=date(2025, 1, 1))
    assert "pytest_profile_test_key_2025-01-01.json" in path

@patch("tools.daily_cache.get_active_profile", return_value="../../evil")
def test_cache_path_sanitizes_profile_and_key(mock_profile, tmp_path):
    with patch("tools.daily_cache.CACHE_DIR", str(tmp_path)):
        path = _cache_path("../../escape/key", for_date=date(2025, 1, 1))

    assert os.path.commonpath([str(tmp_path), path]) == str(tmp_path)
    assert ".." not in os.path.relpath(path, str(tmp_path))
    assert "/" not in os.path.basename(path)

@patch("tools.daily_cache.get_active_profile", return_value="pytest_profile")
def test_set_and_get_cached(mock_profile, tmp_path):
    with patch("tools.daily_cache.CACHE_DIR", str(tmp_path)):
        # Test normal set and get
        set_cached("test_key", {"status": "ok"})
        result = get_cached("test_key")
        assert result == {"status": "ok"}

        # Test error not cached
        set_cached("error_key", {"error": "bad"})
        assert get_cached("error_key") is None

@patch("tools.daily_cache.get_active_profile", return_value="pytest_profile")
def test_get_or_compute(mock_profile, tmp_path):
    with patch("tools.daily_cache.CACHE_DIR", str(tmp_path)):
        compute_fn = MagicMock(return_value={"computed": True})

        # Should call compute_fn and cache
        res = get_or_compute("compute_key", compute_fn)
        assert res == {"computed": True}
        compute_fn.assert_called_once()

        # Should return cached, not call compute_fn again
        compute_fn.reset_mock()
        res2 = get_or_compute("compute_key", compute_fn)
        assert res2 == {"computed": True}
        compute_fn.assert_not_called()

@patch("tools.daily_cache.get_active_profile", return_value="pytest_profile")
def test_cleanup_old(mock_profile, tmp_path):
    with patch("tools.daily_cache.CACHE_DIR", str(tmp_path)):
        from tools.daily_cache import _get_today
        today = _get_today()

        # Create an old cache file (10 days old)
        old_date = today - timedelta(days=10)
        old_path = os.path.join(str(tmp_path), f"pytest_profile_old_key_{old_date.isoformat()}.json")
        with open(old_path, 'w') as f:
            f.write("{}")

        # Create a new cache file (today)
        new_path = os.path.join(str(tmp_path), f"pytest_profile_new_key_{today.isoformat()}.json")
        with open(new_path, 'w') as f:
            f.write("{}")

        # Run cleanup
        removed = cleanup_old(max_age_days=7)

        assert removed == 1
        assert not os.path.exists(old_path)
        assert os.path.exists(new_path)
