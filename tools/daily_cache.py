"""
Daily Cache Module
File-based daily caching that persists across server restarts.
Stores JSON files in user_data/daily_cache/ with date-stamped filenames.
Cache keys are automatically prefixed with the active profile name so
that different profiles never share cached data (benchmark, news, etc.).
"""
import glob
import json
import math
import os
import re
import zoneinfo
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from tools.user_profile import get_active_profile

# Cache directory inside user_data
# Cache directory inside user_data
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_data", "daily_cache")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _is_error_result(data: Any) -> bool:
    """Return True if data looks like an error that should NOT be cached."""
    if isinstance(data, dict):
        if "error" in data:
            return True
        # Check Alpha Vantage rate limits masquerading as valid data
        if "Information" in data and "API" in str(data["Information"]):
            return True
        if "Note" in data and "API" in str(data["Note"]):
            return True
    if isinstance(data, str) and data.startswith("Error"):
        return True
    return False


def _strip_non_finite(obj: Any) -> Any:
    """Replace NaN/Infinity floats with None, recursively.

    Only ever called after a strict serialization attempt has already failed,
    so the walk costs nothing on the overwhelmingly common clean payload.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strip_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_non_finite(v) for v in obj]
    return obj


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_today() -> date:
    """Get today's date in US/Eastern timezone."""
    # This ensures consistency for the user's localized 'day'
    eastern = zoneinfo.ZoneInfo("US/Eastern")
    return datetime.now(eastern).date()


def _safe_cache_part(value: str) -> str:
    """Sanitize profile names and cache keys for use as one filename segment."""
    safe = _SAFE_FILENAME_RE.sub("_", str(value or "cache")).strip("._-")
    return safe[:180] or "cache"


def _cache_path(key: str, for_date: date | None = None) -> str:
    """Get the file path for a cache key on a given date, scoped to the active profile."""
    d = for_date or _get_today()
    profile = get_active_profile()

    # If this is a test run profile, write to the profile directory to avoid cache poisoning/pollution
    if profile.startswith("pytest_"):
        from tools.user_profile import get_data_path
        profile_dir = os.path.dirname(get_data_path("__cleanup__"))
        cache_dir = os.path.join(profile_dir, "daily_cache")
    else:
        cache_dir = CACHE_DIR

    profile_safe = _safe_cache_part(profile)
    cache_key = _safe_cache_part(key)
    # Format: {profile}_{key}_{YYYY-MM-DD}.json
    filename = f"{profile_safe}_{cache_key}_{d.isoformat()}.json"
    path = os.path.abspath(os.path.join(cache_dir, filename))
    cache_root = os.path.abspath(cache_dir)
    if os.path.commonpath([cache_root, path]) != cache_root:
        raise ValueError(f"Cache path escapes daily cache directory: {key}")
    return path


def get_cached(key: str, ttl_seconds: int | None = None) -> Any | None:
    """
    Get cached value for a key, or None if not cached or expired.
    If ttl_seconds is provided, checks file modification time.
    """
    path = _cache_path(key)
    if os.path.exists(path):
        # Check TTL if provided
        if ttl_seconds is not None:
            mtime = os.path.getmtime(path)
            if (datetime.now().timestamp() - mtime) > ttl_seconds:
                # Cache expired
                return None

        try:
            with open(path) as f:
                # NaN/Infinity are Python-json extensions, not valid JSON, and a
                # cache file written before the set_cached guard below can still
                # hold them. Reading them back as None keeps a poisoned file from
                # 500ing whatever endpoint serves it — parse_constant fires only
                # on those three tokens, so a clean file pays nothing.
                return json.load(f, parse_constant=lambda _token: None)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def set_cached(key: str, data: Any) -> None:
    """Store data in today's cache file. Skips error results.

    Never writes NaN or Infinity. Python's json emits them as bare `NaN` /
    `Infinity` tokens, which are not valid JSON: one NaN from an upstream feed
    lands in the file, and every consumer that re-serializes it (a JSONResponse,
    the iOS client) fails for the rest of the day, because only the date rollover
    replaces the file. A single bad number should not be a day-long outage.
    """
    if _is_error_result(data):
        return
    try:
        payload = json.dumps(data, indent=2, default=str, allow_nan=False)
    except ValueError:
        payload = json.dumps(_strip_non_finite(data), indent=2, default=str, allow_nan=False)
    path = _cache_path(key)
    # Ensure directory containing the cache file exists (crucial for isolated profile paths)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Serialize fully before opening: a mid-write failure would otherwise leave a
    # truncated file that reads back as a JSONDecodeError.
    with open(path, 'w') as f:
        f.write(payload)


def get_or_compute(key: str, compute_fn: Callable[[], Any], ttl_seconds: int | None = None) -> Any:
    """
    Return today's cached data for `key`.
    If no cache exists or is expired (based on ttl_seconds), call compute_fn().
    """
    cached = get_cached(key, ttl_seconds=ttl_seconds)
    if cached is not None and not _is_error_result(cached):
        return cached

    result = compute_fn()
    set_cached(key, result)
    return result


def cleanup_old(max_age_days: int = 7) -> int:
    """Remove cache files older than max_age_days. Returns count of removed files."""
    _ensure_dir()
    cutoff = _get_today() - timedelta(days=max_age_days)
    removed = 0
    for filepath in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        fname = os.path.basename(filepath)
        # Extract date from filename: key_YYYY-MM-DD.json
        try:
            date_str = fname.rsplit("_", 1)[-1].replace(".json", "")
            file_date = date.fromisoformat(date_str)
            if file_date < cutoff:
                os.remove(filepath)
                removed += 1
        except (ValueError, IndexError):
            continue
    return removed
