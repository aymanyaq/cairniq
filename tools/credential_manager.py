"""
Credential Manager — API key rate-limit tracking.

Supports tracking rate-limits for API keys. When a request
fails with a rate-limit error (HTTP 429, "too many requests", etc.), the
manager marks the key as rate-limited to avoid immediate retries.

Usage:
    from tools.credential_manager import get_api_key, report_rate_limit

    key = get_api_key("FMP_API_KEY")          # Returns key if not in cooldown
    response = requests.get(url, params={"apikey": key})
    if response.status_code == 429:
        report_rate_limit("FMP_API_KEY")      # Marks current key as limited
"""

import os
import threading
import time

from agent.utils import safe_print

# How long (seconds) a key stays "rate-limited" before being retried
RATE_LIMIT_COOLDOWN = 300  # 5 minutes

# Thread-safe lock for state mutations
_lock = threading.Lock()

# Tracks rate-limit timestamps: {"FMP_API_KEY": timestamp, "FMP_API_KEY_2": timestamp}
_rate_limited_until: dict[str, float] = {}

# Supported services and their env vars
SUPPORTED_SERVICES = {
    "FMP_API_KEY":            ["FMP_API_KEY"],
    "POLYGON_API_KEY":        ["POLYGON_API_KEY"],
    "FINNHUB_API_KEY":        ["FINNHUB_API_KEY"],
    "TAVILY_API_KEY":         ["TAVILY_API_KEY"],
    "ALPHA_VANTAGE_API_KEY":  ["ALPHA_VANTAGE_API_KEY"],
    "FRED_API_KEY":           ["FRED_API_KEY"],
    "ALPACA_API_KEY":         ["ALPACA_API_KEY"],
}


def get_api_key(service: str, default: str = "") -> str:
    """
    Return the best available API key for the given service.

    Strategy: try secondary key first (to preserve primary budget),
    skip any key currently in rate-limit cooldown, fall back to primary.
    """
    now = time.time()
    candidates = SUPPORTED_SERVICES.get(service, [service])

    with _lock:
        # First pass: prefer keys that are NOT rate-limited
        for env_var in candidates:
            if _rate_limited_until.get(env_var, 0) > now:
                continue  # still in cooldown
            key = os.environ.get(env_var, "")
            if key:
                return key

        # Second pass: all keys are rate-limited — return the one whose
        # cooldown expires soonest (best chance of working)
        best_var = None
        best_expiry = float("inf")
        for env_var in candidates:
            key = os.environ.get(env_var, "")
            if not key:
                continue
            expiry = _rate_limited_until.get(env_var, 0)
            if expiry < best_expiry:
                best_expiry = expiry
                best_var = env_var

        if best_var:
            # Clear the expired cooldown so it's usable
            _rate_limited_until.pop(best_var, None)
            return os.environ.get(best_var, default)

    return os.environ.get(service, default)


def report_rate_limit(service: str, key_value: str | None = None) -> None:
    """
    Mark the current key for a service as rate-limited.

    If key_value is provided, only that specific key is marked.
    Otherwise, the first matching key in the candidate list is marked.
    """
    now = time.time()
    candidates = SUPPORTED_SERVICES.get(service, [service])

    with _lock:
        if key_value:
            # Find which env var holds this exact key
            for env_var in candidates:
                if os.environ.get(env_var, "") == key_value:
                    _rate_limited_until[env_var] = now + RATE_LIMIT_COOLDOWN
                    safe_print(f"⚠️ Credential rotation: {env_var} rate-limited, cooldown {RATE_LIMIT_COOLDOWN}s")
                    return
        else:
            # Mark the first non-empty, non-limited key as limited
            for env_var in candidates:
                key = os.environ.get(env_var, "")
                if key and _rate_limited_until.get(env_var, 0) <= now:
                    _rate_limited_until[env_var] = now + RATE_LIMIT_COOLDOWN
                    safe_print(f"⚠️ Credential rotation: {env_var} rate-limited, cooldown {RATE_LIMIT_COOLDOWN}s")
                    return


def is_rate_limit_error(response_or_exception) -> bool:
    """
    Convenience helper to detect rate-limit signals from a requests.Response
    or an Exception.
    """
    if hasattr(response_or_exception, "status_code"):
        return response_or_exception.status_code == 429
    msg = str(response_or_exception).lower()
    return any(kw in msg for kw in ["429", "rate limit", "too many requests", "throttl"])


def get_credential_status() -> dict:
    """Return a diagnostic summary of credential health (for health_check)."""
    now = time.time()
    status = {}
    for service, candidates in SUPPORTED_SERVICES.items():
        keys_info = []
        for env_var in candidates:
            key = os.environ.get(env_var, "")
            limited_until = _rate_limited_until.get(env_var, 0)
            keys_info.append({
                "env_var": env_var,
                "configured": bool(key),
                "rate_limited": limited_until > now,
                "cooldown_remaining": max(0, int(limited_until - now)) if limited_until > now else 0,
            })
        status[service] = keys_info
    return status
