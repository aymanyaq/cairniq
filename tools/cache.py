"""
Cache and Parallel Fetching Utilities
Provides session-level caching and parallel data fetching for performance.
"""
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any

import tools.daily_cache as daily_cache

# --- SESSION CACHE (In-Memory for transient state) ---
_cache: dict[str, dict[str, Any]] = {}
CACHE_TTL = 900  # Increased to 15 minutes as per user request for performance optimization

_hits = 0
_misses = 0

def cache_stats() -> dict[str, Any]:
    """Return cache hit/miss stats and number of cached keys."""
    return {"hits": _hits, "misses": _misses, "keys": len(_cache)}

def get_cached(key: str) -> Any:
    """Get cached value from in-memory SESSION cache."""
    global _hits, _misses
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry['timestamp'] < CACHE_TTL:
            return entry['value']
        else:
            del _cache[key]
    return None

def set_cached(key: str, value: Any) -> None:
    """Set value in in-memory SESSION cache."""
    _cache[key] = {
        'value': value,
        'timestamp': time.time()
    }

def clear_cache() -> None:
    """Clear all cached data."""
    _cache.clear()

def _stamp_fetch(result: Any) -> Any:
    """Apply the 5.8 `_as_of` fetch stamp, never raising.

    A freshness stamp is EVIDENCE, and a caching layer that started throwing
    because it could not annotate a payload would take the data down with it.
    An unstamped payload is reported as *unverified* downstream, which is the
    correct degradation: absence of proof is not proof of freshness.
    """
    try:
        from tools.freshness import stamp
        return stamp(result)
    except Exception:
        return result


def cached(key_func: Callable = None, ttl: int = 3600, stamp: bool = True):
    """
    Decorator to cache function results PERSISTENTLY using DailyCache.
    Default TTL is 1 hour (3600s).
    Error results are NOT cached to avoid poisoning the cache.

    Every cached dict result is `_as_of`-stamped at FETCH time (Roadmap 5.8).
    This decorator is the one place all ~95 cached tools pass through, and it is
    the only place the distinction the stamp exists for can be drawn: the stamp
    goes on in the cache-MISS branch, so a later hit replays the original fetch
    time instead of restarting the clock on every read. Stamping at any call
    site downstream of the cache would make a stale payload look permanently
    fresh, which is the exact bug 5.8 was opened for.

    `stamp` is idempotent, so a tool that already stamps itself at its true
    fetch moment (get_stock_data, boc_valet) keeps its own more precise value.
    Non-dict results pass through untouched — a list or a string has nowhere to
    carry a stamp, and `tools.freshness` reports those as *unverified* rather
    than fresh, which is the honest answer.

    Pass ``stamp=False`` for MAP-shaped returns, where the dict's keys are data
    rather than field names ({TICKER: {...}}, {CUSIP: {...}}). An in-band stamp
    on one of those is not metadata — it is a phantom entry, and every consumer
    that iterates `.items()` reads `_as_of` as if it were a ticker and crashes
    on its string value. That silently killed the entire 13F universe producer
    (`'str' object has no attribute 'get'` × every tracked manager). Reference
    maps have no meaningful fetch-freshness anyway: a CUSIP→holding table from a
    quarterly filing is not a market observation, so reporting it as
    *unverified* costs nothing. Record-shaped payloads keep the stamp.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key scoped by function name and args
            raw_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            if key_func:
                raw_key = key_func(*args, **kwargs)

            # 1. Check persistent daily cache (File-based)
            cached_persistent = daily_cache.get_cached(raw_key, ttl_seconds=ttl)
            if cached_persistent is not None:
                # Skip cached errors — force re-execution
                is_error = False
                if isinstance(cached_persistent, dict) and "error" in cached_persistent:
                    is_error = True
                elif isinstance(cached_persistent, str) and cached_persistent.startswith("Error"):
                    is_error = True

                if not is_error:
                    global _hits
                    _hits += 1
                    return cached_persistent

            # 2. Cache miss (or cached error skipped) - execute
            global _misses
            _misses += 1
            result = func(*args, **kwargs)

            # 3. Store results persistently — but NEVER cache errors
            is_result_error = False
            if isinstance(result, dict) and "error" in result:
                is_result_error = True
            elif isinstance(result, str) and result.startswith("Error"):
                is_result_error = True
            # An `unavailable()` payload is a statement about the SOURCE right
            # now — a missing key, an exhausted quota, an outage — not a value.
            # Caching it pins the outage for the full TTL and keeps reporting a
            # dead source after the quota resets, which is the same "stale state
            # presented as current" failure the fetch-time stamp above exists to
            # prevent. Treated exactly like an error: not stored, retried next call.
            elif isinstance(result, dict) and result.get("status") == "unavailable":
                is_result_error = True

            if not is_result_error:
                # Stamp BEFORE storing, so the cached copy carries the fetch
                # time and every later hit replays it (Roadmap 5.8).
                if stamp:
                    result = _stamp_fetch(result)
                daily_cache.set_cached(raw_key, result)
            return result
        return wrapper
    return decorator


# --- PARALLEL FETCHING ---
def fetch_parallel(tasks: dict[str, Callable], timeout: int = 60, max_workers: int = 5) -> dict[str, Any]:
    """
    Execute multiple fetch functions in parallel.

    Args:
        tasks: Dict of {name: callable} where callable takes no args
        timeout: Maximum seconds to wait
        max_workers: Maximum number of concurrent threads

    Returns:
        Dict of {name: result}

    Example:
        results = fetch_parallel({
            'fundamentals': lambda: get_stock_data('AAPL'),
            'technicals': lambda: get_comprehensive_technicals('AAPL'),
            'sentiment': lambda: get_full_sentiment('AAPL')
        })
    """
    results = {}

    import concurrent.futures

    from agent.utils import get_st_aware_func

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_name = {executor.submit(get_st_aware_func(fn)): name for name, fn in tasks.items()}

        try:
            for future in concurrent.futures.as_completed(future_to_name, timeout=timeout):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {"error": str(e)}
        except concurrent.futures.TimeoutError:
            # For any futures that didn't yield before the timeout, mark them as errors
            for future, name in future_to_name.items():
                if name not in results:
                    results[name] = {"error": "TimeoutError"}

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def fetch_stock_data_parallel(symbol: str) -> dict[str, Any]:
    """
    Fetch fundamentals, technicals, and sentiment in parallel for a single stock.
    Results are cached for 5 minutes.
    """
    cache_key = f"stock_parallel:{symbol.upper()}"
    cached_val = get_cached(cache_key)
    if cached_val is not None:
        return cached_val

    from tools.market_data import get_stock_data
    from tools.sentiment_analysis import get_full_sentiment
    from tools.technicals import get_comprehensive_technicals

    tasks = {
        'fundamentals': lambda: get_stock_data(symbol),
        'technicals': lambda: get_comprehensive_technicals(symbol),
        'sentiment': lambda: get_full_sentiment(symbol)
    }

    results = fetch_parallel(tasks)
    # Only cache if at least one sub-result succeeded (no "error" key)
    if any(not (isinstance(v, dict) and "error" in v) for v in results.values()):
        set_cached(cache_key, results)

    return results
