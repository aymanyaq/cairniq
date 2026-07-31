"""`_as_of` stamping inside the cache decorator (Roadmap 5.8 — remainder).

The alert-path slice stamped three producers by hand. The remainder is the other
~95 `@cached` tools, and the point of doing it in the decorator rather than at
95 call sites is not brevity — it is that the decorator is the only place that
can tell a FETCH from a READ.

So the test that matters is not "does a stamp appear". It is: does a cache HIT
replay the original fetch time? A stamp applied on read would restart the clock
on every hit and make an hour-old payload permanently claim to be current, which
is precisely the bug 5.8 exists to close.
"""
from datetime import datetime, timedelta

import pytest

import tools.cache as cache_mod
from tools.freshness import AS_OF_KEY, age_minutes, as_of


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    """A per-test in-memory stand-in for the persistent daily cache."""
    store: dict = {}

    def get_cached(key, ttl_seconds=None):
        return store.get(key)

    def set_cached(key, value):
        store[key] = value

    monkeypatch.setattr(cache_mod.daily_cache, "get_cached", get_cached)
    monkeypatch.setattr(cache_mod.daily_cache, "set_cached", set_cached)
    return store


def test_a_cached_dict_result_carries_a_fetch_stamp():
    calls = []

    @cache_mod.cached()
    def fetch_quote(symbol):
        calls.append(symbol)
        return {"symbol": symbol, "price": 101.5}

    result = fetch_quote("VTI")

    assert as_of(result) is not None
    assert result["price"] == 101.5  # the payload is otherwise untouched


def test_a_cache_hit_replays_the_original_fetch_time(monkeypatch):
    """The whole reason this lives in the decorator. A hit an hour later must
    report an hour-old payload, not a brand-new one."""
    fetched_at = datetime(2026, 7, 27, 9, 0, 0)
    monkeypatch.setattr(cache_mod, "_stamp_fetch", lambda r: {**r, AS_OF_KEY: fetched_at.isoformat()})

    @cache_mod.cached()
    def fetch_quote(symbol):
        return {"symbol": symbol, "price": 101.5}

    fetch_quote("VTI")                      # miss — stamped at 09:00
    later = fetch_quote("VTI")              # hit, much later

    assert as_of(later) == fetched_at
    assert age_minutes(later, now=fetched_at + timedelta(hours=6)) == pytest.approx(360, abs=1)


def test_the_underlying_function_runs_once_and_the_stamp_does_not_change_that():
    calls = []

    @cache_mod.cached()
    def fetch_quote(symbol):
        calls.append(symbol)
        return {"symbol": symbol}

    fetch_quote("VTI")
    fetch_quote("VTI")

    assert calls == ["VTI"]


def test_a_tool_that_stamps_itself_keeps_its_own_more_precise_time():
    """`stamp` is idempotent on purpose: get_stock_data and boc_valet stamp at
    their true fetch moment, which is earlier and better than the decorator's."""
    true_fetch = datetime(2026, 7, 27, 9, 30, 0).isoformat(timespec="seconds")

    @cache_mod.cached()
    def fetch_series():
        return {"series": [1, 2, 3], AS_OF_KEY: true_fetch}

    assert fetch_series()[AS_OF_KEY] == true_fetch


def test_a_non_dict_result_passes_through_untouched():
    """A list has nowhere to carry a stamp. Downstream it reads as *unverified*
    rather than fresh, which is the honest answer — absence of proof is not
    proof of freshness."""

    @cache_mod.cached()
    def fetch_list():
        return [1, 2, 3]

    assert fetch_list() == [1, 2, 3]
    assert as_of(fetch_list()) is None


def test_an_error_result_is_neither_cached_nor_stamped(isolated_cache):
    """A stamped error would look like successfully-fetched evidence."""

    @cache_mod.cached()
    def failing():
        return {"error": "FMP rate limit exceeded"}

    result = failing()

    assert AS_OF_KEY not in result
    assert isolated_cache == {}


def test_a_broken_freshness_module_does_not_take_the_data_down(monkeypatch):
    """A freshness stamp is evidence, not the payload. A caching layer that
    started throwing because it could not annotate would be strictly worse than
    one that returns unstamped data."""
    def boom(_payload):
        raise RuntimeError("freshness unavailable")

    monkeypatch.setattr("tools.freshness.stamp", boom)

    @cache_mod.cached()
    def fetch_quote():
        return {"price": 42.0}

    assert fetch_quote() == {"price": 42.0}
