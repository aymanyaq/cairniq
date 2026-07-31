"""Roadmap 5.5, recorder half.

The measurement these tests protect: there is no shares-outstanding HISTORY to
fetch for a fund from any source we hold, so the series is accrued locally. Two
properties therefore matter more than the arithmetic — a day must never be
silently lost, and a series that is still filling up must never be presented as
a flat one.
"""
import csv
from datetime import date, timedelta

import pytest

from tools import fund_flows
from tools.tool_errors import is_unavailable


def _d(offset_days=0):
    return (date.today() - timedelta(days=offset_days)).isoformat()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect the global store into a tmp file."""
    path = tmp_path / "fund_shares_history.csv"
    monkeypatch.setattr(fund_flows, "history_path", lambda: str(path))
    return path


def _seed(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fund_flows._FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _point(shares, source=fund_flows.SHARES_SOURCE):
    return {"symbol": "X", "shares_outstanding": shares, "source": source, "_as_of": "2026-07-28T17:00:00"}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr("tools.cache.daily_cache.get_cached", lambda *a, **k: None)
    monkeypatch.setattr("tools.cache.daily_cache.set_cached", lambda *a, **k: None)


def test_fetch_parses_the_fmp_row(monkeypatch, no_cache):
    monkeypatch.setattr("tools.fmp_api._fmp_key", lambda: "key")
    monkeypatch.setattr(
        "tools.fmp_api._fmp_get",
        lambda path, params, timeout=5: ([{"symbol": "VIG", "outstandingShares": 2876624738,
                                          "date": "2026-07-28 01:14:10"}], None),
    )
    r = fund_flows.get_shares_outstanding("vig")
    assert r["symbol"] == "VIG"
    assert r["shares_outstanding"] == 2876624738
    # The source is pinned on the row itself; a reader uses it to refuse to
    # difference across a source change.
    assert r["source"] == fund_flows.SHARES_SOURCE
    assert r["source_stamp"] == "2026-07-28 01:14:10"


def test_fetch_without_a_key_is_unavailable_not_zero(monkeypatch, no_cache):
    """A fabricated 0 would read downstream as a total redemption."""
    monkeypatch.setattr("tools.fmp_api._fmp_key", lambda: "")
    r = fund_flows.get_shares_outstanding("VIG")
    assert is_unavailable(r)
    assert "FMP_API_KEY" in r["reason"]


@pytest.mark.parametrize("bad", [float("nan"), None, 0, -5, "n/a"])
def test_unusable_share_counts_never_become_a_data_point(monkeypatch, no_cache, bad):
    monkeypatch.setattr("tools.fmp_api._fmp_key", lambda: "key")
    monkeypatch.setattr("tools.fmp_api._fmp_get",
                        lambda path, params, timeout=5: ([{"outstandingShares": bad}], None))
    assert is_unavailable(fund_flows.get_shares_outstanding("VIG"))


def test_fetch_surfaces_a_transport_error(monkeypatch, no_cache):
    monkeypatch.setattr("tools.fmp_api._fmp_key", lambda: "key")
    monkeypatch.setattr("tools.fmp_api._fmp_get",
                        lambda path, params, timeout=5: (None, "HTTP 402"))
    r = fund_flows.get_shares_outstanding("VIG")
    assert is_unavailable(r) and r["reason"] == "HTTP 402"


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def test_records_one_row_per_fund(store, monkeypatch):
    monkeypatch.setattr(fund_flows, "get_shares_outstanding",
                        lambda s: {**_point(1000.0), "symbol": s})
    report = fund_flows.record_fund_shares(["VIG", "ZQQ.TO"])

    assert report["recorded"] == 2
    rows = fund_flows.read_history()
    assert {r["symbol"] for r in rows} == {"VIG", "ZQQ.TO"}
    assert all(r["source"] == fund_flows.SHARES_SOURCE for r in rows)
    # Stamped at FETCH time, not read time (5.8) — the fetch stamp rides along
    # from the cache layer rather than being invented on write.
    assert all(r["as_of"] == "2026-07-28T17:00:00" for r in rows)


def test_second_run_the_same_day_records_nothing(store, monkeypatch):
    calls = []

    def _fetch(s):
        calls.append(s)
        return {**_point(1000.0), "symbol": s}

    monkeypatch.setattr(fund_flows, "get_shares_outstanding", _fetch)
    fund_flows.record_fund_shares(["VIG"])
    second = fund_flows.record_fund_shares(["VIG"])

    assert second["recorded"] == 0
    assert second["skipped_existing"] == 1
    # Idempotence must be cheap: the source is not called again.
    assert calls == ["VIG"]
    assert len(fund_flows.read_history()) == 1


def test_force_replaces_todays_row_rather_than_duplicating_it(store, monkeypatch):
    monkeypatch.setattr(fund_flows, "get_shares_outstanding",
                        lambda s: {**_point(1000.0), "symbol": s})
    fund_flows.record_fund_shares(["VIG"])
    monkeypatch.setattr(fund_flows, "get_shares_outstanding",
                        lambda s: {**_point(2000.0), "symbol": s})
    fund_flows.record_fund_shares(["VIG"], force=True)

    rows = fund_flows.read_history("VIG")
    assert len(rows) == 1
    assert float(rows[0]["shares_outstanding"]) == 2000.0


def test_a_failed_source_writes_nothing_and_says_which(store, monkeypatch):
    from tools.tool_errors import unavailable

    def _fetch(s):
        if s == "BAD":
            return unavailable(fund_flows.SHARES_SOURCE, "HTTP 402", symbol=s)
        return {**_point(1000.0), "symbol": s}

    monkeypatch.setattr(fund_flows, "get_shares_outstanding", _fetch)
    report = fund_flows.record_fund_shares(["VIG", "BAD"])

    assert report["recorded"] == 1
    assert report["failed"] == 1
    assert report["failures"]["BAD"] == "HTTP 402"
    assert {r["symbol"] for r in fund_flows.read_history()} == {"VIG"}


def test_universe_is_the_union_of_funds_held_and_excludes_equities(store, monkeypatch):
    monkeypatch.setattr(fund_flows, "is_fund", lambda s: {
        "VIG": {"is_fund": True}, "ZQQ.TO": {"is_fund": True}, "AAPL": {"is_fund": False},
    }[s])
    monkeypatch.setattr("tools.user_profile.list_available_profiles",
                        lambda: [{"name": "a"}, {"name": "b"}, {"name": "pytest_x"}, {"name": "_unbound"}])
    holdings = {"a": ["VIG", "AAPL"], "b": ["ZQQ.TO", "AAPL"]}
    monkeypatch.setattr("tools.user_profile.run_under_profile",
                        lambda name, fn, *a, **k: (holdings.get(name, []) if fn.__name__ == "get_tradeable_symbols"
                                                   else fn(*a, **k)))

    universe = fund_flows.collect_fund_universe()
    assert universe["funds"] == ["VIG", "ZQQ.TO"]
    assert universe["non_funds"] == 1  # AAPL, deduped across both profiles
    assert universe["profiles_read"] == 2


def test_an_unresolvable_symbol_is_named_not_dropped(store, monkeypatch):
    from tools.tool_errors import unavailable

    monkeypatch.setattr(fund_flows, "is_fund", lambda s: (
        unavailable("yahoo:quoteType", "no quoteType in response", symbol=s) if s == "WEIRD"
        else {"is_fund": True}))
    monkeypatch.setattr("tools.user_profile.list_available_profiles", lambda: [{"name": "a"}])
    monkeypatch.setattr("tools.user_profile.run_under_profile",
                        lambda name, fn, *a, **k: (["VIG", "WEIRD"] if fn.__name__ == "get_tradeable_symbols"
                                                   else fn(*a, **k)))

    universe = fund_flows.collect_fund_universe()
    assert universe["funds"] == ["VIG"]
    assert universe["unresolved"] == ["WEIRD"]


# ---------------------------------------------------------------------------
# Reading — the honesty half
# ---------------------------------------------------------------------------
def test_no_rows_is_no_data_not_zero_flow(store):
    r = fund_flows.get_flow_series("VIG")
    assert r["status"] == "no_data"
    assert "NOT a zero-flow reading" in r["note"]


def test_one_point_is_accruing_and_reports_no_number(store):
    """The failure mode this guards: drawing 0.0% flows on day 1."""
    _seed(store, [{"date": _d(0), "symbol": "VIG", "shares_outstanding": 1000,
                   "source": fund_flows.SHARES_SOURCE, "as_of": ""}])
    r = fund_flows.get_flow_series("VIG")
    assert r["status"] == "accruing"
    assert r["wow"] is None
    assert r["days_recorded"] == 1


def test_two_points_a_week_apart_are_ready(store):
    _seed(store, [
        {"date": _d(7), "symbol": "VIG", "shares_outstanding": 1000,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
        {"date": _d(0), "symbol": "VIG", "shares_outstanding": 1100,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
    ])
    r = fund_flows.get_flow_series("VIG")
    assert r["status"] == "ready"
    assert r["wow"]["percent_change"] == 10.0
    assert r["wow"]["direction"] == "creations"
    assert r["wow"]["gap_days"] == 7
    assert r["wow"]["stale_window"] is False


def test_redemptions_are_signed(store):
    _seed(store, [
        {"date": _d(7), "symbol": "VIG", "shares_outstanding": 1000,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
        {"date": _d(0), "symbol": "VIG", "shares_outstanding": 900,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
    ])
    wow = fund_flows.get_flow_series("VIG")["wow"]
    assert wow["direction"] == "redemptions"
    assert wow["percent_change"] == -10.0


def test_two_points_two_days_apart_stay_accruing(store):
    """A 2-day gap labelled week-over-week would overstate what was measured."""
    _seed(store, [
        {"date": _d(2), "symbol": "VIG", "shares_outstanding": 1000,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
        {"date": _d(0), "symbol": "VIG", "shares_outstanding": 1100,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
    ])
    r = fund_flows.get_flow_series("VIG")
    assert r["status"] == "accruing"
    assert r["wow"] is None
    assert r["days_until_ready"] == 3


def test_a_series_never_spans_two_sources(store):
    """FMP and Yahoo disagree on SPY by ~15%. Differencing across the switch
    would report that definitional gap as a creation event."""
    _seed(store, [
        {"date": _d(7), "symbol": "SPY", "shares_outstanding": 917782016,
         "source": "yahoo:info", "as_of": ""},
        {"date": _d(0), "symbol": "SPY", "shares_outstanding": 1060327851,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
    ])
    r = fund_flows.get_flow_series("SPY")
    assert r["status"] == "source_change"
    assert r["wow"] is None
    assert "15%" in r["note"]


def test_a_gap_wider_than_a_week_reports_the_flow_and_flags_the_span(store):
    """Missed days must not be hidden: the number is real but the label isn't."""
    _seed(store, [
        {"date": _d(21), "symbol": "VIG", "shares_outstanding": 1000,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
        {"date": _d(0), "symbol": "VIG", "shares_outstanding": 1100,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
    ])
    r = fund_flows.get_flow_series("VIG")
    assert r["status"] == "ready"
    assert r["wow"]["stale_window"] is True
    assert "wider than a week" in r["note"]


def test_a_nan_in_the_store_is_not_a_point(store):
    _seed(store, [
        {"date": _d(7), "symbol": "VIG", "shares_outstanding": "nan",
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
        {"date": _d(0), "symbol": "VIG", "shares_outstanding": 1100,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
    ])
    r = fund_flows.get_flow_series("VIG")
    assert r["days_recorded"] == 1
    assert r["status"] == "accruing"



# ---------------------------------------------------------------------------
# The scheduler task — a missed day cannot be recovered from any source, so the
# gating rules are part of the contract, not an implementation detail.
# ---------------------------------------------------------------------------
@pytest.fixture
def sched_task(monkeypatch):
    """Drive task_fund_shares_record with the clock and profile plumbing stubbed."""
    import asyncio

    from tools import scheduler as sched

    state = {"marked": [], "outcomes": [], "reports": []}

    monkeypatch.setattr(sched, "_is_trading_weekday", lambda dt: True)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: False)
    monkeypatch.setattr(sched, "_mark_done_today", lambda key: state["marked"].append(key))
    monkeypatch.setattr(sched, "_note_engine_outcome",
                        lambda worked, produced, reason, detail="": state["outcomes"].append(
                            {"worked": worked, "produced": produced, "reason": reason, "detail": detail}))
    monkeypatch.setattr("tools.user_profile.run_under_profile", lambda name, fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr("tools.user_profile.list_available_profiles", lambda: [{"name": "a"}])
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: state.get("enabled", True))

    def run(after_close=True, report=None, enabled=True):
        state["enabled"] = enabled
        monkeypatch.setattr(sched, "_after_market_close", lambda dt: after_close)
        if report is not None:
            monkeypatch.setattr("tools.fund_flows.record_fund_shares",
                                lambda *a, **k: state["reports"].append(1) or report)
        asyncio.run(sched.task_fund_shares_record())
        return state

    return run


def test_task_declines_before_the_close(sched_task):
    state = sched_task(after_close=False, report={"recorded": 1, "universe": 1})
    assert state["reports"] == []          # the source was never called
    assert state["marked"] == []
    assert state["outcomes"][0]["reason"] == "before market close"


def test_task_declines_when_no_profile_wants_background_work(sched_task):
    """SCHEDULER_ENABLED is off by default; a user who left it off has not asked
    us to poll a vendor every evening. Matches task_funnel_signal_scan."""
    state = sched_task(report={"recorded": 1, "universe": 1}, enabled=False)
    assert state["reports"] == []
    assert state["marked"] == []
    assert state["outcomes"][0]["reason"] == "scheduler disabled on every profile"


def test_task_marks_the_day_done_once_rows_land(sched_task):
    state = sched_task(report={"recorded": 16, "universe": 16, "failed": 0, "unresolved": []})
    assert state["marked"] == ["fund_shares_recorded"]
    # 2.6: production is rows recorded — the count that proves the whole chain.
    assert state["outcomes"][0]["produced"] == 16
    assert state["outcomes"][0]["worked"] == 1


def test_a_totally_failed_sweep_does_not_burn_the_day(sched_task):
    """The marker is what stops the retry. Setting it on a run that wrote nothing
    would write off the whole day, and no vendor sells the history back."""
    state = sched_task(report={"recorded": 0, "universe": 16, "failed": 16,
                              "failures": {}, "unresolved": []})
    assert state["marked"] == []
    # Ran and produced nothing: an idle streak SHOULD accrue here.
    assert state["outcomes"][0] == {"worked": 1, "produced": 0, "reason": "",
                                    "detail": "0/16 funds recorded, 16 failed"}


def test_series_is_scoped_to_the_symbol_asked_for(store):
    _seed(store, [
        {"date": _d(7), "symbol": "VIG", "shares_outstanding": 1000,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
        {"date": _d(0), "symbol": "ZQQ.TO", "shares_outstanding": 500,
         "source": fund_flows.SHARES_SOURCE, "as_of": ""},
    ])
    assert fund_flows.get_flow_series("VIG")["days_recorded"] == 1
    assert fund_flows.get_flow_series("ZQQ.TO")["latest_shares"] == 500.0
