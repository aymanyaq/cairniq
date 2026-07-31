"""Holdings event radar (Advisor Roadmap 3.5).

The calendar tools shipped weeks ago and nothing merged them against the actual
book, so the dates existed and nobody was ever told. These tests are weighted
toward the two ways a calendar engine goes wrong: inventing a date it does not
have, and turning into a daily countdown nobody reads by the time it matters.

All sources are injected — this is an offline suite by construction, not by
mocking the network out from under a live call.
"""
from datetime import date, timedelta

import pytest

import tools.event_radar as radar

# The day every fixture is dated against.
_TODAY = date(2026, 7, 25)


def _earnings(days=None, ex_div_days=None, error=None):
    """A get_earnings_info-shaped payload, dated `days` out from `_TODAY`.

    The date is DERIVED from `days` here. This fixture used to pin "2026-08-01" for
    every value of `days` — a date and a countdown that contradicted each other,
    which is precisely the defect the engine then shipped to the panel. A fixture
    that models the bug as normal cannot catch it.
    """
    if error:
        return {"error": error}
    out = {
        "next_earnings_date": "Not available",
        "days_until_earnings": None,
        "ex_dividend_date": None,
        "days_until_ex_dividend": None,
    }
    if days is not None:
        out["next_earnings_date"] = (_TODAY + timedelta(days=days)).isoformat()
        out["days_until_earnings"] = days
    if ex_div_days is not None:
        out["ex_dividend_date"] = (_TODAY + timedelta(days=ex_div_days)).isoformat()
        out["days_until_ex_dividend"] = ex_div_days
    return out


def _build(holdings, per_symbol, fomc=None, today=_TODAY):
    return radar.build_event_radar(
        holdings_fn=lambda: holdings,
        earnings_fn=lambda s: per_symbol.get(s, _earnings()),
        fomc_fn=lambda: fomc or {},
        today=today,
    )


# ---------------------------------------------------------------------------
# A missing date is silence, never a guess
# ---------------------------------------------------------------------------

def test_a_symbol_with_no_earnings_date_produces_no_event():
    """Inferring 'about a quarter after the last one' would put a fabricated
    date in front of someone sizing a position."""
    result = _build(["NODATE"], {"NODATE": _earnings()})

    assert result["events"] == []
    assert result["unknown"] == ["NODATE"]


def test_undated_symbols_are_surfaced_not_silently_dropped():
    """'No earnings coming' and 'the provider did not tell us' are different
    facts, and only one of them is safe to act on."""
    result = _build(
        ["GOOD", "BROKEN"],
        {"GOOD": _earnings(days=10), "BROKEN": _earnings(error="404 from provider")},
    )

    assert [e["symbol"] for e in result["events"]] == ["GOOD"]
    assert result["unknown"] == ["BROKEN"]


@pytest.mark.parametrize("junk", ["Not available", "N/A", "unknown", "", "None"])
def test_provider_refusals_are_not_treated_as_dates(junk):
    payload = {"next_earnings_date": junk, "days_until_earnings": 3}
    result = _build(["X"], {"X": payload})

    assert result["events"] == []


def test_one_bad_symbol_never_aborts_the_sweep():
    def _explode(symbol):
        if symbol == "BOOM":
            raise RuntimeError("provider exploded")
        return _earnings(days=5)

    result = radar.build_event_radar(
        holdings_fn=lambda: ["AAA", "BOOM", "ZZZ"],
        earnings_fn=_explode,
        fomc_fn=lambda: {},
        today=date(2026, 7, 25),
    )

    assert {e["symbol"] for e in result["events"]} == {"AAA", "ZZZ"}
    assert result["unknown"] == ["BOOM"]
    assert result["checked"] == 3


# ---------------------------------------------------------------------------
# T-3 / T-1 only, each firing once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("days,fires", [
    (10, False), (4, False), (3, True), (2, False), (1, True), (0, False),
])
def test_only_t3_and_t1_fire(days, fires):
    """Not a daily countdown. 'Earnings in 9 days... 8 days... 7 days' trains the
    reader to skip it, and by T-1 it is wallpaper."""
    result = _build(["X"], {"X": _earnings(days=days)})

    assert bool(radar.due_alerts(result)) is fires


def test_t3_and_t1_are_distinct_alerts_for_the_same_event():
    """They must not collapse into one dedup key, or the T-1 warning — the one
    that actually matters — would be swallowed as a duplicate of T-3."""
    t3 = radar.due_alerts(_build(["X"], {"X": _earnings(days=3)}))[0]
    t1 = radar.due_alerts(_build(["X"], {"X": _earnings(days=1)}))[0]

    assert t3["dedup_key"] != t1["dedup_key"]
    assert t3["severity"] == "info" and t1["severity"] == "warning"


def test_a_rescheduled_event_is_a_new_alert_not_a_suppressed_one():
    """The dedup key pins the event DATE. If a company moves its print, the old
    key must not silently swallow the new warning.

    Each radar is built at its own T-3, because that is the day each one fires: a
    moved date fires on a different day, and the two keys must still differ. The
    moved payload carries no countdown field at all — the engine does not need one.
    """
    original = radar.due_alerts(_build(["X"], {"X": _earnings(days=3)}))[0]

    moved_to = date(2026, 8, 8)
    rescheduled = radar.due_alerts(_build(
        ["X"],
        {"X": {"next_earnings_date": moved_to.isoformat()}},
        today=moved_to - timedelta(days=3),
    ))[0]

    assert original["dedup_key"] != rescheduled["dedup_key"]


def test_ex_dividend_fires_only_at_t1():
    """A three-day warning on a dividend is noise for a decade-horizon holder;
    the actionable fact is 'own it before tomorrow'."""
    at_t3 = radar.due_alerts(_build(["X"], {"X": _earnings(ex_div_days=3)}))
    at_t1 = radar.due_alerts(_build(["X"], {"X": _earnings(ex_div_days=1)}))

    assert at_t3 == []
    assert len(at_t1) == 1
    assert "ex-dividend" in at_t1[0]["title"]


def test_fomc_rides_the_same_rail_without_a_symbol():
    result = _build([], {}, fomc={"upcoming_meetings": [
        {"date": "2026-07-28", "days_until": 3},
        {"date": "2026-09-16", "days_until": 53},
    ]})

    alerts = radar.due_alerts(result)
    assert len(alerts) == 1
    assert alerts[0]["data"]["symbol"] is None
    assert "FOMC" in alerts[0]["title"]


def test_a_broken_fomc_lookup_does_not_cost_the_holdings_half():
    def _explode():
        raise RuntimeError("fed calendar down")

    result = radar.build_event_radar(
        holdings_fn=lambda: ["X"],
        earnings_fn=lambda s: _earnings(days=3),
        fomc_fn=_explode,
        today=date(2026, 7, 25),
    )

    assert len(result["events"]) == 1


# ---------------------------------------------------------------------------
# The countdown is derived from the date, never read alongside it
# ---------------------------------------------------------------------------

def test_the_countdown_is_derived_from_the_date_not_the_providers_own_field():
    """Observed live 2026-07-29: the panel showed AAPL and AMZN dated 2026-07-30 —
    tomorrow — and captioned both "Today".

    Both fields came from get_earnings_info and nothing compared them.
    `days_until_earnings` was `(event_datetime - datetime.now()).days`, a truncated
    timedelta, so a print dated tomorrow at 06:00 read as 0 from any moment after
    06:00 today. The DATE is authoritative: it is the provider's own timestamp
    rendered to a day, and the countdown was arithmetic layered on top of it.
    """
    provider = {"next_earnings_date": "2026-07-26", "days_until_earnings": 0}

    (event,) = _build(["AAPL"], {"AAPL": provider})["events"]

    assert event["date"] == "2026-07-26"
    assert event["days_until"] == 1, "0 renders as 'Today' beside tomorrow's date"


@pytest.mark.parametrize("date_says,provider_says,fires", [
    (3, 0, True),    # the observed drift, at T-3: must still fire
    (1, 0, True),    # ...and at T-1, the one that actually matters
    (7, 3, False),   # a drifting countdown must not fire a week early
    (3, 4, True),    # ...nor suppress the alert that is genuinely due
    (2, 1, False),
])
def test_alerts_fire_on_the_day_the_date_says(date_says, provider_says, fires):
    """The serious half. due_alerts() gates on `days_until in (3, 1)`, so a one-day
    drift fired T-3 and T-1 notifications on the wrong day or skipped them outright
    — a warning about tomorrow's print arriving the day after it printed.
    """
    provider = {
        "next_earnings_date": (_TODAY + timedelta(days=date_says)).isoformat(),
        "days_until_earnings": provider_says,
    }

    alerts = radar.due_alerts(_build(["X"], {"X": provider}))

    assert bool(alerts) is fires
    if fires:
        assert alerts[0]["data"]["days_until"] == date_says
        assert alerts[0]["dedup_key"].endswith(f"T{date_says}")


def test_the_fomc_countdown_is_derived_too():
    """get_fomc_calendar built a midnight datetime and subtracted datetime.now(),
    so every meeting read one day nearer than it was — and the radar put that
    number on the same T-3/T-1 rail without checking it against the date.
    """
    result = _build([], {}, fomc={"upcoming_meetings": [
        {"date": "2026-07-28", "days_until": 2},  # provider one day short
    ]})

    (event,) = result["events"]
    assert event["days_until"] == 3
    assert radar.due_alerts(result)[0]["dedup_key"].endswith("2026-07-28:T3")


def test_the_ex_dividend_countdown_is_derived_too():
    """Ex-dividend is T-1 only, so a one-day drift is the whole warning: the alert
    would land on a day when the shares can no longer be bought in time."""
    result = _build(["X"], {"X": {
        "ex_dividend_date": "2026-07-26", "days_until_ex_dividend": 0,
    }})

    (event,) = [e for e in result["events"] if e["kind"] == "ex_dividend"]
    assert event["days_until"] == 1
    assert len(radar.due_alerts(result)) == 1


def test_a_date_that_will_not_parse_was_never_established():
    """_is_usable only rejects the refusal strings a provider is known to send, so
    "Q3 2026" passed it and travelled into the panel verbatim, carrying whatever
    countdown happened to arrive beside it. Parsing is the stronger test."""
    result = _build(["X"], {"X": {"next_earnings_date": "Q3 2026", "days_until_earnings": 3}})

    assert result["events"] == []
    assert result["unknown"] == ["X"]


def test_a_provider_date_already_in_the_past_is_not_an_event():
    """The radar looks forward. A stale date behind `today` is not an event, and it
    is not a countdown of 3 either just because the provider still says so."""
    result = _build(["X"], {"X": {"next_earnings_date": "2026-07-01", "days_until_earnings": 3}})

    assert result["events"] == []
    assert result["unknown"] == ["X"]


def test_an_event_dated_today_is_zero_days_out_and_does_not_fire():
    """"Today" stays reachable and now means it. The offsets are T-3 and T-1: by the
    morning of the print there is nothing left to warn anyone about."""
    result = _build(["X"], {"X": {"next_earnings_date": _TODAY.isoformat()}})

    (event,) = result["events"]
    assert event["days_until"] == 0
    assert radar.due_alerts(result) == []


def test_a_full_iso_timestamp_still_yields_a_day():
    """A provider that starts returning timestamps instead of dates must not
    silently empty the radar — take the date half rather than reject the row."""
    (event,) = _build(["X"], {"X": {"next_earnings_date": "2026-07-28T06:00:00"}})["events"]

    assert (event["date"], event["days_until"]) == ("2026-07-28", 3)


# ---------------------------------------------------------------------------
# It reports a calendar; it does not tell anyone to trade
# ---------------------------------------------------------------------------

def test_the_alert_states_what_the_date_means_without_recommending_a_trade():
    """Sizing and action belong to the advisor's gated path (2.2 pre-check), not
    to a zero-LLM date sweep that has never seen the position."""
    alert = radar.due_alerts(_build(["X"], {"X": _earnings(days=1)}))[0]
    body = alert["message"].lower()

    assert "volatility event" in body
    for verb in ("sell ", "buy ", "trim ", "reduce your", "we recommend"):
        assert verb not in body, f"the radar recommended an action: {verb!r}"


def test_events_are_sorted_nearest_first():
    result = _build(
        ["FAR", "NEAR"],
        {"FAR": _earnings(days=40), "NEAR": _earnings(days=2)},
    )

    assert [e["symbol"] for e in result["events"]] == ["NEAR", "FAR"]


# ---------------------------------------------------------------------------
# The tick, and what it reports to the 2.6 heartbeat
# ---------------------------------------------------------------------------

def test_the_tick_counts_names_swept_not_alerts_raised():
    """A book with nothing inside three days is the normal state for most of the
    year. Counting alerts would report this engine dead through every quiet
    stretch — the exact mistake 2.6 exists to prevent."""
    quiet = _build(["A", "B", "C"], {s: _earnings(days=40) for s in ("A", "B", "C")})

    result = radar.run_event_radar_tick(radar=quiet, raise_fn=lambda **kw: None)

    assert result["checked"] == 3
    assert result["alerts"] == 0


def test_a_delivery_failure_does_not_lose_the_remaining_alerts():
    fired = []

    def _flaky(**kw):
        if "AAA" in kw["title"]:
            raise RuntimeError("inbox unavailable")
        fired.append(kw["title"])

    due = _build(["AAA", "BBB"], {s: _earnings(days=1) for s in ("AAA", "BBB")})
    result = radar.run_event_radar_tick(radar=due, raise_fn=_flaky)

    assert result["alerts"] == 1
    assert len(fired) == 1


def test_the_scheduler_registers_it_and_it_declares_production():
    """3.5 is the first engine added since 2.6's coverage guard. If the guard
    works, a new task cannot ship without declaring what it produced."""
    import inspect

    import tools.scheduler as sched

    registry = dict((n, f) for n, f, _c, _t in sched.SCHEDULED_TASKS)
    assert "event_radar" in registry
    assert "event_radar" in sched.DEFAULT_SCHEDULER_SETTINGS

    source = inspect.getsource(registry["event_radar"])
    assert "_note_engine_outcome" in source or "note_production" in source
