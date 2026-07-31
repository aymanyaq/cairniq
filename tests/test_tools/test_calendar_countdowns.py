"""Calendar-days arithmetic in the two date providers behind the event radar.

Both `get_earnings_info` and `get_fomc_calendar` publish a date and a days-until as
separate fields, and both used to compute the second by subtracting
`datetime.now()` from a datetime. `timedelta.days` truncates, so that subtraction
loses a day whenever the event's time of day falls earlier than the current one —
and the two published fields then contradict each other. Observed live 2026-07-29
on the 3.5b panel: AAPL and AMZN dated 2026-07-30 and captioned "Today".

tools/event_radar.py now derives its own countdown and ignores theirs, so the panel
and the T-3/T-1 alerts are safe regardless. These fields are read directly too —
the agent calls get_earnings_info, and tools/macro_strategy.py reads the FOMC
countdown — so the arithmetic is fixed at the source as well, and pinned here.

Offline by construction: the provider is injected via monkeypatch, not mocked out
from under a live call.
"""
from datetime import date

import pandas as pd

import tools.earnings_calendar as earnings
import tools.fed_calendar as fed


def _frozen(day):
    """A `date` whose today() is pinned, for patching into a module's namespace.

    A subclass rather than a stub, because both modules also CONSTRUCT dates
    (`date(year, month, day)`) and subtract them.
    """
    class _Frozen(date):
        @classmethod
        def today(cls):
            return day

    return _Frozen


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

def _ticker(*stamps):
    class _Ticker:
        info = {"forwardEps": 7.5, "trailingEps": 6.5}
        earnings_dates = pd.DataFrame(index=list(stamps))
        earnings_history = None

    return _Ticker


def test_earnings_countdown_is_calendar_days_not_a_truncated_timedelta(monkeypatch):
    """The reported bug, at its source.

    A premarket print TOMORROW read as 0 days from any moment after 06:00 today,
    because `(event - datetime.now()).days` threw away the rest of today.
    """
    monkeypatch.setattr(
        earnings.yf, "Ticker",
        lambda symbol: _ticker(pd.Timestamp("2026-07-30 06:00:00", tz="America/New_York")),
    )
    monkeypatch.setattr(earnings, "date", _frozen(date(2026, 7, 29)))

    out = earnings.get_earnings_info.__wrapped__("AAPL")

    assert out["next_earnings_date"] == "2026-07-30"
    assert out["days_until_earnings"] == 1
    assert out["earnings_warning"].startswith("⚠️")


def test_a_print_scheduled_for_today_stays_visible_all_day(monkeypatch):
    """Selecting on instants (`> datetime.now()`) dropped an event the moment its
    own timestamp passed, silently advancing to next quarter's date. On the morning
    of a print the radar would have shown a date three months out."""
    monkeypatch.setattr(
        earnings.yf, "Ticker",
        lambda symbol: _ticker(
            pd.Timestamp("2026-07-29 06:00:00", tz="America/New_York"),
            pd.Timestamp("2026-10-29 06:00:00", tz="America/New_York"),
        ),
    )
    monkeypatch.setattr(earnings, "date", _frozen(date(2026, 7, 29)))

    out = earnings.get_earnings_info.__wrapped__("AAPL")

    assert out["next_earnings_date"] == "2026-07-29"
    assert out["days_until_earnings"] == 0


def test_a_naive_earnings_index_does_not_raise(monkeypatch):
    """yfinance normally returns a tz-aware index, and the old code called
    tz_localize(None) unconditionally on the strength of that. A naive index must
    still produce a date rather than falling into the bare-except error return."""
    monkeypatch.setattr(
        earnings.yf, "Ticker", lambda symbol: _ticker(pd.Timestamp("2026-08-05 16:30:00")),
    )
    monkeypatch.setattr(earnings, "date", _frozen(date(2026, 7, 29)))

    out = earnings.get_earnings_info.__wrapped__("AAPL")

    assert "error" not in out
    assert (out["next_earnings_date"], out["days_until_earnings"]) == ("2026-08-05", 7)


# ---------------------------------------------------------------------------
# FOMC
# ---------------------------------------------------------------------------

def test_every_fomc_countdown_matches_its_own_date():
    """The invariant the old arithmetic broke for every single meeting: subtracting
    a real `now` from a midnight datetime left every countdown one day short.

    Deliberately NOT frozen. This is the one check here that runs against the real
    clock, so it fails for the actual reason rather than because a patch target
    moved — and the invariant holds on any day, at any time of day, which is exactly
    what "calendar days" means.
    """
    today = date.today()
    cal = fed.get_fomc_calendar.__wrapped__(num_meetings=6)
    if cal.get("error"):
        # FOMC_SCHEDULE is a hardcoded table; when it runs out this stops being a
        # statement about arithmetic. The frozen tests below keep covering it.
        return

    for meeting in cal["upcoming_meetings"]:
        expected = (date.fromisoformat(meeting["date"]) - today).days
        assert meeting["days_until"] == expected, meeting["date"]
        assert meeting["days_until"] >= 0, meeting["date"]


def test_a_meeting_scheduled_for_today_is_not_dropped(monkeypatch):
    """2026-07-29 is a scheduled FOMC day. Under `meeting_date > datetime.now()` its
    midnight was already in the past, so on the day of a rate decision this tool
    reported the September meeting as the next one — 48 days out, itself one short.
    """
    monkeypatch.setattr(fed, "date", _frozen(date(2026, 7, 29)))

    cal = fed.get_fomc_calendar.__wrapped__(num_meetings=2)

    assert cal["next_meeting"]["date"] == "2026-07-29"
    assert cal["next_meeting"]["days_until"] == 0
    assert cal["upcoming_meetings"][1]["days_until"] == 49
