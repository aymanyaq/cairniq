"""Earnings-proximity flag: CALENDAR-day arithmetic.

The earnings date parses at midnight, so subtracting a mid-day ``datetime.now()``
and taking ``.days`` truncated toward zero: every count read one day low, and on
the earnings day itself the difference went NEGATIVE, failing the ``0 <= days``
gate and dropping the flag entirely on the single riskiest day of the cycle.

These assertions are clock-independent — each case pins the earnings date a fixed
number of calendar days from today, so the expected count is exact whatever time
of day the suite runs.
"""
from datetime import date, timedelta

import pytest

import tools.opportunity_scanner as opp


class _FakeTicker:
    def __init__(self, earnings_on: date):
        self.calendar = {"Earnings Date": [earnings_on]}


@pytest.fixture
def _only_earnings(monkeypatch):
    """Silence the insider/short and management-tone probes (network)."""
    import tools.earnings_nlp
    import tools.insider_data

    monkeypatch.setattr(tools.insider_data, "get_insider_and_short_data", lambda s: {})
    monkeypatch.setattr(tools.earnings_nlp, "analyze_management_tone", lambda s: {})


@pytest.mark.parametrize("days_out", [0, 1, 2, 3, 7])
def test_earnings_proximity_counts_calendar_days(monkeypatch, _only_earnings, days_out):
    target = date.today() + timedelta(days=days_out)
    monkeypatch.setattr(opp, "yf", type("YF", (), {"Ticker": lambda self_or_sym: _FakeTicker(target)}))

    assert opp._headwind_check("QRS").get("days_to_earnings") == days_out


def test_earnings_day_itself_still_flags(monkeypatch, _only_earnings):
    """days_to == -1 on the morning of the report silently un-flagged the
    highest-risk day; the scanner reported no event risk at all."""
    monkeypatch.setattr(
        opp, "yf", type("YF", (), {"Ticker": lambda self_or_sym: _FakeTicker(date.today())})
    )

    result = opp._headwind_check("QRS")

    assert result.get("days_to_earnings") == 0


@pytest.mark.parametrize("days_out", [8, 11, 30])
def test_earnings_beyond_the_window_are_not_flagged(monkeypatch, _only_earnings, days_out):
    """The widened arithmetic must not widen the 7-day window itself — a name
    reporting 11 days out is correctly silent, not a coverage gap."""
    target = date.today() + timedelta(days=days_out)
    monkeypatch.setattr(opp, "yf", type("YF", (), {"Ticker": lambda self_or_sym: _FakeTicker(target)}))

    assert "days_to_earnings" not in opp._headwind_check("DEF.TO")


def test_past_earnings_are_not_flagged(monkeypatch, _only_earnings):
    target = date.today() - timedelta(days=1)
    monkeypatch.setattr(opp, "yf", type("YF", (), {"Ticker": lambda self_or_sym: _FakeTicker(target)}))

    assert "days_to_earnings" not in opp._headwind_check("QRS")
