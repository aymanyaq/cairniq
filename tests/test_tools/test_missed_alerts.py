"""Roadmap 7.1 number 3 — the missed-crossing replay.

7.1's standing rule is that a figure quoted without its caveat is the failure
this whole theme keeps rediscovering, and this module is the easiest place in the
codebase to break it: every unmeasurable window renders as a zero unless
something stops it. So the tests below spend most of their effort on the four
states that are NOT a count.

Two of them are the subtle ones and neither is about missing data:

  * A bar interval wider than the gap cannot see inside it. Replaying a 4-hour
    outage against DAILY bars measures a different window, and reporting 0 from
    it would be a clean record manufactured out of a resolution limit.
  * A condition never evaluated before the outage would have VOIDED, not fired —
    `evaluate_conditions` voids a trigger that is already true on its first check
    because it is noise rather than news. Counting those as misses would inflate
    the figure with the cheapest possible false positives.
"""

from datetime import datetime, timedelta

import pytest

from tools.missed_alerts import (
    armed_during,
    get_missed_alerts,
    interval_for,
    replay_gap,
)

GAP_START = datetime(2026, 7, 10, 9, 35)
GAP_END = datetime(2026, 7, 10, 13, 55)


def _gap(start=GAP_START, end=GAP_END, lost=260.0):
    return {"start": start.isoformat(timespec="seconds"),
            "end": end.isoformat(timespec="seconds"),
            "hours": round((end - start).total_seconds() / 3600, 2),
            "window_minutes_lost": lost}


def _condition(symbol="SPY", metric="price", operator="<=", threshold=500.0,
               created="2026-07-01T09:00:00", checked="2026-07-10T09:30:00",
               **kw):
    base = {"id": "wc_test", "symbol": symbol, "metric": metric,
            "operator": operator, "threshold": threshold, "label": "test trigger",
            "status": "active", "created_at": created, "checked_at": checked,
            "expires_at": "2026-08-01T09:00:00"}
    base.update(kw)
    return base


def _bars(values, start=GAP_START):
    """Hourly bars whose high and low both sit at each supplied level."""
    return [{"at": (start + timedelta(hours=i)).isoformat(timespec="seconds"),
             "high": v, "low": v, "close": v}
            for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# Who was armed
# ---------------------------------------------------------------------------
def test_a_condition_created_after_the_gap_was_not_armed_during_it():
    assert armed_during(_condition(created="2026-07-20T09:00:00"),
                        GAP_START, GAP_END) is False


def test_a_condition_resolved_before_the_gap_was_not_armed_during_it():
    assert armed_during(_condition(resolved_at="2026-07-01T10:00:00"),
                        GAP_START, GAP_END) is False


def test_a_condition_that_has_since_fired_was_still_armed_back_then():
    """Reading only ACTIVE conditions would shrink three-week-old exposure to
    whatever happens to have survived until today."""
    assert armed_during(_condition(status="fired",
                                   resolved_at="2026-07-25T10:00:00"),
                        GAP_START, GAP_END) is True


def test_an_undated_condition_is_not_assumed_to_have_been_armed():
    """It cannot be placed in time, and assuming it was live manufactures
    exposure out of a missing field."""
    assert armed_during(_condition(created=None), GAP_START, GAP_END) is False


def test_an_expired_condition_is_not_armed_after_its_expiry():
    assert armed_during(_condition(expires_at="2026-07-05T09:00:00"),
                        GAP_START, GAP_END) is False


# ---------------------------------------------------------------------------
# The four non-counts
# ---------------------------------------------------------------------------
def test_a_gap_outside_the_coverage_window_is_a_real_zero():
    res = replay_gap(_gap(lost=0.0), [_condition()])
    assert res["status"] == "outside_window"
    assert res["missed"] == 0


def test_a_gap_with_nothing_armed_is_a_real_zero_and_says_what_it_does_not_mean():
    res = replay_gap(_gap(), [])
    assert res["status"] == "no_conditions_armed"
    assert res["missed"] == 0
    assert "says nothing about whether the market moved" in res["note"]


def test_a_bar_wider_than_the_gap_is_unmeasurable_not_clean():
    """THE test for this module.

    A four-hour outage a year ago can only be replayed against daily bars. That
    is not a coarse measurement of the right answer — it is a measurement of a
    different window, and `missed: 0` from it would be a clean record made out of
    a provider limitation.
    """
    # Beyond ~2 years the provider serves DAILY bars and nothing finer, so a
    # four-hour window inside one of them is invisible by construction.
    old_start = datetime(2024, 1, 10, 9, 35)
    res = replay_gap(_gap(old_start, old_start + timedelta(hours=4)),
                     [_condition(created="2023-01-01T00:00:00",
                                 expires_at="2026-08-01T00:00:00")],
                     now=datetime(2026, 7, 30))
    assert res["status"] == "resolution_too_coarse"
    assert res["missed"] is None
    assert res["unmeasurable"] == 1
    assert "not the same as clean" in res["note"]


def test_a_symbol_with_no_history_is_unmeasurable_rather_than_uncrossed():
    res = replay_gap(_gap(), [_condition()],
                     bars_fn=lambda *a: [], now=datetime(2026, 7, 12))
    assert res["status"] == "measured"
    assert res["missed"] == 0
    assert res["unmeasurable"] == 1
    assert res["unmeasurable_conditions"][0]["symbol"] == "SPY"


# ---------------------------------------------------------------------------
# The count itself
# ---------------------------------------------------------------------------
def test_a_crossing_inside_the_outage_is_counted_as_a_miss():
    res = replay_gap(
        _gap(), [_condition(operator="<=", threshold=500.0)],
        bars_fn=lambda *a: _bars([510.0, 505.0, 498.0, 502.0]),
        now=datetime(2026, 7, 12),
    )
    assert res["status"] == "measured"
    assert res["missed"] == 1
    assert res["crossings"][0]["outcome"] == "would_have_fired"


def test_no_crossing_is_a_measured_zero_when_the_bars_actually_cover_the_gap():
    res = replay_gap(
        _gap(), [_condition(operator="<=", threshold=400.0)],
        bars_fn=lambda *a: _bars([510.0, 505.0, 498.0, 502.0]),
        now=datetime(2026, 7, 12),
    )
    assert res["missed"] == 0
    assert res["unmeasurable"] == 0


def test_a_condition_never_evaluated_before_the_gap_would_have_voided_not_fired():
    """`evaluate_conditions` voids a trigger satisfied on its FIRST check — it was
    already true when written, which is noise rather than news. Counting that as
    a missed alert is the cheapest possible false positive."""
    res = replay_gap(
        _gap(), [_condition(operator="<=", threshold=500.0, checked=None)],
        bars_fn=lambda *a: _bars([498.0, 497.0]),
        now=datetime(2026, 7, 12),
    )
    assert res["missed"] == 0
    assert res["would_have_voided"] == 1
    assert res["crossings"][0]["outcome"] == "would_have_voided"


def test_the_crossing_is_read_from_the_bar_range_not_only_its_close():
    """A threshold exists to catch the spike. Replaying closes only misses
    precisely the move the trigger was written for."""
    spike = [{"at": GAP_START.isoformat(timespec="seconds"),
              "high": 510.0, "low": 495.0, "close": 508.0},
             {"at": (GAP_START + timedelta(hours=1)).isoformat(timespec="seconds"),
              "high": 509.0, "low": 507.0, "close": 508.0}]
    res = replay_gap(_gap(), [_condition(operator="<=", threshold=500.0)],
                     bars_fn=lambda *a: spike, now=datetime(2026, 7, 12))
    assert res["missed"] == 1
    assert res["bound"] == "upper"


def test_pct_change_without_a_previous_close_is_unmeasurable_not_zero():
    """`pct_change` is measured against the previous close. Substituting the
    window's own opening price would silently redefine the metric mid-replay."""
    res = replay_gap(
        _gap(), [_condition(metric="pct_change", operator="<=", threshold=-3.0)],
        bars_fn=lambda *a: _bars([500.0, 480.0]),
        prev_close_fn=lambda *a: None,
        now=datetime(2026, 7, 12),
    )
    assert res["missed"] == 0
    assert res["unmeasurable"] == 1
    assert "previous close" in res["unmeasurable_conditions"][0]["reason"]


def test_pct_change_replays_against_the_previous_close_when_there_is_one():
    res = replay_gap(
        _gap(), [_condition(metric="pct_change", operator="<=", threshold=-3.0)],
        bars_fn=lambda *a: _bars([500.0, 480.0]),
        prev_close_fn=lambda *a: 500.0,
        now=datetime(2026, 7, 12),
    )
    assert res["missed"] == 1


# ---------------------------------------------------------------------------
# Interval selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("age_days,expected", [
    (1, "1m"), (7, "1m"), (30, "5m"), (200, "1h"), (1000, "1d"),
])
def test_the_interval_follows_what_the_provider_will_still_serve(age_days, expected):
    """Requesting 1m data for a 40-day-old window returns EMPTY rather than an
    error, which would land as `no_data` and read as nothing to see."""
    now = datetime(2026, 7, 30, 12, 0)
    assert interval_for(now - timedelta(days=age_days), now) == expected


# ---------------------------------------------------------------------------
# The read surface
# ---------------------------------------------------------------------------
def test_an_absent_availability_record_is_not_a_report_that_nothing_was_missed(monkeypatch):
    import tools.missed_alerts as ma

    monkeypatch.setattr("tools.availability.measure_availability",
                        lambda *a, **k: {"status": "no_data"})
    res = ma.get_missed_alerts(conditions=[])
    assert res["status"] == "no_availability_data"
    assert "missed_crossings" not in res


def test_the_summary_distinguishes_no_windows_from_no_measurable_windows(monkeypatch):
    import tools.missed_alerts as ma

    monkeypatch.setattr("tools.availability.measure_availability", lambda *a, **k: {
        "status": "measured", "window_minutes_lost": 260.0, "span_days": 31.9,
        "gaps": [_gap(datetime(2024, 1, 10, 9, 35),
                      datetime(2024, 1, 10, 13, 55))],
    })
    res = ma.get_missed_alerts(conditions=[_condition(created="2023-01-01T00:00:00")],
                               now=datetime(2026, 7, 30))
    assert res["status"] == "unmeasurable"
    assert res["measurable_windows"] == 0
    assert res["unmeasurable_windows"] == 1
    assert "not a clean one" in res["summary"]


def test_a_profile_that_never_armed_anything_is_told_so_plainly(monkeypatch):
    import tools.missed_alerts as ma

    monkeypatch.setattr("tools.availability.measure_availability", lambda *a, **k: {
        "status": "measured", "window_minutes_lost": 260.0, "span_days": 31.9,
        "gaps": [_gap()],
    })
    res = ma.get_missed_alerts(conditions=[])
    assert res["conditions_considered"] == 0
    assert "never armed a watch condition" in res["summary"]


def test_the_total_counts_only_measurable_windows_and_says_the_rest_are_a_floor(monkeypatch):
    import tools.missed_alerts as ma

    monkeypatch.setattr("tools.availability.measure_availability", lambda *a, **k: {
        "status": "measured", "window_minutes_lost": 520.0, "span_days": 31.9,
        "gaps": [
            _gap(),                                                   # replayable
            _gap(datetime(2024, 1, 10, 9, 35), datetime(2024, 1, 10, 13, 55)),
        ],
    })
    res = ma.get_missed_alerts(
        conditions=[_condition(created="2023-01-01T00:00:00")],
        bars_fn=lambda *a: _bars([510.0, 498.0]),
        now=datetime(2026, 7, 12),
    )
    assert res["missed_crossings"] == 1
    assert res["measurable_windows"] == 1
    assert res["unmeasurable_windows"] == 1
    assert "at least this" in res["summary"]


def test_availability_points_at_this_surface_rather_than_folding_it_in():
    """Same call 7.1 made for delivery latency: a watch condition belongs to one
    person and that report is global. The pointer is what keeps the two from
    drifting apart."""
    from tools.availability import _open_measurements

    row = next(m for m in _open_measurements()
               if m["number"] == "alerts that should have fired and did not")
    assert "GET /api/alerts/missed" in row["how"]
    assert "PER-PROFILE" in row["state"]
