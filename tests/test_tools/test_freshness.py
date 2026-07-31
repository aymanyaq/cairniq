"""As-of freshness stamps (Advisor Roadmap Theme 5.8 — alert-path slice).

The point of this module is to make a claim *provable*: that a number an alert
fired on was actually observed when it says it was. So the tests are weighted
toward the ways a freshness check can lie — a stamp applied on read instead of
fetch, an unverified payload being reported as fresh, and minute-age being
applied to daily bars, where it is meaningless.
"""
from datetime import datetime, timedelta

import pytest

import tools.freshness as fr


def _at(h, m=0, day=23):
    return datetime(2026, 7, day, h, m)


# ---------------------------------------------------------------------------
# stamp / as_of — the stamp must record FETCH time and survive a cache replay
# ---------------------------------------------------------------------------

def test_stamp_records_the_fetch_time_and_is_read_back():
    payload = fr.stamp({"price": 100.0}, now=_at(9, 42))

    assert fr.as_of(payload) == _at(9, 42)
    assert payload["price"] == 100.0  # payload otherwise untouched


def test_stamp_does_not_overwrite_an_existing_stamp():
    """The core anti-lie property: re-stamping on read would reset the clock and
    make an hour-old cached quote look permanently fresh."""
    payload = fr.stamp({"price": 100.0}, now=_at(9, 0))
    fr.stamp(payload, now=_at(15, 0))  # a later read must not move it

    assert fr.as_of(payload) == _at(9, 0)
    assert fr.age_minutes(payload, now=_at(15, 0)) == 360.0


def test_stamp_passes_non_dicts_through_untouched():
    assert fr.stamp("not a dict") == "not a dict"
    assert fr.stamp(None) is None


def test_unstamped_and_unparseable_payloads_read_as_none():
    assert fr.as_of({"price": 1}) is None
    assert fr.as_of({fr.AS_OF_KEY: "not-a-timestamp"}) is None
    assert fr.as_of({fr.AS_OF_KEY: ""}) is None
    assert fr.as_of("nope") is None


# ---------------------------------------------------------------------------
# age_minutes — quote data
# ---------------------------------------------------------------------------

def test_age_minutes_measures_from_the_stamp():
    payload = fr.stamp({}, now=_at(10, 0))
    assert fr.age_minutes(payload, now=_at(10, 30)) == 30.0


def test_age_of_an_unstamped_payload_is_none_not_zero():
    """Zero would read as 'brand new' — the exact overclaim to avoid."""
    assert fr.age_minutes({"price": 1}, now=_at(10, 0)) is None


def test_a_future_stamp_is_clamped_rather_than_reported_as_negative():
    payload = fr.stamp({}, now=_at(11, 0))
    assert fr.age_minutes(payload, now=_at(10, 0)) == 0.0


# ---------------------------------------------------------------------------
# is_stale / is_verified — stale and unverified are different things
# ---------------------------------------------------------------------------

def test_is_stale_only_when_provably_too_old():
    fresh = fr.stamp({}, now=_at(10, 0))
    assert fr.is_stale(fresh, 45, now=_at(10, 30)) is False
    assert fr.is_stale(fresh, 45, now=_at(11, 30)) is True


def test_an_unverified_payload_is_not_reported_as_stale():
    unverified = {"price": 1}
    assert fr.is_stale(unverified, 45, now=_at(10, 0)) is False
    assert fr.is_verified(unverified) is False


def test_is_verified_true_only_with_a_readable_stamp():
    assert fr.is_verified(fr.stamp({}, now=_at(10, 0))) is True
    assert fr.is_verified({fr.AS_OF_KEY: "garbage"}) is False


# ---------------------------------------------------------------------------
# is_current_session — daily bars, where minute-age is meaningless
# ---------------------------------------------------------------------------

def test_a_bar_dated_today_is_current_even_when_hours_old():
    """A daily bar stamped 00:00 is 10 hours 'old' at 10am and still current —
    which is why the bar path must not use minute-age."""
    bar = fr.stamp({}, now=_at(0, 0))

    assert fr.is_current_session(bar, now=_at(10, 0)) is True
    assert fr.age_minutes(bar, now=_at(10, 0)) == 600.0  # would look ancient


def test_a_bar_from_a_previous_session_is_not_current():
    bar = fr.stamp({}, now=_at(16, 0, day=22))
    assert fr.is_current_session(bar, now=_at(10, 0, day=23)) is False


def test_current_session_is_none_when_unstamped():
    assert fr.is_current_session({"price": 1}, now=_at(10, 0)) is None


# ---------------------------------------------------------------------------
# describe — never overclaims
# ---------------------------------------------------------------------------

def test_describe_states_the_clock_and_the_age():
    payload = fr.stamp({}, now=_at(9, 42))
    assert fr.describe(payload, now=_at(9, 45)) == "as of 09:42 (3 min ago)"


def test_describe_says_unverified_rather_than_guessing():
    assert fr.describe({"price": 1}) == "as-of unverified"


def test_describe_never_claims_live_or_realtime():
    """`data_freshness` already lies this way ('Real-time' on an hour-old cache
    hit); this module must not add a second voice saying it."""
    for now, stamped in [(_at(9, 43), _at(9, 42)), (_at(15, 0), _at(9, 0))]:
        text = fr.describe(fr.stamp({}, now=stamped), now=now).lower()
        assert "live" not in text
        assert "real-time" not in text and "realtime" not in text


@pytest.mark.parametrize("delta,expected_fragment", [
    (timedelta(seconds=20), "just now"),
    (timedelta(minutes=30), "30 min ago"),
    (timedelta(hours=5), "5.0h ago"),
    (timedelta(days=3), "3d ago"),
])
def test_describe_scales_its_units(delta, expected_fragment):
    base = _at(9, 0)
    payload = fr.stamp({}, now=base)
    assert expected_fragment in fr.describe(payload, now=base + delta)
