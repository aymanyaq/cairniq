"""The weekly one-page review.

The load-bearing test in this file is the FIRST one, and it is the reason the
module is written the way it is: a report is the highest-risk surface in this
codebase for the 2026-07-21 failure, where truthiness-gated blocks emitted
nothing, the reader expected a full page, and the silence got back-filled with
real-sounding content. So the empty profile is the primary case, not the edge
case — every section must survive it, by name, saying it has nothing.

The rest follow the same shape as the readiness surface's tests: does it count
what the consumer reads, does an unreadable store read as unknown rather than as
empty, and does it refuse to start work it is only supposed to report on.
"""
from datetime import datetime, timedelta

import pytest

import tools.memory as mem
import tools.weekly_review as wr

_ALL_SECTIONS = {"goal", "market", "scorecard", "advice", "engines", "readiness"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Every store the page reads, homed in a tmp profile — a test that wrote
    into the live profile would corrupt the data the page reports on."""
    import tools.feedback as fb
    import tools.risk_verdict_log as rvl

    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(fb, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(rvl, "get_data_path", lambda name: str(tmp_path / name))
    return tmp_path


def _by_key(review):
    return {s["key"]: s for s in review["sections"]}


# ---------------------------------------------------------------------------
# THE contract: nothing is ever omitted
# ---------------------------------------------------------------------------

def test_an_empty_profile_still_renders_every_section():
    """The whole reason this module exists. A section with nothing to say must
    say so — an omitted section is an invitation to fill the gap."""
    review = wr.build_weekly_review()

    assert set(_by_key(review)) == _ALL_SECTIONS
    assert review["counts"]["total"] == len(_ALL_SECTIONS)


def test_every_blank_section_states_that_it_is_blank():
    """An empty section without a note is a silent omission wearing a header."""
    review = wr.build_weekly_review()

    for section in review["sections"]:
        if section["status"] in (wr.STATUS_EMPTY, wr.STATUS_UNREADABLE):
            assert section["note"].strip(), f"{section['key']} is blank and says nothing"


def test_a_blank_section_never_reports_a_figure():
    """The failure mode is a report that fills a gap with something plausible.
    A blank section may explain the absence; it may not produce a number that
    reads as a result."""
    review = wr.build_weekly_review()

    for section in review["sections"]:
        if section["status"] != wr.STATUS_EMPTY:
            continue
        for key in ("horizons", "confidence", "partial", "calls", "goal_success_rate"):
            assert not section.get(key), f"{section['key']} is empty but carries {key}"


def test_a_broken_section_reads_as_unknown_not_as_empty(monkeypatch):
    """An unreadable store and an empty one are different findings, and
    collapsing them would hide a fault behind a legitimate blank."""
    def boom(period):
        raise RuntimeError("store is corrupt")

    monkeypatch.setattr(wr, "_BUILDERS", (boom,))
    monkeypatch.setattr(wr, "_TITLES", {"boom": ("boom", "Boom")})

    review = wr.build_weekly_review()
    section = review["sections"][0]

    assert section["status"] == wr.STATUS_UNREADABLE
    assert "unavailable, not empty" in section["note"]
    assert "store is corrupt" in section["error"]


def test_one_broken_section_does_not_break_the_page():
    """An instrument that goes dark when one input does has acquired the fault
    it was built to catch."""
    original = wr._goal_section

    def boom(period):
        raise RuntimeError("goal store unreadable")

    wr._BUILDERS = tuple(boom if b is original else b for b in wr._BUILDERS)
    try:
        review = wr.build_weekly_review()
        assert set(_by_key(review)) >= _ALL_SECTIONS - {"goal"}
        assert review["counts"]["unreadable"] == 1
    finally:
        wr._BUILDERS = tuple(original if b is boom else b for b in wr._BUILDERS)


# ---------------------------------------------------------------------------
# It reads; it never generates
# ---------------------------------------------------------------------------

def test_a_cold_pulse_cache_is_reported_not_generated(monkeypatch):
    """A weekly page that kicks off a multi-minute network job would block on
    every cold open. The absence is the report."""
    started = []
    monkeypatch.setattr("tools.cache.get_cached", lambda key: None)
    monkeypatch.setattr(
        "tools.market_sentinel.generate_market_pulse",
        lambda *a, **k: started.append(True),
    )

    section = _by_key(wr.build_weekly_review())["market"]

    assert section["status"] == wr.STATUS_EMPTY
    assert not started, "the review started a pulse generation"


def test_a_cached_pulse_is_reported(monkeypatch):
    monkeypatch.setattr(
        "tools.cache.get_cached",
        lambda key: {"regime": "NEUTRAL", "headline": "SPY +0.4% | F&G 55"} if key == "market_pulse" else None,
    )

    section = _by_key(wr.build_weekly_review())["market"]

    assert section["status"] == wr.STATUS_OK
    assert section["regime"] == "NEUTRAL"


# ---------------------------------------------------------------------------
# It counts what the consumer reads
# ---------------------------------------------------------------------------

def test_the_scorecard_reports_distinct_calls_beside_the_row_count(monkeypatch):
    """On the live ledger 9 graded rows were 4 calls. A reader shown only the
    row count is being told the sample is more than twice its real size."""
    monkeypatch.setattr(
        "tools.memory.get_scored_recommendations_data",
        lambda: {
            "stats": {"2w": {"total": 0}, "1m": {"total": 0}, "3m": {"total": 0}},
            "confidence_stats": {},
            "partial_stats": {"total": 4, "hits": 4, "hit_rate": 100.0, "graded_rows": 9},
            "recommendations": [],
        },
    )

    section = _by_key(wr.build_weekly_review())["scorecard"]

    assert section["status"] == wr.STATUS_OK
    assert section["partial"]["total"] == 4
    assert "4 distinct" in section["sample_note"]
    assert "9 graded rows" in section["sample_note"]


def test_nothing_scored_says_how_many_are_logged(monkeypatch):
    """"Nothing scored" and "nothing logged" are different states, and the
    difference is what tells you whether to wait or to look for a bug."""
    monkeypatch.setattr(
        "tools.memory.get_scored_recommendations_data",
        lambda: {
            "stats": {"2w": {"total": 0}, "1m": {"total": 0}, "3m": {"total": 0}},
            "confidence_stats": {},
            "partial_stats": {"total": 0, "graded_rows": 0},
            "recommendations": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
        },
    )

    section = _by_key(wr.build_weekly_review())["scorecard"]

    assert section["status"] == wr.STATUS_EMPTY
    assert section["logged"] == 2
    assert "2 calls logged" in section["note"]


# ---------------------------------------------------------------------------
# The period boundary
# ---------------------------------------------------------------------------

def test_only_calls_inside_the_period_are_reported(monkeypatch):
    now = datetime(2026, 7, 28, 9, 0, 0)
    inside = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    outside = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    monkeypatch.setattr(
        "tools.memory.load_memory",
        lambda: {"past_recommendations": [
            {"ticker": "AAPL", "action": "BUY", "date": inside},
            {"ticker": "MSFT", "action": "SELL", "date": outside},
        ]},
    )

    section = _by_key(wr.build_weekly_review(now=now))["advice"]

    assert section["call_count"] == 1
    assert section["calls"][0]["ticker"] == "AAPL"


def test_an_unparseable_date_excludes_the_record_without_breaking_the_page(monkeypatch):
    monkeypatch.setattr(
        "tools.memory.load_memory",
        lambda: {"past_recommendations": [{"ticker": "AAPL", "action": "BUY", "date": "not a date"}]},
    )

    section = _by_key(wr.build_weekly_review())["advice"]

    assert section["status"] == wr.STATUS_EMPTY


# ---------------------------------------------------------------------------
# The heartbeat line
# ---------------------------------------------------------------------------

def test_the_heartbeat_line_proves_the_chain_ran():
    """It reports the section counts, not a rare event. An engine whose detail
    line only fills in on an interesting week is indistinguishable from one that
    stopped."""
    line = wr.summarize_for_heartbeat(wr.build_weekly_review())

    assert "sections" in line
    assert "reported" in line
    assert "empty" in line
