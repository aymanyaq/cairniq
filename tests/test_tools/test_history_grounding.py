"""Empty history blocks must assert their emptiness, not fall silent.

Regression suite for the "invented rejection history" class of bug: a Today's Priority
brief narrated two tickers as past rotation targets that "failed strict entry rules",
and put a fabricated entry plan on the radar — with past_recommendations and
active_theses BOTH empty and neither name recorded anywhere. The tickers had only ever
been seen by a funnel scan, which stores nothing but a last-seen date per symbol.

The failure was not a bad number, it was silence: a block omitted because it had no
rows reads as "not provided" rather than "there is none", and the model back-fills it.
Worse, the invented rejection reasons were quoted from REAL user lessons, which is
exactly what made the fabrication read as sourced.

So the contract is: an empty ledger emits an explicit, authoritative negative.
"""

from typing import Any

import pytest

import tools.memory as memory_module
from tools.memory import get_user_context, set_recommendation_execution


def _blank_memory() -> dict[str, Any]:
    return {
        "user_profile": {"name": "Test User", "base_currency": "CAD"},
        "key_facts": [],
        "conversation_summaries": [],
        "past_recommendations": [],
        "lessons_learned": ["Never catch a falling knife."],
        "active_theses": [],
        "secular_themes": [],
    }


@pytest.fixture
def context(monkeypatch):
    """Render the injected memory context over an empty-history profile."""
    monkeypatch.setattr("tools.memory.load_memory", _blank_memory)
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: {"AAPL"})
    return get_user_context()


# --- prior recommendations -------------------------------------------------------


def test_empty_recommendations_state_none_on_record(context):
    assert "Prior recommendations — NONE ON RECORD" in context


def test_empty_recommendations_forbid_inventing_an_evaluation(context):
    """The exact fabrication: naming a scan ticker as a past target that failed."""
    assert "screened" in context and "rejected" in context
    assert "only SEEN" in context


def test_absence_from_portfolio_is_not_a_reason(context):
    """not-held never reveals WHY — declined, never-recommended and pending are alike."""
    assert "does not reveal why it is absent" in context


# --- active theses ---------------------------------------------------------------


def test_empty_theses_state_none_on_record(context):
    assert "NONE ON RECORD" in context
    assert "no active theses" in context


def test_empty_theses_forbid_a_fabricated_watching_tag(context):
    """The invented 'XYZ [WATCHING] ... Stop: $600' radar line."""
    assert "[WATCHING]/[HELD] tag" in context
    assert "entry zone, stop, target" in context


def test_empty_blocks_are_marked_authoritative_not_missing(context):
    """A model told only 'empty' may treat the block as a data gap worth filling."""
    assert context.count("complete and authoritative") == 2


# --- populated ledger ------------------------------------------------------------


def test_populated_ledger_declares_execution_status_unknown(monkeypatch):
    """The reported gap: "I did not execute on that buy recommendation."

    The record has no `executed` field — it stores what was advised, never what was
    done — so an acted-on call and a declined one are indistinguishable here. That
    must be said out loud rather than left for the model to guess at.
    """
    mem = _blank_memory()
    mem["past_recommendations"] = [
        {"date": "2026-07-18", "ticker": "TSLA", "action": "BUY", "price_at_advice": 512.0}
    ]
    monkeypatch.setattr("tools.memory.load_memory", lambda: mem)
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: {"AAPL"})

    context = get_user_context()

    assert "user action: NOT RECORDED" in context
    assert "what was advised, never what was done" in context
    assert "Prior recommendations — NONE ON RECORD" not in context


def test_superseded_only_ledger_still_reads_as_populated(monkeypatch):
    """A ledger whose every row is superseded renders no rows but is NOT 'none on
    record' — calls were made, they were just closed out. Claiming an empty history
    there would be its own fabrication."""
    mem = _blank_memory()
    mem["past_recommendations"] = [
        {"date": "2026-07-18", "ticker": "TSLA", "action": "BUY", "superseded": True}
    ]
    monkeypatch.setattr("tools.memory.load_memory", lambda: mem)
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: {"AAPL"})

    context = get_user_context()

    assert "Prior recommendations — NONE ON RECORD" not in context
    assert "NONE OPEN" in context
    assert "do not claim an empty recommendation history" in context
    # The closed rows are withheld, so their details must stay unstatable.
    assert "do not name, date, or assign an outcome or reason" in context


# --- executed flag ---------------------------------------------------------------


@pytest.fixture
def ledger(monkeypatch):
    """One open BUY on TSLA, execution status unrecorded."""
    mem = _blank_memory()
    mem["past_recommendations"] = [
        {"date": "2026-07-18", "ticker": "TSLA", "action": "BUY",
         "price_at_advice": 512.0, "executed": None}
    ]
    monkeypatch.setattr("tools.memory.load_memory", lambda: mem)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: mem.update(m))
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: {"AAPL"})
    return mem


def test_declined_recommendation_renders_as_declined(ledger):
    """The reported case: a BUY was advised and the user did not act on it."""
    set_recommendation_execution("TSLA", executed=False, note="waiting for a pullback")

    assert ledger["past_recommendations"][0]["executed"] is False
    context = get_user_context()
    assert "user action: DECLINED (user did NOT act on this)" in context
    assert "waiting for a pullback" in context


def test_executed_recommendation_renders_as_executed(ledger):
    set_recommendation_execution("TSLA", executed=True)

    context = get_user_context()
    assert "user action: EXECUTED" in context
    assert "DECLINED" not in context


def test_unrecorded_execution_says_so_and_carries_the_caveat(ledger):
    context = get_user_context()

    assert "user action: NOT RECORDED" in context
    assert "never read it off the portfolio" in context


def test_caveat_drops_once_every_row_is_known(ledger):
    """The 'not recorded' warning is noise once the fact is actually on file."""
    set_recommendation_execution("TSLA", executed=False)

    context = get_user_context()
    assert "NOT RECORDED" not in context
    assert "never read it off the portfolio" not in context


def test_unknown_ticker_does_not_manufacture_a_row(ledger):
    """A report about advice we have no record of giving is a discrepancy to surface,
    not a ledger row to invent — the same fabrication in reverse."""
    result = set_recommendation_execution("NVDA", executed=False)

    assert len(ledger["past_recommendations"]) == 1
    assert "No open recommendation on record for NVDA" in result


def test_execution_status_is_not_inferred_at_write_time(monkeypatch):
    """add_recommendation must leave `executed` unset even though holdings are
    readable — inferring it is precisely the guess that produces false history."""
    mem = _blank_memory()
    monkeypatch.setattr("tools.memory.load_memory", lambda: mem)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: mem.update(m))
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: {"AAPL"})

    memory_module.add_recommendation("AAPL", "BUY", "held name, still a buy")

    # AAPL *is* held, which is the tempting-but-wrong signal to stamp executed=True.
    assert mem["past_recommendations"][-1]["executed"] is None
