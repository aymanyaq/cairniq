"""Held-vs-watching lifecycle for active investment theses.

Regression suite for the "Today's Priority told me to sell a stock I never bought"
class of bug: a BUY thesis on a name the user does not hold yet is an ENTRY PLAN
being monitored for execution, not an open position. Nothing about it may render as
an exit, and its absence from the portfolio is its resting state — never a
contradiction to resolve by deleting the thesis.
"""

from datetime import datetime, timedelta
from typing import Any

import pytest

import tools.memory as memory_module
from tools.memory import _enrich_thesis_with_price_context, add_recommendation


def _stub_price(monkeypatch, price: float, beta: str = "1.0") -> None:
    monkeypatch.setattr(
        "tools.market_data.get_stock_data",
        lambda symbol: {"current_price": str(price), "beta": beta},
    )


def _flags(thesis: dict[str, Any]) -> str:
    return " | ".join(thesis["_health_flags"])


# --- normalization -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MU", "MU"),
        ("$NVDA", "NVDA"),          # the thesis extractor stores a leading '$'
        ("keel.to", "KEEL"),        # listing suffix stripped to match a held KEEL.TO
        ("  aapl  ", "AAPL"),
        (None, ""),
    ],
)
def test_normalize_thesis_symbol(raw, expected):
    assert memory_module._normalize_thesis_symbol(raw) == expected


def test_position_state_unknown_when_portfolio_unreadable():
    """None from the holdings read means UNKNOWN, never 'not held' — a failed
    portfolio fetch must not be able to reclassify a real position as a watchlist
    entry (or vice versa)."""
    thesis = {"symbol": "MU", "action": "BUY"}
    assert memory_module._thesis_position_state(thesis, None) == "unknown"
    assert memory_module._thesis_position_state(thesis, set()) == "watching"
    assert memory_module._thesis_position_state(thesis, {"MU"}) == "held"


# --- WATCHING: the reported bug -------------------------------------------------


def test_watching_thesis_at_stop_never_says_sell(monkeypatch):
    """THE REGRESSION: MU entered as BUY, not held, price at/below the stop.

    Old behaviour fired 'STOP BREACHED — the thesis is invalidated, close it', which
    Lane 0 rendered as a sell directive for a position that never existed.
    """
    _stub_price(monkeypatch, 60.0)
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "Accumulate near $80 on AI memory cycle",
        "stop_loss": "70",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis), held=set())

    assert enriched["_position_state"] == "watching"
    flags = _flags(enriched)
    assert "SETUP BROKEN PRE-ENTRY" in flags
    assert "Nothing to sell" in flags
    # None of the HELD branch's exit directives may appear. Checked as directives, not
    # bare words — the flag text legitimately contains "sell" inside "Nothing to sell".
    for exit_directive in ("STOP BREACHED", "close it", "take profit", "Decide: take profit"):
        assert exit_directive not in flags, f"exit language leaked into a not-held thesis: {flags}"


def test_watching_thesis_in_entry_zone_fires_entry_triggered(monkeypatch):
    """The flag the whole feature exists for, and which previously did not exist:
    price inside the buy zone on a name being monitored produced NO flag at all, so
    Lane 0 never fired and the entry was never surfaced."""
    _stub_price(monkeypatch, 75.0)
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "Accumulate near $80 on AI memory cycle",
        "stop_loss": "60",
        "target_price": "140",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis), held=set())

    flags = _flags(enriched)
    assert "ENTRY TRIGGERED" in flags
    assert "$75.00" in flags and "$80.00" in flags
    assert "SETUP BROKEN" not in flags


def test_watching_thesis_above_target_is_not_take_profit(monkeypatch):
    """Price blew through the target without them: re-base or drop. There is no
    profit to take on a position that was never opened."""
    _stub_price(monkeypatch, 150.0)
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "Accumulate near $80",
        "target_price": "140",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis), held=set())

    flags = _flags(enriched)
    assert "TARGET REACHED PRE-ENTRY" in flags
    assert "Nothing to take profit on" in flags
    # Must not ALSO claim a plain TARGET REACHED, which Lane 0 reads as an exit.
    assert "TARGET REACHED —" not in flags
    assert "Decide: take profit" not in flags


def test_watching_thesis_far_above_entry_is_missed_not_chased(monkeypatch):
    _stub_price(monkeypatch, 120.0, beta="0.8")  # low beta -> 5% missed threshold
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "Accumulate near $80",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis), held=set())

    flags = _flags(enriched)
    assert "ENTRY MISSED" in flags
    assert "do NOT chase" in flags


# --- HELD: existing lifecycle must survive intact --------------------------------


def test_held_thesis_keeps_exit_flags(monkeypatch):
    """The HELD branch is the pre-existing behaviour and must not regress: a real
    position at its stop still gets closed, at its target still gets a decision."""
    _stub_price(monkeypatch, 60.0)
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "Accumulate near $80",
        "stop_loss": "70",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis), held={"MU"})

    assert enriched["_position_state"] == "held"
    flags = _flags(enriched)
    assert "STOP BREACHED" in flags
    assert "close it" in flags
    assert "PRE-ENTRY" not in flags


def test_held_thesis_at_target_reaches_target(monkeypatch):
    _stub_price(monkeypatch, 150.0)
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "Accumulate near $80",
        "target_price": "140",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis), held={"MU"})

    flags = _flags(enriched)
    assert "TARGET REACHED" in flags
    assert "take profit" in flags
    assert "PRE-ENTRY" not in flags


def test_held_match_survives_listing_suffix(monkeypatch):
    """A thesis pinned as 'KEEL' must resolve against a held 'KEEL.TO'."""
    _stub_price(monkeypatch, 5.0)
    thesis = {
        "symbol": "KEEL",
        "action": "BUY",
        "conditions": "Accumulate near $8",
        "stop_loss": "6",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(
        dict(thesis), held=memory_module._normalize_thesis_symbol("KEEL.TO") and {"KEEL"}
    )

    assert enriched["_position_state"] == "held"
    assert "STOP BREACHED" in _flags(enriched)


# --- UNKNOWN: never guess --------------------------------------------------------


def test_unknown_held_status_issues_no_directive(monkeypatch):
    """Portfolio unreadable: emit context, but never an entry or exit directive that
    depends on held status."""
    _stub_price(monkeypatch, 60.0)
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: None)
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "Accumulate near $80",
        "stop_loss": "70",
        "created_at": datetime.now().isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis))

    assert enriched["_position_state"] == "unknown"
    flags = _flags(enriched)
    assert "HELD STATUS UNVERIFIED" in flags
    assert "STOP BREACHED" not in flags
    assert "SETUP BROKEN" not in flags
    assert enriched["_live_price"] == 60.0


def test_stale_only_path_does_not_read_portfolio(monkeypatch):
    """A thesis with no live price returns before held status is ever resolved —
    the no-data path must stay network-free."""
    def boom():
        raise AssertionError("portfolio must not be read when there is no price")

    monkeypatch.setattr(memory_module, "_held_base_symbols", boom)
    monkeypatch.setattr(
        "tools.market_data.get_stock_data",
        lambda symbol: (_ for _ in ()).throw(RuntimeError("quote service down")),
    )
    thesis = {
        "symbol": "ZZZ",
        "action": "BUY",
        "conditions": "core",
        "stop_loss": "10",
        "created_at": (datetime.now() - timedelta(days=120)).isoformat(),
    }

    enriched = _enrich_thesis_with_price_context(dict(thesis))

    assert any("STALE" in f for f in enriched["_health_flags"])


# --- CONTRADICTED stamping -------------------------------------------------------


def _ledger_memory() -> dict[str, Any]:
    return {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [],
        "active_theses": [
            {
                "id": "t1",
                "symbol": "MU",
                "action": "BUY",
                "conditions": "Accumulate near $80",
            }
        ],
    }


def test_sell_does_not_contradict_a_watching_thesis(monkeypatch):
    """A SELL logged against a name the user never owned contradicts nothing.
    Stamping exit_signal here is what drove the model to cancel a live entry plan."""
    test_memory = _ledger_memory()
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: set())

    add_recommendation("MU", "SELL", "trimming AI exposure")

    assert "exit_signal" not in test_memory["active_theses"][0]


def test_sell_contradicts_a_held_thesis(monkeypatch):
    """The original reconciliation behaviour, preserved for real positions."""
    test_memory = _ledger_memory()
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: {"MU"})

    add_recommendation("MU", "SELL", "thesis played out")

    assert test_memory["active_theses"][0]["exit_signal"]["action"] == "SELL"


def test_unverified_holdings_do_not_stamp_contradiction(monkeypatch):
    """A false CONTRADICTED costs the user a live thesis; a missing one costs a
    nudge. When held status is unknown, take the cheap error."""
    test_memory = _ledger_memory()
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))
    monkeypatch.setattr(memory_module, "_held_base_symbols", lambda: None)

    add_recommendation("MU", "SELL", "trimming AI exposure")

    assert "exit_signal" not in test_memory["active_theses"][0]
