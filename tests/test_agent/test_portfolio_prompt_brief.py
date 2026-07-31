"""The portfolio block injected into DeepReasoning prompts.

DeepReasoning used to inject `str(get_portfolio_summary())` — a raw Python repr —
at three prompt sites per heavy turn. The repr carried ~20 fields per holding,
five of them different renderings of the same number (value_native / _base / _usd
/ _cad, and purchase_price / _raw), measured at ~523 chars per holding.

The replacement is a compact text block. What these tests pin is not the size but
the thing that makes shrinking it safe: the aggregates synthesis actually reads —
liquidity, accounts, winners/losers, the return — have to survive, because the
obvious cheaper option (reusing the holdings-verification block) drops them.
"""
import pytest

from agent.nodes.deep_reasoning import _format_portfolio_brief


def _holding(i: int) -> dict:
    """A holding with the full redundant field set the repr used to emit."""
    return {
        "symbol": f"SYM{i:02d}",
        "shares": 100 + i,
        "current_price": 100.0 + i,
        "purchase_price": 90.0 + i,
        "purchase_price_raw": 90.0 + i,
        "gain_loss": f"+${1000 + i:,.2f} (+11.1%)",
        "account": "Questrade TFSA",
        "source": "broker",
        "currency": "USD",
        "value_native": 10000.0 + i,
        "value_base": 10000.0 + i,
        "value_usd": 10000.0 + i,
        "value_cad": 14000.0 + i,
        "is_cash_or_pension": False,
        "sector": "Technology",
    }


@pytest.fixture
def summary() -> dict:
    return {
        "holdings": [_holding(i) for i in range(40)],
        "base_currency": "CAD",
        "total_value_base": 400_000.0,
        "total_value_cad": 400_000.0,
        "total_value_usd": 285_714.29,
        "last_sync_time": "2026-07-31T07:00:00",
        "top_winners": ["SYM01: +$1,001.00"],
        "top_losers": ["SYM39: -$500.00"],
        "liquidity": {
            "total_liquid_cash": "$50,000.00 CAD",
            "pure_cash": "$20,000.00 CAD",
            "cash_equivalents": "$30,000.00 CAD",
            "locked_pension_value": "$0.00 CAD",
        },
        "accounts": {"Questrade TFSA": {"value_base": 200_000.0}, "RRSP": {"value_base": 200_000.0}},
        "summary": {
            "total_invested": "$360,000.00 CAD",
            "current_value": "$400,000.00 CAD",
            "total_gain_loss": "+$40,000.00 CAD",
            "total_return": "+11.1%",
            "number_of_positions": 40,
            "exchange_rate_used": "1 USD = 1.40 CAD",
        },
        "sync_errors": [],
        "integration_notices": [],
    }


def test_brief_is_far_smaller_than_the_repr_it_replaced(summary):
    brief = _format_portfolio_brief(summary)
    per_holding = len(brief) / len(summary["holdings"])
    # The repr this replaced measured ~523 chars/holding on a real-shaped book.
    # Asserting per-holding rather than a whole-string ratio keeps the bound
    # meaningful even though this fixture carries fewer junk fields than
    # production holdings do.
    assert per_holding < 150, f"{per_holding:.0f} chars/holding"
    assert len(brief) < len(str(summary)) / 3


def test_every_holding_is_still_listed(summary):
    brief = _format_portfolio_brief(summary)
    for h in summary["holdings"]:
        assert h["symbol"] in brief


def test_the_aggregates_synthesis_reads_survive(summary):
    """The reason this is not just the verification block."""
    brief = _format_portfolio_brief(summary)
    # Dry powder — an accumulation decision turns on this figure.
    assert "$50,000.00 CAD" in brief
    assert "locked pension" in brief
    # Return, position count, FX and the per-account split.
    assert "+11.1%" in brief
    assert "40 positions" in brief
    assert "1 USD = 1.40 CAD" in brief
    assert "RRSP" in brief
    assert "SYM01: +$1,001.00" in brief


def test_redundant_currency_twins_are_gone(summary):
    """Four renderings of one number is what made the repr expensive."""
    brief = _format_portfolio_brief(summary)
    for field in ("value_native", "value_usd", "purchase_price_raw", "is_cash_or_pension"):
        assert field not in brief


def test_holdings_are_capped_and_the_omission_is_stated(summary):
    brief = _format_portfolio_brief(summary, max_holdings=10)
    assert "SYM09" in brief
    assert "SYM10" not in brief
    # Silence about a truncation reads as "that is the whole book".
    assert "30 additional holdings omitted" in brief


def test_sync_errors_are_surfaced_not_swallowed(summary):
    summary["sync_errors"] = ["Questrade: token expired"]
    summary["integration_notices"] = ["Wealthsimple: not configured"]
    brief = _format_portfolio_brief(summary)
    assert "token expired" in brief
    # "never asked" and "asked and failed" are different answers about whether
    # the totals above are complete.
    assert "Wealthsimple: not configured" in brief


def test_an_errored_summary_says_so_rather_than_rendering_an_empty_book():
    brief = _format_portfolio_brief({"error": "broker sync failed"})
    assert "Portfolio unavailable" in brief
    assert "broker sync failed" in brief
    assert "Holdings:" not in brief
