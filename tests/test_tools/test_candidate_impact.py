"""Pre-trade candidate impact preview tests (Advisor Roadmap 4.9) — offline.

The two heavy collaborators are mocked at their seams:
  - `_decision_context` (portfolio holdings/total)
  - `_impact_returns` (the base-currency return fetch)
and `run_ips_precheck` is stubbed (it has its own suite). What's exercised here
is 4.9's own composition: size resolution, weight construction, and the
before/after vol / beta / CVaR / correlation deltas.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import tools.candidate_impact as ci


def _ctx(total=200_000.0, base="USD"):
    return {
        "total_value_base": total,
        "base_currency": base,
        "holdings": [
            {"symbol": "AAPL", "value_base": 60_000.0, "is_cash_or_pension": False},
            {"symbol": "MSFT", "value_base": 40_000.0, "is_cash_or_pension": False},
            {"symbol": "CASH", "value_base": 100_000.0, "is_cash_or_pension": True},
        ],
    }


def _returns_frame(symbols, n=260, seed=0):
    """Deterministic pseudo-return matrix with a stable FX attrs stamp."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    data = {s: rng.normal(0.0005, 0.012, n) for s in symbols}
    df = pd.DataFrame(data, index=idx)
    df.attrs["fx"] = {"base_currency": "USD", "converted": [], "unavailable": []}
    return df


def test_synthesized_buy_text_parses():
    """Integration seam: the text preview_candidate_impact feeds run_ips_precheck
    must actually extract a SIZED trade — "Buy TICKER $X" does not, "Buy $X of
    TICKER" does. If this regresses, every IPS cap check silently degrades to
    NOT_EVALUATED and the preview stops catching breaches."""
    from tools.ips_precheck import extract_proposed_trades

    for size_pct, entry, stop in [(8.0, None, None), (8.0, 180.5, 165.0)]:
        captured = {}

        def _spy(text, candidate_tickers=None):
            trades = extract_proposed_trades(text, candidate_tickers=candidate_tickers, total_value=300_000.0)
            captured["trades"] = trades
            return {"trades": trades, "rows": [], "violations": [], "block": ""}

        with patch.object(ci, "_decision_context", return_value=_ctx(total=300_000.0)), \
             patch.object(ci, "_impact_returns", return_value=(_returns_frame(["AAPL", "MSFT", "NVDA", "SPY"]), ["AAPL", "MSFT", "NVDA", "SPY"])), \
             patch("tools.ips_precheck.run_ips_precheck", _spy):
            ci.preview_candidate_impact("NVDA", size_pct=size_pct, entry=entry, stop=stop)

        assert captured["trades"], "synthesized buy text failed to parse a trade"
        t = captured["trades"][0]
        assert t["ticker"] == "NVDA"
        assert t["size_usd"] == pytest.approx(24_000.0)  # 8% of 300k
        if stop:
            assert t["stop"] == pytest.approx(stop)
            assert t["stated_entry"] == pytest.approx(entry)


def _stub_precheck(rows=None, violations=None):
    return {
        "trades": [{"ticker": "NVDA"}],
        "rows": rows if rows is not None else [
            {"trade": "BUY NVDA", "check": "position cap", "computed": "0.0% now → 5.0% post-trade",
             "limit": "≤10%", "verdict": "PASS"},
            {"trade": "BUY NVDA", "check": "account location", "computed": "held in: X", "limit": "informational", "verdict": "INFO"},
        ],
        "violations": violations or [],
        "block": "<ips_precheck>...</ips_precheck>",
    }


# ---------------------------------------------------------------------------
# Size resolution
# ---------------------------------------------------------------------------

def test_default_probe_size():
    with patch.object(ci, "_decision_context", return_value=_ctx(total=200_000.0)), \
         patch.object(ci, "_impact_returns", return_value=(_returns_frame(["AAPL", "MSFT", "NVDA", "SPY"]), ["AAPL", "MSFT", "NVDA", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("NVDA")
    assert "assumed 5% probe" in r["proposed_size"]["basis"]
    # 5% of 200k = 10k
    assert r["proposed_size"]["dollars"] == "$10,000"


def test_size_pct_and_usd_resolution():
    with patch.object(ci, "_decision_context", return_value=_ctx(total=200_000.0)), \
         patch.object(ci, "_impact_returns", return_value=(_returns_frame(["AAPL", "MSFT", "NVDA", "SPY"]), ["AAPL", "MSFT", "NVDA", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        by_pct = ci.preview_candidate_impact("NVDA", size_pct=10.0)
        by_usd = ci.preview_candidate_impact("NVDA", size_usd=20_000.0)
    assert by_pct["proposed_size"]["dollars"] == "$20,000"
    assert "10% of portfolio" in by_pct["proposed_size"]["basis"]
    assert by_usd["proposed_size"]["dollars"] == "$20,000"
    assert by_usd["proposed_size"]["basis"] == "stated"


def test_shares_size_converts_listing_currency_to_base():
    """`shares × price` is in the security's currency, not the portfolio's.

    Every figure downstream — candidate_weight, pct_of_current_portfolio, the
    risk deltas — is compared against a base-currency total, so an unconverted
    US price on a CAD book understates the position by the whole FX rate.
    """
    ctx = _ctx(total=200_000.0, base="CAD")
    with patch.object(ci, "_decision_context", return_value=ctx), \
         patch.object(ci, "_impact_returns",
                      return_value=(_returns_frame(["AAPL", "MSFT", "ISRG", "SPY"]), ["AAPL", "MSFT", "ISRG", "SPY"])), \
         patch.object(ci, "_candidate_quote", return_value=(118.00, "USD")), \
         patch.object(ci, "_get_fx_rate", side_effect=lambda f, t: 1.40 if (f, t) == ("USD", "CAD") else 0.0), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("ISRG", shares=74)

    # 74 × $118 USD = $8,732 → × 1.40 = $12,225 CAD
    assert r["proposed_size"]["dollars"] == "$12,225"
    assert r["proposed_size"]["basis"] == "74 shares ≈$12,225 CAD (at $118.00 USD)"
    # 6.1% of the CAD book, not the 4.4% face value would have claimed
    assert r["proposed_size"]["pct_of_current_portfolio"] == "6.1%"


def test_shares_size_takes_currency_from_the_holding_without_a_quote():
    """A held name carries its currency — no quote round-trip needed."""
    ctx = _ctx(total=200_000.0, base="CAD")
    ctx["holdings"][0]["currency"] = "USD"  # AAPL
    quote_calls = []

    def _boom(symbol):
        quote_calls.append(symbol)
        return (None, "")

    with patch.object(ci, "_decision_context", return_value=ctx), \
         patch.object(ci, "_impact_returns",
                      return_value=(_returns_frame(["AAPL", "MSFT", "SPY"]), ["AAPL", "MSFT", "SPY"])), \
         patch.object(ci, "_candidate_quote", side_effect=_boom), \
         patch.object(ci, "_get_fx_rate", side_effect=lambda f, t: 1.40 if (f, t) == ("USD", "CAD") else 0.0), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("AAPL", shares=40, entry=200.0)

    assert quote_calls == []  # the holding answered both price and currency
    assert r["proposed_size"]["dollars"] == "$11,200"  # 40 × $200 USD × 1.40
    assert r["proposed_size"]["basis"] == "40 shares ≈$11,200 CAD (at $200.00 USD)"


def test_shares_size_abstains_when_currency_or_rate_is_missing():
    """No currency, or no rate, means no honest number — error, never a guess.

    A wrong size here is not a cosmetic slip: it propagates into the IPS caps
    and the CVaR delta, so the preview would understate a breach it exists to
    catch.
    """
    ctx = _ctx(total=200_000.0, base="CAD")
    returns = (_returns_frame(["AAPL", "MSFT", "ISRG", "SPY"]), ["AAPL", "MSFT", "ISRG", "SPY"])

    # (a) quote omits the currency
    with patch.object(ci, "_decision_context", return_value=ctx), \
         patch.object(ci, "_impact_returns", return_value=returns), \
         patch.object(ci, "_candidate_quote", return_value=(118.00, "")), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("ISRG", shares=74)
    assert "error" in r and "currency" in r["error"]

    # (b) currency known, but the pair has no rate
    with patch.object(ci, "_decision_context", return_value=ctx), \
         patch.object(ci, "_impact_returns", return_value=returns), \
         patch.object(ci, "_candidate_quote", return_value=(118.00, "USD")), \
         patch.object(ci, "_get_fx_rate", return_value=0.0), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("ISRG", shares=74)
    assert "error" in r and "USD→CAD" in r["error"]


def test_shares_size_unlabelled_price_still_priced_for_a_usd_profile():
    """USD base needs no conversion — abstaining there would break the common path."""
    ctx = _ctx(total=200_000.0, base="USD")
    with patch.object(ci, "_decision_context", return_value=ctx), \
         patch.object(ci, "_impact_returns",
                      return_value=(_returns_frame(["AAPL", "MSFT", "ISRG", "SPY"]), ["AAPL", "MSFT", "ISRG", "SPY"])), \
         patch.object(ci, "_candidate_quote", return_value=(118.00, "")), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("ISRG", shares=74)

    assert r["proposed_size"]["dollars"] == "$8,732"
    assert r["proposed_size"]["basis"] == "74 shares × $118.00"


def test_missing_symbol_and_bad_context():
    assert "error" in ci.preview_candidate_impact("")
    with patch.object(ci, "_decision_context", return_value={"error": "no portfolio"}):
        assert "error" in ci.preview_candidate_impact("NVDA")
    with patch.object(ci, "_decision_context", return_value={"total_value_base": 0}):
        assert "error" in ci.preview_candidate_impact("NVDA")


# ---------------------------------------------------------------------------
# Risk-delta composition
# ---------------------------------------------------------------------------

def test_full_report_shape_and_deltas():
    frame = _returns_frame(["AAPL", "MSFT", "NVDA", "SPY"], seed=1)
    with patch.object(ci, "_decision_context", return_value=_ctx()), \
         patch.object(ci, "_impact_returns", return_value=(frame, ["AAPL", "MSFT", "NVDA", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("NVDA", size_pct=5.0)

    rd = r["risk_deltas"]
    assert "error" not in rd
    # candidate excluded from the "current" book (not held) but present in proposed
    assert rd["symbols_analyzed"] == ["AAPL", "MSFT", "NVDA"]
    assert set(rd["volatility"]) == {"current", "proposed", "delta"}
    assert rd["beta"]["current"] is not None and rd["beta"]["proposed"] is not None
    assert "cvar_95_annual" in rd
    assert rd["cvar_95_annual"]["delta_dollars"].startswith("$")
    # account-location rows are stripped from the preview
    assert all(row["check"] != "account location" for row in r["ips_checks"]["rows"])
    # headline is assembled
    assert "NVDA" in r["headline"] and "IPS" in r["headline"]


def test_ips_flags_surface_in_headline():
    frame = _returns_frame(["AAPL", "MSFT", "NVDA", "SPY"], seed=2)
    breach = _stub_precheck(
        rows=[{"trade": "BUY NVDA", "check": "position cap", "computed": "0% → 45% post-trade",
               "limit": "≤10%", "verdict": "FAIL"}],
        violations=["IPS Pre-check FAIL: BUY NVDA — post-trade position 45% exceeds the 10% single name cap."],
    )
    with patch.object(ci, "_decision_context", return_value=_ctx()), \
         patch.object(ci, "_impact_returns", return_value=(frame, ["AAPL", "MSFT", "NVDA", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=breach):
        r = ci.preview_candidate_impact("NVDA", size_pct=45.0)
    assert r["ips_checks"]["flags"]
    assert "IPS flag" in r["headline"]


def test_candidate_already_held_does_not_duplicate_column():
    # NVDA is already a holding: the current book includes it, proposed increases it.
    ctx = _ctx()
    ctx["holdings"].append({"symbol": "NVDA", "value_base": 20_000.0, "is_cash_or_pension": False})
    frame = _returns_frame(["AAPL", "MSFT", "NVDA", "SPY"], seed=3)
    with patch.object(ci, "_decision_context", return_value=ctx), \
         patch.object(ci, "_impact_returns", return_value=(frame, ["AAPL", "MSFT", "NVDA", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("NVDA", size_pct=5.0)
    rd = r["risk_deltas"]
    assert rd["symbols_analyzed"].count("NVDA") == 1
    assert rd["symbols_analyzed"] == ["AAPL", "MSFT", "NVDA"]


def test_candidate_without_history_reports_partial():
    # candidate missing from valid symbols → risk error, but report still returns
    frame = _returns_frame(["AAPL", "MSFT", "SPY"], seed=4)
    with patch.object(ci, "_decision_context", return_value=_ctx()), \
         patch.object(ci, "_impact_returns", return_value=(frame, ["AAPL", "MSFT", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("ZZZZ", size_pct=5.0)
    assert "error" in r["risk_deltas"]
    assert "ips_checks" in r  # compliance block still present


def test_dropped_holding_noted_and_excluded():
    # One holding (MSFT) has no price history → excluded from the risk math, noted.
    frame = _returns_frame(["AAPL", "NVDA", "SPY"], seed=5)
    with patch.object(ci, "_decision_context", return_value=_ctx()), \
         patch.object(ci, "_impact_returns", return_value=(frame, ["AAPL", "NVDA", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("NVDA", size_pct=5.0)
    rd = r["risk_deltas"]
    assert "MSFT" in rd["data_note"]
    assert "MSFT" not in rd["symbols_analyzed"]


def test_beta_omitted_without_spy():
    frame = _returns_frame(["AAPL", "MSFT", "NVDA"], seed=6)  # no SPY column
    with patch.object(ci, "_decision_context", return_value=_ctx()), \
         patch.object(ci, "_impact_returns", return_value=(frame, ["AAPL", "MSFT", "NVDA"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("NVDA", size_pct=5.0)
    assert "beta" not in r["risk_deltas"]
    assert "volatility" in r["risk_deltas"]  # the rest still computes


# ---------------------------------------------------------------------------
# Numeric sanity: an uncorrelated add at weight w should move vol predictably
# ---------------------------------------------------------------------------

def test_diversifying_add_reduces_or_holds_volatility():
    # Build a candidate that is negatively correlated with the book → adding a
    # slice should not raise (and typically lowers) modeled volatility.
    n = 300
    idx = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(7)
    market = rng.normal(0.0005, 0.012, n)
    frame = pd.DataFrame({
        "AAPL": market + rng.normal(0, 0.002, n),
        "MSFT": market + rng.normal(0, 0.002, n),
        "HEDGE": -market + rng.normal(0, 0.002, n),   # inverse of the book
        "SPY": market,
    }, index=idx)
    frame.attrs["fx"] = {"base_currency": "USD", "converted": [], "unavailable": []}
    with patch.object(ci, "_decision_context", return_value=_ctx()), \
         patch.object(ci, "_impact_returns", return_value=(frame, ["AAPL", "MSFT", "HEDGE", "SPY"])), \
         patch("tools.ips_precheck.run_ips_precheck", return_value=_stub_precheck()):
        r = ci.preview_candidate_impact("HEDGE", size_pct=20.0)
    rd = r["risk_deltas"]
    cur = float(rd["volatility"]["current"].rstrip("%"))
    prop = float(rd["volatility"]["proposed"].rstrip("%"))
    assert prop < cur  # the inverse asset diversifies the book down
    assert float(rd["candidate_correlation_to_portfolio"]) < 0
    assert "diversifies" in rd["diversification_note"]
