"""
Tests for the live-review remediation batch:
  1. Cross-listing/CDR equivalence — a US candidate whose Canadian twin is
     held (MA vs MA.TO) must be flagged as overlap, never "not held".
  2. Falling-knife guard — a Lagging-sector candidate can't surface as a
     clean dip-buy: penalty, risk flag, watchlist-only.
  3. Flow provenance — proxy-derived flow signals carry structured evidence
     and keep the "proxy" qualifier in their summary strings.
"""

import pytest

import tools.opportunity_scanner as opp
from tools.ticker_equivalence import economic_key, find_equivalent_holding

# ---------------------------------------------------------------------------
# 1. Ticker equivalence
# ---------------------------------------------------------------------------

def test_cdr_twin_shares_economic_key():
    assert economic_key("MA.TO") == economic_key("MA") == "MA"
    assert economic_key("NVDA.NE") == "NVDA"
    assert economic_key("RDDT.TO") == "RDDT"


def test_interlisted_same_root_matches_both_directions():
    assert find_equivalent_holding("SU.TO", ["SU"]) == "SU"
    assert find_equivalent_holding("SU", ["SU.TO"]) == "SU.TO"
    assert find_equivalent_holding("RY", ["RY.TO", "VDY.TO"]) == "RY.TO"


def test_no_naive_suffix_stripping():
    # T.TO is Telus (US: TU), NOT AT&T.
    assert economic_key("T.TO") == "TU"
    assert find_equivalent_holding("T", ["T.TO"]) is None
    assert find_equivalent_holding("TU", ["T.TO"]) == "T.TO"
    # MG.TO is Magna (US: MGA), NOT Mistras Group.
    assert find_equivalent_holding("MG", ["MG.TO"]) is None
    assert find_equivalent_holding("MGA", ["MG.TO"]) == "MG.TO"


def test_unknown_canadian_root_never_crosses_markets():
    assert find_equivalent_holding("WEED", ["WEED.TO"]) is None
    assert economic_key("WEED.TO") == "WEED.TO"


def test_exact_listing_is_not_reported_as_twin():
    assert find_equivalent_holding("MA.TO", ["MA.TO"]) is None


# ---------------------------------------------------------------------------
# 2. Portfolio-fit overlay: economic twin detection
# ---------------------------------------------------------------------------

def _fit_ctx(*symbols, value_usd=5000.0):
    return {
        "holdings": [{"symbol": s, "value_usd": value_usd} for s in symbols],
        "total_value_usd": 700_000.0,
    }


def test_portfolio_fit_flags_cdr_twin_even_when_sector_unresolvable(monkeypatch):
    # MA.TO resolving to "Unknown" is the exact path that mislabeled MA as
    # "not held": the twin never entered the sector loop.
    monkeypatch.setattr(opp, "_get_sector_for_ticker", lambda s, fund=None: "Unknown")
    out = opp._portfolio_fit_adjustment("MA", "Financial Services", _fit_ctx("MA.TO", "CM.TO"))

    assert out["portfolio_fit"]["economic_equivalent_held"] == "MA.TO"
    assert any("twin" in f.lower() for f in out["risk_flags"])
    assert out["risk_adjust"] == 0.0  # overlap informs sizing, never punishes score


def test_portfolio_fit_exact_held_checked_across_all_holdings(monkeypatch):
    monkeypatch.setattr(opp, "_get_sector_for_ticker", lambda s, fund=None: "Unknown")
    out = opp._portfolio_fit_adjustment("OVV.TO", "Energy", _fit_ctx("OVV.TO"))

    assert out["portfolio_fit"]["candidate_already_held"] is True
    assert out["portfolio_fit"]["economic_equivalent_held"] is None


def test_portfolio_fit_no_twin_no_flags(monkeypatch):
    monkeypatch.setattr(opp, "_get_sector_for_ticker", lambda s, fund=None: "Unknown")
    out = opp._portfolio_fit_adjustment("CBOE", "Financial Services", _fit_ctx("CM.TO", "RY.TO"))

    assert out["portfolio_fit"]["economic_equivalent_held"] is None
    assert out["risk_flags"] == []


# ---------------------------------------------------------------------------
# 3. Falling-knife guard in _deep_score_v2
# ---------------------------------------------------------------------------

@pytest.fixture
def _no_live_earnings(monkeypatch):
    monkeypatch.setattr(
        "tools.market_mechanics.predict_earnings_surprise",
        lambda symbol: {"beat_rate": "78%"},
    )


def _energy_theme(trend):
    return {
        "theme": "Energy",
        "sector": "Energy",
        "canonical_sector": "Energy",
        "theme_score": 0.40,
        "cycle_stage": "late",
        "drivers": [],
        "trend": trend,
    }


def _value_fund():
    return {
        "symbol": "SU",
        "forward_pe": 9.8,
        "trailing_pe": 14.5,
        "analyst_target": 100,
        "current_price": 76,
        "revenue_growth": 0.05,
        "earnings_growth": 0.08,
        "profit_margin": 0.09,
        "free_cashflow": 4_000_000,
        "total_debt": 8_000_000,
        "total_cash": 2_000_000,
        "recommendation": "buy",
        "description": "Integrated energy",
        "sector_yf": "Energy",
        "industry": "Oil & Gas Integrated",
        "news_headlines": [],
    }


def _value_tech():
    return {
        "price": 76,
        "rsi": 36,
        "sma50": 82,
        "above_sma50": False,
        "golden_cross": False,
        "drawdown_pct": -18,
        "month_return": -7,
        "three_month_return": -15,
        "vol_spike": 1.0,
    }


def _score_with_trend(trend):
    return opp._deep_score_v2(
        "SU",
        _value_fund(),
        _value_tech(),
        trend,
        {},
        [],
        theme_context=_energy_theme(trend),
        rs_alpha=-10,
        flow_data={},
        setup_data={},
        apply_entry_gate=False,
    )


def test_lagging_sector_triggers_falling_knife_guard(_no_live_earnings):
    result = _score_with_trend("Lagging 🔴")

    assert result["watchlist_only"] is True
    assert any("Falling-knife" in f for f in result["risk_flags"])
    assert result["promotion_condition"]
    assert result["conviction"] not in ("Exceptional", "High Conviction")
    assert any(a["type"] == "falling_knife" for a in result["risk_adjustments"])


def test_lagging_scores_below_identical_improving_name(_no_live_earnings):
    lagging = _score_with_trend("Lagging 🔴")
    improving = _score_with_trend("Improving 🔵")

    assert lagging["score"] < improving["score"]
    assert improving["watchlist_only"] is False
    assert improving["promotion_condition"] is None


# ---------------------------------------------------------------------------
# 4. Flow evidence provenance
# ---------------------------------------------------------------------------

def test_flow_confirmations_carry_structured_proxy_evidence(monkeypatch):
    monkeypatch.setattr(
        "tools.dark_pool.scan_dark_pool_proxy",
        lambda symbol: {
            "alerts_count": 2,
            "alerts": [
                {"signature": "DARK POOL PRINT (Hidden)", "time": "10:31",
                 "volume": "1,200,000", "price": "$247.10", "magnitude": "6.2x Normal"},
                {"signature": "AGGRESSIVE BUY", "time": "14:02",
                 "volume": "900,000", "price": "$247.60", "magnitude": "4.1x Normal"},
            ],
        },
    )
    monkeypatch.setattr("tools.options.check_whale_accumulation", lambda symbol: {"count": 2})
    monkeypatch.setattr("tools.options.scan_unusual_activity", lambda symbol: {"alerts": ["BULLISH CALL SWEEP"]})

    flow = opp._flow_confirmation_for_symbol("CBOE", insider_signal="🟢 Insiders BUYING recently")

    assert flow["flow_signal_count"] == 4
    signals = {e["signal"] for e in flow["flow_evidence"]}
    assert signals == {"dark_pool_proxy", "itm_call_sweep_proxy", "unusual_options_proxy", "insider_buying"}

    dark = next(e for e in flow["flow_evidence"] if e["signal"] == "dark_pool_proxy")
    assert "proxy" in dark["source"]
    assert dark["count"] == 2
    assert dark["prints"][0]["time"] == "10:31"

    # The summary strings must keep the proxy qualifier so downstream prose
    # can't upgrade them to "dark-pool block prints confirm accumulation".
    assert any("proxy" in c for c in flow["flow_confirmations"])
