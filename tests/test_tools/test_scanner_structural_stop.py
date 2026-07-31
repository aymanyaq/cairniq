"""Every scan type must carry a structural stop, not just broad scans.

`_setup_check_parallel` (screener.check_setup) is gated on `is_broad`, so themed
scans — "Value & Defensive", mega-cap, growth — reached MarketAnalyst with no
stop and rendered "—" on every row. The market_dip lens requires a structural
stop per pick and judge Rule 4 flags unanchored ones, so the model filled the
gap with invented round numbers and was penalised for it. The stop is
now derived in _compute_technicals_batch from OHLC already downloaded.
"""
import numpy as np
import pandas as pd
import pytest

import tools.opportunity_scanner as sc


def _ohlc(n=120, seed=7):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0.15, 1.2, n)))
    return pd.DataFrame({
        "Close": close,
        "High": close + np.abs(rng.normal(0.8, 0.3, n)),
        "Low": close - np.abs(rng.normal(0.8, 0.3, n)),
        "Volume": pd.Series(rng.integers(1_000_000, 3_000_000, n).astype(float)),
    })


def _reference_stop(df):
    """screener.check_setup's basis, computed independently here.

    Deliberately a re-implementation rather than an import: it pins the two
    code paths to the SAME documented basis, so a change to either without the
    other fails loudly instead of quietly producing two different stops for the
    same ticker depending on which scan surfaced it.
    """
    close, high, low = df["Close"], df["High"], df["Low"]
    current_price = float(close.iloc[-1])
    prev_close = close.shift()
    true_range = (
        (high - low).to_frame("hl")
        .join((high - prev_close).abs().to_frame("hc"))
        .join((low - prev_close).abs().to_frame("lc"))
        .max(axis=1)
    )
    atr_14 = float(true_range.rolling(window=14).mean().iloc[-1])
    swing_low_20 = float(low.tail(20).min())
    candidate = min(current_price - 2 * atr_14, swing_low_20)
    if not (candidate < current_price):
        candidate = swing_low_20 if swing_low_20 < current_price else current_price * 0.92
    return round(candidate, 2)


@pytest.mark.parametrize("seed", [7, 11, 23, 101])
def test_stop_matches_the_check_setup_basis(seed):
    df = _ohlc(seed=seed)

    tech = sc._compute_technicals_batch(df, ["TEST"])["TEST"]

    assert tech["stop_loss"] == _reference_stop(df)
    assert "20d swing low" in tech["stop_basis"]
    assert "2x ATR" in tech["stop_basis"]


@pytest.mark.parametrize("seed", [7, 11, 23, 101])
def test_stop_always_sits_below_price_with_positive_risk(seed):
    tech = sc._compute_technicals_batch(_ohlc(seed=seed), ["TEST"])["TEST"]

    assert 0 < tech["stop_loss"] < tech["price"]
    assert tech["risk_pct"] > 0
    expected = round((tech["price"] - tech["stop_loss"]) / tech["price"] * 100, 1)
    assert tech["risk_pct"] == expected


def test_missing_high_low_degrades_to_no_stop_not_a_crash():
    """Close-only frames must still yield technicals, just without a stop."""
    df = _ohlc()[["Close", "Volume"]]

    tech = sc._compute_technicals_batch(df, ["TEST"])["TEST"]

    assert tech["stop_loss"] is None
    assert tech["risk_pct"] is None
    assert tech["price"] > 0          # the rest of the screen is unaffected
    assert tech["rsi"] is not None


def test_themed_scan_pick_inherits_the_stop_when_the_setup_gate_never_ran():
    """The actual regression: setup_data is {} on every non-broad scan."""
    tech = sc._compute_technicals_batch(_ohlc(), ["TEST"])["TEST"]
    setup_data = {}  # what _setup_check_parallel returns when is_broad is False

    stop = sc._safe_float(setup_data.get("stop_loss")) or sc._safe_float(tech.get("stop_loss"))
    risk = sc._safe_float(setup_data.get("risk_pct")) or sc._safe_float(tech.get("risk_pct"))

    assert stop == tech["stop_loss"]
    assert risk == tech["risk_pct"]


def test_setup_gate_wins_when_it_did_run():
    """Broad scans keep check_setup's value — the fallback must not override it."""
    tech = sc._compute_technicals_batch(_ohlc(), ["TEST"])["TEST"]
    setup_data = {"stop_loss": 12.34, "stop_basis": "from check_setup", "risk_pct": 4.2}

    stop = sc._safe_float(setup_data.get("stop_loss")) or sc._safe_float(tech.get("stop_loss"))
    basis = setup_data.get("stop_basis") or tech.get("stop_basis")

    assert stop == 12.34
    assert basis == "from check_setup"


def test_tech_cache_prefix_is_versioned_past_the_stopless_schema():
    """A same-day cache hit under the old key would serve stop-less entries."""
    assert sc._TECH_CACHE_PREFIX != "funnel_tech_"
