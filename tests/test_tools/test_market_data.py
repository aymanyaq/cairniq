import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.market_data import (
    _safe_yf_call,
    get_etf_holdings,
    get_fundamentals_detailed,
    get_historical_performance,
    get_realtime_quote,
    get_stock_data,
    suppress_stderr,
)


def test_safe_yf_call():
    # Test successful call
    func = MagicMock(return_value="success")
    assert _safe_yf_call(func) == "success"
    assert func.call_count == 1

    # Test closed file retry
    func_fails = MagicMock(side_effect=[ValueError("I/O operation on closed file"), "success_after_retry"])
    with patch("tools.market_data.time.sleep"):
        assert _safe_yf_call(func_fails) == "success_after_retry"
        assert func_fails.call_count == 2

    # Test unhandled exception
    func_hard_fail = MagicMock(side_effect=RuntimeError("Some error"))
    with pytest.raises(RuntimeError):
        _safe_yf_call(func_hard_fail)

@patch("tools.fmp_api.get_fmp_analyst_estimates")
@patch("tools.fmp_api.get_fmp_dcf")
@patch("tools.fmp_api.get_fmp_profile")
@patch("tools.fmp_api.get_fmp_quote")
def test_get_stock_data_fmp(mock_quote, mock_profile, mock_dcf, mock_analyst):
    # Mock FMP responses
    mock_quote.return_value = {"price": 150.0, "market_cap": 2500000000000, "pe": 25.5, "year_high": 160.0, "year_low": 120.0}
    mock_profile.return_value = {"sector": "Technology", "industry": "Consumer Electronics", "currency": "USD", "description": "Apple Inc."}
    mock_dcf.return_value = 160.0
    mock_analyst.return_value = {"target_mean": 170.0, "consensus": "Buy"}

    # Needs clear cache to run
    from tools.cache import _cache
    _cache.clear()

    with patch("tools.market_data.yf.Ticker"):
        data = get_stock_data("AAPL")

        assert data["symbol"] == "AAPL"
        assert data["current_price"] == "$150.00"
        assert data["sector"] == "Technology"
        assert "Hybrid" in data["source"] # Due to the enrichment step

def test_get_stock_data_manual(monkeypatch):
    class MockNodes:
        def get(self, sym, default=None):
            return {"asset_type": "Private"} if sym == "PRIVATE_FUND" else default

    monkeypatch.setattr("tools.graph_memory.graph_memory.graph.nodes", MockNodes())
    monkeypatch.setattr("tools.daily_cache.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.daily_cache.set_cached", lambda *args, **kwargs: None)

    # Should skip API and return static format
    data = get_stock_data("PRIVATE_FUND")
    assert data["symbol"] == "PRIVATE_FUND"
    assert data["source"] == "Manual Entry (Skipped API)"

@patch("tools.market_data.yf.Ticker")
def test_get_historical_performance(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    # Create fake history dataframe
    dates = pd.date_range("2013-01-01", periods=252*10+10, freq="B")
    data = {"Close": [100.0] * len(dates)}
    data["Close"][-1] = 200.0 # Double in 10 years
    data["Close"][-252] = 180.0 # 1 year ago
    data["Close"][-252*5] = 150.0 # 5 years ago

    df = pd.DataFrame(data, index=dates)
    mock_ticker.history.return_value = df

    perf = get_historical_performance("AAPL")
    assert perf["symbol"] == "AAPL"
    assert "performance" in perf
    assert "1_year" in perf["performance"]
    assert "10_year" in perf["performance"]

@patch("tools.market_data.yf.Ticker")
@patch("tools.fmp_api.get_fmp_etf_holdings")
def test_get_etf_holdings_fmp(mock_fmp, mock_ticker):
    mock_fmp.return_value = [{"asset": "MSFT", "weightPercentage": 15.5}, {"asset": "AAPL", "weightPercentage": 14.2}]

    holdings = get_etf_holdings("QQQ")
    assert holdings["symbol"] == "QQQ"
    assert holdings["count"] == 2
    assert "MSFT: 15.50%" in holdings["top_holdings"]

@patch("tools.market_data.get_stock_data")
def test_get_realtime_quote(mock_get_stock):
    mock_get_stock.return_value = {"symbol": "AAPL", "current_price": "$150.00", "recent_trend": "5.00%", "market_cap": "$2,500,000,000,000"}

    quote = get_realtime_quote("AAPL")
    assert quote["symbol"] == "AAPL"
    assert quote["price"] == "$150.00"
    assert quote["change"] == "5.00%"

@patch("tools.market_data.yf.Ticker")
def test_get_fundamentals_detailed(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {
        "trailingPE": 25.5,
        "profitMargins": 0.25,
        "revenueGrowth": 0.10
    }

    funds = get_fundamentals_detailed("AAPL")
    assert funds["symbol"] == "AAPL"
    assert funds["valuation"]["pe_ratio"] == 25.5
    assert funds["profitability"]["profit_margin"] == "25.00%"


def test_suppress_stderr_survives_concurrent_overlap():
    """Regression: sys.stderr is one process-wide object, but suppress_stderr()
    is called from ThreadPoolExecutor pools all over the app (health checks,
    screeners, parallel tool execution). The old implementation had each call
    independently open its own devnull and save/restore sys.stderr — two
    overlapping calls could interleave so one thread restored a devnull that a
    concurrent thread had already closed, permanently wedging sys.stderr closed
    for the rest of the process's life (surfaced in production as "ValueError:
    I/O operation on closed file" in seemingly unrelated code, since any later
    stderr write anywhere would then fail)."""
    original_stderr = sys.stderr
    errors = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        try:
            for _ in range(10):
                with suppress_stderr():
                    time.sleep(0.001)
                    print("noisy library output", file=sys.stderr)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert sys.stderr is original_stderr
    assert not sys.stderr.closed
