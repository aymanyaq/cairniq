from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from tools.backtesting import backtest_strategy


@patch("yfinance.Ticker")
def test_backtest_rsi_strategy(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    # Need many periods to calculate RSI(14) properly and trigger signals
    # Prices: 100 -> 200 -> 50 -> 200
    prices = list(np.linspace(100, 200, 30)) + list(np.linspace(200, 50, 30)) + list(np.linspace(50, 200, 30))
    dates = pd.date_range(end="2024-01-01", periods=len(prices))
    mock_ticker.history.return_value = pd.DataFrame({"Close": prices}, index=dates)

    result = backtest_strategy("rsi", ["AAPL"], params={"buy_threshold": 40, "sell_threshold": 60})

    assert "total_trades" in result or "message" in result
    if "total_trades" in result:
        assert result["symbol"] == "AAPL"

@patch("yfinance.Ticker")
def test_backtest_lump_sum(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    # Simple growth: 100 to 200
    prices = np.linspace(100, 200, 20)
    dates = pd.date_range(end="2024-01-01", periods=20)
    mock_ticker.history.return_value = pd.DataFrame({"Close": prices}, index=dates)

    result = backtest_strategy("lump_sum", ["AAPL"], params={"initial_capital": 10000})

    assert result["strategy"] == "LUMP_SUM"
    assert "100.00%" in result["total_return"]

@patch("yfinance.Ticker")
def test_backtest_dca(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    # Create 2 years of monthly data
    dates = pd.date_range(end="2024-01-01", periods=24, freq="ME")
    prices = np.linspace(100, 200, 24)
    mock_ticker.history.return_value = pd.DataFrame({"Close": prices}, index=dates)

    result = backtest_strategy("dca", ["AAPL"], period="2y", params={"initial_capital": 2400})

    assert result["strategy"] == "DCA"
    assert "final_value" in result

def test_backtest_invalid_strategy():
    result = backtest_strategy("invalid_strategy", ["AAPL"])
    assert "error" in result
    assert "Unknown strategy" in result["error"]

@patch("yfinance.Ticker")
def test_backtest_insufficient_data(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.history.return_value = pd.DataFrame({"Close": [100, 101]}, index=pd.date_range(end="2024-01-01", periods=2))

    result = backtest_strategy("rsi", ["AAPL"])
    assert "error" in result
    assert "Insufficient historical data" in result["error"]
