from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def test_analyze_correlation_high():
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Linear growth for both
        prices = np.linspace(100, 200, 100)
        mock_ticker.history.return_value = pd.DataFrame({"Close": prices}, index=pd.date_range(end="2024-01-01", periods=100))

        from tools.correlation import analyze_correlation
        res = analyze_correlation.__wrapped__(["AAPL", "MSFT"])

        assert res["average_correlation"] == "1.00"
        assert res["diversification_assessment"] == "Concentrated"

def test_analyze_correlation_low():
    with patch("yfinance.Ticker") as mock_ticker_class:
        # Create two series with zero correlation
        # s1: [100, 110, 100, 110...] -> returns [0.1, -0.09, 0.1...]
        # s2: [100, 100, 110, 110...] -> returns [0, 0.1, 0...]

        mock_ticker1 = MagicMock()
        s1 = [100, 110, 100, 110] * 10
        mock_ticker1.history.return_value = pd.DataFrame({"Close": s1}, index=pd.date_range(end="2024-01-01", periods=len(s1)))

        mock_ticker2 = MagicMock()
        s2 = [100, 100, 110, 110] * 10
        mock_ticker2.history.return_value = pd.DataFrame({"Close": s2}, index=pd.date_range(end="2024-01-01", periods=len(s2)))

        mock_ticker_class.side_effect = [mock_ticker1, mock_ticker2]

        from tools.correlation import analyze_correlation
        res = analyze_correlation.__wrapped__(["AAPL", "GLD"])

        assert float(res["average_correlation"]) < 0.5
        assert res["diversification_assessment"] == "Well diversified"

def test_analyze_correlation_insufficient_input():
    from tools.correlation import analyze_correlation
    res = analyze_correlation.__wrapped__(["AAPL"])
    assert "error" in res

def test_analyze_correlation_insufficient_data():
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        from tools.correlation import analyze_correlation
        res = analyze_correlation.__wrapped__(["AAPL", "MSFT"])
        assert "error" in res
