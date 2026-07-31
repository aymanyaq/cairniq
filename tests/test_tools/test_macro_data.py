from unittest.mock import MagicMock, patch

from tools.macro_data import _fetch_single_ticker


@patch("yfinance.Ticker")
def test_fetch_single_ticker_success(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    import pandas as pd
    hist_data = pd.DataFrame({"Close": [100.0, 105.0]}, index=pd.date_range(end="2024-01-01", periods=2))
    mock_ticker.history.return_value = hist_data

    res = _fetch_single_ticker("Gold", "GC=F")
    assert res["price"] == "105.00"
    assert res["change_pct"] == "+5.00%"
    assert res["trend"] == "🟢"

@patch("yfinance.Ticker")
def test_fetch_single_ticker_fallback(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    # history fails
    mock_ticker.history.return_value = MagicMock(empty=True)
    # fast_info works
    mock_ticker.fast_info = {"last_price": 2000.0, "previous_close": 1980.0}

    res = _fetch_single_ticker("Gold", "GC=F")
    assert res["price"] == "2,000.00"
    assert res["change_pct"] == "+1.01%"

@patch("tools.macro_data._fetch_single_ticker")
def test_get_global_market_snapshot(mock_fetch):
    mock_fetch.return_value = {"name": "Test", "price": "100.00", "change_pct": "0.00%"}

    # Bypass cache
    with patch("tools.macro_data.cached", lambda **kwargs: lambda f: f):
        from importlib import reload

        import tools.macro_data
        reload(tools.macro_data)

        res = tools.macro_data.get_global_market_snapshot()
        assert res["source"] == "Yahoo Finance"
        assert len(res["indices"]) > 0
        assert res["indices"][0]["name"] == "S&P 500 (US)"
