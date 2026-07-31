import asyncio
import types
from datetime import datetime, timedelta

import pandas as pd


def _hist(days=45, start=100.0):
    idx = pd.date_range("2026-01-01", periods=days, freq="B")
    return pd.DataFrame(
        {
            "Close": [start + i for i in range(days)],
            "Open": [start + i - 0.5 for i in range(days)],
            "High": [start + i + 1 for i in range(days)],
            "Low": [start + i - 1 for i in range(days)],
            "Volume": [1000 + i for i in range(days)],
        },
        index=idx,
    )


class _Ticker:
    def __init__(self, info=None, calendar=None, history=None, news=None, funds_data=None, fail_info=False):
        self._info = info or {}
        self.calendar = calendar
        self._history = history if history is not None else _hist()
        self.news = news or []
        self.funds_data = funds_data
        self.fail_info = fail_info
        self.fast_info = types.SimpleNamespace(
            last_price=22.0,
            previous_close=21.5,
            currency="USD",
            exchange="NMFQS",
            market_cap=123456,
        )

    @property
    def info(self):
        if self.fail_info:
            raise ValueError("I/O operation on closed file")
        return self._info

    def history(self, *args, **kwargs):
        return self._history


def test_market_data_fmp_path_adds_calendar_and_hybrid_enrichment(monkeypatch):
    import tools.market_data as md

    monkeypatch.setattr("tools.fmp_api.get_fmp_quote", lambda symbol: {
        "price": 150.0,
        "market_cap": 2500000,
        "pe": 25.5,
        "year_high": 200.0,
        "year_low": 100.0,
    })
    monkeypatch.setattr("tools.fmp_api.get_fmp_profile", lambda symbol: {
        "sector": "Technology",
        "industry": "Software",
        "currency": "USD",
        "description": "A business",
    })
    monkeypatch.setattr("tools.fmp_api.get_fmp_dcf", lambda symbol: 175.0)
    monkeypatch.setattr("tools.fmp_api.get_fmp_analyst_estimates", lambda symbol: {
        "target_mean": 180.0,
        "consensus": "Buy",
    })

    future_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
    ticker = _Ticker(
        info={"beta": 1.65},
        calendar={"Earnings Date": [future_date]},
        history=_hist(50, 100),
        news=[{"content": {"title": "New product", "provider": {"displayName": "Wire"}}}],
    )
    monkeypatch.setattr(md.yf, "Ticker", lambda symbol: ticker)

    result = md.get_stock_data.__wrapped__("AAPL")

    assert result["source"] == "Hybrid (FMP + Free Tools)"
    assert result["earnings_date"] == future_date
    assert result["dcf_valuation"] == "$175.00"
    assert result["volatility_warning"].startswith("⚠️ HIGH VOLATILITY")
    assert "New product" in result["recent_news"]
    assert result["recent_trend"] != "N/A"


def test_market_data_polygon_and_yfinance_fallback_paths(monkeypatch):
    import tools.market_data as md

    monkeypatch.setattr("tools.fmp_api.get_fmp_quote", lambda symbol: {"error": "no fmp"})
    monkeypatch.setattr("tools.polygon_api.get_polygon_quote", lambda symbol: {
        "price": 88.0,
        "day_high": 90.0,
        "day_low": 85.0,
    })
    monkeypatch.setattr("tools.polygon_api.get_polygon_profile", lambda symbol: {
        "market_cap": 999,
        "industry": "Finance",
        "currency": "usd",
        "description": "Polygon profile",
    })
    def failing_ticker(symbol):
        raise RuntimeError("yfinance failed")
    monkeypatch.setattr(md.yf, "Ticker", failing_ticker)

    polygon_result = md.get_stock_data.__wrapped__("XYZ")
    assert polygon_result["source"] == "Polygon.io (Fallback)"
    assert polygon_result["currency"] == "USD"

    monkeypatch.setattr("tools.polygon_api.get_polygon_quote", lambda symbol: {"error": "no polygon"})
    seen_symbols = []
    calendar_df = pd.DataFrame({"Earnings Date": [(datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d")]})

    def fake_ticker(symbol):
        seen_symbols.append(symbol)
        return _Ticker(
            info={
                "currentPrice": 70.0,
                "financialCurrency": "CAD",
                "trailingPE": 12.0,
                "marketCap": 100000,
                "fiftyTwoWeekHigh": 100.0,
                "fiftyTwoWeekLow": 50.0,
                # A PERCENT, as the provider sends it (AAPL: 0.32 for 0.32%).
                # Was 0.03 here, i.e. the fraction the reader wrongly assumed —
                # which is why the suite stayed green over a 100x error.
                "dividendYield": 3.0,
                "quoteType": "EQUITY",
                "beta": 1.7,
                "targetMeanPrice": 75.0,
                "recommendationKey": "buy",
                "sector": "Financial Services",
                "industry": "Banks",
                "longBusinessSummary": "Canadian bank",
            },
            calendar=calendar_df,
            history=_hist(50, 50),
        )

    monkeypatch.setattr(md.yf, "Ticker", fake_ticker)
    yf_result = md.get_stock_data.__wrapped__("RY CANADA")

    assert seen_symbols[0] == "RY.TO"
    assert yf_result["source"] == "yfinance (fallback)"
    assert yf_result["52_week_position"] == "40.0%"
    assert yf_result["dividend_yield"] == "3.00%"
    assert yf_result["earnings_warning"].startswith("📅 EARNINGS")


def test_market_data_fast_info_absolute_and_error_fallbacks(monkeypatch):
    import tools.market_data as md

    # The error path below retries with a real 3.0s backoff. What is under test is
    # the fallback *selection*, not the wait — tests/test_tools/test_market_data.py
    # already patches this the same way for the same reason.
    monkeypatch.setattr(md.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr("tools.fmp_api.get_fmp_quote", lambda symbol: {"error": "no fmp"})
    monkeypatch.setattr("tools.polygon_api.get_polygon_quote", lambda symbol: {"error": "no polygon"})
    monkeypatch.setattr(md.yf, "Ticker", lambda symbol: _Ticker(fail_info=True, history=_hist(20, 20)))

    result = md.get_stock_data.__wrapped__("FUNDX")
    assert result["current_price"] == "$22.00"
    assert result["market_cap"] == "$123,456"

    class BrokenTicker:
        @property
        def info(self):
            raise RuntimeError("info gone")

        @property
        def fast_info(self):
            raise RuntimeError("fast gone")

        def history(self, *args, **kwargs):
            raise RuntimeError("history gone")

    monkeypatch.setattr(md.yf, "Ticker", lambda symbol: BrokenTicker())
    broken = md.get_stock_data.__wrapped__("BROKEN")
    assert broken["current_price"] == "N/A"
    assert broken["source"] == "yfinance (fallback)"


def test_market_data_etf_holdings_and_error_branches(monkeypatch):
    import tools.market_data as md

    monkeypatch.setattr("tools.fmp_api.get_fmp_etf_holdings", lambda symbol: [])
    funds_data = types.SimpleNamespace(
        top_holdings=pd.DataFrame({"weight": [0.1, 0.2]}, index=["AAPL", "MSFT"])
    )
    monkeypatch.setattr(md.yf, "Ticker", lambda symbol: _Ticker(info={"quoteType": "ETF"}, funds_data=funds_data))

    holdings = md.get_etf_holdings("ETF")
    assert holdings["top_holdings"] == ["AAPL: 10.00%", "MSFT: 20.00%"]

    monkeypatch.setattr(md.yf, "Ticker", lambda symbol: _Ticker(info={"quoteType": "EQUITY"}))
    assert md.get_etf_holdings("AAPL")["message"].startswith("Not an ETF")

    monkeypatch.setattr(md.yf, "Ticker", lambda symbol: _Ticker(history=pd.DataFrame()))
    assert md.get_historical_performance("NEW")["error"] == "No historical data available"

    monkeypatch.setattr(md.yf, "Ticker", lambda symbol: (_ for _ in ()).throw(RuntimeError("bad ticker")))
    assert "error" in md.get_fundamentals_detailed("BAD")


def test_chat_endpoint_rejects_when_agent_not_ready(monkeypatch):
    import api.routers.chat as chat

    monkeypatch.setattr(chat, "get_agent", lambda: None)
    req = chat.ChatRequest(message="hello", thread_id="t1", request_id="r1")
    response = asyncio.run(chat.chat_endpoint(req))

    assert response.status_code == 530
    assert b"Agent engine is still starting up" in response.body
