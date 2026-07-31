import types
from unittest.mock import MagicMock

import pandas as pd
from fastapi.testclient import TestClient


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_fmp_request_helper_rotates_keys_and_handles_errors(monkeypatch):
    import tools.fmp_api as fmp

    monkeypatch.setattr(fmp, "_fmp_key", MagicMock(side_effect=["KEY1", "KEY2", "KEY3", "KEY4"]))
    reported = []
    monkeypatch.setattr(fmp, "report_rate_limit", lambda name, key: reported.append((name, key)))

    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, dict(params), timeout))
        if len(calls) == 1:
            return _Response(429, {})
        return _Response(200, [{"symbol": "AAPL"}])

    monkeypatch.setattr(fmp.requests, "get", fake_get)
    data, err = fmp._fmp_get("quote", {"symbol": "AAPL"}, timeout=7)

    assert err is None
    assert data == [{"symbol": "AAPL"}]
    assert reported == [("FMP_API_KEY", "KEY1")]
    assert calls[1][1]["apikey"] == "KEY2"

    monkeypatch.setattr(fmp.requests, "get", lambda *args, **kwargs: _Response(500, {}))
    assert fmp._fmp_get("quote")[1] == "HTTP 500"

    monkeypatch.setattr(fmp.requests, "get", lambda *args, **kwargs: _Response(200, ValueError("bad json")))
    assert fmp._fmp_get("quote")[1] == "Invalid JSON response"


def test_fmp_endpoints_parse_success_empty_and_fallback_paths(monkeypatch):
    import tools.fmp_api as fmp

    responses = {
        "quote": ([{
            "symbol": "AAPL",
            "price": 190.5,
            "change": 1.2,
            "changePercentage": 0.63,
            "dayLow": 188,
            "dayHigh": 192,
            "yearHigh": 210,
            "yearLow": 150,
            "marketCap": 3000,
            "pe": 28,
            "eps": 6.5,
            "volume": 100,
            "exchange": "NASDAQ",
        }], None),
        "profile": ([{
            "symbol": "AAPL",
            "companyName": "Apple",
            "industry": "Hardware",
            "sector": "Technology",
            "description": "Phones",
            "ceo": "CEO",
            "beta": 1.1,
            "currency": "USD",
            "country": "US",
            "exchange": "NASDAQ",
            "isEtf": False,
            "isFund": False,
        }], None),
        "income-statement": ([{"revenue": 1, "netIncome": 2, "grossProfit": 3, "eps": 4, "date": "2026-01-01"}], None),
        "etf/holdings": ([{"asset": "MSFT", "weightPercentage": 15.1}] * 20, None),
        "discounted-cash-flow": ([{"dcf": 201.25}], None),
        "price-target-consensus": ([{"targetConsensus": 200, "targetHigh": 240, "targetLow": 160}], None),
        "insider-trading": ([{"symbol": "AAPL"}], None),
        "senate-disclosure": ([{"symbol": "AAPL"}], None),
        "quote-short/AAPL": ([{"symbol": "AAPL", "shortPercentFloat": 0.02, "shortInterest": 123}], None),
        "economic_calendar": ([{"event": "CPI Inflation", "country": "US", "date": "2026-05-01", "estimate": "0.2", "previous": "0.1", "unit": "%"}], None),
        "earning_call_transcript/AAPL": ([{"content": "x" * 12050, "date": "2026-01-20", "quarter": 1, "year": 2026}], None),
    }

    def fake_fmp_get(path, params=None, timeout=5):
        return responses.get(path, (None, "missing"))

    monkeypatch.setattr(fmp, "_fmp_get", fake_fmp_get)

    assert fmp.get_fmp_quote.__wrapped__("AAPL")["price"] == 190.5
    assert fmp.get_fmp_profile.__wrapped__("AAPL")["sector"] == "Technology"
    assert fmp.get_fmp_financials.__wrapped__("AAPL")["net_income"] == 2
    assert len(fmp.get_fmp_etf_holdings.__wrapped__("SPY")) == 15
    assert fmp.get_fmp_dcf.__wrapped__("AAPL") == 201.25
    assert fmp.get_fmp_analyst_estimates.__wrapped__("AAPL")["target_high"] == 240
    assert fmp.get_fmp_insider_trades.__wrapped__("AAPL") == [{"symbol": "AAPL"}]
    assert fmp.get_fmp_senate_disclosures.__wrapped__("AAPL") == [{"symbol": "AAPL"}]
    assert fmp.get_economic_calendar.__wrapped__()[0]["event"] == "CPI Inflation"
    assert "MIDDLE SECTION SKIPPED" in fmp.get_earnings_transcript.__wrapped__("AAPL")
    assert fmp.get_short_interest.__wrapped__("AAPL")["source"] == "FMP"

    monkeypatch.setattr(fmp, "_fmp_get", lambda *args, **kwargs: ([], None))
    monkeypatch.setattr("tools.web_search.search_news", lambda *args, **kwargs: [{"title": "Calendar", "href": "https://example.com"}])
    assert fmp.get_fmp_quote.__wrapped__("MSFT")["error"] == "No data found"
    assert fmp.get_economic_calendar.__wrapped__()[0]["event"] == "Web Search Results"


def test_fmp_short_interest_yfinance_fallback(monkeypatch):
    import tools.fmp_api as fmp

    monkeypatch.setattr(fmp, "_fmp_get", lambda *args, **kwargs: ([], None))
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda symbol: types.SimpleNamespace(
            info={
                "shortPercentOfFloat": 0.04,
                "sharesShort": 1000,
                "sharesShortPriorMonth": 900,
                "shortRatio": 1.7,
            }
        ),
    )

    result = fmp.get_short_interest.__wrapped__("AAPL")
    assert result["source"] == "yfinance"
    assert result["short_float_pct"] == "4.00%"


def test_fred_helpers_fallbacks_and_live_series(monkeypatch):
    import tools.fred_api as fred

    monkeypatch.setattr(fred, "get_api_key", lambda name: "")
    assert "fallback" in fred.get_fed_funds_rate.__wrapped__()["note"]
    assert fred.get_inflation_data.__wrapped__()["headline_inflation"] == "4.2%"
    assert fred.get_treasury_yields.__wrapped__()["curve_status"] == "Normal"
    assert fred.get_canada_metrics.__wrapped__()["error"] == "Missing API Key"

    def _monthly(n, newest_value):
        """n monthly obs, newest first, starting 2026-01 and walking back with
        valid calendar dates; values decrease by 1 per month."""
        out, val, year, month = [], newest_value, 2026, 1
        for _ in range(n):
            out.append({"date": f"{year}-{month:02d}-01", "value": float(val)})
            val -= 1
            month -= 1
            if month == 0:
                year, month = year - 1, 12
        return out

    obs = _monthly(13, 13)  # 2026-01=13.0 ... 2025-01=1.0

    def fake_fetch(series_id, limit=10):
        if series_id in {"FEDFUNDS", "UNRATE"}:
            return obs[:limit]
        if series_id in {"CPIAUCSL", "CPILFESL"}:
            return _monthly(13, 113)[:limit]  # 2026-01=113 ... 2025-01=101
        if series_id == "A191RL1Q225SBEA":
            return [{"date": "2026-Q1", "value": 2.5}, {"date": "2025-Q4", "value": 2.0}]
        if series_id == "DGS10":
            return [{"date": "2026-01-01", "value": 4.5}]
        if series_id == "DGS2":
            return [{"date": "2026-01-01", "value": 4.9}]
        if series_id == "IRSTCI01CAM156N":
            return [{"date": "2026-01-01", "value": 3.75}]
        if series_id == "CPALTT01CAM661S":
            return [{"date": f"2025-{i:02d}-01", "value": 100 + i} for i in range(13, 0, -1)]
        if series_id == "BAMLH0A0HYM2":
            return [{"date": "2026-01-01", "value": 5.5}]
        if series_id == "M2SL":
            return [{"date": f"2025-{i:02d}-01", "value": 100 + i} for i in range(13, 0, -1)]
        return []

    monkeypatch.setattr(fred, "get_api_key", lambda name: "KEY")
    monkeypatch.setattr(fred, "_fetch_series", fake_fetch)

    # Date-matched year-ago: 2026-01 (13.0) vs 2025-01 (1.0)
    assert fred.get_fed_funds_rate.__wrapped__()["change_1y"] == "+12.00%"
    assert fred.get_inflation_data.__wrapped__()["status"] == "Above Target"
    assert fred.get_gdp_growth.__wrapped__()["trend"] == "Accelerating"
    assert fred.get_unemployment.__wrapped__()["trend"] == "Worsening"
    assert "INVERTED" in fred.get_treasury_yields.__wrapped__()["curve_status"]
    assert fred.get_canada_metrics.__wrapped__()["source"] == "FRED API"
    assert fred.get_systemic_risk_indicators.__wrapped__()["crash_risk"] == "High"
    assert "summary" in fred.get_all_macro_indicators.__wrapped__()


def test_fred_request_and_fetch_series(monkeypatch):
    import tools.fred_api as fred

    reported = []
    monkeypatch.setattr("tools.credential_manager.report_rate_limit", lambda name, key: reported.append((name, key)))
    monkeypatch.setattr("tools.credential_manager.get_api_key", MagicMock(side_effect=["KEY1", "KEY2", "KEY3", "KEY4"]))
    calls = []

    def fake_get(url, params, timeout):
        calls.append(dict(params))
        if len(calls) == 1:
            return _Response(429, {})
        return _Response(200, {"observations": [{"date": "2026-01-01", "value": "4.5"}, {"date": "2025-12-01", "value": "."}]})

    monkeypatch.setattr(fred.requests, "get", fake_get)
    data, err = fred._fred_get({"series_id": "FEDFUNDS"})
    assert err is None
    assert data["observations"][0]["value"] == "4.5"
    assert reported == [("FRED_API_KEY", "KEY1")]

    series = fred._fetch_series("FEDFUNDS")
    assert series == [{"date": "2026-01-01", "value": 4.5}]

    monkeypatch.setattr(fred.requests, "get", lambda *args, **kwargs: _Response(500, {}))
    assert fred._fred_get({"series_id": "FEDFUNDS"})[1] == "HTTP Error 500"


def test_portfolio_router_template_upload_and_benchmark(monkeypatch):
    import api.routers.portfolio as portfolio_router
    import server
    from tools.user_profile import get_active_profile

    client = TestClient(server.app)
    client.cookies.set("profile", get_active_profile())
    template = client.get("/api/portfolio/download-template")
    assert template.status_code == 200
    assert "AAPL" in template.text

    rejected = client.post("/api/portfolio/upload", files={"file": ("bad.txt", b"nope", "text/plain")})
    assert rejected.status_code == 400

    uploaded = client.post(
        "/api/portfolio/upload",
        files={"file": ("holdings.csv", b"symbol,shares\nAAPL,1\n", "text/csv")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["status"] == "success"

    history = pd.DataFrame(
        [
            {"date": "2026-01-02", "total_value_usd": 1000.0, "percent_return": 5.0},
            {"date": "2026-01-03", "total_value_usd": 1100.0, "percent_return": 15.0},
        ]
    )
    monkeypatch.setattr("tools.portfolio_tracker.snapshot_portfolio", lambda: None)
    monkeypatch.setattr("tools.portfolio_tracker.get_portfolio_history", lambda period: history)

    class FakeTicker:
        def history(self, start=None):
            idx = pd.to_datetime(["2026-01-02", "2026-01-03"])
            return pd.DataFrame({"Close": [100.0, 105.0]}, index=idx)

    monkeypatch.setattr("yfinance.Ticker", lambda symbol: FakeTicker())

    benchmark = portfolio_router._compute_benchmark_data()
    assert benchmark["portfolio_return"] == "+10.0%"
    assert benchmark["spy_return"] == "+5.0%"
    assert len(benchmark["spy_points"]) == 2

    monkeypatch.setattr(portfolio_router, "get_or_compute", lambda key, fn: {"cached": True})
    assert client.get("/api/benchmark").json() == {"cached": True}


def test_chat_management_direct_paths(monkeypatch):
    import api.routers.chat as chat

    cancel_ctx = chat.ChatRunContext("abc")
    chat._active_chat_runs.clear()
    chat._active_chat_runs["abc"] = cancel_ctx

    class Request:
        async def json(self):
            return {"thread_id": "abc"}

    import asyncio

    cancelled = asyncio.run(chat.chat_stop_endpoint(Request()))
    assert cancelled["status"] == "cancelled"
    assert cancel_ctx.cancel_event.is_set()
    chat._active_chat_runs.clear()

    class BadRequest:
        async def json(self):
            raise ValueError("bad")

    chat._active_chat_runs["one"] = chat.ChatRunContext("one")
    chat._active_chat_runs["two"] = chat.ChatRunContext("two")
    all_cancelled = asyncio.run(chat.chat_stop_endpoint(BadRequest()))
    assert all_cancelled["cancelled_threads"] == ["one", "two"]

    monkeypatch.setattr(chat, "get_session_list", lambda: [{"id": "s1"}])
    monkeypatch.setattr(chat, "load_session", lambda sid: {"messages": [{"role": "user", "content": "hi"}], "session_cost_cad": 1.25} if sid == "s1" else None)
    monkeypatch.setattr(chat, "delete_session", lambda sid: sid == "s1")
    monkeypatch.setattr("tools.memory.extract_thesis_from_text", lambda text: {"ticker": "AAPL"})

    assert asyncio.run(chat.extract_thesis_from_chat({"text": ""}))["error"] == "No text provided"
    assert asyncio.run(chat.extract_thesis_from_chat({"text": "AAPL thesis"}))["ticker"] == "AAPL"
    assert asyncio.run(chat.list_chats()).body == b'{"sessions":[{"id":"s1"}]}'
    loaded = asyncio.run(chat.get_chat("s1"))
    assert b'"session_cost_cad":1.25' in loaded.body
    missing = asyncio.run(chat.get_chat("missing"))
    assert missing.status_code == 404
    removed = asyncio.run(chat.remove_chat("s1"))
    assert removed.body == b'{"success":true}'
