from unittest.mock import MagicMock, patch


def test_av_get_rate_limit_rotation():
    from tools.alpha_vantage import _av_get

    with patch("tools.alpha_vantage.requests.get") as mock_requests_get:
        # First response: Rate limit "Note"
        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = {"Note": "Standard API rate limit..."}

        # Second response: Success
        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = {"Global Quote": {"05. price": "150.00"}}

        mock_requests_get.side_effect = [mock_resp1, mock_resp2]

        # KEY1 triggers rotation, KEY2 works
        with patch("tools.alpha_vantage._av_key", side_effect=["KEY1", "KEY2"]):
            with patch("tools.credential_manager.report_rate_limit") as mock_report:
                params = {"symbol": "AAPL"}
                data, err = _av_get(params)

                assert err is None
                assert data is not None
                assert "Global Quote" in data
                assert data["Global Quote"]["05. price"] == "150.00"
                mock_report.assert_called_with("ALPHA_VANTAGE_API_KEY", "KEY1")

def test_get_quote_standard():
    from tools.alpha_vantage import get_quote
    with patch("tools.alpha_vantage._av_get") as mock_av_get:
        mock_av_get.return_value = (
            {"Global Quote": {"01. symbol": "AAPL", "05. price": "150.25"}},
            None
        )
        res = get_quote.__wrapped__("AAPL")
        assert res["price"] == 150.25

def test_get_quote_uses_yfinance_fallback_on_rate_limit():
    from tools.alpha_vantage import get_quote

    fallback = {"symbol": "AAPL", "price": 150.25, "source": "yfinance"}
    with patch("tools.alpha_vantage._av_get", return_value=(None, "Rate limit")):
        with patch("tools.alpha_vantage._quote_from_yfinance", return_value=fallback):
            res = get_quote.__wrapped__("AAPL")

    assert res == fallback

def test_get_quote_uses_yfinance_fallback_on_empty_quote():
    from tools.alpha_vantage import get_quote

    fallback = {"symbol": "AAPL", "price": 150.25, "source": "yfinance"}
    with patch("tools.alpha_vantage._av_get", return_value=({}, None)):
        with patch("tools.alpha_vantage._quote_from_yfinance", return_value=fallback):
            res = get_quote.__wrapped__("AAPL")

    assert res == fallback

def test_get_daily_prices_fallback():
    from tools.alpha_vantage import get_daily_prices
    with patch("tools.alpha_vantage._av_get") as mock_av_get:
        mock_av_get.return_value = ({}, None)

        with patch("yfinance.Ticker") as mock_yf:
            import pandas as pd
            mock_ticker = MagicMock()
            mock_yf.return_value = mock_ticker
            dates = pd.date_range(end="2024-01-01", periods=5)
            mock_ticker.history.return_value = pd.DataFrame({
                "Open": [100.0]*5, "High": [105.0]*5, "Low": [95.0]*5, "Close": [102.0]*5, "Volume": [1000]*5
            }, index=dates)

            res = get_daily_prices.__wrapped__("AAPL", days=5)
            assert res.get("source") == "yfinance"
            assert len(res["prices"]) == 5


# --- Quote freshness labelling ---
#
# GLOBAL_QUOTE is end-of-day on this key tier. Mid-session on 2026-07-15 it returned
# the prior session's close while the symbol traded well above it, and the agent
# narrated that close as "today". The payload must carry its own session date so
# callers can't make that read.


def _et_today():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("US/Eastern")).date()


def _quote_payload(latest_trading_day):
    # Shaped like the cached AV GLOBAL_QUOTE payload from the 2026-07-15 incident.
    return (
        {"Global Quote": {
            "01. symbol": "XYZ", "05. price": "47.37", "09. change": "-0.28",
            "10. change percent": "-0.5876%", "06. volume": "9416032",
            "07. latest trading day": latest_trading_day, "08. previous close": "47.65",
        }},
        None,
    )


def test_get_quote_flags_previous_session_close_as_stale():
    from datetime import timedelta

    from tools.alpha_vantage import get_quote

    stale_day = (_et_today() - timedelta(days=2)).isoformat()
    with patch("tools.alpha_vantage._av_get", return_value=_quote_payload(stale_day)):
        res = get_quote.__wrapped__("XYZ")

    assert res["is_stale"] is True
    assert res["as_of"] == stale_day
    assert stale_day in res["staleness_note"]
    assert res["price"] == 47.37  # payload still returned — labelled, not rejected


def test_get_quote_from_todays_session_is_not_flagged():
    from tools.alpha_vantage import get_quote

    today = _et_today().isoformat()
    with patch("tools.alpha_vantage._av_get", return_value=_quote_payload(today)):
        res = get_quote.__wrapped__("XYZ")

    assert res["is_stale"] is False
    assert res["as_of"] == today
    assert "staleness_note" not in res  # nothing to warn about


def test_non_iso_trading_day_is_left_unannotated():
    # The yfinance fallback reports "Real-time (yfinance)"; there's no date to
    # compare, so claiming staleness either way would be a guess.
    from tools.alpha_vantage import _annotate_quote_freshness

    for day in ("Real-time (yfinance)", "N/A", ""):
        res = _annotate_quote_freshness({"symbol": "XYZ", "price": 55.23, "latest_trading_day": day})
        assert "is_stale" not in res
        assert "staleness_note" not in res
