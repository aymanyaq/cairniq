from unittest.mock import MagicMock, patch

from tools.finnhub_api import _finnhub_get, get_finnhub_company_news, get_finnhub_market_news


@patch("tools.finnhub_api.requests.get")
@patch("tools.finnhub_api._finnhub_key")
def test_finnhub_get_rotation(mock_key, mock_get):
    mock_key.side_effect = ["KEY1", "KEY2"]

    mock_resp1 = MagicMock()
    mock_resp1.status_code = 429

    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.json.return_value = [{"id": 1, "headline": "News"}]

    mock_get.side_effect = [mock_resp1, mock_resp2]

    with patch("tools.finnhub_api.report_rate_limit") as mock_report:
        data, err = _finnhub_get("news")
        assert err is None
        assert data[0]["id"] == 1
        mock_report.assert_called_with("FINNHUB_API_KEY", "KEY1")

def test_get_finnhub_market_news():
    with patch("tools.finnhub_api._finnhub_get") as mock_get:
        mock_get.return_value = ([{"headline": "Mkt News"}], None)

        # Bypass cache
        res = get_finnhub_market_news.__wrapped__(limit=1)
        assert len(res) == 1
        assert res[0]["headline"] == "Mkt News"

def test_get_finnhub_company_news():
    with patch("tools.finnhub_api._finnhub_get") as mock_get:
        mock_get.return_value = ([{"headline": "AAPL News"}], None)

        # Bypass cache
        res = get_finnhub_company_news.__wrapped__("AAPL", limit=1)
        assert len(res) == 1
        assert res[0]["headline"] == "AAPL News"
