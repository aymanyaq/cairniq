from unittest.mock import MagicMock, patch

from tools.polygon_api import _polygon_get, get_polygon_profile, get_polygon_quote


@patch("tools.polygon_api.requests.get")
@patch("tools.polygon_api._polygon_key")
def test_polygon_get_rate_limit_rotation(mock_key, mock_get):
    # Setup: First key is 429, second key is 200
    mock_key.side_effect = ["KEY1", "KEY2"]

    mock_resp1 = MagicMock()
    mock_resp1.status_code = 429

    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.json.return_value = {"status": "OK"}

    mock_get.side_effect = [mock_resp1, mock_resp2]

    with patch("tools.polygon_api.report_rate_limit") as mock_report:
        data, err = _polygon_get("v2/aggs/ticker/AAPL/prev")

        assert err is None
        assert data["status"] == "OK"
        mock_report.assert_called_with("POLYGON_API_KEY", "KEY1")

def test_get_polygon_quote_standard():
    with patch("tools.polygon_api._polygon_get") as mock_get:
        mock_get.return_value = (
            {"results": [{"T": "AAPL", "c": 150.0, "o": 149.0, "v": 1000000}]},
            None
        )

        # Bypass cache
        res = get_polygon_quote.__wrapped__("AAPL")
        assert res["price"] == 150.0
        assert res["symbol"] == "AAPL"

def test_get_polygon_profile_standard():
    with patch("tools.polygon_api._polygon_get") as mock_get:
        mock_get.return_value = (
            {"results": {"ticker": "AAPL", "name": "Apple Inc.", "market_cap": 2000000000000}},
            None
        )

        # Bypass cache
        res = get_polygon_profile.__wrapped__("AAPL")
        assert res["company_name"] == "Apple Inc."
        assert res["market_cap"] == 2000000000000
