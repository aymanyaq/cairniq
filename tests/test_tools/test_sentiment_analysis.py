from unittest.mock import MagicMock, patch

from tools.sentiment_analysis import (
    _get_yfinance_news_sentiment,
    get_fear_greed_index,
    get_full_sentiment,
    get_reddit_sentiment,
    get_social_buzz,
)


@patch("tools.sentiment_analysis.requests.get")
def test_get_fear_greed_index_success(mock_get):
    # Mock CNN response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "fear_and_greed": {"score": 20, "rating": "Extreme Fear"}
    }
    mock_get.return_value = mock_response

    # Clear cache for test
    from tools.cache import _cache
    _cache.clear()

    result = get_fear_greed_index()
    assert result["score"] == 20
    assert "EXTREME FEAR" in result["implication"]

@patch("tools.sentiment_analysis.requests.get")
def test_get_fear_greed_index_fallback(mock_get):
    # Mock failure
    mock_get.side_effect = Exception("Network Error")

    # Clear cache
    from tools.cache import _cache
    _cache.clear()

    result = get_fear_greed_index()
    assert result["rating"] == "Neutral"
    assert "Live data unavailable" in result["note"]

@patch("tools.sentiment_analysis.yf.Ticker")
def test_get_social_buzz(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    # Mock high volume spike
    import pandas as pd
    dates = pd.date_range(end="2024-01-01", periods=30)
    volumes = [1000] * 25 + [3000] * 5 # High volume recently
    mock_ticker.history.return_value = pd.DataFrame({"Volume": volumes}, index=dates)

    result = get_social_buzz("AAPL")
    assert result["social_buzz_estimate"] == "Very High 🔥"
    assert result["volume_ratio"] >= 2.0

@patch("tools.sentiment_analysis.yf.Ticker")
def test_get_yfinance_news_sentiment(mock_ticker_class):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.news = [
        {"title": "Stock surges to record high after earnings beat", "link": "..."}
    ]

    result = _get_yfinance_news_sentiment("AAPL")
    assert result["recent_headlines"][0]["sentiment"] == "Positive"

@patch("tools.web_search.search_news")
@patch("tools.sentiment_analysis.yf.Ticker")
def test_get_reddit_sentiment(mock_ticker_class, mock_search):
    mock_search.return_value = "WallStreetBets yolo moon squeeze buy bull rocket breakout upgrade undervalued gem"
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.history.return_value = MagicMock(empty=True) # Skip volume check

    result = get_reddit_sentiment("AAPL")
    assert result["reddit_hype_score"] > 70
    assert "Bullish Chatter" in result["verdict"]

@patch("tools.sentiment_analysis.get_fear_greed_index")
@patch("tools.sentiment_analysis.get_news_sentiment")
@patch("tools.sentiment_analysis.get_analyst_consensus")
@patch("tools.sentiment_analysis.get_social_buzz")
@patch("tools.sentiment_analysis.get_reddit_sentiment")
def test_get_full_sentiment_aggregation(mock_reddit, mock_buzz, mock_analysts, mock_news, mock_fg):
    # Setup mocks to lean bullish
    mock_fg.return_value = {"score": 20} # Extreme Fear = Contrarian Bull
    mock_news.return_value = {"sentiment_score": 0.3} # Bullish
    mock_analysts.return_value = {"buy_percentage": 80} # Bullish
    mock_buzz.return_value = {"social_buzz_estimate": "High"}
    mock_reddit.return_value = {"reddit_hype_score": 80} # Bullish

    from tools.cache import _cache
    _cache.clear()

    result = get_full_sentiment("AAPL")
    assert "BULLISH" in result["overall_sentiment"]
    assert result["signal_breakdown"]["bullish"] > result["signal_breakdown"]["bearish"]
