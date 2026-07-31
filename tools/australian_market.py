from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.yf_utils import dividend_yield_display


def is_australian_ticker(symbol: str) -> bool:
    """Check if the ticker is an Australian stock (.AX)"""
    return symbol.upper().endswith('.AX')

@cached(key_func=lambda symbol: f"australian_quote:{symbol.upper()}")
@log_exceptions()
def get_australian_quote(symbol: str) -> dict[str, Any]:
    """Get real-time quote and fundamentals for an Australian stock."""
    if not is_australian_ticker(symbol):
        return {"error": "Not a valid Australian ticker"}

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol,
            "company_name": info.get("longName", "Unknown"),
            "price": info.get("currentPrice", info.get("regularMarketPrice")),
            "currency": info.get("currency", "AUD"),
            "exchange": info.get("exchange", "ASX"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            # A LABELLED string, not a bare float. This payload goes straight
            # to the model, and `"dividend_yield": 0.32` names no unit — it is
            # 0.32% read as 32% by anything that assumes a fraction, which is
            # what every other reader in this codebase assumed.
            "dividend_yield": dividend_yield_display(info),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry")
        }
    except Exception as e:
        return {"error": str(e)}

@cached(key_func=lambda symbol: f"australian_analyst:{symbol.upper()}")
@log_exceptions()
def get_australian_analyst_estimates(symbol: str) -> dict[str, Any]:
    """Get Australian analyst consensus from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "target_mean": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "number_of_analysts": info.get("numberOfAnalystOpinions", 0)
        }
    except Exception as e:
        return {"error": str(e)}

@cached(key_func=lambda symbol: f"australian_news:{symbol.upper()}")
@log_exceptions()
def get_australian_news(symbol: str) -> str:
    """Get specific Australian news and press releases using web search."""
    from tools.web_search import search_news
    company_name = symbol.split(".")[0]
    query = f"{company_name} ASX Australia press release earnings ASX.com.au"

    try:
        results = search_news(query, max_results=5)
        return results
    except Exception as e:
        return f"Error fetching Australian news: {e}"
