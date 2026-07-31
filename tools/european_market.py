from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.yf_utils import dividend_yield_display

# European exchange suffixes
EUROPEAN_SUFFIXES = (
    '.L',   # London Stock Exchange
    '.PA',  # Euronext Paris
    '.DE',  # XETRA (Frankfurt)
    '.AS',  # Euronext Amsterdam
    '.MI',  # Borsa Italiana (Milan)
    '.MC',  # Bolsa de Madrid
    '.SW',  # SIX Swiss Exchange
    '.ST',  # Nasdaq Stockholm
    '.HE',  # Nasdaq Helsinki
    '.CO',  # Nasdaq Copenhagen
    '.OL',  # Oslo Bors
    '.BR',  # Euronext Brussels
    '.LS',  # Euronext Lisbon
    '.IR',  # Euronext Dublin
    '.VI',  # Vienna Stock Exchange
    '.WA',  # Warsaw Stock Exchange
)

def is_european_ticker(symbol: str) -> bool:
    """Check if the ticker is a European stock."""
    return symbol.upper().endswith(EUROPEAN_SUFFIXES)

@cached(key_func=lambda symbol: f"european_quote:{symbol.upper()}")
@log_exceptions()
def get_european_quote(symbol: str) -> dict[str, Any]:
    """Get real-time quote and fundamentals for a European stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol,
            "company_name": info.get("longName", "Unknown"),
            "price": info.get("currentPrice", info.get("regularMarketPrice")),
            "currency": info.get("currency", "EUR"),
            "exchange": info.get("exchange", "Unknown"),
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

@cached(key_func=lambda symbol: f"european_analyst:{symbol.upper()}")
@log_exceptions()
def get_european_analyst_estimates(symbol: str) -> dict[str, Any]:
    """Get European analyst consensus from Yahoo Finance."""
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

@cached(key_func=lambda symbol: f"european_news:{symbol.upper()}")
@log_exceptions()
def get_european_news(symbol: str) -> str:
    """Get specific European news and press releases using web search."""
    from tools.web_search import search_news
    company_name = symbol.split(".")[0]
    # Determine exchange for better search targeting
    suffix = "." + symbol.split(".")[-1] if "." in symbol else ""
    exchange_map = {
        ".L": "London Stock Exchange LSE",
        ".PA": "Euronext Paris",
        ".DE": "XETRA Frankfurt DAX",
        ".AS": "Euronext Amsterdam",
        ".MI": "Borsa Italiana Milan",
    }
    exchange_name = exchange_map.get(suffix.upper(), "European stock exchange")
    query = f"{company_name} {exchange_name} press release earnings results"

    try:
        results = search_news(query, max_results=5)
        return results
    except Exception as e:
        return f"Error fetching European news: {e}"
