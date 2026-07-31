from typing import Any

import requests

from tools.cache import cached
from tools.credential_manager import get_api_key, report_rate_limit
from tools.exception_logger import log_exceptions
from tools.tool_errors import missing_key_reason

BASE_URL = "https://api.polygon.io"

@log_exceptions()
def _polygon_key() -> str:
    """Get the best available Polygon API key."""
    return get_api_key("POLYGON_API_KEY")

@log_exceptions()
def _polygon_get(url_path: str, params: dict = None, timeout: int = 5):
    """
    Shared Polygon request helper with automatic key rotation on 429.
    Returns (response_json, error_string_or_None).
    """
    if params is None:
        params = {}
    key = _polygon_key()
    if not key:
        return None, missing_key_reason("POLYGON_API_KEY")
    params["apiKey"] = key
    url = f"{BASE_URL}/{url_path}" if not url_path.startswith("http") else url_path

    resp = requests.get(url, params=params, timeout=timeout)

    if resp.status_code == 429:
        # Rotate: mark this key as limited and retry with next
        report_rate_limit("POLYGON_API_KEY", key)
        next_key = _polygon_key()
        if next_key and next_key != key:
            params["apiKey"] = next_key
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                report_rate_limit("POLYGON_API_KEY", next_key)
                return None, "Rate limit on all Polygon keys"
        else:
            return None, "Rate limit (no secondary key available)"

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"

    try:
        return resp.json(), None
    except Exception:
        return None, "Invalid JSON response"

@cached(key_func=lambda symbol: f"polygon_quote:{symbol.upper()}")
@log_exceptions()
def get_polygon_quote(symbol: str) -> dict[str, Any]:
    """Get previous day's close / quote from Polygon."""
    try:
        data, err = _polygon_get(f"v2/aggs/ticker/{symbol.upper()}/prev")
        if err or not data or not data.get("results"):
            return {"error": err or "No data found"}

        item = data["results"][0]
        return {
            "symbol": item.get("T", symbol.upper()),
            "price": item.get("c"),
            "open": item.get("o"),
            "day_high": item.get("h"),
            "day_low": item.get("l"),
            "volume": item.get("v"),
            "vwap": item.get("vw")
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "polygon_api", "Polygon quote fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": str(e)}

@cached(key_func=lambda symbol: f"polygon_profile:{symbol.upper()}")
@log_exceptions()
def get_polygon_profile(symbol: str) -> dict[str, Any]:
    """Get company profile (Sector, Description, Market Cap)."""
    try:
        data, err = _polygon_get(f"v3/reference/tickers/{symbol.upper()}")
        if err or not data or not data.get("results"):
            return {"error": err or "Profile not found"}

        item = data["results"]
        return {
            "symbol": item.get("ticker", symbol.upper()),
            "company_name": item.get("name"),
            "market_cap": item.get("market_cap"),
            "description": item.get("description"),
            "industry": item.get("sic_description"),
            "currency": item.get("currency_name"),
            "homepage_url": item.get("homepage_url"),
            "employees": item.get("total_employees")
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "polygon_api", "Polygon profile fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": str(e)}

if __name__ == "__main__":
    # print("--- Quote: AAPL ---")
    # print(get_polygon_quote("AAPL"))
    # print("\n--- Profile: AAPL ---")
    # print(get_polygon_profile("AAPL"))
    pass
