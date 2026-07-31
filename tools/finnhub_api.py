from datetime import datetime, timedelta

import requests

from tools.cache import cached
from tools.credential_manager import get_api_key, report_rate_limit
from tools.exception_logger import log_exceptions
from tools.tool_errors import missing_key_reason

BASE_URL = "https://finnhub.io/api/v1"

@log_exceptions()
def _finnhub_key() -> str:
    """Get the best available Finnhub API key."""
    return get_api_key("FINNHUB_API_KEY")

@log_exceptions()
def _finnhub_get(url_path: str, params: dict = None, timeout: int = 10):
    """
    Shared Finnhub request helper with automatic key rotation on 429.
    Returns (response_json, error_string_or_None).
    """
    if params is None:
        params = {}
    key = _finnhub_key()
    if not key:
        return None, missing_key_reason("FINNHUB_API_KEY")
    params["token"] = key
    url = f"{BASE_URL}/{url_path}" if not url_path.startswith("http") else url_path

    resp = requests.get(url, params=params, timeout=timeout)

    if resp.status_code == 429:
        # Rotate: mark this key as limited and retry with next
        report_rate_limit("FINNHUB_API_KEY", key)
        next_key = _finnhub_key()
        if next_key and next_key != key:
            params["token"] = next_key
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                report_rate_limit("FINNHUB_API_KEY", next_key)
                return None, "Rate limit on all Finnhub keys"
        else:
            return None, "Rate limit (no secondary key available)"

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text}"

    try:
        return resp.json(), None
    except Exception:
        return None, "Invalid JSON response"

@cached(key_func=lambda limit=10: f"finnhub_market_news:{limit}")
@log_exceptions()
def get_finnhub_market_news(limit: int = 10) -> list:
    """Fetches general market news. Returns a list of dicts."""
    data, err = _finnhub_get("news", {"category": "general"})
    if err or not isinstance(data, list):
        return []
    return data[:limit]

@cached(key_func=lambda symbol, limit=5: f"finnhub_company_news:{symbol.upper()}:{limit}")
@log_exceptions()
def get_finnhub_company_news(symbol: str, limit: int = 5) -> list:
    """Fetches company news for the last 14 days. Returns a list of dicts."""
    clean_sym = symbol.upper().split(".")[0] # Finnhub might choke on .TO
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

    data, err = _finnhub_get("company-news", {"symbol": clean_sym, "from": from_date, "to": to_date})
    if err or not isinstance(data, list):
        return []
    return data[:limit]

if __name__ == "__main__":
    print("--- Market News ---")
    print(get_finnhub_market_news(2))
    print("\n--- Company News AAPL ---")
    print(get_finnhub_company_news("AAPL", 2))
