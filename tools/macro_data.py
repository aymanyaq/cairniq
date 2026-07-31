
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions


@log_exceptions()
def _fetch_single_ticker(name: str, ticker: str) -> dict[str, Any]:
    """Fetch a single ticker's price and daily change with multiple fallback strategies."""
    try:
        # Strategy 1: yf.Ticker.history() — guaranteed single-ticker, no MultiIndex issues
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", timeout=40)

        if not hist.empty and 'Close' in hist.columns:
            closes = hist['Close'].dropna()
            if not closes.empty:
                price = float(closes.iloc[-1])
                change_pct = 0.0
                if len(closes) > 1:
                    prev = float(closes.iloc[-2])
                    change_pct = ((price - prev) / prev) * 100 if prev != 0 else 0.0
                return {
                    "name": name,
                    "ticker": ticker,
                    "price": f"{price:,.2f}",
                    "change_pct": f"{change_pct:+.2f}%",
                    "trend": "🟢" if change_pct >= 0 else "🔴"
                }
    except Exception:
        pass

    try:
        # Strategy 2: fast_info (lighter, works for many tickers)
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = fi.get("last_price") or fi.get("regularMarketPrice")
        prev_close = fi.get("previous_close") or fi.get("regularMarketPreviousClose")

        if price:
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
            return {
                "name": name,
                "ticker": ticker,
                "price": f"{price:,.2f}",
                "change_pct": f"{change_pct:+.2f}%",
                "trend": "🟢" if change_pct >= 0 else "🔴"
            }
    except Exception:
        pass

    try:
        # Strategy 3: ticker.info (slowest but most complete)
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")

        if price:
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
            return {
                "name": name,
                "ticker": ticker,
                "price": f"{price:,.2f}",
                "change_pct": f"{change_pct:+.2f}%",
                "trend": "🟢" if change_pct >= 0 else "🔴"
            }
    except Exception:
        pass

    return {"name": name, "ticker": ticker, "error": "No recent data"}


@cached(key_func=lambda: "global_market_snapshot")
@log_exceptions()
def get_global_market_snapshot() -> dict[str, Any]:
    """
    Fetches a snapshot of major global market indices to gauge macro sentiment.
    Returns current price and daily percent change.
    Uses per-ticker fetching with multiple fallbacks to avoid batch data gaps.
    """
    indices = {
        "S&P 500 (US)": "^GSPC",
        "Nasdaq (Tech)": "^IXIC",
        "TSX (Canada)": "^GSPTSE",
        "UK FTSE 100": "^FTSE",
        "Japan Nikkei 225": "^N225",
        "Emerging Markets": "EEM",
        "Bitcoin": "BTC-USD",
        "Gold": "GC=F",
        "Crude Oil": "CL=F"
    }

    snapshot = []

    # Fetch all tickers in parallel for speed
    from agent.utils import get_st_aware_func
    executor = ThreadPoolExecutor(max_workers=len(indices))
    try:
        future_to_name = {
            executor.submit(get_st_aware_func(_fetch_single_ticker), name, ticker): name
            for name, ticker in indices.items()
        }
        results = {}
        for future in as_completed(future_to_name, timeout=30):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"name": name, "error": f"Fetch failed: {e}"}

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    # Preserve original order
    # Preserve original order
    for name in indices:
        snapshot.append(results.get(name, {"name": name, "error": "Unknown error"}))

    return {
        "source": "Yahoo Finance",
        "indices": snapshot
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_global_market_snapshot(), indent=2))
