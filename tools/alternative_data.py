from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.web_search import search_news


@cached(key_func=lambda symbol: f"alt_data:{symbol.upper()}")
@log_exceptions()
def get_alternative_data_signal(symbol: str) -> dict[str, Any]:
    """
    Proxies alternative data (web traffic, app downloads, sentiment shifts).
    Used as a Leading Indicator before earnings.
    """
    try:
        # We proxy this via search sentiment + yf news volume and recent price divergence
        # Real hedge funds use credit card data APIs, but we simulate the signal logically
        ticker = yf.Ticker(symbol)

        # Pull recent headlines focusing on "app downloads", "user growth", "site traffic"
        search_q = f"{symbol} stock app downloads OR user growth OR website traffic trend"
        res = search_news(search_q, max_results=5)

        # Analyze simple sentiment and volume
        res_lower = res.lower() if isinstance(res, str) else str(res).lower()

        score = 0
        signals = []
        if "surge" in res_lower or "spike" in res_lower or "record" in res_lower:
            score += 2
            signals.append("Web traffic/app downloads indicate a recent SURGE.")
        if "drop" in res_lower or "slowdown" in res_lower or "weak" in res_lower:
            score -= 2
            signals.append("Web traffic/app downloads indicate a recent SLOWDOWN.")

        # Get short term price vs 50ma to see if smart money is pricing this in
        hist = ticker.history(period="3mo")
        if not hist.empty and len(hist) > 50:
            current = hist["Close"].iloc[-1]
            ma50 = hist["Close"].iloc[-50:].mean()

            if current > ma50 * 1.05:
                 score += 1
                 signals.append("Smart money is bidding the stock up pre-earnings.")
            elif current < ma50 * 0.95:
                 score -= 1
                 signals.append("Smart money is distributing (selling) pre-earnings.")

        # Verdict
        if score >= 2:
            verdict = "🟢 BULLISH SIGNAL: Alternative data proxies suggest an upcoming earnings BEAT."
        elif score <= -2:
            verdict = "🔴 BEARISH SIGNAL: Alternative data proxies suggest an upcoming earnings MISS."
        else:
            verdict = "⚪ NEUTRAL: No strong alternative data divergence."

        # Fake metrics based on inference (for demo purposes)
        # In production this would hit endpoints like Similarweb

        return {
            "symbol": symbol,
            "verdict": verdict,
            "signals": signals if signals else ["No significant alternative data trends detected."]
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

if __name__ == "__main__":
    print(get_alternative_data_signal("AAPL"))
