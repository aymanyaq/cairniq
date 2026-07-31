"""
Price Targets & Entry Points Tool
Fetches analyst price targets and calculates optimal entry points.
"""
try:
    import yfinance as yf
except ImportError:
    from agent.utils import safe_print
    safe_print("Warning: yfinance import failed in price_targets.py (curl_cffi issue?)")
    yf = None
from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol: f"price_targets:{symbol.upper()}")
@log_exceptions()
def get_price_targets(symbol: str) -> dict:
    """
    Get analyst price targets and calculate entry points.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        target_high = info.get('targetHighPrice', 0)
        target_low = info.get('targetLowPrice', 0)
        target_mean = info.get('targetMeanPrice', 0)
        target_median = info.get('targetMedianPrice', 0)

        # Get 52-week data for support/resistance
        info.get('fiftyTwoWeekHigh', 0)
        week_52_low = info.get('fiftyTwoWeekLow', 0)

        # Calculate key levels
        upside_potential = ((target_mean - current_price) / current_price * 100) if current_price > 0 else 0

        # Calculate support levels (pullback entry points)
        support_1 = current_price * 0.95  # 5% pullback
        support_2 = current_price * 0.90  # 10% pullback
        week_52_low * 1.05 if week_52_low else current_price * 0.85  # Near 52-week low

        # Calculate risk/reward ratio from market structure, not a fixed percent.
        stop_loss = None
        stop_basis = "Data Unavailable"
        try:
            hist = ticker.history(period="3mo", interval="1d")
            if hist is not None and len(hist) >= 20:
                high_low = hist["High"] - hist["Low"]
                high_close = (hist["High"] - hist["Close"].shift()).abs()
                low_close = (hist["Low"] - hist["Close"].shift()).abs()
                true_range = high_low.to_frame("hl").join(high_close.to_frame("hc")).join(low_close.to_frame("lc")).max(axis=1)
                atr_14 = true_range.rolling(window=14).mean().iloc[-1]
                recent_swing_low = hist["Low"].tail(20).min()
                atr_stop = current_price - (2 * atr_14)
                stop_loss = min(atr_stop, recent_swing_low)
                stop_basis = f"lower of 20-day swing low and 2x ATR (${atr_14:.2f})"
        except Exception:
            stop_loss = None

        target = target_mean if target_mean > 0 else current_price * 1.15
        risk = current_price - stop_loss if stop_loss and current_price > stop_loss else 0
        reward = target - current_price
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # Recommendation
        if current_price > 0 and target_mean > 0:
            if upside_potential > 20:
                recommendation = "Strong Buy - Significant upside potential"
            elif upside_potential > 10:
                recommendation = "Buy - Good upside potential"
            elif upside_potential > 0:
                recommendation = "Hold - Limited upside"
            else:
                recommendation = "Caution - Trading above analyst targets"
        else:
            recommendation = "Insufficient analyst data"

        return {
            "symbol": symbol,
            "current_price": f"${current_price:.2f}",
            "analyst_targets": {
                "low": f"${target_low:.2f}" if target_low else "N/A",
                "mean": f"${target_mean:.2f}" if target_mean else "N/A",
                "median": f"${target_median:.2f}" if target_median else "N/A",
                "high": f"${target_high:.2f}" if target_high else "N/A"
            },
            "upside_potential": f"{upside_potential:.1f}%",
            "entry_points": {
                "aggressive": f"${current_price:.2f} (current price)",
                "moderate": f"${support_1:.2f} (5% pullback)",
                "conservative": f"${support_2:.2f} (10% pullback)"
            },
            "stop_loss_suggestion": (
                f"${stop_loss:.2f} ({stop_basis})"
                if stop_loss
                else "Data Unavailable (requires ATR/support history)"
            ),
            "risk_reward_ratio": f"{risk_reward_ratio:.2f}:1" if risk > 0 else "Data Unavailable",
            "recommendation": recommendation
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol}
