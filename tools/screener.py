from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol: f"setup:{symbol.upper()}")
@log_exceptions()
def check_setup(symbol: str) -> dict[str, Any]:
    """Check a single symbol for setups."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo")
        if df.empty or len(df) < 50:
            return None

        close = df["Close"]
        # Cast to native float at extraction: numpy scalars leak as
        # "np.float64(...)" into str()-rendered tool output.
        current_price = float(close.iloc[-1])

        # Calculate Indicators
        rsi = float(RSIIndicator(close=close, window=14).rsi().iloc[-1])
        sma_50 = float(SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])

        # MACD
        macd = MACD(close=close)
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]

        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]

        setup = None

        # Setup 1: Oversold Bounce (RSI < 30 OR Price at Lower BB)
        if rsi < 30:
            setup = f"🟢 OVERSOLD (RSI {rsi:.0f})"
        elif current_price <= bb_lower * 1.01:
            setup = "🟢 BB BOUNCE (Price at Lower Band)"

        # Setup 2: Momentum Breakout (RSI 50-70 + Price > SMA50 + MACD Bullish)
        elif rsi > 50 and rsi < 75 and current_price > sma_50 and macd_line > signal_line:
            setup = "🚀 MOMENTUM BREAKOUT (Bullish MACD + Trend)"

        # Setup 3: Overbought Pullback Warning
        elif rsi > 75 or current_price >= bb_upper:
            setup = f"🔴 EXTENDED (RSI {rsi:.0f} or Upper BB)"

        if setup:
            # Structural stop from the OHLC already in hand — no extra network round-trip.
            # Basis mirrors get_price_targets: the lower of the 20-day swing low and a
            # 2x-ATR(14) volatility stop. df is guaranteed >= 50 rows above, so an
            # actionable pick ALWAYS carries a computable stop (never "Data Unavailable"),
            # which is what downstream risk-based sizing and the RiskManager stop-gate need.
            stop_loss = None
            stop_basis = None
            risk_pct = None
            try:
                high, low = df["High"], df["Low"]
                prev_close = close.shift()
                high_low = high - low
                high_close = (high - prev_close).abs()
                low_close = (low - prev_close).abs()
                true_range = high_low.to_frame("hl").join(high_close.to_frame("hc")).join(low_close.to_frame("lc")).max(axis=1)
                atr_14 = float(true_range.rolling(window=14).mean().iloc[-1])
                swing_low_20 = float(low.tail(20).min())
                candidate = min(current_price - 2 * atr_14, swing_low_20)
                # A stop must sit below entry; on a fresh breakout the ATR stop can land
                # above price, so fall back to the swing low, then a conservative floor.
                if not (candidate < current_price):
                    candidate = swing_low_20 if swing_low_20 < current_price else current_price * 0.92
                if 0 < candidate < current_price:
                    stop_loss = round(candidate, 2)
                    stop_basis = f"lower of 20d swing low and 2x ATR (${atr_14:.2f})"
                    risk_pct = round((current_price - stop_loss) / current_price * 100, 1)
            except Exception:
                stop_loss = None

            return {
                "symbol": symbol,
                "price": current_price,
                "setup": setup,
                "rsi": round(rsi, 1),
                "stop_loss": stop_loss,
                "stop_basis": stop_basis,
                "risk_pct": risk_pct,
            }
        return None

    except Exception:
        return None

@log_exceptions()
def find_breakout_candidates(symbols_str: str) -> dict[str, Any]:
    """
    Scans a list of comma-separated symbols for actionable technical setups.
    Setups: Oversold Bounce (RSI<30), Momentum Breakout, Overbought.
    """
    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

    results = []

    from agent.utils import get_st_aware_func
    executor = ThreadPoolExecutor(max_workers=10)
    try:
        future_to_symbol = {executor.submit(get_st_aware_func(check_setup), sym): sym for sym in symbols}

        try:
            for future in as_completed(future_to_symbol, timeout=60):
                try:
                    res = future.result(timeout=40)
                    if res:
                        results.append(res)
                except Exception:
                    pass
        except TimeoutError:
            pass  # Proceed with results we have

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    # Sort by Setup type
    results.sort(key=lambda x: x['setup'])

    if not results:
        return {"status": "No breakouts found", "scanned_count": len(symbols)}

    return {
        "actionable_candidates": results,
        "scanned_count": len(symbols),
        "note": "These are algorithmic matches, not guaranteed winners."
    }

if __name__ == "__main__":
    print(find_breakout_candidates("AAPL, NVDA, TSLA, AMD, F, T, AMC, GME"))
