import math
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tools.exception_logger import log_exceptions


def _finite(value) -> float | None:
    """The value as a float, or None if it is not a usable number.

    NaN is the shape this guard exists for. It is a float, it compares False
    against everything without raising, and it formats as the perfectly
    respectable-looking string "$nan" — so an unusable price does not fail here,
    it travels. Observed 2026-07-29: a NaN last Close silently classified AMZN
    "Bearish" (`nan > ema_21` is False) and shipped a trade plan quoting a real
    EMA21 beside a current price of "$nan", leaving the stop unverifiable.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


@log_exceptions()
def structure_trade_setup(symbol: str, risk_reward_ratio: float = 2.0, timeframe: str = "swing") -> dict[str, Any]:
    """
    Architects a professional trade setup with precise Entry, Stop Loss, and Targets.
    Uses ATR (Average True Range) for volatility-based stops.

    Args:
        symbol: Ticker symbol (e.g. "AAPL")
        risk_reward_ratio: Desired R:R (default 2.0)
        timeframe: "day" (tight stops) or "swing" (looser stops)
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch enough data for ATR-14 and EMAs
        hist = ticker.history(period="3mo", interval="1d")

        if hist.empty or len(hist) < 20:
            return {"error": "Insufficient historical data for analysis"}

        # Drop rows with no Close before anything is derived from them. yfinance
        # can return a placeholder row for a session that has not settled, and a
        # NaN there poisons the last-row reads below rather than raising.
        hist = hist[hist["Close"].notna()].copy()
        if len(hist) < 20:
            return {"error": "Insufficient historical data for analysis"}

        # Current Price
        current_price = hist["Close"].iloc[-1]

        # 1. Calculate Technical Levels
        # EMA 21 (Trend Support)
        hist["EMA_21"] = hist["Close"].ewm(span=21, adjust=False).mean()
        ema_21 = hist["EMA_21"].iloc[-1]

        # Recent Swing Low (Support)
        hist["Low"].tail(20).min()

        # 2. Calculate ATR (Volatility)
        high_low = hist["High"] - hist["Low"]
        high_close = np.abs(hist["High"] - hist["Close"].shift())
        low_close = np.abs(hist["Low"] - hist["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr_14 = true_range.rolling(window=14).mean().iloc[-1]

        # Every level below is derived from these three, and every one of them can
        # come back NaN from a gappy High/Low column without anything raising. Refuse
        # the setup here rather than formatting "$nan" into a stop the user cannot
        # act on and the compliance judge cannot verify.
        levels = {
            "current price": _finite(current_price),
            "EMA21": _finite(ema_21),
            "ATR14": _finite(atr_14),
        }
        missing = [name for name, val in levels.items() if val is None]
        if missing:
            return {
                "error": (
                    f"Trade setup unavailable for {symbol.upper()}: "
                    f"{', '.join(missing)} is not a usable number in the price history "
                    "(non-numeric or missing data). No entry, stop, or target is quoted."
                )
            }
        current_price = levels["current price"]
        ema_21 = levels["EMA21"]
        atr_14 = levels["ATR14"]

        # 3. Architect the Setup

        # Multiplier depends on timeframe
        # Swing: 1.5x ATR buffer. Day: 1.0x ATR buffer.
        atr_multiplier = 1.5 if timeframe == "swing" else 1.0
        buffer = atr_14 * atr_multiplier

        # Logic:
        # If bullish (Price > EMA21), look for buy.
        # If bearish (Price < EMA21), warn or look for short (assuming long-only for now).

        trend = "Bullish" if current_price > ema_21 else "Bearish"

        if trend == "Bearish":
            def fmt(val): return f"${val:.2f}"
            return {
                "symbol": symbol.upper(),
                "setup_type": f"{timeframe.capitalize()} Trade ({trend} Trend)",
                "current_price": fmt(current_price),
                "volatility_atr": fmt(atr_14),
                "trade_plan": {
                    "entry_zone": "No new long entry while price is below EMA21",
                    "stop_loss": "N/A",
                    "take_profit_1": "N/A",
                    "take_profit_2": "N/A",
                    "risk_per_share": "N/A",
                    "risk_reward_ratio": "N/A"
                },
                "rationale": (
                    f"Price is below trend support (EMA21 at {fmt(ema_21)}). "
                    "The tool does not force a starter position when the technical structure is bearish."
                )
            }

        # Entry Strategy:
        # Ideal entry is a pullback to EMA21 or current price if momentum is strong.
        # For simplicity in this tool, we assume "Enter Now" or "Limit at EMA"

        entry_price = current_price

        # Stop Loss: Below Low or Volatility Based
        # Placing stop below EMA21 - Buffer
        stop_loss = ema_21 - buffer

        # If Stop is too close (current price > EMA), use price - 2*ATR
        if stop_loss > current_price: # Should not happen if Bullish
             stop_loss = current_price - (2 * atr_14)
        elif (current_price - stop_loss) < (0.5 * atr_14): # Too tight
             stop_loss = current_price - (2 * atr_14)

        risk_per_share = entry_price - stop_loss

        # Take Targets (Fibonacci logic or R:R)
        # Target 1: 1R
        target_1 = entry_price + risk_per_share
        # Target 2: Desired R:R
        target_2 = entry_price + (risk_per_share * risk_reward_ratio)

        # Format Currency
        def fmt(val): return f"${val:.2f}"

        return {
            "symbol": symbol.upper(),
            "setup_type": f"{timeframe.capitalize()} Trade ({trend} Trend)",
            "current_price": fmt(current_price),
            "volatility_atr": fmt(atr_14),
            "trade_plan": {
                "entry_zone": fmt(entry_price),
                "stop_loss": fmt(stop_loss),
                "take_profit_1": fmt(target_1),
                "take_profit_2": fmt(target_2),
                "risk_per_share": fmt(risk_per_share),
                "risk_reward_ratio": f"1:{risk_reward_ratio}"
            },
            "rationale": (
                f"Stop placed {atr_multiplier}x ATR below trend support (EMA21 at {fmt(ema_21)}). "
                f"Targets set at 1x and {risk_reward_ratio}x risk distance."
            )
        }

    except Exception as e:
        return {"error": f"Trade Architect failed: {str(e)}"}
