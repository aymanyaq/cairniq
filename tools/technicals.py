"""
Advanced Technical Analysis Tool (Backend Only)
Computes comprehensive technical indicators to feed the AI Agent.
"""
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol: f"technicals:{symbol.upper()}")
@log_exceptions()
def get_comprehensive_technicals(symbol: str) -> dict:
    """
    Performs a deep-dive technical analysis on a stock.
    Returns a structured dictionary of signals for the AI to interpret.
    """
    result = {"symbol": symbol.upper(), "error": None}

    try:
        # Fetch 2 years of data to ensure 200 SMA is valid
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")

        if df.empty or len(df) < 200:
            result["error"] = "Insufficient data (needs >200 days)"
            return result

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        df["Volume"]
        # Cast indicator values to native float at extraction: numpy scalars
        # (np.float64/np.bool_) leak as "np.float64(...)" into any str()-rendered
        # tool output, and keep boolean comparisons below native too.
        current_price = float(close.iloc[-1])

        # --- 1. TREND ANALYSIS ---
        sma_50 = float(SMAIndicator(close, window=50).sma_indicator().iloc[-1])
        sma_200 = float(SMAIndicator(close, window=200).sma_indicator().iloc[-1])
        ema_20 = float(EMAIndicator(close, window=20).ema_indicator().iloc[-1])

        # Crosses
        prev_sma_50 = float(SMAIndicator(close, window=50).sma_indicator().iloc[-2])
        prev_sma_200 = float(SMAIndicator(close, window=200).sma_indicator().iloc[-2])

        golden_cross = prev_sma_50 < prev_sma_200 and sma_50 > sma_200
        death_cross = prev_sma_50 > prev_sma_200 and sma_50 < sma_200

        # ADX (Trend Strength)
        adx = float(ADXIndicator(high, low, close, window=14).adx().iloc[-1])

        trend_status = "SIDEWAYS"
        if current_price > sma_50 > sma_200:
            trend_status = "STRONG UPTREND 🟢"
        elif current_price < sma_50 < sma_200:
            trend_status = "STRONG DOWNTREND 🔴"
        elif current_price > sma_200:
            trend_status = "BULLISH (Above 200 MA)"

        result["trend"] = {
            "current_price": round(current_price, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
            "ema_20": round(ema_20, 2),
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "adx_strength": round(adx, 1), # >25 means strong trend
            "status": trend_status
        }

        # --- 2. MOMENTUM ---
        rsi = float(RSIIndicator(close, window=14).rsi().iloc[-1])
        macd = MACD(close)
        macd_line = float(macd.macd().iloc[-1])
        signal_line = float(macd.macd_signal().iloc[-1])
        histogram = float(macd.macd_diff().iloc[-1])

        # Stochastic RSI
        stoch = StochRSIIndicator(close, window=14)
        k_line = float(stoch.stochrsi_k().iloc[-1])
        stoch.stochrsi_d().iloc[-1]

        result["momentum"] = {
            "rsi_14": round(rsi, 1),
            "macd_line": round(macd_line, 3),
            "signal_line": round(signal_line, 3),
            "histogram": round(histogram, 3),
            "macd_bullish": macd_line > signal_line,
            "stoch_k": round(k_line, 2),
            "oversold": rsi < 30,
            "overbought": rsi > 70
        }

        # --- 3. VOLATILITY & BANDS ---
        bb = BollingerBands(close, window=20, window_dev=2)
        bb_high = float(bb.bollinger_hband().iloc[-1])
        bb_low = float(bb.bollinger_lband().iloc[-1])
        bb_mid = float(bb.bollinger_mavg().iloc[-1])

        atr = float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])

        result["volatility"] = {
            "bb_upper": round(bb_high, 2),
            "bb_lower": round(bb_low, 2),
            "atr": round(atr, 2),
            "squeeze": (bb_high - bb_low) / bb_mid < 0.10 # Basic squeeze check
        }

        # --- 4. SUPPORT & RESISTANCE (Clustering) ---
        # Find local mins and maxes over last 6 months
        last_6mo = df.iloc[-126:]

        # Simple algorithm: Round prices to nearest dollar/50c and find high volume nodes
        # Better simple approach: Recent local swing highs/lows
        window = 5
        # Use simple boolean indexing (implied .loc) instead of .iloc for boolean masks
        mask_min = (last_6mo['Low'].shift(window) > last_6mo['Low']) & (last_6mo['Low'].shift(-window) > last_6mo['Low'])
        mask_max = (last_6mo['High'].shift(window) < last_6mo['High']) & (last_6mo['High'].shift(-window) < last_6mo['High'])

        local_min = last_6mo[mask_min]['Low']
        local_max = last_6mo[mask_max]['High']

        # Get 3 nearest levels to current price
        supports = sorted([float(x) for x in local_min if x < current_price], reverse=True)[:3]
        resistances = sorted([float(x) for x in local_max if x > current_price])[:3]

        result["levels"] = {
            "nearest_support": [round(x, 2) for x in supports],
            "nearest_resistance": [round(x, 2) for x in resistances]
        }

        # --- 5. PATTERN DETECTION (Basic) ---

        # Bull Flag (Simple logic: Sharp rise then consolidation)
        # Head and Shoulders (Visual pattern difficult to code simply, usually requires library)
        # We'll rely on Moving Average patterns and Candle patterns if available.
        # Let's add simple Fibonacci from YTD high/low
        ytd_start = pd.Timestamp(f"{pd.Timestamp.now().year}-01-01").tz_localize(df.index.dtype.tz)
        if ytd_start < df.index[0]: ytd_start = df.index[0] # Handle if data < 1 year

        try:
           ytd_data = df.loc[ytd_start:]
        except Exception:
           ytd_data = df.iloc[-252:] # Fallback to 1yr

        ytd_high = float(ytd_data["High"].max())
        ytd_low = float(ytd_data["Low"].min())

        result["fibonacci"] = {
            "ytd_high": round(ytd_high, 2),
            "ytd_low": round(ytd_low, 2),
            "fib_382": round(ytd_high - (ytd_high - ytd_low) * 0.382, 2),
            "fib_500": round(ytd_high - (ytd_high - ytd_low) * 0.5, 2),
            "fib_618": round(ytd_high - (ytd_high - ytd_low) * 0.618, 2)
        }

        # --- SUMMARY FOR LLM ---
        # Generate a paragraph describing the setup
        summary_parts = [f"Technical Analysis for {symbol} (Price: ${current_price:.2f})."]
        summary_parts.append(f"Trend is {trend_status}. ADX is {result['trend']['adx_strength']}.")

        if golden_cross: summary_parts.append("ALERT: Golden Cross detected (Bullish).")
        if death_cross: summary_parts.append("ALERT: Death Cross detected (Bearish).")

        if rsi < 30: summary_parts.append("RSI is Oversold (Bullish bounce likely).")
        elif rsi > 70: summary_parts.append("RSI is Overbought (Pullback likely).")

        if macd_line > signal_line: summary_parts.append("MACD is Bullish.")
        else: summary_parts.append("MACD is Bearish.")

        if supports: summary_parts.append(f"Support levels: {', '.join(map(str, result['levels']['nearest_support']))}.")
        if resistances: summary_parts.append(f"Resistance levels: {', '.join(map(str, result['levels']['nearest_resistance']))}.")

        result["llm_summary"] = " ".join(summary_parts)

        return result

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

if __name__ == "__main__":
    # Test
    print(get_comprehensive_technicals("AAPL"))
