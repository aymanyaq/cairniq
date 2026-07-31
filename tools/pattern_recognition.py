"""
Pattern Recognition Tool
Detects chart patterns and explains them in SIMPLE terms for non-experts.
Uses price data from yfinance - no external API needed.
"""
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema

from tools.cache import cached
from tools.exception_logger import log_exceptions


@log_exceptions()
def get_price_data(symbol: str, days: int = 120) -> pd.DataFrame | None:
    """Fetch historical price data."""
    try:
        import logging
        # Silence yfinance logger instead of redirecting stdout (which is not thread-safe)
        yf_logger = logging.getLogger("yfinance")
        old_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
        finally:
            yf_logger.setLevel(old_level)

        if df.empty:
            return None
        return df
    except Exception:
        return None


@log_exceptions()
def find_support_resistance(symbol: str, num_levels: int = 3) -> dict[str, Any]:
    """
    Find key price levels where the stock tends to bounce or reverse.
    Returns simple explanations for non-experts.
    """
    df = get_price_data(symbol, days=180)
    if df is None or len(df) < 20:
        return {"error": f"Not enough price history for {symbol}"}

    closes = df['Close'].values
    # Cast to native float: numpy scalars leak as "np.float64(...)" when the
    # result dict is str()-rendered (round() does not strip the numpy type).
    current_price = float(closes[-1])

    # Find local minima (support) and maxima (resistance)
    order = max(5, len(closes) // 20)

    local_min_idx = argrelextrema(closes, np.less_equal, order=order)[0]
    local_max_idx = argrelextrema(closes, np.greater_equal, order=order)[0]

    # Get support levels (below current price)
    support_levels = sorted([float(closes[i]) for i in local_min_idx if closes[i] < current_price], reverse=True)[:num_levels]

    # Get resistance levels (above current price)
    resistance_levels = sorted([float(closes[i]) for i in local_max_idx if closes[i] > current_price])[:num_levels]

    # Count how many times each level was tested
    def count_tests(level, prices, tolerance=0.02):
        return sum(1 for p in prices if abs(p - level) / level < tolerance)

    # Build simple explanations
    support_info = []
    for lvl in support_levels:
        tests = count_tests(lvl, closes)
        pct_below = ((current_price - lvl) / current_price) * 100
        strength = "very reliable" if tests >= 3 else "fairly reliable" if tests >= 2 else "possible"
        support_info.append({
            "price": round(lvl, 2),
            "distance": f"{pct_below:.1f}% below current price",
            "reliability": strength,
            "plain_english": f"The stock has bounced off ${lvl:.2f} about {tests} time(s) before. This is a {strength} buying zone."
        })

    resistance_info = []
    for lvl in resistance_levels:
        tests = count_tests(lvl, closes)
        pct_above = ((lvl - current_price) / current_price) * 100
        strength = "very strong" if tests >= 3 else "moderate" if tests >= 2 else "possible"
        resistance_info.append({
            "price": round(lvl, 2),
            "distance": f"{pct_above:.1f}% above current price",
            "strength": strength,
            "plain_english": f"The stock has struggled to break above ${lvl:.2f} about {tests} time(s). This is a {strength} ceiling."
        })

    # Simple summary
    if support_info and resistance_info:
        nearest_support = support_info[0]["price"]
        nearest_resistance = resistance_info[0]["price"]
        room_up = ((nearest_resistance - current_price) / current_price) * 100
        room_down = ((current_price - nearest_support) / current_price) * 100

        summary = (
            f"📊 {symbol} is currently at ${current_price:.2f}. "
            f"It has about {room_up:.0f}% room to go UP before hitting resistance at ${nearest_resistance:.2f}, "
            f"and {room_down:.0f}% room to go DOWN before hitting support at ${nearest_support:.2f}."
        )
    else:
        summary = f"📊 {symbol} is at ${current_price:.2f}. Limited historical levels detected."

    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "summary": summary,
        "support_levels": support_info,
        "resistance_levels": resistance_info,
        "what_this_means": (
            "🛡️ SUPPORT = A price 'floor' where buyers tend to step in. Good place to consider buying. "
            "🧱 RESISTANCE = A price 'ceiling' where sellers tend to appear. May struggle to go higher. "
            "If the stock breaks ABOVE resistance, that's often a bullish signal to buy. "
            "If it breaks BELOW support, that's often a bearish signal to sell or avoid."
        )
    }


@log_exceptions()
def check_ma_crossover(symbol: str) -> dict[str, Any]:
    """
    Check if the stock's trend is changing direction.
    Uses moving averages - but explains everything simply.
    """
    df = get_price_data(symbol, days=250)
    if df is None or len(df) < 200:
        return {"error": f"Need more price history for {symbol}"}

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    current = df.iloc[-1]
    prev_5 = df.iloc[-5]
    prev_20 = df.iloc[-20]

    price = float(current['Close'])
    signals = []

    # Check Golden/Death Cross (50 vs 200) - THE BIG ONES
    if current['MA50'] > current['MA200'] and prev_20['MA50'] <= prev_20['MA200']:
        signals.append({
            "signal": "🌟 GOLDEN CROSS",
            "meaning": "BUY SIGNAL",
            "plain_english": (
                "This is a classic bullish signal! The short-term trend just crossed above the long-term trend. "
                "Historically, this often signals the START of a big upward move. Many investors see this as a green light to buy."
            )
        })
    elif current['MA50'] < current['MA200'] and prev_20['MA50'] >= prev_20['MA200']:
        signals.append({
            "signal": "💀 DEATH CROSS",
            "meaning": "SELL SIGNAL",
            "plain_english": (
                "This is a classic bearish signal! The short-term trend just crossed below the long-term trend. "
                "Historically, this often signals the START of a downward move. Many investors see this as a warning to sell or avoid."
            )
        })

    # Check short-term momentum (20 vs 50)
    if current['MA20'] > current['MA50'] and prev_5['MA20'] <= prev_5['MA50']:
        signals.append({
            "signal": "📈 Momentum Turning Up",
            "meaning": "Short-term bullish",
            "plain_english": "The stock's recent momentum is improving. It may be starting a short-term rally."
        })
    elif current['MA20'] < current['MA50'] and prev_5['MA20'] >= prev_5['MA50']:
        signals.append({
            "signal": "📉 Momentum Turning Down",
            "meaning": "Short-term bearish",
            "plain_english": "The stock's recent momentum is weakening. It may be starting a short-term pullback."
        })

    # Overall trend assessment in plain English
    if price > current['MA50'] > current['MA200']:
        trend = "🟢 UPTREND"
        trend_explanation = "The stock is in a healthy uptrend. It's trading above its average prices for both the past 2 months AND the past year. This is generally a good sign."
    elif price < current['MA50'] < current['MA200']:
        trend = "🔴 DOWNTREND"
        trend_explanation = "The stock is in a downtrend. It's trading below its average prices. This suggests weakness and caution is warranted."
    else:
        trend = "⚪ MIXED/SIDEWAYS"
        trend_explanation = "The stock's trend is unclear. It's bouncing around without a clear direction. Wait for a clearer signal."

    return {
        "symbol": symbol,
        "current_price": round(price, 2),
        "overall_trend": trend,
        "trend_explanation": trend_explanation,
        "signals": signals if signals else [{"signal": "No major signals", "plain_english": "The trend is stable with no recent changes."}],
        "what_this_means": (
            "📊 We look at the stock's AVERAGE price over different time periods. "
            "When shorter averages cross ABOVE longer averages = trend turning UP (bullish). "
            "When shorter averages cross BELOW longer averages = trend turning DOWN (bearish). "
            "Think of it like: if today's weather is warmer than the monthly average, summer might be coming!"
        )
    }


@log_exceptions()
def detect_rsi_divergence(symbol: str) -> dict[str, Any]:
    """
    Check if the stock might be due for a reversal.
    RSI = measures if a stock has been bought or sold too aggressively.
    """
    df = get_price_data(symbol, days=60)
    if df is None or len(df) < 30:
        return {"error": f"Need more data for {symbol}"}

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    current_rsi = float(df['RSI'].iloc[-1])

    # Determine zone with plain English
    if current_rsi > 70:
        zone = "🔴 OVERBOUGHT"
        zone_explanation = (
            f"The stock's momentum score is {current_rsi:.0f}/100. "
            "Above 70 means it's been bought very aggressively recently. "
            "It might be 'tired' and due for a pullback. Consider waiting for a dip before buying."
        )
    elif current_rsi < 30:
        zone = "🟢 OVERSOLD"
        zone_explanation = (
            f"The stock's momentum score is {current_rsi:.0f}/100. "
            "Below 30 means it's been sold very aggressively recently. "
            "It might be 'oversold' and due for a bounce. Could be a buying opportunity!"
        )
    else:
        zone = "⚪ NEUTRAL"
        zone_explanation = (
            f"The stock's momentum score is {current_rsi:.0f}/100. "
            "Between 30-70 is normal territory. No extreme buying or selling pressure."
        )

    # Look for divergences in last 20 bars
    recent = df.tail(20)
    price_min_idx = recent['Close'].idxmin()
    price_max_idx = recent['Close'].idxmax()
    current_price = recent['Close'].iloc[-1]

    divergences = []

    # Bullish divergence
    if price_min_idx != recent.index[-1]:
        price_at_min = recent.loc[price_min_idx, 'Close']
        rsi_at_min = recent.loc[price_min_idx, 'RSI']

        if current_price <= price_at_min * 1.02 and current_rsi > rsi_at_min + 5:
            divergences.append({
                "signal": "🟢 HIDDEN STRENGTH",
                "meaning": "Potential reversal UP",
                "plain_english": (
                    "Interesting! The price is at recent lows, BUT the underlying momentum is actually getting stronger. "
                    "This 'hidden strength' often appears before a stock bounces back up. Could be a sneaky buying opportunity."
                )
            })

    # Bearish divergence
    if price_max_idx != recent.index[-1]:
        price_at_max = recent.loc[price_max_idx, 'Close']
        rsi_at_max = recent.loc[price_max_idx, 'RSI']

        if current_price >= price_at_max * 0.98 and current_rsi < rsi_at_max - 5:
            divergences.append({
                "signal": "🔴 HIDDEN WEAKNESS",
                "meaning": "Potential reversal DOWN",
                "plain_english": (
                    "Warning! The price looks strong at recent highs, BUT the underlying momentum is actually getting weaker. "
                    "This 'hidden weakness' often appears before a stock drops. Consider taking profits or being cautious."
                )
            })

    return {
        "symbol": symbol,
        "momentum_score": round(current_rsi, 0),
        "zone": zone,
        "zone_explanation": zone_explanation,
        "hidden_signals": divergences if divergences else [{"signal": "None detected", "plain_english": "No hidden reversal signals at this time."}],
        "what_this_means": (
            "📊 We measure the stock's 'momentum' - how aggressively it's being bought or sold. "
            "Score 0-30: Oversold (everyone's selling, might bounce). "
            "Score 70-100: Overbought (everyone's buying, might drop). "
            "Score 30-70: Normal range."
        )
    }


@cached(key_func=lambda symbol: f"patterns:{symbol.upper()}")
@log_exceptions()
def detect_patterns(symbol: str) -> dict[str, Any]:
    """
    MAIN FUNCTION: Complete pattern analysis with simple explanations.
    Combines all signals into an easy-to-understand summary.
    """
    results = {
        "symbol": symbol,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Get all analyses
    sr = find_support_resistance(symbol)
    ma = check_ma_crossover(symbol)
    momentum = detect_rsi_divergence(symbol)

    # Count bullish vs bearish signals
    bullish_points = 0
    bearish_points = 0
    key_findings = []

    # Analyze support/resistance
    if "error" not in sr:
        results["price_levels"] = sr
        price = sr["current_price"]

        # Check if near support (bullish) or resistance (bearish)
        for sup in sr.get("support_levels", [])[:1]:
            if price < sup["price"] * 1.03:
                bullish_points += 1
                key_findings.append(f"✅ Near a price floor at ${sup['price']} (good buying zone)")

        for res in sr.get("resistance_levels", [])[:1]:
            if price > res["price"] * 0.97:
                bearish_points += 1
                key_findings.append(f"⚠️ Near a price ceiling at ${res['price']} (may struggle to go higher)")

    # Analyze trend
    if "error" not in ma:
        results["trend_analysis"] = ma

        if "UPTREND" in (ma.get("overall_trend") or ""):
            bullish_points += 2
            key_findings.append("✅ Stock is in an uptrend (price above average)")
        elif "DOWNTREND" in (ma.get("overall_trend") or ""):
            bearish_points += 2
            key_findings.append("⚠️ Stock is in a downtrend (price below average)")

        for signal in ma.get("signals", []):
            if "GOLDEN" in (signal.get("signal") or ""):
                bullish_points += 3
                key_findings.append("🌟 GOLDEN CROSS detected - major buy signal!")
            elif "DEATH" in (signal.get("signal") or ""):
                bearish_points += 3
                key_findings.append("💀 DEATH CROSS detected - major warning!")
            elif "Turning Up" in (signal.get("signal") or ""):
                bullish_points += 1
                key_findings.append("✅ Short-term momentum improving")
            elif "Turning Down" in (signal.get("signal") or ""):
                bearish_points += 1
                key_findings.append("⚠️ Short-term momentum weakening")

    # Analyze momentum
    if "error" not in momentum:
        results["momentum_analysis"] = momentum

        if "OVERSOLD" in (momentum.get("zone") or ""):
            bullish_points += 2
            key_findings.append("✅ Stock is oversold - may be due for a bounce")
        elif "OVERBOUGHT" in (momentum.get("zone") or ""):
            bearish_points += 1
            key_findings.append("⚠️ Stock is overbought - may be due for a pullback")

        for signal in momentum.get("hidden_signals", []):
            if "STRENGTH" in (signal.get("signal") or ""):
                bullish_points += 2
                key_findings.append("✅ Hidden strength detected - reversal up likely")
            elif "WEAKNESS" in (signal.get("signal") or ""):
                bearish_points += 2
                key_findings.append("⚠️ Hidden weakness detected - reversal down possible")

    # Generate verdict
    total_points = bullish_points + bearish_points

    if total_points == 0:
        verdict = "⚪ NEUTRAL"
        verdict_explanation = (
            f"No strong signals detected for {symbol}. "
            "The stock isn't showing clear direction right now. "
            "Consider waiting for a clearer signal before making a move."
        )
        action = "WAIT - No clear signal"
    elif bullish_points > bearish_points:
        strength = "STRONG" if bullish_points >= 4 else "MODERATE"
        verdict = f"🟢 BULLISH ({strength})"
        verdict_explanation = (
            f"The technical signals for {symbol} lean POSITIVE. "
            f"Found {bullish_points} bullish signals vs {bearish_points} bearish signals. "
            "The odds favor the stock going UP from here, though nothing is guaranteed."
        )
        action = "CONSIDER BUYING" if bullish_points >= 4 else "LEAN BULLISH - Watch for entry"
    else:
        strength = "STRONG" if bearish_points >= 4 else "MODERATE"
        verdict = f"🔴 BEARISH ({strength})"
        verdict_explanation = (
            f"The technical signals for {symbol} lean NEGATIVE. "
            f"Found {bearish_points} bearish signals vs {bullish_points} bullish signals. "
            "The odds favor the stock going DOWN from here. Caution is warranted."
        )
        action = "CONSIDER SELLING" if bearish_points >= 4 else "LEAN BEARISH - Be cautious"

    results["verdict"] = verdict
    results["verdict_explanation"] = verdict_explanation
    results["suggested_action"] = action
    results["key_findings"] = key_findings if key_findings else ["No significant patterns detected"]
    results["signal_score"] = {"bullish": bullish_points, "bearish": bearish_points}

    # Simple bottom line
    results["bottom_line"] = (
        f"📊 {symbol} ANALYSIS: {verdict}\n"
        f"💡 {verdict_explanation}\n"
        f"🎯 Suggested action: {action}"
    )

    return results


if __name__ == "__main__":
    print("=== Testing Pattern Recognition ===\n")

    symbol = "NVDA"
    print(f"Analyzing {symbol}...\n")

    result = detect_patterns(symbol)
    print(result["bottom_line"])
    print("\nKey Findings:")
    for finding in result["key_findings"]:
        print(f"  {finding}")
