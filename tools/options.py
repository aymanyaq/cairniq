from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol: f"options:{symbol.upper()}")
@log_exceptions()
def analyze_options(symbol: str) -> dict[str, Any]:
    """
    Analyzes options chain data and explains it in plain English.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get available expiration dates
        expirations = ticker.options
        if not expirations:
            return {"error": f"No options data available for {symbol}"}

        # Use the nearest expiration
        nearest_exp = expirations[0]

        # Get options chain
        opts = ticker.option_chain(nearest_exp)
        calls = opts.calls
        puts = opts.puts

        if calls.empty or puts.empty:
            return {"error": "Options chain is empty"}

        # Get current stock price
        info = ticker.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        # Calculate key metrics
        # 1. Put/Call volume ratio
        total_call_volume = calls["volume"].sum() if "volume" in calls else 0
        total_put_volume = puts["volume"].sum() if "volume" in puts else 0
        pc_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0

        # 2. Find ATM options (closest to current price)
        calls["distance"] = abs(calls["strike"] - current_price)
        puts["distance"] = abs(puts["strike"] - current_price)

        atm_call = calls.loc[calls["distance"].idxmin()] if not calls.empty else None
        puts.loc[puts["distance"].idxmin()] if not puts.empty else None

        # 3. Implied Volatility at ATM
        atm_iv = atm_call["impliedVolatility"] if atm_call is not None else None

        # Sentiment interpretation with PLAIN ENGLISH explanations
        if pc_ratio > 1.2:
            sentiment = "🔴 BEARISH - More traders are betting the stock will GO DOWN"
            sentiment_explanation = (
                f"For every 100 people betting the stock will rise, {int(pc_ratio * 100)} are betting it will fall. "
                "This suggests pessimism."
            )
        elif pc_ratio < 0.8:
            sentiment = "🟢 BULLISH - More traders are betting the stock will GO UP"
            sentiment_explanation = (
                "Traders are making more bets that the stock will rise than fall. "
                "This suggests optimism in the market."
            )
        else:
            sentiment = "⚪ NEUTRAL - No strong bias either way"
            sentiment_explanation = "Roughly equal bets on the stock going up vs down."

        # Explain implied volatility in plain terms
        if atm_iv:
            iv_pct = atm_iv * 100
            if iv_pct > 60:
                iv_explanation = f"🔥 VERY HIGH VOLATILITY ({iv_pct:.0f}%) - Traders expect BIG price swings. Risky but potentially rewarding."
            elif iv_pct > 35:
                iv_explanation = f"⚡ ELEVATED VOLATILITY ({iv_pct:.0f}%) - Expect some price movement. Stock could swing 2-5% easily."
            else:
                iv_explanation = f"📊 NORMAL VOLATILITY ({iv_pct:.0f}%) - Stock is expected to be relatively calm."
        else:
            iv_explanation = "Volatility data not available"

        return {
            "symbol": symbol.upper(),
            "current_price": f"${current_price:.2f}",
            "what_traders_are_betting": sentiment,
            "explanation": sentiment_explanation,
            "expected_price_swings": iv_explanation,
            "bottom_line": (
                "Traders seem optimistic about this stock" if pc_ratio < 0.8
                else "Traders seem worried about this stock" if pc_ratio > 1.2
                else "Traders are unsure which way this stock will go"
            )
        }

    except Exception as e:
        return {"error": f"Options analysis failed: {e}"}

@cached(key_func=lambda symbol: f"unusual_options:{symbol.upper()}")
@log_exceptions()
def scan_unusual_activity(symbol: str) -> dict[str, Any]:
    """
    Scans for 'Unusual Options Activity' - High IV, Volume > Open Interest, or massive skew.
    Used for finding 'Gamma Squeeze' or explosive move candidates.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Check nearest expiration for immediate action
        if not ticker.options:
            return {"symbol": symbol, "status": "No options"}

        near_date = ticker.options[0]
        chain = ticker.option_chain(near_date)
        calls = chain.calls
        puts = chain.puts

        alerts = []

        # 1. High IV Alert
        avg_iv = calls['impliedVolatility'].mean() * 100
        if avg_iv > 80:
            alerts.append(f"🔥 EXTREME VOLATILITY: IV is {avg_iv:.1f}%. Options are very expensive!")
        elif avg_iv > 50:
             alerts.append(f"⚡ High Volatility: IV is {avg_iv:.1f}%. Expect large moves.")

        # 2. Volume > Open Interest (Smart Money/Hedge Fund activity?)
        # Filter for significant volume (>500 contracts) to avoid noise
        unusual_calls = calls[(calls['volume'] > 500) & (calls['volume'] > calls['openInterest'])]

        for _, row in unusual_calls.iterrows():
            strike = row['strike']
            vol = row['volume']
            oi = row['openInterest']
            alerts.append(f"🐋 UNUSUAL CALL BUYING: Strike ${strike} (Vol: {vol} > OI: {oi}) - Possible pump/breakout targeting ${strike}")

        unusual_puts = puts[(puts['volume'] > 500) & (puts['volume'] > puts['openInterest'])]
        for _, row in unusual_puts.iterrows():
            strike = row['strike']
            vol = row['volume']
            oi = row['openInterest']
            alerts.append(f"📉 UNUSUAL PUT BUYING: Strike ${strike} (Vol: {vol} > OI: {oi}) - Possible crash targeting ${strike}")

        # 3. Gamma Exposure Proxy (Net Call Volume vs Put Volume)
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()

        if total_call_vol > 3 * total_put_vol and total_call_vol > 10000:
            alerts.append(f"🐂 BULLISH FRENZY: Call volume is {total_call_vol/total_put_vol:.1f}x higher than Puts!")

        verdict = "NORMAL"
        if len(alerts) > 2:
            verdict = "🚨 HIGH ACTIVITY DETECTED"
        elif len(alerts) > 0:
            verdict = "⚠️ MODERATE ACTIVITY"

        return {
            "symbol": symbol,
            "verdict": verdict,
            "avg_iv": f"{avg_iv:.1f}%",
            "alerts": alerts,
            "note": "Unusual activity usually precedes a major move (up or down)."
        }

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

@cached(key_func=lambda symbol: f"option_walls:{symbol.upper()}")
@log_exceptions()
def get_option_walls(symbol: str) -> dict[str, Any]:
    """
    Identifies 'Call Walls' (Resistance) and 'Put Walls' (Support)
    based on the strikes with the highest Open Interest.
    """
    try:
        ticker = yf.Ticker(symbol)
        if not ticker.options: return {}

        # Look at nearest expiry with volume
        expiry = ticker.options[0]
        chain = ticker.option_chain(expiry)

        # Call Wall = Strike with max Call OI (Dealers short gamma, defends level)
        call_wall = chain.calls.loc[chain.calls['openInterest'].idxmax()]

        # Put Wall = Strike with max Put OI (Dealers long gamma, supports level)
        put_wall = chain.puts.loc[chain.puts['openInterest'].idxmax()]

        return {
            "expiry": expiry,
            "call_wall": call_wall['strike'],
            "call_wall_oi": int(call_wall['openInterest']),
            "put_wall": put_wall['strike'],
            "put_wall_oi": int(put_wall['openInterest']),
            "interpretation": f"Support likely at ${put_wall['strike']} (Put Wall), Resistance at ${call_wall['strike']} (Call Wall)."
        }
    except Exception:
        return {}

@cached(key_func=lambda symbol: f"whale_accum:{symbol.upper()}")
@log_exceptions()
def check_whale_accumulation(symbol: str) -> dict[str, Any]:
    """
    Scans for 'Whale' accumulation: Deep ITM Call Sweeps.
    This is often used by institutions to replace stock exposure with leverage.
    """
    try:
        ticker = yf.Ticker(symbol)
        if not ticker.options: return {}

        # Check separately for longer dated options (more significant)
        # Look 2-3 months out
        target_date = ticker.options[0]
        if len(ticker.options) > 2:
            target_date = ticker.options[2]

        chain = ticker.option_chain(target_date)
        calls = chain.calls

        # Logic: ITM Calls (Strike < Current Price), High Volume
        current_price = ticker.info.get("currentPrice", 0)
        if current_price == 0:
             hist = ticker.history(period="1d")
             if not hist.empty: current_price = hist["Close"].iloc[-1]

        itm_calls = calls[ (calls['strike'] < current_price) & (calls['volume'] > 500) ].copy()

        whales = []
        for _, row in itm_calls.iterrows():
            strike = row['strike']
            vol = row['volume']
            oi = row['openInterest']
            if vol > oi:
                whales.append(f"🐋 WHALE ALERT: Sweeping ITM Calls ${strike} Exp {target_date} (Vol: {vol} > OI: {oi})")

        return {
            "whale_alerts": whales,
            "count": len(whales)
        }
    except Exception:
        return {}

if __name__ == "__main__":
    print(analyze_options("AAPL"))
    print(scan_unusual_activity("NVDA"))
    print(check_whale_accumulation("AMZN"))

@cached(key_func=lambda symbol: f"dealer_gex:{symbol.upper()}")
@log_exceptions()
def calculate_dealer_gex(symbol: str) -> dict[str, Any]:
    """
    Approximates Dealer Gamma Exposure (GEX) to predict explosive short-term moves.
    When dealers are 'Short Gamma', volatility expands and Gamma Squeezes occur.
    """
    try:
        ticker = yf.Ticker(symbol)
        if not ticker.options:
            return {"symbol": symbol, "status": "No Options Data"}

        near_date = ticker.options[0]
        chain = ticker.option_chain(near_date)
        calls = chain.calls
        puts = chain.puts

        current_price = ticker.info.get("currentPrice", 0)
        if current_price == 0:
             hist = ticker.history(period="1d")
             if not hist.empty: current_price = hist["Close"].iloc[-1]

        # Simplify GEX by looking at Open Interest within 15% of spot
        upper_bound = current_price * 1.15
        lower_bound = current_price * 0.85

        near_calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)]
        near_puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)]

        # Approximate ITM vs OTM Open Interest Proxy
        call_oi_atm = near_calls['openInterest'].sum()
        put_oi_atm = near_puts['openInterest'].sum()

        if put_oi_atm == 0 and call_oi_atm == 0:
            return {"symbol": symbol, "gex_status": "Neutral", "interpretation": "Low Liquidity"}

        ratio = call_oi_atm / (put_oi_atm + 0.0001)

        gex_status = "Neutral"
        interpretation = "Market makers are balanced. Expected volatility is normal."

        if ratio > 2.5 and call_oi_atm > 5000:
             gex_status = "Short Gamma (Squeeze Risk Up)"
             interpretation = "🚨 Dealers are heavily short call options. A small move up could force them to buy the stock, triggering an explosive Gamma Squeeze upward."
        elif ratio < 0.4 and put_oi_atm > 5000:
             gex_status = "Short Gamma (Squeeze Risk Down)"
             interpretation = "🚨 Dealers are heavily short put options. A small move down could force them to short the stock, triggering a violent selloff."
        elif call_oi_atm > 10000 or put_oi_atm > 10000:
             gex_status = "Long Gamma (Price Magnet)"
             interpretation = "Dealers are long gamma. Volatility will likely be suppressed, and the stock will 'pin' around current levels."

        return {
            "symbol": symbol,
            "expiry": near_date,
            "call_oi_proxy": float(call_oi_atm),
            "put_oi_proxy": float(put_oi_atm),
            "gex_status": gex_status,
            "interpretation": interpretation
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
