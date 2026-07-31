
"""
Market Mechanics Tools
Analyzes Sector Rotation and Relative Strength to identify market leadership.
"""
import time
from typing import Any

import pandas as pd
import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLY": "Consumer Discret",
    "XLC": "Comm Services",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials"
}

@cached(key_func=lambda: "mm_sector_rotation")
@log_exceptions()
def detect_sector_rotation() -> dict[str, Any]:
    """
    Analyzes the 11 Major S&P 500 Sectors to detect rotation. TREND version.

    ⚠️ There is a SECOND live function with this exact name:
    `tools.sector_rotation.detect_sector_rotation`. They are both correct and
    they measure DIFFERENT things — importing the wrong one silently changes
    what a caller is asking:

        this one (market_mechanics)  → TREND QUADRANT (RRG-style)
            "is this sector up over 1M and 3M?"
            signs of mom_1m / mom_3m -> Leading / Weakening / Improving / Lagging
            one batched 6mo download for all 11 ETFs; adds RSI(14)
            feeds: the `check_sector_rotation` agent tool, scan-universe
            assembly, DeepReasoning preflight

        sector_rotation             → FLOW / ACCELERATION
            "is this sector outpacing its own 3-month run rate?"
            return_1m - (return_3m / 3) -> INFLOW / OUTFLOW / NEUTRAL
            11 sequential per-ticker fetches over 4mo
            feeds: the market-pulse heatmap

    They can label one sector "Leading 🟢" here and "🔴 OUTFLOW" there in the
    same turn, and that is NOT a contradiction to be unified away: a sector that
    rose 12% over three months and 1% in the last one IS both leading on trend
    and decelerating on flow. Forcing them to agree would delete a real signal.
    The payload carries `methodology` so a consumer can tell which it holds.
    """
    try:
        symbols = list(SECTOR_ETFS.keys())
        # Robust download with retry for I/O errors in concurrent environments
        data = None
        for attempt in range(3):
            try:
                data = yf.download(symbols, period="6mo", progress=False, threads=False)
                if not data.empty:
                    break
            except (OSError, ValueError) as e:
                if "closed file" in str(e).lower() and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

        if data is None or data.empty:
            return {"error": "Could not fetch sector data (Empty response)"}

        # yfinance structure varies (MultiIndex).
        # If MultiIndex columns (Price, Ticker), we need to extract 'Close'
        if isinstance(data.columns, pd.MultiIndex):
             close = data["Close"]
        else:
             close = data["Close"] if "Close" in data else data

        results = []
        for symbol, name in SECTOR_ETFS.items():
            if symbol not in close.columns:
                continue

            prices = close[symbol].dropna()
            if len(prices) < 20:
                continue

            # Cast to native float at extraction: numpy scalars leak as
            # "np.float64(...)" into str()-rendered tool output (round() does
            # not strip the numpy type).
            current = float(prices.iloc[-1])
            one_month = float(prices.iloc[-22] if len(prices) >= 22 else prices.iloc[0])
            three_month = float(prices.iloc[-66] if len(prices) >= 66 else prices.iloc[0])

            mom_1m = ((current - one_month) / one_month) * 100
            mom_3m = ((current - three_month) / three_month) * 100

            # --- NEW: RSI for Reversal Detection ---
            rsi_val = 50.0
            try:
                from ta.momentum import RSIIndicator
                rsi_series = RSIIndicator(close=prices, window=14).rsi()
                rsi_val = float(rsi_series.iloc[-1])
            except ImportError:
                pass # ta lib might not be installed

            # Simple RRG-like Logic (Relative Rotation Graph concept)
            # Leading: Positive 3M, Positive 1M
            # Weakening: Positive 3M, Negative 1M
            # Improving: Negative 3M, Positive 1M
            # Lagging: Negative 3M, Negative 1M

            status_icon = ""
            if rsi_val > 70: status_icon = "⚠️ Overbought"
            elif rsi_val < 30: status_icon = "🛒 Oversold"

            if mom_3m > 0 and mom_1m > 0:
                trend = f"Leading 🟢 {status_icon}"
            elif mom_3m > 0 and mom_1m < 0:
                trend = f"Weakening 🟡 {status_icon}"
            elif mom_3m < 0 and mom_1m > 0:
                trend = f"Improving 🔵 {status_icon}"
            else:
                trend = f"Lagging 🔴 {status_icon}"

            results.append({
                "symbol": symbol,
                "sector": name,
                "trend": trend,
                "rsi": round(rsi_val, 1),
                "1m_change": round(mom_1m, 2),
                "3m_change": round(mom_3m, 2),
                "combined_score": round(mom_1m + mom_3m, 2)
            })

        # Sort by combined momentum (short + medium term)
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        # Ranking is by combined (1M+3M) momentum, so show both horizons —
        # a lone 1M figure misreads (e.g. a +43% 3M leader in a -5% 1M pullback).
        top_sectors = [f"{r['sector']} (1M {r['1m_change']:+.1f}%, 3M {r['3m_change']:+.1f}%)" for r in results[:3]]
        bottom_sectors = [f"{r['sector']} (1M {r['1m_change']:+.1f}%, 3M {r['3m_change']:+.1f}%)" for r in results[-3:]]

        return {
            "methodology": "trend_quadrant",
            "market_status": "Bullish Rotation" if results and results[0]["3m_change"] > 0 else "Bearish/Defensive",
            "leading_sectors": top_sectors,
            "lagging_sectors": bottom_sectors,
            "full_rotation_map": results,
            "interpretation": (
                "Money is flowing INTO 'Leading' and 'Improving' sectors. "
                "Money is flowing OUT of 'Weakening' and 'Lagging' sectors."
            )
        }

    except Exception as e:
        return {"error": f"Sector rotation analysis failed: {str(e)}"}


@cached(key_func=lambda symbols, benchmark="SPY": f"rel_strength:{','.join(symbols).upper() if isinstance(symbols, list) else symbols.upper()}:{benchmark.upper()}")
@log_exceptions()
def rank_relative_strength(symbols: str, benchmark: str = "SPY") -> dict[str, Any]:
    """
    Ranks a list of symbols by Relative Strength vs Benchmark.
    Identifies Leaders (Outperforming) vs Laggards (Underperforming).
    """
    try:
        if isinstance(symbols, list):
            sym_list = [s.strip().upper() for s in symbols]
        else:
            sym_list = [s.strip().upper() for s in symbols.split(",")]
        if not sym_list:
            return {"error": "No symbols provided"}

        # Add benchmark
        fetch_list = sym_list + [benchmark]

        # Robust fetch with retry for I/O errors in concurrent environments
        data = None
        for attempt in range(3):
            try:
                data = yf.download(fetch_list, period="6mo", progress=False, threads=False)
                if not data.empty:
                    break
            except (OSError, ValueError) as e:
                if "closed file" in str(e).lower() and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

        if data is None or data.empty:
            return {"error": "Could not fetch data for any of the symbols provided."}

        if isinstance(data.columns, pd.MultiIndex):
             close = data["Close"]
        else:
             # Handle single symbol case if yfinance returns simple DataFrame
             if "Close" in data:
                 # If it returned [Benchmark, Sym] but only Close exists as level
                 # or if it's a simple DF with Close column
                 if isinstance(data["Close"], pd.Series):
                      # If 1 ticker, close is Series. We want it as DF.
                      # This usually shouldn't happen with 2+ symbols but let's be safe
                      close = data[["Close"]]
                 else:
                      close = data["Close"]
             else:
                 close = data

        # Ensure close is a DataFrame
        if isinstance(close, pd.Series):
            close = close.to_frame()

        if benchmark not in close.columns:
            # Try to see if benchmark is the ONLY ticker and close is just one column
            if len(close.columns) == 1 and sym_list[0] == benchmark:
                 pass # expected
            else:
                 return {"error": f"Benchmark {benchmark} not found in data. Available: {list(close.columns)}"}

        benchmark_prices = close[benchmark].dropna()
        if len(benchmark_prices) < 2:
            return {"error": f"Insufficient benchmark data for {benchmark}"}

        # Get 3-month return (approx 66 trading days)
        bench_curr = benchmark_prices.iloc[-1]
        bench_start = benchmark_prices.iloc[-66] if len(benchmark_prices) >= 66 else benchmark_prices.iloc[0]
        bench_ret_3m = (bench_curr / bench_start) - 1

        rankings = []
        for sym in sym_list:
            if sym not in close.columns:
                continue

            prices = close[sym].dropna()
            if len(prices) < 5: # Lower threshold: need at least 1 week
                continue

            # Calculate RS Score (Price performance relative to SPY)
            # Not just raw return, but the ratio trend

            # Simple Absolute Return Comparison
            curr = prices.iloc[-1]
            # Use 66 days if available, otherwise use earliest available
            start_3m = prices.iloc[-66] if len(prices) >= 66 else prices.iloc[0]

            ret_3m = (curr / start_3m) - 1

            # Relative Performance (Alpha)
            # Adjust benchmark return if the local symbol has < 66 days
            if len(prices) < 66:
                # Get benchmark return for the EXACT same timeframe as the symbol
                bench_aligned = benchmark_prices.loc[benchmark_prices.index.intersection(prices.index)]
                if len(bench_aligned) >= 2:
                    local_bench_ret = (bench_aligned.iloc[-1] / bench_aligned.iloc[0]) - 1
                else:
                    local_bench_ret = bench_ret_3m
            else:
                local_bench_ret = bench_ret_3m

            rel_perf = ret_3m - local_bench_ret

            rankings.append({
                "symbol": sym,
                "3m_return": f"{ret_3m*100:.1f}%",
                "vs_benchmark": f"{rel_perf*100:.1f}%",
                "status": "Leader 🏆" if rel_perf > 0.05 else "Outperformer 🟢" if rel_perf > 0 else "Laggard 🔴",
                "raw_rel_score": float(rel_perf)
            })

        # Sort by relative score
        rankings.sort(key=lambda x: x["raw_rel_score"], reverse=True)

        return {
            "benchmark": benchmark,
            "benchmark_3m_return": f"{bench_ret_3m*100:.1f}%",
            "rankings": rankings,
            "summary": f"Top Leader: {rankings[0]['symbol']}" if rankings else "No valid data"
        }

    except Exception as e:
        return {"error": f"RS ranking failed: {str(e)}"}


@cached(key_func=lambda symbol: f"earnings_surprise:{symbol.upper()}")
@log_exceptions()
def predict_earnings_surprise(symbol: str) -> dict[str, Any]:
    """
    Calculates the probability of an earnings beat based on historical data.
    Analyzes the 'Surprise %' from the last 4-8 quarters.
    """
    try:
        ticker = yf.Ticker(symbol)

        # yfinance often returns 'earnings_history' or we can infer from quarterly financials?
        # Actually 'earnings_dates' or 'calendar' might show past beats.
        # But `get_earnings_history` isn't standard in all yf versions.
        # Let's try `earnings_dates` which often has 'Surprise(%)' column in newer versions.

        # Alternative: Use quarterly earnings to see trend.
        # But 'Estimates' vs 'Actuals' is what defines a surprise.
        # Ideally we'd use `ticker.earnings_history` if available.
        # A robust fallback is to fetch checking the 'calendar' sometimes has previous.

        # Let's try accessing the protected method or common property if available.
        # Some versions expose `earnings_history` dataframe.

        # For this implementation, I will simulate logic if direct history is unavailable,
        # or rely on `quarterly_financials` growth consistency as a proxy for "operational excellence".

        # BETTER: Use `ticker.earnings_dates`.
        # HARD TIMEOUT: yfinance has no request timeout and this property has been
        # observed to hang for ~1000s on flaky tickers, blocking the whole scan
        # pipeline. Bound it to 8s; a timeout is treated as "no data" (graceful).
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as _FTimeout
        with ThreadPoolExecutor(max_workers=1) as _ex:
            try:
                dates = _ex.submit(lambda: ticker.earnings_dates).result(timeout=8)
            except _FTimeout:
                return {"symbol": symbol, "error": "earnings_dates timed out"}
        if dates is None or dates.empty:
             return {"symbol": symbol, "error": "No earnings history found"}

        # Filter for past dates with 'Surprise(%)'
        # Columns usually: EPS Estimate, Reported EPS, Surprise(%)
        if "Surprise(%)" not in dates.columns:
            # Try to calculate if we have Reported and Estimate
            if "Reported EPS" in dates.columns and "EPS Estimate" in dates.columns:
                 dates["Surprise(%)"] = (dates["Reported EPS"] - dates["EPS Estimate"]) / dates["EPS Estimate"].abs()
            else:
                 return {"symbol": symbol, "error": "Earnings surprise data missing from feed"}

        # Get last 8 quarters (2 years)
        past_earnings = dates.dropna(subset=["Surprise(%)"]).head(8)

        if past_earnings.empty:
            return {"symbol": symbol, "note": "Insufficient past surprise data"}

        beats = past_earnings[past_earnings["Surprise(%)"] > 0]
        beat_rate = len(beats) / len(past_earnings)

        avg_surprise = past_earnings["Surprise(%)"].mean() * 100

        classification = "Wildcard 🎲"
        if beat_rate >= 0.85:
            classification = "High Probability Beat 🚀 (Priced for Perfection)"
        elif beat_rate >= 0.7:
             classification = "Likely Beat 📈"
        elif beat_rate <= 0.3:
             classification = "Likely Miss 🔻"

        return {
            "symbol": symbol,
            "history_analyzed": f"{len(past_earnings)} quarters",
            "beat_rate": f"{beat_rate*100:.0f}%",
            "average_surprise": f"{avg_surprise:.2f}%",
            "classification": classification,
            "interpretation": (
                f"Historically beats estimates {int(beat_rate*100)}% of the time. "
                + ("⚠️ High beat rate indicates expectations may be priced for perfection; a 'whisper miss' could cause asymmetric downside risk." if beat_rate >= 0.85 else "")
            )
        }

    except Exception as e:
        return {"symbol": symbol, "error": f"Surprise prediction failed: {str(e)}"}
