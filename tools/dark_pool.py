
from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol: f"dark_pool:{symbol.upper()}")
@log_exceptions()
def scan_dark_pool_proxy(symbol: str) -> dict[str, Any]:
    """
    Scans for 'Dark Pool' signatures by analyzing 1-minute intraday volume spikes.
    Institutional block trades often appear as massive volume spikes on the 1-minute chart
    that don't immediately move price (iceberg orders).
    """
    try:
        ticker = yf.Ticker(symbol)
        # Get 1 day of 1 minute data
        df = ticker.history(period="1d", interval="1m")

        if df.empty:
            return {"symbol": symbol, "error": "No intraday data available"}

        # Calculate stats
        mean_vol = df['Volume'].mean()
        std_vol = df['Volume'].std()

        # Threshold: 3 Sigma (99.7% percentile events)
        # Or simplistic: > 5x average volume
        threshold = mean_vol + (3 * std_vol)

        # Find spikes
        spikes = df[df['Volume'] > threshold].copy()

        alerts = []
        if not spikes.empty:
            for index, row in spikes.iterrows():
                # Format time
                time_str = index.strftime("%H:%M")
                vol = int(row['Volume'])
                price = row['Close']
                limit_ratio = vol / mean_vol

                # Check for price movement during spike
                # Real dark pools often happen with minimal price impact (Doji candle)
                open_p = row['Open']
                close_p = row['Close']
                pct_move = abs((close_p - open_p) / open_p)

                signature = "BLOCK TRADE"
                if pct_move < 0.0005: # Less than 0.05% move
                    signature = "DARK POOL PRINT (Hidden)"
                elif close_p > open_p:
                    signature = "AGGRESSIVE BUY"
                else:
                    signature = "AGGRESSIVE SELL"

                alerts.append({
                    "time": time_str,
                    "volume": f"{vol:,}",
                    "price": f"${price:.2f}",
                    "signature": signature,
                    "magnitude": f"{limit_ratio:.1f}x Normal"
                })

        # Summary
        total_vol = df['Volume'].sum()
        dark_vol = spikes['Volume'].sum()
        dark_pct = (dark_vol / total_vol) * 100 if total_vol > 0 else 0

        return {
            "symbol": symbol.upper(),
            "dark_pool_activity_pct": f"{dark_pct:.1f}%",
            "alerts_count": len(alerts),
            "alerts": alerts,
            "note": "Alerts represent statistical volume anomalies (>3σ) suggestive of institutional block trades."
        }

    except Exception as e:
        return {"symbol": symbol, "error": f"Dark Pool scan failed: {e}"}

if __name__ == "__main__":
    print(scan_dark_pool_proxy("SPY"))
