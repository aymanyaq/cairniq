
from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol, years=10: f"seasonality:{symbol.upper()}:{years}")
@log_exceptions()
def analyze_seasonality(symbol: str, years: int = 10) -> dict[str, Any]:
    """
    Analyzes monthly seasonality patterns for a given symbol over the last N years.
    Returns average monthly return and win rate (percent of positive months).
    """
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=f"{years}y")

        if len(history) < 252: # Need at least 1 year
             return {"symbol": symbol, "error": "Insufficient data"}

        # Calculate monthly returns
        # Resample to month end, get percent change
        monthly_data = history['Close'].resample('ME').last().pct_change()

        # Group by month (1=Jan, 12=Dec)
        # We need a DataFrame to group
        df_monthly = monthly_data.to_frame(name="return")
        df_monthly['month'] = df_monthly.index.month

        seasonality = []
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        for m in range(1, 13):
            month_returns = df_monthly[df_monthly['month'] == m]['return']

            if len(month_returns) > 0:
                # Cast to native float: Series.mean()/.sum() yield numpy scalars
                # that leak as "np.float64(...)" when the dict is str()-rendered.
                avg_ret = float(month_returns.mean())
                win_rate = float((month_returns > 0).sum() / len(month_returns))

                seasonality.append({
                    "month": month_names[m-1],
                    "average_return_pct": round(avg_ret * 100, 2),
                    "win_rate_pct": round(win_rate * 100, 0)
                })

        return {
            "symbol": symbol.upper(),
            "period_years": years,
            "seasonality": seasonality,
            "best_month": max(seasonality, key=lambda x: x['average_return_pct'])['month'] if seasonality else "N/A",
            "worst_month": min(seasonality, key=lambda x: x['average_return_pct'])['month'] if seasonality else "N/A"
        }

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

if __name__ == "__main__":
    print(analyze_seasonality("NVDA"))
