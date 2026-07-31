from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbols, period="1y": f"correlation:{','.join(sorted(s.upper() for s in symbols))}:{period}")
@log_exceptions()
def analyze_correlation(symbols: list[str], period: str = "1y") -> dict[str, Any]:
    """
    Analyzes correlation between multiple assets.
    Helps identify diversification opportunities and concentration risks.
    """
    try:
        if len(symbols) < 2:
            return {"error": "Need at least 2 symbols for correlation analysis"}

        # Fetch historical data
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if not hist.empty:
                data[symbol.upper()] = hist["Close"]

        if len(data) < 2:
            return {"error": "Insufficient data for correlation analysis"}

        # Create DataFrame and calculate returns
        df = pd.DataFrame(data)
        returns = df.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Find highest correlations (excluding self-correlation)
        high_corr_pairs = []
        for i, sym1 in enumerate(corr_matrix.columns):
            for j, sym2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr = corr_matrix.loc[sym1, sym2]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append({
                            "pair": f"{sym1}-{sym2}",
                            "correlation": f"{corr:.2f}",
                            "warning": "High concentration risk" if corr > 0.7 else "Strong negative correlation"
                        })

        # Calculate average correlation
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.where(mask).stack().mean()

        diversification = "Well diversified" if avg_corr < 0.5 else "Concentrated"

        return {
            "symbols_analyzed": list(data.keys()),
            "period": period,
            "average_correlation": f"{avg_corr:.2f}",
            "diversification_assessment": diversification,
            "high_correlation_pairs": high_corr_pairs if high_corr_pairs else "None found",
            "recommendation": "Consider adding uncorrelated assets" if avg_corr > 0.6 else "Portfolio shows good diversification"
        }

    except Exception as e:
        return {"error": f"Correlation analysis failed: {e}"}

if __name__ == "__main__":
    print(analyze_correlation(["AAPL", "MSFT", "GOOGL", "VTI"]))
