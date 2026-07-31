from typing import Any

import pandas as pd
import yfinance as yf

from tools.exception_logger import log_exceptions
from tools.yf_utils import dividend_yield_fraction


@log_exceptions()
def project_portfolio_income(symbols: list[str] = None, amounts: list[float] = None) -> dict[str, Any]:
    """
    Project portfolio income based on trailing 12-month dividends.
    Can optionally take a list of symbols and amounts (shares or value).
    If no amounts provided, assumes equal weight or requires external context (this tool is flexible).

    Args:
        symbols: List of ticker symbols
        amounts: List of share counts (if available) or values.
                 For simplicity in this agent, we often pass the *current portfolio* context if available.
                 If amounts is empty, we return Yield % and Estimated Income per $10k invested.
    """
    if amounts is None:
        amounts = []
    if symbols is None:
        symbols = []
    results = []
    total_projected_income = 0.0
    portfolio_value = 0.0

    # Heuristic: If amounts not provided, we analyze yield only.
    analyze_yield_only = len(amounts) != len(symbols)

    income_growth_rate = 0.05 # Conservative 5% dividend growth assumption

    for i, sym in enumerate(symbols):
        clean_sym = sym.strip().upper()
        shares = amounts[i] if not analyze_yield_only else 0

        try:
            ticker = yf.Ticker(clean_sym)

            # 1. Get Current Price
            try:
                hist = ticker.history(period="5d")
                price = hist["Close"].iloc[-1] if not hist.empty else 0
            except Exception:
                price = 0

            if price == 0:
                # Fallback for funds?
                price = 100 # Dummy base for yield calc if price fails, but better to skip

            # 2. Get Dividend History (reliable method)
            # Fetch 2 years to be safe for TTM
            div_hist = ticker.dividends

            ttm_div = 0.0
            yield_pct = 0.0

            if not div_hist.empty:
                # Filter last 365 days
                cutoff = pd.Timestamp.now(tz=div_hist.index.tz) - pd.Timedelta(days=365)
                recent_divs = div_hist[div_hist.index > cutoff]
                ttm_div = recent_divs.sum()

                if price > 0:
                    yield_pct = (ttm_div / price) * 100

            # 3. Fallback to info if history is empty (sometimes happens for new funds)
            #
            # This path read `dividendYield` — a PERCENT — as though it were a
            # fraction, so a 0.32% payer projected annual income at 32% of price:
            # a 100x overstatement of the one number this tool exists to produce.
            # It only fired when the dividend history came back empty, which is
            # why it was never noticed against the (correct) history path above.
            if ttm_div == 0:
                info = ticker.info
                yield_frac = dividend_yield_fraction(info)
                if yield_frac:
                    yield_pct = yield_frac * 100
                    ttm_div = price * yield_frac

            # 4. Calculate Income
            current_value = shares * price if not analyze_yield_only else 10000 # Assume 10k for analysis
            annual_income = ttm_div * shares if not analyze_yield_only else (yield_pct/100) * 10000

            results.append({
                "symbol": clean_sym,
                "yield_pct": f"{yield_pct:.2f}%",
                "annual_income_projected": f"${annual_income:.2f}",
                "metric_used": "Trailing 12m Dividends" if not div_hist.empty else "Info Yield"
            })

            total_projected_income += annual_income
            portfolio_value += current_value

        except Exception as e:
            results.append({"symbol": clean_sym, "error": str(e)})

    # Projection Table
    future_income = {}
    current_inc = total_projected_income

    for year in [1, 5, 10, 15]:
        future_inc = current_inc * ((1 + income_growth_rate) ** year)
        future_income[f"Year {year}"] = f"${future_inc:,.0f}"

    avg_yield = (total_projected_income / portfolio_value * 100) if portfolio_value > 0 else 0

    return {
        "summary": {
            "total_annual_income": f"${total_projected_income:,.2f}",
            "portfolio_yield": f"{avg_yield:.2f}%",
            "assumption": f"Based on ${portfolio_value:,.0f} value (or $10k/symbol if generic)"
        },
        "growth_projection": {
            "assumed_cagr": "5%",
            "timeline": future_income
        },
        "details": results
    }

if __name__ == "__main__":
    # Test
    print(project_portfolio_income(
        ["XESG.TO", "FTEC", "O"],
        [100, 50, 200]
    ))
