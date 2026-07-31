"""
Macro Analysis Tools (Tool Chains)

These tools orchestrate multiple underlying data fetches in parallel to provide
comprehensive analysis in a single round-trip, reducing latency and token usage.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Import underlying tools
# Stock Analysis
from tools.alpha_vantage import get_company_overview, get_quote
from tools.comprehensive_data import get_earnings_calendar, get_institutional_ownership
from tools.exception_logger import log_exceptions
from tools.insider_data import get_detailed_insider_activity, get_insider_and_short_data
from tools.portfolio_analytics import analyze_correlation, calculate_currency_exposure, calculate_portfolio_metrics

# Portfolio Analysis
from tools.portfolio_csv import get_portfolio_summary
from tools.sector_analysis import check_portfolio_allocation
from tools.sentiment_analysis import get_analyst_consensus
from tools.technicals import get_comprehensive_technicals


@log_exceptions()
def run_stock_deep_dive(symbol: str) -> dict[str, Any]:
    """
    Performs a 360-degree deep dive on a specific stock.
    Executes 6+ data checks in parallel:
    - Real-time Price
    - Valuation & Fundamentals
    - Technical Analysis (RSI, Trends)
    - Analyst Ratings & Consensus
    - Insider Trading
    - Institutional Ownership
    - Earnings Data
    """
    clean_symbol = symbol.strip().upper()
    results = {"symbol": clean_symbol}

    from agent.utils import get_st_aware_func
    executor = ThreadPoolExecutor(max_workers=10)
    try:
        # Define tasks
        future_quote = executor.submit(get_st_aware_func(get_quote), clean_symbol)
        future_fund = executor.submit(get_st_aware_func(get_company_overview), clean_symbol)
        future_tech = executor.submit(get_st_aware_func(get_comprehensive_technicals), clean_symbol)
        future_analyst = executor.submit(get_st_aware_func(get_analyst_consensus), clean_symbol)
        future_insider = executor.submit(get_st_aware_func(get_insider_and_short_data), clean_symbol)
        # Coded insider detail (open-market buys/sells separated from grants,
        # exercises and issuer buybacks). Venue-neutral: this is the only insider
        # depth a Canadian listing gets, since it has no Form 4 on EDGAR.
        future_insider_detail = executor.submit(get_st_aware_func(get_detailed_insider_activity), clean_symbol)
        future_inst = executor.submit(get_st_aware_func(get_institutional_ownership), clean_symbol)
        future_earn = executor.submit(get_st_aware_func(get_earnings_calendar), clean_symbol)

        # Collect results (using simplistic error handling per component)
        try: results["price"] = future_quote.result()
        except Exception as e: results["price"] = f"Error: {e}"

        try: results["fundamentals"] = future_fund.result()
        except Exception as e: results["fundamentals"] = f"Error: {e}"

        try: results["technicals"] = future_tech.result()
        except Exception as e: results["technicals"] = f"Error: {e}"

        try: results["analyst_ratings"] = future_analyst.result()
        except Exception as e: results["analyst_ratings"] = f"Error: {e}"

        try: results["insider_activity"] = future_insider.result()
        except Exception as e: results["insider_activity"] = f"Error: {e}"

        try: results["insider_transactions_coded"] = future_insider_detail.result()
        except Exception as e: results["insider_transactions_coded"] = f"Error: {e}"

        try: results["institutional"] = future_inst.result()
        except Exception as e: results["institutional"] = f"Error: {e}"

        try: results["earnings"] = future_earn.result()
        except Exception as e: results["earnings"] = f"Error: {e}"

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results
    return results


@log_exceptions()
def assess_portfolio_risk() -> dict[str, Any]:
    """
    Performs a complete portfolio risk assessment.
    Executes in parallel:
    - Portfolio Snapshot (Values & P&L)
    - Sector Allocation (True Exposure)
    - Risk Metrics (Sharpe, Beta, Volatility)
    - Correlation Matrix (Concentration Risk)
    - Currency Exposure (USD vs CAD)
    """
    # 1. Fetch Portfolio Snapshot (Source of Truth)
    portfolio = get_portfolio_summary()

    if "error" in portfolio or not portfolio.get("holdings"):
        return {"error": "Could not fetch portfolio data", "details": portfolio}

    holdings = portfolio["holdings"]

    # Extract data for analytics
    # Filter out cash/pension for pure equity risk analysis if needed,
    # but for allocation we want everything.

    symbols = []
    market_values = []

    # helper for currency tool
    holdings_dict = {}
    currencies_dict = {}

    for h in holdings:
        sym = h["symbol"]
        val = h.get("value_usd", 0.0)

        symbols.append(sym)
        market_values.append(val)
        holdings_dict[sym] = holdings_dict.get(sym, 0.0) + val
        # Guess currency using general suffixes
        guess = "USD"
        for suffix, cur in {".TO": "CAD", ".V": "CAD", ".VN": "CAD", ".CN": "CAD", ".L": "GBP", ".DE": "EUR", ".PA": "EUR", ".MI": "EUR", ".AS": "EUR", ".AX": "AUD", ".T": "JPY"}.items():
            if suffix in sym.upper():
                guess = cur
                break
        currencies_dict[sym] = h.get("currency") or guess

    total_value = sum(market_values)
    weights = [v / total_value for v in market_values] if total_value > 0 else []

    # Convert totals to CAD/Base Currency for display
    import os
    usd_cad_rate = portfolio.get("usd_cad_rate", float(os.environ.get("USD_TO_CAD", "1.44")))
    total_value_cad = portfolio.get("total_value_cad", total_value * usd_cad_rate)

    base_currency = "USD"
    try:
        from tools.memory import get_profile_base_currency
        base_currency = get_profile_base_currency()
    except Exception:
        base_currency = os.environ.get("BASE_CURRENCY") or os.environ.get("CAIRNIQ_BASE_CURRENCY") or "USD"

    rate_to_base = 1.0
    if base_currency == "CAD":
        total_value_base = total_value_cad
        rate_to_base = usd_cad_rate
    elif base_currency == "USD":
        total_value_base = total_value
    else:
        from tools.portfolio_csv import get_exchange_rate
        rate_to_base = get_exchange_rate("USD", base_currency)
        total_value_base = total_value * rate_to_base

    results = {
        "snapshot": {
            "total_value_base": f"${total_value_base:,.0f} {base_currency}",
            "total_value_cad": f"${total_value_cad:,.0f} CAD",
            "total_value_usd": f"${total_value:,.0f} USD",
            "exchange_rate": f"1 USD = {usd_cad_rate:.2f} CAD",
            "total_gain_loss_pct": portfolio.get("percent_return"),
            "top_winners": portfolio.get("top_winners"),
            "top_losers": portfolio.get("top_losers")
        }
    }

    # Create filtered lists for tradeable symbols only
    from tools.portfolio_csv import get_tradeable_symbols
    tradeable_symbols_set = set(get_tradeable_symbols())

    tradeable_symbols = []
    tradeable_weights = []
    for sym, weight in zip(symbols, weights):
        if sym.upper() in tradeable_symbols_set:
            tradeable_symbols.append(sym)
            tradeable_weights.append(weight)

    from agent.utils import get_st_aware_func
    executor = ThreadPoolExecutor(max_workers=10)
    try:
        # Define tasks
        # Allocation needs ALL symbols and raw amounts to show total exposure
        future_allocation = executor.submit(get_st_aware_func(check_portfolio_allocation), symbols, market_values)

        # Risk Metrics needs tradeable symbols and weights
        future_metrics = executor.submit(get_st_aware_func(calculate_portfolio_metrics), tradeable_symbols, tradeable_weights)

        # Correlation needs tradeable symbols
        future_correlation = executor.submit(get_st_aware_func(analyze_correlation), tradeable_symbols)

        # Currency needs dict {symbol: value, currencies: currencies_dict}
        future_fx = executor.submit(get_st_aware_func(calculate_currency_exposure.invoke), {"holdings": holdings_dict, "currencies": currencies_dict})

        # Collect results
        try: results["allocation"] = future_allocation.result()
        except Exception as e: results["allocation"] = f"Error: {e}"

        try: results["risk_metrics"] = future_metrics.result()
        except Exception as e: results["risk_metrics"] = f"Error: {e}"

        try: results["correlation_matrix"] = future_correlation.result()
        except Exception as e: results["correlation_matrix"] = f"Error: {e}"

        try: results["currency_exposure"] = future_fx.result()
        except Exception as e: results["currency_exposure"] = f"Error: {e}"

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results
    return results
