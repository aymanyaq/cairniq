import contextlib
import os
import sys
import threading
import time
from typing import Any

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.graph_memory import graph_memory
from tools.yf_utils import dividend_yield_display

_stderr_lock = threading.Lock()
_stderr_depth = 0
_stderr_real = None
_stderr_devnull = None


@contextlib.contextmanager
def suppress_stderr():
    """Helper to silence noisy library outputs (like yfinance 404s) on stderr.

    Reference-counted and lock-guarded: sys.stderr is one process-wide object, but
    this runs inside ThreadPoolExecutor pools throughout the app (health checks,
    screeners, parallel market-data fetches), so calls overlap constantly. The old
    implementation had each call open its own devnull and independently save/restore
    sys.stderr; two overlapping calls could interleave so one thread restored a
    devnull that a concurrent thread had already closed, permanently wedging
    sys.stderr on a closed file for the rest of the process's life. Every later
    stderr write anywhere in the process — not just yfinance calls — then raises
    "ValueError: I/O operation on closed file" (this is what surfaced as the rare
    "Planner failure" crashes in agent/nodes/deep_reasoning.py). Only the outermost
    caller now swaps/restores the stream; nested/concurrent callers just share it.
    """
    global _stderr_depth, _stderr_real, _stderr_devnull
    with _stderr_lock:
        if _stderr_depth == 0:
            _stderr_real = sys.stderr
            _stderr_devnull = open(os.devnull, 'w')
            sys.stderr = _stderr_devnull
        _stderr_depth += 1
    try:
        yield
    finally:
        with _stderr_lock:
            _stderr_depth -= 1
            if _stderr_depth == 0:
                sys.stderr = _stderr_real
                _stderr_devnull.close()
                _stderr_devnull = None
                _stderr_real = None

@log_exceptions()
def _safe_yf_call(func, max_retries=3, initial_delay=0.5):
    """
    Wrapper to handle yfinance 'I/O operation on closed file' race condition.
    Also suppresses stderr to hide noisy 404/API error spam.
    """
    for attempt in range(max_retries):
        try:
            with suppress_stderr():
                return func()
        except (OSError, ValueError, Exception) as e:
            if "closed file" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(initial_delay * (attempt + 1))
                continue
            raise
    return None

from tools.cache import cached


def _normalize_yfinance_symbol(symbol: str) -> str:
    """Normalize common natural-language market suffixes for yfinance."""
    symbol_clean = str(symbol or "").upper().strip()
    if any(keyword in symbol_clean for keyword in ["CANADA", "TSX"]):
        for keyword in ["CANADA", "TSX"]:
            symbol_clean = symbol_clean.replace(keyword, "")
        symbol_clean = " ".join(symbol_clean.split()).strip()
        if symbol_clean and "." not in symbol_clean:
            symbol_clean = f"{symbol_clean}.TO"
    return symbol_clean


@cached(key_func=lambda symbol: f"stock_data:{symbol.upper()}")
@log_exceptions()
def get_stock_data(symbol: str) -> dict[str, Any]:
    """
    Fetches comprehensive stock/ETF data.
    PRIORITY: Financial Modeling Prep (API) -> yfinance (Fallback)
    """
    yf_symbol = _normalize_yfinance_symbol(symbol)
    symbol_clean = yf_symbol.upper().replace(".TO", "")
    data = {}

    # 0. Handle Custom/Manual Assets immediately to avoid API noise
    # Standard tickers do NOT contain spaces. If it has a space or matches manual keywords, skip API.
    kg_node = graph_memory.graph.nodes.get(symbol_clean)
    is_manual = (
        " " in symbol_clean or
        (kg_node and kg_node.get("asset_type", "").lower() == "private")
    )

    if is_manual:
        name = kg_node.get("name") if (kg_node and isinstance(kg_node, dict) and kg_node.get("name")) else symbol.upper()
        return {
            "symbol": symbol.upper(),
            "name": name,
            "description": "Custom/Private Asset",
            "current_price": "$0.00",
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "beta": "0.0",
            "dividend_yield": "N/A",
            "sector": "Other",
            "industry": "Private/Manual",
            "source": "Manual Entry (Skipped API)",
            "data_freshness": "Static"
        }

    # 1. Try FMP (Financial Modeling Prep)
    try:
        from tools.fmp_api import get_fmp_analyst_estimates, get_fmp_dcf, get_fmp_profile, get_fmp_quote

        # FMP uses 'TO' suffix for Toronto? Let's check docs.
        # User docs don't specify, but standard is .TO or .TRT
        # We will try exact symbol.

        quote = get_fmp_quote(symbol)

        if "error" not in quote:
            profile = get_fmp_profile(symbol) # Enrich with profile
            if "error" in profile: profile = {}

            # Enrich with Advanced Data (DCF + Analyst)
            dcf = get_fmp_dcf(symbol)
            analyst = get_fmp_analyst_estimates(symbol)

            # FMP reports the day move directly. Derive the percentage from `change`
            # rather than reading change_pct, whose upstream field name has shifted
            # between FMP API versions and can come back empty; fall back to it only
            # when the absolute change is missing.
            fmp_price = quote.get("price")
            fmp_change = quote.get("change")
            fmp_previous_close = None
            fmp_day_change_pct = None
            if isinstance(fmp_price, (int, float)) and isinstance(fmp_change, (int, float)):
                fmp_previous_close = fmp_price - fmp_change
                if fmp_previous_close > 0:
                    fmp_day_change_pct = (fmp_change / fmp_previous_close) * 100
            elif isinstance(quote.get("change_pct"), (int, float)):
                fmp_day_change_pct = quote["change_pct"]

            # Construct standard object
            name = quote.get("name") or profile.get("companyName") or symbol.upper()
            data = {
                "symbol": symbol.upper(),
                "name": name,
                "current_price": f"${quote.get('price', 0):,.2f}",
                "previous_close": fmp_previous_close,
                "day_change_pct": fmp_day_change_pct,
                "market_cap": f"${quote.get('market_cap', 0):,.0f}" if quote.get('market_cap') else "N/A",
                "pe_ratio": f"{quote.get('pe', 0):.2f}" if quote.get('pe') else "N/A",
                "beta": "N/A",  # FMP quote doesn't include beta directly
                "dcf_valuation": f"${dcf:,.2f}" if dcf else "N/A",
                "analyst_target": f"${analyst.get('target_mean', 0):,.2f}" if analyst.get('target_mean') else "N/A",
                "analyst_consensus": analyst.get("consensus", "N/A"),
                "dividend_yield": "N/A", # FMP quote doesn't have yield. Need metrics endpoint.
                "52_week_high": f"${quote.get('year_high', 0):,.2f}",
                "52_week_low": f"${quote.get('year_low', 0):,.2f}",
                "52_week_position": "N/A",
                "earnings_date": "N/A",
                "earnings_warning": None,
                "volatility_warning": None,
                "sector": profile.get("sector", "N/A"),
                "industry": profile.get("industry", "N/A"),
                "currency": profile.get("currency", "USD"),
                "description": profile.get("description"),
                "source": "FMP (Official API)",
                "data_freshness": "Real-time"
            }

            # 1.5 Try to fill Earnings Date via yfinance if possible (FMP doesn't provide it in quote)
            try:
                # Quick check - don't block main thread too long
                yf_ticker = yf.Ticker(symbol)
                cal = _safe_yf_call(lambda: yf_ticker.calendar)
                if cal is not None:
                     # yfinance calendar keys vary.
                    if isinstance(cal, dict) and 'Earnings Date' in cal:
                        dates = cal['Earnings Date']
                        if dates and len(dates) > 0:
                            data["earnings_date"] = str(dates[0])[:10]
                    # DataFrame fallback (older versions)
                    elif hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                        next_date = cal.iloc[0]['Earnings Date']
                        if next_date:
                            data["earnings_date"] = str(next_date)[:10]
            except Exception:
                pass
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "market_data", "FMP data fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        safe_print(f"⚠️ FMP Failed for {symbol}: {e}")

    # 1.5 Fallback to yfinance if FMP failed
    if not data:
        # print(f"🔄 Invoking Fallback (yfinance) for {symbol}...")
        try:
            # Heuristic: If symbol explicitly asks for Canada/TSX but lacks suffix,
            # normalize it to yfinance's .TO convention before making the API call.
            symbol_clean = yf_symbol

            ticker = yf.Ticker(symbol_clean)

            # Get info with error handling and retry logic (ETFs may have limited data)
            # Mutual funds often fail on 'info', so we try 'fast_info' as fallback
            import logging
            logging.getLogger("yfinance").setLevel(logging.CRITICAL) # Silence noisy 404s

            try:
                info = _safe_yf_call(lambda: ticker.info) or {}
            except Exception:
                # Fallback to fast_info for Mutual Funds
                try:
                    fast_info = ticker.fast_info
                    info = {
                        "symbol": symbol,
                        "quoteType": "MUTUALFUND",
                        "currentPrice": fast_info.last_price,
                        "previousClose": fast_info.previous_close,
                        "currency": fast_info.currency,
                        "exchange": fast_info.exchange,
                        "marketCap": fast_info.market_cap
                    }
                except Exception:
                    # Absolute fallback
                    info = {"symbol": symbol, "quoteType": "MUTUALFUND"}

            # Safe extraction with fallbacks
            # For Mutual Funds, use 'navPrice' or 'previousClose' if currentPrice is missing
            currency = info.get("financialCurrency") or info.get("currency")
            if not currency:
                currency = "CAD" if symbol_clean.endswith(".TO") or ".TO" in symbol.upper() else "USD"

            previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            live_price = (
                info.get("currentPrice") or
                info.get("regularMarketPrice") or
                info.get("navPrice")  # Mutual Fund NAV
            )
            current_price = live_price or previous_close

            # Only a genuinely live price yields a day move. When the quote itself fell
            # back to the previous close, the two are the same number and the move would
            # read a flat 0.00% — asserting the market hasn't moved when the truth is
            # that we have no current price at all.
            day_change_pct = None
            if live_price and previous_close and previous_close > 0:
                day_change_pct = ((live_price - previous_close) / previous_close) * 100

            pe_ratio = info.get("trailingPE")

            # Mutual funds/ETFs use 'totalAssets' instead of marketCap
            market_cap = info.get("marketCap") or info.get("totalAssets") or info.get("netAssets")

            fifty2_high = info.get("fiftyTwoWeekHigh")
            fifty2_low = info.get("fiftyTwoWeekLow")

            # Get price from history if info fails
            if not current_price:
                try:
                    hist = _safe_yf_call(lambda: ticker.history(period="5d", timeout=40))
                    if hist is not None and not hist.empty:
                        current_price = hist["Close"].iloc[-1]
                except Exception:
                    pass  # Keep going even if history fails

            # Dividend info. `dividendYield` is a PERCENT, so multiplying it by
            # 100 reported a 0.31% payer as "31.00%" on the main quote panel.
            dividend_yield_str = dividend_yield_display(info)

            # 52-week position
            position_52w = "N/A"
            if fifty2_high and fifty2_low and current_price:
                range_52w = fifty2_high - fifty2_low
                if range_52w > 0:
                    position_pct = ((current_price - fifty2_low) / range_52w) * 100
                    position_52w = f"{position_pct:.1f}%"

            # Earnings date (stocks only, not ETFs)
            earnings_date = "N/A" # Simplified for brevity
            try:
                if info.get("quoteType") != "ETF":
                    calendar = _safe_yf_call(lambda: ticker.calendar)
                    if calendar is not None:
                        # Handle dictionary format (new yfinance)
                        if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                            dates = calendar['Earnings Date']
                            if dates and len(dates) > 0:
                                earnings_date = str(dates[0])[:10]
                        # Handle DataFrame format
                        elif hasattr(calendar, 'columns'):
                            if 'Earnings Date' in calendar.columns:
                                 next_earnings = calendar.iloc[0]['Earnings Date']
                                 if next_earnings:
                                     earnings_date = str(next_earnings)[:10]
                            elif 0 in calendar.columns: # Transposed format
                                 next_earnings = calendar.iloc[0][0]
                                 if next_earnings:
                                     earnings_date = str(next_earnings)[:10]
            except Exception:
                pass

            # Beta for volatility assessment
            beta = info.get("beta")
            volatility_warning = None
            if beta and beta > 1.5:
                volatility_warning = f"⚠️ HIGH VOLATILITY (Beta: {beta:.2f}) - Position size carefully"
            elif beta and beta > 1.2:
                volatility_warning = f"🟡 Moderate Volatility (Beta: {beta:.2f})"

            # Earnings warning (if within 14 days)
            earnings_warning = None
            if earnings_date != "N/A":
                try:
                    from datetime import datetime
                    earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
                    # CALENDAR days — see tools/opportunity_scanner._headwind_check.
                    # An elapsed-timedelta subtraction reads one day low and goes
                    # negative on the earnings day, hiding the warning entirely.
                    days_to_earnings = (earnings_dt.date() - datetime.now().date()).days
                    if 0 <= days_to_earnings <= 14:
                        earnings_warning = f"📅 EARNINGS IN {days_to_earnings} DAYS - Expect volatility"
                except Exception:
                    pass

            name = info.get("longName") or info.get("shortName") or info.get("companyName") or symbol.upper()
            data = {
                "symbol": symbol.upper(),
                "name": name,
                "current_price": f"${current_price:,.2f}" if current_price else "N/A",
                "previous_close": previous_close,
                "day_change_pct": day_change_pct,
                "market_cap": f"${market_cap:,.0f}" if market_cap else "N/A",
                "pe_ratio": f"{pe_ratio:.2f}" if pe_ratio else "N/A",
                "beta": f"{beta:.2f}" if beta else "N/A",
                "dcf_valuation": "N/A", # yfinance doesn't provide DCF
                "analyst_target": str(info.get("targetMeanPrice", "N/A")),
                "analyst_consensus": info.get("recommendationKey", "N/A"),
                "dividend_yield": dividend_yield_str,
                "52_week_high": f"${fifty2_high:,.2f}" if fifty2_high else "N/A",
                "52_week_low": f"${fifty2_low:,.2f}" if fifty2_low else "N/A",
                "52_week_position": position_52w,
                "earnings_date": earnings_date,
                "earnings_warning": earnings_warning,
                "volatility_warning": volatility_warning,
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "currency": currency,
                "description": info.get("longBusinessSummary"), # yfinance has longBusinessSummary
                "source": "yfinance (fallback)",
                "data_freshness": "Delayed 15-20min"
            }

        except Exception as e:
            safe_print(f"⚠️ yfinance Failed for {symbol}: {e}")

    # 2. Fallback to Polygon.io if FMP and yfinance failed
    if not data:
        try:
            from tools.polygon_api import get_polygon_profile, get_polygon_quote
            quote = get_polygon_quote(symbol)

            if "error" not in quote:
                profile = get_polygon_profile(symbol)
                if "error" in profile: profile = {}

                name = profile.get("name") or symbol.upper()
                data = {
                    "symbol": symbol.upper(),
                    "name": name,
                    "current_price": f"${quote.get('price', 0):,.2f}" if quote.get('price') else "N/A",
                    # get_polygon_quote reads v2/aggs/.../prev, so this "price" IS the
                    # previous session's close. There is no live price to move against,
                    # and comparing it to itself would report a flat 0.00% as if the
                    # market were quiet. Report no direction instead.
                    "previous_close": None,
                    "day_change_pct": None,
                    "market_cap": f"${profile.get('market_cap', 0):,.0f}" if profile.get('market_cap') else "N/A",
                    "pe_ratio": "N/A",
                    "beta": "N/A",
                    "dcf_valuation": "N/A",
                    "analyst_target": "N/A",
                    "analyst_consensus": "N/A",
                    "dividend_yield": "N/A",
                    "52_week_high": f"${quote.get('day_high', 0):,.2f}" if quote.get('day_high') else "N/A",
                    "52_week_low": f"${quote.get('day_low', 0):,.2f}" if quote.get('day_low') else "N/A",
                    "52_week_position": "N/A",
                    "earnings_date": "N/A",
                    "earnings_warning": None,
                    "volatility_warning": None,
                    "sector": profile.get("industry", "N/A"),
                    "industry": profile.get("industry", "N/A"),
                    "currency": profile.get("currency", "usd").upper(),
                    "description": profile.get("description"),
                    "source": "Polygon.io (Fallback)",
                    "data_freshness": "Previous Close / Snapshot"
                }
        except Exception as e:
            import logging

            from agent.logger import log_to_component
            log_to_component("tools", "market_data", "Polygon data fetch failed", {
                "symbol": symbol,
                "error": str(e),
                "error_type": type(e).__name__
            }, level=logging.WARNING)
            safe_print(f"⚠️ Polygon Failed for {symbol}: {e}")

    # 2.5 Final fallback: If everything failed, raise error
    if not data:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "market_data", "All data sources failed", {
            "symbol": symbol,
            "error": "FMP, yfinance, and Polygon all failed",
        }, level=logging.ERROR)
        return {"error": "All data sources failed"}

    # 3. Enrich with Hybrid Data (RSI, News, Trend, Beta) - ALWAYS RUNS if data was successfully fetched by FMP or yfinance
    # We rely on yfinance for this free data

    # Initialize Enrichment Defaults
    rsi_val = "N/A"
    recent_trend = "N/A"
    news_items = []

    if data: # Only attempt enrichment if we have some base data
        # Retry wrapper for yfinance I/O issues in concurrent environments
        def safe_yf_call(func, max_retries=3):
            """Wrapper to handle yfinance 'I/O operation on closed file' race condition."""
            for attempt in range(max_retries):
                try:
                    return func()
                except ValueError as e:
                    if "closed file" in str(e).lower() and attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                    raise
                except Exception as e:
                    if "closed file" in str(e).lower() and attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise
            return None

        try:
            ticker = yf.Ticker(yf_symbol)  # Use normalized yfinance symbol for news/history

            # A. Beta & Volatility (if not already set)
            if data.get("beta") == "N/A" or data.get("beta") is None:
                try:
                    info = safe_yf_call(lambda: ticker.info)
                    if info:
                        beta = info.get("beta")
                        if beta:
                            data["beta"] = f"{beta:.2f}"
                            if beta > 1.5:
                                data["volatility_warning"] = f"⚠️ HIGH VOLATILITY (Beta: {beta:.2f}) - Position size carefully"
                            elif beta > 1.2:
                                data["volatility_warning"] = f"🟡 Moderate Volatility (Beta: {beta:.2f})"
                except Exception:
                    pass

            # B. News (Top 3 headlines)
            try:
                raw_news = safe_yf_call(lambda: ticker.news)
                if raw_news:
                    for n in raw_news[:3]:
                        # yfinance news structure: {'content': {'title': ...}}
                        content = n.get('content', n)
                        title = content.get('title') or content.get('headline') or "No Title"
                        pub = content.get('provider', {}).get('displayName') or "Unknown"
                        news_items.append(f"- {title} ({pub})")
            except: pass

            # C. Technicals (RSI + Trend)
            hist = safe_yf_call(lambda: ticker.history(period="2mo", timeout=40)) # Need enough data for 14-day RSI

            if hist is not None and not hist.empty:
                # Trend
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                pct_change = ((end_price - start_price) / start_price) * 100
                recent_trend = f"{pct_change:.2f}% (2mo)"

                # RSI Calculation
                try:
                    delta = hist["Close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    rsi_val = f"{current_rsi:.1f}"
                except: pass

        except Exception as e:
            import logging

            from agent.logger import log_to_component
            log_to_component("tools", "market_data", "Hybrid enrichment failed", {
                "symbol": symbol,
                "error": str(e),
                "error_type": type(e).__name__
            }, level=logging.WARNING)
            safe_print(f"Hybrid Enrichment Failed: {e}")

    # Update the data object if we have one
    if isinstance(data, dict) and "error" not in data:
        data["rsi_14d"] = rsi_val
        data["recent_trend"] = recent_trend
        data["recent_news"] = "\n".join(news_items) if news_items else "N/A"
        if data.get("source") == "FMP (Official API)":
             data["source"] = "Hybrid (FMP + Free Tools)"

    # As-of stamp (Roadmap 5.8). Written HERE, inside the @cached function, so the
    # timestamp that gets stored is the moment of the actual fetch. A cache hit
    # then replays the original time rather than the read time, which is what lets
    # a caller compute the quote's true age. `data_freshness` above describes the
    # SOURCE ("Real-time", "Delayed 15-20min") and says nothing about this
    # observation — a 1-hour cache TTL means a "Real-time" label can be an hour
    # old, which is how an EOD print once got presented as the live tape.
    from tools.freshness import stamp
    stamp(data)

    return data

@log_exceptions()
def get_historical_performance(symbol: str) -> dict[str, Any]:
    """
    Get 1Y, 3Y, 5Y, and 10Y annualized returns (CAGR).
    Essential for long-term investment validation.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = _safe_yf_call(lambda: ticker.history(period="10y", timeout=20))

        if hist is None or hist.empty:
            return {"symbol": symbol, "error": "No historical data available"}

        current = hist["Close"].iloc[-1]

        def calculate_cagr(years):
            # Trading days approx 252 * years
            lookback = 252 * years
            if len(hist) < lookback:
                return "N/A (IPO too recent)"

            start_price = hist["Close"].iloc[-lookback]
            # CAGR = (End/Start)^(1/n) - 1
            cagr = ((current / start_price) ** (1/years)) - 1
            total_return = ((current - start_price) / start_price) * 100

            return {
                "cagr": f"{cagr*100:.1f}%",
                "total_return": f"{total_return:.0f}%"
            }

        return {
            "symbol": symbol,
            "current_price": f"${current:.2f}",
            "performance": {
                "1_year": calculate_cagr(1),
                "3_year": calculate_cagr(3),
                "5_year": calculate_cagr(5),
                "10_year": calculate_cagr(10)
            },
            "summary": "Compounding machine" if "N/A" not in str(calculate_cagr(5)) and float(calculate_cagr(5)["cagr"].strip("%")) > 15 else "Standard performer"
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "market_data", "Historical performance calculation failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Performance calc failed: {str(e)}"}

@log_exceptions()
def get_etf_holdings(symbol: str) -> dict[str, Any]:
    """
    Get top holdings for an ETF.
    Useful for seeing what you actually own (e.g., FTEC -> MSFT, AAPL, NVDA).
    """
    # Manual fallback for popular ETFs if API fails
    FALLBACK_HOLDINGS = {
        "FTEC": ["MSFT (16%)", "AAPL (15%)", "NVDA (14%)", "AVGO (4%)", "ADBE (2%)"],
        "SCHD": ["KO (4%)", "ABBV (4%)", "HD (4%)", "CSCO (4%)", "CVX (4%)"],
        "VTI": ["MSFT (6%)", "AAPL (6%)", "NVDA (5%)", "AMZN (3%)", "META (2%)"],
        "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO"],
        "JEPI": ["MSFT", "AMZN", "AAPL", "MA", "V"]
    }

    try:
        # 1. Try FMP First (New capability)
        try:
            from tools.fmp_api import get_fmp_etf_holdings
            holdings_fmp = get_fmp_etf_holdings(symbol)
            if holdings_fmp:
                clean_holdings = []
                for h in holdings_fmp:
                    # FMP returns {asset: 'MSFT', weightPercentage: 0.15, ...}
                    asset = h.get('asset') or h.get('symbol')
                    weight = h.get('weightPercentage', 0)
                    clean_holdings.append(f"{asset}: {weight:.2f}%")

                return {
                    "symbol": symbol,
                    "top_holdings": clean_holdings,
                    "count": len(clean_holdings),
                    "source": "FMP (Live)"
                }
        except Exception as e:
            import logging

            from agent.logger import log_to_component
            log_to_component("tools", "market_data", "FMP ETF holdings fetch failed", {
                "symbol": symbol,
                "error": str(e),
                "error_type": type(e).__name__
            }, level=logging.WARNING)
            safe_print(f"FMP ETF Error: {e}")

        # 2. Fallback to yfinance
        ticker = yf.Ticker(symbol)

        # Check if it's an ETF
        info = ticker.info
        if info.get("quoteType") != "ETF":
             return {"symbol": symbol, "message": "Not an ETF. Holdings data unavailable for single stocks."}

        # Get holdings (yfinance often returns top 10 as DataFrame)
        # Note: 'funds_data' or 'holdings' property might vary
        # Using generic 'funds_data' access if available or print top_holdings
        try:
            # Newer yfinance versions
            holdings = ticker.funds_data.top_holdings
            # It's a dataframe usually
            if isinstance(holdings, pd.DataFrame):
                # Convert to dict
                holdings.to_dict()
                # DataFrame format usually: Index=Name, Column=Percent or similar
                # We want a clean list
                clean_holdings = []
                # Iterate rows
                for idx, row in holdings.iterrows():
                    # idx is usually Ticker or Name
                    val = row.iloc[0] # Percent
                    clean_holdings.append(f"{idx}: {val*100:.2f}%")
                return {
                    "symbol": symbol,
                    "top_holdings": clean_holdings,
                    "count": len(clean_holdings)
                }
        except Exception:
             pass

        # Fallback to hardcoded list if available
        if symbol.upper() in FALLBACK_HOLDINGS:
            return {
                "symbol": symbol,
                "top_holdings": FALLBACK_HOLDINGS[symbol.upper()],
                "count": 5,
                "note": "Using cached/fallback holdings data."
            }

        # Fallback to info 'holdings' if available (rare)
        return {"symbol": symbol, "message": "Detailed ETF holdings data not exposed by current data feed."}

    except Exception as e:
         import logging

         from agent.logger import log_to_component
         log_to_component("tools", "market_data", "ETF holdings fetch failed", {
             "symbol": symbol,
             "error": str(e),
             "error_type": type(e).__name__
         }, level=logging.ERROR)
         return {"error": f"ETF fetch failed: {str(e)}"}

@log_exceptions()
def get_realtime_quote(symbol: str) -> dict[str, Any]:
    """
    Get Real-Time (or delayed) quote: Price, Change %, Volume.
    """
    try:
        data = get_stock_data(symbol)
        if "error" in data: return data

        return {
            "symbol": data.get("symbol"),
            "price": data.get("current_price"),
            # The listing currency of `price`. Every live source sets it (FMP
            # profile, yfinance financialCurrency, Polygon); it is absent only
            # for manual/private assets, which quote $0.00 anyway. Callers that
            # compare a quote against a portfolio total need it — see
            # tools.ips_precheck._to_base.
            "currency": data.get("currency"),
            "change": data.get("recent_trend"), # Using recent trend as proxy if daily change unavailable
            "market_cap": data.get("market_cap"),
            "volume_note": "Volume data accessible in full detailed view"
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "market_data", "Realtime quote fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": str(e)}

@log_exceptions()
def get_fundamentals_detailed(symbol: str) -> dict[str, Any]:
    """
    Get detailed valuation metrics: PE, PEG, Price/Sales, Price/Book, Margins.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol.upper(),
            "valuation": {
                "pe_ratio": info.get("trailingPE", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "peg_ratio": info.get("pegRatio", "N/A"),
                "price_to_sales": info.get("priceToSalesTrailing12Months", "N/A"),
                "price_to_book": info.get("priceToBook", "N/A"),
                "enterprise_value_ebitda": info.get("enterpriseToEbitda", "N/A")
            },
            "profitability": {
                "gross_margin": f"{info.get('grossMargins', 0)*100:.2f}%",
                "profit_margin": f"{info.get('profitMargins', 0)*100:.2f}%",
                "operating_margin": f"{info.get('operatingMargins', 0)*100:.2f}%",
                "return_on_equity": f"{info.get('returnOnEquity', 0)*100:.2f}%"
            },
            "growth": {
                "revenue_growth": f"{info.get('revenueGrowth', 0)*100:.2f}%",
                "earnings_growth": f"{info.get('earningsGrowth', 0)*100:.2f}%"
            },
            "financial_strength": {
                "free_cashflow": info.get("freeCashflow", "N/A"),
                "operating_cashflow": info.get("operatingCashflow", "N/A"),
                "total_cash": info.get("totalCash", "N/A"),
                "total_debt": info.get("totalDebt", "N/A"),
                "ebitda": info.get("ebitda", "N/A"),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                "beta": info.get("beta", "N/A")
            }
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "market_data", "Detailed fundamentals fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.WARNING)
        safe_print(f"⚠️ Fundamentals failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "valuation": {"pe_ratio": "N/A", "peg_ratio": "N/A"},
            "profitability": {},
            "growth": {},
            "error": "Data unavailable (likely Mutual Fund/crypto)"
        }

if __name__ == "__main__":
    # print(get_stock_data("AAPL"))
    pass


@tool
@log_exceptions()
def get_dividend_analysis(ticker: str) -> str:
    """
    Analyzes dividend safety, yield, and growth history.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        payout = info.get('payoutRatio', 0)
        rate = info.get('dividendRate', 0)
        ex_date = info.get('exDividendDate', 'N/A')

        # Safety Check
        safety = "Unknown"
        if isinstance(payout, (int, float)):
            if payout < 0.60: safety = "🟢 Safe (<60%)"
            elif payout < 0.90: safety = "🟡 At Risk (60-90%)"
            else: safety = "🔴 Dangerous (>90%)"

        # Derive the yield from rate/price where both are present: two fields in
        # units that cannot be confused beats one field whose unit has changed
        # between yfinance releases. (Checked live: AAPL 1.08/338.19 = 0.32%,
        # which agrees with `dividendYield` read as a percent.)
        #
        # The `# Fix scaling` note this replaces is the trace of somebody hitting
        # the units problem here and routing around it locally rather than at the
        # read. The fallback it left behind was `if yield_val > 1.0` — the same
        # magnitude heuristic that cannot fire for a sub-1% payer. That fallback
        # is not a rare branch either: ETFs carry neither `dividendRate` nor
        # `currentPrice` (BND and VOO return None for both), so every fund came
        # through the broken half.
        formatted_yield = "N/A"
        try:
            current_price = info.get("currentPrice") or info.get("previousClose")
            if rate and current_price:
                formatted_yield = f"{rate / current_price:.2%}"
            else:
                formatted_yield = dividend_yield_display(info)
        except Exception:
            formatted_yield = dividend_yield_display(info)

        return (
            f"### 💰 Dividend Analysis for {ticker}\n\n"
            f"**Income Metrics:**\n"
            f"- **Yield:** {formatted_yield}\n"
            f"- **Annual Payout:** ${rate} per share\n"
            f"- **Payout Ratio:** {f'{payout:.1%}' if isinstance(payout, float) else payout}\n"
            f"- **Safety Score:** {safety}\n\n"
            f"**Timing:**\n"
            f"- **Ex-Dividend Date:** {ex_date if ex_date != 'N/A' else 'Not Available'}"
        )
    except Exception as e:
        return f"Error analyzing dividends for {ticker}: {e}"
