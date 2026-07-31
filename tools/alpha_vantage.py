"""
Alpha Vantage API Integration
Provides reliable real-time market data for ALL markets including Canadian (.TO) stocks.
"""
import math
from typing import Any

import requests

from tools.cache import cached
from tools.credential_manager import get_api_key
from tools.exception_logger import log_exceptions
from tools.yf_utils import dividend_yield_fraction


@log_exceptions()
def _av_key() -> str:
    """Get the best available Alpha Vantage API key (supports secondary key rotation)."""
    return get_api_key("ALPHA_VANTAGE_API_KEY", "")

BASE_URL = "https://www.alphavantage.co/query"


def _annotate_quote_freshness(quote: dict[str, Any]) -> dict[str, Any]:
    """Label an AV GLOBAL_QUOTE with the session it actually came from.

    GLOBAL_QUOTE is end-of-day on this key tier: mid-session on 2026-07-15 it
    returned the prior session's close while the symbol traded well above it on
    news, and the agent narrated that close as "today" — the price it reported
    was a day old (see the 2026-07-15 false-SOURCE-FRAUD incident). Nothing here rejects
    the payload: on a weekend, a holiday, or pre-open, the prior session's close
    IS the right answer. It just has to be labelled so callers cannot mistake it
    for a live tick.

    is_stale means "not today's session" — not "wrong". Compared against the ET
    date (the same day boundary daily_cache keys on, so a cached annotation stays
    true for the life of its cache file).
    """
    latest_day = str(quote.get("latest_trading_day") or "").strip()
    try:
        from datetime import date, datetime
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("US/Eastern")).date()
        quote_day = date.fromisoformat(latest_day)
    except Exception:
        # Non-ISO latest_trading_day (e.g. the yfinance fallback's "Real-time
        # (yfinance)") or a clock/tz failure — no date to compare, so say nothing.
        return quote

    quote["as_of"] = latest_day
    quote["is_stale"] = quote_day < today
    if quote_day < today:
        quote["staleness_note"] = (
            f"⚠️ END-OF-DAY quote: this is the CLOSE from {latest_day}, not today's "
            f"({today.isoformat()}) session. price/change/volume describe {latest_day}. "
            "Do NOT present it as a live/current/real-time price or as today's move; "
            "if the user asked what a stock did TODAY, say this data predates today "
            "and treat any intraday move as unknown from this tool."
        )
    return quote


# Alpha Vantage sends every OVERVIEW value as a STRING — all 55 of them, with zero
# non-string values in the payload (verified 2026-07-30 against live OVERVIEW
# responses for IBM, TSLA, RIVN, LCID and PLTR). A missing number is not omitted and
# is not an empty string: the payload carries the literal "None" (RIVN PERatio and
# PEGRatio, TSLA DividendYield) or the literal "-" (RIVN and LCID ForwardPE), so a
# bare float() on this data raises rather than yielding a number.
_AV_MISSING = frozenset({"none", "-", ""})


def _av_number(value: Any) -> float | None:
    """Coerce one Alpha Vantage OVERVIEW value to a float, or None if it isn't one.

    None means "no number to report", which is the same thing the yfinance fallback
    says by letting a missing key pass through as None. It deliberately does NOT mean
    zero: 0.0 in a payload the model reads asserts a fact ("this pays no dividend")
    that a missing field never stated, and the two are not interchangeable.

    The sentinel test compares the WHOLE trimmed string and never a prefix or a
    stripped character, because one payload carries both meanings of a minus sign:
    RIVN's ForwardPE is "-" (missing) while its EPS is "-2.92" and its ProfitMargin
    "-0.636" (real, and negative for a real reason). Anything that strips or
    startswith()-tests the "-" turns a loss-making company's fundamentals into nulls.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is an int subclass; a flag is not a measurement.
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None

    text = str(value).strip()
    if text.lower() in _AV_MISSING:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    # float() happily accepts "NaN" and "Infinity" without raising, and json.dumps
    # then emits them as bare NaN/Infinity tokens that strict parsers reject — the
    # shape of the Market Pulse outage. Neither is a number worth reporting.
    return number if math.isfinite(number) else None


def _av_int(value: Any) -> int | None:
    """Same coercion, for the one OVERVIEW field the yfinance path returns as an int."""
    number = _av_number(value)
    return None if number is None else int(number)


def _fast_info_value(fast_info, key: str, default=None):
    try:
        value = getattr(fast_info, key)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        return fast_info.get(key, default)
    except Exception:
        return default


def _quote_from_yfinance(symbol: str) -> dict[str, Any] | None:
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        fast = ticker.fast_info
        last_price = _fast_info_value(fast, "last_price")
        previous_close = _fast_info_value(fast, "previous_close")
        if last_price is None:
            return None

        last_price = float(last_price)
        previous_close = float(previous_close) if previous_close else 0.0
        change = last_price - previous_close if previous_close else 0.0
        change_percent = ((last_price / previous_close) - 1) * 100 if previous_close else 0.0
        volume = _fast_info_value(fast, "last_volume", 0) or 0

        return {
            "symbol": symbol.upper(),
            "price": round(last_price, 4),
            "change": round(change, 4),
            "change_percent": f"{change_percent:.2f}%",
            "volume": int(volume),
            "latest_trading_day": "Real-time (yfinance)",
            "previous_close": round(previous_close, 4),
            "source": "yfinance",
        }
    except Exception:
        return None


def _daily_prices_from_yfinance(symbol: str, days: int) -> dict[str, Any] | None:
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d")
        if hist.empty:
            return None
        prices = []
        for date, row in hist.sort_index(ascending=False).iterrows():
            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"])
            })
        return {
            "symbol": symbol.upper(),
            "days": len(prices),
            "prices": prices,
            "latest_close": prices[0]["close"] if prices else None,
            "period_high": max(p["high"] for p in prices) if prices else None,
            "period_low": min(p["low"] for p in prices) if prices else None,
            "source": "yfinance"
        }
    except Exception:
        return None


def _company_overview_from_yfinance(symbol: str) -> dict[str, Any] | None:
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or info.get("quoteType") is None:
            return None
        desc = info.get("longBusinessSummary", "")
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "peg_ratio": info.get("pegRatio"),
            "eps": info.get("trailingEps"),
            # A FRACTION, so that this fallback and the Alpha Vantage path below
            # put the same unit under the same key. They did not: AV's
            # `DividendYield` is a fraction (verified against a live OVERVIEW
            # payload — IBM 0.0296, consistent with its $6.73 per share), while
            # yfinance's `dividendYield` is a percent. `get_company_overview`
            # therefore changed the units of its own output depending on which
            # provider answered, with `source` the only thing that said which.
            #
            # `or None` because this payload carries the raw number rather than a
            # rendered string, and the helper returns 0.0 for "no number to
            # report" — missing, unparseable, real-zero and clamped alike. Emitted
            # as 0.0 into a payload the model reads, that asserts a non-payer on
            # the strength of what may have been a unit error. None is the same
            # answer the field gave before, and the honest one.
            "dividend_yield": dividend_yield_fraction(info) or None,
            "dividend_per_share": info.get("dividendRate"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "50_day_ma": info.get("fiftyDayAverage"),
            "200_day_ma": info.get("twoHundredDayAverage"),
            "beta": info.get("beta"),
            "analyst_target": info.get("targetMeanPrice"),
            "forward_pe": info.get("forwardPE"),
            "profit_margin": info.get("profitMargins"),
            "description": (desc[:300] + "...") if len(desc) > 300 else desc,
            "source": "yfinance"
        }
    except Exception:
        return None


def _av_get(params: dict, timeout: int = 10):
    from tools.credential_manager import report_rate_limit

    key = _av_key()
    if not key:
        return None, "Missing ALPHA_VANTAGE_API_KEY"

    params["apikey"] = key

    response = requests.get(BASE_URL, params=params, timeout=timeout)
    data = response.json()

    # Check for rate limits
    if "Information" in data and "API" in str(data["Information"]):
        report_rate_limit("ALPHA_VANTAGE_API_KEY", key)
        next_key = _av_key()
        if next_key and next_key != key:
            params["apikey"] = next_key
            response = requests.get(BASE_URL, params=params, timeout=timeout)
            data = response.json()

            if "Information" in data and "API" in str(data["Information"]):
                report_rate_limit("ALPHA_VANTAGE_API_KEY", next_key)
                return None, "Rate limit on all AV keys"
        else:
            return None, "Rate limit (no secondary key available)"

    if "Note" in data and "API" in str(data["Note"]):
        report_rate_limit("ALPHA_VANTAGE_API_KEY", key)
        next_key = _av_key()
        if next_key and next_key != key:
            params["apikey"] = next_key
            response = requests.get(BASE_URL, params=params, timeout=timeout)
            data = response.json()

            if "Note" in data and "API" in str(data["Note"]):
                report_rate_limit("ALPHA_VANTAGE_API_KEY", next_key)
                return None, "Rate limit on all AV keys"
        else:
            return None, "Rate limit (no secondary key available)"

    return data, None



@cached(key_func=lambda symbol: f"av_quote:{symbol.upper()}")
@log_exceptions()
def get_quote(symbol: str) -> dict[str, Any]:
    """
    Get real-time quote for a stock/ETF.
    Works reliably for US, Canadian (.TO), and international stocks.

    Returns: price, change, change_percent, volume, latest_trading_day
    """
    try:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper(),
            "apikey": _av_key()
        }
        data, err = _av_get(params, timeout=10)
        if err:
            fallback = _quote_from_yfinance(symbol)
            if fallback:
                return fallback
            return {"error": err}

        if not isinstance(data, dict) or "Global Quote" not in data or not data["Global Quote"]:
            fallback = _quote_from_yfinance(symbol)
            if fallback:
                return fallback
            return {"error": f"No data found for {symbol}. Check if symbol is correct."}

        quote = data["Global Quote"]
        return _annotate_quote_freshness({
            "symbol": quote.get("01. symbol", symbol),
            "price": float(quote.get("05. price", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_percent": quote.get("10. change percent", "0%"),
            "volume": int(quote.get("06. volume", 0)),
            "latest_trading_day": quote.get("07. latest trading day", "N/A"),
            "previous_close": float(quote.get("08. previous close", 0)),
            "open": float(quote.get("02. open", 0)),
            "high": float(quote.get("03. high", 0)),
            "low": float(quote.get("04. low", 0))
        })
    except Exception as e:
        fallback = _quote_from_yfinance(symbol)
        if fallback:
            return fallback

        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "alpha_vantage", "Failed to fetch quote", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Failed to fetch quote for {symbol}: {str(e)}"}


@cached(key_func=lambda symbol, days=30: f"av_daily:{symbol.upper()}:{days}")
@log_exceptions()
def get_daily_prices(symbol: str, days: int = 30) -> dict[str, Any]:
    """
    Get historical daily prices (OHLCV) for analysis.

    Returns: List of {date, open, high, low, close, volume}
    """
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": "compact",  # Last 100 data points
            "apikey": _av_key()
        }
        data, err = _av_get(params, timeout=10)
        if err:
            fallback = _daily_prices_from_yfinance(symbol, days)
            if fallback:
                return fallback
            return {"error": err}

        if "Time Series (Daily)" not in data:
            fallback = _daily_prices_from_yfinance(symbol, days)
            if fallback:
                return fallback
            return {"error": f"No price history found for {symbol}"}

        time_series = data["Time Series (Daily)"]
        prices = []
        for date, values in list(time_series.items())[:days]:
            prices.append({
                "date": date,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"])
            })

        return {
            "symbol": symbol.upper(),
            "days": len(prices),
            "prices": prices,
            "latest_close": prices[0]["close"] if prices else None,
            "period_high": max(p["high"] for p in prices) if prices else None,
            "period_low": min(p["low"] for p in prices) if prices else None,
            "source": "alphavantage"
        }
    except Exception as e:
        fallback = _daily_prices_from_yfinance(symbol, days)
        if fallback:
            return fallback

        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "alpha_vantage", "Failed to fetch price history", {
            "symbol": symbol,
            "days": days,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Failed to fetch price history for {symbol}: {str(e)}"}


@cached(key_func=lambda symbol: f"av_overview:{symbol.upper()}")
@log_exceptions()
def get_company_overview(symbol: str) -> dict[str, Any]:
    """
    Get fundamental data: PE ratio, EPS, dividend yield, market cap, etc.
    Note: Only works for US stocks (not ETFs or Canadian stocks).
    """
    try:
        params = {
            "function": "OVERVIEW",
            "symbol": symbol.upper(),
            "apikey": _av_key()
        }
        data, err = _av_get(params, timeout=10)
        if err:
            fallback = _company_overview_from_yfinance(symbol)
            if fallback:
                return fallback
            return {"error": err}

        if not data or "Symbol" not in data:
            fallback = _company_overview_from_yfinance(symbol)
            if fallback:
                return fallback
            return {"error": f"No fundamental data for {symbol}. May be an ETF or non-US stock."}

        # Every numeric field goes through _av_number so that this branch and the
        # yfinance fallback above put the same TYPE under the same key. They did not:
        # Alpha Vantage sends strings, yfinance sends real ints and floats, and
        # `source` was the only field that told a caller which one it was holding.
        # Nothing downstream does arithmetic on this dict today — both callers hand it
        # straight to the model — so this closes a latent hazard rather than a live
        # defect: the first consumer to write `overview["pe_ratio"] > 15` would have
        # got a TypeError on one provider and a silently wrong answer on the other.
        return {
            "symbol": data.get("Symbol"),
            "name": data.get("Name"),
            "sector": data.get("Sector"),
            "industry": data.get("Industry"),
            # int, to match the yfinance fallback's `marketCap`.
            "market_cap": _av_int(data.get("MarketCapitalization")),
            "pe_ratio": _av_number(data.get("PERatio")),
            "peg_ratio": _av_number(data.get("PEGRatio")),
            "eps": _av_number(data.get("EPS")),
            # A FRACTION on both paths as of 9324369, which routed the yfinance
            # fallback through yf_utils.dividend_yield_fraction — yfinance's own
            # `dividendYield` is a PERCENT, and the two providers were read on the
            # same day at IBM 2.99 against Alpha Vantage's 0.0296. That fix settled
            # the UNIT; the coercion here settles the TYPE. The pair has to land in
            # that order, because matching types remove the type mismatch that would
            # otherwise be the last visible tell that two providers disagree about
            # what the number means.
            "dividend_yield": _av_number(data.get("DividendYield")),
            "dividend_per_share": _av_number(data.get("DividendPerShare")),
            "52_week_high": _av_number(data.get("52WeekHigh")),
            "52_week_low": _av_number(data.get("52WeekLow")),
            "50_day_ma": _av_number(data.get("50DayMovingAverage")),
            "200_day_ma": _av_number(data.get("200DayMovingAverage")),
            "beta": _av_number(data.get("Beta")),
            "analyst_target": _av_number(data.get("AnalystTargetPrice")),
            "forward_pe": _av_number(data.get("ForwardPE")),
            "profit_margin": _av_number(data.get("ProfitMargin")),
            "description": data.get("Description", "")[:300] + "..." if data.get("Description") else None,
            "source": "alphavantage"
        }
    except Exception as e:
        fallback = _company_overview_from_yfinance(symbol)
        if fallback:
            return fallback

        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "alpha_vantage", "Failed to fetch company overview", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Failed to fetch company overview for {symbol}: {str(e)}"}


@cached(key_func=lambda keywords: f"av_search:{keywords.lower()}")
@log_exceptions()
def search_symbol(keywords: str) -> dict[str, Any]:
    """
    Search for stock symbols by company name or keywords.
    Useful when user mentions a company but doesn't know the ticker.
    """
    try:
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
            "apikey": _av_key()
        }
        data, err = _av_get(params, timeout=10)
        if err:
            return {"error": err}

        if "bestMatches" not in data:
            return {"error": f"No symbols found for '{keywords}'"}

        matches = []
        for match in data["bestMatches"][:5]:  # Top 5 results
            matches.append({
                "symbol": match.get("1. symbol"),
                "name": match.get("2. name"),
                "type": match.get("3. type"),
                "region": match.get("4. region"),
                "currency": match.get("8. currency")
            })

        return {
            "query": keywords,
            "matches": matches
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "alpha_vantage", "Symbol search failed", {
            "keywords": keywords,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Symbol search failed: {str(e)}"}


if __name__ == "__main__":
    # Test with Canadian stock that failed before
    print("Testing ZWC.TO (Canadian ETF):")
    print(get_quote("ZWC.TO"))

    print("\nTesting AAPL:")
    print(get_quote("AAPL"))
