
import re
from typing import Any

import requests

from agent.utils import retry_with_backoff, safe_print
from tools.cache import cached
from tools.credential_manager import get_api_key, report_rate_limit
from tools.exception_logger import log_exceptions
from tools.tool_errors import missing_key_reason, unavailable

# Verified Base URL correctly (no /api/ prefix for /stable/)
BASE_URL = "https://financialmodelingprep.com/stable"

@log_exceptions()
def _fmp_key() -> str:
    """Get the best available FMP API key (supports secondary key rotation)."""
    return get_api_key("FMP_API_KEY")

@log_exceptions()
def _fmp_get(url_path: str, params: dict = None, timeout: int = 5):
    """
    Shared FMP request helper with automatic key rotation on 429.
    Returns (response_json, error_string_or_None).
    """
    if params is None:
        params = {}
    key = _fmp_key()
    if not key:
        return None, missing_key_reason("FMP_API_KEY")
    params["apikey"] = key
    url = f"{BASE_URL}/{url_path}" if not url_path.startswith("http") else url_path

    resp = requests.get(url, params=params, timeout=timeout)

    if resp.status_code == 429:
        # Rotate: mark this key as limited and retry with next
        report_rate_limit("FMP_API_KEY", key)
        next_key = _fmp_key()
        if next_key and next_key != key:
            params["apikey"] = next_key
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                report_rate_limit("FMP_API_KEY", next_key)
                return None, "Rate limit on all FMP keys"
        else:
            return None, "Rate limit (no secondary key available)"

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"

    try:
        return resp.json(), None
    except Exception:
        return None, "Invalid JSON response"

@cached(key_func=lambda symbol: f"fmp_quote:{symbol.upper()}")
@log_exceptions()
def get_fmp_quote(symbol: str) -> dict[str, Any]:
    """Get real-time quote from FMP."""
    try:
        data, err = _fmp_get("quote", {"symbol": symbol})
        if data and isinstance(data, list):
            item = data[0]
            return {
                "symbol": item.get("symbol"),
                "price": item.get("price"),
                "change": item.get("change"),
                "change_pct": item.get("changePercentage"),
                "day_low": item.get("dayLow"),
                "day_high": item.get("dayHigh"),
                "year_high": item.get("yearHigh"),
                "year_low": item.get("yearLow"),
                "market_cap": item.get("marketCap"),
                "pe": item.get("pe"),
                "eps": item.get("eps"),
                "volume": item.get("volume"),
                "exchange": item.get("exchange")
            }
        return {"error": "No data found"}
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fmp_api", "FMP quote fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": str(e)}

@cached(key_func=lambda symbol: f"fmp_profile:{symbol.upper()}")
@log_exceptions()
def get_fmp_profile(symbol: str) -> dict[str, Any]:
    """Get company profile (Sector, Description, Beta)."""
    try:
        data, err = _fmp_get("profile", {"symbol": symbol})
        if data and isinstance(data, list):
            item = data[0]
            return {
                "symbol": item.get("symbol"),
                "company_name": item.get("companyName"),
                "industry": item.get("industry"),
                "sector": item.get("sector"),
                "description": item.get("description"),
                "ceo": item.get("ceo"),
                "beta": item.get("beta"),
                "currency": item.get("currency"),
                "country": item.get("country"),
                "exchange": item.get("exchange"),
                "is_etf": item.get("isEtf"),
                "is_fund": item.get("isFund")
            }
        return {"error": "Profile not found"}
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fmp_api", "FMP profile fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": str(e)}

@cached(key_func=lambda symbol: f"fmp_financials:{symbol.upper()}")
@log_exceptions()
def get_fmp_financials(symbol: str) -> dict[str, Any]:
    """Get Income Statement (Revenue, Net Income)."""
    try:
        data, err = _fmp_get("income-statement", {"symbol": symbol, "limit": 1})
        if data and isinstance(data, list):
            item = data[0]
            return {
                "revenue": item.get("revenue"),
                "net_income": item.get("netIncome"),
                "gross_profit": item.get("grossProfit"),
                "eps": item.get("eps"),
                "date": item.get("date")
            }
        return unavailable("FMP", err or f"No income statement data for {symbol}")
    except Exception as e:
        return unavailable("FMP", str(e))

@cached(key_func=lambda symbol: f"fmp_etf_holdings:{symbol.upper()}")
@log_exceptions()
def get_fmp_etf_holdings(symbol: str) -> list:
    """Get top holdings for an ETF."""
    try:
        data, err = _fmp_get("etf/holdings", {"symbol": symbol})
        if err or not data:
            return []
        if isinstance(data, list):
            return data[:15]
        return []
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fmp_api", "FMP ETF holdings fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return []

# stamp=False: map-shaped ({sector: fraction}) — an in-band `_as_of` reads as a
# sector whose weight is a date string, and the consumer sums it. See cached().
@cached(key_func=lambda symbol: f"fmp_etf_sectors:{symbol.upper()}", stamp=False)
@log_exceptions()
def get_fmp_etf_sector_weightings(symbol: str) -> dict[str, float]:
    """
    Get an ETF/fund's sector weightings as {sector_name: fraction}.

    Used to decompose broad/thematic funds (SPYX, SCHD, VNQ, sector SPDRs, …)
    that yfinance returns no `sector` for and would otherwise land in the
    "Unclassified Fund" bucket. Returns {} when FMP has no coverage (e.g. most
    Canadian .TO listings, bond funds) so the caller can fall back cleanly.
    """
    try:
        data, err = _fmp_get("etf/sector-weightings", {"symbol": symbol})
        if err or not data or not isinstance(data, list):
            return {}
        out: dict[str, float] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            sec = (row.get("sector") or "").strip()
            w = row.get("weightPercentage")
            if not sec or w is None:
                continue
            if isinstance(w, str):
                w = w.replace("%", "").strip()
            try:
                frac = float(w)
            except (TypeError, ValueError):
                continue
            # FMP reports percentages (e.g. 30.5); some feeds give fractions.
            if frac > 1.5:
                frac = frac / 100.0
            if frac > 0:
                out[sec] = out.get(sec, 0.0) + frac
        return out
    except Exception:
        return {}

@cached(key_func=lambda symbol: f"fmp_dcf:{symbol.upper()}")
@log_exceptions()
def get_fmp_dcf(symbol: str) -> float:
    """Get Discounted Cash Flow (Intrinsic Value)."""
    try:
        data, err = _fmp_get("discounted-cash-flow", {"symbol": symbol})
        if err or not data:
            return None
        if isinstance(data, list) and data:
            return data[0].get("dcf")
        return None
    except Exception:
        return None

@cached(key_func=lambda symbol: f"fmp_analyst:{symbol.upper()}")
@log_exceptions()
def get_fmp_analyst_estimates(symbol: str) -> dict[str, Any]:
    """Get Analyst Price Targets and Consensus."""
    try:
        target, err = _fmp_get("price-target-consensus", {"symbol": symbol})
        if err:
            return unavailable("FMP", err)
        if not target:
            return {}
        target_data = {}
        if isinstance(target, list) and target:
            target_data = target[0]

        # Fix: Keys are camelCase in API response
        # Note: FMP returns 'targetConsensus' and 'targetMedian', no 'targetMean'.
        return {
            "target_mean": target_data.get("targetConsensus"), # Use consensus as mean proxy
            "target_high": target_data.get("targetHigh"),
            "target_low": target_data.get("targetLow"),
            "consensus": target_data.get("targetConsensus")
        }
    except Exception:
        return {}

@cached(key_func=lambda symbol: f"fmp_insider:{symbol.upper()}")
@log_exceptions()
def get_fmp_insider_trades(symbol: str) -> list | dict:
    """Get recent insider trading activity. Returns an 'unavailable' dict on fetch failure."""
    try:
        data, err = _fmp_get("insider-trading", {"symbol": symbol, "limit": 10})
        if err:
            return unavailable("FMP", err)
        # Empty list is a real answer: no recent insider trades for this symbol.
        return data or []
    except Exception as e:
        return unavailable("FMP", str(e))

@cached(key_func=lambda symbol: f"fmp_senate:{symbol.upper()}")
@log_exceptions()
def get_fmp_senate_disclosures(symbol: str) -> list | dict:
    """Get recent US Senate trading activity. Returns an 'unavailable' dict on fetch failure."""
    try:
        data, err = _fmp_get("senate-disclosure", {"symbol": symbol, "limit": 5})
        if err:
            return unavailable("FMP", err)
        # Empty list is a real answer: no recent Senate disclosures for this symbol.
        return data or []
    except Exception as e:
        return unavailable("FMP", str(e))

@cached(key_func=lambda: "fmp_econ_calendar")
@log_exceptions()
def get_economic_calendar() -> list:
    """Get upcoming economic events (CPI, Fed, Jobs) for the next 14 days."""
    try:
        import datetime
        today = datetime.date.today()
        future = today + datetime.timedelta(days=14)

        data, err = _fmp_get("economic_calendar", {"from": str(today), "to": str(future)})

        if data:
            # Filter for high impact only
            important_keywords = ["CPI", "GDP", "Interest Rate", "Fed", "Non Farm", "Unemployment", "FOMC", "Inflation", "PCE"]

            filtered = []
            for event in data:
                event_name = event.get("event", "")
                if any(k in event_name for k in important_keywords) and event.get("country") == "US":
                    filtered.append({
                        "date": event.get("date"),
                        "event": event_name,
                        "estimate": event.get("estimate"),
                        "previous": event.get("previous"),
                        "unit": event.get("unit")
                    })

            if filtered:
                return filtered

        # Fallback: Web Search if API empty or fails
        from tools.web_search import search_news
        safe_print("⚠️ FMP Calendar empty/failed. Switching to Web Search...")
        search_query = f"Economic calendar key events for week of {today}"
        search_results = search_news(search_query, max_results=3)

        return [{
            "date": "This Week (Live Search)",
            "event": "Web Search Results",
            "estimate": str(search_results)[:500] + "...", # Truncate to avoid huge context
            "previous": "N/A",
            "unit": "News"
        }]

    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fmp_api", "Economic calendar fetch failed", {
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        safe_print(f"Error fetching calendar: {e}")
        return [{
            "date": "Error",
            "event": f"Failed to fetch data: {str(e)}",
            "estimate": "N/A",
            "previous": "N/A",
            "unit": "Error"
        }]

# The header every REAL transcript carries. Detection is positive on purpose:
# this function's failure paths return prose that reads like content ("API Limit
# Reached … here is a web summary instead"), and a caller testing for failure
# words has to enumerate them all correctly forever. Testing for the header
# instead fails safe — an unrecognised payload is "not a transcript", never
# "a transcript with nothing in it". That distinction is not academic: the tone
# analyser used to word-count the rate-limit fallback and report the result as
# NEUTRAL MANAGEMENT TONE.
TRANSCRIPT_HEADER = "### 📞 Earnings Call Transcript:"


def is_real_transcript(payload) -> bool:
    """True only for an actual transcript payload from :func:`get_earnings_transcript`."""
    return isinstance(payload, str) and payload.lstrip().startswith(TRANSCRIPT_HEADER)


def transcript_body(payload) -> str:
    """The spoken transcript, with the header this module prepends removed.

    The header is OUR metadata — a title, the symbol, the period, a date line —
    not management's words. Leaving it in adds a fixed number of tokens to every
    transcript, which dilutes per-1,000-word rates more in a short call than a
    long one and puts a length dependency into a comparison specifically designed
    not to have one.
    """
    if not is_real_transcript(payload):
        return ""
    parts = payload.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else ""


def parse_transcript_period(payload) -> tuple[int, int] | None:
    """(year, quarter) from a real transcript's header, or None.

    Needed to anchor a quarter-over-quarter comparison: the "latest" fetch does
    not say which quarter it returned, and assuming the calendar quarter would
    silently compare a company's Q2 against its own Q2 whenever reporting lags.
    """
    if not is_real_transcript(payload):
        return None
    match = re.search(r"\(Q([1-4])\s+(\d{4})\)", payload)
    if not match:
        return None
    return int(match.group(2)), int(match.group(1))


@cached(key_func=lambda symbol, year=None, quarter=None: f"fmp_transcript:{symbol.upper()}:{year}:{quarter}")
@retry_with_backoff(max_retries=3, initial_delay=2)
@log_exceptions()
def get_earnings_transcript(symbol: str, year: int = None, quarter: int = None) -> str:
    """
    Get the earnings call transcript for a specific quarter.
    If year/quarter not provided, fetches the most recent one.
    Returns the transcript text (truncated if too long).
    """
    try:
        # If no specific date, just get the latest list and pick the first
        params = {}
        if year:
            params["quarter"] = quarter
            params["year"] = year

        data, err = _fmp_get(f"earning_call_transcript/{symbol}", params, timeout=10)

        # A rate limit (or any other fetch failure) falls back to a web-search
        # summary below — never to a bare error string. Raising here used to bypass
        # the fallback: the raise was caught by this function's own `except`, so
        # retry_with_backoff never fired and the caller got a dead error string.
        if err or not isinstance(data, list):
            # Fallback to search
            from tools.web_search import search_news
            safe_print("⚠️ FMP Transcript API failed/limited. Falling back to search...")
            search_query = f"{symbol} earnings call transcript summary {year or ''} Q{quarter or ''}"
            search_results = search_news(search_query, max_results=3)
            return f"⚠️ API Limit Reached for full transcript. Here is a web summary instead:\n\n{search_results}"

        if data:
            transcript = data[0].get("content", "")
            date = data[0].get("date", "Unknown Date")
            q = data[0].get("quarter", "?")
            y = data[0].get("year", "?")

            # Truncate to avoid context window explosion (e.g. first 15k chars ≈ 3-4k tokens)
            # We focus on the intro and management section usually at the start.
            # But Q&A is at the end.
            # Strategy: Take first 5000 chars (Presentation) + last 5000 chars (Q&A)
            if len(transcript) > 10000:
                truncated = transcript[:5000] + "\n\n... [MIDDLE SECTION SKIPPED] ...\n\n" + transcript[-5000:]
            else:
                truncated = transcript

            return f"### 📞 Earnings Call Transcript: {symbol} (Q{q} {y})\n**Date:** {date}\n\n{truncated}"

        # Fallback to search if empty list
        from tools.web_search import search_news
        search_query = f"{symbol} earnings call transcript summary {year or ''} Q{quarter or ''}"
        search_results = search_news(search_query, max_results=3)
        return f"⚠️ No transcript found in DB. Here is a web summary instead:\n\n{search_results}"
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fmp_api", "Earnings transcript fetch failed", {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return f"Error fetching transcript: {e}"

@cached(key_func=lambda symbol: f"fmp_short_interest:{symbol.upper()}")
@log_exceptions()
def get_short_interest(symbol: str) -> dict[str, Any]:
    """
    Get Short Interest data (Short % of Float, Days to Cover).
    Tries FMP first, falls back to yfinance if FMP returns no data.
    """
    try:
        # FMP 'quote-short' endpoint
        data, err = _fmp_get(f"quote-short/{symbol}")

        if not err and data:
            if isinstance(data, list) and data and data[0].get("shortPercentFloat"):
                item = data[0]
                short_pct = item.get('shortPercentFloat', 0)
                return {
                    "symbol": item.get("symbol"),
                    "short_float_pct": f"{short_pct * 100:.2f}%",
                    "short_interest": item.get("shortInterest"),
                    "days_to_cover": "N/A",
                    "mechanics_note": "Short float alone is not a bullish signal; borrow cost, utilization, days-to-cover, and catalyst are needed for squeeze analysis.",
                    "source": "FMP"
                }

        # ── yfinance fallback ─────────────────────────────────────
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            short_pct_float = info.get("shortPercentOfFloat")
            shares_short = info.get("sharesShort")
            days_to_cover = info.get("shortRatio")  # shortRatio = days to cover

            if short_pct_float is not None or shares_short is not None:
                return {
                    "symbol": symbol.upper(),
                    "short_float_pct": f"{short_pct_float * 100:.2f}%" if short_pct_float else "N/A",
                    "short_interest": shares_short,
                    "shares_short_prior": info.get("sharesShortPriorMonth"),
                    "days_to_cover": round(days_to_cover, 2) if days_to_cover else "N/A",
                    "short_ratio": days_to_cover,
                    "mechanics_note": "Short float alone is not a bullish signal; borrow cost, utilization, days-to-cover, and catalyst are needed for squeeze analysis.",
                    "source": "yfinance"
                }
            return {"error": f"Short data not found for {symbol}"}
        except Exception as yf_err:
            return {"error": f"Short data not found (FMP + yfinance both failed: {yf_err})"}
        # ─────────────────────────────────────────────────────────

    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fmp_api", "Short interest fetch failed", {
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": str(e)}

if __name__ == "__main__":
    print(get_fmp_quote("AAPL"))
    print("Insider:", len(get_fmp_insider_trades("AAPL")))
