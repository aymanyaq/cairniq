"""
FRED API Integration (Federal Reserve Economic Data)
Provides macro-economic indicators for market context and investment decisions.
"""
from typing import Any

import requests

from agent.utils import safe_print
from tools.cache import cached
from tools.credential_manager import get_api_key
from tools.exception_logger import log_exceptions

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fred_get(params: dict, timeout: int = 10):
    from tools.credential_manager import get_api_key, report_rate_limit
    from tools.tool_errors import missing_key_reason

    key = get_api_key("FRED_API_KEY")
    if not key:
        return None, missing_key_reason("FRED_API_KEY")
    params["api_key"] = key

    response = requests.get(BASE_URL, params=params, timeout=timeout)

    if response.status_code == 429:
        report_rate_limit("FRED_API_KEY", key)
        next_key = get_api_key("FRED_API_KEY")
        if next_key and next_key != key:
            params["api_key"] = next_key
            response = requests.get(BASE_URL, params=params, timeout=timeout)
            if response.status_code == 429:
                report_rate_limit("FRED_API_KEY", next_key)
                return None, "Rate limit on all FRED keys"
        else:
            return None, "Rate limit (no secondary key available)"

    if response.status_code != 200:
        return None, f"HTTP Error {response.status_code}"

    return response.json(), None


# Fallback data when API key is missing (updated periodically)
# Last updated: July 2026 (verified against raw FRED observations)
FALLBACK_DATA = {
    "fed_funds_rate": {"value": 3.63, "date": "2026-06-01", "trend": "Easing cycle through H1 2026"},
    "inflation_cpi": {"value": 4.2, "date": "2026-05-01", "status": "Above 2% target, headline re-accelerating"},
    "core_inflation_cpi": {"value": 2.8, "date": "2026-05-01", "status": "Moderating toward target"},
    "gdp_growth": {"value": 2.1, "date": "2026-Q1", "trend": "Moderate growth"},
    "unemployment": {"value": 4.3, "date": "2026-05-01", "status": "Low but softening"},
    "treasury_10y": {"value": 4.44, "date": "2026-06-30"},
    "treasury_2y": {"value": 4.14, "date": "2026-06-30"}
}


def _find_year_ago(obs: list[dict]) -> dict | None:
    """Return the observation dated exactly 12 months before the latest one.

    FRED frequently publishes the newest month as a '.' placeholder, which
    _fetch_series filters out AFTER the API applied `limit` — so positional
    indexing (obs[12] for "a year ago") silently shifts the window and can
    shrink the list below the expected length. Matching by calendar month is
    immune to both. Callers should fetch with a few months of buffer
    (e.g. limit=15-18 for monthly series).
    """
    if not obs:
        return None
    y, m = obs[0]["date"][:7].split("-")
    target = f"{int(y) - 1}-{m}"
    for o in obs[1:]:
        if o["date"][:7] == target:
            return o
    return None


@log_exceptions()
def _fetch_series(series_id: str, limit: int = 10) -> list[dict]:
    """Helper to fetch a FRED data series."""
    try:
        params = {
            "series_id": series_id,
            "api_key": get_api_key("FRED_API_KEY"),
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit
        }
        data, err = _fred_get(params, timeout=10)
        if err or not data or "observations" not in data:
            return []

        return [
            {"date": obs["date"], "value": float(obs["value"]) if obs["value"] != "." else None}
            for obs in data["observations"]
            if obs["value"] != "."
        ]
    except Exception:
        return []


@cached(key_func=lambda: "fred_fed_rate", ttl=86400)
@log_exceptions()
def get_fed_funds_rate() -> dict[str, Any]:
    """
    Get the current Federal Funds Effective Rate.
    This is THE interest rate that affects all borrowing costs.
    """
    if not get_api_key("FRED_API_KEY"):
        fallback = FALLBACK_DATA["fed_funds_rate"]
        return {
            "indicator": "Federal Funds Rate",
            "current_rate": f"{fallback['value']:.2f}%",
            "as_of": fallback["date"],
            "year_ago": "N/A",
            "change_1y": "N/A",
            "interpretation": (
                "High rates (>5%) = Tight policy, slows economy, bad for stocks/bonds. "
                "Low rates (<2%) = Easy policy, stimulates economy, good for risk assets."
            ),
            "note": "Using cached fallback data because FRED_API_KEY is not configured."
        }


    try:
        observations = _fetch_series("FEDFUNDS", limit=15)

        if not observations:
            return {"error": "Could not fetch Fed Funds Rate. Check FRED_API_KEY."}

        current = observations[0]
        year_ago = _find_year_ago(observations)

        return {
            "indicator": "Federal Funds Rate",
            "current_rate": f"{current['value']:.2f}%",
            "as_of": current["date"],
            "year_ago": f"{year_ago['value']:.2f}%" if year_ago else "N/A",
            "change_1y": f"{current['value'] - year_ago['value']:+.2f}%" if year_ago else "N/A",
            "interpretation": (
                "High rates (>5%) = Tight policy, slows economy, bad for stocks/bonds. "
                "Low rates (<2%) = Easy policy, stimulates economy, good for risk assets."
            )
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fred_api", "Fed Funds Rate fetch failed", {
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Failed to fetch Fed Funds Rate: {str(e)}"}


@cached(key_func=lambda: "fred_inflation", ttl=86400)
@log_exceptions()
def get_inflation_data() -> dict[str, Any]:
    """
    Get Consumer Price Index (CPI) inflation data.
    Shows how fast prices are rising (eroding purchasing power).
    """
    if not get_api_key("FRED_API_KEY"):
        fb_h = FALLBACK_DATA["inflation_cpi"]
        fb_c = FALLBACK_DATA["core_inflation_cpi"]
        return {
            "indicator": "Inflation (CPI)",
            "headline_inflation": f"{fb_h['value']:.1f}%",
            "core_inflation": f"{fb_c['value']:.1f}%",
            "as_of": fb_h["date"],
            "fed_target": "2.0%",
            "status": fb_h["status"],
            "interpretation": (
                "High inflation (>3%) = Fed likely to raise rates, bad for bonds. "
                "Low inflation (<1.5%) = Fed may cut rates, good for bonds. "
                "Core CPI (ex-food & energy) is the Fed's preferred gauge."
            ),
            "note": "Using cached fallback data because FRED_API_KEY is not configured."
        }

    def _calc_yoy(obs: list[dict]) -> float | None:
        """YoY %: compare the latest observation to the one 12 calendar months
        earlier (date-matched — see _find_year_ago for why not obs[12])."""
        prev = _find_year_ago(obs)
        if obs and prev and prev["value"]:
            return ((obs[0]["value"] - prev["value"]) / prev["value"]) * 100
        return None

    try:
        # Headline CPI (CPIAUCSL). limit=18 leaves buffer for '.' placeholders.
        headline_obs = _fetch_series("CPIAUCSL", limit=18)
        headline_yoy = _calc_yoy(headline_obs)

        # Core CPI (CPILFESL — excludes food & energy, more important for Fed)
        core_obs = _fetch_series("CPILFESL", limit=18)
        core_yoy = _calc_yoy(core_obs)

        # Use fallback values only for what's missing
        fb_headline = FALLBACK_DATA["inflation_cpi"]
        fb_core = FALLBACK_DATA["core_inflation_cpi"]

        headline_val = headline_yoy if headline_yoy is not None else fb_headline["value"]
        core_val = core_yoy if core_yoy is not None else fb_core["value"]
        # as_of must describe the value actually reported: when the headline
        # falls back to the cached estimate, don't stamp it with the live date.
        as_of = headline_obs[0]["date"] if (headline_obs and headline_yoy is not None) else fb_headline["date"]

        # Determine data source transparency
        source_note = []
        if headline_yoy is None:
            source_note.append("headline CPI using cached estimate")
        if core_yoy is None:
            source_note.append("core CPI using cached estimate")

        status = "Above Target" if headline_val > 2.5 else "Near Target" if headline_val > 1.5 else "Below Target"

        result = {
            "indicator": "Inflation (CPI)",
            "headline_inflation": f"{headline_val:.1f}%",
            "core_inflation": f"{core_val:.1f}%",
            "as_of": as_of,
            "fed_target": "2.0%",
            "status": status,
            "interpretation": (
                "High inflation (>3%) = Fed likely to raise rates, bad for bonds. "
                "Low inflation (<1.5%) = Fed may cut rates, good for bonds. "
                "Core CPI (ex-food & energy) is the Fed's preferred gauge."
            )
        }
        if source_note:
            result["note"] = f"Partial fallback: {', '.join(source_note)}."
        return result

    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fred_api", "Inflation data fetch failed", {
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        # Last-resort fallback to hardcoded data
        fb_h = FALLBACK_DATA["inflation_cpi"]
        fb_c = FALLBACK_DATA["core_inflation_cpi"]
        return {
            "indicator": "Inflation (CPI)",
            "headline_inflation": f"{fb_h['value']:.1f}%",
            "core_inflation": f"{fb_c['value']:.1f}%",
            "as_of": fb_h["date"],
            "fed_target": "2.0%",
            "status": fb_h["status"],
            "interpretation": (
                "High inflation (>3%) = Fed likely to raise rates, bad for bonds. "
                "Low inflation (<1.5%) = Fed may cut rates, good for bonds. "
                "Core CPI (ex-food & energy) is the Fed's preferred gauge."
            ),
            "note": "Using cached fallback data (API unavailable)."
        }


@cached(key_func=lambda: "fred_gdp", ttl=86400)
@log_exceptions()
def get_gdp_growth() -> dict[str, Any]:
    """
    Get Real GDP Growth Rate (annualized quarter-over-quarter).
    Shows whether the economy is expanding or contracting.
    """
    try:
        observations = _fetch_series("A191RL1Q225SBEA", limit=8)  # Real GDP growth rate

        if not observations:
            return {"error": "Could not fetch GDP data"}

        current = observations[0]
        prev_quarter = observations[1] if len(observations) > 1 else None

        return {
            "indicator": "Real GDP Growth (Annualized)",
            "current_rate": f"{current['value']:.1f}%",
            "as_of": current["date"],
            "previous_quarter": f"{prev_quarter['value']:.1f}%" if prev_quarter else "N/A",
            "trend": "Accelerating" if prev_quarter and current["value"] > prev_quarter["value"] else "Decelerating",
            "interpretation": (
                "Positive growth (>2%) = Healthy economy, good for stocks. "
                "Negative growth = Recession warning, defensive positioning recommended."
            )
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fred_api", "GDP data fetch failed", {
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Failed to fetch GDP data: {str(e)}"}


@cached(key_func=lambda: "fred_unemployment", ttl=86400)
@log_exceptions()
def get_unemployment() -> dict[str, Any]:
    """
    Get Unemployment Rate.
    Key indicator of labor market health.
    """
    try:
        observations = _fetch_series("UNRATE", limit=15)

        if not observations:
            return {"error": "Could not fetch unemployment data"}

        current = observations[0]
        year_ago = _find_year_ago(observations)

        return {
            "indicator": "Unemployment Rate",
            "current_rate": f"{current['value']:.1f}%",
            "as_of": current["date"],
            "year_ago": f"{year_ago['value']:.1f}%" if year_ago else "N/A",
            "trend": ("Improving" if current["value"] < year_ago["value"]
                      else "Stable" if current["value"] == year_ago["value"]
                      else "Worsening") if year_ago else "N/A",
            "interpretation": (
                "Low unemployment (<4%) = Strong economy, potential wage inflation. "
                "Rising unemployment = Recession risk, Fed may cut rates."
            )
        }
    except Exception as e:
        try:
            from tools.web_search import search_news
            safe_print("⚠️ Unemployment API failed. Searching web...")
            search = search_news("Current US Unemployment Rate", max_results=1)
            return {
                "indicator": "Unemployment Rate",
                "current_rate": "See Notes",
                "status": "Check Search",
                "interpretation": f"Source: Web Search. {str(search)[:200]}..."
            }
        except Exception:
            return {"error": f"Failed to fetch unemployment data: {str(e)}"}


@cached(key_func=lambda: "fred_treasury", ttl=86400)
@log_exceptions()
def get_treasury_yields() -> dict[str, Any]:
    """
    Get Treasury Yields (10-Year and 2-Year) for yield curve analysis.
    Inverted yield curve (2Y > 10Y) historically predicts recessions.
    """
    # Use fallback if no API key
    if not get_api_key("FRED_API_KEY"):
        ten_y = FALLBACK_DATA["treasury_10y"]["value"]
        two_y = FALLBACK_DATA["treasury_2y"]["value"]
        spread = ten_y - two_y
        return {
            "indicator": "Treasury Yields",
            "10_year_yield": f"{ten_y:.2f}%",
            "2_year_yield": f"{two_y:.2f}%",
            "yield_spread": f"{spread:.2f}%",
            "curve_status": "Normal" if spread > 0 else "INVERTED (Recession Warning)",
            "as_of": FALLBACK_DATA["treasury_10y"]["date"],
            "note": "Using cached data (no FRED API key).",
            "interpretation": (
                "Normal curve (10Y > 2Y) = Healthy economy. "
                "Inverted curve = Historically precedes recessions by 6-18 months."
            )
        }

    try:
        ten_year = _fetch_series("DGS10", limit=5)
        two_year = _fetch_series("DGS2", limit=5)

        if not ten_year or not two_year:
            return {"error": "Could not fetch Treasury yield data"}

        ten_y = ten_year[0]["value"]
        two_y = two_year[0]["value"]
        spread = ten_y - two_y

        return {
            "indicator": "Treasury Yields",
            "10_year_yield": f"{ten_y:.2f}%",
            "2_year_yield": f"{two_y:.2f}%",
            "yield_spread": f"{spread:.2f}%",
            "curve_status": "Normal" if spread > 0 else "INVERTED (Recession Warning)",
            "as_of": ten_year[0]["date"],
            "interpretation": (
                "Normal curve (10Y > 2Y) = Healthy economy. "
                "Inverted curve = Historically precedes recessions by 6-18 months."
            )
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fred_api", "Treasury yields fetch failed", {
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Failed to fetch Treasury yields: {str(e)}"}


@cached(key_func=lambda: "fred_treasury_curve", ttl=86400)
@log_exceptions()
def get_treasury_curve() -> dict[str, Any]:
    """
    Get live 1/2/3/5-year Treasury yields for bond/GIC ladder construction.
    The 4-year point is linearly interpolated between 3-year and 5-year since
    FRED has no standalone DGS4 series.
    """
    fallback_curve = {1: 4.4, 2: 4.2, 3: 4.1, 4: 4.05, 5: 4.0}

    if not get_api_key("FRED_API_KEY"):
        return {
            "curve": fallback_curve,
            "as_of": None,
            "source": "Fallback estimate (no FRED_API_KEY configured)",
        }

    try:
        tenor_series = {1: "DGS1", 2: "DGS2", 3: "DGS3", 5: "DGS5"}
        curve: dict[int, float] = {}
        as_of = None

        for tenor, series_id in tenor_series.items():
            observations = _fetch_series(series_id, limit=5)
            if observations:
                curve[tenor] = observations[0]["value"]
                as_of = as_of or observations[0]["date"]

        if 3 in curve and 5 in curve:
            curve[4] = round((curve[3] + curve[5]) / 2, 3)

        if not curve:
            return {"error": "Could not fetch Treasury curve. Check FRED_API_KEY."}

        return {"curve": curve, "as_of": as_of, "source": "FRED"}
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fred_api", "Treasury curve fetch failed", {
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Failed to fetch Treasury curve: {str(e)}"}


@cached(key_func=lambda: "fred_all_macro", ttl=86400)
@log_exceptions()
def get_all_macro_indicators() -> dict[str, Any]:
    """
    Get a comprehensive summary of all major macro indicators.
    Perfect for getting quick market context.
    """
    return {
        "fed_funds": get_fed_funds_rate(),
        "inflation": get_inflation_data(),
        "gdp": get_gdp_growth(),
        "unemployment": get_unemployment(),
        "treasury_yields": get_treasury_yields(),
        "summary": (
            "Use these indicators together: "
            "High inflation + Low unemployment = Fed raises rates (bearish bonds). "
            "Inverted yield curve + Rising unemployment = Recession risk (defensive stocks). "
            "Low inflation + Falling rates = Risk-on environment (bullish stocks)."
        )
    }

@cached(key_func=lambda: "fred_canada", ttl=86400)
@log_exceptions()
def get_canada_metrics() -> dict[str, Any]:
    """
    Get Key Canadian Macro Metrics (BoC Rate, Inflation).
    Critical for CAD-based investors (TFSA/RRSP).
    """
    # Fallback (Manual Update Jan 2026)
    # BoC Rate approx 3.75% (lagging Fed slightly)
    # CPI approx 2.1%
    if not get_api_key("FRED_API_KEY"):
         return {"error": "Missing API Key", "details": "Please set FRED_API_KEY in .env"}


    try:
        # Attempt to fetch real data
        # IRSTCI01CAM156N = Can Immediate Rates
        # CPALTT01CAM661S = Can CPI
        rate_data = _fetch_series("IRSTCI01CAM156N", limit=1)
        cpi_data = _fetch_series("CPALTT01CAM661S", limit=13)

        rate_val = rate_data[0]['value'] if rate_data else 3.75

        cpi_val = 2.1
        if cpi_data and len(cpi_data) >= 13:
            curr = cpi_data[0]['value']
            prev = cpi_data[12]['value']
            cpi_val = ((curr - prev) / prev) * 100

        return {
            "interest_rate": f"{rate_val:.2f}%",
            "inflation": f"{cpi_val:.1f}%",
            "gdp_trend": "Data Unavailable",
            "source": "FRED API"
        }

    except Exception:
        return {"interest_rate": "3.75%", "inflation": "2.1%", "note": "Fallback"}

@cached(key_func=lambda: "fred_systemic_risk", ttl=86400)
@log_exceptions()
def get_systemic_risk_indicators() -> dict[str, Any]:
    """
    Get 'Hidden' Systemic Risk Indicators.
    1. High Yield Option-Adjusted Spread (BAMLH0A0HYM2): The best predictor of recessions/crashes.
       - Rising > 5% = Danger.
       - Low < 3% = Safe.
    2. M2 Money Supply (M2SL): Liquidity gauge.
    """
    try:
        # 1. Credit Spreads (BAMLH0A0HYM2)
        spreads = _fetch_series("BAMLH0A0HYM2", limit=1)
        # 2. Money Supply (M2SL)
        m2 = _fetch_series("M2SL", limit=13)

        spread_val = spreads[0]['value'] if spreads else None

        m2_curr = m2[0]['value'] if m2 else 0
        m2_prev = m2[12]['value'] if len(m2) >= 13 else 0
        m2_growth = ((m2_curr - m2_prev) / m2_prev) * 100 if m2_prev else 0

        return {
            "credit_spread": f"{spread_val:.2f}%" if spread_val else "N/A",
            "m2_growth_yoy": f"{m2_growth:.1f}%",
            "liquidity_status": "Expanding" if m2_growth > 0 else "Contracting",
            "crash_risk": "High" if spread_val and spread_val > 5.0 else "Elevated" if spread_val and spread_val > 4.0 else "Low"
        }
    except Exception as e:
        import logging

        from agent.logger import log_to_component
        log_to_component("tools", "fred_api", "Systemic risk indicators fetch failed", {
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        return {"error": f"Systemic risk data failed: {str(e)}"}



if __name__ == "__main__":
    print("Testing FRED API...")
    print(get_all_macro_indicators())
    print("\n=== Canada ===")
    print(get_canada_metrics())
    print("\n=== Systemic Risk ===")
    print(get_systemic_risk_indicators())
