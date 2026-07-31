"""
Fed Calendar Tool
Tracks FOMC meeting dates and historical rate decision impacts.
Critical for timing trades around Fed policy decisions.
"""

from datetime import date
from typing import Any

from tools.cache import cached
from tools.exception_logger import log_exceptions

# FOMC Meeting Schedule for 2024-2026 (publicly available from Federal Reserve)
# Format: (month, day) for each scheduled meeting
FOMC_SCHEDULE = {
    2024: [
        (1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)
    ],
    2025: [
        (1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (11, 5), (12, 17)
    ],
    2026: [
        (1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (11, 4), (12, 16)
    ],
    2027: [
        (1, 27), (3, 17), (4, 28), (6, 16), (7, 28), (9, 15), (11, 3), (12, 15)
    ]
}

# Historical rate decision impacts (average S&P 500 move)
DECISION_IMPACTS = {
    "rate_hike": {"day_of": -0.5, "week_after": -1.2, "description": "Markets typically sell off on hikes"},
    "rate_cut": {"day_of": +1.0, "week_after": +2.5, "description": "Markets typically rally on cuts"},
    "hold_hawkish": {"day_of": -0.3, "week_after": -0.8, "description": "Hawkish hold = mild negative"},
    "hold_dovish": {"day_of": +0.5, "week_after": +1.2, "description": "Dovish hold = mild positive"},
    "surprise_hike": {"day_of": -2.0, "week_after": -3.5, "description": "Unexpected hikes cause volatility"},
    "surprise_cut": {"day_of": +2.5, "week_after": +4.0, "description": "Unexpected cuts cause rallies"},
}


@cached(key_func=lambda num_meetings=6: f"fomc_calendar:{num_meetings}")
@log_exceptions()
def get_fomc_calendar(num_meetings: int = 6) -> dict[str, Any]:
    """
    Get upcoming FOMC meeting dates with countdown and market impact notes.

    Args:
        num_meetings: Number of upcoming meetings to return (default 6)

    Returns:
        Dictionary with next meeting info, full calendar, and trading guidance
    """
    try:
        # Calendar days, not instants. FOMC_SCHEDULE holds dates, so building a
        # midnight datetime and subtracting datetime.now() truncated away the rest
        # of today: every countdown came back one short, and `> today` dropped a
        # meeting the moment its own midnight passed — so on the morning of a
        # decision this tool reported the NEXT meeting instead. Both fixed by
        # comparing days to days.
        today = date.today()
        upcoming = []

        # Collect all meetings from today forward
        for year in sorted(FOMC_SCHEDULE.keys()):
            for month, day in FOMC_SCHEDULE[year]:
                meeting_date = date(year, month, day)
                if meeting_date >= today:
                    days_until = (meeting_date - today).days

                    # Determine alert level
                    if days_until <= 3:
                        alert = "🚨 CRITICAL"
                        warning = "High volatility expected. Consider reducing position sizes."
                    elif days_until <= 7:
                        alert = "⚠️ IMMINENT"
                        warning = "Fed meeting this week. Avoid major new positions."
                    elif days_until <= 14:
                        alert = "📌 UPCOMING"
                        warning = "Fed meeting in ~2 weeks. Start reviewing rate-sensitive positions."
                    else:
                        alert = "📅 SCHEDULED"
                        warning = None

                    upcoming.append({
                        "date": meeting_date.strftime("%Y-%m-%d"),
                        "day_of_week": meeting_date.strftime("%A"),
                        "days_until": days_until,
                        "alert": alert,
                        "warning": warning
                    })

                    if len(upcoming) >= num_meetings:
                        break
            if len(upcoming) >= num_meetings:
                break

        if not upcoming:
            return {"error": "No upcoming FOMC meetings found in schedule"}

        next_meeting = upcoming[0]

        # Generate trading guidance based on proximity
        if next_meeting["days_until"] <= 7:
            guidance = [
                "📉 Expect increased volatility in rate-sensitive sectors (Tech, REITs, Utilities)",
                "🛡️ Consider hedging with VIX calls or reducing leverage",
                "💵 Banks (XLF) may move sharply based on rate decision",
                "⏳ Avoid opening large positions until after the announcement"
            ]
        elif next_meeting["days_until"] <= 14:
            guidance = [
                "📊 Monitor Fed Funds Futures for rate expectations",
                "📰 Watch for Fed governor speeches that may signal policy direction",
                "🔄 Consider rebalancing rate-sensitive positions"
            ]
        else:
            guidance = [
                "✅ No imminent Fed meeting pressure",
                "📈 Focus on fundamentals and earnings"
            ]

        return {
            "next_meeting": next_meeting,
            "upcoming_meetings": upcoming,
            "historical_impacts": DECISION_IMPACTS,
            "trading_guidance": guidance,
            "data_source": "Federal Reserve Official FOMC Calendar"
        }

    except Exception as e:
        return {"error": f"Fed calendar failed: {str(e)}"}


@cached(key_func=lambda: "rate_expectations")
@log_exceptions()
def get_rate_expectations() -> dict[str, Any]:
    """
    Get current market expectations for Fed rate decisions.
    Uses CME FedWatch-style probability estimates.

    Note: This provides a simplified estimate. For precise probabilities,
    integrate with CME FedWatch Tool API.
    """
    try:
        import yfinance as yf

        from tools.fred_api import FALLBACK_DATA, get_fed_funds_rate

        # Get Fed Funds Futures proxy indicators
        # The 2-year Treasury yield is a good proxy for rate expectations
        tnx = yf.Ticker("^TNX")  # 10-year
        irx = yf.Ticker("^IRX")  # 3-month T-bill

        tnx_price = tnx.fast_info.get("lastPrice", 0)
        irx.fast_info.get("lastPrice", 0) if hasattr(irx, 'fast_info') else 0

        # Pull the live Fed Funds target rate from FRED instead of a hardcoded
        # constant that silently goes stale as policy changes.
        current_fed_rate = None
        fed_data = get_fed_funds_rate()
        raw_rate = fed_data.get("current_rate") if isinstance(fed_data, dict) else None
        if isinstance(raw_rate, str) and raw_rate.endswith("%"):
            try:
                current_fed_rate = float(raw_rate[:-1])
            except ValueError:
                current_fed_rate = None
        if current_fed_rate is None:
            # FRED unreachable and nothing parsed - fall back to fred_api's
            # own FALLBACK_DATA constant so this stays in sync with the one
            # source of truth instead of a second, independently-stale hardcode.
            current_fed_rate = FALLBACK_DATA["fed_funds_rate"]["value"]

        # Rough estimate: if 2Y yield < Fed rate, markets expect cuts
        spread = tnx_price - current_fed_rate if tnx_price else 0

        if spread < -0.5:
            expectation = "CUTS EXPECTED"
            probability = "High probability of rate cuts in next 6 months"
        elif spread < 0:
            expectation = "MILD EASING EXPECTED"
            probability = "Market pricing in 1-2 cuts"
        elif spread < 0.25:
            expectation = "HOLD EXPECTED"
            probability = "No change expected in near term"
        else:
            expectation = "HIKES POSSIBLE"
            probability = "Market pricing in potential tightening"

        return {
            "market_expectation": expectation,
            "interpretation": probability,
            "indicators": {
                "10y_yield": f"{tnx_price:.2f}%" if tnx_price else "N/A",
                "current_fed_target": f"{current_fed_rate:.2f}%",
                "spread": f"{spread:+.2f}%"
            },
            "note": "For precise probabilities, check CME FedWatch Tool"
        }

    except Exception as e:
        return {"error": f"Rate expectations failed: {str(e)}"}


@log_exceptions()
def get_meeting_countdown() -> str:
    """Quick helper to get a single-line countdown to next FOMC meeting."""
    cal = get_fomc_calendar(num_meetings=1)
    if "error" in cal:
        return cal["error"]

    next_mtg = cal["next_meeting"]
    return f"Next FOMC: {next_mtg['date']} ({next_mtg['days_until']} days) - {next_mtg['alert']}"


if __name__ == "__main__":
    import json

    print("=== FOMC Calendar ===")
    print(json.dumps(get_fomc_calendar(), indent=2))

    print("\n=== Rate Expectations ===")
    print(json.dumps(get_rate_expectations(), indent=2))

    print("\n=== Quick Countdown ===")
    print(get_meeting_countdown())
