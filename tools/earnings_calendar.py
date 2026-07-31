"""
Earnings Calendar Tool
Fetches upcoming earnings dates and historical earnings surprise data.
"""
from datetime import date, datetime

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol: f"earnings_info:{symbol.upper()}")
@log_exceptions()
def get_earnings_info(symbol: str) -> dict:
    """
    Get upcoming earnings date and historical earnings performance.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get next earnings date
        earnings_dates = ticker.earnings_dates

        next_earnings = None
        days_until_earnings = None
        earnings_warning = None

        if earnings_dates is not None and len(earnings_dates) > 0:
            # Find the next upcoming earnings date.
            #
            # Both the comparison and the countdown work in CALENDAR DAYS, which is
            # the unit an earnings date is quoted in. They used to work in instants:
            # `(event - datetime.now()).days` truncates a timedelta, so a print dated
            # tomorrow at 06:00 reported 0 days from any moment after 06:00 today —
            # the date said tomorrow and the countdown said today (observed on AAPL
            # and AMZN, 2026-07-29). The ex-dividend block below already reduced both
            # sides to .date() before subtracting; this is the same reduction.
            #
            # `>= today` rather than `> now` also keeps a print scheduled for TODAY
            # visible for the whole day. Comparing instants dropped it the moment it
            # happened and silently advanced to next quarter's date.
            today = date.today()
            for stamp in earnings_dates.index:
                # The index is tz-aware in exchange time. Drop the zone rather than
                # converting it — the earnings day is the exchange's day, and this is
                # the reduction the reported date string has always used.
                naive = stamp.tz_localize(None) if stamp.tzinfo is not None else stamp
                event_day = naive.date()
                if event_day >= today:
                    next_earnings = event_day.isoformat()
                    days_until = (event_day - today).days
                    days_until_earnings = days_until

                    if days_until <= 7:
                        earnings_warning = "⚠️ CAUTION: Earnings within 7 days! High volatility expected."
                    elif days_until <= 14:
                        earnings_warning = "⚡ NOTE: Earnings within 2 weeks."
                    else:
                        earnings_warning = None
                    break

        # Get historical earnings data
        earnings_history = ticker.earnings_history

        avg_surprise = None
        beat_rate = None
        if earnings_history is not None and len(earnings_history) > 0:
            surprises = []
            beats = 0
            for _, row in earnings_history.iterrows():
                if 'surprisePercent' in row and row['surprisePercent'] is not None:
                    surprises.append(row['surprisePercent'])
                    if row['surprisePercent'] > 0:
                        beats += 1

            if surprises:
                avg_surprise = sum(surprises) / len(surprises)
                beat_rate = (beats / len(surprises)) * 100

        # Get EPS estimates if available
        eps_estimate = info.get('forwardEps', 'N/A')
        eps_ttm = info.get('trailingEps', 'N/A')

        # Ex-dividend date, read off the `info` payload already in hand — the
        # event radar (roadmap 3.5) needs it and this costs no extra request.
        # Absent stays absent: an unknown ex-div date is reported as None and the
        # radar says nothing, rather than guessing a quarter forward.
        ex_div_date = None
        ex_div_days_until = None
        raw_ex_div = info.get('exDividendDate')
        if raw_ex_div:
            try:
                ex_dt = datetime.fromtimestamp(float(raw_ex_div))
                ex_div_date = ex_dt.strftime("%Y-%m-%d")
                ex_div_days_until = (ex_dt.date() - datetime.now().date()).days
            except (TypeError, ValueError, OSError, OverflowError):
                ex_div_date = None
                ex_div_days_until = None

        return {
            "symbol": symbol,
            "next_earnings_date": next_earnings or "Not available",
            "days_until_earnings": days_until_earnings,
            "ex_dividend_date": ex_div_date,
            "days_until_ex_dividend": ex_div_days_until,
            "earnings_warning": earnings_warning,
            "eps_estimate_forward": eps_estimate,
            "eps_ttm": eps_ttm,
            "historical_performance": {
                "avg_surprise_percent": f"{avg_surprise:.1f}%" if avg_surprise else "N/A",
                "beat_rate": f"{beat_rate:.0f}%" if beat_rate else "N/A"
            },
            "trading_note": "Consider reducing position size or using options to hedge before earnings" if earnings_warning else "No imminent earnings event"
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol}
