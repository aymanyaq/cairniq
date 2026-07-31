"""
As-of freshness stamps (Advisor Roadmap Theme 5.8 — alert-path slice).

The problem this closes: `data_freshness` on a market payload is a static label
about the SOURCE ("Real-time", "Delayed 15-20min"), not a statement about THIS
observation. `get_stock_data` is cached with a 1-hour TTL, so a quote fetched at
09:00 and read at 15:00 still calls itself "Real-time". That is the exact shape
of the 2026-07-15 stale-quote incident (an end-of-day print presented as the live tape) — and now
that 3.3 watch-conditions and 3.4 the intraday sentinel fire alerts on their own,
with desktop notifications and no human in the loop, an unprovable quote becomes
an unprompted false alarm.

The fix is a timestamp taken at FETCH time, carried inside the payload:

    stamp(payload)          # at the moment the data is actually fetched
    as_of(payload)          # -> datetime | None, read back later
    age_minutes(payload)    # -> float | None, true age even through a cache hit

Because the stamp is written inside the returned object BEFORE the caching layer
stores it, a cache hit replays the ORIGINAL fetch time rather than the read time.
That is what makes the age honest.

Two freshness notions, because the two alert paths read different things:

  - QUOTE data (a point-in-time price, e.g. get_stock_data) → `age_minutes`.
    A quote is fresh if it was fetched recently.
  - DAILY-BAR data (an OHLC series, e.g. the batch downloads behind the sentinel)
    → `is_current_session`. The last bar's DATE is what matters: during a session
    the provider returns a partial bar for today, so a last bar dated yesterday
    means the feed has not updated, however recently it was downloaded. Minute-age
    is meaningless here — a bar stamped 00:00 today is 10 hours "old" at 10am and
    still perfectly current.

Policy for callers on the alert path: absence of proof is not proof of freshness.
A payload with no readable stamp is `unverified` — callers must not describe it
as live. Whether an unverified payload may still fire is the caller's call;
`describe()` gives text that never overclaims either way.
"""
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

AS_OF_KEY = "_as_of"

_MARKET_TZ = ZoneInfo("US/Eastern")


def stamp(payload: Any, now: datetime | None = None) -> Any:
    """Record the fetch time inside `payload`, in place. Returns it unchanged
    otherwise, so it can wrap a return expression directly.

    Call this where the data is actually FETCHED, never where it is read — a
    stamp applied on read would restart the clock on every cache hit and make a
    stale payload look permanently fresh, which is the bug this module exists to
    prevent. Non-dict payloads pass through untouched.
    """
    if isinstance(payload, dict) and AS_OF_KEY not in payload:
        payload[AS_OF_KEY] = (now or datetime.now()).isoformat(timespec="seconds")
    return payload


def as_of(payload: Any) -> datetime | None:
    """Read the stamp back. None when absent or unparseable (never raises)."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get(AS_OF_KEY)
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def age_minutes(payload: Any, now: datetime | None = None) -> float | None:
    """Minutes since the payload was fetched. None when it carries no stamp.

    For QUOTE data. A negative age (a clock skew, or a stamp from the future) is
    clamped to 0 rather than reported as impossibly fresh.
    """
    stamped = as_of(payload)
    if stamped is None:
        return None
    reference = now or datetime.now()
    try:
        if stamped.tzinfo and not reference.tzinfo:
            stamped = stamped.replace(tzinfo=None)
        elif reference.tzinfo and not stamped.tzinfo:
            reference = reference.replace(tzinfo=None)
        return max(0.0, (reference - stamped).total_seconds() / 60.0)
    except (TypeError, ValueError):
        return None


def is_stale(payload: Any, max_age_minutes: float, now: datetime | None = None) -> bool:
    """True when a QUOTE is older than `max_age_minutes`.

    An unstamped payload is NOT reported as stale here — it is *unverified*, a
    different thing, and conflating the two would let a caller claim it proved
    something it merely failed to measure. Use `is_verified` to tell them apart.
    """
    age = age_minutes(payload, now)
    return age is not None and age > max_age_minutes


def is_verified(payload: Any) -> bool:
    """True when the payload carries a readable fetch stamp."""
    return as_of(payload) is not None


def is_current_session(payload: Any, now: datetime | None = None) -> bool | None:
    """True when DAILY-BAR data's stamp falls on today's US/Eastern date.

    None when unstamped. This is the right freshness test for an OHLC series: a
    provider serving a session in progress returns a partial bar dated today, so
    a last bar dated earlier means the feed has not caught up — regardless of how
    recently the download ran.
    """
    stamped = as_of(payload)
    if stamped is None:
        return None
    reference = now or datetime.now(_MARKET_TZ)
    try:
        return _eastern_date(stamped) == _eastern_date(reference)
    except (TypeError, ValueError):
        return None


def _eastern_date(value: datetime) -> date:
    """The US/Eastern calendar date of `value`; naive input is taken as already
    being market-local, which is what every bar index in this codebase is."""
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(_MARKET_TZ).date()


def describe(payload: Any, now: datetime | None = None) -> str:
    """Human phrasing that never overclaims — for alert bodies and prose.

    "as of 09:42 (3 min ago)" when provable; "as-of unverified" when not. It
    deliberately has no wording for "live" or "real-time": this module can prove
    when data was fetched, never that a venue was actually trading.
    """
    stamped = as_of(payload)
    if stamped is None:
        return "as-of unverified"
    age = age_minutes(payload, now)
    clock = stamped.strftime("%H:%M")
    if age is None:
        return f"as of {clock}"
    if age < 1:
        return f"as of {clock} (just now)"
    if age < 90:
        return f"as of {clock} ({int(round(age))} min ago)"
    hours = age / 60.0
    if hours < 48:
        return f"as of {stamped.strftime('%b %d %H:%M')} ({hours:.1f}h ago)"
    return f"as of {stamped.strftime('%b %d %H:%M')} ({int(hours // 24)}d ago)"
