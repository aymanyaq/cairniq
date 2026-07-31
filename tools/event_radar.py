"""Holdings event radar — Advisor Roadmap 3.5.

Calendar-driven advice is table stakes: an advisor that discusses a position
without knowing it reports earnings on Thursday is not advising, it is
commentating. `tools/earnings_calendar.py` and `tools/fed_calendar.py` have been
shipped and standalone for weeks — nothing merged them against the actual book,
so the dates existed and nobody was ever told.

This is that merge: one deterministic pass over HELD names, joining earnings
dates, ex-dividend dates and the FOMC calendar into a single dated view, plus
T-3 / T-1 alerts on the 3.2 rail.

Design constraints carried from the rest of this codebase:

  - **A missing date is silence, never a guess.** A symbol whose provider has no
    earnings date is reported as unknown and produces no alert. Inferring "about
    a quarter after the last one" would put a fabricated date in front of someone
    sizing a position.
  - **The countdown is DERIVED from the date, never read alongside it.** Providers
    hand back a date and a days-until as two independent fields, and they drift.
    `get_earnings_info` computed its countdown as `(event_datetime - now).days` — a
    timedelta truncation against a clock carrying a time of day — so a print dated
    tomorrow at 06:00 read as 0 days from any moment after 06:00 today. Observed
    2026-07-29: AAPL and AMZN both dated 2026-07-30 and captioned "Today", and,
    the half that matters, T-3/T-1 alerts firing against the same off-by-one. One
    parse of one field now feeds both halves of every event.
  - **T-3 and T-1 only, each firing once.** Not a daily countdown. An advisor
    that says "earnings in 9 days… 8 days… 7 days" trains the reader to skip it,
    and by T-1 it is wallpaper. Dedup keys pin each alert to its event DATE, so a
    rescheduled event is a new alert rather than a suppressed one.
  - **Zero LLM.** Dates are facts; nothing here needs a model.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

# Days-until values that fire. Deliberately sparse — see the module docstring.
ALERT_OFFSETS = (3, 1)

# Ex-dividend is T-1 only: the actionable fact is "own it before tomorrow", and a
# three-day warning on a dividend is noise for a decade-horizon holder.
EX_DIV_OFFSETS = (1,)


def _fetch_holdings() -> list[str]:
    from tools.portfolio_csv import get_tradeable_symbols
    return list(get_tradeable_symbols() or [])


def _fetch_earnings(symbol: str) -> dict[str, Any]:
    from tools.earnings_calendar import get_earnings_info
    return get_earnings_info(symbol) or {}


def _fetch_fomc() -> dict[str, Any]:
    from tools.fed_calendar import get_fomc_calendar
    return get_fomc_calendar(num_meetings=2) or {}


@log_exceptions()
def build_event_radar(
    holdings_fn: Callable[[], list[str]] | None = None,
    earnings_fn: Callable[[str], dict[str, Any]] | None = None,
    fomc_fn: Callable[[], dict[str, Any]] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """The merged calendar for held names. Every source is injectable for tests.

    Returns `{"as_of", "events": [...], "checked": n, "unknown": [...]}` where
    each event is `{kind, symbol|None, date, days_until, label}`, sorted nearest
    first. `unknown` names the symbols whose date could not be established —
    surfaced rather than dropped, because "no earnings coming" and "the provider
    did not tell us" are different facts and only one of them is safe to act on.

    `days_until` is DERIVED from `date` against `today` — every source's own
    countdown field is ignored, whatever clock it was computed against. A date that
    cannot be parsed is a date that was not established, so it lands in `unknown`
    alongside the missing ones.
    """
    holdings_fn = holdings_fn or _fetch_holdings
    earnings_fn = earnings_fn or _fetch_earnings
    fomc_fn = fomc_fn or _fetch_fomc
    today = today or date.today()

    events: list[dict[str, Any]] = []
    unknown: list[str] = []
    symbols = holdings_fn()

    for symbol in symbols:
        try:
            info = earnings_fn(symbol) or {}
        except Exception as e:  # noqa: BLE001 — one bad symbol must not abort the sweep
            logger.debug(f"event radar: {symbol} lookup failed: {e}")
            unknown.append(symbol)
            continue
        if info.get("error"):
            unknown.append(symbol)
            continue

        # The provider's own days_until_earnings is deliberately not read. See the
        # module docstring: it drifted from the date it was supposed to describe.
        dated = _dated(info.get("next_earnings_date"), today)
        if dated:
            iso_date, days = dated
            events.append({
                "kind": "earnings",
                "symbol": symbol,
                "date": iso_date,
                "days_until": days,
                "label": f"{symbol} reports earnings",
            })
        else:
            unknown.append(symbol)

        dated = _dated(info.get("ex_dividend_date"), today)
        if dated:
            iso_date, days = dated
            events.append({
                "kind": "ex_dividend",
                "symbol": symbol,
                "date": iso_date,
                "days_until": days,
                "label": f"{symbol} goes ex-dividend",
            })

    try:
        fomc = fomc_fn() or {}
        for meeting in (fomc.get("upcoming_meetings") or fomc.get("calendar") or []):
            dated = _dated(meeting.get("date"), today)
            if dated:
                iso_date, days = dated
                events.append({
                    "kind": "fomc",
                    "symbol": None,
                    "date": iso_date,
                    "days_until": days,
                    "label": "FOMC rate decision",
                })
    except Exception as e:  # noqa: BLE001 — the macro half must not cost the holdings half
        logger.debug(f"event radar: FOMC lookup failed: {e}")

    events.sort(key=lambda e: (e["days_until"], e["kind"], e["symbol"] or ""))
    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "checked": len(symbols),
        "events": events,
        "unknown": sorted(set(unknown)),
    }


_RADAR_CACHE_KEY = "event_radar"

# Six hours. The dates behind this move on a quarterly (earnings, ex-div) or
# multi-week (FOMC) cadence, so re-deriving them per page load buys nothing —
# and the daily cache is date-stamped per profile, so an entry cannot outlive
# the Eastern day that produced it however long this is set.
_RADAR_TTL_SECONDS = 6 * 3600


def build_event_radar_cached() -> dict[str, Any]:
    """The radar for READERS — dashboards, the radar panel, the agent tool.

    Same payload as build_event_radar(), served from the profile's daily cache.
    Worth its own seam because none of the uncached call's cost is visible from the
    call site. It walks held names one at a time (a cached earnings lookup each,
    10.8s cold on a 10-name book), and it re-reads the whole book through
    get_tradeable_symbols() -> load_portfolio(). On the dashboard both ran on every
    request even when the portfolio summary itself was a cache hit, so the
    allocation chart waited on earnings dates it does not display.

    That second read is cheap TODAY and arms itself later: load_portfolio() performs
    a live Questrade AND Alpaca sync, but only for a broker that is actually
    configured, and with both unlinked it currently just parses the CSV. Worth
    stating precisely — the first write-up of this called it a live sync per page
    load, which is what the code says and not what this deployment does.

    `as_of` is the time the radar was BUILT, not the time it was read, so a
    cached payload keeps reporting its true age rather than restarting the clock
    (see tools/cache.py on why the stamp goes on at fetch).

    The ALERT path deliberately does not come through here — run_event_radar_tick
    calls build_event_radar() directly. T-3/T-1 firing is driven by `days_until`,
    and a cached countdown is the one consumer for which staleness changes the
    answer rather than just aging it.
    """
    # Under pytest the cache is skipped entirely. Tests patch build_event_radar
    # and drive these readers through TestClient, whose requests are re-bound to
    # the real default profile by profile_middleware (see the protect_real_user_data
    # fixture) — so a cached call here would write a MOCK radar into the user's
    # real user_data/daily_cache/ and then serve it back for six hours.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return build_event_radar()

    from tools.daily_cache import get_cached, set_cached

    hit = get_cached(_RADAR_CACHE_KEY, ttl_seconds=_RADAR_TTL_SECONDS)
    if hit is not None:
        return hit

    radar = build_event_radar()
    # log_exceptions on build_event_radar means a failure returns a payload rather
    # than raising; caching one would pin the failure for the full TTL.
    if isinstance(radar, dict) and not radar.get("error"):
        set_cached(_RADAR_CACHE_KEY, radar)
    return radar


def _is_usable(value: Any) -> bool:
    """A date string that is actually a date, not a provider's polite refusal."""
    if not value or not isinstance(value, str):
        return False
    return value.strip().lower() not in ("not available", "n/a", "unknown", "none", "")


def _event_day(value: Any) -> date | None:
    """The calendar day `value` names, or None if it does not name one.

    `_is_usable` only rejects the refusal strings a provider is known to send; it
    would pass "Q3 2026" as a date. Parsing is the stronger test, and a value that
    will not parse is treated the same way a missing one is — as a date that could
    not be established, which is the module's standing rule.
    """
    if not _is_usable(value):
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    # A provider that starts returning full ISO timestamps should not silently
    # empty the radar; take the date half rather than reject the row.
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _dated(value: Any, today: date) -> tuple[str, int] | None:
    """`(iso_date, days_until)` for an event date, or None if it has none.

    Both halves come out of ONE parse of ONE field, counted against the `today`
    build_event_radar was handed. That is the whole point: the panel's date and its
    countdown can no longer disagree, because there is nothing left for them to
    disagree about, and the T-3/T-1 gate in due_alerts() now counts the same days
    the reader is shown. A date already in the past returns None — the radar looks
    forward, and a stale provider date is not an event.
    """
    day = _event_day(value)
    if day is None:
        return None
    days = (day - today).days
    if days < 0:
        return None
    return day.isoformat(), days


def due_alerts(radar: dict[str, Any]) -> list[dict[str, Any]]:
    """The subset of `radar` that should fire today, as alert specs.

    An event fires at T-3 and T-1 (T-1 only for ex-dividends). Each spec's
    dedup_key pins the EVENT DATE and the offset, so: the same event cannot fire
    twice at the same offset, T-3 and T-1 are distinct alerts, and an event that
    gets RESCHEDULED produces a fresh alert rather than being silently swallowed
    by the old key.
    """
    specs = []
    for event in radar.get("events") or []:
        offsets = EX_DIV_OFFSETS if event["kind"] == "ex_dividend" else ALERT_OFFSETS
        days = event["days_until"]
        if days not in offsets:
            continue
        when = "tomorrow" if days == 1 else f"in {days} days"
        subject = event["symbol"] or "Markets"
        specs.append({
            "title": f"{subject}: {event['label']} {when}",
            "message": (
                f"{event['label']} on {event['date']} ({when}). "
                + _guidance(event["kind"], days)
            ),
            "severity": "warning" if days == 1 else "info",
            "source": "radar",
            "dedup_key": f"radar:{event['kind']}:{event['symbol'] or 'macro'}:{event['date']}:T{days}",
            "data": {k: event[k] for k in ("kind", "symbol", "date", "days_until")},
        })
    return specs


def _guidance(kind: str, days: int) -> str:
    """One line of what the date MEANS. Deliberately not a recommendation.

    The radar reports a calendar; it does not tell anyone to trade. Sizing and
    action belong to the advisor's own gated path (2.2 pre-check), not to a
    zero-LLM date sweep that has never seen the position.
    """
    if kind == "earnings":
        return ("Expect a volatility event. Any level committed before this print "
                "was set without knowing the result.")
    if kind == "ex_dividend":
        return "Shares must be held before the ex-date to receive this dividend."
    if kind == "fomc":
        return ("Rate decision. Rate-sensitive sleeves — bonds, utilities, long-duration "
                "growth — reprice on the statement, not the vote.")
    return ""


@log_exceptions()
def run_event_radar_tick(
    radar: dict[str, Any] | None = None,
    raise_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build the radar and deliver today's due alerts. Never raises.

    Returns the counts the 2.6 heartbeat needs: `checked` is HELD NAMES SWEPT,
    not alerts raised — a book with nothing inside three days is the normal case
    for most of the year, and counting alerts would report this engine dead
    through every quiet stretch.
    """
    if raise_fn is None:
        from tools.alerts import raise_alert
        raise_fn = raise_alert

    radar = radar if radar is not None else build_event_radar()
    delivered = 0
    for spec in due_alerts(radar):
        try:
            raise_fn(**spec)
            delivered += 1
        except Exception as e:  # noqa: BLE001 — one delivery failure must not lose the rest
            logger.warning(f"event radar alert delivery failed: {e}")

    return {
        "checked": radar.get("checked", 0),
        "events": len(radar.get("events") or []),
        "alerts": delivered,
        "unknown": len(radar.get("unknown") or []),
    }
