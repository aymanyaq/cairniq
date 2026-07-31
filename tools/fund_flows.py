"""ETF / mutual-fund creation-redemption recorder (Roadmap 5.5, recorder half).

WHY THIS IS A RECORDER AND NOT A FETCH — measured 2026-07-28, before a line of it
was written:

  * ``yfinance.Ticker.get_shares_full()`` returns ``None`` for the ENTIRE fund
    class: 16 of 16 funds actually held, and 6 of 6 ETF controls (SPY, QQQ, VTI,
    XIC.TO, ZSP.TO, VFV.TO). An equity control returns 31 rows at a 4.5-day
    median cadence, so the accessor works — just never for a fund. **It is a
    fund-class gap, not a Canadian-listing gap**; US and TSX funds fail
    identically, which is not what the roadmap predicted.
  * No vendor sells the history on the plans we hold: FMP
    ``historical/shares-float`` → 404, ``etf/info`` → 402.
  * A dated daily POINT does exist for both venues — FMP ``shares-float`` →
    ``outstandingShares`` (SCHD 2.88B, SPY 1.06B, XQQ.TO 81.6M, XESG.TO 14.9M).

So there is no series to fetch. There is a point to record, and the series has to
be accrued locally one day at a time.

**This module cannot answer on the day it ships**, and says so instead of
pretending otherwise: the first week-over-week delta lands ~7 days after the
first recorded row, and a credible price-vs-flow divergence read takes 2-4 weeks.
:func:`get_flow_series` reports ``status="accruing"`` with the day count rather
than drawing a 0.0% flow — a confidently-drawn empty number is the failure this
codebase has already paid for in the Market Pulse outage, in a tone verdict for
calls that were never read, and in an insider table of ``Unknown`` rows.

SHARES, NOT AUM, on purpose. A share count is unit-free, so no FX conversion
enters the flow arithmetic and a CAD-listed fund's series is directly comparable
with a USD-listed one. AUM would have needed both a currency and a NAV return
subtracted out to separate a real creation from a market move.

ONE SOURCE PER SERIES. FMP and Yahoo disagree on SPY's share count by ~15%
(1.060B vs 0.918B) — both internally consistent, differently defined. Only
deltas matter for flows, so the level gap is harmless right up until a single
series spans both sources, which would manufacture a 15% creation event out of a
source switch. Every row therefore records the source that produced it, and
:func:`get_flow_series` refuses to difference across a change instead of
reporting the artefact.
"""
import csv
import os
from datetime import date, datetime, timedelta
from typing import Any

from agent.logger import log_to_component
from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.tool_errors import is_unavailable, missing_key_reason, unavailable

# The pinned source for the shares series. Recorded on every row; a reader must
# never difference two rows that disagree on it. See the module docstring.
SHARES_SOURCE = "fmp:shares-float"

_HISTORY_FILE = "fund_shares_history.csv"
_FIELDS = ["date", "symbol", "shares_outstanding", "source", "as_of"]

# A week-over-week comparison needs a gap that is actually about a week. Below
# the floor it is a day-over-day reading wearing a WoW label; above the ceiling
# the recorder missed days and the delta silently spans a longer period.
_WOW_MIN_DAYS = 5
_WOW_MAX_DAYS = 10


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def history_path() -> str:
    """Absolute path of the shares-outstanding store.

    GLOBAL, not per-profile: shares outstanding is a fact about the fund, not
    about whoever holds it. A per-profile store would poll the same ETF once per
    profile and — worse — a profile added later would start its series from
    scratch instead of inheriting the weeks already recorded.
    """
    return os.path.join(_project_root(), "user_data", _HISTORY_FILE)


def _finite(value: Any) -> float | None:
    """Coerce to a finite float or None. NaN/inf never reach the store — a bare
    NaN is not valid JSON, and one of them once took an endpoint down for a day."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


# ---------------------------------------------------------------------------
# Fetch — one dated point, per symbol
# ---------------------------------------------------------------------------
@cached(key_func=lambda symbol: f"fund_shares:{symbol.upper()}", ttl=6 * 3600)
@log_exceptions()
def get_shares_outstanding(symbol: str) -> dict[str, Any]:
    """Today's shares-outstanding point for `symbol`, from the pinned source.

    Works for US and TSX listings alike (measured). Returns the canonical
    degraded payload rather than a zero when the source cannot answer — a
    fabricated 0 would read downstream as a total redemption.
    """
    from tools.fmp_api import _fmp_get, _fmp_key

    symbol = str(symbol or "").upper().strip()
    if not symbol:
        return unavailable(SHARES_SOURCE, "no symbol given")
    if not _fmp_key():
        return unavailable(SHARES_SOURCE, missing_key_reason("FMP_API_KEY"), symbol=symbol)

    data, err = _fmp_get("shares-float", {"symbol": symbol}, timeout=12)
    if err:
        return unavailable(SHARES_SOURCE, err, symbol=symbol)

    row = (data or [{}])[0] if isinstance(data, list) else (data or {})
    shares = _finite(row.get("outstandingShares"))
    if not shares or shares <= 0:
        return unavailable(
            SHARES_SOURCE,
            f"no usable outstandingShares in the response for {symbol}",
            symbol=symbol,
        )

    return {
        "symbol": symbol,
        "shares_outstanding": shares,
        "source": SHARES_SOURCE,
        # The vendor's own stamp is about ITS data age; `_as_of` from the cache
        # decorator is our fetch time. Keep both — they answer different
        # questions, and conflating them is what 5.8 was opened for.
        "source_stamp": str(row.get("date") or ""),
    }


@cached(key_func=lambda symbol: f"is_fund:{symbol.upper()}", ttl=24 * 3600)
@log_exceptions()
def is_fund(symbol: str) -> dict[str, Any]:
    """Whether `symbol` is an ETF or mutual fund, by Yahoo's quoteType.

    Returned as a dict rather than a bool so the cache layer can stamp it and so
    an unresolved symbol is distinguishable from a resolved "no".
    """
    import yfinance as yf

    symbol = str(symbol or "").upper().strip()
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as e:  # noqa: BLE001 — one unresolvable symbol must not stop a sweep
        return unavailable("yahoo:quoteType", f"lookup failed: {e}", symbol=symbol)

    qtype = str(info.get("quoteType") or "").upper()
    if not qtype:
        return unavailable("yahoo:quoteType", "no quoteType in response", symbol=symbol)
    return {"symbol": symbol, "quote_type": qtype, "is_fund": qtype in ("ETF", "MUTUALFUND")}


# ---------------------------------------------------------------------------
# Universe — the funds anyone actually holds
# ---------------------------------------------------------------------------
def _classify_fund_symbols(symbols: set[str], profiles_read: int) -> dict[str, Any]:
    """Classify a supplied holding set without changing its profile scope."""
    funds, non_funds, unresolved = [], [], []
    for sym in sorted(symbols):
        verdict = is_fund(sym)
        if is_unavailable(verdict):
            unresolved.append(sym)
        elif verdict.get("is_fund"):
            funds.append(sym)
        else:
            non_funds.append(sym)

    return {
        "funds": funds,
        "non_funds": len(non_funds),
        "unresolved": unresolved,
        "profiles_read": profiles_read,
    }


@log_exceptions()
def collect_active_profile_fund_universe() -> dict[str, Any]:
    """Classify ETF and mutual-fund holdings for the current profile only.

    The shares-outstanding history is intentionally global, but an interactive
    request must never reveal which symbols another profile holds. Scheduler
    work that needs the global recording universe uses ``collect_fund_universe``.
    """
    from tools.portfolio_csv import get_tradeable_symbols

    symbols = {
        str(symbol).upper().strip()
        for symbol in (get_tradeable_symbols() or [])
        if str(symbol).strip()
    }
    return _classify_fund_symbols(symbols, profiles_read=1)


@log_exceptions()
def collect_fund_universe() -> dict[str, Any]:
    """Union of fund symbols held across all real profiles.

    A union, because the store is global: recording SCHD once serves every
    profile that holds it. Unresolved symbols are counted and named rather than
    dropped silently — a classifier that quietly excludes a holding would shrink
    the universe without anyone being able to see it happen.
    """
    from tools.portfolio_csv import get_tradeable_symbols
    from tools.user_profile import list_available_profiles, run_under_profile

    profiles = run_under_profile("default", list_available_profiles)
    names = [
        p["name"] for p in profiles
        if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
    ]

    symbols: set[str] = set()
    for name in names:
        try:
            symbols.update(run_under_profile(name, get_tradeable_symbols) or [])
        except Exception as e:  # noqa: BLE001 — one bad profile must not abort the sweep
            log_to_component("tools", "fund_flows",
                             f"could not read holdings for a profile: {e}", level=30)

    return _classify_fund_symbols(symbols, profiles_read=len(names))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
@log_exceptions()
def read_history(symbol: str | None = None) -> list[dict[str, Any]]:
    """Every recorded row, oldest first, optionally for one symbol."""
    path = history_path()
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("date") and r.get("symbol")]
    if symbol:
        want = str(symbol).upper().strip()
        rows = [r for r in rows if r["symbol"].upper() == want]
    return sorted(rows, key=lambda r: (r["date"], r["symbol"]))


def _write_all(rows: list[dict[str, Any]]) -> None:
    """Rewrite the store atomically — a truncated CSV would lose the whole
    accrued series, which is the one thing here that cannot be re-fetched."""
    path = history_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _FIELDS})
    os.replace(tmp, path)


@log_exceptions()
def record_fund_shares(symbols: list[str] | None = None, force: bool = False) -> dict[str, Any]:
    """Record one shares-outstanding point per fund for today.

    Idempotent per (date, symbol): a second run on the same day records nothing
    unless `force`. Returns a report whose `recorded` count is what proves the
    chain ran end to end — source reachable, rows parsed, store written. Zero
    recorded on a day with a non-empty universe is the dead-engine signal, so it
    must not be smoothed over.
    """
    if symbols is None:
        universe = collect_fund_universe()
        symbols = universe["funds"]
        unresolved = universe["unresolved"]
    else:
        symbols = [str(s).upper().strip() for s in symbols if str(s).strip()]
        unresolved = []

    today = date.today().isoformat()
    rows = read_history()
    existing = {(r["date"], r["symbol"].upper()) for r in rows}

    recorded, skipped_existing, failed = [], [], {}
    for sym in symbols:
        if (today, sym) in existing and not force:
            skipped_existing.append(sym)
            continue

        point = get_shares_outstanding(sym)
        if is_unavailable(point):
            failed[sym] = point.get("reason", "unavailable")
            continue

        new_row = {
            "date": today,
            "symbol": sym,
            "shares_outstanding": point["shares_outstanding"],
            "source": point["source"],
            # Fetch time, not read time (5.8): a stamp applied on read makes a
            # stale point look permanently fresh.
            "as_of": point.get("_as_of") or datetime.now().isoformat(timespec="seconds"),
        }
        rows = [r for r in rows if not (r["date"] == today and r["symbol"].upper() == sym)]
        rows.append(new_row)
        recorded.append(sym)

    if recorded:
        _write_all(sorted(rows, key=lambda r: (r["date"], r["symbol"])))

    if failed:
        log_to_component("tools", "fund_flows",
                         f"{len(failed)} of {len(symbols)} funds could not be recorded",
                         {"failed": failed}, level=30)

    return {
        "date": today,
        "universe": len(symbols),
        "recorded": len(recorded),
        "recorded_symbols": recorded,
        "skipped_existing": len(skipped_existing),
        "failed": len(failed),
        "failures": failed,
        "unresolved": unresolved,
        "source": SHARES_SOURCE,
        "store": history_path(),
    }


# ---------------------------------------------------------------------------
# Read — deliberately honest about a series that is still accruing
# ---------------------------------------------------------------------------
@log_exceptions()
def get_flow_series(symbol: str, weeks: int = 8) -> dict[str, Any]:
    """Week-over-week creation/redemption for one fund, or why there isn't one yet.

    `status` is the field to read first:
      * ``no_data``       — nothing recorded for this symbol at all.
      * ``accruing``      — recording works, but no two points sit a week apart
                            yet. Carries `days_recorded` and `days_until_ready`.
                            **There is no flow number in this state, by design.**
      * ``source_change`` — the two candidate points came from different sources,
                            so differencing them would report a definitional gap
                            (~15% on SPY) as a creation event.
      * ``ready``         — `wow` holds the real delta.
    """
    symbol = str(symbol or "").upper().strip()
    rows = read_history(symbol)
    if not rows:
        return {
            "symbol": symbol,
            "status": "no_data",
            "days_recorded": 0,
            "note": (
                f"No shares-outstanding rows recorded for {symbol}. Either the recorder "
                "has not run since it was added to the portfolio, or the source cannot "
                "answer for this listing — check the recorder's `failures`. This is NOT "
                "a zero-flow reading."
            ),
        }

    cutoff = (date.today() - timedelta(days=weeks * 7 + 3)).isoformat()
    window = [r for r in rows if r["date"] >= cutoff]
    points = []
    for r in window:
        shares = _finite(r.get("shares_outstanding"))
        if shares and shares > 0:
            points.append({"date": r["date"], "shares": shares, "source": r.get("source", "")})

    latest = points[-1] if points else None
    span_days = 0
    if len(points) >= 2:
        span_days = (date.fromisoformat(points[-1]["date"]) - date.fromisoformat(points[0]["date"])).days

    base = {
        "symbol": symbol,
        "days_recorded": len(points),
        "first_date": points[0]["date"] if points else "",
        "latest_date": latest["date"] if latest else "",
        "latest_shares": latest["shares"] if latest else None,
        "source": latest["source"] if latest else "",
        "points": points,
    }

    # The week-ago comparator: the point closest to 7 days back, accepted only if
    # the gap is actually week-shaped.
    prior = None
    if latest:
        target = date.fromisoformat(latest["date"]) - timedelta(days=7)
        candidates = [p for p in points if p["date"] < latest["date"]]
        if candidates:
            prior = min(candidates, key=lambda p: abs((date.fromisoformat(p["date"]) - target).days))

    if not prior:
        return {**base, "status": "accruing", "wow": None,
                "days_until_ready": max(0, _WOW_MIN_DAYS - span_days),
                "note": (
                    f"Recording is live ({len(points)} point(s) since "
                    f"{base['first_date'] or 'today'}), but a week-over-week flow needs two "
                    f"points at least {_WOW_MIN_DAYS} days apart. No flow number is being "
                    "reported — this is an accruing series, not a flat one."
                )}

    gap = (date.fromisoformat(latest["date"]) - date.fromisoformat(prior["date"])).days
    if gap < _WOW_MIN_DAYS:
        return {**base, "status": "accruing", "wow": None,
                "days_until_ready": _WOW_MIN_DAYS - gap,
                "note": (
                    f"Closest earlier point is only {gap} day(s) back. Labelling that a "
                    "week-over-week flow would overstate what was measured, so no number "
                    "is reported yet."
                )}

    if prior["source"] != latest["source"]:
        return {**base, "status": "source_change", "wow": None,
                "note": (
                    f"The two points come from different sources ({prior['source']} → "
                    f"{latest['source']}). Differencing them would report a definitional "
                    "gap as a creation/redemption event — sources disagree on SPY's share "
                    "count by ~15%. Re-base the series on one source before reading a flow."
                )}

    delta = latest["shares"] - prior["shares"]
    pct = (delta / prior["shares"]) * 100 if prior["shares"] else None
    return {
        **base,
        "status": "ready",
        "wow": {
            "from_date": prior["date"],
            "to_date": latest["date"],
            "gap_days": gap,
            "shares_change": delta,
            "percent_change": round(pct, 4) if pct is not None else None,
            "direction": "creations" if delta > 0 else ("redemptions" if delta < 0 else "flat"),
            "stale_window": gap > _WOW_MAX_DAYS,
        },
        "note": (
            f"Flow measured over {gap} days from a single source ({latest['source']})."
            + (f" NOTE: {gap} days is wider than a week — the recorder missed days, so this "
               "spans a longer period than the label suggests."
               if gap > _WOW_MAX_DAYS else "")
        ),
    }
