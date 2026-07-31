"""7.1 number 3 — the crossings a dead box could not see, replayed.

7.1 measured the exposure and refused to convert it: *"`window_minutes_lost` and
the dated `gaps` are the exposure. Turning that into a count of missed crossings
needs the market data for those windows replayed against the armed conditions,
which this module does not do and must not estimate."* This module is that
replay, and it is the last computable half of the item — the other one (the sleep
remedy) is blocked on a cause the instrument cannot name retroactively.

**Four refusals, and three of them are about resolution rather than data.**

  * **A gap with no price history is `no_data`, never zero.** The whole point of
    this number is that a missed crossing leaves no log line; an absent replay
    leaves no line either, and reporting it as "no alerts missed" would be the
    second silence stacked on the first.

  * **A bar interval wider than the gap cannot see inside it.** yfinance serves
    1-minute bars for about a week, hourly for two years, and daily beyond that.
    A four-hour outage replayed against DAILY bars is not a coarse measurement of
    the right answer — it is a measurement of a different window. Those windows
    report `resolution_too_coarse` and are counted as unmeasurable, not as clean.

  * **A high/low inside a bar is a CROSSING; a close is not.** Replaying closes
    only would miss precisely the spike a threshold exists to catch, so the
    replay reads each bar's high and low. That makes the count an upper bound on
    what a 30-minute evaluator would have caught — the tick engine samples, and a
    spike between two of its own ticks was never going to fire either. Both
    bounds are named on the payload rather than one of them being quoted.

  * **A condition never checked before the gap would have VOIDED, not fired.**
    `evaluate_conditions` voids a condition satisfied on its first check, because
    a trigger already true when it was written is noise rather than news. So a
    crossing during a gap on a condition with no `checked_at` before that gap is
    not a missed alert — it is a missed VOID, which costs nothing. Conflating the
    two would inflate this count with the cheapest possible false positives.

**Per-profile, and deliberately NOT on `/api/availability`.** Watch conditions
belong to one person; that report is global. This is the same call 7.1 made for
delivery latency (number 4), and for the same reason: a count summed across
profiles would answer nobody's question.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from tools.exception_logger import log_exceptions

# AUTHORED CONSTANT (2.7). How many bars must fall inside a gap before the replay
# will call the window measured. Two, because one bar cannot show a crossing —
# it shows a level. Not a statistical threshold; it is the minimum arithmetic.
MIN_BARS_IN_WINDOW = 2

# Bar intervals by gap age, mirroring what the provider will actually serve.
# Requesting 1m data for a 40-day-old window returns EMPTY rather than an error,
# which would land as `no_data` and read as "nothing to see".
_INTERVAL_BY_AGE_DAYS: tuple[tuple[int, str], ...] = (
    (7, "1m"),
    (59, "5m"),
    (729, "1h"),
)
_FALLBACK_INTERVAL = "1d"

# Bar width in minutes, used to decide whether an interval can see inside a gap.
_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440}


def interval_for(gap_start: datetime, now: datetime | None = None) -> str:
    """The finest bar interval the provider will still serve for that date."""
    now = now or datetime.now()
    age_days = max(0, (now - gap_start).days)
    for max_age, interval in _INTERVAL_BY_AGE_DAYS:
        if age_days <= max_age:
            return interval
    return _FALLBACK_INTERVAL


# ---------------------------------------------------------------------------
# Which conditions were armed when
# ---------------------------------------------------------------------------
def _dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None


def armed_during(condition: dict[str, Any], start: datetime,
                 end: datetime) -> bool:
    """Whether `condition` was live and unresolved for any part of [start, end].

    The store keeps one current status per record, so "was it armed then" has to
    be reconstructed from three stamps: it existed (`created_at`), it had not
    been retired (`resolved_at`), and it had not expired (`expires_at`). A record
    missing `created_at` is treated as NOT armed — an undated condition cannot be
    placed in time, and assuming it was live would manufacture exposure.
    """
    created = _dt(condition.get("created_at"))
    if created is None or created > end:
        return False
    resolved = _dt(condition.get("resolved_at"))
    if resolved is not None and resolved < start:
        return False
    expires = _dt(condition.get("expires_at"))
    if expires is not None and expires < start:
        return False
    return True


def _was_checked_before(condition: dict[str, Any], start: datetime) -> bool:
    """Whether the evaluator had already seen this condition unsatisfied.

    Decides `would_have_fired` vs `would_have_voided`. `checked_at` is stamped by
    `evaluate_conditions` on every tick, so a condition with no stamp before the
    gap had never been evaluated — and a first evaluation finding it true VOIDS
    it rather than firing.
    """
    checked = _dt(condition.get("checked_at"))
    return checked is not None and checked < start


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------
def _crosses(bar: dict[str, float], operator: str, threshold: float) -> bool:
    """Whether a bar's RANGE reaches the threshold, not just its close."""
    high, low = bar.get("high"), bar.get("low")
    if high is None or low is None:
        return False
    if operator in (">=", ">"):
        return high >= threshold if operator == ">=" else high > threshold
    if operator in ("<=", "<"):
        return low <= threshold if operator == "<=" else low < threshold
    return False


def _pct_bars(bars: list[dict[str, Any]], prev_close: float | None) -> list[dict[str, Any]]:
    """Convert price bars into day-change-percent bars.

    `pct_change` is measured against the PREVIOUS CLOSE, so without one the
    metric cannot be replayed at all — and this returns an empty list rather than
    substituting the window's own opening price, which would silently redefine
    the metric mid-replay.
    """
    if not prev_close or prev_close <= 0:
        return []
    return [{
        "at": b["at"],
        "high": (b["high"] / prev_close - 1.0) * 100.0,
        "low": (b["low"] / prev_close - 1.0) * 100.0,
    } for b in bars if b.get("high") is not None and b.get("low") is not None]


@log_exceptions()
def replay_gap(gap: dict[str, Any], conditions: list[dict[str, Any]],
               bars_fn: Callable[[str, datetime, datetime, str], list[dict[str, Any]]] | None = None,
               prev_close_fn: Callable[[str, date], float | None] | None = None,
               now: datetime | None = None) -> dict[str, Any]:
    """Replay one dated outage window against the conditions armed across it.

    `gap` is a row from `availability.find_gaps` — `{start, end, hours,
    window_minutes_lost}`. Read `status`:

      * ``outside_window``       — the gap cost no window minutes (overnight, or
        a closed market). Nothing could have fired; this is a real zero.
      * ``no_conditions_armed``  — nothing was watching. Also a real zero.
      * ``resolution_too_coarse``— the finest available bar is wider than the
        gap. UNMEASURABLE, not clean.
      * ``no_data``              — no price history came back. UNMEASURABLE.
      * ``measured``             — `missed` is the count, and `crossings` the why.
    """
    start, end = _dt(gap.get("start")), _dt(gap.get("end"))
    if start is None or end is None:
        return {"status": "no_data", "reason": "the gap carries no readable timestamps"}

    base = {
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "hours": gap.get("hours"),
        "window_minutes_lost": gap.get("window_minutes_lost"),
    }

    if not (gap.get("window_minutes_lost") or 0) > 0:
        return {**base, "status": "outside_window", "missed": 0, "unmeasurable": 0,
                "note": ("This gap cost no coverage-window minutes — it fell "
                         "overnight or on a closed market. Nothing was due to "
                         "evaluate, so zero here is a real zero.")}

    armed = [c for c in conditions if armed_during(c, start, end)]
    if not armed:
        return {**base, "status": "no_conditions_armed", "missed": 0, "unmeasurable": 0,
                "conditions_armed": 0,
                "note": ("No watch condition was armed across this window, so "
                         "nothing could have fired. Zero here is a real zero — "
                         "and it says nothing about whether the market moved.")}

    interval = interval_for(start, now)
    gap_minutes = (end - start).total_seconds() / 60.0
    bar_minutes = _INTERVAL_MINUTES.get(interval, 1440)

    if bar_minutes * MIN_BARS_IN_WINDOW > gap_minutes:
        return {
            **base, "status": "resolution_too_coarse", "missed": None,
            "unmeasurable": len(armed), "conditions_armed": len(armed),
            "interval": interval, "bar_minutes": bar_minutes,
            "note": (
                f"The finest bar the provider still serves for {start.date()} is "
                f"{interval} ({bar_minutes} min), and this gap is "
                f"{gap_minutes:.0f} min long. Fewer than {MIN_BARS_IN_WINDOW} bars "
                "fall inside it, so a crossing during the outage cannot be seen. "
                "This window is UNMEASURABLE, which is not the same as clean."
            ),
        }

    fetch = bars_fn or _default_bars
    prev_close = prev_close_fn or _default_prev_close

    crossings: list[dict[str, Any]] = []
    unmeasurable: list[dict[str, Any]] = []
    price_cache: dict[str, list[dict[str, Any]]] = {}

    for cond in armed:
        symbol = str(cond.get("symbol") or "").upper()
        if symbol not in price_cache:
            try:
                price_cache[symbol] = fetch(symbol, start, end, interval) or []
            except Exception:  # noqa: BLE001 — one bad symbol must not abort the replay
                price_cache[symbol] = []
        bars = price_cache[symbol]

        if len(bars) < MIN_BARS_IN_WINDOW:
            unmeasurable.append({**_cond_row(cond), "reason":
                                 f"only {len(bars)} bar(s) of {interval} history "
                                 f"inside the window"})
            continue

        if cond.get("metric") == "pct_change":
            base_px = prev_close(symbol, start.date())
            metric_bars = _pct_bars(bars, base_px)
            if not metric_bars:
                unmeasurable.append({**_cond_row(cond), "reason":
                                     "no previous close, so day-change percent "
                                     "cannot be reconstructed for this window"})
                continue
        else:
            metric_bars = bars

        hit = next((b for b in metric_bars
                    if _crosses(b, str(cond.get("operator")), float(cond.get("threshold")))),
                   None)
        if hit is None:
            continue

        checked_before = _was_checked_before(cond, start)
        crossings.append({
            **_cond_row(cond),
            "crossed_at": hit["at"],
            "bar_high": round(hit["high"], 4),
            "bar_low": round(hit["low"], 4),
            # The distinction that keeps this count honest.
            "outcome": "would_have_fired" if checked_before else "would_have_voided",
            "outcome_note": (
                "The evaluator had already seen this condition unsatisfied, so a "
                "crossing here is a genuinely missed alert."
                if checked_before else
                "This condition had never been evaluated before the outage. A "
                "first check finding it true VOIDS it rather than firing — the "
                "trigger was already true when written, which is noise, not news. "
                "Not counted as a miss."
            ),
        })

    missed = [c for c in crossings if c["outcome"] == "would_have_fired"]
    voided = [c for c in crossings if c["outcome"] == "would_have_voided"]

    return {
        **base,
        "status": "measured",
        "interval": interval,
        "conditions_armed": len(armed),
        "missed": len(missed),
        "would_have_voided": len(voided),
        "unmeasurable": len(unmeasurable),
        "crossings": crossings,
        "unmeasurable_conditions": unmeasurable,
        "bound": "upper",
        "bound_note": (
            "Counted from each bar's HIGH and LOW, so an intraday spike counts. "
            "The live evaluator samples every 30 minutes and would have missed "
            "some of these anyway — so this is an UPPER bound on what the outage "
            "actually cost, and the lower bound (close-to-close) is not computed "
            "because a threshold exists precisely to catch the spike."
        ),
    }


def _cond_row(cond: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": cond.get("id"),
        "symbol": str(cond.get("symbol") or "").upper(),
        "metric": cond.get("metric"),
        "operator": cond.get("operator"),
        "threshold": cond.get("threshold"),
        "label": cond.get("label"),
        "status_now": cond.get("status"),
    }


# ---------------------------------------------------------------------------
# The read surface
# ---------------------------------------------------------------------------
@log_exceptions()
def get_missed_alerts(path: str | None = None,
                      bars_fn: Callable | None = None,
                      prev_close_fn: Callable | None = None,
                      conditions: list[dict[str, Any]] | None = None,
                      now: datetime | None = None) -> dict[str, Any]:
    """7.1 number 3 for THIS profile: what the measured outages plausibly cost.

    Reads the same gaps `/api/availability` reports and replays each one. The two
    surfaces must agree on the gaps and deliberately do not share a payload — the
    availability report is global and a watch condition belongs to one person.

    `measurable_windows` against `windows` is the figure to read before `missed`.
    A total of zero across four windows nobody could replay is not a clean record,
    and this is the same bound-versus-estimate distinction the coverage figure
    itself carries.
    """
    from tools.availability import measure_availability
    from tools.watch_conditions import get_conditions

    report = measure_availability(path)
    if report.get("status") != "measured":
        return {
            "status": "no_availability_data",
            "note": (
                "The availability instrument has no readable record, so there are "
                "no dated outages to replay. This is an absent measurement, not a "
                "report that nothing was missed."
            ),
        }

    if conditions is None:
        # Every status, not just active: a condition that has since fired or
        # expired was still armed during an outage three weeks ago, and reading
        # only the live ones would shrink the exposure to whatever survived.
        conditions = get_conditions(status=None, limit=0)

    gaps = [g for g in report.get("gaps", []) if (g.get("window_minutes_lost") or 0) > 0]
    replays = [replay_gap(g, conditions, bars_fn=bars_fn,
                          prev_close_fn=prev_close_fn, now=now)
               for g in gaps]

    measured = [r for r in replays if r.get("status") == "measured"]
    unmeasurable_windows = [r for r in replays
                            if r.get("status") in {"resolution_too_coarse", "no_data"}]
    real_zeros = [r for r in replays if r.get("status") == "no_conditions_armed"]

    total_missed = sum(r["missed"] for r in measured)
    total_voided = sum(r["would_have_voided"] for r in measured)
    partial = sum(r["unmeasurable"] for r in measured)

    return {
        "status": "measured" if measured else "unmeasurable",
        "windows": len(gaps),
        "measurable_windows": len(measured),
        "unmeasurable_windows": len(unmeasurable_windows),
        "windows_with_nothing_armed": len(real_zeros),
        "conditions_considered": len(conditions),
        "missed_crossings": total_missed,
        "would_have_voided": total_voided,
        "conditions_unmeasurable_in_measured_windows": partial,
        "window_minutes_lost": report.get("window_minutes_lost"),
        "span_days": report.get("span_days"),
        "replays": replays,
        "bound": "upper",
        "summary": _summarize(len(gaps), len(measured), len(unmeasurable_windows),
                              total_missed, total_voided, len(conditions)),
    }


def _summarize(windows: int, measured: int, unmeasurable: int, missed: int,
               voided: int, conditions: int) -> str:
    if not windows:
        return ("No outage in the measured span cost any coverage-window minutes, "
                "so there is nothing to replay.")
    if not conditions:
        return (f"{windows} outage window(s) cost coverage-window minutes, and this "
                "profile has never armed a watch condition — so nothing could have "
                "fired in any of them.")
    if not measured:
        return (f"{windows} outage window(s) to replay and NONE could be measured: "
                "the price history needed is either absent or coarser than the gap. "
                "This is an unmeasurable record, not a clean one.")
    parts = [f"{missed} crossing(s) would have fired during {measured} replayable "
             f"outage window(s)"]
    if voided:
        parts.append(f"{voided} more would have been voided as already-true rather "
                     f"than fired, and are not counted as misses")
    if unmeasurable:
        parts.append(f"{unmeasurable} window(s) could not be replayed at all, so the "
                     f"true figure is at least this")
    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Providers — network, injected away in tests
# ---------------------------------------------------------------------------
def _default_bars(symbol: str, start: datetime, end: datetime,
                  interval: str) -> list[dict[str, Any]]:
    """OHLC bars covering [start, end] at `interval`. Empty on any failure."""
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(
            start=(start - timedelta(minutes=1)).isoformat(),
            end=(end + timedelta(minutes=1)).isoformat(),
            interval=interval,
        )
        if hist is None or hist.empty:
            return []
        out: list[dict[str, Any]] = []
        for idx, row in hist.iterrows():
            stamp = idx.to_pydatetime().replace(tzinfo=None)
            if not (start <= stamp <= end):
                continue
            out.append({"at": stamp.isoformat(timespec="seconds"),
                        "high": float(row["High"]), "low": float(row["Low"]),
                        "close": float(row["Close"])})
        return out
    except Exception:  # noqa: BLE001 — an unreachable provider is an unmeasurable window
        return []


def _default_prev_close(symbol: str, on_date: date) -> float | None:
    """The close before `on_date`, which is what `pct_change` is measured against."""
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(start=(on_date - timedelta(days=7)).isoformat(),
                                         end=on_date.isoformat())
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":  # pragma: no cover — operator convenience
    import json

    print(json.dumps(get_missed_alerts(), indent=2, default=str))
