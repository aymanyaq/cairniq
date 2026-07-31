"""7.1 Step 1 — the availability floor, measured rather than argued.

Theme 3 is titled "an advisor that calls first", and five shipped engines in it
can only call from a machine that is on. A crossing missed because the box was
asleep is not deferred, it is gone, and — unlike a failed fetch — it leaves no
log line behind. That makes availability a correctness property of the push
layer rather than an infrastructure preference, and it is the one number in this
product that no amount of advisor intelligence raises.

Step 1 is measurement, and this module is it. Four choices in it are load-bearing
and each one is a correction to how 7.1 was originally written:

  * **The watchdog probe, not `scheduler_runs.json`.** 7.1 proposed deriving
    outages from "a gap between consecutive ticks of a known-cadence task", on
    the belief that the registry already held that history. It does not:
    ``tools.scheduler._record_run`` does ``registry[task] = time.time()`` and
    rewrites the file, so the store holds ONE timestamp per task and no history
    whatsoever. There is nothing to difference. ``logs/cairniq.watchdog.log`` is
    an append-only ~2-minute liveness probe going back weeks, which is the
    retroactive record 7.1 assumed it had.

  * **Coverage of the trading window, not wall-clock uptime.** A box that sleeps
    every night at 02:00 has poor uptime and perfect availability for this
    product. The figure that means something is the fraction of the pre-open
    precompute plus the cash session that was covered.

  * **Trading days, not calendar weekdays.** A gap on a day the market is shut
    costs nothing. Counting one makes the floor look worse than it is, and the
    first version of this measurement did exactly that — it scored a 57.8h gap
    over the Independence Day long weekend as 412 lost window-minutes, which was
    the single largest "miss" in the report and was worth nothing at all. The
    denominator has to be trading days. See ``MARKET_HOLIDAYS``.

  * **What it cannot measure, it declines to report.** Of 7.1's four numbers,
    one is computable from disk today, one is computable only as an exposure,
    and two are not computable at all. They are named in ``open_measurements``
    on every report rather than quietly folded into a single reassuring figure.
    This file's own recurring finding is a count that reads what is easy where
    the thing that matters is coverage; a coverage number that silently omits
    its own blind spots is that same failure one level up.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from datetime import time as dtime
from typing import Any

from tools.exception_logger import log_exceptions

_PROBE_LOG = "cairniq.watchdog.log"

# The probe's own cadence is ~2 minutes. Two consecutive misses is the smallest
# thing worth calling an outage; anything tighter reports scheduler jitter.
GAP_THRESHOLD_MINUTES = 5.0

# AUTHORED CONSTANTS: the window this product actually needs to be up for, in
# the host's local time. Not a measured figure — it is the 7.0 premarket task's
# start through half an hour past the cash close, which is the span in which
# every push engine in Theme 3 has something to do.
WINDOW_OPEN = dtime(7, 0)
WINDOW_CLOSE = dtime(16, 30)

# AUTHORED DATA, and the years it covers are part of its contract. The scheduler
# is deliberately holiday-blind ("a holiday costs a few wasted ticks"), which is
# fine when the cost is a wasted tick and wrong when the cost is a denominator.
# A date outside these years is NOT silently assumed to be a trading day — see
# `holiday_table_years` and `days_outside_holiday_table` on the report.
MARKET_HOLIDAYS: frozenset[str] = frozenset({
    # 2026 US equity market closures.
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Washington's Birthday
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed — Jul 4 is a Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas Day
})

HOLIDAY_TABLE_YEARS: frozenset[int] = frozenset(
    int(d[:4]) for d in MARKET_HOLIDAYS
)

# 7.1 STEP 2 — THE STATED SLO. Set by the operator on 2026-07-29, after Step 1
# measured 97.93% over 31.9 days, and written down here BEFORE any remedy so the
# remedy can be judged against a number rather than described as an improvement.
#
# Chosen against the arithmetic that makes this window meaningful: at 570
# window-minutes per trading day, ONE lost morning is ~2% of a month. So 98% is
# precisely the bar at which a single lost market morning is a violation, and the
# measured record fails it by nine minutes — a live bar, not a rubber stamp.
# 99% would mean no lost morning ever, which this hardware cannot promise
# (wake-on-network is impossible on it and FileVault blocks unattended reboot);
# 95% would have let the 4.3-hour July 10 outage pass unremarked.
SLO_WINDOW_COVERAGE_PCT = 98.0

# 22 trading days x 570 window-minutes. Used ONLY to express the SLO as an
# allowance in minutes, which is the unit an operator can act on; the coverage
# figure itself is always computed from the real span.
_NOMINAL_MONTH_WINDOW_MINUTES = 22 * 570

_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+(.*)$")

# Probe messages embed a pid and an attempt counter. Keying the tally on the raw
# string fragments every restart into its own entry with a count of 1, which is
# how "the server was restarted seven times" renders as seven unrelated events.
# Found on the real log; the fixtures had no pids in them.
_DIGITS_RE = re.compile(r"\d+")

# The serving-probe vocabulary, mirroring SERVING_ERROR_MARKER and
# STALE_CODE_MARKER in scripts/cairniq_watchdog.py. Those strings are a contract
# between the writer and this reader and must change together.
#
# Matched against the RAW status text, before `_DIGITS_RE` runs — digit
# normalisation turns "HTTP 500" into "HTTP <n>", which is right for grouping
# restarts by message and would destroy the status code this axis is made of.
_SERVING_ERROR_RE = re.compile(r"SERVING-ERRORS HTTP (\d{3})")
_STALE_CODE_MARKER = "STALE-CODE:"
_NOT_SERVING_MARKER = "did not complete an HTTP exchange"

# The HEALTHY serving line, `ok HTTP 200`. Without it the serving probe could only
# ever prove itself by FAILING: a bare `ok` is what the old tcp-only probe wrote,
# so a healthy host's log carried no evidence a request had been made and
# `serving_probe_active` stayed False while the probe ran perfectly every two
# minutes. Found by deploying it. It also must be excluded from the unhealthy
# tally, or the healthy majority lands in `unhealthy_probes` as `ok HTTP <n>`.
_HEALTHY_RE = re.compile(r"^ok HTTP (\d{3})$")


def probe_log_path() -> str:
    """Absolute path of the watchdog probe log.

    Not per-profile: the host is either up or it is not, and availability is a
    fact about the machine rather than about a book. Contrast 4.10a's store,
    which is per-profile for exactly the opposite reason.
    """
    from agent.logger import LOG_BASE_DIR

    return os.path.join(LOG_BASE_DIR, _PROBE_LOG)


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------
def is_trading_day(day: date) -> bool:
    """Whether the US cash market is open on `day`, per the authored table.

    A date whose year the table does not cover is treated as a trading day if it
    is a weekday — the conservative direction, since it can only make measured
    coverage look worse. `days_outside_holiday_table` reports how much of the
    window rests on that assumption so the figure is never quoted as exact when
    it is a bound.
    """
    return day.weekday() < 5 and day.isoformat() not in MARKET_HOLIDAYS


def _covered_by_holiday_table(day: date) -> bool:
    return day.year in HOLIDAY_TABLE_YEARS


def _day_span(start: datetime, end: datetime):
    """Yield (day_date, chunk_start, chunk_end) for each calendar day in [start, end)."""
    cur = start
    while cur < end:
        midnight = datetime.combine(cur.date(), dtime.max)
        chunk_end = min(end, midnight)
        yield cur.date(), cur, chunk_end
        cur = midnight + timedelta(microseconds=1)


def window_minutes(start: datetime, end: datetime) -> float:
    """Minutes of [start, end) falling inside the coverage window on trading days."""
    total = 0.0
    for day, chunk_start, chunk_end in _day_span(start, end):
        if not is_trading_day(day):
            continue
        lo = max(chunk_start, datetime.combine(day, WINDOW_OPEN))
        hi = min(chunk_end, datetime.combine(day, WINDOW_CLOSE))
        if hi > lo:
            total += (hi - lo).total_seconds() / 60.0
    return total


def _uncovered_window_minutes(start: datetime, end: datetime) -> float:
    """Window minutes in [start, end) whose year the holiday table does not cover."""
    total = 0.0
    for day, chunk_start, chunk_end in _day_span(start, end):
        if _covered_by_holiday_table(day) or not is_trading_day(day):
            continue
        lo = max(chunk_start, datetime.combine(day, WINDOW_OPEN))
        hi = min(chunk_end, datetime.combine(day, WINDOW_CLOSE))
        if hi > lo:
            total += (hi - lo).total_seconds() / 60.0
    return total


# ---------------------------------------------------------------------------
# Probe log
# ---------------------------------------------------------------------------
def _read_probes_with_stats(
    path: str | None = None,
) -> tuple[list[tuple[datetime, str]], int]:
    """`(probes, duplicates_dropped)` — the reader `read_probes` wraps.

    Split out so the duplicate tally reaches the report as a return value rather
    than as state hanging off the module, which a concurrent read would corrupt.
    """
    path = path or probe_log_path()
    if not os.path.exists(path):
        return [], 0
    out: list[tuple[datetime, str]] = []
    seen: set[tuple[datetime, str]] = set()
    duplicates = 0
    with open(path, errors="replace", encoding="utf-8") as fh:
        for line in fh:
            m = _LINE_RE.match(line.strip())
            if not m:
                continue
            try:
                probe = (datetime.fromisoformat(m.group(1)), m.group(2).strip())
            except ValueError:
                continue
            if probe in seen:
                duplicates += 1
                continue
            seen.add(probe)
            out.append(probe)
    out.sort(key=lambda pair: pair[0])
    return out, duplicates


@log_exceptions()
def read_probes(path: str | None = None) -> list[tuple[datetime, str]]:
    """Every DISTINCT parseable probe line as (stamp, status), oldest first.

    Unparseable lines are dropped rather than guessed at. Never raises: an
    observability layer that breaks on a malformed log is not one.

    Exact `(stamp, status)` repeats are collapsed, because on the real log they
    are not probes — they are a second writer. Two watchdog instances appended to
    this file from 06-27 to 07-18 and the ratio of lines to distinct lines was
    2.00 to two decimal places on every one of those days, ending at the minute
    of the 07-18 kickstart. At second resolution against a 2-minute cadence, a
    genuine pair of probes cannot share a timestamp, so a repeat is always the
    duplicate writer.

    Coverage never depended on this — a 0-minute interval is below any gap
    threshold — but two figures did: `probes` is what 2.6 reports as production,
    and it read 2x high for two thirds of the record, while `probe_cadence_minutes`
    reported 1.38 for a probe that runs every 2 minutes. `duplicate_probe_lines`
    keeps the second writer visible on the report instead of smoothing it out;
    quietly correcting a figure and discarding the evidence of why is the same
    move this module exists to refuse.
    """
    return _read_probes_with_stats(path)[0]


def find_gaps(
    stamps: list[datetime], threshold_minutes: float = GAP_THRESHOLD_MINUTES
) -> list[dict[str, Any]]:
    """Intervals between consecutive probes longer than `threshold_minutes`.

    A gap means the probe stopped reporting: the host slept, lost power, or the
    watchdog itself died. All three are unavailability from the push layer's
    point of view, and this module deliberately does not try to tell them apart —
    naming a cause it cannot observe is the fabrication 4.10a's `unclassified`
    contract exists to prevent, and the same rule applies here.
    """
    gaps: list[dict[str, Any]] = []
    for prev, cur in zip(stamps, stamps[1:]):
        minutes = (cur - prev).total_seconds() / 60.0
        if minutes <= threshold_minutes:
            continue
        gaps.append({
            "start": prev.isoformat(timespec="seconds"),
            "end": cur.isoformat(timespec="seconds"),
            "hours": round(minutes / 60.0, 2),
            "window_minutes_lost": round(window_minutes(prev, cur), 1),
        })
    return gaps


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------
@log_exceptions()
def measure_availability(
    path: str | None = None, threshold_minutes: float = GAP_THRESHOLD_MINUTES
) -> dict[str, Any]:
    """7.1 Step 1: coverage of the trading window, and every gap behind it.

    Read `status` first:
      * ``no_data``   — the probe log is absent or unparseable. NOT 100% uptime.
      * ``measured``  — `window_coverage_pct` is the figure; `gaps` is the why.

    `window_coverage_pct` is an UPPER BOUND on availability, not an estimate of
    it, and `open_measurements` says why: the probe answers "is something
    listening on :8000", so a surface returning 5xx to every request scores as
    fully available. That is not a hypothetical — it is a recorded five-hour
    outage in which the probe never once reported anything but `ok`.
    """
    probes, duplicates = _read_probes_with_stats(path)
    if len(probes) < 2:
        return {
            "status": "no_data",
            "probes": len(probes),
            "duplicate_probe_lines": duplicates,
            "note": (
                "The watchdog probe log holds fewer than two readable lines, so no "
                "interval can be measured. This is an ABSENT instrument, not a "
                "perfect record — do not read it as 100% availability."
            ),
            "open_measurements": _open_measurements(),
        }

    stamps = [p[0] for p in probes]
    first, last = stamps[0], stamps[-1]
    span_s = (last - first).total_seconds()

    gaps = find_gaps(stamps, threshold_minutes)
    downtime_s = sum(
        (datetime.fromisoformat(g["end"]) - datetime.fromisoformat(g["start"])).total_seconds()
        for g in gaps
    )

    window_total = window_minutes(first, last)
    window_lost = sum(g["window_minutes_lost"] for g in gaps)
    outside_table = _uncovered_window_minutes(first, last)

    statuses: dict[str, int] = {}
    for _stamp, status in probes:
        # Both spellings of healthy: the old bare `ok` and the serving probe's
        # `ok HTTP 200`. Letting the latter through would file the healthy
        # majority under `unhealthy_probes` as a single huge `ok HTTP <n>` entry.
        if status == "ok" or _HEALTHY_RE.match(status):
            continue
        key = _DIGITS_RE.sub("<n>", status)
        statuses[key] = statuses.get(key, 0) + 1
    unhealthy = dict(sorted(statuses.items(), key=lambda kv: kv[1], reverse=True))

    in_window = [g for g in gaps if g["window_minutes_lost"] > 0]

    coverage = (round(100.0 * (1 - window_lost / window_total), 2)
                if window_total else None)
    serving = _measure_serving(probes)

    return {
        "status": "measured",
        "first_probe": first.isoformat(timespec="seconds"),
        "last_probe": last.isoformat(timespec="seconds"),
        "span_days": round(span_s / 86400.0, 2),
        "probes": len(probes),
        # Non-zero means a second writer appended to this log. It does not move
        # coverage, and it did move `probes` and `probe_cadence_minutes` — see
        # `read_probes`.
        "duplicate_probe_lines": duplicates,
        "probe_cadence_minutes": round(span_s / 60.0 / max(1, len(probes) - 1), 2),
        # Wall clock, reported for completeness and explicitly NOT the headline.
        "wall_clock_uptime_pct": round(100.0 * (1 - downtime_s / span_s), 2) if span_s else None,
        "downtime_hours": round(downtime_s / 3600.0, 2),
        # The figure that means something for Theme 3.
        "window": f"{WINDOW_OPEN.isoformat(timespec='minutes')}-"
                  f"{WINDOW_CLOSE.isoformat(timespec='minutes')} local, trading days",
        "window_minutes_total": round(window_total, 1),
        "window_minutes_lost": round(window_lost, 1),
        "window_coverage_pct": coverage,
        "gaps": gaps,
        "gap_count": len(gaps),
        "gaps_in_window": len(in_window),
        "gap_threshold_minutes": threshold_minutes,
        "unhealthy_probes": unhealthy,
        # Whether the trading-day denominator rests on the authored table or on
        # the weekday fallback. Non-zero means the coverage figure is a bound.
        "holiday_table_years": sorted(HOLIDAY_TABLE_YEARS),
        "window_minutes_outside_holiday_table": round(outside_table, 1),
        **evaluate_slo(coverage),
        **serving,
        "open_measurements": _open_measurements(serving),
    }


def _measure_serving(
    probes: list[tuple[datetime, str]]
) -> dict[str, Any]:
    """The 5xx axis (7.1 Step 3), read from the serving probe's vocabulary.

    Before the watchdog issued an actual request there was nothing on disk to
    read here, which is why `window_coverage_pct` shipped as an upper bound: a
    half-deploy served 500s for five hours and every probe reported `ok`.

    `serving_probe_active` is the load-bearing field, and it is what keeps this
    honest across the boundary. The probe log reaches back weeks BEFORE the
    serving probe existed, so most of any current record has no 5xx information
    in it at all — and zero errors found in a stretch nobody was looking at is
    not a clean bill of health. `first_serving_probe` dates the boundary so the
    figure is only ever quoted over the span it actually covers.
    """
    error_windows: list[dict[str, Any]] = []
    codes: dict[str, int] = {}
    stale_code_probes = 0
    not_serving_probes = 0
    ok_probes = 0
    first_seen: datetime | None = None

    for stamp, status in probes:
        healthy = _HEALTHY_RE.match(status)
        if healthy:
            ok_probes += 1
        m = _SERVING_ERROR_RE.search(status)
        if m:
            codes[m.group(1)] = codes.get(m.group(1), 0) + 1
            error_windows.append({
                "at": stamp.isoformat(timespec="seconds"),
                "code": int(m.group(1)),
                "in_window": window_minutes(stamp, stamp + timedelta(minutes=1)) > 0,
            })
        if _STALE_CODE_MARKER in status:
            stale_code_probes += 1
        if _NOT_SERVING_MARKER in status:
            not_serving_probes += 1
        # Any marker dates the boundary — INCLUDING the healthy one, which is the
        # whole reason `ok HTTP 200` exists. A bare `ok` still cannot, because the
        # old tcp-only probe wrote that too, so it proves nothing was requested.
        if first_seen is None and (healthy or m or _STALE_CODE_MARKER in status
                                  or _NOT_SERVING_MARKER in status):
            first_seen = stamp

    return {
        "serving_probe_active": first_seen is not None,
        "first_serving_probe": first_seen.isoformat(timespec="seconds") if first_seen else None,
        "serving_error_probes": len(error_windows),
        "serving_error_codes": dict(sorted(codes.items())),
        "serving_errors_in_window": sum(1 for e in error_windows if e["in_window"]),
        "serving_error_sightings": error_windows[:50],
        "stale_code_probes": stale_code_probes,
        "not_serving_probes": not_serving_probes,
        # Confirmed-serving probes. This is the count that makes a clean 5xx record
        # mean something — without it, "no errors" is indistinguishable from
        # "nothing was ever asked".
        "serving_ok_probes": ok_probes,
    }


def evaluate_slo(coverage_pct: float | None) -> dict[str, Any]:
    """Judge a measured coverage figure against the stated SLO (7.1 Step 2).

    The asymmetry here is the whole point and it follows from coverage being an
    UPPER bound rather than an estimate. A bound below the bar is a DEFINITE
    breach — the true figure can only be lower. A bound above the bar proves
    nothing on its own, because the unmeasured 5xx time sits underneath it. So
    this reports `breached` / `not_proven_met`, and never "met", until the serving
    probe has covered the span.

    Reported, not alerted. Step 2's purpose is to let Step 3 be judged against a
    number written down beforehand, and an SLO that pages is a separate decision.
    """
    if coverage_pct is None:
        return {
            "slo_target_pct": SLO_WINDOW_COVERAGE_PCT,
            "slo_state": "unknown",
            "slo_note": "no coverage figure to judge — the instrument is absent, not passing.",
        }

    allowed = "at most %.0f window-minutes lost per 22 trading days" % (
        _NOMINAL_MONTH_WINDOW_MINUTES * (1 - SLO_WINDOW_COVERAGE_PCT / 100.0)
    )
    if coverage_pct < SLO_WINDOW_COVERAGE_PCT:
        return {
            "slo_target_pct": SLO_WINDOW_COVERAGE_PCT,
            "slo_state": "breached",
            "slo_note": (
                f"{coverage_pct}% is below the {SLO_WINDOW_COVERAGE_PCT}% floor "
                f"({allowed}). This is DEFINITE: coverage is an upper bound, so the "
                f"true figure can only be lower."
            ),
        }
    return {
        "slo_target_pct": SLO_WINDOW_COVERAGE_PCT,
        "slo_state": "not_proven_met",
        "slo_note": (
            f"{coverage_pct}% clears the {SLO_WINDOW_COVERAGE_PCT}% floor ({allowed}), "
            f"but coverage is an upper bound and the 5xx axis is not covered for the "
            f"whole span — so this is 'not breached', not 'met'."
        ),
    }


def _open_measurements(serving: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """7.1's four numbers, and the honest state of each.

    Kept as data on every report rather than as prose in a docstring, because the
    thing this file gets wrong repeatedly is a figure quoted without the caveat
    that was written next to it.

    The 5xx row is computed from `serving` rather than authored, because its state
    genuinely changed when Step 3 shipped the emitter — and it changes back to a
    caveat for any span predating the emitter. Hardcoding "MEASURED" here the day
    the probe landed would have quietly claimed coverage of the weeks before it.
    """
    if serving and serving.get("serving_probe_active"):
        fivexx = {
            "number": "surfaces returning 5xx",
            "state": "MEASURED — from the serving probe",
            "how": (
                f"the watchdog GETs /api/health and records the status code. Active "
                f"since {serving.get('first_serving_probe')}; "
                f"{serving.get('serving_ok_probes')} probe(s) confirmed a serving "
                f"surface and {serving.get('serving_error_probes')} saw a 5xx, "
                f"{serving.get('serving_errors_in_window')} of them inside the window. "
                f"Any span BEFORE that date has no 5xx information in it, and zero "
                f"errors found where nobody was looking is not a clean record."
            ),
        }
    else:
        fivexx = {
            "number": "surfaces returning 5xx",
            "state": "NOT MEASURED — emitter shipped, no data yet",
            "how": (
                "the watchdog now GETs /api/health and records the status code, but "
                "this log has no serving-probe line in it — so this span predates the "
                "emitter, or the new watchdog is not deployed. Until a marker appears, "
                "coverage stays an upper bound: the old probe only checked that :8000 "
                "was bound, and a half-deploy served 500s for five hours while every "
                "probe reported ok."
            ),
        }
    return [
        {
            "number": "process availability",
            "state": "MEASURED",
            "how": "gaps in the watchdog probe, ~2-minute resolution",
        },
        fivexx,
        {
            "number": "alerts that should have fired and did not",
            "state": "MEASURED PER-PROFILE — not on this report",
            "how": (
                "`window_minutes_lost` and the dated `gaps` here are the EXPOSURE, and "
                "they are all this report will ever carry. The replay that turns them "
                "into a count of missed crossings lives in `tools.missed_alerts`, "
                "reads THESE gaps, and is exposed per-profile at "
                "`GET /api/alerts/missed` — a watch condition belongs to one person "
                "and this report is deliberately global, exactly as with delivery "
                "latency below. Read `measurable_windows` against `windows` there "
                "before reading the count: a bar interval wider than the gap cannot "
                "see inside it, and such a window is reported UNMEASURABLE rather "
                "than clean."
            ),
        },
        {
            "number": "delivery latency (produced-at vs read-at)",
            "state": "MEASURED PER-PROFILE — not on this report",
            "how": (
                "`read_at` and `read_via` are now stamped when an alert is marked "
                "read, so latency is `read_at - ts`. It is NOT folded in here: an "
                "inbox belongs to one person and this report is deliberately global, "
                "so a single number across profiles would mean nothing. Read it from "
                "`GET /api/alerts/delivery` (tools.alerts.get_delivery_latency). A "
                "mark-all click is counted and never timed, and alerts read before "
                "the stamp existed are reported as unmeasurable rather than dropped."
            ),
        },
    ]


@log_exceptions()
def get_availability_report(path: str | None = None) -> dict[str, Any]:
    """Read surface for the endpoint and the 2.6 production report.

    Adds a plain-language `summary` to `measure_availability()` and nothing else,
    so the two can never disagree about a number.
    """
    report = measure_availability(path)
    if report.get("status") != "measured":
        return report

    coverage = report.get("window_coverage_pct")
    lost = report.get("window_minutes_lost") or 0
    in_window = report.get("gaps_in_window") or 0
    verdict = {
        "breached": f"BELOW the {SLO_WINDOW_COVERAGE_PCT}% floor",
        "not_proven_met": f"not below the {SLO_WINDOW_COVERAGE_PCT}% floor",
    }.get(report.get("slo_state", ""), "unjudged")
    bound = (
        "This is an upper bound for the span before the serving probe: it cannot "
        "see a surface that is up and returning errors."
        if not report.get("serving_probe_active")
        else f"The 5xx axis is covered from {report.get('first_serving_probe')} onward."
    )
    report["summary"] = (
        f"{coverage}% of the trading-day coverage window was covered over "
        f"{report['span_days']} days — {lost:.0f} window-minute(s) missed across "
        f"{in_window} incident(s), {verdict}. {bound}"
    )
    return report


if __name__ == "__main__":  # pragma: no cover — operator convenience
    import json

    print(json.dumps(get_availability_report(), indent=2))
