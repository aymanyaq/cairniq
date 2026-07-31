"""4.10 — benchmark-relative attribution, time-weighted, and the four gates it fails today.

Nothing in this app has ever graded the PORTFOLIO. It grades positions, calls,
theses and scans; the one number a household actually lives on — *did the book
beat a benchmark matching what it holds* — has no producer. This module is that
producer, and most of what is in it is the machinery for refusing to answer.

**Why the refusals are the substance.** A percent return is already on
`portfolio_history.csv`, computed as value-over-cost-basis, and it is not a
return in the sense anyone means: it moves when a deposit lands and it moves
when a position is sold at a gain. Replacing it with a chain-linked TWR that
quietly treats a $10,000 contribution as performance would be strictly worse than
the current number, because it would LOOK correct. So this module computes TWR
only when it can prove four things, and names which one failed otherwise:

  1. **Coverage, not span.** ``goal_projection._history_span_days`` returns
     ``(max - min).days``, the distance between two endpoints, and is blind to
     holes between them. That gate opens on schedule with ~8% of its days never
     recorded and nothing says so. TWR chain-links BETWEEN observations, so a
     hole is not neutral — the link spans straight across it, and a hole
     containing a deposit is exactly the corruption TWR exists to prevent.
     ``coverage()`` reports rows-out-of-days and the gate reads that.

  2. **Every flow named.** 4.10a records what moved; 4.10a's classification step
     records why, stated by a human, never inferred. One unclassified delta
     inside the window blocks the number — it could be a deposit or it could be
     return, and a series cannot chain-link through a maybe.

  3. **Every flow PRICED.** The classification store holds quantities: shares for
     a security, currency units for cash. TWR needs money, in base currency, on
     the flow's own date. `amount_base` is where that comes from and it is often
     absent, so this is a separate gate from (2) — a window can be fully
     classified and still unusable, and reporting `complete` alone would call it
     ready.

  4. **A valuation ON each flow date.** TWR breaks the series at every flow, so a
     flow on the 14th needs a portfolio value on the 14th. A monthly series does
     not satisfy this and neither does a daily series with a hole where the flow
     landed.

**The benchmark's weights are an assumption and are stamped as one.** A blended
CAD/US benchmark has to be weighted by the book's actual currency mix, and the
only mix this app can read is TODAY's. Applying it backwards across a year in
which the mix changed is a modelling choice, not a measurement, so
`weights_as_of` and `weights_basis` ride on every payload and the weights can be
overridden by a caller who knows better.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from tools.exception_logger import log_exceptions

# AUTHORED CONSTANT (2.7). The window a benchmark-relative number is quoted over.
# One year, because that is the shortest span over which tracking error means
# anything and the span the roadmap sized this item against.
DEFAULT_WINDOW_DAYS = 365

# AUTHORED CONSTANT. The fraction of calendar days inside the window that must
# actually carry a snapshot. Not a measured figure and not a statistical
# threshold — it is the point at which chain-linking stops being a description of
# the series and starts being an interpolation over it. Stated here so a reader
# can disagree with the number rather than with an invisible default.
MIN_COVERAGE_PCT = 90.0

# The blended benchmark's legs. A book held in CAD and USD is not measured
# against either index alone: XIC is the broad Canadian market, SPY the US one.
# Both are price series; see `benchmark_note` on what that excludes.
BENCHMARK_LEGS: dict[str, str] = {"CAD": "XIC.TO", "USD": "SPY"}


# ---------------------------------------------------------------------------
# Coverage — rows out of days, which is the figure the 365-day gate is missing
# ---------------------------------------------------------------------------
@log_exceptions()
def coverage(dates: list[str], window_days: int = DEFAULT_WINDOW_DAYS,
             as_of: date | None = None) -> dict[str, Any]:
    """Observed days against the calendar window, and every hole inside it.

    Deliberately NOT `(max - min).days`. That figure is what
    `goal_projection._history_span_days` computes, and two series with the same
    span can differ by 8% of their days — the live one does. `sufficient` is the
    gate, and it reads `coverage_pct`, never the span.
    """
    as_of = as_of or date.today()
    window_start = as_of - timedelta(days=window_days - 1)

    parsed = sorted({d for d in (_as_date(x) for x in dates) if d is not None
                     and window_start <= d <= as_of})

    if not parsed:
        return {
            "observed_days": 0,
            "window_days": window_days,
            "window_start": window_start.isoformat(),
            "window_end": as_of.isoformat(),
            "coverage_pct": 0.0,
            "span_days": 0,
            "missing_days": window_days,
            "gaps": [],
            "sufficient": False,
            "min_coverage_pct": MIN_COVERAGE_PCT,
            "note": "No portfolio valuation falls inside the window.",
        }

    gaps = [
        {"after": parsed[i - 1].isoformat(), "before": parsed[i].isoformat(),
         "missing_days": (parsed[i] - parsed[i - 1]).days - 1}
        for i in range(1, len(parsed))
        if (parsed[i] - parsed[i - 1]).days > 1
    ]
    coverage_pct = round(100.0 * len(parsed) / window_days, 2)
    span = (parsed[-1] - parsed[0]).days + 1

    return {
        "observed_days": len(parsed),
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "window_end": as_of.isoformat(),
        "first_observed": parsed[0].isoformat(),
        "last_observed": parsed[-1].isoformat(),
        "coverage_pct": coverage_pct,
        # Reported alongside, and labelled, so the two are never confused again.
        "span_days": span,
        "span_minus_coverage_days": span - len(parsed),
        "missing_days": window_days - len(parsed),
        "gaps": gaps,
        "gap_count": len(gaps),
        "sufficient": coverage_pct >= MIN_COVERAGE_PCT,
        "min_coverage_pct": MIN_COVERAGE_PCT,
        "note": (
            f"{len(parsed)} of {window_days} calendar days carry a valuation "
            f"({coverage_pct}%). The recorded SPAN is {span} days, which is "
            f"{span - len(parsed)} day(s) wider than the coverage — a span figure "
            f"cannot see a hole between its endpoints, and a chain-linked return "
            f"steps straight across one."
        ),
    }


def _as_date(value: Any) -> date | None:
    """Parse a date from a string, a date, or a pandas Timestamp. None on failure."""
    if isinstance(value, date):
        return value
    text = str(value or "")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# The chain-linked return
# ---------------------------------------------------------------------------
@log_exceptions()
def chain_link(valuations: list[tuple[str, float]],
               flows: list[tuple[str, float]]) -> dict[str, Any]:
    """Time-weighted return over `valuations`, with `flows` removed.

    `valuations` are `(date, value)` in base currency, and `flows` are
    `(date, amount)` where a deposit is POSITIVE and a withdrawal negative.

    Each sub-period runs between consecutive valuations and its return is
    ``(V_end - F) / V_start - 1``, where `F` is the net flow dated at the END of
    the sub-period. That END convention is a choice — a flow on the morning of
    the 14th is really a beginning-of-period flow for the 14th→15th sub-period —
    and it is the conservative one here because the store dates a change by the
    snapshot that FIRST SHOWED it, which is the close of the day it appeared.
    `flow_convention` is on the payload rather than in this docstring alone.

    Returns an `error` rather than a number if any sub-period starts from a
    non-positive value: a portfolio that went to zero and was refunded has no
    meaningful chain link across that boundary, and dividing by it would produce
    a spectacular return out of an accounting artefact.
    """
    valuations = sorted(((d, float(v)) for d, v in valuations), key=lambda p: p[0])
    if len(valuations) < 2:
        return {"error": "at least two valuations are needed to chain-link a return"}

    by_date: dict[str, float] = {}
    for d, amount in flows:
        by_date[str(d)[:10]] = by_date.get(str(d)[:10], 0.0) + float(amount)

    periods: list[dict[str, Any]] = []
    growth = 1.0
    for (d0, v0), (d1, v1) in zip(valuations, valuations[1:]):
        if v0 <= 0:
            return {
                "error": f"sub-period starting {d0} opens at {v0}, which cannot be "
                         "chain-linked through",
            }
        flow = by_date.get(str(d1)[:10], 0.0)
        r = (v1 - flow) / v0 - 1.0
        growth *= (1.0 + r)
        periods.append({
            "from": str(d0)[:10], "to": str(d1)[:10],
            "start_value": round(v0, 2), "end_value": round(v1, 2),
            "flow": round(flow, 2), "return_pct": round(r * 100, 6),
        })

    total = (growth - 1.0) * 100.0
    days = (_as_date(valuations[-1][0]) - _as_date(valuations[0][0])).days or 1
    annualized = ((growth ** (365.0 / days)) - 1.0) * 100.0 if growth > 0 else None

    net_flow = round(sum(by_date.values()), 2)
    return {
        "twr_pct": round(total, 4),
        "annualized_pct": round(annualized, 4) if annualized is not None else None,
        "sub_periods": len(periods),
        "periods": periods,
        "net_external_flow": net_flow,
        "first_date": str(valuations[0][0])[:10],
        "last_date": str(valuations[-1][0])[:10],
        "days": days,
        "flow_convention": (
            "A flow is applied at the END of the sub-period it is dated in: "
            "r = (V_end - flow) / V_start - 1. The reconciliation store dates a "
            "change by the snapshot that first showed it, which is a close, so an "
            "end-of-period convention matches how the flow was observed."
        ),
    }


# ---------------------------------------------------------------------------
# Flows, priced, from the classification store
# ---------------------------------------------------------------------------
@log_exceptions()
def window_flows(window_start: date, window_end: date) -> dict[str, Any]:
    """Every classified external flow inside the window, and what still blocks it.

    Walks the WHOLE reconciliation series rather than only the latest pair —
    `get_reconciliation` reports the most recent two snapshots, which is the right
    read surface for "what moved yesterday" and useless for a year of return.

    Three refusals live here, and each is a separate `blocked_by` reason:
      * ``unclassified`` — a delta nobody has named. Might be a flow.
      * ``unpriced``     — named, but with no `amount_base`, so it cannot be
        removed from the return in money.
      * neither          — `flows` is complete for the window.
    """
    from tools.portfolio_classification import apply_classifications
    from tools.portfolio_reconciliation import detect_changes, read_history, snapshot_dates

    rows = read_history()
    dates = [d for d in snapshot_dates(rows)
             if (dt := _as_date(d)) and window_start <= dt <= window_end]

    if len(dates) < 2:
        return {"flows": [], "unclassified": [], "unpriced": [],
                "snapshots_in_window": len(dates),
                "blocked_by": "no_position_history",
                "note": (
                    "Fewer than two position snapshots fall inside this window, so "
                    "no change can be observed — and an unobserved change is not "
                    "an absent one. 4.10a's recorder is the source; check that its "
                    "`position_snapshot` scheduler task is running."
                )}

    changes: list[dict[str, Any]] = []
    for prior, current in zip(dates, dates[1:]):
        changes.extend(detect_changes(prior, current, rows))

    annotated = apply_classifications(changes)
    unclassified = [c for c in annotated if not c.get("classified")]
    external = [c for c in annotated if c.get("is_external_flow")]
    unpriced = [c for c in external if c.get("amount_base") is None]

    blocked_by = None
    if unclassified:
        blocked_by = "unclassified_changes"
    elif unpriced:
        blocked_by = "unpriced_flows"

    return {
        "flows": [{"date": c["current_date"], "amount_base": c.get("amount_base"),
                   "cause": c.get("cause"), "symbol": c.get("symbol"),
                   "account": c.get("account")}
                  for c in external if c.get("amount_base") is not None],
        "unclassified": [{"date": c["current_date"], "symbol": c.get("symbol"),
                          "account": c.get("account"), "delta": c.get("delta")}
                         for c in unclassified[:50]],
        "unpriced": [{"date": c["current_date"], "symbol": c.get("symbol"),
                      "cause": c.get("cause"), "delta": c.get("delta")}
                     for c in unpriced[:50]],
        "changes_seen": len(annotated),
        "unclassified_count": len(unclassified),
        "unpriced_count": len(unpriced),
        "snapshots_in_window": len(dates),
        "blocked_by": blocked_by,
    }


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------
@log_exceptions()
def currency_weights(holdings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The book's CAD/USD split, and an explicit statement that it is point-in-time.

    Weighted by market value where a value is present and by position count
    otherwise — and the payload says which, because a count-weighted mix on a book
    with one huge US position and nine small Canadian ones is not an approximation
    of the value-weighted one, it is a different answer.
    """
    if holdings is None:
        from tools.portfolio_csv import load_portfolio

        holdings = load_portfolio()
    if not isinstance(holdings, list):
        return {"weights": {}, "basis": "unavailable",
                "note": "The portfolio could not be read, so no currency mix is known."}

    by_currency: dict[str, float] = {}
    counts: dict[str, int] = {}
    valued = 0
    for h in holdings:
        if not isinstance(h, dict) or "_sync_errors" in h:
            continue
        cur = str(h.get("currency") or "").upper().strip() or "USD"
        counts[cur] = counts.get(cur, 0) + 1
        value = h.get("market_value") or h.get("value") or h.get("current_value")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value and value > 0:
            by_currency[cur] = by_currency.get(cur, 0.0) + value
            valued += 1

    if by_currency and valued == sum(counts.values()):
        total = sum(by_currency.values())
        weights = {k: round(v / total, 6) for k, v in by_currency.items()}
        basis = "market value"
    elif counts:
        total = sum(counts.values())
        weights = {k: round(v / total, 6) for k, v in counts.items()}
        basis = "position count"
    else:
        return {"weights": {}, "basis": "unavailable",
                "note": "No readable holding carries a currency."}

    return {
        "weights": weights,
        "basis": basis,
        "positions": sum(counts.values()),
        "positions_valued": valued,
        "as_of": date.today().isoformat(),
        "note": (
            f"Weighted by {basis}. This is TODAY's mix. Applying it across a window "
            "in which the mix changed is a modelling choice, not a measurement — "
            "override `weights` if the book was materially different earlier in the "
            "window."
            + ("" if basis == "market value" else
               " No market value was available for every holding, so this is a "
               "COUNT-weighted mix: one large position and nine small ones weigh "
               "the same here, which the value-weighted answer would not.")
        ),
    }


@log_exceptions()
def blended_benchmark(window_start: date, window_end: date,
                      weights: dict[str, float] | None = None,
                      series_fn: Callable[[str, date, date], list[tuple[str, float]]] | None = None,
                      ) -> dict[str, Any]:
    """A benchmark weighted to the book's actual currency mix.

    Each leg's return is its own price series over the window; the blend is the
    weighted sum. A currency with no mapped leg is reported in `uncovered` and its
    weight is EXCLUDED from the blend and named — silently renormalising over the
    covered legs would present a two-currency benchmark as if it covered three.
    """
    weight_source = "supplied"
    if weights is None:
        cw = currency_weights()
        weights = cw.get("weights") or {}
        weight_source = cw.get("basis", "unavailable")
    if not weights:
        return {"status": "no_weights",
                "note": "No currency mix is known, so no blend can be built."}

    fetch = series_fn or _default_series
    legs: list[dict[str, Any]] = []
    uncovered: dict[str, float] = {}
    unavailable: list[str] = []

    for currency, weight in weights.items():
        symbol = BENCHMARK_LEGS.get(currency)
        if not symbol:
            uncovered[currency] = weight
            continue
        series = fetch(symbol, window_start, window_end)
        if not series or len(series) < 2:
            unavailable.append(symbol)
            continue
        start_px, end_px = float(series[0][1]), float(series[-1][1])
        if start_px <= 0:
            unavailable.append(symbol)
            continue
        legs.append({
            "currency": currency, "symbol": symbol, "weight": weight,
            "return_pct": round((end_px / start_px - 1.0) * 100, 4),
            "first": series[0][0], "last": series[-1][0], "points": len(series),
            "series": series,
        })

    if not legs:
        return {"status": "no_legs", "uncovered": uncovered,
                "unavailable": unavailable,
                "note": "No benchmark leg could be priced over this window."}

    covered_weight = sum(leg["weight"] for leg in legs)
    blended = sum(leg["return_pct"] * leg["weight"] for leg in legs) / covered_weight

    return {
        "status": "measured" if not uncovered and not unavailable else "partial",
        "return_pct": round(blended, 4),
        "legs": [{k: v for k, v in leg.items() if k != "series"} for leg in legs],
        "covered_weight": round(covered_weight, 6),
        "uncovered_currencies": uncovered,
        "unavailable_legs": unavailable,
        "weights_basis": weight_source,
        "benchmark_note": (
            "Legs are PRICE series, so distributions are excluded from the "
            "benchmark's return while the portfolio's TWR includes reinvested "
            "income. That biases alpha UPWARD by roughly the benchmark's yield. "
            + (f"{round((1 - covered_weight) * 100, 2)}% of the book's currency "
               f"weight has no mapped index and is excluded rather than "
               f"renormalised away: {sorted(uncovered)}." if uncovered else "")
        ),
        "_legs_with_series": legs,
    }


def _default_series(symbol: str, start: date, end: date) -> list[tuple[str, float]]:
    """Daily closes for one benchmark leg. Network; injected away in tests."""
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(start=start.isoformat(),
                                         end=(end + timedelta(days=1)).isoformat())
        if hist is None or hist.empty:
            return []
        return [(idx.date().isoformat(), float(row["Close"]))
                for idx, row in hist.iterrows()]
    except Exception:  # noqa: BLE001 — an unreachable benchmark is a missing leg
        return []


# ---------------------------------------------------------------------------
# Tracking error
# ---------------------------------------------------------------------------
def tracking_error(portfolio_periods: list[dict[str, Any]],
                   benchmark_series: list[tuple[str, float]]) -> dict[str, Any]:
    """Annualized stdev of the return difference, over the dates BOTH cover.

    Aligned on the portfolio's sub-period boundaries, and it reports how many of
    them found a benchmark price. A tracking error computed over four of fifty-two
    aligned weeks is not a small-sample version of the right answer — `aligned`
    and `unaligned` are on the payload so the caller can refuse it.
    """
    prices = {str(d)[:10]: float(p) for d, p in benchmark_series}
    diffs: list[float] = []
    unaligned = 0

    for period in portfolio_periods:
        p0, p1 = prices.get(period["from"]), prices.get(period["to"])
        if not p0 or not p1 or p0 <= 0:
            unaligned += 1
            continue
        bench_r = (p1 / p0 - 1.0) * 100.0
        diffs.append(period["return_pct"] - bench_r)

    if len(diffs) < 2:
        return {"tracking_error_pct": None, "aligned": len(diffs), "unaligned": unaligned,
                "note": "Fewer than two sub-periods aligned to a benchmark price."}

    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    stdev = variance ** 0.5
    # Sub-periods here are day-to-day snapshots, so 252 is the annualisation
    # factor. Stated rather than assumed: a weekly series would need 52.
    return {
        "tracking_error_pct": round(stdev * (252 ** 0.5), 4),
        "period_stdev_pct": round(stdev, 6),
        "mean_excess_per_period_pct": round(mean, 6),
        "aligned": len(diffs),
        "unaligned": unaligned,
        "annualization_factor": 252,
        "note": (
            f"{len(diffs)} sub-period(s) aligned to a benchmark price; {unaligned} "
            "did not and are excluded. Annualized at sqrt(252), which assumes the "
            "sub-periods are daily."
        ),
    }


# ---------------------------------------------------------------------------
# Per-position attribution
# ---------------------------------------------------------------------------
@log_exceptions()
def position_attribution(window_start: date, window_end: date,
                         price_fn: Callable[[str, str], float | None] | None = None,
                         ) -> dict[str, Any]:
    """What each holding contributed, from recorded shares and dated prices.

    4.10a records SHARES per (account, symbol) per day and no values, so a
    contribution needs a price on the window's two endpoints. Where a price is
    missing the position is named in `unpriced` and EXCLUDED, and the whole report
    is marked `complete: False` — a contribution table that silently drops the two
    positions it could not price still sums to something, and that sum reads as
    the portfolio.
    """
    from tools.portfolio_reconciliation import is_cash, read_history, snapshot_dates

    rows = read_history()
    dates = [d for d in snapshot_dates(rows)
             if (dt := _as_date(d)) and window_start <= dt <= window_end]
    if len(dates) < 2:
        return {"status": "no_data", "snapshots_in_window": len(dates),
                "note": "Fewer than two position snapshots fall inside this window."}

    first, last = dates[0], dates[-1]
    start_rows = {(r.get("account"), r["symbol"]): r for r in rows if r["date"] == first}
    end_rows = {(r.get("account"), r["symbol"]): r for r in rows if r["date"] == last}
    fetch = price_fn or _default_price_on

    contributions: list[dict[str, Any]] = []
    unpriced: list[dict[str, Any]] = []
    start_total = end_total = 0.0

    for key in sorted(set(start_rows) | set(end_rows)):
        account, symbol = key
        if is_cash(symbol):
            continue
        s_shares = _float(start_rows.get(key, {}).get("shares")) or 0.0
        e_shares = _float(end_rows.get(key, {}).get("shares")) or 0.0
        p0, p1 = fetch(symbol, first), fetch(symbol, last)
        if p0 is None or p1 is None:
            unpriced.append({"symbol": symbol, "account": account,
                             "missing": [d for d, p in ((first, p0), (last, p1)) if p is None]})
            continue
        s_value, e_value = s_shares * p0, e_shares * p1
        start_total += s_value
        end_total += e_value
        contributions.append({
            "symbol": symbol, "account": account,
            "start_shares": s_shares, "end_shares": e_shares,
            "start_price": round(p0, 4), "end_price": round(p1, 4),
            "start_value": round(s_value, 2), "end_value": round(e_value, 2),
            "price_return_pct": round((p1 / p0 - 1.0) * 100, 4) if p0 > 0 else None,
        })

    if not contributions:
        return {"status": "unpriced", "unpriced": unpriced, "from": first, "to": last,
                "note": "No position could be priced on both endpoints of the window."}

    for c in contributions:
        # Contribution to the START-weighted return. Share changes inside the
        # window make this an approximation, and `share_change` marks the rows
        # where that bites rather than leaving the reader to notice.
        c["weight_start"] = round(c["start_value"] / start_total, 6) if start_total else None
        c["contribution_pct"] = (round((c["end_value"] - c["start_value"]) / start_total * 100, 4)
                                 if start_total else None)
        c["share_change"] = c["end_shares"] != c["start_shares"]

    contributions.sort(key=lambda c: (c["contribution_pct"] is None, -(c["contribution_pct"] or 0)))
    moved = [c["symbol"] for c in contributions if c["share_change"]]

    return {
        "status": "measured",
        "complete": not unpriced,
        "from": first, "to": last,
        "start_value": round(start_total, 2),
        "end_value": round(end_total, 2),
        "positions": contributions,
        "unpriced": unpriced,
        "positions_with_share_changes": moved,
        "note": (
            "Contribution is the change in each position's market value as a share "
            "of the book's starting value, priced from recorded share counts. "
            + (f"{len(unpriced)} position(s) could not be priced on both endpoints "
               f"and are EXCLUDED — the contributions below do not sum to the "
               f"portfolio's return." if unpriced else
               "Every position was priced on both endpoints.")
            + (f" {len(moved)} position(s) changed share count inside the window, so "
               f"their contribution mixes price return with the trade or flow that "
               f"changed the size: {moved[:10]}." if moved else "")
        ),
    }


def _default_price_on(symbol: str, on_date: str) -> float | None:
    """Close for `symbol` on or immediately before `on_date`. Network; injectable."""
    try:
        import yfinance as yf

        d = _as_date(on_date)
        if d is None:
            return None
        hist = yf.Ticker(symbol).history(start=(d - timedelta(days=7)).isoformat(),
                                         end=(d + timedelta(days=1)).isoformat())
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:  # noqa: BLE001 — a missing price is a named exclusion, not a crash
        return None


def _float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


# ---------------------------------------------------------------------------
# The read surface
# ---------------------------------------------------------------------------
@log_exceptions()
def get_attribution_report(window_days: int = DEFAULT_WINDOW_DAYS,
                           as_of: date | None = None,
                           weights: dict[str, float] | None = None,
                           series_fn: Callable | None = None,
                           price_fn: Callable | None = None,
                           history: list[tuple[str, float]] | None = None,
                           ) -> dict[str, Any]:
    """4.10's answer, or the named reason there isn't one.

    `status` is the field to read first and there are five:

      * ``no_history``          — no portfolio valuation series at all.
      * ``insufficient_coverage`` — the window is not densely enough recorded.
        Read `coverage.coverage_pct`, NOT `coverage.span_days`.
      * ``flows_incomplete``    — a change inside the window has no stated cause,
        or has one and no amount. `blocked_by` says which.
      * ``flow_date_unvalued``  — a flow landed on a day with no valuation, so the
        series cannot be broken at it.
      * ``measured``            — `twr_pct` is the number, and `alpha_pct` is it
        minus a benchmark blended to the book's currency mix.

    Every non-measured status is a REFUSAL with a named unblocker, not an error.
    """
    as_of = as_of or date.today()
    window_start = as_of - timedelta(days=window_days - 1)

    series = history if history is not None else _load_history()
    if not series:
        return {
            "status": "no_history",
            "window": {"start": window_start.isoformat(), "end": as_of.isoformat()},
            "note": (
                "No portfolio valuation series exists. `portfolio_history.csv` is "
                "written by `tools.portfolio_tracker.snapshot_portfolio`; if it is "
                "empty the snapshot task has never produced a readable summary."
            ),
        }

    in_window = [(d, v) for d, v in series
                 if (dt := _as_date(d)) and window_start <= dt <= as_of]
    cov = coverage([d for d, _ in in_window], window_days=window_days, as_of=as_of)

    flows = window_flows(window_start, as_of)

    base = {
        "window": {"start": window_start.isoformat(), "end": as_of.isoformat(),
                   "days": window_days},
        "coverage": cov,
        "flows": flows,
    }

    if not cov["sufficient"]:
        return {
            **base,
            "status": "insufficient_coverage",
            "note": (
                f"{cov['coverage_pct']}% of the {window_days}-day window carries a "
                f"valuation, below the {MIN_COVERAGE_PCT}% this engine will "
                f"chain-link across. Read `coverage.coverage_pct`, not "
                f"`coverage.span_days` — the span is {cov.get('span_days')} days and "
                f"cannot see the {cov.get('gap_count', 0)} hole(s) inside it."
            ),
        }

    if flows.get("blocked_by"):
        reason = {
            "unclassified_changes": (
                f"{flows['unclassified_count']} observed change(s) in this window "
                "have no stated cause. Any one of them could be money in or out, so "
                "the return would be wrong by an unknown amount in an unknown "
                "direction. Classify them at /api/portfolio/classify."
            ),
            "unpriced_flows": (
                f"{flows['unpriced_count']} classified external flow(s) carry no "
                "`amount_base`. The store records SHARES and currency units; a "
                "time-weighted return needs the flow in money on its own date, and "
                "this engine will not price a share delta itself."
            ),
            "no_position_history": flows.get("note"),
        }.get(flows["blocked_by"], flows.get("note"))
        return {**base, "status": "flows_incomplete", "blocked_by": flows["blocked_by"],
                "note": reason}

    valued_dates = {str(d)[:10] for d, _ in in_window}
    unvalued = sorted({f["date"] for f in flows["flows"]} - valued_dates)
    if unvalued:
        return {
            **base,
            "status": "flow_date_unvalued",
            "unvalued_flow_dates": unvalued,
            "note": (
                f"{len(unvalued)} flow(s) fall on a date with no portfolio valuation "
                f"({unvalued[:5]}). TWR breaks the series AT a flow, so a flow "
                "without a same-day value cannot be removed — chain-linking over it "
                "would leave the contribution inside the return."
            ),
        }

    linked = chain_link(in_window, [(f["date"], f["amount_base"]) for f in flows["flows"]])
    if "error" in linked:
        return {**base, "status": "flow_date_unvalued", "note": linked["error"]}

    bench = blended_benchmark(window_start, as_of, weights=weights, series_fn=series_fn)
    result: dict[str, Any] = {
        **base,
        "status": "measured",
        "twr_pct": linked["twr_pct"],
        "annualized_pct": linked["annualized_pct"],
        "sub_periods": linked["sub_periods"],
        "net_external_flow": linked["net_external_flow"],
        "flow_convention": linked["flow_convention"],
        "benchmark": {k: v for k, v in bench.items() if k != "_legs_with_series"},
    }

    if bench.get("status") in {"measured", "partial"}:
        result["alpha_pct"] = round(linked["twr_pct"] - bench["return_pct"], 4)
        # Tracking error against the heaviest leg. A blended DAILY series would
        # need each leg aligned to every snapshot date and re-weighted, which the
        # currency mix (point-in-time by construction) cannot honestly support.
        heaviest = max(bench["_legs_with_series"], key=lambda leg: leg["weight"])
        te = tracking_error(linked["periods"], heaviest["series"])
        result["tracking_error"] = {**te, "against": heaviest["symbol"],
                                    "against_weight": heaviest["weight"]}
    else:
        result["alpha_pct"] = None
        result["alpha_note"] = "No benchmark could be priced, so there is no alpha."

    result["positions"] = position_attribution(window_start, as_of, price_fn=price_fn)
    result["summary"] = (
        f"Time-weighted return {linked['twr_pct']:+.2f}% over "
        f"{linked['days']} days ({cov['observed_days']} of {window_days} days "
        f"recorded), with {linked['net_external_flow']:+,.0f} of external flow "
        f"removed"
        + (f". Benchmark {bench['return_pct']:+.2f}%, alpha "
           f"{result['alpha_pct']:+.2f}%." if result.get("alpha_pct") is not None
           else ", against no priced benchmark.")
    )
    return result


def _load_history() -> list[tuple[str, float]]:
    """`(date, total value in base currency)` from `portfolio_history.csv`.

    Reads the BASE-currency column rather than always CAD: this app's base
    currency is user-configurable, and a TWR quoted in the wrong currency mixes
    the book's return with the FX pair's.
    """
    try:
        from tools.portfolio_tracker import get_portfolio_history

        df = get_portfolio_history("all")
        if df is None or df.empty:
            return []
    except Exception:  # noqa: BLE001
        return []

    base = "CAD"
    try:
        from tools.memory import get_profile_base_currency

        base = (get_profile_base_currency() or "CAD").upper()
    except Exception:  # noqa: BLE001 — an unreadable profile falls back, and the
        # fallback is named on the payload by the caller rather than hidden here.
        pass

    column = f"total_value_{base.lower()}"
    if column not in df.columns:
        column = "total_value_cad" if "total_value_cad" in df.columns else None
    if column is None:
        return []

    out: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        value = _float(row.get(column))
        stamp = _as_date(row.get("date"))
        if value is not None and stamp is not None:
            out.append((stamp.isoformat(), value))
    return sorted(out)


if __name__ == "__main__":  # pragma: no cover — operator convenience
    import json

    print(json.dumps(get_attribution_report(), indent=2, default=str))
