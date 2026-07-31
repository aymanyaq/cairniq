"""
Portfolio-change reconciliation — the dated record of what actually moved (4.10a).

**Why this exists.** Nothing in this app records a transaction. `trade_journal`
is empty, and `portfolio_history.csv` holds six portfolio-LEVEL columns (value
and cost basis) with no per-account, per-holding detail and no external-flow
column. `my_portfolio.csv` carries the `Account` column but is OVERWRITTEN on
every sync, so yesterday's quantities are gone. Two shipped items are blocked on
exactly that gap: 4.10's time-weighted return needs dated external flows, and
4.7's superficial-loss/wash-sale half needs a repurchase to detect.

**What this is: a RECORDER, not a differ over existing history.** Measured
2026-07-29 — there is no prior per-position state anywhere to diff against, so
this accrues from zero exactly the way 5.5's fund-shares recorder does. On day
one it can say nothing, and it says *that* rather than "no changes detected".
Those are opposite claims, and shipping the second one is the failure this
codebase has now hit in Market Pulse, in 5.4's tone verdict and in 5.9's
`Unknown` rows.

**What it must never do: invent a cause.** A quantity delta can be a buy or a
sale — and equally a deposit, a withdrawal, a transfer in kind, a dividend
reinvestment, a fee, an FX conversion, a split or a corporate action. Every
change this module emits is `cause: "unclassified"` and stays that way until an
activity feed supplies the reason or the user confirms one. **An unclassified
delta must never be promoted into an external flow for 4.10 or a tax event for
4.7.** A plausible number in a returns series is indistinguishable from a real
one after the fact.

**Coverage, not span.** The existing 365-day gate
(``goal_projection._history_span_days``) measures ``(max - min).days`` — the
distance between two endpoints, blind to holes between them. A history can be
82 rows across an 88-day span with 7 days missing in 5 gaps. TWR
chain-links between observations, so a hole is not neutral: the link spans
straight across it, and a gap containing a deposit is precisely the corruption
TWR exists to prevent. Everything here reports observed days against the
calendar window, and any change detected across a gap is flagged as such.
"""

from __future__ import annotations

import csv
import os
from datetime import date, datetime, timedelta
from typing import Any

from agent.logger import log_to_component
from tools.exception_logger import log_exceptions

_HISTORY_FILE = "position_history.csv"
_FIELDS = ["date", "account", "symbol", "currency", "shares", "private", "as_of"]

# AUTHORED CONSTANT (2.7): a floor for what counts as a cash movement, in the
# holding's own currency. Not a measured threshold — it exists only to keep
# sub-dollar FX and rounding noise out of a ledger a human reads. Share changes
# have no floor on purpose: a fractional increase is exactly the shape of a
# dividend reinvestment, which is a real event worth seeing.
MATERIAL_CASH_MIN = 1.0

# Cash is held as a position row whose "shares" are units of currency.
_CASH_SYMBOLS = {"CASH", "USD", "CAD", "EUR", "GBP", "AUD", "JPY"}


def history_path() -> str:
    """Absolute path of this PROFILE's position store.

    Per-profile, unlike 5.5's global fund store: shares outstanding is a fact
    about a fund, but a position is a fact about the holder. Two profiles must
    never reconcile against each other's holdings — cross-profile contamination
    has actually happened in this codebase.
    """
    from tools.user_profile import get_data_path

    return get_data_path(_HISTORY_FILE)


def _finite(value: Any) -> float | None:
    """Coerce to a finite float or None. NaN/inf never reach the store — a bare
    NaN is not valid JSON and one of them once took an endpoint down for a day."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def is_cash(symbol: str) -> bool:
    """Whether a position row represents currency rather than a security."""
    return str(symbol or "").upper().strip() in _CASH_SYMBOLS


def _base_currency() -> str:
    """The profile's base currency, for labelling the amount a flow is stated in.

    Falls back to CAD rather than raising: a missing label is a worse failure
    here than a wrong-but-visible one, because the alternative is an unlabelled
    money box. The base currency is user-configurable and must never be assumed
    to match the holding's own currency.
    """
    try:
        from tools.memory import get_profile_base_currency

        return (get_profile_base_currency() or "CAD").upper()
    except Exception:  # noqa: BLE001 — an unreadable profile still gets a label
        return "CAD"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
@log_exceptions()
def read_history(on_date: str | None = None) -> list[dict[str, Any]]:
    """All recorded position rows, oldest first. Never raises on a bad store."""
    path = history_path()
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("date") and r.get("symbol")]
    if on_date:
        rows = [r for r in rows if r["date"] == on_date]
    return sorted(rows, key=lambda r: (r["date"], r.get("account", ""), r["symbol"]))


def _write_all(rows: list[dict[str, Any]]) -> None:
    """Rewrite the store atomically. A truncated CSV would lose the whole accrued
    series, and — as with 5.5 — no vendor will sell these days back."""
    path = history_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _FIELDS})
    os.replace(tmp, path)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    """The identity of a position: an account and a symbol.

    Account is part of the key deliberately. The same ticker in a TFSA and an
    RRSP are different positions with different tax treatment (4.7), and
    collapsing them would hide a transfer between accounts entirely — the two
    legs would net to zero.
    """
    return (str(row.get("account") or "Unknown").strip(), str(row["symbol"]).upper().strip())


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------
@log_exceptions()
def snapshot_positions(force: bool = False) -> dict[str, Any]:
    """Record one row per (account, symbol) held today.

    Idempotent per calendar day: a second run records nothing unless `force`.
    Returns a report whose `recorded` count is what proves the chain ran end to
    end — portfolio readable, rows parsed, store written. Zero recorded against a
    non-empty portfolio is the dead-recorder signal and is reported, not smoothed
    (2.5/2.6).
    """
    from tools.portfolio_csv import load_portfolio

    today = date.today().isoformat()
    holdings = load_portfolio()

    if isinstance(holdings, dict) and "error" in holdings:
        return {
            "date": today,
            "recorded": 0,
            "positions": 0,
            "error": holdings["error"],
            "store": history_path(),
        }
    if not isinstance(holdings, list):
        return {"date": today, "recorded": 0, "positions": 0,
                "error": "portfolio unreadable", "store": history_path()}

    rows = read_history()
    if any(r["date"] == today for r in rows) and not force:
        return {
            "date": today,
            "recorded": 0,
            "positions": len(holdings),
            "declined": "already recorded today",
            "store": history_path(),
        }

    stamp = datetime.now().isoformat(timespec="seconds")
    new_rows, skipped, sync_errors = [], 0, []
    for h in holdings:
        if not isinstance(h, dict):
            skipped += 1
            continue
        # `load_portfolio` appends a METADATA SENTINEL to the holdings list —
        # `{"_sync_errors": [...]}` — rather than returning it alongside. It is
        # not a position, and counting it as an unreadable one would add exactly
        # one phantom parse failure every single day, which is how a REAL parse
        # failure becomes invisible. Only `get_portfolio_summary` filters it;
        # every other consumer of load_portfolio() iterates straight past it.
        if "_sync_errors" in h:
            sync_errors.extend(h.get("_sync_errors") or [])
            continue

        symbol = str(h.get("symbol") or "").upper().strip()
        shares = _finite(h.get("shares"))
        if not symbol or shares is None:
            # A row we cannot read is NOT a zero position — recording it as one
            # would manufacture a full disposal on the next reconciliation.
            skipped += 1
            continue
        new_rows.append({
            "date": today,
            "account": str(h.get("account") or "Unknown").strip(),
            "symbol": symbol,
            "currency": str(h.get("currency") or "").upper().strip(),
            "shares": shares,
            "private": "1" if h.get("is_private_asset") else "",
            # Fetch time, not read time (5.8).
            "as_of": stamp,
        })

    if new_rows:
        rows = [r for r in rows if r["date"] != today]
        _write_all(sorted(rows + new_rows, key=lambda r: (r["date"], r.get("account", ""), r["symbol"])))

    if skipped:
        log_to_component(
            "tools", "portfolio_reconciliation",
            f"{skipped} holding(s) had no readable share count and were not recorded",
            level=30,
        )

    return {
        "date": today,
        "recorded": len(new_rows),
        # Positions, not list length: the sentinel is not a holding.
        "positions": len(new_rows) + skipped,
        "skipped_unreadable": skipped,
        # Surfaced rather than swallowed: a broker that failed to sync means the
        # day's snapshot may be PARTIAL, and a partial snapshot reconciles as a
        # mass disposal tomorrow if nobody is told.
        "sync_errors": sync_errors,
        "store": history_path(),
    }


# ---------------------------------------------------------------------------
# Coverage — observed days against the calendar window
# ---------------------------------------------------------------------------
def snapshot_dates(rows: list[dict[str, Any]] | None = None) -> list[str]:
    """Distinct dates that actually have rows, oldest first."""
    rows = read_history() if rows is None else rows
    return sorted({r["date"] for r in rows})


@log_exceptions()
def get_coverage(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Observed days against the calendar window, plus the gaps between them.

    **This is the figure `goal_projection._history_span_days` does not compute.**
    That gate returns `(max - min).days` and therefore cannot see a hole between
    its two endpoints. `observed_days` / `calendar_days` can, and `gaps` names
    them — because a chain-linked return spans a hole silently, and a hole that
    contained a deposit is the exact error TWR exists to prevent.
    """
    dates = snapshot_dates(rows)
    if not dates:
        return {"observed_days": 0, "calendar_days": 0, "missing_days": 0,
                "first_date": "", "latest_date": "", "gaps": []}

    parsed = [date.fromisoformat(d) for d in dates]
    calendar_days = (parsed[-1] - parsed[0]).days + 1
    gaps = [
        {"after": dates[i - 1], "before": dates[i], "missing_days": (parsed[i] - parsed[i - 1]).days - 1}
        for i in range(1, len(parsed))
        if (parsed[i] - parsed[i - 1]).days > 1
    ]
    return {
        "observed_days": len(dates),
        "calendar_days": calendar_days,
        "missing_days": calendar_days - len(dates),
        "first_date": dates[0],
        "latest_date": dates[-1],
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Reconcile — observed changes, cause deliberately withheld
# ---------------------------------------------------------------------------
@log_exceptions()
def detect_changes(prior_date: str, current_date: str,
                   rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Changes observed between two recorded snapshots.

    Every change carries ``cause: "unclassified"``. A quantity delta is equally
    consistent with a trade, a deposit, a transfer in kind, a reinvested
    dividend, a fee, an FX conversion, a split or a corporate action, and this
    module has no way to tell them apart. Naming one would be a fabrication that
    later reads as a recorded fact.
    """
    rows = read_history() if rows is None else rows
    prior = {_key(r): r for r in rows if r["date"] == prior_date}
    current = {_key(r): r for r in rows if r["date"] == current_date}

    gap_days = (date.fromisoformat(current_date) - date.fromisoformat(prior_date)).days
    spans_gap = gap_days > 1

    changes: list[dict[str, Any]] = []
    for key in sorted(set(prior) | set(current)):
        account, symbol = key
        was = _finite((prior.get(key) or {}).get("shares"))
        now = _finite((current.get(key) or {}).get("shares"))
        row = current.get(key) or prior.get(key) or {}

        if was is None and now is None:
            continue
        if was is not None and now is not None and was == now:
            continue

        cash = is_cash(symbol)
        delta = (now or 0.0) - (was or 0.0)
        # The cash floor keeps FX rounding out of a human-read ledger; shares
        # have no floor, because a fractional increase IS the DRIP signal.
        if cash and abs(delta) < MATERIAL_CASH_MIN:
            continue

        if was is None:
            kind = "position_opened"
        elif now is None:
            kind = "position_closed"
        elif cash:
            kind = "cash_increase" if delta > 0 else "cash_decrease"
        else:
            kind = "quantity_increase" if delta > 0 else "quantity_decrease"

        changes.append({
            "kind": kind,
            "account": account,
            "symbol": symbol,
            "currency": str(row.get("currency") or ""),
            "is_cash": cash,
            "prior_shares": was,
            "current_shares": now,
            "delta": delta,
            "prior_date": prior_date,
            "current_date": current_date,
            # Never inferred. See the module docstring.
            "cause": "unclassified",
            # A change seen across a gap cannot be attributed to a single day,
            # and must not be treated as one dated event by any consumer.
            "spans_gap": spans_gap,
            "gap_days": gap_days,
        })
    return changes


@log_exceptions()
def get_reconciliation(limit: int = 50) -> dict[str, Any]:
    """The read surface: what moved since the previous snapshot, and what is known.

    `status` is the field to read first:
      * ``no_data``   — nothing has ever been recorded. NOT a quiet portfolio.
      * ``accruing``  — exactly one snapshot exists, so there is nothing to
                        compare it against yet. **No change list in this state,
                        by design** — a first snapshot with zero changes and a
                        genuinely unchanged portfolio are different claims.
      * ``ready``     — `changes` holds the observed deltas.

    Each change's `cause` starts at ``"unclassified"`` and stays there unless a
    human has stated one through `tools.portfolio_classification`. Nothing in
    this module or that one infers a cause; `classified` is the flag that says
    whether the value came from a person. The `classification` block summarises
    how much of the window is still unstated, which is what 4.10 must gate on.
    """
    rows = read_history()
    coverage = get_coverage(rows)
    dates = snapshot_dates(rows)

    if not dates:
        return {
            "status": "no_data",
            "coverage": coverage,
            "changes": [],
            "note": (
                "No position snapshots have been recorded yet. This is NOT a report "
                "that nothing changed — the recorder has not run, or has never found "
                "a readable portfolio. Check the scheduler's `position_snapshot` task."
            ),
        }

    if len(dates) < 2:
        return {
            "status": "accruing",
            "coverage": coverage,
            "changes": [],
            "snapshots": len(dates),
            "days_until_ready": 1,
            "note": (
                f"Recording is live (first snapshot {dates[0]}), but a reconciliation "
                "needs two snapshots to compare. No change list is being reported — "
                "this is an accruing record, not an unchanged portfolio."
            ),
        }

    prior_date, current_date = dates[-2], dates[-1]
    changes = detect_changes(prior_date, current_date, rows)
    spans_gap = bool(changes and changes[0]["spans_gap"])
    base_currency = _base_currency()

    # Overlay any cause a human has stated. This is the ONLY way a change stops
    # being unclassified — see tools/portfolio_classification on why there is no
    # inference step here.
    from tools.portfolio_classification import apply_classifications, flow_summary

    changes = apply_classifications(changes)
    classification = flow_summary(changes)

    unstated = classification["unclassified_count"]
    note = (
        f"{len(changes)} change(s) observed between {prior_date} and {current_date}. "
        f"{unstated} of them have no stated cause: a quantity delta is equally "
        "consistent with a trade, a deposit, a transfer, a reinvested dividend, a fee "
        "or a corporate action. An unclassified change may not be counted as an "
        "external flow (4.10) or a tax event (4.7)."
        if unstated else
        f"{len(changes)} change(s) observed between {prior_date} and {current_date}, "
        "all with a stated cause."
    )
    if spans_gap:
        note += (
            f" These two snapshots are {changes[0]['gap_days']} days apart, so the "
            "changes cannot be attributed to a single date."
        )

    return {
        "status": "ready",
        "coverage": coverage,
        "prior_date": prior_date,
        "current_date": current_date,
        "spans_gap": spans_gap,
        # The currency an `amount_base` must be stated in, and it is on the
        # payload because it is usually NOT the currency of the row beside it.
        # A cash change on a USD account still has to be valued in the base
        # currency for 4.10 to chain-link it, and a form that shows "USD" in the
        # symbol column while silently expecting CAD in the amount box is a units
        # trap of exactly the kind this codebase keeps finding.
        "base_currency": base_currency,
        "changes": changes[:limit],
        "change_count": len(changes),
        "truncated": len(changes) > limit,
        "material_cash_min": MATERIAL_CASH_MIN,
        # Computed over ALL changes, not the truncated page: a completeness
        # claim derived from the rows that happened to fit on screen would be
        # the same "count that reads as coverage" error this file keeps finding.
        "classification": classification,
        "note": note,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_reconciliation(), indent=2, default=str))
