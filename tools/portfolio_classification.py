"""
4.10a, second half: attaching a CAUSE to an observed position change.

`tools/portfolio_reconciliation` observes that a position row appeared, grew,
shrank or went away, and refuses to say why — every change it emits carries
``cause: "unclassified"``. That refusal is correct and it is also a dead end:
4.10 (benchmark-relative TWR) cannot chain-link a return series while it does not
know which deltas were the investor adding money, and 4.7's superficial-loss half
cannot run without a dated disposition. No amount of waiting resolves either.
This module is the only path from observed to classified, and the path runs
through a human.

**Nothing here infers a cause, ever.** There is no heuristic, no "a cash decrease
plus a share increase on the same day is probably a buy" — that pattern is also a
transfer in kind settling a day late, and a guess that later reads as a recorded
fact is the failure mode this whole item was written around. A classification
exists only because someone stated it, and every record carries who and when.

**It is demand-driven, not a chore list.** The roadmap is explicit that this must
not become routine data entry. `pending_for()` returns only the changes that
actually block a named consumer, so the question put to a human is always "4.10
needs this one to compute your return", never "here are 400 rows, label them".
Most deltas never need a cause at all and should die unclassified.

**The distinction TWR needs is narrower than the taxonomy.** Time-weighted return
does not care whether a delta was a trade, a DRIP or a split — all three leave the
capital base alone. It cares about exactly one thing: did the investor put money
in or take money out. `EXTERNAL_FLOW_CAUSES` is that set, and it is deliberately
two entries long. Everything else is return, and reporting a trade as a flow
would neutralise the very performance TWR is trying to measure.

**And it never asserts a tax event.** `tax_review_causes` marks a change as worth
4.7's attention; it does not say a disposition occurred, because whether one did
depends on the account's shelter, the jurisdiction and rules this module has no
business encoding. See the standing lesson that tax rules are not parameters.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from agent.logger import log_to_component
from tools.exception_logger import log_exceptions

_STORE_FILE = "position_classifications.jsonl"

# The resting state, and the value every unclassified change keeps carrying.
UNCLASSIFIED = "unclassified"

# The causes a human may assign. Each entry is (label, description) — the
# description is what the entry screen shows, and it exists so the person
# choosing is choosing between meanings rather than between words.
CAUSES: dict[str, tuple[str, str]] = {
    "external_inflow": (
        "Money in",
        "Cash or securities you moved INTO the portfolio from outside it — a "
        "contribution, a deposit, a transfer in from an account not tracked here.",
    ),
    "external_outflow": (
        "Money out",
        "Cash or securities you moved OUT of the portfolio — a withdrawal, or a "
        "transfer to an account not tracked here.",
    ),
    "trade": (
        "Trade",
        "A buy or a sell inside the portfolio. Cash became a security or a "
        "security became cash; the amount of your own money invested did not change.",
    ),
    "internal_transfer": (
        "Between your accounts",
        "Moved between two accounts that are both tracked here. Nothing entered "
        "or left the portfolio as a whole.",
    ),
    "income": (
        "Income received",
        "A dividend, distribution or interest payment arriving as cash. This is "
        "return, not money you added.",
    ),
    "drip": (
        "Dividend reinvested",
        "A distribution that bought more shares automatically. Return, taken in "
        "shares instead of cash.",
    ),
    "corporate_action": (
        "Corporate action",
        "A split, consolidation, merger, spin-off or symbol change. The share "
        "count moved without anyone trading.",
    ),
    "fee": (
        "Fee or charge",
        "An account fee, commission or other charge deducted. A cost, which "
        "reduces return rather than removing your capital.",
    ),
    "fx_conversion": (
        "Currency conversion",
        "One currency converted into another inside the portfolio.",
    ),
}

# The ONLY causes that move the capital base. TWR removes these and nothing
# else; a trade or a DRIP counted here would cancel out the very return 4.10 is
# trying to measure.
EXTERNAL_FLOW_CAUSES = frozenset({"external_inflow", "external_outflow"})

# Causes that mean 4.7 should look, NOT that a taxable event occurred. Whether
# one did depends on the account's shelter and the jurisdiction, which this
# module deliberately does not encode.
TAX_REVIEW_CAUSES = frozenset({"trade", "external_outflow", "corporate_action"})

# What each consumer cannot proceed without. `pending_for` reads this, so the
# question a human is asked is always tied to something that is actually blocked.
CONSUMER_REQUIREMENTS: dict[str, str] = {
    "4.10": (
        "Time-weighted return must remove money you added or withdrew. Until a "
        "change is classified, it could be either a flow or a return, and the "
        "series cannot be chain-linked through it."
    ),
    "4.7": (
        "The superficial-loss and asset-location checks need a dated disposition. "
        "An unclassified decrease might be a sale, a transfer or a fee."
    ),
}


def store_path() -> str:
    """Absolute path of THIS profile's classification store.

    Per-profile for the same reason the position store is: a classification is a
    statement by one holder about one holding, and cross-profile contamination
    has actually happened in this codebase.
    """
    from tools.user_profile import get_data_path

    return get_data_path(_STORE_FILE)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def change_id(change: dict[str, Any]) -> str:
    """A stable id for one observed change.

    The four fields that identify WHICH change this is. Account is in the key for
    the same reason it is in the position key: the same ticker in two shelters is
    two positions, and collapsing them would let a classification of one leg of a
    transfer silently apply to the other.
    """
    return "|".join((
        str(change.get("prior_date") or ""),
        str(change.get("current_date") or ""),
        str(change.get("account") or "Unknown").strip(),
        str(change.get("symbol") or "").upper().strip(),
    ))


def fingerprint(change: dict[str, Any]) -> str:
    """The VALUES the classification was made against.

    Separate from the id on purpose. The id says which row a human was looking
    at; this says what that row said at the time. `_write_all` rewrites the whole
    position store, and a corrected or re-imported snapshot can change a delta
    while keeping its date, account and symbol. A classification of "money in,
    $5,000" must not silently carry over to a delta that is now $50,000 — so the
    two are compared on read and a mismatch downgrades the record rather than
    trusting it.
    """
    def _n(v: Any) -> str:
        return "" if v is None else f"{float(v):.6f}"

    return "|".join((_n(change.get("prior_shares")),
                     _n(change.get("current_shares")),
                     _n(change.get("delta"))))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
@log_exceptions()
def read_classifications() -> dict[str, dict[str, Any]]:
    """Every classification on file, keyed by change id, last write winning.

    Append-only on disk (the audit trail is the point — a reclassification must
    not erase what was said first), collapsed to current state on read.
    """
    path = store_path()
    if not os.path.exists(path):
        return {}

    latest: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # One malformed line must not cost the whole ledger.
                continue
            cid = rec.get("change_id")
            if cid:
                latest[cid] = rec
    return latest


@log_exceptions()
def classify_change(change: dict[str, Any], cause: str, note: str = "",
                    classified_by: str = "user",
                    amount_base: float | None = None,
                    base_currency: str = "") -> dict[str, Any]:
    """Record a human's statement of what caused one observed change.

    Rejects an unknown cause rather than storing it: a free-text cause would be
    unreadable by `is_external_flow`, and a flow the TWR engine cannot recognise
    is worse than one that is still openly unclassified.

    Passing ``cause=UNCLASSIFIED`` is legal and is how a classification is
    RETRACTED — it appends a new record rather than deleting the old one, so the
    ledger still shows that someone changed their mind.

    **`amount_base` is what makes a flow usable by 4.10, and it is optional on
    purpose.** This store records QUANTITIES: shares for a security row, currency
    units for a cash row. A time-weighted return needs the flow in MONEY, in the
    base currency, on the flow's own date — and neither a share count nor a
    foreign cash balance is that. The two ways to get it are to price the delta
    retroactively (a historical quote for a transfer in kind, plus that day's FX)
    or to ask the person who already knows, at the moment they are stating the
    cause. This is the second. Where it is absent, 4.10 reports the flow as
    UNPRICED and withholds the return rather than valuing the delta itself —
    a stated cause without an amount is still a real answer to a different
    question, and it must not be turned into a number nobody gave.

    It is only meaningful for the two external-flow causes; storing it on a trade
    or a DRIP would invite a consumer to net it against something. Accepted for
    any cause, ignored by `flow_summary`, which reads it only for flows.
    """
    cause = str(cause or "").strip().lower()
    if cause != UNCLASSIFIED and cause not in CAUSES:
        return {
            "ok": False,
            "error": f"unknown cause {cause!r}",
            "valid_causes": sorted(CAUSES) + [UNCLASSIFIED],
        }

    cid = change_id(change)
    if not cid.strip("|"):
        return {"ok": False, "error": "change is missing the fields that identify it"}

    if amount_base is not None:
        try:
            amount_base = float(amount_base)
        except (TypeError, ValueError):
            return {"ok": False, "error": "amount_base must be numeric"}
        if amount_base != amount_base or amount_base in (float("inf"), float("-inf")):
            # A bare NaN is not valid JSON and one of them once took an endpoint
            # down for a day. It never enters the store.
            return {"ok": False, "error": "amount_base must be finite"}
        # Sign is derived from the cause, never from what was typed: an inflow is
        # positive and an outflow negative, so a user entering "-5000" for a
        # withdrawal and another entering "5000" produce the same record.
        if cause in EXTERNAL_FLOW_CAUSES:
            magnitude = abs(amount_base)
            amount_base = magnitude if cause == "external_inflow" else -magnitude

    record = {
        "change_id": cid,
        "cause": cause,
        "note": str(note or "").strip(),
        "classified_by": str(classified_by or "user").strip() or "user",
        "classified_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint(change),
        # Denormalised so the ledger is readable on its own, without needing the
        # position store to still hold the row it describes.
        "account": str(change.get("account") or "Unknown").strip(),
        "symbol": str(change.get("symbol") or "").upper().strip(),
        "prior_date": change.get("prior_date"),
        "current_date": change.get("current_date"),
        "delta": change.get("delta"),
        # None means "not stated", which is a different thing from zero. 4.10
        # reads the distinction and blocks on the first, not the second.
        "amount_base": amount_base,
        "base_currency": str(base_currency or "").upper().strip() or None,
    }

    path = store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log_to_component("tools", "portfolio_classification",
                     f"{cid} classified as {cause}", {"by": record["classified_by"]})
    return {"ok": True, "record": record}


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def is_external_flow(cause: str) -> bool:
    """Whether this cause moves the capital base. The one question TWR asks."""
    return cause in EXTERNAL_FLOW_CAUSES


@log_exceptions()
def apply_classifications(changes: list[dict[str, Any]],
                          store: dict[str, dict[str, Any]] | None = None,
                          ) -> list[dict[str, Any]]:
    """Annotate observed changes with any cause a human has supplied.

    Returns NEW dicts; the reconciliation engine's output is not mutated.

    Three states come out of this, and the third is the one worth having:

      * no record          -> stays ``unclassified``
      * record, fingerprint matches -> the stated cause applies
      * record, fingerprint DIFFERS -> ``unclassified``, with `stale_classification`
        naming what was said and against which numbers.

    That third case is a snapshot having been rewritten under a classification.
    Silently keeping the old cause would attach a human's statement about $5,000
    to a delta that now reads $50,000; silently dropping it would erase that they
    ever answered. It reverts to unclassified AND says so.
    """
    store = read_classifications() if store is None else store
    out: list[dict[str, Any]] = []

    for change in changes:
        annotated = dict(change)
        rec = store.get(change_id(change))

        if not rec:
            annotated["cause"] = UNCLASSIFIED
            annotated["is_external_flow"] = False
            annotated["classified"] = False
            out.append(annotated)
            continue

        if rec.get("fingerprint") != fingerprint(change):
            annotated["cause"] = UNCLASSIFIED
            annotated["is_external_flow"] = False
            annotated["classified"] = False
            annotated["stale_classification"] = {
                "cause": rec.get("cause"),
                "classified_at": rec.get("classified_at"),
                "against_delta": rec.get("delta"),
                "now_delta": change.get("delta"),
                "note": ("This change was classified, then the underlying snapshot "
                         "changed. The earlier answer is not being applied to "
                         "different numbers — it needs restating."),
            }
            out.append(annotated)
            continue

        cause = rec.get("cause") or UNCLASSIFIED
        annotated["cause"] = cause
        annotated["is_external_flow"] = is_external_flow(cause)
        annotated["classified"] = cause != UNCLASSIFIED
        annotated["classified_at"] = rec.get("classified_at")
        annotated["classified_by"] = rec.get("classified_by")
        annotated["cause_label"] = CAUSES.get(cause, (cause, ""))[0]
        # Carried through so a consumer never has to re-open the store, and kept
        # as None rather than 0.0 when unstated — see `classify_change`.
        annotated["amount_base"] = rec.get("amount_base")
        annotated["amount_base_currency"] = rec.get("base_currency")
        if rec.get("note"):
            annotated["classification_note"] = rec["note"]
        out.append(annotated)

    return out


# ---------------------------------------------------------------------------
# Demand — who is blocked, and by what
# ---------------------------------------------------------------------------
def _relevant_to(consumer: str, change: dict[str, Any]) -> bool:
    """Whether one change could block one consumer.

    4.10 is blocked by ANY unclassified change: until someone says otherwise, a
    delta might be a flow, and a chain-linked series cannot step over a maybe.

    4.7 is narrower — it needs dispositions, so an INCREASE with no cash leg
    cannot be one. This is a filter on what to ASK about, never an inference
    about what happened.
    """
    if consumer == "4.10":
        return True
    if consumer == "4.7":
        delta = change.get("delta")
        try:
            return float(delta) < 0
        except (TypeError, ValueError):
            return True
    return True


@log_exceptions()
def pending_for(consumer: str, changes: list[dict[str, Any]],
                store: dict[str, dict[str, Any]] | None = None,
                limit: int = 25) -> dict[str, Any]:
    """The changes `consumer` cannot proceed without, and why it cannot.

    The whole point of routing every request through here: a human is only ever
    asked about a delta that something is actually waiting on. A change nobody
    needs stays unclassified forever, which is the correct outcome and not a
    backlog.
    """
    consumer = str(consumer or "").strip()
    if consumer not in CONSUMER_REQUIREMENTS:
        return {"consumer": consumer, "error": f"unknown consumer {consumer!r}",
                "known": sorted(CONSUMER_REQUIREMENTS)}

    annotated = apply_classifications(changes, store=store)
    blocking = [c for c in annotated
                if not c.get("classified") and _relevant_to(consumer, c)]
    restatable = [c for c in blocking if c.get("stale_classification")]

    return {
        "consumer": consumer,
        "requirement": CONSUMER_REQUIREMENTS[consumer],
        "blocked": bool(blocking),
        "pending_count": len(blocking),
        "needs_restating": len(restatable),
        "pending": blocking[:limit],
        "truncated": len(blocking) > limit,
    }


@log_exceptions()
def flow_summary(changes: list[dict[str, Any]],
                 store: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """What 4.10 would read: classified external flows, and what is still unknown.

    `complete` is the field that matters and it is deliberately strict. A summary
    of the flows found so far, quoted without it, is exactly the "count that
    reads as coverage" failure this project keeps rediscovering — a TWR computed
    over three known deposits while a fourth sits unclassified is not slightly
    wrong, it is wrong in an unknown direction by an unknown amount.
    """
    annotated = apply_classifications(changes, store=store)

    inflows = [c for c in annotated if c.get("cause") == "external_inflow"]
    outflows = [c for c in annotated if c.get("cause") == "external_outflow"]
    unclassified = [c for c in annotated if not c.get("classified")]

    def _total(rows: list[dict[str, Any]]) -> float:
        total = 0.0
        for r in rows:
            try:
                total += abs(float(r.get("delta") or 0.0))
            except (TypeError, ValueError):
                continue
        return round(total, 6)

    flows = inflows + outflows
    unpriced = [c for c in flows if c.get("amount_base") is None]

    return {
        "complete": not unclassified,
        "changes_seen": len(annotated),
        "classified_count": len(annotated) - len(unclassified),
        "unclassified_count": len(unclassified),
        "external_inflows": len(inflows),
        "external_outflows": len(outflows),
        # Units are SHARES for a security row and CURRENCY UNITS for a cash row.
        # Not summed across the two, and not called a dollar amount: this store
        # records quantities, and pricing them is 4.10's job, not this module's.
        "inflow_units": _total(inflows),
        "outflow_units": _total(outflows),
        # A SECOND completeness axis, and it is separate from the first because a
        # flow can be fully classified and still unusable. TWR needs the flow in
        # money on its own date; `amount_base` is the only place that number can
        # come from, and where nobody stated it the flow is priced-unknown rather
        # than zero. `complete` alone would report this window as ready.
        "priced": not unpriced,
        "unpriced_flow_count": len(unpriced),
        "flow_amount_base": (
            round(sum(float(c["amount_base"]) for c in flows), 2) if flows and not unpriced
            else None
        ),
        "note": (
            "Every observed change has a stated cause; external flows are complete "
            "for this window."
            if not unclassified else
            f"{len(unclassified)} of {len(annotated)} observed change(s) have no stated "
            "cause. Any of them could be money in or out, so the flows below are a "
            "LOWER BOUND and a time-weighted return computed on them would be wrong "
            "by an unknown amount in an unknown direction."
        ),
    }


@log_exceptions()
def tax_review_changes(changes: list[dict[str, Any]],
                       store: dict[str, dict[str, Any]] | None = None,
                       ) -> list[dict[str, Any]]:
    """Classified changes 4.7 should LOOK at. Not a list of taxable events.

    Whether any of these realises a gain depends on the account's shelter and the
    jurisdiction's rules — neither of which lives here, and both of which this
    project has already got wrong once by treating tax rules as parameters.
    """
    return [c for c in apply_classifications(changes, store=store)
            if c.get("cause") in TAX_REVIEW_CAUSES]


if __name__ == "__main__":
    from tools.portfolio_reconciliation import get_reconciliation

    recon = get_reconciliation()
    print(json.dumps({
        "status": recon.get("status"),
        "flows": flow_summary(recon.get("changes") or []),
        "blocking_4_10": pending_for("4.10", recon.get("changes") or [])["pending_count"],
    }, indent=2, default=str))
