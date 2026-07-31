"""4.7 — the loss-deferral policy engine: versioned, jurisdiction-gated, and mostly refusals.

**This is not a rule table with a country column, and the roadmap's own 07-25
entry was wrong to size it as one.** It said the US wash-sale rule and the
Canadian superficial-loss rule are "the same 30-day repurchase mechanic differing
only in whose accounts count". They are not. They differ in what TRIGGERS them,
in what the denied loss then DOES, in which of the holder's own accounts can
spring the trap, and Canada has a test the US regime has no analogue for. Each of
those changes the answer, not a constant — which is why the rules live inside
per-jurisdiction modules that carry their own version and their own explicit
coverage matrix, and why the engine holds only the mechanism for selecting,
applying and reporting them.

**`not_covered` is a first-class result and it BLOCKS.** It is never a
fall-through to the nearest jurisdiction and never a silent skip. This is P2 of
3.8's execution gate: a TRIM-and-rebuy proposal in an account whose jurisdiction
this engine does not cover must stop, not proceed under Canadian rules because
Canada was first in the dict. The precedent is 4.7's own shipped defect —
`classify_account_type` put TFSA, Roth and ISA in one bucket and charged all
three the Canadian TFSA treatment, inventing a withholding leak for a US user's
Roth and then recommending a swap to fix it.

**Nothing here decides "substantially identical".** The US trigger is broader
than the same ticker, it reaches options and contracts to acquire, and it is a
judgment call — so this engine flags a CANDIDATE and says who has to decide.
Reporting "this is a wash sale" about a sell-XYZ/buy-a-similar-ETF pair would be
asserting a legal conclusion the code cannot reach.

**No module has been reviewed by a tax professional, and every payload says so.**
The roadmap names that review as a deliverable of this item rather than a
disclaimer to append afterwards, so `advice_ready` is False on every output until
`professional_review` records a reviewer against a specific version. The rules
still COMPUTE — a flag the advisor can act on defensively (stop, ask) is useful
before it is reviewed; a rule the advisor QUOTES is not.

Sources read while writing this, neither of which is a substitute for that review:
IRS Publication 550 (https://www.irs.gov/publications/p550) and CRA's
capital-losses guidance.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from tools.exception_logger import log_exceptions

# The engine's own version, separate from each module's. Bump when the SELECTION
# or REPORTING mechanism changes; a module version tracks its rules.
ENGINE_VERSION = "2026-07-30.1"

# The one shared number, and it is shared by coincidence rather than by design:
# both regimes happen to use 30 days on each side. It lives on each module, not
# here, so a jurisdiction with a different window does not have to fight a
# constant that reads like a law of nature.
_US_WINDOW_DAYS = 30
_CA_WINDOW_DAYS = 30


# ---------------------------------------------------------------------------
# The per-jurisdiction modules
# ---------------------------------------------------------------------------
POLICY_MODULES: dict[str, dict[str, Any]] = {
    "US": {
        "jurisdiction": "US",
        "rule_name": "Wash sale",
        "authority": "IRC §1091; IRS Publication 550; Rev. Rul. 2008-5",
        "version": "2026-07-30.1",
        "window_days_before": _US_WINDOW_DAYS,
        "window_days_after": _US_WINDOW_DAYS,
        # The trigger, and it is deliberately not the same string as Canada's.
        "identity_test": "substantially_identical",
        "identity_note": (
            "Broader than the same ticker. It reaches options and contracts to "
            "acquire, and whether two different funds tracking the same index are "
            "substantially identical is a JUDGMENT CALL that this engine does not "
            "make."
        ),
        # Which of the holder's own accounts can spring it.
        "affiliated_scope": ["self", "spouse", "own_ira", "own_roth"],
        "affiliated_note": (
            "A purchase in the taxpayer's own IRA or Roth triggers the rule, and a "
            "spouse's purchase counts."
        ),
        # What the denied loss then DOES. The single biggest divergence.
        "disallowed_loss_treatment": "added_to_basis_of_replacement",
        "registered_plan_treatment": "permanently_disallowed",
        "registered_plan_note": (
            "Per Rev. Rul. 2008-5, a loss washed by a purchase inside the "
            "taxpayer's IRA or Roth is PERMANENTLY DISALLOWED with no basis "
            "add-back anywhere. This is the only case in either regime where the "
            "loss is simply destroyed rather than deferred."
        ),
        "still_owned_test": False,
        "coverage": {
            "rules": ["wash_sale"],
            "account_types": ["TAXABLE", "IRA", "ROTH_IRA", "401K", "ROTH_401K", "403B"],
            "instrument_classes": ["equity", "etf", "mutual_fund"],
            "excluded": ["options", "futures", "crypto", "bonds_with_different_cusips"],
            "excluded_note": (
                "Options and contracts to acquire ARE within §1091's reach; they are "
                "excluded here because this engine cannot evaluate them, which is a "
                "limitation of the engine and not of the rule."
            ),
        },
        "professional_review": {"reviewed": False, "reviewed_version": None,
                                "reviewed_at": None, "reviewer": None},
    },
    "CA": {
        "jurisdiction": "CA",
        "rule_name": "Superficial loss",
        "authority": "Income Tax Act s. 40(2)(g)(i) and s. 54; CRA capital-losses guidance",
        "version": "2026-07-30.1",
        "window_days_before": _CA_WINDOW_DAYS,
        "window_days_after": _CA_WINDOW_DAYS,
        "identity_test": "identical_property",
        "identity_note": (
            "NARROWER and more mechanical than the US test. Identical property is "
            "closer to the same security than to 'anything economically similar', "
            "so a swap into a different issuer's index fund is treated differently "
            "here than it would be under §1091."
        ),
        "affiliated_scope": ["self", "spouse_or_common_law", "controlled_corporation",
                             "certain_trusts", "their_registered_plans"],
        "affiliated_note": (
            "'Affiliated persons' is a DIFFERENT SET from the US one, not a "
            "renaming of it: it reaches corporations the taxpayer controls and "
            "certain trusts, as well as those persons' registered plans."
        ),
        "disallowed_loss_treatment": "added_to_affiliated_acb",
        "registered_plan_treatment": "denied_outright",
        "registered_plan_note": (
            "The denied loss is generally added to the AFFILIATED PERSON's adjusted "
            "cost base — deferred and recoverable later, not destroyed. The "
            "exception is a transfer into a registered plan, where it is denied "
            "outright."
        ),
        # The test the US regime has no analogue for.
        "still_owned_test": True,
        "still_owned_note": (
            "The property must STILL BE OWNED by an affiliated person at the END of "
            "the 30-day window after the sale. An in-and-out that closes inside the "
            "window is therefore treated differently, and this test has no US "
            "counterpart — it is the reason the two rules cannot share one engine."
        ),
        "coverage": {
            "rules": ["superficial_loss"],
            "account_types": ["TAXABLE", "NON_REGISTERED", "RRSP", "TFSA", "RRIF",
                              "LIRA", "FHSA"],
            "instrument_classes": ["equity", "etf", "mutual_fund"],
            "excluded": ["options", "futures", "crypto", "debt_forgiveness_cases"],
            "excluded_note": (
                "Also out of scope: the s. 40(3.3)/(3.4) stop-loss rules for "
                "corporations and partnerships, and the 30-day rule's interaction "
                "with a deemed disposition on emigration or death."
            ),
        },
        "professional_review": {"reviewed": False, "reviewed_version": None,
                                "reviewed_at": None, "reviewer": None},
    },
}

# Jurisdictions this engine KNOWS it does not cover. Listed so `not_covered`
# reports "no module" rather than "unknown jurisdiction" for a shelter the
# account classifier can name — a UK ISA is recognised, and there is still no UK
# module, and those are different failures with different fixes.
KNOWN_UNCOVERED: frozenset[str] = frozenset({"UK", "AU", "EU"})


def policy_version(jurisdiction: str) -> str | None:
    module = POLICY_MODULES.get(str(jurisdiction or "").upper())
    return module["version"] if module else None


@log_exceptions()
def coverage_matrix() -> dict[str, Any]:
    """What this engine covers, per jurisdiction, and what it explicitly does not.

    A consumer that cannot read this should not be calling `check_disposition`.
    3.8's gate reads `advice_ready` from here before it lets a rebuy proposal
    through.
    """
    return {
        "engine_version": ENGINE_VERSION,
        "jurisdictions": {
            j: {
                "rule_name": m["rule_name"],
                "policy_version": m["version"],
                "authority": m["authority"],
                "identity_test": m["identity_test"],
                "coverage": m["coverage"],
                "professional_review": m["professional_review"],
                "advice_ready": bool(m["professional_review"]["reviewed"]),
            }
            for j, m in POLICY_MODULES.items()
        },
        "known_uncovered": sorted(KNOWN_UNCOVERED),
        "advice_ready": all(m["professional_review"]["reviewed"]
                            for m in POLICY_MODULES.values()),
        "advice_ready_note": (
            "No jurisdiction module has been reviewed by a tax professional. Until "
            "one is — recorded against a specific `policy_version` — these results "
            "may be used DEFENSIVELY (stop, flag, ask) and must not be quoted to a "
            "user as the tax treatment of their trade."
        ),
    }


def _not_covered(jurisdiction: str | None, reason: str,
                 **extra: Any) -> dict[str, Any]:
    """The blocking result. Never a fall-through, never a nearest-neighbour match."""
    return {
        "status": "not_covered",
        "blocks": True,
        "jurisdiction": jurisdiction,
        "engine_version": ENGINE_VERSION,
        "policy_version": None,
        "reason": reason,
        "known_jurisdictions": sorted(POLICY_MODULES),
        "note": (
            "This engine has no rules for that jurisdiction and will not apply "
            "another one's. A loss-deferral rule differs between regimes in what "
            "triggers it, in what the denied loss does and in which accounts "
            "count — substituting the nearest available module produces a "
            "confident answer about the wrong law."
        ),
        **extra,
    }


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
@log_exceptions()
def check_disposition(sale: dict[str, Any],
                      acquisitions: list[dict[str, Any]] | None = None,
                      jurisdiction: str | None = None) -> dict[str, Any]:
    """Whether a loss-making disposition sits inside a repurchase window.

    `sale` is ``{symbol, date, account, proceeds?, cost_basis?, gain_loss?}``.
    `acquisitions` are ``{symbol, date, account, shares?}`` — every purchase the
    caller knows about, in ANY account, because the account set is exactly what
    differs between the two regimes.

    Read `status`:
      * ``not_covered``  — BLOCKS. No module for that jurisdiction.
      * ``not_a_loss``   — the disposition is a gain or flat, so no rule applies.
      * ``no_candidates``— a loss, and nothing acquired inside the window.
      * ``candidate``    — a loss with an acquisition inside the window. This is a
        FLAG, not a determination: whether the two securities meet the
        jurisdiction's identity test is a judgment call.

    Never returns "this is a wash sale". The closest it comes is `candidate` with
    `determination_required_by` naming who has to decide.
    """
    jurisdiction = str(jurisdiction or "").upper().strip() or None
    if not jurisdiction:
        return _not_covered(None, "No jurisdiction was supplied or could be "
                                  "derived from the account name.")
    module = POLICY_MODULES.get(jurisdiction)
    if not module:
        return _not_covered(
            jurisdiction,
            f"No policy module for {jurisdiction}."
            + (" This jurisdiction is recognised and deliberately uncovered."
               if jurisdiction in KNOWN_UNCOVERED else ""),
        )

    sale_date = _as_date(sale.get("date"))
    if sale_date is None:
        return {"status": "no_date", "blocks": True, "jurisdiction": jurisdiction,
                "policy_version": module["version"], "engine_version": ENGINE_VERSION,
                "note": (
                    "The disposition carries no usable date. Both regimes are dated "
                    "windows, so an undated sale cannot be checked — and an "
                    "INFERRED date must never be used here. See 4.10a: an "
                    "unclassified delta is not a tax event."
                )}

    loss = _loss_amount(sale)
    stamp = {
        "jurisdiction": jurisdiction,
        "rule_name": module["rule_name"],
        "authority": module["authority"],
        "policy_version": module["version"],
        "engine_version": ENGINE_VERSION,
        "advice_ready": bool(module["professional_review"]["reviewed"]),
        "advice_ready_note": (
            "This module has NOT been reviewed by a tax professional. Use it to "
            "stop and ask, never to tell the user what their tax treatment is."
        ),
    }

    if loss is None:
        return {**stamp, "status": "unknown_result", "blocks": True,
                "note": ("Neither a gain/loss nor both a cost basis and proceeds "
                         "were supplied, so whether this disposition is a LOSS is "
                         "unknown. The rule only bites on a loss, and assuming one "
                         "either way is a fabrication.")}
    if loss <= 0:
        return {**stamp, "status": "not_a_loss", "blocks": False,
                "gain_loss": -loss if loss else 0.0,
                "note": (f"{module['rule_name']} applies only to a realised LOSS. "
                         "This disposition is not one.")}

    before = sale_date - timedelta(days=module["window_days_before"])
    after = sale_date + timedelta(days=module["window_days_after"])

    candidates: list[dict[str, Any]] = []
    for acq in acquisitions or []:
        acq_date = _as_date(acq.get("date"))
        if acq_date is None or not (before <= acq_date <= after):
            continue
        if str(acq.get("symbol") or "").upper() != str(sale.get("symbol") or "").upper():
            # A different ticker can still trigger the US test and generally not
            # the Canadian one — but deciding that is not this engine's call, so
            # a non-matching symbol is not silently dropped either.
            candidates.append({**_acq_row(acq, acq_date, sale_date),
                               "same_symbol": False,
                               "why": ("Different symbol. Under "
                                       f"{module['identity_test']}, whether this "
                                       "counts is a judgment call.")})
            continue
        candidates.append({**_acq_row(acq, acq_date, sale_date), "same_symbol": True,
                           "why": "Same symbol, inside the window."})

    if not candidates:
        return {**stamp, "status": "no_candidates", "blocks": False,
                "loss_amount": round(loss, 2),
                "window": {"from": before.isoformat(), "to": after.isoformat()},
                "note": (
                    f"A realised loss with no acquisition of this security inside "
                    f"the {module['window_days_before']}-day window either side. "
                    "This is only as complete as the acquisitions supplied — see "
                    "`scan_dispositions` on why this app's transaction record is "
                    "the binding limit."
                )}

    shelter_hits = [c for c in candidates if c.get("in_registered_plan")]
    consequence = (module["registered_plan_note"] if shelter_hits
                   else _consequence_note(module))

    return {
        **stamp,
        "status": "candidate",
        # Blocks a 3.8 rebuy proposal: a candidate is exactly the case a human
        # must clear before the advisor recommends the trade.
        "blocks": True,
        "loss_amount": round(loss, 2),
        "window": {"from": before.isoformat(), "to": after.isoformat()},
        "candidates": candidates,
        "identity_test": module["identity_test"],
        "identity_note": module["identity_note"],
        "affiliated_scope": module["affiliated_scope"],
        "affiliated_note": module["affiliated_note"],
        "consequence_if_triggered": consequence,
        "still_owned_test": module["still_owned_test"],
        **({"still_owned_note": module["still_owned_note"]}
           if module.get("still_owned_note") else {}),
        "determination_required_by": (
            "a tax professional. This engine reports that a loss and an "
            "acquisition fall inside the same window. Whether the two securities "
            f"meet the {module['identity_test']} test — and therefore whether the "
            "loss is denied — is a judgment call it does not make."
        ),
        "note": (
            f"{len(candidates)} acquisition(s) inside the "
            f"{module['window_days_before']}-day window around a "
            f"{loss:,.2f} loss. Treat this as a STOP, not as a determination."
        ),
    }


def _acq_row(acq: dict[str, Any], acq_date: date, sale_date: date) -> dict[str, Any]:
    from tools.asset_location import classify_account

    account = str(acq.get("account") or "")
    classified = classify_account(account)
    return {
        "symbol": str(acq.get("symbol") or "").upper(),
        "date": acq_date.isoformat(),
        "account": account or "Unknown",
        "shelter": classified.get("shelter"),
        "account_jurisdiction": classified.get("jurisdiction"),
        "in_registered_plan": classified.get("tax_class") in {"TAX_FREE", "TAX_DEFERRED"},
        "days_from_sale": (acq_date - sale_date).days,
        "side": "before" if acq_date < sale_date else ("after" if acq_date > sale_date
                                                       else "same_day"),
    }


def _consequence_note(module: dict[str, Any]) -> str:
    return {
        "added_to_basis_of_replacement": (
            "If triggered, the disallowed loss is added to the cost basis of the "
            "replacement shares — deferred, not destroyed."
        ),
        "added_to_affiliated_acb": (
            "If triggered, the denied loss is generally added to the AFFILIATED "
            "PERSON's adjusted cost base — deferred and recoverable later, not "
            "destroyed."
        ),
    }.get(module["disallowed_loss_treatment"], "Consequence not encoded.")


def _loss_amount(sale: dict[str, Any]) -> float | None:
    """Positive magnitude of a loss, 0 for a gain, None when it cannot be known."""
    gl = sale.get("gain_loss")
    if gl is not None:
        try:
            value = float(gl)
        except (TypeError, ValueError):
            return None
        return -value if value < 0 else 0.0

    proceeds, basis = sale.get("proceeds"), sale.get("cost_basis")
    if proceeds is None or basis is None:
        return None
    try:
        value = float(proceeds) - float(basis)
    except (TypeError, ValueError):
        return None
    return -value if value < 0 else 0.0


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Jurisdiction resolution — from the account, never from the profile
# ---------------------------------------------------------------------------
@log_exceptions()
def resolve_jurisdiction(account_name: str) -> dict[str, Any]:
    """The tax jurisdiction of ONE ACCOUNT, or an explicit failure to name it.

    **Tax jurisdiction is a property of the account, not of the profile.** One
    household can hold accounts in two countries, and `REGIONAL_LOCALE` is a
    DISPLAY locale — free text typed at install to pick a currency default.
    Deriving tax treatment from a language setting is the class of error 5.9
    found in the insider router and 4.7 shipped in its first shelter table.

    **Two sources, and what the user STATED wins (4.7a).**
    ``account_jurisdictions`` in user memory holds a country per account, entered
    at Context › Account Jurisdictions. Absent that, the account NAME is the
    evidence: TFSA/RRSP name Canada, Roth/401(k)/IRA name the US, ISA/SIPP name
    the UK. A name that identifies a class without a country ("Registered",
    "Pension", "Brokerage") resolves to None and FAILS CLOSED — as does an
    account the user has explicitly marked UNKNOWN, which is an ANSWER and is
    reported as one rather than as an open question.

    ``basis`` says which source answered, because a jurisdiction the user stated
    and one guessed from a substring are not equally good evidence for a rule
    that decides whether a loss is deferred or destroyed.
    """
    from tools.asset_location import classify_account

    classified = classify_account(account_name)
    jurisdiction = classified.get("jurisdiction")
    source = classified.get("jurisdiction_source")

    if not jurisdiction:
        declared = source == "declared_unknown"
        return {
            "account": account_name, "jurisdiction": None,
            "shelter": classified.get("shelter"),
            "tax_class": classified.get("tax_class"),
            "resolved": False,
            "basis": source,
            "note": (
                (f"{account_name!r} is on file with no jurisdiction you could name, "
                 "so no rule is applied to it. That is a recorded answer, not a gap."
                 ) if declared else
                (f"{account_name!r} names an account class without naming a country, "
                 "and no jurisdiction has been stated for it. Set one at Context › "
                 "Account Jurisdictions. Nothing is inferred from the profile's "
                 "display locale.")
            ),
        }
    stated = source == "stated"
    return {
        "account": account_name, "jurisdiction": jurisdiction,
        "shelter": classified.get("shelter"),
        "tax_class": classified.get("tax_class"),
        "resolved": True,
        "covered": jurisdiction in POLICY_MODULES,
        "policy_version": policy_version(jurisdiction),
        "basis": "stated on the account" if stated else "account name",
        # Reported even when the two agree: the disagreement is only meaningful
        # if a reader can see that both were consulted.
        "inferred_from_name": classified.get("jurisdiction_inferred"),
        "conflict": bool(classified.get("jurisdiction_conflict")),
        "basis_note": (
            "Stated by the user for this account, which is the authority here."
            if stated else
            "Derived from the account NAME — an inference from a shelter keyword, "
            "not a stated fact. Set the account's country at Context › Account "
            "Jurisdictions to replace it."
        ),
    }


# ---------------------------------------------------------------------------
# What the store can actually feed it
# ---------------------------------------------------------------------------
@log_exceptions()
def scan_dispositions(lookback_days: int = 400) -> dict[str, Any]:
    """Dated dispositions this app can actually prove, and why there are so few.

    Two sources, and both are nearly empty by construction:

      * `trade_journal` — the thesis log. It records entries and exits a human
        typed, and on the live profile it is `[]`.
      * 4.10a's classified reconciliation — a position decrease whose cause a
        human stated as a `trade`. An UNCLASSIFIED decrease is NOT offered here:
        it might be a sale, a transfer, a fee or a corporate action, and a rule
        engine over inferred transactions is this project's most-repeated mistake.

    `status: "no_data"` is the expected answer today and it means the store is
    empty, NOT that no disposition happened.
    """
    dispositions: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}

    dispositions.extend(_journal_dispositions(lookback_days, sources))
    dispositions.extend(_classified_dispositions(lookback_days, sources))

    if not dispositions:
        return {
            "status": "no_data",
            "dispositions": [],
            "sources": sources,
            "note": (
                "No dated disposition is on file. This is a statement about the "
                "RECORD, not about the portfolio: `trade_journal` holds what a "
                "human typed, and 4.10a's ledger holds only changes a human has "
                "classified as a trade. Neither infers a sale from a share count "
                "going down, and this engine will not either."
            ),
        }

    dispositions.sort(key=lambda d: d["date"], reverse=True)
    return {
        "status": "ready",
        "dispositions": dispositions,
        "count": len(dispositions),
        "sources": sources,
        "note": (
            f"{len(dispositions)} dated disposition(s). Completeness is bounded by "
            "what a human recorded — an unclassified position decrease is excluded "
            "on purpose and is listed in the reconciliation surface instead."
        ),
    }


def _journal_dispositions(lookback_days: int, sources: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from tools.trade_journal import get_trade_history

        history = get_trade_history() or []
    except Exception:  # noqa: BLE001
        sources["trade_journal"] = {"readable": False}
        return []

    cutoff = date.today() - timedelta(days=lookback_days)
    out: list[dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        exit_date = _as_date(entry.get("exit_date") or entry.get("closed_at"))
        if exit_date is None or exit_date < cutoff:
            continue
        # `log_trade` stores the entry price under `price` and the size under
        # `quantity`; `close_trade` adds `exit_price` and `exit_date`. Reading
        # `entry_price` (the obvious name, and the wrong one) silently yields
        # None on every row, which would make every closed thesis look like a
        # disposition whose result is unknown.
        entry_price = _num(entry.get("price")) or _num(entry.get("entry_price"))
        exit_price = _num(entry.get("exit_price"))
        shares = _num(entry.get("quantity")) or _num(entry.get("shares"))
        gain_loss = None
        if entry_price is not None and exit_price is not None and shares:
            gain_loss = (exit_price - entry_price) * shares
        out.append({
            "symbol": str(entry.get("symbol") or "").upper(),
            "date": exit_date.isoformat(),
            "account": entry.get("account") or "",
            "gain_loss": gain_loss,
            "source": "trade_journal",
        })

    sources["trade_journal"] = {"readable": True, "entries": len(history),
                                "dated_exits": len(out)}
    return out


def _classified_dispositions(lookback_days: int, sources: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from tools.portfolio_classification import apply_classifications
        from tools.portfolio_reconciliation import (
            detect_changes,
            is_cash,
            read_history,
            snapshot_dates,
        )

        rows = read_history()
        dates = snapshot_dates(rows)
    except Exception:  # noqa: BLE001
        sources["reconciliation"] = {"readable": False}
        return []

    cutoff = date.today() - timedelta(days=lookback_days)
    changes: list[dict[str, Any]] = []
    for prior, current in zip(dates, dates[1:]):
        stamp = _as_date(current)
        if stamp is None or stamp < cutoff:
            continue
        changes.extend(detect_changes(prior, current, rows))

    annotated = apply_classifications(changes)
    out = [
        {
            "symbol": c["symbol"],
            "date": c["current_date"],
            "account": c.get("account") or "",
            # Deliberately absent: this store records SHARES, so it cannot say
            # whether the sale realised a loss. `check_disposition` returns
            # `unknown_result` on it rather than assuming either way.
            "gain_loss": None,
            "shares_disposed": abs(float(c.get("delta") or 0.0)),
            "source": "reconciliation (cause stated by user)",
            "spans_gap": c.get("spans_gap"),
        }
        for c in annotated
        if c.get("cause") == "trade" and not is_cash(c["symbol"])
        and float(c.get("delta") or 0.0) < 0
    ]

    unclassified = sum(1 for c in annotated if not c.get("classified"))
    sources["reconciliation"] = {
        "readable": True, "snapshots": len(dates), "changes_seen": len(annotated),
        "stated_trades_out": len(out), "unclassified_excluded": unclassified,
        "note": (f"{unclassified} observed change(s) have no stated cause and are "
                 "EXCLUDED. An unclassified decrease is not a sale."
                 if unclassified else "Every observed change has a stated cause."),
    }
    return out


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


# ---------------------------------------------------------------------------
# The pre-trade gate (3.8 P2)
# ---------------------------------------------------------------------------
@log_exceptions()
def precheck_rebuy(symbol: str, account: str,
                   proposed_date: str | None = None) -> dict[str, Any]:
    """Whether a proposed BUY of `symbol` into `account` can proceed (3.8 P2).

    This is the gate a TRIM-and-rebuy proposal has to clear before it ships. It
    returns `allowed: False` in three cases and only one of them is a rule hit:

      * the account's jurisdiction cannot be named → fail closed;
      * this engine has no module for that jurisdiction → `not_covered`, BLOCKS;
      * a recorded loss on the same security sits inside the window → candidate.

    A missing transaction record makes this gate WEAK, not passing, and
    `evidence_complete` says so on every payload. An empty store cannot clear a
    trade; it can only fail to object.
    """
    proposed = _as_date(proposed_date) or date.today()
    resolution = resolve_jurisdiction(account)

    if not resolution["resolved"]:
        return {
            "allowed": False, "reason": "jurisdiction_unresolved",
            "engine_version": ENGINE_VERSION,
            "account": account, "symbol": str(symbol or "").upper(),
            "resolution": resolution,
            "note": (
                "The account's tax jurisdiction could not be named from its name, "
                "and this engine fails closed rather than defaulting. Name the "
                "account for its shelter (e.g. 'RRSP', 'Roth IRA') or add a tax "
                "residency to it."
            ),
        }

    jurisdiction = resolution["jurisdiction"]
    if jurisdiction not in POLICY_MODULES:
        return {**_not_covered(jurisdiction,
                               f"No policy module for {jurisdiction}."),
                "allowed": False, "reason": "not_covered",
                "account": account, "symbol": str(symbol or "").upper()}

    scan = scan_dispositions()
    module = POLICY_MODULES[jurisdiction]
    window_start = proposed - timedelta(days=module["window_days_after"])
    window_end = proposed + timedelta(days=module["window_days_before"])

    relevant = [
        d for d in scan.get("dispositions", [])
        if d["symbol"] == str(symbol or "").upper()
        and (dt := _as_date(d["date"])) and window_start <= dt <= window_end
    ]

    evidence_complete = scan.get("status") == "ready"
    base = {
        "symbol": str(symbol or "").upper(),
        "account": account,
        "jurisdiction": jurisdiction,
        "rule_name": module["rule_name"],
        "policy_version": module["version"],
        "engine_version": ENGINE_VERSION,
        "proposed_date": proposed.isoformat(),
        "window": {"from": window_start.isoformat(), "to": window_end.isoformat()},
        "advice_ready": bool(module["professional_review"]["reviewed"]),
        "evidence_complete": evidence_complete,
        "evidence_note": (
            scan.get("note") if not evidence_complete else
            f"Checked against {scan.get('count')} recorded disposition(s)."
        ),
        "sources": scan.get("sources", {}),
    }

    if relevant:
        return {**base, "allowed": False, "reason": "repurchase_window",
                "dispositions_in_window": relevant,
                "identity_test": module["identity_test"],
                "identity_note": module["identity_note"],
                "consequence_if_triggered": _consequence_note(module),
                "note": (
                    f"{len(relevant)} recorded disposition(s) of {symbol} fall inside "
                    f"the {module['rule_name'].lower()} window around this buy. "
                    "Whether the loss is denied depends on whether it WAS a loss and "
                    "on the identity test — neither of which this engine decides. "
                    "Stop and check."
                )}

    return {
        **base, "allowed": True, "reason": "no_recorded_conflict",
        "note": (
            "No recorded disposition of this security falls inside the window. "
            + ("" if evidence_complete else
               "NOTE this is a weak pass: the transaction record is empty or "
               "partial, so the engine did not clear the trade — it failed to "
               "object. Do not present it as a confirmation that no wash sale "
               "applies.")
        ),
    }


if __name__ == "__main__":  # pragma: no cover — operator convenience
    import json

    print(json.dumps(coverage_matrix(), indent=2))
    print(json.dumps(scan_dispositions(), indent=2, default=str))
