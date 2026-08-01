"""Profile readiness — Advisor Roadmap 2.8.

Every input a shipped engine depends on that only the USER can state, in one
read surface: whether it is on file, and **which shipped feature is inert while
it is not**.

Why it exists: five times now a slice has shipped complete and correct and then
sat dark because the store behind it was empty — 3.7's drawdown playbook, 4.4's
drift band, 2.2's IPS caps, 1.5b's few-shot pool, and (until 2026-07-26) the
wealth goal. Every one of those times the app was RIGHT to refuse to author the
value: a number the software invents gets read back later as the user's own
stated rule, carrying the authority of a promise they made to themselves. And
every one of those times the cost landed as an invisible blank nobody was told
about. 2.5/2.6 made an engine prove it RAN and PRODUCED; this makes a STORE
prove someone FILLED it.

THE CONTRACT, which is the whole point of the item: this module reports
emptiness and names the consequence. It never authors, defaults, suggests,
exemplifies or ranges a value for any input it reports on — not as a
placeholder, not as an "e.g.", not as a "most people". The only things it may
add to a blank are WHERE the value is stated and WHAT stops working until it is.
``tests/test_tools/test_profile_readiness.py`` enforces that by scanning every
string this module emits for an input that is not fully stated.

It is also deliberately not a nag: a blank is a valid answer, and the surface
says so. Nothing here is scored, ranked, or chased.

Three design rules worth keeping if this list grows:

1. **Read through the accessor the engine reads through** (``get_playbook``,
   ``load_ips_constraints``, ``get_financial_goal``, ``get_high_quality_interactions``),
   never straight out of the JSON. A surface that re-derives "is it set" can
   disagree with the consumer it is reporting on, and then it lies in the
   direction that hurts — "stated" while the gate sees nothing.
2. **Count the thing the CONSUMER reads.** The feedback store's own ``total``
   said 100 while the few-shot pool it feeds held zero; the pool is the number
   that belongs on this page.
3. **Prose names capabilities, not code.** Roadmap numbers, endpoints and
   function names go in ``roadmap`` and in these docstrings — never in a line
   the user reads. The volume of this surface is proportional to the number of
   blanks, so it is loudest on a fresh profile: every row carries a one-line
   ``cost`` for the collapsed view, and ``inert`` is the per-field detail the
   page keeps behind a disclosure.
"""
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Status vocabulary. "unreadable" is distinct from "empty" on purpose: a store
# that could not be read is UNVERIFIED, and reporting it as empty would be this
# surface inventing a fact about the user in the one place that must not.
#
# "not_stated" is the fourth word and the newest, and it is distinct from "empty"
# for the mirror-image reason: an OPTIONAL input nobody has stated is COMPLETE.
# Every other row here reports a blank that switches a shipped feature off, so
# "empty" carries a cost by construction; a row whose blank costs nothing needs
# its own word, or the page starts chasing an answer it has no business chasing
# and this surface becomes the nag it says it is not.
STATUS_SET = "set"
STATUS_PARTIAL = "partial"
STATUS_EMPTY = "empty"
STATUS_NOT_STATED = "not_stated"
STATUS_UNREADABLE = "unreadable"

# Rows whose blank is not a gap. Kept as a set rather than a flag on the row so
# the roll-up below can be read in one place: an optional row never lands in the
# "stated / of" figure and never darkens a capability.
OPTIONAL_KEYS = frozenset({"secular_themes"})

CONTRACT = (
    "A blank is a valid answer. Nothing here is filled in on your behalf — the page "
    "reports what is on file and what stays switched off while it is not."
)


# ---------------------------------------------------------------------------
# Drawdown playbook (3.7's store; 4.4 and 3.9 read out of it too)
# ---------------------------------------------------------------------------

# Field -> the shipped thing that cannot fire without it. One consequence per
# FIELD, not per block: the playbook is four independent instructions and a
# half-written one leaves a specific half dark.
_PLAYBOOK_CONSEQUENCES = {
    "never_sell": (
        "A deep-drawdown alert has no hold list of yours to read back to you."
    ),
    "buy_first": (
        "A deep-drawdown alert cannot say what your new contributions buy first."
    ),
    "deployment_levels": (
        "A deep-drawdown alert has no ladder of yours to mark live, and the rungs that "
        "would arm as live triggers do not exist. The sentinel evaluates the ladder on "
        "every tick and reports nothing on file."
    ),
    "rebalance_drift_pct": (
        "The drift check reports itself unavailable. The band is your own trigger and "
        "the app supplies none. Note the band is only HALF of what that check needs — "
        "a target allocation says what to drift FROM, and it has its own row below."
    ),
}

# Field -> the capability names that go dark with it, for the one-line summary
# the page shows before the detail above is expanded. Deliberately overlapping:
# three of these fields feed the same alert, and the summary collapses that
# rather than saying it three times.
_PLAYBOOK_CAPABILITIES = {
    "never_sell": ("deep-drawdown alerts",),
    "buy_first": ("deep-drawdown alerts",),
    "deployment_levels": ("deep-drawdown alerts", "the automated deployment ladder"),
    "rebalance_drift_pct": ("drift checks",),
}

_PLAYBOOK_FEEDS = (
    "The sentinel reads these rules back to you when the tape falls deep.",
    "The drift check measures your book against your stored drift band.",
    "The intraday sentinel arms these rungs against peak-to-date drawdown and delivers "
    "your pre-committed action once, at the level you named.",
)

# Roadmap identity stays in the payload and out of the prose: it is how a
# developer traces a row back to the slice that reads it, and it is noise to
# the person deciding whether to fill the box in.
_PLAYBOOK_ROADMAP = ("3.7", "3.9", "4.4")


def _playbook_input() -> dict[str, Any]:
    from tools.drawdown_playbook import get_playbook

    playbook = get_playbook() or {}
    stated = {k: v for k, v in playbook.items() if v not in (None, "", [], {})}
    missing = [key for key in _PLAYBOOK_CONSEQUENCES if key not in stated]
    return _input(
        key="drawdown_playbook",
        label="Drawdown playbook",
        stated=stated,
        missing=missing,
        required=list(_PLAYBOOK_CONSEQUENCES),
        consequences=_PLAYBOOK_CONSEQUENCES,
        capabilities=_PLAYBOOK_CAPABILITIES,
        roadmap=_PLAYBOOK_ROADMAP,
        feeds=_PLAYBOOK_FEEDS,
        entry="Context › Drawdown Playbook",
        # "notes" carries no consequence — it is a message to your future self,
        # so its absence breaks nothing and is not reported as a gap.
        optional_present=sorted(k for k in stated if k not in _PLAYBOOK_CONSEQUENCES),
    )


# ---------------------------------------------------------------------------
# risk_constraints (2.2's caps; 4.4's bounds)
# ---------------------------------------------------------------------------

_CONSTRAINT_CONSEQUENCES = {
    "max_position_pct": (
        "The pre-trade check has no single-name size cap of yours to enforce on a sized "
        "draft, and the optimizer bounds no single name."
    ),
    "max_fund_position_pct": (
        "The pre-trade check has no fund/ETF size cap of yours to enforce, and the "
        "optimizer bounds no fund position."
    ),
    "max_sector_pct": (
        "The pre-trade check has no sector exposure cap of yours to enforce, and the "
        "optimizer solves with sector exposure unbounded."
    ),
    "max_risk_per_trade_pct": (
        "The pre-trade check has no per-trade risk cap of yours to enforce on a draft "
        "that carries a stop."
    ),
}

_CONSTRAINT_CAPABILITIES = {
    "max_position_pct": ("the pre-trade check", "the optimizer's bounds"),
    "max_fund_position_pct": ("the pre-trade check", "the optimizer's bounds"),
    "max_sector_pct": ("the pre-trade check", "the optimizer's bounds"),
    "max_risk_per_trade_pct": ("the pre-trade check",),
}

_CONSTRAINT_FEEDS = (
    "The pre-trade check gates every sized draft against your stated caps before the "
    "judge sees it.",
    "The optimizer bounds each name and each sector by your stated caps.",
)

_CONSTRAINT_ROADMAP = ("2.2", "4.4")


def _risk_constraints_input() -> dict[str, Any]:
    from tools.ips_precheck import execution_readiness, load_ips_constraints, stated_caps

    constraints = load_ips_constraints()
    stated: dict[str, Any] = dict(stated_caps(constraints))
    restricted = list(constraints.get("restricted_symbols") or [])
    readiness = execution_readiness(constraints)
    # An axis the user has confirmed they want unlimited is ANSWERED, not blank.
    # Reporting it as a gap would make the confirmation worth nothing — the row
    # would keep charging a cost for a decision the user already made, which is
    # the nag this whole surface is built not to be. Only the never-asked axes
    # are missing.
    by_choice = list(readiness.get("unconstrained_by_choice") or [])
    missing = [
        key for key in _CONSTRAINT_CONSEQUENCES
        if key not in stated and key not in by_choice
    ]

    entry = _input(
        key="risk_constraints",
        label="Risk limits",
        stated=stated,
        missing=missing,
        required=list(_CONSTRAINT_CONSEQUENCES),
        consequences=_CONSTRAINT_CONSEQUENCES,
        capabilities=_CONSTRAINT_CAPABILITIES,
        roadmap=_CONSTRAINT_ROADMAP,
        feeds=_CONSTRAINT_FEEDS,
        entry="Context › Risk Limits",
        answered=by_choice,
    )
    # An empty restricted list is a COMPLETE answer ("I restrict nothing"), so it
    # is reported and never counted as a gap. Nothing is switched off by it. The
    # confirmed-unlimited axes ride alongside it for the same reason, and in
    # `observed` rather than `stated` because no figure was stated — the user
    # answered the question without naming a number.
    entry["observed"] = {"restricted_symbols": restricted}
    if by_choice:
        entry["observed"]["unconstrained_by_choice"] = by_choice
    return entry


# ---------------------------------------------------------------------------
# Wealth goal (4.5's target; 3.7 pairs with it)
# ---------------------------------------------------------------------------

# Matches build_goal_projection's own required set exactly — target_high is the
# stretch figure and the projection runs without it.
_GOAL_CONSEQUENCES = {
    "target_low": (
        "The goal panel has no target to project against and no required return to "
        "compute, so it reports itself unavailable."
    ),
    "horizon_years": (
        "The goal panel reports itself unavailable: a target with no runway cannot be "
        "scored."
    ),
    "annual_contribution": (
        "The goal panel reports itself unavailable, and a deep-drawdown alert drops the "
        "line telling you whether the plan still funds from here."
    ),
}

_GOAL_CAPABILITIES = {
    "target_low": ("the goal projection",),
    "horizon_years": ("the goal projection",),
    "annual_contribution": (
        "the goal projection",
        "the drawdown alert's still-on-track line",
    ),
}

_GOAL_FEEDS = (
    "The goal panel's Monte Carlo bands, goal-funded rate and required return.",
    "The drawdown alert's answer to 'does the plan still work from here'.",
)

_GOAL_ROADMAP = ("3.7", "4.5")


def _wealth_goal_input() -> dict[str, Any]:
    from tools.memory import get_financial_goal

    goal = get_financial_goal() or {}
    stated = {
        k: v
        for k, v in goal.items()
        if k != "currency" and v not in (None, "", [], {})
    }
    missing = [key for key in _GOAL_CONSEQUENCES if key not in stated]

    entry = _input(
        key="wealth_goal",
        label="Wealth goal",
        stated=stated,
        missing=missing,
        required=list(_GOAL_CONSEQUENCES),
        consequences=_GOAL_CONSEQUENCES,
        capabilities=_GOAL_CAPABILITIES,
        roadmap=_GOAL_ROADMAP,
        feeds=_GOAL_FEEDS,
        entry="Context › Wealth Goal",
        optional_present=["target_high"] if "target_high" in stated else [],
    )
    if goal.get("currency"):
        entry["observed"] = {"currency": goal["currency"]}
    return entry


# ---------------------------------------------------------------------------
# Target allocation (4.4's drift target)
# ---------------------------------------------------------------------------
# Added 2026-07-30, and it is a correction as much as an addition. This surface
# already reported "drift checks" as switched off by the playbook's
# `rebalance_drift_pct` alone — which meant that filling the band in read as
# turning the check ON, and it does not. `check_rebalance_drift` needs a target
# to measure against as well, and until today that target had no store, no
# endpoint and no entry screen anywhere: it existed only as a function
# parameter. So this row exists for the reason 2.8 itself exists, applied to
# 2.8's own output.

_TARGET_ALLOCATION_CONSEQUENCES = {
    "weights": (
        "The drift check reports itself unavailable — there is nothing to measure your "
        "book against. A target allocation is never inferred from what you happen to "
        "hold: that would make every book perfectly on target by definition."
    ),
}

_TARGET_ALLOCATION_CAPABILITIES = {
    "weights": ("drift checks",),
}

_TARGET_ALLOCATION_FEEDS = (
    "The mix your holdings are compared against when you ask whether to rebalance.",
)

_TARGET_ALLOCATION_ROADMAP = ("4.4",)


def _target_allocation_input() -> dict[str, Any]:
    from tools.memory import get_target_allocation_record

    record = get_target_allocation_record()
    weights = (record or {}).get("weights") or {}

    entry = _input(
        key="target_allocation",
        label="Target allocation",
        stated={"weights": f"{len(weights)} sleeves"} if weights else {},
        missing=[] if weights else ["weights"],
        required=list(_TARGET_ALLOCATION_CONSEQUENCES),
        consequences=_TARGET_ALLOCATION_CONSEQUENCES,
        capabilities=_TARGET_ALLOCATION_CAPABILITIES,
        roadmap=_TARGET_ALLOCATION_ROADMAP,
        feeds=_TARGET_ALLOCATION_FEEDS,
        entry="Context › Target Allocation",
    )
    # The sleeves themselves, so the row can be read without opening the editor.
    # In `observed` rather than `stated` because the count is what gates the
    # capability; the individual weights are detail.
    if weights:
        entry["observed"] = {
            "sleeves": sorted(weights),
            "total_pct": (record or {}).get("total_pct"),
        }
    return entry


# ---------------------------------------------------------------------------
# Account jurisdictions (4.7a)
#
# The only row here whose gap is PER-ACCOUNT rather than per-field, and the
# reason it needs its own shape: a profile can be fully answered for three
# accounts and silently skipping every tax rule on the fourth. A single
# set/empty verdict cannot say that, so the count of accounts still failing
# closed is what drives the status.
#
# It is also the only row with a working fallback. An account named "TFSA"
# resolves to Canada with nobody typing anything, so this store is not a switch
# between working and dark — it is the difference between a rule applied on the
# user's statement and one applied on a substring match that has twice been
# wrong in a way that inverted the answer. The consequence line says that,
# rather than claiming the feature is off.

_ACCOUNT_JURISDICTION_CONSEQUENCES = {
    "jurisdictions": (
        "Every jurisdiction-specific tax check is SKIPPED on the accounts with no country "
        "on file — the withholding check on a tax-free shelter, and the loss-deferral "
        "rules behind a sell-and-rebuy. They are skipped rather than guessed, because a "
        "rule from the wrong country does not merely miss, it inverts: the Canadian TFSA "
        "treatment charged to a US Roth invents a leak that does not exist."
    ),
}

_ACCOUNT_JURISDICTION_CAPABILITIES = {
    "jurisdictions": ("asset-location scoring", "the sell-and-rebuy tax check"),
}

_ACCOUNT_JURISDICTION_FEEDS = (
    "Which country's tax rules apply to each account — the thing that decides whether "
    "a dividend in a shelter is taxed at source, and whether a repurchase defers a loss "
    "or destroys it.",
)

_ACCOUNT_JURISDICTION_ROADMAP = ("4.7a",)


def _account_jurisdictions_input() -> dict[str, Any]:
    from tools.asset_location import portfolio_account_jurisdictions

    view = portfolio_account_jurisdictions()
    counts = view.get("counts") or {}
    needed = int(counts.get("need_jurisdiction") or 0)
    unanswered = int(counts.get("unanswered") or 0)
    stated = int(counts.get("stated") or 0)
    declared = int(counts.get("declared_unknown") or 0)
    inferred = int(counts.get("inferred_from_name") or 0)

    listed = len(view.get("accounts") or [])

    # THREE cases, and the first is the one that must not read as success.
    #
    # No account could be listed at all — an empty or unreadable portfolio — so
    # nothing about these checks has been VERIFIED. Reported as a gap, because
    # the alternative is a lit capability standing on a portfolio nobody read,
    # which is the failure this whole surface exists to catch.
    #
    # Accounts listed and none of them sheltered IS an answer: income taxed at
    # the marginal rate has no shelter rule to get wrong, and reporting a gap
    # there would invent a requirement.
    if listed == 0:
        composition = ""
        gap = True
    elif needed == 0:
        composition = "no sheltered accounts — nothing to place"
        gap = False
    else:
        parts = []
        if stated:
            parts.append(f"{stated} stated")
        if declared:
            parts.append(f"{declared} declared unknown")
        if inferred:
            parts.append(f"{inferred} from the account name")
        if unanswered:
            parts.append(f"{unanswered} unanswered")
        composition = " · ".join(parts)
        gap = bool(unanswered)

    entry = _input(
        key="account_jurisdictions",
        label="Account jurisdictions",
        stated={"jurisdictions": composition} if composition else {},
        missing=["jurisdictions"] if gap else [],
        required=list(_ACCOUNT_JURISDICTION_CONSEQUENCES),
        consequences=_ACCOUNT_JURISDICTION_CONSEQUENCES,
        capabilities=_ACCOUNT_JURISDICTION_CAPABILITIES,
        roadmap=_ACCOUNT_JURISDICTION_ROADMAP,
        feeds=_ACCOUNT_JURISDICTION_FEEDS,
        entry="Context › Account Jurisdictions",
    )
    entry["observed"] = {
        "accounts": [
            {
                "account": a["account"],
                "jurisdiction": a["jurisdiction"],
                "source": a["source"],
            }
            for a in view.get("accounts") or []
            if a.get("jurisdiction_needed")
        ],
        # Named separately because they are the two states that look fine from
        # the summary line and are not: a name-inferred country nobody confirmed,
        # and a stored entry that matches no account the portfolio holds.
        "inferred_from_name": [
            a["account"] for a in view.get("accounts") or []
            if a.get("jurisdiction_needed") and a.get("source") == "inferred_from_name"
        ],
        "conflicts": [
            a["account"] for a in view.get("accounts") or [] if a.get("conflict")
        ],
        "stated_unmatched": view.get("stated_unmatched") or [],
        "set_at": view.get("set_at"),
    }
    return entry


# ---------------------------------------------------------------------------
# Secular themes (3.1's Stage-1 overlay) — the one OPTIONAL row
# ---------------------------------------------------------------------------
# Added 2026-07-31 alongside the entry screen that finally made the store
# writable. It earns a row for the reason every row here exists — the store had
# no writer, so nobody could tell a user who declined to state a theme from a
# user who was never able to — and it breaks the mould for a reason worth
# spelling out, because the temptation is to make it look like the others.
#
# Nothing goes dark without a theme. The daily priority still runs, still ranks,
# still recommends; it simply weighs every holding as a tactical position. That
# is a DIFFERENT DEFAULT, not a switched-off feature, and stating a theme is the
# user choosing to raise the bar on part of their book rather than repairing
# something broken. So this row carries no cost line, darkens no capability, and
# is excluded from the "stated" count: a profile with no theme is complete.
#
# The inverse error is the one that made the store dangerous in the first place —
# treating the blank as a statement. It is not read as "the user holds no
# long-term conviction" here any more than it is anywhere else.

_SECULAR_THEMES_CONSEQUENCE = (
    "A blank here is a complete answer and nothing is switched off by it. With no "
    "conviction on record every holding is weighed as a tactical position, so a name "
    "you mean to hold for a decade can be put up for trimming on the same evidence as "
    "any other. Naming one raises that bar to the exit rules you write for it, and "
    "those rules are the only thing that can clear it."
)

_SECULAR_THEMES_FEEDS = (
    "The long-term convictions your daily priority is read against, and the exit rules "
    "that decide when one of them may be cut.",
)

_SECULAR_THEMES_ROADMAP = ("3.1",)


def _secular_themes_input() -> dict[str, Any]:
    from tools.memory import get_secular_themes

    themes = get_secular_themes()

    return {
        "key": "secular_themes",
        "label": "Structural convictions",
        # Never STATUS_EMPTY. That word is spoken for by the rows whose blank
        # costs something, and the page paints it as a failure.
        "status": STATUS_SET if themes else STATUS_NOT_STATED,
        "authored_by": "you",
        # A count, not the themes themselves — those are the user's own words and
        # ride in `observed`, which the contract test excludes from its scan for
        # exactly this reason.
        "stated": {"themes": len(themes)} if themes else {},
        "observed": {
            "themes": [str(t.get("theme") or "") for t in themes],
            "set_at": max((str(t.get("set_at") or "") for t in themes), default=""),
        },
        # Nothing is required, so nothing can be missing. The page draws its strip
        # of field marks from these two lists and correctly draws none here.
        "required": [],
        "optional_present": ["themes"] if themes else [],
        "answered_blank": [],
        # Carried even though `missing` is empty, because this is the sentence the
        # editor shows while you decide whether to fill the box in — the case
        # `inert` structurally cannot cover. Same reason the limits editor reads
        # this map rather than keeping its own copy of the prose.
        "consequence_by_field": {"themes": _SECULAR_THEMES_CONSEQUENCE},
        "missing": [],
        "feeds": list(_SECULAR_THEMES_FEEDS),
        "inert": [],
        # Empty in BOTH directions, and deliberately so. Listing a capability here
        # would put it on the switchboard, where it would read as lit or dark —
        # and neither is true of a feature that works the same either way.
        "capabilities": [],
        "capabilities_dark": [],
        "cost": "",
        "roadmap": list(_SECULAR_THEMES_ROADMAP),
        "entry": "Context › Structural Convictions",
    }


# ---------------------------------------------------------------------------
# Feedback ratings (1.5's store; 1.5b's few-shot pool)
# ---------------------------------------------------------------------------

_FEEDBACK_FEEDS = (
    "The pool of past answers a tuned prompt draws its worked cases from.",
)

_FEEDBACK_CONSEQUENCE = (
    "The pool holds nothing, so a prompt tuned against your best answers would draw "
    "from zero rows. Capture is running; it is the rating half that is unfilled, and "
    "an unrated store is an EMPTY pool rather than a small one."
)

_FEEDBACK_CAPABILITY = ("tuning on your best answers",)

_FEEDBACK_ROADMAP = ("1.5", "1.5b")


def _feedback_ratings_input() -> dict[str, Any]:
    from tools.feedback import get_high_quality_interactions, get_statistics

    stats = get_statistics() or {}
    captured = int(stats.get("total") or 0)
    rated = int(stats.get("rated") or 0)
    # The pool, not the row count: this is the number the CONSUMER reads, and the
    # one that was silently zero while `total` looked healthy.
    pool = len(get_high_quality_interactions() or [])

    if pool:
        status = STATUS_SET
    elif rated:
        # Ratings exist but none clears the pool's bar. Real, and not a blank.
        status = STATUS_PARTIAL
    else:
        status = STATUS_EMPTY

    return {
        "key": "feedback_ratings",
        "label": "Answer ratings",
        "status": status,
        "authored_by": "you",
        "stated": {},
        "observed": {"captured": captured, "rated": rated, "high_quality_pool": pool},
        # This row is measured rather than typed, but it carries the same shape as
        # the authored ones: the page renders one strip of field marks for every
        # row, and a row missing the key would silently render as having nothing
        # to state — the opposite of what an empty pool means.
        "required": ["rated_interactions"],
        "optional_present": [],
        "answered_blank": [],
        "consequence_by_field": {"rated_interactions": _FEEDBACK_CONSEQUENCE},
        "missing": [] if pool else ["rated_interactions"],
        "feeds": list(_FEEDBACK_FEEDS),
        "inert": [] if pool else [_FEEDBACK_CONSEQUENCE],
        "capabilities": list(_FEEDBACK_CAPABILITY),
        "capabilities_dark": [] if pool else list(_FEEDBACK_CAPABILITY),
        "cost": "" if pool else _cost_line(list(_FEEDBACK_CAPABILITY)),
        "roadmap": list(_FEEDBACK_ROADMAP),
        "entry": "The thumbs-up / thumbs-down control on any chat answer.",
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _cost_line(capability_names: list[str]) -> str:
    """The single line a collapsed row shows in place of the per-field detail.

    Phrased as a list rather than a sentence with a verb, so one capability and
    four read the same way and no agreement bug can creep in. Empty when nothing
    is switched off — a row with no cost says nothing rather than "nothing".
    """
    if not capability_names:
        return ""
    return "Switched off while blank: " + ", ".join(capability_names) + "."


def _input(
    key: str,
    label: str,
    stated: dict[str, Any],
    missing: list[str],
    required: list[str],
    consequences: dict[str, str],
    capabilities: dict[str, tuple[str, ...]],
    roadmap: tuple[str, ...],
    feeds: tuple[str, ...],
    entry: str,
    optional_present: list[str] | None = None,
    answered: list[str] | None = None,
) -> dict[str, Any]:
    """One row: what is on file, what is not, and what the gap switches off.

    `answered` is the third state, and it took a real store to discover it:
    a field can be deliberately left blank as the user's actual answer ("no
    limit here, and I mean it"). That is neither stated nor missing. Counting it
    as stated would put a figure in their mouth; counting it as missing would
    keep charging a cost for a question they have already closed.
    """
    answered = list(answered or [])
    if not stated and not answered:
        status = STATUS_EMPTY
    elif missing:
        status = STATUS_PARTIAL
    else:
        status = STATUS_SET

    # Order-preserving dedupe: several fields commonly feed one capability, and
    # naming it once is the whole point of the summary line.
    lost: list[str] = []
    for field in missing:
        for name in capabilities.get(field, ()):
            if name not in lost:
                lost.append(name)

    # Every capability this store feeds, dark or lit. The page draws a switchboard
    # from it, and a name that only appeared once it went dark would leave the LIT
    # set unknowable — you could not tell a capability that is working from one
    # this surface does not track.
    fed: list[str] = []
    for field in required:
        for name in capabilities.get(field, ()):
            if name not in fed:
                fed.append(name)

    return {
        "key": key,
        "label": label,
        "status": status,
        "authored_by": "you",
        # Verbatim from the store. These are the USER's figures — echoing them
        # back is the opposite of authoring one, and it is what proves this
        # surface can tell a stated value from a blank.
        "stated": stated,
        "missing": list(missing),
        "required": list(required),
        "optional_present": list(optional_present or []),
        # The page draws a distinct mark for these: filled would claim a figure
        # that does not exist, hollow would claim an open question that is
        # closed. Neither is true, so it gets its own.
        "answered_blank": answered,
        "feeds": list(feeds),
        # The same sentences, keyed by field instead of flattened. An editor
        # showing "what this box costs" while you type needs the consequence for
        # a field that is NOT currently missing, which `inert` by construction
        # cannot supply — and the alternative, a second copy of the prose in the
        # template, is how a page starts telling the user a different story from
        # the report underneath it.
        "consequence_by_field": dict(consequences),
        "inert": [consequences[k] for k in missing if k in consequences],
        "capabilities": fed,
        "capabilities_dark": lost,
        # The collapsed view reads `cost`; expanding it reveals `inert`. Both
        # are built from the same `missing` list, so they cannot disagree.
        "cost": _cost_line(lost),
        "roadmap": list(roadmap),
        "entry": entry,
    }


_BUILDERS = (
    _playbook_input,
    _risk_constraints_input,
    _target_allocation_input,
    _account_jurisdictions_input,
    _wealth_goal_input,
    _secular_themes_input,
    _feedback_ratings_input,
)

# Fallback identity for a builder that throws, so a broken read still occupies a
# row instead of vanishing from the list — a missing row would read as "no such
# requirement", which is the failure mode this whole surface exists to end.
_UNREADABLE_LABELS = {
    "_playbook_input": ("drawdown_playbook", "Drawdown playbook"),
    "_risk_constraints_input": ("risk_constraints", "Risk limits"),
    "_target_allocation_input": ("target_allocation", "Target allocation"),
    "_account_jurisdictions_input": ("account_jurisdictions", "Account jurisdictions"),
    "_wealth_goal_input": ("wealth_goal", "Wealth goal"),
    "_secular_themes_input": ("secular_themes", "Structural convictions"),
    "_feedback_ratings_input": ("feedback_ratings", "Answer ratings"),
}


def _unreadable(builder_name: str, error: Exception) -> dict[str, Any]:
    key, label = _UNREADABLE_LABELS.get(builder_name, (builder_name, builder_name))
    logger.warning(f"profile readiness: {key} could not be read: {error}")
    return {
        "key": key,
        "label": label,
        "status": STATUS_UNREADABLE,
        "authored_by": "you",
        "stated": {},
        "missing": [],
        "required": [],
        "optional_present": [],
        "answered_blank": [],
        "consequence_by_field": {},
        "feeds": [],
        # Left empty deliberately, and NOT folded into the dark set: which
        # capabilities this store feeds is knowable, but whether they are lit is
        # not, and a switchboard that marked them dark would be asserting a fact
        # about the user's data that nobody read. The summary counts the
        # unverified STORE instead.
        "capabilities": [],
        "capabilities_dark": [],
        "inert": [
            "This store could not be read, so whether you have stated it is UNKNOWN. "
            "Treat it as unverified rather than as empty."
        ],
        "cost": "Unverified — this store could not be read.",
        "roadmap": [],
        "entry": "",
    }


def build_profile_readiness() -> dict[str, Any]:
    """Every user-authored input the engines depend on, and what each blank costs.

    Never raises: one unreadable store must not take the whole surface down, or
    the instrument built to catch silent gaps acquires one of its own.
    """
    inputs = []
    for builder in _BUILDERS:
        try:
            inputs.append(builder())
        except Exception as e:  # noqa: BLE001 — a bad store yields a row, not a 500
            inputs.append(_unreadable(builder.__name__, e))

    required_rows = [i for i in inputs if i["key"] not in OPTIONAL_KEYS]
    counts = {
        "total": len(inputs),
        STATUS_SET: sum(1 for i in inputs if i["status"] == STATUS_SET),
        STATUS_PARTIAL: sum(1 for i in inputs if i["status"] == STATUS_PARTIAL),
        STATUS_EMPTY: sum(1 for i in inputs if i["status"] == STATUS_EMPTY),
        STATUS_NOT_STATED: sum(1 for i in inputs if i["status"] == STATUS_NOT_STATED),
        STATUS_UNREADABLE: sum(1 for i in inputs if i["status"] == STATUS_UNREADABLE),
        # The "N of M stated" figure, and the reason it is not `set` over `total`:
        # an optional row in that fraction would report a complete profile as
        # incomplete forever, since the only way to close it is to state a
        # conviction the user may simply not have. `total` stays the row count so
        # nothing reading it has to know which rows are optional.
        "required_total": len(required_rows),
        "required_set": sum(1 for i in required_rows if i["status"] == STATUS_SET),
    }
    # The switchboard: distinct capabilities across every store, split by whether
    # anything is dark. Deduped across rows AND within them, which `inert_count`
    # below is not — that figure counts consequence SENTENCES, one per blank
    # field, and three playbook fields all name the same alert. Reporting it as a
    # capability count inflated a fully-blank playbook from three dark features to
    # four. Same row-versus-thing error as the recommendation ledger's.
    all_caps: list[str] = []
    dark: list[str] = []
    for entry in inputs:
        for name in entry.get("capabilities") or []:
            if name not in all_caps:
                all_caps.append(name)
        # Dark wins over lit: a capability fed by two stores needs both, so one
        # blank is enough to switch it off.
        for name in entry.get("capabilities_dark") or []:
            if name not in dark:
                dark.append(name)

    return {
        "generated_at": datetime.now().isoformat(),
        "counts": counts,
        # Kept as-is and still named for what it counts: the weekly review reads
        # it as the volume of consequence prose it is about to summarize.
        "inert_count": sum(len(i["inert"]) for i in inputs),
        "capabilities": {
            "total": len(all_caps),
            "live": [name for name in all_caps if name not in dark],
            "dark": dark,
            # Whether an unreadable store's capabilities are lit is unknown, so
            # they are counted as stores here rather than guessed into either set.
            "unverified_stores": counts[STATUS_UNREADABLE],
        },
        "inputs": inputs,
        "contract": CONTRACT,
    }
