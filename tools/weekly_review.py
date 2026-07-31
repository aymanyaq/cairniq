"""The weekly one-page review — Advisor Roadmap product surface.

One page, once a week, assembled from surfaces that already exist: the wealth
goal as the headline, the week's market state, how past advice actually scored,
what the advisor said in the last seven days, whether the background engines are
alive, and which user-authored inputs are still blank.

**Why this, after four builds nobody could see.** 1.8's grading, 3.9's deployment
ladder, 2.3's provenance and 6.2's substitution counts are all correct, all
shipped, and all invisible on a healthy box — each one exists as a number in a
JSON file that nothing reads on a schedule. This page is where those numbers
become something a person reads. It adds no new intelligence on purpose; it is a
READ surface over intelligence that already shipped.

THE CONTRACT, and it is the whole reason this module is written the way it is:

1. **Every section always renders.** A section with nothing to say says so, by
   name, and is never omitted. This is not a style preference — a report is the
   highest-risk possible surface for the 2026-07-21 failure, where
   truthiness-gated blocks emitted nothing, the reader expected a full page, and
   the silence got back-filled with real-sounding content. An explicit "nothing
   this week" cannot be back-filled. ``tests/test_tools/test_weekly_review.py``
   renders against a completely empty profile and asserts every section is still
   present.
2. **It never generates, only reads.** The market pulse is read from cache and
   never kicked off; no LLM, no scan, no network beyond what a cached read
   implies. A report that triggers work is a report that can time out, cost
   money, or — worst — quietly change the state it is describing.
3. **It never invents a number.** Absent is absent. Where a real figure cannot
   be computed the section says why, in the same spirit as
   ``goal_projection.realized_return_status``.

Three rules carried over from ``tools.profile_readiness``, which this module is
modelled on and which learned all three the hard way:

  - **Read through the accessor the consumer reads through** — never straight
    out of the JSON. A surface that re-derives a figure can disagree with the
    engine it is describing, and then it lies in the direction that hurts.
  - **Count the thing the CONSUMER reads.** For advice scoring that means
    DISTINCT CALLS, never ledger rows: on the live ledger 9 graded rows were 4
    calls, and a row count would have reported a sample more than twice its
    true size.
  - **Prose names capabilities, not code.** Roadmap numbers, endpoints and
    function names live in the ``roadmap`` field and in these docstrings, never
    in a line the reader sees.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

REVIEW_PERIOD_DAYS = 7

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_UNREADABLE = "unreadable"

CONTRACT = (
    "Every section is present whether or not it has anything to report. A blank "
    "section states that it is blank; nothing here is inferred, defaulted or "
    "filled in."
)


def _parse_ts(value: Any) -> datetime | None:
    """Lenient ISO parse. Returns None rather than raising — a malformed stamp
    excludes a record from the period, it does not break the page."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _section(
    key: str,
    title: str,
    status: str,
    *,
    note: str = "",
    roadmap: str = "",
    **payload: Any,
) -> dict[str, Any]:
    """One section of the page. `note` is what the reader is shown when there is
    nothing to report, and it is mandatory for an empty or unreadable section —
    that sentence is the entire anti-fabrication guarantee."""
    return {
        "key": key,
        "title": title,
        "status": status,
        "note": note,
        "roadmap": roadmap,
        **payload,
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _goal_section(period: dict[str, Any]) -> dict[str, Any]:
    """Headline: are we on track. Reads the same builder the dashboard panel does."""
    from tools.goal_projection import build_goal_projection

    projection = build_goal_projection()
    if not projection.get("available"):
        return _section(
            "goal", "Wealth goal", STATUS_EMPTY,
            note=(
                f"No projection this week — {projection.get('reason', 'the goal is not set')}. "
                "Stating a target, a horizon and an annual contribution turns this section on."
            ),
            roadmap="4.5",
            missing=projection.get("missing") or [],
        )

    return _section(
        "goal", "Wealth goal", STATUS_OK,
        roadmap="4.5",
        currency=projection.get("currency"),
        current_value=projection.get("current_value"),
        goal=projection.get("goal"),
        horizon_years=projection.get("horizon_years"),
        goal_success_rate=projection.get("goal_success_rate"),
        required_annual_return=projection.get("required_annual_return") or {},
        # Carried verbatim: this is a STATUS explaining why realized return is
        # withheld, and dropping it would leave the required return looking like
        # a comparison against something.
        realized_annual_return=projection.get("realized_annual_return") or {},
        assumptions=projection.get("assumptions") or {},
    )


def _market_section(period: dict[str, Any]) -> dict[str, Any]:
    """The week's market state, from the CACHED pulse only.

    Never starts a generation. The pulse is a multi-minute job with network fan-out;
    a weekly page that kicks it off would block, and on a cold cache would block
    every time it is opened.
    """
    from tools.cache import get_cached

    pulse = get_cached("market_pulse")
    if not isinstance(pulse, dict) or "regime" not in pulse:
        return _section(
            "market", "Market state", STATUS_EMPTY,
            note=(
                "No market briefing is cached right now, so this week's market state "
                "is not shown. Opening the dashboard generates one."
            ),
        )

    return _section(
        "market", "Market state", STATUS_OK,
        regime=pulse.get("regime"),
        headline=pulse.get("headline"),
        fear_greed=pulse.get("fear_greed"),
        macro_flags=pulse.get("macro_flags") or [],
        sector_trends=pulse.get("sector_trends") or [],
        portfolio_alerts=pulse.get("portfolio_alerts") or [],
        generated_at=pulse.get("generated_at") or pulse.get("scan_date"),
    )


def _scorecard_section(period: dict[str, Any]) -> dict[str, Any]:
    """How past advice actually scored.

    Reports distinct calls and the row count beside them. A hit rate over rows
    credits one correct call once per restatement — the trap that landed inside
    the fix for the corpus problem it was built to solve.
    """
    from tools.memory import get_scored_recommendations_data

    data = get_scored_recommendations_data()
    stats = data.get("stats") or {}
    partial = data.get("partial_stats") or {}

    scored_horizons = sum((stats.get(k) or {}).get("total", 0) for k in ("2w", "1m", "3m"))
    partial_calls = partial.get("total", 0)

    if not scored_horizons and not partial_calls:
        total_logged = len(data.get("recommendations") or [])
        return _section(
            "scorecard", "How past advice scored", STATUS_EMPTY,
            note=(
                f"Nothing has scored yet ({total_logged} calls logged, none matured). "
                "Calls score once their horizon elapses or a later call supersedes them."
            ),
            roadmap="1.8",
            logged=total_logged,
        )

    return _section(
        "scorecard", "How past advice scored", STATUS_OK,
        roadmap="1.8",
        horizons=stats,
        confidence=data.get("confidence_stats") or {},
        partial=partial,
        # Stated so the reader cannot mistake the row count for the sample size.
        sample_note=(
            f"{partial_calls} distinct partial-hold calls from "
            f"{partial.get('graded_rows', 0)} graded rows"
            if partial_calls else ""
        ),
    )


def _advice_section(period: dict[str, Any]) -> dict[str, Any]:
    """What the advisor actually said in the period, and how the risk gate judged it."""
    from tools.memory import load_memory
    from tools.risk_verdict_log import get_recent_verdicts

    start, end = period["start"], period["end"]

    recs = []
    for rec in (load_memory().get("past_recommendations") or []):
        ts = _parse_ts(rec.get("date"))
        if ts and start <= ts <= end:
            recs.append({
                "ticker": rec.get("ticker"),
                "action": rec.get("action"),
                "date": rec.get("date"),
                "confidence": rec.get("confidence_grade"),
                "superseded": bool(rec.get("superseded")),
            })

    verdicts = []
    for record in get_recent_verdicts(limit=500):
        ts = _parse_ts(record.get("ts"))
        if ts and start <= ts <= end:
            verdicts.append({
                "ts": record.get("ts"),
                "score": record.get("score"),
                "risk_result": record.get("risk_result"),
                "violations": record.get("violations") or [],
            })

    if not recs and not verdicts:
        return _section(
            "advice", "What the advisor said this week", STATUS_EMPTY,
            note="No specific calls were made and no advice was judged in this period.",
            roadmap="2.1",
        )

    scores = [v["score"] for v in verdicts if isinstance(v.get("score"), (int, float))]
    return _section(
        "advice", "What the advisor said this week", STATUS_OK,
        roadmap="2.1",
        calls=recs,
        call_count=len(recs),
        verdict_count=len(verdicts),
        avg_verdict_score=round(sum(scores) / len(scores), 1) if scores else None,
        flagged=[v for v in verdicts if v.get("violations")],
    )


def _engines_section(period: dict[str, Any]) -> dict[str, Any]:
    """Is anything quietly dead — and what the quiet builds actually did.

    The `concerning` list is the whole point of the liveness view; an empty one
    is a real, reportable result rather than a reason to omit the section.
    """
    from tools.engine_heartbeat import get_engine_health

    health = get_engine_health()
    concerning = health.get("concerning") or []
    beats = health.get("engines") or health.get("heartbeats") or {}

    produced = []
    if isinstance(beats, dict):
        for name, rec in beats.items():
            if not isinstance(rec, dict):
                continue
            detail = rec.get("last_detail") or ""
            if detail:
                produced.append({"engine": name, "detail": detail, "status": rec.get("last_status")})

    return _section(
        "engines", "Background engines", STATUS_OK if (concerning or produced) else STATUS_EMPTY,
        note=(
            "" if (concerning or produced) else
            "No engine has reported anything since the last restart. That is itself "
            "worth a look — the engines record what they did on every run."
        ),
        roadmap="2.5/2.6",
        concerning=concerning,
        concerning_count=len(concerning),
        reported=produced,
    )


def _readiness_section(period: dict[str, Any]) -> dict[str, Any]:
    """Inputs only the user can state, and what stays switched off without them.

    Delegates entirely rather than re-deriving: that surface already refuses to
    author, default or exemplify a value, and a second implementation would be a
    second chance to break that promise.
    """
    from tools.profile_readiness import build_profile_readiness

    readiness = build_profile_readiness()
    counts = readiness.get("counts") or {}
    gaps = [i for i in (readiness.get("inputs") or []) if i.get("status") != "set"]

    if not gaps:
        return _section(
            "readiness", "Inputs still needed", STATUS_EMPTY,
            note="Nothing is missing — every input the engines depend on is on file.",
            roadmap="2.8",
            counts=counts,
        )

    capabilities = readiness.get("capabilities") or {}
    return _section(
        "readiness", "Inputs still needed", STATUS_OK,
        roadmap="2.8",
        counts=counts,
        inert_count=readiness.get("inert_count", 0),
        # The deduped count. `inert_count` beside it is consequence SENTENCES,
        # one per blank field, and several fields commonly name one capability —
        # rendering that figure as a capability count overstated what is dark.
        dark_capabilities=len(capabilities.get("dark") or []),
        gaps=[
            {
                "field": g.get("field"),
                "label": g.get("label"),
                "status": g.get("status"),
                "cost": g.get("cost"),
                "where": g.get("where"),
            }
            for g in gaps
        ],
    )


_BUILDERS = (
    _goal_section,
    _market_section,
    _scorecard_section,
    _advice_section,
    _engines_section,
    _readiness_section,
)

# Titles must survive a builder blowing up: an unreadable section still has to
# render under its own name, or the failure becomes an omission — which is the
# one outcome this module exists to prevent.
_TITLES = {
    "_goal_section": ("goal", "Wealth goal"),
    "_market_section": ("market", "Market state"),
    "_scorecard_section": ("scorecard", "How past advice scored"),
    "_advice_section": ("advice", "What the advisor said this week"),
    "_engines_section": ("engines", "Background engines"),
    "_readiness_section": ("readiness", "Inputs still needed"),
}


def _unreadable(builder_name: str, error: Exception) -> dict[str, Any]:
    key, title = _TITLES.get(builder_name, (builder_name, builder_name))
    logger.warning("weekly review section %s failed: %s", builder_name, error)
    return _section(
        key, title, STATUS_UNREADABLE,
        note=(
            "This section could not be read this week. Its data is unavailable, "
            "not empty — treat it as unknown rather than as nothing to report."
        ),
        error=f"{type(error).__name__}: {error}",
    )


def build_weekly_review(now: datetime | None = None) -> dict[str, Any]:
    """The whole page, as data. Never raises.

    One failed section yields an `unreadable` row, never a broken page: an
    instrument that goes dark when one of its inputs does has acquired the fault
    it was built to catch.
    """
    now = now or datetime.now()
    start = now - timedelta(days=REVIEW_PERIOD_DAYS)
    period = {
        "start": start,
        "end": now,
        "days": REVIEW_PERIOD_DAYS,
        "label": f"{start:%b %-d} – {now:%b %-d, %Y}",
    }

    sections = []
    for builder in _BUILDERS:
        try:
            sections.append(builder(period))
        except Exception as e:  # noqa: BLE001 — a bad store yields a section, not a 500
            sections.append(_unreadable(builder.__name__, e))

    counts = {
        "total": len(sections),
        STATUS_OK: sum(1 for s in sections if s["status"] == STATUS_OK),
        STATUS_EMPTY: sum(1 for s in sections if s["status"] == STATUS_EMPTY),
        STATUS_UNREADABLE: sum(1 for s in sections if s["status"] == STATUS_UNREADABLE),
    }

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "period": {
            "start": start.isoformat(timespec="seconds"),
            "end": now.isoformat(timespec="seconds"),
            "days": REVIEW_PERIOD_DAYS,
            "label": period["label"],
        },
        "counts": counts,
        "sections": sections,
        "contract": CONTRACT,
    }


def summarize_for_heartbeat(review: dict[str, Any]) -> str:
    """The one line the scheduler records — chosen to prove the CHAIN ran.

    Deliberately the section counts rather than a rare event: an engine whose
    detail line only fills in on an interesting week is indistinguishable from
    an engine that stopped, which is the exact failure the liveness work exists
    to catch.
    """
    counts = review.get("counts") or {}
    return (
        f"{counts.get('total', 0)} sections · {counts.get(STATUS_OK, 0)} reported · "
        f"{counts.get(STATUS_EMPTY, 0)} empty · {counts.get(STATUS_UNREADABLE, 0)} unreadable"
    )
