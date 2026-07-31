"""Observation log — Advisor Roadmap 1.7.

A raw, **prompt-invisible** tier of per-turn behavioural observations. Nothing in
here is ever injected into a prompt. The only path from this store to the model
runs through ``tools/observation_consolidation.py`` → ``tools/pending_lessons.py``
→ a human clicking confirm, and ``tests/test_observation_invisibility.py``
enforces that structurally rather than by convention.

WHY THIS EXISTS — the store it replaces is measurably near-mute, not noisy.
Measured on a densely-used profile: ``key_facts`` = 2 against
``conversation_summaries`` = 20 (at cap) and ``lessons_learned`` = 10 (at cap).
A week of heavy use fills the summary store completely and yields two durable
facts. The defect is the SEAM, not the prompt: ``process_user_message`` fires on
the FIRST supervisor pass, before any tool has run and before the answer exists,
so the extractor judges one isolated message with no conversation, no answer and
no outcome — and its own guard tells it to return nothing for "casual
market-analysis requests", which is most of what a user types.

THREE RULES THIS MODULE IS BUILT ON:

1. **Not a fourth raw store.** Verbatim turns already persist in ``feedback.json``
   (1.5), ``chat_history.json`` and ``checkpoints.sqlite``. A row here is a
   DERIVED record keyed to those: kind, timestamp, thread_id, interaction_id, and
   a short quoted span of the USER's own words. An assistant answer is never
   copied into this file.

2. **Observe behaviour, not statements.** The store this replaces recorded things
   the user SAID. The signal is in what they DO: which calls they act on, which
   they ignore, what they push back on, what they ask for over and over.

3. **Detect, never judge.** Every writer here is deterministic — regex cues,
   ticker shapes, a shares comparison. No LLM runs at this seam. Putting a
   one-shot model judgment on each turn is precisely the defect 1.7 exists to
   fix; the judging happens once, later, in the gated consolidation pass, over
   accumulated evidence a human then reviews.
"""
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic
from tools.user_profile import get_data_path

# Observation kinds. Kept as constants because the consolidation pass, the API
# and the tests all group on them.
KIND_ASKED = "asked"
KIND_PUSHBACK = "pushback"
KIND_DECLINE = "decline"
KIND_REC_ISSUED = "rec_issued"
KIND_REC_FOLLOWED = "rec_followed"
KIND_REC_IGNORED = "rec_ignored"

OBSERVATION_KINDS = (
    KIND_ASKED,
    KIND_PUSHBACK,
    KIND_DECLINE,
    KIND_REC_ISSUED,
    KIND_REC_FOLLOWED,
    KIND_REC_IGNORED,
)

# Retention. Small derived rows, so this holds several weeks of dense use. A row
# that falls off has either been consolidated already or was never corroborated.
MAX_OBSERVATIONS = 500

# Quoted spans are evidence, not transcript. The roadmap asks each observation to
# carry one so a drafted rule can be audited back to the words that produced it.
SPAN_CHARS = 200

# Only actionable calls get a rec_issued row. A HOLD has no observable
# follow-through — "did nothing" is indistinguishable from "ignored it" — and
# recording one would manufacture an ignored-advice signal out of agreement.
FOLLOW_THROUGH_ACTIONS = frozenset({"BUY", "SELL", "TRIM", "ADD"})

# Position changes below this (relative) are DRIP drips, fractional-share dust
# and rounding, not a decision.
_SHARES_REL_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@log_exceptions()
def _observations_file() -> str:
    """Return the profile-specific observation-log path."""
    return get_data_path("observations.json")


@log_exceptions()
def load_observations() -> dict[str, Any]:
    """Load the observation log (profile-scoped). Never raises."""
    try:
        path = _observations_file()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("observations"), list):
                    data.setdefault("last_consolidated_at", None)
                    return data
    except Exception as e:
        safe_print(f"⚠️ Error loading observations: {e}")

    return {"observations": [], "last_consolidated_at": None}


@log_exceptions()
def save_observations(data: dict[str, Any]) -> bool:
    """Persist the observation log (profile-scoped). Never raises."""
    try:
        write_json_atomic(_observations_file(), data)
        return True
    except Exception as e:
        safe_print(f"⚠️ Error saving observations: {e}")
        return False


def _span(text: str) -> str:
    """A short, single-line excerpt of the user's own words."""
    return " ".join(str(text or "").split())[:SPAN_CHARS]


@log_exceptions()
def record_observation(
    kind: str,
    *,
    thread_id: str | None = None,
    interaction_id: str | None = None,
    span: str = "",
    tickers: list[str] | None = None,
    lens: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append one derived observation. Returns the row, or None if not written."""
    if kind not in OBSERVATION_KINDS:
        return None

    data = load_observations()
    row = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread_id,
        "interaction_id": interaction_id,
        "span": _span(span),
        "tickers": sorted({str(t).upper() for t in (tickers or []) if str(t).strip()}),
        "lens": lens,
        "detail": detail or {},
        # Set by the consolidation pass once this row has been read into a batch.
        # Kept on the ROW rather than as a high-water mark so trimming the store
        # can never make already-consolidated evidence look new again.
        "consolidated": False,
    }
    data["observations"] = (data["observations"] + [row])[-MAX_OBSERVATIONS:]
    save_observations(data)
    return row


# ---------------------------------------------------------------------------
# Deterministic detectors
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b")

# All-caps words that share the ticker shape and are never the subject of a
# question here. Deliberately holds no real symbol: dropping a genuine ticker
# costs a mention, while keeping a false one puts a phantom name into the
# evidence a rule gets drafted from.
_NOT_TICKERS = frozenset({
    "THE", "AND", "BUT", "FOR", "NOT", "YOU", "ALL", "ANY", "CAN", "GET", "HOW",
    "WHY", "WHAT", "WHEN", "WHO", "OK", "OKAY", "YES", "NO", "MY", "ME", "WE",
    "US", "USA", "CAD", "USD", "EUR", "GBP", "AUD", "CHF", "JPY",
    "BUY", "SELL", "HOLD", "TRIM", "ADD", "CASH", "RISK", "TAX", "PLAN", "GOAL",
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "ETFS", "EPS", "PE", "PEG", "ROE",
    "ROI", "ROIC", "YTD", "EOD", "ATH", "ATL", "DCA", "NAV", "AUM", "TSX", "NYSE",
    "FED", "FOMC", "CPI", "PPI", "GDP", "PMI", "BOC", "ECB", "SEC", "IRS", "CRA",
    "TFSA", "RRSP", "RESP", "LIRA", "FHSA", "ESG", "IPS", "AI", "ML", "EV", "FX",
    "QOQ", "YOY", "TTM", "FY", "Q1", "Q2", "Q3", "Q4", "PM", "AM", "EST", "ET",
})


def extract_tickers(text: str, holdings: dict[str, float] | None = None) -> list[str]:
    """Ticker mentions in a user message.

    Two passes with different precision profiles: uppercase tokens of ticker
    shape (minus the stopword set), plus — when the caller has holdings on hand —
    a case-insensitive match on symbols the user actually owns, which is how
    "what's going on with pltr" gets attributed at all.
    """
    body = str(text or "")
    found = {t for t in _TICKER_RE.findall(body) if t not in _NOT_TICKERS}

    if holdings:
        lowered = body.lower()
        for symbol in holdings:
            base = str(symbol or "").upper()
            if not base:
                continue
            if re.search(rf"\b{re.escape(base.lower())}\b", lowered):
                found.add(base)

    return sorted(found)


# Correction cues. Curated multi-word phrases rather than a bare "no", because a
# false pushback is fabricated evidence — it would be quoted back in a drafted
# rule as something the user objected to. Precision over recall, on purpose.
_PUSHBACK_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bthat'?s (?:wrong|incorrect|not right|not what)\b",
    r"\byou'?re (?:wrong|mistaken)\b",
    r"\b(?:i|we) (?:already )?told you\b",
    r"\bas i (?:said|mentioned|explained)\b",
    r"\bnot what i (?:asked|meant|said|wanted)\b",
    r"\byou (?:missed|forgot|ignored|keep ignoring)\b",
    r"\bno,\s*(?:i|it|that|the|this|you)\b",
    r"\bwrong (?:ticker|number|stock|figure|price|currency|account)\b",
    r"\bstop (?:recommending|suggesting|telling|showing)\b",
    r"\bdon'?t (?:recommend|suggest|keep|tell me|show me)\b",
    r"\bthat'?s (?:made up|fabricated|invented)\b",
    r"\bwhere did (?:that|you get that) (?:number|figure|come from)\b",
))

# Declination cues — the user turning down a suggestion. Same precision bar.
_DECLINE_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bi'?m not (?:going to |gonna )?(?:sell|buy|trim|add|do|touch)\b",
    r"\bi (?:won'?t|will not) (?:sell|buy|trim|add|do|touch)\b",
    r"\bnot (?:selling|buying|trimming|adding)\b",
    r"\bnot interested in\b",
    r"\b(?:skip|drop) (?:that|it|this) (?:one|idea|name)?\b",
    r"\bi'?ll pass\b",
    r"\bno thanks\b",
    r"\bleave (?:it|that|them) alone\b",
    r"\bnot doing that\b",
))


def _first_match(text: str, patterns) -> str:
    """The matched cue, or "" — the cue itself is stored so a drafted rule can be
    traced to the exact words that triggered the observation."""
    for pattern in patterns:
        m = pattern.search(text or "")
        if m:
            return m.group(0)
    return ""


# ---------------------------------------------------------------------------
# Per-turn writer
# ---------------------------------------------------------------------------

@log_exceptions()
def observe_turn(
    user_query: str,
    *,
    thread_id: str | None = None,
    interaction_id: str | None = None,
    prior_answer: str | None = None,
    holdings: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Write this turn's observations. Called from the POST-turn seam only.

    ``prior_answer`` is what the advisor said immediately before this message in
    the same thread. Pushback is only recorded when there IS one: "no, that's
    wrong" as the opening line of a thread is a correction of something outside
    this app, and attributing it to the advisor would invent the grievance.

    Returns the rows written (possibly empty — most turns say nothing durable,
    and that is the correct outcome, not a failure).
    """
    body = str(user_query or "").strip()
    if not body:
        return []

    # Defence in depth. The caller gates on is_private_turn with the ghost flag
    # in hand; this catches a future caller that forgets. A privacy toggle that
    # depends on every call site remembering is not a privacy toggle.
    from agent.utils import is_private_turn
    if is_private_turn(body):
        return []

    from agent.lenses import extract_lens

    lens = extract_lens(body)
    tickers = extract_tickers(body, holdings)
    written: list[dict[str, Any]] = []

    row = record_observation(
        KIND_ASKED,
        thread_id=thread_id,
        interaction_id=interaction_id,
        span=body,
        tickers=tickers,
        lens=lens,
    )
    if row:
        written.append(row)

    if prior_answer:
        cue = _first_match(body, _PUSHBACK_RES)
        if cue:
            row = record_observation(
                KIND_PUSHBACK,
                thread_id=thread_id,
                interaction_id=interaction_id,
                span=body,
                tickers=tickers,
                lens=lens,
                detail={"cue": cue},
            )
            if row:
                written.append(row)

    cue = _first_match(body, _DECLINE_RES)
    if cue:
        row = record_observation(
            KIND_DECLINE,
            thread_id=thread_id,
            interaction_id=interaction_id,
            span=body,
            tickers=tickers,
            lens=lens,
            detail={"cue": cue},
        )
        if row:
            written.append(row)

    return written


@log_exceptions()
def record_rec_issued(
    ticker: str,
    action: str,
    *,
    shares_at_advice: float | None,
    thread_id: str | None = None,
    interaction_id: str | None = None,
) -> dict[str, Any] | None:
    """Anchor an actionable call so its follow-through can be measured later.

    ``shares_at_advice`` is the position size at the moment of the call. It has
    to be captured HERE: ``portfolio_history.csv`` stores totals, not positions,
    so there is no way to reconstruct what was held on a past date. That is also
    why this signal only accrues forward from the day it ships, and why nothing
    back-fills it.

    ``None`` shares means the portfolio could not be read on this turn — recorded
    as unknown rather than as zero, since zero is a claim that the user owned
    none of it.
    """
    action_u = str(action or "").strip().upper()
    ticker_u = str(ticker or "").strip().upper()
    if not ticker_u or action_u not in FOLLOW_THROUGH_ACTIONS:
        return None

    return record_observation(
        KIND_REC_ISSUED,
        thread_id=thread_id,
        interaction_id=interaction_id,
        tickers=[ticker_u],
        detail={
            "action": action_u,
            "shares_at_advice": shares_at_advice,
            "resolved_by": None,
        },
    )


# ---------------------------------------------------------------------------
# Follow-through sweep (zero-LLM)
# ---------------------------------------------------------------------------

@log_exceptions()
def load_holdings_map() -> dict[str, float] | None:
    """{SYMBOL: shares} for the bound profile, or None if unreadable.

    None and {} are different answers and the caller must treat them that way: an
    empty portfolio is a real state, an unreadable one is not evidence of
    anything. Blocking I/O — call it off the event loop.
    """
    try:
        from tools.portfolio_csv import load_portfolio
        holdings = load_portfolio()
    except Exception as e:
        safe_print(f"⚠️ Observation follow-through: portfolio unreadable ({e})")
        return None

    if holdings is None:
        return None

    # load_portfolio() reports an unreadable file by RETURNING {"error": ...},
    # not by raising, so the except above never sees it. Iterating that dict walks
    # its keys — strings — every .get() raises AttributeError, the per-row except
    # swallows all of them, and the function returns {} : "you hold nothing".
    # That is the exact reading resolve_rec_follow_through's docstring forbids —
    # it marks every aged SELL as followed and writes that to the store for good.
    if isinstance(holdings, dict):
        safe_print(f"⚠️ Observation follow-through: portfolio unreadable ({holdings.get('error')})")
        return None

    out: dict[str, float] = {}
    for h in holdings:
        try:
            symbol = str(h.get("symbol") or "").strip().upper()
            if symbol:
                out[symbol] = out.get(symbol, 0.0) + float(h.get("shares") or 0.0)
        except Exception:
            continue
    return out


def _changed(before: float, after: float) -> bool:
    """True when the position moved by more than dust."""
    floor = max(abs(before), abs(after)) * _SHARES_REL_TOLERANCE
    return abs(after - before) > max(floor, 1e-9)


def _follow_through_verdict(action: str, before: float, after: float) -> str | None:
    """followed / ignored for one call, or None when it cannot be told."""
    if not _changed(before, after):
        return KIND_REC_IGNORED
    if action in ("BUY", "ADD"):
        return KIND_REC_FOLLOWED if after > before else KIND_REC_IGNORED
    if action in ("SELL", "TRIM"):
        return KIND_REC_FOLLOWED if after < before else KIND_REC_IGNORED
    return None


@log_exceptions()
def resolve_rec_follow_through(
    min_age_days: int = 7,
    holdings: dict[str, float] | None = None,
) -> dict[str, int]:
    """Mark aged calls as acted on or not, by comparing the position then and now.

    This is the "which recs they act on vs. ignore" signal 1.7 asks for, and it
    is entirely deterministic — no model, no interpretation.

    An unreadable portfolio resolves NOTHING. Reading a failed load as "no
    position" would mark every open SELL as followed, which is a fabricated
    behavioural record about the user, in the store whose whole purpose is
    holding evidence about the user.
    """
    summary = {"resolved": 0, "followed": 0, "ignored": 0, "pending": 0}

    holdings_map = load_holdings_map() if holdings is None else holdings
    if holdings_map is None:
        return summary

    data = load_observations()
    rows = data["observations"]
    cutoff = datetime.now() - timedelta(days=min_age_days)
    new_rows: list[dict[str, Any]] = []

    for row in rows:
        if row.get("kind") != KIND_REC_ISSUED:
            continue
        detail = row.get("detail") or {}
        if detail.get("resolved_by"):
            continue

        try:
            issued_at = datetime.fromisoformat(str(row.get("timestamp")))
        except (TypeError, ValueError):
            continue
        if issued_at > cutoff:
            summary["pending"] += 1
            continue

        before = detail.get("shares_at_advice")
        if before is None:
            # Unknown starting size — nothing to compare against. Left open
            # rather than guessed; it ages out with the store.
            summary["pending"] += 1
            continue

        ticker = (row.get("tickers") or [None])[0]
        if not ticker:
            continue
        after = float(holdings_map.get(ticker, 0.0))
        verdict = _follow_through_verdict(str(detail.get("action") or ""), float(before), after)
        if verdict is None:
            summary["pending"] += 1
            continue

        resolution_id = uuid.uuid4().hex[:12]
        new_rows.append({
            "id": resolution_id,
            "kind": verdict,
            "timestamp": datetime.now().isoformat(),
            "thread_id": row.get("thread_id"),
            "interaction_id": row.get("interaction_id"),
            "span": "",
            "tickers": [ticker],
            "lens": None,
            "detail": {
                "action": detail.get("action"),
                "shares_at_advice": before,
                "shares_now": after,
                "issued_at": row.get("timestamp"),
                "held_days": (datetime.now() - issued_at).days,
                "from_observation": row.get("id"),
            },
            "consolidated": False,
        })
        detail["resolved_by"] = resolution_id
        row["detail"] = detail
        summary["resolved"] += 1
        summary["followed" if verdict == KIND_REC_FOLLOWED else "ignored"] += 1

    if new_rows or summary["resolved"]:
        data["observations"] = (rows + new_rows)[-MAX_OBSERVATIONS:]
        save_observations(data)

    return summary


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------

@log_exceptions()
def mark_consolidated(observation_ids: list[str]) -> int:
    """Flag rows as read by a consolidation pass. Returns how many were flagged."""
    wanted = {str(i) for i in observation_ids if i}
    if not wanted:
        return 0

    data = load_observations()
    marked = 0
    for row in data["observations"]:
        if row.get("id") in wanted and not row.get("consolidated"):
            row["consolidated"] = True
            marked += 1
    if marked:
        data["last_consolidated_at"] = datetime.now().isoformat()
        save_observations(data)
    return marked


@log_exceptions()
def get_unconsolidated(limit: int | None = None) -> list[dict[str, Any]]:
    """Rows no consolidation pass has read yet, oldest first."""
    rows = [r for r in load_observations()["observations"] if not r.get("consolidated")]
    return rows[-limit:] if limit else rows


@log_exceptions()
def get_observation_stats() -> dict[str, Any]:
    """Counts for the /context read surface and the scheduler's heartbeat detail.

    Reports what is actually in the store, including zero. A log that has never
    been written to and a log nobody reads produce identical silence otherwise —
    which is the failure mode this whole item was opened over.
    """
    from tools.observation_consolidation import CONSOLIDATION_GATE_N

    data = load_observations()
    rows = data["observations"]
    by_kind = {kind: 0 for kind in OBSERVATION_KINDS}
    for row in rows:
        kind = row.get("kind")
        if kind in by_kind:
            by_kind[kind] += 1

    unconsolidated = sum(1 for r in rows if not r.get("consolidated"))
    return {
        "total": len(rows),
        "by_kind": by_kind,
        "unconsolidated": unconsolidated,
        "gate_n": CONSOLIDATION_GATE_N,
        "gate_met": unconsolidated >= CONSOLIDATION_GATE_N,
        "last_consolidated_at": data.get("last_consolidated_at"),
        "capacity": MAX_OBSERVATIONS,
    }


@log_exceptions()
def get_recent_observations(limit: int = 25) -> list[dict[str, Any]]:
    """Most recent rows first, for the read surface."""
    return list(reversed(load_observations()["observations"][-limit:]))
