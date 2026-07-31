"""Behavioural consolidation — Advisor Roadmap 1.7, the deliverable half.

The observation log (``tools/observations.py``) is a write-only tier until
something reads it. This is that reader: on a gate, it summarizes accumulated
behavioural evidence into candidate rules and **drafts** them for a human.

WHERE THIS SITS IN THE CHAIN, and why the boundaries are hard:

    per-turn detectors → observations.json → [this pass] → pending_lessons.json
                                                        → human clicks confirm
                                                        → lessons_learned

``lessons_learned`` is injected into every prompt and capped, so an unreviewed
entry does not add noise — it competes with a rule the user wrote themselves.
This module therefore never imports ``add_lesson``; the only write it performs is
``add_pending_lesson``, and ``tests/test_observation_invisibility.py`` asserts
that on the source rather than trusting the convention.

THREE GUARDS, each paid for by a past failure in this codebase:

* **The n gate.** Nothing runs until enough evidence exists (1.4 and 3.8 are
  gated the same way). A summarizer pointed at a nearly-empty store is exactly
  the setup that produced invented rotation history and a fabricated thesis here
  before — silence gets back-filled.
* **Citation or discard.** Every candidate must cite at least two observation
  ids from the batch it was shown. A candidate citing an id that is not in the
  batch is DROPPED, never repaired: an uncheckable claim about the user's own
  behaviour is worse than no rule at all.
* **A cap per pass.** At most three drafts, so one run cannot fill the 25-slot
  pending queue and turn the confirm gate into a thing nobody reads.
"""
import json
import logging
from typing import Any

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

# Unconsolidated observations required before a pass runs. Roughly a week of
# ordinary use. Below this the pass has nothing to generalize FROM — a rule
# drafted off three rows is a rule about three rows.
CONSOLIDATION_GATE_N = 20

# Drafts per pass. Deliberately small: the deliverable is rules a human actually
# reads, and a queue of twenty is a queue of zero.
MAX_DRAFTS_PER_PASS = 3

# Citations required per proposal, and therefore the floor below which even a
# forced pass provably cannot produce a valid rule. This is derived from the
# citation requirement rather than chosen: with fewer rows than this, every
# possible proposal fails validation, so calling a model would spend money to
# generate output that is discarded — and hand it the one setup (an almost-empty
# store) that has produced invented history here before. `force` skips the
# CADENCE gate; it does not skip this.
MIN_CITATIONS = 2

# Evidence rows shown to the model in one pass. Bounds the prompt and keeps the
# citation check meaningful.
MAX_EVIDENCE_ROWS = 120

_SYSTEM_PROMPT = (
    "You are reviewing a log of OBSERVED USER BEHAVIOUR from a financial advisory app, "
    "to propose standing instructions for the advisor.\n\n"
    "Each row is one observation with an id. Row kinds:\n"
    "  asked         — the user asked about these tickers / this topic\n"
    "  pushback      — the user corrected or objected to the advisor\n"
    "  decline       — the user turned down a suggestion\n"
    "  rec_issued    — the advisor made an actionable call\n"
    "  rec_followed  — the user's position moved in the direction advised\n"
    "  rec_ignored   — the user's position did not move\n\n"
    "Propose standing instructions ONLY for patterns the rows actually show. Rules:\n"
    "1. Every proposal MUST cite at least 2 observation ids that support it, copied "
    "exactly from the rows shown. Never cite an id that is not in the list.\n"
    "2. Never infer a reason the user did not state. 'The user declined three trims of "
    "X' is supported; 'the user is bearish on X' is not.\n"
    "3. A proposal must be an instruction to the ADVISOR about how to behave, phrased "
    "in one or two sentences. Not a summary, not a description of the user.\n"
    "4. Do not propose anything about portfolio holdings or position sizes — those are "
    "tracked from the live portfolio, not from behaviour.\n"
    "5. If the rows show no durable pattern, return an empty list. That is a correct "
    "and expected answer; inventing a pattern to fill the output is a failure.\n\n"
    f"Return at most {MAX_DRAFTS_PER_PASS} proposals as valid JSON, nothing else:\n"
    '{\n  "proposals": [\n    {\n      "rule": "one or two sentences",\n'
    '      "evidence_ids": ["abc123", "def456"]\n    }\n  ]\n}'
)


def _render_row(row: dict[str, Any]) -> str:
    """One observation as a single evidence line, id first."""
    date = str(row.get("timestamp") or "")[:10]
    tickers = ",".join(row.get("tickers") or []) or "-"
    detail = row.get("detail") or {}
    bits = [f"[{row.get('id')}]", str(row.get("kind")), date, tickers]

    if detail.get("action"):
        bits.append(str(detail["action"]))
    if row.get("lens"):
        bits.append(f"lens={row['lens']}")
    if detail.get("shares_now") is not None:
        bits.append(f"shares {detail.get('shares_at_advice')}→{detail.get('shares_now')}")
    if row.get("span"):
        bits.append(f'"{row["span"]}"')
    return " | ".join(bits)


def _parse_proposals(raw: str) -> list[dict[str, Any]]:
    """Pull the proposals list out of a model response. Never raises."""
    body = str(raw or "").strip()
    if body.startswith("```"):
        lines = body.split("\n")
        body = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(body)
    except Exception:
        return []
    proposals = data.get("proposals") if isinstance(data, dict) else None
    return proposals if isinstance(proposals, list) else []


def _validated(
    proposals: list[dict[str, Any]],
    known_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """Keep proposals whose citations check out. Returns (kept, dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0

    for proposal in proposals:
        if not isinstance(proposal, dict):
            dropped += 1
            continue
        rule = str(proposal.get("rule") or "").strip()
        cited = proposal.get("evidence_ids")
        cited = [str(c).strip() for c in cited if str(c).strip()] if isinstance(cited, list) else []
        unique = sorted(set(cited))

        # An uncheckable citation is the whole failure mode. Drop, never repair:
        # trimming the bad ids off and keeping the rule would leave a rule whose
        # stated support is not the support it was actually written from.
        if not rule or len(unique) < MIN_CITATIONS or any(c not in known_ids for c in unique):
            dropped += 1
            continue

        kept.append({"rule": rule, "evidence_ids": unique})
        if len(kept) >= MAX_DRAFTS_PER_PASS:
            break

    return kept, dropped


@log_exceptions()
def consolidate_observations(force: bool = False) -> dict[str, Any]:
    """Read the unconsolidated log and draft candidate rules for confirmation.

    ``force`` skips the n gate for an explicit human click — it does NOT skip any
    of the validation, and an empty log still drafts nothing.

    Returns a report: what it read, what it drafted, and — when it did nothing —
    WHY, because "gated" and "broken" have to be distinguishable from the
    outside. Never raises.
    """
    from agent.utils import get_llm, llm_ready, safe_invoke
    from tools.observations import get_unconsolidated, mark_consolidated
    from tools.pending_lessons import add_pending_lesson

    report: dict[str, Any] = {
        "observations_read": 0,
        "drafted": 0,
        "dropped": 0,
        "gated": False,
        "reason": "",
    }

    rows = get_unconsolidated()
    report["observations_read"] = len(rows)

    if not rows:
        report["gated"] = True
        report["reason"] = "no unconsolidated observations"
        return report

    if len(rows) < CONSOLIDATION_GATE_N and not force:
        report["gated"] = True
        report["reason"] = f"{len(rows)} of {CONSOLIDATION_GATE_N} observations toward the next pass"
        return report

    if len(rows) < MIN_CITATIONS:
        # Applies to a forced pass too — see MIN_CITATIONS.
        report["gated"] = True
        report["reason"] = (
            f"{len(rows)} unread observation(s); a rule needs {MIN_CITATIONS} to cite"
        )
        return report

    ready, why = llm_ready()
    if not ready:
        # Nothing is marked consolidated on this path: evidence the pass never
        # actually read must stay unread, or a provider outage silently burns a
        # week of accumulated behaviour.
        report["reason"] = f"LLM unavailable: {why}"
        return report

    batch = rows[-MAX_EVIDENCE_ROWS:]
    known_ids = {str(r.get("id")) for r in batch}
    evidence = "\n".join(_render_row(r) for r in batch)

    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.memory import _content_to_str

    try:
        response = safe_invoke(get_llm(), [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Observation rows:\n\n{evidence}"),
        ])
    except Exception as e:
        # `llm_ready()` clearing is not the same as a client that BUILDS: a
        # provider selected without its model id passes the readiness check and
        # then raises here. Reported as a clean skip with the evidence left
        # unread — the same treatment as an unready provider, for the same
        # reason (see _skip_if_llm_unready: a persistent credential fault does
        # not fix itself by being retried, and this pass costs real model spend).
        report["reason"] = f"LLM unavailable: {e}"
        logger.warning(f"Observation consolidation could not reach the model: {e}")
        return report

    proposals = _parse_proposals(_content_to_str(response.content))
    kept, dropped = _validated(proposals, known_ids)
    report["dropped"] = dropped

    for proposal in kept:
        cited = [r for r in batch if str(r.get("id")) in set(proposal["evidence_ids"])]
        draft = add_pending_lesson(
            proposal["rule"],
            source="observation_consolidation",
            evidence={
                "observation_ids": proposal["evidence_ids"],
                "kinds": sorted({str(r.get("kind")) for r in cited}),
                "spans": [r.get("span") for r in cited if r.get("span")][:3],
                "thread_ids": sorted({str(r.get("thread_id")) for r in cited if r.get("thread_id")}),
            },
        )
        if draft:
            report["drafted"] += 1

    # Mark the whole batch read, drafted or not. Evidence that produced no rule
    # is evidence that has been considered — leaving it unconsolidated would
    # re-propose the same non-pattern on every future pass.
    mark_consolidated([str(r.get("id")) for r in batch])

    if not report["reason"]:
        report["reason"] = (
            f"{report['drafted']} drafted from {len(batch)} observations"
            + (f", {dropped} dropped for uncheckable citations" if dropped else "")
        )
    logger.info(f"Observation consolidation: {report['reason']}")
    return report
