"""Turn-wide data provenance — Advisor Roadmap 2.3 (with 5.8's remainder).

Two contracts already exist for describing degraded data, and both are per-tool:
``tools.tool_errors.unavailable`` says "I could not check", and ``tools.freshness``
stamps when a payload was actually fetched. What has been missing is the TURN-level
view — a single answer to "what was the evidence behind this answer actually worth?"

Without it the failure is silent in the way this codebase keeps paying for. One
tool returns ``unavailable`` because an API key is missing, the model reads past
it, and the advice reads exactly like advice built on complete data. Nobody is
lying; there is simply nowhere that the degradation is counted.

**This module reads the rendered tool-execution context** — the same block the
RiskManager assembles as the judge's grounding evidence — rather than hooking the
tool executors. That is deliberate:

  - It sees precisely what the judge sees. A provenance summary derived from a
    different collection point could disagree with the evidence it is describing,
    and then it lies in the direction that hurts: "all sources live" beside a
    context full of unavailable payloads.
  - The tools run concurrently across three different analyst nodes. A collector
    threaded through all of them is more moving parts for a strictly worse view.

**It never invents a status.** A payload with no readable stamp is `unverified`,
not `fresh` — absence of proof is not proof of freshness, which is the standing
policy in ``tools.freshness``. The distinction is the whole point: this module
exists to stop degraded evidence reading as complete evidence, so a summary that
guessed optimistically would be worse than none at all.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# Blocks rendered by risk_manager._build_tool_execution_context. The format is
# internal but stable, and shared with deep_reasoning's context merge.
_BLOCK_RE = re.compile(r"^### Tool Call: ([A-Za-z0-9_]+)\(", re.MULTILINE)

# Split on the header so whole blocks can be merged by identity.
_BLOCK_SPLIT_RE = re.compile(r"(?=^### Tool Call: )", re.MULTILINE)

# An `unavailable()` payload, in either repr or JSON quoting — the rendering
# depends on which node stringified it, and both reach the judge.
_UNAVAILABLE_RE = re.compile(r"['\"]status['\"]\s*:\s*['\"]unavailable['\"]")
_SOURCE_RE = re.compile(r"['\"]source['\"]\s*:\s*['\"]([^'\"]{1,60})['\"]")
_REASON_RE = re.compile(r"['\"]reason['\"]\s*:\s*['\"]([^'\"]{1,200})['\"]")
_AS_OF_RE = re.compile(r"['\"]_as_of['\"]\s*:\s*['\"]([0-9T:\-\.\+ ]{10,32})['\"]")

# Beyond this, a fetch is old enough that advice resting on it should say so.
# Matches the watch-conditions quote gate rather than inventing a second number.
STALE_AFTER_MINUTES = 45

STATUS_UNAVAILABLE = "unavailable"
STATUS_STALE = "stale"
STATUS_FRESH = "fresh"
STATUS_UNVERIFIED = "unverified"

# 6.2 marks a recovered call in the result body. Counting it here rather than in
# the nodes keeps the one-collection-point rule this module was built on: the
# figure describes the same evidence the judge reads, and on a healthy box it is
# the only thing that will ever show 6.2 fired.
_SUBSTITUTED_RE = re.compile(re.escape("⚠️ SUBSTITUTED SOURCE"))
_SUBSTITUTE_NAME_RE = re.compile(r"came from `([A-Za-z0-9_]{1,64})` instead")


def _split_blocks(tool_ctx: str) -> list[tuple[str, str]]:
    """[(tool_name, block_body)] for each rendered tool call, in order."""
    if not tool_ctx or not isinstance(tool_ctx, str):
        return []
    matches = list(_BLOCK_RE.finditer(tool_ctx))
    blocks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tool_ctx)
        blocks.append((m.group(1), tool_ctx[m.start():end]))
    return blocks


def merge_tool_contexts(prior: str, current: str) -> str:
    """Union two rendered tool-execution contexts, `current` winning on collision.

    Keyed on the full "### Tool Call: name(args)" header, so re-running the same
    call with the same args refreshes that block in place rather than duplicating
    it, while a call only ONE side ran is kept.

    Two callers, one rule. DeepReasoning uses it across a compliance retry, whose
    narrower tool set must not delete the evidence the draft under revision was
    grounded on. The RiskManager uses it across NODES within a turn: an analyst's
    tool results live in the graph's ToolMessages while the deep path's live only
    in ``data_context``, and a judge shown either half alone reads the other
    half's true figures as Rule 8 source fraud.
    """
    merged: dict[str, str] = {}
    for block in _BLOCK_SPLIT_RE.split(prior or "") + _BLOCK_SPLIT_RE.split(current or ""):
        block = block.strip()
        if block:
            merged[block.split("\n", 1)[0]] = block
    return "\n\n".join(merged.values())


def _age_minutes(raw: str, now: datetime) -> float | None:
    try:
        fetched = datetime.fromisoformat(raw.strip())
    except (TypeError, ValueError):
        return None
    if fetched.tzinfo is not None:
        fetched = fetched.replace(tzinfo=None)
    return max(0.0, (now - fetched).total_seconds() / 60.0)


def _humanize_age(minutes: float) -> str:
    """'live' / '20m' / '3h' — the phrasing the footer uses."""
    if minutes < 5:
        return "live"
    if minutes < 90:
        return f"{int(round(minutes))}m"
    return f"{int(round(minutes / 60))}h"


def classify_block(name: str, body: str, now: datetime) -> dict[str, Any]:
    """Classify ONE tool result. Never raises, never guesses in the safe direction."""
    entry: dict[str, Any] = {"tool": name, "status": STATUS_UNVERIFIED}

    # 6.2: the block header carries the FAILED tool's name (it is rendered from
    # the planner's tool_call), so the body is the only place the stand-in is
    # visible. Recorded beside the status rather than as one, because a
    # substituted payload still has its own freshness to classify.
    if _SUBSTITUTED_RE.search(body):
        entry["substituted"] = True
        stand_in = _SUBSTITUTE_NAME_RE.search(body)
        if stand_in:
            entry["substitute"] = stand_in.group(1)

    if _UNAVAILABLE_RE.search(body):
        entry["status"] = STATUS_UNAVAILABLE
        source = _SOURCE_RE.search(body)
        reason = _REASON_RE.search(body)
        if source:
            entry["source"] = source.group(1)
        if reason:
            entry["reason"] = reason.group(1)
        return entry

    stamped = _AS_OF_RE.search(body)
    if not stamped:
        # No stamp is UNVERIFIED, never fresh. See the module docstring.
        return entry

    age = _age_minutes(stamped.group(1), now)
    if age is None:
        return entry
    entry["as_of"] = stamped.group(1)
    entry["age_minutes"] = round(age, 1)
    entry["status"] = STATUS_STALE if age > STALE_AFTER_MINUTES else STATUS_FRESH
    return entry


def summarize_tool_context(tool_ctx: str, now: datetime | None = None) -> dict[str, Any]:
    """The turn's data-quality summary, for ``data_context['data_quality']``.

    Returns ``{"sources", "counts", "degraded", "footer"}``. ``degraded`` is True
    when anything was unavailable or stale — it is the flag the judge's cap reads,
    and it is deliberately NOT set by merely-unverified evidence: almost nothing
    outside the cached surface carries a stamp yet, so treating unverified as
    degraded would cap essentially every verdict and the signal would be worthless
    within a day.
    """
    now = now or datetime.now()
    sources = []
    try:
        for name, body in _split_blocks(tool_ctx):
            sources.append(classify_block(name, body, now))
    except Exception:  # noqa: BLE001 — provenance must never break a turn
        sources = []

    counts = {
        "total": len(sources),
        STATUS_UNAVAILABLE: sum(1 for s in sources if s["status"] == STATUS_UNAVAILABLE),
        STATUS_STALE: sum(1 for s in sources if s["status"] == STATUS_STALE),
        STATUS_FRESH: sum(1 for s in sources if s["status"] == STATUS_FRESH),
        STATUS_UNVERIFIED: sum(1 for s in sources if s["status"] == STATUS_UNVERIFIED),
        "substituted": sum(1 for s in sources if s.get("substituted")),
    }
    # A recovered call does NOT mark the turn degraded: 6.2 substitutes only when
    # a curated equivalent actually returned data, so the evidence is complete —
    # it simply came from the second-choice source, which the footer names.
    degraded = bool(counts[STATUS_UNAVAILABLE] or counts[STATUS_STALE])
    return {
        "sources": sources,
        "counts": counts,
        "degraded": degraded,
        "footer": build_footer(sources),
    }


def build_footer(sources: list[dict[str, Any]]) -> str:
    """The one-line provenance footer: "quotes live · insider data unavailable".

    Empty string when there is nothing worth saying — no tools ran, or every
    result was unverified, in which case a footer would be pure noise. Naming an
    unavailable source is the load-bearing half: that is the sentence the user
    needs in order to weigh the answer, and it is the one nobody writes by hand.
    """
    if not sources:
        return ""

    parts: list[str] = []
    for entry in sources:
        if entry["status"] == STATUS_UNAVAILABLE:
            label = entry.get("source") or entry["tool"]
            parts.append(f"{label} unavailable")
        elif entry.get("substituted"):
            # 6.2: the answer stands, but not on the source that was asked for.
            # Naming the stand-in is the same contract as naming a dead source —
            # the user needs it to weigh the answer.
            stand_in = entry.get("substitute")
            parts.append(
                f"{entry['tool']} via {stand_in}" if stand_in else f"{entry['tool']} via a fallback source"
            )

    ages = [s["age_minutes"] for s in sources if s["status"] in (STATUS_FRESH, STATUS_STALE)]
    if ages:
        # One phrase for the whole fetched set, keyed on the OLDEST reading —
        # a footer claiming "live" because one of six calls was fresh would be
        # the overclaim this line exists to prevent.
        parts.insert(0, f"data {_humanize_age(max(ages))}")

    if not parts:
        return ""
    return " · ".join(dict.fromkeys(parts))
