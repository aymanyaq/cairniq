"""Earnings-call tone analysis — Advisor Roadmap 5.4.

Three things, in the order they had to be done:

**1. It has to know when it does not know.** The previous version guarded with
``if not transcript_data or "Error" in transcript_data``. But
``get_earnings_transcript`` does not return an error string on failure — a rate
limit returns *"⚠️ API Limit Reached for full transcript. Here is a web summary
instead: …"*, which is truthy and contains no "Error". So the guard passed, the
word counter scored the SEARCH SNIPPET, found fewer than five sentiment words,
and returned **"Neutral · Insufficient forward-looking vocabulary detected"** —
a tone verdict for a call whose transcript was never read. Measured live against
MSFT on 2026-07-28. Detection is now positive (``is_real_transcript``) and the
no-data path returns ``unavailable()``, so the degradation is counted by the
turn-provenance summary and can be substituted for; neither could see this tool
before.

**2. A twelve-word lexicon is not a signal.** The previous lists held twelve
positive and twelve negative words. The categories below are drawn from the
Loughran-McDonald financial sentiment dictionary — the finance-specific standard,
built because general-purpose lexicons misread words like *liability*, *cost* and
*capital* that are negative in ordinary English and neutral in a filing.

  **This is an explicit SUBSET, not the master dictionary.** LM's full lists run
  to several thousand terms; what is embedded here is a few hundred that actually
  recur in earnings calls, and this module says so rather than implying full
  coverage. The consequence is precise and worth stating: **absolute counts are
  LOWER BOUNDS.** The comparison in :func:`compare_management_tone` is
  unaffected, because the same lexicon scores both quarters and a consistent
  undercount cancels in the difference — which is the argument for shipping the
  delta on a subset rather than waiting for the full dictionary.

**3. The delta is the signal, not the level.** A team that always speaks
cautiously is not a sell, and one that always sounds confident is not a buy; what
institutional readers watch is the CHANGE from the same team's previous call.
Counts are normalised per 1,000 words before differencing, because transcript
lengths vary enormously between quarters and raw counts would report a longer
call as a more negative one.

*Also removed: the old interpretation strings asserted that confident language
"historically precedes upward earnings revisions" and that cautious language
means "high risk of a guidance cut". Neither claim was measured anywhere in this
codebase. Describing what the words did is this module's job; predicting what
follows is not, and an unmeasured prediction reads as a finding.*
"""
import collections
import re
from typing import Any

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.fmp_api import (
    get_earnings_transcript,
    is_real_transcript,
    parse_transcript_period,
    transcript_body,
)
from tools.tool_errors import unavailable

# --- Loughran-McDonald categories (subset — see the module docstring) ---------
# Lowercase, four characters or more: the tokenizer drops shorter words, so
# anything below that length would be dead weight in these sets.

NEGATIVE_WORDS = frozenset({
    "adverse", "adversely", "aggravate", "anomalies", "anomaly", "bankruptcy",
    "barrier", "barriers", "breach", "burden", "burdened", "burdensome",
    "cancel", "cancellation", "cancelled", "cautious", "cautiously", "challenge",
    "challenged", "challenges", "challenging", "closure", "closures",
    "complaint", "complaints", "compression", "concern", "concerned", "concerns",
    "constrain", "constrained", "contraction", "correction", "curtail",
    "curtailed", "cutback", "damage", "damages", "decelerate", "decelerating",
    "deceleration", "decline", "declined", "declines", "declining", "decrease",
    "decreased", "decreases", "defer", "deferral", "deferred", "deficiency",
    "deficit", "delay", "delayed", "delays", "deteriorate", "deteriorated",
    "deterioration", "difficult", "difficulties", "difficulty", "diminish",
    "diminished", "disappoint", "disappointed", "disappointing",
    "disappointment", "disruption", "disruptions", "downgrade", "downgraded",
    "downturn", "erode", "eroded", "erosion", "failure", "failures", "forgo",
    "fraud", "harm", "harmful", "headwind", "headwinds", "hurt", "impair",
    "impaired", "impairment", "impede", "inability", "inadequate",
    "inefficiency", "instability", "insufficient", "lagged", "lagging",
    "layoff", "layoffs", "litigation", "loss", "losses", "macro", "misconduct",
    "missed", "negative", "negatively", "obsolete", "outage", "overcapacity",
    "penalties", "penalty", "poor", "pressure", "pressured", "pressures",
    "problem", "problems", "recall", "recession", "reduction", "reductions",
    "restructuring", "setback", "severe", "shortage", "shortfall", "shrink",
    "shrinking", "slowdown", "slower", "slowing", "sluggish", "softening",
    "softer", "softness", "strain", "stress", "suspend", "suspended",
    "terminate", "terminated", "termination", "tough", "unable", "underperform",
    "underperformance", "underperformed", "unfavorable", "unfavourable",
    "unprofitable", "weak", "weaken", "weakened", "weakening", "weaker",
    "weakness", "worse", "worsen", "worsened", "worsening", "writedown",
    "writeoff",
})

POSITIVE_WORDS = frozenset({
    "accelerate", "accelerated", "accelerating", "acceleration", "accomplish",
    "accomplished", "achieve", "achieved", "achievement", "achievements",
    "advance", "advanced", "advantage", "advantages", "attractive",
    "beneficial", "benefit", "benefited", "benefits", "best", "better", "boost",
    "boosted", "breakthrough", "compelling", "confidence", "confident",
    "constructive", "delight", "delighted", "differentiated", "durable",
    "efficiencies", "efficiency", "efficient", "encouraged", "encouraging",
    "enhance", "enhanced", "enhancement", "enthusiasm", "enthusiastic",
    "exceed", "exceeded", "exceeding", "excellent", "exceptional", "excited",
    "expand", "expanded", "expanding", "expansion", "favorable", "favourable",
    "gain", "gained", "gains", "great", "greater", "growing", "growth",
    "healthy", "impressive", "improve", "improved", "improvement",
    "improvements", "improving", "incremental", "innovation", "innovative",
    "leadership", "leading", "momentum", "opportunities", "opportunity",
    "optimism", "optimistic", "outperform", "outperformance", "outperformed",
    "outstanding", "pleased", "positive", "positively", "premier", "profitable",
    "progress", "proud", "raised", "rebound", "record", "resilience",
    "resilient", "robust", "solid", "stability", "stable", "strength",
    "strengthen", "strengthened", "strengthening", "strong", "stronger",
    "strongest", "success", "successful", "successfully", "surpass",
    "surpassed", "tailwind", "tailwinds", "terrific", "tremendous", "upgrade",
    "upgraded", "upside", "winning",
})

# LM's UNCERTAINTY category. Tracked separately rather than folded into negative
# because it is a different claim: "we do not know" is not "it went badly", and
# rising hedging alongside flat results is the classic pre-guidance-cut tell.
UNCERTAINTY_WORDS = frozenset({
    "almost", "ambiguity", "ambiguous", "anticipate", "anticipated",
    "apparently", "appear", "appeared", "appears", "approximate",
    "approximately", "assume", "assumed", "assumes", "assumption",
    "assumptions", "believe", "believed", "believes", "clarification",
    "conditional", "contingency", "contingent", "could", "depend", "depended",
    "dependence", "dependent", "depending", "depends", "differ", "differed",
    "doubt", "doubtful", "exposure", "fluctuate", "fluctuated", "fluctuation",
    "fluctuations", "hopeful", "imprecise", "improbable", "indefinite",
    "indeterminate", "likelihood", "maybe", "might", "nearly", "occasionally",
    "pending", "perhaps", "possible", "possibly", "precaution", "predict",
    "predicted", "prediction", "preliminary", "presumably", "probable",
    "probably", "risk", "risks", "risky", "roughly", "seems", "seldom",
    "should", "sometimes", "somewhat", "speculate", "speculation", "sporadic",
    "sudden", "suggest", "suggests", "tentative", "turbulence", "uncertain",
    "uncertainly", "uncertainties", "uncertainty", "unclear", "unconfirmed",
    "undecided", "undetermined", "unforeseen", "unknown", "unpredictable",
    "unproven", "unusual", "vague", "variability", "variable", "varied", "vary",
    "volatile", "volatility",
})

# LM's LITIGIOUS category — legal-process language. A step change here often
# precedes a disclosure, and it is invisible to a positive/negative axis.
LITIGIOUS_WORDS = frozenset({
    "adjudicate", "adjudication", "allegation", "allegations", "allege",
    "alleged", "appeal", "appealed", "arbitration", "attorney", "attorneys",
    "claimant", "compliance", "contractual", "counterclaim", "court",
    "defendant", "deposition", "indemnify", "indemnity", "indictment",
    "injunction", "investigation", "investigations", "judicial", "juries",
    "jurisdiction", "jury", "lawsuit", "lawsuits", "legal", "liable",
    "litigate", "plaintiff", "plaintiffs", "prosecution", "regulatory",
    "settlement", "settlements", "statutory", "subpoena", "sued", "testimony",
    "verdict",
})

# LM's CONSTRAINING category — language about limits on the firm's freedom to
# act (covenants, obligations, restrictions). Rises when balance-sheet pressure
# is building, ahead of it reaching the reported numbers.
CONSTRAINING_WORDS = frozenset({
    "abide", "bound", "commit", "commitment", "commitments", "committed",
    "compel", "compelled", "comply", "compulsory", "constraint", "constraints",
    "covenant", "covenants", "encumber", "encumbrance", "forbid", "forbidden",
    "impose", "imposed", "imposing", "limitation", "limitations", "limited",
    "mandate", "mandated", "mandatory", "obligate", "obligated", "obligation",
    "obligations", "oblige", "prohibit", "prohibited", "prohibition", "require",
    "required", "requirement", "requirements", "restrict", "restricted",
    "restriction", "restrictions", "stipulate", "stipulation",
})

CATEGORIES: dict[str, frozenset[str]] = {
    "negative": NEGATIVE_WORDS,
    "positive": POSITIVE_WORDS,
    "uncertainty": UNCERTAINTY_WORDS,
    "litigious": LITIGIOUS_WORDS,
    "constraining": CONSTRAINING_WORDS,
}

LEXICON_NOTE = (
    "Loughran-McDonald categories, curated subset "
    f"({sum(len(words) for words in CATEGORIES.values())} terms) — absolute counts "
    "are lower bounds; quarter-over-quarter changes are unaffected, because both "
    "calls are scored with the same lexicon."
)

# Below this a transcript is too short to characterise. Deliberately distinct
# from "no transcript": this one WAS read.
MIN_SCORABLE_WORDS = 200

# Per 1,000 words. A category must move by more than this before the change is
# called out — transcripts differ in length, speakers and structure from quarter
# to quarter, and a signal that fires on every wobble stops being read.
MATERIAL_DELTA_PER_1K = 0.5

_TOKEN_RE = re.compile(r"\b[a-z]{4,}\b")

_TRANSCRIPT_SOURCE = "FMP earnings transcript"


def _tokenize(text: str) -> collections.Counter:
    return collections.Counter(_TOKEN_RE.findall(text.lower()))


def score_text(text: str) -> dict[str, Any]:
    """Category counts and per-1,000-word rates for one transcript body.

    Pure and injectable: the tone logic is testable without a network call, which
    is what would have caught "neutral for a call nobody read" years earlier.
    """
    counts = _tokenize(text)
    total_words = sum(counts.values())
    hits = {name: sum(counts.get(w, 0) for w in words) for name, words in CATEGORIES.items()}
    per_1k = {
        name: round((n / total_words) * 1000, 2) if total_words else 0.0
        for name, n in hits.items()
    }
    return {"total_words": total_words, "counts": hits, "per_1k": per_1k}


def _classify(scored: dict[str, Any]) -> dict[str, Any]:
    """Tone label from the positive/negative balance. Describes; does not predict."""
    pos = scored["counts"]["positive"]
    neg = scored["counts"]["negative"]
    ratio = pos / (neg + 0.0001)

    if ratio > 2.5 and pos >= 10:
        return {
            "tone_status": "Highly Confident (Bullish)",
            "nlp_score": 10,
            "interpretation": (
                f"Management used markedly more confident than cautious language "
                f"({pos} positive vs {neg} cautious terms)."
            ),
        }
    if ratio < 0.6 and neg >= 10:
        return {
            "tone_status": "Highly Cautious (Bearish)",
            "nlp_score": -10,
            "interpretation": (
                f"Management language is notably defensive "
                f"({neg} cautious vs {pos} positive terms)."
            ),
        }
    return {
        "tone_status": "Neutral",
        "nlp_score": 0,
        "interpretation": f"Management language is balanced ({pos} positive, {neg} cautious terms).",
    }


@cached(key_func=lambda symbol: f"management_tone:{symbol.upper()}")
@log_exceptions()
def analyze_management_tone(symbol: str) -> dict[str, Any]:
    """Tone of the latest earnings call, or an explicit statement that it was unread.

    Returns ``unavailable()`` when no real transcript could be fetched. That is
    the point of the rewrite: previously this returned a Neutral tone verdict in
    exactly that case, and no caller could tell a balanced call from a call
    nobody read.
    """
    payload = get_earnings_transcript(symbol)

    if not is_real_transcript(payload):
        return unavailable(
            _TRANSCRIPT_SOURCE,
            f"no earnings-call transcript could be retrieved for {symbol.upper()} — the "
            "provider returned a limit notice or had no filing.",
            symbol=symbol.upper(),
        )

    # Score what management said, not the header this codebase prepended.
    scored = score_text(transcript_body(payload))
    if scored["total_words"] < MIN_SCORABLE_WORDS:
        return unavailable(
            _TRANSCRIPT_SOURCE,
            f"the transcript retrieved for {symbol.upper()} is too short to characterise "
            f"({scored['total_words']} words).",
            symbol=symbol.upper(),
        )

    result = {
        "symbol": symbol.upper(),
        **_classify(scored),
        "counts": scored["counts"],
        "per_1k": scored["per_1k"],
        "transcript_words": scored["total_words"],
        "lexicon": LEXICON_NOTE,
    }
    period = parse_transcript_period(payload)
    if period:
        result["year"], result["quarter"] = period
    return result


def _previous_period(year: int, quarter: int) -> tuple[int, int]:
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


@cached(key_func=lambda symbol: f"management_tone_delta:{symbol.upper()}")
@log_exceptions()
def compare_management_tone(symbol: str) -> dict[str, Any]:
    """Quarter-over-quarter tone CHANGE — 5.4's actual signal.

    Both quarters are scored with the same lexicon and compared per 1,000 words,
    so neither the lexicon's incompleteness nor a difference in transcript length
    moves the result.

    Returns ``unavailable()`` unless BOTH calls were genuinely read. A delta
    against a quarter that could not be fetched is not a weaker signal — it is a
    different number wearing the same name.
    """
    latest_payload = get_earnings_transcript(symbol)
    if not is_real_transcript(latest_payload):
        return unavailable(
            _TRANSCRIPT_SOURCE,
            f"no current transcript for {symbol.upper()}, so there is nothing to compare.",
            symbol=symbol.upper(),
        )

    period = parse_transcript_period(latest_payload)
    if not period:
        return unavailable(
            _TRANSCRIPT_SOURCE,
            f"the transcript for {symbol.upper()} does not state its quarter, so the prior "
            "call cannot be identified without guessing at it.",
            symbol=symbol.upper(),
        )

    year, quarter = period
    prior_year, prior_quarter = _previous_period(year, quarter)
    prior_payload = get_earnings_transcript(symbol, year=prior_year, quarter=prior_quarter)
    if not is_real_transcript(prior_payload):
        return unavailable(
            _TRANSCRIPT_SOURCE,
            f"the Q{prior_quarter} {prior_year} call for {symbol.upper()} could not be "
            "retrieved, so no quarter-over-quarter comparison is possible.",
            symbol=symbol.upper(),
        )

    latest = score_text(transcript_body(latest_payload))
    prior = score_text(transcript_body(prior_payload))
    if min(latest["total_words"], prior["total_words"]) < MIN_SCORABLE_WORDS:
        return unavailable(
            _TRANSCRIPT_SOURCE,
            f"one of the two transcripts for {symbol.upper()} is too short to compare.",
            symbol=symbol.upper(),
        )

    deltas = {
        name: round(latest["per_1k"][name] - prior["per_1k"][name], 2)
        for name in CATEGORIES
    }
    shifts = [
        {"category": name, "change_per_1k": value}
        for name, value in sorted(deltas.items(), key=lambda kv: -abs(kv[1]))
        if abs(value) >= MATERIAL_DELTA_PER_1K
    ]

    return {
        "symbol": symbol.upper(),
        "current": {"year": year, "quarter": quarter, "per_1k": latest["per_1k"],
                    "words": latest["total_words"]},
        "prior": {"year": prior_year, "quarter": prior_quarter, "per_1k": prior["per_1k"],
                  "words": prior["total_words"]},
        "delta_per_1k": deltas,
        "material_shifts": shifts,
        "interpretation": _describe_shift(shifts, quarter, prior_quarter),
        "lexicon": LEXICON_NOTE,
        # 2.7: every number here was counted from two transcripts that were
        # actually read, not assumed.
        "basis": "measured",
    }


def _describe_shift(shifts: list[dict[str, Any]], quarter: int, prior_quarter: int) -> str:
    """Plain reading of the change — including an explicit no-change statement.

    "No material shift" is a RESULT and is stated as one. Emitting nothing for a
    quiet quarter is how a live signal comes to look identical to a dead one.
    """
    if not shifts:
        return (
            f"No material change in management language between Q{prior_quarter} and "
            f"Q{quarter}. Both calls were read and compared; the tone did not move."
        )

    phrases = {
        "negative": ("more cautious language", "less cautious language"),
        "positive": ("more confident language", "less confident language"),
        "uncertainty": ("more hedging", "less hedging"),
        "litigious": ("more legal-process language", "less legal-process language"),
        "constraining": ("more language about obligations and limits",
                         "less language about obligations and limits"),
    }
    parts = []
    for shift in shifts:
        up, down = phrases[shift["category"]]
        direction = up if shift["change_per_1k"] > 0 else down
        parts.append(f"{direction} ({shift['change_per_1k']:+.2f} per 1,000 words)")
    return f"Versus Q{prior_quarter}, management used " + ", ".join(parts) + "."
