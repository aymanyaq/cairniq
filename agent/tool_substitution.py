"""Substitute a failed tool with a genuine equivalent — Advisor Roadmap 6.2.

When a tool fails, the analyst nodes append ``Error executing tool: ...`` to the
message list and move on. Inside a two-cycle budget that is often terminal: the
planner sees a dead result, has one cycle left, and the answer ships with a Data
Gap that a sibling tool could have filled. Two hand-rolled instances of the fix
already shipped per-source (a rate-limited FMP transcript falling through to web
search, a Tavily quota tripping a circuit); this is the general form.

**Why not ``TOOL_RELATIONSHIPS``.** The roadmap's own phrasing was "the first
healthy sibling from the existing clusters", and that is the one thing this
module deliberately does NOT do. That graph exists for *retrieval expansion* —
it means "related", not "equivalent". Its entry for ``get_macro_overview`` lists
``get_canada_macro``, so a first-healthy-sibling rule answers a question about US
macro with Canadian macro and says nothing. Substituting a DIFFERENT answer for
the requested one is the failure mode this codebase has paid for repeatedly, and
it is worse than the Data Gap it replaces, because a gap is visible and a wrong
source is not.

It is also, measurably, not a table you would want to build on: of its 154 edges
only 55 have both ends registered in ``ALL_TOOLS`` — 21 of 46 keys and 35 of 111
distinct values name tools that no longer exist. The retriever silently skips the
dead ones, so nothing ever complained.

So ``TOOL_SUBSTITUTES`` below is curated by hand, verified name-by-name against
the live registry, and every entry answers the SAME question as the tool it
stands in for. Where no honest equivalent exists — market sentiment, macro
indicators — there is no entry, and a Data Gap remains the correct outcome.

**Three guards, because a substitution that misfires is worse than none:**

1. *Arg compatibility, per call.* The substitute must accept every argument the
   failed call actually passed, and must not require one the call did not.
   Nothing is invented to make a substitution fit — inventing a ``"PORTFOLIO"``
   sentinel or a default year to satisfy a signature would be authoring input.
2. *Never chain, never repeat.* One substitution per failed call, never a
   substitute for a substitute, and never a tool already attempted this turn.
3. *Never silent.* The result carries :data:`SUBSTITUTION_MARKER` and a notice
   naming both tools. The rendered ``<tool_execution_context>`` block header
   carries the ORIGINAL tool's name — it is built from the planner's tool_call,
   not from what ran — so without the in-content notice the judge would audit the
   substitute's numbers against the failed tool's name and Rule 8 would be right
   to call it source fraud.

*On circuit breakers:* the one that exists (Tavily, ``tools.web_search``) degrades
rather than dies — an open circuit routes searches to DuckDuckGo instead of
failing them. Blocking substitution toward a search tool on that signal would
remove a working fallback, so there is no health-probe hook here. Add one only
for a circuit that makes a tool genuinely unusable.
"""
from __future__ import annotations

import re
from typing import Any

from tools.tool_errors import is_unavailable

# A stringified `unavailable()` payload, in either repr or JSON quoting. Some
# nodes hand the raw dict to this module and some have already str()'d it; both
# must be recognised as failure. Same pattern tools/provenance.py matches on.
_UNAVAILABLE_TEXT_RE = re.compile(r"['\"]status['\"]\s*:\s*['\"]unavailable['\"]")
_REASON_TEXT_RE = re.compile(r"['\"]reason['\"]\s*:\s*['\"]([^'\"]{1,200})['\"]")
_SOURCE_TEXT_RE = re.compile(r"['\"]source['\"]\s*:\s*['\"]([^'\"]{1,60})['\"]")

# Present in the ToolMessage content of every substituted result. tools/provenance
# counts occurrences of this to report a per-turn `substituted` figure — which is
# the only thing that will ever prove this module fired on a healthy box.
SUBSTITUTION_MARKER = "⚠️ SUBSTITUTED SOURCE"

# Curated equivalents, in preference order. Every name here is registered in
# ALL_TOOLS (enforced by tests/test_agent/test_tool_substitution.py, which is the
# check TOOL_RELATIONSHIPS never had). An entry means: "if the key fails, the
# value answers the same question from a different source."
TOOL_SUBSTITUTES: dict[str, tuple[str, ...]] = {
    # --- QUOTE / PRICE ---
    "get_realtime_quote": ("get_stock_quote", "fetch_fundamentals"),
    "get_stock_quote": ("get_realtime_quote", "fetch_fundamentals"),

    # --- FUNDAMENTALS / VALUATION ---
    "fetch_fundamentals": ("get_fundamentals_detailed", "get_valuation_metrics"),
    "get_fundamentals_detailed": ("fetch_fundamentals", "get_valuation_metrics"),
    "get_valuation_metrics": ("get_fundamentals_detailed", "fetch_fundamentals"),

    # --- INSIDER ACTIVITY ---
    "get_insider_activity": ("get_insider_short_interest", "check_smart_money"),
    "check_smart_money": ("get_insider_activity", "get_insider_short_interest"),
    "get_insider_short_interest": ("get_insider_activity", "check_smart_money"),

    # --- SHORT INTEREST ---
    "get_short_interest_data": ("get_insider_short_interest",),

    # --- CORRELATION ---
    "check_portfolio_correlation": ("get_correlation_analysis",),
    "get_correlation_analysis": ("check_portfolio_correlation",),

    # --- TECHNICALS ---
    # The first three are near-duplicates of one another. The last three are
    # one-directional on purpose: a full technical read answers "where is
    # support?", but support/resistance alone does not answer "analyse the
    # technicals", so the reverse edges are absent rather than partial.
    "analyze_technicals": ("run_technical_analysis", "analyze_technical_chart"),
    "run_technical_analysis": ("analyze_technicals", "analyze_technical_chart"),
    "analyze_technical_chart": ("analyze_technicals", "run_technical_analysis"),
    "analyze_patterns": ("analyze_technicals", "analyze_technical_chart"),
    "get_support_resistance": ("analyze_patterns", "analyze_technicals"),
    "get_ma_signals": ("analyze_technicals", "analyze_patterns"),

    # --- COMPANY NEWS (crosses providers: yfinance sentiment vs. web search) ---
    "get_stock_news": ("search_stock_news",),
    "search_stock_news": ("get_stock_news",),

    # --- WEB SEARCH ---
    # One-directional, and the asymmetry is the point. search_multi_source
    # appends "market analysis" / "impact stocks" / "TSX news" to every query;
    # its own docstring warns that this pollutes a non-financial search. So a
    # failed multi-angle search degrades safely to the plain one, while a failed
    # plain search must NOT be answered by the query-rewriting tool — that would
    # change the question, which is the thing this module refuses to do.
    "search_multi_source": ("perform_search",),

    # --- ANALYST COVERAGE ---
    "get_analyst_ratings": ("get_analyst_targets",),
    "get_analyst_targets": ("get_analyst_ratings",),

    # --- PORTFOLIO RISK METRICS ---
    "get_portfolio_risk_metrics": ("check_risk_metrics", "analyze_portfolio_risk"),
    "analyze_portfolio_risk": ("get_portfolio_risk_metrics", "check_risk_metrics"),
    "check_risk_metrics": ("get_portfolio_risk_metrics", "analyze_portfolio_risk"),

    # --- HOLDINGS ---
    "get_portfolio_snapshot": ("get_my_portfolio",),
    "get_my_portfolio": ("get_portfolio_snapshot",),

    # --- SECTOR EXPOSURE ---
    "get_portfolio_sectors": ("analyze_sectors",),
    "analyze_sectors": ("get_portfolio_sectors",),

    # --- EARNINGS DATES ---
    "get_earnings_data": ("get_earnings_calendar",),
    "get_earnings_calendar": ("get_earnings_data",),
}


# Declared parameter renames for edges whose two tools call the same thing by
# different names. This passes the SAME value under the substitute's parameter
# name; it never invents, drops or reshapes a value. `search_multi_source(topic=X)`
# and `perform_search(query=X)` both take a search string, and without this the
# strict arg check would reject every substitution in the news analyst — whose
# five-tool map is otherwise entirely uncovered, and which sits on Tavily, the
# flakiest source in the app and the origin of one of the two per-source patches
# 6.2 generalizes.
ARG_RENAMES: dict[tuple[str, str], dict[str, str]] = {
    ("search_multi_source", "perform_search"): {"topic": "query"},
}


class Substitution:
    """A chosen stand-in: which tool to run, and the args to run it with."""

    __slots__ = ("name", "args")

    def __init__(self, name: str, args: dict[str, Any]):
        self.name = name
        self.args = args

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"Substitution(name={self.name!r}, args={self.args!r})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Substitution)
            and self.name == other.name
            and self.args == other.args
        )


def rename_args(failed_name: str, substitute_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Apply the declared renames for this edge. Unmapped keys pass through."""
    mapping = ARG_RENAMES.get((failed_name, substitute_name))
    if not mapping:
        return dict(args or {})
    return {mapping.get(k, k): v for k, v in (args or {}).items()}


def soft_failure_reason(observation: Any) -> str | None:
    """Reason string if ``observation`` is a degraded `unavailable()` payload, else None.

    This is the failure kind that raises nothing. A missing API key returns a
    dict, the executor records it as a success, and the model narrates over it —
    it is both the most common real degradation and the one a
    substitute-on-exception rule would miss entirely. A legitimately EMPTY result
    is not a failure here: ``tools.tool_errors`` reserves the natural empty shape
    for "no data exists", which is a real answer and must not trigger a retry.
    """
    if is_unavailable(observation):
        source = observation.get("source") or "source"
        reason = observation.get("reason") or "unavailable"
        return f"{source} unavailable: {reason}"

    if isinstance(observation, str) and _UNAVAILABLE_TEXT_RE.search(observation):
        source = _SOURCE_TEXT_RE.search(observation)
        reason = _REASON_TEXT_RE.search(observation)
        return (
            f"{source.group(1) if source else 'source'} unavailable: "
            f"{reason.group(1) if reason else 'unavailable'}"
        )
    return None


def _arg_spec(tool: Any) -> dict[str, Any] | None:
    """The tool's argument schema, or None when it cannot be read.

    Unreadable means no substitution: `pick_substitute` refuses rather than
    guessing that a call will fit.
    """
    try:
        args = tool.args
    except Exception:  # noqa: BLE001 — a schema that raises is a schema we can't trust
        return None
    return args if isinstance(args, dict) else None


def accepts_args(tool: Any, args: dict[str, Any]) -> bool:
    """True when ``tool`` can be invoked with exactly ``args``.

    Required is derived from the absence of a ``default`` in the JSON-schema-ish
    dict LangChain exposes as ``tool.args`` — which avoids branching on the
    pydantic major version, and matches what ``model_fields[...].is_required()``
    reports.
    """
    spec = _arg_spec(tool)
    if spec is None:
        return False

    provided = set(args or {})
    if not provided.issubset(spec):
        return False

    required = {name for name, field in spec.items() if not (isinstance(field, dict) and "default" in field)}
    return required.issubset(provided)


def pick_substitute(
    failed_name: str,
    args: dict[str, Any],
    tool_map: dict[str, Any],
    attempted: set[str] | None = None,
) -> Substitution | None:
    """The best registered, arg-compatible, not-yet-tried equivalent — or None.

    ``attempted`` is every tool name already run this turn (successes included).
    Re-running one that already failed just spends the budget again, and
    re-running one that already SUCCEEDED would hand the planner a duplicate
    payload under a second name.
    """
    attempted = attempted or set()
    for candidate in TOOL_SUBSTITUTES.get(failed_name, ()):
        if candidate in attempted or candidate == failed_name:
            continue
        tool = tool_map.get(candidate)
        if tool is None:
            continue
        candidate_args = rename_args(failed_name, candidate, args)
        if not accepts_args(tool, candidate_args):
            continue
        return Substitution(candidate, candidate_args)
    return None


def substitution_notice(failed_name: str, substitute_name: str, reason: str) -> str:
    """The header prepended to a substituted result.

    Written at the model, not the user: it has to survive into
    ``<tool_execution_context>``, where the block is titled with the FAILED
    tool's name, and stop the judge attributing these numbers to a tool that
    never ran.
    """
    return (
        f"{SUBSTITUTION_MARKER}: `{failed_name}` failed ({reason}) and returned no data.\n"
        f"The result below came from `{substitute_name}` instead — a different source "
        f"answering the same question. Attribute it to `{substitute_name}`; do not "
        f"present it as output of `{failed_name}`, and say which source was used if "
        f"the distinction affects the answer.\n\n"
    )


def is_substituted(text: Any) -> bool:
    """True if a rendered tool result carries the substitution marker."""
    return isinstance(text, str) and SUBSTITUTION_MARKER in text


# Shorter than any node's primary tool budget on purpose: a substitution is a
# recovery attempt inside a turn that has already spent time failing, and the
# alternative to a slow stand-in is a Data Gap, not a better answer.
SUBSTITUTION_TIMEOUT_SECONDS = 60


def run_substitute(
    tool: Any,
    args: dict[str, Any],
    config: Any = None,
    timeout: int = SUBSTITUTION_TIMEOUT_SECONDS,
) -> tuple[Any, str | None]:
    """Run one substitute in a bounded thread. Returns ``(observation, error)``.

    ``error`` is non-None when the stand-in raised, timed out, or came back
    `unavailable()` in its turn — in every one of those cases the caller keeps
    the original failure, because 6.2 substitutes once and never chains.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FTimeout

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        if hasattr(tool, "invoke"):
            future = executor.submit(tool.invoke, args, config=config)
        else:
            future = executor.submit(lambda: tool(**args))
        try:
            observation = future.result(timeout=timeout)
        except _FTimeout:
            return None, f"substitute timed out after {timeout}s"
        except Exception as e:  # noqa: BLE001 — a dead stand-in leaves the gap intact
            return None, str(e)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    soft = soft_failure_reason(observation)
    if soft:
        return None, soft
    return observation, None
