"""Regression tests for the NewsAnalyst coverage guarantee + data-integrity guards.

Covers the fixes from the adversarial review of the US/global/Canadian coverage change:
  * The coverage floor (US + global + Canadian buckets) actually reaches synthesis on
    broad runs — including when the planner answers in prose with NO tool calls
    (the `if result.tool_calls or tool_outputs:` gate).
  * Narrow single-name queries (planner used only get_specific_news) SKIP the broad
    macro force-fetches.
  * Degraded/placeholder strings (search-throttle, "no news available") never reach
    synthesis — they are filtered, and a degraded *planner* value is overwritten.
  * `_quote_is_sane` drops split/stale-quote artifacts but keeps real large moves.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agent.nodes.news_analyst as na
import agent.utils as autils
import tools.canadian_market as cm
import tools.news_sources as ns

USABLE = (
    "**Headline about the market today**\n*Source: Wire | 2026-06-21*\n"
    "Link: https://example.com/story\nSummary: a real story with enough length to pass."
)
THROTTLE = (
    "⚠️ [Search Throttled]\n\nAnother search is already in progress. "
    "Results will be available shortly.\n\nQuery: global stock markets today europe asia"
)


def _planner_result(tool_names, content=""):
    """A fake planner AIMessage with the given tool calls (or prose if empty)."""
    tcs = [{"name": n, "args": {}, "id": f"id{i}", "type": "tool_call"}
           for i, n in enumerate(tool_names)]
    return AIMessage(content=content, tool_calls=tcs)


@pytest.fixture
def run_node(monkeypatch):
    """Drive news_analyst_node with all I/O mocked; return the captured synthesis prompt.

    The returned callable takes (planner_tool_names, planner_tool_outputs, fetch_overrides)
    and returns the SystemMessage text fed to the synthesis LLM, whose `## <key> Results:`
    headers reveal exactly which buckets reached synthesis.
    """
    captured = {}

    class _Chunk:
        content = "SYNTHETIC REPORT BODY"

    def fake_safe_stream(llm, messages, is_cancelled):
        captured["system"] = messages[0].content
        yield _Chunk()

    # Synthesis plumbing
    monkeypatch.setattr(autils, "safe_stream", fake_safe_stream)
    monkeypatch.setattr(autils, "send_stream", lambda *a, **k: None)
    monkeypatch.setattr(na, "send_status", lambda *a, **k: None)
    monkeypatch.setattr(na, "get_sonnet_llm", lambda *a, **k: object())
    monkeypatch.setattr(na, "get_user_context_string", lambda *a, **k: "")

    # Default forced-fetch sources return usable news; movers return a populated screen.
    monkeypatch.setattr(ns, "get_market_news", lambda *a, **k: USABLE, raising=True)
    monkeypatch.setattr(ns, "get_global_market_news", lambda *a, **k: USABLE, raising=True)
    monkeypatch.setattr(ns, "get_company_news", lambda *a, **k: USABLE, raising=True)
    monkeypatch.setattr(cm, "get_canadian_market_news", lambda *a, **k: USABLE, raising=True)
    monkeypatch.setattr(cm, "scan_tsx_movers", lambda *a, **k: {
        "market_status": "TSX +0.1%", "top_gainers": [{"symbol": "X.TO"}],
        "top_losers": "None", "most_active_large_cap": "None", "note": "live",
    }, raising=True)
    # Holdings source for the per-position news pull (cash excluded; top-by-value).
    import tools.portfolio_csv as pcsv
    monkeypatch.setattr(pcsv, "get_portfolio_summary", lambda *a, **k: {
        "holdings": [
            {"symbol": "AAPL", "value_cad": 5000, "is_cash_or_pension": False},
            {"symbol": "SHOP.TO", "value_cad": 3000, "is_cash_or_pension": False},
            {"symbol": "CASH", "value_cad": 9999, "is_cash_or_pension": True},
        ],
    }, raising=True)

    def _run(tool_names, tool_outputs=None, fetch_overrides=None,
             stream_impl=None, invoke_impl=None, return_final=False):
        for name, fn in (fetch_overrides or {}).items():
            target = ns if hasattr(ns, name) else cm
            monkeypatch.setattr(target, name, fn, raising=True)
        if stream_impl is not None:
            monkeypatch.setattr(autils, "safe_stream", stream_impl, raising=True)
        if invoke_impl is not None:
            monkeypatch.setattr(na, "safe_invoke", invoke_impl, raising=True)
        result = _planner_result(tool_names, content="prose" if not tool_names else "")
        state = {"messages": [HumanMessage(content="what's happening in markets?")]}
        outs = dict(tool_outputs or {})

        def fake_gather(_state):
            return result, list(state["messages"]) + [result], outs

        monkeypatch.setattr(na, "gather_news_tool_outputs", fake_gather)
        out = na.news_analyst_node(state)
        msgs = out.get("messages", []) if isinstance(out, dict) else []
        captured["final"] = str(msgs[-1].content) if msgs else ""
        return captured["final"] if return_final else captured.get("system", "")

    return _run


def test_broad_run_includes_all_regions(run_node):
    sys = run_node(["get_market_headlines"])
    assert "## get_market_headlines Results:" in sys
    assert "## global_market_news Results:" in sys
    assert "## canadian_market_news Results:" in sys
    assert "## scan_tsx_movers Results:" in sys


def test_broad_run_includes_holdings_news(run_node):
    # Per-position news must reach synthesis so the report can assess impact on the
    # user's ACTUAL holdings, not just whatever surfaces in the macro buckets.
    sys = run_node(["get_market_headlines"])
    assert "## holdings_news Results:" in sys


def test_prose_only_planner_still_synthesizes_with_coverage(run_node):
    # No tool calls at all: the old `if result.tool_calls:` gate would discard everything.
    sys = run_node([])
    assert sys, "synthesis prompt should be built even when the planner emits no tool calls"
    assert "## global_market_news Results:" in sys
    assert "## canadian_market_news Results:" in sys


def test_narrow_ticker_query_skips_macro_buckets(run_node):
    sys = run_node(["get_specific_news"], tool_outputs={"get_specific_news": USABLE})
    assert "## get_specific_news Results:" in sys
    assert "## global_market_news Results:" not in sys
    assert "## canadian_market_news Results:" not in sys
    assert "## scan_tsx_movers Results:" not in sys
    assert "## holdings_news Results:" not in sys


def test_throttle_placeholder_is_not_stored_as_news(run_node):
    sys = run_node(
        ["get_market_headlines"],
        fetch_overrides={"get_global_market_news": lambda *a, **k: THROTTLE},
    )
    assert "## global_market_news Results:" not in sys  # placeholder dropped
    assert "Another search is already in progress" not in sys


def test_degraded_planner_value_is_overwritten_by_forced_fetch(run_node):
    # Planner produced a throttled get_market_headlines; the forced fetch must replace it.
    sys = run_node(
        ["get_market_headlines"],
        tool_outputs={"get_market_headlines": THROTTLE},
    )
    assert "## get_market_headlines Results:" in sys
    assert "Another search is already in progress" not in sys
    assert "https://example.com/story" in sys  # the usable forced-fetch content


CONTENT_FILTER_ERR = (
    "Error code: 400 - {'choices': [{'message': {'content': ''}, "
    "'finish_reason': 'content_filter', 'content_filter_results': {'error': "
    "{'code': 'content_filter', 'message': 'ResponsibleAI result indicated block action.'}}}]}"
)


def test_content_filter_block_recovers_via_neutral_retry(run_node):
    """Azure RAI blocks pass 1 (empty completion). The node must retry once with
    neutral framing and render THAT report — not collapse to the raw fallback dump."""
    def blocked_stream(llm, messages, is_cancelled):
        raise RuntimeError(CONTENT_FILTER_ERR)
        yield  # pragma: no cover  (makes this a generator)

    seen = {}

    def neutral_invoke(llm, messages, *a, **k):
        seen["system"] = messages[0].content
        return AIMessage(content="NEUTRAL MARKET REPORT — energy-supply risk, sources below.")

    final = run_node(
        ["get_market_headlines"],
        stream_impl=blocked_stream,
        invoke_impl=neutral_invoke,
        return_final=True,
    )
    assert "NEUTRAL MARKET REPORT" in final
    assert "data summary" not in final  # did NOT fall back to the raw dump
    assert "rendering_constraint" in seen["system"]  # neutral framing was applied on retry


def test_double_block_renders_clean_fallback_not_raw_dict(run_node):
    """Both passes blocked -> clean fallback. The Fear & Greed dict must be
    formatted, never dumped as a raw '{...}' Python repr."""
    def blocked_stream(llm, messages, is_cancelled):
        raise RuntimeError(CONTENT_FILTER_ERR)
        yield  # pragma: no cover

    def blocked_invoke(llm, messages, *a, **k):
        raise RuntimeError(CONTENT_FILTER_ERR)

    fg = ("{'indicator': 'Fear & Greed Index', 'score': 28, 'rating': 'fear', "
          "'implication': 'Market pessimistic, potential opportunity'}")
    final = run_node(
        ["get_market_headlines"],
        tool_outputs={"get_fear_greed": fg},
        stream_impl=blocked_stream,
        invoke_impl=blocked_invoke,
        return_final=True,
    )
    assert "data summary" in final          # clean fallback engaged
    assert "Fear & Greed Index: 28/100" in final  # formatted, not raw
    assert "'indicator'" not in final       # raw dict repr never leaks


# ── pure formatter unit tests ────────────────────────────────────────────────
def test_format_fear_greed_renders_clean_line():
    raw = "{'score': 35, 'rating': 'fear', 'implication': 'overly negative'}"
    out = na._format_fear_greed(raw)
    assert "Fear & Greed Index: 35/100 — Fear" in out
    assert "overly negative" in out
    assert "{" not in out


def test_format_movers_renders_lines_not_json():
    movers = [{"symbol": "ATD.TO", "name": "COUCHE-TARD", "price": "C$91.87", "change": "+11.68%"}]
    out = na._format_movers(movers)
    assert "ATD.TO (COUCHE-TARD): C$91.87 +11.68%" in out
    assert "{" not in out
    assert na._format_movers("None") == ""  # non-list input is safe


# ── pure data-integrity guard ────────────────────────────────────────────────
def test_quote_is_sane_keeps_real_move_drops_artifacts():
    real = {"regularMarketPrice": 41.92, "regularMarketPreviousClose": 51.40,
            "regularMarketChangePercent": -18.44}
    split_artifact = {"regularMarketPrice": 10.0, "regularMarketPreviousClose": 20.0,
                      "regularMarketChangePercent": -2.0}
    no_price = {"regularMarketPrice": None, "regularMarketPreviousClose": 5.0,
                "regularMarketChangePercent": -1.0}
    missing_prev = {"regularMarketPrice": 8.0, "regularMarketChangePercent": 3.0}

    assert cm._quote_is_sane(real) is True          # genuine crash kept
    assert cm._quote_is_sane(split_artifact) is False  # inconsistent -> dropped
    assert cm._quote_is_sane(no_price) is False
    assert cm._quote_is_sane(missing_prev) is True   # can't validate -> keep
