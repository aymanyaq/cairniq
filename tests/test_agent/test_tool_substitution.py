"""Roadmap 6.2 — the substitution decision, tested against the real tool registry.

Deliberately NOT a re-derivation of the table: the 2.3 build shipped a first
draft of its judge tests that re-implemented the cap arithmetic in the test file
and passed against nothing. So every assertion here either drives the real
function against real registered tools, or checks a property of the table that
the table itself cannot satisfy by construction.
"""
from __future__ import annotations

import pytest

from agent.tool_registry import ALL_TOOLS
from agent.tool_substitution import (
    SUBSTITUTION_MARKER,
    TOOL_SUBSTITUTES,
    accepts_args,
    is_substituted,
    pick_substitute,
    run_substitute,
    soft_failure_reason,
    substitution_notice,
)
from tools.tool_errors import unavailable

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

def test_every_name_in_the_table_is_a_registered_tool():
    """The check TOOL_RELATIONSHIPS never had.

    That graph carries 21 dead keys and 35 dead values; the retriever skips them
    silently, so nothing ever failed. A substitution table that rots the same way
    would degrade into "no substitute found" without saying so.
    """
    names = set(TOOL_SUBSTITUTES) | {v for vs in TOOL_SUBSTITUTES.values() for v in vs}
    dead = sorted(n for n in names if n not in TOOL_MAP)
    assert not dead, f"TOOL_SUBSTITUTES names tools that are not registered: {dead}"


def test_no_tool_substitutes_for_itself():
    for name, subs in TOOL_SUBSTITUTES.items():
        assert name not in subs, f"{name} lists itself as its own substitute"
        assert len(set(subs)) == len(subs), f"{name} has duplicate substitutes"


def test_no_substitute_across_a_jurisdiction_or_subject_change():
    """Encodes the reason this table is curated rather than derived.

    TOOL_RELATIONSHIPS answers a failed `get_macro_overview` with
    `get_canada_macro` — Canadian macro presented as the US macro that was asked
    for. No entry may reintroduce that, and where no honest equivalent exists the
    correct outcome is a Data Gap, not a near-miss.
    """
    assert "get_macro_overview" not in TOOL_SUBSTITUTES
    assert "get_sentiment" not in TOOL_SUBSTITUTES
    for subs in TOOL_SUBSTITUTES.values():
        assert "get_canada_macro" not in subs
        assert "analyze_reddit_sentiment" not in subs


# ---------------------------------------------------------------------------
# Failure detection — the soft half is the half that matters
# ---------------------------------------------------------------------------

def test_unavailable_payload_is_a_failure_with_its_reason():
    payload = unavailable("FMP", "FMP_API_KEY not configured — add it in Settings")
    reason = soft_failure_reason(payload)
    assert reason is not None
    assert "FMP" in reason and "not configured" in reason


def test_stringified_unavailable_payload_is_still_a_failure():
    """Some nodes str() the observation before this module sees it."""
    payload = str(unavailable("Tavily", "plan quota exhausted"))
    reason = soft_failure_reason(payload)
    assert reason is not None
    assert "Tavily" in reason


@pytest.mark.parametrize("observation", [
    {},
    [],
    None,
    "",
    {"symbol": "AAPL", "price": 1.0},
    [{"symbol": "AAPL"}],
])
def test_a_legitimately_empty_result_is_not_a_failure(observation):
    """`tools.tool_errors` reserves the empty shape for "no data exists" — a real
    answer. Substituting on it would re-ask a question that was already answered."""
    assert soft_failure_reason(observation) is None


# ---------------------------------------------------------------------------
# Arg compatibility, against real schemas
# ---------------------------------------------------------------------------

def test_accepts_exact_args():
    assert accepts_args(TOOL_MAP["get_realtime_quote"], {"symbol": "AAPL"})


def test_rejects_an_argument_the_substitute_does_not_take():
    # get_price_history takes (symbol, days); get_realtime_quote takes (symbol).
    assert not accepts_args(TOOL_MAP["get_realtime_quote"], {"symbol": "AAPL", "days": 30})


def test_optional_args_need_not_be_supplied():
    # `days` carries a default, so symbol alone is a valid call.
    assert accepts_args(TOOL_MAP["get_price_history"], {"symbol": "AAPL"})


def test_missing_required_arg_is_rejected():
    assert not accepts_args(TOOL_MAP["get_realtime_quote"], {})


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def test_picks_the_first_compatible_equivalent():
    picked = pick_substitute("get_realtime_quote", {"symbol": "AAPL"}, TOOL_MAP)
    assert picked.name == "get_stock_quote"
    assert picked.args == {"symbol": "AAPL"}


def test_falls_through_to_the_second_when_the_first_was_already_attempted():
    picked = pick_substitute(
        "get_realtime_quote", {"symbol": "AAPL"}, TOOL_MAP,
        attempted={"get_realtime_quote", "get_stock_quote"},
    )
    assert picked.name == "fetch_fundamentals"


def test_returns_none_when_every_equivalent_was_already_attempted():
    picked = pick_substitute(
        "get_realtime_quote", {"symbol": "AAPL"}, TOOL_MAP,
        attempted={"get_realtime_quote", "get_stock_quote", "fetch_fundamentals"},
    )
    assert picked is None


def test_returns_none_for_a_tool_with_no_curated_equivalent():
    assert pick_substitute("get_macro_overview", {}, TOOL_MAP) is None


def test_returns_none_when_the_substitute_is_not_in_the_node_tool_map():
    """Analyst nodes carry a RETRIEVED subset, not the full registry."""
    narrow = {"get_realtime_quote": TOOL_MAP["get_realtime_quote"]}
    assert pick_substitute("get_realtime_quote", {"symbol": "AAPL"}, narrow) is None


def test_incompatible_args_block_the_substitution_rather_than_reshaping_the_call():
    # A quote call carrying `days` fits get_price_history but not the curated
    # equivalents — nothing may be dropped to make it fit.
    assert pick_substitute("get_realtime_quote", {"symbol": "AAPL", "days": 30}, TOOL_MAP) is None


def test_never_substitutes_a_tool_for_itself_even_if_the_table_were_wrong():
    assert pick_substitute("get_stock_quote", {"symbol": "AAPL"}, TOOL_MAP).name != "get_stock_quote"


# ---------------------------------------------------------------------------
# Declared arg renames
# ---------------------------------------------------------------------------

def test_declared_rename_passes_the_same_value_under_the_new_name():
    """search_multi_source(topic=X) → perform_search(query=X). Same string, and
    the news analyst has no other covered edge at all."""
    picked = pick_substitute("search_multi_source", {"topic": "NVDA earnings"}, TOOL_MAP)
    assert picked.name == "perform_search"
    assert picked.args == {"query": "NVDA earnings"}


def test_the_search_rename_is_one_directional():
    """perform_search runs a query as-is; search_multi_source appends market
    keywords to it. Substituting the latter for the former would change the
    question, not the source."""
    assert pick_substitute("perform_search", {"query": "is rust worth learning"}, TOOL_MAP) is None


def test_renames_are_declared_only_for_registered_pairs():
    from agent.tool_substitution import ARG_RENAMES
    for (failed, sub), mapping in ARG_RENAMES.items():
        assert failed in TOOL_MAP and sub in TOOL_MAP, f"dead rename edge {failed}->{sub}"
        assert sub in TOOL_SUBSTITUTES.get(failed, ()), f"rename {failed}->{sub} has no substitution edge"
        for target in mapping.values():
            assert target in TOOL_MAP[sub].args, f"{sub} has no parameter {target}"


# ---------------------------------------------------------------------------
# The notice
# ---------------------------------------------------------------------------

def test_notice_names_both_tools_and_carries_the_marker():
    notice = substitution_notice("get_realtime_quote", "get_stock_quote", "timeout after 120s")
    assert SUBSTITUTION_MARKER in notice
    assert "get_realtime_quote" in notice
    assert "get_stock_quote" in notice
    assert "timeout after 120s" in notice
    assert is_substituted(notice)


def test_a_plain_result_is_not_marked_substituted():
    assert not is_substituted("Result:\n{'symbol': 'AAPL', 'price': 1.0}")
    assert not is_substituted(None)


# ---------------------------------------------------------------------------
# Running a stand-in
# ---------------------------------------------------------------------------

class _FakeTool:
    """Stands in for a LangChain BaseTool: has .invoke and an args schema."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.args = {"symbol": {"title": "Symbol", "type": "string"}}

    def invoke(self, args, config=None):
        if self._raises:
            raise self._raises
        return self._result


def test_run_substitute_returns_the_observation_on_success():
    observation, error = run_substitute(_FakeTool(result={"price": 101.5}), {"symbol": "AAPL"})
    assert error is None
    assert observation == {"price": 101.5}


def test_run_substitute_reports_a_raising_stand_in_rather_than_propagating():
    observation, error = run_substitute(_FakeTool(raises=RuntimeError("upstream 500")), {"symbol": "AAPL"})
    assert observation is None
    assert "upstream 500" in error


def test_run_substitute_treats_an_unavailable_stand_in_as_a_failure():
    """6.2 substitutes ONCE. A stand-in that is itself unavailable leaves the
    original Data Gap in place rather than starting a chain."""
    dead = _FakeTool(result=unavailable("yfinance", "rate limited"))
    observation, error = run_substitute(dead, {"symbol": "AAPL"})
    assert observation is None
    assert "rate limited" in error


# ---------------------------------------------------------------------------
# The seam with 2.3 — these two modules must agree on the wire format
# ---------------------------------------------------------------------------

def test_provenance_counts_a_real_substitution_notice():
    """Drives the ACTUAL notice text through the ACTUAL parser.

    The two are coupled by a string: provenance reads what substitution_notice
    writes. A test that hand-rolled the marker would keep passing after a reword
    while the live counter silently went to zero — the 6.2 version of the trap
    2.3's first judge tests fell into.
    """
    import tools.provenance as prov

    notice = substitution_notice("get_realtime_quote", "get_stock_quote", "error: HTTP 503")
    ctx = f"### Tool Call: get_realtime_quote({{'symbol': 'AAPL'}})\nResult:\n{notice}{{'price': 101.5}}"

    summary = prov.summarize_tool_context(ctx)

    assert summary["counts"]["substituted"] == 1
    assert summary["sources"][0]["substituted"] is True
    assert summary["sources"][0]["substitute"] == "get_stock_quote"


def test_the_footer_names_the_stand_in():
    import tools.provenance as prov

    notice = substitution_notice("get_realtime_quote", "get_stock_quote", "timed out after 120s")
    ctx = f"### Tool Call: get_realtime_quote({{'symbol': 'AAPL'}})\nResult:\n{notice}{{'price': 101.5}}"

    footer = prov.summarize_tool_context(ctx)["footer"]

    assert "get_realtime_quote via get_stock_quote" in footer


def test_a_recovered_call_does_not_mark_the_turn_degraded():
    """The stand-in returned real data, so the evidence is complete — it just
    came from the second choice. Capping the verdict for that would punish the
    recovery and make the degraded flag meaningless."""
    import tools.provenance as prov

    notice = substitution_notice("get_realtime_quote", "get_stock_quote", "error: HTTP 503")
    ctx = f"### Tool Call: get_realtime_quote({{'symbol': 'AAPL'}})\nResult:\n{notice}{{'price': 101.5}}"

    assert prov.summarize_tool_context(ctx)["degraded"] is False


def test_an_unrecovered_unavailable_still_marks_the_turn_degraded():
    """The complement: when no substitute ran, 2.3's behaviour is unchanged."""
    import tools.provenance as prov

    ctx = (
        "### Tool Call: get_insider_activity({'symbol': 'AAPL'})\nResult:\n"
        "{'status': 'unavailable', 'source': 'FMP', 'reason': 'FMP_API_KEY not configured'}"
    )

    summary = prov.summarize_tool_context(ctx)
    assert summary["degraded"] is True
    assert summary["counts"]["substituted"] == 0
