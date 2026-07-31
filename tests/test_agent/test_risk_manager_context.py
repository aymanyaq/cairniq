"""Context selection for the RiskManager judge.

Regression cover for ways the judge audited the wrong thing:
- stale-`data_context` false-"source fraud": on a fast-path turn the judge must
  audit against THIS turn's tool outputs, not a heavy-path
  `tool_execution_context` left over from an earlier turn.
- half-the-evidence false-"source fraud": on a turn where an analyst ran tools AND
  the deep path then ran more, the judge must see BOTH — the analyst's in-state
  ToolMessages and the deep path's `data_context` publication, which never enters
  `messages` at all.
- retry-pass anchoring: on a compliance retry the judge must audit the REVISED
  draft, not re-issue its own prior failing verdict against the original draft.
- missing cost basis in the verification brief: Rule 8 orders the judge to verify
  cost-basis claims against the brief, so dropping those fields made every TRUE
  drawdown claim read as fabricated (2026-07-15: a true deep drawdown called SOURCE FRAUD).
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.nodes.risk_manager import (
    _build_judge_context,
    _build_portfolio_verification_brief,
    _build_tool_execution_context,
    _format_cost_basis,
)
from agent.utils import current_turn_key


def _analyst_tool_turn(user_text, tc_id, tool_name, tool_output):
    """A user message + an analyst AIMessage(tool_calls) + its ToolMessage result."""
    return [
        HumanMessage(content=user_text),
        AIMessage(content="", tool_calls=[{"id": tc_id, "name": tool_name, "args": {}}]),
        ToolMessage(content=tool_output, tool_call_id=tc_id, name=tool_name),
    ]


def test_fast_path_prefers_fresh_tools_over_stale_upstream():
    # This turn's analyst produced fresh evidence in-state...
    messages = _analyst_tool_turn(
        "scan my watchlist", "tc_watch", "run_watchlist_scan", "PFE fwd P/E 11.2; LLY insider buy $2M"
    )
    # ...while data_context still holds an EARLIER heavy-path (Trump) turn's context.
    stale_upstream = "### Tool Call: get_latest_trump_yaps({})\nResult:\nTrump posted about NATO."

    ctx = _build_tool_execution_context(messages, stale_upstream)

    assert "PFE fwd P/E 11.2" in ctx and "LLY insider buy" in ctx
    assert "get_latest_trump_yaps" not in ctx  # stale upstream must NOT be used
    assert "NATO" not in ctx


def test_heavy_path_falls_back_to_upstream_when_no_in_state_tools():
    # Heavy path (Trump) returns only its synthesis — no ToolMessages in state.
    messages = [
        HumanMessage(content="[System Instruction: Catalyst] EventScenario source=trump"),
        AIMessage(content="[DeepReasoning]: ### 📢 CATALYST ...", name="DeepReasoning"),
    ]
    fresh_upstream = "### Tool Call: get_latest_trump_yaps({})\nResult:\n50% tariff on chips announced."

    ctx = _build_tool_execution_context(messages, fresh_upstream)

    assert ctx == fresh_upstream  # only source of this turn's evidence


def test_no_tools_and_no_upstream_reports_none():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="[DeepReasoning]: Hello!", name="DeepReasoning"),
    ]
    ctx = _build_tool_execution_context(messages, None)
    assert ctx == "No tool calls executed in recent context."


def test_retry_directive_does_not_hide_this_turns_tools():
    # Fast-path turn's real tools, THEN the retry gate injects a synthetic correction
    # HumanMessage. The boundary must skip that directive so the tools remain visible.
    messages = _analyst_tool_turn(
        "should I trim NVDA?", "tc_nvda", "run_stock_deep_dive", "NVDA RSI 63.8; price $1128.02"
    )
    messages += [
        HumanMessage(content="<compliance_correction_required>\nFix the sizing.\n</compliance_correction_required>"),
        AIMessage(content="[DeepReasoning]: revised advice ...", name="DeepReasoning"),
    ]

    ctx = _build_tool_execution_context(messages, "STALE upstream from another turn")

    assert "NVDA RSI 63.8" in ctx and "price $1128.02" in ctx
    assert "STALE upstream" not in ctx


def test_only_current_turn_tools_included_not_prior_turns():
    # A prior analyst turn, then a NEW user turn whose analyst ran a different tool.
    messages = _analyst_tool_turn(
        "look at KO", "tc_ko", "run_stock_deep_dive", "KO div yield 3.1%"
    )
    messages += _analyst_tool_turn(
        "now look at PEP", "tc_pep", "run_stock_deep_dive", "PEP div yield 2.8%"
    )

    ctx = _build_tool_execution_context(messages, None)

    assert "PEP div yield 2.8%" in ctx     # current turn present
    assert "KO div yield 3.1%" not in ctx  # prior turn excluded (scoped to current turn)


# --- Retry-pass judge context (_build_judge_context) ---

_FAILING_VERDICT = (
    "[RiskManager]: \n\n---\n### 🛡️ Risk Assessment\n"
    "⚖️ **Verdict: [2/10]** — margin deployment violates sizing rules.\n\n"
    "🔴 **Risks:**\n- Deploying 40% margin into a single ticker (MU)"
)
_REVISION = "[DeepReasoning]: Revised: cancel the margin deployment and discard the MU idea."
_DEFAULT_CHECK_MSG = "Quick risk check on the above advice. Reply '✅ Risk Check Passed' if safe."


def _retry_pass_history():
    """Message state exactly as the compliance retry gate leaves it when the
    judge runs a second time: original draft → failing verdict → injected
    correction directive → DeepReasoning's revision."""
    return [
        HumanMessage(content="should I deploy margin into MU?"),
        AIMessage(content="[DeepReasoning]: Deploy 40% margin into MU immediately.", name="DeepReasoning"),
        AIMessage(content=_FAILING_VERDICT, name="RiskManager"),
        HumanMessage(content=(
            "<compliance_correction_required>\n"
            "Your previous response was flagged by the Risk Manager with CRITICAL violations.\n"
            f"{_FAILING_VERDICT}\n"
            "</compliance_correction_required>"
        )),
        AIMessage(content=_REVISION, name="DeepReasoning"),
    ]


def test_retry_pass_excludes_prior_verdict_from_judge_context():
    # The judge must not see its own 2/10 verdict (directly, or embedded in the
    # correction directive) — that's what it anchors on to re-flag the OLD draft.
    ctx = _build_judge_context(_retry_pass_history(), retry_count=1)

    joined = "\n".join(str(m.content) for m in ctx)
    assert "Verdict: [2/10]" not in joined
    assert not any(getattr(m, "name", None) == "RiskManager" for m in ctx)
    assert "cancel the margin deployment" in joined  # revision still under review


def test_retry_pass_check_message_pins_audit_to_the_revision():
    ctx = _build_judge_context(_retry_pass_history(), retry_count=1)

    assert isinstance(ctx[-1], HumanMessage)
    check = str(ctx[-1].content)
    assert "ONLY the most recent assistant message" in check
    assert "REVISION" in check
    assert "historical" in check
    # parse_risk_verdict's bare-pass fast path depends on this exact phrase
    assert "Reply '✅ Risk Check Passed' if safe." in check
    # ...and the message directly above the check IS the revision
    assert str(ctx[-2].content) == _REVISION


def test_retry_detected_from_correction_directive_when_counter_lost():
    # Same guards even if risk_retry_count didn't survive to this pass: the
    # correction directive is the current turn's last HumanMessage.
    ctx = _build_judge_context(_retry_pass_history(), retry_count=0)

    joined = "\n".join(str(m.content) for m in ctx)
    assert "Verdict: [2/10]" not in joined
    assert "REVISION" in str(ctx[-1].content)


def test_later_normal_turn_is_not_treated_as_retry():
    # A later turn in the same thread: the stale correction directive is still
    # in history, but a genuine user message follows it — normal audit rules.
    messages = _retry_pass_history() + [
        AIMessage(content="[RiskManager]: ⚖️ **Verdict: [9/10]** — clean.", name="RiskManager"),
        HumanMessage(content="thanks — now what about my NVDA position?"),
        AIMessage(content="[DeepReasoning]: NVDA looks fully valued here.", name="DeepReasoning"),
    ]

    ctx = _build_judge_context(messages, retry_count=0)

    assert str(ctx[-1].content) == _DEFAULT_CHECK_MSG
    # prior verdicts are only stripped on retry passes
    assert any(getattr(m, "name", None) == "RiskManager" for m in ctx)


# --- Cost basis in the verification brief (_format_cost_basis / _build_portfolio_verification_brief) ---

# A deep-loss position held in USD and reported in CAD: the judge sees only the
# base-currency position value, so without a cost basis the drawdown is unverifiable.
# Figures are invented and internally consistent: 40 × $42.00 USD × 1.40 = $2,352.00 CAD.
_LOSS_HOLDING = {
    "symbol": "XYZ",
    "shares": 40,
    "purchase_price": "$200.00",
    "current_price": "$42.00",
    "gain_loss": "-79.0%",
    "currency": "USD",
    "account": "Brokerage",
    "source": "Manual",
    "value_base": 2352.00,
    "allocation_pct": 0.47,
    "is_cash_or_pension": False,
}


def test_cost_basis_line_exposes_the_fields_rule_8_demands():
    line = _format_cost_basis(_LOSS_HOLDING)

    # Cost basis + current price + return must all be groundable from this line.
    assert "$200.00" in line
    assert "$42.00" in line
    assert "-79.0%" in line
    # Native currency is labelled so the judge does not read it as conflicting
    # with the CAD position value printed on the same brief line.
    assert "USD" in line


def test_missing_or_zero_cost_basis_still_flagged_unverifiable():
    # Rule 8's TRUE positive: no cost basis on file => claims really are unverifiable.
    for holding in ({**_LOSS_HOLDING, "purchase_price": "$0.00"}, {"symbol": "ABC"}):
        assert "unverifiable" in _format_cost_basis(holding)


def test_cash_pension_reports_no_cost_basis_without_false_alarm():
    line = _format_cost_basis({
        "symbol": "CASH", "shares": 5000, "purchase_price": "$1.00",
        "gain_loss": "+2.0%", "currency": "CAD", "is_cash_or_pension": True,
    })
    assert "cash/pension" in line
    assert "unverifiable" not in line  # cash has no cost basis by nature — not fraud


def test_brief_renders_cost_basis_for_each_holding(monkeypatch):
    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda *a, **k: {
            "profile": "alpha", "as_of": "2026-07-15T13:36:00", "is_stale": False,
            "sync_errors": [], "base_currency": "CAD", "total_value_base": 500000.0,
            "holdings": [_LOSS_HOLDING],
        },
    )

    brief = _build_portfolio_verification_brief()

    assert "XYZ" in brief
    assert "$2,352.00 CAD" in brief   # position value in base currency, as before
    assert "$200.00 USD cost" in brief  # ...and now the cost basis Rule 8 needs
    assert "-79.0%" in brief


# --- compliance-retry evidence union ------------------------------------------
# A retry re-plans from scratch and usually runs a NARROWER tool set than the pass
# it is fixing, while the revision is expected to carry figures forward from that
# first draft. Replacing the heavy path's tool_execution_context therefore deletes
# the result that grounds a carried-forward number, and the judge reads a true
# figure as Rule 8 SOURCE FRAUD. Observed 2026-07-21: a first draft whose Sharpe
# and Sortino ratios the judge had confirmed as "verified risk metrics" came back
# from its own retry at 4/10 CRITICAL_FAIL for "completely hallucinating" the same
# Sharpe ratio — the retry had simply not re-run check_risk_metrics.

_ROUND1_CTX = (
    "### Tool Call: check_risk_metrics({})\n"
    "Result:\nsharpe_ratio: 1.31, sortino_ratio: 2.05, volatility: 12.4%\n\n"
    "### Tool Call: get_portfolio_sectors({})\n"
    "Result:\nTechnology 41.0%"
)
_ROUND2_CTX = (
    "### Tool Call: get_portfolio_sectors({})\n"
    "Result:\nTechnology 41.0% (refreshed)\n\n"
    "### Tool Call: verify_portfolio_holdings({})\n"
    "Result:\nAAPL, MSFT, SHOP.TO, GME all held"
)


def test_retry_merge_keeps_evidence_the_retry_did_not_re_run():
    from agent.nodes.deep_reasoning import _merge_tool_contexts

    merged = _merge_tool_contexts(_ROUND1_CTX, _ROUND2_CTX)

    assert "sharpe_ratio: 1.31" in merged        # the carried-forward figure stays grounded
    assert "verify_portfolio_holdings" in merged  # the retry's own evidence is there too


def test_retry_merge_refreshes_a_re_run_tool_in_place():
    from agent.nodes.deep_reasoning import _merge_tool_contexts

    merged = _merge_tool_contexts(_ROUND1_CTX, _ROUND2_CTX)

    assert merged.count("### Tool Call: get_portfolio_sectors({})") == 1
    assert "Technology 41.0% (refreshed)" in merged


# --- heavy-path evidence on a turn that ALSO had analyst tools ----------------
# A turn's tool results arrive by two routes: an analyst's calls become
# ToolMessages in the graph state, while DeepReasoning's heavy path returns only
# its synthesis and publishes its results through data_context. Preferring the
# in-state route outright made the second route invisible whenever the first one
# had anything at all. Observed on a live turn: four MarketAnalyst
# ToolMessages were the entire evidence view, so EMA21 levels from
# structure_trade_setup and a decomposed sector weight from
# check_portfolio_allocation — both genuinely fetched, one cycle later — were
# called invented in a 2/10 SOURCE FRAUD verdict.

_HEAVY_PATH_CTX = (
    "### Tool Call: structure_trade_setup({'symbol': 'ZZA'})\n"
    "Result:\nPrice is below trend support (EMA21 at $242.77).\n\n"
    "### Tool Call: check_portfolio_allocation({})\n"
    "Result:\n{'sector_allocation': {'Technology': '44.0%'}, "
    "'basis': 'look-through; ETFs and funds decomposed into sector sleeves'}"
)


def _mixed_route_turn():
    """The incident's shape: an analyst tool in state, the deep path's results not."""
    messages = _analyst_tool_turn(
        "what should I do today?", "tc_pulse", "get_market_pulse", "VIX 14.2; SPY +0.4%"
    )
    messages.append(AIMessage(content="[DeepReasoning]: ... EMA21 $242.77 ...", name="DeepReasoning"))
    return messages


def test_this_turns_heavy_path_evidence_joins_the_analysts():
    messages = _mixed_route_turn()

    ctx = _build_tool_execution_context(
        messages, _HEAVY_PATH_CTX, upstream_turn_key=current_turn_key(messages)
    )

    # Both routes, one evidence view.
    assert "VIX 14.2" in ctx                    # analyst, from the ToolMessages
    assert "EMA21 at $242.77" in ctx            # deep path, from data_context
    assert "'Technology': '44.0%'" in ctx


def test_upstream_from_another_turn_is_still_refused():
    # The stale-evidence guard survives the union: a key that isn't this turn's
    # buys the upstream copy nothing, however fresh its contents look.
    messages = _mixed_route_turn()

    ctx = _build_tool_execution_context(
        messages, _HEAVY_PATH_CTX, upstream_turn_key="some-earlier-turn"
    )

    assert "VIX 14.2" in ctx
    assert "EMA21 at $242.77" not in ctx


def test_unstamped_upstream_is_refused_when_this_turn_has_its_own_tools():
    # No stamp at all (a checkpoint written before the stamp existed): unprovable
    # provenance, so it may not join evidence this turn definitely produced.
    messages = _mixed_route_turn()

    ctx = _build_tool_execution_context(messages, _HEAVY_PATH_CTX)

    assert "VIX 14.2" in ctx
    assert "EMA21 at $242.77" not in ctx


def test_pure_heavy_path_turn_still_uses_upstream_without_a_stamp():
    # No in-state ToolMessages: the upstream copy is this turn's ONLY evidence,
    # which is the pre-existing fallback and must not regress.
    messages = [
        HumanMessage(content="[System Instruction: Catalyst] source=x"),
        AIMessage(content="[DeepReasoning]: ### 📢 CATALYST ...", name="DeepReasoning"),
    ]
    assert _build_tool_execution_context(messages, _HEAVY_PATH_CTX) == _HEAVY_PATH_CTX


def test_union_prefers_the_heavy_paths_copy_of_a_re_run_call():
    # The deep path runs downstream of the analysts, so on a collision its result
    # is the newer of the two.
    messages = _analyst_tool_turn(
        "sector check", "tc_alloc", "check_portfolio_allocation", "stale first read"
    )
    upstream = "### Tool Call: check_portfolio_allocation({})\nResult:\nTechnology 44.0%"

    ctx = _build_tool_execution_context(
        messages, upstream, upstream_turn_key=current_turn_key(messages)
    )

    assert ctx.count("### Tool Call: check_portfolio_allocation({})") == 1
    assert "Technology 44.0%" in ctx
    assert "stale first read" not in ctx


def test_a_retry_pass_resolves_to_the_same_turn_key():
    # The retry gate injects a synthetic HumanMessage mid-turn. If that changed
    # the key, the revision's judge pass would silently lose the heavy path's
    # evidence — the exact failure this stamp exists to prevent.
    messages = _mixed_route_turn()
    before = current_turn_key(messages)
    messages += [
        HumanMessage(content="<compliance_correction_required>\nfix it\n</compliance_correction_required>"),
        AIMessage(content="[DeepReasoning]: revised ...", name="DeepReasoning"),
    ]

    assert current_turn_key(messages) == before

    ctx = _build_tool_execution_context(messages, _HEAVY_PATH_CTX, upstream_turn_key=before)
    assert "EMA21 at $242.77" in ctx
    assert "VIX 14.2" in ctx


def test_turn_key_changes_when_the_user_speaks_again():
    turn_one = [HumanMessage(content="look at KO", id="m1")]
    turn_two = turn_one + [
        AIMessage(content="[DeepReasoning]: KO ...", name="DeepReasoning"),
        HumanMessage(content="now PEP", id="m2"),
    ]

    assert current_turn_key(turn_one) != current_turn_key(turn_two)
    assert current_turn_key([]) == ""  # no user message => no key, never matches


def test_deep_reasoning_and_risk_manager_agree_on_the_key():
    """The stamp is only useful if writer and reader compute it identically."""
    from agent.nodes.deep_reasoning import current_turn_key as writer_key

    messages = _mixed_route_turn()
    assert writer_key(messages) == current_turn_key(messages) != ""


# --- the writer's half of the contract (_publish_tool_evidence) ---------------

def _react_loop_messages(tc_id, tool_name, args, output):
    """The planner turn + result as they appear INSIDE the heavy path's ReAct
    loop — messages that never reach the graph state."""
    return [
        AIMessage(content="", tool_calls=[{"id": tc_id, "name": tool_name, "args": args}]),
        ToolMessage(content=output, tool_call_id=tc_id, name=tool_name),
    ]


def test_heavy_path_stamps_the_turn_it_published_for():
    from agent.nodes.deep_reasoning import _publish_tool_evidence

    state = {"messages": [HumanMessage(content="what should I do?", id="turn-7")], "data_context": {}}
    loop = _react_loop_messages(
        "tc_1", "structure_trade_setup", {"symbol": "ZZA"}, "EMA21 at $242.77"
    )

    published = _publish_tool_evidence(state, loop, {})

    assert "EMA21 at $242.77" in published["tool_execution_context"]
    assert published["tool_execution_turn"] == "turn-7"


def test_published_evidence_is_the_full_result_not_the_compacted_replay():
    from agent.nodes.deep_reasoning import _publish_tool_evidence

    state = {"messages": [HumanMessage(content="q", id="t1")], "data_context": {}}
    loop = _react_loop_messages("tc_1", "check_risk_metrics", {}, "…[truncated replay]")

    published = _publish_tool_evidence(
        state, loop, {"tc_1": "sharpe_ratio: 1.31, sortino_ratio: 2.05"}
    )

    assert "sharpe_ratio: 1.31" in published["tool_execution_context"]
    assert "truncated replay" not in published["tool_execution_context"]


def test_a_toolless_pass_leaves_earlier_evidence_and_its_stamp_alone():
    from agent.nodes.deep_reasoning import _publish_tool_evidence

    state = {
        "messages": [HumanMessage(content="q", id="t1")],
        "data_context": {"tool_execution_context": _ROUND1_CTX, "tool_execution_turn": "t1"},
    }

    published = _publish_tool_evidence(state, [AIMessage(content="no tools needed")], {})

    assert published["tool_execution_context"] == _ROUND1_CTX
    assert published["tool_execution_turn"] == "t1"


def test_writer_and_reader_round_trip_through_data_context():
    """End to end across the seam: what the deep path publishes is what the judge
    reads, joined to the analyst evidence already in state."""
    from agent.nodes.deep_reasoning import _publish_tool_evidence

    messages = _analyst_tool_turn(
        "check my tech weight", "tc_pulse", "get_market_pulse", "VIX 14.2"
    )
    published = _publish_tool_evidence(
        {"messages": messages, "data_context": {}},
        _react_loop_messages(
            "tc_alloc", "check_portfolio_allocation", {}, "Technology 44.0% (look-through)"
        ),
        {},
    )
    messages.append(AIMessage(content="[DeepReasoning]: …", name="DeepReasoning"))

    ctx = _build_tool_execution_context(
        messages,
        published["tool_execution_context"],
        published["tool_execution_turn"],
    )

    assert "VIX 14.2" in ctx
    assert "Technology 44.0% (look-through)" in ctx
