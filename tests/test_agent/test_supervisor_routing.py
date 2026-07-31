"""Regression tests for the agent's routing backbone.

These lock in the deterministic routing contracts that the app's compliance
story depends on:
- advice-producing workers always pass through the RiskManager gate,
- health-check queries bypass it (both in supervisor and graph edge routing),
- the safety valve terminates runaway turns,
- @mention overrides win.

No LLM is involved: supervisor_node's routing is deterministic; only the
memory side-effects are stubbed out.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.constants import HEALTH_CHECK_KEYWORDS
from agent.graph import after_deep_reasoning, after_market_analyst
from agent.nodes.supervisor import supervisor_node


@pytest.fixture(autouse=True)
def _stub_memory_side_effects(monkeypatch):
    """supervisor_node writes to user memory on the first pass — keep tests pure."""
    monkeypatch.setattr("tools.memory.process_user_message", lambda content: None)


def _state(messages, **extra):
    state = {"messages": messages, "next": "", "data_context": {},
             "ghost": False, "summary": "", "user_framework": ""}
    state.update(extra)
    return state


def _worker_reply(name, content="analysis complete"):
    return AIMessage(content=f"[{name}] {content}", name=name)


# --- Default and override routing -------------------------------------------

def test_fresh_query_defaults_to_deep_reasoning():
    result = supervisor_node(_state([HumanMessage("Should I buy AAPL?")]))
    assert result["next"] == "DeepReasoning"


@pytest.mark.parametrize("mention,expected", [
    ("@DeepReasoning look at NVDA", "DeepReasoning"),
    ("@MarketAnalyst scan the market", "MarketAnalyst"),
    ("[MarketAnalyst lens=portfolio_audit] audit me", "MarketAnalyst"),
    ("@PortfolioManager review my holdings", "PortfolioManager"),
    ("@NewsAnalyst what happened today", "NewsAnalyst"),
])
def test_explicit_mention_overrides(mention, expected):
    result = supervisor_node(_state([HumanMessage(mention)]))
    assert result["next"] == expected


def test_no_human_message_finishes():
    result = supervisor_node(_state([_worker_reply("DeepReasoning")]))
    assert result["next"] == "FINISH"


# --- RiskManager compliance gate ---------------------------------------------

@pytest.mark.parametrize("worker", ["DeepReasoning", "MarketAnalyst", "PortfolioManager"])
def test_advice_worker_response_routes_to_risk_manager(worker):
    messages = [HumanMessage("Should I buy AAPL?"), _worker_reply(worker)]
    result = supervisor_node(_state(messages))
    assert result["next"] == "RiskManager"


def test_finish_after_risk_manager():
    messages = [
        HumanMessage("Should I buy AAPL?"),
        _worker_reply("DeepReasoning"),
        _worker_reply("RiskManager", "risk assessment done"),
    ]
    result = supervisor_node(_state(messages))
    assert result["next"] == "FINISH"


def test_news_analyst_skips_risk_manager():
    messages = [HumanMessage("What's in the news?"), _worker_reply("NewsAnalyst")]
    result = supervisor_node(_state(messages))
    assert result["next"] == "FINISH"


# --- Health-check bypass ------------------------------------------------------

@pytest.mark.parametrize("keyword", HEALTH_CHECK_KEYWORDS)
def test_health_check_bypasses_risk_manager_in_supervisor(keyword):
    messages = [HumanMessage(f"please run a {keyword} now"), _worker_reply("DeepReasoning")]
    result = supervisor_node(_state(messages))
    assert result["next"] == "FINISH"


@pytest.mark.parametrize("keyword", HEALTH_CHECK_KEYWORDS)
def test_health_check_bypasses_risk_manager_after_deep_reasoning(keyword):
    state = _state([HumanMessage(f"please run a {keyword} now"), _worker_reply("DeepReasoning")])
    assert after_deep_reasoning(state) == "Supervisor"


def test_advice_query_gets_risk_gate_after_deep_reasoning():
    state = _state([HumanMessage("Should I buy AAPL?"), _worker_reply("DeepReasoning")])
    assert after_deep_reasoning(state) == "RiskManager"


def test_scan_opportunities_diagnostics_key_does_not_bypass_risk_manager():
    """Regression: scan_opportunities always returns a dict with a literal
    "diagnostics" key (tools/opportunity_scanner.py). A blind substring scan
    of ToolMessage content for HEALTH_CHECK_KEYWORDS previously matched that
    key and skipped RiskManager on every scan turn — this must not happen in
    after_deep_reasoning or in supervisor_node."""
    tool_output = str({"sector": "All", "top_picks": [], "diagnostics": {"candidates": 86}})
    messages = [
        HumanMessage("[MarketAnalyst lens=external_screen] Find new external tickers"),
        ToolMessage(content=tool_output, tool_call_id="1", name="scan_opportunities"),
        _worker_reply("DeepReasoning"),
    ]
    assert after_deep_reasoning(_state(messages)) == "RiskManager"

    result = supervisor_node(_state(messages))
    assert result["next"] == "RiskManager"


# --- MarketAnalyst handoff ----------------------------------------------------

def test_market_analyst_handoff_to_deep_reasoning():
    state = _state([
        HumanMessage("scan then analyze"),
        AIMessage(content="[MarketAnalyst] found candidates [handoff: DeepReasoning]", name="MarketAnalyst"),
    ])
    assert after_market_analyst(state) == "DeepReasoning"


def test_market_analyst_default_goes_to_risk_manager():
    state = _state([
        HumanMessage("scan the market"),
        _worker_reply("MarketAnalyst"),
    ])
    assert after_market_analyst(state) == "RiskManager"


# --- Safety valve ---------------------------------------------------------------

def test_safety_valve_finishes_runaway_turn():
    messages = [HumanMessage("Should I buy AAPL?")]
    messages += [_worker_reply("DeepReasoning", f"pass {i}") for i in range(11)]
    result = supervisor_node(_state(messages))
    assert result["next"] == "FINISH"


# --- Privacy / ghost mode -------------------------------------------------------

def test_ghost_mode_skips_memory_capture(monkeypatch):
    captured = []
    monkeypatch.setattr("tools.memory.process_user_message", lambda content: captured.append(content))

    supervisor_node(_state([HumanMessage("secret question")], ghost=True))
    assert captured == []

    supervisor_node(_state([HumanMessage("normal question")]))
    assert captured == ["normal question"]
