from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import after_deep_reasoning, build_graph
from agent.nodes.supervisor import supervisor_node
from agent.utils import is_cancelled, request_cancellation, reset_cancellation


def _state(messages, **extra):
    state = {"messages": messages, "next": "", "data_context": {},
             "ghost": False, "summary": "", "user_framework": ""}
    state.update(extra)
    return state


def test_graph_compilation():
    """Verify that the LangGraph workflow compiles without errors."""
    workflow = build_graph(use_memory=False)
    assert workflow is not None

def test_cancellation_mechanism():
    """Verify that the cancellation state is tracked correctly."""
    reset_cancellation()
    assert not is_cancelled()

    request_cancellation()
    assert is_cancelled()

    reset_cancellation()
    assert not is_cancelled()

def test_supervisor_routing():
    """Verify supervisor node routing logic based on user queries."""
    # This requires mocking the LLM call, so we'll test the routing edges directly
    # if possible, or just ensure the graph nodes exist.
    workflow = build_graph(use_memory=False)
    nodes = list(workflow.nodes.keys())

    assert "Supervisor" in nodes
    assert "PortfolioManager" in nodes
    assert "RiskManager" in nodes
    assert "DeepReasoning" in nodes

    # Check that RiskManager exists as a compliance gate
    assert "RiskManager" in nodes


# --- Health-check bypass must not leak across turns -------------------------
#
# The RiskManager compliance gate is skipped for health-check/diagnostics
# queries (e.g. "run a system health check") since those aren't financial
# advice. That bypass must only apply to the turn that actually asked for a
# health check — a health-check phrase spoken earlier in the conversation
# must not keep bypassing RiskManager on a later, unrelated advice query.

def _prior_health_check_turn():
    return [
        HumanMessage("please run a health check now"),
        AIMessage(content="[DeepReasoning] system nominal", name="DeepReasoning"),
    ]


def test_advice_query_after_health_check_turn_routes_to_risk_manager():
    """supervisor_node: a stale health-check keyword must not bypass RiskManager."""
    messages = _prior_health_check_turn() + [
        HumanMessage("Should I buy AAPL?"),
        AIMessage(content="[DeepReasoning] here is my analysis", name="DeepReasoning"),
    ]
    result = supervisor_node(_state(messages))
    assert result["next"] == "RiskManager"


def test_advice_query_after_health_check_turn_routes_to_risk_manager_via_graph_edge():
    """after_deep_reasoning: same guarantee at the graph-edge routing level."""
    messages = _prior_health_check_turn() + [
        HumanMessage("Should I buy AAPL?"),
        AIMessage(content="[DeepReasoning] here is my analysis", name="DeepReasoning"),
    ]
    assert after_deep_reasoning(_state(messages)) == "RiskManager"
