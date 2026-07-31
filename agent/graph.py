from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from agent.constants import HEALTH_CHECK_KEYWORDS
from agent.nodes import (
    deep_reasoning_node,
    market_analyst_node,
    news_analyst_node,
    portfolio_manager_node,
    risk_manager_node,
    supervisor_node,
)
from agent.state import AgentState

# Global checkpointer for conversation memory with JSON serializer (supports AIMessage)
# memory = MemorySaver(serde=JsonPlusSerializer())


def after_market_analyst(state: AgentState):
    """MarketAnalyst exit routing: honor an explicit handoff, else the RiskManager gate.

    market_analyst_handoff (set by market_analyst_node) is the real signal — it's
    True for lenses whose output makes an implicit buy/sell/entry-timing call
    (external_screen, guru_validation, market_dip), so DeepReasoning gets a chance
    to reconcile it against portfolio/risk context before RiskManager ever sees it.
    The content-marker check is kept for backward compatibility with any caller
    that still emits "[handoff: DeepReasoning]" directly.
    """
    if state.get("market_analyst_handoff"):
        return "DeepReasoning"
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = str(getattr(last_msg, "content", ""))
        if "[handoff: DeepReasoning]" in content:
            return "DeepReasoning"
    return "RiskManager"


def after_deep_reasoning(state: AgentState):
    """DeepReasoning exit routing: health checks skip the RiskManager compliance gate."""
    messages = state.get("messages", [])

    # Scope the keyword scan to the current turn (the triggering human message
    # onward) so a health-check phrase from an earlier turn can't keep bypassing
    # the compliance gate on later, unrelated advice queries.
    last_human_idx = -1
    for i, msg in enumerate(messages):
        msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
        if msg_type == "human":
            last_human_idx = i

    if last_human_idx != -1:
        for msg in messages[last_human_idx:]:
            msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
            if msg_type == "tool":
                # Raw tool output is untrusted data, not user intent — e.g.
                # scan_opportunities always returns a dict with a literal
                # "diagnostics" key, which would otherwise false-positive this
                # check on every scan turn and skip RiskManager unconditionally.
                continue
            content = str(getattr(msg, "content", msg)).lower()
            if any(kw in content for kw in HEALTH_CHECK_KEYWORDS):
                return "Supervisor"
    return "RiskManager"


def after_risk_manager(state: AgentState):
    """
    RiskManager exit routing with one-retry compliance gate.

    Reads the risk_assessment and risk_retry_count from state (set by risk_manager_node).
    On first CRITICAL_FAIL, route back to DeepReasoning for one correction attempt.
    On second failure or any non-critical result, route to Supervisor.

    HARD CAP: risk_retry_count >= 1 always routes to Supervisor, preventing infinite loops.
    """
    risk_result = state.get("risk_assessment", "PASS")
    retry_count = state.get("risk_retry_count", 0)

    if risk_result == "CRITICAL_FAIL" and retry_count <= 1:
        # risk_manager_node already incremented the counter and injected the
        # correction message — we just need to route.
        return "DeepReasoning"

    # All other cases: route to Supervisor (PASS, FAIL, or exhausted retries)
    return "Supervisor"


def build_graph(use_memory: bool = True, use_async_memory: bool = False):
    """
    Builds the CairnIQ agent graph.

    Args:
        use_memory: If True, enables conversation memory for multi-turn interactions
        use_async_memory: If True, uses async checkpointer (required for API streaming)
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("PortfolioManager", portfolio_manager_node)
    workflow.add_node("NewsAnalyst", news_analyst_node)
    workflow.add_node("MarketAnalyst", market_analyst_node)
    workflow.add_node("DeepReasoning", deep_reasoning_node)
    workflow.add_node("RiskManager", risk_manager_node)

    # Edges
    # Start at Supervisor
    workflow.set_entry_point("Supervisor")

    # Supervisor decides where to go
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {
            "PortfolioManager": "PortfolioManager",
            "NewsAnalyst": "NewsAnalyst",
            "MarketAnalyst": "MarketAnalyst",
            "DeepReasoning": "DeepReasoning",
            "RiskManager": "RiskManager",
            "FINISH": END
        }

    )

    # Workers route through RiskManager for mandatory compliance review
    # PortfolioManager -> RiskManager -> Supervisor
    workflow.add_edge("PortfolioManager", "RiskManager")
    # NewsAnalyst -> Supervisor (no financial advice, skip risk check)
    workflow.add_edge("NewsAnalyst", "Supervisor")
    # Conditional Routing from MarketAnalyst
    workflow.add_conditional_edges(
        "MarketAnalyst",
        after_market_analyst,
        {
            "DeepReasoning": "DeepReasoning",
            "RiskManager": "RiskManager"
        }
    )
    # RiskManager routes conditionally: retry on first critical fail, else to Supervisor
    workflow.add_conditional_edges(
        "RiskManager",
        after_risk_manager,
        {
            "DeepReasoning": "DeepReasoning",
            "Supervisor": "Supervisor",
        }
    )

    # Conditional Routing from DeepReasoning
    workflow.add_conditional_edges(
        "DeepReasoning",
        after_deep_reasoning,
        {
            "Supervisor": "Supervisor",
            "RiskManager": "RiskManager"
        }
    )

    # Compile with optional checkpointer
    if use_memory:
        if use_async_memory:
             # Return UNCOMPILED workflow so the caller (API) can attach
             # an async checkpointer in an async context (lifespan).
             return workflow
        else:
             from agent.checkpointer import ProfileRoutingSaver

             # Resolves its SQLite file from the profile bound to each CALL, not
             # from whatever was active when the graph was built. server.py
             # builds this inside lifespan, before any request binds a profile —
             # a saver constructed against get_data_path() here would resolve to
             # the multi-user guard's _unbound sentinel and become the shared
             # conversation store for every profile (see agent/checkpointer.py).
             return workflow.compile(checkpointer=ProfileRoutingSaver())

    return workflow.compile()
