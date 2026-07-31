from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent workflow."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    data_context: dict  # For passing raw data between split nodes (e.g. Data -> Reasoning)
    ghost: bool        # Privacy mode (No interest capture)
    summary: str       # A running summary of the current user plan/goal
    user_framework: str  # Sticky system-instruction block from the user, replayed every turn
    risk_retry_count: int  # Track compliance retries to prevent infinite loops
    risk_assessment: str   # Store the compliance verdict text
    market_analyst_handoff: bool  # True when MarketAnalyst's lens output needs a DeepReasoning judgment pass before the RiskManager gate
