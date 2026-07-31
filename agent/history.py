"""Bounding conversation-history growth in the graph state.

MarketAnalyst and NewsAnalyst append their whole ReAct loop to `messages` — the
planner AIMessage, every ToolMessage, and the final answer. A ToolMessage here is
a full scan table or a set of news bodies, and `ProfileRoutingSaver` persists all
of it, so the growth survives restarts: by turn 5 the planner was re-sending
40-100k tokens of previous turns' tool dumps on every call.

The obvious fix — have the analysts return only their final message, the way
DeepReasoning does at the end of `deep_reasoning_node` — is WRONG here, and
expensively so. DeepReasoning can drop its ToolMessages because it republishes
them through `data_context["tool_execution_context"]`
(`_publish_tool_evidence`), which is the only route its evidence has to the
RiskManager. The analysts have no such route: for their turns, the in-state
ToolMessages ARE the judge's grounding evidence
(`risk_manager._build_tool_execution_context`), and evidence the judge cannot see
it is free to call fabricated under Rule 8. That failure is documented, real, and
has cost genuinely-fetched numbers a 2/10 SOURCE FRAUD verdict before.

So the split is by TURN, not by node. The judge only ever looks at the current
turn — it scans forward from the last genuine user message — and DeepReasoning's
two ToolMessage checks are both scoped to `messages[last_human_idx + 1:]`. Once a
turn is over, nothing reads its tool traffic again; it is pure re-sent weight.
This prunes completed turns and leaves the live one untouched.
"""
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

# The retry gate injects this as a HumanMessage. It is not user-authored, so it
# must not be mistaken for the start of a new turn — the judge applies the same
# rule when it locates the turn boundary.
_CORRECTION_PREFIX = "<compliance_correction_required>"


def _current_turn_start(messages) -> int:
    """Index of the last genuine user message, i.e. where the live turn begins."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and not str(
            getattr(m, "content", "")
        ).lstrip().startswith(_CORRECTION_PREFIX):
            return i
    return 0


def prune_completed_turns(messages) -> list[RemoveMessage]:
    """RemoveMessage entries dropping tool traffic from turns that are OVER.

    Returns the removals to merge into a node's `messages` update; `add_messages`
    applies them by id. Never touches the current turn, so the RiskManager's
    grounding evidence is intact when it audits this turn's advice.

    A planner AIMessage and its ToolMessages are removed TOGETHER. Dropping the
    results while keeping the calls would leave unresolved tool_use blocks, which
    providers reject outright (Bedrock raises ValidationException) — the history
    would be smaller and every subsequent turn would fail. What survives a
    completed turn is what a conversation actually is: the question and the
    answer.
    """
    boundary = _current_turn_start(messages)
    if boundary <= 0:
        return []

    removals: list[RemoveMessage] = []
    for msg in messages[:boundary]:
        msg_id = getattr(msg, "id", None)
        if not msg_id:
            # Without an id there is nothing for add_messages to match on.
            continue
        if isinstance(msg, ToolMessage) or (
            isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
        ):
            removals.append(RemoveMessage(id=msg_id))
    return removals
