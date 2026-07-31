"""Old turns' tool dumps are pruned; the live turn's are not.

The analysts append their whole ReAct loop to graph state — planner AIMessage,
every ToolMessage (a full scan table, a set of news bodies), final answer — and
the checkpointer persists it, so by turn 5 the planner was re-sending 40-100k
tokens of previous turns' tool output on every call.

The dangerous version of this fix is returning only the final message, the way
DeepReasoning does. DeepReasoning can do that because it republishes its tool
results through data_context; the analysts have no such route, so their in-state
ToolMessages ARE the RiskManager's grounding evidence. Removing them would make
the judge call genuinely-fetched numbers fabricated under Rule 8.

So the tests that matter here are the ones about what is KEPT.
"""
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from agent.history import prune_completed_turns


def _turn(n: int, with_tools: bool = True) -> list:
    """One complete turn: question, planner + tool results, answer."""
    msgs: list = [HumanMessage(content=f"question {n}", id=f"h{n}")]
    if with_tools:
        msgs += [
            AIMessage(
                content="",
                id=f"plan{n}",
                tool_calls=[{"name": "scan_opportunities", "args": {}, "id": f"tc{n}"}],
            ),
            ToolMessage(content="A" * 5000, tool_call_id=f"tc{n}", name="scan_opportunities", id=f"tm{n}"),
        ]
    msgs.append(AIMessage(content=f"answer {n}", id=f"a{n}", name="MarketAnalyst"))
    return msgs


def _removed_ids(messages) -> set[str]:
    out = prune_completed_turns(messages)
    assert all(isinstance(r, RemoveMessage) for r in out)
    return {r.id for r in out}


def test_the_live_turns_tool_messages_are_never_pruned():
    """The judge audits THIS turn; its evidence has to still be there."""
    messages = _turn(1) + _turn(2)
    removed = _removed_ids(messages)
    assert "tm2" not in removed, "pruned the current turn's grounding evidence"
    assert "plan2" not in removed
    assert "tm1" in removed


def test_a_single_turn_conversation_prunes_nothing():
    assert _removed_ids(_turn(1)) == set()


def test_the_planner_and_its_tool_results_are_removed_together():
    """Half a pair is worse than neither.

    An AIMessage with tool_calls whose ToolMessages are gone is an unresolved
    tool_use block; providers reject the whole request (Bedrock raises
    ValidationException), so every later turn would fail.
    """
    messages = _turn(1) + _turn(2) + _turn(3)
    removed = _removed_ids(messages)
    for n in (1, 2):
        assert f"plan{n}" in removed and f"tm{n}" in removed, f"turn {n} pruned as half a pair"


def test_questions_and_answers_survive():
    """What is left of a completed turn is the conversation itself."""
    messages = _turn(1) + _turn(2)
    removed = _removed_ids(messages)
    for kept in ("h1", "a1", "h2", "a2"):
        assert kept not in removed


def test_a_compliance_retry_directive_does_not_start_a_new_turn():
    """The retry gate injects a synthetic HumanMessage.

    If it were treated as the turn boundary, the real turn's tool evidence would
    fall into 'completed' and be pruned out from under the judge mid-retry —
    exactly when it is re-auditing.
    """
    messages = _turn(1) + _turn(2) + [
        HumanMessage(content="<compliance_correction_required> fix it", id="corr"),
        AIMessage(content="revised answer", id="rev"),
    ]
    removed = _removed_ids(messages)
    assert "tm2" not in removed, "retry pass pruned the turn currently under audit"
    assert "tm1" in removed


def test_messages_without_ids_are_skipped():
    """add_messages matches removals by id; there is nothing to match without one."""
    messages = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "x"}]),
        ToolMessage(content="payload", tool_call_id="x", name="t"),
        AIMessage(content="a1"),
        HumanMessage(content="q2", id="h2"),
    ]
    for m in messages[:4]:
        m.id = None
    assert prune_completed_turns(messages) == []


def test_pruning_is_idempotent_across_consecutive_nodes():
    """Two analysts can run in one graph pass.

    The second sees state the first already pruned, so it must not re-emit
    removals for ids that are gone — add_messages raises on an unknown id.
    """
    messages = _turn(1) + _turn(2)
    first = _removed_ids(messages)
    assert first

    surviving = [m for m in messages if getattr(m, "id", None) not in first]
    assert _removed_ids(surviving) == set()
