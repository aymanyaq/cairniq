"""risk_manager_node persists every verdict to risk_verdicts.jsonl (Theme 2.1).

Runs the node with the LLM stack mocked and asserts the audit-trail record
lands with the right score / risk_result / retry outcome for each gate path:
clean pass, critical fail → retry, critical fail → retries exhausted, the
DeepReasoning bypass, and ghost-mode query scrubbing.
"""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agent.nodes.risk_manager as rm
import tools.risk_verdict_log as rvl

ADVICE = AIMessage(
    content="[DeepReasoning]: Accumulate CDE between $14.50 and $15.50 with a staged "
    "buy trigger, sized against your 2% risk budget. " + "x" * 60,
    name="DeepReasoning",
)

CRITICAL_VERDICT = (
    "⚖️ **Verdict: [2/10]** — Plan drains the mandatory cash buffer.\n\n"
    "🔴 **Risks:**\n"
    "- Profile Violation: deploys the entire 2% cash buffer.\n\n"
    "🤔 **Devil's Advocate:** Mining names are rate-sensitive."
)


@pytest.fixture
def verdict_file(monkeypatch, tmp_path):
    path = tmp_path / "risk_verdicts.jsonl"
    monkeypatch.setattr(rvl, "get_data_path", lambda filename: str(path))
    return path


def _records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().strip().split("\n") if line.strip()]


def _mock_node_internals(monkeypatch, judge_reply: str):
    monkeypatch.setattr(rm, "get_llm", lambda: MagicMock())
    monkeypatch.setattr(rm, "create_agent", lambda *a, **k: MagicMock())
    monkeypatch.setattr(rm, "has_stream_callback", lambda: False)
    monkeypatch.setattr(rm, "send_status", lambda *a, **k: None)
    monkeypatch.setattr(rm, "safe_invoke", lambda agent, payload: AIMessage(content=judge_reply))
    monkeypatch.setattr(rm, "get_user_context_string", lambda: "test user context")
    monkeypatch.setattr(rm, "_build_portfolio_verification_brief", lambda: "no holdings")
    for audit in (
        "run_deterministic_grounding_audit",
        "run_deterministic_total_audit",
        "run_deterministic_price_audit",
        "run_deterministic_allocation_audit",
    ):
        monkeypatch.setattr(rm, audit, lambda text: [])


def _state(messages, **overrides):
    state = {
        "messages": messages,
        "data_context": {},
        "summary": "",
        "user_framework": "",
        "ghost": False,
        "risk_retry_count": 0,
    }
    state.update(overrides)
    return state


def test_clean_pass_logs_accepted_verdict(monkeypatch, verdict_file):
    _mock_node_internals(monkeypatch, "✅ Risk Check Passed")

    result = rm.risk_manager_node(_state([HumanMessage(content="scan my watchlist"), ADVICE]))

    assert result["risk_assessment"] == "PASS"
    records = _records(verdict_file)
    assert len(records) == 1
    rec = records[0]
    assert rec["event"] == "verdict"
    assert rec["score"] == 10
    assert rec["risk_result"] == "PASS"
    assert rec["is_compliant"] is True
    assert rec["retry_outcome"] == "accepted"
    assert rec["advice_node"] == "DeepReasoning"
    assert rec["query"] == "scan my watchlist"
    assert rec["grounding_violations"] == [] and rec["llm_violations"] == []


def test_critical_fail_first_pass_logs_retry_scheduled(monkeypatch, verdict_file):
    _mock_node_internals(monkeypatch, CRITICAL_VERDICT)

    result = rm.risk_manager_node(_state([HumanMessage(content="build a dip plan"), ADVICE]))

    assert result["risk_assessment"] == "CRITICAL_FAIL"
    rec = _records(verdict_file)[0]
    assert rec["score"] == 2
    assert rec["risk_result"] == "CRITICAL_FAIL"
    assert rec["retry_count"] == 0
    assert rec["retry_outcome"] == "retry_scheduled"
    assert any("cash buffer" in v for v in rec["llm_violations"])
    assert "Verdict: [2/10]" in rec["verdict_text"]


def test_critical_fail_after_retry_logs_retries_exhausted(monkeypatch, verdict_file):
    _mock_node_internals(monkeypatch, CRITICAL_VERDICT)

    state = _state(
        [HumanMessage(content="build a dip plan"), ADVICE],
        risk_retry_count=1,
    )
    result = rm.risk_manager_node(state)

    assert result["risk_assessment"] == "CRITICAL_FAIL"
    rec = _records(verdict_file)[0]
    assert rec["retry_count"] == 1
    assert rec["retry_outcome"] == "retries_exhausted"


def test_grounding_violation_capped_and_logged(monkeypatch, verdict_file):
    _mock_node_internals(monkeypatch, "⚖️ **Verdict: [9/10]** — Looks fine.\n\n🔴 **Risks:** None flagged")
    monkeypatch.setattr(
        rm, "run_deterministic_grounding_audit",
        lambda text: ["Grounding Error: Advice references MU which is not held."],
    )

    result = rm.risk_manager_node(_state([HumanMessage(content="should I add MU?"), ADVICE]))

    assert result["risk_assessment"] == "CRITICAL_FAIL"
    rec = _records(verdict_file)[0]
    assert rec["score"] == 6  # grounding errors cap the score
    assert rec["grounding_violations"] and "MU" in rec["grounding_violations"][0]


def test_bypass_logs_bypassed_marker(monkeypatch, verdict_file):
    _mock_node_internals(monkeypatch, "unused")

    embedded = AIMessage(
        content="[RiskManager]: \n\n---\n### 🛡️ Risk Assessment\n⚖️ **Verdict: [10/10]** — clean",
        name="RiskManager",
    )
    result = rm.risk_manager_node(_state([HumanMessage(content="deep dive NVDA"), embedded]))

    assert result["risk_assessment"] == "PASS"
    rec = _records(verdict_file)[0]
    assert rec["event"] == "bypassed"
    assert rec["risk_result"] == "PASS"
    assert "score" not in rec


def test_ghost_mode_scrubs_query_but_still_logs(monkeypatch, verdict_file):
    _mock_node_internals(monkeypatch, "✅ Risk Check Passed")

    result = rm.risk_manager_node(
        _state([HumanMessage(content="@Ghost very private question"), ADVICE], ghost=True)
    )

    assert result["risk_assessment"] == "PASS"
    rec = _records(verdict_file)[0]
    assert rec["ghost"] is True
    assert rec["query"] == ""
    assert rec["score"] == 10


def test_retry_pass_logs_genuine_query_not_correction_directive(monkeypatch, verdict_file):
    _mock_node_internals(monkeypatch, "✅ Risk Check Passed")

    messages = [
        HumanMessage(content="build a dip plan"),
        ADVICE,
        HumanMessage(content="<compliance_correction_required>\nFix the sizing.\n</compliance_correction_required>"),
        AIMessage(content="[DeepReasoning]: revised, compliant advice " + "y" * 80, name="DeepReasoning"),
    ]
    rm.risk_manager_node(_state(messages, risk_retry_count=1))

    rec = _records(verdict_file)[0]
    assert rec["query"] == "build a dip plan"
