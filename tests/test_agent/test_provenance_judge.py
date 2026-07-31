"""Degraded-evidence handling in the judge (Advisor Roadmap 2.3).

The danger in wiring provenance to the judge is not that it under-reacts — it is
that it over-reacts. This codebase has already paid twice for a judge that
invented a fault (a fabricated profile rule the advisor then obeyed; a false
SOURCE FRAUD on a stale quote), and a missing API key is the easiest possible
trigger for a third: it is present on most turns, it is nobody's misconduct, and
routing it through the existing grounding-violation path would turn every
unconfigured data source into a CRITICAL_FAIL.

So the load-bearing assertions here are about the CEILING, not the floor: a
degraded turn loses its clean pass and keeps its answer.
"""
from dataclasses import replace

import pytest

import agent.nodes.risk_manager as rm
import tools.provenance as prov

_UNAVAILABLE = (
    "### Tool Call: get_insider_trades({})\nResult:\n"
    "{'status': 'unavailable', 'source': 'FMP', 'reason': 'FMP_API_KEY not configured'}"
)
_CLEAN = "### Tool Call: get_quote({})\nResult:\n{'price': 101.5}"


# ---------------------------------------------------------------------------
# The block the judge is shown
# ---------------------------------------------------------------------------

def test_a_clean_turn_adds_no_provenance_block():
    """Its presence is the signal, so a clean turn must spend no tokens on it."""
    dq = prov.summarize_tool_context(_CLEAN)

    assert rm._build_provenance_block(dq) == ""


def test_a_degraded_turn_states_the_fact_and_the_duty_that_follows():
    dq = prov.summarize_tool_context(_UNAVAILABLE)
    block = rm._build_provenance_block(dq)

    assert "FMP UNAVAILABLE" in block
    assert "FMP_API_KEY not configured" in block
    assert "MUST say" in block


def test_the_block_forbids_manufacturing_a_fault():
    """The instruction that keeps this from becoming the third invented-fault
    incident: advice that never touched the degraded source is not at fault."""
    block = rm._build_provenance_block(prov.summarize_tool_context(_UNAVAILABLE))

    assert "do not manufacture one" in block
    assert "not rely on the degraded source" in block


def test_the_block_is_never_shown_to_the_user():
    block = rm._build_provenance_block(prov.summarize_tool_context(_UNAVAILABLE))

    assert "do not mention this block" in block


# ---------------------------------------------------------------------------
# The cap — a ceiling, never a CRITICAL_FAIL
# ---------------------------------------------------------------------------

@pytest.fixture
def judge(monkeypatch):
    """Drive the REAL judge_advice seam with the LLM boundary stubbed.

    Re-deriving the cap arithmetic in the test would let production drift away
    from it silently, which is the one thing these assertions exist to catch.
    """
    def _run(verdict_text, tool_ctx, grounding=None):
        for audit in ("run_deterministic_grounding_audit", "run_deterministic_total_audit",
                      "run_deterministic_price_audit", "run_deterministic_allocation_audit"):
            monkeypatch.setattr(rm, audit, lambda _t, _g=grounding, _a=audit: (
                list(_g) if _g and _a == "run_deterministic_grounding_audit" else []
            ))
        monkeypatch.setattr(
            "tools.ips_precheck.run_ips_precheck",
            lambda text, tickers: {"trades": [], "rows": [], "violations": [], "block": ""},
        )
        monkeypatch.setattr(rm, "get_llm", lambda: object())
        monkeypatch.setattr(rm, "create_agent", lambda llm, tools, system: object())
        monkeypatch.setattr(
            rm, "safe_invoke",
            lambda *a, **k: type("R", (), {"content": verdict_text})(),
        )
        monkeypatch.setattr(rm, "has_stream_callback", lambda: False)
        return rm.judge_advice(
            "Buy 10 shares of VTI at the open.",
            judge_messages=[],
            tool_execution_ctx=tool_ctx,
        )

    return _run


# The judge's mandated template — parse_risk_verdict fails CLOSED to 5 on
# anything else, so these must be the real shape or the test measures nothing.
_VERDICT_10 = "Verdict: 10/10"
_VERDICT_3 = "Verdict: 3/10\nRisks:\n- Position sizing is reckless."


def test_a_missing_api_key_never_produces_a_critical_fail(judge):
    """The whole point. An unconfigured source must not stop the advisor
    answering a question the user still needs answered."""
    outcome = judge(_VERDICT_10, _UNAVAILABLE)

    assert outcome.risk_result != "CRITICAL_FAIL"
    assert outcome.score == rm.PROVENANCE_SCORE_CAP


def test_a_degraded_turn_cannot_score_a_clean_pass(judge):
    outcome = judge(_VERDICT_10, _UNAVAILABLE)

    assert outcome.score < 8, "a turn built on unavailable evidence scored as fully evidenced"
    assert outcome.data_quality["degraded"] is True


def test_the_provenance_cap_is_weaker_than_the_grounding_cap():
    """Two separate tracks: fabricated numbers are misconduct, a dead API is
    not. If these ever converge, a missing key starts blocking answers."""
    assert rm.PROVENANCE_SCORE_CAP > 6


def test_a_clean_turn_is_left_alone(judge):
    outcome = judge(_VERDICT_10, _CLEAN)

    assert outcome.score == 10
    assert outcome.risk_result == "PASS"
    assert outcome.data_quality["degraded"] is False


def test_a_grounding_violation_still_dominates_a_degraded_turn(judge):
    outcome = judge(_VERDICT_10, _UNAVAILABLE, grounding=["fabricated portfolio total"])

    assert outcome.risk_result == "CRITICAL_FAIL"


def test_the_cap_never_raises_a_low_score(judge):
    """A cap is a ceiling. A genuinely bad draft must not be lifted to 7 just
    because its data was also degraded."""
    outcome = judge(_VERDICT_3, _UNAVAILABLE)

    assert outcome.score == 3
    assert outcome.risk_result == "CRITICAL_FAIL"


# ---------------------------------------------------------------------------
# Shape carried out of the judge
# ---------------------------------------------------------------------------

def test_the_outcome_carries_the_provenance_for_the_response_layer():
    outcome = rm.JudgeOutcome(score=9, risk_result="PASS", is_compliant=True)

    assert outcome.data_quality == {}
    assert replace(outcome, data_quality={"degraded": True}).data_quality["degraded"] is True


# ---------------------------------------------------------------------------
# The user-visible footer
# ---------------------------------------------------------------------------

def test_a_clean_turn_gets_no_footer():
    """A footer under every answer is read for a week and then never again —
    and the one time it says something load-bearing it is wallpaper."""
    assert rm._provenance_footer_line(prov.summarize_tool_context(_CLEAN)) == ""


def test_a_degraded_turn_names_the_source_in_one_line():
    line = rm._provenance_footer_line(prov.summarize_tool_context(_UNAVAILABLE))

    assert "FMP unavailable" in line
    assert line.count("\n---\n") == 1
    assert len(line.strip().splitlines()[-1]) < 120


def test_the_footer_never_reaches_the_persisted_verdict(judge):
    """The 2.1 audit trail records the judge's own words. A footer folded into
    verdict_text would later read as something the judge said."""
    outcome = judge(_VERDICT_10, _UNAVAILABLE)

    assert "Data provenance" not in outcome.verdict_text


def test_an_empty_provenance_summary_is_handled():
    assert rm._provenance_footer_line({}) == ""
    assert rm._provenance_footer_line(None) == ""
