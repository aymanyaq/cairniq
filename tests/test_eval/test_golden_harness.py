"""Golden-set eval harness skeleton (Advisor Roadmap Theme 2.4).

Runs the corpus in deterministic mode inside the regular suite, so the risk
layer's front-half guards (grounding + IPS) are regression-checked on every
change — the whole point of the harness. Also asserts the harness itself is
falsifiable (it can FAIL) so a green run means something.
"""
import pytest

from agent.eval.golden_harness import (
    _BOOK,
    SCENARIOS,
    Scenario,
    run_all,
    run_scenario,
)


def test_every_deterministic_scenario_passes():
    """The corpus must be green in deterministic mode — this is the gate."""
    results = run_all()
    failures = [(r.scenario.name, r.reason) for r in results if r.status == "FAIL"]
    assert not failures, f"golden-set regressions: {failures}"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_status_is_expected(scenario):
    """Each scenario individually is PASS (deterministic) or SKIP (live-only)."""
    result = run_scenario(scenario)
    assert result.status in ("PASS", "SKIP"), f"{scenario.name}: {result.reason}"
    # Live-judge scenarios are the only ones allowed to SKIP.
    if result.status == "SKIP":
        assert scenario.expect_judge_flag is not None


def test_corpus_covers_the_documented_failure_modes():
    """Guard the skeleton's breadth so it can't silently shrink below its brief."""
    rules = " ".join(s.rule for s in SCENARIOS).lower()
    for needed in ("not-held", "total", "currency", "current price", "allocation", "ips", "judge"):
        assert needed in rules, f"corpus lost coverage of: {needed}"
    # At least ~10 scenarios (roadmap: "~10-scenario skeleton"), incl. mirror cases.
    assert len(SCENARIOS) >= 10
    assert sum(s.expect_clean for s in SCENARIOS) >= 3, "must guard against OVER-flagging too"


def test_live_seam_does_not_persist_verdicts(monkeypatch):
    """The judge seam must be side-effect-free: a live harness run must never
    write to the per-profile risk_verdicts.jsonl audit trail. Uses a stubbed
    judge_advice so no real model call is made — this guards the CONTRACT
    (run_scenario_live calls the seam, nothing persists), not the model.
    """
    import agent.eval.golden_harness as gh
    import agent.nodes.risk_manager as rm
    import tools.risk_verdict_log as rvl

    writes = []
    monkeypatch.setattr(rvl, "log_risk_verdict", lambda *a, **k: writes.append(a))
    monkeypatch.setattr(
        rm, "judge_advice",
        lambda draft, **kw: rm.JudgeOutcome(score=2, risk_result="CRITICAL_FAIL", is_compliant=False),
    )

    result = gh.run_scenario_live(SCENARIOS[0])
    assert result.status in ("PASS", "FAIL")  # adjudicated, not errored
    assert writes == [], "live harness run must not touch the verdict audit log"


def test_live_seam_reports_errors_not_raises(monkeypatch):
    """A judge call that raises is reported as ERROR, never propagated."""
    import agent.eval.golden_harness as gh
    import agent.nodes.risk_manager as rm

    def _boom(draft, **kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(rm, "judge_advice", _boom)
    result = gh.run_scenario_live(SCENARIOS[0])
    assert result.status == "ERROR"
    assert "provider down" in result.reason


def test_harness_is_falsifiable():
    """A deliberately-wrong expectation must FAIL — proves a green run is real."""
    # A clean draft that (wrongly) expects a flag → must FAIL.
    bogus_flag = Scenario(
        name="bogus_expects_flag", rule="meta",
        draft="Your portfolio is worth $250,000 USD.", portfolio=_BOOK,
        expect_flag="this substring will never appear",
    )
    assert run_scenario(bogus_flag).status == "FAIL"

    # A known-bad draft that (wrongly) expects clean → must FAIL.
    bogus_clean = Scenario(
        name="bogus_expects_clean", rule="meta",
        draft="I'd trim your GME position by half.", portfolio=_BOOK,
        expect_clean=True,
    )
    assert run_scenario(bogus_clean).status == "FAIL"
