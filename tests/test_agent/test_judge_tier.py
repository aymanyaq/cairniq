"""The compliance judge's model tier is a knob, and its default is deep.

The judge emits ~200 words, which looks like obvious fast-tier work. It is not
being moved there: this node has invented a profile rule that the advisor then
obeyed, and has called genuinely-fetched numbers fabricated under Rule 8. A
cheaper judge that hallucinates a violation costs a full compliance-retry cycle
and corrupts the advice — it does not save money.

What these tests pin is that the seam exists and that the DEFAULT did not move.
Flipping it is gated on a golden-harness `--live` A/B, not on this code.
"""
import pytest

import agent.nodes.risk_manager as rm


@pytest.fixture
def tiers(monkeypatch):
    """Report which tier was asked for, without building a real client."""
    monkeypatch.setattr(rm, "get_llm", lambda *a, **k: "DEEP")
    monkeypatch.setattr(rm, "get_sonnet_llm", lambda *a, **k: "FAST")
    monkeypatch.delenv("AIDLC_JUDGE_TIER", raising=False)


def test_the_default_is_the_deep_tier(tiers):
    assert rm.judge_llm() == "DEEP"


def test_unset_and_empty_both_mean_deep(tiers, monkeypatch):
    monkeypatch.setenv("AIDLC_JUDGE_TIER", "")
    assert rm.judge_llm() == "DEEP"
    monkeypatch.setenv("AIDLC_JUDGE_TIER", "   ")
    assert rm.judge_llm() == "DEEP"


@pytest.mark.parametrize("value", ["fast", "FAST", " Fast ", "sonnet"])
def test_the_knob_selects_the_fast_tier(tiers, monkeypatch, value):
    monkeypatch.setenv("AIDLC_JUDGE_TIER", value)
    assert rm.judge_llm() == "FAST"


def test_an_unrecognised_value_falls_back_to_deep(tiers, monkeypatch):
    """A typo must not silently downgrade the compliance gate."""
    monkeypatch.setenv("AIDLC_JUDGE_TIER", "cheep")
    assert rm.judge_llm() == "DEEP"


def _capture_llm(monkeypatch):
    """Record which model judge_advice builds its agent from, then stop."""
    captured = {}

    def fake_create_agent(llm, tools, system_prompt):
        captured["llm"] = llm
        raise RuntimeError("stop here — we only care which model was chosen")

    monkeypatch.setattr(rm, "create_agent", fake_create_agent)
    return captured


def test_judge_advice_honours_the_knob_when_no_llm_is_injected(tiers, monkeypatch):
    """So the eval harness can A/B by setting the env var alone."""
    captured = _capture_llm(monkeypatch)
    monkeypatch.setenv("AIDLC_JUDGE_TIER", "fast")

    with pytest.raises(RuntimeError):
        rm.judge_advice("Buy 100 shares of ABC.")
    assert captured["llm"] == "FAST"


def test_an_explicitly_injected_llm_still_wins(tiers, monkeypatch):
    """The pure seam is what the harness drives; the env must not override it."""
    captured = _capture_llm(monkeypatch)
    monkeypatch.setenv("AIDLC_JUDGE_TIER", "fast")

    with pytest.raises(RuntimeError):
        rm.judge_advice("Buy 100 shares of ABC.", llm="INJECTED")
    assert captured["llm"] == "INJECTED"
