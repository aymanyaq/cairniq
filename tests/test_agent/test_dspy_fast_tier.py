"""Mechanical DSPy work belongs on the fast slot.

DSPy has exactly one global LM and it is built from the primary slot, so every
DSPy call was billed at deep-tier rates — including ContextExtraction, which
pulls a couple of JSON fields out of one short message on every non-ghost turn.

`fast_dspy_context()` scopes a block to a second LM built from the fast slot. It
has to degrade to the global LM rather than failing, because losing the
extraction entirely is worse than paying deep-tier rates for it.
"""
import pytest

dspy_setup = pytest.importorskip("agent.dspy_setup")

if not dspy_setup.DSPY_AVAILABLE:
    pytest.skip("dspy not installed", allow_module_level=True)


@pytest.fixture(autouse=True)
def fresh_fast_lm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-real")
    dspy_setup.reset_fast_dspy_lm()
    yield
    dspy_setup.reset_fast_dspy_lm()


def test_no_separate_lm_when_both_slots_are_the_same_model(monkeypatch):
    monkeypatch.setenv("AIDLC_MODEL_ID", "same-model")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "same-model")
    # Building a second client for the same model buys nothing.
    assert dspy_setup.get_fast_dspy_lm() is None


def test_fast_lm_is_built_when_the_slots_differ(monkeypatch):
    monkeypatch.setenv("AIDLC_MODEL_ID", "deep-model")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "quick-model")
    lm = dspy_setup.get_fast_dspy_lm()
    assert lm is not None
    assert "quick-model" in lm.model


def test_fast_lm_is_built_once_and_reused(monkeypatch):
    monkeypatch.setenv("AIDLC_MODEL_ID", "deep-model")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "quick-model")
    assert dspy_setup.get_fast_dspy_lm() is dspy_setup.get_fast_dspy_lm()


def test_a_failing_build_is_not_retried_every_turn(monkeypatch):
    monkeypatch.setenv("AIDLC_MODEL_ID", "deep-model")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "quick-model")

    calls = {"n": 0}

    def _boom(model_id, region):
        calls["n"] += 1
        raise RuntimeError("no credentials")

    monkeypatch.setattr(dspy_setup, "_build_litellm_lm", _boom)

    assert dspy_setup.get_fast_dspy_lm() is None
    assert dspy_setup.get_fast_dspy_lm() is None
    # One attempt per process, not one per turn.
    assert calls["n"] == 1


def test_context_swaps_the_active_lm_and_restores_it(monkeypatch):
    monkeypatch.setenv("AIDLC_MODEL_ID", "deep-model")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "quick-model")

    dspy = dspy_setup.dspy
    fast = dspy_setup.get_fast_dspy_lm()
    before = getattr(dspy.settings, "lm", None)

    with dspy_setup.fast_dspy_context():
        assert dspy.settings.lm is fast

    assert getattr(dspy.settings, "lm", None) is before


def test_context_is_a_passthrough_when_no_fast_model_exists(monkeypatch):
    monkeypatch.setenv("AIDLC_MODEL_ID", "same-model")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "same-model")

    dspy = dspy_setup.dspy
    before = getattr(dspy.settings, "lm", None)
    # Must not raise, and must not disturb the global LM: extraction still runs,
    # just on the primary slot.
    with dspy_setup.fast_dspy_context():
        assert getattr(dspy.settings, "lm", None) is before
