"""Roadmap 6.3 — ThesisEvaluation is grounded in evidence.

Two things are under test:

1. The judge actually RECEIVES the evidence at its production call site
   (MarketAnalyst's ToT upgrade) and emits a ``strongest_objection``. These tests
   drive the real node function with only the LM boundary faked — the fake
   predictor is built FROM the real signature class and asserts the exact
   input-field set, so a call site that forgets ``evidence_context`` fails here
   rather than in production. (DeepReasoning had a second, unreachable ToT branch;
   see the note at the foot of this file for why it and its tests are gone.)

2. The objection cannot be fabricated. An evidence block with no substance forces the
   explicit insufficiency sentinel, and figures that are not in the evidence are
   flagged. This is the 2026-07-21 history-fabrication shape: a truthiness-gated block
   emits nothing, and the model back-fills the silence with real-sounding specifics.

No live model calls: ``dspy.ChainOfThought`` is replaced for the duration of each test.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from agent.dspy_setup import dspy
from agent.modules import (
    INSUFFICIENT_OBJECTION,
    NO_EVIDENCE_MARKER,
    UNVERIFIED_FIGURE_NOTICE,
    build_evidence_context,
    evidence_is_empty,
    ground_strongest_objection,
    ungrounded_figures,
)
from agent.signatures import ThesisEvaluation
from agent.state import AgentState

# A fluent, entirely invented objection — the exact output shape this feature must
# refuse to print when there is nothing behind it.
FABRICATED_OBJECTION = (
    "The bull case ignores that management guided FY26 operating margin down to 18.2% "
    "on the Q3 call, and the $4.1B buyback was quietly suspended in March."
)


# --------------------------------------------------------------------------- #
# LM boundary fake
# --------------------------------------------------------------------------- #

class PredictorRegistry:
    """Records every predictor call and supplies per-signature output overrides."""

    def __init__(self, overrides=None):
        self.calls = []
        self.overrides = overrides or {}

    def kwargs_for(self, signature_name):
        return [kw for name, kw in self.calls if name == signature_name]

    def only_call(self, signature_name):
        calls = self.kwargs_for(signature_name)
        assert len(calls) == 1, f"expected exactly 1 {signature_name} call, got {len(calls)}"
        return calls[0]


class FakePredictor:
    """Stands in for ``dspy.ChainOfThought`` — enforces the REAL signature contract."""

    def __init__(self, signature, registry):
        self.signature = signature
        self.registry = registry
        self.name = getattr(signature, "__name__", str(signature))

    def __call__(self, **kwargs):
        expected = set(self.signature.input_fields)
        missing = expected - set(kwargs)
        extra = set(kwargs) - expected
        assert not missing, f"{self.name} invoked without required input(s) {sorted(missing)}"
        assert not extra, f"{self.name} invoked with unknown input(s) {sorted(extra)}"
        self.registry.calls.append((self.name, dict(kwargs)))
        outputs = {field: f"<{field}>" for field in self.signature.output_fields}
        outputs.update(self.registry.overrides.get(self.name, {}))
        return dspy.Prediction(**outputs)


def install_fake_lm(monkeypatch, overrides=None):
    registry = PredictorRegistry(overrides)
    monkeypatch.setattr(
        dspy, "ChainOfThought", lambda signature, *a, **k: FakePredictor(signature, registry)
    )
    return registry


# --------------------------------------------------------------------------- #
# Signature contract
# --------------------------------------------------------------------------- #

def test_thesis_evaluation_signature_takes_evidence_and_returns_an_objection():
    assert "evidence_context" in ThesisEvaluation.input_fields
    assert "strongest_objection" in ThesisEvaluation.output_fields
    desc = ThesisEvaluation.output_fields["strongest_objection"].json_schema_extra["desc"]
    # The instruction must name the escape hatch, not just forbid invention.
    assert "INSUFFICIENT EVIDENCE" in desc


# --------------------------------------------------------------------------- #
# Evidence rendering + emptiness
# --------------------------------------------------------------------------- #

def test_build_evidence_context_states_absence_out_loud():
    ctx = build_evidence_context("AAPL", fundamentals=["P/E 28.4 | mkt cap 3.1T"], news=[])
    assert "Symbol: AAPL" in ctx
    assert "P/E 28.4 | mkt cap 3.1T" in ctx
    # Missing sections are named and marked, never silently dropped.
    for label in ("FUNDAMENTALS", "TECHNICALS", "NEWS & SENTIMENT", "MACRO"):
        assert f"{label}:" in ctx
    assert ctx.count(NO_EVIDENCE_MARKER) == 3
    # A plain string section is accepted as well as a list.
    assert "RSI 71" in build_evidence_context("AAPL", technicals="RSI 71")


def test_evidence_is_empty_sees_through_scaffolding():
    empty = build_evidence_context("AAPL")
    assert evidence_is_empty(empty)
    # Length is not substance: a directive riding along in the block is not evidence.
    assert evidence_is_empty(
        empty + "\nCRITICAL INSTRUCTION: Generate an EXTREMELY CONCISE, bullet-point only response."
    )
    assert evidence_is_empty("")
    assert evidence_is_empty(None)
    # One real datum flips it.
    assert not evidence_is_empty(build_evidence_context("AAPL", news=["Q3 revenue beat"]))


# --------------------------------------------------------------------------- #
# Objection grounding
# --------------------------------------------------------------------------- #

def test_empty_evidence_forces_the_sentinel_over_a_fluent_invention():
    empty = build_evidence_context("AAPL")
    grounded = ground_strongest_objection(FABRICATED_OBJECTION, empty)
    assert grounded == INSUFFICIENT_OBJECTION
    assert "18.2%" not in grounded
    assert "buyback" not in grounded


@pytest.mark.parametrize("blank", ["", "   ", None, "N/A", "None.", "unknown"])
def test_blank_or_placeholder_objection_becomes_the_sentinel(blank):
    ctx = build_evidence_context("AAPL", fundamentals=["P/E 28.4"])
    assert ground_strongest_objection(blank, ctx) == INSUFFICIENT_OBJECTION


def test_ungrounded_figures_are_flagged_and_grounded_ones_are_not():
    ctx = build_evidence_context(
        "AAPL",
        fundamentals=["P/E 28.4 | gross margin 44.1%"],
        news=["Guidance cut announced"],
    )
    objection = (
        "Gross margin of 44.1% is already peak-cycle, and the $4.1B buyback was suspended."
    )
    missing = ungrounded_figures(objection, ctx)
    assert any("4.1B" in m for m in missing)
    assert not any(m.strip() == "44.1%" for m in missing)

    grounded = ground_strongest_objection(objection, ctx)
    assert UNVERIFIED_FIGURE_NOTICE in grounded
    assert "$4.1B" in grounded.split(UNVERIFIED_FIGURE_NOTICE)[1]
    # The prose itself survives — the flag annotates, it does not censor.
    assert "peak-cycle" in grounded


def test_fully_grounded_objection_passes_through_unannotated():
    ctx = build_evidence_context("AAPL", fundamentals=["P/E 28.4", "gross margin 44.1%"])
    objection = "At a P/E of 28.4 the multiple already prices in the margin story."
    grounded = ground_strongest_objection(objection, ctx)
    assert grounded == objection
    assert UNVERIFIED_FIGURE_NOTICE not in grounded


def test_plain_counts_are_not_mistaken_for_fabricated_metrics():
    """False grounding errors have cost this codebase real retries. 'the next 2
    quarters' is not an invented financial metric."""
    ctx = build_evidence_context("AAPL", fundamentals=["P/E 28.4"])
    objection = "Over the next 2 quarters, 3 of the catalysts named in the bull case land after the guide."
    assert ungrounded_figures(objection, ctx) == []
    assert ground_strongest_objection(objection, ctx) == objection


def test_pathological_numbers_do_not_crash_the_gate():
    """Evidence is raw tool output: ids, hashes and digit runs long enough to
    overflow a float all show up. The gate must survive them, not raise."""
    ctx = build_evidence_context("AAPL", fundamentals=["trace_id " + "9" * 400, "P/E 28.4"])
    objection = "The $" + "9" * 400 + "B claim has no source, unlike the 28.4 multiple."
    assert ground_strongest_objection(objection, ctx) == objection


def test_grounding_is_idempotent():
    ctx = build_evidence_context("AAPL", fundamentals=["P/E 28.4"])
    once = ground_strongest_objection("The $4.1B buyback was suspended.", ctx)
    assert ground_strongest_objection(once, ctx) == once
    assert once.count(UNVERIFIED_FIGURE_NOTICE) == 1


# --------------------------------------------------------------------------- #
# Real path 1 — MarketAnalyst's Tree-of-Thought upgrade
# --------------------------------------------------------------------------- #

class DummyTool:
    def __init__(self, name, return_value):
        self.name = name
        self.return_value = return_value

    def invoke(self, args):
        return self.return_value


def _run_market_analyst(monkeypatch, registry, fundamentals_payload):
    state = AgentState(messages=[HumanMessage(content="[MarketAnalyst]: analyze AAPL")])

    monkeypatch.setattr("agent.nodes.market_analyst.get_sonnet_llm", lambda *a, **k: MagicMock())
    monkeypatch.setattr("agent.nodes.market_analyst.get_user_context", lambda: "Risk Tolerance: Aggressive")
    monkeypatch.setattr("agent.nodes.market_analyst.has_stream_callback", lambda: False)
    monkeypatch.setattr(
        "agent.tool_retriever.get_semantic_tools_with_metadata",
        lambda query, k: ([], {"tool_count": 0, "selected_tool_names": []}),
    )

    planner = MagicMock()
    planner.tool_calls = [{"name": "fetch_fundamentals", "args": {"symbol": "AAPL"}, "id": "c1"}]
    planner.content = "Planning fundamentals fetch."
    synthesis = MagicMock()
    synthesis.content = "Synthesis narrative."

    calls = {"n": 0}

    def fake_safe_invoke(agent_or_llm, msgs):
        calls["n"] += 1
        return planner if calls["n"] == 1 else synthesis

    monkeypatch.setattr("agent.nodes.market_analyst.safe_invoke", fake_safe_invoke)

    class MockToolRetriever:
        @property
        def tool_map(self):
            return {"fetch_fundamentals": DummyTool("fetch_fundamentals", fundamentals_payload)}

    monkeypatch.setattr("agent.tool_retriever.ToolRetriever", MockToolRetriever)

    from agent.nodes.market_analyst import market_analyst_node

    return market_analyst_node(state)


def test_market_analyst_judge_gets_the_evidence_and_renders_the_objection(monkeypatch):
    registry = install_fake_lm(
        monkeypatch,
        overrides={
            "ThesisEvaluation": {
                "selected_thesis": "BULL",
                "confidence_level": "MEDIUM",
                "strongest_objection": "A P/E of 28.4 leaves no room for the guide to slip.",
            }
        },
    )

    payload = {"symbol": "AAPL", "pe_ratio": 28.4, "price": 175.0}
    out = _run_market_analyst(monkeypatch, registry, payload)
    content = out["messages"][-1].content

    judge_kwargs = registry.only_call("ThesisEvaluation")
    evidence = judge_kwargs["evidence_context"]
    # The judge saw the raw tool observation, not just the three theses.
    assert "28.4" in evidence
    assert "Symbol: AAPL" in evidence
    assert judge_kwargs["thesis_bull"] == "<thesis>"
    # And the generators were fed the same block.
    assert registry.only_call("BullThesisGeneration")["context"] == evidence

    assert "### 🥊 Strongest Objection" in content
    assert "A P/E of 28.4 leaves no room" in content


def test_market_analyst_renders_the_sentinel_when_the_judge_omits_the_objection(monkeypatch):
    """A judge that returns nothing must not produce a blank section — blank reads as
    'there is no case against this', which is a claim the report never earned."""
    registry = install_fake_lm(
        monkeypatch,
        overrides={
            "ThesisEvaluation": {
                "selected_thesis": "BULL",
                "confidence_level": "HIGH",
                "strongest_objection": "   ",
            }
        },
    )
    out = _run_market_analyst(monkeypatch, registry, {"symbol": "AAPL", "pe_ratio": 28.4})
    content = out["messages"][-1].content
    assert "### 🥊 Strongest Objection" in content
    assert "INSUFFICIENT EVIDENCE" in content


# --------------------------------------------------------------------------- #
# Removed: DeepReasoning "Path A"
#
# There used to be a second integration path here, driving DeepReasoning's
# Tree-of-Thought branch. That branch was deleted on 2026-07-31 as unreachable —
# it was gated on `data_context['symbol']`, which nothing in production ever
# wrote, so these tests were exercising code that could not run.
#
# Neither property they covered is now untested:
#
#   - "empty evidence must force the sentinel, not a fluent invention" is the
#     `ground_strongest_objection` / `evidence_is_empty` gate above, which is
#     where the logic actually lives. It cannot arise on the MarketAnalyst path:
#     with no fundamentals and no macro, the ToT block is never entered at all.
#
#   - "the length directive must not leak into the judge's evidence field" was a
#     hazard specific to Path A, which appended `length_rule` to the evidence
#     block. MarketAnalyst passes `build_evidence_context(...)` to the generators
#     and the judge unchanged, so there is nothing to leak.
# --------------------------------------------------------------------------- #
