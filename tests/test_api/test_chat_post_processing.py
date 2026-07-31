"""Regression tests for _run_chat_post_processing's gates.

scan_opportunities/guru/portfolio-audit turns never authorize a trade call by
lens contract, and RiskManager independently confirms that on the same turn —
so they must not get auto-logged into the recommendation ledger. Only the gate
decision is tested here; the LLM-driven extraction and summary logic are
pre-existing and exercised separately.

Roadmap 1.7 added a second consumer at this seam — the observation log — and its
privacy gate is tested the same way. Nothing here may reach a live profile, so
both the observation writer and the portfolio read are stubbed.
"""
import asyncio

import api.routers.chat as chat
import tools.observations as obs


def _run(monkeypatch, human_query, **kwargs):
    calls = []

    async def _extract(text, **_kwargs):
        calls.append("extract")

    monkeypatch.setattr(chat, "_extract_and_log_recommendations", _extract)
    monkeypatch.setattr(chat, "_update_conversation_summary", lambda thread_id: _noop())
    monkeypatch.setattr(obs, "load_holdings_map", lambda: {})
    monkeypatch.setattr(
        obs, "observe_turn",
        lambda *a, **k: calls.append("observe") or [],
    )
    asyncio.run(chat._run_chat_post_processing(
        "some response text", "thread-1", human_query, **kwargs
    ))
    return calls


async def _noop():
    return None


def test_screener_lens_skips_recommendation_extraction(monkeypatch):
    for lens in ("portfolio_audit", "external_screen", "guru_validation"):
        calls = _run(monkeypatch, f"[MarketAnalyst lens={lens}] do the thing")
        assert "extract" not in calls, f"lens={lens} should skip extraction"


def test_market_dip_still_extracts_recommendations(monkeypatch):
    assert "extract" in _run(monkeypatch, "[MarketAnalyst lens=market_dip] build a deployment plan")


def test_plain_query_still_extracts_recommendations(monkeypatch):
    assert "extract" in _run(monkeypatch, "Should I buy AAPL?")


# ---------------------------------------------------------------------------
# Observation log (roadmap 1.7)
# ---------------------------------------------------------------------------

def test_an_ordinary_turn_reaches_the_observation_log(monkeypatch):
    """The seam 1.7 moved the write path TO. A unit test on the log proves the
    log works; this proves something calls it."""
    assert "observe" in _run(monkeypatch, "Should I buy AAPL?")


def test_a_screener_lens_still_observes(monkeypatch):
    """The lens gate is about the ADVICE LEDGER, not about memory. A portfolio
    audit is still a turn the user took."""
    calls = _run(monkeypatch, "[MarketAnalyst lens=portfolio_audit] review this")
    assert calls == ["observe"]


def test_a_ghost_turn_observes_nothing(monkeypatch):
    assert _run(monkeypatch, "Should I buy AAPL?", ghost=True) == ["extract"]


def test_a_private_tagged_turn_observes_nothing(monkeypatch):
    assert _run(monkeypatch, "@Private should I buy AAPL?") == ["extract"]


def test_one_failed_step_does_not_take_the_others_with_it(monkeypatch):
    """The extractor builds its LLM client outside its own try, so an
    unconfigured provider used to raise straight out of the coroutine and
    silently take the summary and the observation log with it. The log is
    deterministic and needs no model — losing it to someone else's provider
    outage would be losing the one tier that still works."""
    calls = []

    async def _boom(text, **_kwargs):
        raise ValueError("LLM_PROVIDER=bedrock but AIDLC_MODEL_ID is not set.")

    monkeypatch.setattr(chat, "_extract_and_log_recommendations", _boom)
    monkeypatch.setattr(
        chat, "_update_conversation_summary",
        lambda thread_id: calls.append("summary") or _noop(),
    )
    monkeypatch.setattr(obs, "load_holdings_map", lambda: {})
    monkeypatch.setattr(obs, "observe_turn", lambda *a, **k: calls.append("observe") or [])

    asyncio.run(chat._run_chat_post_processing("answer", "thread-1", "Should I buy AAPL?"))

    assert calls == ["summary", "observe"]
