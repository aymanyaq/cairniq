import threading

import pytest

from agent.cost_tracker import (
    USD_TO_CAD,
    accumulate_cost,
    accumulate_grounding,
    get_session_breakdown,
    get_session_cost,
    get_session_stats,
    get_session_tokens,
    reset_session_cost,
    track_dspy_calls,
    track_embedding_cost,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Deterministic environment: known FX rate, no stray price config, and a
    known slot→model mapping. The cost tracker prices the *slot*, not the model,
    so a slot is only billed when its AIDLC_PRICE_<SLOT> is set."""
    monkeypatch.setenv("USD_TO_CAD", str(USD_TO_CAD))
    monkeypatch.setenv("AIDLC_MODEL_ID", "primary-model")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "fast-model")

    # accumulate_cost feeds the persistent budget meter, which writes to disk on
    # every single call. The concurrency tests below make thousands of calls, and
    # the real writes dominated this file's runtime (~6.9ms each). What is under
    # test here is the in-memory accounting; llm_budget has its own tests.
    monkeypatch.setattr("agent.llm_budget.record", lambda **kwargs: None)
    for k in (
        "LLM_PROVIDER", "AIDLC_EMBED_MODEL_ID",
        "AIDLC_PRICE_PRIMARY", "AIDLC_PRICE_FAST", "AIDLC_PRICE_EMBED", "AIDLC_PRICE_OTHER",
        "AIDLC_PRICE_GROUNDING",
    ):
        monkeypatch.delenv(k, raising=False)
    reset_session_cost()
    yield
    reset_session_cost()


def test_tokens_tracked_even_when_unpriced():
    # No AIDLC_PRICE_* set → tokens are still tracked, but cost stays 0 and the
    # session is flagged unpriced (instead of being silently mispriced).
    res = accumulate_cost(100_000, 10_000, model_id="primary-model")
    assert res == 0.0
    stats = get_session_stats()
    assert stats["input_tokens"] == 100_000
    assert stats["output_tokens"] == 10_000
    assert stats["cost_cad"] == 0.0
    assert stats["any_unpriced"] is True
    assert get_session_tokens() == 110_000


def test_priced_primary_slot_computes_cost(monkeypatch):
    # primary slot priced at $5 in / $25 out per 1M; 1M in + 1M out = 30 USD.
    monkeypatch.setenv("AIDLC_PRICE_PRIMARY", "5/25")
    res = accumulate_cost(1_000_000, 1_000_000, model_id="primary-model")
    assert res == 30.0 * USD_TO_CAD
    assert get_session_stats()["any_unpriced"] is False


def test_priced_fast_slot_computes_cost(monkeypatch):
    # fast slot priced at $0.15 in / $0.60 out per 1M; 1M+1M = 0.75 USD.
    monkeypatch.setenv("AIDLC_PRICE_FAST", "0.15/0.60")
    res = accumulate_cost(1_000_000, 1_000_000, model_id="fast-model")
    assert round(res, 4) == round(0.75 * USD_TO_CAD, 4)


def test_slot_mapping_by_configured_model():
    # primary-model → primary, fast-model → fast, titan-embed → embed,
    # an unrecognized id → other. (No prices set, so we assert on tokens/slots.)
    accumulate_cost(1000, 100, model_id="primary-model")
    accumulate_cost(2000, 200, model_id="fast-model")
    accumulate_cost(300, 0, model_id="titan-embed")
    accumulate_cost(50, 5, model_id="some-unknown-model")
    bd = get_session_breakdown()
    assert bd["primary"]["input_tokens"] == 1000
    assert bd["fast"]["input_tokens"] == 2000
    assert bd["embed"]["input_tokens"] == 300
    assert bd["other"]["input_tokens"] == 50


def test_cache_read_discount(monkeypatch):
    # primary priced "in/out/cache" = 3/15/0.3. 1M total, 500k cached:
    # (0.5 * 3) + (0.5 * 0.3) = 1.65 USD.
    monkeypatch.setenv("AIDLC_PRICE_PRIMARY", "3/15/0.3")
    res = accumulate_cost(1_000_000, 0, model_id="primary-model", cache_read_tokens=500_000)
    assert round(res, 4) == round(1.65 * USD_TO_CAD, 4)


def test_track_embedding_routes_to_embed_slot(monkeypatch):
    monkeypatch.setenv("AIDLC_PRICE_EMBED", "0.02/0")
    track_embedding_cost(1_000_000)
    assert round(get_session_cost(), 6) == round(0.02 * USD_TO_CAD, 6)
    assert get_session_breakdown()["embed"]["input_tokens"] == 1_000_000


def test_thread_safety():
    def worker():
        for _ in range(100):
            accumulate_cost(1000, 1000, "primary-model")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = get_session_stats()
    assert stats["input_tokens"] == 100 * 10 * 1000
    assert stats["output_tokens"] == 100 * 10 * 1000


def test_reset_session(monkeypatch):
    monkeypatch.setenv("AIDLC_PRICE_PRIMARY", "5/25")
    accumulate_cost(1000, 1000, model_id="primary-model")
    assert get_session_cost() > 0
    assert get_session_tokens() == 2000
    reset_session_cost()
    assert get_session_cost() == 0
    assert get_session_tokens() == 0
    assert get_session_stats()["input_tokens"] == 0
    assert get_session_breakdown() == {}


# --- Grounding: billed per request, not in tokens -------------------------


def test_grounding_is_counted_but_adds_no_tokens():
    accumulate_grounding(requests=1, queries=3)
    stats = get_session_stats()
    assert stats["grounding_requests"] == 1
    assert stats["grounding_queries"] == 3
    # A grounded search is NOT a token charge — the tokens of the same response
    # are already counted through the ordinary usage path.
    assert stats["input_tokens"] == 0
    assert stats["output_tokens"] == 0
    assert get_session_tokens() == 0


def test_grounding_priced_per_thousand_requests(monkeypatch):
    monkeypatch.setenv("AIDLC_PRICE_GROUNDING", "35")
    accumulate_grounding(requests=1000)
    assert round(get_session_cost(), 6) == round(35.0 * USD_TO_CAD, 6)
    assert get_session_stats()["any_unpriced"] is False


def test_unpriced_grounding_flags_the_session():
    accumulate_grounding(requests=2)
    stats = get_session_stats()
    assert stats["cost_cad"] == 0.0
    # Requests happened but no price is configured — say so rather than report
    # a confident zero.
    assert stats["any_unpriced"] is True


def test_grounding_appears_in_breakdown_only_once_it_fires():
    assert "grounding" not in get_session_breakdown()
    accumulate_grounding(requests=1, queries=2)
    assert get_session_breakdown()["grounding"]["requests"] == 1
    reset_session_cost()
    assert get_session_breakdown() == {}


# --- DSPy history flush ---------------------------------------------------


class _FakeLM:
    """Stands in for a dspy LM: a history list and a cursor attribute."""
    def __init__(self, entries):
        self.history = list(entries)


def test_track_dspy_calls_accumulates_and_advances_the_cursor():
    lm = _FakeLM([
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20}, "model": "primary-model"},
        {"usage": {"input_tokens": 50, "output_tokens": 10}, "model": "fast-model"},
    ])
    assert track_dspy_calls(lm) == 2
    assert get_session_breakdown()["primary"]["input_tokens"] == 100
    assert get_session_breakdown()["fast"]["output_tokens"] == 10

    # A second flush with nothing new must not re-count.
    assert track_dspy_calls(lm) == 0
    assert get_session_breakdown()["primary"]["input_tokens"] == 100


def test_metered_lm_flushes_after_history_is_written():
    """The flush must hang off __call__, not forward().

    dspy.BaseLM.__call__ runs forward() and only then appends the history entry
    (in _process_lm_response). A flush placed inside forward() would therefore
    read a history that does not yet contain the call just made — lagging one
    call behind forever and losing the last call of every session. This drives
    the real CleaningLM with that exact ordering: forward returns first, history
    lands second, and the usage must still be counted.
    """
    dspy_setup = pytest.importorskip("agent.dspy_setup")
    if not dspy_setup.DSPY_AVAILABLE:
        pytest.skip("dspy not installed")

    lm = dspy_setup.CleaningLM("openai/gpt-4o-mini", api_key="test-not-real")
    lm.forward = lambda prompt=None, messages=None, **kw: "RAW"

    def _process(response, prompt, messages, **kw):
        lm.history.append(
            {"usage": {"prompt_tokens": 7, "completion_tokens": 3}, "model": "primary-model"}
        )
        return ["out"]

    lm._process_lm_response = _process

    assert lm(messages=[{"role": "user", "content": "hi"}]) == ["out"]
    assert get_session_breakdown()["primary"]["input_tokens"] == 7
    assert get_session_breakdown()["primary"]["output_tokens"] == 3


def test_track_dspy_calls_does_not_double_count_under_concurrency():
    # The cursor is shared mutable state on one global LM, and scans issue DSPy
    # calls from worker threads. Two flushes racing on the same marker would
    # bill the overlap twice.
    lm = _FakeLM([
        {"usage": {"prompt_tokens": 10, "completion_tokens": 1}, "model": "primary-model"}
        for _ in range(200)
    ])
    threads = [threading.Thread(target=track_dspy_calls, args=(lm,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = get_session_stats()
    assert stats["input_tokens"] == 200 * 10
    assert stats["output_tokens"] == 200 * 1
