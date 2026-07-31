"""Resilience of the background scheduler against failure storms.

Covers the three coordinated guards added after the 2026-07-24 incident (a
missing Vertex key drove 152 failed LLM-client builds across the morning because
the LLM tasks retried every 5 min and swallowed their own per-profile errors):

  1. llm_ready()            — skip cleanly when the provider is unconfigured.
  2. _raise_if_total_failure — surface an all-profiles-failed tick to the breaker.
  3. circuit breaker         — pause a task after N consecutive failures.
"""
import asyncio

import pytest

import tools.scheduler as sched

# --- 1. llm_ready() -------------------------------------------------------

def test_llm_ready_vertex_missing_then_present(monkeypatch):
    from agent.utils import llm_ready
    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_KEY", raising=False)
    ok, why = llm_ready()
    assert ok is False
    assert "GOOGLE_SERVICE_ACCOUNT_KEY" in why

    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_KEY", '{"project_id": "x"}')
    ok, why = llm_ready()
    assert ok is True
    assert why == ""


def test_llm_ready_azure_needs_both(monkeypatch):
    from agent.utils import llm_ready
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    ok, why = llm_ready()
    assert ok is False and "AZURE_OPENAI_ENDPOINT" in why


def test_llm_ready_bedrock_is_never_blocked(monkeypatch):
    """Bedrock auth (IAM role / ~/.aws) isn't offline-verifiable — never block it."""
    from agent.utils import llm_ready
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    ok, why = llm_ready()
    assert ok is True and why == ""


# --- 2. _raise_if_total_failure ------------------------------------------

def test_total_failure_raises_when_every_enabled_profile_failed():
    with pytest.raises(RuntimeError):
        sched._raise_if_total_failure("t", {"a": "error: boom", "b": "failed (will retry next tick)"})


def test_partial_success_does_not_raise():
    sched._raise_if_total_failure("t", {"a": "ran", "b": "error: boom"})  # no raise


@pytest.mark.parametrize("results", [
    {"a": "scheduler disabled", "b": "scheduler disabled"},  # nobody enabled
    {"a": "already done today"},                              # already have it
    {},                                                       # no profiles
])
def test_no_raise_when_nothing_actually_failed(results):
    sched._raise_if_total_failure("t", results)  # no raise


def test_disabled_profiles_are_excluded_from_the_verdict():
    # one enabled profile ran; the other opted out -> not a total failure
    sched._raise_if_total_failure("t", {"a": "ran", "b": "scheduler disabled"})


# --- 3. circuit breaker in _try_run --------------------------------------

@pytest.fixture
def isolated_runner(monkeypatch):
    """A CairnIQScheduler whose _try_run guards are neutralised except the
    breaker, so tests exercise the breaker deterministically."""
    monkeypatch.setattr(sched, "get_scheduler_settings", lambda: {})
    monkeypatch.setattr(sched, "get_scheduler_cooldowns", lambda: {})
    monkeypatch.setattr(sched, "_can_run", lambda *a, **k: True)
    monkeypatch.setattr(sched, "_record_run", lambda *a, **k: None)
    import tools.engine_heartbeat as hb
    monkeypatch.setattr(hb, "begin", lambda *a, **k: None)
    monkeypatch.setattr(hb, "record_run", lambda *a, **k: None)
    alerts: list[tuple[str, int]] = []
    monkeypatch.setattr(sched, "_alert_circuit_open", lambda t, n: alerts.append((t, n)))
    return sched.CairnIQScheduler(), alerts


def test_breaker_opens_after_consecutive_failures_and_stops_running(isolated_runner):
    s, alerts = isolated_runner
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise RuntimeError("nope")

    async def drive():
        for _ in range(6):
            await s._try_run("faily", boom, cooldown=0, timeout=5)

    asyncio.run(drive())

    # the task ran exactly the threshold number of times, then was paused
    assert calls["n"] == sched.MAX_CONSECUTIVE_FAILURES
    assert alerts == [("faily", sched.MAX_CONSECUTIVE_FAILURES)]
    assert s._circuit_is_open("faily") is True


def test_a_success_resets_the_streak(isolated_runner):
    s, alerts = isolated_runner
    state = {"fail": True, "calls": 0}

    async def flaky():
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("transient")

    async def drive():
        # two failures, then a success, then two more failures -> never 3 in a row
        for pattern in (True, True, False, True, True):
            state["fail"] = pattern
            await s._try_run("flaky", flaky, cooldown=0, timeout=5)

    asyncio.run(drive())

    assert alerts == []                      # breaker never tripped
    assert s._circuit_is_open("flaky") is False
    assert state["calls"] == 5               # every attempt actually ran


def test_timeout_counts_as_a_failure(isolated_runner, monkeypatch):
    s, alerts = isolated_runner

    async def slow():
        await asyncio.sleep(10)

    async def drive():
        for _ in range(sched.MAX_CONSECUTIVE_FAILURES + 2):
            await s._try_run("slow", slow, cooldown=0, timeout=0.01)

    asyncio.run(drive())
    assert alerts == [("slow", sched.MAX_CONSECUTIVE_FAILURES)]
    assert s._circuit_is_open("slow") is True
