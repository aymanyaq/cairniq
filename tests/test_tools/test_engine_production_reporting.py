"""Production reporting across every scheduled engine (Roadmap 2.6).

2.5 gave all ten engines LIVENESS for free with one hook in the runner, but the
IDLE signal — the one that catches an engine running forever and producing
nothing — needs each engine to declare what it did. Only two ever called
`note_production`, and neither of them was an engine whose outage motivated the
work: the funnel signal log sat dark 07-02 -> 07-18, and the recommendation
ledger had zero scored calls ever. Both were in the uninstrumented eight.

The distinction every test here defends: a tick that DECLINED (already done
today, outside the window, disabled) must record a SKIP, while a tick that RAN
and produced nothing must accrue idleness. Collapse those two and the instrument
either cries wolf on healthy engines or goes silent on dead ones — this codebase
has now shipped both failures once each.
"""
import inspect

import pytest

import tools.engine_heartbeat as hb
import tools.scheduler as sched


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(hb, "_HEARTBEAT_PATH", str(tmp_path / "engine_heartbeat.json"))
    hb._current.update({"engine": None, "produced": 0, "detail": "", "skipped": None})
    sched.scheduler._circuit_opened_on.clear()
    sched.scheduler._failure_streak.clear()
    return tmp_path


# Engines deliberately left liveness-only, with the reason. Anything NOT in here
# must declare production, or the coverage guard below fails.
LIVENESS_ONLY = {
    "cache_cleanup": "0 files removed is the healthy steady state — reporting it "
                     "would flag a working engine inside a fortnight",
}


def test_every_scheduled_engine_declares_production_or_is_a_named_exception():
    """The guard that makes 2.6 stick.

    The failure being prevented is procedural, not logical: someone adds a task
    to SCHEDULED_TASKS, it runs for months, and nobody notices it produces
    nothing because it never opted into idleness tracking. Silence is the
    symptom of both health and death here, so opting in has to be enforced
    rather than remembered.
    """
    missing = []
    for name, fn, _cooldown, _timeout in sched.SCHEDULED_TASKS:
        if name in LIVENESS_ONLY:
            continue
        source = inspect.getsource(fn)
        if "note_production" not in source and "_note_engine_outcome" not in source:
            missing.append(name)

    assert not missing, (
        f"These engines never declare production, so a dead one is invisible: {missing}. "
        f"Either call note_production/_note_engine_outcome with a count that proves the "
        f"chain ran, or add the engine to LIVENESS_ONLY with the reason."
    )


def test_the_two_engines_whose_outages_motivated_this_are_covered():
    """Named explicitly because 2.5 shipped without covering either of them, and
    a generic coverage test would have passed just as happily while they stayed
    dark."""
    for name in ("funnel_signal_scan", "score_recommendations"):
        fn = dict((n, f) for n, f, _c, _t in sched.SCHEDULED_TASKS)[name]
        source = inspect.getsource(fn)
        assert "note_production" in source or "_note_engine_outcome" in source, name


# ---------------------------------------------------------------------------
# Declined vs idle — the distinction the whole item turns on
# ---------------------------------------------------------------------------

def test_a_tick_where_every_profile_already_did_the_work_is_a_skip():
    """portfolio_snapshot is re-checked every ~5 min but snapshots once a day.
    If the ~200 remaining ticks counted as 'ran and produced nothing', it would
    sit on a 200-run idle streak by nightfall — every marker-gated engine would
    read as dead by morning."""
    hb.begin("portfolio_snapshot")
    sched._note_engine_outcome(
        worked=0, produced=0, declined_reason="all profiles already snapshotted today"
    )
    rec = hb.record_run("portfolio_snapshot")

    assert rec["last_status"] == hb.STATUS_SKIPPED
    assert rec.get("consecutive_idle", 0) == 0


def test_a_tick_that_ran_and_produced_nothing_accrues_idleness():
    """The other half. An engine that reached its logic and did no work is the
    dead-engine signal, and must NOT be softened into a skip."""
    for _ in range(12):
        hb.begin("edgar_events")
        sched._note_engine_outcome(worked=2, produced=0, declined_reason="unused")
        hb.record_run("edgar_events")

    health = hb.get_engine_health(idle_threshold=10)

    assert health["engines"]["edgar_events"]["consecutive_idle"] == 12
    assert any(c["engine"] == "edgar_events" for c in health["concerning"])


def test_real_work_clears_the_streak():
    hb.begin("edgar_events")
    sched._note_engine_outcome(worked=2, produced=37, declined_reason="unused")
    rec = hb.record_run("edgar_events")

    assert rec["last_produced"] == 37
    assert rec["consecutive_idle"] == 0
    assert hb.get_engine_health()["concerning"] == []


# ---------------------------------------------------------------------------
# What counts as "worked"
# ---------------------------------------------------------------------------

def test_declines_are_not_work_but_real_outcomes_are():
    results = {
        "a": "already done today",
        "b": "scheduler disabled",
        "c": "ran",
        "d": "error: boom",
    }

    assert sched._count_worked(results) == 1


def test_an_attempt_that_failed_counts_as_worked_so_it_can_show_as_idle():
    """priority_precompute returns 'failed (will retry next tick)' when the run
    did not complete. That attempted the work, so the tick is NOT a skip — and
    because it produced nothing, a run of failures accrues a visible streak
    instead of looking like a quiet morning."""
    results = {"a": "failed (will retry next tick)"}

    assert sched._count_worked(results) == 1


def test_errors_are_not_counted_as_work():
    """An exception is already surfaced by the runner's error status; counting it
    as work would double-report it and muddy the produced number."""
    assert sched._count_worked({"a": "error: kaboom"}) == 0


# ---------------------------------------------------------------------------
# Production counts must prove the CHAIN ran, not that something interesting
# happened. An engine judged on rare events reads as dead every quiet week.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine,rare_thing,chain_thing", [
    ("edgar_events", "alerts", "symbols"),
    ("score_recommendations", "scored", "rows"),
    ("observation_consolidation", "drafted", "observations"),
])
def test_production_is_measured_on_the_chain_not_on_rare_events(engine, rare_thing, chain_thing):
    """EDGAR counts SYMBOLS POLLED, not alerts raised (a material 8-K is rare and
    legitimately 0 for weeks). The ledger scorer counts LEDGER ROWS WALKED, not
    rows newly scored (scoring only fires at the 14-day horizon, so 'no change'
    is the normal daily outcome). Getting this backwards would make both engines
    report themselves dead while working perfectly."""
    fn = dict((n, f) for n, f, _c, _t in sched.SCHEDULED_TASKS)[engine]
    source = inspect.getsource(fn)

    assert f'{chain_thing}["' in source or f'"{chain_thing}"' in source, (
        f"{engine} should count {chain_thing} (proof the chain ran)"
    )


# ---------------------------------------------------------------------------
# Driven through the REAL runner and the REAL task, not a mock of either
# ---------------------------------------------------------------------------
# The checks above read source, which proves a call EXISTS but not that it fires
# on the production path. These drive CairnIQScheduler._try_run against the
# actual task coroutine — the step whose absence let 3.3 ship dead with 56 green
# tests.

def _drive(monkeypatch, task_name, task_fn):
    import asyncio

    monkeypatch.setattr(sched, "get_scheduler_settings", lambda: {task_name: True})
    monkeypatch.setattr(sched, "get_scheduler_cooldowns", lambda: {})
    monkeypatch.setattr(sched, "_can_run", lambda *a, **k: True)
    monkeypatch.setattr(sched, "_record_run", lambda *a, **k: None)
    asyncio.run(sched.scheduler._try_run(task_name, task_fn, 0, 30))
    return hb.get_heartbeats()[task_name]


def test_an_out_of_window_snapshot_records_a_skip_through_the_real_runner(monkeypatch):
    """Before 2.6 this early-return recorded `ok` — the coroutine completed, so
    a permanently dead engine and an overnight no-op looked identical, which is
    the exact `scheduler_runs.json` ambiguity 2.5 was built to delete and then
    reproduced inside itself."""
    from datetime import datetime

    monkeypatch.setattr(sched, "_eastern_now", lambda: datetime(2026, 7, 25, 12, 0))  # Saturday

    rec = _drive(monkeypatch, "portfolio_snapshot", sched.task_portfolio_snapshot)

    assert rec["last_status"] == hb.STATUS_SKIPPED
    assert rec["last_detail"] == "before market close"
    assert rec.get("consecutive_idle", 0) == 0


def test_a_dead_exchange_rate_feed_shows_as_produced_zero_through_the_real_runner(monkeypatch):
    """One rate is the whole job, so there is no quiet-day reading: a provider
    returning nothing is unambiguously a broken feed and must accrue idleness."""
    import yfinance as yf

    class _EmptyTicker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            import pandas as pd
            return pd.DataFrame()

    monkeypatch.setattr(yf, "Ticker", _EmptyTicker)

    rec = _drive(monkeypatch, "exchange_rate", sched.task_exchange_rate)

    assert rec["last_status"] == hb.STATUS_OK
    assert rec["reports_production"] is True
    assert rec["last_produced"] == 0
    assert rec["consecutive_idle"] == 1


def test_an_aborted_funnel_scan_reports_zero_rather_than_skipping():
    """The 07-02 -> 07-18 shape, precisely. A scan that starts and aborts before
    the signal-log write has ATTEMPTED and produced nothing; recording it as a
    skip would restore the exact silence that let the log sit dark for sixteen
    days with every test green."""
    source = inspect.getsource(sched.task_funnel_signal_scan)

    assert "note_production(0" in source, (
        "an aborted scan must report produced=0, not note_skipped — otherwise a "
        "permanently-aborting scanner is indistinguishable from a quiet evening"
    )


# ---------------------------------------------------------------------------
# A switch held OFF is not a dead engine
# ---------------------------------------------------------------------------
# cache_warm is the one engine gated by its own flag (CAIRNIQ_CACHE_WARM, or a
# deployment that opts out) rather than by SCHEDULED_TASKS' enabled setting. The
# runner therefore still invokes it every 10 minutes, so whatever it reports on a
# disabled install repeats ~144 times a day.

def _drive_disabled_warm(monkeypatch):
    # Pinned through both gates: warm_enabled() is already False under pytest, and
    # this keeps the test on the disabled branch if that guard ever moves.
    monkeypatch.setenv("CAIRNIQ_CACHE_WARM", "false")
    from tools.cache_warm import warm_enabled
    assert warm_enabled() is False, "precondition: this test drives the disabled branch"
    return _drive(monkeypatch, "cache_warm", sched.task_cache_warm)


def test_a_disabled_cache_warm_records_a_skip_with_a_reason(monkeypatch):
    """It reported `ok` with produced=0 — i.e. "ran and did nothing", the
    dead-engine signal — for a switch someone turned off on purpose."""
    rec = _drive_disabled_warm(monkeypatch)

    assert rec["last_status"] == hb.STATUS_SKIPPED
    assert rec["last_detail"] == "cache warming disabled"
    assert rec.get("consecutive_idle", 0) == 0
    # Not just "idle == 0": the skip must never have entered production accounting
    # at all, or the next real run's streak arithmetic starts from a lie.
    assert "last_produced" not in rec


def test_a_day_of_disabled_ticks_never_presents_as_a_dead_engine(monkeypatch):
    """The operational claim. At a 600s cooldown a disabled install ticks ~144
    times a day, so the old reading crossed the idle threshold before lunch and
    parked a switched-off engine on the ops view permanently."""
    for _ in range(15):
        _drive_disabled_warm(monkeypatch)

    health = hb.get_engine_health(idle_threshold=10)
    rec = health["engines"]["cache_warm"]

    assert rec["skips"] == 15
    assert rec.get("consecutive_idle", 0) == 0
    assert not [c for c in health["concerning"] if c["engine"] == "cache_warm"]


def test_an_enabled_warm_that_warmed_nothing_still_accrues_idleness(monkeypatch):
    """The other half, and the reason the fix is scoped to the disabled branch
    only: with warming ON there is no market-hours or once-a-day window to
    decline, so warming zero profiles means every profile's summary AND radar
    failed. That is a broken feed and must stay visible as idleness."""
    import tools.cache_warm as cw

    monkeypatch.setattr(cw, "warm_enabled", lambda: True)
    monkeypatch.setattr(cw, "warm_all_profiles",
                        lambda: {"default": {"summary": False, "radar": False}})

    rec = _drive(monkeypatch, "cache_warm", sched.task_cache_warm)

    assert rec["last_status"] == hb.STATUS_OK
    assert rec["reports_production"] is True
    assert rec["last_produced"] == 0
    assert rec["consecutive_idle"] == 1


def test_a_warm_that_refreshed_profiles_reports_them_as_production(monkeypatch):
    import tools.cache_warm as cw

    monkeypatch.setattr(cw, "warm_enabled", lambda: True)
    monkeypatch.setattr(cw, "warm_all_profiles", lambda: {
        "a": {"summary": True, "radar": True},
        "b": {"summary": True, "radar": False},   # half a warm is still a warm
        "c": {"summary": False, "radar": False},
    })

    rec = _drive(monkeypatch, "cache_warm", sched.task_cache_warm)

    assert rec["last_status"] == hb.STATUS_OK
    assert rec["last_produced"] == 2
    assert rec["consecutive_idle"] == 0
