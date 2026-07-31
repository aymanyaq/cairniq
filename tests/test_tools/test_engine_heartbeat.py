"""Engine liveness heartbeat (Advisor Roadmap 2.5).

The module exists because three separate engines ran dead without anyone
noticing (funnel signal log dark 07-02 -> 07-18; recommendation ledger with zero
scored calls ever; 3.3 watch-conditions harvesting nothing on its first real
morning). The tests that matter here are therefore not "does it store a dict" —
they are: does a dead engine become VISIBLE, and does a healthy-but-quiet engine
stay quiet so the signal is worth reading.
"""
import json

import pytest

import tools.engine_heartbeat as hb


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    """Isolate the global heartbeat file and reset the reporting window."""
    monkeypatch.setattr(hb, "_HEARTBEAT_PATH", str(tmp_path / "engine_heartbeat.json"))
    hb._current.update({"engine": None, "produced": 0, "detail": "", "skipped": None})
    # The scheduler singleton carries breaker state across tests in one process,
    # and an open breaker is now a `concerning` row — so clear it, or an
    # unrelated test could make these assertions depend on execution order.
    import tools.scheduler as sched
    sched.scheduler._circuit_opened_on.clear()
    sched.scheduler._failure_streak.clear()
    return tmp_path


def _run(engine, produced=None, skipped=None, status=hb.STATUS_OK):
    """Simulate one scheduler-driven run of `engine`.

    `produced=None` models an UNINSTRUMENTED engine (never calls
    note_production); `produced=0` models an instrumented engine reporting that
    it did no work. The two must not behave the same — see the tests below.
    """
    hb.begin(engine)
    if skipped is not None:
        hb.note_skipped(skipped)
    if produced is not None:
        hb.note_production(produced)
    return hb.record_run(engine, status=status)


# ---------------------------------------------------------------------------
# The distinction the whole module exists for
# ---------------------------------------------------------------------------

def test_an_uninstrumented_engine_is_never_flagged_as_dead():
    """Caught in production on the first deploy: engines that do not report
    production (portfolio_snapshot, premarket_pulse, ...) were accruing an idle
    streak, so within the hour the ops view would have filled with false "never
    produced" alarms. An engine cannot be judged on production it was never
    asked to declare — and an instrument that cries wolf stops being read."""
    for _ in range(40):
        _run("portfolio_snapshot")            # never calls note_production

    health = hb.get_engine_health(idle_threshold=10)

    assert health["engines"]["portfolio_snapshot"]["last_status"] == hb.STATUS_OK
    assert health["engines"]["portfolio_snapshot"].get("consecutive_idle", 0) == 0
    assert health["concerning"] == []


def test_reporting_zero_is_a_dead_signal_not_an_absence_of_instrumentation():
    """The flip side, and the reason the distinction exists: watch-conditions
    reports 0 when its store is empty, which is exactly the 3.3 outage. That must
    accrue idleness even though the number is the same 0 the uninstrumented
    engines show."""
    for _ in range(12):
        _run("watch_conditions", produced=0)

    health = hb.get_engine_health(idle_threshold=10)

    assert health["engines"]["watch_conditions"]["consecutive_idle"] == 12
    assert any(c["engine"] == "watch_conditions" for c in health["concerning"])


def test_a_dead_engine_becomes_visible():
    """The 3.3 outage shape: the engine runs on schedule, completes cleanly, and
    produces NOTHING, every time. scheduler_runs.json calls this healthy. Here it
    must surface as concerning, and say it has never produced."""
    for _ in range(12):
        _run("watch_conditions", produced=0)

    health = hb.get_engine_health(idle_threshold=10)

    assert health["engines"]["watch_conditions"]["consecutive_idle"] == 12
    assert health["engines"]["watch_conditions"]["last_status"] == hb.STATUS_OK
    concerning = {c["engine"]: c["why"] for c in health["concerning"]}
    assert "watch_conditions" in concerning
    assert "NEVER produced" in concerning["watch_conditions"]


def test_a_working_engine_is_not_flagged():
    """A quiet engine that DOES do real work must stay out of `concerning`, or
    the view cries wolf and stops being read — which is how we got here."""
    for _ in range(30):
        _run("watch_conditions", produced=4)  # 4 conditions checked each tick

    health = hb.get_engine_health(idle_threshold=10)

    assert health["engines"]["watch_conditions"]["consecutive_idle"] == 0
    assert health["concerning"] == []


def test_skipped_runs_never_accrue_an_idle_streak():
    """Overnight and weekend ticks early-return by design. If those counted as
    idle, every market-hours engine would look dead by Monday and the signal
    would be worthless."""
    for _ in range(50):
        _run("intraday_sentinel", skipped="outside market hours")

    rec = hb.get_heartbeats()["intraday_sentinel"]

    assert rec["last_status"] == hb.STATUS_SKIPPED
    assert rec["skips"] == 50
    assert rec.get("consecutive_idle", 0) == 0
    assert hb.get_engine_health()["concerning"] == []


def test_production_resets_the_idle_streak():
    for _ in range(8):
        _run("intraday_sentinel", produced=0)
    assert hb.get_heartbeats()["intraday_sentinel"]["consecutive_idle"] == 8

    _run("intraday_sentinel", produced=11)
    rec = hb.get_heartbeats()["intraday_sentinel"]

    assert rec["consecutive_idle"] == 0
    assert rec["last_produced"] == 11
    assert rec["last_produced_at"] is not None
    assert rec["total_produced"] == 11


def test_an_engine_that_stopped_producing_is_flagged_with_its_last_success():
    """Distinct from never-produced: this one worked, then went dark. The last
    good timestamp is the thing that makes it diagnosable."""
    _run("funnel_signal_scan", produced=69)
    for _ in range(10):
        _run("funnel_signal_scan", produced=0)

    concerning = {c["engine"]: c["why"] for c in hb.get_engine_health(idle_threshold=10)["concerning"]}

    assert "funnel_signal_scan" in concerning
    assert "NEVER" not in concerning["funnel_signal_scan"]
    assert "since" in concerning["funnel_signal_scan"]


def test_errors_are_always_surfaced_regardless_of_idle_count():
    _run("edgar_events", status=hb.STATUS_ERROR)

    health = hb.get_engine_health()

    assert health["engines"]["edgar_events"]["errors"] == 1
    assert any(c["engine"] == "edgar_events" for c in health["concerning"])


# ---------------------------------------------------------------------------
# Multi-profile accumulation + robustness
# ---------------------------------------------------------------------------

def test_production_accumulates_across_profiles_in_one_tick():
    """watch_conditions and intraday_sentinel report once per profile; the
    engine's production for the tick is the sum, not the last writer's value."""
    hb.begin("watch_conditions")
    hb.note_production(3)   # profile A
    hb.note_production(2)   # profile B
    rec = hb.record_run("watch_conditions")

    assert rec["last_produced"] == 5


def test_reporting_outside_a_run_window_is_a_safe_noop():
    """Calling the engine directly (not via the scheduler) must not crash or
    leak counts into whatever runs next."""
    hb.note_production(5)
    hb.note_skipped("nope")

    rec = _run("watch_conditions", produced=1)

    assert rec["last_produced"] == 1


def test_an_unwritable_store_never_breaks_the_caller(monkeypatch):
    """Observability that can break the engine it observes is worse than none."""
    monkeypatch.setattr(hb, "_HEARTBEAT_PATH", "/nonexistent-root/nope/hb.json")

    rec = hb.record_run("watch_conditions")   # must not raise

    assert isinstance(rec, dict)              # degrades, does not propagate
    assert hb.get_heartbeats() == {}          # nothing persisted, still readable


def test_corrupt_store_is_treated_as_empty_not_fatal(store):
    (store / "engine_heartbeat.json").write_text("{not json")

    assert hb.get_heartbeats() == {}
    _run("watch_conditions", produced=1)
    assert hb.get_heartbeats()["watch_conditions"]["last_produced"] == 1


def test_store_is_valid_json_on_disk(store):
    _run("watch_conditions", produced=2)

    data = json.loads((store / "engine_heartbeat.json").read_text())

    assert data["watch_conditions"]["last_produced"] == 2


# ---------------------------------------------------------------------------
# The OTHER half of the contract: proving an engine still RUNS
# ---------------------------------------------------------------------------
# The original view only asked "is it producing?". An engine that stops being
# invoked altogether keeps its last healthy record forever — `last_status: ok`,
# a timestamp that never moves — so the one view built to find dead engines
# reported the deadest possible engine as fine.

def _write(store_dir, name, ran_minutes_ago=0, **fields):
    """Put a heartbeat record on disk with an explicit age.

    Writing the store directly rather than patching the clock: the staleness
    question is entirely about what a stamp on disk means later, and a fake
    `now` would test the patch instead of the record.
    """
    from datetime import datetime, timedelta
    path = store_dir / "engine_heartbeat.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    ran = datetime.now() - timedelta(minutes=ran_minutes_ago)
    data[name] = {
        "last_ran_at": ran.isoformat(timespec="seconds"),
        "last_status": hb.STATUS_OK,
        "last_detail": "",
        "runs": 1,
        **fields,
    }
    path.write_text(json.dumps(data))


def _registry(monkeypatch, **engines):
    """Pin the scheduler roster this view checks against."""
    monkeypatch.setattr(hb, "_registered_engines", lambda: {
        name: {"cooldown_s": cd, "enabled": True, "circuit_open": False}
        if not isinstance(cd, dict) else cd
        for name, cd in engines.items()
    })


def test_an_engine_that_stopped_running_is_flagged_though_its_last_run_was_ok(store, monkeypatch):
    """The gap this check closes. watch_conditions runs every 30 min; six hours
    of silence means it is not being invoked at all. Its record still says `ok`
    with a zero idle streak — nothing about the produced-side signal can ever
    notice, because the engine is not reaching the producing code."""
    _registry(monkeypatch, watch_conditions=1800)
    _write(store, "watch_conditions", ran_minutes_ago=360, consecutive_idle=0,
           last_produced=4, last_produced_at="2026-07-24T15:37:51")

    health = hb.get_engine_health()

    assert health["engines"]["watch_conditions"]["last_status"] == hb.STATUS_OK
    why = {c["engine"]: c["why"] for c in health["concerning"]}
    assert "watch_conditions" in why
    assert "has not run" in why["watch_conditions"]


def test_an_engine_inside_its_own_cooldown_is_not_stale(store, monkeypatch):
    """The daily ledger scorer legitimately sits idle for ~24h. Staleness scales
    with each engine's own cooldown or the view cries wolf on every slow job."""
    _registry(monkeypatch, score_recommendations=86400)
    _write(store, "score_recommendations", ran_minutes_ago=20 * 60)

    assert hb.get_engine_health()["concerning"] == []


def test_a_paused_circuit_breaker_is_surfaced_by_name(store, monkeypatch):
    """The scheduler's breaker pauses a task for the day after 3 consecutive
    failures — deliberately freezing it in exactly the state above. One alert
    fires on the trip; if it is missed, this view must not then call the paused
    engine healthy."""
    monkeypatch.setattr(hb, "_registered_engines", lambda: {
        "priority_precompute": {"cooldown_s": 300, "enabled": True, "circuit_open": True},
    })
    _write(store, "priority_precompute", ran_minutes_ago=1)

    why = {c["engine"]: c["why"] for c in hb.get_engine_health()["concerning"]}

    assert "circuit breaker OPEN" in why["priority_precompute"]


def test_a_registered_engine_that_never_ran_is_visible_as_an_absence(store, monkeypatch):
    """An engine missing from the store is the most complete failure there is,
    and it was the one shape the view could not see: it iterated the records it
    had, so an engine that never started simply was not in the loop."""
    _registry(monkeypatch, watch_conditions=1800, edgar_events=1800)
    _write(store, "watch_conditions", ran_minutes_ago=2 * 24 * 60)  # store is 2 days old

    why = {c["engine"]: c["why"] for c in hb.get_engine_health()["concerning"]}

    assert "NEVER reported a run" in why["edgar_events"]


def test_a_fresh_store_does_not_flag_every_engine_at_once(store, monkeypatch):
    """The cries-wolf guard, and this module shipped without one once already.
    On a first deploy every engine is unseen; flagging ten at once would make
    the view noise on the exact day someone is watching it."""
    _registry(monkeypatch, watch_conditions=1800, edgar_events=1800, exchange_rate=3600)
    _write(store, "watch_conditions", ran_minutes_ago=0)

    assert hb.get_engine_health()["concerning"] == []


def test_an_engine_disabled_in_config_is_never_flagged(store, monkeypatch):
    """Turned off by instruction is not dead — the same reasoning that keeps a
    `skipped` run from accruing an idle streak."""
    monkeypatch.setattr(hb, "_registered_engines", lambda: {
        "funnel_signal_scan": {"cooldown_s": 1800, "enabled": False, "circuit_open": False},
    })
    _write(store, "funnel_signal_scan", ran_minutes_ago=5 * 24 * 60)

    assert hb.get_engine_health()["concerning"] == []


def test_the_view_degrades_to_heartbeats_when_no_scheduler_is_present(store, monkeypatch):
    """An observability layer that hard-depends on the thing it observes is not
    one. With no roster available the produced-side signal must still work."""
    monkeypatch.setattr(hb, "_registered_engines", lambda: {})
    for _ in range(12):
        _run("watch_conditions", produced=0)

    health = hb.get_engine_health(idle_threshold=10)

    assert health["registry"] == {}
    assert any(c["engine"] == "watch_conditions" for c in health["concerning"])


def test_the_roster_comes_from_the_real_scheduler_registry():
    """Driven through the REAL registry, not a fixture of it: if a task is added
    to SCHEDULED_TASKS it is covered by liveness automatically, and if this seam
    ever stops resolving, the checks above silently become no-ops."""
    import tools.scheduler as sched

    registry = hb._registered_engines()

    assert {name for name, *_ in sched.SCHEDULED_TASKS} == set(registry)
    assert registry["score_recommendations"]["cooldown_s"] == 86400
    assert registry["watch_conditions"]["enabled"] is True


# ---------------------------------------------------------------------------
# The scheduler seam — driven through the REAL runner, not a mock of it
# ---------------------------------------------------------------------------
# 3.3 shipped dead with 56 green tests because every one of them bypassed the
# production path and exercised a seam the tests themselves invented. So this
# drives CairnIQScheduler._try_run for real: if the hook is ever dropped from the
# runner, or a task's reporting stops propagating across the asyncio.to_thread /
# run_under_profile hops, these fail.

def test_the_real_scheduler_runner_records_a_heartbeat(monkeypatch):
    import asyncio

    import tools.scheduler as sched

    monkeypatch.setattr(sched, "get_scheduler_settings", lambda: {"probe": True})
    monkeypatch.setattr(sched, "_can_run", lambda name, cooldown: True)
    monkeypatch.setattr(sched, "_record_run", lambda name: None)

    async def probe():
        hb.note_production(7)

    asyncio.run(sched.scheduler._try_run("probe", probe, 0, 10))

    rec = hb.get_heartbeats()["probe"]
    assert rec["last_status"] == hb.STATUS_OK
    assert rec["last_produced"] == 7
    assert rec["consecutive_idle"] == 0


def test_the_real_scheduler_runner_records_a_crash_as_an_error(monkeypatch):
    """A task that raises is swallowed by the runner by design (one bad task must
    not stop the loop) — so the heartbeat is the only place it stays visible."""
    import asyncio

    import tools.scheduler as sched

    monkeypatch.setattr(sched, "get_scheduler_settings", lambda: {"probe": True})
    monkeypatch.setattr(sched, "_can_run", lambda name, cooldown: True)

    async def boom():
        raise RuntimeError("universe read failed")

    asyncio.run(sched.scheduler._try_run("probe", boom, 0, 10))

    rec = hb.get_heartbeats()["probe"]
    assert rec["last_status"] == hb.STATUS_ERROR
    assert "universe read failed" in rec["last_detail"]
    assert any(c["engine"] == "probe" for c in hb.get_engine_health()["concerning"])


def test_market_hours_gate_reports_a_skip_not_a_dead_engine(monkeypatch):
    """The overnight case, end to end through the real task: outside market hours
    both 3.3 and 3.4 early-return, and that must read as 'skipped', never as an
    engine producing nothing."""
    import asyncio
    from datetime import datetime

    import tools.scheduler as sched

    monkeypatch.setattr(sched, "get_scheduler_settings", lambda: {"watch_conditions": True})
    monkeypatch.setattr(sched, "_can_run", lambda name, cooldown: True)
    monkeypatch.setattr(sched, "_record_run", lambda name: None)
    # A Saturday — outside any trading session.
    monkeypatch.setattr(sched, "_eastern_now", lambda: datetime(2026, 7, 25, 12, 0))

    asyncio.run(sched.scheduler._try_run(
        "watch_conditions", sched.task_watch_conditions, 0, 10))

    rec = hb.get_heartbeats()["watch_conditions"]
    assert rec["last_status"] == hb.STATUS_SKIPPED
    assert rec.get("consecutive_idle", 0) == 0
    assert hb.get_engine_health()["concerning"] == []


# ---------------------------------------------------------------------------
# The deployment ladder's liveness line (Roadmap 3.9)
# ---------------------------------------------------------------------------
#
# 3.9 ships inert — the playbook is empty on every profile — which is exactly
# the case this module exists for. These pin the one thing that makes an inert
# ladder honest: it reports, with an explicit 0, and it never reports a count it
# did not actually measure.

def _ladder_detail(ladders):
    from tools.scheduler import _report_deployment_ladder

    hb.begin("intraday_sentinel")
    hb.note_production(7)  # the real production count: holdings checked
    _report_deployment_ladder(ladders)
    return hb.record_run("intraday_sentinel")


def test_an_empty_ladder_reports_inert_rather_than_reporting_nothing(store):
    rec = _ladder_detail([
        {"armed": 0, "fired": 0, "seeded": 0, "levels": 0, "evaluated": True},
        {"armed": 0, "fired": 0, "seeded": 0, "levels": 0, "evaluated": True},
    ])

    assert "INERT" in rec["last_detail"]
    assert "0 rungs on file across 2 profile(s)" in rec["last_detail"]


def test_reporting_the_ladder_does_not_inflate_the_production_count(store):
    """Production stays holdings-checked — the number that proves the scan ran.
    Folding rung counts into it would let an armed ladder disguise a dead scan."""
    rec = _ladder_detail([{"armed": 3, "fired": 0, "seeded": 0, "levels": 3, "evaluated": True}])

    assert rec["last_produced"] == 7


def test_a_ladder_that_was_never_looked_at_is_not_reported_as_empty(store):
    """The distinction that stops a fabricated liveness number: no fresh SPY
    reading means unmeasured, which must not be printed as '0 rungs armed'."""
    rec = _ladder_detail([{"armed": 0, "fired": 0, "seeded": 0, "levels": 0, "evaluated": False}])

    assert "not evaluated" in rec["last_detail"]
    assert "INERT" not in rec["last_detail"]


def test_an_armed_ladder_reports_armed_and_fired_counts(store):
    rec = _ladder_detail([
        {"armed": 2, "fired": 1, "seeded": 0, "levels": 3, "evaluated": True},
        {"armed": 1, "fired": 0, "seeded": 1, "levels": 2, "evaluated": True},
    ])

    assert "3/5 rungs armed, 1 fired" in rec["last_detail"]
    assert "1 already past when first armed" in rec["last_detail"]
