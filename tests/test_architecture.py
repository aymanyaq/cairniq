import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

import tools.alpaca as alpaca_module
import tools.questrade as questrade_module
from tools.daily_cache import CACHE_DIR
from tools.user_profile import get_active_profile, get_data_path, reset_profile, set_active_profile


@pytest.fixture(autouse=True)
def mock_scheduler_enabled(monkeypatch):
    import tools.scheduler as sched
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: True)


def test_data_path_resolution_default_profile():
    """Ensure the default profile targets the user_data/ directory."""
    set_active_profile("default")
    try:
        path = get_data_path("test.csv")
        assert path.startswith(os.path.join(PROJECT_ROOT, "user_data")), f"Path leaked: {path}"
        assert path.endswith("test.csv")
    finally:
        # Reset is manual or rely on contextvar
        pass

def test_data_path_resolution_custom_profile():
    """Ensure custom profiles target user_data/profiles/<name>."""
    set_active_profile("test_user_123")
    try:
        path = get_data_path("portfolio.csv")
        expected_dir = os.path.join(PROJECT_ROOT, "user_data", "profiles", "test_user_123")
        assert path.startswith(expected_dir), f"Path leaked: {path}"
        assert path.endswith("portfolio.csv")
    finally:
        pass

def test_data_path_rejects_profile_traversal():
    """Ensure unsafe profile names cannot escape user_data/."""
    token = set_active_profile("../../../tmp/pwn")
    try:
        path = get_data_path("portfolio.csv")
        assert path.startswith(os.path.join(PROJECT_ROOT, "user_data")), f"Path leaked: {path}"
        assert ".." not in os.path.relpath(path, PROJECT_ROOT)
    finally:
        reset_profile(token)

def test_daily_cache_directory():
    """Ensure the daily cache directory is safely inside user_data/."""
    assert "user_data" in CACHE_DIR, f"Daily cache directory leaked: {CACHE_DIR}"
    assert "daily_cache" in CACHE_DIR

@pytest.mark.real_trade_journal_path
def test_runtime_state_paths_stay_in_user_data():
    """Ensure runtime history files are profile-scoped under user_data/.

    Marked so the autouse `isolate_trade_journal` fixture leaves `_journal_file`
    alone. That fixture redirects the journal to a tmp path for every other test,
    and from inside this process a redirected resolver is indistinguishable from
    the leaked resolver this test exists to catch — asserting against the patched
    attribute would check the fixture, not the code. The marker costs no isolation:
    the fixture blocks the journal's writers for marked tests instead.
    """
    from tools.market_sentinel import _history_path
    from tools.trade_journal import _journal_file

    history_path = _history_path()
    journal_path = _journal_file()

    assert history_path.startswith(os.path.join(PROJECT_ROOT, "user_data")), f"Path leaked: {history_path}"
    assert journal_path.startswith(os.path.join(PROJECT_ROOT, "user_data")), f"Path leaked: {journal_path}"

def test_broker_env_isolation():
    """Ensure broker tools load environment variables from user_data/.env."""
    # Check Alpaca
    with open(alpaca_module.__file__) as f:
        alpaca_code = f.read()
    assert '"user_data", ".env"' in alpaca_code, "Alpaca missing user_data env path."

    # Check Questrade
    with open(questrade_module.__file__) as f:
        questrade_code = f.read()
    assert '"user_data", ".env"' in questrade_code, "Questrade missing user_data env path."

def test_profile_contextvar_does_not_cross_bare_thread():
    """Document the hazard: a manually-spawned thread does NOT inherit the
    request-scoped profile ContextVar — it falls back to 'default'. This is the
    exact mechanism that leaked one user's portfolio into another's analysis."""
    import threading

    set_active_profile("alice")
    seen = {}

    def worker():
        seen["profile"] = get_active_profile()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # The bare thread does NOT see 'alice' — it falls back (to 'default' or the
    # ACTIVE_PROFILE env). This is the leak vector the wrapper below closes.
    assert seen["profile"] != "alice"


def test_run_under_profile_preserves_profile_across_thread():
    """Regression for the cross-user portfolio leak: background/agent workers run
    in bare threads, so the captured profile must be re-applied inside the thread.
    _run_under_profile is the wrapper that guarantees this invariant."""
    import threading

    from api.background import _run_under_profile

    set_active_profile("alice")
    captured = get_active_profile()  # what a request handler would capture
    seen = {}

    def worker():
        seen["profile"] = get_active_profile()
        seen["portfolio_path"] = get_data_path("my_portfolio.csv")

    t = threading.Thread(target=_run_under_profile, args=(captured, worker))
    t.start()
    t.join()

    assert seen["profile"] == "alice"
    # And the per-profile portfolio path must point at alice's dir, never the
    # shared user_data/my_portfolio.csv (the 'default' file that held the leak).
    expected_dir = os.path.join(PROJECT_ROOT, "user_data", "profiles", "alice")
    assert seen["portfolio_path"].startswith(expected_dir), seen["portfolio_path"]


def test_multiuser_guard_blocks_env_profile_fallback():
    """Core cross-user leak regression: a worker that lost the profile ContextVar
    must NOT fall back to the process-global ACTIVE_PROFILE env var (shared across
    household members). With the multi-user guard on, it resolves to the isolated,
    empty UNBOUND_PROFILE instead of leaking whichever profile last wrote the env
    var — exactly the cross-user verification/risk mixup."""
    import threading

    from tools.user_profile import (
        UNBOUND_PROFILE,
        enable_multiuser_guard,
        is_multiuser_guard_enabled,
    )

    prev_guard = is_multiuser_guard_enabled()
    prev_env = os.environ.get("ACTIVE_PROFILE")
    os.environ["ACTIVE_PROFILE"] = "bob"  # simulate another user's global state
    seen = {}

    def worker():
        seen["profile"] = get_active_profile()

    try:
        # Without the guard, a bare worker leaks the env profile (documents the bug).
        enable_multiuser_guard(False)
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen["profile"] == "bob"

        # With the guard, the same worker refuses the env fallback.
        enable_multiuser_guard(True)
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen["profile"] == UNBOUND_PROFILE
        assert seen["profile"] != "bob"
    finally:
        enable_multiuser_guard(prev_guard)
        if prev_env is None:
            os.environ.pop("ACTIVE_PROFILE", None)
        else:
            os.environ["ACTIVE_PROFILE"] = prev_env


def test_unbound_profile_fallback_lands_in_the_component_log():
    """The guard's warning has to survive root-logger reconfiguration.

    It did not: the warning went to logging.getLogger("tools.user_profile"), whose
    NullHandler counts as "handled", so logging never fell through to lastResort on
    stderr — and nothing in this app configures root. The guard fired in production
    from 2026-06-28 onward and left zero lines in cairniq.stderr.log and zero under
    logs/, which is why one process holding every profile's checkpoint store went
    unnoticed. Routing it through log_to_component() puts it on a channel with its
    own FileHandler and propagate=False, so root can be muted (as it is below) and
    the record still reaches logs/tools/tools.jsonl.
    """
    import logging
    import threading

    import tools.user_profile as up
    from agent.logger import get_component_logger
    from tools.user_profile import (
        UNBOUND_PROFILE,
        enable_multiuser_guard,
        is_multiuser_guard_enabled,
    )

    component_logger = get_component_logger("tools")
    captured = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = Capture()
    component_logger.addHandler(handler)

    root = logging.getLogger()
    prev_root_handlers = list(root.handlers)
    prev_root_level = root.level
    prev_guard = is_multiuser_guard_enabled()
    up._unbound_warn_counts.clear()
    seen = {}

    def worker():  # a bare thread never inherits the profile ContextVar
        seen["profile"] = get_active_profile()

    try:
        # Reproduce the swallow: no root handlers, root muted outright.
        root.handlers.clear()
        root.setLevel(logging.CRITICAL)

        enable_multiuser_guard(True)
        t = threading.Thread(target=worker, name="unbound-probe")
        t.start()
        t.join()
        assert seen["profile"] == UNBOUND_PROFILE
    finally:
        enable_multiuser_guard(prev_guard)
        component_logger.removeHandler(handler)
        root.handlers[:] = prev_root_handlers
        root.setLevel(prev_root_level)
        up._unbound_warn_counts.clear()

    assert len(captured) == 1, f"expected exactly one warning, got {len(captured)}"
    record = captured[0]
    assert record.levelno == logging.WARNING
    assert record.component == "tools"
    assert record.phase == "Profile"
    # The historical grep string operators already search for.
    assert "Active-profile ContextVar missing" in record.getMessage()

    # "Logs loudly so the offending worker boundary is diagnosable" — the record has
    # to name the boundary, not merely report that one exists.
    data = record.data
    assert data["resolved_profile"] == UNBOUND_PROFILE
    assert data["thread"] == "unbound-probe"
    assert "test_architecture.py" in data["caller"] and "worker" in data["caller"]
    assert any("test_architecture.py" in frame for frame in data["stack"])

    # The channel it landed on writes to a real file and cannot be reconfigured
    # away by uvicorn/litellm — the two properties the old path lacked.
    assert component_logger.propagate is False
    file_handlers = [h for h in component_logger.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers, "component logger has no FileHandler"
    assert file_handlers[0].baseFilename.endswith(os.path.join("logs", "tools", "tools.jsonl"))


def test_unbound_profile_fallback_is_throttled_but_never_muted():
    """One lost binding takes the fallback on every get_active_profile() call in
    that worker, so an unthrottled warning would bury the tools channel. Only the
    1st/10th/100th... occurrence per call site is written, and each record carries
    the running count — bounded, but never silent (silence is the bug being fixed).
    """
    import logging

    import tools.user_profile as up
    from agent.logger import get_component_logger
    from tools.user_profile import enable_multiuser_guard, is_multiuser_guard_enabled

    component_logger = get_component_logger("tools")
    captured = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    handler = Capture()
    component_logger.addHandler(handler)

    prev_guard = is_multiuser_guard_enabled()
    token = up._profile_ctx.set(None)  # simulate the lost binding in-thread
    up._unbound_warn_counts.clear()
    try:
        enable_multiuser_guard(True)
        for _ in range(100):
            up.get_active_profile()
    finally:
        enable_multiuser_guard(prev_guard)
        up._profile_ctx.reset(token)
        component_logger.removeHandler(handler)
        up._unbound_warn_counts.clear()

    counts = [r.data["occurrences_at_caller"] for r in captured]
    assert counts == [1, 10, 100], counts
    """Even with the guard on, a worker explicitly wrapped with run_under_profile
    resolves the captured profile. This is how the deep_reasoning executors and
    background workers stay correct instead of degrading to UNBOUND_PROFILE."""
    import threading

    from tools.user_profile import (
        enable_multiuser_guard,
        is_multiuser_guard_enabled,
        run_under_profile,
    )

    prev_guard = is_multiuser_guard_enabled()
    seen = {}

    def worker():
        seen["profile"] = get_active_profile()

    try:
        enable_multiuser_guard(True)
        t = threading.Thread(target=run_under_profile, args=("alice", worker))
        t.start()
        t.join()
        assert seen["profile"] == "alice"
    finally:
        enable_multiuser_guard(prev_guard)


def test_get_st_aware_func_preserves_profile_in_threadpool():
    """ThreadPoolExecutor workers do not inherit ContextVars; get_st_aware_func
    re-binds the scheduling thread's profile so portfolio tools submitted to the
    pool (risk metrics, correlation, FX) resolve the correct profile rather than
    the guard fallback. The guard is enabled so a regression would surface as the
    UNBOUND sentinel, not a coincidental 'default'."""
    from concurrent.futures import ThreadPoolExecutor

    from agent.utils import get_st_aware_func
    from tools.user_profile import enable_multiuser_guard, is_multiuser_guard_enabled

    prev_guard = is_multiuser_guard_enabled()
    token = set_active_profile("alice")

    def read_profile():
        return get_active_profile()

    try:
        enable_multiuser_guard(True)
        with ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(get_st_aware_func(read_profile)).result(timeout=10)
        assert result == "alice"
    finally:
        enable_multiuser_guard(prev_guard)
        reset_profile(token)


def test_resolve_risk_symbols_rejects_unheld_tickers(monkeypatch):
    """Risk tools must not compute on tickers the user does not hold. When none of
    the requested symbols are in the verified holdings (a leaked profile's set or
    fabricated tickers), _resolve_risk_symbols falls back to the real portfolio
    and records a reconciliation note."""
    import agent.tool_registry as tr
    import tools.portfolio_csv as pcsv

    monkeypatch.setattr(
        pcsv, "get_portfolio_decision_context",
        lambda *a, **k: {"owned_symbols": ["AAPL", "MSFT"]},
    )
    monkeypatch.setattr(pcsv, "get_tradeable_symbols", lambda: ["AAPL", "MSFT"])

    # None of the requested tickers are held -> fall back to the actual portfolio.
    syms, meta = tr._resolve_risk_symbols("XIC.TO, RY.TO, ZSP.TO")
    assert syms == ["AAPL", "MSFT"]
    assert meta["note"] and "verified holdings" in meta["note"]

    # Empty request -> portfolio scope.
    syms_empty, meta_empty = tr._resolve_risk_symbols("")
    assert syms_empty == ["AAPL", "MSFT"]
    assert meta_empty["scope"] == "portfolio"

    # Mixed -> keep the supplied set but flag the unheld ticker.
    syms_mixed, meta_mixed = tr._resolve_risk_symbols("AAPL, TSLA")
    assert syms_mixed == ["AAPL", "TSLA"]
    assert meta_mixed["not_held"] == ["TSLA"]


def test_graph_memory_no_cross_profile_contamination():
    """Regression: the GraphMemory singleton holds one in-memory graph shared
    across requests. Concurrent writes from two profiles must not bleed into each
    other's graph — the @_synchronized lock makes each reload->mutate->save
    atomic per the active profile. Uses placeholder profiles + synthetic tickers
    only (no real account data)."""
    import shutil
    import threading

    from tools.graph_memory import graph_memory

    PA, PB = "kgtest_alpha", "kgtest_beta"
    A = [f"AAA{i}" for i in range(20)]
    B = [f"BBB{i}" for i in range(20)]

    def worker(profile, tickers):
        set_active_profile(profile)
        for t in tickers:
            graph_memory.add_entity(t, "Stock", {"owned": True})
            graph_memory.add_relationship("Portfolio", t, "OWNS")

    ta = threading.Thread(target=worker, args=(PA, A))
    tb = threading.Thread(target=worker, args=(PB, B))
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    def owns(profile):
        set_active_profile(profile)
        graph_memory._ensure_profile_sync()
        return {t for _, t, d in graph_memory.graph.edges(data=True) if d.get("relation") == "OWNS"}

    try:
        a_owns, b_owns = owns(PA), owns(PB)
        assert set(A) <= a_owns, f"alpha lost its own writes: {set(A) - a_owns}"
        assert not (set(B) & a_owns), f"alpha contaminated with beta: {set(B) & a_owns}"
        assert set(B) <= b_owns, f"beta lost its own writes: {set(B) - b_owns}"
        assert not (set(A) & b_owns), f"beta contaminated with alpha: {set(A) & b_owns}"
    finally:
        for p in (PA, PB):
            shutil.rmtree(os.path.join(PROJECT_ROOT, "user_data", "profiles", p), ignore_errors=True)


def test_legacy_data_folder_absence():
    """Ensure no tools have accidentally recreated the root data/ folder."""
    legacy_data_path = os.path.join(PROJECT_ROOT, "data")
    if os.path.exists(legacy_data_path):
        # We allow it to exist ONLY if it's completely empty (sometimes created by git or OS)
        if os.path.isdir(legacy_data_path):
            contents = os.listdir(legacy_data_path)
            assert len(contents) == 0, f"Legacy data/ folder contains files! Migration failed. Contents: {contents}"
        else:
            pytest.fail("Legacy data path exists and is not a directory.")


def test_opportunity_scanner_parallel_helpers_rebind_profile(monkeypatch):
    """Regression: the opportunity scanner submits per-symbol work to
    ThreadPoolExecutors whose workers do NOT inherit the request ContextVar.
    Each parallel helper must re-bind the active profile (run_under_profile) so
    the per-profile daily cache and any profile-scoped reads resolve the caller's
    profile — not the empty '_unbound' sentinel under the multi-user guard."""
    import tools.opportunity_scanner as osc
    import tools.screener as screener
    from tools.user_profile import (
        UNBOUND_PROFILE,
        enable_multiuser_guard,
        is_multiuser_guard_enabled,
    )

    seen: dict[str, str] = {}

    def _rec(kind):
        def _fn(symbol, *a, **k):
            seen[kind] = get_active_profile()
            return {}
        return _fn

    monkeypatch.setattr(osc, "_headwind_check", _rec("headwind"))
    monkeypatch.setattr(osc, "_flow_confirmation_for_symbol", _rec("flow"))
    monkeypatch.setattr(screener, "check_setup", _rec("setup"))

    prev_guard = is_multiuser_guard_enabled()
    try:
        enable_multiuser_guard(True)
        set_active_profile("alice")
        osc._headwind_check_parallel(["AAPL"], max_workers=1)
        osc._flow_confirmation_parallel(["AAPL"], max_workers=1)
        osc._setup_check_parallel(["AAPL"], max_workers=1)
        osc._warm_cache_parallel(_rec("warm"), ["AAPL"], max_workers=1, overall_budget=5.0)
    finally:
        enable_multiuser_guard(prev_guard)

    assert seen == {"headwind": "alice", "flow": "alice", "setup": "alice", "warm": "alice"}, seen
    assert UNBOUND_PROFILE not in seen.values()


def test_scheduler_portfolio_snapshot_binds_each_profile(monkeypatch):
    """Regression: the background scheduler runs with no request profile bound, so
    task_portfolio_snapshot must re-bind and snapshot each real profile rather than
    resolving to the empty '_unbound' profile (which silently snapshots nothing).
    Transient pytest_* and the '_unbound' sentinel profile must be skipped."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import tools.portfolio_tracker as tracker
    import tools.scheduler as sched
    import tools.user_profile as up
    from tools.user_profile import (
        UNBOUND_PROFILE,
        enable_multiuser_guard,
        is_multiuser_guard_enabled,
    )

    fake_profiles = [
        {"name": "default"}, {"name": "alice"},
        {"name": "pytest_tmp"}, {"name": UNBOUND_PROFILE},
    ]
    snapshotted: list[str] = []

    def _fake_snapshot(force=False):
        snapshotted.append(get_active_profile())
        return "ok"

    # Thursday, well after the 16:15 ET close-time gate.
    after_close = datetime(2026, 7, 9, 16, 30, tzinfo=ZoneInfo("US/Eastern"))

    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(tracker, "snapshot_portfolio", _fake_snapshot)
    monkeypatch.setattr(sched, "_eastern_now", lambda: after_close)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: False)
    monkeypatch.setattr(sched, "_mark_done_today", lambda key: None)

    prev_guard = is_multiuser_guard_enabled()
    try:
        enable_multiuser_guard(True)
        asyncio.run(sched.task_portfolio_snapshot())
    finally:
        enable_multiuser_guard(prev_guard)

    assert set(snapshotted) == {"default", "alice"}, snapshotted
    assert UNBOUND_PROFILE not in snapshotted
    assert not any(p.startswith("pytest_") for p in snapshotted)


def test_scheduler_portfolio_snapshot_skips_before_market_close(monkeypatch):
    """task_portfolio_snapshot must be a no-op before market close (and on
    weekends) — a "close-of-day" snapshot must not fire at, say, 2pm ET, and
    must not fire again the same day once already recorded."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import tools.portfolio_tracker as tracker
    import tools.scheduler as sched
    import tools.user_profile as up

    fake_profiles = [{"name": "default"}]
    snapshotted: list[str] = []
    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(tracker, "snapshot_portfolio", lambda force=False: snapshotted.append(1) or "ok")

    # Thursday mid-afternoon — before the close-time gate.
    mid_afternoon = datetime(2026, 7, 9, 14, 0, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: mid_afternoon)
    asyncio.run(sched.task_portfolio_snapshot())
    assert snapshotted == [], "must not snapshot before market close"

    # Saturday evening — weekday gate should also block it.
    saturday_evening = datetime(2026, 7, 11, 18, 0, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: saturday_evening)
    asyncio.run(sched.task_portfolio_snapshot())
    assert snapshotted == [], "must not snapshot on a non-trading day"

    # After close, but already marked done today — must not re-snapshot.
    after_close = datetime(2026, 7, 9, 16, 30, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: after_close)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: True)
    asyncio.run(sched.task_portfolio_snapshot())
    assert snapshotted == [], "must not re-snapshot once already done today"


def test_scheduler_score_recommendations_binds_each_profile(monkeypatch):
    """Regression: the advice-outcome scorer (Theme 1.1 sub-part c) must run per
    real profile like the snapshot job, not resolve to the empty '_unbound' profile.
    Transient pytest_* and the '_unbound' sentinel profile must be skipped."""
    import asyncio

    import tools.memory as mem
    import tools.scheduler as sched
    import tools.user_profile as up
    from tools.user_profile import (
        UNBOUND_PROFILE,
        enable_multiuser_guard,
        is_multiuser_guard_enabled,
    )

    fake_profiles = [
        {"name": "default"}, {"name": "alice"},
        {"name": "pytest_tmp"}, {"name": UNBOUND_PROFILE},
    ]
    scored: list[str] = []
    saved: list[str] = []

    def _fake_load_memory():
        scored.append(get_active_profile())
        return {"past_recommendations": [{"ticker": "AAPL", "date": "2020-01-01", "scores": {}}]}

    def _fake_score(memory):
        return True  # pretend scoring found something to update

    def _fake_save_memory(memory):
        saved.append(get_active_profile())
        return True

    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(mem, "load_memory", _fake_load_memory)
    monkeypatch.setattr(mem, "score_past_recommendations", _fake_score)
    monkeypatch.setattr(mem, "save_memory", _fake_save_memory)

    prev_guard = is_multiuser_guard_enabled()
    try:
        enable_multiuser_guard(True)
        asyncio.run(sched.task_score_recommendations())
    finally:
        enable_multiuser_guard(prev_guard)

    assert set(scored) == {"default", "alice"}, scored
    assert set(saved) == {"default", "alice"}, saved
    assert UNBOUND_PROFILE not in scored
    assert not any(p.startswith("pytest_") for p in scored)


def test_scheduler_premarket_pulse_binds_each_profile(monkeypatch):
    """Regression: task_premarket_pulse must re-bind and run per real profile
    (same pattern as the other scheduled jobs), only inside the pre-market
    window, and only once per trading day per profile."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import api.background as background
    import tools.scheduler as sched
    import tools.user_profile as up
    from tools.user_profile import UNBOUND_PROFILE

    fake_profiles = [
        {"name": "default"}, {"name": "alice"},
        {"name": "pytest_tmp"}, {"name": UNBOUND_PROFILE},
    ]
    ran: list[str] = []

    def _fake_news(force=False):
        ran.append(get_active_profile())

    # Thursday, inside the 6:00-9:25 ET pre-market window.
    premarket = datetime(2026, 7, 9, 7, 30, tzinfo=ZoneInfo("US/Eastern"))

    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(background, "run_news_agent_in_background", _fake_news)
    monkeypatch.setattr(sched, "_eastern_now", lambda: premarket)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: False)
    monkeypatch.setattr(sched, "_mark_done_today", lambda key: None)
    # Focus on profile binding, not the LLM-readiness gate (no key in test env).
    monkeypatch.setattr(sched, "_skip_if_llm_unready", lambda task_name: False)

    asyncio.run(sched.task_premarket_pulse())

    assert set(ran) == {"default", "alice"}, ran
    assert UNBOUND_PROFILE not in ran
    assert not any(p.startswith("pytest_") for p in ran)


def test_scheduler_premarket_pulse_skips_outside_window(monkeypatch):
    """task_premarket_pulse must be a no-op outside the pre-market window and
    once already run today."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import api.background as background
    import tools.scheduler as sched
    import tools.user_profile as up

    fake_profiles = [{"name": "default"}]
    ran: list[str] = []
    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(background, "run_news_agent_in_background", lambda force=False: ran.append(1))

    # Thursday midday — after the pre-market window closes.
    midday = datetime(2026, 7, 9, 11, 0, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: midday)
    asyncio.run(sched.task_premarket_pulse())
    assert ran == [], "must not run outside the pre-market window"

    # Inside the window, but already marked done today.
    premarket = datetime(2026, 7, 9, 7, 30, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: premarket)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: True)
    asyncio.run(sched.task_premarket_pulse())
    assert ran == [], "must not re-run once already done today"


def test_scheduler_priority_precompute_binds_each_profile(monkeypatch):
    """task_priority_precompute (Theme 3.1) must re-bind and run per real profile
    inside its 7:00-9:25 ET window, marking done-today only on a successful run,
    and must be registered + config-gatable like every other job."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import api.background as background
    import tools.scheduler as sched
    import tools.user_profile as up
    from tools.user_profile import UNBOUND_PROFILE

    # Registration + config-gating surface.
    assert "priority_precompute" in sched.DEFAULT_SCHEDULER_SETTINGS
    assert "priority_precompute" in {name for name, *_ in sched.SCHEDULED_TASKS}

    fake_profiles = [
        {"name": "default"}, {"name": "alice"},
        {"name": "pytest_tmp"}, {"name": UNBOUND_PROFILE},
    ]
    ran: list[str] = []
    marked: list[str] = []

    def _fake_run():
        ran.append(get_active_profile())
        return get_active_profile() != "alice"  # alice's run fails

    # Thursday, inside the 7:00-9:25 ET priority window.
    premarket = datetime(2026, 7, 9, 7, 30, tzinfo=ZoneInfo("US/Eastern"))

    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(background, "run_priority_precompute_in_background", _fake_run)
    monkeypatch.setattr(sched, "get_priority_precompute_profiles", lambda config_path=None: None)
    monkeypatch.setattr(sched, "_eastern_now", lambda: premarket)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: False)
    monkeypatch.setattr(sched, "_mark_done_today", lambda key: marked.append(get_active_profile()))
    # Focus on profile binding, not the LLM-readiness gate (no key in test env).
    monkeypatch.setattr(sched, "_skip_if_llm_unready", lambda task_name: False)

    asyncio.run(sched.task_priority_precompute())

    assert set(ran) == {"default", "alice"}, ran
    assert UNBOUND_PROFILE not in ran
    assert not any(p.startswith("pytest_") for p in ran)
    # A failed run must NOT burn the daily marker — the next tick retries it.
    assert marked == ["default"], marked


def test_scheduler_priority_precompute_skips_outside_window(monkeypatch):
    """task_priority_precompute must be a no-op before its 7:00 ET start (even
    while the broader pre-market pulse window is already open), outside the
    window entirely, and once already done today."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import api.background as background
    import tools.scheduler as sched
    import tools.user_profile as up

    fake_profiles = [{"name": "default"}]
    ran: list[str] = []
    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(background, "run_priority_precompute_in_background", lambda: ran.append(1) or True)

    # Thursday 6:30 ET — pulse window is open, priority window is not yet.
    early = datetime(2026, 7, 9, 6, 30, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: early)
    asyncio.run(sched.task_priority_precompute())
    assert ran == [], "must not run before the 7:00 ET start (pulse/news need a head start)"

    # Thursday midday — after the window closes.
    midday = datetime(2026, 7, 9, 11, 0, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: midday)
    asyncio.run(sched.task_priority_precompute())
    assert ran == [], "must not run outside the pre-market window"

    # Inside the window, but already marked done today.
    premarket = datetime(2026, 7, 9, 7, 30, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: premarket)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: True)
    asyncio.run(sched.task_priority_precompute())
    assert ran == [], "must not re-run once already done today"


def test_scheduler_config_gating_skips_disabled_job(monkeypatch):
    """A job disabled via funnel_config.json's `scheduler` block must be
    skipped by the tick loop without touching its cooldown or task body."""
    import asyncio

    import tools.scheduler as sched

    called = {"count": 0}

    async def _fake_task():
        called["count"] += 1

    monkeypatch.setattr(sched, "get_scheduler_settings", lambda: {"cache_cleanup": False})
    monkeypatch.setattr(sched, "_can_run", lambda name, cooldown: True)

    instance = sched.CairnIQScheduler()
    asyncio.run(instance._try_run("cache_cleanup", _fake_task, cooldown=1, timeout=5))

    assert called["count"] == 0, "disabled job must not execute"


def test_scheduler_funnel_signal_scan_neutral_global_run(monkeypatch):
    """task_funnel_signal_scan (the M5 signal-log producer) must run ONCE
    globally under 'default' — not per profile — with portfolio_context=None
    (the shared signal log must record the funnel's market selection, never
    one profile's M4 portfolio-fit overlay), inside an isolated RunContext
    (so a stale global chat-cancel can't abort it), and only mark the daily
    marker after a fully assembled result."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import tools.opportunity_scanner as scanner
    import tools.scheduler as sched
    import tools.user_profile as up
    from agent.utils import get_run_context
    from tools.user_profile import UNBOUND_PROFILE

    fake_profiles = [
        {"name": "default"}, {"name": "alice"},
        {"name": "pytest_tmp"}, {"name": UNBOUND_PROFILE},
    ]
    calls: list[dict] = []
    marked: list[str] = []

    def _fake_scan(sector, portfolio_context=None, deadline=None):
        calls.append({
            "sector": sector,
            "profile": get_active_profile(),
            "portfolio_context": portfolio_context,
            "run_context": get_run_context(),
        })
        return {"top_picks": [{"symbol": "MU"}], "diagnostics": {}, "summary": "ok"}

    # Thursday, after the 16:15 ET close gate.
    after_close = datetime(2026, 7, 9, 16, 30, tzinfo=ZoneInfo("US/Eastern"))

    monkeypatch.setattr(up, "list_available_profiles", lambda: fake_profiles)
    monkeypatch.setattr(scanner, "_scan_impl", _fake_scan)
    monkeypatch.setattr(sched, "_eastern_now", lambda: after_close)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: False)
    monkeypatch.setattr(sched, "_mark_done_today", lambda key: marked.append(key))

    asyncio.run(sched.task_funnel_signal_scan())

    assert len(calls) == 1, f"must run exactly once globally, got {calls}"
    call = calls[0]
    assert call["sector"] == "All"
    assert call["profile"] == "default", "must bind 'default', not iterate profiles"
    assert call["portfolio_context"] is None, "telemetry scan must be portfolio-neutral"
    assert call["run_context"] is not None, "must run inside an isolated RunContext"
    assert marked == [sched._FUNNEL_SCAN_DONE_KEY]


def test_scheduler_funnel_signal_scan_skips_outside_window_and_when_done(monkeypatch):
    """task_funnel_signal_scan must be a no-op before market close, on
    weekends, and once already done today."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import tools.opportunity_scanner as scanner
    import tools.scheduler as sched
    import tools.user_profile as up

    ran: list[str] = []
    monkeypatch.setattr(up, "list_available_profiles", lambda: [{"name": "default"}])
    monkeypatch.setattr(scanner, "_scan_impl",
                        lambda *a, **k: ran.append("scan") or {"diagnostics": {}, "top_picks": []})

    # Thursday midday — before close.
    midday = datetime(2026, 7, 9, 11, 0, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: midday)
    asyncio.run(sched.task_funnel_signal_scan())
    assert ran == [], "must not run before market close"

    # Saturday evening — weekend.
    saturday = datetime(2026, 7, 11, 18, 0, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: saturday)
    asyncio.run(sched.task_funnel_signal_scan())
    assert ran == [], "must not run on weekends"

    # After close, but already marked done today.
    after_close = datetime(2026, 7, 9, 16, 30, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(sched, "_eastern_now", lambda: after_close)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: True)
    asyncio.run(sched.task_funnel_signal_scan())
    assert ran == [], "must not re-run once already done today"


def test_scheduler_funnel_signal_scan_retries_on_incomplete(monkeypatch):
    """An aborted/empty scan (no 'diagnostics' key — cancellation, deadline
    abort, or no candidates) must NOT set the daily marker, so the next
    eligible tick retries instead of losing the day. And when no real profile
    has the scheduler enabled, the scan must not run at all."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import tools.opportunity_scanner as scanner
    import tools.scheduler as sched
    import tools.user_profile as up

    marked: list[str] = []
    after_close = datetime(2026, 7, 9, 16, 30, tzinfo=ZoneInfo("US/Eastern"))
    monkeypatch.setattr(up, "list_available_profiles", lambda: [{"name": "default"}])
    monkeypatch.setattr(sched, "_eastern_now", lambda: after_close)
    monkeypatch.setattr(sched, "_already_done_today", lambda key: False)
    monkeypatch.setattr(sched, "_mark_done_today", lambda key: marked.append(key))

    # _empty_result shape: no 'diagnostics' key.
    monkeypatch.setattr(scanner, "_scan_impl",
                        lambda *a, **k: {"top_picks": [], "summary": "🛑 Cancelled."})
    asyncio.run(sched.task_funnel_signal_scan())
    assert marked == [], "incomplete scan must not consume the daily marker"

    # All profiles disabled → no scan attempt.
    ran: list[str] = []
    monkeypatch.setattr(scanner, "_scan_impl", lambda *a, **k: ran.append("scan"))
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: False)
    asyncio.run(sched.task_funnel_signal_scan())
    assert ran == [], "must not scan when every profile has the scheduler disabled"
    assert marked == []
