"""
CairnIQ In-Process Scheduler
=============================
Lightweight asyncio-based scheduler for recurring background tasks.

Safety Guards:
- **Cooldown Registry**: Each task records its last run time in user_data/scheduler_runs.json.
  Tasks are skipped if they've run within their configured cooldown period.
- **Overlap Lock**: Each task has an asyncio.Lock. If a tick fires while the task
  is still running, the new tick is silently skipped.
- **Timeout**: Each task execution is wrapped in asyncio.wait_for() with a configurable
  timeout. Stuck tasks are aborted and the lock is released.
- **Graceful Shutdown**: The scheduler loop checks a shutdown event and exits cleanly.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tools.json_store import write_json_atomic

logger = logging.getLogger("cairniq.scheduler")

# ---------------------------------------------------------------------------
# Config gating (funnel_config.json `scheduler` block — same defensive pattern
# as tools/catalyst_extractor.py::get_escalation_settings: missing file/block/
# key falls back to the default, never raises).
# ---------------------------------------------------------------------------

_FUNNEL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "user_data", "funnel_config.json"
)
DEFAULT_SCHEDULER_SETTINGS: dict[str, bool] = {
    "exchange_rate": True,
    "portfolio_snapshot": True,
    "score_recommendations": True,
    "cache_cleanup": True,
    "premarket_pulse": True,
    "priority_precompute": True,
    "funnel_signal_scan": True,
    "edgar_events": True,
    "event_radar": True,
    "watch_conditions": True,
    "intraday_sentinel": True,
    "observation_consolidation": True,
}


def get_priority_precompute_profiles(config_path: str | None = None) -> tuple[str, ...] | None:
    """Optional allowlist of profile names for the Today's Priority precompute
    (the scheduler's most expensive job — one full reasoning-graph pass per
    profile). Read from funnel_config.json's `scheduler.priority_precompute_profiles`
    (a JSON list of names) so real profile names live in machine-local,
    untracked config — never in source. None = no allowlist, all profiles run.
    Never raises."""
    try:
        with open(config_path or _FUNNEL_CONFIG_PATH) as f:
            block = (json.load(f) or {}).get("scheduler")
        names = block.get("priority_precompute_profiles") if isinstance(block, dict) else None
        if isinstance(names, list):
            cleaned = tuple(str(n).strip() for n in names if str(n).strip())
            return cleaned or None
    except Exception:
        pass
    return None


def get_scheduler_settings(config_path: str | None = None) -> dict[str, bool]:
    """Per-job enabled flags from funnel_config.json's `scheduler` block.
    Missing file/block/keys fall back to enabled=True. Never raises."""
    settings = dict(DEFAULT_SCHEDULER_SETTINGS)
    try:
        with open(config_path or _FUNNEL_CONFIG_PATH) as f:
            block = (json.load(f) or {}).get("scheduler")
        if isinstance(block, dict):
            for key in settings:
                if key in block:
                    settings[key] = bool(block[key])
    except Exception:
        pass
    return settings


def get_scheduler_cooldowns(config_path: str | None = None) -> dict[str, float]:
    """Per-job custom cooldown overrides in seconds from funnel_config.json's `scheduler` block.
    Key format: `<task_name>_cooldown_seconds`. Never raises."""
    cooldowns: dict[str, float] = {}
    try:
        with open(config_path or _FUNNEL_CONFIG_PATH) as f:
            block = (json.load(f) or {}).get("scheduler")
        if isinstance(block, dict):
            for key, val in block.items():
                if isinstance(key, str) and key.endswith("_cooldown_seconds"):
                    task_name = key[:-len("_cooldown_seconds")]
                    try:
                        cooldowns[task_name] = float(val)
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return cooldowns


def update_scheduler_settings(new_settings: dict[str, Any], config_path: str | None = None) -> dict[str, bool]:
    """Update `scheduler` block in funnel_config.json cleanly without clobbering other blocks."""
    path = config_path or _FUNNEL_CONFIG_PATH
    data = {}
    try:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f) or {}
    except Exception:
        data = {}

    if "scheduler" not in data or not isinstance(data["scheduler"], dict):
        data["scheduler"] = {}

    for key, val in new_settings.items():
        if key in DEFAULT_SCHEDULER_SETTINGS:
            data["scheduler"][key] = bool(val)
        elif isinstance(key, str) and key.endswith("_cooldown_seconds"):
            try:
                data["scheduler"][key] = float(val)
            except (ValueError, TypeError):
                pass

    try:
        write_json_atomic(path, data)
    except Exception as e:
        logger.warning(f"Failed to update scheduler settings in {path}: {e}")

    return get_scheduler_settings(path)




# ---------------------------------------------------------------------------
# Market-session helpers (US/Eastern; holidays not modeled — a holiday tick
# is a harmless no-op, not a missed trading day, which is an acceptable v1
# simplification for a single-worker local deployment).
# ---------------------------------------------------------------------------

def _eastern_now() -> datetime:
    return datetime.now(ZoneInfo("US/Eastern"))


def _is_trading_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5  # Mon-Fri


def _after_market_close(dt: datetime, close_hour: int = 16, buffer_minutes: int = 15) -> bool:
    """True once `dt` is past market close plus a settlement buffer, same day."""
    threshold = dt.replace(hour=close_hour, minute=buffer_minutes, second=0, microsecond=0)
    return dt >= threshold


def _in_premarket_window(dt: datetime, start: tuple[int, int] = (6, 0), end: tuple[int, int] = (9, 25)) -> bool:
    start_dt = dt.replace(hour=start[0], minute=start[1], second=0, microsecond=0)
    end_dt = dt.replace(hour=end[0], minute=end[1], second=0, microsecond=0)
    return start_dt <= dt <= end_dt


def _in_market_hours(dt: datetime, start: tuple[int, int] = (9, 30), end: tuple[int, int] = (16, 0)) -> bool:
    """Regular US cash session. Holiday-blind by design — a holiday costs a few
    cache-hit price reads and fires nothing, since quotes don't move."""
    start_dt = dt.replace(hour=start[0], minute=start[1], second=0, microsecond=0)
    end_dt = dt.replace(hour=end[0], minute=end[1], second=0, microsecond=0)
    return start_dt <= dt <= end_dt


def _already_done_today(marker_key: str) -> bool:
    """Per-profile, per-calendar-day (US/Eastern) completion marker — reuses
    tools.daily_cache's existing date-stamped, per-profile file naming for
    free once-per-trading-day semantics. Must be called under the profile
    whose completion is being checked (e.g. inside run_under_profile)."""
    from tools.daily_cache import get_cached
    return bool(get_cached(marker_key))


def is_scheduler_enabled() -> bool:
    """Return True if the background scheduler is enabled for the active profile.
    Off by default."""
    from tools.broker_credentials import get_broker_setting
    val = get_broker_setting("SCHEDULER_ENABLED", "false")
    return val.strip().lower() in ("true", "1", "yes", "on")


def _mark_done_today(marker_key: str) -> None:
    from tools.daily_cache import set_cached
    set_cached(marker_key, True)


_SNAPSHOT_DONE_KEY = "portfolio_snapshot_done"
_PREMARKET_DONE_KEY = "premarket_pulse_done"
_PRIORITY_DONE_KEY = "priority_precompute_done"
_FUNNEL_SCAN_DONE_KEY = "funnel_signal_scan_done"
_EDGAR_EVENTS_DONE_KEY = "edgar_events_done"
_EVENT_RADAR_DONE_KEY = "event_radar_done"
_FUND_SHARES_DONE_KEY = "fund_shares_recorded"
_POSITION_SNAPSHOT_DONE_KEY = "position_snapshot_done"


# Log an LLM-unconfigured skip at most once per (task, process) so a whole
# morning window doesn't repeat the same line every tick.
_llm_unready_warned: set[str] = set()


def _skip_if_llm_unready(task_name: str) -> bool:
    """Return True (logging once) when the active provider isn't configured.

    An LLM-driven task calls this after its window/gate checks so it skips CLEANLY
    — returning normally, not raising — instead of building a graph that raises at
    first model use. A clean skip is deliberately NOT a failure, so it never trips
    the circuit breaker; the task simply waits for the key to be set and the
    server restarted (secrets hydrate at startup). This is the primary guard
    against the 2026-07-24 retry storm."""
    from agent.utils import llm_ready
    ready, why = llm_ready()
    if ready:
        _llm_unready_warned.discard(task_name)
        return False
    if task_name not in _llm_unready_warned:
        _llm_unready_warned.add(task_name)
        logger.warning(
            "Skipping %s: %s. Will keep skipping until the LLM is configured "
            "(set the key in Settings, then restart to re-hydrate secrets).",
            task_name, why,
        )
    return True


# A per-profile result that is NOT a failure: the profile ran, already had today's
# output, or opted out of the scheduler. Anything else ("error: …", "failed …")
# counts against the total-failure check below.
_TASK_NONFAILURE_RESULTS = {"ran", "already done today", "scheduler disabled"}


def _raise_if_total_failure(task_name: str, results: dict[str, str]) -> None:
    """Surface a tick where EVERY enabled profile failed to the circuit breaker.

    The per-profile loops swallow individual errors so one bad profile can't abort
    the rest — but that also hides a SYSTEMIC fault (dead LLM, broken dependency)
    from the scheduler, which then retries every tick. Re-raise only when every
    non-disabled profile failed; a single success keeps this quiet, so a lone bad
    profile still never trips the breaker."""
    considered = [r for r in results.values() if r != "scheduler disabled"]
    if considered and all(r not in _TASK_NONFAILURE_RESULTS for r in considered):
        raise RuntimeError(f"{task_name}: all {len(considered)} enabled profile(s) failed: {results}")


# ---------------------------------------------------------------------------
# Cooldown Registry (persisted to disk)
# ---------------------------------------------------------------------------

def _registry_path() -> Path:
    """Path to the scheduler cooldown registry file."""
    try:
        from tools.user_profile import get_data_path
        return Path(get_data_path("scheduler_runs.json"))
    except Exception:
        return Path("user_data/scheduler_runs.json")


def _load_registry() -> dict:
    """Load the cooldown registry from disk."""
    path = _registry_path()
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_registry(registry: dict):
    """Persist the cooldown registry to disk."""
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json_atomic(path, registry)
    except OSError as e:
        logger.warning(f"Failed to save scheduler registry: {e}")


def _can_run(task_name: str, cooldown_seconds: float) -> bool:
    """Check if enough time has elapsed since the last run."""
    registry = _load_registry()
    last_run = registry.get(task_name, 0)
    return (time.time() - last_run) >= cooldown_seconds


def _record_run(task_name: str):
    """Record a successful task execution."""
    registry = _load_registry()
    registry[task_name] = time.time()
    _save_registry(registry)


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

# Roadmap 2.6: outcomes a per-profile task's inner function returns when it
# deliberately did NOT run its logic. These tasks are re-checked roughly every
# tick and are correct no-ops for the rest of the day once the day's work is
# done, so they must report a SKIP — see _note_engine_outcome.
_DECLINED_OUTCOMES = ("scheduler disabled", "already done today")


def _note_engine_outcome(
    worked: int, produced: int, declined_reason: str, detail: str = ""
) -> None:
    """Report this tick to the 2.5 liveness heartbeat (Roadmap 2.6).

    `worked` counts profiles that actually executed the engine's logic;
    `produced` is the work it did (snapshots taken, symbols polled, ledger rows
    walked) — never the count of rare interesting events, because an engine that
    legitimately finds nothing newsworthy for a fortnight must not read as dead.
    `detail` carries the rare-event count that must NOT drive the idle streak but
    still has to be visible on the ops view (e.g. legs graded this pass).

    worked == 0 is a SKIP, not idleness. Every marker-gated task here is
    re-checked at tick resolution and is a deliberate no-op once the day's run is
    complete; counting those as "ran and produced nothing" would put each of them
    on a 200-run idle streak by nightfall, which is exactly the cries-wolf
    failure the heartbeat shipped with once and had to be corrected for.

    Calling this AT ALL (even with produced=0) is what opts an engine into
    idleness tracking, so a genuinely dead producer — the funnel log dark
    07-02 → 07-18, the ledger with zero scored calls ever — now accrues a visible
    streak instead of the silence that hid both.
    """
    from tools.engine_heartbeat import note_production, note_skipped

    if worked:
        note_production(produced, detail)
    else:
        note_skipped(declined_reason)


def _count_worked(results: dict[str, str]) -> int:
    """Profiles in `results` that actually ran the logic (not declined, not errored)."""
    return sum(
        1 for outcome in results.values()
        if outcome not in _DECLINED_OUTCOMES and not str(outcome).startswith("error:")
    )


async def task_exchange_rate():
    """Fetch the latest USD/CAD exchange rate and cache it."""
    import yfinance as yf

    from tools.daily_cache import set_cached
    from tools.engine_heartbeat import note_production

    def _fetch():
        ticker = yf.Ticker("USDCAD=X")
        hist = ticker.history(period="5d", timeout=40)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None

    rate = await asyncio.to_thread(_fetch)
    # 2.6: one rate is the whole job, so 0 is unambiguous — the provider returned
    # nothing. There is no "quiet day" reading here, which makes this the one
    # engine where an idle streak maps directly onto a broken feed.
    note_production(1 if rate else 0)
    if rate:
        os.environ["USD_TO_CAD"] = str(rate)
        # The scheduler runs with no request-scoped profile bound. The daily cache
        # is profile-namespaced, so an unbound write lands in the empty '_unbound'
        # profile under the multi-user guard. The rate is profile-independent —
        # home it in 'default' so it is a real (not '_unbound') cache entry.
        from tools.user_profile import run_under_profile
        run_under_profile("default", set_cached, "usd_cad_rate", rate)
        logger.info(f"Exchange rate updated: USD/CAD = {rate:.4f}")


async def task_cache_warm():
    """Keep every profile's dashboard caches warm so no human pays the cold path.

    The dashboard's summary/allocation/top-performers panels are 31.7s cold and
    12ms warm (measured, 10-symbol book). Nothing used to refresh them, so the
    cold path landed on whoever opened the page — every time, because both TTLs
    are shorter than the gap between two human visits, and the daily cache is
    date-stamped so the first visit of each day was cold regardless.

    Not gated on SCHEDULER_ENABLED: this performs no work a page load would not
    perform itself moments later, it just moves it off the reader's critical path.
    Disable per-deployment with CAIRNIQ_CACHE_WARM=false, or per-install via the
    `scheduler` block in funnel_config.json (which also overrides the cooldown).
    """
    from tools.cache_warm import warm_all_profiles, warm_enabled

    if not warm_enabled():
        # 2.6: a switch held OFF is a SKIP with a reason, not a run that produced
        # nothing. This task is re-checked every 10 minutes and never declines for
        # any other reason, so reporting the disabled state as production would
        # accrue ~144 idle runs a day and present a deliberately-off engine as a
        # dead one — the cries-wolf failure this heartbeat had to be corrected for
        # once already. The reason travels with the skip so the ops view can tell
        # "turned off" from "outside its window".
        _note_engine_outcome(0, 0, "cache warming disabled")
        return

    results = await asyncio.to_thread(warm_all_profiles)
    # 2.6: production is PROFILES WARMED. A run that warmed nothing means every
    # profile's summary AND radar failed — which is a broken feed, not a quiet
    # day, since this task has no market-hours or once-per-day window to decline.
    # worked=1 unconditionally: getting here IS the logic running, and an empty
    # `results` is a fault of its own (the profile listing threw) rather than a
    # decline, so it must keep accruing idleness instead of reporting a skip.
    warmed = sum(1 for r in results.values() if r.get("summary") or r.get("radar"))
    _note_engine_outcome(1, warmed, "")
    if warmed:
        logger.info(f"Cache warm: refreshed {warmed}/{len(results)} profile(s)")


async def task_portfolio_snapshot():
    """Take a close-of-day portfolio snapshot for every real profile, once per
    trading day, after market close.

    The scheduler runs with no request-scoped profile bound, so snapshot_portfolio()
    would otherwise resolve to the empty '_unbound' profile (under the multi-user
    guard) and silently capture nothing. Re-bind each known profile explicitly and
    snapshot it in turn.

    Close-time semantics: this task's own scheduler-level cooldown (SCHEDULED_TASKS
    below) is kept short so it's re-checked roughly every tick — the real gating is
    done here, per profile, via `_already_done_today` (a daily_cache marker, so it
    naturally resets at the next US/Eastern calendar day) combined with
    `_after_market_close`. A long outer cooldown would risk missing the actual
    close window for hours if a tick happened to land before close.
    """
    from tools.engine_heartbeat import note_skipped
    from tools.portfolio_tracker import snapshot_portfolio
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _after_market_close(now)):
        note_skipped("before market close")
        return

    def _snapshot_one() -> str:
        if not is_scheduler_enabled():
            return "scheduler disabled"
        if _already_done_today(_SNAPSHOT_DONE_KEY):
            return "already done today"
        result = str(snapshot_portfolio())[:80]
        _mark_done_today(_SNAPSHOT_DONE_KEY)
        return result

    def _snapshot_all() -> dict[str, str]:
        # list_available_profiles() reads get_active_profile() only to flag the
        # 'active' one; bind 'default' so listing itself doesn't trip the guard.
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _snapshot_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        return results

    result = await asyncio.to_thread(_snapshot_all)
    # 2.6: production is SNAPSHOTS TAKEN. Once the day's snapshot is in, every
    # remaining tick declines by design, so those are skips — see
    # _note_engine_outcome.
    worked = _count_worked(result)
    _note_engine_outcome(worked, worked, "all profiles already snapshotted today")
    logger.info(f"Portfolio snapshot ({len(result)} profiles): {result}")


async def task_position_snapshot():
    """Record one per-account, per-holding position row per profile per day (4.10a).

    Distinct from task_portfolio_snapshot, which records portfolio-LEVEL value
    and cost basis into `portfolio_history.csv`. That series cannot answer what
    moved: it has no per-account detail and no external-flow column, which is
    why 4.10's TWR and 4.7's wash-sale half are both blocked on a store nothing
    was writing. This is that store.

    PER-PROFILE, unlike the 5.5 fund-shares recorder: shares outstanding is a
    fact about a fund and can be shared, but a position is a fact about the
    holder and must never be reconciled across profiles.

    Nothing can backfill this. `my_portfolio.csv` is overwritten on every sync,
    so a day this task does not run is a day whose holdings are unrecoverable —
    the same property that makes 5.5 gated on the close and marked per calendar
    day rather than given a long cooldown.

    2.6: production is ROWS RECORDED — the count proving portfolio read, rows
    parsed and store written. Zero recorded against a non-empty portfolio is a
    genuinely dead recorder and is reported rather than smoothed.
    """
    from tools.portfolio_reconciliation import snapshot_positions
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _after_market_close(now)):
        _note_engine_outcome(0, 0, "before market close")
        return

    def _record_one() -> dict[str, Any]:
        if not is_scheduler_enabled():
            return {"declined": "scheduler disabled"}
        if _already_done_today(_POSITION_SNAPSHOT_DONE_KEY):
            return {"declined": "already done today"}
        report = snapshot_positions()
        # Marked done only when rows were actually written. Writing off a failed
        # read for the whole day would lose holdings no source can return.
        if report.get("recorded"):
            _mark_done_today(_POSITION_SNAPSHOT_DONE_KEY)
        return report

    def _record_all() -> dict[str, dict[str, Any]]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        results: dict[str, dict[str, Any]] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _record_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = {"error": str(e)}
        return results

    results = await asyncio.to_thread(_record_all)
    worked = sum(1 for r in results.values() if r.get("recorded"))
    rows = sum(int(r.get("recorded") or 0) for r in results.values())
    if not worked:
        _note_engine_outcome(0, 0, "every profile already recorded today or disabled")
        return

    detail = f"{rows} position rows across {worked} profile(s)"
    _note_engine_outcome(worked, rows, "", detail)
    logger.info(f"Position snapshot: {detail}")


async def task_cache_cleanup():
    """Remove stale cache files older than 7 days.

    2.6 note — DELIBERATELY liveness-only, the one task left uninstrumented for
    production. Its output is files removed, and zero removed is the HEALTHY
    steady state (nothing has aged past 7 days yet). Reporting that number would
    accrue an idle streak on a perfectly working engine and flag it inside a
    fortnight, which is the cries-wolf failure the heartbeat already shipped with
    once. There is no other count here that means "the chain ran" — so this
    engine is never asked to declare production, and is judged on liveness and
    errors alone.
    """
    from tools.daily_cache import cleanup_old

    removed = await asyncio.to_thread(cleanup_old, 7)
    if removed:
        logger.info(f"Cleaned {removed} old cache files")


async def task_housekeeping():
    """Rotate oversized logs and drop conversations nobody has touched in 30 days.

    Nothing in the tree rotated before this: logs and LangGraph checkpoint
    stores grew to ~450 MB on the production host, and task_cache_cleanup only
    ever looked at user_data/daily_cache.

    2.6: production is PATHS SCANNED, never bytes reclaimed. On a healthy host
    the steady state is nothing to rotate and nothing cold enough to prune, so
    reclaimed bytes is 0 on almost every run — reporting it would accrue an
    idle streak on a working engine, the cries-wolf failure 2.6 was corrected
    for. Paths scanned proves the whole chain: logs dir readable, checkpoint
    stores discovered and opened. Zero means the sweep found nothing at all to
    look at, which is the dead-engine signal.
    """
    from tools.engine_heartbeat import note_production
    from tools.housekeeping import run_housekeeping

    report = await asyncio.to_thread(run_housekeeping)
    note_production(
        int(report.get("scanned") or 0),
        detail=(
            f"{report['reclaimed_bytes'] / (1024 * 1024):.1f}MB reclaimed, "
            f"{len(report['logs']['rotated'])} rotated, "
            f"{report['checkpoints']['threads_pruned']} threads pruned"
        ),
    )
    if report["reclaimed_bytes"]:
        logger.info(
            f"Housekeeping reclaimed {report['reclaimed_bytes'] / (1024 * 1024):.1f} MB"
        )


async def task_score_recommendations():
    """Score each profile's logged recommendations (2w/1m/3m vs SPY) for every profile.

    tools/memory.py::score_past_recommendations previously ran only lazily inside
    get_advisor_scorecard, so scores (and the Theme 1.2 calibration line, which
    reads those scores) never accrued unless a user explicitly asked for the
    scorecard. Runs daily, per real profile, same re-binding pattern as
    task_portfolio_snapshot.
    """
    from tools.memory import (
        count_graded_legs,
        load_memory,
        save_memory,
        score_past_recommendations,
    )
    from tools.user_profile import list_available_profiles, run_under_profile

    # 2.6: production is LEDGER ROWS WALKED, not rows newly scored. Scoring only
    # fires when an entry crosses its 14-day horizon, so "no change" is the
    # normal daily outcome and counting it would flag a healthy scorer. Rows
    # walked proves the whole chain — memory loaded, ledger present, scorer
    # reached it. Zero means the ledger is empty or unreadable, which is exactly
    # the state that hid "zero scored calls ever" for weeks.
    #
    # 1.8: rows walked proves the chain RAN; it says nothing about whether the
    # ledger is still ACCRUING — which is the thing that was silently broken
    # (13 superseded legs, zero of them ever scored). So the legs actually graded
    # this pass — split full-horizon vs graded-at-supersession — ride along in
    # the heartbeat DETAIL, reported every pass INCLUDING when it is zero. Detail
    # rather than the production number on purpose: a day where nothing crosses a
    # horizon is healthy, and counting it as production would put a working
    # scorer on an idle streak (the cries-wolf failure 2.6 was corrected for).
    # Counted by diffing the LEDGER either side of the pass, not from the
    # scorer's own report — a producer's word is not evidence its output landed.
    walked = {"rows": 0, "full": 0, "partial": 0}

    def _score_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]

        def _score_one() -> str:
            if not is_scheduler_enabled():
                return "scheduler disabled"
            memory = load_memory()
            walked["rows"] += len(memory.get("past_recommendations") or [])
            before_full, before_partial = count_graded_legs(memory)
            if score_past_recommendations(memory):
                save_memory(memory)
                after_full, after_partial = count_graded_legs(memory)
                walked["full"] += max(0, after_full - before_full)
                walked["partial"] += max(0, after_partial - before_partial)
                return "updated"
            return "no change"

        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _score_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        return results

    result = await asyncio.to_thread(_score_all)
    graded_total = walked["full"] + walked["partial"]
    graded_detail = (
        f"{graded_total} legs graded ({walked['partial']} at supersession) · "
        f"{walked['rows']} ledger rows walked"
    )
    _note_engine_outcome(
        _count_worked(result),
        walked["rows"],
        "scheduler disabled on every profile",
        graded_detail,
    )
    logger.info(
        f"Recommendation scoring ({len(result)} profiles): {result} — {graded_detail}"
    )


async def task_weekly_review():
    """Assemble the weekly one-page review and tell the user it is ready.

    **Why it delivers into the alerts inbox instead of just writing a file.** A
    producer whose output nothing reads is the exact shape this codebase has been
    burned by repeatedly — the funnel log dark for a fortnight, the ledger that
    never scored, the feedback store with a hundred unrated rows. Snapshotting a
    report to disk on a schedule and hoping someone opens the page would have
    been a fifth instance. The inbox is an existing surface with an existing
    reader, so the weekly cadence has somewhere to land.

    Zero-LLM and read-only: every section reads a surface that already exists and
    the market briefing comes from cache, so this never generates, never calls a
    provider, and cannot change the state it describes. No `llm_ready()` gate is
    needed for that reason — and if prose generation is ever added here, that
    gate and the `_raise_if_total_failure` bridge have to be added with it.

    2.6: production is SECTIONS ASSEMBLED, not "interesting" findings. A quiet
    week is the normal outcome, and counting only weeks with something to report
    would put a working reporter on an idle streak within a month. Sections
    assembled proves the whole chain — profile bound, every underlying surface
    reachable, page built.
    """
    from tools.user_profile import list_available_profiles, run_under_profile
    from tools.weekly_review import build_weekly_review, summarize_for_heartbeat

    assembled = {"sections": 0, "detail": ""}

    def _review_all() -> dict[str, str]:
        now = _eastern_now()
        # Sunday evening, so it is waiting on Monday morning. Checked at tick
        # resolution rather than driven by a 7-day cooldown: a cooldown alone
        # would drift the run to whatever hour the process happened to start.
        if now.weekday() != 6 or now.hour < 18:
            return {}

        week_key = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]

        def _review_one() -> str:
            if not is_scheduler_enabled():
                return "scheduler disabled"

            review = build_weekly_review()
            counts = review.get("counts") or {}
            assembled["sections"] += counts.get("total", 0)
            assembled["detail"] = summarize_for_heartbeat(review)

            from tools.alerts import raise_alert
            raise_alert(
                title="Your weekly review is ready",
                message=(
                    f"{review['period']['label']} — "
                    f"{counts.get('ok', 0)} of {counts.get('total', 0)} sections have "
                    f"something to report."
                ),
                severity="info",
                source="weekly_review",
                # One alert per profile per ISO week; a re-tick refreshes rather
                # than stacks a second copy in the inbox.
                dedup_key=f"weekly_review_{week_key}",
                data={"period": review["period"], "counts": counts},
            )
            return "delivered"

        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _review_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        return results

    result = await asyncio.to_thread(_review_all)
    if not result:
        _note_engine_outcome(0, 0, "outside the weekly review window")
        return

    _note_engine_outcome(
        _count_worked(result),
        assembled["sections"],
        "scheduler disabled on every profile",
        assembled["detail"],
    )
    logger.info(f"Weekly review ({len(result)} profiles): {result} — {assembled['detail']}")


async def task_observation_consolidation():
    """Resolve advice follow-through, then draft behavioural rules on a gate (1.7).

    Two halves with deliberately different rails. The follow-through sweep is
    ZERO-LLM — it compares the position size recorded at the call against the
    position held now — so it runs every day for every profile regardless of
    provider state. The consolidation pass is the LLM half, and it is gated twice:
    once here (`_skip_if_llm_unready`, the 2026-07-24 retry-storm guard) and again
    inside `consolidate_observations`, which needs at least
    CONSOLIDATION_GATE_N unread observations before it will read anything. A
    summarizer pointed at a nearly-empty store is how this codebase produced
    invented history twice.

    2.6: production is OBSERVATIONS WALKED, never drafts. Past the gate, most
    passes correctly draft nothing — the log records ordinary use and ordinary
    use contains no durable rule — so counting drafts would put a working engine
    on an idle streak within a week. Observations walked proves the whole chain:
    profile bound, log readable, sweep and pass both reached it. Zero means the
    log is empty or unreadable, which is precisely the silence 1.7 exists to end.
    Drafts, resolutions and gate state ride in the DETAIL, reported every pass
    including when they are zero.
    """
    from tools.observation_consolidation import consolidate_observations
    from tools.observations import get_observation_stats, resolve_rec_follow_through
    from tools.user_profile import list_available_profiles, run_under_profile

    # Checked once per tick, not once per profile: the answer is process-wide and
    # the helper logs at most one line per process.
    llm_ok = not _skip_if_llm_unready("observation_consolidation")

    walked = {"observations": 0, "resolved": 0, "drafted": 0, "gated": 0, "unread": 0}

    def _run_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]

        def _run_one() -> str:
            if not is_scheduler_enabled():
                return "scheduler disabled"

            follow = resolve_rec_follow_through()
            walked["resolved"] += follow["resolved"]

            stats = get_observation_stats()
            walked["observations"] += stats["total"]
            walked["unread"] += stats["unconsolidated"]

            if llm_ok:
                report = consolidate_observations()
                walked["drafted"] += report.get("drafted", 0)
                if report.get("gated"):
                    walked["gated"] += 1
            return "ran"

        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _run_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        return results

    result = await asyncio.to_thread(_run_all)
    _raise_if_total_failure("observation_consolidation", result)

    detail = (
        f"{walked['drafted']} rules drafted · {walked['resolved']} calls resolved · "
        f"{walked['unread']} unread"
        + (f" · {walked['gated']} profile(s) below the gate" if walked["gated"] else "")
        + ("" if llm_ok else " · LLM unconfigured, sweep only")
    )
    _note_engine_outcome(
        _count_worked(result),
        walked["observations"],
        "scheduler disabled on every profile",
        detail,
    )
    logger.info(f"Observation consolidation ({len(result)} profiles): {result} — {detail}")


async def task_premarket_pulse():
    """Pre-market news digest + chained catalyst scan, once per trading day, per profile.

    Calls api.background.run_news_agent_in_background() directly rather than
    start_news_fetch() (which spawns its own daemon thread) — this task already
    runs inside asyncio.to_thread, so no extra thread is needed. That call already
    chains maybe_start_catalyst_scan_after_news() internally, config-gated via the
    existing `catalyst` block (auto_scan_after_news / auto_scan_min_interval_hours) —
    nothing new to wire there. The chained scan runs in its own detached daemon
    thread and is not awaited here, so this task's timeout only needs to cover the
    news synthesis call itself, not the scan or any Opus escalations it triggers.

    Same close-time-style gating as task_portfolio_snapshot: a short outer cooldown
    (re-checked roughly every tick) plus a per-profile daily_cache marker gated to
    the pre-market window, so a tick landing before or after the window is a cheap
    no-op rather than missing the day entirely.
    """
    from api.background import run_news_agent_in_background
    from tools.engine_heartbeat import note_skipped
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _in_premarket_window(now)):
        note_skipped("outside pre-market window")
        return
    if _skip_if_llm_unready("premarket_pulse"):
        # An absent provider credential is a deliberate decline, not idleness —
        # the retry-storm gate already refused to spend on it. Recording it as a
        # skip keeps the reason visible on the ops view without inflating a
        # dead-engine streak.
        note_skipped("LLM provider credential unavailable")
        return

    def _run_one() -> str:
        if not is_scheduler_enabled():
            return "scheduler disabled"
        if _already_done_today(_PREMARKET_DONE_KEY):
            return "already done today"
        run_news_agent_in_background(force=False)
        _mark_done_today(_PREMARKET_DONE_KEY)
        return "ran"

    def _run_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _run_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        _raise_if_total_failure("premarket_pulse", results)
        return results

    result = await asyncio.to_thread(_run_all)
    _note_engine_outcome(
        _count_worked(result),
        sum(1 for outcome in result.values() if outcome == "ran"),
        "all profiles already pulsed today",
    )
    logger.info(f"Pre-market pulse ({len(result)} profiles): {result}")


async def task_priority_precompute():
    """Precompute the Today's Priority brief, once per trading day, per profile
    (Theme 3.1's remainder — the advisor's morning call, ready before the open).

    Drives the same [QuickAction name=priority] marker the dashboard button
    sends through the FULL reasoning graph (Supervisor → DeepReasoning →
    RiskManager) via api.background.run_priority_precompute_in_background(),
    which caches the brief per profile as `today_priority` for cold reads by
    GET /api/priority.

    The window starts an hour after the pre-market pulse window so the news /
    catalyst caches this brief leans on are fresh. Same resumable gating as
    task_premarket_pulse: a heavy run can outlast one tick's timeout, but the
    per-profile daily marker means the next tick skips finished profiles and
    picks up the rest — a timeout costs a tick, not the day.
    """
    from api.background import run_priority_precompute_in_background
    from tools.engine_heartbeat import note_skipped
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _in_premarket_window(now, start=(7, 0))):
        note_skipped("outside precompute window")
        return
    if _skip_if_llm_unready("priority_precompute"):
        note_skipped("LLM provider credential unavailable")
        return

    def _run_one() -> str:
        if not is_scheduler_enabled():
            return "scheduler disabled"
        if _already_done_today(_PRIORITY_DONE_KEY):
            return "already done today"
        if not run_priority_precompute_in_background():
            return "failed (will retry next tick)"
        _mark_done_today(_PRIORITY_DONE_KEY)
        return "ran"

    def _run_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        allowlist = get_priority_precompute_profiles()
        if allowlist is not None:
            names = [n for n in names if n in allowlist]
        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _run_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        _raise_if_total_failure("priority_precompute", results)
        return results

    result = await asyncio.to_thread(_run_all)
    # Production counts only profiles whose brief was actually generated. A
    # profile that ATTEMPTED and returned "failed (will retry next tick)" is
    # worked-but-unproductive on purpose: a tick where every attempt fails must
    # accrue idleness, because a precompute that never succeeds is the dead
    # engine this instrument exists to surface.
    _note_engine_outcome(
        _count_worked(result),
        sum(1 for outcome in result.values() if outcome == "ran"),
        "all profiles already precomputed today",
    )
    logger.info(f"Priority precompute ({len(result)} profiles): {result}")


async def task_funnel_signal_scan():
    """Nightly portfolio-neutral broad funnel scan — the M5 signal-log producer.

    The walk-forward signal log (user_data/funnel_signal_log/, read by
    tools/funnel_backtest.evaluate_signal_log) previously only accrued when a
    user happened to press Scan — 16 snapshots in a month, dark for weeks at a
    time — so the flip-default gate ("awaits matured signal-log data") could
    never resolve. This task makes the telemetry unconditional: one broad scan
    per trading day, after close, regardless of anyone pressing a button.

    Deliberately calls _scan_impl directly with portfolio_context=None instead
    of scan_sector_opportunities (which injects the active profile's holdings):
    the signal log is GLOBAL and every profile's backtest reads it, so it must
    record the funnel's market selection — not funnel × one profile's M4
    portfolio-fit overlay — and no user's holdings may influence the shared
    file. User-initiated scans stay personalized; only this telemetry run is
    neutral.

    Runs once globally per trading day (the log is global), bound to 'default'
    like task_exchange_rate so the pipeline's daily_cache reads/writes land in
    the real shared cache rather than '_unbound'. Gated on any real profile
    having the scheduler enabled — the corpus serves everyone, so one opted-in
    profile suffices.

    An isolated RunContext is bound for the duration of the scan: without one,
    the pipeline's is_cancelled() phase checks fall back to the GLOBAL chat
    cancel event, and a stale cancel from an earlier user turn would silently
    abort every nightly scan. The daily marker is only set after a completed
    pipeline (an aborted/empty result lacks 'diagnostics'), so a failed run
    retries on the next eligible tick instead of losing the day.
    """
    from agent.utils import activate_run_context, build_run_context, reset_run_context
    from tools.engine_heartbeat import note_production, note_skipped
    from tools.opportunity_scanner import _V2_SCAN_TIMEOUT, _scan_impl
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _after_market_close(now)):
        note_skipped("before market close")
        return

    # 2.6: this is ONE of the two engines whose outage motivated 2.5 and which
    # 2.5 then did not cover. The signal log sat dark 07-02 -> 07-18 with every
    # test green, because a producer that never runs and a producer with nothing
    # to say emitted identical silence. Production here is PICKS LOGGED off a
    # COMPLETED scan: an incomplete run reports nothing (it never reached the
    # signal-log write), and a completed scan yielding 0 picks night after night
    # is a real finding, not a quiet market.
    logged = {"picks": None}

    def _run() -> str:
        if run_under_profile("default", _already_done_today, _FUNNEL_SCAN_DONE_KEY):
            return "already done today"
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        if not any(run_under_profile(n, is_scheduler_enabled) for n in names):
            return "scheduler disabled on every profile"
        token = activate_run_context(build_run_context())
        try:
            deadline = time.perf_counter() + _V2_SCAN_TIMEOUT - 5
            result = run_under_profile(
                "default", _scan_impl, "All",
                portfolio_context=None, deadline=deadline,
            )
        finally:
            reset_run_context(token)
        # 'diagnostics' only exists on a fully assembled result — _empty_result
        # (cancellation / deadline abort / no candidates) never has it, and an
        # incomplete run never reached the signal-log write.
        if not (isinstance(result, dict) and "diagnostics" in result):
            summary = result.get("summary", "") if isinstance(result, dict) else str(result)
            return f"incomplete — will retry next tick ({str(summary)[:80]})"
        run_under_profile("default", _mark_done_today, _FUNNEL_SCAN_DONE_KEY)
        logged["picks"] = len(result.get("top_picks") or [])
        return f"ran ({logged['picks']} picks logged)"

    result = await asyncio.to_thread(_run)
    if logged["picks"] is not None:
        note_production(logged["picks"])
    elif result in ("already done today", "scheduler disabled on every profile"):
        note_skipped(result)
    else:
        # Attempted and aborted before the signal-log write (cancelled, deadline,
        # no candidates). NOT a skip: it tried and produced nothing, and a scan
        # that keeps aborting is precisely the dark-producer state this engine
        # went into for sixteen days. It must accrue an idle streak.
        note_production(0, detail=str(result)[:120])
    logger.info(f"Funnel signal scan: {result}")


async def task_edgar_events():
    """SEC EDGAR event poll for held names (Advisor Roadmap 5.1): material 8-K
    filings and Form 4 insider cluster-buys land in the alerts inbox (3.2).

    Once per trading day, per profile (each profile's holdings differ and each
    has its own inbox). EDGAR data is cached per profile with a 6h submissions
    TTL, and the fetcher is throttled to SEC fair-use limits, so the daily cost
    is a handful of polite keyless requests per held US name. Non-SEC listings
    (.TO etc.) resolve no CIK and are skipped without a request.

    Alert policy: 8-Ks only from the last 3 days (a first run must not flood
    the inbox with month-old filings) and only warning/critical items; cluster
    buys only when the latest buy is <7 days old. Dedup keys pin alerts to the
    accession / latest-buy-date so the daily poll refreshes rather than
    duplicates.
    """
    from datetime import timedelta

    from tools.alerts import raise_alert
    from tools.engine_heartbeat import note_skipped
    from tools.portfolio_csv import get_tradeable_symbols
    from tools.sec_edgar import get_form4_activity, get_recent_8k, resolve_cik
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not _is_trading_weekday(now):
        note_skipped("not a trading weekday")
        return

    recent_buy_cutoff = (now.date() - timedelta(days=7)).isoformat()
    # 2.6: production is SYMBOLS POLLED, never alerts raised. A material 8-K or a
    # cluster buy is rare and legitimately 0 for weeks — counting alerts would
    # report this engine as dead through every quiet stretch. Symbols polled
    # proves the chain: holdings read, CIKs resolved, EDGAR reachable. Zero means
    # no held US names resolved, which is worth seeing.
    polled = {"symbols": 0}

    def _poll_one() -> str:
        if not is_scheduler_enabled():
            return "scheduler disabled"
        if _already_done_today(_EDGAR_EVENTS_DONE_KEY):
            return "already done today"
        symbols = [s for s in get_tradeable_symbols() if resolve_cik(s)]
        polled["symbols"] += len(symbols)
        alerts = 0
        for sym in symbols:
            try:
                eightk = get_recent_8k(sym, days=3)
                for filing in (eightk.get("filings") or []) if isinstance(eightk, dict) else []:
                    if filing.get("severity") not in ("warning", "critical"):
                        continue
                    item_text = "; ".join(i["description"] for i in filing.get("items", [])
                                          if i.get("severity") != "info") or "material items"
                    raise_alert(
                        title=f"{sym}: 8-K filed — {item_text[:120]}",
                        message=(
                            f"{sym} filed a {filing.get('form', '8-K')} on "
                            f"{filing.get('filing_date')}: {item_text}. {filing.get('url', '')}"
                        ),
                        severity=filing["severity"],
                        source="edgar",
                        dedup_key=f"edgar_8k:{sym}:{filing.get('accession')}",
                        data={"symbol": sym, "accession": filing.get("accession"),
                              "items": filing.get("items", [])},
                    )
                    alerts += 1
                activity = get_form4_activity(sym, days=45)
                cluster = activity.get("cluster") or {} if isinstance(activity, dict) else {}
                if cluster.get("cluster_buy") and cluster.get("latest_buy_date", "") >= recent_buy_cutoff:
                    buyers = ", ".join(b["name"] for b in cluster.get("buyers", [])[:4])
                    raise_alert(
                        title=f"{sym}: insider cluster buy — {cluster.get('distinct_buyers')} insiders",
                        message=(
                            f"{cluster.get('distinct_buyers')} distinct insiders made open-market "
                            f"purchases of {sym} totalling ${cluster.get('total_value', 0):,.0f} "
                            f"({buyers}). Latest buy {cluster.get('latest_buy_date')}. "
                            f"Source: SEC Form 4 (code P only)."
                        ),
                        severity="warning",
                        source="edgar",
                        dedup_key=f"edgar_cluster:{sym}:{cluster.get('latest_buy_date')}",
                        data={"symbol": sym, "cluster": cluster},
                    )
                    alerts += 1
            except Exception as e:  # noqa: BLE001 — one bad symbol must not abort the poll
                logger.warning(f"EDGAR poll failed for {sym}: {e}")
        _mark_done_today(_EDGAR_EVENTS_DONE_KEY)
        return f"{len(symbols)} US symbols, {alerts} alerts"

    def _poll_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _poll_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        return results

    result = await asyncio.to_thread(_poll_all)
    _note_engine_outcome(
        _count_worked(result), polled["symbols"], "all profiles already polled today"
    )
    logger.info(f"EDGAR events ({len(result)} profiles): {result}")


async def task_event_radar():
    """Holdings event radar (Advisor Roadmap 3.5): earnings, ex-dividends and
    FOMC for held names, with T-3 / T-1 alerts into the 3.2 inbox.

    Once per trading day, per profile — each profile holds different names and
    has its own inbox. Zero LLM: dates are facts. The provider lookups are the
    already-cached `get_earnings_info`, so a second profile holding the same
    name costs nothing.

    Runs in the morning window rather than after close: a T-1 earnings warning
    delivered at 20:00 the night before is a warning about something that has
    already been priced into the pre-market.
    """
    from tools.engine_heartbeat import note_skipped
    from tools.event_radar import run_event_radar_tick
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not _is_trading_weekday(now):
        note_skipped("not a trading weekday")
        return

    # 2.6: production is HELD NAMES SWEPT, never alerts raised. A book with
    # nothing inside three days is the normal state for most of the year, so
    # counting alerts would report this engine dead through every quiet stretch.
    swept = {"names": 0}

    def _run_one() -> str:
        if not is_scheduler_enabled():
            return "scheduler disabled"
        if _already_done_today(_EVENT_RADAR_DONE_KEY):
            return "already done today"
        result = run_event_radar_tick()
        swept["names"] += int(result.get("checked") or 0)
        _mark_done_today(_EVENT_RADAR_DONE_KEY)
        return (f"{result['checked']} names, {result['events']} dated events, "
                f"{result['alerts']} alerts, {result['unknown']} undated")

    def _run_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _run_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        return results

    result = await asyncio.to_thread(_run_all)
    _note_engine_outcome(
        _count_worked(result), swept["names"], "all profiles already swept today"
    )
    logger.info(f"Event radar ({len(result)} profiles): {result}")


async def task_watch_conditions():
    """Evaluate advisor-authored watch conditions (Advisor Roadmap 3.3).

    The advisor commits to trigger levels in every Today's Priority brief and
    every catalyst scenario; tools/watch_conditions.py stores them, and this is
    the tick that actually checks them. Zero LLM calls — one cached price read
    per distinct symbol per profile — so it is cheap enough to run through the
    session at the tick resolution rather than once a day.

    Deliberately NOT bounded by a daily marker, unlike every other task here: a
    condition can be crossed at 10:04 and back by 10:20, so the value of the
    engine is entirely in re-checking. The 30-minute cooldown is the real
    frequency control (roadmap: "a zero-LLM batch price check every ~30min in
    market hours").

    Market hours only. Outside them the quote would be the same close print on
    every tick, so a fire would say nothing new — and an after-hours print is
    exactly the kind of thin-liquidity level the advisor did not mean.
    """
    from tools.engine_heartbeat import note_production, note_skipped
    from tools.user_profile import list_available_profiles, run_under_profile
    from tools.watch_conditions import evaluate_conditions

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _in_market_hours(now)):
        note_skipped("outside market hours")
        return

    def _check_one() -> str:
        if not is_scheduler_enabled():
            return "scheduler disabled"
        result = evaluate_conditions()
        # Liveness (2.5) counts CONDITIONS CHECKED, not conditions fired: a fire
        # is rare and legitimately 0 for weeks, but an armed store being
        # evaluated proves the capture->store->evaluate chain is intact. Reported
        # BEFORE the empty-store return and including 0 on purpose — an empty
        # store is precisely the dead-engine signal (when 3.3 shipped dead this
        # was 0 on every tick for a full day), so it must land as "ran and
        # produced nothing", never as "not instrumented".
        note_production(int(result.get("checked") or 0))
        if not result.get("checked"):
            return "no active conditions"
        return (
            f"{result['checked']} checked, {result['fired']} fired, "
            f"{result['voided']} voided, {result['expired']} expired, "
            f"{result['unavailable']} no-quote"
        )

    def _check_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        results: dict[str, str] = {}
        for name in names:
            try:
                results[name] = run_under_profile(name, _check_one)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        return results

    result = await asyncio.to_thread(_check_all)
    # Only log profiles that had something to check — an idle store on five
    # profiles would otherwise write a line every half hour, all day.
    interesting = {k: v for k, v in result.items() if v not in ("no active conditions", "scheduler disabled")}
    if interesting:
        logger.info(f"Watch conditions: {interesting}")


def _report_deployment_ladder(ladders: list[dict]) -> None:
    """Say how many deployment rungs are actually armed (Roadmap 3.9 liveness).

    The playbook is empty on every profile today, so this ships inert — which is
    exactly the situation 2.5/2.6 were built for. An inert ladder that reports
    nothing is indistinguishable from a healthy quiet one, and this codebase has
    now counted five surfaces that sat dark behind a green check. So the count
    goes out with an explicit 0 and names WHY it is 0.

    `note_production(0, detail)` adds nothing to the produced count — that stays
    holdings-checked, the number that proves the scan ran — and only sets the
    detail line, which is last-write-wins. One aggregate line across profiles
    therefore replaces what would otherwise be per-profile noise overwriting
    each other in an unpredictable order.
    """
    from tools.engine_heartbeat import note_production

    evaluated = [x for x in ladders if x.get("evaluated")]
    if not evaluated:
        # Distinct from an empty ladder: nothing was looked at, so reporting
        # "0 rungs armed" here would be a fabricated liveness number.
        note_production(0, "deployment ladder: not evaluated (no fresh SPY reading)")
        return

    levels = sum(int(x.get("levels", 0)) for x in evaluated)
    if not levels:
        note_production(
            0,
            f"deployment ladder: INERT — 0 rungs on file across {len(evaluated)} profile(s)",
        )
        return

    armed = sum(int(x.get("armed", 0)) for x in evaluated)
    fired = sum(int(x.get("fired", 0)) for x in evaluated)
    seeded = sum(int(x.get("seeded", 0)) for x in evaluated)
    detail = f"deployment ladder: {armed}/{levels} rungs armed, {fired} fired"
    if seeded:
        detail += f", {seeded} already past when first armed"
    note_production(0, detail)


async def task_intraday_sentinel():
    """Intraday market-state sentinel (Advisor Roadmap 3.4).

    The daily Market Pulse's intraday counterpart. Reuses the sentinel's own
    inputs — `_get_market_snapshot` for the indices and the batch technicals for
    holdings — and fires ONLY on state *changes* (VIX/SPY band crossings, a fresh
    death or golden cross, a >2.5x volume spike), each debounced by hysteresis
    and by a per-profile state store so a standing level is never restated as
    news. Owns the death-cross alert producer left open by 3.2. Zero LLM calls.

    Also arms Roadmap 3.9's cash-deployment ladder off the same SPY drawdown
    reading, so a rung the user wrote at −5% delivers its action at −5% instead
    of waiting for a crossing deep enough to surface the whole playbook.

    Market hours only, same rail as task_watch_conditions: outside the session
    the quote is a stale close print, so a "crossing" would say nothing new.

    The 6-hour cooldown is the real frequency control, and it is deliberately
    slower than 3.3's 30-minute watch-condition tick. A US session runs 6.5
    hours, so this lands roughly twice a session — near the open and again late
    afternoon. That is the right cadence for a long-horizon holder: a band that
    spikes and round-trips inside one day is noise a decade-long plan should
    never be paged about, whereas 3.3's triggers are levels the advisor
    explicitly committed to act on and stay on the faster tick. There is no
    daily marker, so an all-day drift into a deeper band is still caught on the
    second pass.

    The index snapshot is fetched ONCE and shared across profiles (it is global);
    only the per-profile holdings scan and state read/write run under each
    profile.
    """
    from tools.engine_heartbeat import note_production, note_skipped
    from tools.intraday_sentinel import fetch_market_snapshot, run_sentinel_tick
    from tools.user_profile import list_available_profiles, run_under_profile

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _in_market_hours(now)):
        note_skipped("outside market hours")
        return

    snapshot = fetch_market_snapshot()  # one global index read for all profiles

    def _tick_one() -> tuple[str, dict]:
        if not is_scheduler_enabled():
            return "scheduler disabled", {}
        result = run_sentinel_tick(snapshot_fn=lambda: snapshot)
        if result.get("error"):
            return f"error: {result['error']}", {}
        # Liveness (2.5) counts HOLDINGS EVALUATED, not alerts fired — a fresh
        # cross is rare by design, but a sentinel scanning 0 holdings every tick
        # is a broken universe read wearing a healthy face.
        note_production(int(result["checked_holdings"]))
        return (
            f"{result['checked_holdings']} holdings checked, {result['fired']} fired",
            result.get("ladder") or {},
        )

    def _tick_all() -> dict[str, str]:
        profiles = run_under_profile("default", list_available_profiles)
        names = [
            p["name"] for p in profiles
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        results: dict[str, str] = {}
        ladders: list[dict] = []
        for name in names:
            try:
                results[name], ladder = run_under_profile(name, _tick_one)
                if ladder:
                    ladders.append(ladder)
            except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
                results[name] = f"error: {e}"
        _report_deployment_ladder(ladders)
        return results

    result = await asyncio.to_thread(_tick_all)
    # Only log profiles that actually did something (fired an alert), so an idle
    # session doesn't write a line every half hour on every profile.
    interesting = {k: v for k, v in result.items() if not v.endswith("0 fired") and v != "scheduler disabled"}
    if interesting:
        logger.info(f"Intraday sentinel: {interesting}")


async def task_fund_shares_record():
    """Record one shares-outstanding point per held fund, after the close (5.5).

    GLOBAL, not per-profile, and that is the whole reason this task exists in
    this shape: shares outstanding is a fact about the fund, so the store is
    shared and the sweep runs ONCE over the union of funds held across profiles.
    Recording per profile would poll the same ETF repeatedly and — the part that
    actually matters — a profile added next month would begin its series from
    zero instead of inheriting the weeks already accrued.

    There is nothing to backfill: `get_shares_full` is `None` for the entire fund
    class and no vendor sells the history on our plans (measured 2026-07-28), so
    every day this task does not run is a day permanently missing from the series.
    That is why it is gated on the close and marked done per calendar day rather
    than given a long cooldown.

    2.6: production is ROWS RECORDED. That is the count proving the whole chain —
    universe read, classifier resolved, source reachable, store written. Zero
    recorded against a non-empty universe is a genuinely dead recorder and must
    accrue an idle streak, so it is reported rather than smoothed. Once the day's
    rows are in, every later tick declines by design and reports a SKIP.
    """
    from tools.fund_flows import record_fund_shares
    from tools.user_profile import run_under_profile

    now = _eastern_now()
    if not (_is_trading_weekday(now) and _after_market_close(now)):
        _note_engine_outcome(0, 0, "before market close")
        return

    def _record() -> dict[str, Any]:
        # Bound to 'default' so the daily marker and the profile-listing read
        # inside collect_fund_universe() don't trip the multi-user guard.
        if _already_done_today(_FUND_SHARES_DONE_KEY):
            return {"declined": "already done today"}
        # A global task honours the setting the way task_funnel_signal_scan does:
        # run if ANY profile wants background work. SCHEDULER_ENABLED is off by
        # default, and a user who left it off has not asked us to poll a vendor
        # on their behalf every evening. The cost of the gate is that the series
        # does not accrue until someone opts in — which is why this declines with
        # a reason the heartbeat will show, rather than silently doing nothing.
        from tools.user_profile import list_available_profiles
        names = [
            p["name"] for p in list_available_profiles()
            if not p["name"].startswith("pytest_") and p["name"] != "_unbound"
        ]
        if not any(run_under_profile(n, is_scheduler_enabled) for n in names):
            return {"declined": "scheduler disabled on every profile"}
        report = record_fund_shares()
        # Mark done only when something was actually written. A failed source
        # must be retried on the next tick, not silently written off for the day
        # — a missed day cannot be recovered later from any source.
        if report.get("recorded"):
            _mark_done_today(_FUND_SHARES_DONE_KEY)
        return report

    report = await asyncio.to_thread(run_under_profile, "default", _record)

    if report.get("declined"):
        _note_engine_outcome(0, 0, report["declined"])
        return

    detail = (
        f"{report['recorded']}/{report['universe']} funds recorded"
        + (f", {report['failed']} failed" if report.get("failed") else "")
        + (f", {len(report['unresolved'])} unresolved" if report.get("unresolved") else "")
    )
    _note_engine_outcome(1, int(report.get("recorded") or 0), "", detail)
    logger.info(f"Fund shares recorder: {detail}")


async def task_availability_report():
    """Fold the measured window coverage into the heartbeat once a day (7.1 Step 1).

    Deliberately NOT gated on `is_scheduler_enabled()` and not looped over
    profiles: availability is a fact about the HOST, not about a book, and the
    measurement reads a log this process already writes. Nothing is fetched and
    nothing is stored.

    Its purpose here is preservation, not alerting. The measurement is retroactive
    from `logs/cairniq.watchdog.log`, so it needs no accrual — but that log is
    subject to rotation, and once a week rolls off the front of it the coverage
    for that week is gone. Recording the figure in the daily heartbeat detail
    keeps a durable trace of it.

    **The SLO is reported, not alerted.** 7.1's Step 2 set the floor at 98%
    window coverage (see `availability.SLO_WINDOW_COVERAGE_PCT`), and the verdict
    rides in the heartbeat detail so a breach is visible on the ops view. It does
    not page: Step 2's purpose is to let Step 3 be judged against a number fixed
    beforehand, and turning the floor into an alert is a separate decision that
    was not made. Note the verdict can say `breached` but never `met` — coverage
    is an upper bound, so only the failing direction is ever certain.

    2.6: production is PROBES READ — the count proving the log was found, parsed
    and turned into a coverage figure. Zero probes read is a dead instrument (the
    watchdog stopped, or the log moved) and accrues an idle streak, which is the
    correct signal: it means this measurement has gone blind.
    """
    from tools.availability import get_availability_report

    report = await asyncio.to_thread(get_availability_report)

    probes = int(report.get("probes") or 0)
    if report.get("status") != "measured":
        # An absent instrument is a finding, not a skip: something writes that
        # log, and if it stopped, every figure downstream silently reads as fine.
        _note_engine_outcome(1, 0, "", "probe log unreadable — availability is UNKNOWN")
        logger.warning("Availability: probe log unreadable; coverage is unknown, not 100%")
        return

    detail = (
        f"{report['window_coverage_pct']}% window coverage over "
        f"{report['span_days']}d · {report['window_minutes_lost']:.0f} min lost in "
        f"{report['gaps_in_window']} incident(s) · upper bound (5xx unseen)"
    )
    _note_engine_outcome(1, probes, "", detail)
    logger.info(f"Availability: {detail}")


async def task_catalyst_resolution():
    """Score catalyst predictions whose horizon has elapsed (1.3).

    Not gated on `is_scheduler_enabled()` and not looped over profiles: the
    prediction store is global, because "this event implied bullish for AAPL over
    days" is a claim about the world rather than about a book.

    No LLM. The resolver reads prices, so it needs neither `llm_ready()` nor the
    total-failure bridge that a model-backed task does.

    **The horizons are the reason this runs daily and slowly.** A `structural`
    catalyst resolves 90 days after it was recorded, so this task is mostly a
    no-op that occasionally settles a handful. That is the intended shape — a
    resolver that had lots to do every day would mean the emitter was
    re-recording.

    2.6: production is PREDICTIONS RESOLVED. Zero is the normal case on most days
    and must not read as a dead engine, so `checked` (open predictions examined)
    is what proves the chain ran; the detail carries both.
    """
    from tools.catalyst_resolution import resolve_all, scoreboard

    report = await asyncio.to_thread(resolve_all)
    checked, resolved = int(report.get("checked") or 0), int(report.get("resolved") or 0)

    board = await asyncio.to_thread(scoreboard)
    overall = (board.get("overall") or {})
    rate = overall.get("hit_rate")
    detail = (
        f"{resolved} resolved of {checked} open · "
        f"{overall.get('scored', 0)} scored to date · "
        + (f"hit rate {rate:.0%}" if rate is not None
           else f"no rate yet ({overall.get('scored', 0)}/{20} scored)")
    )
    # `checked` rather than `resolved` is the liveness count: a day where nothing
    # has matured is a working resolver with nothing to settle.
    _note_engine_outcome(1, checked, "", detail)
    logger.info(f"Catalyst resolution: {detail}")


# ---------------------------------------------------------------------------
# Task Configuration
# ---------------------------------------------------------------------------

# Each entry: (task_name, coroutine_factory, cooldown_seconds, timeout_seconds).
# portfolio_snapshot and premarket_pulse use a short cooldown (re-checked every
# tick) because their real gating is internal (market-window + daily marker per
# profile) — a long cooldown here would risk the task not even being invoked
# during the actual window on a given day.
SCHEDULED_TASKS = [
    ("exchange_rate",          task_exchange_rate,          3600,    60),   # Every 1 hour, 60s timeout
    ("cache_warm",             task_cache_warm,              600,     300),  # Every 10 min; keeps the dashboard's 900s summary/radar caches from ever expiring under a reader (31.7s cold vs 12ms warm)
    ("portfolio_snapshot",     task_portfolio_snapshot,      300,     120),  # Checked every tick; fires once after close, per profile
    ("position_snapshot",      task_position_snapshot,       300,     180),  # Checked every tick; per-account/per-holding rows once after close, per profile. A missed day is unrecoverable — my_portfolio.csv is overwritten (4.10a)
    ("availability_report",    task_availability_report,     86400,   60),   # Every 24 hours; reads the watchdog probe log, no network, no profile loop (7.1)
    ("catalyst_resolution",    task_catalyst_resolution,     86400,   300),  # Every 24 hours; scores catalysts whose horizon elapsed. Global store, no LLM. Mostly a no-op — a structural horizon takes 90 days to mature (1.3)
    ("score_recommendations",  task_score_recommendations,   86400,   120),  # Every 24 hours, 120s timeout
    ("cache_cleanup",          task_cache_cleanup,           86400,   30),   # Every 24 hours, 30s timeout
    ("housekeeping",           task_housekeeping,            86400,   600),  # Every 24 hours; gzip of a multi-GB log plus a VACUUM needs real headroom
    ("premarket_pulse",        task_premarket_pulse,         300,     180),  # Checked every tick; fires once in the pre-market window, per profile
    ("priority_precompute",    task_priority_precompute,     300,     900),  # Checked every tick; one full-graph run per profile in the 7:00-9:25 window
    ("funnel_signal_scan",     task_funnel_signal_scan,      1800,    240),  # One global neutral broad scan after close; 30min cooldown bounds same-evening retries, 240s covers the 150s scan budget
    ("edgar_events",           task_edgar_events,            1800,    300),  # 8-K + Form 4 cluster poll for held names, once per trading day per profile
    ("event_radar",            task_event_radar,             3600,    180),  # Earnings/ex-div/FOMC T-3 and T-1 sweep for held names, once per trading day per profile
    ("watch_conditions",       task_watch_conditions,        1800,    120),  # Zero-LLM re-check of advisor-authored triggers, every 30min in market hours
    ("intraday_sentinel",      task_intraday_sentinel,      21600,    180),  # Zero-LLM market-state change detector (VIX/SPY bands, fresh crosses, vol spikes); 6h cooldown in market hours = ~twice a session, a long-horizon cadence
    ("observation_consolidation", task_observation_consolidation, 86400, 240),  # Daily; zero-LLM follow-through sweep always, one gated LLM pass per profile past n
    ("weekly_review",          task_weekly_review,           3600,    180),  # Checked hourly; assembles and delivers once in the Sunday-evening window, per profile
    ("fund_shares_record",     task_fund_shares_record,      300,     240),  # Checked every tick; one global post-close sweep per day. A missed day is gone for good — no vendor sells the history (5.5)
]

# How often the scheduler loop ticks (in seconds).
# Tasks won't run more often than their own cooldown, so this is just the
# polling resolution. 300s (5 min) keeps CPU overhead near zero.
TICK_INTERVAL_SECONDS = 300


# ---------------------------------------------------------------------------
# Scheduler Loop
# ---------------------------------------------------------------------------

# Circuit breaker: after this many consecutive failed runs, a task is paused for
# the rest of the Eastern day (or until a restart) instead of retrying on every
# tick. A persistent fault (missing credential, broken dependency, provider
# outage) won't fix itself by hammering — and for the LLM-driven tasks each retry
# is real model cost. A single success resets the streak; one alert fires on trip.
MAX_CONSECUTIVE_FAILURES = 3


def _alert_circuit_open(task_name: str, failures: int) -> None:
    """Best-effort single alert when the breaker trips (under the default profile)."""
    try:
        from tools.alerts import raise_alert
        from tools.user_profile import run_under_profile
        run_under_profile(
            "default", raise_alert,
            f"Scheduler paused '{task_name}'",
            f"'{task_name}' failed {failures} times in a row and is paused until "
            f"tomorrow (US/Eastern) or a restart. See logs/server for the cause.",
            severity="warning",
            source="scheduler",
            dedup_key=f"circuit_open_{task_name}_{_eastern_now():%Y-%m-%d}",
        )
    except Exception:  # noqa: BLE001 — an alerting failure must never break the loop
        pass


class CairnIQScheduler:
    """
    Lightweight in-process task scheduler with cooldown, locking, and timeout guards.
    """

    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._locks: dict[str, asyncio.Lock] = {}
        self._task: asyncio.Task | None = None
        # Circuit-breaker state (in-memory; resets on restart, which is exactly
        # when a fix is applied). Streak counts consecutive failures per task;
        # opened-on records the Eastern date the breaker tripped.
        self._failure_streak: dict[str, int] = {}
        self._circuit_opened_on: dict[str, str] = {}

    def _circuit_is_open(self, task_name: str) -> bool:
        """True while this task is paused by the breaker (same Eastern day it tripped).
        On a date rollover the breaker closes and the task gets a fresh streak."""
        opened = self._circuit_opened_on.get(task_name)
        if opened is None:
            return False
        if opened == _eastern_now().strftime("%Y-%m-%d"):
            return True
        self._circuit_opened_on.pop(task_name, None)
        self._failure_streak[task_name] = 0
        return False

    def get_paused_tasks(self) -> dict[str, str]:
        """Tasks the breaker is currently holding down -> the Eastern date it tripped.

        Read-only on purpose (it does NOT close a rolled-over breaker the way
        `_circuit_is_open` does): this is called from API request threads for the
        2.5 ops view, and an observability read must not mutate scheduler state.
        A paused task keeps a healthy-looking `ok` heartbeat frozen at its last
        good run, so without this the one view built to find dead engines would
        report a deliberately-stopped one as fine.
        """
        today = _eastern_now().strftime("%Y-%m-%d")
        return {t: d for t, d in self._circuit_opened_on.items() if d == today}

    def _record_outcome(self, task_name: str, ok: bool) -> None:
        """Feed a run's outcome to the breaker: a success clears the streak; the
        Nth consecutive failure opens the breaker and fires one alert."""
        if ok:
            self._failure_streak[task_name] = 0
            return
        n = self._failure_streak.get(task_name, 0) + 1
        self._failure_streak[task_name] = n
        if n >= MAX_CONSECUTIVE_FAILURES and self._circuit_opened_on.get(task_name) is None:
            self._circuit_opened_on[task_name] = _eastern_now().strftime("%Y-%m-%d")
            logger.error(
                "Circuit breaker OPEN for '%s': %d consecutive failures. "
                "Pausing it until tomorrow (US/Eastern) or a restart.",
                task_name, n,
            )
            _alert_circuit_open(task_name, n)

    async def start(self):
        """Start the scheduler loop as a background task."""
        # Initialize locks for each task
        for task_name, _, _, _ in SCHEDULED_TASKS:
            self._locks[task_name] = asyncio.Lock()

        self._task = asyncio.create_task(self._loop())
        logger.info(f"Scheduler started with {len(SCHEDULED_TASKS)} tasks, tick interval={TICK_INTERVAL_SECONDS}s")

    async def stop(self):
        """Signal the scheduler to stop and wait for it to finish."""
        self._shutdown_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("Scheduler stopped")

    async def _loop(self):
        """Main scheduler loop — ticks every TICK_INTERVAL_SECONDS."""
        # Small initial delay so startup tasks (e.g., exchange rate fetch) finish first
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)
            return  # Shutdown was requested during initial delay
        except TimeoutError:
            pass  # Normal: initial delay expired, proceed to loop

        while not self._shutdown_event.is_set():
            for task_name, task_fn, cooldown, timeout in SCHEDULED_TASKS:
                if self._shutdown_event.is_set():
                    break
                await self._try_run(task_name, task_fn, cooldown, timeout)

            # Wait for the next tick or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=TICK_INTERVAL_SECONDS
                )
                break  # Shutdown requested
            except TimeoutError:
                pass  # Normal tick expiry

    async def _try_run(self, task_name: str, task_fn, cooldown: float, timeout: float):
        """Attempt to run a single task with all safety guards."""
        # Guard 0: Config gating (funnel_config.json `scheduler` block) — disableable
        # without code changes, per job.
        if not get_scheduler_settings().get(task_name, True):
            return

        # Guard 0.5: Circuit breaker. A task that failed MAX_CONSECUTIVE_FAILURES
        # times in a row is paused for the rest of the day rather than retried every
        # tick — a persistent fault won't self-heal, and re-running an LLM pipeline
        # into a dead provider burns real calls. Auto-closes on the date rollover.
        if self._circuit_is_open(task_name):
            return

        # Guard 1: Cooldown check (custom override from funnel_config.json if specified)
        effective_cooldown = get_scheduler_cooldowns().get(task_name, cooldown)
        if not _can_run(task_name, effective_cooldown):
            return

        # Guard 2: Overlap lock (non-blocking acquire)
        # setdefault, not [task_name]: the lock table is built in start(), so any
        # task reaching the runner without one raised KeyError HERE — before the
        # try block below, and _loop calls this unguarded, so it would escape and
        # kill the whole scheduler task, silently stopping all ten engines at
        # once. Exactly the dark-engine failure 2.5 exists to prevent, so the
        # runner now supplies its own lock instead of trusting the table.
        lock = self._locks.setdefault(task_name, asyncio.Lock())
        if lock.locked():
            logger.debug(f"Skipping {task_name}: previous execution still running")
            return

        async with lock:
            # Roadmap 2.5: one hook here gives EVERY engine liveness for free.
            # Note this is a different question from the cooldown registry above:
            # _record_run means "the coroutine completed", which a task that
            # early-returns outside market hours also satisfies — so a
            # permanently dead engine and a healthy idle one are indistinguishable
            # there. The heartbeat records whether work actually happened.
            from tools import engine_heartbeat as hb
            hb.begin(task_name)
            started = time.time()
            status, detail = hb.STATUS_OK, ""
            try:
                # Guard 3: Timeout protection
                logger.info(f"Running scheduled task: {task_name}")
                await asyncio.wait_for(task_fn(), timeout=timeout)
                _record_run(task_name)
                logger.info(f"Completed scheduled task: {task_name}")
            except TimeoutError:
                status, detail = hb.STATUS_TIMEOUT, f"timed out after {timeout}s"
                logger.error(f"Task {task_name} timed out after {timeout}s — aborting")
            except Exception as e:
                status, detail = hb.STATUS_ERROR, str(e)
                logger.error(f"Task {task_name} failed: {e}")
            finally:
                hb.record_run(
                    task_name, status=status,
                    duration_ms=int((time.time() - started) * 1000), detail=detail,
                )
            # Feed the circuit breaker: a clean run (incl. a deliberate skip, which
            # completes without raising) resets the streak; an error/timeout counts
            # toward the pause threshold.
            self._record_outcome(task_name, ok=(status == hb.STATUS_OK))


# Singleton instance
scheduler = CairnIQScheduler()
