"""
Engine liveness heartbeat — Advisor Roadmap 2.5.

Every background engine must be able to prove it RAN *and* that it DID
something. The recurring failure in this codebase is not a crash — a crash is
loud and gets fixed the same day. It is an engine that runs forever and produces
nothing while every test stays green:

  - the funnel signal log sat dark 07-02 -> 07-18 (its only producer was
    user-initiated, so the nightly corpus never accrued);
  - the recommendation ledger had ZERO scored calls ever, because restatements
    expired entries below the 14-day scoring horizon;
  - 3.3 watch-conditions harvested 0 conditions on its first real morning while
    leaking its own JSON into the user-visible brief.

All three were found by hand, days late, because "nothing to do" and "silently
broken" emit exactly the same silence.

`scheduler_runs.json` cannot answer this. It records that a coroutine COMPLETED,
which is not the same as work happening: a market-hours task that early-returns
at 20:00 every night still stamps a fresh timestamp, so a permanently-dead engine
and a healthy idle one look identical there. This module records the difference.

Three states, and the distinction between the last two is the whole point:

  - ``error`` / ``timeout`` — the loud case, already logged by the scheduler.
  - ``skipped``  — the engine deliberately declined to work (outside market
    hours, disabled for the profile). It never got the chance to produce, so
    this must NOT count against it.
  - ``ok``       — the engine actually ran its logic. If it produced nothing,
    that increments ``consecutive_idle``, and a large idle streak on an engine
    that is supposed to produce is the signal we have been missing.

The store is a TRUE GLOBAL file, deliberately not `get_data_path`: the scheduler
ticks under the `_unbound` profile while the API reads under whichever profile
is serving the request, and a profile-scoped path would silently give them two
different files — the same class of bug this module exists to catch.

Never raises. An observability layer that can break the thing it observes is
worse than no observability layer.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_HEARTBEAT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "user_data", "engine_heartbeat.json"
)

# Statuses. "skipped" is load-bearing — see the module docstring.
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"

# How many cooldowns an engine may miss before "it has not run" is a finding
# rather than ordinary jitter, and the floor below which we never call anything
# stale (the tick resolution is 300s, so a 300s-cooldown engine legitimately
# lands minutes apart).
STALE_MULTIPLIER = 3.0
MIN_STALE_SECONDS = 900

# The engine currently being run by the scheduler, and what it has reported so
# far this tick. A plain module global rather than a ContextVar on purpose: the
# scheduler awaits each task to completion before starting the next (see
# CairnIQScheduler._tick), so there is never more than one engine in flight, and
# a global cannot silently fail to propagate across the asyncio.to_thread and
# run_under_profile hops the way a ContextVar can. A reporting seam that
# no-ops silently would be this module committing the exact sin it audits.
_lock = threading.Lock()
_current: dict[str, Any] = {
    "engine": None, "produced": 0, "detail": "", "skipped": None, "reported": False,
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load() -> dict[str, Any]:
    try:
        if not os.path.exists(_HEARTBEAT_PATH):
            return {}
        with open(_HEARTBEAT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict[str, Any]) -> bool:
    try:
        os.makedirs(os.path.dirname(_HEARTBEAT_PATH), exist_ok=True)
        tmp = _HEARTBEAT_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, _HEARTBEAT_PATH)
        return True
    except Exception as e:  # noqa: BLE001 — observability must never break the caller
        logger.warning(f"engine heartbeat write failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Reporting seam — called from inside a running engine
# ---------------------------------------------------------------------------

def begin(engine: str) -> None:
    """Open a reporting window for `engine` (called by the scheduler runner)."""
    with _lock:
        _current.update({
            "engine": engine, "produced": 0, "detail": "", "skipped": None, "reported": False,
        })


def note_production(count: int, detail: str = "") -> None:
    """Report that the running engine produced `count` units of real output.

    Calling this AT ALL — even with 0 — opts the engine into idleness tracking.
    That distinction is load-bearing: "I ran and produced 0" is a dead-engine
    signal, while an engine that never reports has simply not been instrumented
    and must never be judged on production it was never asked to declare.
    Without the split, every uninstrumented engine accrues an idle streak and the
    ops view fills with false "never produced" alarms within the hour — an
    instrument that cries wolf stops being read, which is how we got here.

    Additive across the tick, because the multi-profile engines (watch
    conditions, intraday sentinel) call this once per profile and the engine's
    production for the tick is the sum. Safe to call when no window is open —
    that is just a direct invocation outside the scheduler.
    """
    try:
        with _lock:
            if _current["engine"] is None:
                return
            _current["reported"] = True
            _current["produced"] += max(0, int(count))
            if detail:
                _current["detail"] = detail
    except Exception:
        pass


def note_skipped(reason: str = "") -> None:
    """Report that the running engine deliberately did no work this tick.

    Outside market hours, disabled for every profile, nothing configured. A skip
    is NOT idleness: the engine never got the chance to produce, so it must not
    accrue an idle streak that would later read as a dark engine.
    """
    try:
        with _lock:
            if _current["engine"] is None:
                return
            _current["skipped"] = reason or "skipped"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Recording — called by the scheduler runner when a task finishes
# ---------------------------------------------------------------------------

def record_run(
    engine: str,
    status: str = STATUS_OK,
    duration_ms: int = 0,
    produced: int | None = None,
    detail: str = "",
) -> dict[str, Any]:
    """Fold one completed run into `engine`'s heartbeat. Never raises.

    `produced`/`detail` default to whatever the engine reported through
    note_production/note_skipped during the run.
    """
    try:
        with _lock:
            reported = dict(_current) if _current["engine"] == engine else {}
            _current.update({
                "engine": None, "produced": 0, "detail": "", "skipped": None, "reported": False,
            })

        # Did the engine declare production this run? An explicit `produced=`
        # from the caller counts as a declaration; otherwise it is whether the
        # engine called note_production (even with 0) during the window.
        reports_production = produced is not None or bool(reported.get("reported"))
        if produced is None:
            produced = int(reported.get("produced", 0) or 0)
        if not detail:
            detail = str(reported.get("detail", "") or "")
        # An engine that reported a skip is never counted as having run its logic.
        if status == STATUS_OK and reported.get("skipped"):
            status = STATUS_SKIPPED
            detail = detail or str(reported["skipped"])

        data = _load()
        rec = data.get(engine) or {}
        stamp = _now()

        rec["last_ran_at"] = stamp
        rec["last_status"] = status
        rec["last_duration_ms"] = int(duration_ms)
        rec["last_detail"] = detail[:300]
        rec["runs"] = int(rec.get("runs", 0)) + 1

        if status in (STATUS_ERROR, STATUS_TIMEOUT):
            rec["errors"] = int(rec.get("errors", 0)) + 1
            rec["last_error_at"] = stamp
        elif status == STATUS_SKIPPED:
            rec["skips"] = int(rec.get("skips", 0)) + 1
            # Deliberately does NOT touch consecutive_idle — see note_skipped.
        elif not reports_production:
            # Ran fine, but this engine does not declare production. Liveness and
            # errors only — accruing an idle streak here would flag every
            # uninstrumented engine as dead and drown the real signal.
            rec["reports_production"] = False
        else:
            rec["reports_production"] = True
            rec["last_produced"] = produced
            if produced > 0:
                rec["last_produced_at"] = stamp
                rec["consecutive_idle"] = 0
                rec["total_produced"] = int(rec.get("total_produced", 0)) + produced
            else:
                rec["consecutive_idle"] = int(rec.get("consecutive_idle", 0)) + 1

        data[engine] = rec
        _save(data)
        return rec
    except Exception as e:  # noqa: BLE001
        logger.warning(f"engine heartbeat record failed for {engine}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------

def get_heartbeats() -> dict[str, Any]:
    """Every engine's heartbeat, newest-run first. {} on any error."""
    data = _load()
    if not isinstance(data, dict):
        return {}
    return dict(
        sorted(data.items(), key=lambda kv: str(kv[1].get("last_ran_at", "")), reverse=True)
    )


def _parse_ts(value: Any) -> datetime | None:
    """Parse a stored isoformat stamp. None on anything unparseable."""
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _registered_engines() -> dict[str, dict[str, Any]]:
    """What the scheduler says SHOULD be running: name -> cooldown/enabled/paused.

    Read through a guarded lazy import: this module must stay importable and
    useful with no scheduler present (tests, scripts, a bare API process), and an
    observability layer that hard-depends on the thing it observes is not one.
    An empty dict simply degrades this view to the heartbeats-only behaviour.
    """
    try:
        from tools.scheduler import (
            SCHEDULED_TASKS,
            get_scheduler_cooldowns,
            get_scheduler_settings,
            scheduler,
        )
        cooldowns = get_scheduler_cooldowns()
        settings = get_scheduler_settings()
        paused = scheduler.get_paused_tasks()
        return {
            name: {
                "cooldown_s": float(cooldowns.get(name, default_cooldown)),
                "enabled": bool(settings.get(name, True)),
                "circuit_open": name in paused,
            }
            for name, _fn, default_cooldown, _timeout in SCHEDULED_TASKS
        }
    except Exception as e:  # noqa: BLE001 — never let the ops view break
        logger.debug(f"engine registry unavailable: {e}")
        return {}


def _instrumented_since(beats: dict[str, Any]) -> datetime | None:
    """Roughly when this store started recording — the earliest run it holds.

    Used as a grace window for "registered but never seen". On a fresh store
    (first deploy, a wiped file) EVERY engine is unseen, and flagging ten engines
    the minute instrumentation lands is the cries-wolf failure this module's own
    docstring warns about — and shipped with once already.
    """
    stamps = [
        ts for ts in (_parse_ts(r.get("last_ran_at")) for r in beats.values()
                      if isinstance(r, dict))
        if ts is not None
    ]
    return min(stamps) if stamps else None


def get_engine_health(idle_threshold: int = 10) -> dict[str, Any]:
    """The ops view: heartbeats, the scheduler's roster, and what to look at.

    The module's contract is that an engine proves it RAN *and* that it DID
    something. `concerning` therefore covers both halves:

      - **stopped** — registered with the scheduler but missing from the store
        entirely, or a `last_ran_at` several cooldowns old. The heartbeat of an
        engine that stops being invoked FREEZES on its last healthy record, so
        without this check a dead engine reads as `ok` forever. The circuit
        breaker (scheduler `MAX_CONSECUTIVE_FAILURES`) manufactures exactly that
        state on purpose, so a paused task is surfaced here by name rather than
        resting on the single alert that fired when it tripped.
      - **idle** — the original signal: it keeps running its logic and keeps
        producing nothing.

    `idle_threshold` is intentionally generous and staleness carries both a
    multiplier and a floor: a genuinely quiet engine (no watch conditions armed,
    no band crossed) is normal, and an instrument that cries wolf stops being
    read. Engines that errored on their most recent run are always surfaced;
    engines disabled in config are never surfaced, for the same reason a
    `skipped` run accrues no idle streak — it declined to work by instruction.
    """
    beats = get_heartbeats()
    registry = _registered_engines()
    since = _instrumented_since(beats)
    now = datetime.now()
    concerning = []

    for name in list(registry) + [n for n in beats if n not in registry]:
        reg = registry.get(name) or {}
        rec = beats.get(name)
        detail = (rec or {}).get("last_detail", "")

        if reg.get("circuit_open"):
            concerning.append({
                "engine": name,
                "why": "circuit breaker OPEN — paused until tomorrow (US/Eastern) or a restart",
                "detail": detail,
            })
            continue

        # Disabled by config is a decision, not a fault.
        if reg and not reg.get("enabled", True):
            continue

        cooldown = float(reg.get("cooldown_s") or 0)
        stale_after = max(STALE_MULTIPLIER * cooldown, MIN_STALE_SECONDS)

        if rec is None:
            # Only a finding once the store has been recording longer than this
            # engine's own cooldown — before that it simply has not come up yet.
            if reg and since is not None and (now - since).total_seconds() > cooldown:
                concerning.append({
                    "engine": name,
                    "why": "registered with the scheduler but has NEVER reported a run",
                    "detail": f"instrumented since {since.isoformat(timespec='seconds')}",
                })
            continue

        status = rec.get("last_status")
        idle = int(rec.get("consecutive_idle", 0) or 0)
        last_ran = _parse_ts(rec.get("last_ran_at"))
        age_s = (now - last_ran).total_seconds() if last_ran else None

        if status in (STATUS_ERROR, STATUS_TIMEOUT):
            concerning.append({"engine": name, "why": f"last run {status}", "detail": detail})
        elif reg and age_s is not None and age_s > stale_after:
            concerning.append({
                "engine": name,
                "why": (f"has not run in {int(age_s // 60)} min — "
                        f"expected every {int(cooldown // 60) or 1} min"),
                "detail": detail,
            })
        elif idle >= idle_threshold and rec.get("last_produced_at") is None:
            concerning.append({
                "engine": name,
                "why": f"{idle} consecutive runs with no output, and it has NEVER produced",
                "detail": detail,
            })
        elif idle >= idle_threshold:
            concerning.append({
                "engine": name,
                "why": f"{idle} consecutive runs with no output since {rec.get('last_produced_at')}",
                "detail": detail,
            })

    return {
        "engines": beats,
        "registry": registry,
        "concerning": concerning,
        "checked_at": _now(),
    }
