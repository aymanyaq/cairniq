"""Persistent LLM usage budget — a restart-safe circuit breaker.

The in-memory cost tracker (agent/cost_tracker.py) resets every process start, so
it cannot stop a *restart* storm (each respawn zeroes the counter). This module
persists rolling per-hour and per-day usage to disk so a runaway — whether an
in-process loop or a crash/restart loop — trips the same breaker and is stopped.

Two enforcement points consume this:
  - a SOFT gate at the chat entry point (refuses politely when over budget), and
  - the external watchdog (scripts/cairniq_watchdog.py), which HARD-kills the
    server when a higher ceiling is breached.

Caps are env-tunable (CAD + call counts). A cap of 0 means "no limit" for that
dimension. Defaults are deliberately generous — high enough that normal
interactive use never trips them, low enough to catch a true runaway.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime

# Global (all-profiles) usage file: a runaway is process-wide, so protect the
# total, not one profile.
_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "user_data", "llm_budget.json"
)
_lock = threading.Lock()


def _f(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "").strip())
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def soft_caps() -> dict:
    """In-app refusal thresholds (the polite stop)."""
    return {
        "calls_per_hour": _f("AIDLC_LLM_MAX_CALLS_PER_HOUR", 600),
        "cost_cad_per_day": _f("AIDLC_LLM_MAX_SPEND_CAD_PER_DAY", 40.0),
    }


def hard_caps() -> dict:
    """Watchdog kill thresholds — default 2.5x the soft caps so the soft gate
    always trips first. Set explicitly to override."""
    sc = soft_caps()
    return {
        "calls_per_hour": _f("AIDLC_LLM_KILL_CALLS_PER_HOUR", sc["calls_per_hour"] * 2.5),
        "cost_cad_per_day": _f("AIDLC_LLM_KILL_SPEND_CAD_PER_DAY", sc["cost_cad_per_day"] * 2.5),
    }


def _now() -> datetime:
    return datetime.now()


def _keys(dt: datetime) -> tuple[str, str]:
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%dT%H")


def _read() -> dict:
    try:
        with open(_STATE_PATH, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write(d: dict) -> None:
    tmp = f"{_STATE_PATH}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    os.replace(tmp, _STATE_PATH)


def _rolled(d: dict, day: str, hour: str) -> dict:
    """Return state with day/hour buckets reset if the clock has advanced."""
    if d.get("day") != day:
        d = {**d, "day": day, "day_calls": 0, "day_cost_cad": 0.0}
    if d.get("hour") != hour:
        d = {**d, "hour": hour, "hour_calls": 0, "hour_cost_cad": 0.0}
    return d


def record(cost_cad: float = 0.0, calls: int = 1) -> None:
    """Add one LLM call's usage to the persistent rolling buckets. Never raises."""
    try:
        day, hour = _keys(_now())
        with _lock:
            d = _rolled(_read(), day, hour)
            d["day_calls"] = int(d.get("day_calls", 0)) + calls
            d["hour_calls"] = int(d.get("hour_calls", 0)) + calls
            d["day_cost_cad"] = round(float(d.get("day_cost_cad", 0.0)) + (cost_cad or 0.0), 6)
            d["hour_cost_cad"] = round(float(d.get("hour_cost_cad", 0.0)) + (cost_cad or 0.0), 6)
            _write(d)
    except Exception:
        pass


def status() -> dict:
    """Current rolling usage (buckets rolled to 'now'). Safe to call anywhere."""
    day, hour = _keys(_now())
    with _lock:
        d = _rolled(_read(), day, hour)
    return {
        "day": day,
        "hour": hour,
        "day_calls": int(d.get("day_calls", 0)),
        "hour_calls": int(d.get("hour_calls", 0)),
        "day_cost_cad": round(float(d.get("day_cost_cad", 0.0)), 4),
        "hour_cost_cad": round(float(d.get("hour_cost_cad", 0.0)), 4),
    }


def _exceeds(caps: dict) -> str:
    """Return a human reason string if usage exceeds `caps`, else ''."""
    s = status()
    cph, cpd = caps["calls_per_hour"], caps["cost_cad_per_day"]
    if cph and s["hour_calls"] >= cph:
        return f"{s['hour_calls']} LLM calls this hour ≥ cap {int(cph)}"
    if cpd and s["day_cost_cad"] >= cpd:
        return f"${s['day_cost_cad']:.2f} CAD spent today ≥ cap ${cpd:.2f}"
    return ""


def over_soft_budget() -> str:
    return _exceeds(soft_caps())


def over_hard_budget() -> str:
    return _exceeds(hard_caps())
