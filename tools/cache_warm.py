"""Dashboard cache pre-warming.

The dashboard's three headline panels — portfolio summary, asset allocation, top
performers — all come from one `/api/dashboard-data` request, and that request is
fast or slow depending purely on whether the profile's daily cache is warm.
Measured on a 10-symbol demo book: **31.7s cold, 12ms warm.**

The caches themselves work fine. The problem this module solves is that nothing
ever warmed them, so the person paying the cold path was always the human who
just opened the page. Both TTLs are shorter than the interval at which anyone
actually reads a dashboard, and the daily cache is date-stamped per profile
(tools/daily_cache.py), so the first visit of every day was cold no matter how
the TTLs were tuned.

Deliberately zero-LLM and read-only. It calls the same two functions the
dashboard calls and keeps their results; it computes nothing the request path
would not have computed itself, which is what makes it safe to run unattended.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def warm_profile() -> dict[str, Any]:
    """Warm the active profile's dashboard caches. Never raises.

    Must be called with a profile bound (see tools.user_profile.run_under_profile)
    — the daily cache is profile-namespaced, so an unbound call warms the empty
    '_unbound' profile and every real one stays cold.
    """
    from tools.event_radar import build_event_radar_cached
    from tools.portfolio_csv import get_portfolio_summary

    out: dict[str, Any] = {"summary": False, "radar": False}

    # Ordered deliberately: the summary primes the FX rates and quote caches that
    # the radar's own load_portfolio() pass would otherwise fetch again.
    try:
        summary = get_portfolio_summary()
        out["summary"] = bool(summary) and not summary.get("error")
    except Exception as e:  # noqa: BLE001 — a warm failure must never surface to a caller
        logger.warning(f"cache warm: portfolio summary failed: {e}")

    try:
        radar = build_event_radar_cached()
        out["radar"] = isinstance(radar, dict) and not radar.get("error")
    except Exception as e:  # noqa: BLE001 — likewise; the summary half already landed
        logger.warning(f"cache warm: event radar failed: {e}")

    return out


def warm_all_profiles() -> dict[str, dict[str, Any]]:
    """Warm every real profile, one at a time. Never raises.

    Serial, not parallel: warming is a background convenience and the work it does
    is a broker sync plus a quote per holding. Running several profiles' worth of
    that concurrently would compete with whatever request the user is actually
    waiting on, which would defeat the point.
    """
    from tools.user_profile import list_available_profiles, run_under_profile

    results: dict[str, dict[str, Any]] = {}
    try:
        # Bind 'default' for the listing itself: list_available_profiles reads
        # get_active_profile() to flag the active entry, and calling it unbound
        # trips the multi-user guard.
        profiles = run_under_profile("default", list_available_profiles)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cache warm: could not list profiles: {e}")
        return results

    for entry in profiles:
        name = entry.get("name") if isinstance(entry, dict) else None
        # pytest_* are per-test scratch profiles whose directories are deleted at
        # teardown; '_unbound' is the guard's placeholder, not a real profile.
        if not name or name.startswith("pytest_") or name == "_unbound":
            continue
        try:
            results[name] = run_under_profile(name, warm_profile)
        except Exception as e:  # noqa: BLE001 — one bad profile must not abort the rest
            logger.warning(f"cache warm: profile {name} failed: {e}")
            results[name] = {"summary": False, "radar": False}

    return results


def warm_enabled() -> bool:
    """Whether pre-warming should run in this process.

    Independent of SCHEDULER_ENABLED, which is off by default and per-profile:
    warming is not background *work*, it is the same read the next request would
    perform anyway, and gating it behind an opt-in flag would leave the slow
    first visit in place for exactly the people who never found the flag.

    Off under pytest, and off when CAIRNIQ_CACHE_WARM is explicitly falsy — a
    suite run (or a laptop on a metered connection) should not be issuing broker
    syncs on a timer.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    val = str(os.environ.get("CAIRNIQ_CACHE_WARM", "true")).strip().lower()
    return val not in ("0", "false", "no", "off")
