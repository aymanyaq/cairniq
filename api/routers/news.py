import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agent.logger import log_to_component
from api.background import (
    catalyst_is_stuck,
    get_catalyst_elapsed_seconds,
    get_news_elapsed_seconds,
    get_priority_elapsed_seconds,
    get_pulse_elapsed_seconds,
    is_catalyst_running,
    is_news_running,
    is_priority_running,
    is_pulse_running,
    news_is_stuck,
    priority_is_stuck,
    pulse_is_stuck,
    start_catalyst_scan,
    start_news_fetch,
    start_priority_precompute,
    start_pulse_fetch,
)
from tools.daily_cache import get_cached

router = APIRouter()

@router.get("/api/news-feed")
def get_news_feed(force: bool = False):
    """Get primary news & event feed (daily cached, generates in background)."""

    # FORCE refresh: always cancel old fetch and start a new one
    if force:
        log_to_component("server", "News", "Force-refresh requested — cancelling any in-progress fetch.")
        start_news_fetch(force=True)
        return JSONResponse({"status": "fetching", "elapsed_seconds": 0})

    # If already fetching AND not stuck, tell the client to wait
    if is_news_running() and not news_is_stuck():
        elapsed = get_news_elapsed_seconds()
        return JSONResponse({"status": "fetching", "elapsed_seconds": elapsed})

    # If not forcing and not currently generating, return cache if available
    cached = get_cached("news_feed")
    if cached and "markdown" in cached:
        return JSONResponse(cached)

    # Either not started yet, or the previous attempt timed out — kick off a fresh run
    if news_is_stuck():
        log_to_component("server", "News", "Previous fetch appears stuck — restarting fresh.", level=logging.WARNING)

    start_news_fetch(force=False)
    return JSONResponse({"status": "fetching", "elapsed_seconds": 0})


@router.get("/api/market-pulse")
def get_market_pulse(force: bool = False):
    """Get daily Market Pulse briefing (daily cached, generates in background)."""

    # FORCE refresh: start a new generation
    if force:
        log_to_component("server", "Pulse", "Force-refresh requested.")
        start_pulse_fetch(force=True)
        return JSONResponse({"status": "fetching", "elapsed_seconds": 0})

    # If already fetching AND not stuck, tell the client to wait
    if is_pulse_running() and not pulse_is_stuck():
        elapsed = get_pulse_elapsed_seconds()
        return JSONResponse({"status": "fetching", "elapsed_seconds": elapsed})

    # If not forcing and not currently generating, return cache if available
    cached = get_cached("market_pulse")
    if cached and "regime" in cached:
        return JSONResponse(cached)

    # Either not started yet, or the previous attempt timed out — kick off fresh
    if pulse_is_stuck():
        log_to_component("server", "Pulse", "Previous generation appears stuck — restarting.", level=logging.WARNING)

    start_pulse_fetch(force=False)
    return JSONResponse({"status": "fetching", "elapsed_seconds": 0})


@router.get("/api/priority")
def get_today_priority(force: bool = False):
    """Get the precomputed Today's Priority brief (Theme 3.1 — daily cached).

    Cost-respectful like /api/catalysts: a plain GET returns the cached brief
    only and NEVER auto-starts a run (a run is a full DeepReasoning graph pass
    — the most expensive single product in the app). The scheduler precomputes
    it pre-market; ``force=true`` regenerates on demand.
    """
    if force:
        log_to_component("server", "Priority", "Refresh requested — starting priority precompute.")
        start_priority_precompute()
        return JSONResponse({"status": "fetching", "elapsed_seconds": 0})

    if is_priority_running() and not priority_is_stuck():
        return JSONResponse({"status": "fetching", "elapsed_seconds": get_priority_elapsed_seconds()})

    cached = get_cached("today_priority")
    if cached and "markdown" in cached:
        return JSONResponse(cached)

    # No cache and not forcing: do not auto-spend — let the client prompt a refresh.
    return JSONResponse({"status": "empty"})


@router.get("/api/catalysts")
def get_catalysts(force: bool = False):
    """Get the ranked, two-lane catalyst list (Catalyst Engine — Layer 2).

    Cost-respectful divergence from /api/news-feed: a plain GET returns cached
    catalysts only and NEVER auto-starts a scan (a scan adds ~2 Sonnet calls).
    The scan runs only on an explicit ``force=true`` refresh.
    """
    if force:
        log_to_component("server", "Catalyst", "Refresh requested — starting catalyst scan.")
        start_catalyst_scan()
        return JSONResponse({"status": "fetching", "elapsed_seconds": 0})

    if is_catalyst_running() and not catalyst_is_stuck():
        return JSONResponse({"status": "fetching", "elapsed_seconds": get_catalyst_elapsed_seconds()})

    cached = get_cached("catalysts")
    if cached and "catalysts" in cached:
        return JSONResponse(cached)

    # No cache and not forcing: do not auto-spend — let the client prompt a refresh.
    return JSONResponse({"status": "empty"})


@router.get("/api/catalysts/scenario/{catalyst_id}")
def get_catalyst_scenario(catalyst_id: str):
    """Cached Layer-3 scenario for one catalyst (Catalyst Engine — auto-escalation).

    Serves the markdown generated by the background auto-escalation, keyed by catalyst
    id. Read-only and never triggers an LLM call — catalysts without a cached scenario
    return not_found, and the UI falls back to the manual "Analyze impact →" drill-down
    (which runs the engine through chat).
    """
    scenarios = get_cached("catalyst_scenarios") or {}
    scenario = scenarios.get(catalyst_id)
    if isinstance(scenario, dict) and scenario.get("markdown"):
        return JSONResponse({"status": "ok", "id": catalyst_id, **scenario})
    return JSONResponse({"status": "not_found", "id": catalyst_id})


@router.get("/api/market-pulse/history")
def get_market_pulse_history(days: int = 30):
    """Get regime history for sparkline chart."""
    from tools.market_sentinel import get_regime_history
    return JSONResponse(get_regime_history(days))
