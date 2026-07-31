import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.logger import log_to_component
from tools.alerts import get_alerts, get_unread_count, mark_read
from tools.watch_conditions import cancel_condition, get_watch_summary

router = APIRouter()


class MarkReadRequest(BaseModel):
    ids: list[str] | None = None
    all: bool = False


@router.get("/api/alerts")
def list_alerts(limit: int = 50, unread: bool = False):
    """Alerts inbox for the current profile, newest first.

    Read-only and cheap: the iOS app can poll this, and the base template
    calls it (limit=1) purely for the unread badge count.
    """
    limit = max(1, min(limit, 200))
    return JSONResponse({
        "alerts": get_alerts(limit=limit, unread_only=unread),
        "unread_count": get_unread_count(),
    })


@router.post("/api/alerts/mark_read")
def mark_alerts_read(req: MarkReadRequest):
    marked = mark_read(alert_ids=req.ids, all_alerts=req.all)
    if marked:
        log_to_component("server", "Alerts", f"Marked {marked} alert(s) read", level=logging.DEBUG)
    return JSONResponse({"marked": marked, "unread_count": get_unread_count()})


@router.get("/api/alerts/delivery")
def alert_delivery_latency():
    """How long alerts wait between being raised and being read (7.1 number 4).

    The counterpart to `/api/availability`: that one shows whether the host was up
    to send, this shows whether anyone was there to receive. Per-profile, because
    an inbox belongs to one person — which is exactly why it is a separate surface
    rather than a field on the global availability report.

    A mark-all click is counted in `bulk_read` and excluded from the timing, so
    the median never reports a button press as a reading time. Read `status`
    first: `no_data` means no stamped read exists yet, which is UNKNOWN latency
    and not fast latency.

    Read-only, network-free, no LLM.
    """
    from tools.alerts import get_delivery_latency
    return JSONResponse(get_delivery_latency())


@router.get("/api/alerts/missed")
def missed_alert_replay():
    """Crossings a measured outage window could have hidden (7.1 number 3).

    The third of 7.1's four numbers, and the last computable one. It reads the
    SAME dated gaps `/api/availability` reports and replays the market data for
    them against the conditions that were armed at the time. Per-profile for the
    same reason delivery latency is: a watch condition belongs to one person, and
    a count summed across profiles would answer nobody's question.

    Read `measurable_windows` against `windows` BEFORE reading
    `missed_crossings`. A bar interval wider than the gap cannot see inside it,
    and such a window reports UNMEASURABLE rather than clean — a total of zero
    across windows nobody could replay is not a clean record.

    The count is an UPPER bound: it reads each bar's high and low, so an intraday
    spike counts, while the live evaluator samples every 30 minutes and would
    have missed some of them anyway.

    Network-bound (it fetches historical bars) and slow on a long record.
    """
    from tools.missed_alerts import get_missed_alerts

    return JSONResponse(get_missed_alerts())


@router.get("/api/watch_conditions")
def list_watch_conditions():
    """Watch conditions the advisor has committed to for this profile (Roadmap 3.3).

    Read-only and network-free — it reports each condition's last checked value,
    never a fresh quote. The store is otherwise invisible between fires, and an
    engine whose pending state cannot be inspected is exactly how silent
    no-op bugs survive (see the dark funnel signal log, the never-scoring
    recommendation ledger).
    """
    return JSONResponse(get_watch_summary())


@router.post("/api/watch_conditions/{condition_id}/cancel")
def cancel_watch_condition(condition_id: str):
    cancelled = cancel_condition(condition_id)
    return JSONResponse({"cancelled": cancelled})


@router.get("/api/engine_health")
def engine_health(idle_threshold: int = 10):
    """Per-engine liveness for every scheduled background job (Roadmap 2.5).

    The one view that answers "is anything quietly dead?". `concerning` lists
    engines that errored on their last run, that STOPPED running (missing from
    the store, several cooldowns overdue, or held down by the circuit breaker),
    or that keep running their logic and keep producing nothing — the state that
    hid the dark funnel signal log, the never-scoring recommendation ledger, and
    3.3 harvesting zero conditions for a full trading day. `registry` is the
    scheduler's own roster, so an engine that never started is visible as an
    absence. Global (not per-profile) because the scheduler ticks under
    `_unbound`: engines are process-wide, so their health is too.

    Read-only, network-free, no LLM.
    """
    from tools.engine_heartbeat import get_engine_health
    return JSONResponse(get_engine_health(idle_threshold=max(1, min(idle_threshold, 500))))


@router.get("/api/availability")
def availability():
    """Measured coverage of the trading-day window this product must be up for (7.1).

    Sits beside `engine_health` because it answers the question that one cannot:
    `engine_health` shows whether an engine ran when the process was alive, and
    this shows how much of the window the process was alive at all. A crossing
    missed to a sleeping host leaves no heartbeat behind to look concerning.

    `window_coverage_pct` is an UPPER BOUND, and `open_measurements` says why —
    the probe checks that :8000 is bound, so a surface serving 500s to every
    request still scores as available. Global, not per-profile: the host is
    either up or it is not.

    Read-only, network-free, no LLM.
    """
    from tools.availability import get_availability_report
    return JSONResponse(get_availability_report())
