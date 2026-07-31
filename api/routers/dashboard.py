import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agent.logger import log_to_component
from tools.portfolio_csv import get_portfolio_summary

router = APIRouter()

def _safe_short_string(value, max_len: int = 200):
    """Coerce ``value`` to a short string or None; long values (likely stack traces) become None."""
    if value is None:
        return None
    s = str(value)
    return s if len(s) <= max_len else None


def _safe_number(value):
    """Coerce ``value`` to a number; non-numerics become None so they can't carry exception text."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_jsonable(value, kind: str = "dict"):
    """Re-construct ``value`` from a fixed shape so exception text can't sneak through.

    For ``kind="dict"``: returns a dict whose values are all coerced to short strings,
    numbers, booleans, or None — recursively at one level.
    For ``kind="list"``: returns a list of similarly-coerced dicts.
    """
    if kind == "dict":
        if not isinstance(value, dict):
            return {}
        out: dict = {}
        for k, v in value.items():
            key = str(k)[:80]
            if isinstance(v, (int, float, bool)) or v is None:
                out[key] = v
            elif isinstance(v, dict):
                out[key] = _safe_jsonable(v, kind="dict")
            elif isinstance(v, list):
                out[key] = _safe_jsonable(v, kind="list")
            else:
                out[key] = _safe_short_string(v)
        return out
    if kind == "list":
        if not isinstance(value, list):
            return []
        return [_safe_jsonable(item, kind="dict") if isinstance(item, dict) else _safe_short_string(item) for item in value]
    return None


_SAFE_SYNC_ERROR_MESSAGES = {
    "questrade": "Questrade sync failed; using last-known-good data where available.",
    "alpaca": "Alpaca sync failed; using last-known-good data where available.",
    "token": "A broker token is invalid or expired. Re-link the account in Settings.",
    "default": "A broker sync failed. Re-link affected accounts in Settings.",
}


def _scrub_sync_errors(raw_errors) -> list[str]:
    """Map internal sync-error strings to a small set of safe, user-facing messages.

    The internal error strings may embed exception traces and file paths; we never
    forward them to the HTTP client. Membership against a fixed table acts as a
    sanitizer barrier — only literal entries from ``_SAFE_SYNC_ERROR_MESSAGES`` reach
    the response.
    """
    if not raw_errors or not isinstance(raw_errors, list):
        return []
    seen: set[str] = set()
    safe: list[str] = []
    for entry in raw_errors:
        text = str(entry or "").lower()
        if "questrade" in text:
            key = "questrade"
        elif "alpaca" in text:
            key = "alpaca"
        elif "token" in text or "auth" in text or "unauthor" in text:
            key = "token"
        else:
            key = "default"
        if key not in seen:
            seen.add(key)
            safe.append(_SAFE_SYNC_ERROR_MESSAGES[key])
    return safe


@router.get("/api/goal_projection")
def get_goal_projection_api(simulations: int = 5000):
    """Are we on track? (Roadmap 4.5 slice 4 / the goal-tracking panel.)

    Monte Carlo bands against the stated target, the goal-funded success rate,
    and the return the plan actually requires. Returns `available: false` with a
    `missing` list whenever the goal, its contribution, or the portfolio value is
    unset, so the UI prompts for the specific gap instead of drawing an empty
    chart — nothing here invents a target, a contribution, or a return.

    Deterministic and network-free apart from the portfolio summary's own cache;
    no LLM. Simulation count is bounded so a query string cannot pin the CPU.
    """
    from tools.goal_projection import build_goal_projection

    return JSONResponse(
        build_goal_projection(num_simulations=max(500, min(simulations, 20000)))
    )


@router.get("/api/contribution_sensitivity")
def get_contribution_sensitivity_api(simulations: int = 5000):
    """What changing the annual contribution does to the goal.

    Same inputs and the same `available: false` / `missing` contract as
    `/api/goal_projection`, because it reads the identical stored goal — a caller
    should not have to learn two shapes.

    Every scenario runs on ONE shared seed, so the differences between rows are
    attributable to the contribution rather than to the RNG. Read `comparable`
    before quoting any single row's absolute success rate.

    Deterministic and network-free apart from the portfolio summary's own cache;
    no LLM. Simulation count is bounded so a query string cannot pin the CPU —
    and this endpoint runs one simulation PER SCENARIO, so the bound matters more
    here than it does on the single-projection route.
    """
    from tools.goal_projection import build_contribution_sensitivity

    return JSONResponse(
        build_contribution_sensitivity(
            num_simulations=max(500, min(simulations, 10000))
        )
    )


@router.get("/api/weekly_review")
def get_weekly_review_api():
    """The weekly one-page review, as data.

    Read-only by construction: every section reads a surface that already exists,
    the market briefing comes from cache and is never generated, and there is no
    LLM anywhere in the path. A report that starts work can time out, cost money,
    or quietly change the state it is describing.

    Sections are always all present — a section with nothing to report says so.
    That is the contract the module is built around, not a rendering detail.
    """
    from tools.weekly_review import build_weekly_review

    return JSONResponse(build_weekly_review())


@router.get("/api/dashboard-data")
def get_dashboard_api():
    try:
        summary = get_portfolio_summary()
        if "error" in summary:
            # Never forward the raw exception text — it may include stack traces or
            # file system paths. Log full detail server-side and return a generic note.
            log_to_component(
                "server",
                "Dashboard",
                f"Portfolio summary unavailable: {summary.get('error')}",
                level=logging.WARNING,
            )
            return JSONResponse({"error": "Portfolio data is temporarily unavailable."}, status_code=503)

        # Format Asset Allocation (Top 6 assets + Others)
        holdings = summary.get("holdings", [])
        holdings.sort(key=lambda x: x.get("value_cad", 0), reverse=True)

        asset_labels = []
        asset_values = []
        other_value = 0.0

        for i, item in enumerate(holdings):
            val = item.get("value_cad", 0)
            if i < 6:
                asset_labels.append(item.get("symbol", "Unk"))
                asset_values.append(val)
            else:
                other_value += val

        if other_value > 0:
            asset_labels.append("Others")
            asset_values.append(other_value)

        # Build the response from a fixed allow-list of typed fields. Each value is
        # passed through ``_safe_jsonable`` which copies dict/list shapes but rejects
        # any string longer than 200 chars (stack traces and exception messages are
        # both far longer than any legitimate label/identifier we surface here).
        data = {
            "summary": _safe_jsonable(summary.get("summary"), kind="dict"),
            "liquidity": _safe_jsonable(summary.get("liquidity"), kind="dict"),
            "accounts": _safe_jsonable(summary.get("accounts"), kind="list"),
            "top_winners": _safe_jsonable(summary.get("top_winners"), kind="list"),
            "top_losers": _safe_jsonable(summary.get("top_losers"), kind="list"),
            "sync_errors": _scrub_sync_errors(summary.get("sync_errors")),
            "is_stale": bool(summary.get("is_stale", False)),
            "last_sync_time": _safe_short_string(summary.get("last_sync_time")),
            "lkg_total_cad": _safe_number(summary.get("lkg_total_cad")),
            "allocation": {
                "labels": [str(label)[:40] for label in asset_labels],
                "values": [_safe_number(v) for v in asset_values],
            },
        }

        # Event Radar Summary (Theme 3.5b)
        try:
            from tools.event_radar import build_event_radar_cached
            radar = build_event_radar_cached()
            upcoming_7d = [e for e in (radar.get("events") or []) if e.get("days_until", 999) <= 7]
            data["event_radar_summary"] = {
                "total_upcoming_7d": len(upcoming_7d),
                "upcoming_events": upcoming_7d[:3],
            }
        except Exception:
            data["event_radar_summary"] = {"total_upcoming_7d": 0, "upcoming_events": []}

        return JSONResponse(data)
    except Exception as e:
        log_to_component("server", "Dashboard", f"Error loading dashboard: {e}", level=logging.ERROR)
        return JSONResponse({"error": "Failed to load dashboard data"}, status_code=500)
