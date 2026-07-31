import logging
import threading
import traceback
from datetime import datetime

from langchain_core.messages import HumanMessage

from agent.logger import log_to_component
from agent.nodes.news_analyst import news_analyst_node
from tools.user_profile import get_active_profile
from tools.user_profile import run_under_profile as _run_under_profile

# _run_under_profile is the canonical profile re-binding wrapper, now defined in
# tools.user_profile so the agent executors and background workers share one
# implementation. Re-exported here under its historical name for the start_*
# helpers below and the regression tests that import it.
__all__ = ["_run_under_profile"]

# ============================================================
# NEWS FEED BACKGROUND WORKER
# ============================================================
_news_fetch_started_at: datetime | None = None
_news_cancel_event: threading.Event = threading.Event()
_news_thread: threading.Thread | None = None
_NEWS_FETCH_TIMEOUT_MINUTES = 3


def news_is_stuck() -> bool:
    """Return True if the background fetch has been running too long (likely died)."""
    if _news_fetch_started_at is None:
        return False
    return (datetime.now() - _news_fetch_started_at).total_seconds() > _NEWS_FETCH_TIMEOUT_MINUTES * 60


def cancel_news_fetch():
    """Signal any in-progress news fetch to abort."""
    global _news_fetch_started_at, _news_thread
    _news_cancel_event.set()
    _news_fetch_started_at = None
    # Don't join — let the old thread die on its own
    _news_thread = None


def run_news_agent_in_background(force: bool = False):
    """Run in a dedicated thread so it's fully isolated from the ASGI event-loop."""
    global _news_fetch_started_at, _news_thread
    try:
        from tools.daily_cache import set_cached
        if force:
            from tools.cache import clear_cache
            clear_cache()
            log_to_component("server", "News", "Forcing fresh run: In-memory cache cleared.")

        log_to_component("server", "News", "Starting background news Intelligence Report generation...")

        # Check cancellation before the expensive LLM call
        if _news_cancel_event.is_set():
            log_to_component("server", "News", "Cancelled before LLM call.", level=logging.WARNING)
            return

        state = {"messages": [HumanMessage(content="Provide a comprehensive market intelligence report covering today's breaking news, the Canadian market, and key thematic trends.")]}
        result = news_analyst_node(state)

        # Check cancellation after the LLM call
        if _news_cancel_event.is_set():
            log_to_component("server", "News", "Cancelled after LLM call — discarding result.", level=logging.WARNING)
            return

        if "messages" not in result or not result["messages"]:
            log_to_component("server", "News", "FAILED: No messages returned from node.", level=logging.ERROR)
            return

        final_msg = result["messages"][-1]

        set_cached("news_feed", {
            "markdown": final_msg.content,
            "generated_at": datetime.now().isoformat()
        })
        log_to_component("server", "News", f"Background fetch complete ({len(final_msg.content)} chars).")

        # Proactive catalyst scan (spec §5 P4-lite): fresh news just landed, so refresh
        # the catalyst lanes too — gated by config + TTL inside the helper, so this does
        # NOT fire on every refresh (the v0.3 cost concern). Defined below; resolved at
        # call time.
        maybe_start_catalyst_scan_after_news()
    except Exception as e:
        log_to_component("server", "News", f"Background fetch error: {e}", level=logging.ERROR)
        log_to_component("server", "News", traceback.format_exc(), level=logging.ERROR)
    finally:
        # Only clear the timestamp if WE are still the active fetch
        if _news_thread is threading.current_thread():
            _news_fetch_started_at = None


def start_news_fetch(force: bool = False):
    """Kick off a news fetch in a dedicated daemon thread."""
    global _news_fetch_started_at, _news_thread, _news_cancel_event

    # Cancel any in-progress fetch
    cancel_news_fetch()

    # Reset the cancel event for the new run
    _news_cancel_event = threading.Event()
    _news_fetch_started_at = datetime.now()

    t = threading.Thread(
        target=_run_under_profile,
        args=(get_active_profile(), run_news_agent_in_background, force),
        name="news-feed-worker",
        daemon=True,
    )
    _news_thread = t
    t.start()


# ============================================================
# MARKET PULSE BACKGROUND WORKER
# ============================================================
_pulse_fetch_started_at: datetime | None = None
_pulse_thread: threading.Thread | None = None
_PULSE_FETCH_TIMEOUT_MINUTES = 3


def pulse_is_stuck() -> bool:
    """Return True if the background pulse generation has been running too long."""
    if _pulse_fetch_started_at is None:
        return False
    return (datetime.now() - _pulse_fetch_started_at).total_seconds() > _PULSE_FETCH_TIMEOUT_MINUTES * 60


def _run_pulse_in_background(force: bool = False):
    """Run the Market Pulse generation in a dedicated daemon thread."""
    global _pulse_fetch_started_at, _pulse_thread
    try:
        from tools.daily_cache import set_cached
        from tools.market_sentinel import generate_market_pulse

        log_to_component("server", "Pulse", "Starting Market Pulse generation...")

        if force:
            # Clear today's cached pulse to force regeneration
            log_to_component("server", "Pulse", "Force refresh requested.")

        briefing = generate_market_pulse()

        if briefing and not briefing.get("error"):
            set_cached("market_pulse", briefing)
            log_to_component("server", "Pulse", f"Generation complete — Regime: {briefing.get('regime')} ({briefing.get('regime_score')})")
        else:
            log_to_component("server", "Pulse", f"Generation returned error: {briefing.get('error', 'unknown')}", level=logging.ERROR)

    except Exception as e:
        log_to_component("server", "Pulse", f"Background generation error: {e}", level=logging.ERROR)
        log_to_component("server", "Pulse", traceback.format_exc(), level=logging.ERROR)
    finally:
        _pulse_fetch_started_at = None


def start_pulse_fetch(force: bool = False):
    """Kick off a pulse generation in a dedicated daemon thread."""
    global _pulse_fetch_started_at, _pulse_thread

    _pulse_fetch_started_at = datetime.now()

    t = threading.Thread(
        target=_run_under_profile,
        args=(get_active_profile(), _run_pulse_in_background, force),
        name="market-pulse-worker",
        daemon=True,
    )
    _pulse_thread = t
    t.start()

# ============================================================
# TODAY'S PRIORITY PRECOMPUTE WORKER  (Theme 3.1)
# ============================================================
# Drives the SAME [QuickAction name=priority] marker the dashboard button sends
# through the full reasoning graph (Supervisor → DeepReasoning → RiskManager),
# so the precomputed brief is byte-for-byte the product a chat run would give —
# including the risk verdict and the coverage-checklist tool floor. The result
# is cached per profile via daily_cache ("today_priority") for cold dashboard
# reads through GET /api/priority.
_priority_runs: dict[str, datetime] = {}
_priority_thread: threading.Thread | None = None
# A heavy DeepReasoning run is two tool cycles + synthesis + the RiskManager
# judge (each leg can take minutes) — treat as stuck only well past that.
_PRIORITY_TIMEOUT_MINUTES = 12


def priority_is_stuck(profile: str | None = None) -> bool:
    if profile is None:
        profile = get_active_profile()
    started_at = _priority_runs.get(profile)
    if started_at is None:
        return False
    return (datetime.now() - started_at).total_seconds() > _PRIORITY_TIMEOUT_MINUTES * 60


def get_priority_elapsed_seconds(profile: str | None = None) -> int:
    if profile is None:
        profile = get_active_profile()
    started_at = _priority_runs.get(profile)
    if started_at:
        return int((datetime.now() - started_at).total_seconds())
    return 0


def is_priority_running(profile: str | None = None) -> bool:
    if profile is None:
        profile = get_active_profile()
    return profile in _priority_runs


def _compose_priority_markdown(final_state: dict) -> str:
    """User-visible markdown from a finished graph run: every AI message after
    the last human turn (DeepReasoning brief + RiskManager assessment), with
    node prefixes and <thinking> blocks stripped — the same messages a chat
    run would emit."""
    from langchain_core.messages import AIMessage as _AIMessage
    from langchain_core.messages import HumanMessage as _HumanMessage

    from agent.utils import extract_visible_text

    messages = (final_state or {}).get("messages", [])
    last_human_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, _HumanMessage):
            last_human_idx = i

    parts = []
    for msg in messages[last_human_idx + 1:]:
        if not isinstance(msg, _AIMessage):
            continue
        text = extract_visible_text(getattr(msg, "content", ""), strip_node_prefix=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def run_priority_precompute_in_background() -> bool:
    """Run the Today's Priority quick action through the reasoning graph and
    cache the result as `today_priority` for the ACTIVE profile. Synchronous —
    the scheduler calls it per profile inside asyncio.to_thread, and
    start_priority_precompute() wraps it in a daemon thread for the API path.
    Returns True when a brief was cached."""
    profile = get_active_profile()
    _priority_runs[profile] = datetime.now()
    try:
        from agent.graph import build_graph
        from tools.daily_cache import set_cached

        log_to_component("server", "Priority", "Starting Today's Priority precompute...")

        # Fresh, memoryless graph: the precompute must never touch chat-thread
        # checkpoints. response_length mirrors the chat default (concise) — the
        # priority prompt itself dictates the brief's structure.
        agent = build_graph(use_memory=False)
        initial_state = {
            "messages": [HumanMessage(content="[DeepReasoning] [QuickAction name=priority]")],
            "ghost": False,
            "data_context": {},
            "risk_retry_count": 0,
        }
        config = {
            "configurable": {
                "thread_id": f"priority-precompute-{get_active_profile()}",
                "profile": get_active_profile(),
                "response_length": "Concise (Save $$)",
            }
        }
        final_state = agent.invoke(initial_state, config=config)

        markdown = _compose_priority_markdown(final_state)
        if not markdown.strip():
            log_to_component("server", "Priority", "FAILED: run produced no visible output.", level=logging.ERROR)
            return False

        # Roadmap 3.3: the NEXT-CHECK trigger board this brief just wrote is the
        # single richest source of advisor-authored conditions, and it runs every
        # trading morning whether or not anyone opens the app. Harvest before the
        # side-channel is stripped from the cached markdown.
        from tools.watch_conditions import capture_watch_conditions, strip_watch_blocks
        capture_watch_conditions(markdown, source="priority")
        markdown = strip_watch_blocks(markdown).strip()

        # Invariant: the machine-readable side-channel must NEVER survive into the
        # brief the dashboard renders. A residual <watch tag means the sanitizer
        # chain changed shape under us (the 2026-07-23 regression shipped raw
        # trigger JSON to the user); fail loudly rather than cache it.
        if "<watch" in markdown.lower():
            log_to_component(
                "server", "Priority",
                "Watch side-channel survived stripping into the cached brief -- "
                "sanitizer regression; not the clean text the user should see.",
                level=logging.ERROR,
            )

        set_cached("today_priority", {
            "markdown": markdown,
            "generated_at": datetime.now().isoformat(),
        })
        log_to_component("server", "Priority", f"Precompute complete ({len(markdown)} chars).")
        return True
    except Exception as e:
        log_to_component("server", "Priority", f"Precompute error: {e}", level=logging.ERROR)
        log_to_component("server", "Priority", traceback.format_exc(), level=logging.ERROR)
        return False
    finally:
        _priority_runs.pop(profile, None)


def start_priority_precompute() -> None:
    """Kick off a priority precompute for the active profile in a daemon thread.

    NOT auto-triggered by a plain GET (a run is a full DeepReasoning graph pass
    — the most expensive single product in the app). Call this from the
    explicit ``force=true`` refresh or the scheduler."""
    global _priority_thread
    profile = get_active_profile()
    if is_priority_running(profile) and not priority_is_stuck(profile):
        return
    t = threading.Thread(
        target=_run_under_profile,
        args=(profile, run_priority_precompute_in_background),
        name="priority-precompute-worker",
        daemon=True,
    )
    _priority_thread = t
    t.start()


# ============================================================
# CATALYST SCAN BACKGROUND WORKER  (Catalyst Engine — Layer 2)
# docs/technical/CATALYST_ENGINE_SPEC.md
# ============================================================
_catalyst_started_at: datetime | None = None
# Scan (~2 Sonnet calls) + up to MAX_AUTO_ESCALATIONS Opus scenario calls — the Opus
# legs run sequentially and can take 1-2 min each, so the stuck-threshold covers the
# full worst case instead of declaring a healthy escalation run dead at 3 minutes.
_CATALYST_TIMEOUT_MINUTES = 10


def catalyst_is_stuck() -> bool:
    if _catalyst_started_at is None:
        return False
    return (datetime.now() - _catalyst_started_at).total_seconds() > _CATALYST_TIMEOUT_MINUTES * 60


def get_catalyst_elapsed_seconds() -> int:
    if _catalyst_started_at:
        return int((datetime.now() - _catalyst_started_at).total_seconds())
    return 0


def _portfolio_brief(summary) -> str:
    """Compact verified-holdings lines for the scenario engine's PORTFOLIO EXPOSURE
    section. Defensive over the summary shape; never raises."""
    try:
        lines = []
        for h in ((summary or {}).get("holdings") or [])[:80]:
            if not isinstance(h, dict) or not h.get("symbol"):
                continue
            parts = [str(h["symbol"])]
            for key in ("value_cad", "market_value", "value"):
                if isinstance(h.get(key), (int, float)):
                    parts.append(f"${h[key]:,.0f}")
                    break
            for key in ("allocation_pct", "allocation", "weight_pct"):
                if isinstance(h.get(key), (int, float)):
                    parts.append(f"{h[key]:.1f}%")
                    break
            lines.append(" | ".join(parts))
        return "\n".join(lines) if lines else "No verified holdings loaded."
    except Exception:
        return "No verified holdings loaded."


def _annotate_scenario_availability(result: dict, scenarios: dict) -> None:
    """Set scenario_available on each catalyst so the UI can offer an instant
    drill-down. The lanes hold the SAME dict objects as result['catalysts'], so
    one pass annotates both views.

    Availability requires a cached scenario with non-empty markdown — not merely a
    key in the dict. An entry can exist with null/empty markdown (a failed or cleared
    generation); marking those available shows a ⚡ Scenario button that the endpoint
    then answers not_found, so the click silently falls back to a live Analyze run —
    making ⚡ Scenario indistinguishable from the Analyze → button. This gate matches
    the endpoint's own check (get_catalyst_scenario), so button and endpoint agree."""
    scenarios = scenarios or {}

    def _has_real_scenario(cid) -> bool:
        entry = scenarios.get(cid)
        return isinstance(entry, dict) and bool((entry.get("markdown") or "").strip())

    for c in result.get("catalysts", []):
        if isinstance(c, dict):
            c["scenario_available"] = _has_real_scenario(c.get("id"))


def _alert_catalyst_escalation(catalyst: dict, markdown: str) -> None:
    """Surface an auto-escalated catalyst in the alerts inbox (3.2's open producer).

    An escalation is the system spending real money to decide something matters
    without being asked; landing it only in a cache means the user learns about
    it by chance. Severity follows portfolio relevance: a catalyst touching a
    held name is a warning (desktop notification), a market-wide one is info.
    Best-effort — a delivery failure must never fail the scan.
    """
    try:
        from tools.alerts import raise_alert

        headline = str(catalyst.get("headline") or "").strip()
        if not headline:
            return
        held = catalyst.get("portfolio_relevance") == "portfolio_impact"
        tickers = [t for t in ((catalyst.get("entities") or {}).get("tickers") or []) if t][:5]
        # The first line of the CATALYST section is the engine's own one-liner;
        # never re-summarize the scenario here (that would be a second LLM call
        # on a path whose whole point is that it already ran one).
        excerpt = next((ln.strip() for ln in markdown.splitlines() if ln.strip() and not ln.startswith("#")), "")
        raise_alert(
            title=f"Catalyst escalated: {headline[:120]}",
            message=(
                f"{headline}\n\n{excerpt[:400]}\n\n"
                f"{'Touches held names' if held else 'No direct portfolio exposure flagged'}"
                f"{': ' + ', '.join(tickers) if tickers else ''}. "
                "Full scenario is on the Catalysts panel."
            ),
            severity="warning" if held else "info",
            source="catalyst",
            dedup_key=f"catalyst_escalation:{catalyst.get('id')}",
            data={"catalyst_id": catalyst.get("id"), "tickers": tickers,
                  "portfolio_relevance": catalyst.get("portfolio_relevance")},
        )
    except Exception as e:
        log_to_component("server", "Catalyst", f"Escalation alert failed (non-fatal): {e}", level=logging.WARNING)


def run_catalyst_scan_in_background() -> None:
    """Classify today's news headlines into a ranked, deduped, two-lane catalyst
    list and cache it as `catalysts`. Best-effort and fully isolated — any failure
    is swallowed and never affects the news feed.

    Layer 3 auto-escalation (spec §3.6): the bounded auto_escalate selection runs the
    Opus scenario engine without a human in the loop; each scenario is cached under its
    catalyst id (`catalyst_scenarios`) for instant drill-down. Config-gated via the
    `catalyst` block of funnel_config.json; the cross-refresh escalation log guarantees
    a recurring story never re-bills.
    """
    global _catalyst_started_at
    _catalyst_started_at = datetime.now()
    try:
        from langchain_core.messages import HumanMessage

        from agent.nodes.news_analyst import gather_news_tool_outputs
        from tools.catalyst_extractor import (
            extract_catalysts,
            get_escalation_settings,
            load_escalated_ids,
            load_seen_ids,
            record_catalyst_ids,
        )
        from tools.daily_cache import get_cached, set_cached

        log_to_component("server", "Catalyst", "Starting catalyst scan...")
        state = {"messages": [HumanMessage(content="Scan today's market news for tradable catalysts.")]}
        _result, _messages, tool_outputs = gather_news_tool_outputs(state)
        if not tool_outputs:
            log_to_component("server", "Catalyst", "No news tool outputs returned; skipping scan.")
            return

        # Load holdings & derive watchlist from active WATCHING theses (Roadmap Item 3.6)
        summary = None
        holdings: list[str] = []
        try:
            from tools.portfolio_csv import get_portfolio_summary
            summary = get_portfolio_summary()
            holdings = [
                h["symbol"] for h in (summary.get("holdings") or [])
                if isinstance(h, dict) and h.get("symbol")
            ]
        except Exception as e:
            log_to_component("server", "Catalyst", f"Holdings load failed (relevance degraded to opportunity-only): {e}", level=logging.WARNING)

        watchlist: list[str] = []
        try:
            from tools.memory import _normalize_thesis_symbol, _thesis_position_state, get_active_theses
            held_set = {_normalize_thesis_symbol(s) for s in holdings} if holdings else set()
            theses = get_active_theses() or []
            for t in theses:
                try:
                    sym = t.get("symbol")
                    if sym and _thesis_position_state(t, held_set) == "watching":
                        watchlist.append(str(sym).upper().strip())
                except Exception:
                    continue  # Skip malformed thesis, don't break the whole watchlist
            watchlist = sorted(set(watchlist))
        except Exception as e:
            log_to_component("server", "Catalyst", f"Watchlist derivation failed: {e}", level=logging.WARNING)

        settings = get_escalation_settings()
        result = extract_catalysts(
            tool_outputs,
            holdings=holdings,
            watchlist=watchlist if watchlist else None,
            seen_ids=load_seen_ids(),
            already_escalated=load_escalated_ids(),
            escalation_cap=int(settings.get("max_auto_escalations", 3)),
            stale_after_days=int(settings.get("stale_event_days", 3)),
        )

        # Cache the list IMMEDIATELY (annotated from scenarios already on disk) so the
        # UI renders it without waiting behind the minutes-long Opus escalation legs.
        scenarios = get_cached("catalyst_scenarios") or {}
        _annotate_scenario_availability(result, scenarios)
        set_cached("catalysts", result)

        # --- Layer 3 auto-escalation (bounded by selection + config) ---
        # `additions` doubles as the record of what escalated: an id is added exactly
        # when its scenario succeeded.
        escalate = result.get("auto_escalate", [])
        additions: dict = {}
        if escalate and settings.get("auto_escalation_enabled"):
            from agent.catalyst_engine import merge_scenario_cache, run_scenario_for_catalyst

            portfolio_brief = _portfolio_brief(summary)
            for cat in escalate:
                cid = cat.get("id")
                if not cid:
                    continue
                headline = str(cat.get("headline", ""))[:100]
                log_to_component("server", "Catalyst", f"Auto-escalating scenario: {headline}")
                markdown = run_scenario_for_catalyst(cat, portfolio_brief)
                if markdown:
                    # Roadmap 3.3 owns the two producers 3.2 left open here: the
                    # scenario's TRIGGER PLAN levels become watched conditions,
                    # and the escalation itself reaches the inbox instead of
                    # sitting silently in a cache the user has to think to open.
                    from tools.watch_conditions import capture_watch_conditions, strip_watch_blocks
                    capture_watch_conditions(markdown, source="catalyst")
                    markdown = strip_watch_blocks(markdown).strip()
                    _alert_catalyst_escalation(cat, markdown)
                    additions[cid] = {
                        "headline": cat.get("headline"),
                        "generated_at": datetime.now().isoformat(),
                        "markdown": markdown,
                    }
                else:
                    log_to_component("server", "Catalyst", f"Scenario failed (non-fatal): {headline}", level=logging.WARNING)

            if additions:
                scenarios = merge_scenario_cache(get_cached("catalyst_scenarios"), additions)
                set_cached("catalyst_scenarios", scenarios)
                _annotate_scenario_availability(result, scenarios)
                set_cached("catalysts", result)
        elif escalate:
            log_to_component(
                "server", "Catalyst",
                f"{len(escalate)} catalyst(s) eligible but auto-escalation disabled by config.",
            )

        record_catalyst_ids(result["catalysts"], escalated_ids=list(additions))

        # 1.3's emitter. Separate from the dedup log above on purpose: that one
        # keeps ids for 7 days to stop a re-billed escalation, this one keeps the
        # CLAIM — direction, horizon, confidence — for as long as it takes the
        # horizon to elapse, which for "structural" is 90 days. Recording what was
        # predicted is not recoverable after the fact, so it happens here or never.
        from tools.catalyst_resolution import record_predictions

        emitted = record_predictions(result["catalysts"], escalated_ids=list(additions))

        log_to_component(
            "server", "Catalyst",
            f"Catalyst scan complete ({len(result['catalysts'])} catalysts cached, "
            f"{len(additions)} scenario(s) auto-escalated, "
            f"{emitted.get('recorded', 0)} prediction(s) recorded for resolution).",
        )
    except Exception as e:
        log_to_component("server", "Catalyst", f"Catalyst scan error: {e}", level=logging.ERROR)
        log_to_component("server", "Catalyst", traceback.format_exc(), level=logging.ERROR)
    finally:
        _catalyst_started_at = None


def start_catalyst_scan() -> None:
    """Kick off a catalyst scan in a dedicated daemon thread.

    NOT auto-triggered by the news refresh (a scan adds ~2 Sonnet calls; auto-firing
    on every refresh would silently grow the bill). Call this explicitly from a
    route/button/schedule. See spec §5 (P4) for the proactive-scan phase.
    """
    t = threading.Thread(
        target=_run_under_profile,
        args=(get_active_profile(), run_catalyst_scan_in_background),
        name="catalyst-scan-worker",
        daemon=True,
    )
    t.start()


def is_catalyst_running() -> bool:
    return _catalyst_started_at is not None


def maybe_start_catalyst_scan_after_news() -> bool:
    """Proactive scan (spec §5 P4-lite): piggyback a catalyst refresh on a completed
    news refresh — config-gated and TTL-throttled so rapid manual news refreshes never
    multiply the bill (a scan adds ~2 Sonnet calls + bounded Opus escalations).

    Returns True only when a scan was actually started. Never raises.
    """
    try:
        from tools.catalyst_extractor import get_escalation_settings
        from tools.daily_cache import get_cached

        settings = get_escalation_settings()
        if not settings.get("auto_scan_after_news"):
            return False
        if is_catalyst_running():
            return False
        interval_s = float(settings.get("auto_scan_min_interval_hours", 6)) * 3600
        if get_cached("catalysts", ttl_seconds=interval_s) is not None:
            return False  # cached list is fresh enough — don't re-spend
        log_to_component("server", "Catalyst", "Proactive scan: news refreshed and catalyst cache is stale — starting scan.")
        start_catalyst_scan()
        return True
    except Exception as e:
        log_to_component("server", "Catalyst", f"Proactive scan check failed (non-fatal): {e}", level=logging.WARNING)
        return False


def get_news_elapsed_seconds() -> int:
    if _news_fetch_started_at:
        return int((datetime.now() - _news_fetch_started_at).total_seconds())
    return 0

def get_pulse_elapsed_seconds() -> int:
    if _pulse_fetch_started_at:
        return int((datetime.now() - _pulse_fetch_started_at).total_seconds())
    return 0

def is_news_running() -> bool:
    return _news_fetch_started_at is not None

def is_pulse_running() -> bool:
    return _pulse_fetch_started_at is not None
