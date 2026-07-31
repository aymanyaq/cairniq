"""
Intraday Sentinel Tick — state-change detector (Advisor Roadmap Theme 3.4).

The daily Market Pulse (tools/market_sentinel.py) paints a once-a-day picture.
This is its intraday counterpart: a zero-LLM tick that runs during market hours
(on a 6-hour cooldown, so ~twice a session) and raises an alert ONLY when a
monitored signal actually *changes state* — never on a level that has been
standing all session.

The cadence is deliberately unhurried. This serves a long-horizon holder, not a
day trader: a band that spikes and round-trips inside one session is noise a
decade-long plan should never be paged about, while a real deterioration — a
death cross, a drawdown deepening a band — persists and is caught on the next
pass regardless of how slowly the tick runs. Because every signal is compared
against remembered state rather than re-derived from scratch, a slower tick
loses nothing but latency.

It reuses the sentinel's own inputs — `_get_market_snapshot` for the indices and
the opportunity-scanner batch technicals for holdings (the same machinery
`_check_portfolio_health` uses) — and adds the two things the daily pulse lacks:

  1. A per-profile STATE STORE (`intraday_sentinel_state.json`). Each signal's
     last-committed state is remembered, so a tick can tell a fresh crossing from
     a level that was already true. The first time a signal is seen it is
     recorded as a silent baseline and NEVER alerted — exactly the "void if
     already true when armed" rule the watch-conditions engine uses. This is the
     guard against the failure mode this project keeps hitting: an alert that
     restates a standing condition as if it were news.

  2. HYSTERESIS. Band boundaries carry a dead-zone (a margin), and a boolean
     signal must clear a lower re-arm level before it can fire again, so a value
     wobbling across a threshold cannot machine-gun the inbox.

Producers (each fires into the 3.2 alerts inbox, source="sentinel"):
  - VIX band crossings (calm → elevated → high → extreme), upward only.
  - SPY drawdown-from-high band crossings (near-high → pullback → correction →
    deep), deepening only.
  - Fresh death cross on a holding — the producer Roadmap 3.2 left open.
  - Fresh golden cross on a holding (info).
  - Volume spike > 2.5x average on a holding, armed with a 2.0x re-arm.
  - A crossed rung of the user's own cash-deployment ladder (Roadmap 3.9),
    armed off the same SPY drawdown reading as the band above it.

Down-transitions (VIX calming, a holding recovering) update the stored state
silently: the sentinel's job is to surface rising risk and fresh signals, not to
narrate every relaxation. Network failures leave the prior state untouched — a
missing quote is never treated as a transition.

NOT FINANCIAL ADVICE — informational only.
"""
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from agent.logger import log_to_component
from tools.user_profile import get_data_path

_STATE_FILENAME = "intraday_sentinel_state.json"

# ---------------------------------------------------------------------------
# Band definitions. Each metric is normalized so a LARGER "severity value" means
# a more severe state, letting one hysteresis routine serve both. Boundaries are
# ascending; `margin` is the half-width of the dead-zone straddling each one.
# ---------------------------------------------------------------------------

# VIX price → {calm, elevated, high, extreme}. severity value = the price itself.
VIX_BOUNDARIES = [20.0, 25.0, 35.0]
VIX_MARGIN = 1.0
VIX_BANDS = [
    {"key": "calm", "label": "calm", "severity": "info"},          # band 0 (baseline)
    {"key": "elevated", "label": "elevated", "severity": "info"},   # band 1
    {"key": "high", "label": "high", "severity": "warning"},        # band 2
    {"key": "extreme", "label": "extreme", "severity": "critical"}, # band 3
]

# SPY drawdown-from-high (a negative %) → depth bands. severity value = -drawdown,
# so a deeper (more negative) drawdown is a larger, more severe value.
#
# The 25% rung was appended for Roadmap 3.7 (drawdown playbook). Appended at the
# END on purpose: persisted state stores the band INDEX, so inserting anywhere
# else would silently re-label every remembered band on existing profiles — a
# machine that thought it was at "correction" would wake up at "pullback" and
# re-fire a crossing it had already reported.
SPY_DD_BOUNDARIES = [5.0, 10.0, 15.0, 25.0]
SPY_DD_MARGIN = 1.0
SPY_DD_BANDS = [
    {"key": "near_high", "label": "near its high", "severity": "info"},          # 0
    {"key": "pullback", "label": "a pullback (>5% off high)", "severity": "info"},        # 1
    {"key": "correction", "label": "a correction (>10% off high)", "severity": "warning"}, # 2
    {"key": "deep", "label": "a deep correction (>15% off high)", "severity": "critical"}, # 3
    {"key": "severe", "label": "a severe drawdown (>25% off high)", "severity": "critical"}, # 4
]

# Volume spike: arm (and alert) at 2.5x average volume; only re-arm below 2.0x.
VOL_ARM = 2.5
VOL_DISARM = 2.0

# Roadmap 3.9 ladder report when the tick never got a fresh SPY reading to arm
# against. `evaluated: False` is the load-bearing field: it separates "the
# ladder is empty" from "the ladder was not looked at", and reporting 0 armed
# rungs for the second case would be a fabricated liveness number.
_LADDER_UNEVALUATED = {"specs": [], "armed": 0, "fired": 0, "seeded": 0, "levels": 0, "evaluated": False}


# ---------------------------------------------------------------------------
# State store (per-profile single JSON object, atomically rewritten each tick).
# ---------------------------------------------------------------------------

def _state_file() -> str:
    return get_data_path(_STATE_FILENAME)


def _load_state() -> dict[str, Any]:
    """Read the per-profile state. {} on any error (a fresh baseline, not a crash)."""
    try:
        path = _state_file()
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(state: dict[str, Any]) -> bool:
    """Atomically replace the state file (tmp + replace). Never raises."""
    try:
        path = _state_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        return True
    except Exception as e:  # noqa: BLE001
        log_to_component("server", "Sentinel", f"state write failed: {e}", level=logging.WARNING)
        return False


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

def _band_with_hysteresis(sev_value: float, boundaries: list[float], margin: float, last_band: int) -> int:
    """Commit a band index for `sev_value`, starting from `last_band`.

    A move UP past boundary k needs the value above it by more than `margin`; a
    move DOWN needs it below by more than `margin`. Between those, the last band
    holds — the dead-zone that stops a value hovering on a boundary from
    flapping. Seeding a cold signal with `last_band=0` yields the plain band (the
    down-moves can't fire), so a first observation records where the value truly
    sits without alerting.
    """
    band = max(0, min(last_band, len(boundaries)))
    while band < len(boundaries) and sev_value >= boundaries[band] + margin:
        band += 1
    while band > 0 and sev_value <= boundaries[band - 1] - margin:
        band -= 1
    return band


# ---------------------------------------------------------------------------
# Alert delivery (routed through one seam so tests can capture without a store).
# ---------------------------------------------------------------------------

def _raise(**kwargs: Any) -> None:
    from tools.alerts import raise_alert
    raise_alert(**kwargs)


def _freshness(payload: Any, now: datetime) -> tuple[bool, str]:
    """(may_fire, note) for one daily-bar payload (Roadmap 5.8).

    A bar from an EARLIER session cannot justify an alert claiming something
    just changed, so it blocks the fire outright. A payload with no readable
    stamp is *unverified* rather than stale — it still fires, but the note says
    so, because refusing every unstamped payload would silently switch the whole
    sentinel off the moment a stamp went missing, and a silent no-op is the
    failure mode this codebase keeps paying for.

    The caller must SKIP a stale payload entirely rather than merely suppress
    its alert: updating state from a stale bar would consume the transition, and
    the alert would then never fire once fresh data arrived.
    """
    from tools.freshness import describe, is_current_session
    return is_current_session(payload, now) is not False, describe(payload, now)


# ---------------------------------------------------------------------------
# Producers — each returns a list of 0 or 1 alert specs and updates its state
# slot in place. An alert spec is {"summary": ..., "raise": {kwargs for _raise}}.
# ---------------------------------------------------------------------------

def _update_band(
    market_state: dict[str, Any],
    key: str,
    sev_value: float,
    display_value: float,
    boundaries: list[float],
    margin: float,
    bands: list[dict[str, str]],
    subject: str,
    describe: Callable[[float, str], str],
    now: datetime,
    note: str = "",
) -> list[dict[str, Any]]:
    stamp = now.isoformat(timespec="seconds")
    entry = market_state.get(key)

    if not isinstance(entry, dict) or "band" not in entry:
        seed = _band_with_hysteresis(sev_value, boundaries, margin, 0)
        market_state[key] = {"band": seed, "value": display_value, "updated_at": stamp}
        return []

    last_band = int(entry["band"])
    new_band = _band_with_hysteresis(sev_value, boundaries, margin, last_band)
    market_state[key] = {"band": new_band, "value": display_value, "updated_at": stamp}

    if new_band <= last_band:
        return []  # unchanged or relaxing — update silently

    dest = bands[new_band]
    was = bands[last_band]["label"]
    return [{
        "summary": {"type": "band", "key": key, "band": dest["key"], "severity": dest["severity"], "value": display_value},
        "raise": {
            "title": f"{subject}: {dest['label']}",
            "message": describe(display_value, was) + (f"\n\nQuote {note}." if note else ""),
            "severity": dest["severity"],
            "source": "sentinel",
            "dedup_key": f"sentinel:{key.lower()}:{dest['key']}:{now.date().isoformat()}",
            "data": {"metric": key, "band": dest["key"], "previous_band": was,
                     "value": display_value, "as_of": note},
        },
    }]


def _attach_drawdown_playbook(alerts: list[dict[str, Any]], drawdown_pct: float) -> None:
    """Roadmap 3.7: on a DEEP drawdown crossing, append the pre-agreed playbook.

    Only the deep bands (see PLAYBOOK_BANDS). Reciting the crash plan on every
    5% dip would make it wallpaper by the time it matters — the whole value of
    this text is that the user has not seen it recently.

    Mutates the specs in place and NEVER raises: the band alert is the load-
    bearing part and must still fire if the playbook store or the goal
    projection is unavailable. A drawdown alert that failed to send because its
    optional footer broke would be a self-inflicted version of the exact silence
    this codebase keeps paying for.
    """
    if not alerts:
        return
    try:
        from tools.drawdown_playbook import PLAYBOOK_BANDS, build_drawdown_message

        labels = {b["key"]: b["label"] for b in SPY_DD_BANDS}
        for spec in alerts:
            band_key = spec.get("summary", {}).get("band")
            if band_key not in PLAYBOOK_BANDS:
                continue
            spec["raise"]["message"] = build_drawdown_message(
                drawdown_pct, labels.get(band_key, band_key)
            )
            spec["raise"]["data"]["playbook_surfaced"] = True
    except Exception as e:  # noqa: BLE001
        log_to_component("server", "Sentinel", f"playbook attach failed: {e}", level=logging.WARNING)


def _evaluate_ladder(state: dict[str, Any], drawdown_pct: float, now: datetime) -> dict[str, Any]:
    """Roadmap 3.9: arm the playbook's deployment rungs. NEVER raises.

    Hosted on this tick rather than beside 3.3's watch conditions because this
    is where the peak already lives: it is handed the very `drawdown_from_high`
    reading the band crossing above just used, so the ladder can never deploy on
    a depth that disagrees with the alert the user was shown. Same reason the
    caller must only reach here through the freshness gate — a bar from an
    earlier session that consumed a rung would spend the user's cash on a level
    the tape may not currently be at, and the rung would not be there to fire
    when it genuinely crossed.

    Never raises, for the same reason `_attach_drawdown_playbook` doesn't: the
    band alert is load-bearing and must still go out if the playbook store is
    unreadable.
    """
    try:
        from tools.drawdown_playbook import evaluate_deployment_ladder

        return evaluate_deployment_ladder(
            drawdown_pct, state.setdefault("deployment_ladder", {}), now
        )
    except Exception as e:  # noqa: BLE001
        log_to_component("server", "Sentinel", f"deployment ladder failed: {e}", level=logging.WARNING)
        return dict(_LADDER_UNEVALUATED)


def _update_cross(
    holding_state: dict[str, Any],
    symbol: str,
    field: str,
    present: bool,
    severity: str,
    title: str,
    message: str,
    now: datetime,
    note: str = "",
) -> list[dict[str, Any]]:
    last = holding_state.get(field)
    holding_state[field] = present
    if last is None or present == last:
        return []  # baseline, or no change
    if not present:
        return []  # the cross cleared — silent
    return [{
        "summary": {"type": field, "symbol": symbol, "severity": severity},
        "raise": {
            "title": title,
            "message": message + (f"\n\nQuote {note}." if note else ""),
            "severity": severity,
            "source": "sentinel",
            "dedup_key": f"sentinel:{field}:{symbol}:{now.date().isoformat()}",
            "data": {"symbol": symbol, "signal": field, "as_of": note},
        },
    }]


def _update_vol(
    holding_state: dict[str, Any],
    symbol: str,
    vol_spike: Any,
    now: datetime,
    note: str = "",
) -> list[dict[str, Any]]:
    if not isinstance(vol_spike, (int, float)):
        return []
    vol = float(vol_spike)
    armed = holding_state.get("vol_spike_armed")

    if armed is None:
        holding_state["vol_spike_armed"] = vol >= VOL_ARM
        return []
    if not armed and vol >= VOL_ARM:
        holding_state["vol_spike_armed"] = True
        return [{
            "summary": {"type": "vol_spike", "symbol": symbol, "severity": "info", "value": round(vol, 2)},
            "raise": {
                "title": f"{symbol}: volume spike",
                "message": (f"{symbol} is trading at {vol:.1f}x its average volume — unusual activity worth a look."
                            + (f"\n\nQuote {note}." if note else "")),
                "severity": "info",
                "source": "sentinel",
                "dedup_key": f"sentinel:vol_spike:{symbol}:{now.date().isoformat()}",
                "data": {"symbol": symbol, "signal": "vol_spike", "vol_spike": round(vol, 2), "as_of": note},
            },
        }]
    if armed and vol < VOL_DISARM:
        holding_state["vol_spike_armed"] = False  # re-armed for the next genuine spike
    return []


# ---------------------------------------------------------------------------
# Data sources (default implementations; injectable for offline testing).
# ---------------------------------------------------------------------------

def fetch_market_snapshot() -> dict[str, Any]:
    """Indices + VIX snapshot — the daily sentinel's own reader, reused."""
    from tools.market_sentinel import _get_market_snapshot
    return _get_market_snapshot()


def fetch_holdings_technicals() -> dict[str, dict[str, Any]]:
    """Per-symbol batch technicals for the bound profile's tradeable holdings.

    Mirrors how `_check_portfolio_health` sources its data (get_tradeable_symbols
    → batch download → batch technicals), but returns the raw signal dict so the
    tick can detect the boolean/threshold *transitions* that function discards.
    """
    from tools.portfolio_csv import get_tradeable_symbols
    symbols = get_tradeable_symbols()
    if not symbols:
        return {}
    from tools.opportunity_scanner import _batch_download, _compute_technicals_batch
    batch = _batch_download(symbols, period="6mo")
    if batch is None or getattr(batch, "empty", True):
        return {}
    technicals = _compute_technicals_batch(batch, symbols)

    # As-of stamp (Roadmap 5.8) = the final bar's date. _compute_technicals_batch
    # is shared by ten callers and returns no timestamps, so rather than change
    # it we read the index off the frame we already hold.
    from tools.market_sentinel import _stamp_last_bar
    for sym, tech in technicals.items():
        try:
            series = batch["Close"] if len(symbols) == 1 else batch[sym]["Close"]
            _stamp_last_bar(tech, series.dropna())
        except Exception:
            pass
    return technicals


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------

def run_sentinel_tick(
    now: datetime | None = None,
    snapshot_fn: Callable[[], dict[str, Any]] | None = None,
    holdings_fn: Callable[[], dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run one intraday tick: detect state changes, persist them, deliver alerts.

    Both data sources are injectable so the whole engine is testable offline.
    State is written BEFORE any alert is delivered, so a delivery failure can
    never lose a recorded transition (and thus never re-fire it next tick).
    Returns a summary; never raises.
    """
    now = now or datetime.now()
    get_snapshot = snapshot_fn or fetch_market_snapshot
    get_holdings = holdings_fn or fetch_holdings_technicals

    try:
        state = _load_state()
        market_state = state.setdefault("market", {})
        holdings_state = state.setdefault("holdings", {})
        pending: list[dict[str, Any]] = []

        # --- Market bands -------------------------------------------------
        snapshot: dict[str, Any] = {}
        try:
            snapshot = get_snapshot() or {}
        except Exception as e:  # noqa: BLE001 — a bad market read must not abort holdings
            log_to_component("server", "Sentinel", f"market snapshot failed: {e}", level=logging.WARNING)

        stale_skipped = 0

        vix_payload = snapshot.get("VIX") or {}
        vix = vix_payload.get("price")
        if isinstance(vix, (int, float)):
            may_fire, note = _freshness(vix_payload, now)
            if may_fire:
                pending += _update_band(
                    market_state, "VIX_BAND", float(vix), round(float(vix), 1),
                    VIX_BOUNDARIES, VIX_MARGIN, VIX_BANDS, "VIX",
                    lambda v, was: f"VIX moved from {was} to a higher band — now {v:.1f}.",
                    now, note,
                )
            else:
                stale_skipped += 1

        spy_payload = snapshot.get("SPY") or {}
        spy_dd = spy_payload.get("drawdown_from_high")
        ladder = dict(_LADDER_UNEVALUATED)
        if isinstance(spy_dd, (int, float)):
            may_fire, note = _freshness(spy_payload, now)
            if may_fire:
                dd_alerts = _update_band(
                    market_state, "SPY_DRAWDOWN_BAND", -float(spy_dd), round(float(spy_dd), 1),
                    SPY_DD_BOUNDARIES, SPY_DD_MARGIN, SPY_DD_BANDS, "SPY",
                    lambda v, was: f"SPY is {v:.1f}% off its 6-month high — deeper than {was}.",
                    now, note,
                )
                _attach_drawdown_playbook(dd_alerts, abs(float(spy_dd)))
                pending += dd_alerts

                # Roadmap 3.9 — arm the ladder off the SAME peak-to-date reading
                # the band above just used, so there is only ever one peak.
                ladder = _evaluate_ladder(state, abs(float(spy_dd)), now)
                pending += ladder.pop("specs", [])
            else:
                stale_skipped += 1

        # --- Holdings -----------------------------------------------------
        tech: dict[str, dict[str, Any]] = {}
        try:
            tech = get_holdings() or {}
        except Exception as e:  # noqa: BLE001
            log_to_component("server", "Sentinel", f"holdings scan failed: {e}", level=logging.WARNING)

        checked = 0
        for sym, t in tech.items():
            if not isinstance(t, dict):
                continue
            # A bar from an earlier session is skipped ENTIRELY — state included.
            # Suppressing only the alert while still advancing state would consume
            # the transition against stale data, and the real crossing would then
            # never fire once the feed caught up.
            may_fire, note = _freshness(t, now)
            if not may_fire:
                stale_skipped += 1
                continue
            checked += 1
            hs = holdings_state.setdefault(sym, {})
            pending += _update_cross(
                hs, sym, "death_cross", bool(t.get("death_cross")), "warning",
                f"{sym}: death cross",
                f"{sym} formed a death cross — its 50-day average crossed below its 200-day. Fresh bearish trend signal on a holding.",
                now, note,
            )
            pending += _update_cross(
                hs, sym, "golden_cross", bool(t.get("golden_cross")), "info",
                f"{sym}: golden cross",
                f"{sym} formed a golden cross — its 50-day average crossed above its 200-day.",
                now, note,
            )
            pending += _update_vol(hs, sym, t.get("vol_spike"), now, note)
            hs["updated_at"] = now.isoformat(timespec="seconds")

        # Prune state for names no longer held — but only when we actually got a
        # universe, so a transient download failure never wipes remembered state.
        if tech:
            live = set(tech.keys())
            for sym in list(holdings_state.keys()):
                if sym not in live:
                    holdings_state.pop(sym, None)

        state["updated_at"] = now.isoformat(timespec="seconds")
        _write_state(state)

        # Deliver after the state is safely persisted.
        for spec in pending:
            try:
                _raise(**spec["raise"])
            except Exception as e:  # noqa: BLE001 — one delivery failure must not lose the rest
                log_to_component("server", "Sentinel", f"alert delivery failed: {e}", level=logging.WARNING)

        if stale_skipped:
            # Visible, never silent: a gate that quietly suppresses everything is
            # indistinguishable from a broken engine.
            log_to_component(
                "server", "Sentinel",
                f"{stale_skipped} signal(s) skipped — bar older than the current session (5.8 freshness gate).",
            )

        return {
            "checked_holdings": checked,
            "fired": len(pending),
            "stale_skipped": stale_skipped,
            "ladder": ladder,
            "alerts": [spec["summary"] for spec in pending],
        }
    except Exception as e:  # noqa: BLE001
        log_to_component("server", "Sentinel", f"run_sentinel_tick failed: {e}", level=logging.WARNING)
        return {"checked_holdings": 0, "fired": 0, "stale_skipped": 0,
                "ladder": dict(_LADDER_UNEVALUATED), "alerts": [], "error": str(e)}
