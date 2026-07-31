"""
Market Sentinel — Daily Market Pulse Engine
=============================================
Generates an on-demand daily market briefing by orchestrating existing tools:
  • Fear & Greed Index (CNN)
  • Batch yf.download (SPY, QQQ, IWM, VIX + portfolio holdings)
  • Fast technical scoring (from opportunity_scanner)
  • Systemic risk indicators (FRED credit spreads, M2)

Designed to be called once per day, cached via daily_cache, and surfaced
on the dashboard as a "Market Pulse" panel.  No background daemon — identical
pattern to the news feed.

NOT FINANCIAL ADVICE — for informational purposes only.
"""
import json
import math
import os
import time
from datetime import date, datetime
from typing import Any

import yfinance as yf

from agent.utils import safe_print
from tools.daily_cache import get_cached, set_cached
from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic
from tools.user_profile import get_data_path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_LEGACY_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_LEGACY_HISTORY_PATH = os.path.join(_LEGACY_DATA_DIR, "sentinel_history.json")


def _history_path() -> str:
    return get_data_path("sentinel_history.json")

# ---------------------------------------------------------------------------
# REGIME DEFINITIONS
# ---------------------------------------------------------------------------
REGIMES = {
    "CRISIS":   {"emoji": "🔴", "label": "CRISIS",   "color": "#ef4444", "min_score": 0,  "max_score": 15},
    "FEAR":     {"emoji": "🟠", "label": "FEAR",     "color": "#f97316", "min_score": 16, "max_score": 30},
    "CAUTIOUS": {"emoji": "🟡", "label": "CAUTIOUS", "color": "#eab308", "min_score": 31, "max_score": 40},
    "NEUTRAL":  {"emoji": "⚪", "label": "NEUTRAL",  "color": "#94a3b8", "min_score": 41, "max_score": 60},
    "BULLISH":  {"emoji": "🟢", "label": "BULLISH",  "color": "#22c55e", "min_score": 61, "max_score": 80},
    "EUPHORIA": {"emoji": "🔵", "label": "EUPHORIA", "color": "#3b82f6", "min_score": 81, "max_score": 100},
}


def _stamp_last_bar(payload: dict, series: Any) -> None:
    """Stamp `payload` with the date of a bar series' final row (Roadmap 5.8).

    Silent on any failure: a freshness stamp is evidence, and evidence that
    can't be obtained must simply be absent rather than guessed. Callers treat
    an unstamped payload as unverified, never as fresh.
    """
    try:
        from tools.freshness import stamp
        last = series.index[-1]
        stamp(payload, now=last.to_pydatetime() if hasattr(last, "to_pydatetime") else last)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. MARKET SNAPSHOT (SPY, QQQ, IWM, VIX)
# ---------------------------------------------------------------------------
@log_exceptions()
def _get_market_snapshot() -> dict[str, Any]:
    """Batch-download major indices and VIX, compute daily change + drawdown."""
    tickers = ["SPY", "QQQ", "IWM", "^VIX"]
    snapshot = {}

    try:
        data = yf.download(
            tickers,
            period="6mo",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=True,
        )

        for symbol in tickers:
            try:
                clean_name = symbol.replace("^", "")
                if len(tickers) == 1:
                    close = data["Close"].dropna()
                else:
                    if symbol not in data.columns.get_level_values(0):
                        continue
                    close = data[symbol]["Close"].dropna()

                if close.empty or len(close) < 2:
                    continue

                current = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                daily_change = ((current - prev_close) / prev_close) * 100
                high_6mo = float(close.max())
                drawdown = ((current - high_6mo) / high_6mo) * 100 if high_6mo > 0 else 0

                # Week and month return
                week_return = ((current - float(close.iloc[-5])) / float(close.iloc[-5]) * 100) if len(close) >= 5 else 0
                month_return = ((current - float(close.iloc[-22])) / float(close.iloc[-22]) * 100) if len(close) >= 22 else 0

                snapshot[clean_name] = {
                    "price": round(current, 2),
                    "daily_change": round(daily_change, 2),
                    "week_return": round(week_return, 1),
                    "month_return": round(month_return, 1),
                    "drawdown_from_high": round(drawdown, 1),
                    "high_6mo": round(high_6mo, 2),
                }
                # As-of stamp (Roadmap 5.8) = the LAST BAR's date, not the
                # download time. For a daily series that is the honest measure:
                # a provider mid-session returns a partial bar dated today, so a
                # last bar dated earlier means the feed has not caught up no
                # matter how recently we downloaded it.
                _stamp_last_bar(snapshot[clean_name], close)
            except Exception:
                pass

    except Exception as e:
        safe_print(f"⚠️ Market snapshot download failed: {e}")

    return snapshot


# ---------------------------------------------------------------------------
# 2. FEAR & GREED
# ---------------------------------------------------------------------------
@log_exceptions()
def _get_fear_greed() -> dict[str, Any]:
    """Fetch Fear & Greed index from existing tool."""
    try:
        from tools.sentiment_analysis import get_fear_greed_index
        return get_fear_greed_index()
    except Exception as e:
        safe_print(f"⚠️ Fear & Greed fetch failed: {e}")
        return {"score": 50, "rating": "Neutral", "error": str(e)}


# ---------------------------------------------------------------------------
# 3. PORTFOLIO HEALTH CHECK
# ---------------------------------------------------------------------------
@log_exceptions()
def _check_portfolio_health() -> list[dict[str, Any]]:
    """Batch-download portfolio tickers, compute RSI + drawdown + cross signals."""
    alerts = []

    try:
        from tools.portfolio_csv import get_tradeable_symbols
        symbols = get_tradeable_symbols()
        if not symbols:
            return []

        # Reuse the opportunity scanner's batch infrastructure
        from tools.opportunity_scanner import _batch_download, _compute_technicals_batch
        batch_data = _batch_download(symbols, period="6mo")

        if batch_data is None or batch_data.empty:
            return []

        technicals = _compute_technicals_batch(batch_data, symbols)

        for sym, tech in technicals.items():
            sym_alerts = []
            rsi = tech.get("rsi", 50)
            dd = tech.get("drawdown_pct", 0)
            tech.get("week_return", 0) / 5 if tech.get("week_return") else 0  # rough daily proxy

            if rsi < 30:
                sym_alerts.append(f"RSI oversold ({rsi:.0f})")
            if dd < -20:
                sym_alerts.append(f"Down {dd:.0f}% from high")
            if tech.get("golden_cross"):
                sym_alerts.append("Golden cross 📈")
            if tech.get("death_cross"):
                sym_alerts.append("Death cross 📉")
            if tech.get("vol_spike", 1) > 2.5:
                sym_alerts.append(f"Volume spike ({tech['vol_spike']:.1f}x)")

            if sym_alerts:
                alerts.append({
                    "symbol": sym,
                    "price": tech.get("price"),
                    "rsi": round(rsi, 1),
                    "drawdown_pct": round(dd, 1),
                    "week_return": tech.get("week_return", 0),
                    "month_return": tech.get("month_return", 0),
                    "alerts": sym_alerts,
                })

    except Exception as e:
        safe_print(f"⚠️ Portfolio health check failed: {e}")

    # Sort by severity (most drawdown first)
    alerts.sort(key=lambda x: x.get("drawdown_pct", 0))
    return alerts[:10]  # Top 10 most concerning


# ---------------------------------------------------------------------------
# 4. OPPORTUNITY SCAN (LIGHTWEIGHT)
# ---------------------------------------------------------------------------
@log_exceptions()
def _scan_opportunities_lite() -> list[dict[str, Any]]:
    """Run a fast technical scan on the full universe — no deep fundamentals."""
    opportunities = []

    try:
        # Funnel V2 M1 retired the static universe file: _load_universe (and the
        # nested sectors/thematic shape this walked) no longer exists, so this
        # whole block raised ImportError on every run and the except below
        # swallowed it — the Sentinel has been reporting zero opportunities
        # unconditionally ever since. _assemble_dynamic_universe returns a flat,
        # already-capped candidate list built from live market state instead.
        from tools.opportunity_scanner import (
            _assemble_dynamic_universe,
            _batch_download,
            _compute_technicals_batch,
            _fast_score,
            _get_sector_for_ticker,
            _get_thematic_tags,
            _stable_unique_symbols,
        )

        try:
            # market_mechanics is the implementation the scanner itself feeds in.
            from tools.market_mechanics import detect_sector_rotation
            rotation_data = detect_sector_rotation()
        except Exception:
            rotation_data = {}  # movers/themes alone still yield a usable pool

        candidates, _provenance = _assemble_dynamic_universe(rotation_data)
        ticker_list = _stable_unique_symbols(candidates)
        if not ticker_list:
            safe_print("⚠️ Sentinel: dynamic universe came back empty — skipping scan")
            return []
        safe_print(f"🔍 Sentinel: Scanning {len(ticker_list)} tickers...")

        batch_data = _batch_download(ticker_list, period="6mo")
        if batch_data is None or batch_data.empty:
            return []

        technicals = _compute_technicals_batch(batch_data, ticker_list)

        # Score each
        scored = []
        for sym, tech in technicals.items():
            score = _fast_score(sym, tech)
            if score >= 50:  # Higher threshold for sentinel — only notable ones
                themes = _get_thematic_tags(sym)
                sector = _get_sector_for_ticker(sym)

                # Build reason list
                reasons = []
                if tech.get("rsi", 50) < 35:
                    reasons.append(f"RSI {tech['rsi']:.0f}")
                if tech.get("drawdown_pct", 0) < -15:
                    reasons.append(f"{tech['drawdown_pct']:.0f}% off high")
                if tech.get("golden_cross"):
                    reasons.append("Golden Cross")
                if tech.get("month_return", 0) > 10:
                    reasons.append(f"+{tech['month_return']:.0f}% momentum")
                if tech.get("vol_spike", 1) > 2.0:
                    reasons.append(f"Vol spike {tech['vol_spike']:.1f}x")
                if themes:
                    reasons.append(f"Theme: {themes[0]}")

                scored.append({
                    "symbol": sym,
                    "score": score,
                    "price": tech.get("price"),
                    "rsi": round(tech.get("rsi", 50), 1),
                    "drawdown_pct": round(tech.get("drawdown_pct", 0), 1),
                    "week_return": round(tech.get("week_return", 0), 1),
                    "month_return": round(tech.get("month_return", 0), 1),
                    "sector": sector,
                    "themes": themes,
                    "reasons": reasons,
                })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        opportunities = scored[:10]  # Top 10

    except Exception as e:
        safe_print(f"⚠️ Opportunity scan failed: {e}")

    return opportunities


# ---------------------------------------------------------------------------
# 5. MACRO FLAGS
# ---------------------------------------------------------------------------
@log_exceptions()
def _get_macro_flags(market_snapshot: dict) -> list[str]:
    """Check systemic risk indicators and return warning flags."""
    flags = []

    # VIX check
    vix = market_snapshot.get("VIX", {})
    vix_price = vix.get("price", 0)
    if vix_price > 35:
        flags.append(f"🔴 VIX at {vix_price} — extreme volatility regime")
    elif vix_price > 25:
        flags.append(f"🟠 VIX at {vix_price} — elevated volatility")
    elif vix_price > 20:
        flags.append(f"🟡 VIX at {vix_price} — above-average volatility")

    # SPY drawdown
    spy_dd = market_snapshot.get("SPY", {}).get("drawdown_from_high", 0)
    if spy_dd < -15:
        flags.append(f"🔴 SPY {spy_dd}% from all-time high — deep correction territory")
    elif spy_dd < -10:
        flags.append(f"🟠 SPY {spy_dd}% from all-time high — correction territory")
    elif spy_dd < -5:
        flags.append(f"🟡 SPY {spy_dd}% from all-time high — pullback")

    # Credit spreads (from FRED)
    try:
        from tools.fred_api import get_systemic_risk_indicators
        risk = get_systemic_risk_indicators()
        if isinstance(risk, dict) and not risk.get("error"):
            crash_risk = risk.get("crash_risk", "Low")
            if crash_risk == "High":
                spread = risk.get("credit_spread", "N/A")
                flags.append(f"🔴 Credit spreads at {spread} — high systemic stress")
            elif crash_risk == "Elevated":
                flags.append("🟠 Credit spreads elevated — corporate bond stress rising")

            liquidity = risk.get("liquidity_status", "")
            m2 = risk.get("m2_growth_yoy", "N/A")
            if liquidity == "Contracting":
                flags.append(f"🟡 Money supply contracting ({m2}) — tighter liquidity")
    except Exception:
        pass

    return flags


# ---------------------------------------------------------------------------
# 5b. SECTOR TRENDS (which sectors are rotating up / down)
# ---------------------------------------------------------------------------
def _parse_pct(value: Any) -> float | None:
    """Pull a float out of a '+2.3%' style string (or a raw number).

    None when there is no usable number — including NaN and infinity, which
    `float()` accepts happily: an upstream NaN formatted as "+nan%" round-trips
    straight back to NaN here, and NaN is not valid JSON, so it 500s the
    endpoint that serves the payload rather than showing a wrong number. None
    rather than 0.0 because a substituted zero is a return that never happened.
    """
    try:
        parsed = float(str(value).replace("%", "").replace("+", "").strip())
    except (TypeError, ValueError):
        return None
    return round(parsed, 1) if math.isfinite(parsed) else None


@log_exceptions()
def _get_sector_trends() -> list[dict[str, Any]]:
    """Rank the 11 S&P sectors by momentum for the pulse heatmap.

    Reuses the (daily-cached) sector-rotation engine and flattens its
    emoji-tagged strings into a compact, numeric payload the dashboard can
    color and sort without re-parsing. Sorted by momentum (accel/decel),
    strongest first — the same order the rotation engine returns.
    """
    try:
        from tools.sector_rotation import detect_sector_rotation
        rotation = detect_sector_rotation()
    except Exception as e:
        safe_print(f"⚠️ Sector trends fetch failed: {e}")
        return []

    if not isinstance(rotation, dict) or rotation.get("error"):
        return []

    trends = []
    for r in rotation.get("sector_performance", []):
        signal = r.get("signal", "")
        if "INFLOW" in signal:
            flow = "inflow"
        elif "OUTFLOW" in signal:
            flow = "outflow"
        else:
            flow = "neutral"
        return_1m = _parse_pct(r.get("return_1m"))
        return_3m = _parse_pct(r.get("return_3m"))
        momentum = _parse_pct(r.get("momentum_score"))
        # A sector whose numbers didn't survive parsing is left OUT of the
        # heatmap rather than plotted at zero — the row is what carries the
        # colour and the sort, and there is nothing here to colour or sort.
        if return_1m is None or return_3m is None or momentum is None:
            continue
        trends.append({
            "symbol": r.get("symbol"),
            "sector": r.get("sector"),
            "character": r.get("character"),
            "return_1m": return_1m,
            "return_3m": return_3m,
            "momentum": momentum,
            "flow": flow,
        })
    return trends


# ---------------------------------------------------------------------------
# 6. REGIME CLASSIFICATION
# ---------------------------------------------------------------------------
def _classify_regime(fear_greed_score: int, vix: float, spy_drawdown: float) -> tuple:
    """
    Compute a composite regime score (0-100) and classify into a regime.
    Lower score = more fear/crisis, Higher score = more greed/euphoria.
    """
    # Normalize components to 0-100 scale
    # Fear & Greed is already 0-100 (higher = greedier)
    fg_normalized = fear_greed_score

    # VIX: 10=calm(100), 20=normal(60), 30=elevated(30), 40+=extreme(0)
    vix_normalized = max(0, min(100, 100 - ((vix - 10) * 3.33)))

    # Drawdown: 0%=perfect(100), -5%=ok(80), -10%=bad(40), -20%=crisis(0)
    dd_normalized = max(0, min(100, 100 + (spy_drawdown * 5)))

    # Weighted composite (Fear & Greed dominates, VIX and drawdown confirm)
    composite = (fg_normalized * 0.45) + (vix_normalized * 0.30) + (dd_normalized * 0.25)
    composite = max(0, min(100, composite))

    # Classify
    if composite <= 15:
        regime = "CRISIS"
    elif composite <= 30:
        regime = "FEAR"
    elif composite <= 40:
        regime = "CAUTIOUS"
    elif composite <= 60:
        regime = "NEUTRAL"
    elif composite <= 80:
        regime = "BULLISH"
    else:
        regime = "EUPHORIA"

    return regime, round(composite, 1)


# ---------------------------------------------------------------------------
# 7. REGIME HISTORY (Append-Only Log)
# ---------------------------------------------------------------------------
def _load_regime_history() -> list[dict]:
    """Load regime history from disk."""
    for path in (_history_path(), _LEGACY_HISTORY_PATH):
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return []


def _save_regime_entry(entry: dict) -> None:
    """Append a new regime entry to history."""
    history_path = _history_path()
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    history = _load_regime_history()

    # Don't duplicate today's entry
    today_str = entry.get("date")
    history = [h for h in history if h.get("date") != today_str]
    history.append(entry)

    # Keep last 90 days
    history = history[-90:]

    write_json_atomic(history_path, history, default=str)


def _get_regime_streak(history: list[dict], current_regime: str) -> int:
    """Count how many consecutive days we've been in this regime."""
    streak = 1
    for entry in reversed(history):
        if entry.get("regime") == current_regime:
            streak += 1
        else:
            break
    return streak


# ---------------------------------------------------------------------------
# 8. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
@log_exceptions()
def generate_market_pulse() -> dict[str, Any]:
    """
    Generate a comprehensive daily market briefing.
    This is the single function called by the server endpoint.
    """
    start_time = time.time()
    safe_print("🛰️ Market Sentinel: Generating daily pulse...")

    # 1. Market Snapshot
    safe_print("  → Downloading market indices...")
    market_snapshot = _get_market_snapshot()

    # 2. Fear & Greed
    safe_print("  → Checking Fear & Greed index...")
    fear_greed = _get_fear_greed()
    fg_score = fear_greed.get("score", 50)
    fg_label = fear_greed.get("rating", "Neutral")

    # 3. Regime Classification
    vix_price = market_snapshot.get("VIX", {}).get("price", 18)
    spy_drawdown = market_snapshot.get("SPY", {}).get("drawdown_from_high", 0)
    regime, regime_score = _classify_regime(fg_score, vix_price, spy_drawdown)
    regime_info = REGIMES[regime]

    # 4. Portfolio Health
    safe_print("  → Scanning portfolio holdings...")
    portfolio_alerts = _check_portfolio_health()

    # 5. Opportunity Scan
    safe_print("  → Running opportunity scan...")
    opportunities = _scan_opportunities_lite()

    # 6. Macro Flags
    safe_print("  → Checking macro indicators...")
    macro_flags = _get_macro_flags(market_snapshot)

    # 6b. Sector Trends (which sectors are rotating up / down)
    safe_print("  → Ranking sector trends...")
    sector_trends = _get_sector_trends()

    # 7. Build headline
    spy_info = market_snapshot.get("SPY", {})
    headline_parts = []
    headline_parts.append(f"{regime_info['emoji']} {regime_info['label']}")
    if spy_info.get("daily_change"):
        headline_parts.append(f"SPY {spy_info['daily_change']:+.1f}%")
    headline_parts.append(f"F&G {fg_score}")
    if vix_price:
        headline_parts.append(f"VIX {vix_price:.0f}")
    if opportunities:
        headline_parts.append(f"{len(opportunities)} opportunities")
    headline = " | ".join(headline_parts)

    # 8. Determine if action is required
    action_required = regime in ("CRISIS", "FEAR") or len(portfolio_alerts) >= 3

    # 9. Build recommendation
    if regime == "CRISIS":
        recommendation = "🚨 This is the environment where generational buying opportunities emerge. Review the flagged opportunities carefully."
    elif regime == "FEAR":
        recommendation = "⚠️ Fear is elevated — historically a favorable time to accumulate quality positions. Don't panic sell."
    elif regime == "EUPHORIA":
        recommendation = "📊 Extreme greed territory — consider trimming winners and raising cash. Don't chase momentum."
    elif regime == "CAUTIOUS":
        recommendation = "🔍 Market showing caution signals. Monitor closely but no urgent action needed."
    elif regime == "BULLISH":
        recommendation = "✅ Healthy bull market conditions. Stay the course with your investment plan."
    else:
        recommendation = "📊 Normal market conditions. Continue with regular portfolio management."

    # 10. Regime history
    history = _load_regime_history()
    streak = _get_regime_streak(history, regime)

    today_str = date.today().isoformat()
    _save_regime_entry({
        "date": today_str,
        "regime": regime,
        "score": regime_score,
        "fear_greed": fg_score,
        "vix": vix_price,
        "spy_drawdown": spy_drawdown,
    })

    # 11. Alerts (Advisor Roadmap Theme 3.2): the advisor calls first.
    # Regime flips and action-required pulses land in the per-profile inbox,
    # broadcast over WebSocket, and (warning+) post a macOS notification.
    # Dedup keys collapse same-day repeats; alerts must never break the pulse.
    try:
        from tools.alerts import raise_alert

        prev_regime = history[-1].get("regime") if history else None
        if prev_regime and prev_regime != regime:
            into_defensive = regime in ("CRISIS", "FEAR")
            raise_alert(
                title=f"Regime flip: {prev_regime} → {regime}",
                message=f"{headline}. {recommendation}",
                severity="warning" if into_defensive else "info",
                source="market_sentinel",
                dedup_key=f"regime-flip-{today_str}-{regime}",
                data={"from": prev_regime, "to": regime, "regime_score": regime_score,
                      "fear_greed": fg_score, "vix": vix_price},
                notify=into_defensive,
            )
        if action_required:
            raise_alert(
                title=f"Action required — {regime_info['label']}",
                message=f"{len(portfolio_alerts)} holding(s) flagged. {recommendation}",
                severity="critical" if regime == "CRISIS" else "warning",
                source="market_sentinel",
                dedup_key=f"action-required-{today_str}",
                data={"regime": regime, "portfolio_alert_count": len(portfolio_alerts),
                      "fear_greed": fg_score, "vix": vix_price},
            )
    except Exception:
        pass

    elapsed = round(time.time() - start_time, 1)
    safe_print(f"🛰️ Market Sentinel: Pulse complete in {elapsed}s — Regime: {regime} ({regime_score})")

    briefing = {
        "date": today_str,
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,

        # Regime
        "regime": regime,
        "regime_score": regime_score,
        "regime_emoji": regime_info["emoji"],
        "regime_label": regime_info["label"],
        "regime_color": regime_info["color"],
        "regime_streak": streak,
        "action_required": action_required,

        # Market Snapshot
        "market_snapshot": {
            "spy": market_snapshot.get("SPY", {}),
            "qqq": market_snapshot.get("QQQ", {}),
            "iwm": market_snapshot.get("IWM", {}),
            "vix": market_snapshot.get("VIX", {}),
            "fear_greed_score": fg_score,
            "fear_greed_label": fg_label,
        },

        # Portfolio
        "portfolio_alerts": portfolio_alerts,
        "portfolio_alert_count": len(portfolio_alerts),

        # Opportunities
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),

        # Macro
        "macro_flags": macro_flags,

        # Sectors (ranked by momentum — which sectors are trending up / down)
        "sector_trends": sector_trends,

        # Summary
        "headline": headline,
        "recommendation": recommendation,

        # History (last 30 days for sparkline)
        "regime_history": history[-30:],
    }

    return briefing


# ---------------------------------------------------------------------------
# 9. AGENT-FACING FUNCTIONS
# ---------------------------------------------------------------------------
def get_market_regime() -> dict[str, Any]:
    """
    Get the current market regime (for agent tool use).
    Returns the cached daily briefing or generates a new one.
    """
    cached = get_cached("market_pulse")
    if cached:
        return {
            "regime": cached.get("regime"),
            "regime_score": cached.get("regime_score"),
            "regime_emoji": cached.get("regime_emoji"),
            "regime_streak": cached.get("regime_streak"),
            "headline": cached.get("headline"),
            "recommendation": cached.get("recommendation"),
            "fear_greed": cached.get("market_snapshot", {}).get("fear_greed_score"),
            "vix": cached.get("market_snapshot", {}).get("vix", {}).get("price"),
            "spy_drawdown": cached.get("market_snapshot", {}).get("spy", {}).get("drawdown_from_high"),
            "portfolio_alerts": cached.get("portfolio_alert_count", 0),
            "opportunities": cached.get("opportunity_count", 0),
            "generated_at": cached.get("generated_at"),
        }

    # No cached data — generate fresh
    briefing = generate_market_pulse()
    set_cached("market_pulse", briefing)
    return get_market_regime()  # Re-read from cache to return slim version


def get_regime_history(days: int = 30) -> dict[str, Any]:
    """
    Get regime history for the last N days (for agent tool use).
    """
    history = _load_regime_history()
    recent = history[-days:]

    if not recent:
        return {"history": [], "note": "No regime history available yet. Open the dashboard to generate your first Market Pulse."}

    # Count regime frequencies
    regime_counts = {}
    for entry in recent:
        r = entry.get("regime", "UNKNOWN")
        regime_counts[r] = regime_counts.get(r, 0) + 1

    return {
        "days_tracked": len(recent),
        "history": recent,
        "regime_distribution": regime_counts,
        "latest": recent[-1] if recent else None,
    }


if __name__ == "__main__":
    result = generate_market_pulse()
    print(json.dumps(result, indent=2, default=str))
