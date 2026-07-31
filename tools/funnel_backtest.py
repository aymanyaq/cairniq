"""
Opportunity Funnel V2 — Tier-1 Backtest Harness (M5 / §11)
==========================================================
Validates funnel picks against realized forward returns using the walk-forward
signal-log that `opportunity_scanner._log_funnel_signals` writes per broad scan.

This is the "Tier-1" backtest of the spec: it measures the RECONSTRUCTABLE outcome
(did the pick beat SPY over the holding window?) for snapshots that have matured.
The flow/entry pillars themselves can't be reconstructed historically — but their
*effect* (the picks they produced) can be scored once enough time has passed.

Usage:
    from tools.funnel_backtest import evaluate_signal_log
    report = evaluate_signal_log(days_forward=21)   # ~1 trading month

Promotion gate (spec §11): flip the funnel to default only once this report shows
the high-conviction cohort beating SPY on hit-rate AND drawdown over a meaningful
sample. Until then the funnel stays as-is and the log keeps accumulating.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from agent.utils import safe_print
from tools.exception_logger import log_exceptions

_SIGNAL_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "user_data", "funnel_signal_log")

# Theme/sector name → SPDR sector ETF, for sector-relative alpha. SPY-relative
# alpha during a sector rotation mostly measures the funnel's sector TILT;
# sector-relative alpha isolates its SELECTION skill within the theme.
# Substring rules (lowercased) so rotation-map variants match ("Consumer
# Discret", "Comm Services", "Basic Materials", ...).
_SECTOR_ETF_RULES: list[tuple[str, str]] = [
    ("tech", "XLK"), ("semiconductor", "XLK"),
    ("industrial", "XLI"),
    ("financ", "XLF"),
    ("health", "XLV"),
    ("real estate", "XLRE"),
    ("discret", "XLY"),
    ("staple", "XLP"),
    ("material", "XLB"),
    ("utilit", "XLU"),
    ("comm", "XLC"),
    ("energy", "XLE"),
]


def _sector_etf(theme: str | None) -> str | None:
    if not theme:
        return None
    tl = theme.lower()
    for substr, etf in _SECTOR_ETF_RULES:
        if substr in tl:
            return etf
    return None


def _load_snapshots(log_dir: str) -> list[dict[str, Any]]:
    """Load every snapshot line from the daily jsonl files, newest-first dir scan."""
    snapshots: list[dict[str, Any]] = []
    if not os.path.isdir(log_dir):
        return snapshots
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".jsonl"):
            continue
        try:
            with open(os.path.join(log_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        snapshots.append(json.loads(line))
        except Exception:
            continue
    return snapshots


def _forward_return(symbol: str, start: datetime, days_forward: int) -> float | None:
    """
    Realized % return from `start` to ~days_forward calendar days later, via yfinance.
    Returns None if data is unavailable. Isolated for mockability in tests.
    """
    try:
        import yfinance as yf
        end = start + timedelta(days=days_forward + 5)  # padding for weekends/holidays
        hist = yf.Ticker(symbol).history(start=start.date().isoformat(), end=end.date().isoformat())
        if hist.empty or len(hist) < 2:
            return None
        close = hist["Close"].dropna()
        if len(close) < 2:
            return None
        entry = float(close.iloc[0])
        # take the close nearest to (but not before) days_forward trading-ish window
        target_idx = min(len(close) - 1, days_forward)
        exit_px = float(close.iloc[target_idx])
        if entry <= 0:
            return None
        return ((exit_px - entry) / entry) * 100.0
    except Exception:
        return None


@log_exceptions()
def evaluate_signal_log(
    days_forward: int = 21,
    as_of: datetime | None = None,
    log_dir: str | None = None,
    forward_return_fn=None,
    dedupe: bool = True,
) -> dict[str, Any]:
    """
    Score matured funnel snapshots against forward returns.

    Args:
        days_forward: holding window (calendar days) to measure the pick over.
        as_of: evaluation date (default: now, UTC). A snapshot is "matured" once
               snapshot_ts + days_forward <= as_of.
        log_dir: signal-log directory (default: user_data/funnel_signal_log).
        forward_return_fn: override for _forward_return (testing).
        dedupe: when True (default), each symbol is scored ONCE from its first
                matured appearance. Daily scans re-surface the same names, so
                scoring every row pseudo-replicates one signal ~10x and lets a
                handful of losers (or winners) dominate the aggregates.
                Pass False for the raw every-row view.

    Each pick is benchmarked against SPY and, when its theme maps to a SPDR
    sector ETF, against that sector (sector_alpha_pct isolates selection skill
    from sector tilt). Returns a report dict with overall + per-conviction +
    per-entry-stage stats, or {"status": "...", ...} when there isn't enough
    matured data yet.
    """
    log_dir = log_dir or _SIGNAL_LOG_DIR
    as_of = as_of or datetime.now(UTC)
    _raw_fwd = forward_return_fn or _forward_return

    # Memoize per (symbol, snapshot-ts): benchmark ETFs repeat across picks.
    _fwd_cache: dict[tuple, float | None] = {}

    def fwd_fn(symbol, start, days):
        key = (symbol, start, days)
        if key not in _fwd_cache:
            _fwd_cache[key] = _raw_fwd(symbol, start, days)
        return _fwd_cache[key]

    snapshots = _load_snapshots(log_dir)
    if not snapshots:
        return {"status": "no_data", "message": f"No signal-log snapshots in {log_dir}.",
                "snapshots": 0}

    matured = []
    for snap in snapshots:
        ts_raw = snap.get("ts")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except Exception:
            continue
        if ts + timedelta(days=days_forward) <= as_of:
            matured.append((ts, snap))

    if not matured:
        return {
            "status": "insufficient_maturity",
            "message": (f"{len(snapshots)} snapshot(s) exist but none have aged "
                        f"{days_forward}d yet. Re-run after they mature."),
            "snapshots": len(snapshots),
        }

    # Chronological order so "first seen" means the signal's first surfacing.
    matured.sort(key=lambda pair: pair[0])

    def _score_population(key: str) -> tuple[list[dict], int]:
        """Score one snapshot population ("picks" or "near_misses") into rows.
        Each population dedupes independently — a symbol can legitimately be a
        near-miss one day and a pick the next."""
        pop_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        dropped = 0
        for ts, snap in matured:
            spy_ret = fwd_fn("SPY", ts, days_forward)
            for item in snap.get(key, []) or []:
                sym = item.get("symbol")
                if not sym:
                    continue
                if dedupe and sym in seen:
                    dropped += 1
                    continue
                item_ret = fwd_fn(sym, ts, days_forward)
                if item_ret is None:
                    continue
                seen.add(sym)
                alpha = item_ret - spy_ret if spy_ret is not None else None
                bench = _sector_etf(item.get("theme"))
                bench_ret = fwd_fn(bench, ts, days_forward) if bench else None
                sector_alpha = item_ret - bench_ret if bench_ret is not None else None
                row = {
                    "symbol": sym,
                    "snapshot": ts.date().isoformat(),
                    "conviction": item.get("conviction", "Unknown"),
                    "entry_stage": item.get("entry_stage", "unknown"),
                    "return_pct": round(item_ret, 2),
                    "spy_return_pct": round(spy_ret, 2) if spy_ret is not None else None,
                    "alpha_pct": round(alpha, 2) if alpha is not None else None,
                    "sector_benchmark": bench,
                    "sector_alpha_pct": round(sector_alpha, 2) if sector_alpha is not None else None,
                }
                if key == "near_misses":
                    row["cut_reason"] = item.get("cut_reason", "unknown")
                pop_rows.append(row)
        return pop_rows, dropped

    rows, duplicate_rows_dropped = _score_population("picks")
    nm_rows, _nm_dropped = _score_population("near_misses")

    if not rows:
        return {"status": "no_priceable_picks", "message": "Matured snapshots had no priceable picks.",
                "matured_snapshots": len(matured)}

    def _agg(subset: list[dict]) -> dict[str, Any]:
        n = len(subset)
        alphas = [r["alpha_pct"] for r in subset if r["alpha_pct"] is not None]
        sec_alphas = [r["sector_alpha_pct"] for r in subset if r["sector_alpha_pct"] is not None]
        rets = [r["return_pct"] for r in subset]
        wins = [a for a in alphas if a > 0]
        sec_wins = [a for a in sec_alphas if a > 0]
        return {
            "n": n,
            "hit_rate_vs_spy": round(len(wins) / len(alphas), 3) if alphas else None,
            "avg_alpha_pct": round(sum(alphas) / len(alphas), 2) if alphas else None,
            "hit_rate_vs_sector": round(len(sec_wins) / len(sec_alphas), 3) if sec_alphas else None,
            "avg_sector_alpha_pct": round(sum(sec_alphas) / len(sec_alphas), 2) if sec_alphas else None,
            "avg_return_pct": round(sum(rets) / n, 2) if n else None,
            "worst_return_pct": round(min(rets), 2) if rets else None,
            "best_return_pct": round(max(rets), 2) if rets else None,
        }

    by_conviction: dict[str, Any] = {}
    for tier in sorted({r["conviction"] for r in rows}):
        by_conviction[tier] = _agg([r for r in rows if r["conviction"] == tier])
    by_entry: dict[str, Any] = {}
    for stage in sorted({r["entry_stage"] for r in rows}):
        by_entry[stage] = _agg([r for r in rows if r["entry_stage"] == stage])

    # ── Miss detector: how did the names the funnel REJECTED perform? ──
    near_miss_report: dict[str, Any] | None = None
    if nm_rows:
        picks_overall = _agg(rows)
        nm_overall = _agg(nm_rows)
        by_cut: dict[str, Any] = {}
        for reason in sorted({r["cut_reason"] for r in nm_rows}):
            by_cut[reason] = _agg([r for r in nm_rows if r["cut_reason"] == reason])

        def _regret(nm_val, picks_val):
            if nm_val is None or picks_val is None:
                return None
            return round(nm_val - picks_val, 2)

        regret_sector = _regret(nm_overall.get("avg_sector_alpha_pct"),
                                picks_overall.get("avg_sector_alpha_pct"))
        regret_spy = _regret(nm_overall.get("avg_alpha_pct"),
                             picks_overall.get("avg_alpha_pct"))
        basis = regret_sector if regret_sector is not None else regret_spy
        if basis is None:
            note = "Not enough benchmark data to compare cut names against picks."
        elif basis > 1.0:
            note = (f"REGRET: cut names outperformed picks by {basis:+.1f}pp — "
                    "one or more gates may be rejecting winners (see by_cut_reason).")
        elif basis < -1.0:
            note = (f"Gates added value: cut names underperformed picks by {abs(basis):.1f}pp.")
        else:
            note = "Cut names performed roughly in line with picks — gates neutral so far."
        near_miss_report = {
            "overall": nm_overall,
            "by_cut_reason": by_cut,
            "regret_vs_picks_alpha_pct": regret_spy,
            "regret_vs_picks_sector_alpha_pct": regret_sector,
            "regret_note": note,
            "detail": nm_rows,
        }

    return {
        "status": "ok",
        "days_forward": days_forward,
        "as_of": as_of.isoformat(),
        "matured_snapshots": len(matured),
        "evaluated_picks": len(rows),
        "dedupe": dedupe,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "history_span_days": (matured[-1][0] - matured[0][0]).days,
        "overall": _agg(rows),
        "by_conviction": by_conviction,
        "by_entry_stage": by_entry,
        "detail": rows,
        **({"near_misses": near_miss_report} if near_miss_report else {}),
        "promotion_note": (
            "Flip-default gate: require a meaningful sample where High Conviction / "
            "Exceptional beat SPY on hit_rate AND show contained worst_return_pct."
        ),
    }


@log_exceptions()
def get_funnel_scorecard_data(horizons: tuple = (14, 21), log_dir: str | None = None) -> dict[str, Any]:
    """
    Compact multi-horizon track record of the opportunity funnel, built for the
    agent to cite when it presents scan results. Deduped (one row per symbol,
    first surfacing), detail rows stripped, honesty caveats attached.
    """
    out: dict[str, Any] = {"status": "ok", "horizons": {}, "caveats": []}
    max_n = 0
    max_span = 0
    for h in horizons:
        r = evaluate_signal_log(days_forward=h, log_dir=log_dir)
        if r.get("status") != "ok":
            out["horizons"][f"{h}d"] = {"status": r.get("status"), "message": r.get("message")}
            continue
        max_n = max(max_n, r["evaluated_picks"])
        max_span = max(max_span, r.get("history_span_days", 0))
        horizon_entry = {
            "matured_snapshots": r["matured_snapshots"],
            "unique_signals": r["evaluated_picks"],
            "overall": r["overall"],
            "by_conviction": r["by_conviction"],
            "by_entry_stage": r["by_entry_stage"],
        }
        nm = r.get("near_misses")
        if nm:
            horizon_entry["near_misses"] = {
                "overall": nm["overall"],
                "by_cut_reason": nm["by_cut_reason"],
                "regret_vs_picks_sector_alpha_pct": nm["regret_vs_picks_sector_alpha_pct"],
                "regret_note": nm["regret_note"],
            }
        out["horizons"][f"{h}d"] = horizon_entry
    if not any(v.get("overall") for v in out["horizons"].values() if isinstance(v, dict)):
        out["status"] = "no_matured_data"
        out["caveats"].append("No matured signals yet — the scanner has no measurable track record.")
        return out
    if max_n < 100:
        out["caveats"].append(
            f"Small sample ({max_n} unique signals) — treat hit rates as indicative, not statistical.")
    if max_span < 90:
        out["caveats"].append(
            f"History spans only ~{max_span} days (a single market regime); results may not generalize.")
    out["caveats"].append(
        "hit_rate_vs_sector / avg_sector_alpha_pct isolate stock selection from sector tilt; "
        "SPY-relative numbers mix both.")
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(evaluate_signal_log())
