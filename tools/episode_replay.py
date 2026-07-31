"""Historical episode replay — Advisor Roadmap 4.3.

What this replaces, and why it matters more than "add a stress test":

  - `tools/simulation.py::simulate_scenario` multiplies each holding's beta by a
    HARDCODED market drop (recession −35%, tech_crash −45%) from a table of round
    numbers labelled "2008-style" and "Dot-com style". Nothing in it was measured.
  - `tools/predictive.py::match_historical_regime` matches today's macro to six
    hand-written regimes and returns hand-written outcomes in fields called
    `forecast_3mo` / `forecast_1yr`.

Both produce confident figures with no data behind them, from functions whose
names assert history. This module answers the same question with the actual
daily paths: what did THESE weights do, in THAT window.

**The honesty problem this design is built around.** Applying today's weights to
2008 needs 2008 prices for today's names, and many do not have them — recent
ETFs, recent listings, anything post-dating the episode. Silently dropping them
and renormalising is worse than useless: the names most likely to be missing are
the newest and most volatile, so the drawdown comes back FLATTERING and reads as
measured. So coverage is computed BY WEIGHT, reported on every result, and a
replay below `MIN_COVERAGE` refuses rather than returns a number. A partial
replay states exactly which weight it could not see.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

# Below this share of portfolio weight with usable history, a replay is refused
# rather than reported. 60% is deliberately permissive — the point is to block
# the "three of forty names existed in 2008" case, not to demand completeness.
MIN_COVERAGE = 0.60

# Real episodes, dated from index peak to trough with the recovery date being the
# first close back at the prior peak. These dates are FACTS about the S&P 500 and
# are the only constants in this module; every number derived from them is
# measured from price data at run time.
EPISODES: dict[str, dict[str, Any]] = {
    "gfc": {
        "name": "Global Financial Crisis",
        "peak": "2007-10-09",
        "trough": "2009-03-09",
        "recovered": "2013-03-28",
        "note": "Systemic leverage unwind. The slowest recovery in the set — about 5.5 years back to the prior peak.",
    },
    "covid": {
        "name": "COVID crash",
        "peak": "2020-02-19",
        "trough": "2020-03-23",
        "recovered": "2020-08-18",
        "note": "The fastest large drawdown on record, and the fastest recovery — about 5 months.",
    },
    "bear_2022": {
        "name": "2022 rate-shock bear market",
        "peak": "2022-01-03",
        "trough": "2022-10-12",
        "recovered": "2024-01-19",
        "note": "Duration repricing: bonds fell WITH equities, so the usual diversifier did not diversify.",
    },
    "dotcom": {
        "name": "Dot-com bust",
        "peak": "2000-03-24",
        "trough": "2002-10-09",
        "recovered": "2007-05-30",
        "note": "Concentrated in technology. Relevant to a book with heavy AI/semiconductor weight.",
    },
    # Added for 4.3b. Shallow by the standards of the others (-19.8%) and the only
    # POLICY-driven episode in the set — a tariff/rate-guidance selloff that resolved
    # on a policy pivot rather than on economic repair. That is the Swing spec's own
    # home turf, which is why the out-of-sample study needs it.
    "q4_2018": {
        "name": "2018 Q4 policy selloff",
        "peak": "2018-09-20",
        "trough": "2018-12-24",
        "recovered": "2019-04-23",
        "note": "Trade-war escalation plus hawkish rate guidance. Recovered in four months once the Fed pivoted — the fastest full recovery in the set after COVID.",
    },
}


def list_episodes() -> list[dict[str, Any]]:
    """The available episodes and their dates, for callers offering a choice."""
    return [{"key": key, **{k: v for k, v in ep.items()}} for key, ep in EPISODES.items()]


def _normalize_weights(symbols: list[str], weights: list[float] | None) -> dict[str, float]:
    """Symbol → weight, summing to 1. Equal-weight when weights are absent."""
    if not symbols:
        return {}
    if not weights or len(weights) != len(symbols):
        return {s: 1.0 / len(symbols) for s in symbols}
    merged: dict[str, float] = {}
    for sym, w in zip(symbols, weights):
        merged[sym] = merged.get(sym, 0.0) + float(w)
    total = sum(merged.values())
    if total <= 0:
        return {s: 1.0 / len(symbols) for s in symbols}
    return {s: w / total for s, w in merged.items()}


@log_exceptions()
def replay_episode(
    symbols: list[str],
    weights: list[float] | None = None,
    episode: str = "covid",
    returns_fn: Any = None,
) -> dict[str, Any]:
    """Apply today's weights to one historical episode's ACTUAL daily returns.

    `returns_fn(symbols, start, end) -> (returns_df, valid_symbols)` is injected
    in tests; production uses the 4.1 estimation layer, so the paths are measured
    in the profile's base currency and a CAD holder sees the FX contribution to
    the drawdown rather than a USD-only view of it.
    """
    key = str(episode or "").strip().lower()
    spec = EPISODES.get(key)
    if not spec:
        return {"error": f"Unknown episode '{episode}'. Known: {', '.join(EPISODES)}"}

    weight_map = _normalize_weights([str(s).upper().strip() for s in symbols or []], weights)
    if not weight_map:
        return {"error": "No symbols supplied"}

    if returns_fn is None:
        from tools.portfolio_analytics import _get_returns

        def returns_fn(syms, start, end):  # noqa: E306
            return _get_returns(syms, start=start, end=end)

    # Fetch the episode window plus the recovery leg, so time-to-recover is
    # measured from the same series rather than assumed from the index's dates.
    fetch_end = spec.get("recovered") or spec["trough"]
    returns, valid = returns_fn(list(weight_map), spec["peak"], fetch_end)

    if returns is None or getattr(returns, "empty", True) or not valid:
        return {
            "error": "No price history available for this episode",
            "episode": spec["name"],
            "coverage_pct": 0.0,
        }

    covered = [s for s in valid if s in weight_map]
    coverage = sum(weight_map[s] for s in covered)
    missing = sorted(set(weight_map) - set(covered))

    if coverage < MIN_COVERAGE:
        return {
            "error": (
                f"Only {coverage * 100:.0f}% of portfolio weight has price history back to "
                f"{spec['peak']} — too little to replay {spec['name']} honestly. The names "
                f"without history are typically the newest and most volatile, so a replay of "
                f"the remainder would understate the drawdown while reading as measured."
            ),
            "episode": spec["name"],
            "coverage_pct": round(coverage * 100, 1),
            "missing_symbols": missing,
            "basis": "measured",
        }

    # Renormalise across what we can actually see, and say so.
    sub_weights = np.array([weight_map[s] for s in covered], dtype=float)
    sub_weights = sub_weights / sub_weights.sum()
    path = returns[covered].dot(sub_weights)

    drawdown_window = path.loc[: spec["trough"]] if spec.get("trough") else path
    result = _measure(path, drawdown_window, returns, covered, sub_weights, spec)
    result.update({
        "episode": spec["name"],
        "episode_key": key,
        "window": {"peak": spec["peak"], "trough": spec["trough"],
                   "index_recovered": spec.get("recovered")},
        "note": spec["note"],
        "coverage_pct": round(coverage * 100, 1),
        "symbols_replayed": covered,
        "missing_symbols": missing,
        # 2.7: this figure was computed from price data, unlike the tools this
        # module replaces. The marker is what lets prose attribute it honestly.
        "basis": "measured",
    })
    if missing:
        result["data_warning"] = (
            f"{len(missing)} holding(s) had no price history in this window and were excluded, "
            f"then the rest renormalised: {', '.join(missing)}. Newer holdings are usually the "
            f"more volatile ones, so treat this as a floor on the drawdown, not an estimate of it."
        )
    return result


def _measure(
    path: pd.Series,
    drawdown_window: pd.Series,
    returns: pd.DataFrame,
    covered: list[str],
    weights: np.ndarray,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Peak-to-trough, time-to-recover, worst positions, correlation spike.

    Drawdown is measured from the EPISODE PEAK, not from a running max inside the
    window. The distinction is not cosmetic: the window already starts at the
    peak by construction, and the first return in the series is
    `P(peak+1)/P(peak)`, so the cumulative product IS the level relative to the
    peak. Using an expanding max instead re-anchors on the first day *after* the
    peak — which, in any episode that falls immediately, is already below it. On
    a 50/30/20 SPY/AGG/QQQ book through COVID that understated the fall by half a
    point (−17.2% vs the true −17.7%), always in the flattering direction. A
    stress test that quietly reports a shallower loss than really happened is
    worse than no stress test.
    """
    cumulative = (1 + drawdown_window).cumprod()
    max_dd = float(cumulative.min() - 1.0) if len(cumulative) else 0.0
    trough_date = cumulative.idxmin() if len(cumulative) else None

    # Time to recover, measured from THIS portfolio's own path rather than the
    # index's recovery date — a different book recovers on a different day.
    full_cum = (1 + path).cumprod()
    recovered_on, days_to_recover = None, None
    if trough_date is not None:
        # Recovery means back to the EPISODE PEAK level (cumulative 1.0), for the
        # same reason the drawdown is measured from it — recovering to a
        # mid-decline local high is not recovering.
        back = full_cum.loc[trough_date:]
        regained = back[back >= 1.0]
        if len(regained):
            recovered_on = regained.index[0]
            days_to_recover = int((recovered_on - trough_date).days)

    # Worst positions over the drawdown leg.
    leg = returns.loc[: spec["trough"], covered] if spec.get("trough") else returns[covered]
    per_symbol = ((1 + leg).prod() - 1).sort_values()
    worst = [
        {"symbol": sym, "return_pct": round(float(val) * 100, 1)}
        for sym, val in per_symbol.head(5).items()
    ]

    # Correlation spike: what stopped diversifying. The MEAN alone is a weak
    # signal and this was caught on real data — a 50/30/20 SPY/AGG/QQQ book
    # showed 0.49 in the fall and 0.49 over the full window, while underneath
    # SPY/QQQ went 0.956 -> 0.985 and the bond leg stayed loose. Averaging a
    # tightening equity block against a decorrelated sleeve hides the only fact
    # worth acting on, so the pair that tightened MOST is reported beside it.
    corr_stress = _mean_pairwise_corr(leg)
    corr_full = _mean_pairwise_corr(returns[covered])
    tightest = _biggest_correlation_shift(leg, returns[covered])

    return {
        "peak_to_trough_pct": round(max_dd * 100, 1),
        "trough_date": str(trough_date.date()) if trough_date is not None else None,
        "portfolio_recovered_on": str(recovered_on.date()) if recovered_on is not None else None,
        "days_to_recover": days_to_recover,
        "recovery_note": (
            None if days_to_recover is not None else
            "This portfolio had not regained its pre-drawdown level by the end of the "
            "fetched window — the recovery took longer than the index's own."
        ),
        "worst_positions": worst,
        "mean_correlation_in_drawdown": corr_stress,
        "mean_correlation_full_window": corr_full,
        "largest_correlation_shift": tightest,
        # The finding is the LEVEL in the drawdown, not only the change. Two
        # sleeves at 0.99 are one position whether they moved from 0.96 or not —
        # an earlier version reported "diversification held" for exactly that
        # case, because the delta was small.
        "correlation_note": _correlation_note(tightest),
    }


def _correlation_note(shift: dict[str, Any] | None) -> str | None:
    """Plain reading of what the correlation pair means, level first."""
    if not shift:
        return None
    a, b = shift["pair"]
    if shift["in_drawdown"] >= 0.85:
        moved = (f" (up from {shift['full']})" if shift["change"] > 0.02
                 else " — and they were already that tight beforehand")
        return (f"{a} and {b} were {shift['in_drawdown']} correlated through the fall{moved}. "
                f"In this episode they behaved as one position, not two.")
    if shift["change"] > 0.15:
        return (f"{a} and {b} tightened from {shift['full']} to {shift['in_drawdown']} during "
                f"the fall — they diversified in calm markets and stopped when it mattered.")
    return "No pair became materially more correlated in this episode."


def _biggest_correlation_shift(leg: pd.DataFrame, full: pd.DataFrame) -> dict[str, Any] | None:
    """The pair whose correlation rose most from the full window to the fall."""
    if leg is None or leg.shape[1] < 2 or len(leg) < 3 or len(full) < 3:
        return None
    a, b = leg.corr(), full.corr()
    best = None
    for i, x in enumerate(leg.columns):
        for y in leg.columns[i + 1:]:
            try:
                delta = float(a.loc[x, y]) - float(b.loc[x, y])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isnan(delta):
                continue
            if best is None or delta > best["change"]:
                best = {
                    "pair": [x, y],
                    "in_drawdown": round(float(a.loc[x, y]), 2),
                    "full": round(float(b.loc[x, y]), 2),
                    "change": round(delta, 2),
                }
    return best


def _mean_pairwise_corr(frame: pd.DataFrame) -> float | None:
    """Mean off-diagonal correlation, or None when it cannot be computed."""
    if frame is None or frame.shape[1] < 2 or len(frame) < 3:
        return None
    corr = frame.corr()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr.values[mask]
    values = values[~np.isnan(values)]
    return round(float(values.mean()), 2) if values.size else None


@log_exceptions()
def replay_all_episodes(
    symbols: list[str],
    weights: list[float] | None = None,
    returns_fn: Any = None,
) -> dict[str, Any]:
    """Replay every episode. Episodes that cannot be covered say so individually.

    Deliberately does NOT aggregate into a single "expected crash" figure:
    averaging a 2008 that this book could not be measured through with a 2022 it
    could would produce exactly the kind of confident composite this module
    exists to remove.
    """
    results, refused = {}, []
    for key in EPISODES:
        outcome = replay_episode(symbols, weights, episode=key, returns_fn=returns_fn)
        results[key] = outcome
        if outcome.get("error"):
            refused.append(key)
    replayed = {k: v for k, v in results.items() if not v.get("error")}
    worst_key = min(
        replayed, key=lambda k: replayed[k]["peak_to_trough_pct"], default=None
    )
    return {
        "episodes": results,
        "replayed": sorted(replayed),
        "not_replayable": refused,
        "worst_episode": (
            {"key": worst_key, "name": replayed[worst_key]["episode"],
             "peak_to_trough_pct": replayed[worst_key]["peak_to_trough_pct"]}
            if worst_key else None
        ),
        "basis": "measured",
    }
