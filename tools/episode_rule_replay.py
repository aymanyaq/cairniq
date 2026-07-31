"""Episode deployment RULE replayer — Advisor Roadmap 4.3b.

4.3 replays WEIGHTS through a window: "what did this book do in COVID". This
replays RULES through a window: triggers firing, tranches deploying, positions
exiting at resolution. It exists to answer one question the Swing spec
(`user_data/docs/SWING_ENGINE_SPEC.md`) cannot answer about itself — does the
Episode Deployment Engine have any edge OUTSIDE the single in-sample year it was
tuned on (Jul 2025 → Jul 2026)?

**Why this inverts the spec's own ordering.** The spec's milestone table makes M2,
its go/no-go gate, depend on M1, the engine. A gate that requires building the
thing it gates is not a gate. This module is the cheap half of M2 — a few hundred
lines and no user surface — run BEFORE anything is committed to.

WHAT IS MEASURED HONESTLY, AND WHAT IS A PROXY
----------------------------------------------
The level rules replay exactly: "deploy 40% of the sleeve at −5% from the running
peak, the remaining 60% at −10%" is a pure price rule, and price history is real.

The **de-escalation bell is not directly replayable and this module does not
pretend otherwise.** The bell fires on classified *resolution events* — a
ceasefire, a walk-back, tariffs paused — read from `trump_tracker`, the catalyst
engine and geopolitical feeds. No historical archive of those classifications
exists for 2018/2020/2022, and hand-typing one from hindsight would make the
study's headline a set of authored constants (exactly what Roadmap 2.7 exists to
stamp out — and it would be authored by someone who already knows which rallies
held).

So the bell arm runs on a **mechanical relief proxy**, which is the spec's own
§5 answer to the same problem for Lane C: a strong up-session with VIX falling,
inside an open episode. This tests the bell's SKELETON — "does adding capital on
confirmed relief beat waiting for the next level" — while stripping its event
classification. The distinction matters in both directions and the report must
carry it:

  * if the proxy bell FAILS, the bell's entire claimed value rests on event
    classification that cannot be validated historically — a much weaker basis
    than "+3.1pp measured", because the in-sample year cannot separate the two;
  * if the proxy bell CLEARS, the mechanical skeleton alone has edge and the
    classifier is upside rather than load-bearing.

An optional `resolution_dates` argument accepts a genuinely sourced event calendar
if one is ever compiled, and the result records which arm ran.

POINT-IN-TIME HYGIENE (spec §10, and the whole validity of this)
---------------------------------------------------------------
* The running peak at day *t* uses closes up to and including *t* — never the
  window's global max, which is the classic way a replay learns the future.
* A trigger evaluated on the close of *t* enters at the **next session's open**.
  Same for exits. Nothing transacts at a price the rule could not have seen.
* Drawdown and returns come from adjusted daily bars; no intraday assumptions,
  so stops are close-triggered rather than touch-triggered (stated, not hidden —
  a touch-triggered stop would report better fills than this data can support).
* Using SPY as the deployment instrument means **no survivorship bias at all**;
  the moment a hardest-hit-names picker is added, today's universe leaks into a
  2008 replay and the report must say so.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

# The user's own tranche plan, per SWING_ENGINE_SPEC §0. Levels are drawdown
# percentages from the running peak; fractions are of the INITIAL sleeve.
TRANCHE_LEVELS: tuple[tuple[float, float], ...] = ((-5.0, 0.40), (-10.0, 0.60))

# An episode is "open" — and the bell armed — once the drawdown clears this.
# Shallower than the first tranche level on purpose: the bell's claimed value in
# the in-sample year came from shallow episodes that never reached −10%.
EPISODE_OPEN_PCT = -3.0

# Mechanical relief proxy for the bell (see module docstring).
RELIEF_SESSION_PCT = 1.5      # index up this much on the day...
RELIEF_VIX_DROP_PCT = -5.0    # ...with VIX down this much, when VIX is available.

DISASTER_STOP_PCT = -25.0     # spec §0

# The spec names "max-hold" as a component but never sets it. That silence is
# load-bearing: at 252 sessions the 2022 legs are forced out in Jan-Feb 2023 at a
# loss and the sleeve then sits in cash through the recovery, while holding to
# resolution turns the same trades positive. So the study runs BOTH and reports
# the sensitivity rather than letting a number chosen here decide the verdict.
MAX_HOLD_SESSIONS = 252
NO_MAX_HOLD = 10**9

# A tranche smaller than this is not a position, it is a rounding artifact — and
# it was corrupting the gate: with the sleeve fully committed at -10%, the bell
# opened a 0.0%-capital leg in COVID that lost 25.8% of nothing and still counted
# as a losing leg, dragging the pooled hit rate from 50% to 44%.
MIN_LEG_CAPITAL = 0.005


@log_exceptions()
def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Daily OHLC for one symbol over a window — the production data path.

    Kept out of :func:`replay_rules`, which takes bars directly, so the state
    machine stays testable offline and no test can accidentally depend on a live
    Yahoo response for a 2008 date.
    """
    from tools.yf_utils import download_safe

    data = download_safe([symbol], period=None, start=start, end=end)
    if data is None or getattr(data, "empty", True):
        return None
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance 1.x returns (field, ticker); collapse to the single ticker.
        data = data.xs(symbol, axis=1, level=1) if symbol in data.columns.get_level_values(1) else data
    return data


@log_exceptions()
def fetch_close_series(symbol: str, start: str, end: str) -> pd.Series | None:
    """Closing series for an auxiliary symbol (VIX for the bell, ^IRX for cash)."""
    frame = fetch_bars(symbol, start, end)
    if frame is None or getattr(frame, "empty", True):
        return None
    cols = {str(c).strip().lower(): c for c in (
        [c[0] for c in frame.columns] if isinstance(frame.columns, pd.MultiIndex) else frame.columns
    )}
    if "close" not in cols:
        return None
    series = frame[cols["close"]] if not isinstance(frame.columns, pd.MultiIndex) else frame.xs("Close", axis=1, level=0).iloc[:, 0]
    return series.dropna()


class _Leg:
    """One deployed tranche. The unit the spec's §5 gate is computed over."""

    __slots__ = ("label", "entry_date", "entry_price", "capital", "exit_date",
                 "exit_price", "exit_reason", "worst_close", "sessions_held")

    def __init__(self, label: str, entry_date: Any, entry_price: float, capital: float):
        self.label = label
        self.entry_date = entry_date
        self.entry_price = float(entry_price)
        self.capital = float(capital)
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.worst_close = float(entry_price)
        self.sessions_held = 0

    @property
    def open(self) -> bool:
        return self.exit_price is None

    def value(self, price: float) -> float:
        return self.capital * (float(price) / self.entry_price)

    def to_dict(self) -> dict[str, Any]:
        ret = (self.exit_price / self.entry_price - 1.0) if self.exit_price else 0.0
        return {
            "label": self.label,
            "entry_date": str(getattr(self.entry_date, "date", lambda: self.entry_date)()),
            "entry_price": round(self.entry_price, 2),
            "exit_date": str(getattr(self.exit_date, "date", lambda: self.exit_date)()),
            "exit_price": round(float(self.exit_price), 2) if self.exit_price else None,
            "exit_reason": self.exit_reason,
            "capital_pct": round(self.capital * 100, 1),
            "return_pct": round(ret * 100, 2),
            # Close-based, because daily bars cannot see an intraday extreme.
            "max_adverse_close_pct": round((self.worst_close / self.entry_price - 1.0) * 100, 2),
            "sessions_held": self.sessions_held,
        }


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Accept yfinance-shaped frames and return lowercase open/close only."""
    frame = bars.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [c[0] for c in frame.columns]
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = {"open", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"bars must have open and close columns; missing {sorted(missing)}")
    return frame[["open", "close"]].dropna()


@log_exceptions()
def replay_rules(
    bars: pd.DataFrame,
    *,
    levels: tuple[tuple[float, float], ...] = TRANCHE_LEVELS,
    bell: bool = False,
    bell_fraction: float = 0.40,
    vix: pd.Series | None = None,
    resolution_dates: list[str] | None = None,
    risk_free: pd.Series | None = None,
    episode_open_pct: float = EPISODE_OPEN_PCT,
    disaster_stop_pct: float = DISASTER_STOP_PCT,
    max_hold_sessions: int = MAX_HOLD_SESSIONS,
    min_leg_capital: float = MIN_LEG_CAPITAL,
) -> dict[str, Any]:
    """Replay the tranche-deployment rules over one window of daily bars.

    The sleeve starts as 1.0 of cash. Idle cash accrues ``risk_free`` (an annual
    percentage series) if supplied — without that, cash-vs-strategy is rigged in
    the strategy's favour, since the comparison is precisely "is deploying this
    dry powder better than leaving it earning the bill rate".

    **The bell competes with the ladder for the same sleeve, and that is not an
    implementation detail.** The user's plan commits 40% + 60% — the entire
    sleeve — by −10%, so a bell firing earlier and shallower can only be funded by
    taking capital the deeper, cheaper level would have used. In 2018 Q4 that is
    exactly what happened: the bell bought at 249.42 with capital the −10% level
    would have spent at 230.71. A model that funded the bell from outside the
    sleeve would report a marginal gain that the plan cannot actually finance.

    Returns legs, sleeve return, and the exit-reason breakdown. All the gate
    statistics are computed by :func:`summarize`.
    """
    frame = _normalize_bars(bars)
    if len(frame) < 2:
        return {"error": "Need at least two sessions to replay (entries are at the NEXT open)."}

    resolution_set = {str(d)[:10] for d in (resolution_dates or [])}
    rf = risk_free.reindex(frame.index).ffill() if risk_free is not None else None

    cash = 1.0
    legs: list[_Leg] = []
    peak = float(frame["close"].iloc[0])
    episode_active = False
    fired_levels: set[float] = set()
    bell_fired = False
    episodes_seen = 0
    bell_signals = 0
    deployed_sessions = 0

    # Actions decided at the close of session i execute at the open of i+1.
    pending_entries: list[tuple[str, float]] = []
    pending_exits: list[tuple[_Leg, str]] = []

    dates = list(frame.index)
    for i, date in enumerate(dates):
        open_px = float(frame["open"].iloc[i])
        close_px = float(frame["close"].iloc[i])

        # --- execute what yesterday's close decided, at today's open ----------
        for leg, reason in pending_exits:
            if leg.open:
                leg.exit_price = open_px
                leg.exit_date = date
                leg.exit_reason = reason
                cash += leg.value(open_px)
        pending_exits = []

        for label, fraction in pending_entries:
            amount = min(fraction, cash)
            if amount >= min_leg_capital:
                legs.append(_Leg(label, date, open_px, amount))
                cash -= amount
        pending_entries = []

        # --- accrue the bill rate on idle cash for this session ---------------
        if rf is not None and cash > 0:
            annual = rf.iloc[i]
            if pd.notna(annual):
                cash *= (1.0 + float(annual) / 100.0) ** (1.0 / 252.0)

        open_legs = [leg for leg in legs if leg.open]
        if open_legs:
            deployed_sessions += 1
        for leg in open_legs:
            leg.sessions_held += 1
            leg.worst_close = min(leg.worst_close, close_px)

        # --- observe the close, decide for tomorrow ---------------------------
        peak = max(peak, close_px)          # point-in-time: closes up to today only
        drawdown = (close_px / peak - 1.0) * 100.0

        if not episode_active and drawdown <= episode_open_pct:
            episode_active = True
            episodes_seen += 1

        if episode_active and drawdown >= 0.0:
            # Resolution: back at the pre-episode peak. Everything comes off.
            for leg in open_legs:
                pending_exits.append((leg, "resolution"))
            episode_active = False
            fired_levels = set()
            bell_fired = False
        elif episode_active:
            for threshold, fraction in levels:
                if threshold not in fired_levels and drawdown <= threshold:
                    fired_levels.add(threshold)
                    pending_entries.append((f"level {threshold:+.0f}%", fraction))

            if bell and not bell_fired and cash * bell_fraction >= min_leg_capital:
                if _bell_rings(frame, vix, i, date, resolution_set):
                    bell_fired = True
                    bell_signals += 1
                    pending_entries.append(("bell", cash * bell_fraction))

        # --- risk exits, evaluated on the close, executed next open -----------
        for leg in open_legs:
            if any(leg is queued for queued, _ in pending_exits):
                continue
            if (close_px / leg.entry_price - 1.0) * 100.0 <= disaster_stop_pct:
                pending_exits.append((leg, "disaster_stop"))
            elif leg.sessions_held >= max_hold_sessions:
                pending_exits.append((leg, "max_hold"))

    # Anything still open at the end of the window is marked as such rather than
    # silently valued as a win — an unresolved leg is not a resolved one.
    final_close = float(frame["close"].iloc[-1])
    for leg in legs:
        if leg.open:
            leg.exit_price = final_close
            leg.exit_date = dates[-1]
            leg.exit_reason = "window_end_unresolved"
            cash += leg.value(final_close)

    return {
        "legs": [leg.to_dict() for leg in legs],
        "sleeve_return_pct": round((cash - 1.0) * 100, 2),
        "episodes_seen": episodes_seen,
        "bell_signals": bell_signals,
        "deployed_session_pct": round(deployed_sessions / len(dates) * 100, 1),
        "sessions": len(dates),
        "bell_arm": (
            "sourced resolution calendar" if resolution_set
            else "mechanical relief proxy" if bell else "levels only"
        ),
        "vix_available": vix is not None,
    }


def _bell_rings(
    frame: pd.DataFrame,
    vix: pd.Series | None,
    i: int,
    date: Any,
    resolution_set: set[str],
) -> bool:
    """Did the de-escalation bell ring at the close of session ``i``?

    With a sourced calendar the date decides. Without one, the mechanical proxy:
    a strong up-session, confirmed by VIX falling when VIX is available. The VIX
    leg matters — a big up-day with VIX still climbing is the classic bear-market
    rally the bell is supposed NOT to buy, and dropping the check because the data
    is inconvenient would quietly widen the rule being tested.
    """
    if resolution_set:
        return str(getattr(date, "date", lambda: date)()) in resolution_set

    if i == 0:
        return False
    prev_close = float(frame["close"].iloc[i - 1])
    session_pct = (float(frame["close"].iloc[i]) / prev_close - 1.0) * 100.0
    if session_pct < RELIEF_SESSION_PCT:
        return False

    if vix is None:
        return True
    try:
        vix_today = float(vix.reindex([frame.index[i]]).iloc[0])
        vix_prev = float(vix.reindex([frame.index[i - 1]]).iloc[0])
    except (KeyError, IndexError, TypeError, ValueError):
        return True
    if pd.isna(vix_today) or pd.isna(vix_prev) or vix_prev == 0:
        return True
    return (vix_today / vix_prev - 1.0) * 100.0 <= RELIEF_VIX_DROP_PCT


@log_exceptions()
def summarize(result: dict[str, Any], gate_hit_rate: float = 55.0,
              gate_profit_factor: float = 1.3) -> dict[str, Any]:
    """Leg statistics and the spec's own §5 gate verdict.

    The gate is the SPEC's (hit rate ≥ 55%, profit factor ≥ 1.3), applied
    unchanged — the point of an out-of-sample study is to hold a rule to the bar
    its author set, not to a bar chosen once the numbers are known.
    """
    legs = result.get("legs", [])
    if not legs:
        return {
            "n_legs": 0, "hit_rate_pct": None, "profit_factor": None,
            "gate_passed": False,
            "gate_note": "No deployments — the rules never triggered in this window.",
        }

    returns = [leg["return_pct"] for leg in legs]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    wilson = _wilson_lower_bound(len(wins), len(returns))
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    # A run with no losing leg has an undefined profit factor, not an infinite
    # edge. Reported as None so it cannot be averaged into a flattering number.
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
    hit_rate = round(len(wins) / len(returns) * 100, 1)

    passed = hit_rate >= gate_hit_rate and (
        profit_factor is None or profit_factor >= gate_profit_factor
    )
    reasons = []
    if hit_rate < gate_hit_rate:
        reasons.append(f"hit rate {hit_rate}% < {gate_hit_rate}%")
    if profit_factor is not None and profit_factor < gate_profit_factor:
        reasons.append(f"profit factor {profit_factor} < {gate_profit_factor}")

    exit_reasons: dict[str, int] = {}
    for leg in legs:
        exit_reasons[leg["exit_reason"]] = exit_reasons.get(leg["exit_reason"], 0) + 1

    return {
        "n_legs": len(legs),
        "hit_rate_pct": hit_rate,
        # The spec's OWN humility measure (§3.6a): confidence is built on the Wilson
        # lower bound precisely so a handful of trades cannot print a high number.
        # Held to the same standard here — three episodes is a small sample no matter
        # how clean the replay is, and the raw hit rate is the flattering statistic.
        "hit_rate_wilson_lower_pct": wilson,
        "small_sample_note": (
            f"Raw hit rate {hit_rate}% on {len(returns)} legs; the Wilson 95% lower "
            f"bound is {wilson}%. A pass here is a non-falsification, not a validation."
        ),
        "avg_leg_return_pct": round(sum(returns) / len(returns), 2),
        "median_leg_return_pct": round(sorted(returns)[len(returns) // 2], 2),
        "best_leg_pct": round(max(returns), 2),
        "worst_leg_pct": round(min(returns), 2),
        "profit_factor": profit_factor,
        "worst_max_adverse_close_pct": round(min(leg["max_adverse_close_pct"] for leg in legs), 2),
        "exit_reasons": exit_reasons,
        "gate_passed": passed,
        "gate_note": (
            "clears the spec's own §5 gate" if passed else "fails: " + "; ".join(reasons)
        ),
        "profit_factor_note": (
            None if profit_factor is not None
            else "undefined — no losing leg in this sample, which is a small-n artifact, not an infinite edge"
        ),
    }


def _wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float | None:
    """Wilson 95% lower bound on the hit rate, in percent.

    Same estimator the spec's confidence score is built on (§3.6a). Reported here
    so the study cannot quote "83% hit rate" off six legs without the number that
    says how little six legs supports.
    """
    if n <= 0:
        return None
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return round(max(0.0, (centre - margin) / denom) * 100, 1)


@log_exceptions()
def benchmark_cash(bars: pd.DataFrame, risk_free: pd.Series | None) -> dict[str, Any]:
    """What the same sleeve earned sitting in bills over the same window."""
    frame = _normalize_bars(bars)
    if risk_free is None:
        return {
            "return_pct": None,
            "note": "No risk-free series available — cash return could not be measured.",
        }
    rf = risk_free.reindex(frame.index).ffill()
    value = 1.0
    for annual in rf:
        if pd.notna(annual):
            value *= (1.0 + float(annual) / 100.0) ** (1.0 / 252.0)
    return {
        "return_pct": round((value - 1.0) * 100, 2),
        "mean_annual_rate_pct": round(float(rf.mean()), 2) if len(rf.dropna()) else None,
    }


@log_exceptions()
def benchmark_buy_and_hold(bars: pd.DataFrame) -> dict[str, Any]:
    """Full sleeve in the instrument from the window's first open to its last close.

    **Read this one with its artifact attached.** These windows run peak → recovery
    by construction, and "recovery" is defined as the index regaining the peak — so
    buy-and-hold over the whole window is mechanically ≈ 0%. It is reported because
    the roadmap asks for both benchmarks, but a strategy beating it here has beaten
    an arithmetic identity, not a competing plan. **Cash is the benchmark that
    means something for this capital.**
    """
    frame = _normalize_bars(bars)
    entry = float(frame["open"].iloc[0])
    exit_px = float(frame["close"].iloc[-1])
    return {
        "return_pct": round((exit_px / entry - 1.0) * 100, 2),
        "artifact_note": (
            "Window is peak-to-recovery, so this figure is ~0% by construction and "
            "flatters any dip-buying rule compared to it."
        ),
    }
