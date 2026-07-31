"""Constrained optimizer + drift-band rebalancing — Advisor Roadmap 4.4.

Two instruments, one estimation spine:

  - ``optimize_portfolio`` — scipy SLSQP max-Sharpe / min-vol / target-vol under
    the profile's OWN IPS constraints (``load_ips_constraints``: single-name and
    fund position caps, the fund-decomposed sector cap, the restricted list).
    Covariance is 4.1's Ledoit-Wolf ``estimate_covariance``; expected returns
    are the historical means of the same window, labelled ``basis:
    "historical_mean"`` with the same caveat 4.5's parameter derivation
    carries: measured from current holdings over a past window — a description
    of this portfolio, not a forecast. There is no default risk-free rate; it
    is 0 unless the caller passes one, and the value used is reported.

  - ``check_rebalance_drift`` — the forward-looking successor to
    ``tools/simulation.py:simulate_rebalancing``'s backward-looking verdict.
    Answers "should I rebalance?" with the user's OWN trigger: the
    ``rebalance_drift_pct`` stored in the 3.7 drawdown playbook. If that band
    is unset the check is ``available: False`` with the reason — nothing here
    invents a band, because a band the user did not choose still gets quoted
    back as theirs. (Roadmap 2.8, the profile-readiness surface, exists to make
    exactly this kind of inert feature visible; the two were specified
    together.) The target allocation is likewise never invented: pass one
    explicitly, or pass an objective and the optimizer's output becomes the
    target, labelled as such.

    When the band is breached, the trade list to return to target is costed:
    turnover in base currency, and the realized-gain exposure of the sells —
    TAXABLE accounts only, per the tax_loss.py shelter rule (TFSA / RRSP / IRA
    / DCPP / PENSION realize no gains). The tax BILL is withheld, not
    estimated: no marginal rate is stated anywhere in the profile, and a bill
    computed from an invented rate would be precisely the kind of number this
    codebase withholds (jurisdiction parameters are 4.7's data, not an
    assumption to bake in).

Never raises from the public functions: any failure degrades to
``available: False`` with a reason. This is computed analysis, not
personalized investment advice.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_OBJECTIVES = ("min_vol", "max_sharpe", "target_vol")
# Below this many daily observations the covariance/means are noise dressed as
# signal — same refusal posture as 4.5's parameter derivation (which uses 400
# for goal projection; 60 is the floor for an exploratory optimization, and the
# observation count is always reported so the caller can judge).
_MIN_OBSERVATIONS = 60
# Account-label substrings that make realized gains untaxed, mirrored from
# tools/tax_loss.py's governance rule (keep the two lists identical).
_SHELTERED_ACCOUNT_HINTS = ("TFSA", "RRSP", "IRA", "DCPP", "PENSION")
# Classification sources that mean "this holding is a diversified fund" — from
# tools/ips_precheck.py's position-cap branch (same constant, same purpose).
_FUND_SOURCES = {"Fund Decomposition DB", "FMP Decomposition", "Knowledge Graph", "Cache (Decomposed)"}
# Sector label that means "we could not classify this" — excluded from the
# sector constraint and reported, never silently capped as a real sector.
_UNKNOWN_SECTOR = "Unknown"


# ---------------------------------------------------------------------------
# Pure core — no I/O, no profile state. Fully deterministic given the inputs.
# ---------------------------------------------------------------------------

def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> dict[str, float]:
    """(annual return, annual vol) for weights, both in decimals."""
    w = np.asarray(weights, dtype=float)
    return {
        "expected_return": float(w @ mu),
        "volatility": float(np.sqrt(max(float(w @ cov @ w), 0.0))),
    }


def _clip_to_feasible_start(caps: np.ndarray) -> np.ndarray:
    """Equal-weight start projected onto the cap box, or None if infeasible."""
    n = len(caps)
    if caps.sum() < 1.0 - 1e-9:
        return None
    w = np.minimum(np.full(n, 1.0 / n), caps)
    # Distribute the shortfall onto names still under their cap.
    for _ in range(n + 1):
        deficit = 1.0 - w.sum()
        if abs(deficit) < 1e-12:
            break
        room = caps - w
        free = room > 1e-12
        if not free.any():
            break
        w[free] += deficit * room[free] / room[free].sum()
    return w


def optimize_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    symbols: list[str],
    objective: str,
    risk_free_rate: float = 0.0,
    target_vol: float | None = None,
    max_weights: dict[str, float] | None = None,
    sector_exposure: dict[str, dict[str, float]] | None = None,
    max_sector_weight: float | None = None,
) -> dict[str, Any]:
    """Solve a constrained long-only portfolio problem.

    Args:
        mu: annualized expected returns (decimals), len == len(symbols).
        cov: annualized covariance matrix aligned to `symbols`.
        objective: "min_vol" | "max_sharpe" | "target_vol".
        risk_free_rate: annual decimal; only max_sharpe reads it. Defaults to
            0 — no rate is assumed for the caller.
        target_vol: annual decimal vol cap; required by target_vol.
        max_weights: per-symbol upper bounds (decimals); absent means 1.
        sector_exposure: symbol -> {sector: fraction of that symbol's weight}.
            Symbols absent or mapped only to "Unknown" are NOT sector-
            constrained (reported under `sector_unclassified`).
        max_sector_weight: post-decomposition sector cap (decimal).

    Returns a dict with `available`; when True, `weights` (symbol -> decimal),
    `stats`, and `binding` (which constraints hold at the boundary). Never
    raises: solver failure is reported, not thrown.
    """
    if objective not in _OBJECTIVES:
        return {"available": False, "reason": f"objective must be one of {_OBJECTIVES}, got {objective!r}"}

    symbols = [str(s).upper().strip() for s in symbols]
    n = len(symbols)
    if n == 0:
        return {"available": False, "reason": "no assets to optimize"}
    if n == 1:
        only = symbols[0]
        stats = portfolio_stats(np.array([1.0]), mu, cov)
        return {
            "available": True,
            "objective": objective,
            "weights": {only: 1.0},
            "stats": stats,
            "binding": {"position_caps": [], "sectors": [], "target_vol": False},
            "note": "single-asset universe — the optimum is 100% by construction",
        }

    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    if mu.shape != (n,) or cov.shape != (n, n):
        return {"available": False, "reason": f"shape mismatch: mu {mu.shape}, cov {cov.shape}, {n} symbols"}
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        return {"available": False, "reason": "non-finite expected returns or covariance"}
    cov = (cov + cov.T) / 2.0  # numerical symmetry

    caps = np.array(
        [float((max_weights or {}).get(s, 1.0)) for s in symbols], dtype=float
    )
    caps = np.clip(caps, 0.0, 1.0)
    if caps.sum() < 1.0 - 1e-9:
        return {
            "available": False,
            "reason": (
                f"position caps are jointly infeasible: they sum to {caps.sum() * 100:.1f}% "
                "but a fully-invested book needs 100%"
            ),
        }

    x0 = _clip_to_feasible_start(caps)
    if x0 is None:
        return {"available": False, "reason": "no feasible fully-invested allocation under these caps"}

    # Sector constraint rows: for each real sector, sum_i w_i * s_ij <= cap.
    sector_rows: list[tuple[str, np.ndarray]] = []
    sector_unclassified: list[str] = []
    if max_sector_weight is not None:
        exposure = sector_exposure or {}
        sectors: set[str] = set()
        for s in symbols:
            vec = exposure.get(s) or {}
            sectors.update(k for k in vec if k != _UNKNOWN_SECTOR)
        for sector in sorted(sectors):
            row = np.array(
                [float((exposure.get(s) or {}).get(sector, 0.0)) for s in symbols], dtype=float
            )
            if row.sum() > 0:
                sector_rows.append((sector, row))
        for s in symbols:
            vec = exposure.get(s) or {}
            if not any(k != _UNKNOWN_SECTOR for k in vec):
                sector_unclassified.append(s)

    if sector_rows:
        # Parallel to the position-cap check above: a sector cap that jointly
        # cannot reach a fully-invested 100% gets a stated reason, not a bare
        # "solver did not converge". Only claimed when nothing is left
        # unconstrained to absorb the remainder — a name whose exposure is
        # wholly or partly Unknown is not sector-capped, so it can.
        classified = np.zeros(n)
        for _sector, row in sector_rows:
            classified = classified + row
        headroom = len(sector_rows) * max_sector_weight
        if bool(np.all(classified >= 1.0 - 1e-9)) and headroom < 1.0 - 1e-9:
            return {
                "available": False,
                "reason": (
                    f"the {max_sector_weight * 100:.1f}% sector cap is jointly infeasible: "
                    f"{len(sector_rows)} sector(s) cover the whole book, capping total exposure "
                    f"at {headroom * 100:.1f}% where a fully-invested book needs 100%"
                ),
            }

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    for sector, row in sector_rows:
        constraints.append({
            "type": "ineq",
            "fun": lambda w, r=row: max_sector_weight - float(r @ w),
        })

    bounds = [(0.0, float(c)) for c in caps]

    def _solve(fun) -> dict[str, Any] | None:
        from scipy.optimize import minimize

        result = minimize(
            fun, x0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            return None
        w = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
        w[w < 1e-6] = 0.0
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        # Re-clipping to caps after the renormalization guard: renorm can only
        # shrink weights, so caps stay respected.
        return {"weights": w, "fun": float(result.fun)}

    def _vol(w: np.ndarray) -> float:
        return float(np.sqrt(max(float(w @ cov @ w), 0.0)))

    def _sharpe(w: np.ndarray) -> float:
        v = _vol(w)
        return (float(w @ mu) - risk_free_rate) / v if v > 1e-12 else 0.0

    try:
        if objective == "min_vol":
            solved = _solve(lambda w: float(w @ cov @ w))
        elif objective == "max_sharpe":
            def neg_sharpe(w: np.ndarray) -> float:
                return -_sharpe(w)

            solved = _solve(neg_sharpe)
            # Sharpe maximization is not convex; a second start from the
            # min-vol corner catches the case where the equal-weight start
            # falls into a worse local optimum.
            alt = optimize_weights(
                mu, cov, symbols, "min_vol",
                max_weights=max_weights,
                sector_exposure=sector_exposure,
                max_sector_weight=max_sector_weight,
            )
            if alt.get("available"):
                # `.get(s, 0.0)`, NOT `[s]`. The returned `weights` dict is built
                # with `if wi > 0`, so every name the min-vol corner zeroed out is
                # ABSENT from it — and a corner solution that drops a name is the
                # normal outcome on a real diversified book, not an edge case.
                # Indexing it directly raised KeyError, which the broad handler
                # below turned into `solver error: '<TICKER>'`: max_sharpe reported
                # a solver failure whose whole message was the dropped symbol.
                # It never fired in tests, where every synthetic name keeps weight.
                mv_weights = alt.get("weights") or {}
                w_mv = np.array([float(mv_weights.get(s, 0.0)) for s in symbols], dtype=float)
                x0_saved = x0
                x0 = w_mv
                retry = _solve(neg_sharpe)
                x0 = x0_saved
                if retry is not None and (solved is None or _sharpe(retry["weights"]) > _sharpe(solved["weights"])):
                    solved = retry
        else:  # target_vol
            if target_vol is None or target_vol <= 0:
                return {"available": False, "reason": "target_vol objective needs a positive target_vol (annual decimal)"}
            mv = optimize_weights(
                mu, cov, symbols, "min_vol",
                max_weights=max_weights,
                sector_exposure=sector_exposure,
                max_sector_weight=max_sector_weight,
            )
            if not mv.get("available"):
                return {"available": False, "reason": f"min-vol feasibility failed: {mv.get('reason')}"}
            min_vol_value = float(mv["stats"]["volatility"])
            if min_vol_value > target_vol * (1 + 1e-6):
                return {
                    "available": False,
                    "reason": (
                        f"target volatility {target_vol * 100:.1f}% is below the achievable "
                        f"minimum of {min_vol_value * 100:.1f}% under these constraints"
                    ),
                }
            constraints.append({
                "type": "ineq",
                "fun": lambda w: target_vol**2 - float(w @ cov @ w),
            })
            solved = _solve(lambda w: -float(w @ mu))
    except Exception as e:  # solver setup failure is a result, not an exception
        logger.warning("optimize_weights solver error: %s", e)
        return {"available": False, "reason": f"solver error: {e}"}

    if solved is None:
        return {
            "available": False,
            "reason": "solver did not converge to a feasible allocation under these constraints",
        }

    w = solved["weights"]
    tol = 1e-4
    binding = {
        "position_caps": sorted(
            s for s, wi, cap in zip(symbols, w, caps) if cap < 1.0 - 1e-9 and wi >= cap - tol
        ),
        "sectors": sorted(
            sector for sector, row in sector_rows
            if float(row @ w) >= max_sector_weight - tol
        ),
        "target_vol": bool(
            objective == "target_vol" and target_vol is not None and _vol(w) >= target_vol - tol
        ),
    }

    out = {
        "available": True,
        "objective": objective,
        "weights": {s: float(wi) for s, wi in zip(symbols, w) if wi > 0},
        "stats": portfolio_stats(w, mu, cov),
        "sharpe": _sharpe(w),
        "risk_free_rate": risk_free_rate,
        "binding": binding,
    }
    if sector_unclassified:
        out["sector_unclassified"] = sector_unclassified
    return out


# ---------------------------------------------------------------------------
# Seams — kept thin so tests mock network and profile state here.
# ---------------------------------------------------------------------------

def _decision_context() -> dict[str, Any]:
    from tools.portfolio_csv import get_portfolio_decision_context
    return get_portfolio_decision_context()


def _optimizer_returns(symbols: list[str], period: str) -> tuple[pd.DataFrame, list[str]]:
    """Base-currency daily returns (the 4.1 estimation layer's fetch)."""
    from tools.portfolio_analytics import _get_returns
    return _get_returns(symbols, period=period)


def _ips_constraints() -> dict[str, Any]:
    from tools.ips_precheck import load_ips_constraints
    return load_ips_constraints()


def _execution_readiness(constraints: dict[str, Any]) -> dict[str, Any]:
    """Same seam the 2.2 gate reports through, so one profile state cannot
    produce two different answers about whether a proposal can be acted on."""
    from tools.ips_precheck import execution_readiness
    return execution_readiness(constraints)


def _where_limits_are_set() -> str:
    from tools.ips_precheck import WHERE_LIMITS_ARE_SET
    return WHERE_LIMITS_ARE_SET


def _symbol_sector_profile(symbol: str) -> tuple[bool, dict[str, float] | None]:
    """(is_fund, sector vector or None) via the fund-decomposition stack —
    the same call the IPS pre-check makes, so a name the gate treats as a fund
    gets the fund cap here too."""
    from tools.sector_analysis import check_portfolio_allocation
    try:
        result = check_portfolio_allocation([symbol], [1.0], allow_network=True)
    except Exception:
        return False, None
    if not isinstance(result, dict):
        return False, None
    raw = result.get("sector_allocation_raw")
    sectors = None
    if isinstance(raw, dict) and raw:
        sectors = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float)) and v > 0}
        if not sectors:
            sectors = None
    details = result.get("holding_details") or []
    source = str(details[0].get("classification_source", "")) if details else ""
    sector_details = str(details[0].get("sector_details", "")) if details else ""
    is_fund = source in _FUND_SOURCES or "Fund" in sector_details
    return is_fund, sectors


def _playbook() -> dict[str, Any] | None:
    from tools.drawdown_playbook import get_playbook
    return get_playbook()


def _stored_target_allocation() -> dict[str, float] | None:
    """The user's stated target sleeve mix, or None. A seam so tests mock here."""
    from tools.memory import get_target_allocation
    return get_target_allocation()


# ---------------------------------------------------------------------------
# Orchestration — composes the seams with the pure core. Never raises.
# ---------------------------------------------------------------------------

def _aggregate_tradeable(holdings: list[dict[str, Any]]) -> dict[str, float]:
    """symbol -> base-currency value, cash/pension excluded, duplicates summed."""
    agg: dict[str, float] = {}
    for h in holdings or []:
        if not isinstance(h, dict) or h.get("is_cash_or_pension"):
            continue
        sym = str(h.get("symbol") or "").upper().strip()
        val = h.get("value_base")
        if sym and isinstance(val, (int, float)) and val > 0:
            agg[sym] = agg.get(sym, 0.0) + float(val)
    return agg


def _estimation_block(
    symbols: list[str], period: str
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]] | dict[str, Any]:
    """Annualized (mu, cov) for the universe, or an unavailable-result dict."""
    returns, valid = _optimizer_returns(symbols, period)
    if returns is None or getattr(returns, "empty", True) or not valid:
        return {"available": False, "reason": f"no price history for the universe over {period}"}
    if len(returns) < _MIN_OBSERVATIONS:
        return {
            "available": False,
            "reason": (
                f"only {len(returns)} daily observations over {period} — "
                f"below the { _MIN_OBSERVATIONS }-day floor for a usable estimate"
            ),
            "observations": int(len(returns)),
        }

    from tools.covariance import estimate_covariance

    frame = returns[valid]
    mu = (frame.mean() * _TRADING_DAYS).to_numpy(dtype=float)
    cov_daily, cov_meta = estimate_covariance(frame)
    cov = (cov_daily.to_numpy(dtype=float) * _TRADING_DAYS)

    meta = {
        "basis": "historical_mean",
        "period": period,
        "observations": int(len(frame)),
        "covariance": cov_meta,
        "caveat": (
            "Expected returns are the historical means of this window, measured from the "
            "CURRENT holdings — a description of this portfolio's past, not a forecast. "
            "Mean-variance optimization amplifies estimation error in exactly these means; "
            "treat the weights as the constrained geometry of the window, not as a verdict "
            "about the future."
        ),
    }
    fx = getattr(returns, "attrs", {}).get("fx") or {}
    if fx.get("unavailable"):
        meta["fx_unavailable"] = list(fx["unavailable"])
    return mu, cov, list(valid), meta


@log_exceptions()
def optimize_portfolio(
    objective: str = "min_vol",
    target_vol_pct: float | None = None,
    period: str = "1y",
    risk_free_rate_pct: float = 0.0,
) -> dict[str, Any]:
    """Optimize the live tradeable book under the profile's own IPS caps.

    objective: "min_vol" | "max_sharpe" | "target_vol" (the last needs
    target_vol_pct). Cash and pension sleeves are excluded (same rule as the
    4.9 candidate preview). A HELD restricted-list name is excluded from the
    universe and reported — restricted means no NEW buys, not a forced sale,
    so its weight is held constant and the remainder is what gets optimized.
    """
    try:
        if objective not in _OBJECTIVES:
            return {"available": False, "reason": f"objective must be one of {_OBJECTIVES}, got {objective!r}"}
        if objective == "target_vol" and (not target_vol_pct or target_vol_pct <= 0):
            return {"available": False, "reason": "target_vol objective needs a positive target_vol_pct"}

        ctx = _decision_context()
        if not isinstance(ctx, dict) or ctx.get("error"):
            return {"available": False, "reason": (ctx or {}).get("error", "portfolio context unavailable")}

        agg = _aggregate_tradeable(ctx.get("holdings") or [])
        if len(agg) < 2:
            return {
                "available": False,
                "reason": f"optimization needs at least two tradeable positions; found {len(agg)}",
            }

        constraints = _ips_constraints()
        readiness = _execution_readiness(constraints)
        restricted = {str(s).upper().strip() for s in (constraints.get("restricted_symbols") or [])}
        total_tradeable = sum(agg.values())
        restricted_held = {s: v for s, v in agg.items() if s in restricted}
        restricted_weight = sum(restricted_held.values()) / total_tradeable if total_tradeable > 0 else 0.0
        universe = {s: v for s, v in agg.items() if s not in restricted}
        if len(universe) < 2:
            return {
                "available": False,
                "reason": "fewer than two tradeable positions remain after the restricted list",
                "restricted_held_constant": sorted(restricted_held),
            }

        estimated = _estimation_block(list(universe), period)
        if isinstance(estimated, dict):
            return estimated
        mu, cov, valid, estimation = estimated

        dropped = sorted(set(universe) - set(valid))
        universe_values = {s: universe[s] for s in valid}
        current_w = {s: v / sum(universe_values.values()) for s, v in universe_values.items()}

        # Names the optimizer cannot move: restricted (no new buys ≠ forced
        # sale) and no-history (no data to trade on). The drift check carries
        # these at current weight when it reuses this output as its target.
        held_constant = {s: v / total_tradeable for s, v in restricted_held.items()}
        for s in dropped:
            held_constant[s] = agg[s] / total_tradeable

        # Per-symbol caps from the user's OWN stated limits; fund vs single-name
        # uses the same classification the IPS gate uses.
        max_weights: dict[str, float] = {}
        sector_exposure: dict[str, dict[str, float]] = {}
        sector_unclassified: list[str] = []
        single_cap = constraints.get("max_position_pct")
        fund_cap = constraints.get("max_fund_position_pct")
        for s in valid:
            is_fund, sectors = _symbol_sector_profile(s)
            cap_pct = fund_cap if is_fund else single_cap
            if cap_pct is not None:
                max_weights[s] = float(cap_pct) / 100.0
            if sectors:
                sector_exposure[s] = sectors
                if not any(k != _UNKNOWN_SECTOR for k in sectors):
                    sector_unclassified.append(s)
            else:
                sector_unclassified.append(s)

        sector_cap = constraints.get("max_sector_pct")
        solved = optimize_weights(
            mu, cov, valid, objective,
            risk_free_rate=float(risk_free_rate_pct or 0.0) / 100.0,
            target_vol=(float(target_vol_pct) / 100.0) if target_vol_pct else None,
            max_weights=max_weights or None,
            sector_exposure=sector_exposure or None,
            max_sector_weight=(float(sector_cap) / 100.0) if sector_cap is not None else None,
        )
        if not solved.get("available"):
            return solved

        current_stats = portfolio_stats(
            np.array([current_w[s] for s in valid]), mu, cov
        )
        current_sharpe = (
            (current_stats["expected_return"] - float(risk_free_rate_pct or 0.0) / 100.0)
            / current_stats["volatility"] if current_stats["volatility"] > 1e-12 else None
        )

        out = {
            "available": True,
            "objective": objective,
            "universe": valid,
            "current_weights_pct": {s: round(w * 100, 2) for s, w in current_w.items()},
            "optimized_weights_pct": {s: round(w * 100, 2) for s, w in solved["weights"].items()},
            "current_stats": {
                "expected_return_pct": round(current_stats["expected_return"] * 100, 2),
                "volatility_pct": round(current_stats["volatility"] * 100, 2),
                "sharpe": round(current_sharpe, 3) if current_sharpe is not None else None,
            },
            "optimized_stats": {
                "expected_return_pct": round(solved["stats"]["expected_return"] * 100, 2),
                "volatility_pct": round(solved["stats"]["volatility"] * 100, 2),
                "sharpe": round(solved["sharpe"], 3),
            },
            "constraints_applied": {
                "max_position_pct": single_cap,
                "max_fund_position_pct": fund_cap,
                "max_sector_pct": sector_cap,
                "sector_unclassified": sorted(set(sector_unclassified)),
                "restricted_held_constant": sorted(restricted_held),
            },
            "binding": solved["binding"],
            # Whether this allocation is something the user could act on, which
            # is NOT the same question as whether it solved. An unbounded solve
            # is a valid answer to "what is optimal with no limits" and a
            # misleading one to "what should I hold" when nobody ever asked the
            # user for limits — the weights come back concentrated exactly where
            # the maths says, with no cap in sight and no note that the profile
            # is missing one. The solve still runs and still reports; this says
            # what it does and does not mean.
            "execution_ready": readiness["execution_ready"],
            "execution_readiness": readiness,
            "estimation": estimation,
            "risk_free_rate_pct": float(risk_free_rate_pct or 0.0),
            "as_of": ctx.get("as_of"),
            "is_stale": bool(ctx.get("is_stale")),
        }
        if held_constant:
            out["held_constant_pct"] = {s: round(w * 100, 2) for s, w in held_constant.items()}
        if restricted_weight > 0:
            out["restricted_note"] = (
                f"{sorted(restricted_held)} are on the restricted list and held constant "
                f"({restricted_weight * 100:.1f}% of tradeable value): restricted means no NEW "
                "buys, not a forced sale. Weights above describe the remaining book."
            )
        if readiness.get("note"):
            # Sits beside `restricted_note` rather than inside `constraints_applied`:
            # that block lists what WAS applied, and this is the opposite claim.
            # The pointer is appended here and not carried in `note`, because the
            # page that shows `note` IS the page it points at.
            out["execution_readiness_note"] = f"{readiness['note']} {_where_limits_are_set()}"
        if dropped:
            out["data_gaps"] = {"no_price_history": dropped}
        if solved.get("sector_unclassified"):
            out["constraints_applied"]["sector_unclassified"] = sorted(
                set(sector_unclassified) | set(solved["sector_unclassified"])
            )
        return out
    except Exception as e:
        logger.exception("optimize_portfolio failed")
        return {"available": False, "reason": f"unexpected failure: {e}"}


def _is_sheltered(account: str) -> bool:
    up = str(account or "").upper()
    return any(hint in up for hint in _SHELTERED_ACCOUNT_HINTS)


def _tax_exposure(
    sells: dict[str, float], ctx: dict[str, Any]
) -> dict[str, Any]:
    """Realized-gain exposure of the required sells, taxable accounts only.

    Returns the gain BASE the rebalance would realize — never a tax bill: no
    marginal rate is stated in any profile, and jurisdiction parameters are
    4.7's data, not something to invent here. Gain fractions are measured in
    each row's listing currency (purchase vs current price) and applied to
    base-currency sale proceeds; FX drift between purchase and sale is not
    separated, which the output says.
    """
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for h in ctx.get("holdings") or []:
        if not isinstance(h, dict) or h.get("is_cash_or_pension"):
            continue
        sym = str(h.get("symbol") or "").upper().strip()
        if sym in sells:
            rows_by_symbol.setdefault(sym, []).append(h)

    per_symbol: dict[str, Any] = {}
    total_gain = 0.0
    unpriced: list[str] = []
    for sym, sale_base in sells.items():
        gain_base = 0.0
        rows = rows_by_symbol.get(sym, [])
        taxable_rows = [r for r in rows if not _is_sheltered(r.get("account"))]
        row_total = sum(
            float(r.get("value_base") or 0.0) for r in taxable_rows
            if isinstance(r.get("value_base"), (int, float)) and r.get("value_base") > 0
        )
        priced_any = False
        for r in taxable_rows:
            value_base = r.get("value_base")
            if not isinstance(value_base, (int, float)) or value_base <= 0 or row_total <= 0:
                continue
            purchase = r.get("purchase_price")
            current = r.get("current_price")
            if not all(isinstance(p, (int, float)) and p > 0 for p in (purchase, current)):
                continue
            priced_any = True
            gain_fraction = max(0.0, (float(current) - float(purchase)) / float(current))
            gain_base += sale_base * (float(value_base) / row_total) * gain_fraction
        if taxable_rows and not priced_any:
            unpriced.append(sym)
        if gain_base > 0:
            per_symbol[sym] = round(gain_base, 2)
            total_gain += gain_base

    out = {
        "taxable_realized_gain_base": round(total_gain, 2),
        "per_symbol": per_symbol,
        "sheltered_note": (
            "Rows in TFSA/RRSP/IRA/DCPP/pension accounts realize no taxable gain and are "
            "excluded (same shelter rule as the tax-loss tool)."
        ),
        "tax_bill": None,
        "tax_bill_withheld_reason": (
            "No marginal tax rate is stated in the profile, and none is invented: the gain "
            "base above is the computed fact; the bill needs jurisdiction parameters "
            "(Roadmap 4.7's data) applied by the user or their accountant."
        ),
        "gain_basis_note": (
            "Gain fractions are measured in each row's listing currency and applied to "
            "base-currency proceeds; FX drift between purchase and sale is not separated."
        ),
    }
    if unpriced:
        out["unpriced_rows"] = {
            "symbols": sorted(unpriced),
            "note": "taxable rows without usable purchase/current prices — their gain is unknown, not zero",
        }
    return out


@log_exceptions()
def check_rebalance_drift(
    target_allocation: dict[str, float] | None = None,
    objective: str | None = None,
    target_vol_pct: float | None = None,
    period: str = "1y",
) -> dict[str, Any]:
    """"Should I rebalance?" — measured against the user's OWN drift band.

    The band is 3.7's stored `rebalance_drift_pct`; unset means unavailable
    with the reason, never a default.

    The target DEFAULTS to the user's stored allocation (4.4's store) — call
    with no target and no objective to get that. `target_allocation` and
    `objective` are overrides for what-if questions: each REPLACES the stored
    plan for that call, and doing so is reported (`stored_target_overridden`,
    and in the verdict) rather than left in the machine-readable
    `target_basis` alone. A breach costs out the return-to-target trades:
    turnover plus the taxable realized-gain exposure of the sells (the bill
    itself is withheld — see `_tax_exposure`).
    """
    try:
        playbook = _playbook()
        band = None
        if isinstance(playbook, dict):
            raw_band = playbook.get("rebalance_drift_pct")
            if isinstance(raw_band, (int, float)) and raw_band > 0:
                band = float(raw_band)
        if band is None:
            return {
                "available": False,
                "reason": (
                    "no rebalance_drift_pct is stored in the drawdown playbook — the drift "
                    "band is the user's own trigger and nothing defaults it. Set it in the "
                    "playbook (the 3.7 surface) and this check goes live."
                ),
                "inert_dependency": (
                    "Roadmap 2.8 (profile readiness surface) exists to surface exactly this "
                    "kind of shipped-but-inert feature; 4.4 was specified to be built with "
                    "it or to name the dependency — this names it."
                ),
            }

        ctx = _decision_context()
        if not isinstance(ctx, dict) or ctx.get("error"):
            return {"available": False, "reason": (ctx or {}).get("error", "portfolio context unavailable")}
        agg = _aggregate_tradeable(ctx.get("holdings") or [])
        total_tradeable = sum(agg.values())
        if total_tradeable <= 0:
            return {"available": False, "reason": "no tradeable value in the book"}

        # Resolve the target: the user's stored plan by default, or an override
        # the caller supplied for this call. Never invented.
        target_basis = None
        target_source = None
        target: dict[str, float] | None = None
        if target_allocation:
            cleaned: dict[str, float] = {}
            for k, v in target_allocation.items():
                sym = str(k or "").upper().strip()
                try:
                    pct = float(v)
                except (TypeError, ValueError):
                    continue
                if sym and pct > 0:
                    cleaned[sym] = pct
            if not cleaned:
                return {"available": False, "reason": "target_allocation has no usable positive weights"}
            total = sum(cleaned.values())
            target = {s: v / total for s, v in cleaned.items()}
            target_basis = "explicit"
            # Named because this rescale is the difference that matters against
            # the stored plan: a caller passing a mix inline means those
            # proportions, but sleeves totalling 90% in the STORE mean 10% held
            # aside, which is why set_target_allocation refuses to normalize.
            target_source = (
                "a target allocation supplied with this call, rescaled to total 100%"
            )
        elif objective:
            optimized = optimize_portfolio(
                objective=objective, target_vol_pct=target_vol_pct, period=period
            )
            if not optimized.get("available"):
                return {
                    "available": False,
                    "reason": f"target from optimizer unavailable: {optimized.get('reason')}",
                }
            # The optimizer covers only its movable universe; restricted or
            # no-history names stay at current weight (they are not forced
            # sales), and the optimized weights scale over the remainder.
            held = {s: p / 100.0 for s, p in (optimized.get("held_constant_pct") or {}).items()}
            held_total = sum(held.values())
            target = dict(held)
            for s, p in optimized["optimized_weights_pct"].items():
                target[s] = target.get(s, 0.0) + (p / 100.0) * (1.0 - held_total)
            target_basis = f"optimizer:{objective}"
            target_source = (
                f"a fresh optimizer solve ({objective}) over the movable universe, "
                "not a plan the user has stated"
            )
        else:
            # The STORED target, and this branch is why the drift check was
            # unreachable in practice. Both routes above need a caller to supply
            # something per-call: an inline dict, or an objective that spends a
            # solve. Neither is a plan the user keeps. `target_allocation` had no
            # store, no endpoint and no entry screen anywhere in the codebase —
            # the same shape `risk_constraints` was in before 2.9, and misfiled
            # the same way: reported as "nothing to drift against" as though the
            # user had declined to say, when nobody could tell us.
            stored = _stored_target_allocation()
            if stored:
                total = sum(stored.values())
                # Stored weights are validated to ~100% at write time; this
                # divide is the percent-to-decimal conversion, not a rescue of a
                # mix that does not add up. set_target_allocation refuses those.
                target = {s: v / total for s, v in stored.items()}
                target_basis = "stored"
                target_source = (
                    "the target allocation the user stored at Context › Target Allocation"
                )
            else:
                return {
                    "available": False,
                    "reason": (
                        "no target allocation is stored, none was supplied, and no optimizer "
                        "objective was given — there is nothing to drift against, and a target "
                        "is never invented. Set one at Context › Target Allocation."
                    ),
                    "entry_screen": "/context › Target Allocation",
                }

        # An override is only half-reported by `target_basis`. When the user HAS
        # a stored plan and the caller supplied something else, every number
        # below is drift against the substitute — and a result that says only
        # "your book has drifted 12 points" reads as drift from the plan the
        # user typed, which was never consulted. Same discipline as `band_source`
        # and the tax bill's withheld reason: name the basis where it is read,
        # not only in a field beside it.
        overridden_stored = None
        if target_basis != "stored":
            try:
                overridden_stored = _stored_target_allocation()
            except Exception:
                # Reporting the substitution is a courtesy. Failing to read a
                # store this call needed nothing from must not take the check
                # down — that would make setting a target allocation able to
                # BREAK the explicit-target path.
                logger.warning("could not read stored target allocation for the override note",
                               exc_info=True)

        names = sorted(set(agg) | set(target))
        current_w = {s: agg.get(s, 0.0) / total_tradeable for s in names}
        target_w = {s: target.get(s, 0.0) for s in names}

        drifts = {
            s: round((current_w[s] - target_w[s]) * 100.0, 2) for s in names
        }
        max_abs = max((abs(d) for d in drifts.values()), default=0.0)
        breached = max_abs > band

        trades = []
        sells: dict[str, float] = {}
        buys_total = 0.0
        for s in names:
            delta = (target_w[s] - current_w[s]) * total_tradeable
            if abs(delta) < 0.01:
                continue
            trades.append({
                "symbol": s,
                "side": "BUY" if delta > 0 else "SELL",
                "amount_base": round(abs(delta), 2),
                "weight_change_pct": round((target_w[s] - current_w[s]) * 100.0, 2),
            })
            if delta < 0:
                sells[s] = abs(delta)
            else:
                buys_total += delta
        sells_total = sum(sells.values())

        out = {
            "available": True,
            "band_pct": band,
            "band_source": "drawdown_playbook.rebalance_drift_pct (user's own)",
            "breached": breached,
            "max_abs_drift_pct": round(max_abs, 2),
            "drift_pct_by_symbol": drifts,
            "current_weights_pct": {s: round(w * 100, 2) for s, w in current_w.items() if w > 0},
            "target_weights_pct": {s: round(w * 100, 2) for s, w in target_w.items() if w > 0},
            "target_basis": target_basis,
            "target_source": target_source,
            "trades_to_target": trades,
            "turnover": {
                "sells_base": round(sells_total, 2),
                "buys_base": round(buys_total, 2),
                "one_way_turnover_pct": round(sells_total / total_tradeable * 100.0, 2),
            },
            "total_tradeable_base": round(total_tradeable, 2),
            "base_currency": ctx.get("base_currency"),
            "as_of": ctx.get("as_of"),
            "is_stale": bool(ctx.get("is_stale")),
        }
        # `trades_to_target` is a sized action list, so it carries the same
        # readiness question the optimizer's weights do — and it carries it on
        # BOTH paths, including an explicitly supplied target: where the target
        # came from says nothing about whether the profile has stated the limits
        # the resulting trades would be checked against.
        drift_readiness = _execution_readiness(_ips_constraints())
        out["execution_ready"] = drift_readiness["execution_ready"]
        out["execution_readiness"] = drift_readiness
        if drift_readiness.get("note"):
            out["execution_readiness_note"] = (
                f"{drift_readiness['note']} {_where_limits_are_set()}"
            )
        if overridden_stored:
            out["stored_target_overridden"] = {
                "stored_weights_pct": {
                    s: round(float(v), 2) for s, v in sorted(overridden_stored.items())
                },
                "measured_against": target_source,
                "note": (
                    "The user HAS a stored target allocation and this check did NOT measure "
                    "against it — the drift, trades and turnover above are against "
                    f"{target_source}. Say which basis was used when reporting this, and re-run "
                    "with no target_allocation and no objective to measure against their own "
                    "stated plan."
                ),
            }
        if breached and sells:
            out["tax"] = _tax_exposure(sells, ctx)
        elif breached:
            out["tax"] = {"taxable_realized_gain_base": 0.0, "note": "no sells required to return to target"}
        out["verdict"] = (
            f"DRIFT BAND BREACHED: max drift {max_abs:.1f}pp exceeds the user's {band:g}% band — "
            "trades and taxable-gain exposure above."
            if breached else
            f"Within band: max drift {max_abs:.1f}pp is inside the user's {band:g}% band — no rebalance triggered."
        )
        # The verdict is the line that gets quoted back, so the substitution has
        # to survive into it — not sit in a sibling field the summary skips.
        if overridden_stored:
            out["verdict"] += (
                f" Measured against {target_source} — the user's OWN stored target allocation "
                "was overridden for this call."
            )
        return out
    except Exception as e:
        logger.exception("check_rebalance_drift failed")
        return {"available": False, "reason": f"unexpected failure: {e}"}
