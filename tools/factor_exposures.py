"""Regression-based factor exposures — Advisor Roadmap 4.2.

What this replaces: `tools/portfolio_analytics.py::analyze_factors`, which
buckets each holding as Growth/Value/Momentum/Quality from `trailingPE`,
`revenueGrowth` and `roe` thresholds (`pe > 30 and rev_growth > 0.15` → Growth)
and then COUNTS the buckets. "Your portfolio is 40% Growth" from a label tally is
not a factor loading — it is a census of adjectives, reported with the confidence
a measured exposure would carry.

A factor exposure is a regression coefficient: how much this portfolio moves when
that factor moves, estimated from returns, with a standard error attached. The
standard error is the point. An exposure of 0.31 that cannot be distinguished
from zero and an exposure of 0.31 that can are different findings, and only one
of them should change a position. Every beta here ships with its t-statistic and
is explicitly labelled when it is not significant.

**Currency convention, stated because it changes the interpretation.** Portfolio
returns are measured in the profile's base currency (via 4.1's `_get_returns`),
while the factor proxies stay in their native USD, and the USD/base move enters
as its OWN regressor. Converting the factors too would smear the same currency
move across every column and make the FX beta collinear with all of them — a
CAD investor would then see "market beta" that silently contained their currency
risk. This way `fx` answers "how much of my variance is the dollar", and the
other betas answer "what am I exposed to in the market itself".

Dependencies: numpy and scipy only. `statsmodels` and `sklearn` are installed in
this environment but are NOT in requirements.txt, so a fresh install would not
have them — the same trap `tools/covariance.py` documents for Ledoit-Wolf.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

# |t| at which a coefficient is treated as distinguishable from zero. 2.0 is the
# usual ~5% two-sided rule at these sample sizes and is stated rather than hidden.
T_SIGNIFICANT = 2.0

# Minimum observations before any of this is worth reporting. Below roughly a
# quarter of daily data the standard errors are wide enough that every beta
# fails significance anyway, and a table of "not significant" rows invites the
# reader to squint at the point estimates — which is the habit this module is
# trying to break.
MIN_OBSERVATIONS = 60

# Factors as long/short SPREADS rather than raw ETF levels. Raw sector or style
# ETFs are ~0.9 correlated with the market, so regressing on their levels gives
# unstable, sign-flipping betas that look precise. A spread isolates the tilt.
FACTOR_SPECS: dict[str, dict[str, Any]] = {
    "market":   {"long": "SPY",  "short": None,  "label": "Market (SPY)"},
    "size":     {"long": "IWM",  "short": "SPY", "label": "Size (small minus large)"},
    "value":    {"long": "IVE",  "short": "IVW", "label": "Value (value minus growth)"},
    "momentum": {"long": "MTUM", "short": "SPY", "label": "Momentum"},
    "quality":  {"long": "QUAL", "short": "SPY", "label": "Quality"},
    "rates":    {"long": "IEF",  "short": None,  "label": "Rates / duration (7-10y Treasuries)"},
}

_PROXY_TICKERS = sorted({
    t for spec in FACTOR_SPECS.values() for t in (spec["long"], spec["short"]) if t
})


def _fetch_factor_returns(period: str) -> pd.DataFrame:
    """Factor proxy returns in NATIVE USD — see the module docstring on currency."""
    from tools.portfolio_analytics import _get_returns

    raw, valid = _get_returns(_PROXY_TICKERS, period=period, base_currency="USD")
    if raw is None or raw.empty:
        return pd.DataFrame()

    built = {}
    for name, spec in FACTOR_SPECS.items():
        long_leg, short_leg = spec["long"], spec["short"]
        if long_leg not in valid:
            continue
        if short_leg is None:
            built[name] = raw[long_leg]
        elif short_leg in valid:
            built[name] = raw[long_leg] - raw[short_leg]
    return pd.DataFrame(built)


def _fetch_fx_factor(base_currency: str, period: str) -> pd.Series | None:
    """USD/base daily return — the currency leg, as its own regressor."""
    base = (base_currency or "USD").upper().strip()
    if base == "USD":
        return None
    try:
        from tools.fx_utils import get_fx_rate_series
        fx = get_fx_rate_series(["USD"], base, period=period)
        if fx is None or fx.empty or "USD" not in fx.columns:
            return None
        return fx["USD"].pct_change(fill_method=None).dropna()
    except Exception as e:  # noqa: BLE001 — a missing FX leg must not kill the regression
        logger.debug(f"FX factor unavailable: {e}")
        return None


def _ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """OLS with an intercept. Returns coefficients, t-stats, R-squared.

    numpy/scipy only, deliberately — see the module docstring.
    """
    n, k = X.shape
    design = np.column_stack([np.ones(n), X])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeffs
    residuals = y - fitted
    dof = n - k - 1
    if dof <= 0:
        return {"error": "Not enough observations for the number of factors"}

    sigma2 = float(residuals @ residuals) / dof
    try:
        xtx_inv = np.linalg.pinv(design.T @ design)
    except np.linalg.LinAlgError:
        return {"error": "Factor matrix is singular"}
    std_errors = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(std_errors > 0, coeffs / std_errors, np.nan)

    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "intercept": float(coeffs[0]),
        "intercept_t": float(t_stats[0]) if np.isfinite(t_stats[0]) else None,
        "betas": coeffs[1:],
        "t_stats": t_stats[1:],
        "r_squared": r_squared,
        "dof": dof,
    }


def _p_value(t_stat: float, dof: int) -> float | None:
    """Two-sided p-value. scipy is a declared dependency; a fallback is not needed."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t_stat), dof))
    except Exception:  # noqa: BLE001
        return None


@log_exceptions()
def estimate_factor_exposures(
    symbols: list[str],
    weights: list[float] | None = None,
    period: str = "1y",
    returns_fn: Any = None,
    factors_fn: Any = None,
    fx_fn: Any = None,
) -> dict[str, Any]:
    """Regress portfolio returns on factor returns. Every source is injectable.

    Reports, per factor: beta, t-statistic, p-value, whether it is
    distinguishable from zero, and its share of explained variance. Plus alpha,
    R², the observation count, and any condition that makes the estimates
    unreliable — stated rather than left for the reader to infer.
    """
    symbols = [str(s).upper().strip() for s in (symbols or []) if str(s).strip()]
    if not symbols:
        return {"error": "No symbols supplied"}

    if returns_fn is None:
        from tools.portfolio_analytics import _get_returns

        def returns_fn(syms, per):  # noqa: E306
            return _get_returns(syms, period=per)

    port_returns, valid = returns_fn(symbols, period)
    if port_returns is None or getattr(port_returns, "empty", True) or not valid:
        return {"error": "Could not fetch price history for these holdings"}

    base_currency = (port_returns.attrs.get("fx", {}) or {}).get("base_currency") or "USD"

    # Weight the holdings into one portfolio series.
    if weights and len(weights) == len(symbols):
        wmap: dict[str, float] = {}
        for sym, w in zip(symbols, weights):
            wmap[sym] = wmap.get(sym, 0.0) + float(w)
    else:
        wmap = {s: 1.0 for s in symbols}
    w = np.array([wmap.get(s, 0.0) for s in valid], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(valid))
    w = w / w.sum()
    portfolio = port_returns[valid].dot(w)

    factors = (factors_fn or _fetch_factor_returns)(period)
    if factors is None or factors.empty:
        return {"error": "Could not fetch factor proxy history"}
    factors = factors.copy()

    fx_series = (fx_fn or _fetch_fx_factor)(base_currency, period)
    if fx_series is not None and not fx_series.empty:
        factors["fx"] = fx_series

    aligned = pd.concat([portfolio.rename("_portfolio"), factors], axis=1).dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        return {
            "error": (
                f"Only {len(aligned)} overlapping observations — below the {MIN_OBSERVATIONS} "
                f"needed for exposures worth reporting. With standard errors this wide every "
                f"beta would read 'not significant', which invites squinting at point "
                f"estimates that are not there."
            ),
            "observations": len(aligned),
        }

    factor_names = [c for c in aligned.columns if c != "_portfolio"]
    y = aligned["_portfolio"].to_numpy(dtype=float)
    X = aligned[factor_names].to_numpy(dtype=float)

    fit = _ols(y, X)
    if "error" in fit:
        return fit

    # Variance share: beta_i * cov(f_i, r) / var(r). Sums to R² across factors,
    # which is what makes it readable as "share of the movement explained" even
    # when the factors are correlated with one another.
    var_y = float(np.var(y, ddof=1))
    exposures = []
    for i, name in enumerate(factor_names):
        beta = float(fit["betas"][i])
        t_stat = float(fit["t_stats"][i]) if np.isfinite(fit["t_stats"][i]) else None
        cov_i = float(np.cov(X[:, i], y, ddof=1)[0, 1])
        share = (beta * cov_i / var_y) if var_y > 0 else 0.0
        significant = t_stat is not None and abs(t_stat) >= T_SIGNIFICANT
        exposures.append({
            "factor": name,
            "label": FACTOR_SPECS.get(name, {}).get("label", "Currency (USD vs base)"),
            "beta": round(beta, 3),
            "t_stat": round(t_stat, 2) if t_stat is not None else None,
            "p_value": round(_p_value(t_stat, fit["dof"]), 4) if t_stat is not None else None,
            "significant": significant,
            "variance_share_pct": round(share * 100, 1),
            # The sentence that `analyze_factors` could never say.
            "reading": (
                f"A 1% move in {name} moves this portfolio {beta:+.2f}%."
                if significant else
                f"Not distinguishable from zero at this sample size (t={t_stat:.2f}) — "
                f"the point estimate of {beta:+.2f} should not be acted on."
            ),
        })
    exposures.sort(key=lambda e: (not e["significant"], -abs(e["variance_share_pct"])))

    warnings = []
    condition = _condition_number(X)
    if condition > 30:
        warnings.append(
            f"Factor proxies are highly collinear (condition number {condition:.0f}); "
            f"individual betas are unstable even though the fit as a whole is not."
        )
    if not any(e["significant"] for e in exposures):
        warnings.append(
            "No factor exposure is distinguishable from zero over this window. That is a "
            "finding, not a gap — do not read the point estimates as tilts."
        )
    missing = sorted(set(symbols) - set(valid))
    if missing:
        warnings.append(f"No price history for: {', '.join(missing)} (excluded, weights renormalised).")

    alpha_annual = fit["intercept"] * 252
    return {
        "period": period,
        "base_currency": base_currency,
        "observations": len(aligned),
        "symbols_analyzed": valid,
        "r_squared": round(fit["r_squared"], 3),
        "exposures": exposures,
        "alpha_annualized_pct": round(alpha_annual * 100, 2),
        "alpha_significant": (
            fit["intercept_t"] is not None and abs(fit["intercept_t"]) >= T_SIGNIFICANT
        ),
        "alpha_reading": (
            "Alpha is not distinguishable from zero — this portfolio's returns are explained "
            "by its factor exposures, not by selection."
            if not (fit["intercept_t"] is not None and abs(fit["intercept_t"]) >= T_SIGNIFICANT)
            else f"Alpha of {alpha_annual * 100:+.1f}%/yr is statistically distinguishable from zero."
        ),
        "unexplained_variance_pct": round((1 - fit["r_squared"]) * 100, 1),
        "method": (
            "OLS of base-currency portfolio returns on long/short factor spreads in native USD, "
            "with the USD/base move as its own regressor. t >= 2 treated as significant."
        ),
        "basis": "measured",
        "warnings": warnings,
    }


def _condition_number(X: np.ndarray) -> float:
    """Condition number of the standardised design — a collinearity read."""
    try:
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        return float(np.linalg.cond(X / sd))
    except Exception:  # noqa: BLE001
        return 0.0
