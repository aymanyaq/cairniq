"""Covariance estimation for portfolio risk.

The sample covariance matrix is a poor estimator when the number of assets is
not small relative to the number of observations: its extreme eigenvalues are
biased (largest too large, smallest too small), and any optimizer or risk
decomposition fed with it amplifies that estimation noise. Two better
estimators live here:

- **Ledoit-Wolf shrinkage** — the analytically optimal convex combination of
  the sample covariance and a scaled-identity target, per Ledoit & Wolf (2004),
  "Honey, I Shrunk the Sample Covariance Matrix". Shrinkage intensity is
  derived from the data, not tuned.
- **EWMA** — exponentially weighted, so recent observations dominate. The
  0.94 default is the RiskMetrics daily convention.

Implemented on numpy rather than scikit-learn: sklearn is not a declared
dependency of this project, and adding it for one estimator would be a heavy
import for a small amount of arithmetic. `tests/test_tools/test_covariance.py`
cross-checks this implementation against sklearn's when it is importable.

All estimators return covariance at the input's native frequency (daily in,
daily out); callers annualize.
"""
from typing import Any

import numpy as np
import pandas as pd

from tools.exception_logger import log_exceptions

# RiskMetrics daily decay factor.
_DEFAULT_EWMA_LAMBDA = 0.94

_METHODS = ("ledoit_wolf", "ewma", "sample")


def _as_matrix(returns: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    clean = returns.dropna(how="any")
    return clean.to_numpy(dtype=float), list(clean.columns)


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrunk covariance and the shrinkage intensity actually used.

    Returns (covariance, shrinkage) where shrinkage is in [0, 1]: 0 means the
    sample covariance was kept as-is, 1 means it was fully replaced by the
    scaled-identity target. Follows the paper in using the MLE (1/T) sample
    covariance, which is what the shrinkage constant is derived against.
    """
    X, columns = _as_matrix(returns)
    T, N = X.shape
    if T < 2 or N < 1:
        raise ValueError(f"Need at least 2 observations and 1 asset, got T={T}, N={N}")

    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T                     # MLE sample covariance
    m = np.trace(S) / N                     # average variance
    F = m * np.eye(N)                       # scaled-identity target

    # Normalized Frobenius norm: ||A||^2 = trace(A A') / N = sum(A^2) / N for
    # the symmetric matrices involved here.
    d2 = float(np.sum((S - F) ** 2) / N)
    if d2 <= 0:
        # S already equals the target (e.g. a single asset) — nothing to shrink.
        return pd.DataFrame(S, index=columns, columns=columns), 0.0

    # b_bar2 = (1/T^2) * sum_t ||x_t x_t' - S||^2, expanded so no T outer
    # products are ever materialized:
    #   ||x_t x_t' - S||^2_F = (x_t'x_t)^2 - 2 x_t' S x_t + ||S||^2_F
    sq_norms = np.sum(Xc**2, axis=1)
    term1 = float(np.sum(sq_norms**2))
    term2 = float(np.sum(np.einsum("ij,jk,ik->i", Xc, S, Xc)))
    term3 = T * float(np.sum(S**2))
    b_bar2 = (term1 - 2 * term2 + term3) / (T**2 * N)

    b2 = min(b_bar2, d2)                    # b2 <= d2 keeps shrinkage in [0, 1]
    shrinkage = float(np.clip(b2 / d2, 0.0, 1.0))
    sigma = shrinkage * F + (1.0 - shrinkage) * S
    return pd.DataFrame(sigma, index=columns, columns=columns), shrinkage


def ewma_covariance(returns: pd.DataFrame, decay: float = _DEFAULT_EWMA_LAMBDA) -> pd.DataFrame:
    """Exponentially weighted covariance; `decay` is the RiskMetrics lambda."""
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay must be in (0, 1), got {decay}")

    X, columns = _as_matrix(returns)
    T, N = X.shape
    if T < 2 or N < 1:
        raise ValueError(f"Need at least 2 observations and 1 asset, got T={T}, N={N}")

    # Most recent row carries the largest weight.
    weights = decay ** np.arange(T - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    Xc = X - np.average(X, axis=0, weights=weights)
    sigma = (Xc * weights[:, None]).T @ Xc
    return pd.DataFrame(sigma, index=columns, columns=columns)


@log_exceptions()
def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    decay: float = _DEFAULT_EWMA_LAMBDA,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate an asset covariance matrix.

    Args:
        returns: DataFrame of periodic returns, one column per asset.
        method: "ledoit_wolf" (default), "ewma", or "sample".
        decay: EWMA decay factor; ignored by the other methods.

    Returns:
        (covariance, meta) where meta describes what was actually computed —
        `method`, `observations`, `assets`, and for Ledoit-Wolf the
        `shrinkage` intensity. On any estimator failure meta carries a
        `fallback_reason` and the sample covariance is returned, so a caller
        always gets a usable matrix rather than an exception.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    X, columns = _as_matrix(returns)
    T, N = X.shape
    meta: dict[str, Any] = {"method": method, "observations": int(T), "assets": int(N)}

    if T < 2 or N < 1:
        raise ValueError(f"Need at least 2 observations and 1 asset, got T={T}, N={N}")

    def _sample() -> pd.DataFrame:
        clean = returns.dropna(how="any")
        if isinstance(clean, pd.Series):
            clean = clean.to_frame()
        return clean.cov()

    if method == "sample":
        return _sample(), meta

    try:
        if method == "ledoit_wolf":
            cov, shrinkage = ledoit_wolf_shrinkage(returns)
            meta["shrinkage"] = round(shrinkage, 4)
            return cov, meta
        cov = ewma_covariance(returns, decay=decay)
        meta["decay"] = decay
        return cov, meta
    except Exception as e:
        # A degenerate estimator must not take down the risk tool that called it.
        meta["method"] = "sample"
        meta["fallback_reason"] = str(e)
        return _sample(), meta
