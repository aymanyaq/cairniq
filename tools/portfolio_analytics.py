"""
Portfolio Analytics
Advanced risk metrics and portfolio analysis using historical price data.
"""
import os
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.opportunity_scanner import _get_sector_for_ticker
from tools.yf_utils import dividend_yield_fraction, get_info_safe

# Last-resort risk-free rate if FRED is unreachable and its own fallback
# data can't be parsed. Kept only as a final safety net, not a live source.
_FALLBACK_RISK_FREE_RATE = 0.05


def _get_risk_free_rate() -> float:
    """Live risk-free rate (Fed Funds Effective Rate) as a decimal, e.g. 0.0525.

    Used as the risk-free rate for Sharpe/Sortino so those ratios track actual
    policy rates instead of a hardcoded constant that goes stale as rates move.
    """
    try:
        from tools.fred_api import get_fed_funds_rate
        data = get_fed_funds_rate()
        raw = data.get("current_rate") if isinstance(data, dict) else None
        if isinstance(raw, str) and raw.endswith("%"):
            return float(raw[:-1]) / 100.0
    except Exception:
        pass
    return _FALLBACK_RISK_FREE_RATE


def _fx_period_covering(start: str) -> str:
    """Smallest yfinance period string reaching back past `start`.

    Coarse on purpose: over-fetching FX history is cheap, under-fetching means a
    historical window silently loses its base-currency conversion.
    """
    try:
        from datetime import date, datetime
        years = (date.today() - datetime.fromisoformat(str(start)[:10]).date()).days / 365.25
    except (TypeError, ValueError):
        return "max"
    for bound, label in ((1, "2y"), (4, "5y"), (9, "10y")):
        if years <= bound:
            return label
    return "max"


@log_exceptions()
def _get_returns(
    symbols: list[str],
    period: str = "1y",
    base_currency: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Helper to fetch price data and calculate returns.

    Prices are converted into the profile's base currency (or the explicit
    ``base_currency``) BEFORE returns are computed, so volatility/VaR/drawdown
    on mixed-currency portfolios include FX risk. A symbol whose FX series is
    unavailable stays in its native currency — numerically identical to a
    constant-rate conversion — and is listed under
    ``returns.attrs["fx"]["unavailable"]`` so callers can surface the gap.

    ``start``/``end`` (ISO dates) select an explicit WINDOW instead of a trailing
    ``period``, which is what Roadmap 4.3's episode replay needs: "the GFC" is a
    pair of dates, not a lookback. Given a window, the FX series is fetched for a
    period long enough to cover it and then reindexed onto the equity dates, so
    historical replays are still measured in the base currency rather than
    quietly reverting to native.

    Returns: (returns_df, valid_symbols)
    """
    try:
        # Use safe download with retry logic for concurrent environments
        from tools.yf_utils import download_safe
        if start:
            data = download_safe(symbols, period=None, start=start, end=end)
        else:
            data = download_safe(symbols, period=period)

        # Handle 'Adj Close' vs 'Close' vs multi-level columns.
        #
        # The two blocks are merged rather than one being chosen, because
        # yfinance returns a PARTIAL 'Adj Close' when some tickers in a batch
        # fail: fetching [ARKK, SMH, NVDA] back to 2007 yields an 'Adj Close'
        # block containing only the failed ARKK column (all NaN) alongside a
        # complete 'Close' block. Preferring 'Adj Close' wholesale then dropped
        # every all-NaN column and returned an EMPTY frame — so one unavailable
        # symbol silently took down the entire fetch, and the caller could not
        # tell "no history" from "one bad ticker". Found via 4.3's episode
        # replay, which trips it constantly (today's names, decade-old windows).
        adj = data["Adj Close"] if "Adj Close" in data else None
        close = data["Close"] if "Close" in data else None
        if adj is None and close is None:
            # Single symbol with no column grouping — it may be the frame itself.
            prices = data
        else:
            frames = [f for f in (adj, close) if f is not None]
            frames = [f.to_frame(name=symbols[0]) if isinstance(f, pd.Series) else f for f in frames]
            frames = [f.dropna(axis=1, how="all") for f in frames]
            prices = frames[0]
            for extra in frames[1:]:
                # Adjusted prices win where present; Close fills the gaps.
                missing = [c for c in extra.columns if c not in prices.columns]
                if missing:
                    prices = prices.join(extra[missing], how="outer")

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=symbols[0])

        # Drop columns (tickers) that have no data
        prices = prices.dropna(axis=1, how="all")

        if prices.empty:
            return pd.DataFrame(), []

        # Get valid symbols from columns
        valid_symbols = prices.columns.tolist()

        # --- BASE-CURRENCY CONVERSION ---
        fx_info: dict[str, Any] = {"base_currency": None, "converted": [], "unavailable": []}
        try:
            from tools.fx_utils import get_fx_rate_series, infer_symbol_currency
            if base_currency is None:
                from tools.memory import get_profile_base_currency
                base_currency = get_profile_base_currency()
            base = str(base_currency).upper().strip()
            fx_info["base_currency"] = base

            symbol_currency = {s: infer_symbol_currency(s) for s in valid_symbols}
            foreign = sorted({c for c in symbol_currency.values() if c != base})
            if foreign:
                prices = prices.copy()
                # For a historical window, ask for a period long enough to reach
                # back past `start` — the reindex+ffill below then aligns it onto
                # the equity dates. Without this an episode replay would find no
                # overlapping FX rows and silently fall back to native currency,
                # which is the one failure this conversion exists to prevent.
                fx_period = _fx_period_covering(start) if start else period
                fx = get_fx_rate_series(foreign, base, period=fx_period)
                if not fx.empty:
                    # FX and equity markets observe different holidays; align on
                    # the equity index and carry the last known rate across gaps.
                    fx = fx.reindex(prices.index).ffill().bfill()
                for sym in valid_symbols:
                    cur = symbol_currency[sym]
                    if cur == base:
                        continue
                    rate = fx[cur] if (not fx.empty and cur in fx.columns) else None
                    if rate is not None and rate.notna().any():
                        prices[sym] = prices[sym] * rate
                        fx_info["converted"].append(sym)
                    else:
                        fx_info["unavailable"].append(sym)
        except Exception as fx_err:
            safe_print(f"FX conversion skipped, using native-currency returns: {fx_err}")

        # Calculate returns
        returns = prices.pct_change(fill_method=None).dropna()
        returns.attrs["fx"] = fx_info
        return returns, valid_symbols
    except Exception as e:
        safe_print(f"Error fetching data: {e}")
        return pd.DataFrame(), []


@log_exceptions()
def calculate_portfolio_metrics(symbols: list[str], weights: list[float] | None = None, period: str = "1y") -> dict[str, Any]:
    """
    Calculate key risk/return metrics for a portfolio.

    Args:
        symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        weights: Optional weights (if None, assumes equal weighting)
        period: Historical period to fetch (e.g., '1y', '5y', '10y')

    Returns:
        Sharpe ratio, Sortino ratio, max drawdown, beta, volatility, returns
    """
    try:
        # --- STABILIZATION CACHE ---
        # For long-term historical baselines (>= 5y), use a 24-hour cache to prevent daily "jitter"
        # in the anchor metrics. 1y metrics remain live.
        from tools.daily_cache import get_cached, set_cached
        from tools.memory import get_profile_base_currency
        base_currency = get_profile_base_currency()
        # Base currency is part of the key: the same symbols measured in a
        # different base currency are different return series.
        cache_key = f"metrics_{period}_{base_currency}_{'_'.join(sorted(symbols))}"
        if period in ["5y", "10y", "max"]:
            cached = get_cached(cache_key, ttl_seconds=86400) # 24 hour TTL
            if cached:
                return cached

        returns, valid_symbols = _get_returns(symbols, period=period)

        if returns.empty or not valid_symbols:
            return {"error": "Could not fetch price data for the given symbols"}

        # Filter weights to match valid symbols
        if weights is not None:
            # Create a map of symbol -> weight
            # Assumes input symbols and weights were aligned
            if len(symbols) == len(weights):
                weight_map = {}
                for s, w in zip(symbols, weights):
                    weight_map[s] = weight_map.get(s, 0) + w
                # extracting weights for valid symbols only
                new_weights = [weight_map.get(s, 0) for s in valid_symbols]
            else:
                # If mismatch, fallback to equal
                new_weights = [1.0 / len(valid_symbols)] * len(valid_symbols)
        else:
            new_weights = [1.0 / len(valid_symbols)] * len(valid_symbols)

        # Array conversion and Renormalize weights to sum to 1.0
        w_array = np.array(new_weights)
        if w_array.sum() > 0:
            w_array = w_array / w_array.sum()

        # Portfolio returns
        # Ensure alignment
        portfolio_returns = returns[valid_symbols].dot(w_array)

        # Annualized metrics (252 trading days)
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe Ratio (live Fed Funds rate as the risk-free proxy)
        risk_free_rate = _get_risk_free_rate()
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # Sortino Ratio (only downside volatility)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Max Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Geometric CAGR — what the portfolio actually compounded at. The
        # arithmetic annual_return above overstates this by roughly half the
        # variance, so the two diverge exactly when volatility is high and the
        # gap matters most. Undefined if the portfolio's cumulative value ever
        # reaches zero.
        years = len(portfolio_returns) / 252
        total_growth = float(cumulative.iloc[-1])
        cagr = (total_growth ** (1 / years) - 1) if (years > 0 and total_growth > 0) else None

        # Beta (vs S&P 500)
        spy_returns, _ = _get_returns(["SPY"], period=period)
        if not spy_returns.empty:
            # Align dates
            common_idx = portfolio_returns.index.intersection(spy_returns.index)
            port_aligned = portfolio_returns.loc[common_idx]
            spy_aligned = spy_returns.loc[common_idx, "SPY"]

            if len(common_idx) > 1:
                covariance = np.cov(port_aligned, spy_aligned)[0, 1]
                market_variance = spy_aligned.var()
                beta = covariance / market_variance if market_variance > 0 else 1.0
            else:
                beta = None
        else:
            beta = None

        beta_available = beta is not None and not np.isnan(beta)
        fx_info = returns.attrs.get("fx", {})
        result = {
            "symbols": symbols,
            "weights": w_array.tolist(),
            "base_currency": fx_info.get("base_currency") or base_currency,
            "metrics": {
                "annual_return": f"{annual_return * 100:.1f}%",
                "cagr": f"{cagr * 100:.1f}%" if cagr is not None else "N/A",
                "annual_volatility": f"{annual_volatility * 100:.1f}%",
                # Cast to native float — numpy scalars repr as "np.float64(1.62)" and
                # leak that wrapper into str()-rendered tool output (e.g. raw dumps).
                "sharpe_ratio": round(float(sharpe_ratio), 2),
                "sortino_ratio": round(float(sortino_ratio), 2),
                "max_drawdown": f"{max_drawdown * 100:.1f}%",
                "beta": round(float(beta), 2) if beta_available else "N/A"
            },
            "metric_notes": {
                "annual_return": "Arithmetic: mean daily return x 252. Used for Sharpe/Sortino by convention.",
                "cagr": "Geometric: the rate the portfolio actually compounded at over this lookback.",
            },
            "interpretation": {
                "sharpe": (
                    "Excellent (>1.5)" if sharpe_ratio > 1.5 else
                    "Good (1.0-1.5)" if sharpe_ratio > 1.0 else
                    "Average (0.5-1.0)" if sharpe_ratio > 0.5 else
                    "Poor (<0.5)"
                ),
                "volatility": (
                    "Low risk (<15%)" if annual_volatility < 0.15 else
                    "Moderate risk (15-25%)" if annual_volatility < 0.25 else
                    "High risk (>25%)"
                ),
                "beta": (
                    "Defensive (<0.8) - Less volatile than market" if beta_available and beta < 0.8 else
                    "Market-like (0.8-1.2)" if beta_available and beta < 1.2 else
                    "Aggressive (>1.2) - More volatile than market" if beta_available else "N/A"
                )
            }
        }

        if fx_info.get("converted"):
            result["fx_note"] = (
                f"Returns measured in {result['base_currency']}: "
                f"{', '.join(fx_info['converted'])} converted from native currency, so FX risk is included."
            )
        if fx_info.get("unavailable"):
            result["data_warning"] = (
                f"FX series unavailable for {', '.join(fx_info['unavailable'])}; "
                "their returns remain in native currency (FX risk not reflected)."
            )

        # Cache long-term results
        if period in ["5y", "10y", "max"]:
            set_cached(cache_key, result)

        return result
    except Exception as e:
        return {"error": f"Portfolio analysis failed: {str(e)}"}


@log_exceptions()
def estimate_marginal_risk_contribution(
    symbols: list[str],
    weights: list[float],
    candidate_symbol: str,
    candidate_weight: float = 0.05,
    period: str = "1y",
    cov_method: str = "ledoit_wolf",
) -> dict[str, Any]:
    """
    Estimate how adding a candidate changes total portfolio volatility.

    Args:
        symbols: Current portfolio symbols.
        weights: Current portfolio weights as decimals or percentages.
        candidate_symbol: Proposed addition.
        candidate_weight: Proposed allocation as decimal (0.05) or percent (5).
        period: Historical period for covariance.
        cov_method: Covariance estimator — "ledoit_wolf" (default), "ewma", or
            "sample". Shrinkage is the default because the raw sample matrix
            makes the volatility delta below noisier than the effect it is
            trying to measure once the holding count approaches the ~252 daily
            observations in a 1y lookback.
    """
    try:
        if not symbols or not weights or len(symbols) != len(weights):
            return {"error": "Need aligned current symbols and weights"}

        candidate_symbol = candidate_symbol.upper().strip()
        if not candidate_symbol:
            return {"error": "candidate_symbol is required"}

        current_symbols = [str(s).upper().strip() for s in symbols if str(s).strip()]
        if candidate_symbol not in current_symbols:
            all_symbols = current_symbols + [candidate_symbol]
        else:
            all_symbols = current_symbols

        returns, valid_symbols = _get_returns(all_symbols, period=period)
        if returns.empty or candidate_symbol not in valid_symbols:
            return {"error": "Insufficient price history for marginal risk analysis"}

        weight_map = {
            str(sym).upper().strip(): float(weight)
            for sym, weight in zip(symbols, weights)
            if str(sym).upper().strip() in valid_symbols
        }
        if not weight_map:
            return {"error": "No current holdings had usable price history"}

        current_weights = np.array([weight_map.get(sym, 0.0) for sym in valid_symbols], dtype=float)
        if current_weights.max(initial=0) > 1.0:
            current_weights = current_weights / 100.0
        if current_weights.sum() <= 0:
            return {"error": "Current weights sum to zero"}
        current_weights = current_weights / current_weights.sum()

        candidate_weight = float(candidate_weight)
        if candidate_weight > 1.0:
            candidate_weight = candidate_weight / 100.0
        candidate_weight = max(0.0, min(candidate_weight, 0.95))

        proposed_weight_map = {sym: weight for sym, weight in zip(valid_symbols, current_weights)}
        for sym in list(proposed_weight_map):
            proposed_weight_map[sym] *= (1.0 - candidate_weight)
        proposed_weight_map[candidate_symbol] = proposed_weight_map.get(candidate_symbol, 0.0) + candidate_weight
        proposed_weights = np.array([proposed_weight_map.get(sym, 0.0) for sym in valid_symbols], dtype=float)
        proposed_weights = proposed_weights / proposed_weights.sum()

        from tools.covariance import estimate_covariance
        cov, cov_meta = estimate_covariance(returns[valid_symbols], method=cov_method)
        current_daily_vol = float(np.sqrt(current_weights.T @ cov.values @ current_weights))
        proposed_daily_vol = float(np.sqrt(proposed_weights.T @ cov.values @ proposed_weights))
        current_annual_vol = current_daily_vol * np.sqrt(252)
        proposed_annual_vol = proposed_daily_vol * np.sqrt(252)

        candidate_returns = returns[candidate_symbol]
        current_port_returns = returns[valid_symbols].dot(current_weights)
        candidate_corr = current_port_returns.corr(candidate_returns)

        if proposed_daily_vol > 0:
            sigma_w = cov.values @ proposed_weights
            idx = valid_symbols.index(candidate_symbol)
            component_daily_vol = proposed_weights[idx] * sigma_w[idx] / proposed_daily_vol
            component_annual_vol = component_daily_vol * np.sqrt(252)
            component_share = component_annual_vol / proposed_annual_vol if proposed_annual_vol > 0 else 0
        else:
            component_annual_vol = 0.0
            component_share = 0.0

        fx_info = returns.attrs.get("fx", {})
        result = {
            "period": period,
            "candidate_symbol": candidate_symbol,
            "candidate_weight": f"{candidate_weight * 100:.1f}%",
            "symbols_analyzed": valid_symbols,
            "covariance_estimator": cov_meta,
            "current_annual_volatility": f"{current_annual_vol * 100:.1f}%",
            "proposed_annual_volatility": f"{proposed_annual_vol * 100:.1f}%",
            "volatility_delta": f"{(proposed_annual_vol - current_annual_vol) * 100:+.1f} pct pts",
            "candidate_correlation_to_current_portfolio": (
                round(float(candidate_corr), 2) if pd.notna(candidate_corr) else "Data Unavailable"
            ),
            "candidate_component_volatility": f"{component_annual_vol * 100:.1f}%",
            "candidate_share_of_total_volatility": f"{component_share * 100:.1f}%",
            "interpretation": (
                "Candidate increases portfolio volatility; size should be justified by expected return or diversification benefit."
                if proposed_annual_vol > current_annual_vol
                else "Candidate does not increase modeled volatility over this lookback."
            ),
        }
        if fx_info.get("base_currency"):
            result["base_currency"] = fx_info["base_currency"]
        if fx_info.get("unavailable"):
            result["data_warning"] = (
                f"FX series unavailable for {', '.join(fx_info['unavailable'])}; "
                "their returns remain in native currency (FX risk not reflected)."
            )
        return result
    except Exception as e:
        return {"error": f"Marginal risk contribution failed: {str(e)}"}


@log_exceptions()
def analyze_correlation(symbols: list[str], period: str = "1y") -> dict[str, Any]:
    """
    Analyze correlation between assets.
    Low correlation = better diversification.
    """
    try:
        returns, valid_symbols = _get_returns(symbols, period=period)

        # Retry once if yfinance dropped symbols under concurrent load
        if returns.empty or len(valid_symbols) < 2:
            import time
            time.sleep(1)
            returns, valid_symbols = _get_returns(symbols, period=period)

        if returns.empty or len(valid_symbols) < 2:
            return {"error": "Need at least 2 valid assets for correlation analysis"}

        # Sanity-check: if a large fraction of requested symbols were dropped,
        # the diversification verdict below will be unreliable. Surface a warning
        # so callers (and the synthesis prompt) don't treat the result as authoritative.
        requested = len(symbols)
        coverage = len(valid_symbols) / requested if requested else 1.0
        data_warning = None
        if requested >= 4 and coverage < 0.5:
            dropped = sorted({s.upper() for s in symbols} - {s.upper() for s in valid_symbols})
            data_warning = (
                f"Only {len(valid_symbols)}/{requested} symbols returned price data "
                f"({coverage:.0%} coverage). Dropped: {', '.join(dropped[:10])}"
                f"{'…' if len(dropped) > 10 else ''}. Diversification verdict may be misleading."
            )

        corr_matrix = returns.corr()

        # Find highest and lowest correlations (excluding diagonal)
        correlations = []
        for i in range(len(valid_symbols)):
            for j in range(i + 1, len(valid_symbols)):
                correlations.append({
                    "pair": f"{valid_symbols[i]} vs {valid_symbols[j]}",
                    # Cast to native float: numpy scalars repr as "np.float64(0.97)"
                    # and leak that wrapper into any str()-rendered tool output.
                    "correlation": round(float(corr_matrix.iloc[i, j]), 2)
                })

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        avg_correlation = float(np.mean([abs(c["correlation"]) for c in correlations])) if correlations else 0.0

        result = {
            "symbols": valid_symbols,
            "correlation_pairs": correlations,
            "average_correlation": round(avg_correlation, 2),
            "diversification_quality": (
                "Excellent (<0.3)" if avg_correlation < 0.3 else
                "Good (0.3-0.5)" if avg_correlation < 0.5 else
                "Moderate (0.5-0.7)" if avg_correlation < 0.7 else
                "Poor (>0.7) - Assets move together"
            ),
            "hidden_correlation_warnings": _check_etf_overlap(valid_symbols),
            "recommendation": (
                "Your portfolio is well-diversified." if avg_correlation < 0.5 else
                "Consider adding uncorrelated assets (bonds, commodities, international)."
            ),
        }
        if data_warning:
            result["data_warning"] = data_warning
            # Override misleading "well-diversified" when coverage is poor.
            result["recommendation"] = (
                "Coverage incomplete — diversification verdict not reliable. "
                + result["recommendation"]
            )

        # --- Store strong correlations in Knowledge Graph ---
        try:
            from tools.graph_memory import graph_memory
            corr_tuples = []
            for c in correlations:
                if abs(c["correlation"]) > 0.7:
                    parts = c["pair"].split(" vs ")
                    if len(parts) == 2:
                        corr_tuples.append((parts[0].strip(), parts[1].strip(), c["correlation"]))
            if corr_tuples:
                graph_memory.add_portfolio_context(holdings=[], correlations=corr_tuples[:5])
        except Exception:
            pass

        return result
    except Exception as e:
        return {"error": f"Correlation analysis failed: {str(e)}"}

@log_exceptions()
def _check_etf_overlap(symbols: list[str]) -> list[str]:
    """Check for hidden overlap between ETFs and individual stocks."""
    warnings = []
    # Simplified top holdings map for common funds
    etf_exposure = {
        "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"],
        "SPY": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
        "VTI": ["AAPL", "MSFT", "NVDA", "AMZN"],
        "FTEC": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE"],
        "FSCSX": ["MSFT", "AAPL", "NVDA", "ORCL"],
        "SMH": ["NVDA", "TSM", "AMD", "AVGO"],
        "XLK": ["MSFT", "AAPL", "NVDA"]
    }

    symbols_set = set(sym.upper() for sym in symbols)

    for etf, holdings in etf_exposure.items():
        if etf in symbols_set:
            overlap = [stock for stock in holdings if stock in symbols_set and stock != etf]
            if overlap:
                warnings.append(f"⚠️ **{etf}** already holds {', '.join(overlap)}. You are doubling down on these positions.")

    return warnings

@log_exceptions()
def analyze_factors(symbols: list[str]) -> dict[str, Any]:
    """
    Count holdings by STYLE LABEL (Growth, Value, Momentum, Quality).

    SUPERSEDED by `tools/factor_exposures.py::estimate_factor_exposures`
    (Roadmap 4.2). This is a census of adjectives, not a factor exposure: each
    name is bucketed from `trailingPE` / `revenueGrowth` / `roe` thresholds and
    the buckets are counted. "40% Growth" from a label tally carries no
    information about how the portfolio actually MOVES when the growth factor
    moves, and no standard error, so it cannot say when the tilt is
    indistinguishable from none at all.

    Kept for the descriptive breakdown it gives (which names look like what), and
    the result is marked so prose built on it cannot present a label count as a
    measured loading.
    """
    factors = {"Growth": 0, "Value": 0, "Momentum": 0, "Quality": 0, "Unknown": 0}
    details = []

    for sym in symbols:
        try:
            info = get_info_safe(sym) or {}
            style = "Unknown"

            # Simple heuristic classification
            pe = info.get("trailingPE")
            rev_growth = info.get("revenueGrowth", 0)
            roe = info.get("returnOnEquity", 0)

            if pe and pe > 30 and rev_growth > 0.15:
                style = "Growth"
            elif pe and pe < 20:
                style = "Value"
            elif roe and roe > 0.20:
                style = "Quality"
            else:
                style = "Core"

            factors[style] = factors.get(style, 0) + 1
            details.append({"symbol": sym, "factor": style})
        except Exception:
            factors["Unknown"] += 1

    return {
        "factor_counts": factors,
        "details": details,
        # Roadmap 2.7/4.2: these are LABEL COUNTS, not regression loadings.
        "basis": "label heuristic",
        "basis_note": (
            "Counts of holdings bucketed by P/E, revenue growth and ROE thresholds — "
            "descriptive only. For measured exposures with significance, use "
            "estimate_factor_exposures (regression betas with t-statistics)."
        ),
    }

@log_exceptions()
def get_geographic_exposure(symbols: list[str]) -> dict[str, Any]:
    """
    Analyze geographic exposure by headquarters country.
    """
    geo = {}

    for sym in symbols:
        try:
            info = get_info_safe(sym) or {}
            country = info.get("country", "Unknown")
            geo[country] = geo.get(country, 0) + 1
        except Exception:
            geo["Unknown"] = geo.get("Unknown", 0) + 1

    return {"geographic_counts": geo}


@log_exceptions()
def calculate_var(symbols: list[str], weights: list[float] | None = None,
                  confidence: float = 0.95, investment: float = 100000, period: str = "1y") -> dict[str, Any]:
    """
    Calculate Value at Risk (VaR) - Maximum expected loss at a given confidence level.

    Args:
        symbols: List of ticker symbols
        weights: Portfolio weights
        confidence: Confidence level (default 95%)
        investment: Investment amount in dollars
        period: Historical period for volatility calculation (default '1y')

    Returns:
        VaR (daily and annual), maximum expected loss
    """
    try:
        returns, valid_symbols = _get_returns(symbols, period=period)

        if returns.empty or not valid_symbols:
            return {"error": "Could not fetch price data"}

        # Filter weights to match valid symbols
        if weights is not None:
             if len(symbols) == len(weights):
                weight_map = dict(zip(symbols, weights))
                new_weights = [weight_map.get(s, 0) for s in valid_symbols]
             else:
                new_weights = [1.0 / len(valid_symbols)] * len(valid_symbols)
        else:
            new_weights = [1.0 / len(valid_symbols)] * len(valid_symbols)

        w_array = np.array(new_weights)
        if w_array.sum() > 0:
            w_array = w_array / w_array.sum()

        # Ensure matrix alignment for dot product
        # returns columns match valid_symbols order from _get_returns
        portfolio_returns = returns[valid_symbols].dot(w_array)

        # Parametric VaR (assuming normal distribution)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        # Z-score for confidence level
        from scipy import stats
        alpha = 1 - confidence
        z_score = stats.norm.ppf(alpha)

        daily_var = -(mean_return + z_score * std_return)
        annual_var = daily_var * np.sqrt(252)

        # Dollar VaR
        daily_var_dollars = daily_var * investment
        annual_var_dollars = annual_var * investment

        # Parametric CVaR / Expected Shortfall — the average loss *given* the
        # VaR threshold is breached. VaR says how far the tail starts; CVaR says
        # how bad it is once you are in it, which is the number that matters for
        # sizing. For a normal, E[X | X <= q_alpha] = mu - sigma * phi(z)/alpha.
        daily_cvar = -(mean_return - std_return * stats.norm.pdf(z_score) / alpha)
        annual_cvar = daily_cvar * np.sqrt(252)
        daily_cvar_dollars = daily_cvar * investment
        annual_cvar_dollars = annual_cvar * investment

        # Historical VaR (empirical percentile of the actual return distribution).
        # Cast to native float — np.percentile returns np.float64, which leaks as
        # "np.float64(...)" if these values are ever surfaced unformatted.
        historical_daily_var = -float(np.percentile(portfolio_returns, alpha * 100))
        historical_daily_var_dollars = historical_daily_var * investment

        # Historical CVaR: mean of the actual observations at or beyond the
        # empirical VaR threshold. Falls back to historical VaR only if the tail
        # is empty (possible for a tiny sample).
        tail_losses = portfolio_returns[portfolio_returns <= -historical_daily_var]
        historical_daily_cvar = (
            -float(tail_losses.mean()) if len(tail_losses) > 0 else historical_daily_var
        )
        historical_daily_cvar_dollars = historical_daily_cvar * investment

        fx_info = returns.attrs.get("fx", {})
        result = {
            "confidence_level": f"{confidence * 100:.0f}%",
            "investment": f"${investment:,.0f}",
            "value_at_risk": {
                "daily_var_pct": f"{daily_var * 100:.2f}%",
                "daily_var_dollars": f"${daily_var_dollars:,.0f}",
                "annual_var_pct": f"{annual_var * 100:.1f}%",
                "annual_var_dollars": f"${annual_var_dollars:,.0f}",
                # Empirical cross-check on the parametric (normal) figures above.
                "historical_daily_var_pct": f"{historical_daily_var * 100:.2f}%",
                "historical_daily_var_dollars": f"${historical_daily_var_dollars:,.0f}",
            },
            "conditional_value_at_risk": {
                "daily_cvar_pct": f"{daily_cvar * 100:.2f}%",
                "daily_cvar_dollars": f"${daily_cvar_dollars:,.0f}",
                "annual_cvar_pct": f"{annual_cvar * 100:.1f}%",
                "annual_cvar_dollars": f"${annual_cvar_dollars:,.0f}",
                "historical_daily_cvar_pct": f"{historical_daily_cvar * 100:.2f}%",
                "historical_daily_cvar_dollars": f"${historical_daily_cvar_dollars:,.0f}",
                "definition": (
                    "Expected Shortfall: the average loss on the days when the VaR "
                    "threshold is breached, not merely the threshold itself."
                ),
            },
            "interpretation": (
                f"With {confidence * 100:.0f}% confidence, your ${investment:,.0f} portfolio "
                f"should not lose more than ${daily_var_dollars:,.0f} in a single day "
                f"or ${annual_var_dollars:,.0f} in a year. On the roughly "
                f"{alpha * 100:.0f}% of days that threshold IS breached, the average "
                f"loss is ${daily_cvar_dollars:,.0f} (CVaR)."
            ),
            "worst_case_note": (
                "Note: This is a statistical estimate. In extreme events (like 2008 or 2020 COVID crash), "
                "actual losses can exceed VaR significantly."
            )
        }
        if fx_info.get("base_currency"):
            result["base_currency"] = fx_info["base_currency"]
        if fx_info.get("unavailable"):
            result["data_warning"] = (
                f"FX series unavailable for {', '.join(fx_info['unavailable'])}; "
                "their returns remain in native currency (FX risk not reflected in VaR)."
            )
        return result
    except ImportError:
        return {"error": "scipy is required for VaR calculation. Install with: pip install scipy"}
    except Exception as e:
        return {"error": f"VaR calculation failed: {str(e)}"}



@log_exceptions()
def get_sector_exposure(symbols: list[str], weights: list[float] | None = None, is_portfolio: bool = False) -> dict[str, Any]:
    """
    Analyze sector concentration for the given symbols.
    Uses weighted analysis if weights are provided.

    is_portfolio: set True only when `symbols`/`weights` represent the user's
    actual whole portfolio. This gates the knowledge-graph write below — a
    single-symbol or arbitrary-subset call (e.g. checking one ticker's sector
    before a buy decision) must never overwrite the graph's portfolio-wide
    EXPOSED_TO edges with a skewed, narrowly-weighted breakdown (a lone stock
    always resolves to 100% of its own sector). That corrupted state would
    otherwise persist and leak into every later turn's injected memory context.
    """
    try:
        sectors = {}
        details = []

        # Normalize weights
        if weights is None:
            w_vals = [1.0 / len(symbols)] * len(symbols)
        else:
            w_vals = weights


        for i, symbol in enumerate(symbols):
            weight = w_vals[i]
            sector = "Unknown"

            # Use the unified 3-tier dynamic lookup (Universe -> API -> Knowledge Graph)
            sector = _get_sector_for_ticker(symbol)
            if sector == "Unknown":
                pass
            else:
                pass  # Default generic industry for sector matches

            # Add to tally (WEIGHTED)
            current_weight = sectors.get(sector, 0)
            sectors[sector] = current_weight + weight

            details.append({
                "symbol": symbol,
                "sector": sector,
                "weight": f"{weight*100:.1f}%"
            })

        # Formulate stats
        sector_pct = {k: round(v * 100, 1) for k, v in sectors.items()}

        # Sort by exposure
        sorted_sectors = sorted(sector_pct.items(), key=lambda x: x[1], reverse=True)
        sector_pct = dict(sorted_sectors)

        # Find concentration issues (>30% in one sector). "Private/Manual Holding"
        # and "Unknown" are wrapper/liquidity buckets (pensions, funds without a
        # resolvable sector), not an actual sector bet — flagging them here would
        # produce a nonsensical "Heavy concentration in Private/Manual Holding"
        # warning once callers pass whole-portfolio (not just tradeable-equity)
        # weights.
        _NON_SECTOR_BUCKETS = {"Diversified Fund", "Broad Market", "Private/Manual Holding", "Unknown"}
        concentrated = [s for s, pct in sector_pct.items() if pct > 30 and s not in _NON_SECTOR_BUCKETS]

        # --- AUTO-POPULATE KNOWLEDGE GRAPH with sector context ---
        # Only persist when this call actually represents the whole portfolio (see
        # is_portfolio docstring above) — otherwise a single-symbol lookup clobbers
        # the real portfolio-wide sector breakdown with a 100%-one-sector snapshot.
        if is_portfolio:
            try:
                from tools.graph_memory import graph_memory
                holdings_with_sectors = [{"symbol": d["symbol"], "sector": d["sector"]} for d in details]
                graph_memory.add_portfolio_context(
                    holdings=holdings_with_sectors,
                    sector_exposure=sector_pct
                )
            except Exception as e:
                safe_print(f"⚠️ Graph update skipped: {e}")

        return {
            "symbols_analyzed": len(symbols),
            "sector_breakdown": sector_pct,
            "details": details,
            "concentration_warning": (
                f"Heavy concentration in {', '.join(concentrated)} ({sector_pct.get(concentrated[0], 0)}%). "
                "Consider diversifying." if concentrated else None
            )
        }
    except Exception as e:
        return {"error": f"Sector analysis failed: {str(e)}"}



@log_exceptions()
def get_fee_income_analysis(symbols: list[str], weights: list[float] | None = None) -> dict[str, Any]:
    """
    Analyze portfolio fees (expense ratios) and income (dividend yield).
    """
    try:
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)

        total_expense = 0.0
        total_yield = 0.0
        high_fees = []
        dividend_payers = []

        # Hardcoded overrides for common tickers where API might miss data
        YIELD_OVERRIDES = {
            "SCHD": 0.034, "CASH.TO": 0.048, "VTI": 0.014, "SPY": 0.013,
            "VOO": 0.013, "QQQ": 0.006, "BND": 0.034, "VXUS": 0.03
        }
        EXPENSE_OVERRIDES = {
            "SCHD": 0.0006, "CASH.TO": 0.001, "VTI": 0.0003, "SPY": 0.0009,
            "VOO": 0.0003, "QQQ": 0.002, "BND": 0.0003, "VXUS": 0.0007
        }

        for i, symbol in enumerate(symbols):
            weight = weights[i]

            # Defaults
            expense = 0.0
            div_yield = 0.0

            if symbol.upper() in YIELD_OVERRIDES:
                div_yield = YIELD_OVERRIDES[symbol.upper()]
            if symbol.upper() in EXPENSE_OVERRIDES:
                expense = EXPENSE_OVERRIDES[symbol.upper()]

            # If still 0, try API (only if not explicitly set to 0 in overrides)
            info = None
            if (expense == 0.0 and symbol.upper() not in EXPENSE_OVERRIDES) or (div_yield == 0.0 and symbol.upper() not in YIELD_OVERRIDES):
                info = get_info_safe(symbol) or {}

            if expense == 0.0 and symbol.upper() not in EXPENSE_OVERRIDES:
                try:
                    er = info.get("expenseRatio") or 0.0
                    expense = float(er)
                except Exception:
                    expense = 0.0

            # A FRACTION, the same unit as YIELD_OVERRIDES above. This read used
            # `dividendYield` (a PERCENT) directly, so the sanity cap below —
            # which was the last line of defence and is now inside the helper —
            # zeroed every holding yielding more than 0.25%. The yield did not
            # come out wrong, it came out MISSING, and a missing yield is
            # indistinguishable from a non-payer in `expected_yield` and
            # `dividend_payers_count`. That silence is why the override table
            # above exists: it was hand-patching the eight tickers somebody
            # noticed reading zero, rather than a gap in the provider's data.
            if div_yield == 0.0 and symbol.upper() not in YIELD_OVERRIDES:
                div_yield = dividend_yield_fraction(info)

            # Accumulate Weighted Averages
            total_expense += (expense * weight)
            total_yield += (div_yield * weight)

            # Flag high fees (>0.50% is high for ETFs usually)
            if expense > 0.005:
                symbol_display = f"{symbol} ({expense*100:.2f}%)"
                if symbol_display not in high_fees:
                    high_fees.append(symbol_display)

            if div_yield > 0.0:
                dividend_payers.append(symbol)

        return {
            "weighted_expense_ratio": round(total_expense, 5),
            "expected_yield": round(total_yield, 5),
            "high_fee_funds": high_fees,
            "dividend_payers_count": len(dividend_payers),
            "fee_rating": (
                "Excellent (<0.1%)" if total_expense < 0.001 else
                "Good (0.1-0.3%)" if total_expense < 0.003 else
                "Moderate (0.3-0.6%)" if total_expense < 0.006 else
                "High (>0.6%) - Check fees!"
            )
        }
    except Exception as e:
        return {"error": f"Fee/Income analysis failed: {str(e)}"}


    print("\n=== Sector Exposure ===")
    print(get_sector_exposure(test_symbols))

@log_exceptions()
def generate_portfolio_charts(symbols: list[str]) -> dict[str, str]:
    """
    Generate Plotly JSON charts for:
    1. Cumulative Returns vs SPY
    2. Drawdown Underwater Plot
    3. Correlation Heatmap
    """
    charts = {}
    try:
        import json

        import plotly.graph_objects as go

        returns, valid_symbols = _get_returns(symbols)
        if returns.empty: return {}

        # 1. Equity Curve vs SPY
        spy_returns, _ = _get_returns(["SPY"])

        # Construct Equal Weighted Portfolio Index
        # (For simplicity of visualization. In real app, we'd use actual weights if passed)
        # Re-using logic from metrics to get portfolio series
        w = np.ones(len(valid_symbols)) / len(valid_symbols)
        port_returns = returns[valid_symbols].dot(w)

        cum_port = (1 + port_returns).cumprod()
        cum_port = cum_port / cum_port.iloc[0] # Normalize to 1

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, name="Portfolio", line=dict(color='#00B0F6', width=2)))

        if not spy_returns.empty:
            cum_spy = (1 + spy_returns["SPY"]).cumprod()
            # Align start dates
            common_idx = cum_port.index.intersection(cum_spy.index)
            if not common_idx.empty:
               cum_spy = cum_spy.loc[common_idx]
               cum_spy = cum_spy / cum_spy.iloc[0] # Normalize
               fig_perf.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy.values, name="S&P 500", line=dict(color='gray', dash='dash')))

        fig_perf.update_layout(title="Portfolio Performance vs Market", template="plotly_dark", hovermode="x unified")
        charts["performance_chart"] = json.dumps(fig_perf.to_dict(), default=str)

        # 2. Drawdown Chart
        rolling_max = cum_port.expanding().max()
        drawdown = (cum_port - rolling_max) / rolling_max

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy', name="Drawdown", line=dict(color='red')))
        fig_dd.update_layout(title="Portfolio Drawdown (Risk)", yaxis_tickformat=".1%", template="plotly_dark")
        charts["drawdown_chart"] = json.dumps(fig_dd.to_dict(), default=str)

        # 3. Correlation Heatmap
        corr = returns.corr()
        fig_corr = go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis',
            zmin=-1, zmax=1
        )
        layout = go.Layout(title="Asset Correlation Matrix", template="plotly_dark")
        fig_heatmap = go.Figure(data=[fig_corr], layout=layout)
        charts["correlation_chart"] = json.dumps(fig_heatmap.to_dict(), default=str)

        return charts

    except Exception as e:
        safe_print(f"Chart generation failed: {e}")
        return {}


@tool
@log_exceptions()
def calculate_currency_exposure(holdings: dict[str, float], currencies: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Analyzes portfolio currency exposure (USD vs CAD).
    Args:
        holdings: Dict of ticker symbols to current market value (e.g. {'AAPL': 15000.0, 'RY.TO': 10000.0})
        currencies: Optional dict mapping ticker symbols to explicit currencies (e.g. {'AAPL': 'USD'})
    Returns:
        Dict with total value, currency breakdown, and raw details.
    """
    def get_profile_base_currency() -> str:
        try:
            from tools.memory import get_profile_base_currency as _get_profile_base_currency
            return _get_profile_base_currency()
        except Exception:
            return os.environ.get("BASE_CURRENCY") or os.environ.get("CAIRNIQ_BASE_CURRENCY") or "USD"

    def guess_currency(ticker_name: str, default: str = "USD") -> str:
        t_up = ticker_name.upper()
        suffix_map = {
            ".TO": "CAD", ".V": "CAD", ".VN": "CAD", ".CN": "CAD",
            ".L": "GBP",
            ".DE": "EUR", ".PA": "EUR", ".MI": "EUR", ".AS": "EUR",
            ".AX": "AUD",
            ".T": "JPY",
            "CAD": "CAD", "USD": "USD", "EUR": "EUR", "GBP": "GBP"
        }
        for suffix, cur in suffix_map.items():
            if suffix in t_up:
                return cur
        return default

    base_currency = get_profile_base_currency()
    total_value = 0.0
    exposure = {}
    details = []

    for ticker, value in holdings.items():
        try:
            # Handle cash or special keys
            if ticker in ["USD", "CAD", "EUR", "GBP", "Cash"]:
                currency = ticker if ticker != "Cash" else base_currency

                exposure[currency] = exposure.get(currency, 0.0) + value
                total_value += value
                details.append({"ticker": ticker, "currency": currency, "value": value})
                continue

            # 1. Use explicit currency dictionary if provided
            if currencies and ticker in currencies:
                currency = currencies[ticker]
            else:
                # 2. Try yfinance
                info = yf.Ticker(ticker).info or {}
                currency = info.get("currency") or guess_currency(ticker, base_currency)

            exposure[currency] = exposure.get(currency, 0.0) + value
            total_value += value
            details.append({"ticker": ticker, "currency": currency, "value": value})

        except Exception as e:
            safe_print(f"⚠️ Error processing {ticker}: {e}")
            currency = guess_currency(ticker, base_currency)
            exposure[currency] = exposure.get(currency, 0.0) + value
            total_value += value
            details.append({"ticker": ticker, "currency": currency, "value": value})

    # Calculate percentages
    if total_value > 0:
        percentages = {k: round((v / total_value) * 100, 1) for k, v in exposure.items()}
    else:
        percentages = {k: 0.0 for k in exposure}

    return {
        "total_value": total_value,
        "breakdown_value": exposure,
        "breakdown_percent": percentages,
        "details": details
    }
