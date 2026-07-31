"""
Pre-trade candidate impact preview (Advisor Roadmap Theme 4.9).

Answers the question an advisor gets most — "should I add this?" — by
recomputing the portfolio WITH the candidate at a proposed size and reporting
the DELTA. It is a delta report, not a verdict: it says what adding the name
does to concentration, sector caps, beta, volatility, and tail risk, and lets
the IPS gate (Theme 2.2) and the human make the call.

Everything it needs already exists; 4.9 wires it together:
  - IPS caps / sizing / dollar-at-risk  → tools.ips_precheck.run_ips_precheck
    (the SAME deterministic engine that gates a live instruction at ship time,
    so a preview can never say "fine" where the gate would later FAIL).
  - Risk deltas (vol, beta, CVaR, marginal contribution) → the 4.1 estimation
    layer: one base-currency return fetch (tools.portfolio_analytics._get_returns)
    + Ledoit-Wolf covariance (tools.covariance.estimate_covariance).

The whole risk block is computed from ONE price fetch of
(tradeable holdings ∪ candidate ∪ SPY), so every before/after number comes
from the same snapshot and the tool makes a single network round-trip.

Never raises: any failure degrades to a partial report with a data note.
This is computed analysis, not personalized investment advice.
"""
from typing import Any

import numpy as np
import pandas as pd

from tools.exception_logger import log_exceptions

_DEFAULT_PROBE_PCT = 5.0      # assumed position size when the caller states none
_CVAR_CONFIDENCE = 0.95
_TRADING_DAYS = 252


# --- seams (kept thin so tests can mock network + heavy collaborators) --------

def _decision_context() -> dict[str, Any]:
    from tools.portfolio_csv import get_portfolio_decision_context
    return get_portfolio_decision_context()


def _impact_returns(symbols: list[str], period: str) -> tuple[pd.DataFrame, list[str]]:
    """Base-currency return series for the impact math (4.1 estimation layer)."""
    from tools.portfolio_analytics import _get_returns
    return _get_returns(symbols, period=period)


def _candidate_quote(symbol: str) -> tuple[float | None, str]:
    """(price, listing currency); currency "" when the feed omits it.

    The currency travels with the price because the caller multiplies it out
    and compares the result against a base-currency portfolio total.
    """
    from tools.market_data import get_realtime_quote
    try:
        quote = get_realtime_quote(symbol)
        if not isinstance(quote, dict):
            return None, ""
        price = quote.get("price")
        price = float(price) if isinstance(price, (int, float)) and price > 0 else None
        currency = str(quote.get("currency") or "").strip().upper()
        return price, (currency if len(currency) == 3 and currency.isalpha() else "")
    except Exception:
        return None, ""


# Currency resolution is delegated to the 2.2 gate rather than reimplemented:
# a preview that priced a trade differently from the gate that later blocks it
# would be worse than no preview at all.
def _held_currency(ctx: dict[str, Any], ticker: str) -> str:
    from tools.ips_precheck import _held_currency as _ips_held_currency
    return _ips_held_currency(ctx, ticker)


def _get_fx_rate(from_currency: str, to_currency: str) -> float:
    from tools.ips_precheck import _get_fx_rate as _ips_fx_rate
    return _ips_fx_rate(from_currency, to_currency)


# --- helpers ------------------------------------------------------------------

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


def _parametric_var_cvar(port_returns: pd.Series, confidence: float) -> tuple[float, float]:
    """(annual VaR, annual CVaR) as positive-loss decimals — same formulas as
    tools.portfolio_analytics.calculate_var, kept consistent on purpose."""
    from scipy import stats

    mean = float(port_returns.mean())
    std = float(port_returns.std())
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    daily_var = -(mean + z * std)
    daily_cvar = -(mean - std * stats.norm.pdf(z) / alpha)
    root = np.sqrt(_TRADING_DAYS)
    return daily_var * root, daily_cvar * root


def _beta(port_returns: pd.Series, spy_returns: pd.Series) -> float | None:
    common = port_returns.index.intersection(spy_returns.index)
    if len(common) < 2:
        return None
    p = port_returns.loc[common]
    s = spy_returns.loc[common]
    var = float(s.var())
    if var <= 0:
        return None
    beta = float(np.cov(p, s)[0, 1] / var)
    return beta if not np.isnan(beta) else None


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _signed_pp(delta: float) -> str:
    return f"{delta * 100:+.1f} pct pts"


# --- risk-delta block ---------------------------------------------------------

@log_exceptions()
def _compute_risk_deltas(
    tradeable: dict[str, float],
    candidate: str,
    candidate_weight: float,
    total_value: float,
    size_usd: float,
    period: str,
) -> dict[str, Any]:
    """Before/after vol, beta, CVaR, correlation, and marginal vol share from a
    single return fetch. candidate_weight is the candidate's FINAL weight in the
    proposed book (existing positions scale by 1 - candidate_weight)."""
    tradeable_symbols = list(tradeable.keys())
    fetch = list(dict.fromkeys(tradeable_symbols + [candidate, "SPY"]))
    returns, valid = _impact_returns(fetch, period)
    if returns is None or getattr(returns, "empty", True):
        return {"error": "Could not fetch price history for the impact analysis."}
    if candidate not in valid:
        return {"error": f"Insufficient price history for {candidate}; risk deltas unavailable."}

    valid_tradeable = [s for s in tradeable_symbols if s in valid]
    if not valid_tradeable:
        return {"error": "No current holdings had usable price history; cannot compute a portfolio delta."}

    # Current weights over the holdings that actually have history.
    cur_mass = sum(tradeable[s] for s in valid_tradeable)
    if cur_mass <= 0:
        return {"error": "Current holdings have no positive value; cannot weight the portfolio."}
    w_cur = {s: tradeable[s] / cur_mass for s in valid_tradeable}

    asset_symbols = valid_tradeable + ([candidate] if candidate not in valid_tradeable else [])
    cand_idx = asset_symbols.index(candidate)

    w_cur_arr = np.array([w_cur.get(s, 0.0) for s in asset_symbols])
    w_prop_arr = w_cur_arr * (1.0 - candidate_weight)
    w_prop_arr[cand_idx] += candidate_weight

    from tools.covariance import estimate_covariance
    cov, cov_meta = estimate_covariance(returns[asset_symbols])
    sigma = cov.values

    daily_vol_cur = float(np.sqrt(w_cur_arr @ sigma @ w_cur_arr))
    daily_vol_prop = float(np.sqrt(w_prop_arr @ sigma @ w_prop_arr))
    ann_vol_cur = daily_vol_cur * np.sqrt(_TRADING_DAYS)
    ann_vol_prop = daily_vol_prop * np.sqrt(_TRADING_DAYS)

    port_cur = returns[valid_tradeable].dot(np.array([w_cur[s] for s in valid_tradeable]))
    port_prop = returns[asset_symbols].dot(w_prop_arr)

    # Beta (vs SPY) before/after.
    beta_cur = beta_prop = None
    if "SPY" in valid:
        beta_cur = _beta(port_cur, returns["SPY"])
        beta_prop = _beta(port_prop, returns["SPY"])

    # Tail risk: % from returns, dollarized (proposed adds new money → larger base).
    var_cur, cvar_cur = _parametric_var_cvar(port_cur, _CVAR_CONFIDENCE)
    var_prop, cvar_prop = _parametric_var_cvar(port_prop, _CVAR_CONFIDENCE)
    cvar_cur_dollars = cvar_cur * total_value
    cvar_prop_dollars = cvar_prop * (total_value + size_usd)

    # Candidate's correlation to the current book + its share of proposed vol.
    cand_corr = port_cur.corr(returns[candidate])
    if daily_vol_prop > 0:
        sigma_w = sigma @ w_prop_arr
        comp_daily = w_prop_arr[cand_idx] * sigma_w[cand_idx] / daily_vol_prop
        cand_vol_share = (comp_daily * np.sqrt(_TRADING_DAYS)) / ann_vol_prop if ann_vol_prop > 0 else 0.0
    else:
        cand_vol_share = 0.0

    conf = int(_CVAR_CONFIDENCE * 100)
    block: dict[str, Any] = {
        "covariance_estimator": cov_meta,
        "symbols_analyzed": asset_symbols,
        "volatility": {
            "current": _pct(ann_vol_cur),
            "proposed": _pct(ann_vol_prop),
            "delta": _signed_pp(ann_vol_prop - ann_vol_cur),
        },
        f"cvar_{conf}_annual": {
            "current": _pct(cvar_cur),
            "proposed": _pct(cvar_prop),
            "current_dollars": f"${cvar_cur_dollars:,.0f}",
            "proposed_dollars": f"${cvar_prop_dollars:,.0f}",
            "delta_dollars": f"${cvar_prop_dollars - cvar_cur_dollars:+,.0f}",
            "definition": (
                f"Expected shortfall at {conf}% — average annual loss in the worst "
                f"{100 - conf}% of outcomes. Proposed dollars are on the larger "
                "post-trade base (new money added)."
            ),
        },
        "candidate_correlation_to_portfolio": (
            round(float(cand_corr), 2) if pd.notna(cand_corr) else "unavailable"
        ),
        "candidate_share_of_proposed_volatility": _pct(cand_vol_share),
        "diversification_note": (
            f"{candidate} is lowly/negatively correlated ({cand_corr:.2f}) to the current book — "
            "it diversifies more than it concentrates."
            if pd.notna(cand_corr) and cand_corr < 0.4 else
            f"{candidate} is highly correlated ({cand_corr:.2f}) to the current book — "
            "it adds exposure more than diversification."
            if pd.notna(cand_corr) else
            "Correlation to the current book could not be computed."
        ),
    }
    if beta_cur is not None and beta_prop is not None:
        block["beta"] = {
            "current": round(beta_cur, 2),
            "proposed": round(beta_prop, 2),
            "delta": round(beta_prop - beta_cur, 2),
        }

    fx_info = returns.attrs.get("fx", {}) if hasattr(returns, "attrs") else {}
    if fx_info.get("base_currency"):
        block["base_currency"] = fx_info["base_currency"]
    dropped = [s for s in tradeable_symbols if s not in valid]
    if dropped:
        block["data_note"] = f"No price history for {', '.join(dropped)} — excluded from the risk math."
    if fx_info.get("unavailable"):
        block["fx_note"] = (
            f"FX series unavailable for {', '.join(fx_info['unavailable'])}; "
            "their returns stayed in native currency (FX risk not fully reflected)."
        )
    return block


# --- public API ---------------------------------------------------------------

@log_exceptions()
def preview_candidate_impact(
    candidate_symbol: str,
    size_usd: float | None = None,
    size_pct: float | None = None,
    shares: float | None = None,
    stop: float | None = None,
    entry: float | None = None,
    period: str = "1y",
) -> dict[str, Any]:
    """Recompute the portfolio WITH `candidate_symbol` at a proposed size and
    report the delta: IPS position/sector/dollar-at-risk checks (the 2.2 gate's
    own engine) plus before/after beta, volatility, and CVaR.

    Size resolution order: size_pct (% of portfolio) → shares (× live quote) →
    size_usd → an assumed 5% probe. All dollar figures in the report are base
    currency; a shares-based size is converted from the security's listing
    currency first, and abstains (returns an error) when that currency or its
    FX rate cannot be established. `size_usd` is taken as base currency —
    that is the caller's contract, not something this function can verify.

    Returns a delta report, never a buy/sell verdict.
    """
    candidate = str(candidate_symbol or "").upper().strip()
    if not candidate:
        return {"error": "candidate_symbol is required"}

    ctx = _decision_context()
    if not isinstance(ctx, dict) or ctx.get("error"):
        return {"error": "Portfolio context unavailable — cannot compute a candidate impact."}
    total_value = ctx.get("total_value_base")
    if not isinstance(total_value, (int, float)) or total_value <= 0:
        return {"error": "Portfolio total value is unavailable or zero."}
    total_value = float(total_value)
    base_currency = str(ctx.get("base_currency") or "USD").upper()

    # --- resolve proposed size to base-currency dollars ---
    size_basis = "stated"
    if size_pct is not None and size_pct > 0:
        size_usd = float(size_pct) / 100.0 * total_value
        size_basis = f"{size_pct:g}% of portfolio"
    elif shares is not None and shares > 0:
        # shares × price lands in the SECURITY's currency, but every figure
        # derived from it below — candidate_weight, pct_of_current_portfolio,
        # the risk deltas, the synthesized pre-check text — is compared against
        # a base-currency total. Converting is not optional: for a CAD profile
        # pricing a US listing, face value understates the position by the whole
        # FX rate. Same rule as the 2.2 gate (tools.ips_precheck._to_base):
        # convert, or abstain — never compare unconverted. A held position's
        # currency also labels a caller-supplied `entry` (same security).
        currency = _held_currency(ctx, candidate)
        price = entry
        if price is None or (not currency and base_currency != "USD"):
            quote_price, quote_currency = _candidate_quote(candidate)
            price = price or quote_price
            currency = currency or quote_currency
        if not price:
            return {"error": f"Could not price {shares:g} shares of {candidate} (no quote/entry)."}

        native = float(shares) * float(price)
        if currency and currency != base_currency:
            try:
                rate = _get_fx_rate(currency, base_currency)
            except Exception:
                rate = 0.0
            if not rate or rate <= 0:
                return {"error": (
                    f"No {currency}→{base_currency} rate available — refusing to size "
                    f"{shares:g} shares of {candidate} by comparing an unconverted "
                    f"{currency} figure against a {base_currency} portfolio."
                )}
            size_usd = native * rate
            # Base currency leads, native in parentheses — the profile's
            # reporting rule, same as the pre-check table.
            size_basis = (
                f"{shares:g} shares ≈${size_usd:,.0f} {base_currency} "
                f"(at ${price:,.2f} {currency})"
            )
        elif currency or base_currency == "USD":
            size_usd = native
            size_basis = f"{shares:g} shares × ${price:,.2f}"
        else:
            return {"error": (
                f"Could not establish the currency of {candidate}'s price — refusing to "
                f"compare it against a {base_currency} portfolio."
            )}
    elif size_usd is not None and size_usd > 0:
        size_usd = float(size_usd)
    else:
        size_usd = _DEFAULT_PROBE_PCT / 100.0 * total_value
        size_basis = f"assumed {_DEFAULT_PROBE_PCT:g}% probe (no size stated)"

    # Candidate's FINAL weight in the proposed book (new money → denominator grows).
    candidate_weight = size_usd / (total_value + size_usd)

    report: dict[str, Any] = {
        "candidate": candidate,
        "base_currency": base_currency,
        "portfolio_total": f"${total_value:,.0f}",
        "proposed_size": {
            "dollars": f"${size_usd:,.0f}",
            "pct_of_current_portfolio": _pct(size_usd / total_value),
            "basis": size_basis,
        },
        "disclaimer": (
            "Computed portfolio-impact analysis (a delta report), not a recommendation "
            "to buy or sell. The IPS checks below use the same deterministic engine that "
            "gates a live instruction at ship time (Theme 2.2)."
        ),
    }

    # --- IPS compliance / sizing block: reuse the 2.2 gate's own engine ---
    # The phrasing matters: ips_precheck's size extractor requires the buy verb
    # immediately before the dollar figure ("Buy $X of TICKER"), so "Buy TICKER
    # $X" parses NO size and every cap check silently degrades to NOT_EVALUATED.
    # tests/test_tools/test_candidate_impact.py::test_synthesized_buy_text_parses
    # guards this exact seam.
    try:
        from tools.ips_precheck import run_ips_precheck
        text = f"Buy ${size_usd:,.0f} of {candidate}"
        if entry:
            text += f" at ${entry:,.2f}"
        if stop:
            text += f" with a stop at ${stop:,.2f}"
        precheck = run_ips_precheck(text + ".", candidate_tickers={candidate})
        rows = [r for r in precheck.get("rows", []) if r.get("check") != "account location"]
        report["ips_checks"] = {
            "rows": rows,
            "flags": precheck.get("violations", []),
            "note": (
                "Position/sector/dollar-at-risk vs the profile's IPS constraints — "
                "the same computation the compliance gate runs before any instruction "
                "ships. A FAIL here is what would block the trade."
            ),
        }
    except Exception:
        report["ips_checks"] = {"rows": [], "flags": [], "note": "IPS pre-check unavailable this pass."}

    # --- risk-metric deltas: the 4.1 estimation layer ---
    tradeable = _aggregate_tradeable(ctx.get("holdings", []))
    report["risk_deltas"] = _compute_risk_deltas(
        tradeable, candidate, candidate_weight, total_value, size_usd, period
    )

    report["headline"] = _headline(report)
    return report


def _headline(report: dict[str, Any]) -> str:
    """One-line human summary the agent can lead with."""
    cand = report["candidate"]
    size = report["proposed_size"]["dollars"]
    bits = [f"Adding {size} of {cand}"]
    risk = report.get("risk_deltas", {})
    if isinstance(risk, dict) and "beta" in risk:
        bits.append(f"beta {risk['beta']['current']}→{risk['beta']['proposed']}")
    if isinstance(risk, dict) and "volatility" in risk:
        bits.append(f"vol {risk['volatility']['delta']}")
    conf_key = next((k for k in risk if k.startswith("cvar_")), None) if isinstance(risk, dict) else None
    if conf_key:
        bits.append(f"CVaR {risk[conf_key]['delta_dollars']}")
    flags = report.get("ips_checks", {}).get("flags", [])
    if flags:
        bits.append(f"⚠️ {len(flags)} IPS flag(s)")
    else:
        bits.append("no IPS breach")
    return " · ".join(bits)
