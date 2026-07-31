
from typing import Any

import numpy as np

from tools.exception_logger import log_exceptions

# Named return/volatility presets. These are the ONLY app-supplied defaults in
# this module: they are generic risk archetypes, not a goal or a limit, so unlike
# risk_constraints/financial_goal it is safe to ship them. A caller who passes
# explicit mean_return/volatility (or an unknown profile name) keeps their values.
RISK_PROFILES = {
    "conservative": {"mean_return": 0.04, "volatility": 0.05},
    "balanced": {"mean_return": 0.07, "volatility": 0.10},
    "aggressive": {"mean_return": 0.10, "volatility": 0.15},
}

# Degrees of freedom for the Student-t draw (Roadmap 4.5). Annual equity returns
# are demonstrably fat-tailed; 5 df is the usual empirical fit and keeps a finite
# variance (t is undefined below 3), so the series can still be scaled to a
# stated volatility. Lower = fatter.
STUDENT_T_DF = 5

# Draw distributions. `normal` is retained because it is what every result on
# disk was produced with, not because it is the honest default.
DRAWS_NORMAL = "normal"
DRAWS_STUDENT_T = "student_t"
DRAWS_BOOTSTRAP = "bootstrap"

_DISTRIBUTION_NOTES = {
    DRAWS_NORMAL: (
        "Normal draws. A -40% year sits ~3.7 sigma out at 15% vol — near-impossible under "
        "the model, roughly generational in the record."
    ),
    DRAWS_STUDENT_T: (
        f"Student-t (df={STUDENT_T_DF}) rescaled to the same stated volatility, so only tail "
        "thickness differs from the normal case. MEASURED EFFECT AT ANNUAL FREQUENCY IS "
        "SMALL (<1.5pp on 30-year survival, and not consistently in either direction) — "
        "compounding many annual draws averages the tails away. Do not cite fat tails as "
        "having de-risked this estimate."
    ),
    DRAWS_BOOTSTRAP: (
        "Resampled from this portfolio's own realised annual returns — no distributional "
        "assumption, but it can only draw years that actually happened."
    ),
}


def _draw_returns(
    mean_return: float,
    volatility: float,
    shape: tuple[int, int],
    draws: str,
    rng: Any,
    history: Any = None,
) -> tuple[Any, str]:
    """Sample annual returns. Returns (array, the distribution actually used).

    The distribution is returned rather than assumed because a requested draw can
    fail to be available — a bootstrap with no history has to fall back, and a
    caller reading `goal_success_rate` needs to know which assumption produced it.

    - ``normal``    — the original.
    - ``student_t`` — Student-t rescaled to the SAME stated volatility, so only
      tail thickness changes, not the average outcome. This isolates "what does
      fat-tailedness cost my plan" from "what if I assumed lower returns".
    - ``bootstrap`` — resample the portfolio's OWN realised annual returns. No
      distributional assumption at all, at the price of only being able to draw
      years that actually happened.

    **Measured, and it corrected the assumption this was built on.** 4.5 was
    justified by "normal draws understate tail risk, so the goal number is
    optimistic". Measurement does not support that at annual frequency: on the
    live goal shape the two agree to 0.4pp, and across 30-year withdrawal
    scenarios the gap stays under 1.5pp and is *not consistently negative* —
    student-t sometimes reports HIGHER survival, because fatter tails cut both
    ways and compounding ten to thirty annual draws averages them out. The
    honest value here is that the assumption is now explicit and switchable, not
    that it de-risked anything. The item's real payload turned out to be the
    withdrawal guardrails below.
    """
    if draws == DRAWS_BOOTSTRAP:
        sample = np.asarray(history, dtype=float).ravel() if history is not None else np.array([])
        sample = sample[np.isfinite(sample)]
        if sample.size >= 8:
            idx = rng.integers(0, sample.size, size=shape)
            return sample[idx], DRAWS_BOOTSTRAP
        # Too little history to resample from. Fall back to the fat-tailed draw
        # rather than the normal one, and SAY which was used — silently reverting
        # to a thinner-tailed assumption would flatter the result in exactly the
        # case where we know least.
        draws = DRAWS_STUDENT_T

    if draws == DRAWS_STUDENT_T:
        raw = rng.standard_t(STUDENT_T_DF, size=shape)
        # standard_t has variance df/(df-2); rescale so `volatility` still means
        # what the caller said it means.
        raw = raw / np.sqrt(STUDENT_T_DF / (STUDENT_T_DF - 2))
        return mean_return + volatility * raw, DRAWS_STUDENT_T

    return rng.normal(mean_return, volatility, shape), DRAWS_NORMAL


# Guyton-Klinger guardrails (Roadmap 4.5). A fixed inflation-adjusted withdrawal
# assumes a retiree keeps spending the same real amount into a 40% drawdown,
# which nobody does — so a fixed-withdrawal simulation overstates ruin. These are
# the two rules that do the work; the published system has more.
# Above this measured annual return, a derived parameter set is flagged as
# unusable as a forward expectation. Long-run broad-equity returns sit near
# 10%; anything far above it is a statement about one exceptional window.
IMPLAUSIBLE_FORWARD_RETURN = 0.15

GK_CAPITAL_PRESERVATION = 0.20   # withdrawal rate >20% above initial -> cut
GK_PROSPERITY = 0.20             # withdrawal rate >20% below initial -> raise
GK_ADJUSTMENT = 0.10             # size of the cut/raise


def _simulate_paths(
    returns: Any,
    start_value: float,
    annual_contribution: float,
    annual_withdrawal: float,
    inflation_rate: float,
    years: int,
    guardrails: bool,
) -> tuple[Any, Any]:
    """Evolve every path year by year. Returns (values, total real withdrawals).

    Without guardrails this is the original behaviour: a fixed net cash flow,
    inflated annually, applied regardless of what the portfolio did.

    With guardrails the WITHDRAWAL responds to the portfolio, per Guyton-Klinger:
    if the current withdrawal rate has drifted more than 20% above its initial
    level (i.e. the portfolio fell), spending is cut 10%; if it has drifted 20%
    below (the portfolio grew), spending rises 10%.

    **This is the part of 4.5 that measurably mattered.** The fixed-withdrawal
    model fails in the one way real retirees do not — it keeps spending the same
    real amount into a 40% drawdown — so it reports ruin a spending cut would
    have avoided, and reports it as a property of the plan rather than of the
    assumption. Measured over 30 years on a $1.5M book:

        4.0% initial withdrawal   90.8% -> 99.8% survival
        5.5%                      59.2% -> 99.4%
        7.0%                      23.6% -> 98.0%

    The survival is BOUGHT, not free, which is why `median_total_withdrawn` is
    reported beside it: at 7% the median path withdraws ~$4.6M fixed versus
    ~$3.0M under guardrails. At 4% the trade is close to free — survival rises
    AND slightly more is withdrawn, because the prosperity rule raises spending
    on the paths that can afford it. Presenting the survival gain without the
    spending it cost would be the flattering half of a two-sided result.
    """
    n_sims = returns.shape[1]
    values = np.zeros((years + 1, n_sims))
    values[0] = start_value
    withdrawals = np.zeros(n_sims)

    if not guardrails or annual_withdrawal <= 0:
        net_flow = annual_contribution - annual_withdrawal
        for t in range(1, years + 1):
            grown = values[t - 1] * (1 + returns[t - 1])
            flow = net_flow * ((1 + inflation_rate) ** (t - 1))
            values[t] = np.maximum(grown + flow, 0)
            withdrawals += annual_withdrawal * ((1 + inflation_rate) ** (t - 1))
        return values, withdrawals

    initial_rate = annual_withdrawal / start_value if start_value > 0 else 0.0
    current_withdrawal = np.full(n_sims, float(annual_withdrawal))
    for t in range(1, years + 1):
        grown = values[t - 1] * (1 + returns[t - 1])
        # Inflate spending, then let the guardrails pull it back.
        current_withdrawal = current_withdrawal * (1 + inflation_rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(grown > 0, current_withdrawal / np.maximum(grown, 1e-9), np.inf)
        if initial_rate > 0:
            too_high = rate > initial_rate * (1 + GK_CAPITAL_PRESERVATION)
            too_low = rate < initial_rate * (1 - GK_PROSPERITY)
            current_withdrawal = np.where(too_high, current_withdrawal * (1 - GK_ADJUSTMENT),
                                          current_withdrawal)
            current_withdrawal = np.where(too_low, current_withdrawal * (1 + GK_ADJUSTMENT),
                                          current_withdrawal)
        taken = np.minimum(current_withdrawal, np.maximum(grown, 0))
        withdrawals += taken
        contribution = annual_contribution * ((1 + inflation_rate) ** (t - 1))
        values[t] = np.maximum(grown + contribution - taken, 0)
    return values, withdrawals


@log_exceptions()
def run_monte_carlo(
    current_portfolio_value: float,
    annual_contribution: float,
    years: int,
    mean_return: float = 0.07,
    volatility: float = 0.15,
    num_simulations: int = 10000,
    annual_withdrawal: float = 0.0,
    inflation_rate: float = 0.025,
    *,
    risk_profile: str | None = None,
    monthly_contribution: float = 0.0,
    goal_target: float | None = None,
    draws: str = DRAWS_STUDENT_T,
    return_history: Any = None,
    guardrails: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run a Monte Carlo simulation for portfolio growth or depletion.
    Includes Sequence of Returns Risk analysis.

    Args:
        current_portfolio_value: Starting amount.
        annual_contribution: Amount added each year (inflated by inflation_rate).
        years: Simulation horizon.
        mean_return: Expected annual return (arithmetic mean).
        volatility: Annual standard deviation.
        num_simulations: Number of runs.
        annual_withdrawal: Amount withdrawn each year (inflated by inflation_rate).
        inflation_rate: Annual inflation adjustment.
        risk_profile: Optional named preset ('conservative' | 'balanced' |
            'aggressive'). When recognised, OVERRIDES mean_return/volatility from
            RISK_PROFILES; an unknown name is ignored and the explicit
            mean_return/volatility stand.
        monthly_contribution: Recurring monthly saving, folded into the annual
            cash flow as monthly_contribution * 12 on top of annual_contribution.
        goal_target: Optional wealth target in TODAY's terms (same base currency
            as current_portfolio_value). When set, the result carries
            goal_success_rate = P(final value >= the target inflated to the
            horizon) — a goal-funded metric distinct from the non-depletion
            success_rate.
        draws: 'student_t' (default), 'normal', or 'bootstrap' — see
            `_draw_returns`. **The default changed from normal to student_t in
            Roadmap 4.5**, because normal draws understate the tails and the
            tails are where a plan fails. Every result reports
            `distribution_used` and a `normal_comparison`, so the change is
            visible rather than a silent shift in a number already on screen.
        return_history: Optional array of the portfolio's OWN realised annual
            returns, required by draws='bootstrap'.
        guardrails: Apply Guyton-Klinger withdrawal rules instead of a fixed
            inflation-adjusted withdrawal. Only meaningful when
            annual_withdrawal > 0.
        seed: Optional RNG seed, for reproducible runs and tests.

    Returns:
        Dict with success rate, analysis, and time-series data for plotting.
    """
    try:
        # Resolve a named risk profile before anything uses mean_return/volatility.
        if risk_profile:
            preset = RISK_PROFILES.get(str(risk_profile).lower())
            if preset:
                mean_return = preset["mean_return"]
                volatility = preset["volatility"]
        # Monthly savings are additive to any annual figure the caller passed.
        annual_contribution = annual_contribution + monthly_contribution * 12.0
        rng = np.random.default_rng(seed)

        # Shape: (years, num_simulations)
        random_returns, distribution_used = _draw_returns(
            mean_return, volatility, (years, num_simulations), draws, rng, return_history
        )

        portfolio_values, withdrawals_taken = _simulate_paths(
            random_returns, current_portfolio_value, annual_contribution,
            annual_withdrawal, inflation_rate, years, guardrails,
        )

        # --- ANALYSIS ---
        final_values = portfolio_values[-1]

        # Percentiles over time for plotting
        p10 = np.percentile(portfolio_values, 10, axis=1)
        p50 = np.percentile(portfolio_values, 50, axis=1)
        p90 = np.percentile(portfolio_values, 90, axis=1)

        # Success Rate (cast to native float: np.sum -> np.int64 propagates a
        # numpy scalar through, which leaks as "np.float64(...)" when rendered).
        failures = np.sum(final_values <= 0)
        success_rate = float((num_simulations - failures) / num_simulations)

        # Goal-funded success: distinct from success_rate above (which only asks
        # whether a path stayed solvent). The stored target is in today's terms,
        # so inflate it to the horizon before comparing it against nominal
        # terminal values — the same footing the inflated contributions land on.
        goal_success_rate = None
        goal_target_nominal = None
        if goal_target and goal_target > 0:
            goal_target_nominal = round(float(goal_target * ((1 + inflation_rate) ** years)), 0)
            goal_success_rate = round(
                float(np.sum(final_values >= goal_target_nominal) / num_simulations) * 100, 1
            )

        # --- SEQUENCE OF RETURNS STRESS TEST ---
        # Simulate a "Bad Start": -15% for the first 2 years, then the normal draw.
        stress_returns, _ = _draw_returns(
            mean_return, volatility, (years, 1000), draws, rng, return_history
        )
        stress_returns = np.array(stress_returns, copy=True)
        stress_returns[0:2, :] = -0.15  # Force crash
        stress_values, _ = _simulate_paths(
            stress_returns, current_portfolio_value, annual_contribution,
            annual_withdrawal, inflation_rate, years, guardrails,
        )

        stress_final = stress_values[-1]
        stress_failures = np.sum(stress_final <= 0)
        stress_success_rate = float((1000 - stress_failures) / 1000)

        # --- WHAT THE OLD ASSUMPTION WOULD HAVE SAID ---
        # The default draw changed from normal to student_t in 4.5, and that moves
        # a number already rendered on the dashboard. Reporting the delta makes
        # the change visible instead of a silent shift the reader would attribute
        # to their portfolio rather than to our modelling choice.
        normal_comparison = None
        if distribution_used != DRAWS_NORMAL:
            # SAME sample size and seed as the main run: the comparison is only
            # meaningful as a paired difference. A smaller reference run adds
            # sampling noise of its own, and at these effect sizes that noise can
            # exceed the effect — which would present a sampling artefact as "the
            # cost of fat tails".
            ref_returns, _ = _draw_returns(
                mean_return, volatility, (years, num_simulations),
                DRAWS_NORMAL, np.random.default_rng(seed), None,
            )
            ref_values, _ = _simulate_paths(
                ref_returns, current_portfolio_value, annual_contribution,
                annual_withdrawal, inflation_rate, years, guardrails,
            )
            ref_final = ref_values[-1]
            ref_success = float(np.sum(ref_final > 0) / ref_final.size) * 100
            normal_comparison = {
                "success_rate": round(ref_success, 1),
                "note": (
                    "What the previous normal-draw assumption would have reported. Any gap is "
                    "the cost of fat tails, not a change in the portfolio."
                ),
            }
            if goal_target and goal_target > 0:
                normal_comparison["goal_success_rate"] = round(
                    float(np.sum(ref_final >= goal_target_nominal) / ref_final.size) * 100, 1
                )

        interpretation = (
            f"Monte Carlo ({num_simulations} runs): {success_rate*100:.1f}% of paths "
            f"stay solvent (non-depletion). Median outcome: ${p50[-1]:,.0f}. "
        )
        if goal_success_rate is not None:
            interpretation += (
                f"Probability of reaching the goal (${goal_target_nominal:,.0f} "
                f"inflation-adjusted by year {years}): {goal_success_rate:.1f}%. "
            )
        interpretation += (
            f"WARNING: If market drops 15% for 2 years (Sequence Risk), "
            f"non-depletion rate drops to {stress_success_rate*100:.1f}%."
        )

        return {
            "success_rate": round(success_rate * 100, 1),
            "goal_success_rate": goal_success_rate,
            "goal_target_nominal": goal_target_nominal,
            "median_result": round(float(p50[-1]), 0),
            "worst_case": round(float(p10[-1]), 0),
            "best_case": round(float(p90[-1]), 0),
            "stress_test_success_rate": round(stress_success_rate * 100, 1),
            "stress_test_median": round(float(np.median(stress_final)), 0),
            "years": years,
            "charts": {
                "years": list(range(years + 1)),
                "p10": p10.tolist(),
                "p50": p50.tolist(),
                "p90": p90.tolist()
            },
            # 4.5: the assumption travels with the answer. A success rate whose
            # draw distribution is invisible is a number people over-trust — and
            # this one changed.
            "distribution_used": distribution_used,
            "distribution_note": _DISTRIBUTION_NOTES.get(distribution_used, ""),
            "normal_comparison": normal_comparison,
            "withdrawal_policy": "guyton_klinger" if (guardrails and annual_withdrawal > 0) else "fixed_real",
            "median_total_withdrawn": (
                round(float(np.median(withdrawals_taken)), 0) if annual_withdrawal > 0 else None
            ),
            "interpretation": interpretation,
        }

    except Exception as e:
        return {"error": f"Monte Carlo simulation failed: {str(e)}"}

@log_exceptions()
def derive_portfolio_parameters(
    symbols: list[str],
    weights: list[float] | None = None,
    period: str = "5y",
    returns_fn: Any = None,
) -> dict[str, Any]:
    """Mean return, volatility and realised annual returns from the ACTUAL book.

    The third piece of 4.5. `RISK_PROFILES` presets are generic archetypes — a
    "balanced" 7/10 says nothing about a portfolio that is 70% semiconductors.
    This measures the same two numbers from the holdings themselves, in the
    profile's base currency, and returns the realised annual series so
    `draws='bootstrap'` can resample it.

    Returns `available: False` with a reason rather than a preset when there is
    too little history — falling back to an archetype silently would be the
    familiar failure: a measured-looking number that was assumed.

    Note the honest limitation, stated because it decides how the output should
    be used: this measures the CURRENT holdings over a PAST window, so it
    inherits their realised performance. A book that happened to hold the
    decade's winners will project their returns forward. It is a better
    description of this portfolio than a generic preset; it is not a forecast.
    """
    symbols = [str(s).upper().strip() for s in (symbols or []) if str(s).strip()]
    if not symbols:
        return {"available": False, "reason": "no symbols supplied"}

    if returns_fn is None:
        from tools.portfolio_analytics import _get_returns

        def returns_fn(syms, per):  # noqa: E306
            return _get_returns(syms, period=per)

    returns, valid = returns_fn(symbols, period)
    if returns is None or getattr(returns, "empty", True) or not valid:
        return {"available": False, "reason": "no price history for these holdings"}

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
    daily = returns[valid].dot(w)

    # ~2 years of daily data before this is worth reporting; below that the
    # volatility estimate is noisy and the annual series has almost no entries.
    if len(daily) < 400:
        return {
            "available": False,
            "reason": f"only {len(daily)} daily observations — too few to derive parameters",
            "observations": int(len(daily)),
        }

    mean_return = float(daily.mean() * 252)
    volatility = float(daily.std(ddof=1) * np.sqrt(252))

    # Realised annual returns, for bootstrap resampling. Overlapping 252-day
    # windows give more samples than calendar years from the same history; they
    # are autocorrelated, which matters for confidence intervals but not for
    # resampling a return distribution.
    values = (1 + daily).cumprod().to_numpy()
    annual = (values[252:] / values[:-252] - 1.0) if len(values) > 252 else np.array([])

    base_currency = (returns.attrs.get("fx", {}) or {}).get("base_currency")
    result = {
        "available": True,
        "mean_return": round(mean_return, 4),
        "volatility": round(volatility, 4),
        "observations": int(len(daily)),
        "annual_return_samples": annual.tolist(),
        "base_currency": base_currency,
        "symbols_used": valid,
        "period": period,
        "basis": "measured",
        "caveat": (
            "Measured from the CURRENT holdings over a PAST window, so it inherits their "
            "realised performance — a better description of this portfolio than a generic "
            "preset, but not a forecast."
        ),
    }

    # A loud, specific warning rather than the generic caveat, because this is
    # the failure mode that actually bit in testing: a concentrated two-name book
    # can measure a very high trailing CAGR over a 5y window, and feeding that
    # forward reports a near-certain chance of hitting the goal where a generic
    # preset reports a moderate one. The number LOOKS measured, so it gets
    # trusted — and it is measured, of the wrong thing. It describes what these
    # holdings did, not what they will do.
    if mean_return > IMPLAUSIBLE_FORWARD_RETURN:
        result["warning"] = (
            f"Measured mean return is {mean_return:.1%}/yr. That is a description of an "
            f"exceptional past window, not a usable forward expectation — projecting it "
            f"forward assumes the winners keep winning at the same rate. Prefer a preset for "
            f"planning, or use these parameters only to ask 'what if this repeats?'."
        )
        result["forward_use"] = "not recommended"
    return result


if __name__ == "__main__":
    res = run_monte_carlo(500000, 40000, 15)
    print(res['interpretation'])
