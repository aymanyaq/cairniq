"""Monte Carlo remainder — Advisor Roadmap 4.5 (fat tails, live params, Guyton-Klinger).

The tests are organised around what MEASUREMENT said, not around what the item
assumed going in. 4.5 was justified by "normal draws understate tail risk, so the
goal number is optimistic"; that turned out not to hold at annual frequency, and
the guardrails — an afterthought in the original scoping — turned out to be the
whole payload. Both facts are pinned here so neither gets re-assumed later.
"""
import numpy as np
import pytest

from tools.monte_carlo import (
    DRAWS_BOOTSTRAP,
    DRAWS_NORMAL,
    DRAWS_STUDENT_T,
    IMPLAUSIBLE_FORWARD_RETURN,
    derive_portfolio_parameters,
    run_monte_carlo,
)

ACCUMULATION = dict(current_portfolio_value=1_000_000, annual_contribution=50_000,
                    years=10, num_simulations=4000, seed=3)
RETIREMENT = dict(current_portfolio_value=1_500_000, annual_contribution=0, years=30,
                  risk_profile="balanced", num_simulations=4000, seed=11)


# ---------------------------------------------------------------------------
# Guardrails — the part that measurably mattered
# ---------------------------------------------------------------------------

def test_guardrails_dramatically_change_survival_at_a_stressed_withdrawal():
    """The fixed-withdrawal model fails in the one way real retirees do not: it
    keeps spending the same real amount into a 40% drawdown. Measured over 30
    years on $1.5M, a 7% initial withdrawal goes from 23.6% to 98.0% survival
    once spending responds."""
    fixed = run_monte_carlo(annual_withdrawal=105_000, guardrails=False, **RETIREMENT)
    guarded = run_monte_carlo(annual_withdrawal=105_000, guardrails=True, **RETIREMENT)

    assert guarded["success_rate"] - fixed["success_rate"] > 40
    assert guarded["withdrawal_policy"] == "guyton_klinger"
    assert fixed["withdrawal_policy"] == "fixed_real"


def test_the_survival_gain_is_bought_with_spending_and_that_is_reported():
    """Presenting the survival number without the spending it cost would be the
    flattering half of a two-sided result."""
    fixed = run_monte_carlo(annual_withdrawal=105_000, guardrails=False, **RETIREMENT)
    guarded = run_monte_carlo(annual_withdrawal=105_000, guardrails=True, **RETIREMENT)

    assert guarded["median_total_withdrawn"] < fixed["median_total_withdrawn"]
    assert fixed["median_total_withdrawn"] is not None


def test_at_a_safe_withdrawal_the_prosperity_rule_lets_spending_rise():
    """Guardrails are not merely a ratchet down. At 4% the trade is close to
    free — survival rises AND slightly more is withdrawn, because good paths are
    allowed to spend more."""
    fixed = run_monte_carlo(annual_withdrawal=60_000, guardrails=False, **RETIREMENT)
    guarded = run_monte_carlo(annual_withdrawal=60_000, guardrails=True, **RETIREMENT)

    assert guarded["success_rate"] > fixed["success_rate"]
    assert guarded["median_total_withdrawn"] >= fixed["median_total_withdrawn"]


def test_guardrails_are_inert_without_withdrawals():
    """An accumulation plan has nothing to guard. The flag must not silently
    alter a projection it does not apply to."""
    plain = run_monte_carlo(risk_profile="balanced", guardrails=False, **ACCUMULATION)
    flagged = run_monte_carlo(risk_profile="balanced", guardrails=True, **ACCUMULATION)

    assert plain["median_result"] == flagged["median_result"]
    assert flagged["withdrawal_policy"] == "fixed_real"


# ---------------------------------------------------------------------------
# Fat tails — the assumption that measurement did NOT support
# ---------------------------------------------------------------------------

def test_fat_tails_do_not_materially_change_long_horizon_outcomes():
    """Pinned deliberately, against the item's original justification.

    4.5 was scoped on "normal draws understate tail risk, so goal_success_rate
    is optimistic". Measured: on the live goal shape the two agree to 0.4pp, and
    across 30-year withdrawal scenarios the gap stays under ~1.5pp and is not
    consistently negative — compounding many annual draws averages the tails
    away. If this test ever starts failing loudly in one direction, the claim
    can be revisited with evidence; until then nobody should cite fat tails as
    having de-risked the estimate.
    """
    normal = run_monte_carlo(risk_profile="balanced", draws=DRAWS_NORMAL,
                             goal_target=2_000_000, **ACCUMULATION)
    fat = run_monte_carlo(risk_profile="balanced", draws=DRAWS_STUDENT_T,
                          goal_target=2_000_000, **ACCUMULATION)

    assert abs(fat["goal_success_rate"] - normal["goal_success_rate"]) < 5.0


def test_student_t_preserves_the_stated_volatility():
    """Only tail thickness changes, not the average outcome — otherwise the
    comparison above would be measuring "lower assumed returns" instead."""
    normal = run_monte_carlo(mean_return=0.07, volatility=0.15, draws=DRAWS_NORMAL,
                             **ACCUMULATION)
    fat = run_monte_carlo(mean_return=0.07, volatility=0.15, draws=DRAWS_STUDENT_T,
                          **ACCUMULATION)

    assert fat["median_result"] == pytest.approx(normal["median_result"], rel=0.06)


def test_student_t_actually_produces_fatter_tails():
    """The mechanism must work even though its effect on the headline is small."""
    rng = np.random.default_rng(0)
    from tools.monte_carlo import _draw_returns

    normal, _ = _draw_returns(0.07, 0.15, (1, 200_000), DRAWS_NORMAL, rng)
    fat, _ = _draw_returns(0.07, 0.15, (1, 200_000), DRAWS_STUDENT_T, rng)

    # More mass beyond 3 sigma, similar standard deviation.
    assert (np.abs(fat - 0.07) > 0.45).mean() > (np.abs(normal - 0.07) > 0.45).mean()
    assert fat.std() == pytest.approx(normal.std(), rel=0.1)


def test_every_result_names_the_distribution_it_used():
    """The default CHANGED in 4.5, and it moves a number already on the
    dashboard. An invisible assumption is one people over-trust."""
    result = run_monte_carlo(risk_profile="balanced", **ACCUMULATION)

    assert result["distribution_used"] == DRAWS_STUDENT_T
    assert "MEASURED EFFECT AT ANNUAL FREQUENCY IS SMALL" in result["distribution_note"]


def test_a_non_normal_run_reports_what_the_old_assumption_would_have_said():
    result = run_monte_carlo(risk_profile="balanced", goal_target=2_000_000, **ACCUMULATION)

    assert result["normal_comparison"] is not None
    assert result["normal_comparison"]["goal_success_rate"] is not None


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_resamples_only_years_that_actually_happened():
    history = [0.10, -0.20, 0.30]
    result = run_monte_carlo(draws=DRAWS_BOOTSTRAP, return_history=history * 4,
                             risk_profile="balanced", **ACCUMULATION)

    assert result["distribution_used"] == DRAWS_BOOTSTRAP


def test_bootstrap_without_enough_history_falls_back_to_the_FATTER_draw():
    """Falling back to the normal draw would flatter the result in exactly the
    case where we know least. The fallback must also be visible."""
    result = run_monte_carlo(draws=DRAWS_BOOTSTRAP, return_history=[0.05, 0.06],
                             risk_profile="balanced", **ACCUMULATION)

    assert result["distribution_used"] == DRAWS_STUDENT_T


# ---------------------------------------------------------------------------
# Parameters from the actual portfolio — and the trap that came with them
# ---------------------------------------------------------------------------

def _fake_returns(daily_mean, daily_sd, n=1200, seed=1):
    """A return series whose sample mean is EXACTLY `daily_mean`.

    The draws are de-meaned and re-centred rather than trusted to land on the
    requested mean. Without that the fixture's sample mean drifts by up to ~2
    standard errors (0.012/sqrt(1200) ~ 3.5bp/day here), which is enough to move
    an annualised figure by 15pp and put a threshold test on the wrong side of
    its boundary — that happened twice while writing this file, and both times
    the guard under test was working correctly.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, daily_sd, n)
    draws = draws - draws.mean() + daily_mean
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    frame = pd.DataFrame({"A": draws}, index=idx)
    frame.attrs["fx"] = {"base_currency": "CAD"}
    return lambda syms, per: (frame, ["A"])


def test_derived_parameters_are_measured_and_carry_the_annual_series():
    params = derive_portfolio_parameters(["A"], returns_fn=_fake_returns(0.0003, 0.01))

    assert params["available"] is True
    assert params["basis"] == "measured"
    assert len(params["annual_return_samples"]) > 0
    assert params["base_currency"] == "CAD"


def test_an_exceptional_measured_return_is_flagged_as_UNUSABLE_going_forward():
    """The trap this hit on real data: a book concentrated in two mega-cap tech
    names measured 33.6%/yr over 5 years, and feeding that forward reported a
    100% chance of hitting the goal against 71% from a generic preset. The
    number looks measured — and it is, of the wrong thing. It describes what
    those holdings DID."""
    # 0.0011/day = exactly 27.7%/yr, well clear of the 15% threshold.
    params = derive_portfolio_parameters(["A"], returns_fn=_fake_returns(0.0011, 0.012))

    assert params["mean_return"] > IMPLAUSIBLE_FORWARD_RETURN
    assert params["forward_use"] == "not recommended"
    assert "winners keep winning" in params["warning"]


def test_an_ordinary_measured_return_is_not_flagged():
    params = derive_portfolio_parameters(["A"], returns_fn=_fake_returns(0.0003, 0.01))

    assert params["mean_return"] < IMPLAUSIBLE_FORWARD_RETURN
    assert "forward_use" not in params


def test_too_little_history_refuses_rather_than_falling_back_to_a_preset():
    """Silently substituting an archetype would produce a measured-looking
    number that was assumed."""
    params = derive_portfolio_parameters(["A"], returns_fn=_fake_returns(0.0003, 0.01, n=100))

    assert params["available"] is False
    assert "too few" in params["reason"]


def test_no_history_at_all_is_unavailable_not_a_default():
    import pandas as pd
    params = derive_portfolio_parameters(["A"], returns_fn=lambda s, p: (pd.DataFrame(), []))

    assert params["available"] is False


# ---------------------------------------------------------------------------
# Backward compatibility — 4.5's earlier slices must still hold
# ---------------------------------------------------------------------------

def test_goal_and_non_depletion_rates_remain_distinct():
    """The metric split 4.5 shipped in slices 1-3 must survive this change."""
    result = run_monte_carlo(risk_profile="conservative", goal_target=5_000_000,
                             **ACCUMULATION)

    assert result["success_rate"] == 100.0
    assert result["goal_success_rate"] < result["success_rate"]


def test_seeding_makes_a_run_reproducible():
    a = run_monte_carlo(risk_profile="balanced", **ACCUMULATION)
    b = run_monte_carlo(risk_profile="balanced", **ACCUMULATION)

    assert a["median_result"] == b["median_result"]
