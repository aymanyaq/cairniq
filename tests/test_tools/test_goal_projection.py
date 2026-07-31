"""Goal projection — Advisor Roadmap 4.5 slice 4 (the goal-tracking panel).

The panel answers "are we on track?", so the tests that matter are not "does it
return a dict". They are: does it REFUSE to answer when it has not been told the
goal, and is the number it gives when it does answer the right one. A goal panel
that renders a confident projection off assumed inputs is the most damaging
version of this codebase's recurring failure mode — a decade of decisions would
anchor to it.
"""
import pytest

from tools.goal_projection import (
    build_goal_projection,
    required_annual_return,
    resolve_risk_profile,
)

# ---------------------------------------------------------------------------
# Required return — the number the closed form gets wrong
# ---------------------------------------------------------------------------

def test_required_return_with_no_contributions_matches_the_closed_form():
    """The one case where (target/current)^(1/n)-1 IS right, so it pins the
    solver to a known answer: doubling in 10 years needs ~7.18%/yr."""
    rate = required_annual_return(
        current_value=1_000_000, target=2_000_000, years=10, annual_contribution=0
    )

    assert rate == pytest.approx(0.0718, abs=0.001)


def test_contributions_lower_the_required_return():
    """Money you add is not return you earned. The whole reason for solving
    numerically: a plan funded by contributions needs LESS from the market."""
    without = required_annual_return(1_500_000, 3_000_000, 10, annual_contribution=0)
    with_flows = required_annual_return(1_500_000, 3_000_000, 10, annual_contribution=65_000)

    assert with_flows < without


def test_the_closed_form_would_overstate_reachability_at_a_real_contribution_rate():
    """Why this is not academic. The naive formula ignores the contribution
    stream entirely, so it reports a HIGHER required return than the truth --
    and a panel comparing that against realized performance would read
    'behind' on a plan that is actually funded. The gap here is several points,
    not a rounding difference."""
    naive = (3_000_000 / 1_500_000) ** (1 / 10) - 1          # ~7.2%
    actual = required_annual_return(1_500_000, 3_000_000, 10, annual_contribution=65_000)

    assert naive - actual > 0.02


def test_an_overfunded_goal_does_not_report_a_fake_positive_requirement():
    """Already past the target: the honest answer is 'you need nothing from the
    market', not a fabricated positive hurdle."""
    rate = required_annual_return(5_000_000, 3_000_000, 10, annual_contribution=65_000)

    assert rate is not None and rate < 0


def test_an_unreachable_goal_returns_none_rather_than_the_search_boundary():
    """A goal that misses even at 100%/yr is unanswerable at this contribution
    rate. Returning the bisection bound would present a search artifact as a
    finding."""
    assert required_annual_return(1_000, 5_000_000, 2, annual_contribution=0) is None


@pytest.mark.parametrize("current,contribution", [(0, 0), (-5, 0)])
def test_nothing_to_grow_and_nothing_added_is_unanswerable(current, contribution):
    assert required_annual_return(current, 3_000_000, 10, contribution) is None


def test_zero_horizon_is_unanswerable():
    assert required_annual_return(1_000_000, 2_000_000, 0, 65_000) is None


# ---------------------------------------------------------------------------
# Refusing to project on inputs the user never gave
# ---------------------------------------------------------------------------

def test_no_goal_means_unavailable_not_a_default_projection():
    """Same contract as risk_constraints and get_financial_goal: unset is
    MEANINGFUL. The panel must say it has no goal, never pick a plausible one."""
    result = build_goal_projection(current_value=1_500_000, goal=None)

    assert result["available"] is False
    assert "no wealth goal" in result["reason"]


def test_a_goal_with_no_contribution_names_the_missing_field():
    """The contribution is the single biggest driver of reachability, so
    assuming one would decide the answer. The UI needs to know WHICH field to
    prompt for, not just that something is wrong."""
    result = build_goal_projection(
        current_value=1_500_000,
        goal={"target_low": 3_000_000, "horizon_years": 10, "annual_contribution": None},
    )

    assert result["available"] is False
    assert result["missing"] == ["annual_contribution"]


def test_a_contribution_with_no_target_is_also_incomplete():
    result = build_goal_projection(
        current_value=1_500_000,
        goal={"target_low": None, "horizon_years": 10, "annual_contribution": 65_000},
    )

    assert result["available"] is False
    assert "target_low" in result["missing"]


def test_no_portfolio_value_is_unavailable_not_zero():
    """Projecting from an assumed 0 would render a real-looking chart of a
    portfolio that does not exist."""
    result = build_goal_projection(
        current_value=0,
        goal={"target_low": 3_000_000, "horizon_years": 10, "annual_contribution": 65_000},
    )

    assert result["available"] is False
    assert result["missing"] == ["current_value"]


# ---------------------------------------------------------------------------
# A complete goal
# ---------------------------------------------------------------------------

@pytest.fixture
def projection():
    return build_goal_projection(
        current_value=1_500_000,
        goal={
            "target_low": 3_000_000,
            "target_high": 5_000_000,
            "horizon_years": 10,
            "annual_contribution": 65_000,
            "currency": "CAD",
        },
        risk_tolerance="Aggressive",
        num_simulations=2000,
    )


def test_a_complete_goal_projects_bands_and_a_goal_funded_probability(projection):
    assert projection["available"] is True
    assert projection["goal_success_rate"] is not None
    assert len(projection["bands"]["p50"]) == 11          # years 0..10 inclusive
    assert projection["bands"]["p10"][0] == 1_500_000     # every path starts from today


def test_both_targets_get_their_own_required_return(projection):
    """The stretch target is not decoration — seeing that $5M needs materially
    more than $3M is the point of showing a range."""
    assert projection["required_annual_return"]["high"] > projection["required_annual_return"]["low"]


def test_the_return_assumption_is_reported_not_hidden(projection):
    """A projection whose assumed return is invisible is a number people
    over-trust. Which preset ran must travel with the answer."""
    assert projection["assumptions"]["risk_profile"] == "aggressive"
    assert projection["assumptions"]["mean_return"] == 0.10
    assert projection["assumptions"]["contributions_inflated"] is True


def test_goal_success_rate_is_never_conflated_with_non_depletion(projection):
    """4.5 shipped these as two DISTINCT metrics and the roadmap is explicit
    that they must not be confused: success_rate is 'did not go to zero',
    goal_success_rate is 'funded the goal'. Both must be present and separate."""
    assert projection["success_rate"] is not None
    assert projection["goal_success_rate"] != projection["success_rate"]


def test_the_target_is_inflated_to_the_horizon_before_being_scored(projection):
    """The stored target is in today's terms; terminal values are nominal.
    Comparing them directly would overstate success every time."""
    assert projection["goal_target_nominal"] > 3_000_000


# ---------------------------------------------------------------------------
# Realized return — withheld on purpose
# ---------------------------------------------------------------------------

def test_realized_return_is_withheld_with_reasons_never_estimated(projection):
    """Showing a wrong realized CAGR beside a correct required CAGR is worse
    than showing neither, because the COMPARISON is the panel. Both blockers
    must be stated: too little history, and not flow-adjusted until 4.10's TWR
    lands (contributions would otherwise count as performance)."""
    realized = projection["realized_annual_return"]

    assert realized["available"] is False
    assert any("flow-adjusted" in b for b in realized["blockers"])


def test_risk_tolerance_maps_onto_a_preset_and_unknown_words_do_not_guess():
    assert resolve_risk_profile("Moderate") == "balanced"
    assert resolve_risk_profile("aggressive") == "aggressive"
    assert resolve_risk_profile("YOLO") is None
    assert resolve_risk_profile(None) is None
