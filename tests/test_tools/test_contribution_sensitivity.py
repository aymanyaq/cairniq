"""
Contribution sensitivity — the last piece of the goal panel's original scope.

The panel answers "are we on track". This answers the only question a person can
act on when the answer is no, and it answers it in the currency this plan is
actually bound by: the stated binding constraints are discipline, cash deployment
and tax location, NOT alpha. So each row's headline is the REQUIRED RETURN — an
extra $10K/yr does not merely raise the odds, it lowers the return the plan needs.

**The shared seed is the feature, and it is what these tests mostly guard.**
Monte Carlo error on a 5,000-path success rate runs to about a percentage point.
The effect of $5K/yr on a book this size can be smaller than that. Run
independently, adjacent scenarios would routinely show MORE contribution
producing a LOWER success rate — the panel would be reporting its own RNG at the
user, in a view whose entire purpose is to make a small difference legible. One
seed across all scenarios makes every row face the identical return paths, so a
row-to-row difference is the contribution and nothing else.

The second guard is the contract: unavailable inputs must produce the SAME
`available: false` / `missing` shape `build_goal_projection` returns, because
both read the identical stored goal and a caller should not learn two shapes.
"""
import pytest

from tools.goal_projection import (
    CONTRIBUTION_DELTAS,
    SENSITIVITY_SEED,
    build_contribution_sensitivity,
)

GOAL = {
    "target_low": 3_000_000, "target_high": 5_000_000,
    "horizon_years": 10, "annual_contribution": 65_000, "currency": "CAD",
}
VALUE = 1_500_000.0

# Enough paths for the run to be stable, few enough to keep the suite quick.
SIMS = 800


def _build(**kw):
    params = dict(current_value=VALUE, goal=GOAL, risk_tolerance="balanced",
                  num_simulations=SIMS)
    params.update(kw)
    return build_contribution_sensitivity(**params)


def _by_delta(result):
    return {s["delta"]: s for s in result["scenarios"]}


# ---------------------------------------------------------------------------
# The shared seed
# ---------------------------------------------------------------------------
def test_more_contribution_never_lowers_the_success_rate():
    """The property that only holds because every scenario shares a seed. Run
    independently these rows would cross, and a panel built to make a small
    difference legible would be showing simulation noise instead."""
    scenarios = [s for s in _build()["scenarios"] if "goal_success_rate" in s]
    scenarios.sort(key=lambda s: s["annual_contribution"])
    rates = [s["goal_success_rate"] for s in scenarios]
    assert rates == sorted(rates), f"success rate is not monotonic in contribution: {rates}"


def test_more_contribution_never_raises_the_required_return():
    """Deterministic — required_annual_return is solved, not simulated — so this
    would hold without the seed. It is the row a user should act on."""
    scenarios = [s for s in _build()["scenarios"] if s.get("required_annual_return") is not None]
    scenarios.sort(key=lambda s: s["annual_contribution"])
    required = [s["required_annual_return"] for s in scenarios]
    assert required == sorted(required, reverse=True), required


def test_the_same_inputs_give_the_same_answer_twice():
    """A sensitivity view that moved between refreshes would be unusable for the
    thing it exists for: comparing two options."""
    a, b = _build(), _build()
    assert [s.get("goal_success_rate") for s in a["scenarios"]] == \
           [s.get("goal_success_rate") for s in b["scenarios"]]


def test_the_seed_is_reported_so_a_reader_can_tell_runs_apart():
    result = _build()
    assert result["assumptions"]["seed"] == SENSITIVITY_SEED
    assert "same simulated return paths" in result["comparable"]
    assert "Monte Carlo error" in result["comparable"]


def test_a_different_seed_gives_a_different_but_still_monotonic_answer():
    """Guards against the monotonicity above being an accident of one seed."""
    scenarios = [s for s in _build(seed=999)["scenarios"] if "goal_success_rate" in s]
    scenarios.sort(key=lambda s: s["annual_contribution"])
    rates = [s["goal_success_rate"] for s in scenarios]
    assert rates == sorted(rates), rates


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
def test_no_goal_returns_the_same_shape_as_the_projection_does():
    result = build_contribution_sensitivity(current_value=VALUE, goal={},
                                            num_simulations=SIMS)
    assert result["available"] is False
    assert "no wealth goal set" in result["reason"]
    assert "target_low" in result["missing"]
    assert "scenarios" not in result


def test_an_incomplete_goal_names_the_missing_field():
    result = build_contribution_sensitivity(
        current_value=VALUE, goal={"target_low": 3_000_000, "horizon_years": 10},
        num_simulations=SIMS)
    assert result["available"] is False
    assert "annual_contribution" in result["missing"]


def test_no_portfolio_value_is_reported_not_assumed():
    """A projection against an invented starting value is the most consequential
    fabrication this panel could make."""
    result = build_contribution_sensitivity(current_value=0, goal=GOAL,
                                            num_simulations=SIMS)
    assert result["available"] is False
    assert result["missing"] == ["current_value"]


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------
def test_the_current_plan_is_present_and_marked():
    """Without a zero row there is nothing for the other rows to be a delta
    against, and the reader has to do the subtraction."""
    rows = _by_delta(_build())
    assert 0.0 in rows
    assert rows[0.0]["is_current"] is True
    assert rows[0.0]["annual_contribution"] == GOAL["annual_contribution"]
    assert rows[0.0]["success_rate_delta_pp"] == 0.0
    assert rows[0.0]["required_return_delta_pp"] == 0.0


def test_downside_scenarios_are_shown_too():
    """"What if we have to cut back" is the same discipline question from the
    other side; a panel showing only upside reads as an advertisement."""
    deltas = [s["delta"] for s in _build()["scenarios"]]
    assert any(d < 0 for d in deltas), deltas
    assert any(d > 0 for d in deltas), deltas


def test_a_negative_contribution_is_clamped_and_flagged():
    """The model inflates contributions as deposits; a negative stream would be
    simulating something else entirely."""
    result = _build(goal={**GOAL, "annual_contribution": 5_000},
                    deltas=(-10_000, 0))
    row = _by_delta(result)[-10_000.0]
    assert row["annual_contribution"] == 0.0
    assert row["clamped_to_zero"] is True
    # The zero row must not be flagged.
    assert _by_delta(result)[0.0]["clamped_to_zero"] is False


def test_deltas_are_measured_against_the_current_plan():
    rows = _by_delta(_build(deltas=(0, 25_000)))
    boost = rows[25_000.0]
    assert boost["success_rate_delta_pp"] == pytest.approx(
        boost["goal_success_rate"] - rows[0.0]["goal_success_rate"], abs=1e-6)
    # Required return falls, so its delta is negative — the actionable number.
    assert boost["required_return_delta_pp"] < 0


def test_required_return_deltas_are_in_percentage_points():
    """required_annual_return is a FRACTION and goal_success_rate is already a
    percentage; mixing the two units would understate the return delta 100x."""
    rows = _by_delta(_build(deltas=(0, 25_000)))
    raw = (rows[25_000.0]["required_annual_return"]
           - rows[0.0]["required_annual_return"])
    assert rows[25_000.0]["required_return_delta_pp"] == pytest.approx(raw * 100, abs=1e-6)


def test_a_delta_is_withheld_rather_than_computed_against_nothing():
    """No zero row means no baseline; a difference against nothing is worse than
    an absent one."""
    result = _build(deltas=(5_000, 10_000))
    assert all(s["success_rate_delta_pp"] is None for s in result["scenarios"])


def test_custom_deltas_are_honoured():
    rows = _by_delta(_build(deltas=(0, 12_345)))
    assert set(rows) == {0.0, 12_345.0}
    assert rows[12_345.0]["annual_contribution"] == GOAL["annual_contribution"] + 12_345


def test_the_defaults_bracket_the_current_plan():
    assert min(CONTRIBUTION_DELTAS) < 0 < max(CONTRIBUTION_DELTAS)
    assert 0 in CONTRIBUTION_DELTAS


# ---------------------------------------------------------------------------
# What it refuses to say
# ---------------------------------------------------------------------------
def test_nothing_is_recommended_and_nothing_is_interpolated():
    note = _build()["note"]
    assert "not interpolated" in note or "Nothing here is interpolated" in note
    assert "extrapolated" in note
    assert "no level is being recommended" in note.lower()


def test_the_assumptions_ride_with_the_answer():
    """A projection whose return assumption is invisible is over-trusted — the
    same rule build_goal_projection states for itself."""
    a = _build()["assumptions"]
    for key in ("risk_profile", "mean_return", "volatility", "inflation_rate",
                "num_simulations", "seed", "contributions_inflated"):
        assert key in a, key
    assert a["contributions_inflated"] is True


def test_the_endpoint_serves_it():
    from fastapi.testclient import TestClient

    from server import app

    res = TestClient(app).get("/api/contribution_sensitivity?simulations=500")
    assert res.status_code == 200
    body = res.json()
    assert "available" in body
    if body["available"]:
        assert body["scenarios"]
        assert "comparable" in body
    else:
        assert body["reason"]


def test_the_simulation_count_is_bounded():
    """This runs one simulation PER SCENARIO, so an unbounded query string costs
    six times what it would on the single-projection route."""
    from fastapi.testclient import TestClient

    from server import app

    res = TestClient(app).get("/api/contribution_sensitivity?simulations=999999")
    assert res.status_code == 200
    body = res.json()
    if body.get("available"):
        assert body["assumptions"]["num_simulations"] <= 10000


def test_the_agent_tool_is_registered():
    from agent.tool_registry import ALL_TOOLS

    assert "get_contribution_sensitivity" in {getattr(t, "name", "") for t in ALL_TOOLS}
