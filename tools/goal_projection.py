"""Goal projection — Advisor Roadmap 4.5 slice 4 / the goal-tracking panel.

The instrument that answers "are we on track?". Without it a decade-long plan
flies on one-year instruments: the dashboard shows today's value and today's
movers, and nothing anywhere says whether today is consistent with the goal.

Three numbers, and the discipline is in what this module REFUSES to produce:

  - **projected bands** — 4.5's Monte Carlo (p10/p50/p90) plus its goal-funded
    `goal_success_rate`, run off the stored goal and the stored contribution.
  - **required return** — the constant annual return that lands exactly on the
    target given today's value and the planned contribution stream. Solved
    numerically rather than as a naive (target/current)^(1/n)-1, because with a
    material contribution stream that closed form is wrong in the optimistic
    direction: it credits the returns with growth the deposits paid for.
  - **realized return** — WITHHELD, with a reason, until it can be computed
    honestly. See `realized_return_status`.

Nothing here supplies a target, a horizon, a contribution, or a return the user
did not state. An unset input produces `available: false` and a reason — never a
placeholder that renders as a real projection. That is the same contract as
`risk_constraints` and `get_financial_goal`, and it exists because a fabricated
number on this panel would be the most consequential kind: it is the one a
decade of decisions gets anchored to.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

# A realized return computed over less than this much history is noise dressed
# as a trend — annualizing a few weeks swings wildly and reads as authoritative.
MIN_HISTORY_DAYS_FOR_REALIZED = 365

# risk_tolerance (a user-facing word) -> monte_carlo.RISK_PROFILES key. The
# presets themselves are generic archetypes and safe to ship (see monte_carlo);
# what must never be silent is WHICH one was used, so every projection reports
# the resolved profile and its parameters back to the caller.
_RISK_TOLERANCE_TO_PROFILE = {
    "conservative": "conservative",
    "moderate": "balanced",
    "balanced": "balanced",
    "aggressive": "aggressive",
}


def resolve_risk_profile(risk_tolerance: str | None) -> str | None:
    """Map a stated risk tolerance onto a Monte Carlo preset, or None."""
    if not risk_tolerance:
        return None
    return _RISK_TOLERANCE_TO_PROFILE.get(str(risk_tolerance).strip().lower())


def required_annual_return(
    current_value: float,
    target: float,
    years: int,
    annual_contribution: float = 0.0,
    inflation_rate: float = 0.025,
) -> float | None:
    """The constant annual return that lands exactly on `target`.

    Solves for r in::

        current*(1+r)^n + SUM_t contribution*(1+infl)^(t-1) * (1+r)^(n-t) = target

    by bisection. The future value is strictly increasing in r for non-negative
    inputs, so bisection is both safe and exact to tolerance — and unlike the
    closed-form (target/current)^(1/n)-1 it does not hand the contributions'
    growth to the return. With ~$65K/yr on a ~$680K book that difference is not
    academic; the closed form understates required return by several points and
    the panel would read "on track" while the plan was not.

    Contributions inflate at `inflation_rate` to match `run_monte_carlo`, so the
    two halves of the panel describe the same cash-flow stream.

    Returns None when the question is unanswerable: no horizon, no target, or no
    way to reach it from nothing (zero starting value AND zero contribution).
    """
    if years <= 0 or target <= 0:
        return None
    if current_value <= 0 and annual_contribution <= 0:
        return None

    def future_value(rate: float) -> float:
        total = current_value * (1 + rate) ** years
        for t in range(1, years + 1):
            flow = annual_contribution * ((1 + inflation_rate) ** (t - 1))
            total += flow * (1 + rate) ** (years - t)
        return total

    low, high = -0.95, 1.0
    # If even a 100%/yr return misses the target, the goal is not reachable at
    # this contribution rate; say so rather than returning the search boundary
    # as though it were an answer.
    if future_value(high) < target:
        return None
    if future_value(low) > target:
        # Already overfunded — the required return is below the search floor.
        return low

    for _ in range(200):
        mid = (low + high) / 2
        if future_value(mid) < target:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 4)


def _history_span_days() -> int | None:
    """Days covered by this profile's portfolio history, or None if unreadable."""
    try:
        import pandas as pd

        from tools.user_profile import get_data_path

        for filename in ("portfolio_history.csv", "demo_portfolio_history.csv"):
            path = get_data_path(filename)
            if not os.path.exists(path):
                continue
            frame = pd.read_csv(path)
            if frame.empty or "date" not in frame.columns:
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
            if len(dates) < 2:
                continue
            return int((dates.max() - dates.min()).days)
    except Exception as e:  # noqa: BLE001 — a missing/odd history must not break the panel
        logger.debug(f"portfolio history span unavailable: {e}")
    return None


def realized_return_status() -> dict[str, Any]:
    """Why realized return is not being shown yet.

    Deliberately a STATUS, not a number. Two independent blockers, and both must
    clear before this can honestly be compared against `required_annual_return`:

      1. *Not enough history.* Annualizing a sub-year sample produces a figure
         that swings enormously and reads as authoritative.
      2. *Not flow-adjusted.* A raw value-to-value return over a period with
         large contributions is mostly the contributions. That is exactly the
         "~$65K/yr masquerading as performance" that roadmap 4.10 exists to fix
         with time-weighted return; until TWR lands, any number here would
         flatter the portfolio by roughly the deposit rate.

    Showing a wrong realized CAGR beside a correct required CAGR is worse than
    showing neither, because the comparison is the whole point of the panel.
    """
    span = _history_span_days()
    blockers = []
    if span is None:
        blockers.append("no portfolio history recorded yet")
    elif span < MIN_HISTORY_DAYS_FOR_REALIZED:
        blockers.append(
            f"only {span} days of history (need {MIN_HISTORY_DAYS_FOR_REALIZED})"
        )
    blockers.append(
        "not flow-adjusted — contributions would count as performance until "
        "roadmap 4.10's time-weighted return lands"
    )
    return {"available": False, "history_days": span, "blockers": blockers}


@log_exceptions()
def build_goal_projection(
    current_value: float | None = None,
    goal: dict[str, Any] | None = None,
    risk_tolerance: str | None = None,
    num_simulations: int = 5000,
) -> dict[str, Any]:
    """Assemble the goal panel payload. Every input is injectable for tests.

    Returns `{"available": False, "reason": ...}` whenever a required input is
    missing, naming WHICH one so the UI can prompt for exactly that instead of
    rendering an empty chart.
    """
    from tools.memory import get_financial_goal, get_profile_base_currency, load_memory
    from tools.monte_carlo import RISK_PROFILES, run_monte_carlo

    profile = load_memory().get("user_profile") or {}
    if goal is None:
        goal = get_financial_goal()
    if risk_tolerance is None:
        risk_tolerance = profile.get("risk_tolerance")

    if not goal:
        return {
            "available": False,
            "reason": "no wealth goal set",
            "missing": ["target_low", "horizon_years", "annual_contribution"],
        }

    missing = [
        key for key in ("horizon_years", "annual_contribution")
        if not goal.get(key)
    ]
    if not (goal.get("target_low") or goal.get("target_high")):
        missing.insert(0, "target_low")
    if missing:
        return {
            "available": False,
            "reason": f"goal is incomplete: {', '.join(missing)} not set",
            "missing": missing,
            "goal": goal,
        }

    if current_value is None:
        from tools.portfolio_csv import get_portfolio_summary
        summary = get_portfolio_summary() or {}
        current_value = summary.get("total_value_base")

    if not current_value or current_value <= 0:
        return {
            "available": False,
            "reason": "portfolio value unavailable",
            "missing": ["current_value"],
            "goal": goal,
        }

    years = int(goal["horizon_years"])
    contribution = float(goal["annual_contribution"])
    currency = goal.get("currency") or get_profile_base_currency(profile)

    profile_key = resolve_risk_profile(risk_tolerance)
    preset = RISK_PROFILES.get(profile_key or "", {})

    # The bands are driven by the LOW target: goal_success_rate answers "did we
    # clear the bar", and the low end is the bar. The high target is reported
    # through required_return alongside it so the stretch case is visible too.
    primary_target = float(goal.get("target_low") or goal.get("target_high"))

    simulation = run_monte_carlo(
        current_portfolio_value=float(current_value),
        annual_contribution=contribution,
        years=years,
        num_simulations=num_simulations,
        risk_profile=profile_key,
        goal_target=primary_target,
    )
    if "error" in simulation:
        return {"available": False, "reason": simulation["error"], "goal": goal}

    required = {}
    for label, key in (("low", "target_low"), ("high", "target_high")):
        target = goal.get(key)
        if target:
            required[label] = required_annual_return(
                current_value=float(current_value),
                target=float(target),
                years=years,
                annual_contribution=contribution,
            )

    return {
        "available": True,
        "currency": currency,
        "current_value": round(float(current_value), 2),
        "goal": goal,
        "horizon_years": years,
        "annual_contribution": contribution,
        "required_annual_return": required,
        "realized_annual_return": realized_return_status(),
        "goal_success_rate": simulation.get("goal_success_rate"),
        "goal_target_nominal": simulation.get("goal_target_nominal"),
        "success_rate": simulation.get("success_rate"),
        "median_result": simulation.get("median_result"),
        "worst_case": simulation.get("worst_case"),
        "best_case": simulation.get("best_case"),
        "stress_test_success_rate": simulation.get("stress_test_success_rate"),
        "bands": simulation.get("charts"),
        # Assumptions are part of the answer, not footnotes: a projection whose
        # return assumption is invisible is a number people over-trust.
        "assumptions": {
            "risk_profile": profile_key,
            "risk_tolerance": risk_tolerance,
            "mean_return": preset.get("mean_return"),
            "volatility": preset.get("volatility"),
            "inflation_rate": 0.025,
            "num_simulations": num_simulations,
            "contributions_inflated": True,
        },
    }


# ---------------------------------------------------------------------------
# Contribution sensitivity — the last piece of the goal panel's original scope
# ---------------------------------------------------------------------------
# AUTHORED CONSTANTS (2.7). Neither is measured; both are the shape of the
# question rather than an answer to it.
#
# The deltas to test, in base-currency units per year. Asymmetric on purpose:
# the downside cases exist because "what if we have to cut back" is the same
# discipline question asked from the other side, and a panel that only shows
# upside reads as an advertisement for saving more.
CONTRIBUTION_DELTAS: tuple[float, ...] = (-10_000, -5_000, 0, 5_000, 10_000, 25_000)

# One seed, shared by every scenario. This is the load-bearing detail of the
# whole feature and it has precedent in `run_monte_carlo` itself, where the
# stress comparison is run at the SAME sample size and seed for the same reason.
# Monte Carlo error on a 5,000-path success rate is on the order of a percentage
# point; the effect of $5K/yr on a book this size can be smaller than that. Run
# independently, adjacent scenarios would routinely show MORE contribution
# producing a LOWER success rate, and the panel would be reporting its own RNG.
# Sharing the seed makes every scenario face the identical sequence of return
# paths, so the difference between two rows is attributable to the contribution
# and nothing else.
SENSITIVITY_SEED = 20260729


@log_exceptions()
def build_contribution_sensitivity(
    deltas: tuple[float, ...] | list[float] | None = None,
    current_value: float | None = None,
    goal: dict[str, Any] | None = None,
    risk_tolerance: str | None = None,
    num_simulations: int = 5000,
    seed: int = SENSITIVITY_SEED,
) -> dict[str, Any]:
    """What changing the annual contribution does to the goal.

    The discipline instrument the goal panel was specified with and shipped
    without. The panel answers "are we on track"; this answers the only question
    a person can actually act on when the answer is no — and it answers it in the
    currency this plan is actually bound by. The stated binding constraints here
    are discipline, cash deployment and tax location, NOT alpha, so the headline
    each row carries is the REQUIRED RETURN: an extra $10K/yr does not just raise
    the odds, it lowers the return the plan needs in order to work. "Save more" and
    "find 3 more points of alpha" are the same row read two ways, and only one of
    them is under anyone's control.

    Every scenario shares one seed — see SENSITIVITY_SEED. Row-to-row differences
    are therefore real; the ABSOLUTE success rate in any row still carries the
    usual Monte Carlo error, and `comparable` says exactly that.

    Unavailable inputs produce `available: false` with the same `missing` list
    `build_goal_projection` returns, because this reads the identical inputs and
    a caller should not have to learn two contracts.
    """
    from tools.memory import get_financial_goal, get_profile_base_currency, load_memory
    from tools.monte_carlo import RISK_PROFILES, run_monte_carlo

    profile = load_memory().get("user_profile") or {}
    if goal is None:
        goal = get_financial_goal()
    if risk_tolerance is None:
        risk_tolerance = profile.get("risk_tolerance")

    # Reuse the base projection purely as the input GATE, so "what is missing"
    # is answered in exactly one place. A second copy of this validation is how
    # the two surfaces end up disagreeing about whether the goal is set.
    base = build_goal_projection(
        current_value=current_value, goal=goal,
        risk_tolerance=risk_tolerance, num_simulations=num_simulations,
    )
    if not base.get("available"):
        return {k: base[k] for k in ("available", "reason", "missing", "goal") if k in base}

    years = int(base["horizon_years"])
    base_contribution = float(base["annual_contribution"])
    value = float(base["current_value"])
    currency = base["currency"]
    target = float(goal.get("target_low") or goal.get("target_high"))

    profile_key = resolve_risk_profile(risk_tolerance)
    preset = RISK_PROFILES.get(profile_key or "", {})

    scenarios: list[dict[str, Any]] = []
    for delta in (CONTRIBUTION_DELTAS if deltas is None else deltas):
        raw = base_contribution + float(delta)
        # A negative contribution is a withdrawal, which this model does not
        # represent — the Monte Carlo inflates contributions as deposits. Clamp
        # and SAY SO rather than silently simulating a stream that means
        # something else.
        contribution = max(0.0, raw)
        clamped = raw < 0

        sim = run_monte_carlo(
            current_portfolio_value=value,
            annual_contribution=contribution,
            years=years,
            num_simulations=num_simulations,
            risk_profile=profile_key,
            goal_target=target,
            seed=seed,
        )
        if "error" in sim:
            scenarios.append({
                "delta": float(delta), "annual_contribution": contribution,
                "error": sim["error"], "clamped_to_zero": clamped,
            })
            continue

        scenarios.append({
            "delta": float(delta),
            "annual_contribution": round(contribution, 2),
            "clamped_to_zero": clamped,
            "goal_success_rate": sim.get("goal_success_rate"),
            "median_result": sim.get("median_result"),
            "required_annual_return": required_annual_return(
                current_value=value, target=target, years=years,
                annual_contribution=contribution,
            ),
            "is_current": float(delta) == 0.0,
        })

    current = next((s for s in scenarios if s.get("is_current")), None)
    for s in scenarios:
        # Deltas against the current plan, which is the only comparison a shared
        # seed licenses. Absent when either side failed to simulate — a missing
        # difference is better than one computed against nothing.
        s["success_rate_delta_pp"] = _pp_delta(s, current, "goal_success_rate")
        s["required_return_delta_pp"] = _pp_delta(s, current, "required_annual_return", scale=100.0)

    return {
        "available": True,
        "currency": currency,
        "current_value": value,
        "horizon_years": years,
        "base_annual_contribution": base_contribution,
        "target": target,
        "scenarios": scenarios,
        "comparable": (
            "Every scenario ran on the same simulated return paths (one shared "
            "seed), so the DIFFERENCES between rows are attributable to the "
            "contribution alone. Each row's absolute success rate still carries "
            "the usual Monte Carlo error of roughly a percentage point."
        ),
        "assumptions": {
            "risk_profile": profile_key,
            "risk_tolerance": risk_tolerance,
            "mean_return": preset.get("mean_return"),
            "volatility": preset.get("volatility"),
            "inflation_rate": 0.025,
            "num_simulations": num_simulations,
            "seed": seed,
            "contributions_inflated": True,
        },
        "note": (
            "Rows show only the contribution levels listed. Nothing here is "
            "interpolated between them or extrapolated beyond them, and no level "
            "is being recommended — this reports what each one implies."
        ),
    }


def _pp_delta(scenario: dict[str, Any], current: dict[str, Any] | None,
              key: str, scale: float = 1.0) -> float | None:
    """Percentage-point difference from the current plan, or None if unknowable."""
    if not current:
        return None
    a, b = scenario.get(key), current.get(key)
    if a is None or b is None:
        return None
    try:
        return round((float(a) - float(b)) * scale, 4)
    except (TypeError, ValueError):
        return None
