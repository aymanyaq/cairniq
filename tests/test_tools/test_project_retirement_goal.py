"""project_retirement_goal must seed the simulation from the profile's base
currency, and thread the stored goal (target + horizon) into the engine.

Regression guard: it previously read summary['total_value_usd'], which is the
wrong figure for a non-USD (e.g. CAD) base — the goal projection would start from
a silently converted number.
"""
import agent.tool_registry as reg


def _capture(monkeypatch):
    calls = {}

    def fake_run_monte_carlo(current_value, annual_contribution, years, **kwargs):
        calls["current_value"] = current_value
        calls["annual_contribution"] = annual_contribution
        calls["years"] = years
        calls.update(kwargs)
        return {"called": "retirement"}

    monkeypatch.setattr("tools.monte_carlo.run_monte_carlo", fake_run_monte_carlo)
    monkeypatch.setattr(reg, "_risk_tolerance_to_profile", lambda default="balanced": "balanced")
    return calls


def test_seeds_from_base_currency_not_usd(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_summary",
        lambda: {"total_value_base": 555000.0, "total_value_usd": 999999.0, "base_currency": "CAD"},
    )
    monkeypatch.setattr("tools.memory.get_financial_goal", lambda: None)

    reg.project_retirement_goal.invoke({})  # current_value=0 -> pull from portfolio

    assert calls["current_value"] == 555000.0  # base, NOT the 999999 USD figure


def test_threads_stored_goal_target_and_horizon(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_summary", lambda: {"total_value_base": 1_500_000.0})
    monkeypatch.setattr(
        "tools.memory.get_financial_goal",
        lambda: {"target_low": 3_000_000.0, "target_high": 5_000_000.0, "horizon_years": 7, "currency": "CAD"},
    )

    reg.project_retirement_goal.invoke({"monthly_contribution": 5000})

    assert calls["goal_target"] == 3_000_000.0
    assert calls["years"] == 7  # from the stored goal, not the 10 fallback
    assert calls["monthly_contribution"] == 5000
    assert calls["annual_contribution"] == 0  # monthly is the sole contribution channel


def test_defaults_to_ten_year_horizon_when_no_goal(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_summary", lambda: {"total_value_base": 100.0})
    monkeypatch.setattr("tools.memory.get_financial_goal", lambda: None)

    reg.project_retirement_goal.invoke({})

    assert calls["years"] == 10
    assert calls["goal_target"] is None
