"""calculate_position_size must never invent the user's risk budget.

It used to default risk_per_trade_pct to 2.0 and stamp "2%" into every result,
which the advisor then reported back as the user's own rule.
"""
import tools.ips_precheck as ips
from tools.position_sizing import calculate_position_size


def _no_constraints(monkeypatch):
    monkeypatch.setattr(ips, "_load_memory", lambda: {"risk_constraints": {}})


def _with_limit(monkeypatch, pct):
    monkeypatch.setattr(ips, "_load_memory", lambda: {"risk_constraints": {"max_risk_per_trade_pct": pct}})


def test_no_stated_limit_returns_no_size(monkeypatch):
    _no_constraints(monkeypatch)

    result = calculate_position_size(100_000, entry_price=200, stop_loss_price=180)

    assert "recommended_shares" not in result
    assert "base_risk_pct" not in result
    assert "no maximum-risk rule" in result["risk_basis"]
    assert "no default 2% rule" in result["sizing_unavailable"]


def test_stated_limit_is_used_and_attributed(monkeypatch):
    _with_limit(monkeypatch, 1.0)

    result = calculate_position_size(100_000, entry_price=200, stop_loss_price=180)

    assert result["base_risk_pct"] == "1%"
    assert result["risk_basis"] == "the user's own stated 1% max-risk rule"
    # $1,000 at risk / $20 per share
    assert result["recommended_shares"] == 50


def test_explicit_pct_is_labelled_as_an_assumption_not_a_rule(monkeypatch):
    _no_constraints(monkeypatch)

    result = calculate_position_size(100_000, 2.0, entry_price=200, stop_loss_price=180)

    assert result["recommended_shares"] == 100
    assert "not a rule from the user's profile" in result["risk_basis"]


def test_explicit_pct_overrides_a_stated_rule_but_says_so(monkeypatch):
    _with_limit(monkeypatch, 1.0)

    result = calculate_position_size(100_000, 5.0, entry_price=200, stop_loss_price=180)

    assert result["base_risk_pct"] == "5%"
    assert "not a rule from the user's profile" in result["risk_basis"]


def test_generic_tier_warning_carries_no_phantom_rule(monkeypatch):
    _with_limit(monkeypatch, 1.0)

    result = calculate_position_size(100_000)

    assert "2% rule" not in result["warning"]
    assert "the user's own stated 1% max-risk rule" in result["warning"]
    assert result["generic_allocations"]["5% Tier"] == "$5,000.00"


def test_unreadable_profile_does_not_fall_back_to_two_percent(monkeypatch):
    def boom():
        raise OSError("profile unreadable")
    monkeypatch.setattr(ips, "_load_memory", boom)

    result = calculate_position_size(100_000, entry_price=200, stop_loss_price=180)

    assert "sizing_unavailable" in result
    assert "recommended_shares" not in result
