"""The wealth goal lives on the user profile and is only ever set by the user.

Folded into user_profile (goal_target_low/high, goal_horizon_years) rather than a
separate store, so there is one home for "what the user wants" and no second
structure to drift. horizon is the goal's OWN field, never derived from
retirement_age.
"""
import json

import pytest

import tools.memory as mem


@pytest.fixture
def profile(monkeypatch, tmp_path):
    path = tmp_path / "user_memory.json"
    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    return path


def test_default_profile_states_no_goal(profile):
    assert mem.get_financial_goal() is None
    p = mem.load_memory()["user_profile"]
    assert p["goal_target_low"] is None
    assert p["goal_target_high"] is None
    assert p["goal_horizon_years"] is None
    assert p["goal_annual_contribution"] is None


def test_set_then_read_round_trips(profile):
    mem.update_profile({"base_currency": "CAD"})
    mem.set_financial_goal({"target_low": 3_000_000, "target_high": 5_000_000, "horizon_years": 10})

    goal = mem.get_financial_goal()
    assert goal == {
        "target_low": 3_000_000.0,
        "target_high": 5_000_000.0,
        "horizon_years": 10,
        # Stated targets do not imply a funding plan: the contribution stays
        # None until the user gives one, and the goal projection reads
        # `available: false` rather than assuming an inflow.
        "annual_contribution": None,
        "currency": "CAD",
    }
    # Persisted onto the profile itself — one home, no parallel block.
    stored = json.loads(profile.read_text())["user_profile"]
    assert stored["goal_target_low"] == 3_000_000.0
    assert "financial_goal" not in json.loads(profile.read_text())


def test_horizon_is_stored_as_int(profile):
    mem.set_financial_goal({"horizon_years": 10.0})
    assert mem.get_financial_goal()["horizon_years"] == 10
    assert isinstance(mem.load_memory()["user_profile"]["goal_horizon_years"], int)


def test_none_clears_a_field(profile):
    mem.set_financial_goal({"target_low": 3_000_000, "horizon_years": 10})
    mem.set_financial_goal({"target_low": None})

    goal = mem.get_financial_goal()
    assert goal["target_low"] is None
    assert goal["horizon_years"] == 10  # untouched


def test_clearing_everything_reads_as_no_goal(profile):
    mem.set_financial_goal({"target_low": 3_000_000, "target_high": 5_000_000, "horizon_years": 10})
    mem.set_financial_goal({"target_low": None, "target_high": None, "horizon_years": None})

    assert mem.get_financial_goal() is None


@pytest.mark.parametrize("bad", [0, -1, "abc", ""])
def test_junk_is_rejected_and_leaves_the_existing_target_standing(profile, bad):
    """Only None clears. A typo must not silently erase a real goal."""
    mem.set_financial_goal({"target_low": 3_000_000})

    mem.set_financial_goal({"target_low": bad})

    assert mem.get_financial_goal()["target_low"] == 3_000_000.0


@pytest.mark.parametrize("bad", [0, -1, "abc", ""])
def test_junk_never_creates_a_target_either(profile, bad):
    mem.set_financial_goal({"target_low": bad})

    assert mem.get_financial_goal() is None


def test_setting_a_goal_leaves_the_rest_of_memory_intact(profile):
    mem.add_lesson("Never catch a falling knife.")
    mem.set_risk_constraints({"max_position_pct": 10})

    mem.set_financial_goal({"target_low": 3_000_000, "horizon_years": 10})

    memory = mem.load_memory()
    assert memory["lessons_learned"] == ["Never catch a falling knife."]
    assert memory["risk_constraints"] == {"max_position_pct": 10.0}
    assert memory["user_profile"]["goal_target_low"] == 3_000_000.0


def test_partial_goal_is_still_a_goal(profile):
    """A target with no horizon yet still reads as set (horizon defaults downstream)."""
    mem.set_financial_goal({"target_low": 3_000_000})

    goal = mem.get_financial_goal()
    assert goal is not None
    assert goal["target_low"] == 3_000_000.0
    assert goal["horizon_years"] is None
