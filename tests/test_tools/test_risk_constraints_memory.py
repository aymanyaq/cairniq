"""Risk limits live in user_memory.json and are only ever set by the user."""
import json
import re

import pytest

import tools.ips_precheck as ips
import tools.memory as mem


@pytest.fixture
def profile(monkeypatch, tmp_path):
    path = tmp_path / "user_memory.json"
    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(ips, "_load_memory", mem.load_memory)
    return path


def test_default_profile_states_no_limits(profile):
    assert mem.load_memory()["risk_constraints"] == {}
    assert ips.stated_caps() == {}


def test_set_then_read_round_trips_through_the_precheck(profile):
    mem.set_risk_constraints({"max_risk_per_trade_pct": 1.5, "max_position_pct": 12})

    assert ips.stated_caps() == {"max_risk_per_trade_pct": 1.5, "max_position_pct": 12.0}
    assert json.loads(profile.read_text())["risk_constraints"]["max_risk_per_trade_pct"] == 1.5


def test_none_clears_a_limit_back_to_unconstrained(profile):
    mem.set_risk_constraints({"max_risk_per_trade_pct": 2.0})
    mem.set_risk_constraints({"max_risk_per_trade_pct": None})

    assert ips.stated_caps() == {}
    assert ips.load_ips_constraints()["max_risk_per_trade_pct"] is None


@pytest.mark.parametrize("bad", [0, -1, "abc", ""])
def test_junk_is_rejected_and_leaves_the_existing_limit_standing(profile, bad):
    """Only None clears. A typo must not silently delete a real protection."""
    mem.set_risk_constraints({"max_risk_per_trade_pct": 2.0})

    mem.set_risk_constraints({"max_risk_per_trade_pct": bad})

    assert ips.stated_caps()["max_risk_per_trade_pct"] == 2.0


@pytest.mark.parametrize("bad", [0, -1, "abc", ""])
def test_junk_never_creates_a_limit_either(profile, bad):
    mem.set_risk_constraints({"max_risk_per_trade_pct": bad})

    assert ips.stated_caps() == {}


def test_unknown_keys_are_ignored(profile):
    mem.set_risk_constraints({"max_leverage": 3, "max_position_pct": 10})

    assert mem.load_memory()["risk_constraints"] == {"max_position_pct": 10.0}


def test_restricted_symbols_normalise(profile):
    mem.set_risk_constraints({"restricted_symbols": ["nvda", " tsla ", "", "NVDA"]})

    assert ips.load_ips_constraints()["restricted_symbols"] == ["NVDA", "TSLA"]


def test_setting_limits_leaves_the_rest_of_memory_intact(profile):
    """risk_constraints is a sibling of lessons — it must not clobber them."""
    mem.add_lesson("Never catch a falling knife.")
    mem.update_profile({"base_currency": "CAD"})

    mem.set_risk_constraints({"max_risk_per_trade_pct": 1.0})

    memory = mem.load_memory()
    assert memory["lessons_learned"] == ["Never catch a falling knife."]
    assert memory["user_profile"]["base_currency"] == "CAD"
    assert memory["risk_constraints"] == {"max_risk_per_trade_pct": 1.0}


def test_lesson_edits_preserve_existing_limits(profile):
    """And the reverse: the older write paths must not drop the new key."""
    mem.set_risk_constraints({"max_position_pct": 8.0})

    mem.add_lesson("Report all values in CAD.")
    mem.update_profile({"age": "37"})

    assert mem.load_memory()["risk_constraints"] == {"max_position_pct": 8.0}


# ---------------------------------------------------------------------------
# "Unset" and "unset ON PURPOSE" are different answers
#
# The store's contract has always been that a blank cap enforces nothing. What
# it could not say is whether the user had ever been ASKED — and for months the
# answer was no, while the gate that reads this block sat empty and said nothing
# about it. The acknowledgement is that missing bit, and nothing else: it never
# creates, implies or defaults a cap.
# ---------------------------------------------------------------------------

def test_a_bare_profile_is_not_execution_ready(profile):
    """No caps, nobody asked. Every axis is an open question."""
    readiness = ips.execution_readiness()

    assert readiness["execution_ready"] is False
    assert sorted(readiness["unanswered"]) == sorted(ips._CONSTRAINT_KEYS)
    assert readiness["unconstrained_by_choice"] == []


def test_stating_every_cap_is_execution_ready_without_any_acknowledgement(profile):
    """The acknowledgement answers blanks. A profile with no blanks needs none."""
    mem.set_risk_constraints({
        "max_position_pct": 10, "max_fund_position_pct": 25,
        "max_sector_pct": 30, "max_risk_per_trade_pct": 2,
    })

    readiness = ips.execution_readiness()

    assert readiness["execution_ready"] is True
    assert readiness["unanswered"] == []
    assert readiness["note"] == ""


def test_acknowledging_the_blanks_makes_an_uncapped_profile_execution_ready(profile):
    """The whole point: "no limits" is a real answer, once it is actually given."""
    mem.set_risk_constraints({"acknowledge_unconstrained": True})

    readiness = ips.execution_readiness()

    assert readiness["execution_ready"] is True
    assert sorted(readiness["unconstrained_by_choice"]) == sorted(ips._CONSTRAINT_KEYS)
    assert readiness["acknowledged_at"]
    # And it authored nothing while doing it.
    assert ips.stated_caps() == {}


def test_acknowledgement_covers_only_the_axes_blank_when_it_was_given(profile):
    """A cap stated in the same write is not "confirmed unlimited"."""
    mem.set_risk_constraints({"max_position_pct": 10, "acknowledge_unconstrained": True})

    readiness = ips.execution_readiness()

    assert readiness["execution_ready"] is True
    assert "max_position_pct" not in readiness["unconstrained_by_choice"]
    assert readiness["stated"] == {"max_position_pct": 10.0}


def test_deleting_a_cap_reopens_that_axis_rather_than_inheriting_the_old_answer(profile):
    """The reason the acknowledgement stores axis NAMES and not a bare flag.

    A user who confirms three blanks and later deletes their one real cap has
    said nothing about the axis they just cleared. A boolean would have quietly
    counted it as confirmed — a limit deleted this minute inheriting a
    confirmation given about entirely different axes.
    """
    mem.set_risk_constraints({"max_position_pct": 10, "acknowledge_unconstrained": True})
    assert ips.execution_readiness()["execution_ready"] is True

    mem.set_risk_constraints({"max_position_pct": None})

    readiness = ips.execution_readiness()
    assert readiness["execution_ready"] is False
    assert readiness["unanswered"] == ["max_position_pct"]


def test_the_acknowledgement_can_be_withdrawn(profile):
    mem.set_risk_constraints({"acknowledge_unconstrained": True})

    mem.set_risk_constraints({"acknowledge_unconstrained": False})

    assert ips.execution_readiness()["execution_ready"] is False
    assert mem.UNCONSTRAINED_ACK_KEY not in mem.load_memory()["risk_constraints"]


def test_an_ordinary_save_leaves_an_existing_acknowledgement_alone(profile):
    """Only True and False touch it. A cap edit that omits the flag is silent."""
    mem.set_risk_constraints({"acknowledge_unconstrained": True})

    mem.set_risk_constraints({"max_sector_pct": 30})

    readiness = ips.execution_readiness()
    assert readiness["execution_ready"] is True
    assert readiness["stated"] == {"max_sector_pct": 30.0}


def test_the_acknowledgement_is_never_mistaken_for_a_cap(profile):
    """It lives in the same block as the caps, so the readers must ignore it."""
    mem.set_risk_constraints({"acknowledge_unconstrained": True})

    constraints = ips.load_ips_constraints()

    assert ips.stated_caps(constraints) == {}
    assert all(constraints[key] is None for key in ips._CONSTRAINT_KEYS)


def test_the_unready_sentence_names_no_figure(profile):
    """It is read by the judge and by the user, and this codebase has already
    shipped a judge that turned an absent limit into a rule it attributed to the
    user. The sentence may name the axis and never a number."""
    note = ips.execution_readiness()["note"]

    assert note
    assert "%" not in note
    assert not re.search(r"\d", note)
