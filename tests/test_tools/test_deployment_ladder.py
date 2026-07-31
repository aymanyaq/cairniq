"""The armed cash-deployment ladder (Advisor Roadmap 3.9).

3.7 stored the rungs; this arms them. That difference is the whole item — a
−5% rung used to be visible only if the tape fell far enough to surface the
entire playbook, so the shallow rungs were decorative.

Every rung fire is an instruction to move real money, which sets what these
tests weigh. Almost all of them are about what must NOT happen: no fire for a
level the tape crossed before the ladder was armed, no second fire for a rung
already deployed, no re-arm on a wobble, no rung consumed by a stale quote. A
duplicate alert here does not cost attention, it costs a tranche.
"""
from datetime import datetime, timedelta

import pytest

import tools.drawdown_playbook as pb
import tools.memory as mem

_LADDER = [
    {"drawdown_pct": 5, "action": "Deploy the first 25% of cash into VTI."},
    {"drawdown_pct": 10, "action": "Deploy the next 25%."},
    {"drawdown_pct": 20, "action": "Deploy everything that is left."},
]

_T0 = datetime(2026, 3, 2, 10, 30)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    return tmp_path


@pytest.fixture
def state():
    return {}


def _run(state, depth, at=_T0):
    """One evaluation. Returns the result with `specs` kept for inspection."""
    return pb.evaluate_deployment_ladder(depth, state, at)


def _armed(state, depth, at=_T0):
    """Evaluate from a recovered baseline, so the first real crossing can fire."""
    _run(state, 0.0, at)  # at a new high — every rung armed, nothing seeded
    return _run(state, depth, at)


# ---------------------------------------------------------------------------
# Nothing on file — inert, and it says so rather than reading as healthy quiet
# ---------------------------------------------------------------------------

def test_no_ladder_on_file_is_inert_but_reports_that_it_ran(state):
    result = _run(state, 12.0)

    assert result["specs"] == []
    assert result["levels"] == 0
    assert result["armed"] == 0
    # The distinction the heartbeat depends on: evaluated-and-empty is NOT the
    # same as never-looked-at, and only one of them is a broken engine.
    assert result["evaluated"] is True


def test_deleting_the_ladder_clears_the_state_it_left_behind(state):
    """Otherwise a re-written −10% rung would inherit the deleted one's fire and
    stay silent through the drawdown it was written for."""
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 12.0)
    assert state["fired"]

    pb.set_playbook({"deployment_levels": None})
    _run(state, 12.0)

    assert state == {}


# ---------------------------------------------------------------------------
# The silent baseline — a ladder armed mid-drawdown fires nothing
# ---------------------------------------------------------------------------

def test_a_ladder_first_seen_mid_drawdown_fires_nothing(state):
    """The failure this guards against is a user writing their rules on a
    Tuesday when SPY is already 12% down and being handed three deployment
    orders at once, for levels that were crossed before anybody armed them."""
    pb.set_playbook({"deployment_levels": _LADDER})

    result = _run(state, 12.0)

    assert result["specs"] == []
    assert result["fired"] == 0
    assert result["seeded"] == 2  # the −5% and −10% rungs were already past
    assert result["armed"] == 1   # only −20% is still live


def test_rungs_passed_at_baseline_are_recorded_as_such_not_as_fires(state):
    """Readable by hand in the state file: a seeded rung must never be
    mistakable for one that actually delivered an instruction."""
    pb.set_playbook({"deployment_levels": _LADDER})
    _run(state, 12.0)

    assert state["fired"]["5"] == pb.LADDER_BASELINE
    assert state["fired"]["10"] == pb.LADDER_BASELINE


def test_a_rung_still_below_the_baseline_depth_fires_normally_later(state):
    pb.set_playbook({"deployment_levels": _LADDER})
    _run(state, 12.0)  # baseline: −5 and −10 seeded, −20 armed

    result = _run(state, 21.0)

    assert result["fired"] == 1
    assert result["specs"][0]["summary"]["drawdown_pct"] == 20


# ---------------------------------------------------------------------------
# Crossing — once, terminally
# ---------------------------------------------------------------------------

def test_crossing_a_rung_delivers_the_users_own_action(state):
    pb.set_playbook({"deployment_levels": _LADDER})

    result = _armed(state, 6.0)

    assert result["fired"] == 1
    body = result["specs"][0]["raise"]
    assert "Deploy the first 25% of cash into VTI." in body["message"]
    assert body["data"]["drawdown_pct"] == 5
    assert body["severity"] == "warning"


def test_the_alert_adds_no_advice_of_its_own(state):
    """The entire authority of this alert is that every word of the instruction
    is the user's. Anything the app contributes dilutes exactly that."""
    pb.set_playbook({"deployment_levels": _LADDER})

    body = _armed(state, 6.0)["specs"][0]["raise"]["message"].lower()

    for invented in ("we recommend", "you should", "consider ", "buy the dip", "opportunity"):
        assert invented not in body, f"the rung alert invented advice: {invented}"
    assert "the decision and the order are yours" in body


def test_a_rung_does_not_fire_twice_in_the_same_episode(state):
    pb.set_playbook({"deployment_levels": _LADDER})
    assert _armed(state, 6.0)["fired"] == 1

    assert _run(state, 6.0)["fired"] == 0
    assert _run(state, 7.0)["fired"] == 0
    assert _run(state, 6.5)["fired"] == 0


def test_a_deepening_drawdown_fires_only_the_newly_crossed_rung(state):
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 6.0)

    result = _run(state, 11.0)

    assert result["fired"] == 1
    assert result["specs"][0]["summary"]["drawdown_pct"] == 10
    assert result["armed"] == 1


def test_a_gap_straight_through_two_rungs_fires_both(state):
    """A tape that opens −22% skipped nothing — both instructions are due."""
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 6.0)

    result = _run(state, 22.0)

    assert result["fired"] == 2
    assert [s["summary"]["drawdown_pct"] for s in result["specs"]] == [10, 20]
    assert result["armed"] == 0


def test_a_recovering_tape_fires_nothing_and_keeps_rungs_spent(state):
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 11.0)

    result = _run(state, 4.0)  # recovering, but the episode is not over

    assert result["fired"] == 0
    assert result["armed"] == 1  # −5 and −10 stay spent


# ---------------------------------------------------------------------------
# Re-arming — only on a new high, and a wobble is not a new high
# ---------------------------------------------------------------------------

def test_the_ladder_re_arms_when_the_index_makes_a_new_high(state):
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 11.0)

    recovered = _run(state, 0.0)

    assert recovered["armed"] == 3
    assert state["fired"] == {}
    # And the next episode genuinely re-deploys.
    assert _run(state, 6.0)["fired"] == 1


def test_a_wobble_short_of_a_new_high_does_not_re_arm(state):
    """The money version of hysteresis. Reset anywhere near the shallowest rung
    and a 1pp round trip would re-deploy a tranche already deployed."""
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 6.0)

    _run(state, pb.LADDER_RESET_PCT + 0.5)  # recovered, but not to a new high

    assert _run(state, 6.0)["fired"] == 0


def test_a_new_episode_carries_its_own_dedup_key(state):
    """Same rung, same depth, different episode — the inbox must not collapse
    the second instruction into a refresh of the first."""
    pb.set_playbook({"deployment_levels": _LADDER})
    first = _armed(state, 6.0)["specs"][0]["raise"]["dedup_key"]

    _run(state, 0.0, _T0 + timedelta(days=200))
    second = _run(state, 6.0, _T0 + timedelta(days=201))["specs"][0]["raise"]["dedup_key"]

    assert first != second


def test_re_arming_at_a_new_high_seeds_nothing(state):
    """A reset is not a first sight: coming back through a new high must leave
    every rung genuinely armed, not silently marked as already passed."""
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 22.0)

    assert _run(state, 0.0)["seeded"] == 0
    assert _run(state, 6.0)["fired"] == 1


# ---------------------------------------------------------------------------
# Editing the ladder
# ---------------------------------------------------------------------------

def test_a_rung_edited_away_does_not_suppress_a_new_one_at_that_depth(state):
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 11.0)  # −5 and −10 now spent

    pb.set_playbook({"deployment_levels": [{"drawdown_pct": 10, "action": "New plan: buy VEA."}]})
    result = _run(state, 11.0)

    assert result["fired"] == 0  # still the same rung depth, still spent
    assert "10" in state["fired"]


def test_adding_a_deeper_rung_mid_episode_arms_it(state):
    pb.set_playbook({"deployment_levels": _LADDER})
    _armed(state, 11.0)

    pb.set_playbook({"deployment_levels": [*_LADDER, {"drawdown_pct": 30, "action": "Last of it."}]})
    result = _run(state, 11.0)

    assert result["armed"] == 2  # −20 and the new −30
    assert result["fired"] == 0
