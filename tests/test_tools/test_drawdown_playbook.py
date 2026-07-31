"""Drawdown playbook (Advisor Roadmap 3.7).

The item exists because a −30% tape inside a ten-year plan is near-certain and
the plan only dies if the holder sells. So the tests that matter are not "does
it store a list". They are: does the app ever author a rule it will later read
back as the user's own, and does the alert stay honest when nothing has been
agreed. A fabricated rule recited during a crash carries the authority of a
promise the user made to themselves — they will act on it.
"""
import pytest

import tools.drawdown_playbook as pb
import tools.memory as mem


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    return tmp_path


# ---------------------------------------------------------------------------
# Nothing is ever authored by the app
# ---------------------------------------------------------------------------

def test_an_unset_playbook_is_none_not_a_sensible_default():
    assert pb.get_playbook() is None


def test_the_message_for_an_unset_playbook_names_the_absence(isolated):
    """The highest-value thing to say at −20% with no rules on file is that
    there are no rules on file — and explicitly that now is the wrong moment to
    invent them. It must not offer any."""
    text = pb.describe_playbook(None, 20.0)

    assert "No drawdown playbook is on file" in text
    assert "Nothing is being recommended here" in text
    # No invented rules of any kind.
    for word in ("never sell", "deploy", "rebalance", "buy "):
        assert word not in text.lower(), f"the empty-state message suggested: {word}"


def test_a_malformed_value_is_rejected_and_leaves_the_existing_rule_standing():
    """A typo must not silently delete a rule the user will be relying on during
    the worst week of the decade."""
    pb.set_playbook({"rebalance_drift_pct": 5})
    pb.set_playbook({"rebalance_drift_pct": "five percent"})

    assert pb.get_playbook()["rebalance_drift_pct"] == 5.0


def test_none_clears_a_field_without_touching_the_rest():
    pb.set_playbook({"never_sell": ["VTI"], "rebalance_drift_pct": 5})
    pb.set_playbook({"never_sell": None})

    playbook = pb.get_playbook()
    assert "never_sell" not in playbook
    assert playbook["rebalance_drift_pct"] == 5.0


def test_clearing_everything_reads_as_no_playbook():
    pb.set_playbook({"never_sell": ["VTI"]})
    pb.set_playbook({"never_sell": None})

    assert pb.get_playbook() is None


# ---------------------------------------------------------------------------
# The rules are instructions, so their shape is load-bearing
# ---------------------------------------------------------------------------

def test_buy_first_preserves_the_users_order():
    """'What new contributions buy first' is a priority sequence. Sorting or
    de-duplicating it would change the instruction."""
    pb.set_playbook({"buy_first": ["Total market index", "Bond sleeve", "Individual names"]})

    assert pb.get_playbook()["buy_first"] == [
        "Total market index", "Bond sleeve", "Individual names"
    ]


def test_the_deployment_ladder_is_sorted_shallowest_first():
    """The ladder is read top-down while the tape falls, so it must read in the
    order it will be hit — not in whatever order it was typed."""
    pb.set_playbook({"deployment_levels": [
        {"drawdown_pct": 30, "action": "deploy the rest"},
        {"drawdown_pct": 10, "action": "deploy a third"},
        {"drawdown_pct": 20, "action": "deploy half"},
    ]})

    depths = [lvl["drawdown_pct"] for lvl in pb.get_playbook()["deployment_levels"]]
    assert depths == [10.0, 20.0, 30.0]


def test_a_rung_with_no_depth_is_dropped_not_defaulted():
    """Guessing a level would put real money on a number nobody chose."""
    pb.set_playbook({"deployment_levels": [
        {"action": "deploy something"},                       # no depth
        {"drawdown_pct": 15, "action": "deploy half"},
        {"drawdown_pct": 20, "action": ""},                   # no action
    ]})

    assert pb.get_playbook()["deployment_levels"] == [
        {"drawdown_pct": 15.0, "action": "deploy half"}
    ]


def test_breached_rungs_are_marked_live_at_the_current_depth():
    """At −22% the user must see which of their own instructions apply NOW,
    without doing arithmetic mid-panic."""
    pb.set_playbook({"deployment_levels": [
        {"drawdown_pct": 10, "action": "deploy a third"},
        {"drawdown_pct": 20, "action": "deploy half"},
        {"drawdown_pct": 30, "action": "deploy the rest"},
    ]})

    text = pb.describe_playbook(pb.get_playbook(), 22.0)

    assert text.count("LIVE NOW") == 2          # the 10% and 20% rungs
    assert "−30%: deploy the rest ←" not in text


def test_a_stated_playbook_is_rendered_verbatim():
    pb.set_playbook({
        "never_sell": ["Core index ETFs", "RRSP bond sleeve"],
        "buy_first": ["Total market index"],
        "notes": "The 2020 low recovered in five months.",
    })

    text = pb.describe_playbook(pb.get_playbook(), 18.0)

    assert "Core index ETFs, RRSP bond sleeve" in text
    assert "1. Total market index" in text
    assert "The 2020 low recovered in five months." in text


# ---------------------------------------------------------------------------
# The goal pairing — the behavioural half, and a fabrication risk
# ---------------------------------------------------------------------------

def test_no_goal_means_silence_not_a_reassuring_generality(monkeypatch):
    """'You're still on track' with no target behind it is fabricated comfort,
    and a drawdown is the worst possible moment to fabricate one."""
    import tools.goal_projection as gp

    monkeypatch.setattr(
        gp, "build_goal_projection",
        lambda **kw: {"available": False, "reason": "no wealth goal set"},
    )

    assert pb.goal_status_line() == ""


def test_a_set_goal_answers_from_todays_depressed_value(monkeypatch):
    """The projection runs off the CURRENT portfolio value, so during a drawdown
    it answers the only question that matters: does the plan still work FROM
    HERE, with contributions continuing?"""
    import tools.goal_projection as gp

    monkeypatch.setattr(gp, "build_goal_projection", lambda **kw: {
        "available": True, "goal_success_rate": 71.4, "currency": "CAD",
        "annual_contribution": 65000.0, "horizon_years": 10,
    })

    line = pb.goal_status_line()

    assert "71% of simulated paths" in line
    assert "65,000 CAD/yr" in line
    assert "Selling converts a paper drawdown into a permanent one" in line


def test_a_broken_goal_projection_never_costs_the_drawdown_alert(monkeypatch):
    """The band alert is the load-bearing part. An optional footer that throws
    must not silence the message the whole item exists to deliver."""
    import tools.goal_projection as gp

    def _boom(**kw):
        raise RuntimeError("monte carlo exploded")

    monkeypatch.setattr(gp, "build_goal_projection", _boom)
    pb.set_playbook({"never_sell": ["VTI"]})

    message = pb.build_drawdown_message(18.0, "a deep correction (>15% off high)")

    assert "VTI" in message
    assert "18.0% off its 6-month high" in message


# ---------------------------------------------------------------------------
# Base-currency resolution — found while verifying 3.7's alert text
# ---------------------------------------------------------------------------

def test_the_page_and_the_store_resolve_an_unset_currency_identically():
    """They did not, and 4.5's wealth goal made it expensive.

    The page resolved an unset base_currency through the persisted .env and a
    locale default (CAD); tools.memory hardcoded USD. A profile that had never
    stated a currency therefore read "CAD" on every screen while its stored goal
    was stamped "USD" — so a 10-year target typed as CAD would be scored as USD,
    a ~40% error on the single number the plan is measured against.
    """
    from api.routers.pages import _configured_base_currency
    from tools.memory import configured_base_currency, get_profile_base_currency

    assert _configured_base_currency() == configured_base_currency()
    assert get_profile_base_currency({}) == configured_base_currency()


def test_a_stated_currency_always_beats_the_deployment_default():
    from tools.memory import get_profile_base_currency

    assert get_profile_base_currency({"base_currency": "EUR"}) == "EUR"
    assert get_profile_base_currency({"base_currency": "JPY"}) == "JPY"


def test_an_unsupported_stated_currency_falls_back_rather_than_propagating():
    from tools.memory import configured_base_currency, get_profile_base_currency

    assert get_profile_base_currency({"base_currency": "XYZ"}) == "USD"
    assert configured_base_currency() in {"USD", "CAD", "EUR", "GBP", "AUD", "JPY"}
