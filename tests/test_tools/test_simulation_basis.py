"""Roadmap 2.7 — `simulate_scenario` must declare its assumed magnitudes.

The scenario table (`recession: -35%`, `tech_crash: -45%`) is a set of round numbers
labelled "2008-style" / "Dot-com style". Nothing measured them. The marker used to be
added by the `run_stress_test` agent-tool wrapper, which meant any other caller got
the constants unlabelled; it now lives on the function that owns the table, and these
tests are what stop it drifting back.
"""
from tools.simulation import simulate_scenario


def test_scenario_payload_declares_authored_basis():
    result = simulate_scenario("AAPL,MSFT", "recession")
    assert result["basis"] == "authored constant"
    assert result["measured_alternative"] == "replay_historical_episode"


def test_basis_note_names_the_specific_assumed_number():
    """A generic caveat is skippable; the actual figure in the sentence is not.

    The reader has to be able to connect "assumed" to the -35% they just read.
    """
    result = simulate_scenario("AAPL", "recession")
    assert "-35%" in result["basis_note"]

    tech = simulate_scenario("AAPL", "tech_crash")
    assert "-45%" in tech["basis_note"]


def test_marker_is_present_on_every_scenario_including_the_upside_one():
    """bull_market's +25% is authored exactly like the drawdowns.

    Worth pinning: an optimistic constant reads as harmless and is the one most
    likely to be quoted back as "historically, recoveries run +25%".
    """
    for scenario in ("recession", "rate_hike", "tech_crash", "bull_market"):
        result = simulate_scenario("SPY", scenario)
        assert result["basis"] == "authored constant", scenario
