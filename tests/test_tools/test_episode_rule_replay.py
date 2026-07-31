"""Tests for the 4.3b rule replayer.

The study's whole validity rests on the state machine below being point-in-time,
so these are mostly hygiene tests rather than output tests: a replayer that peeks
at the future produces a beautiful, worthless report, and it does so silently.
"""
import pandas as pd
import pytest

from tools import episode_rule_replay as err


def _bars(closes, opens=None):
    """Daily bars from a close path. Opens default to the prior close (gapless)."""
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    if opens is None:
        opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame({"open": opens, "close": closes}, index=idx)


# ---------------------------------------------------------------------------
# point-in-time hygiene — the tests that matter most
# ---------------------------------------------------------------------------

def test_entry_is_at_the_next_open_not_the_signal_close():
    """The trigger is visible at the close; the fill cannot be."""
    # Day 2 closes −6% from the peak → level −5% fires. Day 3 opens at 90.
    bars = _bars([100, 100, 94, 96, 101], opens=[100, 100, 99, 90, 95])
    out = err.replay_rules(bars)

    assert len(out["legs"]) == 1
    leg = out["legs"][0]
    assert leg["entry_price"] == 90.0        # day 3's OPEN
    assert leg["entry_date"] == "2020-01-06"  # not day 2


def test_running_peak_is_point_in_time_so_a_later_high_cannot_trigger_earlier():
    """The classic lookahead bug: using the window's global max as the peak.

    Here the price rises to 200 at the very end. Measured against that global
    peak, every earlier session is deep in 'drawdown' and would have deployed.
    Measured correctly, nothing triggers at all.
    """
    bars = _bars([100, 101, 102, 103, 200])
    out = err.replay_rules(bars)
    assert out["legs"] == []
    assert out["episodes_seen"] == 0


def test_drawdown_is_measured_from_the_peak_the_rule_had_seen():
    """A dip that is shallow against a new peak must not fire a deeper level."""
    # Peak walks 100 → 120, then a 6% fall from 120 (to 112.8). That is a −5%
    # level, not a −10% one, even though 112.8 is above the ORIGINAL 100 peak.
    bars = _bars([100, 110, 120, 112.8, 112.8])
    out = err.replay_rules(bars)
    assert [leg["label"] for leg in out["legs"]] == ["level -5%"]


# ---------------------------------------------------------------------------
# the deployment ladder
# ---------------------------------------------------------------------------

def test_both_levels_fire_on_a_deepening_drawdown_in_ladder_order():
    bars = _bars([100, 94, 88, 88, 88])
    out = err.replay_rules(bars)
    labels = [leg["label"] for leg in out["legs"]]
    assert labels == ["level -5%", "level -10%"]
    assert [leg["capital_pct"] for leg in out["legs"]] == [40.0, 60.0]


def test_a_level_fires_once_per_episode_not_once_per_session():
    """Sitting at −7% for a week is one deployment, not five."""
    bars = _bars([100, 93, 93, 93, 93, 93])
    out = err.replay_rules(bars)
    assert len(out["legs"]) == 1


def test_levels_rearm_after_the_episode_resolves():
    # Fall to −6%, recover to a new high (resolution), fall −6% again.
    bars = _bars([100, 94, 101, 101, 95, 95])
    out = err.replay_rules(bars)
    assert [leg["label"] for leg in out["legs"]] == ["level -5%", "level -5%"]
    assert out["episodes_seen"] == 2


def test_resolution_closes_every_open_leg():
    bars = _bars([100, 94, 88, 101, 101])
    out = err.replay_rules(bars)
    assert len(out["legs"]) == 2
    assert {leg["exit_reason"] for leg in out["legs"]} == {"resolution"}


def test_capital_cannot_exceed_the_sleeve():
    """40% + 60% is the whole sleeve; a third trigger has nothing left to spend."""
    bars = _bars([100, 94, 88, 80, 80])
    out = err.replay_rules(bars, levels=((-5.0, 0.4), (-10.0, 0.6), (-18.0, 0.5)))
    assert sum(leg["capital_pct"] for leg in out["legs"]) == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# risk exits
# ---------------------------------------------------------------------------

def test_disaster_stop_fires_at_minus_25_percent_on_the_position():
    bars = _bars([100, 94, 70, 68, 68, 68])
    out = err.replay_rules(bars)
    assert "disaster_stop" in {leg["exit_reason"] for leg in out["legs"]}


def test_max_hold_exit_is_reported_as_its_own_reason():
    """A rule that only 'works' by holding for years must be visible as such."""
    bars = _bars([100, 94] + [94] * 20)
    out = err.replay_rules(bars, max_hold_sessions=5)
    assert out["legs"][0]["exit_reason"] == "max_hold"
    assert out["legs"][0]["sessions_held"] >= 5


def test_legs_still_open_at_the_window_end_are_flagged_unresolved():
    """An unresolved leg is not a resolved one, and must not read as a clean win."""
    bars = _bars([100, 94, 96, 97])
    out = err.replay_rules(bars)
    assert out["legs"][0]["exit_reason"] == "window_end_unresolved"


# ---------------------------------------------------------------------------
# the bell — proxy and sourced-calendar arms
# ---------------------------------------------------------------------------

def test_bell_does_not_ring_outside_an_open_episode():
    """A +2% day in a calm uptrend is not a de-escalation."""
    bars = _bars([100, 102.5, 105, 107.5])
    out = err.replay_rules(bars, bell=True)
    assert out["bell_signals"] == 0
    assert out["legs"] == []


def test_bell_rings_on_a_relief_session_inside_an_episode():
    bars = _bars([100, 96, 94, 96.5, 96.5])
    out = err.replay_rules(bars, bell=True)
    assert out["bell_signals"] == 1
    assert "bell" in [leg["label"] for leg in out["legs"]]


def test_bell_requires_vix_to_fall_when_vix_is_available():
    """A big up-day with VIX still climbing is the bear-market rally the bell
    exists NOT to buy — dropping the check would widen the rule under test."""
    bars = _bars([100, 96, 94, 96.5, 96.5])
    rising_vix = pd.Series([20, 30, 35, 36, 36], index=bars.index)
    out = err.replay_rules(bars, bell=True, vix=rising_vix)
    assert out["bell_signals"] == 0

    falling_vix = pd.Series([20, 30, 35, 28, 27], index=bars.index)
    out = err.replay_rules(bars, bell=True, vix=falling_vix)
    assert out["bell_signals"] == 1


def test_bell_rings_once_per_episode():
    bars = _bars([100, 96, 94, 96.5, 98.5, 100.5, 102.6])
    out = err.replay_rules(bars, bell=True)
    assert out["bell_signals"] == 1


def test_sourced_calendar_arm_overrides_the_proxy_and_is_recorded():
    """If a real resolution calendar is ever compiled, the report must say which
    arm produced the numbers — the two are not the same claim."""
    bars = _bars([100, 96, 94, 94.2, 94.2])
    out = err.replay_rules(bars, bell=True, resolution_dates=["2020-01-06"])
    assert out["bell_arm"] == "sourced resolution calendar"
    assert out["bell_signals"] == 1

    proxy = err.replay_rules(bars, bell=True)
    assert proxy["bell_arm"] == "mechanical relief proxy"
    assert proxy["bell_signals"] == 0  # +0.2% is no relief session


# ---------------------------------------------------------------------------
# cash accrual and benchmarks
# ---------------------------------------------------------------------------

def test_idle_cash_earns_the_bill_rate_so_the_comparison_is_not_rigged():
    """Without this, 'strategy beats cash' is partly just 'cash earned nothing'."""
    bars = _bars([100, 101, 102, 103])
    rf = pd.Series([5.0] * 4, index=bars.index)
    out = err.replay_rules(bars, risk_free=rf)
    assert out["legs"] == []
    assert out["sleeve_return_pct"] > 0  # never deployed, still earned


def test_cash_benchmark_reports_none_rather_than_assuming_a_rate():
    """An assumed 4% would be an authored constant standing in for a measurement."""
    out = err.benchmark_cash(_bars([100, 101]), None)
    assert out["return_pct"] is None
    assert "could not be measured" in out["note"]


def test_buy_and_hold_carries_its_own_artifact_warning():
    out = err.benchmark_buy_and_hold(_bars([100, 90, 100]))
    assert "by construction" in out["artifact_note"]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def test_gate_applies_the_specs_own_thresholds():
    result = {"legs": [
        {"return_pct": 10.0, "exit_reason": "resolution", "max_adverse_close_pct": -2.0},
        {"return_pct": 8.0, "exit_reason": "resolution", "max_adverse_close_pct": -1.0},
        {"return_pct": -5.0, "exit_reason": "disaster_stop", "max_adverse_close_pct": -25.0},
    ]}
    out = err.summarize(result)
    assert out["hit_rate_pct"] == pytest.approx(66.7, abs=0.1)
    assert out["profit_factor"] == pytest.approx(3.6, abs=0.05)
    assert out["gate_passed"] is True


def test_gate_fails_on_a_good_hit_rate_with_a_bad_profit_factor():
    """Two small wins and one large loss is the shape that a hit rate hides."""
    result = {"legs": [
        {"return_pct": 1.0, "exit_reason": "resolution", "max_adverse_close_pct": 0.0},
        {"return_pct": 1.0, "exit_reason": "resolution", "max_adverse_close_pct": 0.0},
        {"return_pct": -20.0, "exit_reason": "disaster_stop", "max_adverse_close_pct": -22.0},
    ]}
    out = err.summarize(result)
    assert out["hit_rate_pct"] > 55
    assert out["gate_passed"] is False
    assert "profit factor" in out["gate_note"]


def test_no_losing_leg_gives_an_undefined_profit_factor_not_an_infinite_one():
    result = {"legs": [
        {"return_pct": 5.0, "exit_reason": "resolution", "max_adverse_close_pct": 0.0},
    ]}
    out = err.summarize(result)
    assert out["profit_factor"] is None
    assert "small-n artifact" in out["profit_factor_note"]


def test_no_deployments_is_reported_as_no_edge_measured_not_as_a_pass():
    out = err.summarize({"legs": []})
    assert out["gate_passed"] is False
    assert out["n_legs"] == 0


# ---------------------------------------------------------------------------
# small-sample humility (the spec's own §3.6a estimator)
# ---------------------------------------------------------------------------

def test_wilson_lower_bound_punishes_small_samples():
    """5/6 legs looks like an 83% hit rate and is not evidence of one.

    This is the whole reason the spec builds confidence on the lower bound: the
    study pools six legs across three episodes, and quoting the raw rate would
    turn a non-falsification into a claimed edge.
    """
    small = err.summarize({"legs": [
        {"return_pct": r, "exit_reason": "resolution", "max_adverse_close_pct": 0.0}
        for r in (5, 5, 5, 5, 5, -5)
    ]})
    assert small["hit_rate_pct"] == pytest.approx(83.3, abs=0.1)
    assert small["hit_rate_wilson_lower_pct"] < 55.0  # below the gate's own bar

    large = err.summarize({"legs": [
        {"return_pct": r, "exit_reason": "resolution", "max_adverse_close_pct": 0.0}
        for r in ([5] * 50 + [-5] * 10)
    ]})
    assert large["hit_rate_wilson_lower_pct"] > small["hit_rate_wilson_lower_pct"]


def test_summary_states_the_sample_size_alongside_the_rate():
    out = err.summarize({"legs": [
        {"return_pct": 5.0, "exit_reason": "resolution", "max_adverse_close_pct": 0.0},
        {"return_pct": -1.0, "exit_reason": "resolution", "max_adverse_close_pct": -1.0},
    ]})
    assert "non-falsification" in out["small_sample_note"]


# ---------------------------------------------------------------------------
# the phantom-leg bug the first study run exposed
# ---------------------------------------------------------------------------

def test_a_zero_capital_tranche_is_not_a_leg():
    """Found by running the study: with the sleeve fully committed at -10%, the
    bell opened a 0.0%-capital position that "lost" 25.8% of nothing and still
    counted against the hit rate, dragging the pooled figure from 50% to 44%."""
    # Levels take the whole sleeve before the relief session arrives.
    bars = _bars([100, 94, 88, 90.5, 90.5, 90.5])
    out = err.replay_rules(bars, bell=True)
    assert all(leg["capital_pct"] > 0 for leg in out["legs"])


def test_min_leg_capital_threshold_is_configurable_and_enforced():
    bars = _bars([100, 94, 94, 94])
    out = err.replay_rules(bars, levels=((-5.0, 0.001),))
    assert out["legs"] == []
