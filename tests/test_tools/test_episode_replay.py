"""Historical episode replay (Advisor Roadmap 4.3).

This replaces two shipped tools that produce confident figures from hand-typed
constants, so the tests are weighted toward the ways a replay can go back to
being fiction: silently dropping the holdings that lack history (which flatters
the drawdown), measuring the fall from the wrong anchor (which also flatters it),
and reporting a number without saying what it could not see.

Price data is injected, so this suite is offline by construction rather than by
mocking a live call out from under itself.
"""
import numpy as np
import pandas as pd
import pytest

import tools.episode_replay as er


def _series(paths: dict[str, list[float]], start="2020-02-20"):
    """Daily returns frame from explicit per-symbol return paths."""
    idx = pd.date_range(start, periods=len(next(iter(paths.values()))), freq="D")
    return pd.DataFrame(paths, index=idx)


def _fn(frame):
    """A returns_fn that yields `frame` regardless of window."""
    return lambda syms, start, end: (frame[[c for c in frame.columns if c in syms]],
                                     [c for c in frame.columns if c in syms])


# ---------------------------------------------------------------------------
# Coverage — the failure that would make this fiction again
# ---------------------------------------------------------------------------

def test_a_replay_below_the_coverage_floor_refuses_rather_than_reports():
    """Holdings that lack history in the window are typically the newest and
    most volatile. Dropping them and renormalising returns a FLATTERING
    drawdown that reads as measured — so below the floor there is no number."""
    frame = _series({"OLD": [-0.02] * 10})
    result = er.replay_episode(
        ["OLD", "NEW1", "NEW2", "NEW3"], [0.25] * 4, episode="gfc", returns_fn=_fn(frame)
    )

    assert "error" in result
    assert result["coverage_pct"] == 25.0
    assert result["missing_symbols"] == ["NEW1", "NEW2", "NEW3"]
    assert "understate" in result["error"]


def test_coverage_is_measured_by_WEIGHT_not_by_count():
    """Three tiny positions with history and one huge one without is NOT 75%
    covered — it is 25%, and the missing quarter is most of the portfolio."""
    frame = _series({"A": [-0.01] * 10, "B": [-0.01] * 10, "C": [-0.01] * 10})
    result = er.replay_episode(
        ["A", "B", "C", "BIG"], [0.05, 0.05, 0.05, 0.85],
        episode="covid", returns_fn=_fn(frame),
    )

    assert "error" in result
    assert result["coverage_pct"] == 15.0


def test_a_partial_but_sufficient_replay_names_what_it_could_not_see():
    frame = _series({"A": [-0.02] * 10, "B": [-0.02] * 10})
    result = er.replay_episode(
        ["A", "B", "GONE"], [0.4, 0.4, 0.2], episode="covid", returns_fn=_fn(frame)
    )

    assert "error" not in result
    assert result["coverage_pct"] == 80.0
    assert result["missing_symbols"] == ["GONE"]
    assert "floor on the drawdown" in result["data_warning"]


def test_no_history_at_all_is_an_error_not_a_zero_percent_drawdown():
    empty = pd.DataFrame()
    result = er.replay_episode(["X"], [1.0], episode="gfc",
                               returns_fn=lambda s, a, b: (empty, []))

    assert "error" in result
    assert result["coverage_pct"] == 0.0


# ---------------------------------------------------------------------------
# The drawdown anchor — caught on real data, always flattering when wrong
# ---------------------------------------------------------------------------

def test_drawdown_is_measured_from_the_episode_peak_not_a_running_max():
    """The window starts AT the peak by construction, so the cumulative product
    is already the level relative to it. Using an expanding max instead
    re-anchors on the first day after the peak — which in any episode that falls
    immediately is already below it, understating the fall. Measured on a real
    50/30/20 SPY/AGG/QQQ book through COVID the gap was 0.5pp, in the flattering
    direction every time.

    Here: a straight -10% then -10% fall is -19%, not the -10% a running max
    anchored on day one would report.
    """
    frame = _series({"A": [-0.10, -0.10]})
    result = er.replay_episode(["A"], [1.0], episode="covid", returns_fn=_fn(frame))

    assert result["peak_to_trough_pct"] == pytest.approx(-19.0, abs=0.1)


def test_recovery_means_back_to_the_peak_not_to_a_mid_decline_high():
    """-50% then +50% is still -25%. A recovery test anchored on a local high
    would call that recovered."""
    frame = _series({"A": [-0.5, 0.5, 0.0]})
    result = er.replay_episode(["A"], [1.0], episode="covid", returns_fn=_fn(frame))

    assert result["days_to_recover"] is None
    assert "had not regained" in result["recovery_note"]


def test_a_genuine_recovery_is_dated_and_counted():
    frame = _series({"A": [-0.20, 0.30, 0.0]})
    result = er.replay_episode(["A"], [1.0], episode="covid", returns_fn=_fn(frame))

    assert result["days_to_recover"] is not None
    assert result["portfolio_recovered_on"] is not None


# ---------------------------------------------------------------------------
# What it reports
# ---------------------------------------------------------------------------

def test_worst_positions_are_ranked_by_actual_episode_return():
    frame = _series({"MILD": [-0.01] * 5, "BAD": [-0.10] * 5, "UP": [0.02] * 5})
    result = er.replay_episode(["MILD", "BAD", "UP"], [1 / 3] * 3,
                               episode="covid", returns_fn=_fn(frame))

    assert result["worst_positions"][0]["symbol"] == "BAD"
    assert result["worst_positions"][-1]["symbol"] == "UP"


def test_a_pair_that_is_already_tight_is_reported_on_LEVEL_not_only_on_CHANGE():
    """Caught on real data: SPY/QQQ went 0.96 -> 0.99 through COVID, and an
    earlier version reported 'diversification held' because the DELTA was small.
    Two sleeves at 0.99 are one position whether they moved or not."""
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 0.02, 60)
    frame = _series({
        "X": (shared + rng.normal(0, 0.001, 60)).tolist(),
        "Y": (shared + rng.normal(0, 0.001, 60)).tolist(),
    })
    result = er.replay_episode(["X", "Y"], [0.5, 0.5], episode="covid",
                               returns_fn=_fn(frame))

    assert result["largest_correlation_shift"]["in_drawdown"] >= 0.85
    assert "one position, not two" in result["correlation_note"]


def test_results_are_marked_as_measured_so_prose_can_attribute_them():
    """Roadmap 2.7: the tools this replaces return authored constants through
    fields that read as measurements. Everything here came from price data and
    says so."""
    frame = _series({"A": [-0.05] * 5})
    result = er.replay_episode(["A"], [1.0], episode="covid", returns_fn=_fn(frame))

    assert result["basis"] == "measured"


def test_an_unknown_episode_is_rejected_rather_than_guessed():
    result = er.replay_episode(["A"], [1.0], episode="the_bad_one")

    assert "Unknown episode" in result["error"]


def test_episode_dates_are_the_only_constants_and_they_are_declared():
    """Every other number is measured at run time. These dates are facts about
    the S&P 500 and are stated openly rather than baked into a multiplier."""
    for key, ep in er.EPISODES.items():
        assert ep["peak"] < ep["trough"] < ep["recovered"], key
        assert ep["note"]


# ---------------------------------------------------------------------------
# replay_all
# ---------------------------------------------------------------------------

def test_replay_all_separates_what_it_could_replay_from_what_it_could_not():
    frame = _series({"A": [-0.05] * 5})

    def partial(syms, start, end):
        # Only the modern episodes have data for this book.
        if start < "2015":
            return pd.DataFrame(), []
        return frame[[c for c in frame.columns if c in syms]], ["A"]

    out = er.replay_all_episodes(["A"], [1.0], returns_fn=partial)

    assert "covid" in out["replayed"] and "bear_2022" in out["replayed"]
    assert "gfc" in out["not_replayable"] and "dotcom" in out["not_replayable"]


def test_replay_all_does_not_average_into_one_expected_crash_number():
    """Averaging an episode this book could be measured through with one it
    could not is exactly the confident composite this module exists to remove."""
    frame = _series({"A": [-0.05] * 5})
    out = er.replay_all_episodes(["A"], [1.0], returns_fn=_fn(frame))

    assert "expected_drawdown" not in out
    assert "average" not in str(out.keys())
    assert out["worst_episode"]["peak_to_trough_pct"] < 0
