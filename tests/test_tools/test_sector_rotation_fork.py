"""Two live functions named `detect_sector_rotation`, and they must stay two.

`tools.market_mechanics` and `tools.sector_rotation` each define one. Both are
correct; they answer different questions:

    market_mechanics -> TREND QUADRANT   "is this up over 1M and 3M?"
                        Leading / Weakening / Improving / Lagging
    sector_rotation  -> FLOW ACCELERATION "is 1M outpacing the 3M run rate?"
                        INFLOW / OUTFLOW / NEUTRAL

The hazard is the shared NAME: an import of the wrong one silently changes the
question being asked, and nothing downstream would notice. These tests pin the
distinction so it survives, and so nobody "fixes" the apparent contradiction by
collapsing them — a sector up 12% over three months and 1% over the last one is
genuinely BOTH leading on trend and decelerating on flow. Collapsing them would
delete a real signal.
"""
import numpy as np
import pandas as pd
import pytest

import tools.market_mechanics as mm
import tools.sector_rotation as sr

SECTORS = list(mm.SECTOR_ETFS.keys())


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr("tools.daily_cache.get_cached", lambda *a, **k: None)
    monkeypatch.setattr("tools.daily_cache.set_cached", lambda *a, **k: None)


def _leading_but_decelerating(length: int = 90) -> pd.Series:
    """Strong 3M, weak 1M — the case the two engines classify oppositely.

    3M return ~ +12%, 1M return ~ +1%. Trend says both windows are up
    (Leading); flow says 1% badly lags the 12%/3 run rate (OUTFLOW).
    """
    idx = pd.date_range("2025-01-01", periods=length, freq="B")
    v = np.empty(length, dtype=float)
    v[: length - 66] = 100.0
    v[length - 66 : length - 22] = np.linspace(100.0, 111.0, 44)
    v[length - 22 :] = np.linspace(111.0, 112.11, 22)
    return pd.Series(v, index=idx)


def _run_trend_engine(monkeypatch, series):
    frame = pd.DataFrame(
        {("Close", s): series for s in SECTORS},
        index=series.index,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(mm.yf, "download", lambda *a, **k: frame)
    return mm.detect_sector_rotation()


def _run_flow_engine(monkeypatch, series):
    hist = pd.DataFrame({"Close": series}, index=series.index)

    class _T:
        def __init__(self, symbol):
            pass

        def history(self, *a, **k):
            return hist

    monkeypatch.setattr(sr.yf, "Ticker", _T)
    return sr.detect_sector_rotation()


def test_each_payload_states_which_question_it_answered(monkeypatch):
    series = _leading_but_decelerating()
    trend = _run_trend_engine(monkeypatch, series)
    flow = _run_flow_engine(monkeypatch, series)

    assert trend["methodology"] == "trend_quadrant"
    assert flow["methodology"] == "flow_acceleration"
    # A consumer holding one of these must be able to tell which it has.
    assert trend["methodology"] != flow["methodology"]


def test_the_two_engines_disagree_on_the_same_prices_and_that_is_correct(monkeypatch):
    """The exact shape the audit called a contradiction. It is not one."""
    series = _leading_but_decelerating()
    trend = _run_trend_engine(monkeypatch, series)
    flow = _run_flow_engine(monkeypatch, series)

    trend_labels = {r["sector"]: r["trend"] for r in trend["full_rotation_map"]}
    flow_labels = {r["sector"]: r["signal"] for r in flow["sector_performance"]}

    shared = set(trend_labels) & set(flow_labels)
    assert shared, "engines returned no sector in common — fixture is wrong"

    for sector in shared:
        assert "Leading" in trend_labels[sector], trend_labels[sector]
        assert "OUTFLOW" in flow_labels[sector], flow_labels[sector]

    # Both readings come off the SAME series: up over both windows, but the last
    # month badly lags the three-month run rate. Unifying these would have to
    # discard one of two true statements.


def test_the_scanner_reads_both_shapes(monkeypatch):
    """_inflowing_sectors is the one consumer that takes either payload."""
    from tools.opportunity_scanner import _inflowing_sectors

    trend_shape = {"full_rotation_map": [
        {"sector": "Technology", "trend": "Leading 🟢"},
        {"sector": "Utilities", "trend": "Lagging 🔴"},
    ]}
    flow_shape = {"sector_performance": [
        {"sector": "Energy", "momentum_score": 4.2},
        {"sector": "Real Estate", "momentum_score": -3.1},
    ]}

    assert _inflowing_sectors(trend_shape) == ["Technology"]
    assert _inflowing_sectors(flow_shape) == ["Energy"]


def test_the_quick_action_prompt_names_a_tool_that_actually_exists():
    """The scan button used to tell the model to call `detect_sector_rotation`.

    That is the internal function name; the REGISTERED tool is
    `check_sector_rotation`. Tool binding is by registered name, so the model was
    being pointed at something it could not call.
    """
    from pathlib import Path

    html = Path("templates/index.html").read_text()
    assert "check_sector_rotation" in html
    assert "detect_sector_rotation" not in html
