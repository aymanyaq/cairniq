"""CVaR and geometric CAGR (Theme 4.1).

Checks the estimators against closed-form/analytic values rather than
snapshotting whatever the code currently emits.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import tools.portfolio_analytics as pa


def _pct(text: str) -> float:
    """'-1.23%' -> -0.0123"""
    return float(text.rstrip("%")) / 100.0


def _dollars(text: str) -> float:
    return float(text.replace("$", "").replace(",", ""))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(pa, "_get_risk_free_rate", lambda: 0.04)


def _single_asset(returns_series: pd.Series, monkeypatch):
    """Patch _get_returns so the portfolio return series IS returns_series."""
    df = returns_series.to_frame(name="X")

    def fake(symbols, period="1y"):
        valid = [s for s in symbols if s in df.columns]
        if not valid:
            return pd.DataFrame(), []
        return df[valid], valid

    monkeypatch.setattr(pa, "_get_returns", fake)


class TestCvar:
    def test_parametric_cvar_matches_closed_form(self, monkeypatch):
        rng = np.random.default_rng(11)
        series = pd.Series(rng.normal(0.0005, 0.012, size=500))
        _single_asset(series, monkeypatch)

        result = pa.calculate_var(["X"], [1.0], confidence=0.95, investment=100_000)

        mu, sigma = series.mean(), series.std()
        alpha = 0.05
        z = stats.norm.ppf(alpha)
        expected_cvar = -(mu - sigma * stats.norm.pdf(z) / alpha)

        assert _pct(result["conditional_value_at_risk"]["daily_cvar_pct"]) == pytest.approx(
            expected_cvar, abs=5e-5
        )

    def test_cvar_exceeds_var(self, monkeypatch):
        rng = np.random.default_rng(12)
        _single_asset(pd.Series(rng.normal(0.0005, 0.012, size=500)), monkeypatch)

        result = pa.calculate_var(["X"], [1.0], confidence=0.95, investment=100_000)
        var = _pct(result["value_at_risk"]["daily_var_pct"])
        cvar = _pct(result["conditional_value_at_risk"]["daily_cvar_pct"])
        # Expected shortfall is an average *beyond* the threshold, so it is
        # always the larger loss.
        assert cvar > var

    def test_normal_cvar_var_ratio_is_about_1_25_at_95pct(self, monkeypatch):
        # For a zero-mean normal at 95%: VaR = 1.645s, CVaR = 2.063s -> ~1.254.
        rng = np.random.default_rng(13)
        _single_asset(pd.Series(rng.normal(0.0, 0.01, size=20_000)), monkeypatch)

        result = pa.calculate_var(["X"], [1.0], confidence=0.95, investment=100_000)
        ratio = (
            _pct(result["conditional_value_at_risk"]["daily_cvar_pct"])
            / _pct(result["value_at_risk"]["daily_var_pct"])
        )
        assert ratio == pytest.approx(2.0627 / 1.6449, rel=0.02)

    def test_historical_cvar_is_mean_of_the_tail(self, monkeypatch):
        rng = np.random.default_rng(14)
        series = pd.Series(rng.normal(0.0005, 0.012, size=500))
        _single_asset(series, monkeypatch)

        result = pa.calculate_var(["X"], [1.0], confidence=0.95, investment=100_000)

        threshold = np.percentile(series, 5)
        expected = -series[series <= threshold].mean()
        assert _pct(
            result["conditional_value_at_risk"]["historical_daily_cvar_pct"]
        ) == pytest.approx(expected, abs=5e-5)

    def test_fat_tails_widen_the_cvar_var_gap(self, monkeypatch):
        """The reason CVaR earns its place: it reacts to tail shape, VaR barely does."""
        rng = np.random.default_rng(15)
        normal = pd.Series(rng.normal(0, 0.01, size=5000))
        fat = pd.Series(rng.standard_t(df=3, size=5000) * 0.01 / np.sqrt(3))

        _single_asset(normal, monkeypatch)
        r_normal = pa.calculate_var(["X"], [1.0], investment=100_000)
        _single_asset(fat, monkeypatch)
        r_fat = pa.calculate_var(["X"], [1.0], investment=100_000)

        def hist_gap(r):
            return (
                _pct(r["conditional_value_at_risk"]["historical_daily_cvar_pct"])
                - _pct(r["value_at_risk"]["historical_daily_var_pct"])
            )

        assert hist_gap(r_fat) > hist_gap(r_normal)

    def test_dollar_cvar_tracks_investment(self, monkeypatch):
        rng = np.random.default_rng(16)
        _single_asset(pd.Series(rng.normal(0.0005, 0.012, size=500)), monkeypatch)

        result = pa.calculate_var(["X"], [1.0], confidence=0.95, investment=250_000)
        cvar_pct = _pct(result["conditional_value_at_risk"]["daily_cvar_pct"])
        cvar_dollars = _dollars(result["conditional_value_at_risk"]["daily_cvar_dollars"])
        assert cvar_dollars == pytest.approx(cvar_pct * 250_000, rel=0.01)

    def test_interpretation_mentions_cvar(self, monkeypatch):
        rng = np.random.default_rng(17)
        _single_asset(pd.Series(rng.normal(0.0005, 0.012, size=500)), monkeypatch)
        result = pa.calculate_var(["X"], [1.0], investment=100_000)
        assert "CVaR" in result["interpretation"]


class TestGeometricCagr:
    def test_cagr_matches_known_compounding(self, monkeypatch):
        # Exactly 252 days of a constant +0.1%/day => (1.001^252 - 1).
        series = pd.Series([0.001] * 252)
        _single_asset(series, monkeypatch)

        result = pa.calculate_portfolio_metrics(["X"], [1.0])
        expected = 1.001**252 - 1
        assert _pct(result["metrics"]["cagr"]) == pytest.approx(expected, abs=1e-3)

    def test_volatility_drag_textbook_case(self, monkeypatch):
        """+10%/-10% alternating: arithmetic says breakeven, compounding says ruin.

        This is the whole reason CAGR is reported — mean x 252 calls this
        portfolio flat while it has actually lost most of its value.
        """
        _single_asset(pd.Series([0.10, -0.10] * 126), monkeypatch)

        result = pa.calculate_portfolio_metrics(["X"], [1.0])
        assert _pct(result["metrics"]["annual_return"]) == pytest.approx(0.0, abs=1e-9)
        assert _pct(result["metrics"]["cagr"]) == pytest.approx(0.99**126 - 1, abs=1e-3)
        assert _pct(result["metrics"]["cagr"]) < -0.5

    def test_drag_grows_with_variance_at_equal_arithmetic_mean(self, monkeypatch):
        """Holding the mean fixed isolates the drag from the compounding gap.

        Both series have an identical arithmetic mean (so an identical
        `annual_return`); only the variance differs, so any CAGR difference is
        volatility drag alone.
        """
        calm = pd.Series([0.0004] * 252)
        wild = pd.Series([0.0004 + 0.02, 0.0004 - 0.02] * 126)

        _single_asset(calm, monkeypatch)
        r_calm = pa.calculate_portfolio_metrics(["X"], [1.0])
        _single_asset(wild, monkeypatch)
        r_wild = pa.calculate_portfolio_metrics(["X"], [1.0])

        assert _pct(r_calm["metrics"]["annual_return"]) == pytest.approx(
            _pct(r_wild["metrics"]["annual_return"]), abs=1e-9
        )
        assert _pct(r_wild["metrics"]["cagr"]) < _pct(r_calm["metrics"]["cagr"])

    def test_negative_series_yields_negative_cagr(self, monkeypatch):
        _single_asset(pd.Series([-0.001] * 252), monkeypatch)
        result = pa.calculate_portfolio_metrics(["X"], [1.0])
        assert _pct(result["metrics"]["cagr"]) < 0

    def test_total_wipeout_reports_na_not_a_crash(self, monkeypatch):
        # A -100% day drives cumulative value to zero; CAGR is undefined there.
        series = pd.Series([0.01] * 50 + [-1.0] + [0.0] * 10)
        _single_asset(series, monkeypatch)

        result = pa.calculate_portfolio_metrics(["X"], [1.0])
        assert result["metrics"]["cagr"] == "N/A"

    def test_both_annualizations_are_reported_and_documented(self, monkeypatch):
        rng = np.random.default_rng(22)
        _single_asset(pd.Series(rng.normal(0.0004, 0.01, size=252)), monkeypatch)

        result = pa.calculate_portfolio_metrics(["X"], [1.0])
        assert "annual_return" in result["metrics"]
        assert "cagr" in result["metrics"]
        # The advisor must be able to tell the reader which is which.
        assert "Arithmetic" in result["metric_notes"]["annual_return"]
        assert "Geometric" in result["metric_notes"]["cagr"]
