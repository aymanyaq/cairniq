"""Regression-based factor exposures (Advisor Roadmap 4.2).

This replaces a label counter that reported "40% Growth" with the confidence a
measured loading would carry. So the tests are weighted toward the thing that
makes a regression honest and a count dishonest: **the standard error**. An
exposure of 0.31 that cannot be distinguished from zero and one that can are
different findings, and only one should move a position.

All price data is injected — offline by construction.
"""
import numpy as np
import pandas as pd
import pytest

import tools.factor_exposures as fx


def _frame(cols: dict[str, np.ndarray]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(next(iter(cols.values()))), freq="B")
    return pd.DataFrame(cols, index=idx)


def _run(portfolio, factors, base="USD", fx_series=None):
    port = _frame({"A": portfolio})
    port.attrs["fx"] = {"base_currency": base}
    return fx.estimate_factor_exposures(
        ["A"],
        returns_fn=lambda syms, per: (port, ["A"]),
        factors_fn=lambda per: _frame(factors),
        fx_fn=lambda b, p: (pd.Series(fx_series, index=port.index) if fx_series is not None else None),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# ---------------------------------------------------------------------------
# The standard error is the point
# ---------------------------------------------------------------------------

def test_a_real_exposure_is_recovered_and_marked_significant(rng):
    """A portfolio built as 1.5x a factor should show beta ~1.5 with a large t."""
    n = 400
    market = rng.normal(0, 0.01, n)
    noise = rng.normal(0, 0.001, n)
    result = _run(1.5 * market + noise, {"market": market})

    beta = next(e for e in result["exposures"] if e["factor"] == "market")
    assert beta["beta"] == pytest.approx(1.5, abs=0.05)
    assert beta["significant"] is True
    assert abs(beta["t_stat"]) > 10


def test_pure_noise_is_reported_as_NOT_distinguishable_from_zero(rng):
    """The finding the label counter could never make. A portfolio unrelated to
    a factor must not be given a tilt just because the point estimate is
    non-zero — every regression produces a non-zero point estimate."""
    n = 400
    result = _run(rng.normal(0, 0.01, n), {"market": rng.normal(0, 0.01, n)})

    beta = next(e for e in result["exposures"] if e["factor"] == "market")
    assert beta["significant"] is False
    assert "should not be acted on" in beta["reading"]
    assert any("not a gap" in w for w in result["warnings"])


def test_the_reading_for_an_insignificant_beta_names_the_t_stat(rng):
    n = 300
    result = _run(rng.normal(0, 0.01, n), {"market": rng.normal(0, 0.01, n)})
    beta = next(e for e in result["exposures"] if e["factor"] == "market")

    assert "t=" in beta["reading"]
    assert beta["p_value"] is not None


def test_significant_exposures_sort_above_insignificant_ones(rng):
    n = 400
    real = rng.normal(0, 0.01, n)
    junk = rng.normal(0, 0.01, n)
    result = _run(2.0 * real + rng.normal(0, 0.001, n), {"market": real, "size": junk})

    assert result["exposures"][0]["factor"] == "market"
    assert result["exposures"][0]["significant"] is True
    assert result["exposures"][-1]["significant"] is False


# ---------------------------------------------------------------------------
# Sample size — refuse rather than emit a table of non-findings
# ---------------------------------------------------------------------------

def test_too_few_observations_refuses_instead_of_reporting_wide_estimates(rng):
    n = 30
    result = _run(rng.normal(0, 0.01, n), {"market": rng.normal(0, 0.01, n)})

    assert "error" in result
    assert result["observations"] == n
    assert "squinting at point" in result["error"]


def test_the_observation_count_is_always_reported(rng):
    n = 300
    result = _run(rng.normal(0, 0.01, n), {"market": rng.normal(0, 0.01, n)})

    assert result["observations"] == n


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------

def test_variance_shares_sum_to_r_squared(rng):
    """The property that makes the shares readable as "portion of the movement
    explained" even when the factors are correlated. Verified on real data too:
    a concentrated two-equity-plus-bond book gave shares summing to 71.8%
    against R²=0.717.
    """
    n = 500
    f1 = rng.normal(0, 0.01, n)
    f2 = rng.normal(0, 0.01, n)
    port = 0.8 * f1 + 0.4 * f2 + rng.normal(0, 0.004, n)
    result = _run(port, {"market": f1, "value": f2})

    total = sum(e["variance_share_pct"] for e in result["exposures"])
    assert total == pytest.approx(result["r_squared"] * 100, abs=0.5)


def test_unexplained_variance_is_reported_not_hidden(rng):
    n = 400
    f1 = rng.normal(0, 0.01, n)
    result = _run(0.5 * f1 + rng.normal(0, 0.02, n), {"market": f1})

    assert result["unexplained_variance_pct"] > 0
    assert result["unexplained_variance_pct"] == pytest.approx(
        (1 - result["r_squared"]) * 100, abs=0.2
    )


# ---------------------------------------------------------------------------
# Alpha and currency
# ---------------------------------------------------------------------------

def test_a_portfolio_with_no_drift_reports_alpha_as_indistinguishable(rng):
    """Residuals are explicitly de-meaned so alpha is genuinely zero.

    Worth recording why: a first version of this test used raw noise at
    sigma=0.002 over n=400, where the intercept's standard error is ~0.0001 — so
    the tiny random drift in the sample WAS significant, and the test failed
    correctly. The estimator was right and the expectation was wrong. Detecting
    a real 5bp/yr drift at that precision is the feature, not a bug.
    """
    n = 400
    f1 = rng.normal(0, 0.01, n)
    residual = rng.normal(0, 0.002, n)
    residual = residual - residual.mean()
    result = _run(0.9 * f1 + residual, {"market": f1})

    assert result["alpha_significant"] is False
    assert "explained by its factor exposures, not by selection" in result["alpha_reading"]


def test_a_portfolio_with_genuine_drift_reports_alpha_as_significant(rng):
    """The mirror. A real, persistent excess return must not be explained away."""
    n = 400
    f1 = rng.normal(0, 0.01, n)
    residual = rng.normal(0, 0.002, n)
    daily_alpha = 0.0004                      # ~10%/yr
    port = 0.9 * f1 + (residual - residual.mean()) + daily_alpha

    result = _run(port, {"market": f1})

    assert result["alpha_significant"] is True
    assert result["alpha_annualized_pct"] == pytest.approx(daily_alpha * 252 * 100, abs=1.0)
    assert "distinguishable from zero" in result["alpha_reading"]


def test_currency_enters_as_its_own_factor_for_a_non_usd_base(rng):
    """A CAD holder of US assets carries real FX exposure. Converting the factor
    proxies too would smear the same move across every column and hide it inside
    "market beta" — verified live: a CAD book of US growth names shows an fx beta
    of +0.81."""
    n = 400
    market = rng.normal(0, 0.01, n)
    usdcad = rng.normal(0, 0.004, n)
    port = 0.7 * market + 0.9 * usdcad + rng.normal(0, 0.001, n)

    result = _run(port, {"market": market}, base="CAD", fx_series=usdcad)

    fx_row = next(e for e in result["exposures"] if e["factor"] == "fx")
    assert fx_row["beta"] == pytest.approx(0.9, abs=0.1)
    assert fx_row["significant"] is True
    assert result["base_currency"] == "CAD"


def test_no_fx_factor_when_the_base_currency_is_usd(rng):
    n = 300
    f1 = rng.normal(0, 0.01, n)
    result = _run(f1, {"market": f1}, base="USD", fx_series=None)

    assert all(e["factor"] != "fx" for e in result["exposures"])


# ---------------------------------------------------------------------------
# Honesty about the estimate itself
# ---------------------------------------------------------------------------

def test_collinear_factors_are_flagged_rather_than_silently_unstable(rng):
    """Raw style ETFs are ~0.9 correlated with the market, which is why the
    factors are built as spreads. When collinearity survives anyway, individual
    betas are unstable and the reader must be told."""
    n = 400
    f1 = rng.normal(0, 0.01, n)
    near_duplicate = f1 + rng.normal(0, 0.00001, n)
    result = _run(f1 + rng.normal(0, 0.002, n), {"market": f1, "quality": near_duplicate})

    assert any("collinear" in w for w in result["warnings"])


def test_results_are_marked_measured_and_state_their_method():
    n = 300
    rng = np.random.default_rng(1)
    f1 = rng.normal(0, 0.01, n)
    result = _run(f1, {"market": f1})

    assert result["basis"] == "measured"
    assert "t >= 2" in result["method"]


def test_the_superseded_label_counter_now_marks_itself_as_a_heuristic():
    """`analyze_factors` is kept for its descriptive breakdown, but prose built
    on it must not present a label tally as a measured loading (2.7)."""
    import inspect

    from tools.portfolio_analytics import analyze_factors

    source = inspect.getsource(analyze_factors)
    assert '"basis": "label heuristic"' in source
    assert "estimate_factor_exposures" in source


def test_missing_holdings_are_named_rather_than_quietly_dropped():
    port = _frame({"A": np.random.default_rng(3).normal(0, 0.01, 300)})
    port.attrs["fx"] = {"base_currency": "USD"}
    result = fx.estimate_factor_exposures(
        ["A", "DELISTED"],
        returns_fn=lambda syms, per: (port, ["A"]),
        factors_fn=lambda per: _frame({"market": np.random.default_rng(4).normal(0, 0.01, 300)}),
        fx_fn=lambda b, p: None,
    )

    assert any("DELISTED" in w for w in result["warnings"])
