"""Covariance estimators (Theme 4.1): Ledoit-Wolf shrinkage, EWMA, sample.

The Ledoit-Wolf tests cross-check against sklearn's reference implementation
when sklearn is importable. sklearn is deliberately NOT a declared dependency
of this project — tools/covariance.py implements the estimator on numpy — so
those checks skip rather than fail where it is absent.
"""
import numpy as np
import pandas as pd
import pytest

from tools.covariance import (
    estimate_covariance,
    ewma_covariance,
    ledoit_wolf_shrinkage,
)


def _returns(n_obs: int = 250, n_assets: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Correlated returns: a common factor plus idiosyncratic noise.
    factor = rng.normal(0, 0.01, size=(n_obs, 1))
    loadings = rng.uniform(0.5, 1.5, size=(1, n_assets))
    noise = rng.normal(0, 0.005, size=(n_obs, n_assets))
    data = factor @ loadings + noise
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])


class TestLedoitWolf:
    def test_matches_sklearn_reference(self):
        sk = pytest.importorskip("sklearn.covariance", reason="sklearn is not a project dependency")
        returns = _returns()
        cov, shrinkage = ledoit_wolf_shrinkage(returns)

        ref_cov, ref_shrinkage = sk.ledoit_wolf(returns.to_numpy())
        np.testing.assert_allclose(cov.to_numpy(), ref_cov, rtol=1e-10, atol=1e-16)
        assert shrinkage == pytest.approx(ref_shrinkage, rel=1e-10)

    def test_matches_sklearn_when_assets_exceed_observations(self):
        """The regime shrinkage exists for: N > T makes the sample matrix singular."""
        sk = pytest.importorskip("sklearn.covariance", reason="sklearn is not a project dependency")
        returns = _returns(n_obs=20, n_assets=30)
        cov, shrinkage = ledoit_wolf_shrinkage(returns)

        ref_cov, ref_shrinkage = sk.ledoit_wolf(returns.to_numpy())
        np.testing.assert_allclose(cov.to_numpy(), ref_cov, rtol=1e-9, atol=1e-16)
        assert shrinkage == pytest.approx(ref_shrinkage, rel=1e-9)

    def test_shrinkage_within_unit_interval(self):
        for n_obs, n_assets in [(250, 5), (60, 40), (20, 30), (10, 3)]:
            _, shrinkage = ledoit_wolf_shrinkage(_returns(n_obs, n_assets))
            assert 0.0 <= shrinkage <= 1.0

    def test_output_is_symmetric_and_psd(self):
        cov, _ = ledoit_wolf_shrinkage(_returns(n_obs=30, n_assets=25))
        M = cov.to_numpy()
        np.testing.assert_allclose(M, M.T, rtol=1e-12)
        # Shrinkage toward a positive-multiple-of-identity target repairs the
        # singularity a sample matrix has when N > T.
        assert np.linalg.eigvalsh(M).min() > 0

    def test_shrinkage_rises_as_observations_shrink(self):
        _, plenty = ledoit_wolf_shrinkage(_returns(n_obs=2000, n_assets=8))
        _, scarce = ledoit_wolf_shrinkage(_returns(n_obs=25, n_assets=8))
        assert scarce > plenty

    def test_preserves_column_labels(self):
        returns = _returns(n_assets=3)
        cov, _ = ledoit_wolf_shrinkage(returns)
        assert list(cov.columns) == ["A0", "A1", "A2"]
        assert list(cov.index) == ["A0", "A1", "A2"]

    def test_single_asset_has_nothing_to_shrink(self):
        cov, shrinkage = ledoit_wolf_shrinkage(_returns(n_assets=1))
        assert shrinkage == 0.0
        assert cov.shape == (1, 1)

    def test_rejects_insufficient_observations(self):
        with pytest.raises(ValueError, match="at least 2 observations"):
            ledoit_wolf_shrinkage(pd.DataFrame({"A": [0.01]}))


class TestEwma:
    def test_recent_observations_dominate(self):
        # Calm for a long stretch, then a volatile tail.
        rng = np.random.default_rng(3)
        calm = rng.normal(0, 0.001, size=(200, 2))
        wild = rng.normal(0, 0.05, size=(40, 2))
        returns = pd.DataFrame(np.vstack([calm, wild]), columns=["A", "B"])

        ewma = ewma_covariance(returns, decay=0.94)
        sample = returns.cov()
        # EWMA weights the volatile tail more heavily than an equal-weighted mean.
        assert ewma.loc["A", "A"] > sample.loc["A", "A"]

    def test_decay_near_one_approaches_sample_covariance(self):
        returns = _returns(n_obs=100, n_assets=3)
        ewma = ewma_covariance(returns, decay=0.999999)
        # With near-flat weights this is the MLE (1/T) covariance; pandas .cov()
        # uses 1/(T-1), hence the T/(T-1) rescale before comparing.
        T = len(returns)
        expected = returns.cov().to_numpy() * (T - 1) / T
        np.testing.assert_allclose(ewma.to_numpy(), expected, rtol=1e-4, atol=1e-12)

    def test_output_is_symmetric(self):
        cov = ewma_covariance(_returns(n_assets=4))
        np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T, rtol=1e-12)

    @pytest.mark.parametrize("decay", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_out_of_range_decay(self, decay):
        with pytest.raises(ValueError, match="decay must be in"):
            ewma_covariance(_returns(), decay=decay)


class TestEstimateCovariance:
    def test_default_is_ledoit_wolf_and_reports_shrinkage(self):
        cov, meta = estimate_covariance(_returns())
        assert meta["method"] == "ledoit_wolf"
        assert 0.0 <= meta["shrinkage"] <= 1.0
        assert meta["observations"] == 250
        assert meta["assets"] == 5
        assert cov.shape == (5, 5)

    def test_sample_method_matches_pandas(self):
        returns = _returns(n_assets=3)
        cov, meta = estimate_covariance(returns, method="sample")
        assert meta["method"] == "sample"
        np.testing.assert_allclose(cov.to_numpy(), returns.cov().to_numpy(), rtol=1e-12)

    def test_sample_and_shrunk_differ(self):
        returns = _returns(n_obs=40, n_assets=15)
        shrunk, _ = estimate_covariance(returns, method="ledoit_wolf")
        sample, _ = estimate_covariance(returns, method="sample")
        assert not np.allclose(shrunk.to_numpy(), sample.to_numpy())

    def test_ewma_method_reports_decay(self):
        cov, meta = estimate_covariance(_returns(), method="ewma", decay=0.9)
        assert meta["method"] == "ewma"
        assert meta["decay"] == 0.9
        assert cov.shape == (5, 5)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            estimate_covariance(_returns(), method="magic")

    def test_drops_rows_with_missing_data(self):
        returns = _returns(n_obs=50, n_assets=3)
        returns.iloc[5, 1] = np.nan
        _, meta = estimate_covariance(returns)
        assert meta["observations"] == 49

    def test_estimator_failure_falls_back_to_sample(self, monkeypatch):
        import tools.covariance as cvm

        def boom(returns):
            raise RuntimeError("degenerate input")

        monkeypatch.setattr(cvm, "ledoit_wolf_shrinkage", boom)
        returns = _returns(n_assets=3)
        cov, meta = estimate_covariance(returns, method="ledoit_wolf")

        # A risk tool must still get a usable matrix, with the reason recorded.
        assert meta["method"] == "sample"
        assert "degenerate input" in meta["fallback_reason"]
        np.testing.assert_allclose(cov.to_numpy(), returns.cov().to_numpy(), rtol=1e-12)


class TestMarginalRiskWiring:
    @pytest.fixture
    def returns(self):
        return _returns(n_obs=120, n_assets=4)

    def _patch(self, monkeypatch, returns):
        import tools.portfolio_analytics as pa
        cols = list(returns.columns)
        monkeypatch.setattr(
            pa, "_get_returns",
            lambda symbols, period="1y": (returns[[s for s in symbols if s in cols]],
                                          [s for s in symbols if s in cols]),
        )
        return pa

    def test_uses_shrinkage_by_default(self, monkeypatch, returns):
        pa = self._patch(monkeypatch, returns)
        result = pa.estimate_marginal_risk_contribution(
            ["A0", "A1", "A2"], [0.4, 0.3, 0.3], "A3", candidate_weight=0.1,
        )
        meta = result["covariance_estimator"]
        assert meta["method"] == "ledoit_wolf"
        assert 0.0 <= meta["shrinkage"] <= 1.0

    def test_honors_explicit_sample_method(self, monkeypatch, returns):
        pa = self._patch(monkeypatch, returns)
        result = pa.estimate_marginal_risk_contribution(
            ["A0", "A1", "A2"], [0.4, 0.3, 0.3], "A3", candidate_weight=0.1,
            cov_method="sample",
        )
        assert result["covariance_estimator"]["method"] == "sample"
