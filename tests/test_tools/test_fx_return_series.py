"""Base-currency return series (Theme 4.1): FX conversion inside _get_returns.

Mixed .TO/US portfolios must have their return series measured in the profile's
base currency so volatility/VaR/drawdown include FX risk. These tests use canned
price frames — no network.
"""
import numpy as np
import pandas as pd
import pytest

import tools.portfolio_analytics as pa
from tools.fx_utils import get_fx_rate_series, infer_symbol_currency

IDX = pd.date_range("2026-01-05", periods=3, freq="B")


def _yf_frame(prices: dict[str, list[float]]) -> pd.DataFrame:
    """Shape a dict of ticker->closes like yf.download output (MultiIndex)."""
    return pd.concat({"Close": pd.DataFrame(prices, index=IDX)}, axis=1)


def _fake_download(equities: dict[str, list[float]], fx: dict[str, list[float]] | None = None):
    """download_safe stub serving equity tickers and (optionally) FX pairs."""
    calls: list[list[str]] = []

    def fake(symbols, period="1y", threads=False, **kwargs):
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        calls.append(syms)
        if any("=X" in s for s in syms):
            if not fx:
                return pd.DataFrame()
            available = {s: v for s, v in fx.items() if s in syms}
            if not available:
                return pd.DataFrame()
            return _yf_frame(available)
        return _yf_frame({s: equities[s] for s in syms if s in equities})

    fake.calls = calls
    return fake


class TestInferSymbolCurrency:
    @pytest.mark.parametrize("symbol,expected", [
        ("AAPL", "USD"),
        ("RY.TO", "CAD"),
        ("PSA.TO", "CAD"),
        ("WN.V", "CAD"),
        ("VOD.L", "GBP"),
        ("SAP.DE", "EUR"),
        ("AIR.PA", "EUR"),
        ("BHP.AX", "AUD"),
        ("7203.T", "JPY"),
        ("BTC-USD", "USD"),
        ("ETH-CAD", "CAD"),
        ("", "USD"),
    ])
    def test_suffix_mapping(self, symbol, expected):
        assert infer_symbol_currency(symbol) == expected

    def test_dot_to_not_confused_with_dot_t(self):
        # ".TO" must resolve to CAD, never fall through to the JPY ".T" rule.
        assert infer_symbol_currency("SHOP.TO") == "CAD"


class TestGetFxRateSeries:
    def test_direct_pair(self, monkeypatch):
        import tools.yf_utils as yfu
        monkeypatch.setattr(
            yfu, "download_safe",
            _fake_download({}, fx={"USDCAD=X": [1.40, 1.41, 1.40]}),
        )
        out = get_fx_rate_series(["USD"], "CAD")
        assert list(out.columns) == ["USD"]
        assert out["USD"].tolist() == [1.40, 1.41, 1.40]

    def test_inverse_pair_reciprocal(self, monkeypatch):
        import tools.yf_utils as yfu
        # Direct CADUSD=X missing; only the inverse USDCAD=X is available.
        monkeypatch.setattr(
            yfu, "download_safe",
            _fake_download({}, fx={"USDCAD=X": [1.40, 1.41, 1.40]}),
        )
        out = get_fx_rate_series(["CAD"], "USD")
        assert list(out.columns) == ["CAD"]
        np.testing.assert_allclose(out["CAD"].tolist(), [1 / 1.40, 1 / 1.41, 1 / 1.40])

    def test_base_only_is_empty_without_download(self, monkeypatch):
        import tools.yf_utils as yfu
        def boom(*a, **k):
            raise AssertionError("no download expected")
        monkeypatch.setattr(yfu, "download_safe", boom)
        assert get_fx_rate_series(["CAD", "cad", ""], "CAD").empty

    def test_unfetchable_pair_omitted(self, monkeypatch):
        import tools.yf_utils as yfu
        monkeypatch.setattr(yfu, "download_safe", _fake_download({}, fx=None))
        assert get_fx_rate_series(["JPY"], "CAD").empty


class TestGetReturnsBaseCurrency:
    def test_mixed_portfolio_converted_to_base(self, monkeypatch):
        import tools.yf_utils as yfu
        fake = _fake_download(
            {"AAPL": [100.0, 101.0, 102.0], "RY.TO": [50.0, 50.0, 51.0]},
            fx={"USDCAD=X": [1.40, 1.41, 1.40]},
        )
        monkeypatch.setattr(yfu, "download_safe", fake)

        returns, valid = pa._get_returns(["AAPL", "RY.TO"], base_currency="CAD")

        assert set(valid) == {"AAPL", "RY.TO"}
        fx_info = returns.attrs["fx"]
        assert fx_info["base_currency"] == "CAD"
        assert fx_info["converted"] == ["AAPL"]
        assert fx_info["unavailable"] == []

        # r_base = (1 + r_local) * (1 + r_fx) - 1
        expected_aapl = [(101 * 1.41) / (100 * 1.40) - 1, (102 * 1.40) / (101 * 1.41) - 1]
        np.testing.assert_allclose(returns["AAPL"].tolist(), expected_aapl)
        # CAD-native symbol untouched.
        np.testing.assert_allclose(returns["RY.TO"].tolist(), [0.0, 1 / 50])

    def test_all_base_currency_skips_fx_download(self, monkeypatch):
        import tools.yf_utils as yfu
        fake = _fake_download({"RY.TO": [50.0, 51.0, 52.0], "CNR.TO": [150.0, 149.0, 151.0]})
        monkeypatch.setattr(yfu, "download_safe", fake)

        returns, valid = pa._get_returns(["RY.TO", "CNR.TO"], base_currency="CAD")

        assert len(fake.calls) == 1  # equities only — no FX round-trip
        assert not any("=X" in s for call in fake.calls for s in call)
        assert returns.attrs["fx"]["converted"] == []
        assert returns.attrs["fx"]["unavailable"] == []

    def test_fx_unavailable_falls_back_to_native(self, monkeypatch):
        import tools.yf_utils as yfu
        fake = _fake_download(
            {"AAPL": [100.0, 101.0, 102.0], "RY.TO": [50.0, 50.0, 51.0]},
            fx=None,  # FX download yields nothing
        )
        monkeypatch.setattr(yfu, "download_safe", fake)

        returns, _ = pa._get_returns(["AAPL", "RY.TO"], base_currency="CAD")

        assert returns.attrs["fx"]["unavailable"] == ["AAPL"]
        # Native returns — same numbers a constant-rate conversion would give.
        np.testing.assert_allclose(returns["AAPL"].tolist(), [0.01, 102 / 101 - 1])

    def test_profile_base_currency_resolved_when_not_passed(self, monkeypatch):
        import tools.memory as mem
        import tools.yf_utils as yfu
        monkeypatch.setattr(mem, "get_profile_base_currency", lambda profile=None: "CAD")
        fake = _fake_download(
            {"AAPL": [100.0, 101.0, 102.0]},
            fx={"USDCAD=X": [1.40, 1.41, 1.40]},
        )
        monkeypatch.setattr(yfu, "download_safe", fake)

        returns, _ = pa._get_returns(["AAPL"])
        assert returns.attrs["fx"]["base_currency"] == "CAD"
        assert returns.attrs["fx"]["converted"] == ["AAPL"]


class TestMetricsAndVarSurfaceFx:
    @pytest.fixture
    def mixed_market(self, monkeypatch):
        import tools.memory as mem
        import tools.yf_utils as yfu
        monkeypatch.setattr(mem, "get_profile_base_currency", lambda profile=None: "CAD")
        monkeypatch.setattr(pa, "_get_risk_free_rate", lambda: 0.04)
        fake = _fake_download(
            {
                "AAPL": [100.0, 101.0, 102.0],
                "RY.TO": [50.0, 50.0, 51.0],
                "SPY": [500.0, 502.0, 501.0],
            },
            fx={"USDCAD=X": [1.40, 1.41, 1.40]},
        )
        monkeypatch.setattr(yfu, "download_safe", fake)

    def test_metrics_reports_base_currency_and_fx_note(self, mixed_market):
        result = pa.calculate_portfolio_metrics(["AAPL", "RY.TO"], [0.5, 0.5])
        assert result["base_currency"] == "CAD"
        assert "AAPL" in result.get("fx_note", "")
        assert "data_warning" not in result
        assert "sharpe_ratio" in result["metrics"]

    def test_var_reports_base_currency(self, mixed_market):
        result = pa.calculate_var(["AAPL", "RY.TO"], [0.5, 0.5], investment=50000)
        assert result["base_currency"] == "CAD"
        assert "value_at_risk" in result

    def test_var_warns_when_fx_missing(self, monkeypatch):
        import tools.memory as mem
        import tools.yf_utils as yfu
        monkeypatch.setattr(mem, "get_profile_base_currency", lambda profile=None: "CAD")
        fake = _fake_download(
            {"AAPL": [100.0, 101.0, 102.0], "RY.TO": [50.0, 50.0, 51.0]},
            fx=None,
        )
        monkeypatch.setattr(yfu, "download_safe", fake)

        result = pa.calculate_var(["AAPL", "RY.TO"], [0.5, 0.5], investment=50000)
        assert "AAPL" in result.get("data_warning", "")
