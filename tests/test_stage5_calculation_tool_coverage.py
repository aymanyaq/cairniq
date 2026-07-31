
import numpy as np
import pandas as pd


def _price_history(rows=80, start=100.0, freq="D"):
    index = pd.date_range("2025-01-01", periods=rows, freq=freq)
    close = pd.Series(np.linspace(start, start + rows - 1, rows), index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": [1000] * rows,
        },
        index=index,
    )


def _wavy_history(rows=420, start=100.0):
    """Recent daily OHLCV with genuine ups and downs.

    Pure-linear data pins RSI at 100 and trips NaN/exception paths, so use a
    cumulative-sine walk to exercise the real momentum/volatility branches.
    """
    # tz-aware index to mirror real yfinance history (technicals reads index.tz).
    index = pd.date_range(
        end=pd.Timestamp.now().normalize(), periods=rows, freq="D", tz="America/New_York"
    )
    walk = np.cumsum(np.sin(np.linspace(0, 30, rows)) + 0.05)
    close = pd.Series(start + walk, index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, rows),
        },
        index=index,
    )


def _multi_close(symbols, rows=130):
    """A yfinance-style MultiIndex (Close, <symbol>) download frame."""
    index = pd.date_range(end=pd.Timestamp.now().normalize(), periods=rows, freq="D")
    data = {}
    for i, sym in enumerate(symbols):
        walk = np.cumsum(np.sin(np.linspace(0, 20 + i, rows)) + 0.03)
        data[("Close", sym)] = 100 + i + walk
    return pd.DataFrame(data, index=index, columns=pd.MultiIndex.from_tuples(data.keys()))


def _assert_no_numpy_leak(result, where=""):
    """Fail if any numpy scalar (np.float64/np.int64/np.bool_) reaches a tool's
    returned dict. Such values str()/repr() as ``np.float64(0.97)`` and leak that
    wrapper into raw tool dumps (e.g. the Market Analyst fallback summary)."""
    text = repr(result)
    assert "np.float64" not in text, f"{where}: np.float64 leaked -> {text[:300]}"
    assert "np.int" not in text, f"{where}: np.int leaked -> {text[:300]}"
    assert "np.bool" not in text and "np.True" not in text and "np.False" not in text, (
        f"{where}: np.bool leaked -> {text[:300]}"
    )

    def _walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}[{k!r}]")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")
        else:
            assert not isinstance(obj, np.generic), (
                f"{where}{path} is numpy {type(obj).__name__}: {obj!r}"
            )

    _walk(result, "")


def test_position_sizing_branches():
    from tools.position_sizing import calculate_position_size

    # The risk % is explicit in both calls: there is no default, and with no
    # risk rule in the profile an omitted pct returns no size at all.
    sized = calculate_position_size(100000, 2, 100, 92, volatility_adjustment=True, asset_beta=2)
    generic = calculate_position_size(100000, 2, volatility_adjustment=True, asset_beta=2)

    assert sized["recommended_shares"] == 125
    assert "Reduced by 50.0%" in sized["note"]
    assert "2.5% Tier (Adj to 1.2%)" in generic["generic_allocations"]


def test_monte_carlo_success_and_error(monkeypatch):
    import tools.monte_carlo as mc

    # 4.5 routes every draw through _draw_returns so the distribution is
    # switchable and reported. That is now the determinism seam: patching
    # np.random.normal no longer intercepts anything, because the engine uses a
    # Generator instance rather than the legacy global.
    monkeypatch.setattr(mc, "_draw_returns",
                        lambda mean, vol, shape, draws, rng, history=None: (np.full(shape, mean), draws))
    result = mc.run_monte_carlo(100000, 12000, 3, mean_return=0.05, volatility=0.0, num_simulations=5)
    assert result["success_rate"] == 100.0
    assert result["charts"]["years"] == [0, 1, 2, 3]
    # np.sum/np.percentile/np.median feed these fields — must be native floats.
    _assert_no_numpy_leak(result, "monte_carlo.run_monte_carlo")

    monkeypatch.setattr(mc, "_draw_returns",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rng failed")))
    assert "error" in mc.run_monte_carlo(100000, 0, 1)


def test_monte_carlo_risk_profile_presets(monkeypatch):
    """A named preset OVERRIDES mean_return/volatility; explicit floats and
    unknown names fall back to whatever mean_return was passed."""
    import tools.monte_carlo as mc

    # Deterministic: every path grows by exactly `mean` each year.
    # 4.5 routes every draw through _draw_returns so the distribution is
    # switchable and reported. That is now the determinism seam: patching
    # np.random.normal no longer intercepts anything, because the engine uses a
    # Generator instance rather than the legacy global.
    monkeypatch.setattr(mc, "_draw_returns",
                        lambda mean, vol, shape, draws, rng, history=None: (np.full(shape, mean), draws))

    # aggressive -> 10% ; conservative -> 4% ; over one year, no contributions.
    assert mc.run_monte_carlo(100000, 0, 1, risk_profile="aggressive")["median_result"] == 110000.0
    assert mc.run_monte_carlo(100000, 0, 1, risk_profile="conservative")["median_result"] == 104000.0
    # explicit mean_return wins when no profile is named
    assert mc.run_monte_carlo(100000, 0, 1, mean_return=0.05)["median_result"] == 105000.0
    # an unrecognised profile is ignored -> the explicit/default mean_return stands
    assert mc.run_monte_carlo(100000, 0, 1, mean_return=0.05, risk_profile="banana")["median_result"] == 105000.0


def test_monte_carlo_monthly_contribution_equals_annual_x12(monkeypatch):
    import tools.monte_carlo as mc

    # 4.5 routes every draw through _draw_returns so the distribution is
    # switchable and reported. That is now the determinism seam: patching
    # np.random.normal no longer intercepts anything, because the engine uses a
    # Generator instance rather than the legacy global.
    monkeypatch.setattr(mc, "_draw_returns",
                        lambda mean, vol, shape, draws, rng, history=None: (np.full(shape, mean), draws))

    monthly = mc.run_monte_carlo(100000, 0, 5, mean_return=0.06, monthly_contribution=1000)
    annual = mc.run_monte_carlo(100000, 12000, 5, mean_return=0.06)
    assert monthly["median_result"] == annual["median_result"]
    assert monthly["charts"]["p50"] == annual["charts"]["p50"]


def test_monte_carlo_goal_funded_success(monkeypatch):
    """goal_success_rate is distinct from non-depletion success_rate and compares
    against the target INFLATED to the horizon."""
    import tools.monte_carlo as mc

    # 4.5 routes every draw through _draw_returns so the distribution is
    # switchable and reported. That is now the determinism seam: patching
    # np.random.normal no longer intercepts anything, because the engine uses a
    # Generator instance rather than the legacy global.
    monkeypatch.setattr(mc, "_draw_returns",
                        lambda mean, vol, shape, draws, rng, history=None: (np.full(shape, mean), draws))

    # Deterministic terminal value: 100000 * 1.10^10 = 259374.25
    res = mc.run_monte_carlo(100000, 0, 10, risk_profile="aggressive", goal_target=100000)
    assert res["goal_target_nominal"] == round(100000 * (1.025 ** 10), 0)  # inflated
    assert res["goal_success_rate"] == 100.0  # 259k clears the ~128k inflated bar
    _assert_no_numpy_leak(res, "monte_carlo.run_monte_carlo goal")

    # An out-of-reach target -> 0% funded, but the run itself never depletes.
    miss = mc.run_monte_carlo(100000, 0, 10, risk_profile="aggressive", goal_target=1_000_000)
    assert miss["goal_success_rate"] == 0.0
    assert miss["success_rate"] == 100.0

    # No goal supplied -> the goal fields are simply absent/None.
    none = mc.run_monte_carlo(100000, 0, 10, risk_profile="aggressive")
    assert none["goal_success_rate"] is None
    assert none["goal_target_nominal"] is None


def test_fixed_income_ladder_rates_and_error(monkeypatch):
    import tools.fixed_income as fixed

    # Rates now come live from the providers (fixing the bug where hardcoded 2026
    # "estimated" constants were presented as current market data) - mock them so
    # the test stays deterministic. CAD reads the Bank of Canada's posted GIC curve
    # first (Roadmap 5.7) and only falls back to the FRED policy-rate proxy; this
    # case exercises that fallback, so the BoC leg is stubbed unavailable.
    monkeypatch.setattr(
        "tools.boc_valet.get_cad_gic_curve",
        lambda: {"status": "unavailable", "source": "Bank of Canada Valet", "reason": "stubbed"},
    )
    monkeypatch.setattr("tools.fred_api.get_canada_metrics", lambda: {"interest_rate": "4.10%"})
    monkeypatch.setattr(
        "tools.fred_api.get_treasury_curve",
        lambda: {"curve": {1: 4.4, 2: 4.2, 3: 4.1, 4: 4.05, 5: 4.0}, "as_of": "2026-01-01", "source": "FRED"},
    )

    cad = fixed.construct_bond_ladder(100000, "GIC", "CAD")
    usd_rates, usd_note = fixed._fetch_current_rates("Treasury", "USD")

    assert cad["strategy"] == "5-Year GIC Ladder (CAD)"
    assert cad["rungs"][0]["annual_income"] == "$820.00"
    assert usd_rates[1] == 4.4
    assert "FRED" in usd_note

    monkeypatch.setattr(fixed, "_fetch_current_rates", lambda *args: (_ for _ in ()).throw(RuntimeError("rates down")))
    assert "error" in fixed.construct_bond_ladder(100000)


def test_trade_structuring_success_error_and_insufficient(monkeypatch):
    import tools.trade_structuring as trade

    class Ticker:
        def __init__(self, frame):
            self.frame = frame

        def history(self, *args, **kwargs):
            return self.frame

    monkeypatch.setattr(trade.yf, "Ticker", lambda symbol: Ticker(_price_history(60)))
    setup = trade.structure_trade_setup("aapl", risk_reward_ratio=3, timeframe="day")
    assert setup["symbol"] == "AAPL"
    assert setup["trade_plan"]["risk_reward_ratio"] == "1:3"

    monkeypatch.setattr(trade.yf, "Ticker", lambda symbol: Ticker(_price_history(10)))
    assert "Insufficient" in trade.structure_trade_setup("NEW")["error"]

    monkeypatch.setattr(trade.yf, "Ticker", lambda symbol: (_ for _ in ()).throw(RuntimeError("bad ticker")))
    assert "Trade Architect failed" in trade.structure_trade_setup("BAD")["error"]


# --- NaN never reaches a quoted price level ----------------------------------
# 2026-07-29: a trailing row whose Close was NaN classified the name "Bearish"
# (`nan > ema_21` is False, silently) and shipped a plan quoting a real EMA21
# beside `current_price: "$nan"` — a stop the user could not act on and the
# compliance judge could not verify. NaN is a float: nothing raises, nothing
# compares True, and it formats as a plausible-looking string.


def _history_with_unsettled_last_row(rows=60):
    """History whose final session has no Close yet — the yfinance shape that bit."""
    frame = _price_history(rows)
    frame.iloc[-1, frame.columns.get_loc("Close")] = np.nan
    return frame


def test_trade_setup_ignores_an_unsettled_trailing_row(monkeypatch):
    import tools.trade_structuring as trade

    class Ticker:
        def __init__(self, frame):
            self.frame = frame

        def history(self, *args, **kwargs):
            return self.frame

    monkeypatch.setattr(
        trade.yf, "Ticker", lambda symbol: Ticker(_history_with_unsettled_last_row(60))
    )
    setup = trade.structure_trade_setup("AAPL")

    # The prior settled Close carries the setup; no "$nan" anywhere in the output.
    assert "error" not in setup
    assert "nan" not in str(setup).lower()
    assert setup["current_price"] == "$158.00"


def test_trade_setup_refuses_when_every_close_is_nan(monkeypatch):
    import tools.trade_structuring as trade

    frame = _price_history(60)
    frame["Close"] = np.nan

    class Ticker:
        def history(self, *args, **kwargs):
            return frame

    monkeypatch.setattr(trade.yf, "Ticker", lambda symbol: Ticker())
    result = trade.structure_trade_setup("AAPL")

    assert "error" in result
    assert "trade_plan" not in result  # no stop is quoted off unusable data


def test_trade_setup_refuses_when_a_derived_level_is_not_finite(monkeypatch):
    """Close is clean but ATR is not — the guard covers each level, not just price."""
    import tools.trade_structuring as trade

    frame = _price_history(60)
    # No intraday range on any bar, so every true-range component — and with it
    # ATR14 — is NaN while Close (and the EMA off it) stays perfectly good.
    frame["High"] = np.nan
    frame["Low"] = np.nan

    class Ticker:
        def history(self, *args, **kwargs):
            return frame

    monkeypatch.setattr(trade.yf, "Ticker", lambda symbol: Ticker())
    result = trade.structure_trade_setup("AAPL")

    assert "ATR14" in result["error"]
    assert "AAPL" in result["error"]


def test_finite_helper_rejects_the_values_that_do_not_raise():
    from tools.trade_structuring import _finite

    assert _finite(12.5) == 12.5
    assert _finite(float("nan")) is None
    assert _finite(float("inf")) is None
    assert _finite(None) is None
    assert _finite("n/a") is None


def test_sector_allocation_fund_api_unknown_and_insights(monkeypatch):
    import tools.sector_analysis as sector

    class Ticker:
        def __init__(self, symbol):
            if symbol == "AAPL":
                self.info = {"sector": "Information Technology"}
            elif symbol == "ETF":
                self.info = {"sector": "Unknown", "quoteType": "ETF"}
            else:
                raise RuntimeError("missing")

    monkeypatch.setattr(sector.yf, "Ticker", Ticker)
    # Keep the test hermetic: no live FMP calls for the ETF/BAD fallbacks.
    monkeypatch.setattr(sector, "_fmp_decompose", lambda sym: None)
    result = sector.check_portfolio_allocation(["XLK", "AAPL", "ETF", "BAD"], [50, 25, 15, 10])

    assert result["sector_allocation"]["Technology"] == "75.0%"
    assert result["sector_allocation"]["Unclassified Fund"] == "15.0%"
    assert result["sector_allocation"]["Unknown"] == "10.0%"
    assert result["key_insights"]


def test_sector_allocation_fmp_cash_and_diversified_fund(monkeypatch):
    import tools.daily_cache as daily_cache
    import tools.sector_analysis as sector

    class Ticker:
        def __init__(self, symbol):
            # A fund with no single sector; the target-date fund name errors out.
            if symbol == "ZZETF":
                self.info = {"quoteType": "ETF"}
            else:
                raise RuntimeError("missing")

    monkeypatch.setattr(sector.yf, "Ticker", Ticker)
    # check_portfolio_allocation imports these from tools.daily_cache at call time,
    # so patch them at the source to stay off the shared on-disk cache.
    monkeypatch.setattr(daily_cache, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(daily_cache, "set_cached", lambda *a, **k: None)
    # ZZETF isn't in the static DB, so it reaches the FMP decomposition fallback.
    monkeypatch.setattr(
        sector,
        "_fmp_decompose",
        lambda sym: {"Technology": 0.6, "Financial Services": 0.4} if sym == "ZZETF" else None,
    )

    # Use fabricated symbols not in the static DB or knowledge graph so the
    # FMP-fallback and diversified-fund heuristic paths are what get exercised.
    result = sector.check_portfolio_allocation(
        ["ZZETF", "CASH", "ACME 2045 TARGET FUND"], [50, 25, 25]
    )
    alloc = result["sector_allocation"]

    # FMP-decomposed: 50% * {0.6 tech, 0.4 fin}
    assert alloc["Technology"] == "30.0%"
    assert alloc["Financial Services"] == "20.0%"
    # Cash line -> its own bucket, not Unknown
    assert alloc["Cash"] == "25.0%"
    # Named target-date fund with no market quote -> Diversified Fund, not Unknown
    assert alloc["Diversified Fund"] == "25.0%"
    assert "Unknown" not in alloc


def test_fmp_decompose_survives_a_stamped_weightings_map(monkeypatch):
    """An in-band `_as_of` must not take the whole sector breakdown down.

    get_fmp_etf_sector_weightings is map-shaped ({sector: fraction}), and it was
    cached with the default stamp=True for two days. The stamp arrived as a
    phantom sector whose weight is a date string, and `norm.get(k, 0.0) + frac`
    raised `float + str` — killing check_portfolio_allocation outright rather
    than degrading it, so "am I overweight Tech?" returned nothing at all.
    Caches written before the fix still carry the stamp for the rest of their
    TTL, so the consumer has to survive it too.
    """
    import tools.fmp_api as fmp_api
    import tools.sector_analysis as sector

    monkeypatch.setattr(
        fmp_api,
        "get_fmp_etf_sector_weightings",
        lambda sym: {
            "Technology": 0.6,
            "Financial Services": 0.4,
            "_as_of": "2026-07-29T07:05:02.681605",
        },
    )

    out = sector._fmp_decompose("ZZETF")

    assert out == {"Technology": 0.6, "Financial Services": 0.4}
    assert "_as_of" not in out


def test_income_projection_dividends_info_and_error(monkeypatch):
    import tools.income_analytics as income

    recent_index = pd.date_range(pd.Timestamp.now() - pd.Timedelta(days=120), periods=4, freq="30D")

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol
            if symbol == "BAD":
                raise RuntimeError("bad")

        def history(self, *args, **kwargs):
            return _price_history(5, 100)

        @property
        def dividends(self):
            if self.symbol == "DIV":
                return pd.Series([0.25, 0.25, 0.25, 0.25], index=recent_index)
            return pd.Series(dtype=float)

        @property
        def info(self):
            # A PERCENT (4%), as the provider sends it. Was 0.04, the fraction
            # the reader assumed, which hid a 100x income overstatement.
            return {"dividendYield": 4.0}

    monkeypatch.setattr(income.yf, "Ticker", Ticker)
    with_amounts = income.project_portfolio_income(["DIV", "INFO", "BAD"], [10, 5, 1])
    yield_only = income.project_portfolio_income(["INFO"], [])

    assert with_amounts["details"][0]["metric_used"] == "Trailing 12m Dividends"
    assert with_amounts["details"][1]["metric_used"] == "Info Yield"
    assert "error" in with_amounts["details"][2]
    assert yield_only["summary"]["portfolio_yield"] == "4.00%"


def test_wealth_management_tools(monkeypatch):
    import yfinance as yf

    import tools.market_data as mdata
    import tools.portfolio_analytics as analytics

    class Ticker:
        def __init__(self, symbol):
            if symbol == "BAD":
                raise RuntimeError("bad")
            self.info = {
                "currency": "CAD" if symbol.endswith(".TO") else "USD",
                "trailingPE": 20,
                "forwardPE": 18,
                "pegRatio": 0.8,
                "priceToBook": 5,
                "revenueGrowth": 0.12,
                "earningsGrowth": 0.2,
                "debtToEquity": 50,
                "currentRatio": 1.8,
                "returnOnEquity": 0.25,
                "dividendYield": 2.0,  # a PERCENT; rate/price agrees at 2.00%
                "payoutRatio": 0.55,
                "dividendRate": 2.0,
                "currentPrice": 100.0,
                "exDividendDate": "2026-05-01",
            }

    monkeypatch.setattr(yf, "Ticker", Ticker)

    exposure = analytics.calculate_currency_exposure.invoke({"holdings": {"AAPL": 100, "RY.TO": 50, "EURX": 25, "Cash": 10, "BAD": 5}})
    dividends = mdata.get_dividend_analysis.invoke({"ticker": "AAPL"})

    assert exposure["breakdown_percent"]["USD"] > exposure["breakdown_percent"]["CAD"]
    assert "Safe" in dividends


def test_seasonality_price_targets_and_earnings(monkeypatch):
    import tools.earnings_calendar as earnings
    import tools.price_targets as targets
    import tools.seasonality as seasonality

    class SeasonalityTicker:
        def history(self, *args, **kwargs):
            return _price_history(420, 100, freq="D")

    monkeypatch.setattr(seasonality.yf, "Ticker", lambda symbol: SeasonalityTicker())
    result = seasonality.analyze_seasonality.__wrapped__("AAPL", years=2)
    assert result["symbol"] == "AAPL"
    assert len(result["seasonality"]) == 12
    # Series.mean()/.sum() drive average_return_pct/win_rate_pct — must be native.
    _assert_no_numpy_leak(result, "seasonality.analyze_seasonality")

    class TargetTicker:
        info = {
            "currentPrice": 100,
            "targetHighPrice": 150,
            "targetLowPrice": 80,
            "targetMeanPrice": 130,
            "targetMedianPrice": 125,
            "fiftyTwoWeekLow": 70,
        }

    monkeypatch.setattr(targets.yf, "Ticker", lambda symbol: TargetTicker())
    target_result = targets.get_price_targets.__wrapped__("AAPL")
    assert target_result["recommendation"].startswith("Strong Buy")
    assert target_result["risk_reward_ratio"] == "Data Unavailable"
    assert "ATR/support" in target_result["stop_loss_suggestion"]

    future = pd.Timestamp.now() + pd.Timedelta(days=5)
    class EarningsTicker:
        info = {"forwardEps": 7.5, "trailingEps": 6.5}
        earnings_dates = pd.DataFrame(index=[future])
        earnings_history = pd.DataFrame({"surprisePercent": [5.0, -2.0, 3.0]})

    monkeypatch.setattr(earnings.yf, "Ticker", lambda symbol: EarningsTicker())
    earnings_result = earnings.get_earnings_info.__wrapped__("AAPL")
    assert earnings_result["earnings_warning"].startswith("⚠️")
    assert earnings_result["historical_performance"]["beat_rate"] == "67%"


def test_alternative_dark_pool_earnings_nlp_and_visualizer(monkeypatch):
    import tools.alternative_data as alt
    import tools.dark_pool as dark
    import tools.earnings_nlp as nlp
    import tools.visualizer as visualizer

    class AltTicker:
        def history(self, *args, **kwargs):
            return _price_history(80, 100)

    monkeypatch.setattr(alt.yf, "Ticker", lambda symbol: AltTicker())
    monkeypatch.setattr(alt, "search_news", lambda *args, **kwargs: "record surge in app downloads")
    assert "BULLISH" in alt.get_alternative_data_signal.__wrapped__("APP")["verdict"]

    index = pd.date_range("2026-01-01 09:30", periods=60, freq="min")
    intraday = pd.DataFrame(
        {
            "Open": [100] * 60,
            "Close": [100] * 60,
            "Volume": [100] * 59 + [10000],
        },
        index=index,
    )
    class DarkTicker:
        def history(self, *args, **kwargs):
            return intraday

    monkeypatch.setattr(dark.yf, "Ticker", lambda symbol: DarkTicker())
    dark_result = dark.scan_dark_pool_proxy.__wrapped__("SPY")
    assert dark_result["alerts_count"] == 1
    assert "DARK POOL" in dark_result["alerts"][0]["signature"]

    # 5.4: the stub must now be a REAL transcript payload. A bare string is no
    # longer scored — that is the whole fix, since the provider's rate-limit
    # notice is also a bare string and used to be read as a neutral tone.
    from tools.fmp_api import TRANSCRIPT_HEADER
    positive_text = " ".join(["growth", "strong", "record", "momentum", "confident"] * 60)
    monkeypatch.setattr(
        nlp, "get_earnings_transcript",
        lambda symbol, year=None, quarter=None: (
            f"{TRANSCRIPT_HEADER} AAPL (Q2 2026)\n**Date:** 2026-04-25\n\n{positive_text}"
        ),
    )
    assert nlp.analyze_management_tone.__wrapped__.__wrapped__("AAPL")["tone_status"].startswith("Highly Confident")

    class ChartTicker:
        options = []
        info = {"currentPrice": 100}

        def history(self, *args, **kwargs):
            return _price_history(90, 100)

    monkeypatch.setattr(visualizer.yf, "Ticker", lambda symbol: ChartTicker())
    ascii_chart = visualizer.generate_ascii_chart("AAPL")
    price_chart = visualizer.generate_price_chart("AAPL")
    mc_chart = visualizer.generate_monte_carlo_chart({"charts": {"years": [0, 1], "p10": [90, 95], "p50": [100, 110], "p90": [120, 140]}})

    assert "CHART AAPL" in ascii_chart
    assert price_chart.startswith("[PLOTLY_JSON:") or "CHART AAPL" in price_chart
    assert mc_chart == "" or mc_chart.startswith("[PLOTLY_JSON:")


def test_market_scanner_live_and_fallback_paths(monkeypatch):
    import tools.market_scanner as scanner

    broad = pd.DataFrame(
        {
            ("Close", "SPY"): [100.0, 101.0],
            ("Close", "QQQ"): [200.0, 198.0],
            ("Close", "IWM"): [150.0, 153.0],
            ("Close", "^VIX"): [15.0, 16.5],
        }
    )

    def live_fmp(endpoint):
        if endpoint == "sector-performance-snapshot":
            return [{"sector": "Technology", "changesPercentage": 1.25}]
        if endpoint == "biggest-gainers":
            return [{"symbol": "WIN", "name": "Winner Incorporated", "price": 12.5, "changesPercentage": 8.2}]
        if endpoint == "biggest-losers":
            return [{"symbol": "LOSE", "name": "Loser Incorporated", "price": 8.0, "changesPercentage": -5.5}]
        if endpoint == "most-actives":
            return [{"symbol": "BUSY", "name": "Busy Incorporated", "price": 20.0, "changesPercentage": 1.0}]
        return []

    monkeypatch.setattr(scanner.yf, "download", lambda *args, **kwargs: broad)
    monkeypatch.setattr(scanner, "_fmp_get", live_fmp)
    live = scanner.scan_intraday_movers.__wrapped__("SPY")

    assert live["top_gainers"][0]["symbol"] == "WIN"
    assert live["sector_performance"] == ["Technology: +1.25%"]
    assert "SPY: +1.00%" in live["market_status"]

    fallback_prices = pd.DataFrame(
        {
            ("Close", "NVDA"): [100.0, 104.0],
            ("Close", "TSLA"): [100.0, 97.0],
        }
    )
    downloads = iter([broad, fallback_prices])
    monkeypatch.setattr(scanner.yf, "download", lambda *args, **kwargs: next(downloads))
    monkeypatch.setattr(scanner, "_fmp_get", lambda endpoint: [])
    fallback = scanner.scan_intraday_movers.__wrapped__("QQQ")

    assert fallback["note"].startswith("FMP live feeds unavailable")
    assert [m["symbol"] for m in fallback["active_movers"]] == ["NVDA", "TSLA"]

    monkeypatch.setattr(scanner.yf, "download", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("download failed")))
    assert "Scanner failed" in scanner.scan_intraday_movers.__wrapped__("SPY")["error"]


def test_technicals_outputs_are_native(monkeypatch):
    import tools.technicals as technicals

    class Ticker:
        def history(self, *args, **kwargs):
            return _wavy_history(420)

    monkeypatch.setattr(technicals.yf, "Ticker", lambda symbol: Ticker())
    result = technicals.get_comprehensive_technicals.__wrapped__("AAPL")

    assert result.get("error") is None
    assert "momentum" in result and "trend" in result and "levels" in result
    # current_price/sma/rsi/macd/bb/atr + golden_cross all come off .iloc[-1].
    _assert_no_numpy_leak(result, "technicals.get_comprehensive_technicals")
    assert isinstance(result["trend"]["golden_cross"], bool)
    assert isinstance(result["momentum"]["rsi_14"], float)


def test_screener_setup_outputs_are_native(monkeypatch):
    import tools.screener as screener

    class Ticker:
        def history(self, *args, **kwargs):
            return _wavy_history(90)

    monkeypatch.setattr(screener.yf, "Ticker", lambda symbol: Ticker())
    result = screener.check_setup.__wrapped__("AAPL")
    if result is not None:
        _assert_no_numpy_leak(result, "screener.check_setup")
        assert isinstance(result["price"], float)
        assert isinstance(result["rsi"], float)


def test_pattern_recognition_outputs_are_native(monkeypatch):
    import tools.pattern_recognition as pattern

    monkeypatch.setattr(pattern, "get_price_data", lambda symbol, days=120: _wavy_history(420))

    sr = pattern.find_support_resistance("AAPL")
    ma = pattern.check_ma_crossover("AAPL")
    rsi = pattern.detect_rsi_divergence("AAPL")

    _assert_no_numpy_leak(sr, "pattern.find_support_resistance")
    _assert_no_numpy_leak(ma, "pattern.check_ma_crossover")
    _assert_no_numpy_leak(rsi, "pattern.detect_rsi_divergence")
    assert isinstance(sr["current_price"], float)


def test_sector_rotation_outputs_are_native(monkeypatch):
    import tools.sector_rotation as sr_mod

    class Ticker:
        def history(self, *args, **kwargs):
            return _wavy_history(130)

    monkeypatch.setattr(sr_mod.yf, "Ticker", lambda symbol: Ticker())
    result = sr_mod.detect_sector_rotation.__wrapped__()

    assert "error" not in result
    _assert_no_numpy_leak(result, "sector_rotation.detect_sector_rotation")
    assert isinstance(result["sector_performance"][0]["momentum_score"], float)


def test_market_mechanics_outputs_are_native(monkeypatch):
    import tools.market_mechanics as mm

    def fake_download(symbols, *args, **kwargs):
        if isinstance(symbols, str):
            symbols = [symbols]
        return _multi_close(list(symbols))

    monkeypatch.setattr(mm.yf, "download", fake_download)

    rotation = mm.detect_sector_rotation.__wrapped__()
    rs = mm.rank_relative_strength.__wrapped__("AAPL,MSFT", "SPY")

    assert "error" not in rotation
    assert "error" not in rs
    _assert_no_numpy_leak(rotation, "market_mechanics.detect_sector_rotation")
    _assert_no_numpy_leak(rs, "market_mechanics.rank_relative_strength")
    assert isinstance(rs["rankings"][0]["raw_rel_score"], float)


def test_social_buzz_output_is_native(monkeypatch):
    import tools.sentiment_analysis as sa

    class Ticker:
        def history(self, *args, **kwargs):
            return _wavy_history(30)

    monkeypatch.setattr(sa.yf, "Ticker", lambda symbol: Ticker())
    result = sa.get_social_buzz("AAPL")

    _assert_no_numpy_leak(result, "sentiment_analysis.get_social_buzz")
    assert isinstance(result["volume_ratio"], float)
