def test_tool_registry_portfolio_report_paths(monkeypatch):
    import agent.tool_registry as reg

    monkeypatch.setattr(reg, "get_portfolio_summary", lambda: {
        "holdings": [
            {"symbol": "AAPL", "value_usd": 700.0, "gain_loss": "+10%"},
            {"symbol": "MSFT", "value_usd": 300.0, "gain_loss": "+5%"},
            {"symbol": "PRIVATE_FUND", "value_usd": 500.0, "gain_loss": "0%"},
        ],
        "summary": {"current_value": "$1,500"},
        "usd_cad_rate": 1.4,
    })
    monkeypatch.setattr("tools.portfolio_csv.get_tradeable_symbols", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(reg, "calculate_portfolio_metrics", lambda symbols, weights=None, period="1y": {
        "metrics": {
            "sharpe_ratio": 1.2,
            "beta": 0.9,
            "max_drawdown": "-8%",
            "annual_volatility": "12%",
            "annual_return": "10%",
        },
        "interpretation": {
            "sharpe": "Good",
            "beta": "Market-like",
            "volatility": "Low",
        },
    })
    monkeypatch.setattr(reg, "calculate_var", lambda symbols, weights=None, investment=1000000: {
        "value_at_risk": {"daily_var_dollars": "$1,000"}
    })
    monkeypatch.setattr(reg, "analyze_correlation", lambda symbols: {
        "diversification_quality": "Good",
        "average_correlation": 0.4,
        "correlation_pairs": [{"pair": "AAPL vs MSFT", "correlation": 0.5}],
    })
    monkeypatch.setattr(reg, "get_sector_exposure", lambda symbols, weights=None, is_portfolio=False: {
        "sector_breakdown": {"Technology": 100},
        "concentration_warning": "Heavy tech",
    })
    monkeypatch.setattr(reg, "get_fee_income_analysis", lambda symbols, weights=None: {
        "weighted_expense_ratio": 0.002,
        "fee_rating": "Good",
        "high_fee_funds": ["FUNDX"],
        "expected_yield": 0.015,
        "dividend_payers_count": 2,
    })
    monkeypatch.setattr(reg, "analyze_macro_context", lambda: {
        "current_regime": "Expansion",
        "plain_english": "Risk-on",
        "canadian_strategy": "Hedge CAD",
        "strategy": {
            "tactical_opportunity": ["Technology"],
            "sectors_to_underweight": ["Utilities"],
        },
        "key_indicators": {
            "Systemic Risk": "Low",
            "Liquidity (M2)": "Expanding",
            "Inflation (US)": "2.5%",
        },
    })

    listed_report = reg.analyze_portfolio_risk.invoke({"symbols": "AAPL,MSFT", "investment_amount": 500000})
    portfolio_report = reg.analyze_portfolio_risk.invoke({"symbols": "PORTFOLIO"})

    assert "Institutional Portfolio Report" in listed_report
    assert "My Portfolio" in portfolio_report
    assert "High Fee Alert" in portfolio_report

    monkeypatch.setattr(reg, "get_portfolio_summary", lambda: {"error": "missing"})
    assert "Error loading portfolio" in reg.analyze_portfolio_risk.invoke({"symbols": "PORTFOLIO"})
    assert "No valid symbols" in reg.analyze_portfolio_risk.invoke({"symbols": ""})


def test_tool_registry_thin_wrappers_route_to_underlying_tools(monkeypatch):
    import agent.tool_registry as reg

    calls = []

    def stub(name):
        def _inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            return {"called": name, "args": args, "kwargs": kwargs}
        return _inner

    direct_stubs = {
        "analyze_options": "options",
        "search_news": "search",
        "get_price_targets": "targets",
        "run_mc_engine": "monte_carlo",
        "generate_monte_carlo_chart": "mc_chart",
        "get_earnings_info": "earnings",
        "calculate_position_size": "position",
        "get_insider_and_short_data": "insider_short",
        "av_get_quote": "quote",
        "get_daily_prices": "prices",
        "get_company_overview": "overview",
        "get_all_macro_indicators": "macro_all",
        "analyze_macro_context": "macro_strategy",
        "detect_patterns": "patterns",
        "find_support_resistance": "support",
        "check_ma_crossover": "ma",
        "get_full_sentiment": "sentiment",
        "get_fear_greed_index": "fear",
        "get_analyst_consensus": "analysts",
        "generate_price_chart": "chart",
        "scan_sector_opportunities": "scan_sector",
        "scan_geopolitical_opportunities": "geo_scan",
        "get_ticker_geopolitical_context": "geo_context",
        "get_supply_chain_exposure": "supply_chain",
        "calculate_dealer_gex": "gex",
        "get_alternative_data_signal": "alt",
        "analyze_management_tone": "tone",
        "check_crowded_trade": "crowded",
        "analyze_correlation": "correlation",
        "simulate_rebalancing": "rebalance",
        "simulate_scenario": "scenario",
        "get_reddit_sentiment": "reddit",
        "scan_unusual_activity": "unusual",
        "find_breakout_candidates": "breakouts",
        "_raw_get_earnings_calendar": "earnings_raw",
        "get_insider_trading": "insider",
        "get_institutional_ownership": "institutional",
        "detect_sector_rotation": "rotation",
        "rank_relative_strength": "relative",
        "predict_earnings_surprise": "surprise",
        "get_upcoming_ipos": "ipos",
        "macro_stock_deep_dive": "deep_dive",
        "macro_portfolio_risk": "portfolio_risk",
        "tool_scan_intraday_movers": "movers",
        "get_market_regime": "regime",
        "get_regime_history": "regime_history",
        "compare_assets": "compare",
    }
    for attr, name in direct_stubs.items():
        monkeypatch.setattr(reg, attr, stub(name))
    monkeypatch.setattr(reg, "run_mc_engine", lambda **kwargs: {"interpretation": "Monte Carlo ok"})
    monkeypatch.setattr(reg, "generate_monte_carlo_chart", lambda result: "mc_chart")

    monkeypatch.setattr("tools.sector_analysis.check_portfolio_allocation", stub("sector_alloc"))
    monkeypatch.setattr("tools.compare_assets.compare_assets", stub("compare"))
    monkeypatch.setattr("tools.health_check.run_tool_health_check", lambda: {
        "health_summary": {
            "overall_status": "ok",
            "operational": 3,
            "failed": 1,
            "total_checked": 4,
            "missing_prerequisites": ["KEY"],
        },
        "tool_results": [{"tool": "bad", "status": "ERR"}],
    })
    monkeypatch.setattr(reg, "_get_portfolio_data", lambda: (["AAPL", "MSFT"], {"AAPL": 60, "MSFT": 40}, "ctx"))

    assert reg.analyze_options_chain.invoke({"symbol": "AAPL"})["called"] == "options"
    assert reg.search_stock_news.invoke({"symbol": "AAPL"})["called"] == "search"
    assert reg.perform_search.invoke({"query": "rates"})["called"] == "search"
    assert reg.analyze_sectors.invoke({"symbols": "AAPL,MSFT"})["called"] == "sector_alloc"
    assert reg.compare_stocks.invoke({"symbols": "AAPL,MSFT", "mode": "performance"})["called"] == "compare"
    assert reg.get_analyst_targets.invoke({"symbol": "AAPL"})["called"] == "targets"
    assert "mc_chart" in reg.run_retirement_simulation.invoke({"years": 10, "annual_contribution": 1000, "start_value": 50000})
    assert reg.get_earnings_calendar.invoke({"symbol": "AAPL"})["called"] == "earnings"
    assert reg.calculate_position.invoke({"portfolio_value": 100000, "entry_price": 150, "stop_loss_price": 140})["called"] == "position"
    assert reg.get_insider_short_interest.invoke({"symbol": "AAPL"})["called"] == "insider_short"
    assert reg.get_realtime_quote.invoke({"symbol": "AAPL"})["called"] == "quote"
    assert reg.get_stock_quote.invoke({"symbol": "AAPL"})["called"] == "quote"
    assert reg.get_price_history.invoke({"symbol": "AAPL", "days": 5})["called"] == "prices"
    assert reg.get_fundamentals_detailed.invoke({"symbol": "AAPL"})["called"] == "overview"
    assert reg.get_macro_overview.invoke({})["called"] == "macro_all"
    assert reg.get_macro_strategy.invoke({})["called"] == "macro_strategy"
    assert reg.analyze_patterns.invoke({"symbol": "AAPL"})["called"] == "patterns"
    assert reg.get_support_resistance.invoke({"symbol": "AAPL"})["called"] == "support"
    assert reg.get_ma_signals.invoke({"symbol": "AAPL"})["called"] == "ma"
    assert reg.get_sentiment.invoke({"symbol": "AAPL"})["called"] == "sentiment"
    assert reg.get_fear_greed.invoke({})["called"] == "fear"
    assert reg.get_analyst_ratings.invoke({"symbol": "AAPL"})["called"] == "analysts"
    assert reg.visualize_stock_chart.invoke({"symbol": "AAPL", "period": "1mo"})["called"] == "chart"
    assert reg.scan_opportunities.invoke({"sector": "Tech"})["called"] == "scan_sector"
    monkeypatch.setattr("tools.funnel_backtest.get_funnel_scorecard_data",
                        lambda: {"status": "ok", "horizons": {}, "caveats": []})
    assert reg.get_funnel_scorecard.invoke({})["status"] == "ok"
    assert reg.scan_geopolitical_events.invoke({"event": ""})["called"] == "geo_scan"
    assert reg.check_ticker_geopolitical_context.invoke({"symbol": "XOM"})["called"] == "geo_context"
    assert reg.check_supply_chain.invoke({"country": "Qatar"})["called"] == "supply_chain"
    assert reg.run_diagnostics.invoke({})["overall_status"] == "ok"
    assert reg.dealer_gamma_exposure.invoke({"symbol": "AAPL"})["called"] == "gex"
    assert reg.get_alt_data.invoke({"symbol": "AAPL"})["called"] == "alt"
    assert reg.check_management_tone.invoke({"symbol": "AAPL"})["called"] == "tone"
    assert reg.analyze_crowded_trade.invoke({"symbol": "AAPL"})["called"] == "crowded"
    assert reg.check_portfolio_correlation.invoke({"symbols": ""})["called"] == "correlation"
    assert reg.check_portfolio_correlation.invoke({"symbols": "AAPL"})["error"].startswith("Need")
    assert reg.simulate_portfolio_rebalancing.invoke({"adjustments": '{"AAPL":50}'})["called"] == "rebalance"
    assert reg.run_stress_test.invoke({"scenario": "recession"})["called"] == "scenario"
    assert reg.analyze_reddit_sentiment.invoke({"symbol": "AAPL"})["called"] == "reddit"
    assert reg.scan_options_chain.invoke({"symbol": "AAPL"})["called"] == "unusual"
    assert reg.scan_technical_breakouts.invoke({"symbols": "AAPL,MSFT"})["called"] == "breakouts"
    assert reg.get_earnings_data.invoke({"symbol": "AAPL"})["called"] == "earnings_raw"
    # get_insider_activity is EDGAR-first (Roadmap 5.1): Form 4 when EDGAR
    # answers, the yfinance insider table only as fallback.
    monkeypatch.setattr("tools.sec_edgar.get_form4_activity",
                        lambda s: {"called": "form4_edgar"})
    assert reg.get_insider_activity.invoke({"symbol": "AAPL"})["called"] == "form4_edgar"
    from tools.tool_errors import unavailable as _unavail
    monkeypatch.setattr("tools.sec_edgar.get_form4_activity",
                        lambda s: _unavail("SEC EDGAR", "down"))
    assert reg.get_insider_activity.invoke({"symbol": "AAPL"})["called"] == "insider"
    assert reg.get_institutional_data.invoke({"symbol": "AAPL"})["called"] == "institutional"
    assert reg.check_sector_rotation.invoke({})["called"] == "rotation"
    assert reg.get_relative_strength.invoke({"symbols": "AAPL,MSFT"})["called"] == "relative"
    assert reg.predict_surprise.invoke({"symbol": "AAPL"})["called"] == "surprise"
    assert reg.find_ipos.invoke({})["called"] == "ipos"
    assert reg.run_stock_deep_dive.invoke({"symbol": "AAPL"})["called"] == "deep_dive"
    assert reg.assess_portfolio_risk.invoke({})["called"] == "portfolio_risk"
    assert reg.scan_intraday_movers.invoke({})["called"] == "movers"
    assert reg.get_market_pulse_data.invoke({})["called"] == "regime"
    assert reg.get_market_regime_data.invoke({"days": 5})["called"] == "regime_history"

    assert calls


def test_tool_registry_forecast_and_late_wrappers(monkeypatch):
    import agent.tool_registry as reg

    monkeypatch.setattr(reg, "send_status", lambda message: None)
    monkeypatch.setattr(reg, "_get_portfolio_data", lambda: (["AAPL"], {"AAPL": 100}, "Portfolio context"))
    monkeypatch.setattr(reg, "DSPY_AVAILABLE", False)
    monkeypatch.setattr("tools.fred_api.get_inflation_data", lambda: {"headline_inflation": "3.1%"})
    monkeypatch.setattr("tools.fred_api.get_fed_funds_rate", lambda: {"current_rate": "4.5%"})
    monkeypatch.setattr("tools.fred_api.get_systemic_risk_indicators", lambda: {
        "credit_spread": "3.2%",
        "crash_risk": "Low",
        "liquidity_status": "Expanding",
        "m2_growth_yoy": "4.0%",
    })
    monkeypatch.setattr("tools.fred_api.get_treasury_yields", lambda: {
        "yield_spread": "0.5%",
        "curve_status": "Normal",
    })
    monkeypatch.setattr("tools.market_data.get_stock_data", lambda symbol: {
        "pe_ratio": "22.0",
        "recent_trend": "6.0% (2mo)",
        "current_price": "$15.00",
    })
    monkeypatch.setattr("tools.market_mechanics.detect_sector_rotation", lambda: {
        "leading_sectors": ["Technology"],
        "lagging_sectors": ["Utilities"],
    })
    monkeypatch.setattr("tools.fmp_api.get_economic_calendar", lambda: [
        {"date": "2026-05-01", "event": "CPI", "estimate": "0.2%"}
    ])
    monkeypatch.setattr("tools.predictive.match_historical_regime", lambda *args: {
        "matched_regime": "Soft landing",
        "similarity_score": 82,
        "description": "Moderate growth",
        "authored_scenario_3mo": "Up modestly",
        "authored_scenario_1yr": "Positive",
        "key_risks": "Inflation",
        "basis": "authored constant",
        "methodology_note": "Test analogue",
    })

    forecast = reg.generate_future_forecast.invoke({"query": "next year outlook"})
    assert "Soft landing" in forecast
    assert "Evidence Quality" in forecast
    # 2.7: the authored half must be attributed wherever this renders, including
    # the non-LLM path this test exercises.
    assert "authored" in forecast.lower()
    assert "not measured" in forecast.lower()

    calls = []

    def stub(name):
        def _inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            return {"called": name, "args": args, "kwargs": kwargs}
        return _inner

    reg.PORTFOLIO_CACHE = {"data": None, "timestamp": 0}
    monkeypatch.setattr(reg, "get_portfolio_summary", lambda: {"summary": "ok"})
    assert reg.get_portfolio_snapshot.invoke({})["summary"] == "ok"
    assert reg.get_portfolio_snapshot.invoke({})["summary"] == "ok"

    monkeypatch.setattr(reg, "scan_sector_opportunities", stub("screen"))
    monkeypatch.setattr(reg, "fetch_historical_perf", stub("historical"))
    monkeypatch.setattr(reg, "compare_assets", stub("competitors"))
    monkeypatch.setattr(reg, "get_etf_holdings", stub("etf"))
    monkeypatch.setattr(reg, "get_all_macro_indicators", stub("macro"))
    monkeypatch.setattr(reg, "get_trade_setup", stub("trade_setup"))
    monkeypatch.setattr(reg, "get_portfolio_proxy", stub("hypothetical"))
    monkeypatch.setattr(reg, "get_options_strat", stub("options_strategy"))
    monkeypatch.setattr(reg, "get_bond_ladder", stub("bond_ladder"))
    monkeypatch.setattr("tools.technicals.get_comprehensive_technicals", stub("technical"))
    monkeypatch.setattr("tools.seasonality.analyze_seasonality", stub("seasonality"))
    monkeypatch.setattr("tools.macro_data.get_global_market_snapshot", stub("macro_data"))
    monkeypatch.setattr("tools.monte_carlo.run_monte_carlo", stub("retirement"))
    monkeypatch.setattr("tools.fx_utils.analyze_my_portfolio_fx", stub("fx"))
    monkeypatch.setattr("tools.portfolio_analytics.calculate_portfolio_metrics", lambda symbols: {"metrics": {"sharpe": 1}, "interpretation": {"risk": "ok"}})
    monkeypatch.setattr("tools.portfolio_analytics.calculate_var", lambda symbols: {"value_at_risk": {"daily": "$1"}})
    monkeypatch.setattr("tools.portfolio_csv.get_tradeable_symbols", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr("tools.esg_analytics.check_esg_scores", stub("esg"))
    monkeypatch.setattr("tools.fund_analytics.analyze_mutual_funds", stub("funds"))
    monkeypatch.setattr("tools.portfolio_csv.load_portfolio", lambda: [{"symbol": "AAPL", "shares": 10}])
    monkeypatch.setattr("tools.income_analytics.project_portfolio_income", stub("income"))
    monkeypatch.setattr("tools.backtesting.backtest_strategy", stub("backtest"))
    monkeypatch.setattr("tools.sector_analysis.check_portfolio_allocation", stub("allocation"))
    monkeypatch.setattr("tools.web_search.search_news", stub("web"))
    monkeypatch.setattr(reg, "read_web_page", stub("reader"))

    assert reg.screen_stocks.invoke({"criteria": "Energy"})["called"] == "screen"
    assert reg.get_historical_performance.invoke({"symbol": "AAPL"})["called"] == "historical"
    assert reg.get_competitors.invoke({"symbol": "AAPL"})["called"] == "competitors"
    assert reg.get_etf_holdings_data.invoke({"symbol": "SPY"})["called"] == "etf"
    assert reg.structure_trade_setup.invoke({"symbol": "AAPL"})["called"] == "trade_setup"
    assert reg.get_hypothetical_portfolio.invoke({"persona": "Balanced"})["called"] == "hypothetical"
    assert reg.model_options_strategy.invoke({"symbol": "AAPL", "strategy_type": "collar"})["called"] == "options_strategy"
    assert reg.construct_bond_ladder.invoke({"amount": 50000, "currency": "USD"})["called"] == "bond_ladder"
    assert reg.run_technical_analysis.invoke({"symbol": "AAPL"})["called"] == "technical"
    assert reg.get_seasonality_data.invoke({"symbol": "AAPL"})["called"] == "seasonality"
    assert reg.get_global_indices.invoke({})["called"] == "macro_data"
    assert reg.project_retirement_goal.invoke({"current_value": 1000, "monthly_contribution": 100, "years": 5})["called"] == "retirement"
    assert reg.check_fx_impact.invoke({"base_currency": "CAD"})["called"] == "fx"
    assert reg.check_risk_metrics.invoke({"symbols": "AAPL,MSFT"})["risk_metrics"] == {"sharpe": 1}
    assert reg.check_esg_scores.invoke({"symbols": "AAPL,MSFT"})["called"] == "esg"
    assert reg.analyze_mutual_funds.invoke({"symbols": "FXAIX"})["called"] == "funds"
    assert reg.project_portfolio_income.invoke({})["called"] == "income"
    assert reg.backtest_strategy.invoke({"strategy_type": "rsi", "symbols": "AAPL", "period": "1y", "details": "30,70"})["called"] == "backtest"
    assert reg.check_portfolio_allocation.invoke({})["called"] == "allocation"
    assert reg.read_url.invoke({"url": "https://example.com"})["called"] == "reader"

    monkeypatch.setattr("tools.portfolio_csv.get_tradeable_symbols", lambda: [])
    assert reg.check_portfolio_earnings.invoke({}) == "Portfolio empty or includes no tradeable positions."


def test_tool_registry_portfolio_risk_metrics_report(monkeypatch):
    import agent.tool_registry as reg

    monkeypatch.setattr(reg, "get_portfolio_summary", lambda: {
        "holdings": [
            {"symbol": "AAPL", "value_usd": 1000.0},
            {"symbol": "MSFT", "value_usd": 1000.0},
            {"symbol": "PRIVATE_FUND", "value_usd": 500.0},
        ],
        "total_value_usd": 2500.0,
        "total_value_cad": 3500.0,
    })
    monkeypatch.setattr(reg, "get_tradeable_symbols", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr("tools.portfolio_analytics.calculate_portfolio_metrics", lambda symbols, weights=None: {
        "metrics": {
            "annual_return": "8.8%",
            "sharpe_ratio": 1.4,
            "sortino_ratio": 2.0,
            "beta": 0.8,
            "max_drawdown": "-9%",
            "annual_volatility": "11%",
        },
        "interpretation": {"sharpe": "Good", "beta": "Defensive", "volatility": "Low"},
    })
    monkeypatch.setattr("tools.portfolio_analytics.calculate_var", lambda symbols, weights=None, investment=1000000: {
        "value_at_risk": {"daily_var_dollars": "$2,500"}
    })
    monkeypatch.setattr("tools.portfolio_analytics.analyze_correlation", lambda symbols: {
        "average_correlation": 0.42,
        "correlation_pairs": [{"pair": "AAPL/MSFT", "correlation": 0.55}],
    })
    monkeypatch.setattr("tools.portfolio_analytics.get_sector_exposure", lambda symbols, weights=None, is_portfolio=False: {
        "concentration_warning": "Technology is concentrated"
    })
    monkeypatch.setattr("tools.portfolio_analytics.get_fee_income_analysis", lambda symbols, weights=None: {
        "weighted_expense_ratio": 0.0009,
        "expected_yield": 0.0162,
        "fee_rating": "Excellent",
    })
    monkeypatch.setattr("tools.portfolio_analytics.generate_portfolio_charts", lambda symbols: {
        "risk": '{"data":[]}'
    })

    portfolio_report = reg.get_portfolio_risk_metrics.invoke({"symbols": "PORTFOLIO", "investment": 100000})
    assert "My Portfolio" in portfolio_report
    assert "Sharpe Ratio" in portfolio_report
    assert "Key Metrics (Canonical Calculated Values)" in portfolio_report
    assert "| Expected Return | 8.8% annualized | Reasonable |" in portfolio_report
    assert "| Dividend Income | $57/yr (1.62% yield) | Estimated from weighted dividend yield |" in portfolio_report
    assert "| Daily VaR (95%) | -$2,500 | Worst expected single-day loss |" in portfolio_report
    assert "Technology is concentrated" in portfolio_report
    assert "[PLOTLY_JSON:" in portfolio_report

    many_symbols = ",".join(f"S{i}" for i in range(18))
    condensed = reg.get_portfolio_risk_metrics.invoke({"symbols": many_symbols, "investment": 250000})
    assert "Institutional Portfolio Report" in condensed
    assert "Sharpe 1.4" in condensed

    monkeypatch.setattr(reg, "get_portfolio_summary", lambda: {"error": "missing csv"})
    assert "Error loading portfolio" in reg.get_portfolio_risk_metrics.invoke({"symbols": "PORTFOLIO"})
    assert "No valid symbols" in reg.get_portfolio_risk_metrics.invoke({"symbols": ""})


def test_tool_registry_portfolio_risk_metrics_sector_uses_whole_portfolio_weights(monkeypatch):
    """Regression: sector exposure must be computed against whole-portfolio weights,
    not the tradeable-only subset renormalized to sum to 100% — excluding a large
    pension/fund holding otherwise inflates whatever sector the remaining tradeable
    equities concentrate in (the source of a real "Technology at 100.0%" false
    concentration warning against a portfolio that was actually ~34% tech)."""
    import agent.tool_registry as reg

    monkeypatch.setattr(reg, "get_portfolio_summary", lambda: {
        "holdings": [
            {"symbol": "AAPL", "value_usd": 1000.0},
            {"symbol": "PENSION_FUND", "value_usd": 4000.0},  # large non-tradeable holding
        ],
        "total_value_usd": 5000.0,
        "total_value_cad": 6500.0,
    })
    monkeypatch.setattr(reg, "get_tradeable_symbols", lambda: ["AAPL"])
    monkeypatch.setattr("tools.portfolio_analytics.calculate_portfolio_metrics", lambda symbols, weights=None: {"metrics": {}, "interpretation": {}})
    monkeypatch.setattr("tools.portfolio_analytics.calculate_var", lambda symbols, weights=None, investment=1000000: {"value_at_risk": {}})
    monkeypatch.setattr("tools.portfolio_analytics.analyze_correlation", lambda symbols: {})
    monkeypatch.setattr("tools.portfolio_analytics.get_fee_income_analysis", lambda symbols, weights=None: {})
    monkeypatch.setattr("tools.portfolio_analytics.generate_portfolio_charts", lambda symbols: {})

    captured = {}
    def _fake_sector_exposure(symbols, weights=None, is_portfolio=False):
        captured["symbols"] = list(symbols)
        captured["weights"] = list(weights) if weights is not None else None
        captured["is_portfolio"] = is_portfolio
        return {}
    monkeypatch.setattr("tools.portfolio_analytics.get_sector_exposure", _fake_sector_exposure)

    reg.get_portfolio_risk_metrics.invoke({"symbols": "PORTFOLIO", "investment": 100000})

    assert "PENSION_FUND" in captured["symbols"], captured
    assert "AAPL" in captured["symbols"], captured
    aapl_weight = captured["weights"][captured["symbols"].index("AAPL")]
    # AAPL's TRUE portfolio-wide weight is 1000/5000 = 20%, not 100% (its
    # tradeable-only-renormalized weight, since it's the only tradeable symbol).
    assert abs(aapl_weight - 0.2) < 1e-6, captured
    # A whole-portfolio call must be flagged so it's allowed to persist to the
    # knowledge graph; a single-symbol lookup elsewhere must not be (see
    # test_get_sector_exposure_does_not_persist_narrow_symbol_lookups).
    assert captured["is_portfolio"] is True, captured


def test_tool_registry_portfolio_sectors_uses_whole_portfolio_weights(monkeypatch):
    """Same regression as above, for the dedicated get_portfolio_sectors tool,
    whose docstring promises the "full, verified, dollar-weighted holdings" but
    previously only ever aggregated the tradeable-equity subset."""
    import agent.tool_registry as reg

    monkeypatch.setattr(reg, "get_portfolio_summary", lambda: {
        "holdings": [
            {"symbol": "AAPL", "value_usd": 1000.0},
            {"symbol": "PENSION_FUND", "value_usd": 4000.0},
        ],
        "total_value_usd": 5000.0,
        "total_value_cad": 6500.0,
    })
    monkeypatch.setattr(reg, "get_tradeable_symbols", lambda: ["AAPL"])
    # get_portfolio_sectors uses the module-level (not a locally re-imported)
    # get_sector_exposure/get_geographic_exposure/analyze_factors names, so patch
    # them as bound in agent.tool_registry's own namespace.
    monkeypatch.setattr(reg, "get_geographic_exposure", lambda symbols: {})
    monkeypatch.setattr(reg, "analyze_factors", lambda symbols: {})

    captured = {}
    def _fake_sector_exposure(symbols, weights=None, is_portfolio=False):
        captured["symbols"] = list(symbols)
        captured["weights"] = list(weights) if weights is not None else None
        captured["is_portfolio"] = is_portfolio
        return {}
    monkeypatch.setattr(reg, "get_sector_exposure", _fake_sector_exposure)

    reg.get_portfolio_sectors.invoke({"symbols": "PORTFOLIO"})

    assert "PENSION_FUND" in captured["symbols"], captured
    aapl_weight = captured["weights"][captured["symbols"].index("AAPL")]
    assert abs(aapl_weight - 0.2) < 1e-6, captured
    assert captured["is_portfolio"] is True, captured
