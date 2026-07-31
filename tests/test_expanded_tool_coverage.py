import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def no_persistent_daily_cache(monkeypatch):
    monkeypatch.setattr("tools.daily_cache.get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.daily_cache.set_cached", lambda *args, **kwargs: None)


class _InfoTicker:
    def __init__(self, info=None, history=None, news=None, calendar=None):
        self.info = info or {}
        self._history = history if history is not None else pd.DataFrame()
        self.news = news or []
        self.calendar = calendar

    def history(self, *args, **kwargs):
        return self._history


def _price_history(length=90, start=100.0, step=1.0, volume=1000):
    idx = pd.date_range("2025-01-01", periods=length, freq="B")
    close = pd.Series(start + np.arange(length) * step, index=idx)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 2,
            "Low": close - 2,
            "Volume": np.full(length, volume),
        },
        index=idx,
    )


def _staged_price_history(base=100.0, old_step=0.0, recent_step=0.0, length=90):
    idx = pd.date_range("2025-01-01", periods=length, freq="B")
    values = np.full(length, base, dtype=float)
    for i in range(1, length):
        values[i] = values[i - 1] + (recent_step if i >= length - 22 else old_step)
    close = pd.Series(values, index=idx)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 2,
            "Low": close - 2,
            "Volume": np.full(length, 1000),
        },
        index=idx,
    )


def test_esg_and_mutual_fund_analysis_cover_database_and_fallback_paths(monkeypatch):
    import tools.esg_analytics as esg
    import tools.fund_analytics as funds

    monkeypatch.setattr(
        esg.yf,
        "Ticker",
        lambda symbol: _InfoTicker({"sector": "Energy" if symbol == "XOM2" else "Technology"}),
    )
    esg_result = esg.check_esg_scores(["MSFT", "XOM2", "UNKNOWN"])

    assert esg_result["portfolio_avg_esg_score"] > 0
    assert any(item["symbol"] == "XOM2" for item in esg_result["details"])
    assert any("Controversial" in warning for warning in esg_result["warnings"])

    monkeypatch.setattr(
        funds.yf,
        "Ticker",
        lambda symbol: _InfoTicker(
            {
                "longName": "Live Growth Fund",
                "expenseRatio": 0.006,
                "category": "Growth",
            }
        ),
    )
    class MockNodes:
        def get(self, sym, default=None):
            return {"asset_type": "Private", "expense_ratio": 0.0185, "name": "Private Fund"} if sym == "PRIVATE_FUND" else default

    monkeypatch.setattr("tools.graph_memory.graph_memory.graph.nodes", MockNodes())

    fund_result = funds.analyze_mutual_funds(["PRIVATE_FUND", "LIVEFUND"])

    assert fund_result["funds_analyzed"] == 2
    assert fund_result["details"][0]["data_source"] == "Knowledge Graph"
    assert fund_result["average_expense_ratio"] != "0.00%"
    assert fund_result["warnings"]


def test_sector_rotation_detects_risk_on_and_helpers(monkeypatch):
    import tools.sector_rotation as rotation

    sector_curves = {
        "XLK": _staged_price_history(100, old_step=0.0, recent_step=2.0),
        "XLY": _staged_price_history(90, old_step=0.0, recent_step=1.8),
        "XLF": _staged_price_history(80, old_step=0.0, recent_step=1.6),
        "XLI": _staged_price_history(85, old_step=0.0, recent_step=1.5),
        "XLU": _staged_price_history(160, old_step=0.2, recent_step=-0.8),
        "XLP": _staged_price_history(150, old_step=0.2, recent_step=-0.7),
        "XLV": _staged_price_history(140, old_step=0.2, recent_step=-0.6),
    }

    def fake_ticker(symbol):
        return _InfoTicker(history=sector_curves.get(symbol, _price_history(90, 100, 0.05)))

    monkeypatch.setattr(rotation.yf, "Ticker", fake_ticker)

    result = rotation.detect_sector_rotation()
    assert result["sector_performance"][0]["momentum_score"] >= result["sector_performance"][-1]["momentum_score"]
    assert any(p["pattern"] == "RISK-ON ROTATION" for p in result["rotation_patterns"])
    assert rotation.get_sector_momentum_ranking()
    assert rotation.is_risk_off_rotation() is False


def test_portfolio_analytics_risk_factor_sector_fee_and_chart_paths(monkeypatch):
    import tools.portfolio_analytics as pa

    idx = pd.date_range("2025-01-01", periods=80, freq="B")
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.01, 0.02, len(idx)),
            "MSFT": np.linspace(0.015, -0.005, len(idx)),
            "SPY": np.linspace(0.002, 0.008, len(idx)),
        },
        index=idx,
    )

    def fake_returns(symbols, period="1y"):
        valid = [s for s in symbols if s in returns.columns]
        if not valid:
            return pd.DataFrame(), []
        return returns[valid], valid

    monkeypatch.setattr(pa, "_get_returns", fake_returns)
    monkeypatch.setitem(
        sys.modules,
        "tools.graph_memory",
        types.SimpleNamespace(
            graph_memory=types.SimpleNamespace(add_portfolio_context=lambda **kwargs: None)
        ),
    )
    monkeypatch.setattr(
        pa.yf,
        "Ticker",
        lambda symbol: _InfoTicker(
            {
                "trailingPE": 35 if symbol == "AAPL" else 12,
                "revenueGrowth": 0.2,
                "returnOnEquity": 0.25,
                "country": "United States",
                "expenseRatio": 0.007,
                "dividendYield": 2.0,  # a PERCENT, as the provider sends it
                "sector": "Technology",
                "industry": "Software",
            }
        ),
    )

    metrics = pa.calculate_portfolio_metrics(["AAPL", "MSFT"], [0.7, 0.3])
    corr = pa.analyze_correlation(["AAPL", "MSFT", "SPY"])
    factors = pa.analyze_factors(["AAPL", "MSFT"])
    geo = pa.get_geographic_exposure(["AAPL", "MSFT"])
    var = pa.calculate_var(["AAPL", "MSFT"], [0.5, 0.5], investment=50000)
    sector = pa.get_sector_exposure(["AAPL", "MSFT"], [0.5, 0.5])
    fee = pa.get_fee_income_analysis(["AAPL", "MSFT"], [0.5, 0.5])
    charts = pa.generate_portfolio_charts(["AAPL", "MSFT"])

    assert metrics["metrics"]["beta"] != "N/A"
    assert any("SPY" in warning for warning in corr["hidden_correlation_warnings"])
    assert factors["factor_counts"]["Growth"] >= 1
    assert geo["geographic_counts"]["United States"] == 2
    assert var["investment"] == "$50,000"
    assert sector["sector_breakdown"]["Technology"] == 100.0
    assert fee["high_fee_funds"]
    assert {"performance_chart", "drawdown_chart", "correlation_chart"} <= set(charts)


def test_get_sector_exposure_does_not_persist_narrow_symbol_lookups(monkeypatch):
    """A single-symbol sector lookup (e.g. checking MU's sector before a buy
    decision) must NOT overwrite the knowledge graph's portfolio-wide EXPOSED_TO
    edges — a lone stock always resolves to 100% of its own sector, and writing
    that in as if it were the whole portfolio's exposure corrupts every later
    turn's injected memory context with a bogus "100% Technology" claim. Only a
    call that actually represents the whole portfolio (is_portfolio=True) may
    persist to the graph.
    """
    import tools.portfolio_analytics as pa

    mock_graph_memory = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "tools.graph_memory",
        types.SimpleNamespace(graph_memory=mock_graph_memory),
    )
    monkeypatch.setattr(pa, "_get_sector_for_ticker", lambda symbol: "Technology")

    pa.get_sector_exposure(["MU"])
    assert not mock_graph_memory.add_portfolio_context.called, (
        "Narrow single-symbol get_sector_exposure call persisted to the graph, "
        "clobbering the real portfolio-wide sector breakdown."
    )

    pa.get_sector_exposure(["MU"], is_portfolio=True)
    assert mock_graph_memory.add_portfolio_context.called


def test_opportunity_scanner_scores_and_pipeline_without_network(monkeypatch):
    import tools.opportunity_scanner as opp

    monkeypatch.setattr(opp, "_tv_sector_constituents", lambda sec, **k: ["AAPL", "MSFT", "PLTR"] if "tech" in sec.lower() or "technol" in sec.lower() else ["XOM"])
    monkeypatch.setattr(opp, "is_cancelled", lambda: False)

    assert opp._get_sector_tickers("tech") == ["AAPL", "MSFT", "PLTR"]
    assert "Technology" in opp._get_all_sector_names()
    assert opp._get_sector_for_ticker("AAPL") == "Technology"

    hist = _price_history(230, 80, 0.4)
    technicals = opp._compute_technicals_batch(hist, ["AAPL"])
    assert technicals["AAPL"]["price"] > 0
    assert opp._fast_score("AAPL", {**technicals["AAPL"], "golden_cross": True, "vol_spike": 2.5}) > 40

    news_score, news_reasons, labels = opp._score_news_catalysts(
        ["Company beats expectations and raises guidance", "Analyst downgrade after lawsuit"]
    )
    assert news_score > 0
    assert "Earnings Beat" in labels
    assert opp._count_signal_categories(news_reasons) >= 1
    assert "On Sale" in opp._classify_opportunity(["Thematic Tailwind"], {"rsi": 25}, -25, True)

    import tools.insider_data as insider_data

    monkeypatch.setattr(
        insider_data,
        "get_insider_and_short_data",
        lambda symbol: {
            "insider_signal": "Insiders neutral",
            "short_interest": {"short_percent_of_float": "12%"},
        },
    )

    fund = {
        "revenue_growth": 0.25,
        "earnings_growth": 0.4,
        "profit_margin": 0.26,
        "forward_pe": 20,
        "trailing_pe": 35,
        "peg_ratio": 1.0,
        "analyst_target": 150,
        "current_price": 100,
        "52_week_high": 160,
        "recommendation": "strong_buy",
        "description": "Quality software business",
        "news_headlines": ["New product launch and partnership with major bank"],
    }
    tech = {
        "price": 100,
        "above_sma50": True,
        "golden_cross": True,
        "rsi": 28,
        "drawdown_pct": -25,
        "vol_spike": 2.3,
        "three_month_return": 30,
        "month_return": 12,
    }
    scored = opp._deep_score_value_v1(
        "AAPL",
        fund,
        tech,
        "Leading",
        {"is_macro_favored": True, "liquidity": "Expanding"},
        ["AI"],
        headwind_data={"short_pct_float": 0.12, "days_to_earnings": 3},
    )
    assert scored["score"] >= 100
    assert scored["risk_flags"]
    assert scored["foundation_check"]["grade"] in {"Mixed", "Unproven"}
    assert not any("Fallen Angel" in reason for reason in scored["reasons"])
    assert opp._select_qualified_opportunities([{"symbol": "B", "score": 30}, {"symbol": "A", "score": 50}])[0]["symbol"] == "A"

    monkeypatch.setattr(opp, "_batch_download", lambda symbols: pd.DataFrame({"ok": [1]}))
    monkeypatch.setattr(opp, "_compute_technicals_batch", lambda data, symbols: {s: tech for s in symbols})
    monkeypatch.setattr(opp, "_fetch_fundamentals_parallel", lambda symbols: {s: fund for s in symbols})
    monkeypatch.setattr(opp, "_headwind_check_parallel", lambda symbols, **kw: {"AAPL": {"short_pct_float": 0.11}})
    monkeypatch.setattr("tools.market_mechanics.detect_sector_rotation", lambda: {
        "leading_sectors": ["Technology"],
        "lagging_sectors": ["Utilities"],
        "full_rotation_map": [{"sector": "Technology", "trend": "Leading"}],
        "market_status": "Risk-On",
    })
    monkeypatch.setattr("tools.fred_api.get_systemic_risk_indicators", lambda: {
        "liquidity_status": "Expanding",
        "m2_growth_yoy": "5%",
        "crash_risk": "Low",
    })
    monkeypatch.setattr("tools.fred_api.get_treasury_yields", lambda: {"curve_status": "Normal"})
    monkeypatch.setattr("tools.macro_strategy.analyze_macro_context", lambda: {
        "strategy": {"tactical_opportunity": ["Technology"], "sectors_to_underweight": []}
    })

    scan = opp._scan_impl("Technology")
    assert scan["top_picks"]
    assert "Scanned" in scan["summary"]


def test_marginal_risk_contribution_without_network(monkeypatch):
    import tools.portfolio_analytics as pa

    idx = pd.date_range("2025-01-01", periods=80, freq="B")
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.01, 0.012, len(idx)),
            "MSFT": np.linspace(-0.008, 0.01, len(idx)),
            "AFRM": np.linspace(-0.03, 0.035, len(idx)),
        },
        index=idx,
    )

    monkeypatch.setattr(pa, "_get_returns", lambda symbols, period="1y": (returns[[s for s in symbols if s in returns]], [s for s in symbols if s in returns]))

    result = pa.estimate_marginal_risk_contribution(
        ["AAPL", "MSFT"],
        [0.6, 0.4],
        "AFRM",
        candidate_weight=0.10,
    )

    assert result["candidate_symbol"] == "AFRM"
    assert "volatility_delta" in result
    assert "candidate_correlation_to_current_portfolio" in result


def test_geopolitical_scanner_mapping_scan_reverse_lookup_and_quick_check(monkeypatch):
    import tools.gdelt_api
    import tools.geopolitical_scanner as geo

    monkeypatch.setattr(tools.gdelt_api, "get_gdelt_crisis_alerts", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools.gdelt_api, "search_gdelt_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools.gdelt_api, "scan_gdelt_geopolitical", lambda *args, **kwargs: {})

    monkeypatch.setattr(geo, "is_cancelled", lambda: False)
    monkeypatch.setattr(geo, "search_news", lambda *args, **kwargs: "Russia sanctions oil and wheat supply shock")
    monkeypatch.setattr(
        geo,
        "_quick_price_check",
        lambda symbol: {
            "symbol": symbol,
            "price": 50.0,
            "weekly_change_pct": 6.0,
            "monthly_change_pct": 18.0,
            "six_month_change_pct": 25.0,
            "pct_from_6mo_high": -10.0,
            "high_6mo": 60.0,
            "low_6mo": 35.0,
            "volume_spike": 2.2,
            "trending": True,
        },
    )

    assert geo._normalize_country("South Korea") == "south_korea"
    assert {"russia", "ukraine"} <= set(geo._detect_countries_in_text("Russia and Ukraine wheat war"))
    assert {"war", "sanctions"} <= set(geo._detect_event_types("war sanctions blockade"))
    assert "PPLT" in geo._get_tickers_for_commodity("palladium")
    assert "Crude" in geo._get_commodity_description("oil")

    premium = geo._analyze_conflict_premium("oil", 70.0)
    assert premium["status"] == "NOT_PRICED_IN"

    scan = geo.scan_geopolitical_opportunities("Russia sanctions oil and wheat exports")
    assert scan["status"] == "opportunities_found"
    assert scan["top_picks"]

    exposure = geo.get_supply_chain_exposure("Russia")
    assert exposure["commodities"]

    context = geo.get_ticker_geopolitical_context("XOM")
    assert context["exposed"] is True
    assert context["commodity_exposure"]

    quick = geo.quick_geopolitical_check()
    assert quick["alert"] is True


def test_deep_reasoning_helpers_cover_timeout_and_market_pulse(monkeypatch):
    import concurrent.futures

    import agent.nodes.deep_reasoning as dr

    assert dr._is_health_check_query("please run diagnostics")
    assert dr._query_needs_market_pulse("portfolio risk", cache_warm=False)
    assert dr._run_with_timeout(lambda: "done", 1) == "done"
    tagged_memory = dr._format_user_profile_memory_tag("risk_tolerance: aggressive")
    assert tagged_memory.startswith("<user_profile_memory>")
    assert "risk_tolerance: aggressive" in tagged_memory
    assert "<user_memory>" not in tagged_memory

    with pytest.raises(concurrent.futures.TimeoutError):
        dr._run_with_timeout(lambda: __import__("time").sleep(0.05), 0.001)

    monkeypatch.setattr("tools.daily_cache.get_cached", lambda key: {"cached": True})
    monkeypatch.setitem(
        sys.modules,
        "tools.market_sentinel",
        types.SimpleNamespace(
            get_market_regime=lambda: {
                "regime": "Risk-On",
                "regime_score": 72,
                "regime_streak": 4,
                "headline": "Momentum improving",
                "recommendation": "Stay invested",
                "fear_greed": 60,
                "vix": 14,
                "spy_drawdown": "-2%",
            },
            get_regime_history=lambda days=7: {
                "history": [
                    {"date": "2026-04-28", "regime": "Neutral", "score": 50},
                    {"date": "2026-04-29", "regime": "Risk-On", "score": 65},
                ]
            },
        ),
    )

    brief = dr._get_market_pulse_brief("portfolio")
    assert "Current Regime: Risk-On" in brief

    sent_chunks = []
    monkeypatch.setattr(dr, "has_stream_callback", lambda: True)
    monkeypatch.setattr(dr, "send_stream", sent_chunks.append)
    monkeypatch.setattr(dr, "is_cancelled", lambda: False)
    monkeypatch.setattr(dr._time, "sleep", lambda *_: None)
    dr._stream_text_in_chunks("abcdef", chunk_size=2, delay_seconds=0)
    assert sent_chunks == ["ab", "cd", "ef"]

    failing_agent = object()
    monkeypatch.setattr(dr, "safe_invoke", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    message, meta = dr._invoke_planner_with_timeout(failing_agent, [], 2, planner_timeout_seconds=1)
    assert meta["failed"] is True
    assert "Planner failed" in message.content


def test_portfolio_action_gate_catches_keep_exit_framing():
    """Regression: keep/hold/exit framing must trip the verified-holdings gate.

    This phrasing shape once slipped past _is_portfolio_action_query (no
    trim/sell/risk/portfolio keyword), so the heavy-path synthesis writer got no
    grounded portfolio total and fabricated one. Keep/hold/exit framing on an
    owned asset is a portfolio-action query and must fetch verification.
    """
    import agent.nodes.deep_reasoning as dr

    keep_exit_queries = [
        "should i keep my gold etf, i heard it is heading for a crash",
        "should i hold my nvda through earnings",
        "is it time to exit my energy position",
        "thinking of cashing out my tsla",
        "should i dump my bond etf",
        "what should i do with my portfolio",
    ]
    for q in keep_exit_queries:
        assert dr._is_portfolio_action_query(q), q

    # Classic action keywords still trip it.
    for q in ("trim nvda", "rebalance into value", "sell my losers for tax-loss"):
        assert dr._is_portfolio_action_query(q), q

    # A pure market/macro question with no holding intent still does not.
    for q in ("what is the fed doing this week", "how did the s&p 500 close"):
        assert not dr._is_portfolio_action_query(q), q


def test_cycle2_coverage_checklist_helpers():
    """Tiered coverage checklist (Theme 6.1): Tier-2 expectations are inferred and
    soft (named with an opt-out), Tier-1 mandates are hard; nothing is suggested
    for generic queries or unregistered tools."""
    import agent.nodes.deep_reasoning as dr

    tool_map = {"verify_portfolio_holdings": object(), "check_portfolio_allocation": object()}

    # Tier-2 expectations fire only for portfolio-action query shapes...
    assert dr._expected_tools_for_query("should i keep my gold etf", tool_map) == [
        "verify_portfolio_holdings",
        "check_portfolio_allocation",
    ]
    # ...never for generic market questions (the accuracy concern)...
    assert dr._expected_tools_for_query("what is the fed doing this week", tool_map) == []
    # ...never for tools that aren't registered, and never double-listed with Tier 1.
    assert dr._expected_tools_for_query("should i keep my gold etf", {}) == []
    assert dr._expected_tools_for_query(
        "should i keep my gold etf", tool_map, exclude=["verify_portfolio_holdings"]
    ) == ["check_portfolio_allocation"]

    # Checklist prompt: None when nothing is missing (generic prompt is used instead).
    assert dr._coverage_checklist_prompt([], []) is None
    hard = dr._coverage_checklist_prompt(["get_latest_trump_yaps"], [])
    assert "REQUIRED" in hard and "get_latest_trump_yaps" in hard
    soft = dr._coverage_checklist_prompt([], ["verify_portfolio_holdings"])
    assert "unless clearly irrelevant" in soft
    assert "REQUIRED" not in soft  # soft tier must not read as a mandate


def test_coverage_backstop_runs_only_all_optional_arg_tools():
    """The Tier-1 deterministic backstop auto-invokes only tools whose args are all
    optional; tools needing args we can't guess are skipped (not failed), and
    successful output is recorded for synthesis."""
    from langchain_core.tools import tool as lc_tool

    import agent.nodes.deep_reasoning as dr

    @lc_tool
    def no_arg_probe() -> str:
        """Probe tool with no args."""
        return "probe-ok"

    @lc_tool
    def needs_symbol(symbol: str) -> str:
        """Probe tool with a required arg."""
        return symbol

    assert dr._tool_required_args(no_arg_probe) == []
    assert dr._tool_required_args(needs_symbol) == ["symbol"]

    recorded = []
    outcomes = []
    executed = dr._run_coverage_backstop(
        ["no_arg_probe", "needs_symbol", "not_registered"],
        {"no_arg_probe": no_arg_probe, "needs_symbol": needs_symbol},
        lambda name, content: recorded.append((name, str(content))),
        outcomes,
    )
    assert executed == ["no_arg_probe"]
    assert recorded == [("no_arg_probe", "probe-ok")]
    # needs_symbol (required arg) and not_registered are skipped silently — no
    # spurious failure outcomes for tools we knowingly could not invoke.
    assert [o["name"] for o in outcomes] == ["no_arg_probe"]
    assert outcomes[0]["success"] is True


def test_financial_institution_foundation_quality():
    import tools.opportunity_scanner as opp

    # Non-financial company (e.g. standard software/retail company)
    non_financial_fund = {
        "symbol": "AAPL",
        "sector_yf": "Technology",
        "industry": "Consumer Electronics",
        "gross_margin": 0.45,
        "free_cashflow": 10000.0,
        "total_debt": 50000.0,
        "total_cash": 10000.0,
        "ebitda": 20000.0,
    }
    # Net debt / EBITDA = (50000 - 10000) / 20000 = 2.0 (Fails the <= 1.5 corporate leverage check)
    res_non_financial = opp._assess_foundation_quality(non_financial_fund)
    assert res_non_financial["grade"] == "Mixed"  # Failed leverage check (2.0 > 1.5)

    # Financial company (e.g. Affirm — detected via sector + industry, NOT hardcoded symbol list)
    financial_fund = {
        "symbol": "AFRM",
        "sector_yf": "Financial Services",
        "industry": "Credit Services",
        "profit_margin": 0.15,
        "free_cashflow": -100.0,  # Structurally negative/volatile is normal for loan outflows
        "total_debt": 100000.0,   # Highly leveraged
        "total_cash": 1000.0,
        "ebitda": -50.0,          # ebitda often missing/irrelevant
    }
    res_financial = opp._assess_foundation_quality(financial_fund)
    assert res_financial["grade"] == "Strong"  # Profit margin passes, cash flow and leverage checked as standard/normal!

    # Insurance company — never in any hardcoded list, detected purely by industry
    insurance_fund = {
        "symbol": "PGR",
        "sector_yf": "Financial Services",
        "industry": "Insurance—Property & Casualty",
        "profit_margin": 0.12,
        "free_cashflow": -500.0,
        "total_debt": 200000.0,
        "total_cash": 5000.0,
        "ebitda": -100.0,
    }
    res_insurance = opp._assess_foundation_quality(insurance_fund)
    assert res_insurance["grade"] == "Strong"  # Insurance leverage is structurally normal

    # Regional bank — never hardcoded, detected by industry field
    bank_fund = {
        "symbol": "FITB",
        "sector_yf": "Financial Services",
        "industry": "Banks—Regional",
        "profit_margin": 0.28,
        "free_cashflow": -2000.0,
        "total_debt": 500000.0,
        "total_cash": 20000.0,
        "ebitda": 3000.0,
    }
    res_bank = opp._assess_foundation_quality(bank_fund)
    assert res_bank["grade"] == "Strong"  # Bank leverage is structurally normal


def test_sector_for_ticker_3_tier_lookup(monkeypatch):
    import tools.opportunity_scanner as opp

    # Seed universe cache with a small set
    opp._universe_cache = {
        "sectors": {
            "Technology": {"tickers": ["AAPL", "MSFT"], "gems": []},
            "Energy": {"tickers": ["XOM"], "gems": []},
        },
        "thematic": {}
    }

    # Tier 1: Universe file hit — should return "Technology"
    assert opp._get_sector_for_ticker("AAPL") == "Technology"
    assert opp._get_sector_for_ticker("XOM") == "Energy"

    # Unknown ticker with NO fund — falls through all tiers to "Unknown"
    # (Mock KG to avoid real file I/O)
    import types
    mock_graph = types.SimpleNamespace(has_node=lambda s: False, nodes={})
    mock_gm = types.SimpleNamespace(graph=mock_graph)
    monkeypatch.setattr("tools.opportunity_scanner.graph_memory", mock_gm, raising=False)
    # Need to patch the lazy import inside the function
    import tools.graph_memory as gm_mod
    monkeypatch.setattr(gm_mod, "graph_memory", mock_gm)

    # Mock yfinance dynamic fetch
    monkeypatch.setattr("tools.yf_utils.get_info_safe", lambda s: None)

    assert opp._get_sector_for_ticker("RDDT") == "Unknown"

    # Tier 2: Unknown ticker WITH yfinance fund data — should resolve dynamically
    fund_rddt = {"sector_yf": "Communication Services", "industry": "Internet Content & Information"}
    assert opp._get_sector_for_ticker("RDDT", fund=fund_rddt) == "Communication Services"

    # Tier 1 still wins over Tier 2 when both are available
    fund_aapl = {"sector_yf": "Consumer Electronics Bogus"}
    assert opp._get_sector_for_ticker("AAPL", fund=fund_aapl) == "Technology"  # Universe wins


def test_health_check_new_keywords_and_portfolio_integrity(monkeypatch):
    import agent.nodes.deep_reasoning as dr
    # Test keywords
    assert dr._is_health_check_query("Check system health and portfolio integrity.")
    assert dr._is_health_check_query("portfolio integrity check")
    assert dr._is_health_check_query("system health status")

    # Test synthesize health check mocks
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=MagicMock(content="[DeepReasoning]: System Health OK. Portfolio Integrity OK."))
    monkeypatch.setattr(dr, "get_sonnet_llm", lambda: mock_llm)
    monkeypatch.setattr(dr, "has_stream_callback", lambda: False)
    monkeypatch.setattr(dr, "send_status", lambda *args, **kwargs: None)

    # Mock get_portfolio_decision_context
    mock_context = {
        "is_stale": False,
        "sync_errors": [],
        "total_value_cad": 100000.0,
        "total_value_usd": 70000.0,
        "as_of": "2026-05-20",
        "holdings": [{"symbol": "AAPL"}]
    }
    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: mock_context)

    health_report = {
        "health_summary": {
            "overall_status": "🟢 ALL SYSTEMS GO",
            "operational": 90,
            "failed": 0,
            "total_checked": 90,
            "prerequisites": {},
            "missing_prerequisites": []
        },
        "tool_results": [],
        "agent_instructions": "Check OK"
    }

    res = dr._synthesize_health_check(health_report)
    assert res["messages"]
    assert "System Health OK" in res["messages"][0].content


def test_holdings_dispute_avoidance_on_system_instructions(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    import agent.nodes.deep_reasoning as dr

    # Mock get_portfolio_decision_context
    mock_context = {
        "is_stale": False,
        "sync_errors": [],
        "total_value_cad": 100000.0,
        "total_value_usd": 70000.0,
        "as_of": "2026-05-20",
        "holdings": [{"symbol": "AAPL", "account": "Brokerage", "value_cad": 2900.00, "allocation": 0.029, "source": "Manual"}]
    }
    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: mock_context)
    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_summary", lambda: (_ for _ in ()).throw(ValueError("Bypassed dispute check successfully!")))
    monkeypatch.setattr(dr, "has_stream_callback", lambda: False)
    monkeypatch.setattr(dr, "send_status", lambda *args, **kwargs: None)

    # Mock LLM to avoid real API calls. The HEAVY path also builds the planner via
    # get_sonnet_llm(max_tokens=...), which constructs a real Bedrock client and fails
    # in credential-less CI before the test reaches its assertion — mock it too.
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=AIMessage(content="Mocked LLM Response"))
    monkeypatch.setattr(dr, "get_llm", lambda: mock_llm)
    monkeypatch.setattr(dr, "get_sonnet_llm", lambda *args, **kwargs: mock_llm)

    # Case 1: System instruction containing "fabricate"
    state_system = {
        "messages": [
            HumanMessage(content="[DeepReasoning] [System Instruction: Political & Social Media Market Impact Analyst] Do not fabricate facts. Analyze Trump's post.")
        ],
        "data_context": {},
        "summary": "",
        "user_framework": "",
    }

    # This should bypass the holdings dispute check and proceed to the main path, where it will eventually call get_portfolio_summary
    # resulting in our ValueError.
    res_system = dr.deep_reasoning_node(state_system)
    assert "messages" in res_system
    content = res_system["messages"][0].content
    assert "Bypassed dispute check successfully!" in content

    # Case 2: Normal prompt containing "fabricate" without system instruction headers
    state_normal = {
        "messages": [
            HumanMessage(content="Did you fabricate this list? Please verify my portfolio.")
        ],
        "data_context": {},
        "summary": "",
        "user_framework": "",
    }

    # This should NOT bypass the holdings dispute check, meaning it returns the holdings dispute response instead of proceeding.
    res_normal = dr.deep_reasoning_node(state_normal)
    assert "messages" in res_normal
    content_normal = res_normal["messages"][0].content
    assert "You're right to challenge that." in content_normal
    assert "AAPL" in content_normal
    assert "Bypassed dispute check successfully!" not in content_normal


def test_compliance_retry_regenerates_from_tools_not_stale_context(monkeypatch):
    """A RiskManager <compliance_correction_required> retry must re-verify with tools.

    Originally this guarded against the retry falling into "Path A", a
    Tree-of-Thought branch that re-ran the DSPy pipeline purely from the cached
    data_context fundamentals/technicals/news left over from the failed first
    pass — never reading the correction text, so it regenerated the very figures
    the RiskManager had just rejected. The correction text quotes its own
    critique (e.g. "violates your active investment thesis"), which contains
    deep-dive keywords like "thesis" and used to false-positive the routing.

    Path A was removed on 2026-07-31 as unreachable (its `data_context['symbol']`
    gate was never written by anything in production), so the specific trapdoor
    is gone. The property it protected is not: a retry must not answer out of the
    stale context that failed. Asserting that directly, rather than against a
    sentinel for code that no longer exists.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    import agent.nodes.deep_reasoning as dr

    monkeypatch.setattr(dr, "has_stream_callback", lambda: False)
    monkeypatch.setattr(dr, "send_status", lambda *args, **kwargs: None)

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=AIMessage(content="Mocked LLM Response"))
    monkeypatch.setattr(dr, "get_llm", lambda: mock_llm)
    monkeypatch.setattr(dr, "get_sonnet_llm", lambda *args, **kwargs: mock_llm)

    correction_msg = HumanMessage(content=(
        "<compliance_correction_required>\n"
        "Your previous response was flagged by the Risk Manager with CRITICAL violations.\n"
        "Thesis Contradiction (Rule 10): Recommending an $850 stop loss for MU directly violates "
        "your active investment thesis, which strictly anchors MU's stop loss at $600.\n"
        "</compliance_correction_required>"
    ))

    state = {
        "messages": [
            HumanMessage(content="Build a market-dip deployment plan: what quality names to accumulate, at what levels."),
            AIMessage(content="[RiskManager]: prior risk verdict text", name="RiskManager"),
            correction_msg,
        ],
        "data_context": {
            "symbol": "MU",
            "fundamentals": ["stale fundamentals from the rejected first pass"],
            "technicals": ["stale technicals from the rejected first pass"],
            "news": ["stale news from the rejected first pass"],
        },
        "summary": "",
        "user_framework": "",
    }

    result = dr.deep_reasoning_node(state)

    content = result["messages"][0].content
    # The retry must not simply replay the rejected pass's cached context.
    assert "stale fundamentals from the rejected first pass" not in content
    assert "stale technicals from the rejected first pass" not in content
    assert "stale news from the rejected first pass" not in content


def test_grounding_audit_ignores_technical_and_title_acronyms(monkeypatch):
    """Common technical-indicator and C-suite acronyms (EMA, SMA, MACD, RSI, CEO,
    CFO...) are 1-5 uppercase letters, matching the ticker-shape regex. "close" is
    also a routine word in this domain for a PRICE close ("close above its 21-day
    EMA"), not an instruction to close a position — so it must not count as a sell
    verb. Both false-positive sources previously produced bogus "Recommended to
    sell/trim EMA/SMA/CEO, but ... is NOT currently held" grounding errors on
    ordinary technical-analysis prose, tanking an otherwise-correct response's
    compliance score.
    """
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "AAPL"}]},
    )

    text = (
        "MU must reclaim and close above its 21-day EMA ($1,016.76) to confirm strength. "
        "It is trading above its 50-day SMA ($880.08), but the MACD is bearish and RSI is neutral. "
        "CEO Sanjay Mehrotra sold $84.7 million USD in shares between late May and late June."
    )
    violations = rm.run_deterministic_grounding_audit(text)
    assert violations == [], violations


def test_grounding_audit_ignores_third_party_and_market_selling(monkeypatch):
    """Insider/institutional selling and market sell-offs describe what OTHERS
    did — they are evidence the advice reasons FROM, not instructions to the
    user. Real regression: "Keep MU on WATCHING. Heavy insider selling (CEO sold
    ~$84M) confirms delaying entry" was flagged as "Recommended to sell/trim MU",
    the exact opposite of the advice (stay out, keep watching). The audit feeds
    the judge as a CONFIRMED fact, so it also tanked an otherwise-10/10 verdict
    and fired the user-facing compliance-warning banner on both attempts.
    """
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "AAPL"}]},
    )

    for text in (
        "Keep MU on WATCHING. Heavy insider selling (CEO sold ~$84M) confirms delaying entry.",
        "MU faces institutional selling pressure after the print.",
        "A broad sell-off dragged MU down 6% this week.",
        "Shares of MU were sold by insiders throughout June.",
    ):
        assert rm.run_deterministic_grounding_audit(text) == [], text


def test_grounding_audit_ignores_policy_acronyms_and_modifiers(monkeypatch):
    """The advisor's own compliance vocabulary is not a shopping list. "Enforce
    the IPS Limit: you must trim your Technology exposure" put a sell verb 20
    chars from the token "IPS", so the audit reported "Recommended to sell/trim
    IPS, but IPS is NOT currently held" — and since a grounding error caps the
    verdict, a clean draft was capped at 6/10 over a word the advisor wrote
    itself. The judge then explained the parser misfire to the user, who has no
    visibility into it.

    Two guards: the acronym ignore-list (IPS, FX, TSX, IRA...), and the general
    modifier form — a ticker-shaped token whose every mention sits in front of a
    policy noun ("the XYZ limit") is vocabulary, not a holding.
    """
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "AAPL"}]},
    )

    for text in (
        "Enforce the IPS Limit: you must trim your Technology exposure back down to the 30% limit.",
        "Your unhedged FX exposure argues for trimming the US sleeve.",
        "Reduce the position to respect your ESG mandate and the TSX concentration cap.",
        "You are breaching the XYZ limit — sell down to the threshold.",
    ):
        assert rm.run_deterministic_grounding_audit(text) == [], text


def test_policy_modifier_guard_keeps_real_tickers_scannable(monkeypatch):
    """The modifier guard fires only when EVERY mention is modifier-shaped. A
    name discussed as a policy breach AND as a trade must stay auditable."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "AAPL"}]},
    )

    text = "NVDA breaches your NVDA limit at 14% of the book. Trim NVDA back to 10%."
    assert any("NVDA" in v for v in rm.run_deterministic_grounding_audit(text))


def test_grounding_audit_does_not_bleed_sell_verbs_across_bullets(monkeypatch):
    """A sell verb belongs to the bullet it is written in. The proximity window
    used a bare ±60 characters, which reaches over a line break into the NEXT
    list item, so the verb got pinned on whatever ticker leads it.

    Real regression, from the same run as the IPS false positive: a watchlist
    block whose first bullet ended "...weight is trimmed back below the 30%
    policy limit." followed by "* Watchlist (MU): the Micron thesis remains
    dropped/closed ... We do not chase." produced "Recommended to sell/trim MU,
    but MU is NOT currently held" — the verb was about the Technology sleeve,
    and the draft had explicitly said it would not touch MU.
    """
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "XLK"}]},
    )

    text = (
        "🔭 Early Signals / Watch\n\n"
        "* Watchlist (NVDA / PLTR): We are actively monitoring these for potential entry. "
        "Trigger: entries execute only once your Technology sector weight is trimmed back "
        "below the 30% policy limit.\n"
        "* Watchlist (MU): The Micron (MU) thesis remains dropped/closed, as the price "
        "gapped well above our entry target. We do not chase.\n"
    )
    assert rm.run_deterministic_grounding_audit(text) == []


def test_grounding_audit_catches_sell_verbs_inside_the_same_bullet(monkeypatch):
    """The block clamp must not disarm the check within one list item."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "XLK"}]},
    )

    text = (
        "1. Clean the Clutter: Liquidate the remnants of GME and AMC.\n"
        "2. Enforce the limit: trim your Technology exposure to 30%.\n"
    )
    violations = rm.run_deterministic_grounding_audit(text)
    assert len(violations) == 2, violations
    assert any("GME" in v for v in violations) and any("AMC" in v for v in violations), violations


def test_grounding_audit_still_catches_real_unheld_sell_recommendations(monkeypatch):
    """Sanity check that suppressing the EMA/SMA/CEO/"close" false positives didn't
    also disable the real check: a genuine recommendation to sell/trim a ticker the
    user doesn't hold must still be flagged."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "AAPL"}]},
    )

    text = "You should sell your NVDA position to fund this purchase."
    violations = rm.run_deterministic_grounding_audit(text)
    assert any("NVDA" in v for v in violations), violations


_TOTAL_AUDIT_CTX = {
    "holdings": [
        {"symbol": "AAPL", "allocation_pct": 2.94},
        {"symbol": "BCE.TO", "allocation_pct": 5.0},
    ],
    "total_value_base": 450000.0,
    "total_value_cad": 450000.0,
    "total_value_usd": 330000.0,
    "base_currency": "CAD",
}


def test_total_audit_accepts_correct_headline_figures(monkeypatch):
    """A portfolio-total headline that matches the verified CAD or USD total
    (with or without an explicit currency label) must not be flagged."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: _TOTAL_AUDIT_CTX)

    assert rm.run_deterministic_total_audit("Your portfolio is worth $450,000 CAD today.") == []
    assert rm.run_deterministic_total_audit("Your portfolio is valued at $330,000 USD.") == []


def test_total_audit_catches_wrong_total_and_currency_mislabel(monkeypatch):
    """A headline total that matches neither verified total, or that labels a
    figure with the wrong currency, must be flagged."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: _TOTAL_AUDIT_CTX)

    wrong = rm.run_deterministic_total_audit("Your portfolio is worth $600,000 CAD today.")
    assert any("600,000" in v for v in wrong), wrong

    mislabeled = rm.run_deterministic_total_audit("Your portfolio is worth $330,000 CAD today.")
    assert any("labels portfolio total" in v for v in mislabeled), mislabeled


def test_total_audit_ignores_unrelated_dollar_figures(monkeypatch):
    """A dollar figure that merely follows the word 'portfolio' in the same
    sentence (a holding's price, a cash balance) without valuation-keyword
    framing must not be treated as a portfolio-total claim."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: _TOTAL_AUDIT_CTX)

    text = "Your portfolio holds AAPL at $232 and a cash balance of $15,000."
    assert rm.run_deterministic_total_audit(text) == []


def test_total_audit_ignores_figures_derived_from_the_total(monkeypatch):
    """A percentage OF the total is not a claim ABOUT the total.

    "your total portfolio's 2% maximum risk limit ($9,000 CAD)" clears the
    valuation-keyword gate on the word "total", so the audit used to report the
    correctly-computed 2% limit as a hallucinated portfolio total — a false
    grounding error, which caps the verdict at ≤6/10 and forces a retry of
    advice that was right.
    """
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: _TOTAL_AUDIT_CTX)

    derived = [
        "Dollar-at-risk is measured against your total portfolio's 2% maximum risk limit ($9,000 CAD).",
        "That leaves portfolio headroom of $45,000 CAD before the sector cap binds.",
        "Total portfolio drawdown at the stop would be $12,500 CAD.",
    ]
    for text in derived:
        assert rm.run_deterministic_total_audit(text) == [], text

    # The guard must not blind the audit to a real wrong-total claim in the
    # same sentence shape.
    wrong = rm.run_deterministic_total_audit("Your total portfolio is worth $600,000 CAD.")
    assert any("600,000" in v for v in wrong), wrong


def test_total_audit_is_currency_agnostic_not_cad_usd_only(monkeypatch):
    """CairnIQ's base currency is user-configurable across USD/CAD/EUR/GBP/AUD/JPY
    (tools.memory.SUPPORTED_BASE_CURRENCIES) — a correct headline in the
    profile's actual base currency (here EUR, via total_value_base) must not
    be flagged just because it isn't CAD or USD."""
    import agent.nodes.risk_manager as rm

    eur_ctx = {
        "holdings": [],
        "total_value_base": 380000.0,
        "total_value_cad": 570000.0,
        "total_value_usd": 415000.0,
        "base_currency": "EUR",
    }
    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: eur_ctx)

    assert rm.run_deterministic_total_audit("Your portfolio is worth €380,000 today.") == []
    assert rm.run_deterministic_total_audit("Your portfolio is worth $380,000 EUR today.") == []
    # The CAD-equivalent conversion the app also computes is an equally valid citation.
    assert rm.run_deterministic_total_audit("Your portfolio is worth $570,000 CAD.") == []

    wrong = rm.run_deterministic_total_audit("Your portfolio is worth €999,000 today.")
    assert any("999,000" in v and "EUR" in v for v in wrong), wrong


def test_price_audit_catches_stale_current_price_claim(monkeypatch):
    """An explicit CURRENT-price claim ("trading at $X") that doesn't match the
    cached live quote must be flagged; a correct one must not."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.market_data.get_realtime_quote", lambda symbol: {"price": 692.86})

    assert rm.run_deterministic_price_audit("LITE is currently trading at $692.86, a strong setup.") == []

    wrong = rm.run_deterministic_price_audit("LITE is currently trading at $450.00, a strong setup.")
    assert any("LITE" in v and "450.00" in v for v in wrong), wrong


def test_price_audit_ignores_proposed_entry_stop_target_prices(monkeypatch):
    """Proposed entry/stop/target prices are deliberately different from the
    live price by design and must never be flagged as a mismatch — this is the
    exact false-positive trap that made the not-held-ticker check unreliable
    before it was hardened."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.market_data.get_realtime_quote", lambda symbol: {"price": 692.86})

    text = "LITE Entry Zone: $620.00 - $650.00. Structural Stop: $580.00. Target: $800."
    assert rm.run_deterministic_price_audit(text) == []


def test_allocation_audit_catches_wrong_percentage(monkeypatch):
    """An explicit portfolio-allocation percentage claim that doesn't match the
    verified allocation_pct must be flagged; a correct one must not, and an
    unrelated percentage (e.g. a daily return) must not be mistaken for one."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: _TOTAL_AUDIT_CTX)

    assert rm.run_deterministic_allocation_audit("AAPL is 2.9% of your portfolio.") == []

    wrong = rm.run_deterministic_allocation_audit("AAPL is 12% of your portfolio.")
    assert any("AAPL" in v and "12.0%" in v for v in wrong), wrong

    assert rm.run_deterministic_allocation_audit("AAPL is up 2.9% today.") == []


# --- Regression: EventScenario / DeepReasoning section headers must not be
# mined for phantom tickers, and negated "do not sell" guidance must not be
# read as a sell recommendation (see the Trump/Iran EventScenario console). ---

_EVENT_SCENARIO_ADVICE = (
    "📢 CATALYST\n\n"
    "Trump confirmed military retribution against Iran (July 8, 2026). Confidence: High.\n\n"
    "🔗 EXPOSURE MAP\n\n"
    "Winners: Energy/Defense. Losers: Broad Tech/Transports.\n\n"
    "🎲 SCENARIOS\n\n"
    "CASE\t~PROB\tMECHANISM\tTIME\tCONFIRMS/INVALIDATES\n"
    "Base\t60%\tLocalized strikes, oil premium\t1-4w\tCrude spikes / De-escalation\n"
    "Bear\t30%\tRegional war, risk-off\t1-3m\tVIX >20 / Diplomatic resolution\n\n"
    "💼 PORTFOLIO EXPOSURE\n\n"
    "VET.TO: $5,000.00 CAD (1.50%) - Direct energy exposure.\n"
    "Tech Sleeve: ~25.0% - Vulnerable to risk-off.\n\n"
    "⚡ TRIGGER PLAN\n\n"
    "Hold VET.TO for oil upside.\n"
    "Do not trim Tech unless VIX > 20.\n\n"
    "🔭 EARLY SIGNALS / WATCH\n\n"
    "Watch crude futures and defense sector volume (Medium confidence)."
)


def test_grounding_audit_does_not_flag_section_header_words_as_tickers(monkeypatch):
    """The reported EventScenario bug: all-caps section headers ('🔭 EARLY SIGNALS
    / WATCH') and scenario-table label rows ('CASE ~PROB MECHANISM ...') match the
    ticker shape, and the nearby negated 'Do not trim Tech' supplied a sell verb —
    so EARLY/WATCH were flagged as unheld tickers recommended for trimming. Neither
    the header labels nor the negated verb may produce a grounding violation."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "VET.TO"}]},
    )

    violations = rm.run_deterministic_grounding_audit(_EVENT_SCENARIO_ADVICE)
    assert violations == [], violations
    assert not any("EARLY" in v or "WATCH" in v or "CASE" in v for v in violations)


def test_allcaps_heading_line_detection():
    """The header/label detector fires on emoji-led and table-header label lines
    but leaves prose and a lone bare ticker on its own line scannable."""
    import agent.nodes.risk_manager as rm

    assert rm._is_allcaps_heading_line("🔭 EARLY SIGNALS / WATCH")
    assert rm._is_allcaps_heading_line("💼 PORTFOLIO EXPOSURE")
    assert rm._is_allcaps_heading_line("CASE\t~PROB\tMECHANISM\tTIME\tCONFIRMS/INVALIDATES")
    # Prose (has lowercase) and a lone all-caps token are NOT headers.
    assert not rm._is_allcaps_heading_line("VET.TO: $5,000.00 CAD (1.50%) - Direct energy exposure.")
    assert not rm._is_allcaps_heading_line("NVDA")
    assert not rm._is_allcaps_heading_line("")


def test_grounding_audit_respects_negation(monkeypatch):
    """Negated / keep guidance ('do not sell', "don't trim", 'hold rather than
    sell', 'avoid trimming') must never be read as a sell recommendation, even
    for a genuinely unheld ticker."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda: {"holdings": [{"symbol": "AAPL"}]},
    )

    for phrase in (
        "Do not sell NVDA here — the setup is intact.",
        "Don't trim NVDA on this dip.",
        "Hold rather than sell NVDA into the print.",
        "Avoid trimming NVDA ahead of earnings.",
    ):
        assert rm.run_deterministic_grounding_audit(phrase) == [], phrase

    # But an unqualified sell verb on the same unheld ticker is still caught.
    assert any(
        "NVDA" in v for v in rm.run_deterministic_grounding_audit("Sell NVDA now.")
    )


def test_total_audit_covers_total_portfolio_leading_keyword(monkeypatch):
    """The app's own canonical headline is 'Total Portfolio: $X' — the valuation
    keyword precedes 'portfolio'. Previously the audit only inspected the trailing
    connector, so this primary format was never checked and a wrong/mislabeled
    total slipped through silently."""
    import agent.nodes.risk_manager as rm

    monkeypatch.setattr("tools.portfolio_csv.get_portfolio_decision_context", lambda: _TOTAL_AUDIT_CTX)

    # Correct leading-keyword headline passes.
    assert rm.run_deterministic_total_audit("Total Portfolio: $450,000 CAD.") == []
    # A wrong total in that same phrasing is now actually flagged.
    wrong = rm.run_deterministic_total_audit("Total Portfolio: $600,000 CAD.")
    assert any("600,000" in v for v in wrong), wrong
    # And a currency mislabel in that phrasing is caught too.
    mislabeled = rm.run_deterministic_total_audit("Total Portfolio: $330,000 CAD.")
    assert any("labels portfolio total" in v for v in mislabeled), mislabeled


def test_parse_risk_verdict_captures_violations_with_blank_lines():
    """The judge routinely puts blank lines between 'Risks:' and its bullets. The
    old '\\n\\n' stop captured an EMPTY block, so a real flagged risk went
    uncounted and a >=8 verdict was wrongly marked compliant. Violations must be
    enumerated regardless of blank-line formatting."""
    import agent.nodes.risk_manager as rm

    verdict = (
        "⚖️ **Verdict: 8/10** — Strong integration, but penalized for an arbitrary stop.\n\n"
        "🔴 **Risks:**\n\n\n"
        "INVALID STOPS: The proposed 5% trailing stop on VET.TO is arbitrary.\n\n"
        "🤔 **Devil's Advocate:** The market has digested Middle East escalations for years."
    )
    score, is_compliant, violations = rm.parse_risk_verdict(verdict)
    assert score == 8
    assert any("INVALID STOPS" in v for v in violations), violations
    # A listed violation must block compliance even at a passing score.
    assert is_compliant is False


def test_parse_risk_verdict_clean_pass_still_compliant():
    """A clean, high-score verdict with no flagged risks stays compliant — the
    blank-line fix must not manufacture phantom violations from 'None flagged'."""
    import agent.nodes.risk_manager as rm

    verdict = (
        "⚖️ **Verdict: 9/10** — Well grounded and appropriately sized.\n\n"
        "🔴 **Risks:** None flagged.\n\n"
        "🤔 **Devil's Advocate:** Consider the macro tail risk."
    )
    score, is_compliant, violations = rm.parse_risk_verdict(verdict)
    assert score == 9
    assert violations == []
    assert is_compliant is True

