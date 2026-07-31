"""Characterization tests for the _scan_impl phase pipeline.

The legacy suite only exercised a specific-sector scan ("Technology"). The
phase-split refactor (driver + _ScanContext + _scan_phaseN helpers) preserves
behavior across the broad (v2 scorer) and guru (media-feed) paths too, which
these tests pin down by mocking the I/O boundary deterministically.
"""
import importlib.util

import pandas as pd
import pytest

import tools.opportunity_scanner as opp

# tools.guru_feed is excluded from public/CI builds — the scanner wraps its guru
# imports in try/except ImportError. Patch it only when present, and skip the
# guru-specific test otherwise so the suite stays green on minimal builds.
_HAS_GURU = importlib.util.find_spec("tools.guru_feed") is not None

_TECH = {
    "price": 100.0, "above_sma50": True, "golden_cross": True, "rsi": 34,
    "drawdown_pct": -22, "vol_spike": 2.1, "three_month_return": 18,
    "month_return": 7, "sma50": 95.0, "sma200": 88.0, "volume": 1_000_000,
    "avg_volume": 600_000, "atr_pct": 3.1, "macd_bullish": True,
}
_FUND = {
    "pe_ratio": 22, "peg_ratio": 1.1, "revenue_growth": 0.18, "profit_margin": 0.21,
    "roe": 0.28, "debt_to_equity": 0.4, "current_price": 100.0, "market_cap": 5e11,
    "52_week_high": 130, "free_cash_flow": 1e10, "earnings_growth": 0.2,
    "recommendation": "buy", "description": "Quality business",
    "news_headlines": ["Strong product launch"], "beta": 1.1, "dividend_yield": 0.01,
}


@pytest.fixture
def scanner_io(monkeypatch):
    """Mock every network/data collaborator _scan_impl touches, deterministically."""
    monkeypatch.setattr(opp, "safe_print", lambda *a, **k: None)
    monkeypatch.setattr("agent.logger.log_to_component", lambda *a, **k: None)

    monkeypatch.setattr(opp, "_compute_technicals_cached", lambda candidates: {s: dict(_TECH) for s in candidates})
    monkeypatch.setattr(opp, "_batch_download", lambda symbols, **kw: pd.DataFrame({"ok": [1]}))
    monkeypatch.setattr(opp, "_compute_technicals_batch", lambda data, symbols: {s: dict(_TECH) for s in symbols})
    monkeypatch.setattr(opp, "_fetch_fundamentals_parallel", lambda symbols, **kw: {s: dict(_FUND, symbol=s) for s in symbols})
    monkeypatch.setattr(opp, "_headwind_check_parallel", lambda symbols, **kw: {s: {"short_pct_float": 0.11} for s in symbols})
    monkeypatch.setattr(opp, "_setup_check_parallel", lambda symbols, **kw: {})
    monkeypatch.setattr(opp, "_flow_confirmation_parallel", lambda symbols, headwind_map=None, **kw: {})
    monkeypatch.setattr(opp, "_warm_cache_parallel", lambda *a, **k: None)
    monkeypatch.setattr(opp, "_log_funnel_signals", lambda *a, **k: None)
    monkeypatch.setattr(opp, "_assemble_dynamic_universe", lambda rotation_data, guru_tickers=None: (
        ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD"],
        {"NVDA": ["theme:ai"], "AMD": ["mover:active"]},
    ))
    # The named-sector path does NOT go through _assemble_dynamic_universe: it calls
    # _get_sector_tickers, which fetches TradingView (then Yahoo) live. Leaving it
    # unmocked made the sector test's verdict depend on whether a same-day
    # tv_sector_constituents cache happened to be in user_data/daily_cache/ — the
    # offline guard blocks the fetch, so with no cache the universe is empty and the
    # assertion fails for no change in the code it names. Measured red on 2026-07-30
    # after weeks of green. The empty case is now its own test below.
    monkeypatch.setattr(opp, "_get_sector_tickers",
                        lambda sector_key: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMD"])
    monkeypatch.setattr(opp, "_rank_themes", lambda *a, **k: [
        {"theme": "AI Infrastructure", "theme_score": 0.91, "cycle_stage": "Acceleration",
         "drivers": ["capex"], "canonical_sector": "Technology"},
    ])

    if _HAS_GURU:
        monkeypatch.setattr("tools.guru_feed.get_guru_feed_summary", lambda: {
            "picks": [{"ticker": "NVDA", "signal": "BUY", "freshness": "SWEET_SPOT",
                       "date": "2026-06-01", "headline": "NVDA momentum", "url": "http://x",
                       "mention_count": 3, "segment_type": "feature"}],
            "ticker_metadata": {"NVDA": {"signal": "BUY", "freshness": "SWEET_SPOT",
                                         "mention_count": 3, "headline": "NVDA momentum",
                                         "date": "2026-06-01"}},
            "sweet_spot_count": 1,
        }, raising=False)
        monkeypatch.setattr("tools.guru_feed.get_guru_universe",
                            lambda freshness_filter="active": ["NVDA", "AMD", "AAPL"], raising=False)

    monkeypatch.setattr("tools.market_mechanics.detect_sector_rotation", lambda: {
        "leading_sectors": ["Technology", "Energy"],
        "lagging_sectors": ["Utilities"],
        "full_rotation_map": [
            {"sector": "Technology", "trend": "Leading"},
            {"sector": "Energy", "trend": "Improving"},
            {"sector": "Utilities", "trend": "Lagging"},
        ],
        "market_status": "Risk-On",
    })
    monkeypatch.setattr("tools.market_mechanics.predict_earnings_surprise",
                        lambda symbol: {"surprise_probability": 0.5}, raising=False)
    monkeypatch.setattr("tools.fred_api.get_systemic_risk_indicators", lambda: {
        "liquidity_status": "Expanding", "m2_growth_yoy": "5%", "crash_risk": "Low"})
    monkeypatch.setattr("tools.fred_api.get_treasury_yields", lambda: {"curve_status": "Normal"})
    monkeypatch.setattr("tools.macro_strategy.analyze_macro_context", lambda: {
        "strategy": {"tactical_opportunity": ["Technology"], "sectors_to_underweight": ["Utilities"]}})


def test_broad_scan_uses_v2_path_and_themes(scanner_io):
    scan = opp._scan_impl("All")
    assert scan["sector"] == "Broad Market (High Conviction)"
    assert scan["top_picks"], "broad scan should surface high-conviction picks"
    assert scan["market_status"] == "Risk-On"
    # Phase 1.5 themes flow through to the result.
    assert scan["ranked_themes"] and scan["ranked_themes"][0]["theme"] == "AI Infrastructure"
    assert "across" in scan["summary"]  # broad-only summary fragment


@pytest.mark.skipif(not _HAS_GURU, reason="tools.guru_feed is excluded from this build")
def test_guru_scan_emits_feed_payload(scanner_io):
    scan = opp._scan_impl("GURU")
    assert scan["sector"].startswith("📺 Guru Picks")
    assert scan["top_picks"]
    assert "guru_feed" in scan, "guru path must attach the media-feed payload"
    assert scan["guru_feed"]["picks"], "feed rows should be populated for active picks"


def test_specific_sector_scan_returns_ranked_picks(scanner_io):
    scan = opp._scan_impl("Technology")
    assert scan["sector"] == "Technology"
    assert scan["top_picks"]
    assert "Scanned" in scan["summary"]
    # Ranked descending by score.
    scores = [p.get("score", 0) for p in scan["top_picks"]]
    assert scores == sorted(scores, reverse=True)


def test_sector_scan_with_no_universe_says_so_instead_of_ranking_nothing(scanner_io, monkeypatch):
    """A sector whose constituent feeds return nothing must report that, not a blank ranking.

    This is the state the suite was accidentally testing until 2026-07-30: with the
    universe feeds unreachable, `_scan_impl` produced `top_picks: []` and a summary
    that says no candidates were found. That behaviour is correct and worth pinning —
    what was wrong was asserting the OPPOSITE of it from a test that had no universe.
    Both the live fetch and the web-search fallback are stubbed empty here, so the
    case is reached by construction rather than by whatever the network is doing.
    """
    monkeypatch.setattr(opp, "_get_sector_tickers", lambda sector_key: [])
    monkeypatch.setattr("tools.web_search.search_news",
                        lambda *a, **k: "", raising=False)

    scan = opp._scan_impl("Technology")

    assert scan["sector"] == "Technology"
    assert scan["top_picks"] == []
    assert "No candidates" in scan["summary"], scan["summary"]


def test_sector_finalist_keeps_overweight_penalty_after_phase5_rescore(scanner_io, monkeypatch):
    """Regression: a sector-scan finalist that has BOTH a headwind and concentrated
    portfolio exposure must keep its concentration/over-weight penalty after the
    Phase-5 rescore.

    The legacy v1 Phase-5 `elif hw:` branch (`_deep_score_value_v1(..., headwind_data=hw)`)
    historically omitted `portfolio_context`, so the over-weight penalty applied in
    Phase 4 was silently dropped on rescore and the final score was inflated. The
    `scanner_io` fixture gives every finalist a headwind (`short_pct_float`), which
    forces that rescore branch for the sector path.
    """
    # Deterministic Technology universe — all resolve to "Technology" via the
    # _API_SECTOR_FALLBACKS table (Tier-1, no network).
    monkeypatch.setattr(opp, "_get_sector_tickers",
                        lambda sector: ["AAPL", "MSFT", "NVDA", "AMD", "AVGO"])

    # 40% Technology exposure → over-weight (>25%) → penalty = int((0.40-0.25)*100)*2 = 30.
    # Large enough to be visible, small enough that the finalist still clears the
    # qualification threshold and reaches the Phase-5 rescore.
    concentrated = {"total_value_usd": 1_000_000, "holdings": [
        {"symbol": "MSFT", "value_usd": 400_000},  # Technology
        {"symbol": "JPM", "value_usd": 600_000},   # Financial Services
    ]}

    baseline = opp._scan_impl("Technology")
    penalized = opp._scan_impl("Technology", portfolio_context=concentrated)

    assert penalized["top_picks"], "concentrated sector scan should still surface picks"

    # Every finalist is a Technology name and over-weight, so each must carry the penalty.
    for pick in penalized["top_picks"]:
        overweight_reasons = [r for r in pick.get("reasons", []) if "Overweight Penalty" in r]
        assert overweight_reasons, (
            f"{pick['symbol']} lost its over-weight penalty after the Phase-5 rescore: "
            f"{pick.get('reasons')}"
        )

    # The penalty must actually depress the FINAL (post-rescore) score, not just appear
    # as a reason string: the concentrated run must score strictly below the no-portfolio run.
    baseline_scores = {p["symbol"]: p.get("score", 0) for p in baseline["top_picks"]}
    for pick in penalized["top_picks"]:
        sym = pick["symbol"]
        assert sym in baseline_scores, f"{sym} unexpectedly absent from baseline picks"
        assert pick.get("score", 0) < baseline_scores[sym], (
            f"{sym} final score {pick.get('score')} not reduced vs baseline "
            f"{baseline_scores[sym]} — Phase-5 rescore dropped the concentration penalty"
        )


def test_empty_universe_short_circuits(scanner_io, monkeypatch):
    """A phase that finds no candidates must short-circuit to an empty result."""
    monkeypatch.setattr(opp, "_get_sector_tickers", lambda sector: [])
    # P1 does a local `from tools.web_search import search_news`, so patch the source.
    monkeypatch.setattr("tools.web_search.search_news", lambda *a, **k: "", raising=False)
    scan = opp._scan_impl("Nonexistent Sector XYZ")
    assert scan["top_picks"] == []
