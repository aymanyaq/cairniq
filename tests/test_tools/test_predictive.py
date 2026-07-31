from tools.predictive import _normalize_trend_label, match_historical_regime


def test_normalize_trend_label():
    assert _normalize_trend_label("Bull Market") == "bull"
    assert _normalize_trend_label("Risk-ON") == "bull"
    assert _normalize_trend_label("Bearish") == "bear"
    assert _normalize_trend_label("Correction") == "bear"
    assert _normalize_trend_label("Risk-OFF") == "bear"
    assert _normalize_trend_label("Sideways") == "neutral"
    assert _normalize_trend_label(None) == "neutral"

def test_match_historical_regime_stagflation():
    # 9% inflation, 11% rates should match 1970s Stagflation
    res = match_historical_regime(inflation_rate=9.0, fed_rate=11.0, market_trend="bear", pe_ratio=12.0)
    assert res["matched_regime"] == "1970s Stagflation"
    assert "High inflation" in res["description"]
    assert res["similarity_score"] > 50

def test_match_historical_regime_tech_bubble():
    # 2.5% inflation, 5% rates, 30 PE should match 1999 Tech Bubble
    res = match_historical_regime(inflation_rate=2.5, fed_rate=5.0, market_trend="bull", pe_ratio=30.0)
    assert res["matched_regime"] == "1999 Tech Bubble"
    assert "bubble bursts" in res["authored_scenario_1yr"]

def test_match_historical_regime_ai_boom():
    # Current scenario: 3.4% inflation, 5.25% rates, 24 PE
    res = match_historical_regime(inflation_rate=3.4, fed_rate=5.25, market_trend="bull", pe_ratio=24.0)
    assert res["matched_regime"] == "2023 AI Boom"
    assert "Tech leadership" in res["authored_scenario_3mo"]

def test_match_historical_regime_robustness():
    # Missing/Bad data should still return a result without crashing
    res = match_historical_regime(inflation_rate="invalid", fed_rate=5.25, market_trend=None)
    assert "matched_regime" in res
    assert isinstance(res["similarity_score"], (int, float))


def test_authored_scenarios_are_marked_authored():
    """2.7: the hand-typed outcome strings must declare themselves.

    The whole failure this closes is a field NAMED like a measurement carrying a
    number nobody measured — so both the name and the marker are asserted here.
    """
    res = match_historical_regime(inflation_rate=2.5, fed_rate=5.0, market_trend="bull", pe_ratio=30.0)
    assert res["basis"] == "authored constant"
    assert res["measured_alternative"] == "replay_historical_episode"
    assert "typed into this module by hand" in res["basis_note"]
    # The old names read as derived output; they must not come back as aliases,
    # because an alias is exactly how the unmarked field survives a rename.
    assert "forecast_3mo" not in res
    assert "forecast_1yr" not in res


def test_basis_detail_separates_computed_from_authored():
    """The payload mixes both, so a single blanket label would be a lie either way.

    The similarity score IS computed from the caller's live macro inputs; marking it
    authored would understate the tool, while marking the scenarios computed is the
    failure being fixed.
    """
    res = match_historical_regime(inflation_rate=3.4, fed_rate=5.25, market_trend="bull", pe_ratio=24.0)
    detail = res["basis_detail"]
    assert detail["similarity_score"].startswith("computed")
    assert detail["authored_scenario_3mo"].startswith("authored constant")
    assert detail["authored_scenario_1yr"].startswith("authored constant")
