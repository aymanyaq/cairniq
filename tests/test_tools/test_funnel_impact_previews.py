"""Funnel × candidate-impact wiring (Advisor Roadmap 4.9).

Guards the enrichment seam in tools.opportunity_scanner: only High-Conviction /
Exceptional picks get a preview, only on the personalized path, capped and
best-effort. preview_candidate_impact is stubbed (it has its own suite).
"""
from unittest.mock import patch

import tools.opportunity_scanner as opp


def _report(symbol, flags=None, err=None):
    if err:
        return {"candidate": symbol, "risk_deltas": {"error": err}, "ips_checks": {"flags": []},
                "headline": f"{symbol} n/a", "proposed_size": {"dollars": "$10,000"}}
    return {
        "candidate": symbol,
        "headline": f"Adding $10,000 of {symbol} · beta 0.6→0.7 · no IPS breach",
        "proposed_size": {"dollars": "$10,000"},
        "risk_deltas": {
            "beta": {"current": 0.6, "proposed": 0.7, "delta": 0.1},
            "volatility": {"current": "16%", "proposed": "16.4%", "delta": "+0.4 pct pts"},
            "cvar_95_annual": {"delta_dollars": "$+4,000"},
            "candidate_correlation_to_portfolio": 0.3,
        },
        "ips_checks": {"flags": flags or []},
    }


def _result(convictions):
    return {"top_picks": [{"symbol": f"T{i}", "conviction": c} for i, c in enumerate(convictions)]}


_CTX = {"holdings": [{"symbol": "AAPL", "value_base": 100.0, "is_cash_or_pension": False}]}


def test_previews_only_high_conviction_capped():
    result = _result(["Exceptional", "High Conviction", "High Conviction", "Qualified", "Watchlist"])
    with patch("tools.candidate_impact.preview_candidate_impact", side_effect=lambda s, **k: _report(s)):
        opp._attach_impact_previews(result, _CTX)
    previewed = [p for p in result["top_picks"] if "impact_preview" in p]
    # cap is 3; the two extra High-Conviction + non-qualifying tiers excluded
    assert [p["symbol"] for p in previewed] == ["T0", "T1", "T2"]
    assert all(p["conviction"] in ("Exceptional", "High Conviction") for p in previewed)
    # Qualified/Watchlist never touched
    assert all("impact_preview" not in p for p in result["top_picks"] if p["conviction"] in ("Qualified", "Watchlist"))


def test_compact_shape():
    result = _result(["High Conviction"])
    with patch("tools.candidate_impact.preview_candidate_impact", side_effect=lambda s, **k: _report(s, flags=["IPS FAIL: T0 over cap"])):
        opp._attach_impact_previews(result, _CTX)
    ip = result["top_picks"][0]["impact_preview"]
    assert ip["beta_delta"] == 0.1
    assert ip["volatility_delta"] == "+0.4 pct pts"
    assert ip["cvar_delta_dollars"] == "$+4,000"
    assert ip["correlation_to_portfolio"] == 0.3
    assert ip["ips_fit"] == "would breach IPS"
    assert ip["ips_flags"] == ["IPS FAIL: T0 over cap"]


def test_skips_when_no_portfolio():
    """Neutral nightly path: no holdings → no previews, no crash."""
    result = _result(["High Conviction"])
    with patch("tools.candidate_impact.preview_candidate_impact", side_effect=AssertionError("must not be called")):
        opp._attach_impact_previews(result, None)
        opp._attach_impact_previews(result, {"error": "x"})
        opp._attach_impact_previews(result, {"holdings": []})
    assert all("impact_preview" not in p for p in result["top_picks"])


def test_preview_failure_is_isolated():
    """A raising/failed preview on one pick must not break the others."""
    result = _result(["High Conviction", "High Conviction"])

    def _flaky(symbol, **kwargs):
        if symbol == "T0":
            raise RuntimeError("boom")
        return _report(symbol)

    with patch("tools.candidate_impact.preview_candidate_impact", side_effect=_flaky):
        opp._attach_impact_previews(result, _CTX)
    assert "impact_preview" not in result["top_picks"][0]
    assert "impact_preview" in result["top_picks"][1]


def test_risk_error_still_attaches_ips_fit():
    result = _result(["High Conviction"])
    with patch("tools.candidate_impact.preview_candidate_impact",
               side_effect=lambda s, **k: _report(s, err="no history")):
        opp._attach_impact_previews(result, _CTX)
    ip = result["top_picks"][0]["impact_preview"]
    assert ip["risk_note"] == "no history"
    assert ip["ips_fit"] == "within IPS limits"
    assert "beta_delta" not in ip
