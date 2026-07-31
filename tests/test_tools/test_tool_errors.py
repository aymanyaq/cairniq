"""Tests for the degraded-result contract (tools/tool_errors.py).

The contract: when a tool cannot check (missing key, rate limit, outage) it
must return an explicit 'unavailable' payload instead of a silent {}/[]/None,
while genuine "no data exists" keeps the natural empty shape.
"""
import tools.fmp_api as fmp
from tools.tool_errors import UNAVAILABLE, is_unavailable, missing_key_reason, unavailable


def test_unavailable_shape_and_predicate():
    payload = unavailable("FMP", "rate limited", symbol="AAPL")
    assert payload["status"] == UNAVAILABLE
    assert payload["source"] == "FMP"
    assert payload["reason"] == "rate limited"
    assert payload["symbol"] == "AAPL"
    assert is_unavailable(payload)
    assert not is_unavailable({})
    assert not is_unavailable([])
    assert not is_unavailable(None)
    assert not is_unavailable({"status": "ok"})


def test_missing_key_reason_names_the_env_var():
    reason = missing_key_reason("FMP_API_KEY")
    assert "FMP_API_KEY" in reason
    assert "not configured" in reason


def test_fmp_helper_fails_fast_without_key(monkeypatch):
    monkeypatch.setattr(fmp, "_fmp_key", lambda: "")
    data, err = fmp._fmp_get("quote", {"symbol": "AAPL"})
    assert data is None
    assert "FMP_API_KEY" in err


def test_finnhub_helper_fails_fast_without_key(monkeypatch):
    import tools.finnhub_api as finnhub
    monkeypatch.setattr(finnhub, "_finnhub_key", lambda: "")
    data, err = finnhub._finnhub_get("news")
    assert data is None
    assert "FINNHUB_API_KEY" in err


def test_polygon_helper_fails_fast_without_key(monkeypatch):
    import tools.polygon_api as polygon
    monkeypatch.setattr(polygon, "_polygon_key", lambda: "")
    data, err = polygon._polygon_get("v2/aggs/ticker/AAPL/prev")
    assert data is None
    assert "POLYGON_API_KEY" in err


def test_fred_helper_fails_fast_without_key(monkeypatch):
    import tools.fred_api as fred
    monkeypatch.setattr("tools.credential_manager.get_api_key", lambda service, default="": "")
    data, err = fred._fred_get({"series_id": "FEDFUNDS"})
    assert data is None
    assert "FRED_API_KEY" in err


def test_fmp_insider_trades_distinguish_error_from_empty(monkeypatch):
    # Fetch failure → explicit unavailable payload, not a silent []
    monkeypatch.setattr(fmp, "_fmp_get", lambda *a, **k: (None, "HTTP 401"))
    result = fmp.get_fmp_insider_trades.__wrapped__("AAPL")
    assert is_unavailable(result)
    assert result["reason"] == "HTTP 401"

    # Genuine empty → real answer, keeps the list shape
    monkeypatch.setattr(fmp, "_fmp_get", lambda *a, **k: ([], None))
    assert fmp.get_fmp_insider_trades.__wrapped__("AAPL") == []


def test_fmp_senate_disclosures_distinguish_error_from_empty(monkeypatch):
    monkeypatch.setattr(fmp, "_fmp_get", lambda *a, **k: (None, "HTTP 401"))
    result = fmp.get_fmp_senate_disclosures.__wrapped__("AAPL")
    assert is_unavailable(result)

    monkeypatch.setattr(fmp, "_fmp_get", lambda *a, **k: ([], None))
    assert fmp.get_fmp_senate_disclosures.__wrapped__("AAPL") == []


def test_fmp_financials_and_estimates_propagate_errors(monkeypatch):
    monkeypatch.setattr(fmp, "_fmp_get", lambda *a, **k: (None, "Rate limit on all FMP keys"))
    financials = fmp.get_fmp_financials.__wrapped__("AAPL")
    assert is_unavailable(financials)
    assert "Rate limit" in financials["reason"]

    estimates = fmp.get_fmp_analyst_estimates.__wrapped__("AAPL")
    assert is_unavailable(estimates)
    # The unavailable payload is still .get()-safe for aggregators like market_data
    assert estimates.get("target_mean") is None
