"""market_scanner must not hold its own API key.

It used to: `FMP_KEY = os.environ.get("FMP_API_KEY", "")` evaluated at import
time, with its own `requests.get`. That is three defects wearing one line —

  - the key never rotated, so one 429 on the primary key took this scanner out
    for the remaining life of the process;
  - the 429 was never reported to credential_manager, so the cooldown every
    other FMP caller relies on was never started by this one;
  - whether a key existed at all depended on import order versus environment
    load.

None of it was visible from the call sites, which just saw an empty list and
quietly fell back to yfinance — i.e. the scanner degraded silently and
permanently. These tests pin the routing, because that is what makes the
rotation and the cooldown reachable.
"""
import pytest

import tools.fmp_api as fmp
import tools.market_scanner as scanner


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else [{"symbol": "X"}]

    def json(self):
        return self._payload


@pytest.fixture
def wire(monkeypatch):
    """Capture the apikey on every outbound FMP request."""
    sent = []

    def fake_get(url, params=None, timeout=None):
        sent.append((url, (params or {}).get("apikey")))
        return _Resp()

    monkeypatch.setattr(fmp.requests, "get", fake_get)
    return sent


def test_no_module_level_key_survives():
    """The literal shape of the bug: a key captured once, at import."""
    assert not hasattr(scanner, "FMP_KEY")


def test_key_is_resolved_per_call_not_captured_once(monkeypatch, wire):
    keys = iter(["first-key", "rotated-key"])
    monkeypatch.setattr(fmp, "get_api_key", lambda service, default="": next(keys))

    scanner._fmp_get("biggest-gainers")
    scanner._fmp_get("biggest-losers")

    # A second call gets whatever the credential manager says NOW. Under the old
    # code both would have carried the same import-time value.
    assert [k for _, k in wire] == ["first-key", "rotated-key"]


def test_a_429_rotates_the_key_and_reports_the_rate_limit(monkeypatch):
    """Inherited from the shared helper — the whole point of routing through it."""
    reported = []
    monkeypatch.setattr(fmp, "report_rate_limit",
                        lambda service, key=None: reported.append((service, key)))

    keys = iter(["primary", "secondary", "secondary"])
    monkeypatch.setattr(fmp, "get_api_key", lambda service, default="": next(keys))

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((params or {}).get("apikey"))
        # Primary is limited; the secondary succeeds.
        return _Resp(429) if len(calls) == 1 else _Resp(200)

    monkeypatch.setattr(fmp.requests, "get", fake_get)

    assert scanner._fmp_get("most-actives") == [{"symbol": "X"}]
    assert calls == ["primary", "secondary"]
    assert reported == [("FMP_API_KEY", "primary")]


def test_a_missing_key_degrades_to_an_empty_list(monkeypatch):
    """Callers read [] as 'FMP unavailable' and fall back to yfinance."""
    monkeypatch.setattr(fmp, "get_api_key", lambda service, default="": "")
    assert scanner._fmp_get("biggest-gainers") == []


def test_a_non_list_payload_is_not_passed_through(monkeypatch):
    """FMP returns an error OBJECT on some failures; callers here expect a list."""
    monkeypatch.setattr(fmp, "get_api_key", lambda service, default="": "k")
    monkeypatch.setattr(
        fmp.requests, "get",
        lambda url, params=None, timeout=None: _Resp(200, {"Error Message": "nope"}),
    )
    assert scanner._fmp_get("biggest-gainers") == []


def test_endpoints_still_resolve_against_the_fmp_base(monkeypatch, wire):
    monkeypatch.setattr(fmp, "get_api_key", lambda service, default="": "k")
    scanner._fmp_get("sector-performance-snapshot")
    url = wire[0][0]
    assert url.endswith("/sector-performance-snapshot")
    assert url.startswith(fmp.BASE_URL)
