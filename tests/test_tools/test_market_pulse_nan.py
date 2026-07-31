"""A NaN from a price feed must never reach — or survive in — a JSON payload.

Reconstructed from a live outage: `/api/market-pulse` returned 500 for a full
day with `ValueError: Out of range float values are not JSON compliant: nan`.
The chain had four links and every one of them passed the NaN along:

  1. `detect_sector_rotation` divided by a NaN close price, so `return_1m` came
     out NaN and was formatted into the string "+nan%".
  2. `_parse_pct` was meant to be the guard, but `float("nan")` does not raise —
     it returns NaN, so the guard handed the NaN straight back.
  3. `set_cached` wrote it to disk, because Python's json emits a bare `NaN`
     token by default even though that is not valid JSON.
  4. Starlette's `JSONResponse` serializes with `allow_nan=False` and raised —
     and since only the date rollover replaces a daily cache file, every
     request for the rest of the day hit the same poisoned bytes.

Each test below pins one link.
"""
import json
import math

import pandas as pd
import pytest
from fastapi.responses import JSONResponse

from tools import daily_cache
from tools.market_sentinel import _get_sector_trends, _parse_pct

# --- Link 2: the guard that wasn't -----------------------------------------

@pytest.mark.parametrize("raw", ["+nan%", "nan", "-nan%", "inf", "+inf%", "-Infinity"])
def test_parse_pct_rejects_non_finite_strings(raw):
    """float() accepts every one of these — that is exactly the trap."""
    assert _parse_pct(raw) is None


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
def test_parse_pct_rejects_non_finite_numbers(raw):
    assert _parse_pct(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("+2.3%", 2.3), ("-11.94%", -11.9), (4.0, 4.0), ("0%", 0.0)]
)
def test_parse_pct_still_parses_real_numbers(raw, expected):
    assert _parse_pct(raw) == pytest.approx(expected)


def test_parse_pct_returns_none_not_zero_for_junk():
    """None, not 0.0: a substituted zero is a return that never happened."""
    assert _parse_pct("n/a") is None
    assert _parse_pct(None) is None


# --- Link 1 + 2: the sector row is dropped, not zeroed ----------------------

def _rotation(rows):
    return {"sector_performance": rows}


def _row(symbol, r1m, r3m, momentum):
    return {
        "symbol": symbol, "sector": f"{symbol} sector", "character": "Growth",
        "return_1m": r1m, "return_3m": r3m, "momentum_score": momentum,
        "signal": "🟢 INFLOW",
    }


def test_sector_with_nan_numbers_is_omitted_from_the_heatmap(monkeypatch):
    rows = [
        _row("XLK", "+6.0%", "+11.9%", 2.0),
        _row("XLE", "+nan%", "+nan%", float("nan")),
        _row("XLC", "-3.9%", "+13.8%", -8.5),
    ]
    monkeypatch.setattr(
        "tools.sector_rotation.detect_sector_rotation", lambda: _rotation(rows)
    )
    trends = _get_sector_trends()

    assert [t["symbol"] for t in trends] == ["XLK", "XLC"]
    # And the surviving payload is JSON-serializable, which is the whole point.
    json.dumps(trends, allow_nan=False)


def test_sector_trends_survive_a_fully_nan_rotation(monkeypatch):
    monkeypatch.setattr(
        "tools.sector_rotation.detect_sector_rotation",
        lambda: _rotation([_row("XLE", "+nan%", "+nan%", float("nan"))]),
    )
    assert _get_sector_trends() == []


# --- Link 1 at the source: the producer drops the sector --------------------

def test_detect_sector_rotation_skips_a_sector_with_nan_closes(monkeypatch):
    """A NaN hole in the Close series must not become a '+nan%' string."""
    import tools.sector_rotation as sr

    good = pd.DataFrame({"Close": [100.0 + i for i in range(80)]})
    holed = pd.DataFrame({"Close": [float("nan")] * 80})

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="4mo"):
            return holed if self.symbol == "XLE" else good

    monkeypatch.setattr(sr.yf, "Ticker", _Ticker)
    out = sr.detect_sector_rotation.__wrapped__.__wrapped__()

    symbols = [r["symbol"] for r in out["sector_performance"]]
    assert "XLE" not in symbols
    assert "XLK" in symbols
    for row in out["sector_performance"]:
        assert "nan" not in row["return_1m"].lower()
        assert math.isfinite(row["momentum_score"])


# --- Link 3: the cache refuses to persist invalid JSON ----------------------

@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Route the daily cache at tmp_path (profile name must not start with
    `pytest_`, which _cache_path deliberately redirects to the profile dir)."""
    monkeypatch.setattr(daily_cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(daily_cache, "get_active_profile", lambda: "nanprobe")
    return tmp_path


def test_set_cached_never_writes_a_nan_token(cache_dir):
    daily_cache.set_cached(
        "pulse_nan", {"regime": "BULLISH", "sector_trends": [{"momentum": float("nan")}]}
    )

    written = list(cache_dir.iterdir())
    assert len(written) == 1
    raw = written[0].read_text()
    assert "NaN" not in raw
    json.loads(raw)  # strict parse: would raise on a bare NaN token


def test_set_cached_nulls_the_bad_number_and_keeps_the_rest(cache_dir):
    daily_cache.set_cached(
        "pulse_mixed",
        {"regime": "BULLISH", "vix": 19.4, "bad": float("inf"), "nested": [1.0, float("nan")]},
    )
    out = daily_cache.get_cached("pulse_mixed")

    assert out["regime"] == "BULLISH"
    assert out["vix"] == 19.4
    assert out["bad"] is None
    assert out["nested"] == [1.0, None]


def test_clean_payload_round_trips_unchanged(cache_dir):
    payload = {"regime": "BULLISH", "sector_trends": [{"symbol": "XLK", "momentum": 2.0}]}
    daily_cache.set_cached("pulse_clean", payload)
    assert daily_cache.get_cached("pulse_clean") == payload


# --- Link 4: a file poisoned before the fix still serves ---------------------

def test_get_cached_neutralizes_a_preexisting_nan_file(cache_dir):
    """The outage persisted because only a date rollover replaces the file."""
    path = daily_cache._cache_path("pulse_legacy")
    with open(path, "w") as f:
        json.dump({"regime": "BULLISH", "momentum": float("nan")}, f)  # allow_nan default
    assert "NaN" in open(path).read()  # the poisoned bytes really are on disk

    out = daily_cache.get_cached("pulse_legacy")

    assert out["momentum"] is None
    # The exact call that used to 500 the endpoint.
    JSONResponse(out)
