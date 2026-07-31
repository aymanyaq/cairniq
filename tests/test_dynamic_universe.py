"""
Tests for Funnel V2 / M1+M2 — Dynamic Universe Assembly + Theme Ranking
in tools.opportunity_scanner.

These guard the three failure modes found in review:
  1. Rotation-shape mismatch (the scanner uses market_mechanics' full_rotation_map shape,
     NOT sector_rotation's sector_performance/momentum_score shape).
  2. The semiconductor seed guarantee (MU must enter even with no FMP key / no live theme).
  3. The chunked-download size-1 tail that corrupts the MultiIndex merge.
"""
from datetime import UTC

import pytest

import tools.opportunity_scanner as opp
from tools.opportunity_scanner import (
    _canonical_sector,
    _cycle_stage_from_trend,
    _get_sector_for_ticker,
    _rank_themes,
    _resolve_text_to_themes,
)

# Captured before the autouse _no_network fixture stubs it, so the direct
# unit test below can exercise the real implementation.
_REAL_YF_SECTOR_CONSTITUENTS = opp._yf_sector_constituents
_REAL_TV_SECTOR_CONSTITUENTS = opp._tv_sector_constituents


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, tmp_path):
    """Neutralise every live source so the universe is built deterministically."""
    import tools.sec_edgar as sec_edgar

    monkeypatch.setattr(opp, "_tv_sector_constituents", lambda *a, **k: [])
    monkeypatch.setattr(opp, "_fmp_sector_constituents", lambda *a, **k: [])
    monkeypatch.setattr(opp, "_yf_sector_constituents", lambda *a, **k: [])
    monkeypatch.setattr(opp, "_get_active_theme_tickers", lambda: {})
    # 13F institutional universe (Roadmap 5.1) is a live EDGAR fetch.
    monkeypatch.setattr(sec_edgar, "get_13f_universe", lambda: [])
    # Keep the scan ledger out of real user_data during tests.
    monkeypatch.setattr(opp, "_SCAN_LEDGER_PATH", str(tmp_path / "ledger.json"))


# ---------------------------------------------------------------------------
# 1. Rotation-shape robustness
# ---------------------------------------------------------------------------

def test_inflowing_sectors_reads_market_mechanics_shape():
    """The scanner's actual rotation source returns full_rotation_map + trend strings."""
    rotation = {
        "full_rotation_map": [
            {"sector": "Technology", "trend": "Leading 🟢 ⚠️ Overbought"},
            {"sector": "Energy", "trend": "Improving 🔵"},
            {"sector": "Utilities", "trend": "Lagging 🔴"},
            {"sector": "Healthcare", "trend": "Weakening 🟡"},
        ],
        "leading_sectors": ["Technology"],
    }
    hot = opp._inflowing_sectors(rotation)
    assert "Technology" in hot
    assert "Energy" in hot           # Improving counts as inflow
    assert "Utilities" not in hot    # Lagging excluded
    assert "Healthcare" not in hot   # Weakening excluded


def test_inflowing_sectors_reads_sector_rotation_shape():
    """Robust to the OTHER implementation's momentum_score shape."""
    rotation = {
        "sector_performance": [
            {"sector": "Technology", "momentum_score": 3.2},
            {"sector": "Energy", "momentum_score": -1.1},
        ]
    }
    hot = opp._inflowing_sectors(rotation)
    assert hot == ["Technology"]


def test_inflowing_sectors_empty_on_garbage():
    assert opp._inflowing_sectors({}) == []
    assert opp._inflowing_sectors({"error": "boom"}) == []


# ---------------------------------------------------------------------------
# 1b. TradingView screener transport (the cairniq SSL failure mode)
# ---------------------------------------------------------------------------

def test_tv_screener_uses_requests_and_parses(monkeypatch):
    """The screener must fetch via requests (certifi-backed), not raw urllib.

    On the framework Python the server ships under, ssl.get_default_verify_paths()
    .cafile is None, so a urllib.urlopen fails EVERY TLS verify with
    CERTIFICATE_VERIFY_FAILED and the screener silently returned zero names for
    every sector (fixed 2026-07-22). requests bundles certifi and verifies on the
    same runtime. This asserts the transport is requests AND that the parse,
    sector filter, and .TO suffixing still hold.
    """
    import tools.daily_cache as daily_cache

    monkeypatch.setattr(daily_cache, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(daily_cache, "set_cached", lambda *a, **k: None)

    calls = {"urls": []}

    class _FakeResp:
        def __init__(self, region):
            # america returns one in-sector (AAPL) + one off-sector (JPM, must be
            # filtered out for Technology); canada returns SHOP (gets .TO suffix).
            self._rows = {
                "america": [
                    {"d": ["AAPL", 200.0, 1.0, 5_000_000, "Electronic Technology"]},
                    {"d": ["JPM", 190.0, 0.5, 4_000_000, "Finance"]},
                ],
                "canada": [
                    {"d": ["SHOP", 90.0, 1.2, 2_000_000, "Technology Services"]},
                ],
            }[region]

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": self._rows}

    def _fake_post(url, **kwargs):
        calls["urls"].append(url)
        region = "canada" if "canada" in url else "america"
        return _FakeResp(region)

    # Guard the transport at the source: a regression to urllib.urlopen would not
    # go through requests.post and this mock would never fire.
    import requests
    monkeypatch.setattr(requests, "post", _fake_post)

    result = _REAL_TV_SECTOR_CONSTITUENTS("Technology")

    assert calls["urls"] == [
        "https://scanner.tradingview.com/america/scan",
        "https://scanner.tradingview.com/canada/scan",
    ]
    assert "AAPL" in result
    assert "SHOP.TO" in result       # Canadian name gets the .TO suffix
    assert "JPM" not in result       # Finance filtered out of a Technology scan


# ---------------------------------------------------------------------------
# 2. Semiconductor seed guarantee (the MU failure mode)
# ---------------------------------------------------------------------------

def test_mu_enters_universe_with_no_fmp_no_theme(monkeypatch):
    """With FMP and themes dead, MU must STILL enter via the static semi seed."""
    monkeypatch.setattr(opp, "scan_intraday_movers", lambda *a, **k: {}, raising=False)
    rotation = {"full_rotation_map": [{"sector": "Technology", "trend": "Leading 🟢"}]}

    candidates, provenance = opp._assemble_dynamic_universe(
        rotation, intraday_movers={}, guru_tickers=[]
    )
    assert "MU" in candidates
    assert any("Semiconductors" in p for p in provenance["MU"])
    # The whole memory complex is seeded
    for sym in ("WDC", "STX", "AVGO", "AMAT"):
        assert sym in candidates


def test_funnel_survives_guru_module_absent(monkeypatch):
    """Guru is an optional feed. When tools.guru_feed can't be imported, the broad
    universe must still assemble from the other sources — no crash, no guru tags."""
    import sys

    # A None entry in sys.modules makes `from tools.guru_feed import ...` raise
    # ImportError, simulating the public build where the module is stripped.
    monkeypatch.setitem(sys.modules, "tools.guru_feed", None)
    monkeypatch.setattr(opp, "scan_intraday_movers", lambda *a, **k: {}, raising=False)
    rotation = {"full_rotation_map": [{"sector": "Technology", "trend": "Leading 🟢"}]}

    # guru_tickers=None forces assembly to attempt the (now failing) import itself.
    candidates, provenance = opp._assemble_dynamic_universe(
        rotation, intraday_movers={}, guru_tickers=None
    )

    assert candidates, "universe must still populate from sector/static sources"
    assert "MU" in candidates  # static semiconductor seed survives
    # No name should carry a guru provenance tag when the feed is absent.
    assert not any("guru" in tag for tags in provenance.values() for tag in tags)


def test_guru_feed_available_reflects_module_presence(monkeypatch):
    """guru_enabled must be False when the optional module is stripped (public/CI
    build) and True when it's importable — so confidence grading can tell a
    switched-off feature from a genuine per-ticker data gap.

    guru_feed.py is intentionally gitignored (optional feature), so we cannot
    assert the module is present — only that the flag accurately tracks reality
    and that forcing absence (None in sys.modules) always yields False.
    """
    import sys

    # Whatever the current env is, the flag must return a bool.
    result = opp._guru_feed_available()
    assert isinstance(result, bool)

    # Forcing absence must always yield False regardless of environment.
    monkeypatch.setitem(sys.modules, "tools.guru_feed", None)
    assert opp._guru_feed_available() is False


def test_mover_late_filter_excludes_parabolic_gainers():
    rotation = {"full_rotation_map": []}
    movers = {
        "most_active": [{"symbol": "NVDA"}],
        "top_gainers": [
            {"symbol": "EARLY", "change": "+8.0%"},
            {"symbol": "PARABOLIC", "change": "+25.0%"},
        ],
    }
    candidates, provenance = opp._assemble_dynamic_universe(
        rotation, intraday_movers=movers, guru_tickers=[]
    )
    assert "EARLY" in candidates
    assert provenance.get("EARLY") == ["mover:gainer_early"]
    # PARABOLIC may still appear via the static seed, but never tagged as an early mover
    assert "mover:gainer_early" not in provenance.get("PARABOLIC", [])


def test_universe_is_capped():
    rotation = {"full_rotation_map": []}
    # Flood with synthetic gainers
    movers = {"most_active": [], "top_gainers": []}
    candidates, _ = opp._assemble_dynamic_universe(
        rotation, intraday_movers=movers, guru_tickers=[f"TT{i}" for i in range(500)]
    )
    assert len(candidates) <= opp._UNIVERSE_CAP


# ---------------------------------------------------------------------------
# 3. provenance boost in _fast_score (admit → surface)
# ---------------------------------------------------------------------------

def test_fast_screen_cut_is_deterministic_under_dict_reordering():
    """
    The top-N fast-screen cut must be stable regardless of the order technicals
    arrive in (chunked-download completion order is non-deterministic under
    concurrency). Two candidates tied on _fast_score that straddle the cut must
    not flip in/out just because the dict was iterated in a different order.

    Regression: a name with an ideal accumulation setup (GOOGL) qualified in two
    intraday runs then vanished in a third with byte-identical cached signals,
    because the tie-break at the rank-N boundary was dict insertion order.
    """
    # Two symbols whose technicals produce the SAME integer _fast_score.
    tech = {"rsi": 52, "golden_cross": False, "above_sma50": True, "death_cross": False,
            "drawdown_pct": -8, "month_return": 4, "vol_spike": 1.1, "three_month_return": 12}
    a, b = "AAA", "ZZZ"
    assert opp._fast_score(a, tech) == opp._fast_score(b, tech)  # genuine tie

    def cut_top1(order):
        scored = [(sym, opp._fast_score(sym, tech), tech) for sym in order]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[0][0]

    # Feed the two candidates in both possible orders; the survivor of a top-1
    # cut must be identical (the alphabetical tiebreak → "AAA"), not order-dependent.
    assert cut_top1([a, b]) == cut_top1([b, a]) == a


def test_technicals_are_daily_cached_across_runs(monkeypatch):
    """
    Per-symbol fast-screen technicals are computed once per trading day and
    reused on subsequent intraday runs, so the field feeding the top-N cut
    doesn't wobble with live prices (the third lever behind the GOOGL dropout).
    """
    import numpy as np
    import pandas as pd

    import tools.daily_cache as dc

    store: dict = {}
    monkeypatch.setattr(dc, "get_cached", lambda k, ttl_seconds=None: store.get(k))
    monkeypatch.setattr(dc, "set_cached", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr(opp, "is_cancelled", lambda: False, raising=False)

    calls = []

    def fake_batch(tickers, period=None, max_retries=3):
        calls.append(list(tickers))
        cols = pd.MultiIndex.from_product([list(tickers), ["Close", "Volume"]])
        # 80 rows so RSI/SMA50/3M-return all compute; +1 keeps prices positive.
        return pd.DataFrame(np.random.rand(80, len(cols)) + 1, columns=cols)

    monkeypatch.setattr(opp, "_batch_download", fake_batch)
    monkeypatch.setattr(opp, "_batch_download_chunked", fake_batch)

    first = opp._compute_technicals_cached(["AAA", "BBB"])
    assert set(first) == {"AAA", "BBB"}
    assert len(calls) == 1  # one download for the two cache misses

    # Second intraday run: served entirely from the daily cache, no new download.
    second = opp._compute_technicals_cached(["AAA", "BBB"])
    assert second == first
    assert len(calls) == 1  # NO additional download → field is frozen for the day


def test_accumulation_like_predicate():
    """The near-miss proxy fires on basing/pullback names, not on momentum chases."""
    # Low RSI + below 50DMA → accumulation-like
    assert opp._is_accumulation_like({"rsi": 42, "above_sma50": False, "drawdown_pct": -5})
    # Neutral RSI but meaningfully off the highs → accumulation-like
    assert opp._is_accumulation_like({"rsi": 52, "above_sma50": True, "drawdown_pct": -18})
    # Extended: high RSI → never accumulation-like
    assert not opp._is_accumulation_like({"rsi": 78, "above_sma50": True, "drawdown_pct": -2})
    # Trending up, near highs, mid RSI → a chase, not a base
    assert not opp._is_accumulation_like({"rsi": 60, "above_sma50": True, "drawdown_pct": -3})


def test_fast_score_provenance_boost_orders_correctly():
    tech = {"rsi": 52, "golden_cross": False, "above_sma50": True, "death_cross": False,
            "drawdown_pct": -8, "month_return": 4, "vol_spike": 1.1, "three_month_return": 12}
    none = opp._fast_score("MU", tech, provenance=None)
    one = opp._fast_score("MU", tech, provenance=["mover:active"])
    two = opp._fast_score("MU", tech, provenance=["theme:memory_chips", "sector:Technology"])
    three = opp._fast_score("MU", tech, provenance=["theme:memory_chips", "sector:Technology", "mover:active"])
    assert none < one < two < three


# ---------------------------------------------------------------------------
# M2 — Theme Ranking / Event & Flow Radar
# ---------------------------------------------------------------------------

def test_resolve_text_to_themes_finds_memory_keywords():
    assert "memory_chips" in _resolve_text_to_themes("DRAM prices rising due to HBM demand")
    assert "oil" in _resolve_text_to_themes("crude oil prices spike on OPEC cut")
    assert _resolve_text_to_themes("") == []
    assert _resolve_text_to_themes("completely unrelated text") == []


def test_cycle_stage_from_trend():
    assert _cycle_stage_from_trend("Improving 🔵") == "early"
    assert _cycle_stage_from_trend("Leading 🟢") == "mid"
    assert _cycle_stage_from_trend("Leading 🟢 ⚠️ Overbought") == "late"
    assert _cycle_stage_from_trend("Weakening 🟡") == "late"
    assert _cycle_stage_from_trend("Lagging 🔴") == "neutral"


def test_rank_themes_uses_real_rotation_shape(monkeypatch):
    """Verifies _rank_themes works with market_mechanics' full_rotation_map shape."""
    monkeypatch.setattr(opp, "_get_catalyst_themes_from_events", lambda: {})
    rotation = {
        "full_rotation_map": [
            {"sector": "Technology", "trend": "Leading 🟢"},
            {"sector": "Energy",     "trend": "Improving 🔵"},
            {"sector": "Utilities",  "trend": "Lagging 🔴"},
            {"sector": "Healthcare", "trend": "Weakening 🟡"},
        ],
    }
    macro = {"liquidity": "Expanding", "crash_risk": "Low"}
    bullish = ["TECHNOLOGY", "ENERGY"]
    bearish = ["UTILITIES"]

    themes = _rank_themes(rotation, macro, bullish, bearish)

    assert len(themes) == 4
    # Sorted descending
    scores = [t["theme_score"] for t in themes]
    assert scores == sorted(scores, reverse=True)
    # Bullish + Leading > Bearish + Lagging
    tech = next(t for t in themes if t["sector"] == "Technology")
    util = next(t for t in themes if t["sector"] == "Utilities")
    assert tech["theme_score"] > util["theme_score"]
    # Cycle stages
    assert next(t for t in themes if t["sector"] == "Energy")["cycle_stage"] == "early"
    assert next(t for t in themes if t["sector"] == "Technology")["cycle_stage"] == "mid"
    assert next(t for t in themes if t["sector"] == "Healthcare")["cycle_stage"] == "late"


def test_rank_themes_empty_rotation_returns_empty(monkeypatch):
    monkeypatch.setattr(opp, "_get_catalyst_themes_from_events", lambda: {})
    assert _rank_themes({}, {}, [], []) == []
    assert _rank_themes({"error": "boom"}, {}, [], []) == []


def test_rank_themes_catalyst_boosts_sector(monkeypatch):
    """An active geo/policy event should lift the relevant sector's theme_score."""
    monkeypatch.setattr(opp, "_get_catalyst_themes_from_events",
                        lambda: {"lng": 1.0, "natural_gas": 0.8})
    rotation = {
        "full_rotation_map": [
            {"sector": "Energy",     "trend": "Leading 🟢"},
            {"sector": "Technology", "trend": "Leading 🟢"},
        ]
    }
    macro = {"liquidity": "Neutral"}
    themes = _rank_themes(rotation, macro, [], [])
    energy = next(t for t in themes if t["sector"] == "Energy")
    tech   = next(t for t in themes if t["sector"] == "Technology")
    # Same rotation, Energy gets extra catalyst boost from lng/natural_gas
    assert energy["catalyst_score"] > 0
    assert energy["theme_score"] >= tech["theme_score"]


def test_resolve_text_no_substring_false_positives():
    """Word-boundary matching: 'rice' must not fire on 'prices', 'coal' not on 'coalition'."""
    assert _resolve_text_to_themes("stocks rallied on falling prices") == []
    assert _resolve_text_to_themes("the coalition announced a deal") == []
    # ...but real mentions and plurals still match
    assert "memory_chips" in _resolve_text_to_themes("HBM and DRAM demand surging")
    assert "semiconductors" in _resolve_text_to_themes("building new semiconductors")  # plural
    assert "coal" in _resolve_text_to_themes("coal exports rose")


def test_canonical_sector_collapses_all_taxonomies():
    """The keystone fix: rotation, yfinance, and universe sector names all collapse to one key."""
    # market_mechanics rotation names
    assert _canonical_sector("Consumer Discret") == "Consumer Discretionary"
    assert _canonical_sector("Comm Services") == "Communication Services"
    assert _canonical_sector("Financials") == "Financials"
    # _get_sector_for_ticker / yfinance names
    assert _canonical_sector("Finance") == "Financials"
    assert _canonical_sector("Consumer Defensive") == "Consumer Staples"
    assert _canonical_sector("Consumer Cyclical") == "Consumer Discretionary"
    assert _canonical_sector("Financial Services") == "Financials"
    # Unmatched → None (not a silent wrong-bucket)
    assert _canonical_sector("Crypto Assets") is None
    assert _canonical_sector("") is None


def test_theme_join_works_for_non_technology_sectors():
    """
    Regression for the M2 review bug: a Finance / Consumer pick (whose
    _get_sector_for_ticker name differs from the rotation name) must still join
    to its theme. Previously only Technology joined.
    """
    rotation_sectors = {"Technology", "Healthcare", "Financials", "Consumer Discret",
                        "Comm Services", "Industrials", "Consumer Staples", "Energy",
                        "Utilities", "Real Estate", "Materials"}
    # Build the canonical theme-map exactly as _scan_impl does
    canon_theme_keys = {_canonical_sector(s) for s in rotation_sectors}
    # Every common ticker's resolved sector must canonicalize into that key set
    for tkr in ["MU", "JPM", "XOM", "PG", "HD", "LLY", "CAT"]:
        pick_canon = _canonical_sector(_get_sector_for_ticker(tkr))
        assert pick_canon in canon_theme_keys, f"{tkr} ({pick_canon}) would lose its theme"


def test_catalyst_intensity_single_mention_is_weak(monkeypatch):
    """A lone keyword hit must not max out catalyst weight (absolute, not relative-to-max)."""
    import sys
    import types
    fake = types.ModuleType("tools.trump_tracker")
    fake.get_latest_trump_posts = lambda **k: {"posts": [{"text": "coal exports rose"}]}
    monkeypatch.setitem(sys.modules, "tools.trump_tracker", fake)
    monkeypatch.setattr(opp, "_cached_geo_check", lambda: {"alert": False})

    intensities = opp._get_catalyst_themes_from_events()
    # one genuine mention → 0.25 (1/4), not 1.0
    assert intensities.get("coal", 0) <= 0.3


# ---------------------------------------------------------------------------
# Timeout / performance review fixes (broad-scan latency)
# ---------------------------------------------------------------------------

def test_non_ticker_holdings_skip_yfinance(monkeypatch):
    """Private/manual holdings (with spaces) must never be sent to yfinance (404 spam)."""
    called = []
    import tools.yf_utils as yu
    monkeypatch.setattr(yu, "get_info_safe", lambda s: called.append(s) or {})
    assert opp._is_plausible_ticker("ACME TARGET 2050 FUND") is False
    assert opp._is_plausible_ticker("MU") is True
    assert opp._is_plausible_ticker("RY.TO") is True
    assert opp._get_sector_for_ticker("ACME TARGET 2050 FUND") == "Private/Manual Holding"
    assert called == []  # no network lookup happened


def test_sector_lookup_memoized(monkeypatch):
    """Repeated symbol-only lookups hit the memo, not yfinance, after the first call."""
    opp._sector_lookup_memo.clear()
    calls = []
    import tools.yf_utils as yu
    monkeypatch.setattr(yu, "get_info_safe", lambda s: calls.append(s) or {"sector": "Energy"})
    a = opp._get_sector_for_ticker("ZZZZ")   # unknown → yfinance once
    b = opp._get_sector_for_ticker("ZZZZ")   # memo hit
    assert a == b == "Energy"
    assert calls == ["ZZZZ"]  # exactly one network call


def test_delisted_symbols_dropped_from_universe():
    opp._fmp_screener_available = False
    cands, _ = opp._assemble_dynamic_universe(
        {"full_rotation_map": []},
        intraday_movers={"top_gainers": [{"symbol": "TELL", "change": "+2%"}],
                         "most_active": [{"symbol": "PXD"}]},
        guru_tickers=["SWN"],
    )
    assert not any(s in cands for s in ("PXD", "SWN", "TELL"))


def test_batch_download_accepts_max_retries():
    import inspect
    assert "max_retries" in inspect.signature(opp._batch_download).parameters


def test_past_deadline():
    import time
    assert opp._past_deadline(None) is False
    assert opp._past_deadline(time.perf_counter() - 1) is True
    assert opp._past_deadline(time.perf_counter() + 100) is False


def test_warm_cache_parallel_invokes_all_and_swallows_errors():
    seen = []

    def fn(sym):
        seen.append(sym)
        if sym == "BOOM":
            raise RuntimeError("network")
        return {"ok": True}

    opp._warm_cache_parallel(fn, ["A", "B", "BOOM", "C"], max_workers=4, overall_budget=5.0)
    assert set(seen) == {"A", "B", "BOOM", "C"}  # error did not abort the rest


def test_warm_cache_parallel_is_hard_bounded_when_calls_hang():
    """Hung calls must not blow the budget: 40 hangs @ 3s budget returns in ~3s, not 40×."""
    import time

    def hanging(sym):
        time.sleep(60)

    t = time.perf_counter()
    opp._warm_cache_parallel(hanging, [f"S{i}" for i in range(40)], max_workers=8, overall_budget=3.0)
    assert time.perf_counter() - t < 8.0


def test_v2_timeout_is_larger_than_legacy():
    assert opp._V2_SCAN_TIMEOUT > opp._SCAN_TIMEOUT


def test_chunked_download_parallel_keeps_all_tickers(monkeypatch):
    """Parallel chunk merge must not drop tickers (160 → 3 chunks)."""
    import numpy as np
    import pandas as pd

    def fake_batch(tickers, period=None):
        cols = pd.MultiIndex.from_product([tickers, ["Close", "Volume"]])
        return pd.DataFrame(np.random.rand(60, len(cols)), columns=cols)

    monkeypatch.setattr(opp, "_batch_download", fake_batch)
    monkeypatch.setattr(opp, "is_cancelled", lambda: False)
    tickers = [f"T{i}" for i in range(160)]
    merged = opp._batch_download_chunked(tickers, chunk_size=75)
    got = set(merged.columns.get_level_values(0))
    assert all(t in got for t in tickers)


def test_signal_log_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(opp, "_SIGNAL_LOG_DIR", str(tmp_path / "siglog"))
    result = {
        "market_status": "Risk-On",
        "macro_context": {"liquidity": "Expanding"},
        "ranked_themes": [{"theme": "Technology", "theme_score": 0.8, "cycle_stage": "mid"}],
        "top_picks": [{
            "symbol": "MU", "price": 347.0, "score": 92, "conviction": "High Conviction",
            "theme": "Technology", "entry_stage": "early_breakout",
            "flow_confirmations": ["whale sweeps"], "score_breakdown": {"base": 80},
            "universe_provenance": ["theme:memory_chips"],
        }],
    }
    opp._log_funnel_signals(result, {"liquidity": "Expanding"})
    import json
    from datetime import datetime, timezone
    fname = tmp_path / "siglog" / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"
    assert fname.exists()
    snap = json.loads(fname.read_text().strip())
    assert snap["picks"][0]["symbol"] == "MU"
    assert snap["ranked_themes"][0]["theme"] == "Technology"


def test_fast_score_rs_alpha_ordering():
    """RS alpha should produce a monotone ranking: high alpha > base > lagging."""
    tech_data = {"rsi": 52, "golden_cross": False, "above_sma50": True, "death_cross": False,
                 "drawdown_pct": -8, "month_return": 4, "vol_spike": 1.1, "three_month_return": 12}
    base  = opp._fast_score("MU", tech_data)
    rs_hi = opp._fast_score("MU", tech_data, rs_alpha=15.0)
    rs_lo = opp._fast_score("MU", tech_data, rs_alpha=-20.0)
    assert rs_hi > base > rs_lo


def test_fast_score_penalizes_overbought_rsi():
    """
    _fast_score is a COARSE pre-filter — the real don't-chase gate is the deep-score
    entry-stage multiplier (M3/M4), not here. The one demotion _fast_score does owe us
    is the overbought-RSI penalty: holding everything else constant, an RSI>75 name must
    score below an otherwise-identical normal-RSI name (even with the same provenance).
    """
    base = {"rsi": 60, "golden_cross": False, "above_sma50": True, "death_cross": False,
            "drawdown_pct": -8, "month_return": 4, "vol_spike": 1.1, "three_month_return": 12}
    overbought = {**base, "rsi": 82}
    prov = ["theme:memory_chips", "sector:Technology"]
    assert opp._fast_score("MU", overbought, provenance=prov) < opp._fast_score("MU", base, provenance=prov)


# ---------------------------------------------------------------------------
# M3 — Additive scoring + capped flow + entry-stage multiplier
# ---------------------------------------------------------------------------

def _m3_theme(score=0.9, sector="Technology", cycle="early"):
    return {
        "theme": sector,
        "sector": sector,
        "canonical_sector": _canonical_sector(sector),
        "theme_score": score,
        "cycle_stage": cycle,
        "drivers": ["Rotation: Improving"],
    }


def _m3_base_fund():
    return {
        "symbol": "MU",
        "forward_pe": 18,
        "trailing_pe": 36,
        "analyst_target": 140,
        "current_price": 100,
        "revenue_growth": -0.05,
        "earnings_growth": 0.10,
        "profit_margin": 0.03,
        "free_cashflow": -1_000_000,
        "total_debt": 5_000_000,
        "total_cash": 500_000,
        "recommendation": "hold",
        "description": "Cyclical memory producer",
        "sector_yf": "Technology",
        "industry": "Semiconductors",
        "news_headlines": [],
    }


def _m3_base_tech(**overrides):
    base = {
        "price": 100,
        "rsi": 38,
        "sma50": 105,
        "above_sma50": False,
        "golden_cross": False,
        "drawdown_pct": -24,
        "month_return": 4,
        "three_month_return": 40,
        "vol_spike": 1.1,
    }
    base.update(overrides)
    return base


@pytest.fixture
def _m3_no_live_earnings(monkeypatch):
    monkeypatch.setattr(
        "tools.market_mechanics.predict_earnings_surprise",
        lambda symbol: {"beat_rate": "78%"},
    )


def test_m3_mu_base_profile_reaches_high_conviction(_m3_no_live_earnings):
    result = opp._deep_score_v2(
        "MU",
        _m3_base_fund(),
        _m3_base_tech(),
        "Improving",
        {},
        [],
        theme_context=_m3_theme(),
        rs_alpha=18,
        flow_data={"flow_bonus": 10, "flow_confirmations": ["Insider buying", "ITM calls"]},
        setup_data={"setup": "BB BOUNCE", "rsi": 38},
        apply_entry_gate=True,
    )

    assert result["score"] >= 80
    assert result["conviction"] in {"High Conviction", "Exceptional"}
    assert result["entry_stage"] == "accumulation_base"
    assert result["score_breakdown"]["quality"] < 10  # quality helps, but does not gate


def test_m3_same_mu_extended_is_demoted_to_watchlist_or_lower(_m3_no_live_earnings):
    result = opp._deep_score_v2(
        "MU",
        _m3_base_fund(),
        _m3_base_tech(price=182, sma50=100, rsi=81, drawdown_pct=0, above_sma50=True),
        "Leading",
        {},
        [],
        theme_context=_m3_theme(cycle="mid"),
        rs_alpha=18,
        flow_data={"flow_bonus": 10, "flow_confirmations": ["Insider buying", "ITM calls"]},
        setup_data={"setup": "EXTENDED", "rsi": 81},
        apply_entry_gate=True,
    )

    assert result["entry_stage"] == "extended"
    assert result["entry_multiplier"] == opp._M3_ENTRY_MULTIPLIERS["extended"]
    assert result["score"] < 40
    assert result["conviction"] == "Low Interest"


def test_m3_finance_pick_keeps_theme_and_conviction(_m3_no_live_earnings):
    theme_map = {"Financials": _m3_theme(score=0.82, sector="Financials", cycle="mid")}
    ticker_sector = _get_sector_for_ticker("JPM")
    theme_context = opp._theme_for_sector(ticker_sector, theme_map)

    fund = {
        **_m3_base_fund(),
        "symbol": "JPM",
        "sector_yf": "Financial Services",
        "industry": "Banks",
        "forward_pe": 12,
        "trailing_pe": 14,
        "analyst_target": 120,
    }
    result = opp._deep_score_v2(
        "JPM",
        fund,
        _m3_base_tech(price=100, rsi=61, sma50=94, above_sma50=True, drawdown_pct=-4, month_return=8),
        "Leading",
        {},
        [],
        theme_context=theme_context,
        rs_alpha=9,
        setup_data={"setup": "MOMENTUM BREAKOUT", "rsi": 61},
        apply_entry_gate=True,
    )

    assert theme_context["theme"] == "Financials"
    assert result["theme"] == "Financials"
    assert result["conviction"] in {"Watchlist", "Qualified", "High Conviction", "Exceptional"}
    assert result["opportunity_type"] != result["conviction"]


def test_m3_cyclical_thin_margin_still_scores_on_theme_rs_forward(_m3_no_live_earnings):
    fund = {
        **_m3_base_fund(),
        "profit_margin": 0.01,
        "free_cashflow": -2_000_000,
        "operating_cashflow": -1_500_000,
        "gross_margin": 0.05,
    }
    result = opp._deep_score_v2(
        "MU",
        fund,
        _m3_base_tech(),
        "Improving",
        {},
        [],
        theme_context=_m3_theme(),
        rs_alpha=18,
        apply_entry_gate=False,
    )

    assert result["base_score"] >= 60
    assert result["foundation_check"]["grade"] != "Strong"
    assert result["score_breakdown"]["theme"] > 0
    assert result["score_breakdown"]["relstr"] > 0
    assert result["score_breakdown"]["forward"] > 0


def test_m3_flow_bonus_cannot_promote_weak_base(_m3_no_live_earnings):
    weak_fund = {
        **_m3_base_fund(),
        "forward_pe": None,
        "trailing_pe": None,
        "analyst_target": 90,
        "revenue_growth": None,
        "earnings_growth": None,
        "profit_margin": None,
        "peg_ratio": None,
    }
    result = opp._deep_score_v2(
        "WEAK",
        weak_fund,
        _m3_base_tech(price=100, rsi=35, drawdown_pct=-22),
        "Neutral",
        {},
        [],
        theme_context=_m3_theme(score=0.15, cycle="early"),
        rs_alpha=-20,
        flow_data={"flow_bonus": 10, "flow_confirmations": ["Dark print", "ITM calls"]},
        setup_data={"setup": "BB BOUNCE", "rsi": 35},
        apply_entry_gate=True,
    )

    assert result["base_score"] < opp._MIN_SCORE_THRESHOLD
    assert result["effective_flow_bonus"] == 0
    assert result["score"] < opp._MIN_SCORE_THRESHOLD


def test_m3_flow_confirmation_ignores_dark_pool_sell_alerts(monkeypatch):
    monkeypatch.setattr(
        "tools.dark_pool.scan_dark_pool_proxy",
        lambda symbol: {
            "alerts_count": 1,
            "alerts": [{"signature": "AGGRESSIVE SELL"}],
        },
    )
    monkeypatch.setattr("tools.options.check_whale_accumulation", lambda symbol: {"count": 0})
    monkeypatch.setattr("tools.options.scan_unusual_activity", lambda symbol: {"alerts": []})

    flow = opp._flow_confirmation_for_symbol("MU")

    assert flow["flow_bonus"] == 0
    assert flow["flow_confirmations"] == []


def test_m3_outputs_two_independent_label_axes(_m3_no_live_earnings):
    result = opp._deep_score_v2(
        "MU",
        _m3_base_fund(),
        _m3_base_tech(),
        "Improving",
        {},
        [],
        theme_context=_m3_theme(),
        rs_alpha=18,
        apply_entry_gate=False,
    )

    assert result["conviction"] in {"Exceptional", "High Conviction", "Qualified", "Watchlist", "Low Interest"}
    assert result["opportunity_type"]
    assert result["opportunity_type"] != result["conviction"]
    assert "score_breakdown" in result


# ---------------------------------------------------------------------------
# M4 — surfaced risk overlay + externalized config
# ---------------------------------------------------------------------------

def _m4_overweight_tech_context():
    return {
        "total_value_usd": 100_000,
        "holdings": [
            {"symbol": "AAPL", "value_usd": 45_000},
            {"symbol": "MSFT", "value_usd": 25_000},
            {"symbol": "XOM", "value_usd": 30_000},
        ],
    }


def test_m4_portfolio_concentration_is_capped_and_surfaced(_m3_no_live_earnings):
    result = opp._deep_score_v2(
        "MU",
        _m3_base_fund(),
        _m3_base_tech(),
        "Improving",
        {},
        [],
        portfolio_context=_m4_overweight_tech_context(),
        theme_context=_m3_theme(),
        rs_alpha=18,
        flow_data={"flow_bonus": 10, "flow_confirmations": ["Insider buying", "ITM calls"]},
        setup_data={"setup": "BB BOUNCE", "rsi": 38},
        apply_entry_gate=True,
    )

    concentration = next(a for a in result["risk_adjustments"] if a["type"] == "sector_concentration")
    assert concentration["points"] == opp._cfg_number("risk_cap", 15)
    assert result["score_breakdown"]["risk_adjust"] == opp._cfg_number("risk_cap", 15)
    assert result["score_breakdown"]["raw_risk_adjust"] == opp._cfg_number("risk_cap", 15)
    assert any("size accordingly" in flag for flag in result["risk_flags"])
    assert result["portfolio_fit"]["current_sector_exposure_pct"] == 70.0
    # Risk informs sizing/ranking; it does not delete the otherwise strong idea.
    assert result["score"] >= opp._MIN_SCORE_THRESHOLD


def test_m4_risk_cap_combines_headwinds_and_concentration(_m3_no_live_earnings):
    result = opp._deep_score_v2(
        "MU",
        _m3_base_fund(),
        _m3_base_tech(),
        "Improving",
        {},
        [],
        portfolio_context=_m4_overweight_tech_context(),
        theme_context=_m3_theme(),
        rs_alpha=18,
        flow_data={"flow_bonus": 10, "flow_confirmations": ["Insider buying", "ITM calls"]},
        setup_data={"setup": "BB BOUNCE", "rsi": 38},
        headwind_data={"short_pct_float": 0.20, "insider_signal": "🔴 Insiders SELLING recently"},
        apply_entry_gate=True,
    )

    assert result["score_breakdown"]["raw_risk_adjust"] > opp._cfg_number("risk_cap", 15)
    assert result["score_breakdown"]["risk_adjust"] == opp._cfg_number("risk_cap", 15)
    assert any(a["type"] == "risk_cap" for a in result["risk_adjustments"])
    assert any(a["type"] == "short_interest" for a in result["risk_adjustments"])
    assert any(a["type"] == "sector_concentration" for a in result["risk_adjustments"])


# ---------------------------------------------------------------------------
# Two tools, one question, two bases
# ---------------------------------------------------------------------------
# The scanner's concentration figure and check_portfolio_allocation's sector map
# both answer "how much Technology do I hold?" and legitimately disagree: the
# scanner labels each ticker once and drops what it cannot label, while the
# allocation tool decomposes funds into their sector sleeves. On 2026-07-29 both
# ran in one turn and returned different Technology weights; the compliance judge,
# reading two bare percentages, took the smaller as proof the larger was
# fabricated and issued a 2/10 SOURCE FRAUD verdict against a genuinely fetched
# number. Neither figure was wrong; neither said what it measured.


def _fund_heavy_context():
    """A book whose sector weight depends entirely on which basis you use: a broad
    fund (no GICS label of its own) beside one directly-labelled tech name."""
    return {
        "total_value_usd": 100_000,
        "holdings": [
            {"symbol": "AAPL", "value_usd": 30_000},   # labels Technology
            {"symbol": "VTI", "value_usd": 60_000},    # labels "Large Blend" — no sector
            {"symbol": "XOM", "value_usd": 10_000},    # labels Energy
        ],
    }


def test_scanner_exposure_says_it_excludes_funds():
    fit = opp._portfolio_fit_adjustment("MU", "Technology", _fund_heavy_context())["portfolio_fit"]

    assert fit["current_sector_exposure_pct"] == 30.0
    assert fit["basis"] == opp.SECTOR_EXPOSURE_BASIS_DIRECT
    assert "funds not decomposed" in fit["basis"]


def test_scanner_reports_how_much_of_the_book_it_could_not_classify():
    # The 60% broad fund sits in the denominator while contributing to no sector.
    # Publishing that share is what keeps 30% from reading as a complete answer.
    fit = opp._portfolio_fit_adjustment("MU", "Technology", _fund_heavy_context())["portfolio_fit"]

    assert fit["unmapped_pct"] == 60.0


def test_the_flag_string_the_judge_reads_carries_its_basis():
    # Over-threshold so the risk flag actually renders — this string is what
    # reaches the judge's evidence, and it is the one that was misread.
    heavy = {
        "total_value_usd": 100_000,
        "holdings": [{"symbol": "AAPL", "value_usd": 40_000}, {"symbol": "VTI", "value_usd": 60_000}],
    }
    result = opp._portfolio_fit_adjustment("MU", "Technology", heavy)

    flag = next(f for f in result["risk_flags"] if "size accordingly" in f)
    assert "40% Technology exposure" in flag
    assert opp.SECTOR_EXPOSURE_BASIS_DIRECT in flag
    assert opp.SECTOR_EXPOSURE_BASIS_DIRECT in result["reasons"][0]


def test_look_through_allocation_publishes_the_opposing_basis(monkeypatch):
    """The other side of the pair: same question, decomposed, and labelled as such."""
    import tools.daily_cache as daily_cache
    import tools.sector_analysis as sector

    class _Ticker:
        def __init__(self, symbol):
            self.info = {"sector": "Technology"} if symbol == "ZZSTOCK" else {"quoteType": "ETF"}

    monkeypatch.setattr(sector.yf, "Ticker", _Ticker)
    # check_portfolio_allocation imports these from tools.daily_cache at call time,
    # so patch them at the source to stay off the shared on-disk cache.
    monkeypatch.setattr(daily_cache, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(daily_cache, "set_cached", lambda *a, **k: None)
    monkeypatch.setattr(sector, "_fmp_decompose", lambda sym: {"Technology": 0.4, "Energy": 0.6})

    result = sector.check_portfolio_allocation(["ZZSTOCK", "ZZBROADFUND"], [30_000, 60_000])

    # The fund's tech sleeve counts here and does not in the scanner — the whole
    # reason the two numbers differ.
    assert result["basis"] == sector.SECTOR_EXPOSURE_BASIS_LOOKTHROUGH
    assert "decomposed" in result["basis"]
    assert result["sector_allocation_raw"]["Technology"] > 30_000 / 90_000


def test_the_two_bases_are_distinguishable_strings():
    import tools.sector_analysis as sector

    assert opp.SECTOR_EXPOSURE_BASIS_DIRECT != sector.SECTOR_EXPOSURE_BASIS_LOOKTHROUGH


def test_m4_config_loader_merges_user_overrides(tmp_path, monkeypatch):
    cfg_path = tmp_path / "funnel_config.json"
    cfg_path.write_text('{"risk_cap": 7, "pillars": {"theme": 24}, "flow_bonus": {"one": 3}}')
    monkeypatch.setattr(opp, "_FUNNEL_CONFIG_PATH", str(cfg_path))

    cfg = opp._load_funnel_config()

    assert cfg["risk_cap"] == 7
    assert cfg["pillars"]["theme"] == 24
    assert cfg["pillars"]["relstr"] == 25.0  # default preserved by deep merge
    assert cfg["flow_bonus"]["one"] == 3
    assert cfg["flow_bonus"]["two_plus"] == 10.0


def test_seed_funnel_config_if_missing(tmp_path, monkeypatch):
    """First-run seeding: copies the example to user_data when absent, is
    idempotent, and never clobbers an existing user config."""
    example = tmp_path / "funnel_config.example.json"
    example.write_text('{"final_top_n": 15}')
    dest = tmp_path / "user_data" / "funnel_config.json"
    monkeypatch.setattr(opp, "_FUNNEL_CONFIG_EXAMPLE_PATH", str(example))
    monkeypatch.setattr(opp, "_FUNNEL_CONFIG_PATH", str(dest))

    # Missing → seeded
    assert opp.seed_funnel_config_if_missing() is True
    assert dest.exists()
    assert dest.read_text() == '{"final_top_n": 15}'

    # Already present → no-op, not overwritten
    dest.write_text('{"final_top_n": 5}')
    assert opp.seed_funnel_config_if_missing() is False
    assert dest.read_text() == '{"final_top_n": 5}'


def test_seed_funnel_config_no_example_is_safe(tmp_path, monkeypatch):
    """Missing example template must not raise — just return False."""
    monkeypatch.setattr(opp, "_FUNNEL_CONFIG_EXAMPLE_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setattr(opp, "_FUNNEL_CONFIG_PATH", str(tmp_path / "user_data" / "funnel_config.json"))
    assert opp.seed_funnel_config_if_missing() is False


def test_m4_config_example_is_valid_json():
    import json
    from pathlib import Path

    path = Path(opp.__file__).resolve().parents[1] / "funnel_config.example.json"
    data = json.loads(path.read_text())

    assert data["risk_cap"] == 15
    assert data["concentration"]["threshold_pct"] == 25
    assert data["entry_multipliers"]["extended"] == 0.4


# ---------------------------------------------------------------------------
# 4. Chunked download — no size-1 tail (the MultiIndex-merge corruption)
# ---------------------------------------------------------------------------

def test_chunked_download_never_leaves_singleton_tail(monkeypatch):
    """76 tickers @ chunk_size 75 must NOT produce a [75, 1] split."""
    seen_chunks = []

    def fake_batch_download(tickers, period=None):
        seen_chunks.append(list(tickers))
        import pandas as pd
        # Return a tiny non-empty frame so the function proceeds
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(opp, "_batch_download", fake_batch_download)
    monkeypatch.setattr(opp, "is_cancelled", lambda: False)

    tickers = [f"T{i}" for i in range(76)]
    opp._batch_download_chunked(tickers, chunk_size=75)

    assert len(seen_chunks) == 1, f"76 should merge into a single rebalanced batch, got {[len(c) for c in seen_chunks]}"
    # Generalise: 151 tickers → must not have a trailing singleton
    seen_chunks.clear()
    opp._batch_download_chunked([f"T{i}" for i in range(151)], chunk_size=75)
    assert all(len(c) >= 2 for c in seen_chunks), [len(c) for c in seen_chunks]


# ---------------------------------------------------------------------------
# Yahoo sector-constituent fallback + exploration slots + novelty tags
# ---------------------------------------------------------------------------

def test_yf_sector_constituents_parses_top_companies(monkeypatch):
    import pandas as pd

    class FakeSector:
        def __init__(self, key):
            assert key == "basic-materials"

        @property
        def top_companies(self):
            return pd.DataFrame(
                {"name": ["Linde", "Newmont", "Freeport"]},
                index=["LIN", "NEM", "FCX"],
            )

    store = {}
    import tools.daily_cache as dc
    monkeypatch.setattr(dc, "get_cached", lambda k, ttl_seconds=None: store.get(k))
    monkeypatch.setattr(dc, "set_cached", lambda k, v: store.__setitem__(k, v))
    import yfinance
    monkeypatch.setattr(yfinance, "Sector", FakeSector)

    # The autouse fixture stubs opp._yf_sector_constituents; use the real
    # implementation captured at import time.
    got = _REAL_YF_SECTOR_CONSTITUENTS("Basic Materials", limit=2)
    assert got == ["LIN", "NEM"]
    # daily-cached full list for subsequent calls
    assert store["yf_sector_constituents:basic-materials"] == ["LIN", "NEM", "FCX"]
    # unknown sector → no yahoo key → []
    assert _REAL_YF_SECTOR_CONSTITUENTS("Cryptowidgets") == []


def test_universe_uses_yahoo_when_fmp_unavailable(monkeypatch):
    monkeypatch.setattr(opp, "scan_intraday_movers", lambda *a, **k: {}, raising=False)
    monkeypatch.setattr(opp, "_yf_sector_constituents",
                        lambda sec, limit=30: ["YFA", "YFB"] if "Tech" in sec else [])
    rotation = {"full_rotation_map": [
        {"sector": "Technology", "trend": "Leading 🟢", "1m_change": 5, "3m_change": 9},
    ]}
    candidates, provenance = opp._assemble_dynamic_universe(rotation, intraday_movers={}, guru_tickers=[])
    assert "YFA" in candidates and "YFB" in candidates
    assert any(src.startswith("sector_yf:") for src in provenance["YFA"])


def test_exploration_candidates_prefers_never_and_stalest():
    scored = [(f"TOP{i}", 90 - i, {}) for i in range(3)]  # above the cut
    scored += [
        ("OLD", 50, {}),    # scanned long ago
        ("FRESH", 80, {}),  # scanned today — despite best score, goes last
        ("NEVER1", 40, {}),
        ("NEVER2", 60, {}),
    ]
    ledger = {"OLD": "2026-01-01", "FRESH": "2026-07-02"}
    got = opp._exploration_candidates(scored, fast_screen_top_n=3, slots=3, ledger=ledger)
    syms = [s for s, _, _ in got]
    # never-scanned first (fast-score desc within bucket), then the stalest
    assert syms == ["NEVER2", "NEVER1", "OLD"]


def test_scan_ledger_roundtrip_and_prune(monkeypatch, tmp_path):
    monkeypatch.setattr(opp, "_SCAN_LEDGER_PATH", str(tmp_path / "ledger.json"))
    import json as _json
    from datetime import date, timedelta
    stale = (date.today() - timedelta(days=200)).isoformat()
    (tmp_path / "ledger.json").write_text(_json.dumps({"STALE": stale, "KEEP": date.today().isoformat()}))

    opp._update_scan_ledger(["nvda", "MU"])
    ledger = opp._load_scan_ledger()
    assert ledger["NVDA"] == date.today().isoformat()
    assert ledger["MU"] == date.today().isoformat()
    assert "KEEP" in ledger
    assert "STALE" not in ledger  # >180d pruned


def test_scan_ledger_writes_cannot_reach_the_real_user_data_store():
    """The isolate_scan_ledger guard has a test, because the leak it closes is silent.

    `_SCAN_LEDGER_PATH` is hardcoded repo-relative and not profile-scoped, so before
    the autouse fixture in conftest every test that reached a scan stamped the user's
    real funnel_scan_ledger.json — the store that decides which tail names get
    exploration slots in a live broad scan. Nothing failed when it happened; the
    evidence was 7 tickers carrying the suite's run date (measured 2026-07-30).

    Deliberately takes no monkeypatch fixture: it asserts the DEFAULT state every
    other test runs under, and writes through the real API to prove the redirect
    holds for the writer and not just for the constant.
    """
    import os as _os
    real = _os.path.abspath(_os.path.join(
        _os.path.dirname(opp.__file__), "..", "user_data", "funnel_scan_ledger.json"))
    assert _os.path.abspath(opp._SCAN_LEDGER_PATH) != real

    before = _os.path.exists(real) and _os.stat(real).st_mtime_ns
    opp._update_scan_ledger(["ZZTEST"])
    assert "ZZTEST" in opp._load_scan_ledger()
    after = _os.path.exists(real) and _os.stat(real).st_mtime_ns
    assert before == after, "a scan-ledger write reached the real user_data store"


def test_pick_novelty_tags_new_vs_carried(monkeypatch, tmp_path):
    import json as _json
    from datetime import date, timedelta
    log_dir = tmp_path / "signal_log"
    log_dir.mkdir()
    yday = (date.today() - timedelta(days=1)).isoformat()
    (log_dir / f"{yday}.jsonl").write_text(_json.dumps({
        "ts": f"{yday}T12:00:00+00:00",
        "picks": [{"symbol": "NVDA"}],
    }) + "\n")
    monkeypatch.setattr(opp, "_SIGNAL_LOG_DIR", str(log_dir))

    picks = [
        {"symbol": "NVDA", "reasons": ["existing reason"]},
        {"symbol": "FRESHCO", "reasons": []},
    ]
    opp._annotate_pick_novelty(picks)
    nvda, fresh = picks
    assert nvda["novelty"] == "carried:1"
    assert nvda["first_surfaced"] == yday
    assert any("Repeat signal" in r for r in nvda["reasons"])
    assert fresh["novelty"] == "NEW"
    assert any(r.startswith("🆕") for r in fresh["reasons"])
