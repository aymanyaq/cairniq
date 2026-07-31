"""The 2026-07-28 "screeners timed out" turn, broken into its four causes.

The user asked for external tickers; both screener tools reported a flat timeout
and the answer fell back to macro-only ETFs. Nothing had actually failed — the
scan ran to completion 60s AFTER the user was served, twice, and its results
were discarded. Four independent defects stacked:

  1. `screen_stocks` and `scan_opportunities` are the same function behind two
     names, so the planner calling both ran two identical pipelines at once.
  2. The caller's batch ceiling (120s) sat BELOW the tool's own budget (150s),
     so the tool's result could never arrive in time to be observed.
  3. Phase 5 had no deadline awareness and per-future (not per-batch) timeouts,
     so it ran ~33s past the scan deadline.
  4. `@cached` stamped `_as_of` into map-shaped payloads, making a phantom
     "_as_of" ticker whose string value crashed every consumer that iterated —
     which is what had silently killed the 13F universe producer.

Each test below pins one of them. They assert on real structures rather than
mocks of them: a stale-schema mock is how the previous instance of this class of
bug stayed green.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import tools.cache as cache_mod
import tools.opportunity_scanner as scanner
import tools.sec_edgar as sec_edgar
from tools.freshness import AS_OF_KEY


# ---------------------------------------------------------------------------
# 1. Duplicate pipeline
# ---------------------------------------------------------------------------
def test_identical_concurrent_scans_run_the_pipeline_once(monkeypatch):
    """screen_stocks('All') + scan_opportunities('All') = ONE pipeline.

    Both tool names call scan_sector_opportunities with the same argument. Before
    the in-flight registry this doubled every network stage — two 143-ticker
    universes, two 30s earnings warms — against shared rate limits.
    """
    executions = []

    def fake_impl(sector, portfolio_context=None, deadline=None):
        executions.append(sector)
        time.sleep(1.0)
        return {"sector": sector, "top_picks": [{"symbol": "XLE"}]}

    monkeypatch.setattr(scanner, "_scan_impl", fake_impl)
    monkeypatch.setattr(scanner, "_attach_impact_previews", lambda r, c: None)

    results = {}
    # Released together, NOT staggered. The planner submits both tool names to
    # one executor, so they arrive microseconds apart; a `time.sleep(0.05)`
    # stagger is wider than the window and passes against a registry that
    # claims its key after the portfolio read instead of under the check's lock.
    gate = threading.Barrier(2)

    def run(name, arg):
        gate.wait()
        results[name] = scanner.scan_sector_opportunities(arg)

    t1 = threading.Thread(target=run, args=("scan_opportunities", "All"))
    t2 = threading.Thread(target=run, args=("screen_stocks", "All"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert executions == ["All"], f"pipeline ran {len(executions)}x, expected 1"
    # The follower gets the real picks, not a degraded "someone else is running" stub.
    assert results["screen_stocks"]["top_picks"] == [{"symbol": "XLE"}]
    assert results["scan_opportunities"]["top_picks"] == [{"symbol": "XLE"}]
    assert scanner._INFLIGHT_SCANS == {}, "registry leaked an entry"


def test_broad_sector_aliases_collapse_to_one_dedup_key(monkeypatch):
    """'All', 'Market' and 'General' are the same scan and must coalesce."""
    executions = []

    def fake_impl(sector, portfolio_context=None, deadline=None):
        executions.append(sector)
        time.sleep(1.0)
        return {"sector": sector, "top_picks": []}

    monkeypatch.setattr(scanner, "_scan_impl", fake_impl)
    monkeypatch.setattr(scanner, "_attach_impact_previews", lambda r, c: None)

    gate = threading.Barrier(3)

    def run(alias):
        gate.wait()
        scanner.scan_sector_opportunities(alias)

    threads = [threading.Thread(target=run, args=(a,))
               for a in ("All", "Market", "General")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(executions) == 1, f"aliases did not coalesce: {executions}"


def test_a_failing_leader_does_not_strand_its_followers(monkeypatch):
    """The follower waits on the leader's claim, so every leader exit path must
    resolve it. An unresolved claim blocks each follower for its own full
    `_V2_SCAN_TIMEOUT` and then reports a timeout for a scan that already died
    in under a second."""
    def boom(sector, portfolio_context=None, deadline=None):
        time.sleep(0.3)
        raise RuntimeError("upstream died")

    monkeypatch.setattr(scanner, "_scan_impl", boom)
    monkeypatch.setattr(scanner, "_attach_impact_previews", lambda r, c: None)

    out = {}
    gate = threading.Barrier(2)

    def run(i):
        gate.wait()
        out[i] = scanner.scan_sector_opportunities("All")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started

    assert elapsed < 10, f"follower blocked {elapsed:.1f}s on an unresolved claim"
    assert len(out) == 2
    for payload in out.values():
        assert "upstream died" in str(payload.get("summary") or payload.get("error"))
    assert scanner._INFLIGHT_SCANS == {}, "registry leaked an entry"


def test_a_different_sector_is_not_deduplicated(monkeypatch):
    """Coalescing is keyed on the sector — 'Energy' must not reuse 'All'."""
    executions = []

    def fake_impl(sector, portfolio_context=None, deadline=None):
        executions.append(sector)
        time.sleep(1.0)
        return {"sector": sector, "top_picks": []}

    monkeypatch.setattr(scanner, "_scan_impl", fake_impl)
    monkeypatch.setattr(scanner, "_attach_impact_previews", lambda r, c: None)

    threads = [
        threading.Thread(target=scanner.scan_sector_opportunities, args=(s,))
        for s in ("All", "Energy")
    ]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()

    assert sorted(executions) == ["All", "Energy"]


# ---------------------------------------------------------------------------
# 2. Timeout ordering
# ---------------------------------------------------------------------------
def test_caller_ceiling_outlives_the_longest_tool_budget():
    """The invariant the whole incident turned on.

    A caller that gives up before its slowest tool's own budget expires can
    never observe that tool's result — every broad scan is reported as a flat
    timeout on principle, however healthy the scan actually was.
    """
    from agent.nodes.market_analyst import _TOOL_BATCH_TIMEOUT

    assert _TOOL_BATCH_TIMEOUT > scanner._V2_SCAN_TIMEOUT, (
        f"batch ceiling {_TOOL_BATCH_TIMEOUT}s <= scan budget "
        f"{scanner._V2_SCAN_TIMEOUT}s — broad scans can never return in time"
    )
    assert _TOOL_BATCH_TIMEOUT > scanner._SCAN_TIMEOUT


# ---------------------------------------------------------------------------
# 3. Phase-5 aggregate budget
# ---------------------------------------------------------------------------
def test_gate_budget_covers_the_batch_not_each_symbol():
    """15 finalists x a 15s per-future timeout was a 225s worst case inside a
    150s scan. The budget must apply to the whole batch."""
    def hang(_sym):
        time.sleep(30)
        return {"flagged": True}

    symbols = [f"SYM{i}" for i in range(15)]
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        future_map = {executor.submit(hang, s): s for s in symbols}
        started = time.perf_counter()
        collected = scanner._collect_bounded(future_map, overall_budget=4.0)
        elapsed = time.perf_counter() - started
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert elapsed < 10, f"batch took {elapsed:.1f}s — budget not enforced"
    assert collected == {}, "nothing finished, so nothing should be reported"


def test_unfinished_symbols_get_the_neutral_default():
    """The flow gate scores every finalist, so each needs an entry. A symbol
    that never returned must read as *no confirmation*, never as confirmed."""
    def hang(_sym):
        time.sleep(30)
        return {"flow_signal_count": 5}

    symbols = [f"SYM{i}" for i in range(6)]
    executor = ThreadPoolExecutor(max_workers=3)
    try:
        future_map = {executor.submit(hang, s): s for s in symbols}
        collected = scanner._collect_bounded(
            future_map, overall_budget=2.0,
            default=lambda: {"flow_bonus": 0.0, "flow_signal_count": 0},
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert set(collected) == set(symbols)
    assert all(v["flow_signal_count"] == 0 for v in collected.values())


def test_phase5_budget_constants_fit_inside_the_scan():
    """The gates plus their reserve must leave the scan room to finish."""
    assert (scanner._PHASE5_GATE_BUDGET_S + scanner._PHASE5_RESERVE_S
            < scanner._V2_SCAN_TIMEOUT)
    assert scanner._PHASE5_MIN_GATE_S < scanner._PHASE5_GATE_BUDGET_S


def test_gates_never_claim_more_time_than_the_deadline_leaves():
    """The constants fitting in isolation is not the property that matters.

    Walk the whole curve of "time actually left" and check what the three gates
    would claim in sequence. A per-gate floor applied WITHOUT consulting the
    clock passes the constants check above while spending a flat ~14.7s at every
    point below 15s — which is +3.6s past the hard timeout when 6s remain, i.e.
    the timeout this budget exists to prevent.
    """
    MIN, RESERVE, GATE = (scanner._PHASE5_MIN_GATE_S,
                          scanner._PHASE5_RESERVE_S,
                          scanner._PHASE5_GATE_BUDGET_S)
    for tenths in range(0, 1200):
        remaining = tenths / 10.0
        if remaining < MIN + RESERVE:
            continue  # caller skips every gate below this
        budget = min(GATE, remaining - RESERVE)
        left = remaining
        claimed = 0.0
        for weight in (0.40, 0.27, 0.33):
            gate = scanner._phase5_gate_slice(budget * weight, left)
            assert gate == 0.0 or gate >= MIN, f"half-run gate {gate:.1f}s"
            claimed += gate
            left -= gate
        assert claimed <= remaining, (
            f"{remaining:.1f}s left but the gates claim {claimed:.1f}s "
            f"— overshoots the deadline by {claimed - remaining:.1f}s"
        )


def test_the_risk_gate_is_the_last_one_to_yield():
    """Under pressure the gates degrade in priority order. Headwind is the risk
    check and runs on every scan; setup and flow are scoring inputs, so a pick
    that loses them is still safe to surface — one that loses the headwind gate
    is not."""
    budget = min(scanner._PHASE5_GATE_BUDGET_S,
                 12.0 - scanner._PHASE5_RESERVE_S)
    headwind = scanner._phase5_gate_slice(budget * 0.40, 12.0)
    setup = scanner._phase5_gate_slice(budget * 0.27, 12.0 - headwind)
    flow = scanner._phase5_gate_slice(budget * 0.33, 12.0 - headwind - setup)

    assert headwind > 0, "the risk gate yielded before the scoring gates"
    assert (setup, flow) == (0.0, 0.0)


def test_no_deadline_means_each_gate_takes_its_nominal_share():
    """A caller with no deadline (a direct tool call) must not be throttled."""
    assert scanner._phase5_gate_slice(12.0, None) == 12.0


# ---------------------------------------------------------------------------
# 4. The cache stamp that poisoned map-shaped payloads
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_cache(monkeypatch):
    store: dict = {}
    monkeypatch.setattr(cache_mod.daily_cache, "get_cached",
                        lambda key, ttl_seconds=None: store.get(key))
    monkeypatch.setattr(cache_mod.daily_cache, "set_cached",
                        lambda key, value: store.__setitem__(key, value))
    return store


def test_map_shaped_payloads_are_not_stamped(isolated_cache):
    """A stamp inside {TICKER: {...}} is a phantom entry, not metadata."""
    @cache_mod.cached(key_func=lambda: "map", stamp=False)
    def ticker_map():
        return {"NVDA": {"cik": "1", "title": "NVIDIA CORP"},
                "AAPL": {"cik": "2", "title": "Apple Inc."}}

    result = ticker_map()
    assert AS_OF_KEY not in result
    assert all(isinstance(v, dict) for v in result.values()), (
        "a non-dict value means a metadata key is masquerading as data"
    )


def test_record_shaped_payloads_still_carry_their_stamp(isolated_cache):
    """The 5.8 freshness contract is unchanged for normal payloads."""
    @cache_mod.cached(key_func=lambda: "record")
    def quote():
        return {"symbol": "NVDA", "price": 100.0}

    assert AS_OF_KEY in quote()


def test_issuer_map_survives_a_legacy_stamped_cache_entry(monkeypatch):
    """Caches written before stamp=False keep serving `_as_of` for the rest of
    their TTL (7 days for the CIK map), so the reader must not trust every key.
    """
    poisoned = {
        "NVDA": {"cik": "0001045810", "title": "NVIDIA CORP"},
        AS_OF_KEY: "2026-07-28T07:18:44",
    }
    monkeypatch.setattr(sec_edgar, "get_cik_map", lambda: poisoned)

    mapping = sec_edgar._issuer_name_to_ticker_map()
    assert mapping.get("NVIDIA") == "NVDA"
    assert "NVDA" in mapping.values()


def test_13f_diff_skips_a_legacy_stamped_holdings_table(monkeypatch):
    """Same poison, one layer down: {CUSIP: {...}} from _fetch_13f_holdings."""
    latest = {
        "AAA": {"name": "NVIDIA CORP", "shares": 200.0, "value": 2.0},
        AS_OF_KEY: "2026-07-28T07:18:44",
    }
    previous = {
        "AAA": {"name": "NVIDIA CORP", "shares": 100.0, "value": 1.0},
        AS_OF_KEY: "2026-07-01T07:18:44",
    }
    monkeypatch.setattr(sec_edgar, "_managers_13f", lambda: {"Test Fund": 1234})
    monkeypatch.setattr(sec_edgar, "_latest_13f_accessions", lambda cik: [
        {"accession": "a1", "filingDate": "2026-07-15"},
        {"accession": "a0", "filingDate": "2026-04-15"},
    ])
    monkeypatch.setattr(sec_edgar, "_fetch_13f_holdings",
                        lambda cik, acc: latest if acc == "a1" else previous)
    monkeypatch.setattr(sec_edgar, "_issuer_name_to_ticker_map",
                        lambda: {"NVIDIA": "NVDA"})

    diff = sec_edgar.get_13f_diff.__wrapped__("Test Fund")

    assert diff["positions_held"] == 2  # raw table length, stamp included
    # The doubled position is reported as an add; the stamp contributes nothing.
    assert [r["ticker"] for r in diff["adds"]] == ["NVDA"]
    assert diff["exits"] == []
    assert diff["new_positions"] == []


# ---------------------------------------------------------------------------
# 5. The 13F cap's tiebreak was doing the selecting
# ---------------------------------------------------------------------------
def _diff(new=(), adds=()):
    return {
        "new_positions": [{"ticker": t, "value": v} for t, v in new],
        "adds": [{"ticker": t, "value": v, "change_pct": c} for t, v, c in adds],
    }


def test_universe_cap_is_broken_by_dollars_not_by_spelling(monkeypatch):
    """Measured 2026-07-28: only 17 of 108 accumulated names were held by more
    than one manager, so 23 of the 40 slots were filled by sorting 91
    single-manager names A→Z and cutting at "CSX" — MSFT ($2.1B accumulated) and
    DAL ($2.6B) were excluded for their spelling while ABR and BYND got in.
    Alphabetical is a determinism tiebreak, never a selector.
    """
    monkeypatch.setattr(sec_edgar, "_managers_13f", lambda: {"F1": 1, "F2": 2})
    monkeypatch.setattr(sec_edgar, "_13F_UNIVERSE_CAP", 3)
    monkeypatch.setattr(sec_edgar, "get_13f_diff", lambda m: (
        _diff(new=[("AAA", 1_000_000), ("ZZZ", 900_000_000)])
        if m == "F1" else
        _diff(new=[("BBB", 5_000_000)], adds=[("AAA", 2_000_000, 10.0)])
    ))

    universe = sec_edgar.get_13f_universe.__wrapped__.__wrapped__()

    # AAA leads on manager count (2). The remaining slots go to the biggest
    # accumulation, not to "BBB" for starting with a B.
    assert universe == ["AAA", "ZZZ", "BBB"]


def test_an_add_contributes_only_the_new_money():
    """A 2% top-up of a mega-position must not outrank a full-size new buy —
    the diff row carries the whole position's value, not the increment."""
    top_up = sec_edgar._accumulated_usd(
        {"value": 1_000_000_000, "change_pct": 2.0})
    new_buy = sec_edgar._accumulated_usd({"value": 100_000_000})

    assert top_up < new_buy
    assert round(top_up) == round(1_000_000_000 * 0.02 / 1.02)


def test_a_trim_or_an_unparseable_change_contributes_nothing():
    assert sec_edgar._accumulated_usd({"value": 5_000_000, "change_pct": -30.0}) == 0.0
    assert sec_edgar._accumulated_usd({"value": 5_000_000, "change_pct": "n/a"}) == 0.0
    assert sec_edgar._accumulated_usd({"change_pct": None}) == 0.0


def test_diff_rows_carry_the_value_the_ranking_needs(monkeypatch):
    """The ranking is only as good as the field it reads. `_row` must emit
    `value`, or every name ranks at $0 and the alphabetical order returns."""
    monkeypatch.setattr(sec_edgar, "_managers_13f", lambda: {"Test Fund": 1234})
    monkeypatch.setattr(sec_edgar, "_latest_13f_accessions", lambda cik: [
        {"accession": "a1", "filingDate": "2026-07-15"},
        {"accession": "a0", "filingDate": "2026-04-15"},
    ])
    monkeypatch.setattr(sec_edgar, "_fetch_13f_holdings", lambda cik, acc: (
        {"AAA": {"name": "NVIDIA CORP", "shares": 200.0, "value": 4_000_000.0}}
        if acc == "a1" else
        {"AAA": {"name": "NVIDIA CORP", "shares": 100.0, "value": 2_000_000.0}}
    ))
    monkeypatch.setattr(sec_edgar, "_issuer_name_to_ticker_map",
                        lambda: {"NVIDIA": "NVDA"})

    diff = sec_edgar.get_13f_diff.__wrapped__.__wrapped__("Test Fund")

    assert diff["adds"][0]["value"] == 4_000_000.0
    # Shares doubled (+100%), so half the position is new money.
    assert round(sec_edgar._accumulated_usd(diff["adds"][0])) == 2_000_000
