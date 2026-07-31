"""Roadmap 4.10 — the TWR engine, and the four gates that stop it lying.

The arithmetic tests here are short because chain-linking is short. Almost
everything below is about the refusals, and that is the right proportion: on the
live book this engine will return a REFUSAL for months, and a refusal that
degrades into a plausible number is the specific failure 4.10 was re-scoped to
prevent. The roadmap's own words — "a plausible-looking TWR over unclassified
deltas is worse than a blank panel".

The load-bearing one is `test_coverage_is_rows_out_of_days_not_endpoint_span`:
two series with an identical span differ by 8% of their days on the live box, and
the existing 365-day gate reads the span.
"""

from datetime import date

import pytest

from tools.attribution import (
    MIN_COVERAGE_PCT,
    blended_benchmark,
    chain_link,
    coverage,
    currency_weights,
    get_attribution_report,
    tracking_error,
)


def _dense_series(start: date, days: int, value: float = 100000.0):
    """A gapless daily valuation series, flat unless a test moves it."""
    from datetime import timedelta

    return [((start + timedelta(days=i)).isoformat(), value) for i in range(days)]


# ---------------------------------------------------------------------------
# Coverage — the gate that has been measuring the wrong thing
# ---------------------------------------------------------------------------
def test_coverage_is_rows_out_of_days_not_endpoint_span():
    """THE test for this item.

    Two series, same endpoints, same span, one with a hole. A span figure calls
    them identical; coverage does not. The live history is 83 rows over an 89-day
    span with 7 days missing in 5 gaps, and the shipped 365-day gate reads the 89.
    """
    as_of = date(2026, 7, 30)
    full = [d for d, _ in _dense_series(date(2026, 7, 21), 10)]
    holed = [d for d in full if d not in {"2026-07-24", "2026-07-25"}]

    a = coverage(full, window_days=10, as_of=as_of)
    b = coverage(holed, window_days=10, as_of=as_of)

    # Identical endpoints, therefore identical SPAN.
    assert a["span_days"] == b["span_days"] == 10
    # And different coverage, which is the figure that matters.
    assert a["coverage_pct"] == 100.0
    assert b["coverage_pct"] == 80.0
    assert b["span_minus_coverage_days"] == 2
    assert b["gap_count"] == 1


def test_coverage_gates_on_the_stated_minimum():
    as_of = date(2026, 7, 30)
    dense = [d for d, _ in _dense_series(date(2026, 7, 21), 10)]
    assert coverage(dense, window_days=10, as_of=as_of)["sufficient"] is True
    assert coverage(dense[:5], window_days=10, as_of=as_of)["sufficient"] is False
    assert coverage(dense, window_days=10, as_of=as_of)["min_coverage_pct"] == MIN_COVERAGE_PCT


def test_coverage_of_an_empty_series_is_zero_not_an_error():
    res = coverage([], window_days=365, as_of=date(2026, 7, 30))
    assert res["coverage_pct"] == 0.0
    assert res["sufficient"] is False


def test_dates_outside_the_window_do_not_count_toward_coverage():
    """A two-year-old row is not coverage of this year."""
    res = coverage(["2024-01-01", "2026-07-30"], window_days=10, as_of=date(2026, 7, 30))
    assert res["observed_days"] == 1


# ---------------------------------------------------------------------------
# Chain-linking
# ---------------------------------------------------------------------------
def test_a_deposit_is_removed_from_the_return():
    """The one thing TWR is for.

    The book goes 100k -> 110k, but 10k of that is a deposit. The return is 0%,
    and the naive value-over-value figure is +10%.
    """
    linked = chain_link([("2026-01-01", 100000.0), ("2026-01-02", 110000.0)],
                        [("2026-01-02", 10000.0)])
    assert linked["twr_pct"] == pytest.approx(0.0, abs=1e-9)
    assert linked["net_external_flow"] == 10000.0


def test_a_withdrawal_does_not_read_as_a_loss():
    linked = chain_link([("2026-01-01", 100000.0), ("2026-01-02", 90000.0)],
                        [("2026-01-02", -10000.0)])
    assert linked["twr_pct"] == pytest.approx(0.0, abs=1e-9)


def test_returns_chain_link_rather_than_add():
    """+10% then +10% is +21%, not +20%. An additive engine is off by the cross
    term, which compounds into a materially wrong annual figure."""
    linked = chain_link([("2026-01-01", 100.0), ("2026-01-02", 110.0),
                         ("2026-01-03", 121.0)], [])
    assert linked["twr_pct"] == pytest.approx(21.0, abs=1e-6)
    assert linked["sub_periods"] == 2


def test_a_sub_period_opening_at_zero_is_refused_rather_than_divided_by():
    linked = chain_link([("2026-01-01", 0.0), ("2026-01-02", 5000.0)], [])
    assert "error" in linked
    assert "twr_pct" not in linked


def test_one_valuation_cannot_be_chain_linked():
    assert "error" in chain_link([("2026-01-01", 100.0)], [])


# ---------------------------------------------------------------------------
# The report's refusals
# ---------------------------------------------------------------------------
def _no_flows(monkeypatch, **overrides):
    """Patch the flow reader to a complete, empty window unless told otherwise."""
    import tools.attribution as attr

    payload = {"flows": [], "unclassified": [], "unpriced": [], "changes_seen": 0,
               "unclassified_count": 0, "unpriced_count": 0, "snapshots_in_window": 2,
               "blocked_by": None}
    payload.update(overrides)
    monkeypatch.setattr(attr, "window_flows", lambda *a, **k: payload)


def _no_positions(monkeypatch):
    import tools.attribution as attr

    monkeypatch.setattr(attr, "position_attribution",
                        lambda *a, **k: {"status": "no_data", "snapshots_in_window": 0})


def test_no_history_is_reported_as_absent_not_as_zero_return():
    res = get_attribution_report(window_days=30, as_of=date(2026, 7, 30), history=[])
    assert res["status"] == "no_history"
    assert "twr_pct" not in res


def test_a_holed_series_blocks_on_coverage_and_says_span_is_not_coverage(monkeypatch):
    _no_flows(monkeypatch)
    _no_positions(monkeypatch)
    sparse = _dense_series(date(2026, 7, 1), 30)[::3]      # every third day
    res = get_attribution_report(window_days=30, as_of=date(2026, 7, 30),
                                 history=sparse)
    assert res["status"] == "insufficient_coverage"
    assert "twr_pct" not in res
    assert "span_days" in res["note"]


def test_one_unclassified_change_withholds_the_whole_number(monkeypatch):
    """A TWR over three known deposits while a fourth sits unclassified is not
    slightly wrong; it is wrong in an unknown direction by an unknown amount."""
    _no_flows(monkeypatch, blocked_by="unclassified_changes", unclassified_count=1)
    _no_positions(monkeypatch)
    res = get_attribution_report(window_days=30, as_of=date(2026, 7, 30),
                                 history=_dense_series(date(2026, 7, 1), 30))
    assert res["status"] == "flows_incomplete"
    assert res["blocked_by"] == "unclassified_changes"
    assert "twr_pct" not in res


def test_a_classified_but_unpriced_flow_is_its_own_refusal(monkeypatch):
    """The second completeness axis. A window can be fully classified and still
    unusable, and `complete` alone would call it ready."""
    _no_flows(monkeypatch, blocked_by="unpriced_flows", unpriced_count=2)
    _no_positions(monkeypatch)
    res = get_attribution_report(window_days=30, as_of=date(2026, 7, 30),
                                 history=_dense_series(date(2026, 7, 1), 30))
    assert res["status"] == "flows_incomplete"
    assert res["blocked_by"] == "unpriced_flows"
    assert "amount_base" in res["note"]


def test_a_flow_on_an_unvalued_date_blocks_rather_than_chaining_over_it(monkeypatch):
    """TWR breaks the series AT a flow. A flow with no same-day valuation cannot
    be removed, and chain-linking across it leaves the deposit inside the return."""
    _no_flows(monkeypatch, flows=[{"date": "2026-07-15", "amount_base": 5000.0,
                                   "cause": "external_inflow", "symbol": "CASH",
                                   "account": "RRSP"}])
    _no_positions(monkeypatch)
    history = [(d, v) for d, v in _dense_series(date(2026, 7, 1), 30)
               if d != "2026-07-15"]
    res = get_attribution_report(window_days=30, as_of=date(2026, 7, 30),
                                 history=history)
    assert res["status"] == "flow_date_unvalued"
    assert res["unvalued_flow_dates"] == ["2026-07-15"]


def test_a_complete_window_produces_a_number_with_the_flow_removed(monkeypatch):
    _no_flows(monkeypatch, flows=[{"date": "2026-07-15", "amount_base": 10000.0,
                                   "cause": "external_inflow", "symbol": "CASH",
                                   "account": "RRSP"}])
    _no_positions(monkeypatch)

    history = _dense_series(date(2026, 7, 1), 30)
    # The deposit lands on the 15th and the book steps up by exactly it.
    history = [(d, v if d < "2026-07-15" else v + 10000.0) for d, v in history]

    res = get_attribution_report(window_days=30, as_of=date(2026, 7, 30),
                                 history=history,
                                 series_fn=lambda *a, **k: [],
                                 weights={"USD": 1.0})
    assert res["status"] == "measured"
    # Every day flat except the deposit day, and the deposit is removed: 0%.
    assert res["twr_pct"] == pytest.approx(0.0, abs=1e-6)
    assert res["net_external_flow"] == 10000.0
    # No benchmark could be priced, so there is no alpha — and it says so rather
    # than reporting alpha == twr.
    assert res["alpha_pct"] is None


def test_alpha_is_the_return_minus_the_blended_benchmark(monkeypatch):
    _no_flows(monkeypatch)
    _no_positions(monkeypatch)

    history = _dense_series(date(2026, 7, 1), 30)
    history = [(d, v * (1.10 if d >= "2026-07-15" else 1.0)) for d, v in history]

    def fake_series(symbol, start, end):
        # SPY up 5% across the window; XIC flat.
        end_px = 105.0 if symbol == "SPY" else 100.0
        return [("2026-07-01", 100.0), ("2026-07-30", end_px)]

    res = get_attribution_report(window_days=30, as_of=date(2026, 7, 30),
                                 history=history, series_fn=fake_series,
                                 weights={"USD": 1.0})
    assert res["status"] == "measured"
    assert res["twr_pct"] == pytest.approx(10.0, abs=1e-4)
    assert res["benchmark"]["return_pct"] == pytest.approx(5.0, abs=1e-4)
    assert res["alpha_pct"] == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# The benchmark's own honesty
# ---------------------------------------------------------------------------
def test_an_uncovered_currency_is_named_rather_than_renormalised_away():
    """Silently renormalising over the covered legs presents a two-currency
    benchmark as if it covered three."""
    res = blended_benchmark(date(2026, 1, 1), date(2026, 7, 30),
                            weights={"USD": 0.6, "CAD": 0.2, "EUR": 0.2},
                            series_fn=lambda s, a, b: [("2026-01-01", 100.0),
                                                       ("2026-07-30", 110.0)])
    assert res["status"] == "partial"
    assert res["uncovered_currencies"] == {"EUR": 0.2}
    assert res["covered_weight"] == pytest.approx(0.8)
    assert "EUR" in res["benchmark_note"]


def test_the_benchmark_states_that_its_legs_are_price_series():
    """Price legs vs a total-return portfolio biases alpha upward by the
    benchmark's yield. That is a systematic bias, not noise, and it must be on
    the payload."""
    res = blended_benchmark(date(2026, 1, 1), date(2026, 7, 30),
                            weights={"USD": 1.0},
                            series_fn=lambda s, a, b: [("2026-01-01", 100.0),
                                                       ("2026-07-30", 110.0)])
    assert "PRICE series" in res["benchmark_note"]
    assert res["return_pct"] == pytest.approx(10.0)


def test_an_unpriceable_benchmark_yields_no_legs_rather_than_a_zero_return():
    res = blended_benchmark(date(2026, 1, 1), date(2026, 7, 30),
                            weights={"USD": 1.0}, series_fn=lambda s, a, b: [])
    assert res["status"] == "no_legs"
    assert "return_pct" not in res


def test_currency_weights_say_when_they_are_count_weighted_not_value_weighted():
    """One large position and nine small ones weigh the same in a count-weighted
    mix. That is a different answer, not a rougher version of the right one."""
    res = currency_weights([{"symbol": "AAPL", "currency": "USD"},
                            {"symbol": "XIC.TO", "currency": "CAD"}])
    assert res["basis"] == "position count"
    assert res["weights"] == {"USD": 0.5, "CAD": 0.5}
    assert "COUNT-weighted" in res["note"]

    valued = currency_weights([{"symbol": "AAPL", "currency": "USD", "market_value": 900.0},
                               {"symbol": "XIC.TO", "currency": "CAD", "market_value": 100.0}])
    assert valued["basis"] == "market value"
    assert valued["weights"]["USD"] == pytest.approx(0.9)


def test_the_sync_error_sentinel_is_not_a_holding_in_the_currency_mix():
    res = currency_weights([{"symbol": "AAPL", "currency": "USD"},
                            {"_sync_errors": ["questrade down"]}])
    assert res["positions"] == 1


# ---------------------------------------------------------------------------
# Tracking error
# ---------------------------------------------------------------------------
def test_tracking_error_reports_how_many_periods_actually_aligned():
    """A tracking error over four of fifty-two weeks is not a small-sample
    version of the right answer, and the caller has to be able to see that."""
    periods = [{"from": "2026-01-01", "to": "2026-01-02", "return_pct": 1.0},
               {"from": "2026-01-02", "to": "2026-01-03", "return_pct": -1.0},
               {"from": "2026-01-03", "to": "2026-01-04", "return_pct": 2.0}]
    series = [("2026-01-01", 100.0), ("2026-01-02", 101.0), ("2026-01-03", 100.0)]
    res = tracking_error(periods, series)
    assert res["aligned"] == 2
    assert res["unaligned"] == 1
    assert res["tracking_error_pct"] is not None


def test_tracking_error_withholds_below_two_aligned_periods():
    res = tracking_error([{"from": "2026-01-01", "to": "2026-01-02", "return_pct": 1.0}],
                         [("2026-01-01", 100.0), ("2026-01-02", 101.0)])
    assert res["tracking_error_pct"] is None


# ---------------------------------------------------------------------------
# window_flows — the whole series, not just the latest pair
# ---------------------------------------------------------------------------
@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Isolated position and classification stores for one test."""
    import tools.portfolio_classification as pc
    import tools.portfolio_reconciliation as pr

    positions = tmp_path / "position_history.csv"
    monkeypatch.setattr(pr, "history_path", lambda: str(positions))
    monkeypatch.setattr(pc, "store_path", lambda: str(tmp_path / "classifications.jsonl"))
    return positions


def _snapshot(path, rows):
    import csv

    import tools.portfolio_reconciliation as pr

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pr._FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in pr._FIELDS})


def test_window_flows_walks_every_snapshot_pair_not_only_the_latest(stores):
    """`get_reconciliation` reports the most recent two snapshots, which is the
    right surface for "what moved yesterday" and useless for a year of return: a
    deposit three months ago is invisible to it."""
    from tools.attribution import window_flows

    _snapshot(stores, [
        {"date": "2026-07-01", "account": "RRSP", "symbol": "CASH", "shares": 1000},
        {"date": "2026-07-02", "account": "RRSP", "symbol": "CASH", "shares": 6000},
        {"date": "2026-07-03", "account": "RRSP", "symbol": "CASH", "shares": 6000},
    ])
    res = window_flows(date(2026, 7, 1), date(2026, 7, 30))
    # The 07-01 -> 07-02 change is only visible if the whole series is walked.
    assert res["changes_seen"] == 1
    assert res["blocked_by"] == "unclassified_changes"


def test_window_flows_is_complete_once_every_change_is_named_and_priced(stores):
    from tools.attribution import window_flows
    from tools.portfolio_classification import classify_change
    from tools.portfolio_reconciliation import detect_changes, read_history

    _snapshot(stores, [
        {"date": "2026-07-01", "account": "RRSP", "symbol": "CASH", "shares": 1000},
        {"date": "2026-07-02", "account": "RRSP", "symbol": "CASH", "shares": 6000},
    ])
    change = detect_changes("2026-07-01", "2026-07-02", read_history())[0]

    classify_change(change, "external_inflow")
    assert window_flows(date(2026, 7, 1), date(2026, 7, 30))["blocked_by"] == "unpriced_flows"

    classify_change(change, "external_inflow", amount_base=5000.0, base_currency="CAD")
    res = window_flows(date(2026, 7, 1), date(2026, 7, 30))
    assert res["blocked_by"] is None
    assert res["flows"] == [{"date": "2026-07-02", "amount_base": 5000.0,
                             "cause": "external_inflow", "symbol": "CASH",
                             "account": "RRSP"}]


def test_fewer_than_two_snapshots_is_named_as_no_position_history(stores):
    """An unobserved change is not an absent one, and the recorder being dead
    must not read as a clean window."""
    from tools.attribution import window_flows

    _snapshot(stores, [{"date": "2026-07-01", "account": "RRSP",
                        "symbol": "CASH", "shares": 1000}])
    res = window_flows(date(2026, 7, 1), date(2026, 7, 30))
    assert res["blocked_by"] == "no_position_history"
    assert "position_snapshot" in res["note"]
