"""Roadmap 4.8 — duration, convexity, and the states a bare "0 bonds" would hide.

Two groups of tests here and they guard different things.

The first group is arithmetic: a par bond prices at par, a zero's Macaulay
duration IS its maturity, convexity is positive, and the second-order estimate
beats the first-order one. These are closed-form facts and they pin the formulas
against a typo that would otherwise produce a plausible wrong number — the worst
possible failure mode for an analytics module, because nothing downstream can
tell a wrong duration from a right one.

The second group is the refusals, and they are the ones that matter for the live
book. The portfolio holds 27 public and 2 private positions and no bonds, so the
only figure this module will report for a long time is an absence — and this
codebase has repeatedly shipped an absence that reads as a measurement. The tests
below keep `no_fixed_income` (measured) and `undetermined` (some holding could not
be read) from ever collapsing into each other.
"""

import pytest

from tools.bond_analytics import (
    bond_metrics,
    classify_fixed_income,
    ladder_rate_sensitivity,
    portfolio_rate_sensitivity,
    rate_hike_duration_leg,
    sensitivity_over,
    shock_table,
)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def test_par_bond_prices_at_par():
    """Coupon == yield must give price == face, at any tenor or frequency.

    The single cheapest check that the discounting is right: it fails on an
    off-by-one in the period count, a misplaced frequency division, or a coupon
    applied annually while discounting semi-annually.
    """
    for years in (1.0, 5.0, 10.0, 30.0):
        for freq in (1, 2, 4):
            m = bond_metrics(coupon_rate=0.045, ytm=0.045, years=years,
                             face=100.0, frequency=freq)
            assert m["price"] == pytest.approx(100.0, abs=1e-6), (years, freq)


def test_zero_coupon_macaulay_duration_equals_its_maturity():
    """The textbook identity, and the reason it is worth a test: it is the one
    duration figure with a known answer that does not depend on the yield."""
    m = bond_metrics(coupon_rate=0.0, ytm=0.05, years=7.0, frequency=0)
    assert m["macaulay_duration"] == pytest.approx(7.0, abs=1e-6)
    assert m["modified_duration"] == pytest.approx(7.0 / 1.05, abs=1e-6)
    assert m["zero_coupon"] is True


def test_modified_duration_is_below_macaulay_and_convexity_is_positive():
    m = bond_metrics(coupon_rate=0.05, ytm=0.04, years=10.0, frequency=2)
    assert 0 < m["modified_duration"] < m["macaulay_duration"]
    assert m["convexity"] > 0
    assert m["dv01"] > 0


def test_a_coupon_bond_has_shorter_duration_than_a_zero_of_the_same_maturity():
    """Coupons pull the weighted average of the cashflow times forward."""
    coupon = bond_metrics(coupon_rate=0.06, ytm=0.05, years=10.0, frequency=2)
    zero = bond_metrics(coupon_rate=0.0, ytm=0.05, years=10.0, frequency=0)
    assert coupon["macaulay_duration"] < zero["macaulay_duration"]


def test_bond_metrics_refuses_a_matured_or_unreadable_instrument():
    assert "error" in bond_metrics(0.04, 0.04, years=0)
    assert "error" in bond_metrics(0.04, 0.04, years=-1)
    assert "error" in bond_metrics(0.04, 0.04, years=5, face=0)
    assert "error" in bond_metrics("x", 0.04, years=5)


# ---------------------------------------------------------------------------
# The shock table
# ---------------------------------------------------------------------------
def test_convexity_makes_the_gain_bigger_than_the_loss():
    """The asymmetry IS the convexity, and it is the whole reason for the table.

    A -100bp move must gain MORE than a +100bp move loses. A table built on
    duration alone reports these as equal and opposite, which is exactly the
    understatement this item exists to remove.
    """
    table = shock_table(coupon_rate=0.04, ytm=0.04, years=20.0, frequency=2)
    rows = {r["shock_bp"]: r for r in table["shocks"]}

    gain = rows[-100]["exact_pct"]
    loss = rows[100]["exact_pct"]
    assert gain > 0 > loss
    assert gain > abs(loss)

    # And duration alone would have called them equal.
    assert rows[-100]["duration_only_pct"] == pytest.approx(-rows[100]["duration_only_pct"],
                                                            abs=1e-9)


def test_adding_convexity_gets_closer_to_the_exact_reprice():
    table = shock_table(coupon_rate=0.04, ytm=0.04, years=20.0, frequency=2)
    for row in table["shocks"]:
        exact = row["exact_pct"]
        first_order_err = abs(exact - row["duration_only_pct"])
        second_order_err = abs(exact - row["duration_convexity_pct"])
        assert second_order_err <= first_order_err + 1e-9, row["shock_bp"]


def test_the_approximation_error_is_reported_and_grows_with_the_shock():
    """A second-order estimate is still an estimate. At 25bp the residual is
    noise; at 200bp it is visible, and a table that hid it would be most
    misleading at exactly the shock size a user cares about."""
    table = shock_table(coupon_rate=0.04, ytm=0.04, years=20.0, frequency=2)
    rows = {r["shock_bp"]: r for r in table["shocks"]}
    assert abs(rows[200]["approximation_error_pct"]) > abs(rows[25]["approximation_error_pct"])


def test_a_non_marketable_instrument_says_its_shocks_are_not_realisable():
    """A GIC ladder must never report a paper loss the holder cannot take."""
    table = shock_table(coupon_rate=0.04, ytm=0.04, years=5.0, frequency=0,
                        marked_to_market=False)
    assert table["marked_to_market"] is False
    assert "NO SECONDARY MARKET" in table["basis_note"]

    marketable = shock_table(coupon_rate=0.04, ytm=0.04, years=5.0, frequency=2)
    assert marketable["marked_to_market"] is True
    assert "PARALLEL" in marketable["basis_note"]


# ---------------------------------------------------------------------------
# Classification — three answers, and None is one of them
# ---------------------------------------------------------------------------
def test_a_known_bond_fund_is_classified_without_any_metadata():
    assert classify_fixed_income("AGG")["is_bond"] is True
    assert classify_fixed_income("xbb.to")["is_bond"] is True


def test_an_etf_with_no_category_is_undetermined_rather_than_equity():
    """The case that must not default. Bond ETFs and equity ETFs share a
    quoteType, so `ETF` alone cannot answer the question — and answering it
    `False` is how a book with bonds in it reports zero rate exposure."""
    verdict = classify_fixed_income("XYZ", {"quoteType": "ETF"})
    assert verdict["is_bond"] is None


def test_an_equity_is_classified_false_and_an_unknown_symbol_is_none():
    assert classify_fixed_income("AAPL", {"quoteType": "EQUITY"})["is_bond"] is False
    assert classify_fixed_income("WHAT")["is_bond"] is None


def test_a_bond_category_is_matched_on_tokens_not_a_substring_sweep():
    """4.7 shipped `"ISA" in "Visa"`. The same class of bug here would read
    "Bondholder Communications" as a bond fund."""
    assert classify_fixed_income("X", {"quoteType": "ETF",
                                       "category": "Intermediate Core Bond"})["is_bond"] is True
    assert classify_fixed_income("Y", {"quoteType": "ETF",
                                       "category": "Large Blend"})["is_bond"] is False


# ---------------------------------------------------------------------------
# The portfolio states — the ones the live book will actually hit
# ---------------------------------------------------------------------------
def test_all_holdings_classified_and_none_a_bond_is_a_measured_zero():
    rows = [{"symbol": "AAPL", "info": {"quoteType": "EQUITY"}},
            {"symbol": "MSFT", "info": {"quoteType": "EQUITY"}}]
    res = sensitivity_over(rows)
    assert res["status"] == "no_fixed_income"
    assert res["unclassified_holdings"] == 0
    assert res["positions_read"] == 2


def test_one_unreadable_holding_downgrades_the_zero_to_undetermined():
    """THE test for this module. One symbol nobody could classify means the book
    cannot be called bond-free, and the difference between these two statuses is
    the difference between a measurement and a silence."""
    rows = [{"symbol": "AAPL", "info": {"quoteType": "EQUITY"}},
            {"symbol": "MYSTERY", "info": {"quoteType": "ETF"}}]
    res = sensitivity_over(rows)
    assert res["status"] == "undetermined"
    assert res["bond_holdings"] == 0
    assert res["unclassified_holdings"] == 1
    assert "NOT a measured zero" in res["note"]
    assert res["unclassified"][0]["symbol"] == "MYSTERY"


def test_bonds_found_with_no_yield_on_file_withhold_the_duration():
    rows = [{"symbol": "AGG", "shares": 100}]
    res = sensitivity_over(rows)
    assert res["status"] == "yields_missing"
    assert "modified_duration" not in res
    assert res["missing_inputs"] == ["AGG"]


def test_supplying_yields_produces_a_measured_sleeve_duration():
    rows = [{"symbol": "AGG", "shares": 100}, {"symbol": "AAPL",
                                               "info": {"quoteType": "EQUITY"}}]
    res = sensitivity_over(rows, yields={"AGG": 0.045}, maturities={"AGG": 8.0})
    assert res["status"] == "measured"
    assert res["bond_holdings"] == 1
    assert res["non_bond_holdings"] == 1
    assert res["modified_duration"] > 0
    assert res["convexity"] > 0
    # The sleeve, never the whole book.
    assert "FIXED-INCOME SLEEVE only" in res["note"]
    down = next(s for s in res["shocks"] if s["shock_bp"] == -100)
    assert down["estimated_pct"] > 0


def test_an_unreadable_portfolio_is_not_a_bond_free_portfolio(monkeypatch):
    import tools.portfolio_csv as pcsv

    monkeypatch.setattr(pcsv, "load_portfolio", lambda *a, **k: {"error": "no csv"})
    res = portfolio_rate_sensitivity()
    assert res["status"] == "no_portfolio"
    assert "not a report that" in res["note"]


def test_the_sync_error_sentinel_is_not_counted_as_a_position(monkeypatch):
    """`load_portfolio` appends `{"_sync_errors": [...]}` to the holdings list.
    Counting it adds one phantom unreadable row on every single call, which is
    how a REAL unreadable row becomes invisible."""
    import tools.portfolio_csv as pcsv

    monkeypatch.setattr(pcsv, "load_portfolio", lambda *a, **k: [
        {"symbol": "AAPL", "shares": 10},
        {"_sync_errors": ["questrade down"]},
    ])
    res = portfolio_rate_sensitivity()
    assert res["positions_read"] == 1


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------
def _fake_rates(_inv_type, _currency):
    return {1: 4.4, 2: 4.2, 3: 4.1, 4: 4.05, 5: 4.0}, "fixture rates"


def test_gic_ladder_duration_is_between_its_shortest_and_longest_rung(monkeypatch):
    import tools.fixed_income as fi

    monkeypatch.setattr(fi, "_fetch_current_rates", _fake_rates)
    res = ladder_rate_sensitivity(100000, "GIC", "CAD")
    assert res["rungs_priced"] == 5
    assert 1.0 < res["modified_duration"] < 5.0
    assert res["total_value"] == pytest.approx(100000.0, rel=1e-9)


def test_a_gic_ladder_states_that_its_marks_are_not_realisable(monkeypatch):
    import tools.fixed_income as fi

    monkeypatch.setattr(fi, "_fetch_current_rates", _fake_rates)
    gic = ladder_rate_sensitivity(50000, "GIC", "CAD")
    assert gic["marked_to_market"] is False
    assert "no secondary market" in gic["note"]

    treasury = ladder_rate_sensitivity(50000, "Treasury", "USD")
    assert treasury["marked_to_market"] is True


def test_the_ladder_states_the_par_assumption_it_rests_on(monkeypatch):
    """Every rung is priced as if bought today at today's rate. A rung bought
    last year has a different duration, and the payload has to say so rather than
    let the reader assume it modelled their actual ladder."""
    import tools.fixed_income as fi

    monkeypatch.setattr(fi, "_fetch_current_rates", _fake_rates)
    res = ladder_rate_sensitivity(100000, "GIC", "CAD")
    assert "priced at par" in res["assumption"]


# ---------------------------------------------------------------------------
# The scenario leg
# ---------------------------------------------------------------------------
def test_the_rate_leg_declines_rather_than_reporting_zero_impact():
    """An absent rate leg reads as "rates cost this book nothing". On a book with
    no bonds the honest answer is `applicable: False` plus the reason."""
    leg = rate_hike_duration_leg(rows=[{"symbol": "AAPL",
                                        "info": {"quoteType": "EQUITY"}}])
    assert leg["applicable"] is False
    assert leg["status"] == "no_fixed_income"
    assert leg["reason"]
    assert "estimated_pct" not in leg


def test_simulate_scenario_carries_a_rate_leg_only_for_rate_hike():
    """The leg belongs to the rate scenario and nowhere else. Attaching a
    duration figure to `tech_crash` would imply the equity constant there had a
    measured component, which it does not."""
    from tools.simulation import simulate_scenario

    hike = simulate_scenario("AAPL", "rate_hike")
    assert "rate_leg" in hike
    # The equity leg keeps its own, weaker, stamp — the two halves of this
    # payload do not have the same standing and must not read as if they do.
    assert hike["basis"] == "authored constant"
    assert hike["rate_leg"]["basis"] == "computed"

    assert "rate_leg" not in simulate_scenario("AAPL", "recession")


def test_the_rate_leg_computes_where_the_equity_leg_only_asserts():
    leg = rate_hike_duration_leg(
        shock_bp=100,
        rows=[{"symbol": "TLT", "shares": 100}],
        yields={"TLT": 0.045}, maturities={"TLT": 20.0},
    )
    assert leg["applicable"] is True
    assert leg["basis"] == "computed"
    assert leg["estimated_pct"] < 0          # a hike hurts a long bond
    assert "PARALLEL" in leg["basis_note"]
