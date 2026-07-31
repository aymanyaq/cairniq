"""Roadmap 4.7 — the loss-deferral policy engine.

Almost every test here asserts that the engine did NOT answer, and that is the
right shape for this item. The roadmap re-scoped 4.7 twice, and the second
re-scope was a correction: the US wash-sale rule and the Canadian
superficial-loss rule are not one mechanic with a country parameter. The tests
below pin the four places they actually diverge, because a parameter table would
pass a test on the 30-day window and get all four of those wrong.

The most important test in the file is
`test_an_uncovered_jurisdiction_blocks_and_never_falls_through`. 4.7 has already
shipped the opposite behaviour once: TFSA, Roth and ISA went into one `TAX_FREE`
bucket and every one of them was charged the Canadian TFSA treatment, which
invented a withholding leak on a US user's Roth and then recommended a swap to
fix it. A rule engine that silently picks the nearest jurisdiction produces a
confident answer about the wrong law.
"""

from datetime import date, timedelta

import pytest

from tools import tax_policy as tp


# ---------------------------------------------------------------------------
# The modules are genuinely different, not one table with a country column
# ---------------------------------------------------------------------------
def test_the_two_regimes_differ_on_all_four_axes_that_change_the_answer():
    us, ca = tp.POLICY_MODULES["US"], tp.POLICY_MODULES["CA"]

    # 1. The trigger. "Substantially identical" is broader than "identical property".
    assert us["identity_test"] == "substantially_identical"
    assert ca["identity_test"] == "identical_property"

    # 2. The account set. Different sets, not a renaming.
    assert set(us["affiliated_scope"]) != set(ca["affiliated_scope"])
    assert "controlled_corporation" in ca["affiliated_scope"]
    assert "controlled_corporation" not in us["affiliated_scope"]

    # 3. What the denied loss DOES.
    assert us["disallowed_loss_treatment"] == "added_to_basis_of_replacement"
    assert ca["disallowed_loss_treatment"] == "added_to_affiliated_acb"
    # The one case where the loss is destroyed rather than deferred.
    assert us["registered_plan_treatment"] == "permanently_disallowed"

    # 4. A test one regime has and the other has no analogue for.
    assert ca["still_owned_test"] is True
    assert us["still_owned_test"] is False


def test_every_module_carries_a_version_and_a_coverage_matrix():
    """A consumer must be able to refuse a version it does not know, and to see
    what a module does not claim to cover."""
    for jurisdiction, module in tp.POLICY_MODULES.items():
        assert module["version"], jurisdiction
        assert module["coverage"]["rules"], jurisdiction
        assert module["coverage"]["excluded"], jurisdiction
        assert module["authority"], jurisdiction


def test_no_module_is_advice_ready_until_a_professional_has_reviewed_it():
    """The roadmap names the review as a DELIVERABLE of this item, not as a
    disclaimer to append. `advice_ready` is the machine-readable form of that."""
    matrix = tp.coverage_matrix()
    assert matrix["advice_ready"] is False
    for entry in matrix["jurisdictions"].values():
        assert entry["advice_ready"] is False
        assert entry["professional_review"]["reviewed"] is False


# ---------------------------------------------------------------------------
# not_covered BLOCKS
# ---------------------------------------------------------------------------
def test_an_uncovered_jurisdiction_blocks_and_never_falls_through():
    """THE test for this item.

    A UK ISA is a jurisdiction this engine RECOGNISES and does not cover. It must
    say so and stop — not quietly apply the Canadian rules because CA happened to
    be in the dict.
    """
    res = tp.check_disposition(
        {"symbol": "VUSA.L", "date": "2026-07-01", "gain_loss": -500.0},
        [{"symbol": "VUSA.L", "date": "2026-07-10"}],
        jurisdiction="UK",
    )
    assert res["status"] == "not_covered"
    assert res["blocks"] is True
    assert res["policy_version"] is None
    assert "candidates" not in res
    assert "UK" in tp.KNOWN_UNCOVERED


def test_a_missing_jurisdiction_is_not_covered_rather_than_defaulted():
    res = tp.check_disposition({"symbol": "AAPL", "date": "2026-07-01",
                                "gain_loss": -100.0}, [], jurisdiction=None)
    assert res["status"] == "not_covered"
    assert res["blocks"] is True


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------
def test_a_gain_is_not_subject_to_the_rule():
    res = tp.check_disposition({"symbol": "AAPL", "date": "2026-07-01",
                                "gain_loss": 500.0},
                               [{"symbol": "AAPL", "date": "2026-07-05"}],
                               jurisdiction="US")
    assert res["status"] == "not_a_loss"
    assert res["blocks"] is False


def test_an_unknowable_result_blocks_rather_than_assuming_a_loss_either_way():
    """The reconciliation store records SHARES, so it cannot say whether a sale
    realised a loss. Assuming it did would manufacture wash-sale flags; assuming
    it did not would suppress real ones."""
    res = tp.check_disposition({"symbol": "AAPL", "date": "2026-07-01"},
                               [{"symbol": "AAPL", "date": "2026-07-05"}],
                               jurisdiction="US")
    assert res["status"] == "unknown_result"
    assert res["blocks"] is True


def test_a_loss_with_a_repurchase_inside_the_window_is_a_candidate_not_a_verdict():
    res = tp.check_disposition(
        {"symbol": "AAPL", "date": "2026-07-01", "proceeds": 900.0, "cost_basis": 1000.0},
        [{"symbol": "AAPL", "date": "2026-07-15", "account": "Taxable Brokerage"}],
        jurisdiction="US",
    )
    assert res["status"] == "candidate"
    assert res["blocks"] is True
    assert res["loss_amount"] == 100.0
    assert res["candidates"][0]["days_from_sale"] == 14
    assert res["candidates"][0]["side"] == "after"
    # It never says "this IS a wash sale".
    assert "tax professional" in res["determination_required_by"]


def test_the_window_reaches_backwards_as_well_as_forwards():
    """30 days BEFORE and after. An engine that only looks forward misses the
    buy-then-sell-the-older-lot case entirely."""
    res = tp.check_disposition(
        {"symbol": "AAPL", "date": "2026-07-01", "gain_loss": -100.0},
        [{"symbol": "AAPL", "date": "2026-06-20", "account": "Taxable"}],
        jurisdiction="US",
    )
    assert res["status"] == "candidate"
    assert res["candidates"][0]["side"] == "before"
    assert res["candidates"][0]["days_from_sale"] == -11


def test_an_acquisition_outside_the_window_is_not_a_candidate():
    res = tp.check_disposition(
        {"symbol": "AAPL", "date": "2026-07-01", "gain_loss": -100.0},
        [{"symbol": "AAPL", "date": "2026-09-01", "account": "Taxable"}],
        jurisdiction="US",
    )
    assert res["status"] == "no_candidates"
    assert res["blocks"] is False


def test_a_different_symbol_inside_the_window_is_surfaced_as_a_judgment_call():
    """Under §1091 a near-identical ETF may well count, and under the Canadian
    test it generally does not. Neither call belongs to this engine, so the row
    is shown with its uncertainty rather than dropped."""
    res = tp.check_disposition(
        {"symbol": "VOO", "date": "2026-07-01", "gain_loss": -100.0},
        [{"symbol": "IVV", "date": "2026-07-05", "account": "Taxable"}],
        jurisdiction="US",
    )
    assert res["status"] == "candidate"
    assert res["candidates"][0]["same_symbol"] is False
    assert "judgment call" in res["candidates"][0]["why"]


def test_a_repurchase_inside_a_registered_plan_reports_the_harsher_us_consequence():
    """Rev. Rul. 2008-5: the loss is destroyed, not deferred. It is the only case
    in either regime with no basis add-back anywhere, and reporting the generic
    'added to basis' consequence there would understate it completely."""
    res = tp.check_disposition(
        {"symbol": "AAPL", "date": "2026-07-01", "gain_loss": -1000.0},
        [{"symbol": "AAPL", "date": "2026-07-10", "account": "Roth IRA"}],
        jurisdiction="US",
    )
    assert res["candidates"][0]["in_registered_plan"] is True
    assert "PERMANENTLY DISALLOWED" in res["consequence_if_triggered"]


def test_the_canadian_still_owned_test_rides_on_the_payload():
    res = tp.check_disposition(
        {"symbol": "XIC.TO", "date": "2026-07-01", "gain_loss": -400.0},
        [{"symbol": "XIC.TO", "date": "2026-07-10", "account": "Non-Registered"}],
        jurisdiction="CA",
    )
    assert res["still_owned_test"] is True
    assert "no US counterpart" in res["still_owned_note"]
    assert "adjusted cost base" in res["consequence_if_triggered"]


def test_an_undated_disposition_is_refused_rather_than_dated_by_inference():
    res = tp.check_disposition({"symbol": "AAPL", "gain_loss": -100.0}, [],
                               jurisdiction="US")
    assert res["status"] == "no_date"
    assert res["blocks"] is True


# ---------------------------------------------------------------------------
# Jurisdiction resolution — from the account, never from the profile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("account,expected", [
    ("RRSP Questrade", "CA"),
    ("TFSA", "CA"),
    ("Roth IRA", "US"),
    ("Fidelity 401(k)", "US"),
    ("Stocks and Shares ISA", "UK"),
])
def test_a_shelter_name_names_its_country(account, expected):
    assert tp.resolve_jurisdiction(account)["jurisdiction"] == expected


@pytest.mark.parametrize("account", ["Registered", "Pension", "Main Brokerage",
                                     "Joint Account", ""])
def test_an_account_class_without_a_country_fails_closed(account):
    """`REGIONAL_LOCALE` is a DISPLAY locale. Nothing here may fall back to it,
    and one household can hold accounts in two countries anyway."""
    res = tp.resolve_jurisdiction(account)
    assert res["resolved"] is False
    assert res["jurisdiction"] is None


def test_a_recognised_but_uncovered_jurisdiction_resolves_and_is_marked_uncovered():
    """Two different failures with two different fixes: 'we cannot tell what
    country' vs 'we know the country and have no rules for it'."""
    res = tp.resolve_jurisdiction("Stocks and Shares ISA")
    assert res["resolved"] is True
    assert res["covered"] is False
    assert res["policy_version"] is None


# ---------------------------------------------------------------------------
# The store, and what it can prove
# ---------------------------------------------------------------------------
@pytest.fixture
def empty_stores(tmp_path, monkeypatch):
    import tools.portfolio_classification as pc
    import tools.portfolio_reconciliation as pr
    import tools.trade_journal as tj

    monkeypatch.setattr(pr, "history_path", lambda: str(tmp_path / "positions.csv"))
    monkeypatch.setattr(pc, "store_path", lambda: str(tmp_path / "classifications.jsonl"))
    monkeypatch.setattr(tj, "get_trade_history", lambda *a, **k: [])
    return tmp_path


def test_an_empty_record_is_no_data_not_no_dispositions(empty_stores):
    res = tp.scan_dispositions()
    assert res["status"] == "no_data"
    assert "statement about the RECORD" in res["note"]


def test_an_unclassified_decrease_is_never_offered_as_a_disposition(empty_stores, monkeypatch):
    """A share count going down is equally a sale, a transfer, a fee or a
    corporate action. A rule engine over inferred transactions is this project's
    most-repeated mistake."""
    import csv

    import tools.portfolio_reconciliation as pr

    today = date.today()
    with open(pr.history_path(), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pr._FIELDS)
        w.writeheader()
        for d, shares in (((today - timedelta(days=2)).isoformat(), 100),
                          ((today - timedelta(days=1)).isoformat(), 40)):
            w.writerow({"date": d, "account": "Non-Registered", "symbol": "XIC.TO",
                        "currency": "CAD", "shares": shares, "private": "", "as_of": ""})

    res = tp.scan_dispositions()
    assert res["status"] == "no_data"
    assert res["sources"]["reconciliation"]["unclassified_excluded"] == 1


def test_a_user_stated_trade_becomes_a_dated_disposition(empty_stores):
    import csv

    import tools.portfolio_reconciliation as pr
    from tools.portfolio_classification import classify_change

    today = date.today()
    prior, current = (today - timedelta(days=2)).isoformat(), (today - timedelta(days=1)).isoformat()
    with open(pr.history_path(), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pr._FIELDS)
        w.writeheader()
        for d, shares in ((prior, 100), (current, 40)):
            w.writerow({"date": d, "account": "Non-Registered", "symbol": "XIC.TO",
                        "currency": "CAD", "shares": shares, "private": "", "as_of": ""})

    change = pr.detect_changes(prior, current, pr.read_history())[0]
    classify_change(change, "trade")

    res = tp.scan_dispositions()
    assert res["status"] == "ready"
    assert res["dispositions"][0]["symbol"] == "XIC.TO"
    # And it still cannot say whether the sale realised a loss.
    assert res["dispositions"][0]["gain_loss"] is None


# ---------------------------------------------------------------------------
# The pre-trade gate (3.8 P2)
# ---------------------------------------------------------------------------
def test_an_empty_record_gives_a_WEAK_pass_that_says_it_is_one(empty_stores):
    """An empty store cannot clear a trade; it can only fail to object. A pass
    presented as a confirmation is the whole failure mode of this gate."""
    res = tp.precheck_rebuy("XIC.TO", "RRSP Questrade")
    assert res["allowed"] is True
    assert res["evidence_complete"] is False
    assert "failed to object" in res["note"]


def test_a_taxable_account_that_names_no_country_still_fails_closed(empty_stores):
    """"Non-Registered" is Canadian WORDING and not a Canadian fact. The shelter
    table deliberately gives it no jurisdiction, and this gate must not read the
    familiarity of the phrase as evidence of a country."""
    res = tp.precheck_rebuy("Non-Registered RBC", "Non-Registered RBC")
    assert res["allowed"] is False
    assert res["reason"] == "jurisdiction_unresolved"


def test_the_gate_fails_closed_on_an_account_it_cannot_place(empty_stores):
    res = tp.precheck_rebuy("AAPL", "Main Brokerage")
    assert res["allowed"] is False
    assert res["reason"] == "jurisdiction_unresolved"


def test_the_gate_blocks_on_an_uncovered_jurisdiction_rather_than_passing(empty_stores):
    res = tp.precheck_rebuy("VUSA.L", "Stocks and Shares ISA")
    assert res["allowed"] is False
    assert res["reason"] == "not_covered"
    assert res["blocks"] is True


def test_the_gate_blocks_a_rebuy_inside_the_window_of_a_recorded_sale(empty_stores, monkeypatch):
    sold_on = (date.today() - timedelta(days=5)).isoformat()
    monkeypatch.setattr(tp, "scan_dispositions", lambda *a, **k: {
        "status": "ready", "count": 1,
        "dispositions": [{"symbol": "XIC.TO", "date": sold_on,
                          "account": "Non-Registered", "gain_loss": -400.0,
                          "source": "trade_journal"}],
        "sources": {},
    })
    res = tp.precheck_rebuy("XIC.TO", "RRSP Questrade")
    assert res["allowed"] is False
    assert res["reason"] == "repurchase_window"
    assert res["evidence_complete"] is True
    assert res["jurisdiction"] == "CA"


def test_the_gate_stamps_the_policy_version_it_judged_against(empty_stores):
    res = tp.precheck_rebuy("XIC.TO", "TFSA")
    assert res["policy_version"] == tp.POLICY_MODULES["CA"]["version"]
    assert res["engine_version"] == tp.ENGINE_VERSION
    assert res["advice_ready"] is False
