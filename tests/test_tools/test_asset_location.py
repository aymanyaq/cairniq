"""
Tests for Asset Location Engine & Tax Efficiency Scoring (Theme 4.7).
"""

from unittest.mock import patch

import pytest

from tools.asset_location import (
    TAX_POLICY_VERSION,
    _evaluate_position_location,
    _get_asset_tax_characteristics,
    _is_us_ticker,
    analyze_asset_location,
    classify_account,
    classify_account_type,
)


def test_classify_account_type():
    """Verify account taxonomy classification."""
    assert classify_account_type("Questrade TFSA") == "TAX_FREE"
    assert classify_account_type("Roth IRA") == "TAX_FREE"
    assert classify_account_type("RRSP Account") == "TAX_DEFERRED"
    assert classify_account_type("401(k) Plan") == "TAX_DEFERRED"
    assert classify_account_type("Traditional IRA") == "TAX_DEFERRED"
    assert classify_account_type("Margin Account") == "TAXABLE"
    assert classify_account_type("Personal Non-Reg") == "TAXABLE"
    assert classify_account_type("") == "TAXABLE"
    assert classify_account_type("Random Vault") == "TAXABLE"


def test_classify_account_resolves_jurisdiction_from_the_shelter_name():
    """The shelter names the country; the tax class alone does not.

    This is the whole reason the withholding rule can be gated: a TFSA and a
    Roth are both TAX_FREE and are governed by different tax systems.
    """
    # `jurisdictions={}` pins the NAME path specifically: this test is about what
    # the shelter keyword resolves to, and passing None would let a stored
    # jurisdiction (4.7a) answer instead and quietly stop testing the table.
    def named(account):
        resolved = classify_account(account, jurisdictions={})
        return {k: resolved[k] for k in ("tax_class", "shelter", "jurisdiction")}

    assert named("Questrade TFSA") == {
        "tax_class": "TAX_FREE", "shelter": "TFSA", "jurisdiction": "CA",
    }
    assert named("Fidelity Roth IRA") == {
        "tax_class": "TAX_FREE", "shelter": "ROTH_IRA", "jurisdiction": "US",
    }
    assert named("Stocks & Shares ISA") == {
        "tax_class": "TAX_FREE", "shelter": "ISA", "jurisdiction": "UK",
    }
    # And the source is reported, so a caller can tell this apart from a country
    # the user actually stated — they are not equally good evidence.
    assert classify_account("Questrade TFSA", jurisdictions={})[
        "jurisdiction_source"] == "inferred_from_name"
    assert classify_account("RRSP Account")["jurisdiction"] == "CA"
    assert classify_account("Vanguard 401(k)")["jurisdiction"] == "US"
    assert classify_account("Traditional IRA")["jurisdiction"] == "US"
    assert classify_account("HL SIPP")["jurisdiction"] == "UK"


def test_classify_account_reports_unknown_jurisdiction_rather_than_guessing():
    """A shelter CLASS that names no country is reported, never defaulted."""
    generic = classify_account("Registered Account")
    assert generic["tax_class"] == "TAX_DEFERRED"
    assert generic["jurisdiction"] is None

    pension = classify_account("Company Pension")
    assert pension["tax_class"] == "TAX_DEFERRED"
    assert pension["jurisdiction"] is None


def test_shelter_rule_order_and_word_boundaries_are_load_bearing():
    """The generic keys are substrings of the specific ones, both ways.

    Every assertion here failed against the original keyword-substring matcher,
    and each is a wrong tax class rather than a cosmetic mislabel.
    """
    # "Non-Registered" contains "REGISTERED": a fully taxable account was being
    # classified as a tax-deferred shelter and scored as if it were sheltered.
    assert classify_account("Non-Registered Account")["tax_class"] == "TAXABLE"
    assert classify_account("Non Registered Investment")["tax_class"] == "TAXABLE"

    # "Roth 401(k)" is tax-FREE and must be settled before the 401(k) rule.
    roth_401k = classify_account("Roth 401(k)")
    assert roth_401k["tax_class"] == "TAX_FREE"
    assert roth_401k["jurisdiction"] == "US"

    # "LIRA" contains "IRA" — a Canadian locked-in account is not a US IRA.
    assert classify_account("LIRA Locked-In")["jurisdiction"] == "CA"

    # "VISA" contains "ISA" — word boundaries, not substrings.
    assert classify_account("Visa Rewards Cash")["tax_class"] == "TAXABLE"


def test_is_us_ticker():
    """Verify only confirmed listings trigger US withholding rules."""
    assert _is_us_ticker("AAPL", {"exchange": "NMS"}) is True
    assert _is_us_ticker("MSFT", {"fullExchangeName": "NasdaqGS"}) is True
    assert _is_us_ticker("TD.TO") is False
    assert _is_us_ticker("XIC.TO") is False
    assert _is_us_ticker("TD") is None
    assert _is_us_ticker("CASH") is False
    assert _is_us_ticker("USD") is False


def test_get_asset_tax_characteristics():
    """Verify asset class and tax profile extraction."""
    # Cash
    cash_tax = _get_asset_tax_characteristics("CASH", {})
    assert cash_tax["asset_type"] == "CASH"
    assert cash_tax["is_high_income"] is True

    # REIT
    reit_info = {"sector": "Real Estate", "industry": "REIT - Industrial", "dividendYield": 5.0}
    reit_tax = _get_asset_tax_characteristics("O", reit_info)
    assert reit_tax["asset_type"] == "REIT"
    assert reit_tax["is_high_income"] is True

    # Bond ETF
    bond_info = {"shortName": "Vanguard Total Bond Market ETF", "dividendYield": 3.8}
    bond_tax = _get_asset_tax_characteristics("BND", bond_info)
    assert bond_tax["asset_type"] == "BOND"
    assert bond_tax["is_high_income"] is True

    # US Equity
    eq_info = {"sector": "Technology", "dividendYield": 1.5, "exchange": "NMS"}
    eq_tax = _get_asset_tax_characteristics("AAPL", eq_info)
    assert eq_tax["asset_type"] == "EQUITY"
    assert eq_tax["is_us_listed"] is True


def test_evaluate_position_location_rules():
    """Verify tax leakage and scoring rules across placement scenarios."""
    # 1. US Dividend Stock in a Canadian TFSA (Withholding Drag)
    us_div_tax = {"asset_type": "EQUITY", "dividend_yield": 0.03, "is_us_listed": True, "is_high_income": True}
    eval_tfsa = _evaluate_position_location(
        "AAPL", "Questrade TFSA", "TAX_FREE", us_div_tax, 10000.0, jurisdiction="CA"
    )
    assert eval_tfsa["score"] < 100
    assert any("withholding" in issue for issue in eval_tfsa["issues"])

    # 2. US Dividend Stock in RRSP (Optimal)
    eval_rrsp = _evaluate_position_location(
        "AAPL", "RRSP Account", "TAX_DEFERRED", us_div_tax, 10000.0, jurisdiction="CA"
    )
    assert eval_rrsp["score"] == 100
    assert not eval_rrsp["issues"]

    # 3. High Yield Bond in Taxable Account (High Tax Drag)
    bond_tax = {"asset_type": "BOND", "dividend_yield": 0.05, "is_us_listed": True, "is_high_income": True}
    eval_taxable = _evaluate_position_location("BND", "Margin Account", "TAXABLE", bond_tax, 10000.0)
    assert eval_taxable["score"] <= 70
    assert any("taxable account" in issue for issue in eval_taxable["issues"])

    # 4. US REIT in a Canadian TFSA (stacking: withholding drag + opportunity cost)
    us_reit_tax = {"asset_type": "REIT", "dividend_yield": 0.05, "is_us_listed": True, "is_high_income": True}
    eval_reit_tfsa = _evaluate_position_location(
        "O", "Questrade TFSA", "TAX_FREE", us_reit_tax, 10000.0, jurisdiction="CA"
    )
    assert eval_reit_tfsa["score"] < 75  # Both -25 (withholding) and -10 (opportunity) = 65
    assert any("withholding" in issue for issue in eval_reit_tfsa["issues"])
    assert any("Opportunity cost" in issue for issue in eval_reit_tfsa["issues"])


def test_us_dividends_in_a_roth_are_not_charged_a_withholding_drag():
    """The defect this gating exists for.

    US-source dividends paid into a US person's Roth are a domestic payment
    into a domestic account: nothing is withheld. Applying the Canadian TFSA
    rule to it invents a 15% leak and then recommends a swap to fix it.
    """
    us_div_tax = {"asset_type": "EQUITY", "dividend_yield": 0.03, "is_us_listed": True, "is_high_income": True}
    eval_roth = _evaluate_position_location(
        "AAPL", "Fidelity Roth IRA", "TAX_FREE", us_div_tax, 10000.0, jurisdiction="US"
    )
    assert eval_roth["score"] == 100
    assert not any("withholding" in issue for issue in eval_roth["issues"])
    assert eval_roth["ideal_placement"] == ["TAX_FREE"]

    # The same holding in a Canadian TFSA IS charged — same rule, same code path,
    # opposite answer, and the jurisdiction is the only thing that differs.
    eval_tfsa = _evaluate_position_location(
        "AAPL", "Questrade TFSA", "TAX_FREE", us_div_tax, 10000.0, jurisdiction="CA"
    )
    assert eval_tfsa["score"] == 75
    assert any("withholding" in issue for issue in eval_tfsa["issues"])


def test_uk_isa_is_charged_but_pointed_at_its_own_remedy():
    """An ISA leaks like a TFSA, and the fix is a SIPP, not an RRSP."""
    us_div_tax = {"asset_type": "EQUITY", "dividend_yield": 0.04, "is_us_listed": True, "is_high_income": True}
    eval_isa = _evaluate_position_location(
        "AAPL", "Stocks & Shares ISA", "TAX_FREE", us_div_tax, 10000.0, jurisdiction="UK"
    )
    assert any("withholding" in issue for issue in eval_isa["issues"])
    joined = " ".join(eval_isa["issues"])
    assert "SIPP" in joined
    assert "RRSP" not in joined


def test_unknown_jurisdiction_skips_the_check_visibly_instead_of_scoring_it():
    """Fail closed and say so — a skipped check must not read as a clean one."""
    us_div_tax = {"asset_type": "EQUITY", "dividend_yield": 0.03, "is_us_listed": True, "is_high_income": True}
    result = _evaluate_position_location(
        "AAPL", "Registered Account", "TAX_FREE", us_div_tax, 10000.0, jurisdiction=None
    )
    # No penalty invented under a jurisdiction we cannot name...
    assert not any("withholding" in issue for issue in result["issues"])
    # ...and the omission is stated rather than silent.
    assert result["notes"]
    assert any("no tax jurisdiction is on file" in n for n in result["notes"])
    # ...and it names where to answer it. A skipped check whose remedy is not
    # stated is a dead end: 4.7a made the account's country enterable, and the
    # note is the only place a reader of this payload learns that.
    assert any("Context › Account Jurisdictions" in n for n in result["notes"])


@patch("tools.asset_location.load_portfolio")
@patch("tools.asset_location.yf.Ticker")
def test_analyze_asset_location_integration(mock_ticker, mock_load_portfolio):
    """Test full portfolio analysis integration."""
    mock_load_portfolio.return_value = [
        {"symbol": "AAPL", "account": "Questrade TFSA", "value_base": 10000.0, "shares": 50, "purchase_price": 200.0},
        {"symbol": "MSFT", "account": "RRSP Account", "value_base": 15000.0, "shares": 40, "purchase_price": 375.0},
        {"symbol": "CASH", "account": "Questrade TFSA", "value_base": 5000.0, "shares": 5000, "purchase_price": 1.0},
    ]

    class FakeTicker:
        def __init__(self, sym):
            self.info = {
                "sector": "Technology",
                "dividendYield": 2.0 if sym == "AAPL" else 0.8,
                "quoteType": "EQUITY",
            }

    mock_ticker.side_effect = lambda sym: FakeTicker(sym)

    res = analyze_asset_location()
    assert "overall_score" in res
    assert "rating" in res
    assert "account_breakdown" in res
    assert len(res["account_breakdown"]) == 2
    assert res["total_value_base"] == 30000.0

    # Coverage travels with the score so a consumer can tell "checked and clean"
    # from "never checked" (2.3 provenance; 3.8 gates on the policy version).
    assert res["tax_policy_version"] == TAX_POLICY_VERSION
    assert res["jurisdictions_covered"] == ["CA"]
    assert res["uncovered_accounts"] == []


@patch("tools.asset_location.load_portfolio")
@patch("tools.asset_location.yf.Ticker")
def test_analysis_reports_accounts_whose_jurisdiction_it_could_not_resolve(
    mock_ticker, mock_load_portfolio
):
    """An unresolved shelter is named in the payload, not absorbed by the score."""
    mock_load_portfolio.return_value = [
        {"symbol": "AAPL", "account": "Registered Account", "value_base": 10000.0},
    ]

    class FakeTicker:
        def __init__(self, sym):
            self.info = {"sector": "Technology", "dividendYield": 3.0,
                         "quoteType": "EQUITY", "exchange": "NMS"}

    mock_ticker.side_effect = lambda sym: FakeTicker(sym)

    res = analyze_asset_location()
    assert res["uncovered_accounts"] == ["Registered Account"]
    assert res["jurisdictions_covered"] == []


@patch("tools.asset_location.load_portfolio")
@patch("tools.asset_location.yf.Ticker")
def test_swaps_are_never_proposed_across_tax_jurisdictions(mock_ticker, mock_load_portfolio):
    """Moving a holding between tax systems is not an asset-location decision."""
    mock_load_portfolio.return_value = [
        # Leaking Canadian TFSA position...
        {"symbol": "AAPL", "account": "Questrade TFSA", "value_base": 10000.0},
        # ...and the only tax-deferred room available is in another country.
        {"symbol": "MSFT", "account": "Vanguard 401(k)", "value_base": 15000.0},
    ]

    class FakeTicker:
        def __init__(self, sym):
            self.info = {
                "sector": "Technology",
                "dividendYield": 3.0 if sym == "AAPL" else 0.4,
                "quoteType": "EQUITY",
                "exchange": "NMS",
            }

    mock_ticker.side_effect = lambda sym: FakeTicker(sym)

    res = analyze_asset_location()
    assert any("withholding" in leak["issue"] for leak in res["tax_leakages"])
    assert res["recommended_swaps"] == []
    assert res["jurisdictions_covered"] == ["CA", "US"]


# ---------------------------------------------------------------------------
# Dividend yield — two provider fields, two different units
# ---------------------------------------------------------------------------
# NOTE the fixtures ABOVE were restated on 2026-07-30 as part of this fix. Every
# one of them wrote `dividendYield` as a fraction (0.03 for 3%), which is what
# the code believed and not what the provider sends. The suite was green because
# the fixtures shared the bug — a mock written from the reader's assumption
# rather than from the source can only ever confirm it. This is the same shape as
# the stale-schema insider mock that kept 5.9's two-vocabulary defect green.
def test_the_percent_field_is_not_read_as_a_fraction():
    """Found on the first live read of the 4.7 panel, 2026-07-30.

    `dividendYield` is a PERCENT (AAPL: 0.32) and `trailingAnnualDividendYield`
    is a FRACTION (AAPL: 0.003058). The old normaliser was
    `if raw_yield > 1.0: /= 100`, which fires for a 4% yielder and CANNOT fire
    for a sub-1% one — so AAPL's 0.32% was read as 32%. High yielders were right
    and low yielders were wrong by 100x, which is why it survived until the
    numbers reached a screen.
    """
    from tools.asset_location import _dividend_yield_fraction

    aapl = {"dividendYield": 0.32, "trailingAnnualDividendYield": 0.003058104}
    assert _dividend_yield_fraction(aapl) == pytest.approx(0.003058104)


def test_the_percent_field_is_converted_when_it_is_the_only_one():
    from tools.asset_location import _dividend_yield_fraction

    assert _dividend_yield_fraction({"dividendYield": 4.2}) == pytest.approx(0.042)
    assert _dividend_yield_fraction({"dividendYield": 0.32}) == pytest.approx(0.0032)


def test_an_implausible_yield_is_unknown_rather_than_high_income():
    """The clamp is what survives the next units change. A 32% equity yield is a
    unit error, and asserting it is worse than asserting nothing — it would flag
    every dividend payer as tax drag, which is exactly what happened."""
    from tools.asset_location import _dividend_yield_fraction

    assert _dividend_yield_fraction({"trailingAnnualDividendYield": 0.32}) == 0.0
    assert _dividend_yield_fraction({"dividendYield": 3200.0}) == 0.0


def test_a_low_yielding_equity_is_no_longer_flagged_as_high_income():
    """The consequence the bug actually had. `is_high_income` triggers at 3%, so
    a 0.31% payer read as 31% was reported as tax drag in every taxable account."""
    from tools.asset_location import _get_asset_tax_characteristics

    chars = _get_asset_tax_characteristics(
        "AAPL", {"quoteType": "EQUITY", "longName": "Apple Inc.",
                 "dividendYield": 0.32, "trailingAnnualDividendYield": 0.003058104})
    assert chars["dividend_yield"] < 0.01
    assert chars["is_high_income"] is False


def test_a_genuine_high_yielder_is_still_flagged():
    from tools.asset_location import _get_asset_tax_characteristics

    chars = _get_asset_tax_characteristics(
        "ENB.TO", {"quoteType": "EQUITY", "longName": "Enbridge Inc.",
                   "dividendYield": 6.1, "trailingAnnualDividendYield": 0.061})
    assert chars["dividend_yield"] == pytest.approx(0.061)
    assert chars["is_high_income"] is True


def test_a_missing_yield_is_zero_not_a_crash():
    from tools.asset_location import _dividend_yield_fraction

    assert _dividend_yield_fraction({}) == 0.0
    assert _dividend_yield_fraction({"dividendYield": None}) == 0.0
    assert _dividend_yield_fraction({"dividendYield": "n/a"}) == 0.0
