"""The percent-vs-fraction split in yfinance's yield fields, and the ten readers of it.

**Every fixture below was captured from a live ``yf.Ticker(...).info`` on
2026-07-30 under yfinance 1.5.1, not written from what a reader expects.** That
distinction is the whole reason this file exists. The bug these tests pin
survived for months with a green suite because the existing fixtures all wrote
``dividendYield`` as a fraction — the units the code believed — so the mocks and
the defect agreed with each other and only disagreed with the provider. A fixture
copied from the reader can only ever confirm the reader.

The same shape has now bitten this codebase twice: the stale-schema insider mock
that kept the two-vocabulary defect green, and this. When a test's input comes
from the code under test rather than from the source, the test measures
self-consistency, not correctness.
"""

import pytest

from tools.yf_utils import (
    IMPLAUSIBLE_YIELD,
    dividend_yield_display,
    dividend_yield_fraction,
)

# --- Captured payloads. Real values, trimmed to the yield-bearing fields. -----
# `dividendYield` is a PERCENT. `trailingAnnualDividendYield` and `yield` are
# FRACTIONS. Note which fields are None per quote type: that is not incidental,
# it decides which branch of the reader a security takes.

# Equity, sub-1% yield. THE case the old `if raw > 1.0: /= 100` guard could not
# catch, because 0.32 never trips a >1.0 test.
AAPL = {"dividendYield": 0.32, "trailingAnnualDividendYield": 0.003058104,
        "yield": None, "dividendRate": 1.08, "currentPrice": 338.19,
        "quoteType": "EQUITY", "longName": "Apple Inc."}

# Equity, high yield. Correct under the OLD code too (5.0 > 1.0, so it was
# divided) — which is exactly why the defect stayed hidden.
ENB_TO = {"dividendYield": 5.0, "trailingAnnualDividendYield": 0.04869922,
          "yield": None, "dividendRate": 3.88, "currentPrice": 77.56,
          "quoteType": "EQUITY", "longName": "Enbridge Inc."}

# Bond ETF. `trailingAnnualDividendYield` is None and `yield` carries the
# fraction — funds have no trailing field at all.
BND = {"dividendYield": 3.95, "trailingAnnualDividendYield": None,
       "yield": 0.0395, "dividendRate": None, "currentPrice": None,
       "quoteType": "ETF", "shortName": "Vanguard Total Bond Market ETF"}

# Equity ETF, same shape as BND.
VOO = {"dividendYield": 1.07, "trailingAnnualDividendYield": None,
       "yield": 0.0107, "dividendRate": None, "currentPrice": None,
       "quoteType": "ETF", "shortName": "Vanguard S&P 500 ETF"}

# Growth ETF, sub-1% yield. The single most damning payload in this file, and the
# reason the fixtures are captured rather than invented: it is under 1.0 in the
# percent field, so the old guard left it alone and reported 0.39% as 39.00% —
# and it carries neither `dividendRate` nor `currentPrice`, so the one call site
# with a correct primary path could not use it and fell to the broken fallback.
VUG = {"dividendYield": 0.39, "trailingAnnualDividendYield": None,
       "yield": 0.0039, "dividendRate": None, "currentPrice": None,
       "quoteType": "ETF", "shortName": "Vanguard Growth ETF"}

# REIT, high yield.
REALTY_INCOME = {
    "dividendYield": 4.96, "trailingAnnualDividendYield": 0.049201276,
    "yield": None, "dividendRate": 3.25, "currentPrice": 65.54,
    "quoteType": "EQUITY", "sector": "Real Estate", "longName": "Realty Income",
}

# A genuine non-payer: the trailing field is a real 0.0 and the percent field is
# absent entirely.
BRK_B = {"dividendYield": None, "trailingAnnualDividendYield": 0.0,
         "yield": None, "dividendRate": None, "currentPrice": 509.16,
         "quoteType": "EQUITY", "longName": "Berkshire Hathaway Inc."}


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

def test_the_percent_field_is_not_read_as_a_fraction():
    """AAPL pays 0.31%, not 31%. The fraction field wins when both are present."""
    assert dividend_yield_fraction(AAPL) == pytest.approx(0.003058104)


def test_a_fund_reads_its_yield_from_the_field_a_fund_actually_has():
    """ETFs carry `yield` and NOT `trailingAnnualDividendYield`.

    If the reader only preferred the trailing field, every fund would fall
    through to the percent field — so this is the case that decides whether the
    second entry in the preference list is load-bearing. It is.
    """
    assert dividend_yield_fraction(BND) == pytest.approx(0.0395)
    assert dividend_yield_fraction(VOO) == pytest.approx(0.0107)
    # BND and VOO both happen to exceed 1.0 in the percent field, so the OLD
    # magnitude guard got them right by luck. VUG does not, and is the one that
    # separates a working reader from a lucky one.
    assert dividend_yield_fraction(VUG) == pytest.approx(0.0039)


def test_the_percent_field_is_converted_when_it_is_the_only_one():
    assert dividend_yield_fraction({"dividendYield": 4.2}) == pytest.approx(0.042)
    assert dividend_yield_fraction({"dividendYield": 0.32}) == pytest.approx(0.0032)


def test_a_high_yielder_survives_the_fix():
    """The half that was already right must stay right — the old guard handled
    these correctly, and a fix that only moved the error would be no fix."""
    assert dividend_yield_fraction(ENB_TO) == pytest.approx(0.04869922)
    assert dividend_yield_fraction(REALTY_INCOME) == pytest.approx(0.049201276)


def test_a_real_zero_is_not_mistaken_for_a_missing_field():
    """BRK-B's trailing field is a true 0.0, and 0.0 must not be read as a
    reason to fall through to the (absent) percent field and invent a number."""
    assert dividend_yield_fraction(BRK_B) == 0.0


def test_an_implausible_yield_is_unknown_rather_than_high_income():
    """The clamp is what survives the NEXT units change.

    A 32% equity yield is a unit error, and asserting it is worse than asserting
    nothing: it flags every dividend payer as tax drag, which is precisely what
    happened. If yfinance flips `dividendYield` back to a fraction, this clamp is
    what keeps the failure quiet instead of confident.
    """
    assert dividend_yield_fraction({"trailingAnnualDividendYield": 0.32}) == 0.0
    assert dividend_yield_fraction({"dividendYield": 3200.0}) == 0.0
    assert dividend_yield_fraction({"yield": IMPLAUSIBLE_YIELD + 0.01}) == 0.0
    # ...and the boundary itself is allowed through.
    assert dividend_yield_fraction({"yield": IMPLAUSIBLE_YIELD}) == IMPLAUSIBLE_YIELD


def test_a_missing_or_unparseable_yield_is_zero_not_a_crash():
    assert dividend_yield_fraction({}) == 0.0
    assert dividend_yield_fraction({"dividendYield": None}) == 0.0
    assert dividend_yield_fraction({"dividendYield": "n/a"}) == 0.0
    assert dividend_yield_fraction({"trailingAnnualDividendYield": "-"}) == 0.0


def test_display_labels_its_unit_and_withholds_a_number_it_does_not_have():
    """0.0 from the reader means "no number", covering missing, unparseable,
    genuinely zero and clamped alike. Rendering that as "0.00%" would assert a
    non-payer on the strength of a unit error, so it stays "N/A"."""
    assert dividend_yield_display(AAPL) == "0.31%"
    assert dividend_yield_display(ENB_TO) == "4.87%"
    assert dividend_yield_display(BND) == "3.95%"
    assert dividend_yield_display(BRK_B) == "N/A"
    assert dividend_yield_display({}) == "N/A"
    assert dividend_yield_display({"dividendYield": 3200.0}) == "N/A"


# ---------------------------------------------------------------------------
# The call sites. One test per confirmed symptom — the same misread surfaced
# differently in each, which is why fixing only the first would have left the
# others looking healthy.
# ---------------------------------------------------------------------------

def test_income_projection_does_not_overstate_by_100x(monkeypatch):
    """`tools/income_analytics.py` — a 100x OVERSTATEMENT of projected income.

    The fallback path (taken when dividend history is empty, which is normal for
    newly listed funds) computed `ttm_div = price * yield_info` against the
    percent field. A 0.32% payer at $338 projected $108/share of annual income
    instead of $1.08 — the single number this tool exists to produce.
    """
    import pandas as pd

    import tools.income_analytics as income

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, *a, **k):
            return pd.DataFrame({"Close": [338.19]})

        @property
        def dividends(self):
            return pd.Series(dtype=float)  # empty -> forces the info fallback

        @property
        def info(self):
            return AAPL

    monkeypatch.setattr(income.yf, "Ticker", Ticker)
    result = income.project_portfolio_income(["AAPL"], [100])

    assert result["details"][0]["metric_used"] == "Info Yield"
    assert result["details"][0]["yield_pct"] == "0.31%"
    # 100 shares * ~$1.03/share of trailing dividend, NOT 100 * $338 * 0.32.
    income_projected = float(
        result["details"][0]["annual_income_projected"].lstrip("$").replace(",", "")
    )
    assert income_projected == pytest.approx(103.4, abs=1.0)


def test_fee_income_analysis_does_not_silently_drop_the_yield(monkeypatch):
    """`tools/portfolio_analytics.py` — the opposite symptom, and invisible.

    Here a `if div_yield > 0.25: div_yield = 0.0` sanity cap caught the percent
    value and zeroed it. Nothing looked wrong: the yield was not overstated, it
    was ABSENT, and an absent yield is indistinguishable from a non-payer in
    `expected_yield` and `dividend_payers_count`. Every holding yielding more
    than 0.25% was dropped.
    """
    import tools.portfolio_analytics as pa

    # Deliberately not in the module's YIELD_OVERRIDES table — those eight
    # hand-patched tickers were the workaround for this very defect.
    monkeypatch.setattr(pa, "get_info_safe", lambda symbol: ENB_TO)

    result = pa.get_fee_income_analysis(["ENB.TO"], [1.0])

    assert result["expected_yield"] == pytest.approx(0.04869922, abs=1e-5)
    assert result["dividend_payers_count"] == 1


def test_a_dividend_analysis_falls_back_without_the_magnitude_guard(monkeypatch):
    """`tools/market_data.py::get_dividend_analysis` — the fallback IS the fund path.

    The primary path (`dividendRate / currentPrice`) was already correct, so this
    looked safe. But ETFs carry neither field — BND, VOO and VUG return None for
    both — so every fund fell to the `if yield_val > 1.0` branch, the same
    magnitude heuristic that cannot fire for a sub-1% payer.

    VUG is the intersection of both failures: a fund (no primary path) yielding
    under 1% (no magnitude rescue). It was reported at 39.00%.
    """
    import tools.market_data as md

    class Ticker:
        def __init__(self, symbol):
            self.info = VUG

    monkeypatch.setattr(md.yf, "Ticker", Ticker)
    out = md.get_dividend_analysis.invoke({"ticker": "VUG"})

    assert "**Yield:** 0.39%" in out
    assert "39.00%" not in out


def test_a_side_by_side_comparison_uses_one_unit_for_both_names(monkeypatch):
    """`tools/compare_assets.py` — the worst shape of this bug.

    AAPL carries the fraction field; a name that carries only the percent field
    fell through to it. So a single table could print one real yield beside one
    100x yield and invite a direct comparison between them — which is the entire
    purpose of the tool.
    """
    import tools.compare_assets as ca

    percent_only = {"dividendYield": 0.31, "currentPrice": 50.0}

    class Ticker:
        def __init__(self, symbol):
            self.info = AAPL if symbol == "AAPL" else percent_only

    monkeypatch.setattr(ca.yf, "Ticker", Ticker)
    rows = ca.compare_assets.__wrapped__(["AAPL", "PEER"])["comparison"]

    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["AAPL"]["dividend_yield"] == "0.31%"
    assert by_symbol["PEER"]["dividend_yield"] == "0.31%"


@pytest.mark.parametrize(
    "module_name, symbol",
    [
        ("tools.canadian_market", "ENB.TO"),
        ("tools.australian_market", "BHP.AX"),
        ("tools.european_market", "SHEL.L"),
    ],
)
def test_regional_quotes_label_the_unit_they_emit(monkeypatch, module_name, symbol):
    """`tools/{canadian,australian,european}_market.py` — a bare
    `"dividend_yield": 0.32` names no unit, and the model reading this payload
    has no way to tell 0.32% from 32%. The value now carries its own unit."""
    import importlib

    module = importlib.import_module(module_name)

    class Ticker:
        def __init__(self, sym):
            self.info = ENB_TO

    monkeypatch.setattr(module.yf, "Ticker", Ticker)
    quote = module.__dict__[f"get_{module_name.split('.')[1].split('_')[0]}_quote"]
    result = quote.__wrapped__(symbol)

    assert result["dividend_yield"] == "4.87%"


def test_both_company_overview_providers_agree_on_units(monkeypatch):
    """`tools/alpha_vantage.py` — one key, two providers, two units.

    `get_company_overview` returns Alpha Vantage's `DividendYield` when the API
    answers and falls back to yfinance when it does not. AV sends a FRACTION
    (verified live: IBM "0.0296", consistent with its $6.73 per share); yfinance
    sends a PERCENT. So the units of `dividend_yield` changed with the provider,
    and `source` was the only field that said which one you got.
    """
    import tools.alpha_vantage as av

    class Ticker:
        def __init__(self, symbol):
            self.info = AAPL if symbol == "AAPL" else BRK_B

    # This module imports yfinance inside the function, so there is no module
    # attribute to patch — the source module is the seam.
    monkeypatch.setattr("yfinance.Ticker", Ticker)

    # The yfinance fallback now emits the same unit AV does — a fraction, in the
    # same range as AV's own IBM value of 0.0296.
    assert av._company_overview_from_yfinance("AAPL")["dividend_yield"] == pytest.approx(
        0.003058104
    )

    # ...and where there is no number, it stays None rather than 0.0. This
    # payload carries the raw value, so a 0.0 here would tell the model "pays no
    # dividend" on the strength of what may have been a unit error. The display
    # sites express the same rule as "N/A".
    assert av._company_overview_from_yfinance("BRK-B")["dividend_yield"] is None
