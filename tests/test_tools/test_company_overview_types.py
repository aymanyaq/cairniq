"""One key, two providers, two TYPES: `get_company_overview`'s numeric fields.

**Every fixture below was captured from a live provider response on 2026-07-30 —
Alpha Vantage OVERVIEW payloads and `yf.Ticker(...).info` under yfinance 1.5.1 —
and none was written from what the reader expects.** This file follows the rule
`test_dividend_yield_units.py` documents, for the reason that file gives: a
fixture copied from the reader can only ever confirm the reader. The units defect
in that file stayed green for months because every existing mock wrote the field
in the units the code believed, so the mocks and the defect agreed with each other
and disagreed only with the provider.

That rule matters more than usual here, because the entire defect under test is a
TYPE that no hand-written mock would reproduce: Alpha Vantage sends every one of
the 55 values in an OVERVIEW payload as a string, including the ones that read as
numbers, and a mock author writing `"PERatio": 20.2` would encode the bug's
absence into the fixture. All 55 values in each captured payload are strings; the
payloads contain zero non-string values.

Nothing downstream does arithmetic on this dict today — `macro_analysis`,
`health_check` and `tool_registry` all hand it straight to the model — so these
tests pin a latent hazard shut rather than proving a live defect fixed.
"""

from unittest.mock import MagicMock, patch

import pytest

from tools.alpha_vantage import _av_number, get_company_overview

# --- Captured Alpha Vantage OVERVIEW payloads --------------------------------
# Trimmed to the fields `get_company_overview` reads. Values verbatim, including
# the quoting: the strings are the point.

# A complete profile — every numeric field populated. IBM is the payload the
# demo key serves, and the one to check the happy path against.
AV_IBM = {
    "Symbol": "IBM", "Name": "International Business Machines",
    "Sector": "TECHNOLOGY", "Industry": "COMPUTER & OFFICE EQUIPMENT",
    "Description": "International Business Machines Corporation provides "
                   "integrated solutions and services worldwide.",
    "MarketCapitalization": "213336916000", "PERatio": "20.2",
    "PEGRatio": "2.308", "EPS": "11.21", "DividendYield": "0.0296",
    "DividendPerShare": "6.73", "52WeekHigh": "332.46", "52WeekLow": "199.19",
    "50DayMovingAverage": "260.43", "200DayMovingAverage": "271.04",
    "Beta": "0.675", "AnalystTargetPrice": "245.33", "ForwardPE": "18.35",
    "ProfitMargin": "0.155",
}

# A non-payer. The "None" sentinel, in the two dividend fields — Alpha Vantage
# does not omit a missing number and does not send an empty string for it.
AV_TSLA = {
    "Symbol": "TSLA", "Name": "Tesla Inc", "Sector": "MANUFACTURING",
    "Industry": "MOTOR VEHICLES & PASSENGER CAR BODIES",
    "MarketCapitalization": "1178229015000", "PERatio": "286.85",
    "PEGRatio": "4.505", "EPS": "1.04", "DividendYield": "None",
    "DividendPerShare": "None", "52WeekHigh": "498.83", "52WeekLow": "297.38",
    "50DayMovingAverage": "394.64", "200DayMovingAverage": "412.75",
    "Beta": "1.802", "AnalystTargetPrice": "399.45", "ForwardPE": "158.73",
    "ProfitMargin": "0.0367",
}

# The most important payload in this file. A loss-making company carries BOTH
# meanings of a minus sign at once: ForwardPE is "-" (the missing-value sentinel)
# while EPS is "-2.92" and ProfitMargin "-0.636" (real numbers that are negative
# because the company lost money). Any coercion that strips "-" or tests
# startswith("-") nulls out a loss-maker's real fundamentals — and would look
# perfectly correct against a profitable fixture like IBM.
AV_RIVN = {
    "Symbol": "RIVN", "Name": "Rivian Automotive Inc", "Sector": "MANUFACTURING",
    "Industry": "MOTOR VEHICLES & PASSENGER CAR BODIES",
    "MarketCapitalization": "24156582000", "PERatio": "None",
    "PEGRatio": "None", "EPS": "-2.92", "DividendYield": "None",
    "DividendPerShare": "None", "52WeekHigh": "22.69", "52WeekLow": "11.57",
    "50DayMovingAverage": "16.27", "200DayMovingAverage": "16.03",
    "Beta": "1.602", "AnalystTargetPrice": "18.85", "ForwardPE": "-",
    "ProfitMargin": "-0.636",
}

# Second witness to the same pair of sentinels, so neither is a one-symbol fluke.
AV_LCID = {
    "Symbol": "LCID", "Name": "Lucid Group Inc", "Sector": "MANUFACTURING",
    "Industry": "MOTOR VEHICLES & PASSENGER CAR BODIES",
    "MarketCapitalization": "3083029000", "PERatio": "None", "PEGRatio": "None",
    "EPS": "-15.97", "DividendYield": "None", "DividendPerShare": "None",
    "52WeekHigh": "25.23", "52WeekLow": "2.37", "50DayMovingAverage": "5.92",
    "200DayMovingAverage": "10.18", "Beta": "0.831",
    "AnalystTargetPrice": "8.3", "ForwardPE": "-", "ProfitMargin": "-2.398",
}

# --- Captured yfinance `.info` payloads --------------------------------------
# The fallback branch's source. Note the types: real ints and floats for the same
# keys Alpha Vantage sends as strings. IBM is deliberately the same company as
# AV_IBM, captured the same day, so the two branches can be compared directly.

YF_IBM = {
    "quoteType": "EQUITY", "longName": "International Business Machines Corporation",
    "sector": "Technology", "industry": "Information Technology Services",
    "longBusinessSummary": "International Business Machines Corporation provides "
                           "integrated solutions and services worldwide.",
    "marketCap": 208852353024, "trailingPE": 19.704887, "pegRatio": 2.18,
    "trailingEps": 11.25, "dividendYield": 2.99, "dividendRate": 6.76,
    "fiftyTwoWeekHigh": 332.46, "fiftyTwoWeekLow": 199.19,
    "fiftyDayAverage": 260.4316, "twoHundredDayAverage": 271.0388,
    "beta": 0.675, "targetMeanPrice": 245.33043, "forwardPE": 16.825922,
    "profitMargins": 0.15522,
    # All three yield fields, because `dividend_yield_fraction` prefers the
    # trailing one and carrying only `dividendYield` would silently route this
    # fixture down the percent-conversion fallback that a real IBM payload never
    # takes. Note the units against AV_IBM's DividendYield "0.0296": the trailing
    # field agrees with it, and `dividendYield` 2.99 is the percent that did not.
    "trailingAnnualDividendYield": 0.029720897, "yield": None,
}

# The fallback's own missing-value shape: absent numbers arrive as None, not as a
# sentinel string and not as 0.0. That meaning is what the AV branch has to match.
YF_RIVN = {
    "quoteType": "EQUITY", "longName": "Rivian Automotive, Inc.",
    "sector": "Consumer Cyclical", "industry": "Auto Manufacturers",
    "longBusinessSummary": "Rivian Automotive, Inc. designs, develops, "
                           "manufactures, and sells electric vehicles.",
    "marketCap": 23606726656, "trailingPE": None, "pegRatio": None,
    "trailingEps": -2.92, "dividendYield": None, "dividendRate": None,
    "fiftyTwoWeekHigh": 22.69, "fiftyTwoWeekLow": 11.57,
    "fiftyDayAverage": 16.3211, "twoHundredDayAverage": 16.045574,
    "beta": 1.602, "targetMeanPrice": 18.84615, "forwardPE": -8.935965,
    "profitMargins": -0.63622004,
    # A real 0.0 in the trailing field for a company that pays nothing, which
    # `dividend_yield_fraction` returns as its "no number to report" 0.0 — the
    # fallback then emits None, the same answer the AV branch gives for "None".
    "trailingAnnualDividendYield": 0.0, "yield": None,
}

# The keys whose type is being unified. `market_cap` is an int on both paths;
# the rest are floats.
NUMERIC_KEYS = (
    "market_cap", "pe_ratio", "peg_ratio", "eps", "dividend_yield",
    "dividend_per_share", "52_week_high", "52_week_low", "50_day_ma",
    "200_day_ma", "beta", "analyst_target", "forward_pe", "profit_margin",
)
TEXT_KEYS = ("symbol", "name", "sector", "industry", "description")


def _overview_from_av(payload):
    """Drive the Alpha Vantage branch. `__wrapped__` skips the cache decorator."""
    with patch("tools.alpha_vantage._av_get", return_value=(payload, None)):
        return get_company_overview.__wrapped__(payload["Symbol"])


def _overview_from_yfinance(info):
    """Drive the fallback branch by failing the AV call the way a rate limit does."""
    ticker = MagicMock()
    ticker.info = info
    with patch("tools.alpha_vantage._av_get", return_value=(None, "Rate limit")):
        with patch("yfinance.Ticker", return_value=ticker):
            return get_company_overview.__wrapped__(info.get("longName", "X"))


# ---------------------------------------------------------------------------
# The captured payloads say what the providers actually send
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [AV_IBM, AV_TSLA, AV_RIVN, AV_LCID])
def test_alpha_vantage_sends_every_value_as_a_string(payload):
    # The premise of the whole file. If a future fixture is hand-written with a
    # real float in it, this fails and says so, rather than quietly weakening
    # every assertion below.
    assert all(isinstance(v, str) for v in payload.values())


# ---------------------------------------------------------------------------
# Both branches now agree on type
# ---------------------------------------------------------------------------

def test_av_branch_returns_numbers_not_strings():
    res = _overview_from_av(AV_IBM)
    for key in NUMERIC_KEYS:
        assert not isinstance(res[key], str), f"{key} came back as a string"
    assert res["pe_ratio"] == 20.2
    assert res["eps"] == 11.21
    assert res["beta"] == 0.675
    assert res["profit_margin"] == 0.155
    assert res["source"] == "alphavantage"


def test_both_branches_agree_on_type_for_every_numeric_key():
    # The actual point of the change: `source` is no longer the only thing that
    # tells a caller what it is holding.
    av = _overview_from_av(AV_IBM)
    yf = _overview_from_yfinance(YF_IBM)
    for key in NUMERIC_KEYS:
        assert type(av[key]) is type(yf[key]), (
            f"{key}: alphavantage {type(av[key]).__name__} "
            f"vs yfinance {type(yf[key]).__name__}"
        )


def test_market_cap_is_an_int_on_both_branches():
    assert _overview_from_av(AV_IBM)["market_cap"] == 213336916000
    assert isinstance(_overview_from_av(AV_IBM)["market_cap"], int)
    assert isinstance(_overview_from_yfinance(YF_IBM)["market_cap"], int)


def test_a_future_consumer_can_compare_without_knowing_the_provider():
    # The hazard in one line. `overview["pe_ratio"] > 15` raised TypeError on the
    # AV path and worked on the yfinance path; now it works on both.
    for res in (_overview_from_av(AV_IBM), _overview_from_yfinance(YF_IBM)):
        assert res["pe_ratio"] > 15
        assert res["market_cap"] > 1_000_000


def test_text_fields_are_left_as_text():
    res = _overview_from_av(AV_IBM)
    for key in TEXT_KEYS:
        assert isinstance(res[key], str), f"{key} should still be a string"
    assert res["symbol"] == "IBM"
    assert res["sector"] == "TECHNOLOGY"


# ---------------------------------------------------------------------------
# The sentinels
# ---------------------------------------------------------------------------

def test_the_None_sentinel_becomes_None_and_not_zero():
    # TSLA pays no dividend and Alpha Vantage says so with the literal "None".
    # 0.0 here would assert a measurement the provider never made; None is the
    # same answer the yfinance branch gives for the same absence.
    res = _overview_from_av(AV_TSLA)
    assert res["dividend_yield"] is None
    assert res["dividend_per_share"] is None
    assert res["pe_ratio"] == 286.85  # the rest of the payload still parses


def test_the_dash_sentinel_becomes_None():
    for payload in (AV_RIVN, AV_LCID):
        assert _overview_from_av(payload)["forward_pe"] is None


def test_missing_ratios_are_None_on_both_branches():
    av = _overview_from_av(AV_RIVN)
    yf = _overview_from_yfinance(YF_RIVN)
    for key in ("pe_ratio", "peg_ratio", "dividend_yield", "dividend_per_share"):
        assert av[key] is None and yf[key] is None, key


def test_a_missing_yield_never_becomes_zero():
    # Guarding the distinction explicitly, because 0.0 and None are both falsy and
    # a `or 0.0` anywhere on this path would erase it. A non-payer and a company
    # whose yield the provider does not carry are not the same claim.
    for payload in (AV_TSLA, AV_RIVN, AV_LCID):
        assert _overview_from_av(payload)["dividend_yield"] is None


# ---------------------------------------------------------------------------
# Negative numbers, which share a character with one of the sentinels
# ---------------------------------------------------------------------------

def test_real_negative_numbers_survive_in_a_payload_that_also_carries_the_dash():
    # RIVN carries "-" (missing ForwardPE) and "-2.92" (real EPS) at once. The
    # sentinel test has to be against the whole string; anything that strips the
    # minus sign or tests a prefix passes IBM and destroys this.
    res = _overview_from_av(AV_RIVN)
    assert res["eps"] == -2.92
    assert res["profit_margin"] == -0.636
    assert res["forward_pe"] is None

    lcid = _overview_from_av(AV_LCID)
    assert lcid["eps"] == -15.97
    assert lcid["profit_margin"] == -2.398
    assert lcid["forward_pe"] is None


# ---------------------------------------------------------------------------
# The coercion helper directly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("20.2", 20.2),            # the ordinary case
    ("213336916000", 213336916000.0),
    ("-2.92", -2.92),          # negative, not a sentinel
    ("-0.636", -0.636),
    ("0", 0.0),                # a real zero is a number, and must not become None
    ("0.0", 0.0),
    (" 20.2 ", 20.2),          # tolerated whitespace
    (20.2, 20.2),              # already a number
    (7, 7.0),
])
def test_av_number_parses_what_is_a_number(raw, expected):
    assert _av_number(raw) == expected


@pytest.mark.parametrize("raw", [
    "None",      # verified live: RIVN PERatio, TSLA DividendYield
    "-",         # verified live: RIVN and LCID ForwardPE
    "",
    "   ",
    None,        # key absent from the payload entirely
    "n/a",       # not observed live, but unparseable and so not a number
    "abc",
    "NaN",       # float() accepts this without raising; json.dumps then emits a
    "Infinity",  # bare NaN/Infinity token that strict parsers reject
    "-Infinity",
    float("nan"),
    float("inf"),
    True,        # bool is an int subclass; a flag is not a measurement
])
def test_av_number_returns_None_for_what_is_not_a_number(raw):
    assert _av_number(raw) is None


def test_av_number_never_confuses_zero_with_missing():
    # Both are falsy, and only one of them is a measurement.
    assert _av_number("0") == 0.0
    assert _av_number("0") is not None
    assert _av_number("None") is None
