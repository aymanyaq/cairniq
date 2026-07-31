from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tools.insider_data import (
    _classify_insider_transaction,
    classify_insider_text,
    get_detailed_insider_activity,
    get_insider_and_short_data,
    is_canadian_listing,
)

# The live yfinance schema, captured from real responses. The previous fixture
# used "Insider Trading" for the name and put the description in "Transaction";
# yfinance now names them "Insider" and "Text" and leaves "Transaction" BLANK.
# Mocking the stale schema is why the suite stayed green while every production
# row came back insider="Unknown", type="UNKNOWN".
LIVE_COLUMNS = ["Shares", "Value", "URL", "Text", "Insider", "Position", "Transaction",
                "Start Date", "Ownership"]


def _frame(rows):
    """Build an insider table in the live yfinance column shape."""
    return pd.DataFrame(
        [dict(zip(LIVE_COLUMNS, r)) for r in rows],
        columns=LIVE_COLUMNS,
    )


def _today(offset_days=0):
    from datetime import date, timedelta
    return (date.today() - timedelta(days=offset_days)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Classification — US vocabulary (unchanged behaviour)
# ---------------------------------------------------------------------------
def test_classify_purchase_and_buy():
    assert _classify_insider_transaction("Purchase at price 12.34 per share.") == "BUY"
    assert _classify_insider_transaction("Open Market Buy") == "BUY"


def test_classify_sale_and_sell():
    assert _classify_insider_transaction("Sale at price 12.34 - 12.66 per share.") == "SELL"
    assert _classify_insider_transaction("Automatic Sell") == "SELL"


def test_classify_ambiguous_is_unknown_not_buy():
    assert _classify_insider_transaction("Stock Gift") == "UNKNOWN"
    assert _classify_insider_transaction("Option Exercise") == "UNKNOWN"
    assert _classify_insider_transaction("") == "UNKNOWN"
    assert _classify_insider_transaction(None) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Classification — Canadian/SEDI vocabulary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,signal", [
    ("Acquisition in the public market at price 40.00 per share.", "BUY"),
    ("Disposition in the public market at price 200.00 per share.", "SELL"),
    ("Exercise of options at price 75.00 per share.", "COMP"),
    ("Exercise of rights", "COMP"),
    ("Exercise for cash", "COMP"),
    ("Grant of rights", "COMP"),
    ("Compensation for services at price 40.00 per share.", "COMP"),
    ("Disposition by gift", "GIFT"),
])
def test_sedi_vocabulary_is_classified(text, signal):
    assert classify_insider_text(text)["signal"] == signal


def test_issuer_buyback_is_not_an_insider_buy():
    """'Redemption, retraction, cancelation, repurchase' is the MOST common row on
    a TSX table and contains the substring 'repurchase' — a keyword match reads
    the company buying back its own stock as insiders buying."""
    coding = classify_insider_text("Redemption, retraction, cancelation, repurchase at price 200.00 per share.")
    assert coding["signal"] == "BUYBACK"
    assert _classify_insider_transaction("Redemption, retraction, cancelation, repurchase") == "UNKNOWN"


def test_ownership_plan_disposition_is_not_a_buy():
    """'Disposition under a purchase/ownership plan' is a SALE whose text contains
    'purchase' and no 'sale'/'sell' — the keyword classifier called it a BUY."""
    assert classify_insider_text("Disposition under a purchase/ownership plan")["signal"] == "PLAN"
    assert _classify_insider_transaction("Disposition under a purchase/ownership plan") == "UNKNOWN"


def test_ownership_plan_acquisition_is_not_conviction():
    assert classify_insider_text("Acquisition under a purchase/ownership plan")["signal"] == "PLAN"


def test_is_canadian_listing():
    assert is_canadian_listing("NGF.TO")
    assert is_canadian_listing("abc.v")
    assert is_canadian_listing("XYZ.CN")
    assert not is_canadian_listing("AAPL")
    assert not is_canadian_listing("BRK.B")


# ---------------------------------------------------------------------------
# get_insider_and_short_data
# ---------------------------------------------------------------------------
@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_live_schema_reads_names_and_types(mock_ticker_class, _g, _s):
    """Regression: with the live column names, insider/type must be real values."""
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {"shortRatio": 2.0, "shortPercentOfFloat": 0.05,
                        "heldPercentInsiders": 0.10, "heldPercentInstitutions": 0.60}
    mock_ticker.insider_transactions = _frame([
        (10000, 450000.0, "", "Sale at price 45.00 per share.", "Jane Doe", "Officer", "", _today(3), "D"),
        (5000, 220000.0, "", "Purchase at price 44.00 per share.", "John Roe", "Director", "", _today(4), "D"),
        (2000, 90000.0, "", "Stock Gift", "Ann Poe", "Director", "", _today(5), "D"),
    ])

    result = get_insider_and_short_data("TESTX")
    activity = result["recent_insider_activity"]

    assert [t["type"] for t in activity] == ["SELL", "BUY", "UNKNOWN"]
    # The name column must actually be read — "Unknown" here means the reader
    # is looking at a column that no longer exists.
    assert [t["insider"] for t in activity] == ["Jane Doe", "John Roe", "Ann Poe"]


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_canadian_buybacks_do_not_create_a_buy_signal(mock_ticker_class, _g, _s):
    """A TSX table dominated by issuer buybacks and grants must NOT read as
    insider buying — and must not read as a clean 'no activity' either."""
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {"heldPercentInsiders": 0.02}
    mock_ticker.insider_transactions = _frame([
        (100000, 20000000.0, "", "Redemption, retraction, cancelation, repurchase at price 200.00 per share.",
         "Northgate Financial Corp", "Issuer", "", _today(2), "D"),
        (1000, 40000.0, "", "Compensation for services at price 40.00 per share.",
         "Tremblay (Renee)", "Director of Issuer", "", _today(3), "I"),
        (50000, 3750000.0, "", "Exercise of options at price 75.00 per share.",
         "Okafor (Daniel)", "Senior Officer of Issuer", "", _today(4), "D"),
    ])

    result = get_insider_and_short_data("NGF.TO")

    assert result["insider_signal"].startswith("⚪ No open-market insider buys or sells")
    assert result["open_market_summary"]["buys"] == 0
    assert result["open_market_summary"]["sells"] == 0
    assert result["open_market_summary"]["currency"] == "CAD"
    assert result["filings_source"] == "SEDI (Canadian)"


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_canadian_open_market_selling_is_detected(mock_ticker_class, _g, _s):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {}
    mock_ticker.insider_transactions = _frame([
        (50000, 10000000.0, "", "Disposition in the public market at price 200.00 per share.",
         "Okafor (Daniel)", "Senior Officer of Issuer", "", _today(2), "D"),
    ])

    result = get_insider_and_short_data("NGF.TO")

    assert result["insider_signal"] == "🔴 Insiders SELLING recently"
    assert result["open_market_summary"]["sells"] == 1


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_dollar_weighting_beats_row_counting(mock_ticker_class, _g, _s):
    """Two token buys must not outvote one large sale — the signal is weighted by
    dollars, not by row count."""
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {}
    mock_ticker.insider_transactions = _frame([
        (100, 1000.0, "", "Acquisition in the public market", "A", "Director of Issuer", "", _today(1), "D"),
        (100, 1000.0, "", "Acquisition in the public market", "B", "Director of Issuer", "", _today(2), "D"),
        (100000, 5000000.0, "", "Disposition in the public market", "C", "Senior Officer of Issuer", "", _today(3), "D"),
    ])

    result = get_insider_and_short_data("XYZ.TO")

    assert result["insider_signal"] == "🔴 Insiders SELLING recently"
    assert result["open_market_summary"]["net_value"] == -4998000.0


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_nan_values_never_escape(mock_ticker_class, _g, _s):
    """A bare NaN is not valid JSON and poisons every downstream consumer."""
    import json
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {}
    mock_ticker.insider_transactions = _frame([
        (30104, float("nan"), "", "Sale at price 10.00 per share.", "Jane", "Officer", "", _today(1), "D"),
    ])

    result = get_insider_and_short_data("TESTX")

    assert "NaN" not in json.dumps(result)


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_canadian_short_interest_gap_is_labelled_not_silent(mock_ticker_class, _g, _s):
    """No short %-of-float for a TSX name is a COVERAGE gap, not a low-short
    reading — it must say which."""
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {"shortRatio": 1.1}
    mock_ticker.insider_transactions = _frame([])

    result = get_insider_and_short_data("NGF.TO")

    signal = result["short_interest"]["signal"]
    assert "TSX" in signal
    assert "coverage gap" in signal.lower()


# ---------------------------------------------------------------------------
# get_detailed_insider_activity
# ---------------------------------------------------------------------------
@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_detailed_activity_separates_conviction_from_mechanics(mock_ticker_class, _g, _s):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.insider_transactions = _frame([
        (1000, 50000.0, "", "Acquisition in the public market", "Buyer One", "Director of Issuer", "", _today(1), "D"),
        (2000, 100000.0, "", "Acquisition in the public market", "Buyer Two", "Senior Officer of Issuer", "", _today(2), "D"),
        (500, 25000.0, "", "Exercise of options", "Buyer One", "Director of Issuer", "", _today(3), "D"),
        (900, 45000.0, "", "Acquisition under a purchase/ownership plan", "Buyer Three", "Director of Issuer", "", _today(4), "D"),
        (99999, 5000000.0, "", "Redemption, retraction, cancelation, repurchase", "The Issuer", "Issuer", "", _today(5), "D"),
    ])

    r = get_detailed_insider_activity("XYZ.TO")
    s = r["summary"]

    assert s["open_market_buys"] == 2
    assert s["open_market_buy_value"] == 150000.0
    assert s["distinct_open_market_buyers"] == 2
    assert s["compensation_transactions"] == 1
    assert s["ownership_plan_transactions"] == 1
    # The buyback is the company, not an insider — separated, not counted.
    assert r["issuer_buybacks"]["transactions"] == 1
    assert r["issuer_buybacks"]["value"] == 5000000.0
    assert r["currency"] == "CAD"


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_detailed_activity_detects_cluster_buys(mock_ticker_class, _g, _s):
    """Multiple distinct insiders buying on the open market in a window — the
    strongest insider signal, and one a TSX name could not previously produce."""
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.insider_transactions = _frame([
        (1000, 100000.0, "", "Acquisition in the public market", f"Buyer {i}", "Director of Issuer", "", _today(i + 1), "D")
        for i in range(3)
    ])

    r = get_detailed_insider_activity("ABC.TO")

    assert r["cluster"]["cluster_buy"] is True
    assert r["cluster"]["distinct_buyers"] == 3


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_empty_table_is_a_coverage_gap_not_no_activity(mock_ticker_class, _g, _s):
    """Some TSX names have no insider table at all. Reporting that as 'no insider
    activity' converts missing data into a clean bill of health."""
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.insider_transactions = _frame([])

    r = get_detailed_insider_activity("DEF.TO")

    assert r["coverage"] == "none"
    assert "COVERAGE gap" in r["note"]


# ---------------------------------------------------------------------------
# Venue routing — the actual gap: a Canadian name must not be sent to EDGAR
# ---------------------------------------------------------------------------
def test_canadian_ticker_never_queries_edgar(monkeypatch):
    """EDGAR has no Form 4 for a TSX issuer. Querying it returns a 'not a US
    filer' note that reads like 'no insider data exists' — the exact reason a
    Canadian name came back with nothing to reason over."""
    from agent import tool_registry as reg

    called = []
    monkeypatch.setattr("tools.sec_edgar.get_form4_activity",
                        lambda s: called.append(s) or {"source": "edgar"})
    monkeypatch.setattr("tools.insider_data.get_detailed_insider_activity",
                        lambda s, days=90: {"source": "sedi", "symbol": s})

    assert reg.get_insider_activity.invoke({"symbol": "NGF.TO"})["source"] == "sedi"
    assert called == [], "EDGAR must not be queried for a Canadian listing"


def test_non_sec_filer_falls_through_to_detailed(monkeypatch):
    """A US-looking symbol EDGAR has no CIK for must fall through, not surface
    the 'no CIK' note as if it were an answer."""
    from agent import tool_registry as reg

    monkeypatch.setattr("tools.sec_edgar.get_form4_activity",
                        lambda s: {"symbol": s, "not_an_sec_filer": True, "note": "No EDGAR CIK"})
    monkeypatch.setattr("tools.insider_data.get_detailed_insider_activity",
                        lambda s, days=90: {"source": "yahoo", "symbol": s})

    assert reg.get_insider_activity.invoke({"symbol": "WEIRD"})["source"] == "yahoo"


def test_us_ticker_still_prefers_edgar(monkeypatch):
    """Form 4 coding is stronger than the Yahoo table — US names must keep it."""
    from agent import tool_registry as reg

    monkeypatch.setattr("tools.sec_edgar.get_form4_activity",
                        lambda s: {"source": "SEC EDGAR Form 4", "symbol": s})
    assert reg.get_insider_activity.invoke({"symbol": "AAPL"})["source"] == "SEC EDGAR Form 4"


@patch("tools.cache.daily_cache.set_cached")
@patch("tools.cache.daily_cache.get_cached", return_value=None)
@patch("yfinance.Ticker")
def test_stale_rows_outside_window_are_a_real_answer(mock_ticker_class, _g, _s):
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.insider_transactions = _frame([
        (1000, 50000.0, "", "Acquisition in the public market", "Old Buyer", "Director of Issuer", "", _today(400), "D"),
    ])

    r = get_detailed_insider_activity("XYZ.TO", days=90)

    assert r["coverage"] == "full"
    assert "IS a real answer" in r["note"]
