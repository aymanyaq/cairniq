"""SEC EDGAR pipeline tests (Advisor Roadmap 5.1) — fully offline.

Every network touchpoint (_sec_get_json / _sec_get_text / the cached wrappers)
is patched; fixtures are minimal but structurally faithful EDGAR payloads.
"""
from unittest.mock import patch

from tools.sec_edgar import (
    _issuer_to_ticker,
    _normalize_issuer,
    _parse_form4_xml,
    detect_cluster_buys,
    get_13f_diff,
    get_13f_universe,
    get_form4_activity,
    get_institutional_moves,
    get_recent_8k,
    resolve_cik,
)
from tools.tool_errors import is_unavailable

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CIK_MAP = {
    "AAPL": {"cik": "0000320193", "title": "Apple Inc."},
    "MSFT": {"cik": "0000789019", "title": "Microsoft Corp"},
    "OXY": {"cik": "0000797468", "title": "Occidental Petroleum Corp"},
}


def _form4_xml(code="P", shares=1000, price=50.0, owner="DOE JANE", title="CEO",
               tx_date="2026-07-10", ad_code="A", aff10b5="0"):
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0407</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>{tx_date}</periodOfReport>
  <aff10b5One>{aff10b5}</aff10b5One>
  <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001214156</rptOwnerCik><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>0</isDirector><isOfficer>1</isOfficer><officerTitle>{title}</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>{tx_date}</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>{code}</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{ad_code}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>10000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def _filing_row(form="4", filing_date="2026-07-10", accession="0001-26-000001",
                primary="form4.xml", items=""):
    return {"form": form, "filingDate": filing_date, "accession": accession,
            "primaryDocument": primary, "items": items}


_INFOTABLE_NS = 'xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable"'


def _infotable_xml(rows):
    entries = "".join(
        f"""<infoTable>
  <nameOfIssuer>{name}</nameOfIssuer><titleOfClass>COM</titleOfClass>
  <cusip>{cusip}</cusip><value>{value}</value>
  <shrsOrPrnAmt><sshPrnamt>{shares}</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  {"<putCall>" + putcall + "</putCall>" if putcall else ""}
</infoTable>"""
        for (name, cusip, shares, value, putcall) in rows
    )
    return f'<?xml version="1.0"?><informationTable {_INFOTABLE_NS}>{entries}</informationTable>'


# ---------------------------------------------------------------------------
# CIK resolution + name normalization
# ---------------------------------------------------------------------------

@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_resolve_cik(mock_map):
    assert resolve_cik("aapl") == "0000320193"
    assert resolve_cik("SHOP.TO") is None  # non-SEC listing → real answer, not error


def test_normalize_issuer():
    assert _normalize_issuer("APPLE INC") == "APPLE"
    assert _normalize_issuer("Occidental Petroleum Corp") == "OCCIDENTAL PETROLEUM"
    # Conservative: never strips the whole name away
    assert _normalize_issuer("CORP") == "CORP"
    # Apostrophe divergence between filings ("MACYS INC") and SEC titles
    # ("Macy's Inc") — both sides must land on the same key.
    assert _normalize_issuer("MACYS INC") == _normalize_issuer("Macy's Inc") == "MACYS"


def test_issuer_to_ticker():
    name_map = {"APPLE": "AAPL", "OCCIDENTAL PETROLEUM": "OXY"}
    assert _issuer_to_ticker("APPLE INC", name_map) == "AAPL"
    assert _issuer_to_ticker("OCCIDENTAL PETROLEUM CORP", name_map) == "OXY"
    assert _issuer_to_ticker("UNKNOWN WIDGETS LLC", name_map) is None


# ---------------------------------------------------------------------------
# Form 4 parsing + classification
# ---------------------------------------------------------------------------

def test_parse_form4_open_market_buy():
    parsed = _parse_form4_xml(_form4_xml(code="P", shares=2000, price=25.5))
    assert len(parsed["transactions"]) == 1
    tx = parsed["transactions"][0]
    assert tx["signal"] == "BUY"
    assert tx["code"] == "P"
    assert tx["shares"] == 2000
    assert tx["value"] == 51000.0
    assert tx["owner"] == "DOE JANE"
    assert tx["owner_title"] == "CEO"
    assert tx["rule_10b5_1_plan"] is False


def test_parse_form4_correct_coding():
    """The core 5.1 claim: M (exercise) and A (grant) are NOT buys."""
    for code, expected in [("S", "SELL"), ("M", "COMP"), ("A", "COMP"), ("F", "COMP"), ("G", "GIFT")]:
        tx = _parse_form4_xml(_form4_xml(code=code))["transactions"][0]
        assert tx["signal"] == expected, f"code {code} misclassified as {tx['signal']}"


def test_parse_form4_10b5_1_flag():
    tx = _parse_form4_xml(_form4_xml(code="S", aff10b5="1"))["transactions"][0]
    assert tx["rule_10b5_1_plan"] is True


# ---------------------------------------------------------------------------
# Cluster-buy detection
# ---------------------------------------------------------------------------

def _tx(owner, code="P", value=100_000, tx_date="2026-07-10"):
    return {"owner": owner, "code": code, "value": value, "date": tx_date, "signal": "BUY"}


def test_cluster_three_distinct_buyers():
    cluster = detect_cluster_buys([_tx("A"), _tx("B"), _tx("C")])
    assert cluster["cluster_buy"] is True
    assert cluster["distinct_buyers"] == 3


def test_cluster_two_buyers_large_value():
    cluster = detect_cluster_buys([_tx("A", value=600_000), _tx("B", value=500_000)])
    assert cluster["cluster_buy"] is True


def test_no_cluster_single_buyer_or_awards():
    assert detect_cluster_buys([_tx("A"), _tx("A"), _tx("A")])["cluster_buy"] is False
    # Awards/exercises never form a cluster — code P only
    awards = [dict(_tx(o), code="A") for o in ("A", "B", "C", "D")]
    assert detect_cluster_buys(awards)["cluster_buy"] is False


def test_cluster_window_excludes_old_buys():
    cluster = detect_cluster_buys([
        _tx("A", tx_date="2026-07-10"), _tx("B", tx_date="2026-07-05"),
        _tx("C", tx_date="2026-03-01"),  # far outside the 30d window
    ])
    assert cluster["cluster_buy"] is False
    assert cluster["distinct_buyers"] == 2


# ---------------------------------------------------------------------------
# get_form4_activity end-to-end (mocked fetches)
# ---------------------------------------------------------------------------

@patch("tools.sec_edgar._fetch_form4_doc")
@patch("tools.sec_edgar._recent_filings")
@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_form4_activity_summary(mock_map, mock_filings, mock_doc):
    mock_filings.return_value = [
        _filing_row(accession="acc-1", filing_date="2026-07-10"),
        _filing_row(accession="acc-2", filing_date="2026-07-08"),
        _filing_row(form="8-K", accession="acc-3"),  # ignored
    ]
    mock_doc.side_effect = [
        _parse_form4_xml(_form4_xml(code="P", shares=1000, price=100.0, owner="DOE JANE")),
        _parse_form4_xml(_form4_xml(code="M", shares=5000, price=10.0, owner="ROE RICHARD")),
    ]
    result = get_form4_activity.__wrapped__("AAPL", days=90)
    s = result["summary"]
    assert s["open_market_buys"] == 1
    assert s["open_market_sells"] == 0
    assert s["open_market_buy_value"] == 100_000.0
    assert s["compensation_transactions"] == 1  # the option exercise is NOT a buy
    assert s["distinct_open_market_buyers"] == 1
    assert result["filings_analyzed"] == 2


@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_form4_activity_non_us_symbol(mock_map):
    result = get_form4_activity.__wrapped__("RY.TO")
    assert "note" in result and "CIK" in result["note"]
    assert not is_unavailable(result)  # a real answer, not a degraded fetch


@patch("tools.sec_edgar._recent_filings", side_effect=ConnectionError("edgar down"))
@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_form4_activity_network_failure(mock_map, mock_filings):
    assert is_unavailable(get_form4_activity.__wrapped__("AAPL"))


# ---------------------------------------------------------------------------
# 8-K material events
# ---------------------------------------------------------------------------

@patch("tools.sec_edgar._recent_filings")
@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_8k_severity_mapping(mock_map, mock_filings):
    mock_filings.return_value = [
        _filing_row(form="8-K", filing_date="2026-07-10", accession="a1", items="4.02,9.01"),
        _filing_row(form="8-K", filing_date="2026-07-09", accession="a2", items="2.02,9.01"),
        _filing_row(form="8-K", filing_date="2026-07-08", accession="a3", items="5.02"),
        _filing_row(form="4", filing_date="2026-07-08", accession="a4"),  # ignored
    ]
    result = get_recent_8k.__wrapped__("AAPL", days=30)
    by_acc = {f["accession"]: f for f in result["filings"]}
    assert by_acc["a1"]["severity"] == "critical"   # restatement
    assert by_acc["a2"]["severity"] == "info"       # earnings release
    assert by_acc["a3"]["severity"] == "warning"    # officer departure
    assert result["material_count"] == 2


@patch("tools.sec_edgar._recent_filings")
@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_8k_window_filter(mock_map, mock_filings):
    mock_filings.return_value = [
        _filing_row(form="8-K", filing_date="2020-01-01", accession="old", items="1.03"),
    ]
    result = get_recent_8k.__wrapped__("AAPL", days=30)
    assert result["filings"] == []
    assert "No 8-K" in result["summary"]


# ---------------------------------------------------------------------------
# 13F diffs + universe
# ---------------------------------------------------------------------------

_Q_LATEST = {
    "037833100": {"name": "APPLE INC", "shares": 1_000_000.0, "value": 5000.0},
    "674599105": {"name": "OCCIDENTAL PETROLEUM CORP", "shares": 2_600_000.0, "value": 9000.0},  # +30% add
    "594918104": {"name": "MICROSOFT CORP", "shares": 400_000.0, "value": 2000.0},  # new
}
_Q_PREV = {
    "037833100": {"name": "APPLE INC", "shares": 1_000_000.0, "value": 5000.0},   # unchanged
    "674599105": {"name": "OCCIDENTAL PETROLEUM CORP", "shares": 2_000_000.0, "value": 7000.0},
    "88160R101": {"name": "TESLA INC", "shares": 300_000.0, "value": 1500.0},     # exited
}


@patch("tools.sec_edgar._fetch_13f_holdings")
@patch("tools.sec_edgar._latest_13f_accessions")
@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_13f_diff(mock_map, mock_accessions, mock_holdings):
    mock_accessions.return_value = [
        _filing_row(form="13F-HR", filing_date="2026-05-15", accession="q2"),
        _filing_row(form="13F-HR", filing_date="2026-02-14", accession="q1"),
    ]
    mock_holdings.side_effect = [_Q_LATEST, _Q_PREV]
    diff = get_13f_diff.__wrapped__("Berkshire Hathaway")

    assert [r["ticker"] for r in diff["new_positions"]] == ["MSFT"]
    assert [r["issuer"] for r in diff["exits"]] == ["TESLA INC"]
    assert [r["ticker"] for r in diff["adds"]] == ["OXY"]
    assert diff["adds"][0]["change_pct"] == 30.0
    assert diff["trims"] == []
    assert diff["positions_held"] == 3


@patch("tools.sec_edgar.get_cik_map", return_value=_CIK_MAP)
def test_13f_unknown_manager(mock_map):
    result = get_13f_diff.__wrapped__("Nonexistent Capital")
    assert is_unavailable(result)
    assert "tracked" in result


@patch("tools.sec_edgar.get_13f_diff")
@patch("tools.sec_edgar._managers_13f", return_value={"M1": 1, "M2": 2})
def test_13f_universe_ranks_by_manager_count(mock_mgrs, mock_diff):
    mock_diff.side_effect = [
        {"new_positions": [{"ticker": "MSFT"}], "adds": [{"ticker": "OXY"}]},
        {"new_positions": [{"ticker": "OXY"}], "adds": [{"ticker": None}]},  # unmapped skipped
    ]
    universe = get_13f_universe.__wrapped__()
    assert universe == ["OXY", "MSFT"]  # OXY touched by 2 managers → first


@patch("tools.sec_edgar.get_13f_diff", return_value={"status": "unavailable", "source": "SEC EDGAR", "reason": "down"})
@patch("tools.sec_edgar._managers_13f", return_value={"M1": 1})
def test_institutional_moves_all_unavailable(mock_mgrs, mock_diff):
    assert is_unavailable(get_institutional_moves(None))


# ---------------------------------------------------------------------------
# 13F info-table XML parsing (namespace handling, putCall skip)
# ---------------------------------------------------------------------------

@patch("tools.sec_edgar._sec_get_text")
@patch("tools.sec_edgar._sec_get_json")
def test_fetch_13f_holdings_parses_namespaced_xml(mock_json, mock_text):
    from tools.sec_edgar import _fetch_13f_holdings
    mock_json.return_value = {"directory": {"item": [
        {"name": "primary_doc.xml"}, {"name": "form13fInfoTable.xml"},
    ]}}
    mock_text.return_value = _infotable_xml([
        ("APPLE INC", "037833100", 500, 100, ""),
        ("APPLE INC", "037833100", 250, 50, ""),        # second block, same CUSIP → summed
        ("TESLA INC", "88160R101", 100, 40, "Put"),     # option position → skipped
    ])
    holdings = _fetch_13f_holdings.__wrapped__("0001067983", "acc-x")
    assert holdings["037833100"]["shares"] == 750
    assert "88160R101" not in holdings
