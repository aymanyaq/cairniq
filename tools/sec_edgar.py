"""
SEC EDGAR pipeline (Advisor Roadmap Theme 5.1).

Free, keyless institutional-grade filings data, three signals:

  1. Form 4 insider activity with CORRECT transaction coding — only code P is
     an open-market buy and only code S an open-market sale; option exercises
     (M), grants (A), and tax withholding (F) are compensation mechanics, not
     conviction. yfinance's insider table lumps these together, which is the
     exact gap this module closes. Includes cluster-buy detection (multiple
     distinct insiders buying in a window — the strongest insider signal).
  2. 8-K material-event polling with item-level severity (bankruptcy,
     restatement, auditor change, delisting notice → critical).
  3. 13F quarter-over-quarter diffs for a curated set of long-horizon
     managers — new buys / exits / adds / trims — replacing the scraped
     media/guru feed as the scanner's institutional universe source.

SEC fair-use rules: a declared User-Agent and <10 req/s. All fetches go
through one throttled helper. Results are JSON-safe dicts; degraded fetches
return tools.tool_errors.unavailable() while genuine "no filings" keeps its
natural shape (that IS an answer).
"""
import json
import re
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.tool_errors import unavailable

# ---------------------------------------------------------------------------
# HTTP plumbing — declared UA + polite throttle (SEC fair-use)
# ---------------------------------------------------------------------------
_SEC_HEADERS = {"User-Agent": "CairnIQBot contact@cairniq.local"}
_MIN_REQUEST_INTERVAL = 0.15  # seconds between requests (~6 req/s max)
_REQUEST_TIMEOUT = 15

_throttle_lock = threading.Lock()
_last_request_ts = [0.0]


def _throttled_get(url: str):
    """GET with SEC-required UA and a minimum inter-request interval. Raises on HTTP error."""
    import requests

    with _throttle_lock:
        wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_ts[0])
        if wait > 0:
            time.sleep(wait)
        _last_request_ts[0] = time.monotonic()
    resp = requests.get(url, headers=_SEC_HEADERS, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def _sec_get_json(url: str) -> Any:
    return _throttled_get(url).json()


def _sec_get_text(url: str) -> str:
    return _throttled_get(url).text


# ---------------------------------------------------------------------------
# CIK resolution + issuer-name → ticker mapping (both from company_tickers.json)
# ---------------------------------------------------------------------------
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Trailing tokens stripped when normalizing issuer names for 13F name→ticker
# matching. Deliberately conservative: over-stripping causes collisions.
_NAME_SUFFIX_TOKENS = {
    "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "PLC", "SA", "NV",
    "AG", "HOLDINGS", "HLDGS", "HOLDING", "GROUP", "GRP", "TRUST", "LP",
    "COM", "NEW", "DEL", "CL", "CLASS", "A", "B", "C", "ORD", "SHS", "ADR", "ADS",
}


# stamp=False: map-shaped ({TICKER: {...}}), so an in-band `_as_of` would read
# back as a ticker named "_as_of" whose value is a string. See tools/cache.py.
@cached(key_func=lambda: "sec_edgar:cik_map", ttl=7 * 86400, stamp=False)
@log_exceptions()
def get_cik_map() -> dict[str, dict[str, str]]:
    """{TICKER: {cik: zero-padded-10-digit, title}} from SEC's official mapping."""
    raw = _sec_get_json(_COMPANY_TICKERS_URL)
    out: dict[str, dict[str, str]] = {}
    for entry in (raw or {}).values():
        try:
            ticker = str(entry["ticker"]).upper()
            out[ticker] = {
                "cik": f"{int(entry['cik_str']):010d}",
                "title": str(entry.get("title", "")),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out


def resolve_cik(symbol: str) -> str | None:
    """Zero-padded CIK for a ticker, or None when it isn't an SEC filer
    (e.g. .TO listings, crypto pairs). None is a real answer, not an error."""
    try:
        cik_map = get_cik_map()
    except Exception:
        return None
    if not isinstance(cik_map, dict):
        return None
    entry = cik_map.get(symbol.upper().strip())
    return entry["cik"] if entry else None


def _normalize_issuer(name: str) -> str:
    """Normalize an issuer/company name for matching (uppercase, no punctuation,
    conservative suffix stripping from the right). Apostrophes/periods are
    REMOVED, not spaced: filings say "MACYS INC" where the SEC title says
    "Macy's Inc" — both must normalize to MACYS."""
    up = re.sub(r"[^A-Z0-9 ]", " ", re.sub(r"['’.]", "", str(name).upper()))
    tokens = [t for t in up.split() if t]
    while len(tokens) > 1 and tokens[-1] in _NAME_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _issuer_name_to_ticker_map() -> dict[str, str]:
    """normalized issuer name → ticker. company_tickers.json is ordered by
    market cap, so the first (primary) listing wins on duplicate names."""
    try:
        cik_map = get_cik_map()
    except Exception:
        return {}
    if not isinstance(cik_map, dict):
        return {}
    out: dict[str, str] = {}
    for ticker, entry in cik_map.items():
        # Skip non-record values rather than trusting every key to be a ticker.
        # Caches written before `stamp=False` still carry an `_as_of` string on
        # disk for the rest of their TTL, and one of those took this whole
        # producer down; a map read defensively survives its own history.
        if not isinstance(entry, dict):
            continue
        norm = _normalize_issuer(entry.get("title", ""))
        if norm and norm not in out:
            out[norm] = ticker
    return out


def _issuer_to_ticker(name: str, name_map: dict[str, str]) -> str | None:
    """Map a 13F nameOfIssuer to a ticker: exact normalized match, then one
    trailing-token relaxation. None when unmapped (reported as name+CUSIP)."""
    norm = _normalize_issuer(name)
    if not norm:
        return None
    hit = name_map.get(norm)
    if hit:
        return hit
    tokens = norm.split()
    if len(tokens) > 1:
        return name_map.get(" ".join(tokens[:-1]))
    return None


# ---------------------------------------------------------------------------
# Submissions index (one JSON per filer; drives Form 4, 8-K, and 13F listing)
# ---------------------------------------------------------------------------
def _archive_base(cik: str, accession: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession.replace('-', '')}"
    )


@cached(key_func=lambda cik: f"sec_edgar:submissions:{cik}", ttl=6 * 3600)
@log_exceptions()
def _recent_filings(cik: str) -> list[dict[str, str]]:
    """Recent filings for a CIK as row dicts (form, filingDate, accession,
    primaryDocument, items), newest first — from the submissions API."""
    raw = _sec_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = ((raw or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    rows = []
    for i, form in enumerate(forms):
        try:
            rows.append({
                "form": str(form),
                "filingDate": str((recent.get("filingDate") or [])[i]),
                "accession": str((recent.get("accessionNumber") or [])[i]),
                "primaryDocument": str((recent.get("primaryDocument") or [""] * len(forms))[i]),
                "items": str((recent.get("items") or [""] * len(forms))[i]),
            })
        except IndexError:
            continue
    return rows


# ---------------------------------------------------------------------------
# Form 4 — insider transactions with correct transaction coding
# ---------------------------------------------------------------------------
# SEC transaction codes → (description, signal). Only P/S are open-market
# conviction signals; everything else is compensation mechanics or transfers.
FORM4_TX_CODES: dict[str, tuple[str, str]] = {
    "P": ("Open-market purchase", "BUY"),
    "S": ("Open-market sale", "SELL"),
    "A": ("Grant/award", "COMP"),
    "M": ("Option exercise", "COMP"),
    "F": ("Tax withholding", "COMP"),
    "D": ("Disposition to issuer", "OTHER"),
    "G": ("Gift", "GIFT"),
    "C": ("Conversion of derivative", "OTHER"),
    "X": ("In-the-money option exercise", "OTHER"),
    "J": ("Other acquisition/disposition", "OTHER"),
    "L": ("Small acquisition", "OTHER"),
    "W": ("Acquisition by will/inheritance", "OTHER"),
    "I": ("Discretionary transaction", "OTHER"),
    "U": ("Tender of shares", "OTHER"),
}

# Cluster-buy thresholds: either N distinct open-market buyers in the window,
# or a smaller group whose combined purchases clear a dollar bar.
CLUSTER_WINDOW_DAYS = 30
CLUSTER_MIN_BUYERS = 3
CLUSTER_ALT_MIN_BUYERS = 2
CLUSTER_ALT_MIN_VALUE = 1_000_000
_FORM4_MAX_FILINGS = 15


def _xml_text(elem, path: str) -> str:
    """findtext that also unwraps EDGAR's <value> child convention."""
    if elem is None:
        return ""
    node = elem.find(path)
    if node is None:
        return ""
    if node.text and node.text.strip():
        return node.text.strip()
    value = node.find("value")
    return (value.text or "").strip() if value is not None and value.text else ""


@log_exceptions()
def _parse_form4_xml(xml_text: str) -> dict[str, Any]:
    """Parse one Form 4 ownershipDocument: owners, non-derivative transactions
    (coded + classified), and the 10b5-1 plan checkbox where present."""
    import defusedxml.ElementTree as ET

    root = ET.fromstring(xml_text)

    owners = []
    for owner in root.findall("reportingOwner"):
        rel = owner.find("reportingOwnerRelationship")
        title = _xml_text(rel, "officerTitle")
        if not title and rel is not None:
            if (_xml_text(rel, "isDirector") or "0") in ("1", "true"):
                title = "Director"
            elif (_xml_text(rel, "isTenPercentOwner") or "0") in ("1", "true"):
                title = "10% owner"
        owners.append({
            "name": _xml_text(owner, "reportingOwnerId/rptOwnerName"),
            "title": title,
        })

    aff10b5 = (root.findtext("aff10b5One") or "").strip() in ("1", "true")

    transactions = []
    table = root.find("nonDerivativeTable")
    for tx in table.findall("nonDerivativeTransaction") if table is not None else []:
        code = _xml_text(tx, "transactionCoding/transactionCode").upper()
        desc, signal = FORM4_TX_CODES.get(code, (f"Code {code}", "OTHER"))
        try:
            shares = float(_xml_text(tx, "transactionAmounts/transactionShares") or 0)
        except ValueError:
            shares = 0.0
        try:
            price = float(_xml_text(tx, "transactionAmounts/transactionPricePerShare") or 0)
        except ValueError:
            price = 0.0
        transactions.append({
            "date": _xml_text(tx, "transactionDate"),
            "code": code,
            "code_description": desc,
            "signal": signal,
            "acquired_disposed": _xml_text(tx, "transactionAmounts/transactionAcquiredDisposedCode").upper(),
            "shares": shares,
            "price": price,
            "value": round(shares * price, 2),
            "owner": owners[0]["name"] if owners else "",
            "owner_title": owners[0]["title"] if owners else "",
            "rule_10b5_1_plan": aff10b5,
        })

    return {"owners": owners, "transactions": transactions, "rule_10b5_1_plan": aff10b5}


@cached(key_func=lambda cik, accession, doc: f"sec_edgar:form4:{accession}", ttl=30 * 86400)
@log_exceptions()
def _fetch_form4_doc(cik: str, accession: str, doc: str) -> dict[str, Any]:
    """Fetch + parse one Form 4 filing (immutable once filed → long TTL)."""
    # primaryDocument sometimes carries an XSL-rendering path prefix; the raw
    # XML lives at the bare filename.
    filename = doc.split("/")[-1]
    xml_text = _sec_get_text(f"{_archive_base(cik, accession)}/{filename}")
    return _parse_form4_xml(xml_text) or {"owners": [], "transactions": []}


def detect_cluster_buys(
    transactions: list[dict[str, Any]], window_days: int = CLUSTER_WINDOW_DAYS
) -> dict[str, Any]:
    """Cluster-buy detection over classified transactions: distinct insiders
    making open-market (code P) purchases inside a window anchored on the most
    recent buy. The strongest insider signal there is."""
    buys = [t for t in transactions if t.get("code") == "P" and t.get("date")]
    if not buys:
        return {"cluster_buy": False, "distinct_buyers": 0, "window_days": window_days}

    latest = max(t["date"] for t in buys)
    try:
        anchor = datetime.strptime(latest, "%Y-%m-%d").date()
    except ValueError:
        anchor = date.today()
    cutoff = (anchor - timedelta(days=window_days)).isoformat()
    windowed = [t for t in buys if t["date"] >= cutoff]

    by_owner: dict[str, float] = {}
    for t in windowed:
        owner = t.get("owner") or "unknown"
        by_owner[owner] = by_owner.get(owner, 0.0) + float(t.get("value") or 0)

    distinct = len(by_owner)
    total_value = round(sum(by_owner.values()), 2)
    is_cluster = distinct >= CLUSTER_MIN_BUYERS or (
        distinct >= CLUSTER_ALT_MIN_BUYERS and total_value >= CLUSTER_ALT_MIN_VALUE
    )
    return {
        "cluster_buy": is_cluster,
        "distinct_buyers": distinct,
        "total_value": total_value,
        "buyers": [
            {"name": name, "value": round(val, 2)}
            for name, val in sorted(by_owner.items(), key=lambda kv: -kv[1])
        ],
        "latest_buy_date": latest,
        "window_days": window_days,
        "basis": (
            f"≥{CLUSTER_MIN_BUYERS} distinct open-market buyers in {window_days}d, "
            f"or ≥{CLUSTER_ALT_MIN_BUYERS} with combined ≥${CLUSTER_ALT_MIN_VALUE:,}"
        ),
    }


@cached(key_func=lambda symbol, days=90: f"sec_edgar:form4_activity:{symbol.upper()}:{days}", ttl=6 * 3600)
@log_exceptions()
def get_form4_activity(symbol: str, days: int = 90) -> dict[str, Any]:
    """Insider activity from SEC Form 4 filings with correct buy/sell coding.

    Open-market purchases (P) and sales (S) are separated from compensation
    mechanics (grants, option exercises, tax withholding), with per-owner
    aggregates and cluster-buy detection.
    """
    symbol = symbol.upper().strip()
    try:
        cik = resolve_cik(symbol)
    except Exception as e:
        return unavailable("SEC EDGAR", f"CIK lookup failed: {e}", symbol=symbol)
    if not cik:
        return {
            "symbol": symbol,
            "not_an_sec_filer": True,
            "note": (
                "No EDGAR CIK — not a US SEC filer (e.g. Canadian .TO or other "
                "foreign listing). Insider filings unavailable from EDGAR; use "
                "tools.insider_data.get_detailed_insider_activity for this venue."
            ),
        }

    try:
        filings = _recent_filings(cik)
    except Exception as e:
        return unavailable("SEC EDGAR", f"submissions fetch failed: {e}", symbol=symbol)
    if not isinstance(filings, list):
        return unavailable("SEC EDGAR", "submissions fetch failed", symbol=symbol)

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    form4s = [
        f for f in filings
        if f.get("form") == "4" and f.get("filingDate", "") >= cutoff
    ][:_FORM4_MAX_FILINGS]

    transactions: list[dict[str, Any]] = []
    fetch_errors = 0
    for f in form4s:
        try:
            parsed = _fetch_form4_doc(cik, f["accession"], f["primaryDocument"])
            for tx in (parsed or {}).get("transactions") or []:
                tx = dict(tx)
                tx["filing_date"] = f["filingDate"]
                tx["accession"] = f["accession"]
                transactions.append(tx)
        except Exception:
            fetch_errors += 1

    if form4s and fetch_errors == len(form4s):
        return unavailable("SEC EDGAR", "every Form 4 document fetch failed", symbol=symbol)

    open_buys = [t for t in transactions if t["signal"] == "BUY"]
    open_sells = [t for t in transactions if t["signal"] == "SELL"]
    comp = [t for t in transactions if t["signal"] == "COMP"]
    buy_value = round(sum(t["value"] for t in open_buys), 2)
    sell_value = round(sum(t["value"] for t in open_sells), 2)

    result = {
        "symbol": symbol,
        "source": "SEC EDGAR Form 4",
        "window_days": days,
        "filings_analyzed": len(form4s),
        "transactions": transactions,
        "summary": {
            "open_market_buys": len(open_buys),
            "open_market_sells": len(open_sells),
            "open_market_buy_value": buy_value,
            "open_market_sell_value": sell_value,
            "net_open_market_value": round(buy_value - sell_value, 2),
            "distinct_open_market_buyers": len({t["owner"] for t in open_buys}),
            "compensation_transactions": len(comp),
            "coding_note": (
                "Only code P (open-market purchase) and S (open-market sale) are "
                "conviction signals; grants (A), option exercises (M), and tax "
                "withholding (F) are compensation mechanics and are excluded from "
                "buy/sell counts."
            ),
        },
        "cluster": detect_cluster_buys(transactions),
    }
    if not form4s:
        result["note"] = f"No Form 4 filings in the last {days} days — that is a real answer, not a data gap."
    return result


# ---------------------------------------------------------------------------
# 8-K — material corporate events with item-level severity
# ---------------------------------------------------------------------------
# item code → (description, severity). Severity feeds the alerts rail (3.2):
# warning+ raise an inbox alert; info items are context only.
EIGHT_K_ITEMS: dict[str, tuple[str, str]] = {
    "1.01": ("Entry into material agreement", "warning"),
    "1.02": ("Termination of material agreement", "warning"),
    "1.03": ("Bankruptcy or receivership", "critical"),
    "1.05": ("Material cybersecurity incident", "critical"),
    "2.01": ("Completed acquisition or disposition of assets", "warning"),
    "2.02": ("Results of operations (earnings)", "info"),
    "2.03": ("Creation of material financial obligation", "warning"),
    "2.04": ("Triggering event on a financial obligation", "critical"),
    "2.05": ("Exit or disposal costs", "warning"),
    "2.06": ("Material impairment", "critical"),
    "3.01": ("Delisting notice / listing-standard failure", "critical"),
    "3.02": ("Unregistered equity sales", "warning"),
    "3.03": ("Material modification of shareholder rights", "warning"),
    "4.01": ("Change of auditor", "critical"),
    "4.02": ("Non-reliance on prior financials (restatement)", "critical"),
    "5.01": ("Change of control", "warning"),
    "5.02": ("Officer/director departure or appointment", "warning"),
    "5.03": ("Charter/bylaws amendment", "info"),
    "5.07": ("Shareholder vote results", "info"),
    "7.01": ("Regulation FD disclosure", "info"),
    "8.01": ("Other events", "info"),
    "9.01": ("Financial statements and exhibits", "info"),
}
_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@cached(key_func=lambda symbol, days=30: f"sec_edgar:8k:{symbol.upper()}:{days}", ttl=3600)
@log_exceptions()
def get_recent_8k(symbol: str, days: int = 30) -> dict[str, Any]:
    """Recent 8-K filings with parsed item codes and a materiality severity per
    filing (worst item wins). Item codes come from the submissions index — one
    request, no document parsing needed."""
    symbol = symbol.upper().strip()
    try:
        cik = resolve_cik(symbol)
    except Exception as e:
        return unavailable("SEC EDGAR", f"CIK lookup failed: {e}", symbol=symbol)
    if not cik:
        return {"symbol": symbol, "note": "No EDGAR CIK — not a US SEC filer; 8-K polling unavailable."}

    try:
        filings = _recent_filings(cik)
    except Exception as e:
        return unavailable("SEC EDGAR", f"submissions fetch failed: {e}", symbol=symbol)
    if not isinstance(filings, list):
        return unavailable("SEC EDGAR", "submissions fetch failed", symbol=symbol)

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    out = []
    for f in filings:
        if f.get("form") not in ("8-K", "8-K/A") or f.get("filingDate", "") < cutoff:
            continue
        codes = [c.strip() for c in (f.get("items") or "").split(",") if c.strip()]
        items = []
        severity = "info"
        for code in codes:
            desc, sev = EIGHT_K_ITEMS.get(code, (f"Item {code}", "info"))
            items.append({"code": code, "description": desc, "severity": sev})
            if _SEVERITY_ORDER[sev] > _SEVERITY_ORDER[severity]:
                severity = sev
        out.append({
            "form": f["form"],
            "filing_date": f["filingDate"],
            "accession": f["accession"],
            "items": items,
            "severity": severity,
            "url": f"{_archive_base(cik, f['accession'])}/{(f.get('primaryDocument') or '').split('/')[-1]}",
        })

    material = [f for f in out if f["severity"] != "info"]
    return {
        "symbol": symbol,
        "source": "SEC EDGAR 8-K",
        "window_days": days,
        "filings": out,
        "material_count": len(material),
        "summary": (
            f"{len(out)} 8-K filing(s) in {days}d, {len(material)} material"
            if out else f"No 8-K filings in the last {days} days."
        ),
    }


# ---------------------------------------------------------------------------
# 13F — quarter-over-quarter diffs for curated long-horizon managers
# ---------------------------------------------------------------------------
# name → CIK (int). Chosen for long-horizon, conviction-weighted books — the
# signal that fits an accumulation-first funnel. Overridable without a code
# change via funnel_config.json: {"edgar": {"managers_13f": {"name": cik}}}.
MANAGERS_13F: dict[str, int] = {
    "Berkshire Hathaway": 1067983,
    "Pershing Square": 1336528,
    "Duquesne Family Office": 1536411,
    "Third Point": 1040273,
    "Baupost Group": 1061768,
    "Lone Pine Capital": 1061165,
    "Viking Global": 1103804,
    "Coatue Management": 1135730,
    "Tiger Global": 1167483,
    "Scion Asset Management": 1649339,
}
_ADD_TRIM_THRESHOLD_PCT = 25.0   # share-count change that counts as add/trim
_13F_UNIVERSE_CAP = 40


def _managers_13f() -> dict[str, int]:
    """Curated managers, with an optional funnel_config override."""
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "user_data", "funnel_config.json")
    try:
        with open(cfg_path) as f:
            block = (json.load(f) or {}).get("edgar")
        override = block.get("managers_13f") if isinstance(block, dict) else None
        if isinstance(override, dict) and override:
            return {str(k): int(v) for k, v in override.items()}
    except Exception:
        pass
    return dict(MANAGERS_13F)


def _latest_13f_accessions(cik: str, count: int = 2) -> list[dict[str, str]]:
    """Newest `count` 13F-HR filings (skips 13F-NT notices) as filing rows."""
    filings = _recent_filings(cik)
    if not isinstance(filings, list):
        return []
    return [f for f in filings if f.get("form") in ("13F-HR", "13F-HR/A")][:count]


# stamp=False: map-shaped ({CUSIP: {...}}) — see get_cik_map above.
@cached(key_func=lambda cik, accession: f"sec_edgar:13f_holdings:{accession}", ttl=30 * 86400, stamp=False)
@log_exceptions()
def _fetch_13f_holdings(cik: str, accession: str) -> dict[str, dict[str, Any]]:
    """Aggregate holdings from one 13F-HR information table:
    {cusip: {name, shares, value}}. Options positions (putCall) are skipped.
    The info-table XML is discovered via the filing directory's index.json."""
    import defusedxml.ElementTree as ET

    base = _archive_base(cik, accession)
    index = _sec_get_json(f"{base}/index.json")
    files = [
        item.get("name", "")
        for item in ((index or {}).get("directory") or {}).get("item") or []
    ]
    xml_files = [n for n in files if n.lower().endswith(".xml") and "primary_doc" not in n.lower()]
    # Prefer names that look like the info table; fall back to any candidate.
    xml_files.sort(key=lambda n: ("info" not in n.lower(), n))

    for name in xml_files:
        try:
            root = ET.fromstring(_sec_get_text(f"{base}/{name}"))
        except Exception:
            continue
        if not root.tag.endswith("informationTable"):
            continue
        holdings: dict[str, dict[str, Any]] = {}
        for entry in root:
            if not entry.tag.endswith("infoTable"):
                continue
            fields = {child.tag.split("}")[-1]: child for child in entry}
            put_call = fields.get("putCall")
            if put_call is not None and (put_call.text or "").strip():
                continue
            cusip = (fields["cusip"].text or "").strip() if "cusip" in fields else ""
            if not cusip:
                continue
            issuer = (fields["nameOfIssuer"].text or "").strip() if "nameOfIssuer" in fields else ""
            try:
                value = float((fields["value"].text or 0) if "value" in fields else 0)
            except (ValueError, TypeError):
                value = 0.0
            shares = 0.0
            shr = fields.get("shrsOrPrnAmt")
            if shr is not None:
                for child in shr:
                    if child.tag.endswith("sshPrnamt"):
                        try:
                            shares = float(child.text or 0)
                        except (ValueError, TypeError):
                            shares = 0.0
            slot = holdings.setdefault(cusip, {"name": issuer, "shares": 0.0, "value": 0.0})
            slot["shares"] += shares
            slot["value"] += value
        return holdings
    return {}


@cached(key_func=lambda manager: f"sec_edgar:13f_diff:{manager}", ttl=86400)
@log_exceptions()
def get_13f_diff(manager: str) -> dict[str, Any]:
    """Quarter-over-quarter position diff for one tracked manager:
    new positions, exits, adds (> +25% shares), trims (< -25% shares)."""
    managers = _managers_13f()
    match = next((k for k in managers if k.lower() == manager.lower()), None)
    if match is None:
        match = next((k for k in managers if manager.lower() in k.lower()), None)
    if match is None:
        return unavailable(
            "SEC EDGAR", f"unknown 13F manager '{manager}'",
            tracked=sorted(managers),
        )

    cik = f"{managers[match]:010d}"
    try:
        recent = _latest_13f_accessions(cik)
    except Exception as e:
        return unavailable("SEC EDGAR", f"13F listing fetch failed: {e}", manager=match)
    if not recent:
        return {"manager": match, "note": "No 13F-HR filings found on EDGAR for this CIK."}

    try:
        latest = _fetch_13f_holdings(cik, recent[0]["accession"]) or {}
        previous = _fetch_13f_holdings(cik, recent[1]["accession"]) or {} if len(recent) > 1 else {}
    except Exception as e:
        return unavailable("SEC EDGAR", f"13F info-table fetch failed: {e}", manager=match)
    if not latest:
        return unavailable("SEC EDGAR", "13F information table could not be parsed", manager=match)

    name_map = _issuer_name_to_ticker_map()

    def _row(cusip: str, pos: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "issuer": pos["name"],
            "ticker": _issuer_to_ticker(pos["name"], name_map),
            "cusip": cusip,
            "shares": pos["shares"],
            # Reported position value in USD (whole dollars since 2023-Q1).
            # Carried so consumers can rank by dollars rather than by row count.
            "value": float(pos.get("value") or 0.0),
            **extra,
        }

    new_positions, exits, adds, trims = [], [], [], []
    for cusip, pos in latest.items():
        if not isinstance(pos, dict):
            continue  # see _issuer_name_to_ticker_map — legacy stamped cache entry
        prev = previous.get(cusip)
        if not isinstance(prev, dict):
            prev = None
        if prev is None:
            if previous:  # only a real "new" when we have a prior quarter to compare
                new_positions.append(_row(cusip, pos))
            continue
        if prev["shares"] > 0:
            change_pct = (pos["shares"] - prev["shares"]) / prev["shares"] * 100
            if change_pct >= _ADD_TRIM_THRESHOLD_PCT:
                adds.append(_row(cusip, pos, change_pct=round(change_pct, 1)))
            elif change_pct <= -_ADD_TRIM_THRESHOLD_PCT:
                trims.append(_row(cusip, pos, change_pct=round(change_pct, 1)))
    for cusip, pos in previous.items():
        if not isinstance(pos, dict):
            continue
        if cusip not in latest:
            exits.append(_row(cusip, pos, shares_sold=pos["shares"]))

    return {
        "manager": match,
        "source": "SEC EDGAR 13F-HR",
        "latest_quarter_filed": recent[0]["filingDate"],
        "previous_quarter_filed": recent[1]["filingDate"] if len(recent) > 1 else None,
        "positions_held": len(latest),
        "new_positions": sorted(new_positions, key=lambda r: -r["shares"]),
        "exits": sorted(exits, key=lambda r: -r["shares_sold"]),
        "adds": sorted(adds, key=lambda r: -r["change_pct"]),
        "trims": sorted(trims, key=lambda r: r["change_pct"]),
        "note": (
            "13F positions are filed up to 45 days after quarter end — this is a "
            "conviction/accumulation signal, never a timing signal."
        ),
    }


def _accumulated_usd(row: dict[str, Any]) -> float:
    """Dollars a manager ADDED to a name this quarter.

    A new position contributes its whole reported value; an add contributes only
    the incremental slice implied by ``change_pct`` (shares grew by g, so the
    new money is value × g/(1+g)). Ranking on the raw position value instead
    would put a 2% top-up of a mega-position above a full-size new buy, which is
    backwards for an accumulation-first funnel.
    """
    value = float(row.get("value") or 0.0)
    change_pct = row.get("change_pct")
    if change_pct is None:
        return value  # new position — all of it is newly accumulated
    try:
        growth = float(change_pct) / 100.0
    except (TypeError, ValueError):
        return 0.0
    if growth <= 0:
        return 0.0
    return value * growth / (1.0 + growth)


@cached(key_func=lambda: "sec_edgar:13f_universe", ttl=86400)
@log_exceptions()
def get_13f_universe() -> list[str]:
    """Tickers institutional managers are ACCUMULATING (new buys + meaningful
    adds across the curated 13F set, ranked by how many managers touched the
    name). The scanner's institutional universe producer — replaces the
    scraped media/guru feed (Roadmap 5.1)."""
    counts: dict[str, int] = {}
    dollars: dict[str, float] = {}
    managers = _managers_13f()
    failures = 0
    for manager in managers:
        try:
            diff = get_13f_diff(manager)
        except Exception:
            failures += 1
            continue
        if not isinstance(diff, dict) or diff.get("status") == "unavailable":
            failures += 1
            continue
        for row in (diff.get("new_positions") or []) + (diff.get("adds") or []):
            ticker = row.get("ticker")
            if ticker:
                counts[ticker] = counts.get(ticker, 0) + 1
                dollars[ticker] = dollars.get(ticker, 0.0) + _accumulated_usd(row)
    if managers and failures == len(managers):
        # Raise (never cache []) so a network blip doesn't cache-dark this source
        # for a full TTL — the scanner's try/except treats it as source-empty.
        raise RuntimeError("13F universe: every tracked manager fetch failed")
    # Cross-manager confirmation first, then DOLLARS accumulated — the ticker
    # itself is only the last-resort determinism tiebreak.
    #
    # Alphabetical was the *effective* selector, not a tiebreak: measured
    # 2026-07-28, only 17 of 108 accumulated names were touched by more than one
    # manager, so 23 of the 40 slots were filled by sorting 91 single-manager
    # names A→Z and cutting at "CSX" — 68 names excluded for their spelling. The
    # names that reached the funnel were not the ones institutions bought most.
    ranked = sorted(counts, key=lambda t: (-counts[t], -dollars.get(t, 0.0), t))
    return ranked[:_13F_UNIVERSE_CAP]


@log_exceptions()
def get_institutional_moves(manager: str | None = None) -> dict[str, Any]:
    """13F diffs for one manager, or a compact cross-manager report."""
    if manager:
        return get_13f_diff(manager)
    reports = {}
    for name in _managers_13f():
        diff = get_13f_diff(name)
        if not isinstance(diff, dict) or diff.get("status") == "unavailable":
            continue
        reports[name] = {
            "latest_quarter_filed": diff.get("latest_quarter_filed"),
            "new_positions": [
                {"ticker": r.get("ticker"), "issuer": r["issuer"]}
                for r in (diff.get("new_positions") or [])[:10]
            ],
            "exits": [
                {"ticker": r.get("ticker"), "issuer": r["issuer"]}
                for r in (diff.get("exits") or [])[:10]
            ],
            "adds": [
                {"ticker": r.get("ticker"), "issuer": r["issuer"], "change_pct": r.get("change_pct")}
                for r in (diff.get("adds") or [])[:10]
            ],
            "trims": [
                {"ticker": r.get("ticker"), "issuer": r["issuer"], "change_pct": r.get("change_pct")}
                for r in (diff.get("trims") or [])[:10]
            ],
        }
    if not reports:
        return unavailable("SEC EDGAR", "no 13F data could be fetched for any tracked manager")
    try:
        universe = get_13f_universe()
    except Exception:
        universe = []
    return {
        "source": "SEC EDGAR 13F-HR",
        "managers": reports,
        "accumulation_universe": universe,
        "note": "Diffs are latest vs previous 13F-HR; filed up to 45 days after quarter end.",
    }
