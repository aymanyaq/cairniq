"""Free-text statement → draft holding rows, for the accounts that cannot sync.

Questrade and Alpaca arrive over an API. Everything else — a workplace pension
portal, a bank that only emails a PDF, a broker with no API, a spouse's account
read off a screen — arrives as text a human has to retype into the editor one
field at a time. This module takes that pasted text and drafts rows from it.

WHERE THIS SITS, and why the boundary is the whole design:

    pasted text → [this module] → draft rows → the editor's table → user edits
                                                                  → Save Changes
                                                                  → my_portfolio.csv

It does not write. Nothing here touches the CSV, the graph, or the cache. The
draft rows land in the editor as unsaved inputs and the user still presses the
same Save button they would have pressed after typing them by hand. That is not
timidity about an extra endpoint — it is the only structure in which a model may
touch a ledger at all. A model that can write positions can write a position
nobody holds, and a portfolio is the input to every other engine in this app.

THREE GUARDS, each paid for by a failure already in this codebase's history:

* **Citation or discard.** Every row must quote the span of the input it came
  from, and that span is checked against the pasted text HERE, not trusted. A
  row whose quote is not in the input is dropped, never repaired — the same rule
  ``observation_consolidation`` applies to its evidence ids, for the same reason:
  a plausible holding is indistinguishable from a real one once it is in a table.
* **Numbers are echoed, never computed.** Each extracted number must appear as
  digits in its own quoted span. One that does not is blanked and flagged rather
  than shown — the deep-reasoning pass once synthesised a portfolio total that
  was never in its context, and a share count invented the same way would be
  silently wrong forever, in a field nobody re-reads.
* **Cost basis is not market price.** A statement shows both, and they sit in
  adjacent columns. Taking the market price as the entry price pins the position
  at 0% return for as long as it is held, which reads as a fact rather than as
  an error. When no per-unit cost is stated, the field comes back empty with a
  warning; this module will not divide a book-cost total by a quantity to
  manufacture one.

The failure modes are reported, never swallowed: a provider that is not
configured, an input that is too long, and a model that returned nothing all
produce distinct ``reason`` text. "Found nothing" and "could not look" are
different answers, and only one of them is about the user's text.
"""
import json
import logging
import re
from typing import Any

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

# The paste box, bounded. Long enough for a multi-account statement dump
# (a 60-position book with headers and footers runs ~8k), short enough that one
# paste cannot become a five-figure-token prompt. Rejected rather than truncated:
# a silent truncation drops the holdings at the bottom of the statement and there
# is no way for the user to tell which ones those were.
MAX_INPUT_CHARS = 20000

# Below this there is nothing to extract from, and calling a model on it hands it
# the one setup that has produced invented content here before — an almost-empty
# context with an output schema to fill.
MIN_INPUT_CHARS = 12

# Rows kept from one paste. A statement with more positions than this is a CSV
# export job, not a paste job.
MAX_ROWS = 60

# Mirrors tools.memory.SUPPORTED_BASE_CURRENCIES, imported lazily below so this
# module stays cheap to import from a request handler.
_FALLBACK_CURRENCIES = {"USD", "CAD", "EUR", "GBP", "AUD", "JPY"}

# Brokers this app syncs over an API. A manual row filed under one of these
# account names is not merely redundant: load_portfolio() drops CSV rows whose
# account matches a successfully-synced Questrade or Alpaca account, so the typed
# row would vanish on the next load with no error anywhere. This feature exists
# for the accounts that CANNOT sync, so an extracted account naming a broker that
# can is worth saying out loud before it is saved.
_SYNCED_BROKERS = ("QUESTRADE", "ALPACA")

# A ticker, a fund code, or CASH. Deliberately narrow: anything with a space in
# it is a description the model lifted out of the name column, not a symbol.
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-:]{0,14}$")

# Column headings that are shaped exactly like tickers. A portal copy-paste puts
# the header row in the same one-value-per-line stream as the data, so "Symbol"
# and "Currency" arrive looking like any other line — and both sail through the
# regex above. A row built from the header would carry a real-looking quote (the
# header IS in the pasted text) and would be caught by nothing else here.
_HEADER_WORDS = {
    "SYMBOL", "CURRENCY", "QTY", "QUANTITY", "SHARES", "UNITS", "PRICE", "COST",
    "TOTAL", "SUBTOTAL", "ACCOUNT", "TYPE", "NAME", "VALUE", "BALANCE", "EQUITY",
    "MARKET", "AVG", "AVERAGE", "BOOK", "GAIN", "LOSS", "DESCRIPTION", "SECURITY",
    "HOLDING", "HOLDINGS", "POSITION", "TICKER", "PORTFOLIO", "N/A", "NA",
}

# Thousands separators and the spacing used as one. Stripped from both sides
# before a number is looked for, so "1,234.56" in the statement matches 1234.56
# in the extraction.
_SEPARATORS_RE = re.compile(r"[,\s_'’ ]")

_SYSTEM_PROMPT = (
    "You extract investment holdings from the text of a brokerage or pension statement "
    "that a user has pasted. The text may be a table that lost its formatting, an email, "
    "a PDF copy-paste, a web portal copied to the clipboard, or hand-typed notes.\n\n"
    "FORMAT: a copied table usually arrives as ONE VALUE PER LINE, with no row "
    "boundaries — the column headings appear first, then every cell of the first "
    "position in that same order, then every cell of the second, and so on. Read the "
    "heading order first and use it to work out which line is which field. A position's "
    "`source_text` is then the whole run of lines belonging to it, quoted together.\n\n"
    "Return ONE object per position actually listed in the text. Rules:\n"
    "1. Extract ONLY what the text states. Never add a holding that is not there, and "
    "never fill a field from knowledge of the company or the market.\n"
    "2. `source_text` MUST be copied VERBATIM from the input — the line or fragment that "
    "position came from, character for character. It is checked against the input, and a "
    "row whose source_text is not found there is discarded.\n"
    "3. Every number you return must appear in that row's own source_text. Never compute, "
    "convert, total, or divide. If a value is not stated, use null.\n"
    "4. `purchase_price` is the per-unit COST BASIS. Statements print the cost basis and "
    "the current market price in ADJACENT columns and the wrong one is not detectable "
    "later, so choose by the heading:\n"
    "     TAKE  — 'Avg price', 'Average cost', 'Book cost per share', 'Cost basis', "
    "'Price paid'\n"
    "     NEVER — 'Symbol price', 'Last price', 'Market price', 'Current price', 'Close', "
    "'Market value'\n"
    "   If the text has only a market price and no cost, return null. If it states only a "
    "TOTAL book cost, return null — do not divide it by the quantity.\n"
    "5. `shares` is the quantity of units held. Take it from the quantity column ('QTY', "
    "'Units', 'Shares'), never from a '% of portfolio' column, which is a weight and not a "
    "count. A stated CASH BALANCE is a position: use the amount as the quantity and the "
    "symbol \"CASH\".\n"
    "6. `symbol` is the ticker or fund code as printed. If the text gives only a company "
    "name and no symbol, put the name in `name` and set `symbol` to null — do not recall "
    "the ticker from memory. Never return a column heading as a symbol.\n"
    "7. `account` is the account this position is listed UNDER. Many statements show a "
    "combined view: a list of accounts in one place and a single merged holdings table in "
    "another. In that case the table does not say which account each position is in, so "
    "return null — do NOT attach a position to an account because it was the only one "
    "named, or the first, or the one whose type seems to fit.\n"
    "8. `currency` is a 3-letter code, only if the text states it. Otherwise null.\n"
    "9. `asset_type` is \"Private\" ONLY for things with no public market (private equity, "
    "real estate, a pension entitlement). Everything else is \"Public\". A statement's own "
    "'Asset type' column usually means something else entirely ('Fixed income', 'US "
    "equity', 'Intl. equity') — that is a sector or class, not this field. Ignore it.\n"
    "10. Skip totals, subtotals, account summaries, allocation percentages, performance "
    "figures and disclaimer text. 'Total equity', 'Market value', 'Net deposits', 'Total "
    "P&L', 'Rate of return' and asset-mix percentages are NOT positions, however much they "
    "look like one. A cash balance listed among the holdings is the one exception, per "
    "rule 5.\n"
    "11. If the text contains no holdings, return an empty list. That is a correct answer; "
    "inventing a plausible portfolio to fill the output is the worst failure available.\n\n"
    "Return valid JSON and nothing else:\n"
    '{\n  "holdings": [\n    {\n      "symbol": "AAPL",\n      "name": "Apple Inc",\n'
    '      "shares": 30,\n      "purchase_price": 170.25,\n      "account": "RRSP",\n'
    '      "currency": "USD",\n      "asset_type": "Public",\n'
    '      "source_text": "AAPL Apple Inc 30 170.25 5107.50"\n    }\n  ]\n}'
)


def _supported_currencies() -> set[str]:
    try:
        from tools.memory import SUPPORTED_BASE_CURRENCIES
        return set(SUPPORTED_BASE_CURRENCIES)
    except Exception:
        return set(_FALLBACK_CURRENCIES)


def _normalize(text: str) -> str:
    """Casefold and collapse whitespace, for containment checks only.

    Statements lose their column alignment on the way through a clipboard, so a
    quote that is right about the content can be wrong about the run of spaces
    between two fields. Whitespace is the one difference allowed.
    """
    return " ".join(str(text or "").split()).casefold()


def _parse_holdings(raw: str) -> tuple[list[dict[str, Any]], bool]:
    """(holdings, readable) from a model response. Never raises.

    The second value is the difference between "the model answered, and the
    answer was no positions" and "the model's answer could not be read at all" —
    a long statement whose response hit the output cap lands in the second, and
    reporting it as the first would tell the user their own text was empty.
    """
    body = str(raw or "").strip()
    if body.startswith("```"):
        lines = body.split("\n")
        body = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(body)
    except Exception:
        # Some providers wrap the object in a sentence. Take the outermost
        # braces and try once more before giving up.
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            return [], False
        try:
            data = json.loads(body[start:end + 1])
        except Exception:
            return [], False
    holdings = data.get("holdings") if isinstance(data, dict) else data
    if not isinstance(holdings, list):
        return [], False
    return holdings, True


def _number_renderings(value: float) -> set[str]:
    """The digit strings a statement might have printed this number as."""
    forms = {f"{value:g}", f"{value:.2f}"}
    if value == int(value):
        forms.add(str(int(value)))
    trimmed = f"{value:.6f}".rstrip("0").rstrip(".")
    if trimmed:
        forms.add(trimmed)
    return {f for f in forms if f}


def _number_is_stated(value: float, source_text: str) -> bool:
    """Do this number's digits actually appear in the span it was taken from?

    Separator-insensitive on both sides, so 1234.56 matches "1,234.56" and 5000
    matches "5 000". The sign is ignored — a short position prints as (100) or
    -100 and neither is a reason to doubt the magnitude.
    """
    haystack = _SEPARATORS_RE.sub("", str(source_text or ""))
    return any(form in haystack for form in _number_renderings(abs(float(value))))


def _coerce_number(value: Any) -> float | None:
    """A number from the model, or None. Strings are accepted with separators."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if _is_finite(float(value)) else None
    cleaned = _SEPARATORS_RE.sub("", str(value)).replace("$", "").strip()
    # Accounting negatives: (1,234.00) is minus 1234.
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()")
    try:
        number = float(cleaned)
    except (TypeError, ValueError):
        return None
    if not _is_finite(number):
        return None
    return -number if negative else number


def _is_finite(number: float) -> bool:
    # NaN survives float() and json.dumps() emits it as a bare NaN token that no
    # strict JSON parser will read back — it has taken this app's market pulse
    # off the air for a day before. Screen it at the boundary.
    return number == number and number not in (float("inf"), float("-inf"))


def _validated(
    holdings: list[dict[str, Any]],
    source: str,
    default_account: str,
    default_currency: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the rows whose claims check out against the pasted text.

    Returns (rows, drop_reasons). A dropped row is reported by symbol and cause
    rather than counted, because "we found 9 of your 12 positions" is only
    actionable if the user can tell which three are missing.
    """
    haystack = _normalize(source)
    currencies = _supported_currencies()
    rows: list[dict[str, Any]] = []
    drops: list[str] = []
    seen: set[tuple[str, str]] = set()

    for item in holdings:
        if not isinstance(item, dict):
            drops.append("A row came back in an unreadable shape.")
            continue

        label = str(item.get("symbol") or item.get("name") or "unnamed").strip()[:40]
        quote = str(item.get("source_text") or "").strip()

        # Guard one: the quote has to be in the text. Everything below this line
        # is checked AGAINST the quote, so an unverified quote makes every other
        # check meaningless rather than merely weaker.
        if not quote:
            drops.append(f"{label}: no source line quoted.")
            continue
        if _normalize(quote) not in haystack:
            drops.append(f"{label}: quoted a line that is not in the pasted text.")
            continue

        symbol = str(item.get("symbol") or "").strip().upper()
        name = str(item.get("name") or "").strip()[:80]
        warnings: list[str] = []

        if not symbol:
            drops.append(
                f"{name or 'A position'}: no ticker in the text. "
                "Add the row by hand if you know the symbol."
            )
            continue
        if not _SYMBOL_RE.match(symbol):
            drops.append(f"{label}: '{symbol}' is not a usable ticker.")
            continue
        if symbol in _HEADER_WORDS:
            # See _HEADER_WORDS: a row built from the table's own heading quotes
            # real text and would pass every other check here.
            drops.append(f"'{symbol}' is a column heading, not a position.")
            continue
        # The symbol is a claim like any other, and it is the claim that decides
        # WHICH security this row is about.
        if symbol.casefold() not in _normalize(quote):
            drops.append(f"{symbol}: the ticker does not appear in its own quoted line.")
            continue

        shares = _coerce_number(item.get("shares"))
        if shares is not None and not _number_is_stated(shares, quote):
            # Blanked, not dropped, and not kept. The position is real — its
            # ticker was verified — but this number was not read off the text, so
            # showing it would launder a guess into a quantity. An empty field
            # the user fills is the honest form of "we could not read this".
            warnings.append(f"Quantity {shares:g} is not in the quoted line — enter it yourself.")
            shares = None
        if shares is None and not warnings:
            warnings.append("No quantity stated for this position.")

        price = _coerce_number(item.get("purchase_price"))
        if price is not None and not _number_is_stated(price, quote):
            warnings.append(f"Cost {price:g} is not in the quoted line — enter it yourself.")
            price = None
        if price is None:
            warnings.append(
                "No per-unit cost basis stated. Leave it empty and track this row by "
                "return %, or type the cost you paid."
            )
        elif price < 0:
            warnings.append("Cost basis read as negative — check it.")
            price = None

        currency = str(item.get("currency") or "").strip().upper()
        if currency and currency not in currencies:
            warnings.append(f"Currency '{currency}' is not one this app handles.")
            currency = ""
        if not currency:
            currency = default_currency
            if currency:
                warnings.append(f"Currency not stated — defaulted to {currency}.")

        account = str(item.get("account") or "").strip()[:60] or default_account
        if not account:
            warnings.append("No account named — set one before saving.")
        elif any(broker in account.upper() for broker in _SYNCED_BROKERS):
            # See _SYNCED_BROKERS: this row would be discarded on the next load,
            # silently, if that account also syncs.
            warnings.append(
                f"'{account}' names a broker this app syncs directly. A typed row in a "
                "synced account is dropped when the sync runs — file it under a name of "
                "its own."
            )

        asset_type = "Private" if str(item.get("asset_type") or "").strip().lower() == "private" else "Public"

        # One row per symbol-and-account. A statement that prints a position
        # twice (a summary section and a detail section) would otherwise double
        # the holding, and the second copy always looks as legitimate as the first.
        key = (symbol, account.casefold())
        if key in seen:
            drops.append(f"{symbol} in '{account or 'no account'}': listed more than once, kept the first.")
            continue
        seen.add(key)

        rows.append({
            "symbol": symbol,
            "name": name,
            "shares": shares,
            "purchase_price": price,
            "account": account,
            "currency": currency,
            "asset_type": asset_type,
            "source_text": quote[:300],
            "warnings": warnings,
        })

        if len(rows) >= MAX_ROWS:
            drops.append(f"Stopped at {MAX_ROWS} rows — paste the rest separately.")
            break

    return rows, drops


@log_exceptions()
def parse_statement_text(
    text: str,
    default_account: str = "",
    default_currency: str = "",
) -> dict[str, Any]:
    """Draft holding rows from pasted statement text. Never raises.

    ``default_account`` and ``default_currency`` fill only the fields the text
    itself does not state — they are the user's answer to "what am I pasting",
    not a licence to guess. Every row that uses one says so in its warnings.

    Returns a report whose ``reason`` is always populated, including on the
    paths that produce no rows. An empty result from "the model read your text
    and found no positions" and an empty result from "no model was reachable"
    are different claims, and a caller that cannot tell them apart will show the
    user the wrong one.
    """
    from agent.utils import get_fast_llm, llm_ready, safe_invoke

    report: dict[str, Any] = {
        "ok": False,
        "rows": [],
        "row_count": 0,
        "dropped": [],
        "reason": "",
    }

    body = str(text or "").strip()
    if len(body) < MIN_INPUT_CHARS:
        report["reason"] = "Nothing to read — paste the text from your statement first."
        return report
    if len(body) > MAX_INPUT_CHARS:
        report["reason"] = (
            f"That is {len(body):,} characters; the limit is {MAX_INPUT_CHARS:,}. "
            "Paste one account at a time — nothing was read, so nothing was silently cut."
        )
        return report

    ready, why = llm_ready()
    if not ready:
        report["reason"] = (
            f"No language model is configured, so the text cannot be read: {why}. "
            "Add positions with 'Allocate New Position' or upload a CSV."
        )
        return report

    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.memory import _content_to_str

    try:
        # The fast tier: this is extraction against a schema, not reasoning. The
        # cap has to clear MAX_ROWS records with their quoted source spans — a
        # response cut off mid-JSON is unreadable, and the whole paste is lost
        # rather than degraded. Keep these two in step if either moves.
        response = safe_invoke(get_fast_llm(max_tokens=8192), [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Statement text:\n\n{body}"),
        ])
    except Exception as e:
        # llm_ready() clearing is not the same as a client that BUILDS — a
        # provider selected without its model id passes the check above and
        # raises here. Reported as a clean, stated failure for the same reason
        # it is elsewhere in this codebase: the user is owed the difference
        # between "your text had nothing in it" and "we never looked".
        report["reason"] = f"The model could not be reached: {e}"
        logger.warning(f"Statement parse could not reach the model: {e}")
        return report

    holdings, readable = _parse_holdings(_content_to_str(response.content))
    if not readable:
        # A claim about the READER, so it must not come back as ok:True with an
        # empty list — the caller would render "no positions found in your text",
        # which is a claim about the user's statement and is not one we can make.
        report["reason"] = (
            "The model's response could not be read back. If the statement is long, "
            "paste one account at a time."
        )
        logger.warning("Statement parse: unreadable model response")
        return report

    default_currency = str(default_currency or "").strip().upper()
    if default_currency and default_currency not in _supported_currencies():
        default_currency = ""

    rows, drops = _validated(holdings, body, str(default_account or "").strip()[:60], default_currency)

    report["ok"] = True
    report["rows"] = rows
    report["row_count"] = len(rows)
    report["dropped"] = drops

    if rows:
        report["reason"] = f"Drafted {len(rows)} position(s) from the pasted text."
    elif drops:
        report["reason"] = (
            "Nothing could be verified against the text you pasted. "
            "Every candidate was discarded — see the reasons below."
        )
    else:
        report["reason"] = (
            "The text was read but no positions were found in it. "
            "It may be a summary or a performance report rather than a holdings list."
        )

    logger.info(f"Statement parse: {report['reason']}")
    return report
