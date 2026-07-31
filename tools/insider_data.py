"""
Insider Trading & Short Interest Tool

Reads Yahoo Finance's insider-transaction table, which serves BOTH venues but
with two different vocabularies:

  * US listings carry SEC/Form-4 derived wording — "Purchase", "Sale",
    "Stock Award(Grant)", "Stock Gift".
  * TSX/TSXV listings (.TO/.V/.CN) carry SEDI wording — "Acquisition in the
    public market", "Disposition in the public market", "Exercise of options",
    "Acquisition under a purchase/ownership plan", and "Redemption, retraction,
    cancelation, repurchase".

Those SEDI phrases share no keyword with the US ones, which is why a
substring test for "purchase"/"sale" reads a Canadian table as noise — or
worse, backwards: "Redemption, retraction, cancelation, repurchase" (an ISSUER
buyback, and the single most common row on a TSX table) contains "repurchase",
and "Disposition under a purchase/ownership plan" (a sale) contains "purchase".

So classification here is an ordered, most-specific-first rule table over the
description text, and it separates *conviction* (open-market buys and sells)
from *mechanics* (grants, option exercises, automatic ownership-plan accruals,
gifts) and from *issuer* buybacks. Only conviction rows drive the signal —
the same standard :mod:`tools.sec_edgar` applies to Form 4 codes.

Position matters too: the bare position "Issuer" IS the company filing about
its own buyback, not an insider. "Director of Issuer" and "Senior Officer of
Issuer" are real insiders — so this must be an exact match, never a substring.
"""
import re
from typing import Any

import pandas as pd
import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.tool_errors import unavailable

# Venues whose insider filings are Canadian (SEDI), not SEC Form 4.
_CANADIAN_SUFFIXES = (".TO", ".V", ".VN", ".CN", ".NE")

# ---------------------------------------------------------------------------
# Transaction taxonomy
# ---------------------------------------------------------------------------
# (regex, code, signal, description). ORDER IS LOAD-BEARING: the buyback and
# ownership-plan rules must precede the generic purchase/sale rules, because
# their text contains "repurchase" / "purchase" and would otherwise be read as
# an insider buy. `code` mirrors Form 4 coding so results interoperate with
# tools.sec_edgar (P = open-market buy, S = open-market sale, M = exercise,
# A = grant, G = gift), plus B = issuer buyback and PL = ownership plan.
_TX_RULES: list[tuple[str, str, str, str]] = [
    (r"redemption|retraction|cancel+ation|repurchase",
     "B", "BUYBACK", "Issuer redemption/repurchase"),
    (r"(acquisition|disposition).*(purchase\s*/\s*ownership|ownership\s+plan|purchase\s+plan)",
     "PL", "PLAN", "Automatic purchase/ownership plan"),
    (r"gift",
     "G", "GIFT", "Gift"),
    (r"exercise",
     "M", "COMP", "Option/rights exercise"),
    (r"grant|award|compensation for services",
     "A", "COMP", "Grant / compensation"),
    (r"acquisition in the public market",
     "P", "BUY", "Open-market purchase"),
    (r"disposition in the public market",
     "S", "SELL", "Open-market sale"),
    (r"purchase|\bbuy\b|\bbought\b",
     "P", "BUY", "Open-market purchase"),
    (r"\bsale\b|\bsell\b|\bsold\b|disposition",
     "S", "SELL", "Open-market sale"),
]

_CONVICTION = ("BUY", "SELL")


def classify_insider_text(text: Any) -> dict[str, str]:
    """Classify an insider-transaction description into a Form-4-style code.

    Returns ``{code, signal, description, direction}`` where ``signal`` is one
    of BUY, SELL, COMP, PLAN, BUYBACK, GIFT, UNKNOWN. Only BUY and SELL are
    conviction signals; everything else is mechanics and must never be counted
    as insider sentiment.
    """
    raw = str(text or "").strip()
    low = raw.lower()
    if not low:
        return {"code": "", "signal": "UNKNOWN", "description": "Unspecified", "direction": ""}

    for pattern, code, signal, desc in _TX_RULES:
        if re.search(pattern, low):
            direction = ""
            if signal == "PLAN":
                direction = "acquire" if "acquisition" in low else "dispose"
            elif signal == "BUY":
                direction = "acquire"
            elif signal == "SELL":
                direction = "dispose"
            return {"code": code, "signal": signal, "description": desc, "direction": direction}

    return {"code": "", "signal": "UNKNOWN", "description": "Unrecognized", "direction": ""}


def _classify_insider_transaction(transaction_text) -> str:
    """Legacy three-way label: BUY / SELL / UNKNOWN.

    BUY and SELL now mean *open-market conviction only*. Grants, option
    exercises, ownership-plan accruals, gifts and issuer buybacks all collapse
    to UNKNOWN here — they are not insider sentiment. Use
    :func:`classify_insider_text` for the full coding.
    """
    signal = classify_insider_text(transaction_text)["signal"]
    return signal if signal in _CONVICTION else "UNKNOWN"


def is_canadian_listing(symbol: str) -> bool:
    """True for TSX/TSXV/CSE/NEO symbols, whose insiders file on SEDI not EDGAR."""
    return str(symbol or "").upper().strip().endswith(_CANADIAN_SUFFIXES)


# ---------------------------------------------------------------------------
# Row readers — yfinance renamed these columns; tolerate both spellings
# ---------------------------------------------------------------------------
# Current schema: Shares, Value, URL, Text, Insider, Position, Start Date,
# Ownership. The older schema used "Insider Trading" for the name and put the
# description in "Transaction" (now blank). Reading only the old names is why
# every row came back insider="Unknown", type="UNKNOWN".
def _first(row, *names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            val = row[n]
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            return val
    return default


def _num(value) -> float | None:
    """Coerce to a finite float, or None. NaN never escapes this function —
    a bare NaN is not valid JSON and poisons every downstream consumer."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _row_date(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value or "").strip()
    return text[:10] if text and text.lower() != "nat" else ""


def _normalize_rows(insider_df, limit: int = 250) -> list[dict[str, Any]]:
    """Flatten the yfinance insider table into classified, JSON-safe rows."""
    rows: list[dict[str, Any]] = []
    for _, row in insider_df.head(limit).iterrows():
        text = _first(row, "Text", "Transaction", default="")
        coding = classify_insider_text(text)
        position = str(_first(row, "Position", default="") or "").strip()
        rows.append({
            "date": _row_date(_first(row, "Start Date", "Date", default="")),
            "owner": str(_first(row, "Insider", "Insider Trading", default="") or "Unknown")[:60],
            "owner_title": position,
            "is_issuer": position.lower() == "issuer",  # exact: "Director of Issuer" is an insider
            "text": str(text or "")[:120],
            "code": coding["code"],
            "signal": coding["signal"],
            "code_description": coding["description"],
            "shares": _num(_first(row, "Shares")),
            "value": _num(_first(row, "Value")),
        })
    return rows


def _within_window(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    # Rows with no parsable date are kept — dropping them would silently shrink
    # the evidence base without saying so.
    return [r for r in rows if not r["date"] or r["date"] >= cutoff]


# ---------------------------------------------------------------------------
# Detailed activity — venue-neutral counterpart to sec_edgar.get_form4_activity
# ---------------------------------------------------------------------------
@cached(key_func=lambda symbol, days=90: f"insider_detail:{symbol.upper()}:{days}", ttl=6 * 3600)
@log_exceptions()
def get_detailed_insider_activity(symbol: str, days: int = 90) -> dict[str, Any]:
    """Insider activity with true buy/sell coding for ANY venue, via Yahoo.

    This is the Canadian (and other non-SEC) counterpart to
    :func:`tools.sec_edgar.get_form4_activity`, and returns the same contract —
    open-market buys/sells separated from compensation mechanics, per-owner
    aggregates, dollar values, and cluster-buy detection — so a TSX name can be
    analysed to the same depth as a US one instead of coming back empty.
    """
    symbol = str(symbol or "").upper().strip()
    canadian = is_canadian_listing(symbol)
    venue = "SEDI / Canadian regulatory filings" if canadian else "SEC Form 4"

    try:
        ticker = yf.Ticker(symbol)
        insider_df = ticker.insider_transactions
    except Exception as e:
        return unavailable("Yahoo insider table", f"fetch failed: {e}", symbol=symbol)

    if insider_df is None or len(insider_df) == 0:
        return {
            "symbol": symbol,
            "source": f"Yahoo Finance insider table ({venue} vocabulary)",
            "window_days": days,
            "transactions": [],
            "summary": {"open_market_buys": 0, "open_market_sells": 0},
            "note": (
                f"No insider transactions published for {symbol}. This is a COVERAGE gap, "
                "not evidence of insider inactivity — Yahoo carries no insider table for "
                "this listing at all. Do not report it as 'no insider activity'."
            ),
            "coverage": "none",
        }

    try:
        all_rows = _normalize_rows(insider_df)
    except Exception as e:
        return unavailable("Yahoo insider table", f"parse failed: {e}", symbol=symbol)

    rows = _within_window(all_rows, days)

    insiders = [r for r in rows if not r["is_issuer"]]
    buybacks = [r for r in rows if r["is_issuer"]]

    buys = [r for r in insiders if r["signal"] == "BUY"]
    sells = [r for r in insiders if r["signal"] == "SELL"]
    comp = [r for r in insiders if r["signal"] == "COMP"]
    plan = [r for r in insiders if r["signal"] == "PLAN"]
    gifts = [r for r in insiders if r["signal"] == "GIFT"]

    buy_value = round(sum(r["value"] or 0 for r in buys), 2)
    sell_value = round(sum(r["value"] or 0 for r in sells), 2)
    currency = "CAD" if canadian else "USD"

    from tools.sec_edgar import detect_cluster_buys
    cluster = detect_cluster_buys([
        {"code": r["code"], "date": r["date"], "owner": r["owner"], "value": r["value"] or 0}
        for r in buys if r["date"]
    ])

    result = {
        "symbol": symbol,
        "source": f"Yahoo Finance insider table ({venue} vocabulary)",
        "currency": currency,
        "window_days": days,
        "rows_in_window": len(rows),
        "transactions": [
            {k: v for k, v in r.items() if k != "is_issuer"}
            for r in insiders[:25]
        ],
        "summary": {
            "open_market_buys": len(buys),
            "open_market_sells": len(sells),
            "open_market_buy_value": buy_value,
            "open_market_sell_value": sell_value,
            "net_open_market_value": round(buy_value - sell_value, 2),
            "distinct_open_market_buyers": len({r["owner"] for r in buys}),
            "distinct_open_market_sellers": len({r["owner"] for r in sells}),
            "compensation_transactions": len(comp),
            "ownership_plan_transactions": len(plan),
            "gift_transactions": len(gifts),
            "value_units": f"{currency} (native listing currency)",
            "coding_note": (
                "Only open-market purchases and sales are conviction signals. Option "
                "exercises, grants/compensation, automatic purchase-ownership-plan "
                "accruals and gifts are mechanics and are EXCLUDED from the buy/sell "
                "counts. Issuer redemptions/repurchases are the company buying its own "
                "stock — reported separately, never as an insider buy."
            ),
        },
        "issuer_buybacks": {
            "transactions": len(buybacks),
            "shares": round(sum(r["shares"] or 0 for r in buybacks), 2),
            "value": round(sum(r["value"] or 0 for r in buybacks), 2),
            "note": "Issuer activity (buyback/redemption) — a capital-allocation signal, not insider conviction.",
        },
        "cluster": cluster,
        "coverage": "full",
    }

    if not rows:
        oldest = max((r["date"] for r in all_rows if r["date"]), default="")
        result["note"] = (
            f"No insider transactions in the last {days} days"
            + (f" (most recent filing: {oldest})" if oldest else "")
            + ". The table exists and was read — this IS a real answer, not a data gap."
        )
    elif not buys and not sells:
        result["note"] = (
            f"{len(rows)} filings in the last {days} days, but NONE were open-market buys "
            "or sales — all were compensation mechanics, plan accruals or issuer buybacks. "
            "No insider conviction signal either way."
        )
    return result


# ---------------------------------------------------------------------------
# Combined insider + short interest (scanner / deep-dive entry point)
# ---------------------------------------------------------------------------
@cached(key_func=lambda symbol: f"insider_short:{symbol.upper()}")
@log_exceptions()
def get_insider_and_short_data(symbol: str) -> dict:
    """
    Get insider trading activity and short interest data.
    """
    try:
        canadian = is_canadian_listing(symbol)
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Short Interest Data
        short_ratio = info.get('shortRatio')  # Days to cover
        short_percent = info.get('shortPercentOfFloat')

        short_signal = None
        if short_percent:
            if short_percent > 0.20:  # >20% shorted
                short_signal = "⚠️ Heavily shorted - risk signal; squeeze unconfirmed without borrow/utilization and days-to-cover context"
            elif short_percent > 0.10:  # >10% shorted
                short_signal = "🔶 Elevated short interest - investigate short thesis and covering mechanics"
            else:
                short_signal = "✅ Normal short interest"
        elif canadian:
            # TSX short positions are reported bi-monthly by IIROC/CIRO and are not
            # carried in this feed. Saying "data not available" without the reason
            # reads as "checked, nothing there".
            short_signal = (
                "❔ Short % of float is NOT published for TSX/TSXV listings in this feed "
                "(Canadian short positions are reported bi-monthly by CIRO). Absence here "
                "is a coverage gap, not a low-short-interest signal."
            )

        # Insider Holdings
        insider_percent = info.get('heldPercentInsiders')
        institutional_percent = info.get('heldPercentInstitutions')

        # Recent insider transactions, correctly coded. Scanning a window rather
        # than the top 5 rows matters on TSX tables, where the most recent rows
        # are routinely a block of issuer buybacks or director grants that carry
        # no conviction at all.
        insider_transactions: list[dict[str, Any]] = []
        buys = sells = 0
        buy_value = sell_value = 0.0
        window_rows = 0
        try:
            insider_df = ticker.insider_transactions
            if insider_df is not None and len(insider_df) > 0:
                rows = _within_window(_normalize_rows(insider_df), 90)
                window_rows = len(rows)
                for r in rows:
                    if r["is_issuer"]:
                        continue
                    if r["signal"] == "BUY":
                        buys += 1
                        buy_value += r["value"] or 0
                    elif r["signal"] == "SELL":
                        sells += 1
                        sell_value += r["value"] or 0
                    if len(insider_transactions) < 5:
                        insider_transactions.append({
                            "insider": r["owner"][:30],
                            "type": r["signal"] if r["signal"] in _CONVICTION else "UNKNOWN",
                            "activity": r["code_description"],
                            "date": r["date"],
                            "shares": r["shares"] if r["shares"] is not None else "N/A",
                            "value": r["value"] if r["value"] is not None else "N/A",
                        })
        except Exception:
            pass

        # Sentiment from conviction rows only, weighted by dollars — one $10M sale
        # is not offset by three token purchases.
        if buys or sells:
            net = buy_value - sell_value
            if buys > sells and net >= 0:
                insider_signal = "🟢 Insiders BUYING recently"
            elif sells > buys and net <= 0:
                insider_signal = "🔴 Insiders SELLING recently"
            elif net > 0:
                insider_signal = "🟢 Insiders BUYING recently"
            elif net < 0:
                insider_signal = "🔴 Insiders SELLING recently"
            else:
                insider_signal = "⚪ Mixed insider activity"
        elif window_rows:
            insider_signal = (
                f"⚪ No open-market insider buys or sells in 90d ({window_rows} filings, "
                "all grants/exercises/plan accruals or issuer buybacks)"
            )
        else:
            insider_signal = "No recent insider transactions found"

        return {
            "symbol": symbol.upper(),
            "short_interest": {
                "short_percent_of_float": f"{short_percent*100:.1f}%" if short_percent else "N/A",
                "days_to_cover": f"{short_ratio:.1f} days" if short_ratio else "N/A",
                "signal": short_signal or "Data not available",
                "mechanics_note": (
                    "Short interest is not bullish by itself. Confirm days to cover, borrow cost, utilization, "
                    "and a catalyst before treating it as squeeze fuel."
                )
            },
            "insider_holdings": {
                "insider_ownership": f"{insider_percent*100:.1f}%" if insider_percent else "N/A",
                "institutional_ownership": f"{institutional_percent*100:.1f}%" if institutional_percent else "N/A"
            },
            "recent_insider_activity": insider_transactions[:3] if insider_transactions else "No data",
            "insider_signal": insider_signal,
            "open_market_summary": {
                "buys": buys,
                "sells": sells,
                "buy_value": round(buy_value, 2),
                "sell_value": round(sell_value, 2),
                "net_value": round(buy_value - sell_value, 2),
                "currency": "CAD" if canadian else "USD",
                "window_days": 90,
            },
            "filings_source": "SEDI (Canadian)" if canadian else "SEC Form 4 (US)",
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol}


if __name__ == "__main__":
    print(get_insider_and_short_data("GME"))
