from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.yf_utils import dividend_yield_display


def is_canadian_ticker(symbol: str) -> bool:
    """Check if the ticker is a Canadian stock (.TO, .VN, .CN)"""
    return symbol.upper().endswith(('.TO', '.VN', '.CN'))

@cached(key_func=lambda symbol: f"canadian_quote:{symbol.upper()}")
@log_exceptions()
def get_canadian_quote(symbol: str) -> dict[str, Any]:
    """Get real-time quote and fundamentals for a Canadian stock."""
    if not is_canadian_ticker(symbol):
        return {"error": "Not a valid Canadian ticker"}

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol,
            "company_name": info.get("longName", "Unknown"),
            "price": info.get("currentPrice", info.get("regularMarketPrice")),
            "currency": info.get("currency", "CAD"),
            "exchange": info.get("exchange", "TSX"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            # A LABELLED string, not a bare float. This payload goes straight
            # to the model, and `"dividend_yield": 0.32` names no unit — it is
            # 0.32% read as 32% by anything that assumes a fraction, which is
            # what every other reader in this codebase assumed.
            "dividend_yield": dividend_yield_display(info),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry")
        }
    except Exception as e:
        return {"error": str(e)}

@cached(key_func=lambda symbol: f"canadian_analyst:{symbol.upper()}")
@log_exceptions()
def get_canadian_analyst_estimates(symbol: str) -> dict[str, Any]:
    """Get Canadian analyst consensus from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "target_mean": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "number_of_analysts": info.get("numberOfAnalystOpinions", 0)
        }
    except Exception as e:
        return {"error": str(e)}

# Liquid TSX fallback universe (TSX-60-style large caps across sectors). Used ONLY
# when the Yahoo screener endpoint is unavailable — the primary path screens the
# whole exchange dynamically, so this list never gates what scan_tsx_movers can see.
_TSX_FALLBACK_UNIVERSE = [
    # Financials
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO", "MFC.TO", "SLF.TO",
    "IFC.TO", "POW.TO", "BN.TO", "BAM.TO", "FFH.TO",
    # Energy
    "CNQ.TO", "SU.TO", "CVE.TO", "IMO.TO", "ENB.TO", "TRP.TO", "PPL.TO",
    # Materials / Gold
    "NTR.TO", "TECK-B.TO", "ABX.TO", "AEM.TO", "K.TO", "WPM.TO", "FNV.TO", "CCO.TO",
    # Tech
    "SHOP.TO", "CSU.TO", "GIB-A.TO", "OTEX.TO", "BB.TO",
    # Industrials / Transport
    "CNR.TO", "CP.TO", "WCN.TO", "WSP.TO", "TFII.TO", "MG.TO",
    # Consumer / Telecom / Utilities
    "ATD.TO", "L.TO", "MRU.TO", "DOL.TO", "QSR.TO", "BCE.TO", "T.TO", "RCI-B.TO",
    "FTS.TO", "EMA.TO", "H.TO",
]


def _quote_is_sane(q: dict) -> bool:
    """Reject corrupt or internally-inconsistent screener quotes.

    The screener occasionally returns a split/stale ``previousClose`` against an
    already-adjusted price, which inflates ``regularMarketChangePercent`` into a
    fake double-digit "mover". We catch that by recomputing the move from price and
    previousClose: a *real* large move agrees with the reported %; an artifact does
    not. Magnitude alone is NEVER a reason to drop — a genuine −18% crash is kept,
    because its reported % matches (price − prevClose)/prevClose. This is a
    data-integrity guard, not an outlier filter."""
    price = q.get("regularMarketPrice")
    if not isinstance(price, (int, float)) or price <= 0:
        return False
    chg = q.get("regularMarketChangePercent")
    prev = q.get("regularMarketPreviousClose")
    if isinstance(chg, (int, float)) and isinstance(prev, (int, float)) and prev > 0:
        implied = (price - prev) / prev * 100
        # Allow 1 percentage point or 10% of the reported move, whichever is larger.
        if abs(implied - chg) > max(1.0, 0.10 * abs(chg)):
            return False
    return True


def _tsx_screen_lane(extra_filters: list, sort_field: str, sort_asc: bool, limit: int) -> list[dict]:
    """One Yahoo-screener query over the whole Toronto exchange. Raises on failure
    so the caller can fall back; returns a list of compact mover dicts."""
    import yfinance as yf

    base = [
        yf.EquityQuery("eq", ["exchange", "TOR"]),
        yf.EquityQuery("gt", ["dayvolume", 100_000]),   # liquidity floor
        yf.EquityQuery("gt", ["intradayprice", 1.0]),   # no sub-dollar names
    ]
    resp = yf.screen(
        yf.EquityQuery("and", base + extra_filters),
        sortField=sort_field, sortAsc=sort_asc, size=limit,
    )
    movers = []
    for q in (resp or {}).get("quotes", []):
        if not _quote_is_sane(q):
            continue
        sym = q.get("symbol", "")
        chg = q.get("regularMarketChangePercent")
        price = q.get("regularMarketPrice")
        movers.append({
            "symbol": sym,
            "name": (q.get("shortName") or q.get("longName") or "")[:30],
            "price": f"C${price:.2f}" if isinstance(price, (int, float)) else str(price),
            "change": f"{chg:+.2f}%" if isinstance(chg, (int, float)) else str(chg),
            "volume": q.get("regularMarketVolume"),
        })
    return movers


def _tsx_market_context() -> str:
    """TSX Composite + USD/CAD one-liner, best-effort."""
    import yfinance as yf
    parts = []
    for sym, label in (("^GSPTSE", "TSX Composite"), ("CAD=X", "USD/CAD")):
        try:
            fi = yf.Ticker(sym).fast_info
            last, prev = fi.get("lastPrice"), fi.get("previousClose")
            if last and prev:
                parts.append(f"{label}: {((last - prev) / prev) * 100:+.2f}%")
        except Exception:
            continue
    return " | ".join(parts) or "Unavailable"


@cached(key_func=lambda: "tsx_movers")
@log_exceptions()
def scan_tsx_movers() -> dict[str, Any]:
    """Market-wide TSX movers: top gainers, top losers, and large-cap most active.

    Primary path screens the ENTIRE Toronto exchange via the Yahoo screener (with
    liquidity floors), so it is not limited to any curated list. If that endpoint
    fails, falls back to scanning a static liquid TSX-60-style universe via batch
    download.
    """
    import yfinance as yf

    context = _tsx_market_context()

    # ── Primary: dynamic exchange-wide screen ────────────────────────────
    try:
        large_cap = [yf.EquityQuery("gt", ["intradaymarketcap", 2_000_000_000])]
        gainers = _tsx_screen_lane([], "percentchange", False, 8)
        losers = _tsx_screen_lane([], "percentchange", True, 8)
        most_active = _tsx_screen_lane(large_cap, "dayvolume", False, 8)
        if gainers or losers or most_active:
            return {
                "market_status": context,
                "top_gainers": gainers or "None",
                "top_losers": losers or "None",
                "most_active_large_cap": most_active or "None",
                "note": (
                    "Live exchange-wide TSX screen (Toronto exchange, volume>100k, "
                    "price>C$1; most-active lane is market cap >C$2B)."
                ),
            }
    except Exception:
        pass  # fall through to the static-universe fallback

    # ── Fallback: static liquid universe via batch download ──────────────
    try:
        data = yf.download(_TSX_FALLBACK_UNIVERSE, period="5d", interval="1d",
                           progress=False, threads=False)
        close = data["Close"] if "Close" in data else data
        movers = []
        for sym in _TSX_FALLBACK_UNIVERSE:
            if sym not in close.columns:
                continue
            prices = close[sym].dropna()
            if len(prices) < 2:
                continue
            curr, prev = prices.iloc[-1], prices.iloc[-2]
            pct = ((curr - prev) / prev) * 100
            movers.append({"symbol": sym, "price": f"C${curr:.2f}", "change": f"{pct:+.2f}%"})
        movers.sort(key=lambda m: abs(float(m["change"].rstrip("%"))), reverse=True)
        return {
            "market_status": context,
            "top_gainers": [m for m in movers if m["change"].startswith("+")][:8] or "None",
            "top_losers": [m for m in movers if m["change"].startswith("-")][:8] or "None",
            "note": (
                "Yahoo TSX screener unavailable — scanned a static liquid TSX-60-style "
                "universe instead (large caps only; small-cap movers not covered)."
            ),
        }
    except Exception as e:
        return {"error": f"TSX scanner failed: {e}"}


@cached(key_func=lambda: "canadian_market_news")
@log_exceptions()
def get_canadian_market_news() -> str:
    """Canada-wide macro/market news for the report's Canadian section: the Bank of
    Canada and rates, the loonie (USD/CAD), and the TSX's dominant energy & mining
    sectors. This is the macro backdrop for the section — NOT per-ticker mover
    catalysts (a price screen surfaces idiosyncratic single-stock moves that rarely
    have a findable catalyst, which is why we no longer chase one per ticker)."""
    from tools.web_search import search_news
    query = (
        "Canada stock market TSX today Bank of Canada interest rate "
        "Canadian dollar loonie energy oil mining sector"
    )
    try:
        return search_news(query, max_results=6)
    except Exception as e:
        return f"Error fetching Canadian market news: {e}"
