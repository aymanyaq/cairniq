from typing import Any

import pandas as pd
import yfinance as yf

from agent.utils import safe_print
from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.fmp_api import _fmp_get as _fmp_request


@log_exceptions()
def _fmp_get(endpoint: str) -> list:
    """Fetch a list from an FMP endpoint. Returns [] on failure.

    Delegates to the shared helper in tools.fmp_api, which resolves the key
    through credential_manager on EVERY call and rotates to the secondary key
    on a 429.

    This module used to issue its own request against a module-level
    ``FMP_KEY = os.environ.get("FMP_API_KEY", "")`` read AT IMPORT TIME. Three
    things followed from that, none of them visible from here:

      - the key never rotated, so a single 429 on the primary key took this
        scanner out for the remaining life of the process;
      - a rate limit was never reported back to credential_manager, so the
        cooldown every other FMP caller relies on was never started by this one;
      - whether there was a key at all depended on whether the environment
        happened to be loaded before this module was first imported.

    Returning ``[]`` on failure is kept: callers here treat an empty list as
    "FMP unavailable" and fall back to yfinance.
    """
    data, err = _fmp_request(endpoint, timeout=8)
    if err or not isinstance(data, list):
        return []
    return data


@cached(key_func=lambda market_index="SPY": f"intraday_movers:{market_index.upper()}")
@log_exceptions()
def scan_intraday_movers(market_index: str = "SPY") -> dict[str, Any]:
    """
    Scans for intraday movers, unusual volume, and volatility.
    Uses live FMP APIs for biggest gainers/losers/most-active
    with yfinance broad-market context as fallback.
    """
    try:
        # ── 1. Broad Market Context (SPY, QQQ, IWM, VIX) via yfinance ──
        tickers = ["SPY", "QQQ", "IWM", "^VIX"]
        data = yf.download(tickers, period="5d", interval="1d", progress=False, threads=False)

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data["Close"] if "Close" in data else data

        market_summary = []
        for sym in tickers:
            if sym in close.columns:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    curr = prices.iloc[-1]
                    prev = prices.iloc[-2]
                    change = ((curr - prev) / prev) * 100
                    market_summary.append(f"{sym}: {change:+.2f}%")

        # ── 2. Sector Performance (FMP) ──────────────────────────────
        sector_summary = []
        sectors = _fmp_get("sector-performance-snapshot")
        for s in sectors[:11]:
            name = s.get("sector", "")
            chg = s.get("changesPercentage", "")
            if name and chg is not None:
                sector_summary.append(f"{name}: {chg:+.2f}%" if isinstance(chg, (int, float)) else f"{name}: {chg}")

        # ── 3. Live Gainers / Losers / Most Active (FMP) ────────────
        gainers = _fmp_get("biggest-gainers")
        losers = _fmp_get("biggest-losers")
        actives = _fmp_get("most-actives")

        def _format_movers(items, limit=8):
            result = []
            for item in items[:limit]:
                sym = item.get("symbol", "")
                price = item.get("price", 0)
                chg_pct = item.get("changesPercentage", 0)
                name = item.get("name", "")
                result.append({
                    "symbol": sym,
                    "name": name[:30] if name else "",
                    "price": f"${price:.2f}" if isinstance(price, (int, float)) else str(price),
                    "change": f"{chg_pct:+.2f}%" if isinstance(chg_pct, (int, float)) else str(chg_pct),
                })
            return result

        top_gainers = _format_movers(gainers)
        top_losers = _format_movers(losers)
        most_active = _format_movers(actives)

        # If FMP returned data, return the live results
        if top_gainers or top_losers or most_active:
            return {
                "market_status": " | ".join(market_summary),
                "sector_performance": sector_summary if sector_summary else "Unavailable",
                "top_gainers": top_gainers if top_gainers else "None",
                "top_losers": top_losers if top_losers else "None",
                "most_active": most_active if most_active else "None",
                "note": "Live data from FMP: biggest gainers, losers, and most-active across ALL sectors."
            }

        # ── 4. Fallback: yfinance broad watchlist if FMP unavailable ─
        watchlist = [
            "NVDA", "TSLA", "AMD", "META", "AMZN",       # Tech
            "XOM", "CVX", "GOLD", "NEM",                  # Energy/Commodities
            "LMT", "RTX", "NOC",                          # Defence
            "JPM", "GS", "UNH", "LLY",                    # Financials/Health
            "CAT", "DE", "FCX",                            # Industrials
        ]
        movers_data = yf.download(watchlist, period="5d", interval="1d", progress=False, threads=False)
        if isinstance(movers_data.columns, pd.MultiIndex):
            m_close = movers_data["Close"]
        else:
            m_close = movers_data["Close"]

        active_movers = []
        for sym in watchlist:
            if sym not in m_close.columns:
                continue
            prices = m_close[sym].dropna()
            if len(prices) < 2:
                continue
            curr = prices.iloc[-1]
            prev = prices.iloc[-2]
            pct = ((curr - prev) / prev) * 100
            if abs(pct) > 2.0:
                active_movers.append({
                    "symbol": sym,
                    "price": f"${curr:.2f}",
                    "change": f"{pct:+.2f}%",
                })
        active_movers.sort(key=lambda x: abs(float(x["change"].strip('%'))), reverse=True)

        return {
            "market_status": " | ".join(market_summary),
            "sector_performance": sector_summary if sector_summary else "Unavailable",
            "active_movers": active_movers if active_movers else "No major movers detected.",
            "note": "FMP live feeds unavailable — used yfinance fallback watchlist."
        }

    except Exception as e:
        return {"error": f"Scanner failed: {str(e)}"}

if __name__ == "__main__":
    safe_print(scan_intraday_movers())
