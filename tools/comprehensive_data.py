
"""
Comprehensive Data Tools
Provides deep-dive data on Earnings, Insider Trading, and Institutional Ownership.
"""
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions


@cached(key_func=lambda symbol: f"earnings_cal:{symbol.upper()}")
@log_exceptions()
def get_earnings_calendar(symbol: str) -> dict[str, Any]:
    """
    Get next earnings date and recent estimates.
    """
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar

        if cal is None:
            return {"symbol": symbol, "error": "No earnings calendar found"}

        # Format the data
        # yfinance calendar is often a dict with keys like 'Earnings Date', 'Earnings High', etc.
        # Ensure values are serializable

        results = {"symbol": symbol}

        # Handle different yfinance versions/outputs
        if isinstance(cal, dict):
            for k, v in cal.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                     # Often dates are datetime objects
                     val = v[0]
                     if isinstance(val, (datetime, pd.Timestamp)):
                         results[k] = val.strftime("%Y-%m-%d")
                     else:
                         results[k] = val
                elif isinstance(v, (datetime, pd.Timestamp)):
                    results[k] = v.strftime("%Y-%m-%d")
                else:
                    results[k] = v
        else:
             # DataFrame fallback
             results["calendar_data"] = str(cal)

        # Add basic info
        info = ticker.info
        results["forward_pe"] = info.get("forwardPE", "N/A")
        results["trailing_pe"] = info.get("trailingPE", "N/A")
        results["peg_ratio"] = info.get("pegRatio", "N/A")

        return results

    except Exception as e:
        return {"symbol": symbol, "error": f"Earnings fetch failed: {str(e)}"}


@cached(key_func=lambda symbol: f"insider_trading:{symbol.upper()}")
@log_exceptions()
def get_insider_trading(symbol: str) -> dict[str, Any]:
    """
    Get recent insider transactions (Buy/Sell) by executives/directors.
    """
    try:
        ticker = yf.Ticker(symbol)
        insider = ticker.insider_transactions

        if insider is None or insider.empty:
            return {"symbol": symbol, "note": "No recent insider activity found"}

        # Get last 10 transactions
        recent = insider.head(10).copy()

        # Clean for output
        transactions = []
        for _index, row in recent.iterrows():
            # Clean values
            date = row.get("Start Date")
            if hasattr(date, "strftime"):
                 date = date.strftime("%Y-%m-%d")

            shares = row.get("Shares")
            value = row.get("Value")
            if pd.isna(value) or value == 0:
                # Estimate value if missing
                value = "N/A"
            else:
                value = f"${value:,.0f}"

            transactions.append({
                "insider": row.get("Insider", "Unknown"),
                "position": row.get("Position", "Unknown"),
                "transaction_date": date,
                "type": "Sell" if "Sale" in str(row.get("Text", "")) or "Sell" in str(row.get("Text", "")) else "Buy",
                "shares": int(shares) if pd.notna(shares) else 0,
                "value": value,
                "ownership": str(row.get("Ownership", ""))
            })

        return {
            "symbol": symbol,
            "recent_transactions_count": len(transactions),
            "transactions": transactions,
            "summary": "Check for clusters of Buying (Bullish) or Selling (Bearish, though often for tax/liquidity)."
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"Insider data failed: {str(e)}"}


@cached(key_func=lambda symbol: f"institutional:{symbol.upper()}")
@log_exceptions()
def get_institutional_ownership(symbol: str) -> dict[str, Any]:
    """
    Get top institutional holders and mutual funds.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Major Holders
        major = ticker.major_holders
        inst_holders = ticker.institutional_holders

        data = {"symbol": symbol}

        info = ticker.info
        pct = info.get('heldPercentInstitutions') or info.get('institutionsPercentHeld', 0)
        data['institutional_ownership_percentage'] = pct * 100

        # Clean Major Holders
        if major is not None:
             # Often returns a DataFrame with 0: Value, 1: Description
             # Or sometimes named columns
             try:
                 # Generic parsing logic to handle different yfinance versions
                 if isinstance(major, pd.DataFrame):
                     # Convert to list of dicts
                     data["breakdown"] = major.to_dict(orient="records")
             except Exception:
                 data["breakdown"] = "Available (Format parse error)"

        # Clean Institutional Holders
        if inst_holders is not None and not inst_holders.empty:
            holders = []
            for _idx, row in inst_holders.head(10).iterrows():
                date = row.get("Date Reported")
                if hasattr(date, "strftime"):
                    date = date.strftime("%Y-%m-%d")

                pct = row.get("pctHeld", row.get("% Out", 0))
                if isinstance(pct, (float, int)):
                     pct = f"{pct*100:.2f}%"

                holders.append({
                    "holder": row.get("Holder", "Unknown"),
                    "shares": f"{row.get('Shares', 0):,}",
                    "date_reported": date,
                    "percent_held": pct
                })
            data["top_institutions"] = holders
        else:
            data["top_institutions"] = "No data available"

        return data

    except Exception as e:
        return {"symbol": symbol, "error": f"Institutional data failed: {str(e)}"}


@log_exceptions()
def get_upcoming_ipos() -> dict[str, Any]:
    """
    Returns upcoming IPOs in two sections:
    1. Confirmed filings: scraped from Nasdaq IPO calendar (live S-1 filings)
    2. Speculative watchlist: high-profile rumored IPOs, clearly labeled as analyst speculation
    """
    confirmed = []
    speculative = [
        {
            "company": "Stripe",
            "expected": "Q3 2026",
            "est_valuation": "$65B",
            "sector": "Fintech",
            "status": "Rumored — no S-1 filed"
        },
        {
            "company": "Databricks",
            "expected": "Late 2026",
            "est_valuation": "$43B",
            "sector": "AI/Data",
            "status": "Rumored — no S-1 filed"
        },
        {
            "company": "SpaceX (Starlink spin-off)",
            "expected": "TBD",
            "est_valuation": "$180B+",
            "sector": "Aerospace",
            "status": "Speculative — Elon has denied near-term plans"
        },
        {
            "company": "Canva",
            "expected": "2026",
            "est_valuation": "$26B",
            "sector": "Software/Design",
            "status": "Rumored — no S-1 filed"
        },
        {
            "company": "Revolut",
            "expected": "2026",
            "est_valuation": "$45B",
            "sector": "Fintech",
            "status": "Rumored — UK banking license pending"
        }
    ]

    # --- Strategy 1: SEC EDGAR Recent S-1 Filings (Primary — Free & Reliable) ---
    try:
        from datetime import date, timedelta

        import requests
        today = date.today()
        start = (today - timedelta(days=45)).isoformat()
        end = today.isoformat()
        # EDGAR EFTS full-text search — requires User-Agent per SEC policy
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22S-1%22"
            f"&forms=S-1&dateRange=custom&startdt={start}&enddt={end}"
        )
        headers = {"User-Agent": "CairnIQBot contact@cairniq.local"}
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])
            for hit in hits[:8]:
                src = hit.get("_source", {})
                entity = src.get("entity_name") or src.get("display_names", ["Unknown"])[0]
                confirmed.append({
                    "company": entity,
                    "filed_date": src.get("file_date", "N/A"),
                    "form": "S-1",
                    "cik": (src.get("ciks") or ["N/A"])[0],
                    "source": "SEC EDGAR (Live S-1 Filing)"
                })
    except Exception as e:
        confirmed.append({"note": f"EDGAR unavailable: {e}"})

    # --- Strategy 2: Nasdaq IPO Calendar (Fallback — may be bot-blocked) ---
    if not confirmed or (len(confirmed) == 1 and "note" in confirmed[0]):
        try:
            import requests
            from bs4 import BeautifulSoup
            headers = {"User-Agent": "Mozilla/5.0 (compatible; CairnIQBot/1.0)"}
            resp = requests.get("https://www.nasdaq.com/market-activity/ipos", headers=headers, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                rows = soup.select("table tbody tr")
                for row in rows[:10]:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cols) >= 4:
                        confirmed.append({
                            "company": cols[0],
                            "symbol": cols[1] if len(cols) > 1 else "TBD",
                            "exchange": cols[2] if len(cols) > 2 else "N/A",
                            "expected_date": cols[3] if len(cols) > 3 else "N/A",
                            "price_range": cols[4] if len(cols) > 4 else "N/A",
                            "source": "Nasdaq IPO Calendar (Live)"
                        })
        except Exception:
            pass

    # --- Strategy 3: Relevance-Filtered Web Search ---
    live_news = []
    try:
        from tools.web_search import search_news
        # Tight query to avoid noise
        results = search_news(
            "IPO S-1 filing priced 2026 NASDAQ NYSE listing",
            max_results=5
        )
        # Relevance filter: only keep results that mention IPO/listing/S-1
        ipo_keywords = {"ipo", "s-1", "listing", "initial public offering", "priced", "debut"}
        if isinstance(results, list):
            for item in results:
                title = (item.get("title") or item.get("name") or "").lower()
                snippet = (item.get("snippet") or item.get("description") or "").lower()
                combined = title + " " + snippet
                if any(kw in combined for kw in ipo_keywords):
                    live_news.append({
                        "headline": item.get("title") or item.get("name", "N/A"),
                        "snippet": (item.get("snippet") or item.get("description", ""))[:200],
                        "url": item.get("url") or item.get("link", "")
                    })
        elif isinstance(results, str):
            # String result — apply keyword filter
            if any(kw in results.lower() for kw in ipo_keywords):
                live_news.append({"summary": results[:500]})
    except Exception:
        pass

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "confirmed_filings": confirmed if confirmed else [{"note": "No confirmed filings scraped — check nasdaq.com/market-activity/ipos"}],
        "speculative_watchlist": speculative,
        "live_news": live_news if live_news else [{"note": "No relevant IPO news found this week"}],
        "disclaimer": (
            "⚠️ Speculative watchlist = analyst rumor/speculation only. "
            "No confirmed S-1 filings. Dates and valuations subject to change. "
            "Confirmed filings sourced from Nasdaq IPO Calendar and SEC EDGAR."
        )
    }

@cached(key_func=lambda symbol: f"crowded_trade:{symbol.upper()}")
@log_exceptions()
def check_crowded_trade(symbol: str) -> dict[str, Any]:
    """
    Checks if a trade is 'Crowded' (heavily owned by institutions).
    Used to flag 'Pain Trades' where a small earnings miss causes a massive crash
    because all hedge funds run for the exit simultaneously.
    """
    try:
        inst_data = get_institutional_ownership(symbol)
        if not inst_data or "error" in inst_data:
            return {"symbol": symbol, "status": "No data"}

        try:
            # Safely get the ownership percentage
            pct_str = inst_data.get("institutional_ownership_percentage", "0")
            if isinstance(pct_str, str):
                pct_str = pct_str.replace('%', '')
            ownership_pct = float(pct_str)
        except Exception:
            ownership_pct = 0.0

        buyers = inst_data.get("recent_buyers", 0)
        sellers = inst_data.get("recent_sellers", 0)

        status = "Normal"
        risk = "Low"
        interpretation = "Institutional ownership is at normal levels."

        if ownership_pct > 85.0:
            status = "Extremely Crowded"
            risk = "High Pain-Trade Risk"
            interpretation = f"⚠️ Hedge funds own {ownership_pct}% of the float. If the company misses earnings, there's a huge risk of a 'Pain Trade' flash crash as everyone tries to sell at once."
        elif ownership_pct > 70.0:
            status = "Crowded"
            risk = "Elevated"
            interpretation = f"High institutional ownership ({ownership_pct}%). The stock may struggle to find new buyers."
        elif ownership_pct < 20.0:
            status = "Under-Owned"
            risk = "Low"
            interpretation = f"Low institutional ownership ({ownership_pct}%). Might present an opportunity if smart money rotates in."

        return {
            "symbol": symbol,
            "institutional_ownership": f"{ownership_pct}%",
            "buyers": buyers,
            "sellers": sellers,
            "status": status,
            "pain_trade_risk": risk,
            "interpretation": interpretation
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
