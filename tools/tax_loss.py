from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.portfolio_csv import load_portfolio


@log_exceptions()
def _conviction_check(symbol: str) -> dict:
    """
    Forward-looking conviction check for a tax-loss candidate.
    Fetches analyst consensus, recent price momentum, revenue growth,
    and recent news headlines to determine if a turnaround is brewing.
    Returns a dict with conviction signals that the LLM can use to
    override a blind "SELL" recommendation.
    """
    result = {
        "analyst_consensus": "N/A",
        "price_target": "N/A",
        "upside_potential": "N/A",
        "recent_momentum_1m": "N/A",
        "revenue_growth": "N/A",
        "forward_pe": "N/A",
        "recent_news": [],
        "turnaround_signals": 0,
        "conviction_verdict": "NO_DATA",
    }

    turnaround_signals = 0

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # 1. Analyst consensus & price target
        consensus = info.get("recommendationKey", "N/A")
        result["analyst_consensus"] = consensus.replace("_", " ").title() if consensus != "N/A" else "N/A"

        target_price = info.get("targetMeanPrice")
        current_price = info.get("currentPrice") or info.get("previousClose")

        if target_price:
            result["price_target"] = f"${target_price:,.2f}"
        if target_price and current_price and current_price > 0:
            upside = ((target_price - current_price) / current_price) * 100
            result["upside_potential"] = f"{upside:+.1f}%"
            if upside > 20:
                turnaround_signals += 1  # Analysts see significant upside

        # Buy/outperform consensus = turnaround signal
        if consensus and any(x in consensus.lower() for x in ["buy", "outperform", "overweight"]):
            turnaround_signals += 1

        # 2. Forward PE (profitable outlook?)
        fwd_pe = info.get("forwardPE")
        if fwd_pe and fwd_pe > 0:
            result["forward_pe"] = f"{fwd_pe:.1f}"
            if fwd_pe < 30:
                turnaround_signals += 1  # Reasonable forward valuation

        # 3. Revenue growth
        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            result["revenue_growth"] = f"{rev_growth * 100:+.1f}%"
            if rev_growth > 0.10:
                turnaround_signals += 1  # Strong revenue growth despite price decline

        # 4. Recent 1-month price momentum
        try:
            hist = ticker.history(period="1mo")
            if hist is not None and len(hist) >= 2:
                start_p = hist["Close"].iloc[0]
                end_p = hist["Close"].iloc[-1]
                mom_1m = ((end_p - start_p) / start_p) * 100
                result["recent_momentum_1m"] = f"{mom_1m:+.1f}%"
                if mom_1m > 10:
                    turnaround_signals += 1  # Strong recent recovery
        except Exception:
            pass

        # 5. Recent news headlines (top 3)
        try:
            raw_news = ticker.news
            if raw_news:
                for n in raw_news[:3]:
                    content = n.get("content", n)
                    title = content.get("title") or content.get("headline") or "No Title"
                    result["recent_news"].append(title)
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)

    result["turnaround_signals"] = turnaround_signals

    # Verdict logic
    if turnaround_signals >= 3:
        result["conviction_verdict"] = "⚠️ POTENTIAL_TURNAROUND — Review carefully before selling"
    elif turnaround_signals >= 2:
        result["conviction_verdict"] = "🟡 MIXED_SIGNALS — Some recovery indicators present"
    elif turnaround_signals == 1:
        result["conviction_verdict"] = "🟢 WEAK_RECOVERY — Tax-loss harvest likely beneficial"
    else:
        result["conviction_verdict"] = "🟢 NO_RECOVERY — Strong tax-loss harvesting candidate"

    return result


@cached(key_func=lambda: "tax_loss_analysis", ttl=3600)
@log_exceptions()
def analyze_tax_loss_harvesting() -> dict:
    """
    Analyze portfolio for tax-loss harvesting opportunities.
    Identifies positions with >10% unrealized losses that could be sold to offset gains.

    ENHANCED: Each candidate now includes a forward-looking conviction check
    (analyst consensus, price momentum, revenue growth, news) to prevent
    blindly selling stocks with changing outlooks.
    """
    holdings = load_portfolio()
    if isinstance(holdings, dict) and "error" in holdings:
        return holdings

    opportunities = []

    # 1. Parse and Collect Symbols
    candidates = []
    symbols = []

    for item in holdings:
        sym = item.get("symbol", "").upper()
        account = item.get("account", "").upper()

        if not sym or item.get("is_private_asset") or sym == "CASH" or "USD" in sym:
            continue

        # Enforce Governance Rule: Do not report tax loss harvesting for tax-sheltered accounts
        if any(tax_shelter in account for tax_shelter in ["TFSA", "RRSP", "IRA", "DCPP", "PENSION"]):
            continue

        try:
            # Robust parsing
            p_str = str(item.get("purchase_price", "0"))
            cost_basis = float(p_str.replace('$', '').replace(',', '').strip())

            s_str = str(item.get("shares", "0"))
            shares = float(s_str.replace(',', '').strip())

            if cost_basis > 0 and shares > 0:
                candidates.append({
                    "symbol": sym,
                    "cost_basis": cost_basis,
                    "shares": shares
                })
                symbols.append(sym)
        except Exception:
            continue

    if not symbols:
        return {"note": "No valid positions found to analyze."}

    # 2. Batch Fetch Current Prices (Faster)
    try:
        # Download all at once
        data = yf.download(symbols, period="1d", progress=False)['Close']

        # Handle single symbol case (Series vs DataFrame)
        if len(symbols) == 1:
            current_prices = {symbols[0]: data.iloc[-1]}
        else:
            current_prices = data.iloc[-1].to_dict()

    except Exception as e:
        return {"error": f"Failed to fetch market data: {str(e)}"}

    # 3. Identify loss candidates first
    loss_candidates = []
    for c in candidates:
        sym = c["symbol"]
        cost = c["cost_basis"]
        shares = c["shares"]

        # Get price (handle missing data)
        curr = current_prices.get(sym)
        if pd.isna(curr): continue

        loss_pct = (curr - cost) / cost

        # Threshold: -10% loss to be worth harvesting
        if loss_pct <= -0.10:
            unrealized_loss = (cost - curr) * shares

            # Only flag if loss amount is significant (> $200)
            if unrealized_loss > 200:
                loss_candidates.append({
                    "symbol": sym,
                    "loss_pct": loss_pct,
                    "unrealized_loss": unrealized_loss,
                    "current_price": curr,
                    "cost_basis": cost,
                    "shares": shares,
                })

    # 4. Run conviction checks in parallel for all loss candidates
    conviction_results = {}
    if loss_candidates:
        check_symbols = [lc["symbol"] for lc in loss_candidates]
        executor = ThreadPoolExecutor(max_workers=min(len(check_symbols), 5))
        try:
            future_map = {executor.submit(_conviction_check, sym): sym for sym in check_symbols}
            try:
                # Add a 15-second timeout to prevent yfinance hangs
                from concurrent.futures import TimeoutError
                for future in as_completed(future_map, timeout=15):
                    sym = future_map[future]
                    try:
                        conviction_results[sym] = future.result(timeout=1)
                    except Exception as e:
                        conviction_results[sym] = {"conviction_verdict": "NO_DATA", "error": str(e)}
            except TimeoutError:
                # Safely fallback hanging tasks
                for future, sym in future_map.items():
                    if sym not in conviction_results:
                        conviction_results[sym] = {"conviction_verdict": "NO_DATA", "error": "yfinance timeout"}
        finally:
            # Force background threads to die without holding the main interpreter
            executor.shutdown(wait=False, cancel_futures=True)

    # 5. Build final opportunities with conviction data attached
    for lc in loss_candidates:
        sym = lc["symbol"]
        loss_pct = lc["loss_pct"]
        conviction = conviction_results.get(sym, {})

        opportunities.append({
            "symbol": sym,
            "loss_pct": f"{loss_pct*100:.1f}%",
            "est_loss_amount": f"${lc['unrealized_loss']:,.2f}",
            "current_price": f"${lc['current_price']:.2f}",
            "cost_basis": f"${lc['cost_basis']:.2f}",
            "shares": f"{lc['shares']:,.2f}",
            "opportunity": "STRONG" if loss_pct < -0.20 else "MODERATE",
            # Conviction Check results
            "conviction_check": {
                "verdict": conviction.get("conviction_verdict", "NO_DATA"),
                "analyst_consensus": conviction.get("analyst_consensus", "N/A"),
                "price_target": conviction.get("price_target", "N/A"),
                "upside_potential": conviction.get("upside_potential", "N/A"),
                "recent_momentum_1m": conviction.get("recent_momentum_1m", "N/A"),
                "revenue_growth": conviction.get("revenue_growth", "N/A"),
                "forward_pe": conviction.get("forward_pe", "N/A"),
                "turnaround_signals": conviction.get("turnaround_signals", 0),
                "recent_news": conviction.get("recent_news", []),
            }
        })

    # Sort by biggest loss amount
    opportunities.sort(key=lambda x: float(x["est_loss_amount"].replace("$","").replace(",","")), reverse=True)

    return {
        "harvesting_candidates": opportunities,
        "count": len(opportunities),
        "note": (
            "⚠️ Wash Sale Rule: If you sell at a loss, you cannot buy the same or 'substantially identical' "
            "security within 30 days before or after the sale.\n"
            "🔍 Each candidate includes a CONVICTION CHECK — review analyst consensus, momentum, and revenue "
            "growth before selling. Stocks showing turnaround signals may be worth holding despite losses."
        )
    }

if __name__ == "__main__":
    import json
    result = analyze_tax_loss_harvesting()
    print(json.dumps(result, indent=2))
