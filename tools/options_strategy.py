from datetime import datetime
from typing import Any

import yfinance as yf

from tools.exception_logger import log_exceptions


@log_exceptions()
def model_options_strategy(symbol: str, strategy_type: str = "covered_call", shares: int = 100) -> dict[str, Any]:
    """
    Architects advanced options strategies for Income (Covered Calls) or Protection (Collars).

    Args:
        symbol: Ticker symbol (e.g., "AAPL")
        strategy_type: "covered_call", "protective_put", or "collar"
        shares: Number of shares owned (default 100 for 1 contract)
    """
    try:
        ticker = yf.Ticker(symbol)

        # 1. Get Live Data
        hist = ticker.history(period="1d")
        if hist.empty:
            return {"error": f"Could not fetch price for {symbol}"}

        current_price = hist["Close"].iloc[-1]

        # 2. Get Option Chain (Next ~30-45 days ideal for income)
        if not ticker.options:
            return {"error": "No options chain available."}

        # Find expiry ~30-45 days out. Prioritize Monthly (3rd Friday)
        # Standard monthlies are typically the only ones available far out,
        # but near-term we have weeklies.
        expiries = ticker.options
        target_date = ""
        today = datetime.now()

        # 1. Search for a monthly expiry in the 30-50 day window
        candidate_dates = []
        for date_str in expiries:
            exp_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_out = (exp_date - today).days
            if 30 <= days_out <= 60:
                # Heuristic for monthly: 3rd Friday (days 15-21 of month)
                if 15 <= exp_date.day <= 21 and exp_date.weekday() == 4:
                     target_date = date_str
                     break
                candidate_dates.append(date_str)

        # 2. Fallback to nearest date after 30 days if no monthly found
        if not target_date and candidate_dates:
            target_date = candidate_dates[0]

        # 3. Final fallback: pick the second expiry (usually avoids the immediate week)
        if not target_date:
            target_date = expiries[1] if len(expiries) > 1 else expiries[0]

        chain = ticker.option_chain(target_date)
        calls = chain.calls
        puts = chain.puts

        # Filter for liquidity with fallback
        thresholds = [10, 5, 0]
        found_liq = False

        for thresh in thresholds:
            filtered_calls = calls[calls["openInterest"] >= thresh]
            filtered_puts = puts[puts["openInterest"] >= thresh]

            # Check based on strategy需求
            can_proceed = True
            if strategy_type == "covered_call" and filtered_calls.empty:
                can_proceed = False
            elif strategy_type == "protective_put" and filtered_puts.empty:
                can_proceed = False
            elif strategy_type == "collar" and (filtered_calls.empty or filtered_puts.empty):
                can_proceed = False

            if can_proceed:
                calls = filtered_calls
                puts = filtered_puts
                found_liq = True
                break

        if not found_liq:
            return {"error": f"Insufficient liquidity in options for {symbol} at {target_date} even with loosened filters."}

        results = {
            "symbol": symbol,
            "current_price": f"${current_price:.2f}",
            "expiry": target_date,
            "strategy": strategy_type.replace("_", " ").title(),
            "recommendation": [],
            "note": f"Targeted expiry: {target_date} (~{(datetime.strptime(target_date, '%Y-%m-%d') - today).days} days out)."
        }

        # --- STRATEGY LOGIC ---

        if strategy_type == "covered_call":
            # Target Delta ~0.30 (Aggressive Income) and ~0.20 (Conservative)
            # Proxy for Delta: Probability OTM.
            # Strike > Price.
            # 0.30 Delta is roughly where Strike is ~3-5% OTM depending on IV.
            # We will use simple OTM % as a proxy since we don't have Greeks calculated.

            # Conservative: 5-7% OTM
            cons_strike = current_price * 1.07
            # Aggressive: 3-4% OTM
            aggr_strike = current_price * 1.035

            # Find closest strikes
            c_indices = (calls['strike'] - cons_strike).abs().argsort()
            a_indices = (calls['strike'] - aggr_strike).abs().argsort()

            c_opt = calls.iloc[c_indices[:1]].iloc[0] if len(c_indices) > 0 else calls.iloc[0]
            a_opt = calls.iloc[a_indices[:1]].iloc[0] if len(a_indices) > 0 else calls.iloc[0]

            for name, opt in [("Conservative (Lower Risk)", c_opt), ("Aggressive (Higher Income)", a_opt)]:
                premium = opt['lastPrice']
                contracts = shares // 100
                total_income = premium * 100 * contracts
                downside_protection = (premium / current_price) * 100

                # Annualized Return if called away
                # ((Strike - Price + Premium) / Price) * (365/days)
                # But simple Yield is (Premium / Price)
                simple_yield = (premium / current_price) * 100

                results["recommendation"].append({
                    "type": name,
                    "action": f"Sell {contracts}x {target_date} ${opt['strike']} Call",
                    "premium_per_share": f"${premium:.2f}",
                    "total_income": f"${total_income:.0f}",
                    "yield_capture": f"{simple_yield:.1f}% ({simple_yield * 12:.1f}% annualized)",
                    "breakeven": f"${current_price - premium:.2f}",
                    "note": f"Provides {downside_protection:.1f}% downside buffer."
                })

        elif strategy_type == "protective_put":
            # Insurance against crash > 10%
            # Strike = 90% of current price
            target_strike = current_price * 0.90

            p_indices = (puts['strike'] - target_strike).abs().argsort()
            p_opt = puts.iloc[p_indices[:1]].iloc[0] if len(p_indices) > 0 else puts.iloc[0]

            cost = p_opt['lastPrice']
            contracts = shares // 100
            total_cost = cost * 100 * contracts

            results["recommendation"].append({
                "type": "Crash Insurance",
                "action": f"Buy {contracts}x {target_date} ${p_opt['strike']} Put",
                "cost": f"${total_cost:.0f} (${cost:.2f}/share)",
                "protection_level": f"${p_opt['strike']} (-{(1 - p_opt['strike']/current_price)*100:.1f}%)",
                "note": "Protects against any drop below strike price."
            })

        elif strategy_type == "collar":
            # Zero Cost Collar: Buy Put (Protection) financed by Selling Call (Income)
            # 1. Buy 10% OTM Put
            put_strike_target = current_price * 0.90
            p_indices = (puts['strike'] - put_strike_target).abs().argsort()
            p_opt = puts.iloc[p_indices[:1]].iloc[0] if len(p_indices) > 0 else puts.iloc[0]
            put_cost = p_opt['lastPrice']

            # 2. Sell Call with matching premium
            # Find call where price is closest to put_cost
            c_indices = (calls['lastPrice'] - put_cost).abs().argsort()
            c_opt = calls.iloc[c_indices[:1]].iloc[0] if len(c_indices) > 0 else calls.iloc[0]
            call_credit = c_opt['lastPrice']

            net_debit = put_cost - call_credit

            contracts = shares // 100

            results["recommendation"].append({
                "type": "Zero-Cost Collar",
                "action_leg_1": f"Buy {contracts}x ${p_opt['strike']} Put (Cost: ${put_cost:.2f})",
                "action_leg_2": f"Sell {contracts}x ${c_opt['strike']} Call (Credit: ${call_credit:.2f})",
                "net_cost": f"${net_debit:.2f}/share",
                "capped_upside": f"${c_opt['strike']} (+{(c_opt['strike']/current_price - 1)*100:.1f}%)",
                "floored_downside": f"${p_opt['strike']} ( -{(1 - p_opt['strike']/current_price)*100:.1f}%)",
                "note": "Locks in portfolio range. Upside limited, downside protected."
            })

        return results

    except Exception as e:
        return {"error": f"Strategy modeling failed: {str(e)}"}

if __name__ == "__main__":
    print(model_options_strategy("NVDA", "collar"))
