from typing import Any

import yfinance as yf

from tools.exception_logger import log_exceptions
from tools.fed_calendar import get_fomc_calendar
from tools.fred_api import get_all_macro_indicators, get_canada_metrics, get_systemic_risk_indicators


@log_exceptions()
def analyze_macro_context() -> dict[str, Any]:
    """
    Analyze current economic conditions to determine the Market Regime
    and suggest tactical asset allocation shifts.
    """
    try:
        data = get_all_macro_indicators()
        can_data = get_canada_metrics()
        risk_data = get_systemic_risk_indicators()



        # safely parse strings like "4.33%" to float
        def parse_pct(s):
            if isinstance(s, str):
                return float(s.replace("%", "").replace("+", ""))
            return 0.0

        # Extract Core Data
        try:
            fed_funds = parse_pct(data["fed_funds"].get("current_rate", "0"))
            inflation = parse_pct(data["inflation"].get("headline_inflation", "0"))
            unemployment = parse_pct(data["unemployment"].get("current_rate", "0"))

            # Yield Curve
            y10 = parse_pct(data["treasury_yields"].get("10_year_yield", "0"))
            y2 = parse_pct(data["treasury_yields"].get("2_year_yield", "0"))
            curve_spread = y10 - y2

            gdp_trend = data["gdp"].get("trend", "Stable")
        except Exception:
            # Fallback if parsing fails
            return {"error": "Could not parse macro data for strategy."}

        # --- Regime Detection Logic ---
        regime = "Neutral"
        regime_priority = 0
        actions = []
        sectors_bullish = []
        sectors_bearish = []

        def set_regime(candidate: str, priority: int) -> None:
            nonlocal regime, regime_priority
            if priority > regime_priority:
                regime = candidate
                regime_priority = priority

        # --- Advanced Cross-Asset Leading Indicators ---
        cu_au_ratio_warning = False
        hyg_stress_warning = False
        try:
            hg = yf.Ticker("HG=F").history(period="1mo")
            gc = yf.Ticker("GC=F").history(period="1mo")
            if not hg.empty and not gc.empty:
                current_ratio = hg["Close"].iloc[-1] / gc["Close"].iloc[-1]
                past_ratio = hg["Close"].iloc[0] / gc["Close"].iloc[0]
                if current_ratio < past_ratio * 0.90:  # 10% drop in 1 month
                    cu_au_ratio_warning = True

            hyg = yf.Ticker("HYG").history(period="1mo")
            ief = yf.Ticker("IEF").history(period="1mo")
            if not hyg.empty and not ief.empty:
                current_spread = hyg["Close"].iloc[-1] / ief["Close"].iloc[-1]
                past_spread = hyg["Close"].iloc[0] / ief["Close"].iloc[0]
                if current_spread < past_spread * 0.95: # High yield dropping fast vs Treasuries
                    hyg_stress_warning = True
        except Exception:
            pass

        # --- Systemic Risk Check (Override) ---
        # If credit markets are breaking, nothing else matters.
        crash_risk = risk_data.get("crash_risk", "Low")
        risk_data.get("liquidity_status", "Neutral")

        if crash_risk in ["High", "Elevated"] or hyg_stress_warning:
            set_regime("SYSTEMIC RISK ALERT", 5)
            actions.append("CRITICAL: Credit markets are signaling stress (Spreads Widening or HYG dropping vs IEF).")
            sectors_bullish.extend(["Cash (BIL)", "Gold (GLD)", "Volatility (VIX)"])
            sectors_bearish.extend(["High Yield Bonds (HYG)", "Small Caps (IWM)", "Leverage"])

        # 1. Yield Curve Check (Recession Signal)
        elif curve_spread < 0 or cu_au_ratio_warning:
            set_regime("Late Cycle / Recession Risk", 4)
            if cu_au_ratio_warning: actions.append("Copper-to-Gold ratio crashing. Indicates institutional pivot to safe havens.")
            else: actions.append("Yield Curve is Inverted. Historically precedes recession.")
            sectors_bullish.extend(["Consumer Staples (XLP)", "Healthcare (XLV)", "Utilities (XLU)"])
            sectors_bearish.extend(["Industrials (XLI)", "Consumer Discretionary (XLY)"])

        # 2. Inflation & Rates Check
        if inflation > 3.0:
            set_regime("Inflationary", 3)
            actions.append("Inflation is high. Cash purchasing power eroding.")
            sectors_bullish.extend(["Energy (XLE)", "Real Estate (VNQ)", "Commodities"])
            sectors_bearish.extend(["Long-Duration Bonds (TLT)", "High-Growth Tech (ARKK)"])

            if fed_funds > 4.5:
                 actions.append("Fed Policy is TIGHT to fight inflation.")

        elif inflation < 2.0 and fed_funds > 3.0:
            # Easing likely coming
            set_regime("Disinflation / Pivot Watch", 3)
            actions.append("Inflation cooling. Potential for Rate Cuts.")
            sectors_bullish.extend(["Technology (XLK)", "Homebuilders (XHB)", "Small Caps (IWM)"])

        elif inflation < 2.5 and gdp_trend == "Accelerating":
            set_regime("Goldilocks (Growth without Inflation)", 2)
            sectors_bullish.extend(["Technology (XLK)", "Consumer Discretionary (XLY)", "Semiconductors (SMH)"])

        # 3. Unemployment Check
        if unemployment > 4.5:
             set_regime("Economic Slowdown", 4)
             actions.append("Unemployment rising. Consumer spending may weaken.")


        # Deduplicate
        sectors_bullish = list(set(sectors_bullish))
        sectors_bearish = list(set(sectors_bearish))


        # --- Plain English Translation ---
        simple_explanation = ""
        if regime == "SYSTEMIC RISK ALERT":
             simple_explanation = "CRITICAL: Financial stability indicators are flashing red. Credit markets are stressed. Capital preservation is the priority."
        elif regime == "Late Cycle / Recession Risk":
            simple_explanation = "Warning lights are flashing in the bond market. It might be wise to play it safe with defensive stocks (like things people buy regardless of the economy) rather than risky bets."
        elif regime == "Inflationary":
             simple_explanation = "Prices are rising fast, which eats into cash. Assets like Energy or Real Estate often do better when money loses value, while standard bonds might suffer."
        elif regime == "Disinflation / Pivot Watch":
             simple_explanation = "Inflation is cooling down. This usually means interest rates might fall soon, which often supports higher-duration assets such as Tech stocks and smaller companies."
        elif regime == "Goldilocks (Growth without Inflation)":
             simple_explanation = "The economy is in a 'sweet spot'—growing without prices getting out of control. This is generally the best time to own stocks."
        elif regime == "Economic Slowdown":
             simple_explanation = "The job market is weakening, which means people might spend less. It's a time to be cautious and focus on quality companies."
        else:
             simple_explanation = "The data is mixed right now, showing neither strong growth nor immediate danger. A balanced approach is best."

        # --- Canadian Context ---
        # Roadmap 5.7: prefer the Bank of Canada's own Valet series (the policy rate
        # it actually sets, and CPI-trim/median, the core measures its rate statements
        # cite) over FRED's lagged OECD re-publication. FRED stays as the fallback so
        # a Valet outage degrades rather than blanks this block, and the source is
        # always named so the two can never be mistaken for each other.
        can_context = "🇨🇦 Canada: Data unavailable."
        boc_block = None
        boc_rate = None
        can_cpi = None
        can_source = None
        try:
            from tools.boc_valet import get_boc_inflation, get_boc_policy_rate
            from tools.tool_errors import is_unavailable

            boc_policy = get_boc_policy_rate()
            if not is_unavailable(boc_policy):
                boc_rate = boc_policy.get("policy_rate")
                can_source = "Bank of Canada"
                boc_block = {
                    "policy_rate_pct": boc_policy.get("policy_rate_pct"),
                    "observation_date": boc_policy.get("observation_date"),
                    "last_change": boc_policy.get("last_change"),
                    "corra_vs_target_bps": boc_policy.get("corra_vs_target_bps"),
                    "funding_conditions": boc_policy.get("funding_conditions"),
                }

            boc_cpi = get_boc_inflation()
            if not is_unavailable(boc_cpi):
                can_cpi = boc_cpi.get("core_average_pct")
                can_source = can_source or "Bank of Canada"
                if boc_block is not None:
                    boc_block["core_inflation_pct"] = can_cpi
                    boc_block["vs_target"] = boc_cpi.get("vs_target")
        except Exception:
            pass

        if boc_rate is None or can_cpi is None:
            try:
                fallback_rate = parse_pct(can_data.get("interest_rate", "0")) or None
                fallback_cpi = parse_pct(can_data.get("inflation", "0")) or None
                if boc_rate is None and fallback_rate:
                    boc_rate = fallback_rate
                    can_source = can_source or "FRED (OECD re-publication)"
                if can_cpi is None and fallback_cpi:
                    can_cpi = fallback_cpi
                    can_source = can_source or "FRED (OECD re-publication)"
            except Exception:
                pass

        if boc_rate is not None and can_cpi is not None:
            src = f" [{can_source}]" if can_source else ""
            rate_diff = boc_rate - fed_funds
            if rate_diff < -0.5:
                can_context = f"🇨🇦 Canada: BoC Rate ({boc_rate}%) is significantly lower than Fed ({fed_funds}%). This may weaken the CAD. Consider keeping some USD exposure.{src}"
            elif can_cpi > inflation:
                can_context = f"🇨🇦 Canada: Inflation ({can_cpi}%) is running hotter than US ({inflation}%). BoC may need to stay tighter for longer.{src}"
            else:
                can_context = f"🇨🇦 Canada: BoC Rate ({boc_rate}%) and Inflation ({can_cpi}%) are tracking the US. Standard diversification applies.{src}"

        # --- Fed Calendar Integration ---
        fed_warning = None
        try:
            fomc = get_fomc_calendar(num_meetings=1)
            if "next_meeting" in fomc:
                next_mtg = fomc["next_meeting"]
                days_until = next_mtg.get("days_until", 999)
                if days_until <= 3:
                    fed_warning = f"🚨 FOMC MEETING IN {days_until} DAYS ({next_mtg['date']}) - HIGH VOLATILITY EXPECTED"
                    actions.append(fed_warning)
                elif days_until <= 7:
                    fed_warning = f"⚠️ FOMC meeting this week ({next_mtg['date']}) - expect market sensitivity"
                    actions.append(fed_warning)
                elif days_until <= 14:
                    fed_warning = f"📌 FOMC meeting in ~2 weeks ({next_mtg['date']})"
        except Exception:
            pass

        result = {
            "current_regime": regime,
            "plain_english": simple_explanation,
            "canadian_strategy": can_context,
            "canada_source": can_source or "unavailable",
            "fed_meeting_warning": fed_warning,
            "key_indicators": {
                "Systemic Risk": risk_data.get("crash_risk", "Low"),
                "Liquidity (M2)": risk_data.get("liquidity_status", "Neutral"),
                "Inflation (US)": f"{inflation}%",
                "Fed Rate (US)": f"{fed_funds}%",
                "BoC Rate (CA)": f"{boc_rate}%" if boc_rate is not None else "N/A",
                "Canada CPI": f"{can_cpi}%" if can_cpi is not None else "N/A",
                "Yield Curve": f"{curve_spread:.2f}% ({'Inverted' if curve_spread < 0 else 'Normal'})"
            },

            "strategy": {
                "tactical_opportunity": sectors_bullish,
                "sectors_to_underweight": sectors_bearish,
                "commentary": " ".join(actions)
            }
        }
        if boc_block:
            result["boc_detail"] = boc_block
        return result

    except Exception as e:
        return {"error": f"Macro strategy failed: {str(e)}"}

if __name__ == "__main__":
    import json
    print(json.dumps(analyze_macro_context(), indent=2))
