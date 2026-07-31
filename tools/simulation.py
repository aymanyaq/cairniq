"""
Portfolio Simulator Tool
Enables "What If" scenario analysis for the Deep Reasoning node.
Supports counterfactual portfolio analysis and rebalancing simulations.
"""

import json
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from agent.utils import safe_print
from tools.exception_logger import log_exceptions


@log_exceptions()
def simulate_rebalancing(current_holdings: str, adjustments: str) -> dict[str, Any]:
    """
    Simulates a portfolio rebalancing scenario and compares returns/volatility.

    Args:
        current_holdings: JSON string of current portfolio, e.g. '{"NVDA": 50, "AAPL": 30, "GOOGL": 20}'
                          Values are percentage allocations that should sum to 100.
        adjustments: Natural language or JSON adjustment, e.g. "Sell 50% NVDA, Buy SPY"
                     or '{"NVDA": 25, "AAPL": 30, "GOOGL": 20, "SPY": 25}'

    Returns:
        Comparison of original vs. proposed portfolio performance.
    """
    try:
        # Default Tax/Fee Parameters
        TRANS_COST = 0.0 # Modern brokers

        # Parse current holdings
        try:
            original_alloc = json.loads(current_holdings)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON for current_holdings: {current_holdings}"}

        # Validate allocations
        total = sum(original_alloc.values())
        if abs(total - 100) > 1:  # Allow small rounding differences
            # Normalize to 100%
            original_alloc = {k: (v / total) * 100 for k, v in original_alloc.items()}

        # Parse adjustments
        new_alloc = _parse_adjustments(original_alloc, adjustments)
        if "error" in new_alloc:
            return new_alloc

        # Fetch 1-year historical data for all symbols
        all_symbols = list(set(list(original_alloc.keys()) + list(new_alloc.keys())))
        price_data = {}

        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    price_data[symbol] = hist["Close"]
            except Exception as e:
                safe_print(f"Warning: Could not fetch {symbol}: {e}")

        # Filter for symbols with sufficient price history (at least 20 days)
        valid_price_data = {s: p for s, p in price_data.items() if len(p) >= 20}

        if len(valid_price_data) < len(price_data):
            missing = set(price_data.keys()) - set(valid_price_data.keys())
            safe_print(f"Warning: Skipping {missing} due to insufficient history (< 20 days)")

        if len(valid_price_data) < 1:
            return {"error": "Insufficient price data for simulation (< 20 days for all symbols)"}

        # Create aligned DataFrame
        df = pd.DataFrame(valid_price_data).dropna()
        if df.empty:
            return {"error": "No overlapping price data for simulation (try different symbols or periods)"}

        returns = df.pct_change().dropna()

        # Calculate portfolio returns for original allocation
        original_weights = _normalize_weights(original_alloc, list(returns.columns))
        original_port_returns = (returns * pd.Series(original_weights)).sum(axis=1)

        # Calculate portfolio returns for new allocation
        new_weights = _normalize_weights(new_alloc, list(returns.columns))
        new_port_returns = (returns * pd.Series(new_weights)).sum(axis=1)

        # Calculate metrics
        trading_days = 252

        original_annual_return = original_port_returns.mean() * trading_days * 100
        original_volatility = original_port_returns.std() * np.sqrt(trading_days) * 100
        original_sharpe = (original_annual_return - 5) / original_volatility if original_volatility > 0 else 0

        new_annual_return = new_port_returns.mean() * trading_days * 100
        new_volatility = new_port_returns.std() * np.sqrt(trading_days) * 100
        new_sharpe = (new_annual_return - 5) / new_volatility if new_volatility > 0 else 0

        # Calculate deltas
        return_delta = new_annual_return - original_annual_return
        volatility_delta = new_volatility - original_volatility
        sharpe_delta = new_sharpe - original_sharpe

        # Generate recommendation
        if sharpe_delta > 0.1:
            verdict = "✅ RECOMMENDED: Improves risk-adjusted returns"
        elif sharpe_delta < -0.1:
            verdict = "⚠️ NOT RECOMMENDED: Worse risk-adjusted returns"
        else:
            verdict = "➡️ NEUTRAL: Similar risk-adjusted returns"

        return {
            "simulation_type": "rebalancing_comparison",
            "period_analyzed": "1 Year",
            "original_portfolio": {
                "allocations": {k: f"{v:.1f}%" for k, v in original_alloc.items()},
                "annual_return": f"{original_annual_return:.1f}%",
                "volatility": f"{original_volatility:.1f}%",
                "sharpe_ratio": f"{original_sharpe:.2f}"
            },
            "proposed_portfolio": {
                "allocations": {k: f"{v:.1f}%" for k, v in new_alloc.items()},
                "annual_return": f"{new_annual_return:.1f}%",
                "volatility": f"{new_volatility:.1f}%",
                "sharpe_ratio": f"{new_sharpe:.2f}"
            },
            "comparison": {
                "return_change": f"{return_delta:+.1f}%",
                "volatility_change": f"{volatility_delta:+.1f}%",
                "sharpe_change": f"{sharpe_delta:+.2f}"
            },
            "estimated_impacts": {
                "taxes_on_sales": "Estimated ~15% on gains (if applicable)",
                "transaction_costs": f"${TRANS_COST} per trade"
            },
            "verdict": verdict,
            "plain_english": _generate_plain_english(return_delta, volatility_delta, sharpe_delta)
        }

    except Exception as e:
        return {"error": f"Simulation failed: {str(e)}"}


@log_exceptions()
def simulate_scenario(symbols: str, scenario: str) -> dict[str, Any]:
    """
    Simulates how a portfolio would perform under different market scenarios.

    Args:
        symbols: Comma-separated list of symbols, e.g. "NVDA,AAPL,GOOGL"
        scenario: One of "recession", "rate_hike", "tech_crash", "bull_market"

    Returns:
        Estimated impact based on historical drawdowns and beta.
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Overrides for ETFs often missing sector data
        SECTOR_OVERRIDES = {
            "XLE": "Energy", "XLF": "Financials", "XLK": "Technology", "XLV": "Healthcare",
            "XLY": "Consumer Cyclical", "XLP": "Consumer Defensive", "XLU": "Utilities",
            "XLI": "Industrials", "XLB": "Basic Materials", "XLRE": "Real Estate",
            "QQQ": "Technology", "SMH": "Technology", "IGV": "Technology",
            "FTEC": "Technology", "VGT": "Technology"
        }

        # Historical scenario impacts (based on actual market data)
        scenario_impacts = {
            "recession": {
                "market_drop": -35,
                "duration_months": 12,
                "description": "2008-style financial crisis",
                "high_beta_multiplier": 1.5,
                "defensive_reduction": 0.6
            },
            "rate_hike": {
                "market_drop": -20,
                "duration_months": 6,
                "description": "Fed aggressive tightening (2022-style)",
                "high_beta_multiplier": 1.3,
                "defensive_reduction": 0.8
            },
            "tech_crash": {
                "market_drop": -45,
                "duration_months": 24,
                "description": "Dot-com style tech selloff",
                "high_beta_multiplier": 2.0,
                "defensive_reduction": 0.5
            },
            "bull_market": {
                "market_drop": 25,  # Actually a gain
                "duration_months": 12,
                "description": "Strong recovery period",
                "high_beta_multiplier": 1.2,
                "defensive_reduction": 0.9
            }
        }

        params = scenario_impacts.get(scenario.lower())

        # If not standard, try to parse custom scenario "Oil $150" etc.
        if not params:
            if "oil" in scenario.lower():
                 params = {
                    "market_drop": -10,
                    "duration_months": 6,
                    "description": f"Custom Scenario: {scenario}",
                    "high_beta_multiplier": 1.1,
                    "defensive_reduction": 1.0,
                    "sector_impacts": {"Energy": 0.20, "Transportation": -0.15} # Custom logic
                 }
            else:
                 return {"error": f"Unknown scenario: {scenario}"}

        # Fetch beta and sector data for each symbol
        results = []
        # Kept so the 4.8 rate leg below can classify the SAME symbols off the
        # SAME metadata this loop already paid for, instead of re-fetching or —
        # worse — quietly answering for the whole portfolio when the caller named
        # three tickers.
        symbol_infos: dict[str, dict] = {}
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol)
                info = {}
                try:
                    info = ticker.info or {}
                except Exception:
                    info = {}
                symbol_infos[symbol] = info

                beta = info.get("beta", 1.0) or 1.0
                sector = info.get("sector") or "Unknown"

                # Check override
                if symbol in SECTOR_OVERRIDES:
                    sector = SECTOR_OVERRIDES[symbol]

                # Defensive sectors get less impact
                defensive_sectors = ["Utilities", "Consumer Defensive", "Healthcare"]
                is_defensive = sector in defensive_sectors

                sector_impacts = params.get("sector_impacts") or {}

                if sector in sector_impacts:
                    # Precise sector override
                    estimated_impact = sector_impacts[sector] * 100
                    impact_multiplier = 1.0 # handled directly
                elif is_defensive:
                    impact_multiplier = params["defensive_reduction"]
                elif beta > 1.2:
                    impact_multiplier = params["high_beta_multiplier"]
                else:
                    impact_multiplier = 1.0

                if sector not in sector_impacts:
                    estimated_impact = params["market_drop"] * impact_multiplier

                results.append({
                    "symbol": symbol,
                    "sector": sector,
                    "beta": f"{beta:.2f}",
                    "estimated_impact": f"{estimated_impact:+.1f}%",
                    "risk_level": "High" if beta > 1.3 else "Medium" if beta > 0.8 else "Low"
                })

            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })

        # Calculate portfolio-level impact
        impact_values = [float(r["estimated_impact"].replace("%", "").replace("+", ""))
                         for r in results if "estimated_impact" in r]
        avg_impact = float(np.mean(impact_values)) if impact_values else 0.0

        # Roadmap 4.8 — the rate leg this scenario has never had.
        #
        # `rate_hike` scored every named symbol off a -20% equity constant and a
        # beta multiplier. For a bond fund that is not an approximation, it is the
        # wrong instrument's model: a bond ETF's low beta lands it on the 1.0
        # multiplier and it gets marked down 20% by a rule built for equities,
        # while the one quantity that actually governs its response to a hike —
        # duration — was never computed anywhere in this app. This adds it, over
        # the SYMBOLS THE CALLER NAMED, and stamps `basis: "computed"` next to the
        # equity leg's `basis: "authored constant"` so the two are not read as
        # carrying the same weight.
        rate_leg = None
        if scenario.lower() == "rate_hike":
            from tools.bond_analytics import rate_hike_duration_leg

            rate_leg = rate_hike_duration_leg(
                shock_bp=100,
                rows=[{"symbol": s, "info": symbol_infos.get(s)} for s in symbol_list],
            )

        return {
            "scenario": params["description"],
            "market_impact": f"{params['market_drop']:+.0f}%",
            "duration": f"{params['duration_months']} months",
            "portfolio_analysis": results,
            "portfolio_estimated_impact": f"{avg_impact:+.1f}%",
            "recommendation": _scenario_recommendation(scenario.lower(), avg_impact),
            # Roadmap 2.7. The stamp lives HERE, on the function that owns the
            # constants, not only on the agent-tool wrapper that used to add it —
            # `scenario_impacts` above is a table of round numbers labelled
            # "2008-style"/"Dot-com style", and every caller (4.8 will add one for
            # the rate-hike case) must inherit the marker, not just this one.
            #
            # NOTE the stamp applies to the EQUITY leg. `rate_leg` below carries
            # its own `basis`, and it is a different one — see 4.8.
            "basis": "authored constant",
            "basis_note": (
                f"The {params['market_drop']:+.0f}% market move is an assumed constant "
                f"chosen to characterise a '{params['description']}' scenario, not a "
                f"measurement of one. Only each holding's beta is fetched. For a "
                f"measured answer use replay_historical_episode, which replays the "
                f"actual daily paths of the GFC, COVID, 2022 and the dot-com bust."
            ),
            "measured_alternative": "replay_historical_episode",
            # Present only for rate_hike, and present even when it does not apply:
            # an ABSENT key would read as "rates cost this book nothing", which is
            # the claim 4.8 exists to stop being made by omission.
            **({"rate_leg": rate_leg} if rate_leg is not None else {}),
        }

    except Exception as e:
        return {"error": f"Scenario simulation failed: {str(e)}"}


@log_exceptions()
def _parse_adjustments(original: dict[str, float], adjustments: str) -> dict[str, float]:
    """Parse adjustment string into new allocation dict."""
    # Try JSON first
    try:
        return json.loads(adjustments)
    except json.JSONDecodeError:
        pass

    # Parse natural language
    new_alloc = original.copy()
    adjustments.lower()

    # Pattern: "Sell X% SYMBOL" or "Reduce SYMBOL by X%"
    import re


    # Simple heuristic: split on comma, process each instruction
    instructions = adjustments.split(",")
    freed_allocation = 0
    new_symbols = []

    for instr in instructions:
        instr = instr.strip().lower()

        # Handle "sell 50% NVDA" pattern
        sell_match = re.search(r"sell\s+(?:(\d+)%?\s+of\s+)?(\w+)", instr)
        if sell_match:
            pct = int(sell_match.group(1)) if sell_match.group(1) else 50
            symbol = sell_match.group(2).upper()
            if symbol in new_alloc:
                reduction = (pct / 100) * new_alloc[symbol]
                new_alloc[symbol] -= reduction
                freed_allocation += reduction

        # Handle "buy SPY" pattern
        buy_match = re.search(r"(?:buy|add)\s+(?:(\d+)%?\s+)?(\w+)", instr)
        if buy_match and not sell_match:
            symbol = buy_match.group(2).upper()
            new_symbols.append(symbol)

    # Distribute freed allocation to new symbols
    if new_symbols and freed_allocation > 0:
        per_symbol = freed_allocation / len(new_symbols)
        for symbol in new_symbols:
            new_alloc[symbol] = new_alloc.get(symbol, 0) + per_symbol

    # Remove zero allocations
    new_alloc = {k: v for k, v in new_alloc.items() if v > 0.5}

    # Normalize to 100%
    total = sum(new_alloc.values())
    if total > 0:
        new_alloc = {k: (v / total) * 100 for k, v in new_alloc.items()}

    return new_alloc


@log_exceptions()
def _normalize_weights(alloc: dict[str, float], available_symbols: list[str]) -> dict[str, float]:
    """Normalize weights to available symbols only."""
    filtered = {k: v for k, v in alloc.items() if k in available_symbols}
    total = sum(filtered.values())
    if total > 0:
        return {k: v / total for k, v in filtered.items()}
    return {}


@log_exceptions()
def _generate_plain_english(return_delta: float, vol_delta: float, sharpe_delta: float) -> str:
    """Generate a plain English summary of the simulation results."""
    parts = []

    if abs(return_delta) > 1:
        direction = "higher" if return_delta > 0 else "lower"
        parts.append(f"returns would be ~{abs(return_delta):.0f}% {direction}")

    if abs(vol_delta) > 1:
        direction = "more" if vol_delta > 0 else "less"
        parts.append(f"volatility would be {abs(vol_delta):.0f}% {direction}")

    if parts:
        return "The new portfolio would have " + " and ".join(parts) + "."
    return "The portfolios are nearly identical in risk/return profile."


@log_exceptions()
def _scenario_recommendation(scenario: str, avg_impact: float) -> str:
    """Generate recommendation based on scenario analysis."""
    if scenario in ["recession", "tech_crash"]:
        if avg_impact < -30:
            return "⚠️ HIGH RISK: Consider adding defensive positions (utilities, bonds) or increasing cash"
        elif avg_impact < -20:
            return "🟡 MODERATE RISK: Some vulnerability; consider modest hedges"
        else:
            return "✅ RELATIVELY DEFENSIVE: Portfolio shows resilience to this scenario"
    elif scenario == "rate_hike":
        if avg_impact < -15:
            return "⚠️ RATE SENSITIVE: Consider reducing high-growth/duration exposure"
        else:
            return "✅ MANAGEABLE: Portfolio can weather rate hikes"
    else:  # bull_market
        if avg_impact > 20:
            return "✅ WELL POSITIONED: Portfolio captures upside in recovery"
        else:
            return "🟡 CONSERVATIVE: May underperform in bull markets"


if __name__ == "__main__":
    # Test simulation
    print("=== Rebalancing Simulation ===")
    result = simulate_rebalancing(
        '{"NVDA": 50, "AAPL": 30, "GOOGL": 20}',
        'Sell 50% NVDA, Buy SPY'
    )
    print(json.dumps(result, indent=2))

    print("\n=== Scenario Analysis ===")
    result = simulate_scenario("NVDA,AAPL,SPY", "recession")
    print(json.dumps(result, indent=2))
