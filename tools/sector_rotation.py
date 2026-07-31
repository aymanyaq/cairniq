"""
Sector Rotation Tool
Detects when institutional money is rotating between sectors.
Leading indicator for market regime changes.
"""

import math
from datetime import datetime
from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions

# Major sector ETFs for rotation analysis
SECTOR_ETFS = {
    "XLK": {"name": "Technology", "character": "Growth/Risk-On"},
    "XLV": {"name": "Healthcare", "character": "Defensive"},
    "XLF": {"name": "Financials", "character": "Cyclical/Rate-Sensitive"},
    "XLE": {"name": "Energy", "character": "Inflation Hedge/Cyclical"},
    "XLU": {"name": "Utilities", "character": "Defensive/Rate-Sensitive"},
    "XLY": {"name": "Consumer Discretionary", "character": "Growth/Cyclical"},
    "XLP": {"name": "Consumer Staples", "character": "Defensive"},
    "XLI": {"name": "Industrials", "character": "Cyclical/Economic Barometer"},
    "XLB": {"name": "Materials", "character": "Cyclical/Inflation Hedge"},
    "XLRE": {"name": "Real Estate", "character": "Income/Rate-Sensitive"},
    "XLC": {"name": "Communication Services", "character": "Mixed/Growth"},
}


@cached(key_func=lambda: "sr_sector_rotation")
@log_exceptions()
def detect_sector_rotation() -> dict[str, Any]:
    """
    Detect sector rotation by comparing 1-month vs 3-month performance. FLOW version.

    Logic:
    - If 1M performance > 3M/3 (momentum acceleration) = Money flowing IN
    - If 1M performance < 3M/3 (momentum deceleration) = Money flowing OUT

    ⚠️ There is a SECOND live function with this exact name:
    `tools.market_mechanics.detect_sector_rotation`. Both are correct and they
    answer DIFFERENT questions — importing the wrong one silently changes what
    is being asked. This one measures ACCELERATION against a sector's own
    3-month run rate (INFLOW / OUTFLOW / NEUTRAL) and feeds the market-pulse
    heatmap; that one measures TREND DIRECTION (Leading / Weakening / Improving
    / Lagging) and feeds the `check_sector_rotation` agent tool and scan-universe
    assembly. See its docstring for the full comparison.

    A sector can be "Leading" there and "OUTFLOW" here at the same time — up
    strongly over three months, but slowing. That is a real state, not a bug to
    be unified away. The payload carries `methodology` so a consumer can tell
    which of the two it is holding.

    Returns:
        Sector rotation signals with interpretation and trading implications.
    """
    try:
        results = []

        for symbol, info in SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="4mo")

                if hist.empty or len(hist) < 60:
                    continue

                # Calculate returns (cast to native float at extraction: numpy
                # scalars leak as "np.float64(...)" into str()-rendered output).
                current_price = float(hist["Close"].iloc[-1])
                price_1m = float(hist["Close"].iloc[-22])  # ~1 month ago
                price_3m = float(hist["Close"].iloc[-66] if len(hist) >= 66 else hist["Close"].iloc[0])

                return_1m = ((current_price - price_1m) / price_1m) * 100
                return_3m = ((current_price - price_3m) / price_3m) * 100

                # Momentum = 1M return compared to expected (3M/3)
                expected_1m = return_3m / 3
                momentum = return_1m - expected_1m

                # yfinance can return a row whose Close is NaN (a hole in the
                # series, not a missing row), which arithmetic propagates. Drop
                # the sector the same way insufficient history drops it: a NaN
                # formatted as "+nan%" reads as a number downstream, and a
                # substituted 0.0% would be a return this sector never had.
                if not all(math.isfinite(v) for v in (return_1m, return_3m, momentum)):
                    continue

                # Classify
                if momentum > 2:
                    signal = "🟢 INFLOW"
                    interpretation = "Accelerating momentum - money flowing IN"
                elif momentum < -2:
                    signal = "🔴 OUTFLOW"
                    interpretation = "Decelerating momentum - money flowing OUT"
                else:
                    signal = "⚪ NEUTRAL"
                    interpretation = "Stable momentum"

                results.append({
                    "symbol": symbol,
                    "sector": info["name"],
                    "character": info["character"],
                    "return_1m": f"{return_1m:+.1f}%",
                    "return_3m": f"{return_3m:+.1f}%",
                    "momentum_score": round(momentum, 1),
                    "signal": signal,
                    "interpretation": interpretation
                })

            except Exception:
                continue

        if not results:
            return {"error": "Could not fetch sector data"}

        # Sort by momentum (highest first)
        results.sort(key=lambda x: x["momentum_score"], reverse=True)

        # Identify rotation patterns
        inflows = [r for r in results if "INFLOW" in r["signal"]]
        outflows = [r for r in results if "OUTFLOW" in r["signal"]]

        # Pattern recognition
        patterns = []

        # Risk-on to Risk-off rotation
        inflow_sectors = [r["sector"] for r in inflows]
        outflow_sectors = [r["sector"] for r in outflows]

        defensive = ["Utilities", "Consumer Staples", "Healthcare"]
        cyclical = ["Technology", "Consumer Discretionary", "Financials", "Industrials"]

        defensive_inflows = len([s for s in inflow_sectors if s in defensive])
        cyclical_outflows = len([s for s in outflow_sectors if s in cyclical])

        cyclical_inflows = len([s for s in inflow_sectors if s in cyclical])
        defensive_outflows = len([s for s in outflow_sectors if s in defensive])

        if defensive_inflows >= 2 and cyclical_outflows >= 2:
            patterns.append({
                "pattern": "RISK-OFF ROTATION",
                "description": "Money rotating from growth/cyclical into defensive sectors",
                "implication": "Potential market top or recession expectations",
                "action": "Consider increasing defensive exposure (XLU, XLP, XLV)"
            })
        elif cyclical_inflows >= 2 and defensive_outflows >= 2:
            patterns.append({
                "pattern": "RISK-ON ROTATION",
                "description": "Money rotating from defensive into growth/cyclical sectors",
                "implication": "Market optimism / expansion expectations",
                "action": "Consider increasing growth exposure (XLK, XLY)"
            })

        # Energy/Materials rotation (inflation play)
        if "Energy" in inflow_sectors and "Materials" in inflow_sectors:
            patterns.append({
                "pattern": "INFLATION HEDGE ROTATION",
                "description": "Money flowing into commodities/energy",
                "implication": "Market pricing in higher inflation",
                "action": "Consider XLE, XLB, or commodity exposure"
            })

        # Rate-sensitive rotation
        rate_sensitive = ["Utilities", "Real Estate"]
        rs_inflows = [s for s in inflow_sectors if s in rate_sensitive]
        if len(rs_inflows) >= 2:
            patterns.append({
                "pattern": "RATE CUT POSITIONING",
                "description": "Money flowing into rate-sensitive sectors",
                "implication": "Market expecting Fed to cut rates",
                "action": "Consider XLRE, XLU if rate cuts expected"
            })

        if not patterns:
            patterns.append({
                "pattern": "NO CLEAR ROTATION",
                "description": "Sector performance is mixed",
                "implication": "Market in transition or consolidation",
                "action": "Maintain diversified allocation"
            })

        return {
            "methodology": "flow_acceleration",
            "scan_date": datetime.now().strftime("%Y-%m-%d"),
            "sector_performance": results,
            "rotation_patterns": patterns,
            "top_inflows": inflows[:3] if inflows else "None",
            "top_outflows": outflows[:3] if outflows else "None",
            "summary": f"Detected {len(inflows)} sectors with inflows, {len(outflows)} with outflows"
        }

    except Exception as e:
        return {"error": f"Sector rotation analysis failed: {str(e)}"}


@log_exceptions()
def get_sector_momentum_ranking() -> list[dict[str, Any]]:
    """Get a simple ranked list of sectors by momentum."""
    rotation = detect_sector_rotation()
    if "error" in rotation:
        return []
    return rotation.get("sector_performance", [])


@log_exceptions()
def is_risk_off_rotation() -> bool:
    """Quick check: Is the market in a risk-off rotation?"""
    rotation = detect_sector_rotation()
    if "error" in rotation:
        return False

    patterns = rotation.get("rotation_patterns", [])
    return any("RISK-OFF" in (p.get("pattern") or "") for p in patterns)


if __name__ == "__main__":
    import json

    print("=== Sector Rotation Analysis ===")
    result = detect_sector_rotation()
    print(json.dumps(result, indent=2))

    print("\n=== Risk-Off Check ===")
    print(f"Is Risk-Off? {is_risk_off_rotation()}")
