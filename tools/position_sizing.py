from typing import Any

from tools.exception_logger import log_exceptions


def _resolve_risk_pct(risk_per_trade_pct: float | None) -> tuple[float | None, str]:
    """Settle on a risk % and — just as important — on whose number it is.

    An explicit argument is the caller's own working assumption. Otherwise the
    profile's stated max-risk rule is used, if the user set one. There is no
    fallback: this tool used to default to 2.0 and stamp "2%" into every result,
    which is where the phantom "your 2% risk limit" in the advice came from.
    """
    if risk_per_trade_pct is not None:
        return float(risk_per_trade_pct), "assumption supplied for this calculation (not a rule from the user's profile)"
    try:
        from tools.ips_precheck import stated_caps
        stated = stated_caps().get("max_risk_per_trade_pct")
    except Exception:
        stated = None
    if stated is not None:
        return float(stated), f"the user's own stated {stated:g}% max-risk rule"
    return None, "the user's profile states no maximum-risk rule"


@log_exceptions()
def calculate_position_size(
    portfolio_value: float,
    risk_per_trade_pct: float | None = None,  # % of portfolio to risk per trade
    entry_price: float = 0,
    stop_loss_price: float = 0,
    volatility_adjustment: bool = False,
    asset_beta: float = 1.0
) -> dict[str, Any]:
    """
    Calculates recommended position size based on risk management principles.

    Args:
        portfolio_value: Total portfolio value in dollars
        risk_per_trade_pct: % of portfolio to risk. Omit to use the user's own
            stated max-risk rule; if they have not set one there is no default
            and you must pass an explicit figure to get a sizing answer.
        entry_price: Planned entry price
        stop_loss_price: Planned stop loss price
        volatility_adjustment: If True, reduces size for high beta assets
        asset_beta: Beta of the asset (1.0 = market volatility)

    Returns:
        Position sizing recommendation, including `risk_basis` — where the risk
        % came from. Attribute the number to the user ONLY when risk_basis says
        it is their stated rule.
    """
    try:
        risk_per_trade_pct, risk_basis = _resolve_risk_pct(risk_per_trade_pct)
        if risk_per_trade_pct is None:
            return {
                "portfolio_value": f"${portfolio_value:,.2f}",
                "risk_basis": risk_basis,
                "sizing_unavailable": (
                    "No risk budget is defined, so there is no correct position size to report. "
                    "Do not assume one — there is no default 2% rule. Either call this tool again "
                    "with an explicit risk_per_trade_pct and present the result as your own stated "
                    "assumption, or give the user the dollar-at-risk of the trade you are proposing "
                    "and let them judge it."
                ),
            }

        # 1. Base Risk Calculation
        risk_amount = portfolio_value * (risk_per_trade_pct / 100)

        # 2. Volatility Adjustment (Proprietary Logic)
        adjustment_note = "Standard (No volatility adjust)"
        if volatility_adjustment and asset_beta > 1.0:
            # Reduce risk for high beta. E.g. Beta 2.0 -> Half risk.
            # Damping factor: We don't want to be too aggressive in reduction, so sqrt(beta) or just beta.
            # Let's use simple inverse beta for safety.
            original_risk = risk_amount
            risk_amount = risk_amount / asset_beta
            adjustment_note = f"Reduced by {((original_risk - risk_amount)/original_risk)*100:.1f}% due to high Beta ({asset_beta})"

        result = {
            "portfolio_value": f"${portfolio_value:,.2f}",
            "base_risk_pct": f"{risk_per_trade_pct:g}%",
            "risk_basis": risk_basis,
            "volatility_adjusted_risk": f"${risk_amount:,.2f}",
            "sizing_method": "Fixed Fractional Risk" + (" (Volatility Adjusted)" if volatility_adjustment else ""),
            "note": adjustment_note
        }

        # 3. Calculate Shares/Size
        if entry_price > 0 and stop_loss_price > 0:
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share > 0:
                recommended_shares = int(risk_amount / risk_per_share)
                position_value = recommended_shares * entry_price
                position_pct = (position_value / portfolio_value) * 100

                result.update({
                    "entry_price": f"${entry_price:.2f}",
                    "stop_loss": f"${stop_loss_price:.2f}",
                    "risk_per_share": f"${risk_per_share:.2f} ({(risk_per_share/entry_price)*100:.1f}%)",
                    "recommended_shares": recommended_shares,
                    "estimated_position_value": f"${position_value:,.2f}",
                    "portfolio_allocation": f"{position_pct:.1f}%"
                })
        else:
            # Calculate simple allocation based on % of portfolio
            allocations = {}
            for alloc_pct in [2.5, 5, 10]:
                alloc_amount = portfolio_value * (alloc_pct / 100)
                # Apply volatility dampener to raw allocation too?
                if volatility_adjustment and asset_beta > 1.0:
                     alloc_amount /= asset_beta

                label = f"{alloc_pct}% Tier"
                if volatility_adjustment and asset_beta > 1.0:
                    label += f" (Adj to {alloc_pct/asset_beta:.1f}%)"

                allocations[label] = f"${alloc_amount:,.2f}"
            result["generic_allocations"] = allocations
            result["warning"] = (
                "Generic allocation tiers are not a trade setup. Provide an ATR/support-based stop "
                f"to calculate shares and dollar-at-risk against {risk_basis}."
            )

        return result

    except Exception as e:
        return {"error": f"Position sizing calculation failed: {e}"}

if __name__ == "__main__":
    # Example: $100k portfolio, AAPL at $200, stop at $190
    print(calculate_position_size(100000, 2.0, 200, 190))
