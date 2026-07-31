from typing import Any

from tools.exception_logger import log_exceptions


@log_exceptions()
def construct_bond_ladder(amount: float, investment_type: str = "GIC", currency: str = "CAD") -> dict[str, Any]:
    """
    Constructs a 5-Year Bond/GIC Ladder for income and liquidity.

    Args:
        amount: Total capital to deploy (e.g., 100000)
        investment_type: "GIC", "Treasury", "Corporate Bond"
        currency: "CAD" or "USD"
    """
    try:
        rates, data_note = _fetch_current_rates(investment_type, currency)

        # 2. Structure Ladder (Equal Weight: 20% per rung)
        rung_amount = amount / 5
        ladder = []
        total_income = 0
        weighted_rate = 0

        for year in range(1, 6):
            rate = rates.get(year, 3.5) # Default 3.5% safe fall back
            annual_income = rung_amount * (rate / 100)
            total_income += annual_income
            weighted_rate += rate

            ladder.append({
                "maturity": f"{year} Year",
                "rate": f"{rate:.2f}%",
                "principal": f"${rung_amount:,.2f}",
                "annual_income": f"${annual_income:,.2f}",
                "action": f"Buy {year}-Year {investment_type}"
            })

        avg_rate = weighted_rate / 5

        return {
            "strategy": f"5-Year {investment_type} Ladder ({currency})",
            "total_investment": f"${amount:,.2f}",
            "average_yield": f"{avg_rate:.2f}%",
            "estimated_annual_income": f"${total_income:,.2f}",
            "rungs": ladder,
            "data_source": data_note,
            "rationale": (
                "Laddering reduces reinvestment risk. "
                "Every year, 20% of your cash matures, allowing you to re-invest at current rules or use for income."
            ),
            "next_steps": f"Check your broker ({'Questrade/bank' if currency == 'CAD' else 'US Broker'}) for these specific rates."
        }

    except Exception as e:
        return {"error": f"Ladder construction failed: {str(e)}"}

@log_exceptions()
def _fetch_current_rates(inv_type: str, currency: str) -> tuple[dict[int, float], str]:
    """
    Get current per-tenor rates for the ladder and a one-line note on where
    they came from, so the caller can disclose freshness instead of silently
    presenting stale numbers as current.
    """
    if currency == "USD":
        from tools.fred_api import get_treasury_curve
        data = get_treasury_curve()
        curve = data.get("curve") if isinstance(data, dict) else None
        if curve:
            as_of = data.get("as_of")
            source = data.get("source", "FRED")
            note = f"Live US Treasury yields ({source}, as of {as_of})" if as_of else f"US Treasury yields ({source})"
            return {int(k): v for k, v in curve.items()}, note
        # FRED unreachable - last-resort constants, clearly labeled as such.
        return {1: 4.4, 2: 4.2, 3: 4.1, 4: 4.05, 5: 4.0}, "Fallback estimate (FRED unavailable) - NOT live data"

    # CAD: prefer the Bank of Canada's own posted chartered-bank GIC series
    # (Roadmap 5.7). It gives a REAL 1/3/5-year shape — the 2 and 4-year rungs are
    # interpolated between observed points and said to be — instead of the flat
    # policy-rate proxy that used to stand in for a curve here.
    from tools.boc_valet import get_cad_gic_curve
    from tools.tool_errors import is_unavailable

    gic = get_cad_gic_curve()
    if not is_unavailable(gic) and gic.get("curve"):
        curve = {int(k): float(v) for k, v in gic["curve"].items()}
        interpolated = ", ".join(f"{t}y" for t in gic.get("interpolated_tenors", []))
        note = (
            f"Live posted GIC rates from the Bank of Canada (as of {gic.get('observation_date')})"
            + (f"; {interpolated} interpolated between posted points" if interpolated else "")
            + ". Posted rates are a reference, not an offer - promotional GICs often price above them."
        )
        return curve, note

    # BoC unavailable: fall back to the policy-rate proxy via FRED, flat across
    # tenors, and say so rather than implying a curve shape nothing measured.
    from tools.fred_api import get_canada_metrics
    data = get_canada_metrics()
    anchor_rate = None
    raw_rate = data.get("interest_rate") if isinstance(data, dict) else None
    if isinstance(raw_rate, str) and raw_rate.endswith("%"):
        try:
            anchor_rate = float(raw_rate[:-1])
        except ValueError:
            anchor_rate = None

    if anchor_rate is not None:
        note = (
            f"Anchored to the BoC policy-rate proxy ({anchor_rate:.2f}%, via FRED) - "
            "flat across tenors because the Bank of Canada posted GIC curve was unavailable; "
            "confirm actual GIC/bond rates with your broker before buying."
        )
        return {year: anchor_rate for year in range(1, 6)}, note

    return (
        {1: 4.1, 2: 3.9, 3: 3.8, 4: 3.75, 5: 3.7},
        "Fallback estimate (BoC rate unavailable) - NOT live data",
    )

if __name__ == "__main__":
    print(construct_bond_ladder(100000, "GIC", "CAD"))
