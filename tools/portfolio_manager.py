from typing import Any

from tools.exception_logger import log_exceptions


@log_exceptions()
def get_hypothetical_portfolio(persona: str = "Aggressive Growth") -> dict[str, Any]:
    """
    Generates a representative portfolio based on an investor persona.
    Used when the user has not connected a real brokerage account.
    """
    persona = persona.lower()

    portfolios = {
        "aggressive": {
            "name": "Aggressive Growth (High Risk)",
            "holdings": [
                {"symbol": "QQQ", "allocation": "40%", "value": 40000, "description": "Tech Growth ETF"},
                {"symbol": "SMH", "allocation": "20%", "value": 20000, "description": "Semiconductors"},
                {"symbol": "NVDA", "allocation": "15%", "value": 15000, "description": "AI Leaader"},
                {"symbol": "IBIT", "allocation": "10%", "value": 10000, "description": "Bitcoin ETF"},
                {"symbol": "VGT", "allocation": "15%", "value": 15000, "description": "Vanguard Tech"}
            ],
            "total_value": 100000,
            "cash": 5000,
            "focus": "Maximizing capital appreciation. High volatility tolerance."
        },
        "conservative": {
            "name": "Conservative Income (Low Risk)",
            "holdings": [
                {"symbol": "VTI", "allocation": "30%", "value": 30000, "description": "Total Market"},
                {"symbol": "BND", "allocation": "40%", "value": 40000, "description": "Total Bond Market"},
                {"symbol": "SCHD", "allocation": "20%", "value": 20000, "description": "Dividend Equity"},
                {"symbol": "GLD", "allocation": "5%", "value": 5000, "description": "Gold"},
                {"symbol": "BIL", "allocation": "5%", "value": 5000, "description": "T-Bills"}
            ],
            "total_value": 100000,
            "cash": 10000,
            "focus": "Capital preservation and steady income. Low volatility."
        },
        "dividend": {
            "name": "Dividend Growth (Income)",
            "holdings": [
                {"symbol": "SCHD", "allocation": "25%", "value": 25000, "description": "US Dividend Equity"},
                {"symbol": "VIG", "allocation": "20%", "value": 20000, "description": "Dividend Appreciation"},
                {"symbol": "O", "allocation": "15%", "value": 15000, "description": "Realty Income (REIT)"},
                {"symbol": "JEPQ", "allocation": "15%", "value": 15000, "description": "Tech Income"},
                {"symbol": "MAIN", "allocation": "10%", "value": 10000, "description": "BDC Income"},
                {"symbol": "DGRO", "allocation": "15%", "value": 15000, "description": "Dividend Growth"}
            ],
            "total_value": 100000,
            "cash": 2000,
            "focus": "Generating passive income stream. Moderate growth."
        },
        "balanced": { # Default
            "name": "Balanced 60/40",
            "holdings": [
                {"symbol": "SPY", "allocation": "60%", "value": 60000, "description": "S&P 500"},
                {"symbol": "AGG", "allocation": "40%", "value": 40000, "description": "Aggregate Bond"}
            ],
            "total_value": 100000,
            "cash": 5000,
            "focus": "Standard risk/reward balance."
        }
    }

    # Simple keyword matching
    selected = portfolios["balanced"]
    if "aggressive" in persona or "growth" in persona:
        selected = portfolios["aggressive"]
    elif "conservative" in persona or "safe" in persona or "retire" in persona:
        selected = portfolios["conservative"]
    elif "dividend" in persona or "income" in persona:
        selected = portfolios["dividend"]

    return {
        "status": "Simulated Portfolio",
        "rationale": f"User requested analysis for '{persona}' style.",
        "portfolio": selected
    }
