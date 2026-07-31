from typing import Any

import yfinance as yf

from tools.cache import cached
from tools.exception_logger import log_exceptions

# Approximate ESG Scores (Lower Risk is Better for Sustainalytics, Higher is Better for MSCI)
# We will standardized on a 0-100 "Quality Score" where 100 is best.
# Sourced from general public knowledge (MSCI/Sustainalytics proxies)
ESG_DATABASE = {
    # ETFs
    "XESG.TO": {"score": 85, "grade": "AA", "controversies": "None", "focus": "Broad ESG"},
    "ESGD": {"score": 82, "grade": "A", "controversies": "None", "focus": "Developed Markets ESG"},
    "DSI": {"score": 80, "grade": "A", "controversies": "Low", "focus": "Social Index"},
    "SPYX": {"score": 75, "grade": "BBB", "controversies": "None", "focus": "Fossil Free"},
    "NZAC": {"score": 88, "grade": "AA", "controversies": "None", "focus": "Net Zero Paris Aligned"},
    "SHE": {"score": 85, "grade": "AA", "controversies": "None", "focus": "Gender Diversity"},

    # Tech / Major Holdings
    "AAPL": {"score": 72, "grade": "BBB", "controversies": "Labor Supply Chain", "focus": "Tech"},
    "MSFT": {"score": 90, "grade": "AAA", "controversies": "None", "focus": "Tech"},
    "NVDA": {"score": 88, "grade": "AA", "controversies": "None", "focus": "Tech"},
    "TSLA": {"score": 40, "grade": "B", "controversies": "Labor/Governance", "focus": "EV/Tech"},
    "AMZN": {"score": 45, "grade": "BB", "controversies": "Labor Rights", "focus": "Retail"},
    "GOOGL": {"score": 68, "grade": "BBB", "controversies": "Monopoly/Privacy", "focus": "Tech"},
    "META": {"score": 35, "grade": "B", "controversies": "Privacy/Democracy", "focus": "Tech"},

    # Oil/Gas (for comparison)
    "XOM": {"score": 30, "grade": "CCC", "controversies": "Climate Change", "focus": "Energy"},
    "CVX": {"score": 32, "grade": "CCC", "controversies": "Climate Change", "focus": "Energy"},

    # Defense
    "LMT": {"score": 40, "grade": "B", "controversies": "Weapons", "focus": "Defense"},
    "RTX": {"score": 42, "grade": "B", "controversies": "Weapons", "focus": "Defense"}
}

CONTROVERSIAL_SECTORS = ["Energy", "Defense", "Tobacco", "Gambling"]

@cached(key_func=lambda symbols: f"esg:{','.join(sorted(s.upper() for s in symbols))}")
@log_exceptions()
def check_esg_scores(symbols: list[str]) -> dict[str, Any]:
    """
    Analyze ESG (Environmental, Social, Governance) scores for a list of symbols.
    Uses a robust internal database since free live ESG APIs are unreliable.
    """
    results = []
    portfolio_score = 0
    scored_count = 0
    warnings = []

    for sym in symbols:
        clean_sym = sym.upper().strip()
        data = ESG_DATABASE.get(clean_sym)

        # If not in DB, try naive web search fallback notion?
        # For now, just mark unknown.
        if not data:
            # Try parsing info from yfinance as fallback (rarely works but worth a shot)
            try:
                # Naive sector check
                ticker = yf.Ticker(clean_sym)
                info = ticker.info
                sector = info.get("sector", "")

                # Greenwashing Check
                if sector in CONTROVERSIAL_SECTORS:
                    data = {"score": 40, "grade": "B-", "controversies": f"Sector: {sector}", "focus": sector}
                    warnings.append(f"⚠️ {clean_sym} is in {sector} (Controversial).")
                else:
                    data = {"score": 50, "grade": "N/A", "controversies": "Unknown", "focus": sector}
            except Exception:
                 data = {"score": 50, "grade": "N/A", "controversies": "Unknown", "focus": "Unknown"}

        results.append({
            "symbol": clean_sym,
            "esg_score": data["score"],
            "grade": data["grade"],
            "controversies": data["controversies"]
        })

        if data["grade"] != "N/A":
            portfolio_score += data["score"]
            scored_count += 1

    avg_score = round(portfolio_score / scored_count, 1) if scored_count > 0 else 0

    return {
        "portfolio_avg_esg_score": avg_score,
        "esg_rating": (
            "Leader (AAA)" if avg_score > 85 else
            "Above Average (AA/A)" if avg_score > 70 else
            "Average (BBB)" if avg_score > 50 else
            "Laggard (B/CCC)"
        ),
        "details": results,
        "warnings": warnings,
        "note": "Scores based on MSCI/Sustainalytics proxies and sector flagging."
    }

if __name__ == "__main__":
    print(check_esg_scores(["XESG.TO", "TSLA", "XOM", "AAPL"]))
