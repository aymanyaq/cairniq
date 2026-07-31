from typing import Any

import yfinance as yf

from tools.exception_logger import log_exceptions
from tools.graph_memory import graph_memory


@log_exceptions()
def analyze_mutual_funds(symbols: list[str]) -> dict[str, Any]:
    """
    Analyze Mutual Funds and Pension holdings for fees (MER) and performance.
    """
    results = []
    total_fees = 0.0
    count = 0
    warnings = []

    for sym in symbols:
        clean_sym = sym.strip()

        # 1. Check Knowledge Graph first (for private/pension funds)
        kg_node = graph_memory.graph.nodes.get(clean_sym)
        if kg_node and kg_node.get("asset_type", "").lower() == "private" and "expense_ratio" in kg_node:
            er = kg_node["expense_ratio"]
            name = kg_node.get("name", clean_sym)
            cat = kg_node.get("category", "Private Fund")
            src = "Knowledge Graph"
        else:
            # 2. Try Yahoo Finance
            try:
                ticker = yf.Ticker(clean_sym)
                info = ticker.info

                name = info.get("longName", clean_sym)
                # Try multiple fields for fees
                er = info.get("annualReportExpenseRatio") or info.get("expenseRatio") or 0.0
                cat = info.get("category", "Unknown")
                src = "Live API"

                # If API returns success but 0.0 fees (often means data missing for pension funds), force search
                if er == 0.0 and "Fund" in name:
                     raise Exception("Force Web Search for missing fees")

            except Exception:
                # 3. Last Resort: Web Search (Targeting Morningstar/Fund Facts)
                try:
                    from tools.web_search import search_news
                    query = f"{clean_sym} fund facts mer fees holdings morningstar.ca"
                    search_results = search_news(query, max_results=2)

                    if search_results:
                        # Simple extraction heuristic (LLM would be better here, but we use rule-based for now)
                        # We just return the snippet as "Details" and flag it
                        str(search_results)
                        name = f"{clean_sym} (Web Search)"
                        er = 0.0 # Unknown numeric, but text provided
                        cat = "Web Search"
                        src = f"Web Search: {search_results[0].get('title', 'Result')}"
                        warnings.append(f"ℹ️ {clean_sym}: Fees not structured. Check search result: {search_results[0].get('href')}")
                    else:
                        raise Exception("No search results")
                except Exception:
                    name = clean_sym
                    er = 0.0
                    cat = "Unknown"
                    src = "Failed"

        # Convert to percentage for display
        er_pct = er * 100 if er < 1 else er # Handle 0.01 vs 1.0 format

        results.append({
            "symbol": clean_sym,
            "name": name,
            "expense_ratio": f"{er_pct:.2f}%" if src != "Failed" else "N/A",
            "category": cat,
            "data_source": src
        })

        if er_pct > 1.5:
            warnings.append(f"⚠️ {clean_sym}: High Fee Warning ({er_pct:.2f}%). Consider lower-cost ETF alternatives.")

        if er > 0:
            total_fees += er_pct
            count += 1

    avg_fee = total_fees / count if count > 0 else 0

    return {
        "funds_analyzed": len(results),
        "average_expense_ratio": f"{avg_fee:.2f}%",
        "fee_rating": (
            "Low (<0.5%)" if avg_fee < 0.5 else
            "Moderate (0.5-1.0%)" if avg_fee < 1.0 else
            "High (>1.0%)"
        ),
        "details": results,
        "warnings": warnings,
        "recommendation": "Pension funds often have high fees. Compare performance net of fees against VTI/VEQT.",
        "note": "For 'Web Search' results, verify MER manually in the provided link."
    }

if __name__ == "__main__":
    print(analyze_mutual_funds(["FXAIX", "VTSAX", "RY"]))
