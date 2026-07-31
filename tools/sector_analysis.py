import re
from typing import Any

import yfinance as yf

from tools.exception_logger import log_exceptions
from tools.graph_memory import graph_memory

# ETF/Fund Sector Decomposition Database
# Approximate weightings for widely-held funds this app can decompose
FUND_COMPOSITION = {
    # Tech / Growth Funds (The "Hidden" Tech Exposure)
    "FTEC": {"Technology": 1.0},
    "FSCSX": {"Technology": 1.0},
    "TEC.TO": {"Technology": 1.0},
    "QTUM": {"Technology": 0.9, "Industrials": 0.1}, # Quantum often hardware/industrial
    "XLK": {"Technology": 1.0},
    "QQQ": {"Technology": 0.5, "Communication Services": 0.2, "Consumer Cyclical": 0.2, "Healthcare": 0.1},
    "XQQ.TO": {"Technology": 0.5, "Communication Services": 0.2, "Consumer Cyclical": 0.2, "Healthcare": 0.1}, # Cdn-hedged Nasdaq 100 (= QQQ)
    "SMH": {"Technology": 1.0},

    # Single-sector SPDRs / iShares — 100% one sector (pure, stable mappings)
    "XLB": {"Basic Materials": 1.0},
    "XLI": {"Industrials": 1.0},
    "XLV": {"Healthcare": 1.0},
    "XLU": {"Utilities": 1.0},
    "XLY": {"Consumer Cyclical": 1.0},
    "XLP": {"Consumer Defensive": 1.0},
    "XLF": {"Financial Services": 1.0},
    "XLE": {"Energy": 1.0},
    "XLC": {"Communication Services": 1.0},
    "XLRE": {"Real Estate": 1.0},
    "VNQ": {"Real Estate": 1.0},
    "XUT.TO": {"Utilities": 1.0}, # iShares S&P/TSX Capped Utilities

    # Canadian Broad Market (Financials/Energy Heavy)
    "XIU.TO": {"Financial Services": 0.35, "Energy": 0.18, "Industrials": 0.12, "Basic Materials": 0.10, "Technology": 0.08, "Other": 0.17},
    "XIC.TO": {"Financial Services": 0.30, "Energy": 0.15, "Industrials": 0.12, "Basic Materials": 0.10, "Other": 0.33},
    "VCN.TO": {"Financial Services": 0.30, "Energy": 0.15, "Industrials": 0.12, "Basic Materials": 0.10, "Other": 0.33},
    "VDY.TO": {"Financial Services": 0.55, "Energy": 0.30, "Utilities": 0.10, "Communication Services": 0.05},

    # ESG / Diversified
    "XESG.TO": {"Financial Services": 0.32, "Energy": 0.05, "Technology": 0.15, "Industrials": 0.12, "Other": 0.36}, # Lower energy than TSX
    "XEN.TO": {"Financial Services": 0.32, "Energy": 0.05, "Technology": 0.15, "Industrials": 0.12, "Other": 0.36}, # iShares ESG Cdn (~ XESG)
    "ESGD": {"Financial Services": 0.20, "Industrials": 0.15, "Healthcare": 0.12, "Consumer Cyclical": 0.11, "Technology": 0.08, "Other": 0.34},
    "DSI": {"Technology": 0.33, "Healthcare": 0.15, "Financial Services": 0.10, "Consumer Cyclical": 0.10, "Other": 0.32},
    "NZAC": {"Technology": 0.25, "Financial Services": 0.16, "Industrials": 0.11, "Consumer Cyclical": 0.11, "Healthcare": 0.10, "Communication Services": 0.08, "Consumer Defensive": 0.06, "Energy": 0.04, "Basic Materials": 0.04, "Utilities": 0.03, "Real Estate": 0.02}, # SPDR MSCI ACWI Climate (~ global)

    # Broad-market / thematic (approximate GICS weights; auto-refined via FMP when its quota allows)
    "SPYX": {"Technology": 0.33, "Financial Services": 0.14, "Consumer Cyclical": 0.11, "Healthcare": 0.11, "Communication Services": 0.09, "Industrials": 0.09, "Consumer Defensive": 0.06, "Utilities": 0.03, "Basic Materials": 0.02, "Real Estate": 0.02}, # S&P 500 fossil-fuel-free
    "SCHD": {"Industrials": 0.18, "Financial Services": 0.17, "Healthcare": 0.15, "Consumer Defensive": 0.13, "Energy": 0.10, "Technology": 0.10, "Consumer Cyclical": 0.08, "Basic Materials": 0.05, "Communication Services": 0.04}, # Dividend value tilt
    "EMXC": {"Technology": 0.30, "Financial Services": 0.22, "Consumer Cyclical": 0.10, "Basic Materials": 0.09, "Industrials": 0.08, "Energy": 0.06, "Communication Services": 0.06, "Consumer Defensive": 0.05, "Healthcare": 0.04}, # EM ex-China
    "FSEAX": {"Technology": 0.28, "Financial Services": 0.20, "Consumer Cyclical": 0.14, "Communication Services": 0.10, "Consumer Defensive": 0.07, "Industrials": 0.07, "Basic Materials": 0.06, "Energy": 0.04, "Healthcare": 0.04}, # Fidelity Emerging Asia
    "NUKZ": {"Utilities": 0.38, "Industrials": 0.27, "Energy": 0.18, "Basic Materials": 0.12, "Technology": 0.05}, # Nuclear energy theme

    # Pension / Balanced (Approx 60/40 logic)
    # Note: Private/named funds (e.g. target-date pensions) should be stored in the
    # Knowledge Graph with a "sector_breakdown" attribute — see check_portfolio_allocation.
    # Anything unresolved that looks like a named fund falls back to "Diversified Fund".

    # Commodities / Crypto
    "GLD": {"Basic Materials": 1.0}, # Gold
    "FBTC": {"Crypto": 1.0}, "IBIT": {"Crypto": 1.0}, "ETHA": {"Crypto": 1.0},
    "BTC-USD": {"Crypto": 1.0},
    "ETH-USD": {"Crypto": 1.0}
}

SECTOR_MAPPING = {
    # Normalize sector names
    "Information Technology": "Technology",
    "Tech": "Technology",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Finance": "Financial Services",
    "Financials": "Financial Services",
    "Materials": "Basic Materials",
    "Telecommunication Services": "Communication Services",
    "Communication": "Communication Services",
}

# Broker cash lines / money-market ETFs — a real allocation bucket, not "Unknown".
_CASH_SYMBOLS = {
    "CASH", "CAD", "USD",
    "CASH.TO", "MNY.TO", "HISA.TO", "PSA.TO", "CBIL.TO", "ZMMK.TO", "HSAV.TO",
}


def _is_cash(sym: str) -> bool:
    return sym in _CASH_SYMBOLS or sym.startswith("CASH.")


# Named funds carry spaces and/or a target-date year; real tickers do not. Such a
# holding (e.g. a target-date pension like "ACME 2045 TARGET FUND") has no market
# quote, so rather than dropping it into "Unknown" we treat it as a diversified fund.
_TARGET_DATE_RE = re.compile(r"\b20[2-9]\d\b")
_FUND_NAME_HINTS = ("FUND", "TARGET", "RETIRE", "PORTFOLIO", "BALANCED", "PENSION")


def _looks_like_diversified_fund(sym: str) -> bool:
    up = sym.upper()
    if " " not in up:  # market tickers are single tokens
        return False
    return bool(_TARGET_DATE_RE.search(up)) or any(h in up for h in _FUND_NAME_HINTS)


def _apply_breakdown(sector_exposure: dict, breakdown: dict, weight_in_port: float) -> None:
    """Distribute a position's weight across a fund's sector breakdown."""
    for sec, fund_weight in breakdown.items():
        norm_sec = SECTOR_MAPPING.get(sec, sec)
        sector_exposure[norm_sec] = sector_exposure.get(norm_sec, 0.0) + (weight_in_port * fund_weight)


def _fmp_decompose(sym: str) -> dict[str, float] | None:
    """Decompose an ETF/fund into normalized sector fractions via FMP, or None."""
    try:
        from tools.fmp_api import get_fmp_etf_sector_weightings
        raw = get_fmp_etf_sector_weightings(sym)
    except Exception:
        return None
    if not raw:
        return None
    norm: dict[str, float] = {}
    for sec, frac in raw.items():
        # Skip non-numeric weights rather than trusting every key to be a sector.
        # Caches written before `stamp=False` still carry an `_as_of` string on
        # disk for the rest of their TTL, and one of those took the whole sector
        # breakdown down (float + str) for two days; a map read defensively
        # survives its own history.
        if isinstance(frac, bool) or not isinstance(frac, (int, float)):
            continue
        key = SECTOR_MAPPING.get(sec, sec)
        norm[key] = norm.get(key, 0.0) + float(frac)
    total = sum(norm.values())
    if total <= 0:
        return None
    # Normalize so the full position weight is distributed (FMP weights rarely sum to exactly 1).
    return {k: v / total for k, v in norm.items()}

# How this module measures sector exposure. Paired with
# opportunity_scanner.SECTOR_EXPOSURE_BASIS_DIRECT — the two are different
# measures of the same book, and each publishes which one it is.
SECTOR_EXPOSURE_BASIS_LOOKTHROUGH = "look-through; ETFs and funds decomposed into sector sleeves"


@log_exceptions()
def check_portfolio_allocation(symbols: list[str], amounts: list[float] = None, allow_network: bool = True) -> dict[str, Any]:
    """
    Analyze true sector exposure by decomposing ETFs/Funds.

    Args:
        symbols: List of tickers
        amounts: Value or Weight of each position.
                 If empty, assumes equal weight (not recommended for accurate analysis).
        allow_network: When False, cache misses are bucketed as Unknown instead
                 of resolving via yfinance/FMP — for deterministic audit paths
                 (the IPS pre-check) that must never stall on live lookups.
    """
    if amounts is None:
        amounts = []
    if not amounts or len(amounts) != len(symbols):
        # Default to equal weight if no amounts provided
        amounts = [1.0] * len(symbols)

    total_value = sum(amounts)
    sector_exposure = {}
    breakdown_details = []

    for i, sym in enumerate(symbols):
        clean_sym = sym.strip().upper()
        val = amounts[i]
        weight_in_port = (val / total_value) if total_value else 0.0

        # 1. Knowledge Graph (private/pension funds stored with a sector_breakdown)
        kg_node = graph_memory.graph.nodes.get(clean_sym)
        if kg_node and "sector_breakdown" in kg_node:
            fund_breakdown = kg_node["sector_breakdown"]
            source = "Knowledge Graph"
            details_str = f"Decomposed (KG): {fund_breakdown}"
            _apply_breakdown(sector_exposure, fund_breakdown, weight_in_port)

        # 2. Static Fund Database
        elif clean_sym in FUND_COMPOSITION:
            fund_breakdown = FUND_COMPOSITION[clean_sym]
            source = "Fund Decomposition DB"
            details_str = f"Decomposed: {fund_breakdown}"
            _apply_breakdown(sector_exposure, fund_breakdown, weight_in_port)

        # 3. Cash / money-market lines
        elif _is_cash(clean_sym):
            source = "Cash"
            details_str = "Cash / Money Market"
            sector_exposure["Cash"] = sector_exposure.get("Cash", 0.0) + weight_in_port

        else:
            # 4. Resolve via Cache -> yfinance (stocks) -> FMP decomposition (funds)
            from tools.daily_cache import get_cached, set_cached

            cache_key = f"sector_{clean_sym}"
            cached_val = get_cached(cache_key)

            if isinstance(cached_val, dict) and cached_val:
                source = "Cache (Decomposed)"
                details_str = f"Cached Decomposition: {cached_val}"
                _apply_breakdown(sector_exposure, cached_val, weight_in_port)
            elif isinstance(cached_val, str) and cached_val:
                source = "Cache"
                details_str = f"Cached Sector: {cached_val}"
                sector_exposure[cached_val] = sector_exposure.get(cached_val, 0.0) + weight_in_port
            elif not allow_network:
                source = "Offline"
                details_str = "Cache miss (network disabled for this call)"
                sector_exposure["Unknown"] = sector_exposure.get("Unknown", 0.0) + weight_in_port
            else:
                # Try yfinance for a single-stock sector first (fast path for equities).
                norm_sec = None
                is_fund = False
                try:
                    ticker = yf.Ticker(clean_sym)
                    info = ticker.info or {}
                    sec = info.get("sector")
                    norm_sec = SECTOR_MAPPING.get(sec, sec)
                    qtype = (info.get("quoteType") or "").lower()
                    is_fund = "etf" in qtype or "fund" in qtype
                except Exception:
                    norm_sec = None

                if norm_sec and norm_sec != "Unknown":
                    source = "API"
                    details_str = "Single Stock"
                    sector_exposure[norm_sec] = sector_exposure.get(norm_sec, 0.0) + weight_in_port
                    set_cached(cache_key, norm_sec)
                else:
                    # No single sector — decompose the fund via FMP before giving up.
                    fmp_bd = _fmp_decompose(clean_sym)
                    if fmp_bd:
                        source = "FMP Decomposition"
                        details_str = f"Decomposed (FMP): {{{', '.join(f'{k}: {v:.2f}' for k, v in fmp_bd.items())}}}"
                        _apply_breakdown(sector_exposure, fmp_bd, weight_in_port)
                        set_cached(cache_key, fmp_bd)
                    elif is_fund:
                        source = "API"
                        details_str = "Unclassified Fund (no sector data)"
                        sector_exposure["Unclassified Fund"] = sector_exposure.get("Unclassified Fund", 0.0) + weight_in_port
                    elif _looks_like_diversified_fund(clean_sym):
                        source = "Heuristic"
                        details_str = "Named fund — no market quote; treated as diversified"
                        sector_exposure["Diversified Fund"] = sector_exposure.get("Diversified Fund", 0.0) + weight_in_port
                    else:
                        source = "API"
                        details_str = "Unknown"
                        sector_exposure["Unknown"] = sector_exposure.get("Unknown", 0.0) + weight_in_port

        breakdown_details.append({
            "symbol": clean_sym,
            "weight_in_portfolio": f"{weight_in_port*100:.1f}%",
            "classification_source": source,
            "sector_details": details_str
        })

    # Format Output
    sorted_sectors = sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True)
    formatted_exposure = {k: f"{v*100:.1f}%" for k, v in sorted_sectors}

    # Generate Insights
    tech_weight = sector_exposure.get("Technology", 0)
    insights = []

    if tech_weight > 0.40:
        insights.append(f"⚠️ HIGH TECH CONCENTRATION: {tech_weight*100:.1f}% of your portfolio is in Technology.")
        insights.append("   (This includes the 'hidden' exposure inside your tech-sector ETFs)")

    energy_weight = sector_exposure.get("Energy", 0)
    if energy_weight > 0.15:
         insights.append(f"Energy Exposure ({energy_weight*100:.1f}%) is high relative to global benchmarks.")

    unclassified = sector_exposure.get("Unclassified Fund", 0) + sector_exposure.get("Unknown", 0)
    if unclassified > 0.10:
        insights.append(
            f"ℹ️ {unclassified*100:.1f}% is still unclassified — add these funds' holdings "
            "(or a sector_breakdown in memory) to sharpen the sector view."
        )

    return {
        "portfolio_total_value": f"${total_value:,.0f}",
        "sector_allocation": formatted_exposure,
        # Raw fractions (0..1) for numeric consumers (IPS pre-check) — the
        # formatted map above loses precision to display rounding.
        "sector_allocation_raw": {k: round(v, 6) for k, v in sorted_sectors},
        # What these percentages MEAN. The opportunity scanner answers the same
        # "how much Technology do I hold?" question on a different basis — one
        # label per ticker, funds not decomposed — and reports a lower number out
        # of the same book. Stating each basis is what stops the pair reading as a
        # contradiction; unlabelled, the scanner's smaller figure was taken as
        # proof this one was fabricated (2026-07-29, a 2/10 SOURCE FRAUD verdict).
        "basis": SECTOR_EXPOSURE_BASIS_LOOKTHROUGH,
        "key_insights": insights,
        "holding_details": breakdown_details
    }

if __name__ == "__main__":
    # Smoke test with a synthetic allocation
    print(check_portfolio_allocation(
        ["FTEC", "FXAIX", "AAPL", "XIC.TO"],
        [20000, 25000, 15000, 50000]
    ))
