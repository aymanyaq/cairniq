"""
Geopolitical Supply-Chain Opportunity Scanner
----------------------------------------------
Detects geopolitical events (wars, sanctions, natural disasters, trade policy)
and maps them to investable opportunities through a supply-chain knowledge base.

Logic Chain:
  Event (Iran strikes Qatar)
  → Affected Country (Qatar)
  → Affected Commodities (Helium, LNG, Natural Gas)
  → Beneficiary Tickers (GLNG, LNG, AR, EQT)
  → Scored & Ranked Opportunities

The knowledge base is STATIC (not LLM-generated) for speed and reliability.
The LLM interprets news; the knowledge base maps to tickers.
"""

from typing import Any

import yfinance as yf

from agent.utils import is_cancelled, safe_print
from tools.cache import cached
from tools.exception_logger import log_exceptions
from tools.geopolitical_data import (
    COMMODITY_TICKER_MAP,
    CONFLICT_PEAK_PRICES,
    COUNTRY_ALIASES,
    COUNTRY_COMMODITY_MAP,
    DOWNSTREAM_EFFECTS_MAP,
    EVENT_TYPE_MAP,
)
from tools.web_search import search_news

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

@log_exceptions()
def _normalize_country(text: str) -> str:
    """Normalize a country name to our knowledge base key."""
    text = text.lower().strip().replace("-", "_").replace(" ", "_")
    # Check aliases first
    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]
    # Check direct match
    if text in COUNTRY_COMMODITY_MAP:
        return text
    # Check partial match
    for key in COUNTRY_COMMODITY_MAP:
        if key in text or text in key:
            return key
    return text


@log_exceptions()
def _detect_countries_in_text(text: str) -> list[str]:
    """Find all known countries/regions mentioned in text."""
    text_lower = text.lower()
    found = []

    # Check all known countries and aliases
    all_names = list(COUNTRY_COMMODITY_MAP.keys()) + list(COUNTRY_ALIASES.keys())
    for name in all_names:
        # Convert underscores to spaces for matching
        search_term = name.replace("_", " ")
        if search_term in text_lower:
            normalized = _normalize_country(name)
            if normalized not in found:
                found.append(normalized)

    return found


@log_exceptions()
def _detect_event_types(text: str) -> list[str]:
    """Detect what kind of geopolitical event this is."""
    text_lower = text.lower()
    detected = []
    for event_type, _ in EVENT_TYPE_MAP.items():
        if event_type.replace("_", " ") in text_lower or event_type in text_lower:
            detected.append(event_type)
    return detected


@log_exceptions()
def _get_tickers_for_commodity(commodity: str) -> list[str]:
    """Get all unique tickers for a commodity, flattening the category dictionaries."""
    tickers = set()
    entry = COMMODITY_TICKER_MAP.get(commodity, {})
    for key, value in entry.items():
        if key == "desc":
            continue
        if isinstance(value, list):
            for t in value:
                # Handle entries like "platinum → PPLT"
                if "→" in t:
                    t = t.split("→")[-1].strip()
                # Handle entries like "II-VI (COHR)"
                if "(" in t:
                    t = t.split("(")[1].replace(")", "").strip()
                tickers.add(t)
    return list(tickers)


@log_exceptions()
def _get_commodity_description(commodity: str) -> str:
    """Get description for a commodity."""
    entry = COMMODITY_TICKER_MAP.get(commodity, {})
    return entry.get("desc", commodity.replace("_", " ").title())


@log_exceptions()
def _quick_price_check(symbol: str) -> dict[str, Any]:
    """Quick yfinance check: price, recent change, volume spike, and 52-week context."""
    try:
        ticker = yf.Ticker(symbol)
        # Use 6mo for better trend analysis
        hist = ticker.history(period="6mo")
        if hist.empty or len(hist) < 5:
            return {"symbol": symbol, "error": "No data"}

        current = float(hist['Close'].iloc[-1])
        week_ago = float(hist['Close'].iloc[-5]) if len(hist) >= 5 else float(hist['Close'].iloc[0])
        month_ago_idx = max(0, len(hist) - 22)  # ~22 trading days = 1 month
        month_ago = float(hist['Close'].iloc[month_ago_idx])
        six_month_ago = float(hist['Close'].iloc[0])

        weekly_change = ((current - week_ago) / week_ago) * 100 if week_ago else 0
        monthly_change = ((current - month_ago) / month_ago) * 100 if month_ago else 0
        six_month_change = ((current - six_month_ago) / six_month_ago) * 100 if six_month_ago else 0

        # 52-week high/low from the data we have
        high_6mo = float(hist['High'].max())
        low_6mo = float(hist['Low'].min())
        pct_from_high = ((current - high_6mo) / high_6mo) * 100 if high_6mo else 0

        # Volume spike check
        avg_vol = float(hist['Volume'].mean())
        recent_vol = float(hist['Volume'].tail(3).mean())
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        return {
            "symbol": symbol,
            "price": round(current, 2),
            "weekly_change_pct": round(weekly_change, 2),
            "monthly_change_pct": round(monthly_change, 2),
            "six_month_change_pct": round(six_month_change, 2),
            "pct_from_6mo_high": round(pct_from_high, 2),
            "high_6mo": round(high_6mo, 2),
            "low_6mo": round(low_6mo, 2),
            "volume_spike": round(vol_ratio, 2),
            "trending": vol_ratio > 1.5 or abs(weekly_change) > 5,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


@log_exceptions()
def _analyze_conflict_premium(commodity: str, current_price: float) -> dict[str, Any]:
    """
    Analyze how much MORE a commodity-linked asset can rise based on historical conflict peaks.

    Returns a conflict premium assessment:
    - NOT PRICED IN: Current price is well below historical conflict peaks
    - PARTIALLY PRICED: Price has moved but still has significant upside to peak
    - FULLY PRICED: Price is near or above historical conflict peaks
    """
    peak_data = CONFLICT_PEAK_PRICES.get(commodity)
    if not peak_data or not current_price:
        return {"status": "unknown", "commodity": commodity}

    # Get current price of the proxy commodity ticker for comparison
    proxy_ticker = peak_data.get("ticker")
    avg_premium_pct = peak_data.get("historical_avg_conflict_premium_pct", 30)
    peaks = peak_data.get("peaks", [])

    if not peaks:
        return {"status": "unknown", "commodity": commodity}

    # Use the most relevant (most recent) peak
    latest_peak = peaks[0]["peak"]
    latest_event = peaks[0]["event"]

    # Try to get actual commodity price for better analysis
    try:
        if proxy_ticker:
            proxy_data = _quick_price_check(proxy_ticker)
            if "error" not in proxy_data:
                proxy_price = proxy_data["price"]
                pct_below_peak = ((latest_peak - proxy_price) / latest_peak) * 100 if latest_peak else 0

                if pct_below_peak > 30:
                    return {
                        "status": "NOT_PRICED_IN",
                        "verdict": f"⚠️ NOT priced in — {commodity.title()} proxy ({proxy_ticker}) at ${proxy_price:.0f}, still {pct_below_peak:.0f}% below conflict peak of ${latest_peak:.0f} ({latest_event})",
                        "upside_to_peak_pct": round(((latest_peak - proxy_price) / proxy_price) * 100, 1),
                        "avg_conflict_premium_pct": avg_premium_pct,
                        "current_proxy_price": proxy_price,
                        "conflict_peak": latest_peak,
                        "peak_event": latest_event,
                    }
                elif pct_below_peak > 10:
                    return {
                        "status": "PARTIALLY_PRICED",
                        "verdict": f"🟡 Partially priced — {commodity.title()} ({proxy_ticker}) at ${proxy_price:.0f}, {pct_below_peak:.0f}% below peak of ${latest_peak:.0f}. Still has room if conflict escalates.",
                        "upside_to_peak_pct": round(((latest_peak - proxy_price) / proxy_price) * 100, 1),
                        "avg_conflict_premium_pct": avg_premium_pct,
                        "current_proxy_price": proxy_price,
                        "conflict_peak": latest_peak,
                        "peak_event": latest_event,
                    }
                elif pct_below_peak <= 0:
                    # Price is ABOVE the historical conflict peak
                    return {
                        "status": "ABOVE_PEAK",
                        "verdict": f"🔴 Already ABOVE conflict peak — {commodity.title()} ({proxy_ticker}) at ${proxy_price:.0f}, exceeds prior peak of ${latest_peak:.0f} ({latest_event}). Momentum play only.",
                        "upside_to_peak_pct": 0,
                        "avg_conflict_premium_pct": avg_premium_pct,
                        "current_proxy_price": proxy_price,
                        "conflict_peak": latest_peak,
                        "peak_event": latest_event,
                    }
                else:
                    return {
                        "status": "NEAR_PEAK",
                        "verdict": f"🟡 Near conflict peak — {commodity.title()} ({proxy_ticker}) at ${proxy_price:.0f}, only {pct_below_peak:.0f}% below peak of ${latest_peak:.0f}. Limited additional upside.",
                        "upside_to_peak_pct": round(((latest_peak - proxy_price) / proxy_price) * 100, 1),
                        "avg_conflict_premium_pct": avg_premium_pct,
                        "current_proxy_price": proxy_price,
                        "conflict_peak": latest_peak,
                        "peak_event": latest_event,
                    }
    except Exception:
        pass

    # Fallback: use the avg conflict premium as a rough estimate
    return {
        "status": "LIKELY_NOT_PRICED_IN",
        "verdict": f"Historical data shows {commodity.title()} typically spikes +{avg_premium_pct}% during conflicts. Market rarely fully prices in supply disruptions upfront.",
        "avg_conflict_premium_pct": avg_premium_pct,
        "conflict_peak": latest_peak,
        "peak_event": latest_event,
    }


# =============================================================================
# PUBLIC API
# =============================================================================

@cached(key_func=lambda event_description=None: f"geopolitical_scan:{(event_description or 'auto')[:50]}")
@log_exceptions()
def scan_geopolitical_opportunities(event_description: str = None) -> dict[str, Any]:
    """
    Scan for geopolitical event-driven investment opportunities.

    Args:
        event_description: Optional specific event (e.g., "Iran strike on Qatar").
                          If None, auto-detects from latest news.

    Returns:
        Dict with detected events, affected commodities, and ranked ticker opportunities.
    """
    safe_print("🌍 Geopolitical Scanner: Analyzing global supply chain disruptions...")

    # --- Step 1: Detect or parse the event ---
    events = []

    if event_description:
        # User provided a specific event
        safe_print(f"🎯 Analyzing specific event: {event_description}")
        events.append({
            "description": event_description,
            "countries": _detect_countries_in_text(event_description),
            "event_types": _detect_event_types(event_description),
            "source": "User-provided",
        })

    # Also search for latest geopolitical news (whether user-provided or auto-detect)
    # PRIORITY: GDELT Project (global event firehose) → web search (fallback)

    # --- Try GDELT first ---
    gdelt_success = False
    try:
        from tools.gdelt_api import scan_gdelt_geopolitical, search_gdelt_events

        if event_description:
            # User-provided event: search GDELT for related articles
            gdelt_articles = search_gdelt_events(event_description, max_results=10, timespan="7d")
            if gdelt_articles:
                gdelt_success = True
                combined_text = " ".join(a.get("title", "") for a in gdelt_articles)
                countries = _detect_countries_in_text(combined_text)
                event_types = _detect_event_types(combined_text)
                if countries:
                    headlines = "\n".join(f"• {a['title'][:100]} ({a['source']})" for a in gdelt_articles[:5])
                    events.append({
                        "description": f"GDELT Event Monitor:\n{headlines}",
                        "countries": countries,
                        "event_types": event_types,
                        "source": "GDELT Project",
                    })
        else:
            # Auto-detect: scan across all GDELT themes
            gdelt_scan = scan_gdelt_geopolitical(
                themes=["military_conflict", "sanctions_embargo", "supply_chain", "energy_crisis"],
                timespan="3d",
                max_per_theme=5,
            )
            for theme, articles in gdelt_scan.items():
                if not articles:
                    continue
                gdelt_success = True
                combined_text = " ".join(a.get("title", "") for a in articles)
                countries = _detect_countries_in_text(combined_text)
                event_types = _detect_event_types(combined_text)
                if countries:
                    headlines = "\n".join(f"• {a['title'][:100]} ({a['source']})" for a in articles[:3])
                    events.append({
                        "description": f"GDELT [{theme}]:\n{headlines}",
                        "countries": countries,
                        "event_types": event_types,
                        "source": f"GDELT ({theme})",
                    })

        if gdelt_success:
            safe_print("🌐 GDELT: Successfully retrieved global event data")
    except Exception as e:
        safe_print(f"⚠️ GDELT failed: {e}")

    # --- Fallback to web search if GDELT returned nothing ---
    if not gdelt_success:
        safe_print("🔄 GDELT unavailable, falling back to web search...")
        search_queries = [
            "geopolitical supply chain disruption commodity 2026",
            "sanctions embargo trade war commodity prices",
            "military conflict oil gas supply disruption",
        ]

        if event_description:
            search_queries = [event_description + " investment impact commodity"]

        for query in search_queries:
            if is_cancelled():
                break
            try:
                results = search_news(query, max_results=5, timelimit="w")
                if results and results != "No results found.":
                    # Extract countries and event types from search results
                    countries = _detect_countries_in_text(results)
                    event_types = _detect_event_types(results)
                    if countries:
                        events.append({
                            "description": results[:500],
                            "countries": countries,
                            "event_types": event_types,
                            "source": "News search (fallback)",
                        })
            except Exception as e:
                safe_print(f"  ⚠️ Search failed: {e}")

    if not events:
        return {
            "status": "no_events",
            "message": "No significant geopolitical supply-chain disruptions detected in the last week.",
            "opportunities": [],
        }

    # --- Step 2: Map events to commodities ---
    safe_print("🔗 Mapping events to supply chain impacts...")

    all_affected = []  # List of {commodity, country, share, desc, event_context}

    for event in events:
        for country in event["countries"]:
            commodities = COUNTRY_COMMODITY_MAP.get(country, [])
            for com in commodities:
                all_affected.append({
                    "commodity": com["commodity"],
                    "country": country.replace("_", " ").title(),
                    "global_share": com["share"],
                    "supply_desc": com["desc"],
                    "event_source": event.get("source", ""),
                })

        # Add event-type triggered commodities (e.g., "military" → defense + oil)
        for etype in event["event_types"]:
            extra_commodities = EVENT_TYPE_MAP.get(etype, [])
            for com in extra_commodities:
                # Avoid duplicates
                if not any(a["commodity"] == com for a in all_affected):
                    all_affected.append({
                        "commodity": com,
                        "country": "Event-triggered",
                        "global_share": 0,
                        "supply_desc": f"Triggered by {etype} event",
                        "event_source": event.get("source", ""),
                    })

    if not all_affected:
        return {
            "status": "no_supply_chain_impact",
            "message": "Events detected but no clear supply-chain commodity impact identified.",
            "events": [e.get("description", "")[:200] for e in events],
            "opportunities": [],
        }

    # Deduplicate
    seen_commodities = set()
    unique_affected = []
    for a in all_affected:
        if a["commodity"] not in seen_commodities:
            unique_affected.append(a)
            seen_commodities.add(a["commodity"])

    # --- Step 3: Map commodities to tickers and score ---
    safe_print(f"📊 Found {len(unique_affected)} affected commodities. Checking price action...")

    opportunities = []

    for affected in unique_affected:
        if is_cancelled():
            break

        commodity = affected["commodity"]
        tickers = _get_tickers_for_commodity(commodity)

        if not tickers:
            continue

        # Check top 5 tickers for each commodity (avoid rate limits)
        for ticker_sym in tickers[:5]:
            if is_cancelled():
                break

            # Skip non-US tickers that might fail
            if "." in ticker_sym and not ticker_sym.endswith(".TO"):
                continue

            price_data = _quick_price_check(ticker_sym)
            if "error" in price_data:
                continue

            # Score the opportunity
            score = 0
            reasons = []

            # 1. Supply chain significance
            share = affected["global_share"]
            if share >= 40:
                score += 25
                reasons.append(f"🌍 Critical Supply ({affected['supply_desc']})")
            elif share >= 20:
                score += 20
                reasons.append(f"🌍 Major Supply ({affected['supply_desc']})")
            elif share >= 10:
                score += 15
                reasons.append(f"🌍 Significant Supply ({affected['supply_desc']})")
            else:
                score += 10
                reasons.append(f"🌍 Supply Exposure ({affected['supply_desc']})")

            # 2. Price momentum (already moving?)
            weekly = price_data.get("weekly_change_pct", 0)
            monthly = price_data.get("monthly_change_pct", 0)

            if weekly > 5:
                score += 20
                reasons.append(f"🚀 Already Moving (+{weekly:.1f}% this week)")
            elif weekly > 2:
                score += 10
                reasons.append(f"📈 Positive Momentum (+{weekly:.1f}% this week)")
            elif weekly < -5:
                score += 5
                reasons.append(f"📉 Dip Opportunity ({weekly:.1f}% this week)")

            # 2.5 Monthly trend (catches moves like oil $70→$100)
            if monthly > 15:
                score += 15
                reasons.append(f"📈 Strong Monthly Surge (+{monthly:.1f}% this month — trend accelerating)")
            elif monthly > 8:
                score += 10
                reasons.append(f"📈 Monthly Uptrend (+{monthly:.1f}%)")

            # 3. Volume spike (confirming institutional interest)
            vol_spike = price_data.get("volume_spike", 1.0)
            if vol_spike > 2.0:
                score += 15
                reasons.append(f"🔥 Volume Spike ({vol_spike:.1f}x normal)")
            elif vol_spike > 1.5:
                score += 10
                reasons.append(f"📊 Elevated Volume ({vol_spike:.1f}x normal)")

            # 4. CONFLICT PREMIUM ANALYSIS (the "is it priced in?" check)
            conflict_analysis = _analyze_conflict_premium(commodity, price_data.get("price", 0))
            conflict_status = conflict_analysis.get("status", "unknown")

            if conflict_status == "NOT_PRICED_IN":
                score += 25  # Major signal — still well below conflict peaks
                reasons.append(f"⚠️ NOT Priced In — {conflict_analysis.get('upside_to_peak_pct', 0):.0f}% upside to conflict peak")
            elif conflict_status == "LIKELY_NOT_PRICED_IN":
                score += 15
                reasons.append(f"📊 Historically +{conflict_analysis.get('avg_conflict_premium_pct', 30)}% upside in conflicts")
            elif conflict_status == "PARTIALLY_PRICED":
                score += 10
                reasons.append(f"🟡 Partially Priced — still {conflict_analysis.get('upside_to_peak_pct', 0):.0f}% to peak")
            elif conflict_status == "NEAR_PEAK":
                score -= 10
                reasons.append("🟡 Near Conflict Peak — limited upside")
            elif conflict_status == "ABOVE_PEAK":
                score -= 15
                reasons.append("🔴 Above Conflict Peak — momentum play only")

            # 5. Commodity description
            com_desc = _get_commodity_description(commodity)
            reasons.append(f"💎 Commodity: {com_desc}")

            opportunities.append({
                "symbol": ticker_sym,
                "score": min(score, 100),
                "price": price_data.get("price", "N/A"),
                "weekly_change": f"{weekly:+.1f}%",
                "monthly_change": f"{monthly:+.1f}%",
                "commodity": commodity.replace("_", " ").title(),
                "country": affected["country"],
                "reasons": reasons,
                "causal_chain": f"{affected['country']} disruption → {commodity.replace('_', ' ').title()} supply risk → {ticker_sym} benefits",
                "conflict_premium": conflict_analysis.get("verdict", ""),
            })

        # --- DOWNSTREAM BEARISH EFFECTS ---
        if commodity in DOWNSTREAM_EFFECTS_MAP:
            downstream = DOWNSTREAM_EFFECTS_MAP[commodity]
            for bear_tick in downstream["bearish_tickers"]:
                if is_cancelled(): break
                p_data = _quick_price_check(bear_tick)
                if "error" in p_data: continue

                opportunities.append({
                    "symbol": bear_tick,
                    "score": 85,  # High base score for macro bearish flags
                    "price": p_data.get("price", "N/A"),
                    "weekly_change": f"{p_data.get('weekly_change_pct', 0):+.1f}%",
                    "monthly_change": f"{p_data.get('monthly_change_pct', 0):+.1f}%",
                    "commodity": commodity.replace("_", " ").title(),
                    "country": affected["country"],
                    "reasons": [
                         f"📉 DOWNSTREAM VICTIM: {', '.join(downstream['vulnerable_sectors'])}",
                         f"⚠️ THESIS: {downstream['thesis']}"
                    ],
                    "causal_chain": f"{affected['country']} disruption → {commodity.replace('_', ' ').title()} cost spike → Margin compression for {bear_tick}",
                    "conflict_premium": "Vulnerable to short-term shock (Bearish)",
                })


    # --- Step 4: Rank and return ---
    opportunities.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate by ticker (keep highest score)
    seen_tickers = set()
    final_opps = []
    for opp in opportunities:
        if opp["symbol"] not in seen_tickers:
            final_opps.append(opp)
            seen_tickers.add(opp["symbol"])

    # Top 15
    top_picks = final_opps[:15]

    # Event summary
    event_summaries = []
    for e in events[:3]:
        countries_str = ", ".join(e["countries"][:3])
        event_summaries.append({
            "countries": countries_str,
            "event_types": e["event_types"][:3],
            "source": e.get("source", ""),
        })

    commodities_affected = list(set(a["commodity"].replace("_", " ").title() for a in unique_affected))

    return {
        "status": "opportunities_found",
        "events_detected": event_summaries,
        "commodities_affected": commodities_affected,
        "total_opportunities": len(final_opps),
        "top_picks": top_picks,
        "summary": (
            f"Detected {len(events)} geopolitical event(s) affecting "
            f"{len(commodities_affected)} commodities. "
            f"Found {len(final_opps)} investable opportunities. "
            f"Top commodities: {', '.join(commodities_affected[:5])}."
        ),
    }


@log_exceptions()
def get_supply_chain_exposure(country: str) -> dict[str, Any]:
    """
    Quick lookup: What commodities/tickers are exposed to a specific country?

    Args:
        country: Country name (e.g., "Qatar", "Taiwan", "Russia")

    Returns:
        Dict with commodities and associated tickers.
    """
    normalized = _normalize_country(country)
    commodities = COUNTRY_COMMODITY_MAP.get(normalized, [])

    if not commodities:
        return {
            "country": country,
            "error": f"No supply chain data for '{country}'. Known regions: {', '.join(list(COUNTRY_COMMODITY_MAP.keys())[:10])}..."
        }

    result = {
        "country": country,
        "normalized_key": normalized,
        "commodities": [],
    }

    for com in commodities:
        tickers = _get_tickers_for_commodity(com["commodity"])
        result["commodities"].append({
            "name": com["commodity"].replace("_", " ").title(),
            "global_share": f"{com['share']}%",
            "description": com["desc"],
            "investable_tickers": tickers[:8],
        })

    return result


@cached(key_func=lambda symbol: f"ticker_geo_ctx:{symbol.upper()}")
@log_exceptions()
def get_ticker_geopolitical_context(symbol: str) -> dict[str, Any]:
    """
    REVERSE LOOKUP: Given a stock ticker, find its geopolitical supply-chain exposure.

    Maps: Ticker → Commodity → Country → Geopolitical Events & Conflict Premium

    This is the critical function that enables geopolitical context during
    individual stock analysis (e.g., "analyze NTR" automatically surfaces
    fertilizer/potash → Russia/Canada supply chain risks).

    Args:
        symbol: Stock ticker (e.g., "NTR", "LNG", "FCX")

    Returns:
        Dict with commodity exposure, country risks, conflict premium, and recent events.
        Returns {"exposed": False} if the ticker has no geopolitical relevance.
    """
    symbol = symbol.upper().strip()
    safe_print(f"🌍 Geopolitical Context Check: {symbol}")

    # --- Step 1: Reverse lookup — find which commodities this ticker appears in ---
    matched_commodities = []

    for commodity, entry in COMMODITY_TICKER_MAP.items():
        if commodity == "desc":
            continue
        for category_key, tickers_or_desc in entry.items():
            if category_key == "desc":
                continue
            if isinstance(tickers_or_desc, list):
                # Normalize ticker names for matching
                for t in tickers_or_desc:
                    clean_t = t.strip().upper()
                    # Handle special formats like "II-VI (COHR)" or "platinum → PPLT"
                    if "(" in clean_t:
                        clean_t = clean_t.split("(")[1].replace(")", "").strip()
                    if "→" in clean_t:
                        clean_t = clean_t.split("→")[-1].strip()

                    if clean_t == symbol:
                        if commodity not in [m["commodity"] for m in matched_commodities]:
                            matched_commodities.append({
                                "commodity": commodity,
                                "category": category_key,
                                "desc": entry.get("desc", commodity.replace("_", " ").title()),
                            })

    if not matched_commodities:
        return {"exposed": False, "symbol": symbol, "message": f"{symbol} has no known geopolitical supply-chain exposure."}

    # --- Step 2: Find which countries produce these commodities ---
    country_exposures = []

    for match in matched_commodities:
        commodity = match["commodity"]
        for country, commodities_list in COUNTRY_COMMODITY_MAP.items():
            for com_entry in commodities_list:
                if com_entry["commodity"] == commodity:
                    country_exposures.append({
                        "country": country.replace("_", " ").title(),
                        "commodity": commodity.replace("_", " ").title(),
                        "global_share": com_entry["share"],
                        "supply_desc": com_entry["desc"],
                    })

    # --- Step 3: Conflict premium analysis for matched commodities ---
    conflict_premiums = []
    for match in matched_commodities:
        commodity = match["commodity"]
        peak_data = CONFLICT_PEAK_PRICES.get(commodity)
        if peak_data:
            # Get current proxy price
            proxy_ticker = peak_data.get("ticker")
            if proxy_ticker:
                try:
                    proxy_data = _quick_price_check(proxy_ticker)
                    if "error" not in proxy_data:
                        premium = _analyze_conflict_premium(commodity, proxy_data["price"])
                        conflict_premiums.append({
                            "commodity": commodity.replace("_", " ").title(),
                            "verdict": premium.get("verdict", "Unknown"),
                            "status": premium.get("status", "unknown"),
                            "upside_to_peak_pct": premium.get("upside_to_peak_pct", 0),
                            "historical_premium_pct": premium.get("avg_conflict_premium_pct", 0),
                        })
                except Exception:
                    conflict_premiums.append({
                        "commodity": commodity.replace("_", " ").title(),
                        "verdict": f"Historically +{peak_data.get('historical_avg_conflict_premium_pct', 30)}% in conflicts",
                        "status": "LIKELY_NOT_PRICED_IN",
                        "historical_premium_pct": peak_data.get("historical_avg_conflict_premium_pct", 30),
                    })

    # --- Step 4: Quick news check for geopolitical events affecting these commodities ---
    recent_events = []
    commodity_names = [m["commodity"].replace("_", " ") for m in matched_commodities]
    search_query = f"{symbol} {' '.join(commodity_names[:2])} geopolitical supply chain sanctions tariff"

    try:
        news_results = search_news(search_query, max_results=3, timelimit="w")
        if news_results and news_results != "No results found.":
            # Detect countries and event types from the news
            detected_countries = _detect_countries_in_text(news_results)
            detected_events = _detect_event_types(news_results)
            if detected_countries or detected_events:
                recent_events.append({
                    "summary": news_results[:500],
                    "countries_mentioned": [c.replace("_", " ").title() for c in detected_countries[:5]],
                    "event_types": detected_events[:5],
                })
    except Exception as e:
        safe_print(f"  ⚠️ News search failed: {e}")

    # --- Step 5: Build response ---
    return {
        "exposed": True,
        "symbol": symbol,
        "commodity_exposure": [
            {
                "commodity": m["commodity"].replace("_", " ").title(),
                "role": m["category"].replace("_", " ").title(),
                "description": m["desc"],
            }
            for m in matched_commodities
        ],
        "country_supply_chain": country_exposures,
        "conflict_premium_analysis": conflict_premiums,
        "recent_geopolitical_events": recent_events,
        "summary": (
            f"⚠️ {symbol} is exposed to geopolitical risk via "
            f"{', '.join(m['commodity'].replace('_', ' ').title() for m in matched_commodities)}. "
            f"Key supply countries: {', '.join(set(c['country'] for c in country_exposures[:5]))}. "
            + (f"Conflict premium: {conflict_premiums[0]['verdict']}" if conflict_premiums else "")
        ),
    }


@log_exceptions()
def quick_geopolitical_check() -> dict[str, Any]:
    """
    Fast check for the morning briefing: Are there any major geopolitical disruptions right now?
    Returns a brief alert or 'all clear'.
    PRIORITY: GDELT crisis alerts → web search (fallback)
    """
    results_text = ""
    source_used = "Unknown"

    # 1. Try GDELT crisis alerts first
    try:
        from tools.gdelt_api import get_gdelt_crisis_alerts
        alerts = get_gdelt_crisis_alerts(max_results=5)
        if alerts:
            results_text = " ".join(a.get("title", "") for a in alerts)
            source_used = "GDELT"
    except Exception as e:
        safe_print(f"⚠️ GDELT quick check failed: {e}")

    # 2. Fallback to web search
    if not results_text:
        try:
            results_text = search_news(
                "geopolitical crisis supply chain disruption commodity prices",
                max_results=3,
                timelimit="d"  # Last 24 hours only
            )
            source_used = "Web Search"
        except Exception:
            pass

    if not results_text or results_text == "No results found.":
        return {"alert": False, "message": "No major geopolitical supply disruptions detected today."}

    countries = _detect_countries_in_text(results_text)
    event_types = _detect_event_types(results_text)

    if not countries and not event_types:
        return {"alert": False, "message": "No clear supply-chain impact from today's events."}

    # Quick commodity mapping
    affected_commodities = set()
    for country in countries:
        for com in COUNTRY_COMMODITY_MAP.get(country, []):
            if com["share"] >= 15:  # Only significant exposures
                affected_commodities.add(com["commodity"].replace("_", " ").title())

    if not affected_commodities:
        return {"alert": False, "message": "Geopolitical events detected but no major commodity supply risk."}

    return {
        "alert": True,
        "countries": countries,
        "event_types": event_types,
        "commodities_at_risk": list(affected_commodities),
        "source": source_used,
        "message": (
            f"⚠️ Geopolitical Alert ({source_used}): Events in {', '.join(c.replace('_', ' ').title() for c in countries[:3])} "
            f"may disrupt {', '.join(list(affected_commodities)[:4])} supply. "
            f"Use 'scan geopolitical opportunities' for full analysis."
        ),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import json

    print("=== TEST 1: Iran-Qatar Scenario ===")
    result = scan_geopolitical_opportunities("Iran strike on Qatar")
    print(json.dumps(result, indent=2, default=str))

    print("\n=== TEST 2: Supply Chain Lookup ===")
    result2 = get_supply_chain_exposure("Qatar")
    print(json.dumps(result2, indent=2))

    print("\n=== TEST 3: Quick Morning Check ===")
    result3 = quick_geopolitical_check()
    print(json.dumps(result3, indent=2))
