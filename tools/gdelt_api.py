"""
GDELT Project API Connector
----------------------------
Queries the GDELT 2.0 DOC API for real-time global event monitoring.
No API key required — completely free and open.

Endpoints used:
  - /api/v2/doc/doc?mode=artlist  → Article list with metadata
  - /api/v2/doc/doc?mode=tonechart → Aggregate tone over time

Fallback: If GDELT is slow or down, callers should fall back to
web_search (search_news) which is the existing behaviour.
"""

import time
from datetime import datetime
from typing import Any

import requests

from agent.utils import safe_print
from tools.cache import cached
from tools.exception_logger import log_exceptions

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT Theme codes that map to our geopolitical scanner's EVENT_TYPE_MAP
GDELT_THEME_QUERIES = {
    "military_conflict": '(military OR "armed conflict" OR airstrike OR missile OR "troops deployed")',
    "sanctions_embargo": '(sanctions OR embargo OR "trade ban" OR "export controls" OR "asset freeze")',
    "supply_chain":      '("supply chain" OR "supply disruption" OR shortage OR "port closure" OR blockade)',
    "energy_crisis":     '(oil OR "natural gas" OR LNG OR OPEC OR "energy crisis" OR pipeline)',
    "commodity_shock":   '(commodity OR wheat OR copper OR lithium OR "rare earth" OR uranium)',
    "trade_war":         '(tariff OR "trade war" OR "import duty" OR "retaliatory tariff")',
    "geopolitical":      '(geopolitical OR coup OR revolution OR "regime change" OR invasion)',
}


@log_exceptions()
def _gdelt_get(query: str, max_records: int = 10, timespan: str = "3d",
               timeout: int = 12) -> list[dict[str, Any]] | None:
    """
    Core GDELT request helper. Returns a list of article dicts or None on failure.

    GDELT can be slow (~5-10s) and occasionally returns empty/malformed JSON.
    We handle this gracefully and let callers fall back.

    Args:
        query: GDELT query string (supports boolean operators)
        max_records: Number of articles to return (max 250)
        timespan: How far back to look (e.g., '1d', '3d', '7d')
        timeout: HTTP timeout in seconds
    """
    params = {
        "query": f"{query} sourcelang:eng",
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(min(max_records, 250)),
        "timespan": timespan,
        "sort": "datedesc",
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=timeout)

        if resp.status_code != 200:
            safe_print(f"⚠️ GDELT returned HTTP {resp.status_code}")
            return None

        # GDELT sometimes returns empty body or HTML error pages
        if not resp.text or resp.text.strip().startswith("<!"):
            safe_print("⚠️ GDELT returned empty or HTML response")
            return None

        data = resp.json()
        articles = data.get("articles", [])

        if not articles:
            return None

        return articles

    except requests.exceptions.Timeout:
        safe_print("⚠️ GDELT request timed out (>12s)")
        return None
    except requests.exceptions.JSONDecodeError:
        safe_print("⚠️ GDELT returned malformed JSON (rate limit or server issue)")
        return None
    except Exception as e:
        safe_print(f"⚠️ GDELT request failed: {e}")
        return None


def _parse_gdelt_article(article: dict) -> dict[str, Any]:
    """Parse a raw GDELT article dict into our standardized format."""
    # GDELT date format: "20260413T221500Z"
    raw_date = article.get("seendate", "")
    try:
        dt = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ")
        pub_date = dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        pub_date = raw_date or "Unknown"

    return {
        "title": article.get("title", "Untitled"),
        "url": article.get("url", "#"),
        "source": article.get("domain", "Unknown"),
        "pub_date": pub_date,
        "language": article.get("language", "English"),
        "source_country": article.get("sourcecountry", "Unknown"),
    }


@cached(key_func=lambda query, max_results=10, timespan="3d": f"gdelt_search:{query[:40]}:{max_results}:{timespan}")
@log_exceptions()
def search_gdelt_events(query: str, max_results: int = 10,
                        timespan: str = "3d") -> list[dict[str, Any]]:
    """
    Search GDELT for geopolitical events matching a query.

    Args:
        query: Free-text or boolean query (e.g., "Iran sanctions oil")
        max_results: Max number of articles to return
        timespan: How far back ('1d', '3d', '7d')

    Returns:
        List of parsed article dicts, or empty list if GDELT fails.
    """
    articles = _gdelt_get(query, max_records=max_results, timespan=timespan)
    if not articles:
        return []

    return [_parse_gdelt_article(a) for a in articles]


@cached(key_func=lambda themes=None, timespan="3d": f"gdelt_geopolitical:{','.join(themes or ['all'])}:{timespan}")
@log_exceptions()
def scan_gdelt_geopolitical(themes: list[str] = None,
                            timespan: str = "3d",
                            max_per_theme: int = 5) -> dict[str, list[dict[str, Any]]]:
    """
    Scan GDELT across multiple geopolitical themes simultaneously.

    This is the primary entry point for the geopolitical scanner integration.
    It queries GDELT for each theme category and returns organized results.

    Args:
        themes: List of theme keys from GDELT_THEME_QUERIES.
                If None, scans all themes.
        timespan: How far back to look ('1d', '3d', '7d')
        max_per_theme: Max articles per theme category

    Returns:
        Dict mapping theme names to lists of parsed articles.
        Example: {"military_conflict": [...], "sanctions_embargo": [...]}
    """
    if themes is None:
        themes = list(GDELT_THEME_QUERIES.keys())

    results = {}

    for theme in themes:
        query = GDELT_THEME_QUERIES.get(theme)
        if not query:
            continue

        articles = _gdelt_get(query, max_records=max_per_theme, timespan=timespan)

        if articles:
            parsed = [_parse_gdelt_article(a) for a in articles]
            results[theme] = parsed

        # Be polite to GDELT — small delay between theme queries
        time.sleep(0.5)

    return results


@cached(key_func=lambda max_results=10: f"gdelt_quick_alerts:{max_results}")
@log_exceptions()
def get_gdelt_crisis_alerts(max_results: int = 10) -> list[dict[str, Any]]:
    """
    Quick crisis alert scan — combines military + sanctions + supply chain
    into a single query for the morning briefing.

    Returns a list of the most relevant crisis headlines.
    """
    crisis_query = (
        '("military strike" OR "armed conflict" OR sanctions OR embargo '
        'OR "supply chain disruption" OR blockade OR "trade war" OR tariff '
        'OR "missile attack" OR invasion OR "regime change")'
    )

    articles = _gdelt_get(crisis_query, max_records=max_results * 2, timespan="1d")

    if not articles:
        return []

    # Deduplicate by title similarity (GDELT often returns duplicates)
    seen_titles = set()
    unique_articles = []
    for a in articles:
        title = a.get("title", "").lower().strip()
        # Simple dedup: skip if first 40 chars match something we've seen
        title_key = title[:40]
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        unique_articles.append(_parse_gdelt_article(a))

        if len(unique_articles) >= max_results:
            break

    return unique_articles


if __name__ == "__main__":
    import json

    print("=== GDELT Crisis Alerts (24h) ===")
    alerts = get_gdelt_crisis_alerts(3)
    for a in alerts:
        print(f"  • {a['title'][:80]}")
        print(f"    Source: {a['source']} | {a['pub_date']}")

    print("\n=== GDELT Geopolitical Scan ===")
    scan = scan_gdelt_geopolitical(themes=["military_conflict", "energy_crisis"], max_per_theme=2)
    print(json.dumps(scan, indent=2, default=str))
