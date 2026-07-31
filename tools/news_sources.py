
import random

import yfinance as yf

from agent.utils import safe_print
from tools.cache import cached
from tools.exception_logger import log_exceptions


@log_exceptions()
def _score_headline(title: str) -> int:
    """Score a headline by market-wide relevance. Higher = more relevant to broad conditions."""
    title_lower = title.lower()
    score = 0

    # High-value: broad market / macro themes
    macro_keywords = [
        'market', 'markets', 'wall street', 'rally', 'sell-off', 'selloff',
        'recession', 'economy', 'economic', 'gdp', 'inflation', 'cpi',
        'fed ', 'federal reserve', 'interest rate', 'rate cut', 'rate hike',
        'treasury', 'bond', 'yield', 'jobs report', 'unemployment',
        'trade war', 'tariff', 'geopolitical', 'oil prices', 'crude',
        'bank of canada', 'boc ', 'tsx', 'dow', 's&p', 'nasdaq',
        'sector', 'rotation', 'risk-off', 'risk-on', 'bull', 'bear',
        'correction', 'crash', 'surge', 'plunge', 'tumble', 'soar',
        'futures', 'stocks fall', 'stocks rise', 'stock market',
        'earnings season', 'ipo', 'commodit', 'gold ', 'bitcoin',
        'canadian market', 'canadian dollar', 'energy sector', 'tech sector',
        'financial sector', 'mining', 'ai sector', 'chip', 'semiconductor',
    ]
    for kw in macro_keywords:
        if kw in title_lower:
            score += 3

    # Medium-value: sector / multi-company themes
    sector_keywords = [
        'tech stocks', 'energy stocks', 'energy sector', 'oil and gas',
        'bank stocks', 'financ', 'healthcare', 'consumer staple',
        'industrial', 'utilities', 'real estate', 'automotive',
        'magnificent seven', 'faang', 'big tech', 'pipeline',
    ]
    for kw in sector_keywords:
        if kw in title_lower:
            score += 2

    # Penalty: individual stock / consumer product focus
    stock_keywords = [
        'upgraded', 'downgraded', 'price target', 'maintained at',
        'outperform', 'underperform', 'buy rating', 'sell rating',
        'announces dividend', 'stock split', 'insider',
        'energy drink', 'beverage', 'snack', 'restaurant',
    ]
    for kw in stock_keywords:
        if kw in title_lower:
            score -= 2

    return score


@cached(key_func=lambda limit=10: f"market_news:{limit}")
@log_exceptions()
def get_market_news(limit: int = 10) -> str:
    """
    Get latest market news from Finnhub (Fallback: Yahoo Finance).
    Prioritizes broad market conditions over individual stock stories.
    """
    all_news = []

    try:
        from tools.finnhub_api import get_finnhub_market_news
        finnhub_news = get_finnhub_market_news(limit * 3) # Grab more to score
        if finnhub_news:
            for n in finnhub_news:
                title = n.get('headline')
                if not title: continue
                provider = n.get('source', 'Finnhub')
                timestamp = n.get('datetime', 0)
                from datetime import datetime as dt
                try:
                    pub_date = dt.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pub_date = "Unknown Date"
                link = n.get('url', '#')

                relevance = _score_headline(title)
                entry = f"**{title}**\n*Source: {provider} | {pub_date}*\n[Read More]({link})"
                all_news.append({'timestamp': pub_date, 'text': entry, 'score': relevance})

            if all_news:
                all_news.sort(key=lambda x: x['score'], reverse=True)
                return "\n\n".join([x['text'] for x in all_news[:limit]])
    except Exception as e:
        safe_print(f"⚠️ Finnhub market news failed: {e}")

    # --- FALLBACK: Yahoo Finance ---
    # Major indices — US and Canadian
    indices = ["^GSPC", "^DJI", "^IXIC", "^GSPTSE", "CADUSD=X"]
    # US movers
    us_movers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "META"]
    # Canadian movers (Banking, Energy, Tech)
    ca_movers = ["RY.TO", "TD.TO", "SU.TO", "ENB.TO", "SHOP.TO", "CNQ.TO", "BMO.TO"]

    all_news = []
    seen_titles = set()

    # Always pick from BOTH pools to guarantee balanced coverage
    targets = indices + random.sample(us_movers, 3) + random.sample(ca_movers, 2)

    for sym in targets:
        try:
            val = yf.Ticker(sym)
            news = val.news
            if not news: continue

            for n in news:
                if not isinstance(n, dict): continue
                content = n.get('content', n)
                if not isinstance(content, dict): continue

                title = content.get('title') or content.get('headline', 'N/A')

                if title in seen_titles: continue
                seen_titles.add(title)

                pub_date = content.get('pubDate', 'N/A')
                link = content.get('clickThroughUrl', {}).get('url', n.get('link', '#'))
                provider = content.get('provider', {}).get('displayName', 'Yahoo')

                relevance = _score_headline(title)
                entry = f"**{title}**\n*Source: {provider} | {pub_date}*\n[Read More]({link})"

                all_news.append({'timestamp': pub_date, 'text': entry, 'score': relevance})
        except Exception:
            continue

    if not all_news:
        try:
            from tools.web_search import search_news
            safe_print("⚠️ Yahoo Finance news stream returned empty. Falling back to web search for market headlines...")
            return search_news("major financial market news daily headlines S&P 500 TSX Fed macro data", max_results=limit)
        except Exception as e:
            return f"No market news available at this time. (YF Empty, Search Error: {e})"

    # Sort by relevance score (highest first), then return top results
    all_news.sort(key=lambda x: x['score'], reverse=True)
    return "\n\n".join([x['text'] for x in all_news[:limit]])


@cached(key_func=lambda: "global_market_news")
@log_exceptions()
def get_global_market_news() -> str:
    """International / ex-US macro-market news for the report's global section.

    ``get_market_news`` above is US-centric (Finnhub US wire + US/CA Yahoo names), so
    a report built on it alone reads as a US-only view. This complements it with the
    rest of the world — the Fed and ECB, China and broader Asia, Europe, oil &
    commodities, and the cross-border geopolitics that move world markets — so a
    holder of US + Canadian + international exposure sees the full macro backdrop."""
    from tools.web_search import search_news
    query = (
        "global stock markets today Europe Asia China Federal Reserve ECB "
        "interest rates oil commodities geopolitics world economy"
    )
    try:
        return search_news(query, max_results=8)
    except Exception as e:
        return f"Error fetching global market news: {e}"


@cached(key_func=lambda tickers, limit=5: f"company_news:{tickers.upper()}:{limit}")
@log_exceptions()
def get_company_news(tickers: str, limit: int = 5) -> str:
    """Get specific news for a list of tickers (comma-separated)."""
    syms = [s.strip().upper() for s in tickers.split(',')]
    all_news = []
    seen_titles = set()

    # 1. Try Finnhub First
    finnhub_success = False
    try:
        from tools.finnhub_api import get_finnhub_company_news
        for sym in syms:
            fh_news = get_finnhub_company_news(sym, limit)
            if fh_news:
                finnhub_success = True
                for n in fh_news:
                    title = n.get('headline')
                    if not title or title in seen_titles: continue
                    seen_titles.add(title)

                    provider = n.get('source', 'Finnhub')
                    timestamp = n.get('datetime', 0)
                    from datetime import datetime as dt
                    try:
                        pub_date = dt.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        pub_date = "Unknown Date"
                    link = n.get('url', '#')

                    entry = f"**[{sym}] {title}**\n*Source: {provider} | {pub_date}*\n[Read More]({link})"
                    all_news.append(entry)

        if finnhub_success and all_news:
            return "\n\n".join(all_news)
    except Exception as e:
        safe_print(f"⚠️ Finnhub company news failed for {tickers}: {e}")

    # 2. Fallback to Yahoo Finance
    all_news = [] # Reset if partial failure
    seen_titles = set()

    for sym in syms:
        try:
            val = yf.Ticker(sym)
            news = val.news
            if not news: continue
            count = 0
            for n in news:
                if not isinstance(n, dict): continue
                content = n.get('content', n)
                if not isinstance(content, dict): continue

                title = content.get('title') or content.get('headline', 'N/A')

                if title in seen_titles: continue
                seen_titles.add(title)

                pub_date = content.get('pubDate', 'N/A')

                # Link formatting safety
                click_url = content.get('clickThroughUrl')
                if isinstance(click_url, dict):
                    link = click_url.get('url', n.get('link', '#'))
                else:
                    link = n.get('link', '#')

                prov = content.get('provider')
                provider = prov.get('displayName', 'Yahoo') if isinstance(prov, dict) else 'Yahoo'

                entry = f"**[{sym}] {title}**\n*Source: {provider} | {pub_date}*\n[Read More]({link})"
                all_news.append(entry)
                count += 1
                if count >= limit: break
        except Exception as e:
            all_news.append(f"Error fetching {sym}: {e}")

    return "\n\n".join(all_news)

if __name__ == "__main__":
    print("--- MARKET NEWS ---")
    print(get_market_news(5))
    print("\n--- COMPANY NEWS (NVDA) ---")
    print(get_company_news("NVDA", 3))
