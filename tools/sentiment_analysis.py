"""
Sentiment Analysis Tool
Analyzes market sentiment from multiple sources:
- Fear & Greed Index (CNN)
- News headlines sentiment (Alpha Vantage)
- Analyst ratings (yfinance)
"""
from datetime import datetime
from typing import Any

import requests
import yfinance as yf

from tools.cache import cached
from tools.credential_manager import get_api_key
from tools.exception_logger import log_exceptions

# from langchain_community.tools.tavily_search import TavilySearchResults


@log_exceptions()
def get_reddit_sentiment(symbol: str) -> dict[str, Any]:
    """
    Scans Reddit/Social media for "Hype" and retail sentiment using Tavily Search.
    Focuses on "YOLO", "Squeeze", and "Moon" terminology.
    """
    try:
        # 1. Broad Search (Use hardened search_news with Tavily/Google fallback)
        content = ""
        try:
            from tools.web_search import search_news
            query = f"{symbol} stock investor sentiment discussion analysis reddit social media"
            # We use max_results=5 for a good mix of recent sentiment
            content = search_news(query, max_results=5).lower()
        except Exception as e:
            # Fallback if search fails
            from agent.logger import log_event
            log_event("SENTIMENT_ANALYSIS", {"error": f"Social search tool failed: {e}", "symbol": symbol})
            content = f"caution: live social chatter for {symbol} currently unavailable due to search connectivity issues."

        # 2. Text Analysis
        hype_keywords = ["yolo", "moon", "squeeze", "buy", "bull", "rocket", "breakout", "upgrade", "undervalued", "gem"]
        fear_keywords = ["crash", "sell", "bear", "scam", "puts", "drop", "tank", "overvalued", "risk", "warning"]

        hype_count = sum(1 for k in hype_keywords if k in content)
        fear_count = sum(1 for k in fear_keywords if k in content)

        text_score = 50 + ((hype_count - fear_count) * 5)
        text_score = min(100, max(0, text_score))

        # 3. Volume Proxy (Confirmation)
        # Use yfinance volume as a "Hype Check"
        try:
             ticker = yf.Ticker(symbol)
             hist = ticker.history(period="5d")
             if not hist.empty:
                 avg_vol = hist['Volume'].mean()
                 last_vol = hist['Volume'].iloc[-1]
                 if last_vol > avg_vol * 1.5:
                     text_score += 15 # Boost score if volume is spiking
        except Exception:
             pass

        final_score = min(100, text_score)

        if final_score > 70:
            verdict = "🔥 HIGH HYPE (Bullish Chatter)"
        elif final_score < 40:
            verdict = "📉 Negative Sentiment"
        else:
            verdict = "⚪ Neutral/Mixed"

        return {
            "symbol": symbol,
            "reddit_hype_score": final_score,
            "verdict": verdict,
            "recent_discussions": [{
                "snippet": content[:300] + "...",
                "source": "Web & Social Aggregators",
                "sentiment_lean": "Bullish" if final_score > 50 else "Bearish"
            }]
        }

    except Exception as e:
        return {"symbol": symbol, "error": f"Social search failed: {e}"}



@cached(key_func=lambda: "fear_greed")
@log_exceptions()
def get_fear_greed_index() -> dict[str, Any]:
    """
    Get the CNN Fear & Greed Index - measures overall market sentiment.
    0 = Extreme Fear (buy opportunity)
    100 = Extreme Greed (sell signal)
    """
    try:
        # CNN's Fear & Greed API endpoint
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract current value
            current = data.get("fear_and_greed", {})
            score = current.get("score", 50)
            rating = current.get("rating", "Neutral")

            # Determine investment implication
            if score <= 25:
                implication = "🟢 EXTREME FEAR - Historically a strong buying opportunity"
                action = "Consider buying - others are fearful"
            elif score <= 40:
                implication = "🟢 FEAR - Market pessimistic, potential opportunity"
                action = "Lean bullish - sentiment overly negative"
            elif score <= 60:
                implication = "⚪ NEUTRAL - Market balanced"
                action = "No strong sentiment signal"
            elif score <= 75:
                implication = "🟡 GREED - Market optimistic, use caution"
                action = "Be cautious - don't chase"
            else:
                implication = "🔴 EXTREME GREED - Market euphoric, high risk"
                action = "Consider taking profits - others are greedy"

            return {
                "indicator": "Fear & Greed Index",
                "score": round(score),
                "rating": rating,
                "implication": implication,
                "suggested_action": action,
                "interpretation": (
                    "Warren Buffett: 'Be fearful when others are greedy, "
                    "and greedy when others are fearful.'"
                )
            }
    except Exception:
        pass

    # Fallback with reasonable default
    return {
        "indicator": "Fear & Greed Index",
        "score": 55,
        "rating": "Neutral",
        "note": "Live data unavailable - using default",
        "implication": "⚪ NEUTRAL - Check CNN.com for live reading"
    }


@cached(key_func=lambda symbol: f"news_sentiment:{symbol.upper()}")
@log_exceptions()
def get_news_sentiment(symbol: str) -> dict[str, Any]:
    """
    Get sentiment from recent news headlines for a specific stock.
    Uses Alpha Vantage News Sentiment API.
    """
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol.upper(),
            "apikey": get_api_key("ALPHA_VANTAGE_API_KEY", "demo"),
            "limit": 20
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if "feed" not in data:
            # Fallback to yfinance news
            return _get_yfinance_news_sentiment(symbol)

        articles = data["feed"]
        if not articles:
            return {"error": f"No news found for {symbol}"}

        # Analyze sentiment scores
        sentiments = []
        headlines = []

        for article in articles[:10]:
            # Get ticker-specific sentiment
            for ticker_sentiment in article.get("ticker_sentiment", []):
                if ticker_sentiment.get("ticker") == symbol.upper():
                    score = float(ticker_sentiment.get("ticker_sentiment_score", 0))
                    sentiments.append(score)
                    headlines.append({
                        "title": article.get("title", "")[:80],
                        "sentiment": "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral",
                        "score": round(score, 2)
                    })
                    break

        if not sentiments:
            return _get_yfinance_news_sentiment(symbol)

        avg_sentiment = sum(sentiments) / len(sentiments)

        # Categorize
        if avg_sentiment > 0.25:
            overall = "Very Positive 🟢"
            signal = "Strong bullish news flow"
        elif avg_sentiment > 0.1:
            overall = "Positive 🟢"
            signal = "Mildly bullish news"
        elif avg_sentiment < -0.25:
            overall = "Very Negative 🔴"
            signal = "Strong bearish news flow"
        elif avg_sentiment < -0.1:
            overall = "Negative 🔴"
            signal = "Mildly bearish news"
        else:
            overall = "Neutral ⚪"
            signal = "Mixed or neutral news"

        return {
            "symbol": symbol,
            "news_sentiment": overall,
            "sentiment_score": round(avg_sentiment, 2),
            "signal": signal,
            "articles_analyzed": len(sentiments),
            "recent_headlines": headlines[:5],
            "interpretation": (
                "Sentiment score: -1.0 (very negative) to +1.0 (very positive). "
                "News sentiment often leads price by 1-3 days."
            )
        }
    except Exception:
        return _get_yfinance_news_sentiment(symbol)


@log_exceptions()
def _get_yfinance_news_sentiment(symbol: str) -> dict[str, Any]:
    """Fallback: Get basic news from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news

        if not news:
            return {"symbol": symbol, "note": "No recent news available"}

        headlines = []
        for item in news[:5]:
            title = item.get("title", "")
            # Basic keyword sentiment
            positive_words = ["surge", "jump", "gain", "rise", "beat", "upgrade", "buy", "bullish", "record"]
            negative_words = ["fall", "drop", "decline", "miss", "downgrade", "sell", "bearish", "crash", "cut"]

            sentiment = "Neutral"
            if any(word in title.lower() for word in positive_words):
                sentiment = "Positive"
            elif any(word in title.lower() for word in negative_words):
                sentiment = "Negative"

            headlines.append({"title": title[:80], "sentiment": sentiment})

        return {
            "symbol": symbol,
            "news_sentiment": "See headlines",
            "recent_headlines": headlines,
            "note": "Basic sentiment from headlines (Alpha Vantage unavailable)"
        }
    except Exception:
        return {"symbol": symbol, "error": "Could not fetch news"}


@cached(key_func=lambda symbol: f"analyst_consensus:{symbol.upper()}")
@log_exceptions()
def get_analyst_consensus(symbol: str) -> dict[str, Any]:
    """
    Get analyst ratings and price targets from yfinance.
    Shows Buy/Hold/Sell distribution and target price.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get recommendations
        recommendations = ticker.recommendations
        if recommendations is None or recommendations.empty:
            # Fallback to info
            info = ticker.info
            recon = info.get("recommendationKey", "N/A").replace("_", " ").title()
            target = info.get("targetMeanPrice")
            if recon != "N/A" or target:
                return {
                    "symbol": symbol,
                    "consensus": recon if recon != "N/A" else "Mixed ⚪",
                    "target_price": target,
                    "note": "Derived from summary data (detailed history unavailable)"
                }
            return {"symbol": symbol, "note": "No analyst data available"}

        # Get recent recommendations (last 3 months)
        recent = recommendations.tail(20)

        # Count ratings - handle BOTH yfinance formats
        rating_counts = {"Buy": 0, "Hold": 0, "Sell": 0}

        # NEW FORMAT: Summary table with strongBuy, buy, hold, sell, strongSell columns
        if "strongBuy" in recent.columns or "buy" in recent.columns:
            # Use the most recent period (row 0 = current month)
            row = recent.iloc[0]
            rating_counts["Buy"] = int(row.get("strongBuy", 0) or 0) + int(row.get("buy", 0) or 0)
            rating_counts["Hold"] = int(row.get("hold", 0) or 0)
            rating_counts["Sell"] = int(row.get("sell", 0) or 0) + int(row.get("strongSell", 0) or 0)
        # LEGACY FORMAT: Individual analyst actions with "To Grade" column
        elif "To Grade" in recent.columns:
            for _, row in recent.iterrows():
                grade = str(row.get("To Grade", "")).lower()
                if any(x in grade for x in ["buy", "outperform", "overweight", "positive"]):
                    rating_counts["Buy"] += 1
                elif any(x in grade for x in ["sell", "underperform", "underweight", "negative"]):
                    rating_counts["Sell"] += 1
                else:
                    rating_counts["Hold"] += 1
        else:
            # Unknown format - try to read whatever columns exist
            for _, row in recent.iterrows():
                grade = str(row.to_dict()).lower()
                if any(x in grade for x in ["buy", "outperform", "overweight"]):
                    rating_counts["Buy"] += 1
                elif any(x in grade for x in ["sell", "underperform", "underweight"]):
                    rating_counts["Sell"] += 1
                else:
                    rating_counts["Hold"] += 1

        total = sum(rating_counts.values())
        if total == 0:
            return {"symbol": symbol, "note": "No recent analyst ratings"}

        # Calculate percentages
        buy_pct = rating_counts["Buy"] / total * 100

        # Determine consensus
        if buy_pct >= 70:
            consensus = "Strong Buy 🟢"
            signal = "Analysts are very bullish"
        elif buy_pct >= 50:
            consensus = "Buy 🟢"
            signal = "Majority of analysts are bullish"
        elif rating_counts["Sell"] > rating_counts["Buy"]:
            consensus = "Sell 🔴"
            signal = "More analysts are bearish"
        else:
            consensus = "Hold ⚪"
            signal = "Mixed analyst sentiment"

        # Get price target
        info = ticker.info
        target_price = info.get("targetMeanPrice")
        current_price = info.get("currentPrice") or info.get("previousClose")

        upside = None
        if target_price and current_price:
            upside = ((target_price - current_price) / current_price) * 100

        return {
            "symbol": symbol,
            "analyst_consensus": consensus,
            "signal": signal,
            "ratings": {
                "Buy": rating_counts["Buy"],
                "Hold": rating_counts["Hold"],
                "Sell": rating_counts["Sell"]
            },
            "buy_percentage": round(buy_pct, 1),
            "price_target": round(target_price, 2) if target_price else "N/A",
            "current_price": round(current_price, 2) if current_price else "N/A",
            "upside_potential": f"{upside:+.1f}%" if upside else "N/A",
            "interpretation": (
                "⚠️ Analyst consensus is a heavily lagging indicator and subject to career risk. "
                "Do NOT use it as a primary pillar for a thesis. Look for divergence between analyst sentiment and actual fundamentals."
            )
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"Could not fetch analyst data: {str(e)}"}


@log_exceptions()
def get_social_buzz(symbol: str) -> dict[str, Any]:
    """
    Get social media mentions/buzz indicator.
    Note: For full social data, you'd need Twitter/Reddit APIs.
    This provides a basic estimate based on volume spikes.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="30d")

        if hist.empty:
            return {"symbol": symbol, "note": "No data available"}

        # Use volume as a proxy for social interest (cast to native float:
        # Series.mean() yields np.float64 which leaks as "np.float64(...)").
        avg_volume = float(hist['Volume'].mean())
        recent_volume = float(hist['Volume'].tail(5).mean())
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        if volume_ratio > 2:
            buzz = "Very High 🔥"
            interpretation = "Unusually high interest - potential viral/social activity"
        elif volume_ratio > 1.5:
            buzz = "High 📈"
            interpretation = "Above average interest"
        elif volume_ratio < 0.5:
            buzz = "Low 📉"
            interpretation = "Below average interest"
        else:
            buzz = "Normal"
            interpretation = "Average trading activity"

        return {
            "symbol": symbol,
            "social_buzz_estimate": buzz,
            "volume_ratio": round(volume_ratio, 2),
            "interpretation": interpretation,
            "note": "Based on volume as proxy. For real social data, integrate Twitter/Reddit APIs."
        }
    except Exception:
        return {"symbol": symbol, "error": "Could not calculate buzz"}


@cached(key_func=lambda symbol: f"full_sentiment:{symbol.upper()}")
@log_exceptions()
def get_full_sentiment(symbol: str) -> dict[str, Any]:
    """
    Get comprehensive sentiment analysis combining all sources.
    """
    results = {
        "symbol": symbol,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Get all sentiment data
    fear_greed = get_fear_greed_index()
    news = get_news_sentiment(symbol)
    analysts = get_analyst_consensus(symbol)
    buzz = get_social_buzz(symbol)
    reddit = get_reddit_sentiment(symbol)

    results["market_sentiment"] = fear_greed
    results["news_sentiment"] = news
    results["analyst_sentiment"] = analysts
    results["social_buzz"] = buzz
    results["reddit_hype"] = reddit

    # Calculate overall sentiment score
    signals = {"bullish": 0, "bearish": 0}

    # Fear & Greed contribution
    fg_score = fear_greed.get("score", 50)
    if fg_score < 30:
        signals["bullish"] += 2  # Extreme fear = contrarian buy
    elif fg_score < 45:
        signals["bullish"] += 1
    elif fg_score > 70:
        signals["bearish"] += 2  # Extreme greed = contrarian sell
    elif fg_score > 55:
        signals["bearish"] += 1

    # Reddit Hype contribution (NEW)
    hype_score = reddit.get("reddit_hype_score", 50)
    if hype_score > 75:
        signals["bullish"] += 2 # Strong Meme Hype
    elif hype_score < 30:
        signals["bearish"] += 1


    # News sentiment contribution
    news_score = news.get("sentiment_score", 0)
    if news_score > 0.2:
        signals["bullish"] += 2
    elif news_score > 0.1:
        signals["bullish"] += 1
    elif news_score < -0.2:
        signals["bearish"] += 2
    elif news_score < -0.1:
        signals["bearish"] += 1

    # Analyst contribution
    buy_pct = analysts.get("buy_percentage", 50)
    if buy_pct >= 70:
        signals["bullish"] += 1
    elif buy_pct < 30:
        signals["bearish"] += 1

    # Overall signal
    if signals["bullish"] > signals["bearish"] + 1:
        overall = "🟢 BULLISH SENTIMENT"
    elif signals["bearish"] > signals["bullish"] + 1:
        overall = "🔴 BEARISH SENTIMENT"
    else:
        overall = "⚪ MIXED SENTIMENT"

    results["overall_sentiment"] = overall
    results["signal_breakdown"] = signals

    return results


if __name__ == "__main__":
    print("=== Testing Sentiment Analysis ===\n")

    print("--- Fear & Greed Index ---")
    print(get_fear_greed_index())

    print("\n--- News Sentiment (NVDA) ---")
    print(get_news_sentiment("NVDA"))

    print("\n--- Analyst Consensus (NVDA) ---")
    print(get_analyst_consensus("NVDA"))

    print("\n--- Full Sentiment (NVDA) ---")
    print(get_full_sentiment("NVDA"))
