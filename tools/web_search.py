import threading
import time

import requests

from tools.cache import cached
from tools.credential_manager import get_api_key, report_rate_limit
from tools.exception_logger import log_exceptions

# ---------------------------------------------------------------------------
# GLOBAL DDG CONCURRENCY GUARD
# ---------------------------------------------------------------------------
# Only ONE DuckDuckGo search runs at a time system-wide.
# Others queue behind it.  This prevents the "stampede" that occurs when
# Tavily rate-limits and 8+ threads all slam DDG simultaneously.
_ddg_lock = threading.Lock()
_ddg_last_call = 0.0       # epoch of last DDG request (rate-limit spacing)
_DDG_MIN_INTERVAL = 1.5    # minimum seconds between DDG calls
_DDG_TIMEOUT = 15           # seconds — down from 50; DDG either works fast or not at all
# How long a caller waits to acquire the DDG lock before giving up. Deep Reasoning's
# tool-execution timeout is 120s (agent/nodes/deep_reasoning.py), so this must leave
# room for the DDG call itself (up to _DDG_TIMEOUT) once the lock is acquired.
_DDG_LOCK_WAIT = 100

# ---------------------------------------------------------------------------
# TAVILY CIRCUIT BREAKER
# ---------------------------------------------------------------------------
# When Tavily rejects a request for plan/quota exhaustion (HTTP 432, or a body
# saying the plan's usage limit is exceeded), that quota stays dead for hours —
# yet every subsequent search would still POST to Tavily, log an ERROR, and only
# then fall back. Open a circuit for a cooldown and send searches straight to
# DuckDuckGo until it resets. A transient 429 still uses key rotation, not this.
_tavily_circuit_open_until = 0.0
_tavily_circuit_lock = threading.Lock()
_TAVILY_CIRCUIT_COOLDOWN = 3600  # seconds; quota exhaustion is not a momentary blip


def _tavily_circuit_open() -> bool:
    """True while the Tavily circuit is open (quota cooldown active)."""
    return time.time() < _tavily_circuit_open_until


def _trip_tavily_circuit() -> None:
    """Open the Tavily circuit for the cooldown period."""
    global _tavily_circuit_open_until
    with _tavily_circuit_lock:
        _tavily_circuit_open_until = time.time() + _TAVILY_CIRCUIT_COOLDOWN


@log_exceptions()
def _duckduckgo_search_fallback(query: str, max_results: int = 5) -> str:
    """Final fallback: DuckDuckGo search with timeout protection (free, no API key needed).

    Hardened against stampede:
      • Global lock ensures only one DDG request at a time
      • 15s timeout (down from 50s)
      • Rate-limit spacing between calls
      • Queues behind the lock for up to _DDG_LOCK_WAIT seconds instead of
        dropping the search immediately — only gives up if the wait itself
        risks blowing the caller's own tool-execution timeout
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    from agent.logger import log_to_component

    # Queue behind the lock rather than abandoning the search after a few seconds
    acquired = _ddg_lock.acquire(timeout=_DDG_LOCK_WAIT)
    if not acquired:
        log_to_component("tools", "WebSearch", "DuckDuckGo lock contention — gave up after long wait", {
            "query": query[:100],
            "waited_seconds": _DDG_LOCK_WAIT
        }, level=30)
        return (
            "⚠️ [Search Throttled]\n\n"
            f"Waited {_DDG_LOCK_WAIT}s for another search to finish and still couldn't get a turn. "
            "Search is heavily contended right now — try a narrower query or again shortly.\n\n"
            f"Query: {query}"
        )

    try:
        # Rate-limit spacing
        global _ddg_last_call
        elapsed_since_last = time.time() - _ddg_last_call
        if elapsed_since_last < _DDG_MIN_INTERVAL:
            time.sleep(_DDG_MIN_INTERVAL - elapsed_since_last)

        def _run_ddg_search():
            """Run DuckDuckGo NEWS search in a separate thread with timeout protection."""
            try:
                from ddgs import DDGS

                log_to_component("tools", "WebSearch", "Attempting DuckDuckGo news search", {
                    "query": query[:100],
                    "max_results": max_results
                })

                results = []
                with DDGS() as ddgs:
                    # Use ddgs.news() for current articles instead of ddgs.text()
                    # which returns generic web pages (often years old)
                    try:
                        for i, result in enumerate(ddgs.news(query, max_results=max_results)):
                            if i >= max_results:
                                break
                            results.append(result)
                    except Exception:
                        # If news endpoint fails, fall back to text with timelimit
                        for i, result in enumerate(ddgs.text(query, max_results=max_results, timelimit="w")):
                            if i >= max_results:
                                break
                            results.append(result)

                if not results:
                    log_to_component("tools", "WebSearch", "DuckDuckGo returned no results", {
                        "query": query[:100]
                    }, level=30)  # WARNING
                    return None

                log_to_component("tools", "WebSearch", "DuckDuckGo news search successful", {
                    "query": query[:100],
                    "result_count": len(results)
                })

                return results

            except ImportError as e:
                log_to_component("tools", "WebSearch", "DuckDuckGo import failed", {
                    "error": str(e)
                }, level=40)  # ERROR
                raise
            except Exception as e:
                log_to_component("tools", "WebSearch", "DuckDuckGo search failed", {
                    "query": query[:100],
                    "error": str(e),
                    "error_type": type(e).__name__
                }, level=40)  # ERROR
                raise

        # Run DuckDuckGo search with tight timeout
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_run_ddg_search)
            try:
                results = future.result(timeout=_DDG_TIMEOUT)

                if not results:
                    return "⚠️ [All Search Providers Failed]\n\nNo search results available. Tavily hit rate limits, and DuckDuckGo returned no results."

                formatted = []
                for r in results:
                    title = r.get("title", "No Title")
                    # ddgs.news() uses 'url', ddgs.text() uses 'href'
                    link = r.get("url") or r.get("href", "No Link")
                    body = r.get("body", "No summary available")
                    date = r.get("date", "")
                    source = r.get("source", "")

                    date_str = f"\n*{source} | {date}*" if date else ""
                    entry = f"**{title}**{date_str}\nLink: {link}\nSummary: {body}\n---"
                    formatted.append(entry)

                return "\n".join(formatted)

            except FuturesTimeoutError:
                log_to_component("tools", "WebSearch", f"DuckDuckGo search timed out after {_DDG_TIMEOUT}s", {
                    "query": query[:100]
                }, level=40)  # ERROR
                return f"⚠️ [All Search Providers Failed]\n\nSearch timeout: DuckDuckGo took longer than {_DDG_TIMEOUT} seconds.\n\nQuery: {query}"
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        except ImportError:
            return "Error: duckduckgo_search library not installed (should be in requirements)."
        except Exception as e:
            log_to_component("tools", "WebSearch", "DuckDuckGo fallback failed completely", {
                "query": query[:100],
                "error": str(e),
                "error_type": type(e).__name__
            }, level=40)  # ERROR
            return f"⚠️ [All Search Providers Failed]\n\nDuckDuckGo error: {str(e)}\n\nAll search providers (Tavily, DuckDuckGo) are currently unavailable."
    finally:
        _ddg_last_call = time.time()
        _ddg_lock.release()

@cached(key_func=lambda query, max_results=5, timelimit=None: f"search:tavily_ddg:{query.lower()[:60]}:{max_results}:{timelimit}")
@log_exceptions()
def search_news(query: str, max_results: int = 5, timelimit: str = None) -> str:
    """
    Searches for news and context using a 2-tier fallback chain:
    1. Tavily API (Agentic RAG, requires API key, best quality)
    2. DuckDuckGo (free, no API key needed, no rate limits)

    Returns formatted search results with title, link, and summary for each result.
    """
    from agent.logger import log_to_component

    api_key = get_api_key("TAVILY_API_KEY")
    if not api_key:
        log_to_component("tools", "WebSearch", "Tavily API key missing, falling back to DuckDuckGo", {}, level=30)
        return _duckduckgo_search_fallback(query, max_results)

    # Circuit breaker: quota is exhausted — skip Tavily entirely and go straight
    # to DuckDuckGo rather than re-hitting the dead quota and logging an ERROR.
    if _tavily_circuit_open():
        log_to_component("tools", "WebSearch", "Tavily circuit open (quota cooldown), using DuckDuckGo directly", {
            "query": query[:100]
        }, level=30)
        return _duckduckgo_search_fallback(query, max_results)

    log_to_component("tools", "WebSearch", "Attempting Tavily search", {
        "query": query[:100],
        "max_results": max_results,
        "timelimit": timelimit
    })

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced", # Triggers full-page extraction
        "topic": "news" if timelimit else "general",
        "max_results": max_results,
    }

    # Map timelimits (tavily only supports 'days' for news topic)
    if timelimit == 'd':
        payload["days"] = 1
    elif timelimit == 'w':
        payload["days"] = 7

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)

        # Handle rate-limiting with key rotation before falling back to DuckDuckGo
        if response.status_code == 429:
             log_to_component("tools", "WebSearch", "Tavily rate limit hit, attempting key rotation", {
                 "query": query[:100]
             }, level=30)
             report_rate_limit("TAVILY_API_KEY", api_key)
             next_key = get_api_key("TAVILY_API_KEY")
             if next_key and next_key != api_key:
                 payload["api_key"] = next_key
                 response = requests.post(url, json=payload, headers=headers, timeout=20)

        # Handle 4xx or 5xx intelligently with fallback
        if response.status_code != 200:
             # Quota/plan exhaustion (HTTP 432 or an explicit "usage limit" body)
             # is not transient — open the circuit so later searches skip Tavily,
             # and log at WARNING instead of ERROR since it's now a handled state.
             body = response.text.lower()
             quota_exceeded = response.status_code == 432 or any(
                 sig in body for sig in ("usage limit", "exceeds your plan", "upgrade your plan")
             )
             if quota_exceeded:
                 _trip_tavily_circuit()
                 log_to_component("tools", "WebSearch", "Tavily quota exhausted — circuit opened, falling back to DuckDuckGo", {
                     "query": query[:100],
                     "status_code": response.status_code,
                     "cooldown_sec": _TAVILY_CIRCUIT_COOLDOWN
                 }, level=30)
             else:
                 log_to_component("tools", "WebSearch", "Tavily returned non-200 status, falling back to DuckDuckGo", {
                     "query": query[:100],
                     "status_code": response.status_code,
                     "response_text": response.text[:200]
                 }, level=40)  # ERROR
             return _duckduckgo_search_fallback(query, max_results)

        data = response.json()
        results = data.get("results", [])

        # Fallback to general if news finds absolutely nothing
        if not results and payload.get("topic") == "news":
             log_to_component("tools", "WebSearch", "Tavily news search returned no results, retrying with general topic", {
                 "query": query[:100]
             }, level=30)
             payload["topic"] = "general"
             payload.pop("days", None)
             resp2 = requests.post(url, json=payload, headers=headers, timeout=20)
             if resp2.status_code == 200:
                 results = resp2.json().get("results", [])

        if not results:
            log_to_component("tools", "WebSearch", "Tavily returned no results, falling back to DuckDuckGo", {
                "query": query[:100]
            }, level=30)
            return _duckduckgo_search_fallback(query, max_results)

        log_to_component("tools", "WebSearch", "Tavily search successful", {
            "query": query[:100],
            "result_count": len(results)
        })

        formatted = []
        for r in results:
            title = r.get("title", "No Title")
            link = r.get("url", "No Link")
            content = r.get("content", "No content")

            entry = f"Title: {title}\nLink: {link}\nSummary: {content}\n---"
            formatted.append(entry)

        return "\n".join(formatted)

    except requests.exceptions.Timeout as e:
        log_to_component("tools", "WebSearch", "Tavily request timeout, falling back to DuckDuckGo", {
            "query": query[:100],
            "error": str(e)
        }, level=40)
        return _duckduckgo_search_fallback(query, max_results)
    except requests.exceptions.RequestException as e:
        log_to_component("tools", "WebSearch", "Tavily request failed, falling back to DuckDuckGo", {
            "query": query[:100],
            "error": str(e),
            "error_type": type(e).__name__
        }, level=40)
        return _duckduckgo_search_fallback(query, max_results)
    except Exception as e:
        # Network errors or timeouts fall back to DuckDuckGo
        log_to_component("tools", "WebSearch", "Tavily search exception, falling back to DuckDuckGo", {
            "query": query[:100],
            "error": str(e),
            "error_type": type(e).__name__
        }, level=40)
        return _duckduckgo_search_fallback(query, max_results)

if __name__ == "__main__":
    print(search_news("latest stock market news"))
