"""
Comprehensive Tool Health Check & Registry (Optimized & Parallelized)
Runs a deep diagnostic of all 90+ unique tool capabilities across the system.
Organized by category to provide the AI agent with a complete map of its abilities.
"""
import concurrent.futures
import logging
import os
import sys
import threading
import time
from typing import Any

from tools.exception_logger import log_exceptions

# Ensure project root is in path for imports to work when run standalone
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.logger import log_to_component

# MOCK Environment for standalone execution
if not os.environ.get("AIDLC_MODEL_ID"):
    os.environ["AIDLC_MODEL_ID"] = "mock-for-health-check"
if not os.environ.get("AWS_REGION"):
    os.environ["AWS_REGION"] = "us-east-1"

try:
    from agent.tool_registry import ALL_TOOLS
except ImportError:
    ALL_TOOLS = []

# ─────────────────────────────────────────────────────────────────────────────
# Thread-Safe Logging & Deduplication
# ─────────────────────────────────────────────────────────────────────────────

_log_lock = threading.Lock()
_logged_entries = set()  # Track logged entries to prevent duplicates
_health_check_lock = threading.Lock()  # Prevent concurrent health checks
_health_check_running = False  # Track if health check is currently running

def force_release_health_check_lock():
    """
    Emergency function to force-release a stuck health check lock.
    Only use this if a health check has been stuck for several minutes.
    """
    global _health_check_running

    log_to_component("tools", "HealthCheck",
                   "Force-releasing stuck health check lock",
                   {}, level=logging.WARNING)

    _health_check_running = False

    # Try to release the lock if it's held
    try:
        if _health_check_lock.locked():
            _health_check_lock.release()
            log_to_component("tools", "HealthCheck",
                           "Successfully released stuck lock",
                           {})
            return True
    except RuntimeError as e:
        # Lock wasn't held by this thread
        log_to_component("tools", "HealthCheck",
                       f"Could not release lock: {e}",
                       {}, level=logging.ERROR)
        return False

    return True

@log_exceptions()
def _thread_safe_log(component: str, phase: str, message: str, data: dict[str, Any], level: int = logging.INFO):
    """Thread-safe logging with deduplication."""
    # Create a unique key for this log entry
    log_key = (component, phase, message, data.get("tool"), data.get("status"))

    with _log_lock:
        if log_key not in _logged_entries:
            _logged_entries.add(log_key)
            log_to_component(component, phase, message, data, level)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Engine
# ─────────────────────────────────────────────────────────────────────────────

@log_exceptions()
def _probe(name: str, fn, *args, **kwargs) -> dict[str, Any]:
    """
    Run a single tool probe and capture timing + status.
    Improved logic: Only flags ERROR if result contains an explicit error key or is None.
    Enhanced logging: Captures full stack traces and detailed error context.
    """
    start = time.time()
    import sys
    import traceback

    try:
        if fn is None:
             error_detail = {
                "tool": name,
                "status": "💥 IMPORT ERROR",
                "latency_s": 0,
                "error": "Function not found / import failed",
                "error_type": "ImportError",
                "sample": None
            }
             _thread_safe_log("tools", "Probe", f"Probe {name}: IMPORT ERROR", error_detail, level=logging.ERROR)
             return error_detail

        if hasattr(fn, 'invoke'):
            import inspect
            if kwargs:
                invoke_arg = kwargs
            elif args:
                if len(args) == 1 and isinstance(args[0], dict):
                    invoke_arg = args[0]
                else:
                    try:
                        underlying = getattr(fn, 'func', fn)
                        sig = inspect.signature(underlying)
                        param_names = [p for p in sig.parameters if p not in ('self', 'cls')]
                        invoke_arg = dict(zip(param_names, args))
                    except Exception:
                        for candidate in ('symbol', 'ticker', 'query', 'topic', 'keywords'):
                            invoke_arg = {candidate: args[0]}
                            break
            else:
                invoke_arg = {}

            # Log the invocation attempt
            _thread_safe_log("tools", "Probe", f"Invoking {name} with args: {invoke_arg}", {}, level=logging.DEBUG)
            result = fn.invoke(invoke_arg)
        else:
            _thread_safe_log("tools", "Probe", f"Calling {name} with args: {args}, kwargs: {kwargs}", {}, level=logging.DEBUG)
            result = fn(*args, **kwargs)

        elapsed = round(time.time() - start, 2)
        is_error = False
        error_msg = None
        error_type = None

        if result is None:
            if "pattern" not in name.lower() and "breakout" not in name.lower():
                is_error = True
                error_msg = "None returned"
                error_type = "NoneResult"
                _thread_safe_log("tools", "Probe", f"Tool {name} returned None", {
                    "tool": name,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }, level=logging.WARNING)
        elif isinstance(result, dict):
            if result.get("error") is not None:
                is_error = True
                error_msg = str(result["error"])[:150]
                error_type = "DictError"
                # Log full error details
                _thread_safe_log("tools", "Probe", f"Tool {name} returned error dict", {
                    "tool": name,
                    "error": str(result["error"]),
                    "full_result": str(result)[:500],
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }, level=logging.ERROR)
        elif isinstance(result, str) and ("error" in result.lower() or "failed" in result.lower()) and len(result) < 200:
            is_error = True
            error_msg = result[:150]
            error_type = "StringError"
            _thread_safe_log("tools", "Probe", f"Tool {name} returned error string", {
                "tool": name,
                "error": result,
                "args": str(args)[:200],
                "kwargs": str(kwargs)[:200]
            }, level=logging.ERROR)

        res = {
            "tool": name,
            "status": "❌ ERROR" if is_error else "✅ OK",
            "latency_s": elapsed,
            "error": error_msg,
            "error_type": error_type,
            "sample": _summarize(result) if not is_error else None
        }

        if is_error:
            _thread_safe_log("tools", "Probe", f"Probe {name}: {res['status']}", res, level=logging.ERROR)
        else:
            _thread_safe_log("tools", "Probe", f"Probe {name}: {res['status']} ({elapsed}s)", {
                "tool": name,
                "latency_s": elapsed,
                "sample": res["sample"]
            }, level=logging.INFO)

        return res

    except Exception as e:
        elapsed = round(time.time() - start, 2)

        # Capture full stack trace
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # Extract the most relevant error location
        tb_lines = traceback.extract_tb(exc_traceback)
        error_location = None
        if tb_lines:
            last_frame = tb_lines[-1]
            error_location = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"

        res = {
            "tool": name,
            "status": "💥 EXCEPTION",
            "latency_s": elapsed,
            "error": str(e)[:150],
            "error_type": type(e).__name__,
            "error_location": error_location,
            "sample": None
        }

        # Log full exception details
        _thread_safe_log("tools", "Probe", f"Probe {name}: EXCEPTION - {type(e).__name__}: {str(e)}", {
            "tool": name,
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "error_location": error_location,
            "args": str(args)[:200],
            "kwargs": str(kwargs)[:200],
            "stack_trace": stack_trace,
            "latency_s": elapsed
        }, level=logging.ERROR)

        return res

@log_exceptions()
def _summarize(result) -> str:
    """Return a short human-readable summary of a tool result."""
    if isinstance(result, dict):
        keys = list(result.keys())[:5]
        return f"dict with keys: {keys}"
    elif isinstance(result, list):
        return f"list of {len(result)} items"
    elif isinstance(result, str):
        return (result[:100] + "...") if len(result) > 100 else result
    return str(result)[:100]


# Required LLM-provider credentials, keyed by the active LLM_PROVIDER. Only the
# selected provider's keys are required — switching providers changes the set.
_PROVIDER_REQUIRED_KEYS = {
    "bedrock": {
        "AWS_ACCESS_KEY_ID": "Bedrock LLM & embeddings",
        "AWS_SECRET_ACCESS_KEY": "Bedrock LLM & embeddings",
    },
    "azure": {
        "AZURE_OPENAI_API_KEY": "Azure OpenAI LLM",
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI endpoint",
    },
    "openai": {"OPENAI_API_KEY": "OpenAI LLM"},
    "anthropic": {"ANTHROPIC_API_KEY": "Anthropic Claude LLM"},
    "google": {"GOOGLE_API_KEY": "Google Gemini LLM"},
    # vertexai (Gemini on Vertex AI) authenticates with a service-account key
    # (JSON) pasted in Settings and stored in the keychain like any other secret.
    "vertexai": {"GOOGLE_SERVICE_ACCOUNT_KEY": "Vertex AI Gemini service-account key"},
}

# Optional data-provider credentials. These are NOT prerequisites: every dependent
# tool has a graceful fallback (yfinance, web scraping), so an unset key only means
# the enrichment source is idle — never a failure. Deliberately kept OUT of
# missing_prerequisites so the agent never presents them as "missing credentials".
_OPTIONAL_DATA_KEYS = {
    "FMP_API_KEY": "FMP financial data (insider, earnings, calendar, senate)",
    "ALPHA_VANTAGE_API_KEY": "Alpha Vantage quotes & fundamentals",
    "FRED_API_KEY": "FRED macro/economic data",
    "TAVILY_API_KEY": "Tavily web search",
    "FINNHUB_API_KEY": "Finnhub sentiment & news",
    "POLYGON_API_KEY": "Polygon options & technicals",
}


def _classify_credentials(environ=None):
    """Classify credentials into required (per active provider) vs optional.

    Returns ``(prerequisites, missing_results, optional_not_configured)``:
      - ``prerequisites``: dict[name -> status string] for the report.
      - ``missing_results``: probe-style dicts for genuinely-missing REQUIRED keys.
      - ``optional_not_configured``: dict[name -> purpose] for unset optional keys.

    Optional data-provider keys are never reported as missing prerequisites —
    they have graceful fallbacks, so an unset key is informational only.
    """
    env = environ if environ is not None else os.environ
    # Read LLM_PROVIDER from the .env file so the health check always reflects
    # the currently-configured provider, not a stale os.environ value from a
    # server that started before the provider was switched in Settings.
    provider = (env.get("LLM_PROVIDER", "bedrock") or "bedrock").lower()
    if environ is None:
        # os.environ may be stale (server started before the provider was
        # changed in Settings), so read the .env file directly as source of
        # truth.  Skip when a custom environ dict was passed (tests) so the
        # caller's LLM_PROVIDER is respected.
        try:
            from dotenv import dotenv_values as _dv
            _file_vals = _dv(os.path.join(os.getcwd(), "user_data", ".env"))
            provider = (_file_vals.get("LLM_PROVIDER") or provider).lower()
        except Exception:
            pass

    # Build a credential lookup that checks os.environ AND the OS keychain.
    # API keys are stored in the keychain (blank in .env after migration), so
    # env.get() alone returns "" for secrets — get_secret() covers both paths.
    # When a custom environ dict is passed (tests), skip the keychain lookup.
    _check_keychain = environ is None
    def _is_credential_set(key: str) -> bool:
        if env.get(key):
            return True
        if _check_keychain:
            try:
                from tools.secrets_store import get_secret, is_secret_key
                if is_secret_key(key):
                    return bool(get_secret(key))
            except Exception:
                pass
        return False

    required = _PROVIDER_REQUIRED_KEYS.get(provider, {})

    prerequisites: dict[str, str] = {}
    missing_results: list[dict[str, Any]] = []
    for key, purpose in required.items():
        if _is_credential_set(key):
            prerequisites[key] = "✅ Set"
        else:
            prerequisites[key] = f"❌ MISSING — required for {purpose}"
            missing_results.append({
                "tool": f"Prereq: {key}", "status": "❌ MISSING", "latency_s": 0,
                "error": f"Environment variable not set. Needed for: {purpose}", "sample": None,
            })

    optional_not_configured: dict[str, str] = {}
    for key, purpose in _OPTIONAL_DATA_KEYS.items():
        if _is_credential_set(key):
            prerequisites[key] = "✅ Set"
        else:
            prerequisites[key] = f"➖ Optional (not configured) — {purpose}"
            optional_not_configured[key] = purpose

    return prerequisites, missing_results, optional_not_configured


_TOOL_INVENTORY = {
"📊 PORTFOLIO MANAGEMENT": [
    "get_portfolio_summary: Full view of holdings, values, and P&L.",
    "calculate_portfolio_metrics: Sharpe, Beta, Volatility, Max Drawdown.",
    "calculate_var: Value at Risk (95% confidence loss estimate).",
    "analyze_correlation: Correlation matrix to check diversification.",
    "get_sector_exposure: Sector and industry weightings.",
    "analyze_my_portfolio_fx: Currency risk (USD/CAD) analysis.",
    "analyze_tax_loss_harvesting: Identify positions for tax benefits.",
    "get_trade_history: Log of past investment decisions.",
    "get_fee_income_analysis: Analysis of MER fees and dividend yields.",
    "analyze_factors: Growth/Value/Size factor exposure.",
    "get_geographic_exposure: US vs International exposure.",
    "construct_bond_ladder: 5-Year GIC/Bond ladder for safe income.",
    "project_portfolio_income: Detailed dividend and passive income projection.",
    "check_portfolio_allocation: Sector allocation balance check.",
    "get_hypothetical_portfolio: Model portfolio by persona type."
],
"📈 STOCK ANALYSIS": [
    "get_stock_data: Core fundamentals, PE, Market Cap.",
    "av_get_quote: Real-time price quote from AlphaVantage.",
    "get_realtime_quote: Live price from Yahoo Finance.",
    "get_fundamentals_detailed: Extended fundamental metrics.",
    "get_etf_holdings: Top holdings of an ETF.",
    "get_historical_performance: Multi-timeframe return history.",
    "get_comprehensive_technicals: RSI, MACD, Support/Resistance summary.",
    "detect_patterns: Chart pattern recognition (Flags, M&W, etc).",
    "find_support_resistance: Logic-based price level detection.",
    "get_price_targets: Institutional price targets and consensus.",
    "compare_assets: Multi-ticker comparison (multiples/performance).",
    "get_earnings_transcript: Qualitative management tone from calls.",
    "get_short_interest: Squeeze risk and short-seller data.",
    "generate_price_chart: ASCII/Visual chart generation.",
    "predict_earnings_surprise: Probability of earnings beat/miss.",
    "scan_sector_opportunities: Sector-specific opportunity scan.",
    "run_stock_deep_dive: Comprehensive single-stock analysis."
],
"🌍 MACRO & MARKET PULSE": [
    "get_global_market_snapshot: Multi-index view (SPY, QQQ, TSX, FTSE).",
    "get_inflation_data: Latest CPI (Headline and Core) YoY.",
    "get_fed_funds_rate: Current target rate and central bank trend.",
    "get_treasury_yields: 10Y/2Y spread and yield curve status.",
    "get_all_macro_indicators: Composite FRED macro dashboard.",
    "get_canada_macro_snapshot: Bank of Canada policy rate, CORRA, CPI-trim/median, bank rates.",
    "get_boc_fed_divergence: BoC-vs-Fed policy spread and its CAD implication.",
    "get_gdp_growth: Latest GDP growth rate.",
    "get_systemic_risk_indicators: Credit spreads and financial stress.",
    "get_economic_calendar: Upcoming economic events from FMP.",
    "get_dividend_analysis: Dividend history and yield analysis.",
    "calculate_currency_exposure: Portfolio currency breakdown.",
    "get_fear_greed_index: Market sentiment (CNN index).",
    "get_market_news: High-signal macro financial headlines.",
    "search_news: Web-search for recent specific stock events.",
    "get_reddit_sentiment: Retail hype and meme-stock tracking.",
    "analyze_macro_context: Overall market regime and strategy allocation.",
    "detect_sector_rotation: Leading vs Lagging sector identification.",
    "match_historical_regime: Matches current macro to historical analogues."
],
"🐚 OPTIONS & INSTITUTIONAL": [
    "analyze_options: Put/Call ratios, IV rank, and volatility skew.",
    "scan_unusual_activity: Massive institutional options sweeps.",
    "get_option_walls: Gamma-derived support and resistance.",
    "calculate_dealer_gex: Gamma Exposure analysis for squeezes.",
    "check_whale_accumulation: Dark pool and block trade tracking.",
    "scan_dark_pool_proxy: Intraday volume spike detection.",
    "get_fmp_insider_trades: Tracking management buys/sells.",
    "get_fmp_senate_disclosures: US Senate trading activity.",
    "get_insider_and_short_data: Combined insider + short data.",
    "model_options_strategy: Strategy modeling (covered call, etc).",
    "check_crowded_trade: Institutional over-ownership risk.",
    "get_institutional_ownership: 13F filing tracking."
],
"🎯 SENTIMENT & ALT DATA": [
    "get_news_sentiment: Per-ticker news sentiment scoring.",
    "get_full_sentiment: Comprehensive sentiment composite.",
    "get_analyst_consensus: Wall Street consensus rating.",
    "analyze_management_tone: Earnings call NLP tone analysis.",
    "get_alternative_data_signal: Alt data indicators (web traffic, app downloads)."
],
"🎲 SIMULATION & BACKTESTING": [
    "simulate_rebalancing: 'What if' portfolio change analysis.",
    "simulate_scenario: Stress tests (Recession, Tech Crash, Bull Market).",
    "backtest_strategy: Historical technical/DCA strategy backtesting.",
    "run_monte_carlo: Long-term wealth success probability.",
    "calculate_position_size: Risk-based position sizing."
],
"🌐 GEOPOLITICAL & ESG": [
    "scan_geopolitical_opportunities: Geopolitical event impact scan.",
    "get_ticker_geopolitical_context: Stock-specific geopolitical risks.",
    "get_supply_chain_exposure: Country supply chain risk analysis.",
    "check_esg_scores: Environmental/Social/Governance ratings.",
    "analyze_mutual_funds: Mutual fund analysis and comparison."
],
"📅 EARNINGS & CALENDAR": [
    "get_earnings_info: Next/past earnings dates and surprise data."
],
"🔧 WEB & UTILITIES": [
    "read_web_page: Fetch and summarize any web URL.",
    "clean_memory: Clear user memory/context (destructive)."
]
}


def _register_probe_tasks(tasks: list, results: list) -> None:
    """Register all health-check probes, grouped by category.

    Each category guards its imports in its own try/except so an import
    failure in one domain degrades to a single "IMPORT ERROR" result rather
    than aborting the whole check. Mutates ``tasks`` (probe
    ``(name, fn, args, kwargs)`` tuples) and ``results`` (import-verified /
    import-error entries) in place.
    """
    # Category 1: Portfolio
    try:
        from tools.fixed_income import construct_bond_ladder
        from tools.fx_utils import analyze_my_portfolio_fx
        from tools.income_analytics import project_portfolio_income
        from tools.portfolio_analytics import (
            analyze_correlation,
            analyze_factors,
            calculate_portfolio_metrics,
            calculate_var,
            get_fee_income_analysis,
            get_geographic_exposure,
            get_sector_exposure,
        )
        from tools.portfolio_csv import get_portfolio_summary
        from tools.portfolio_manager import get_hypothetical_portfolio
        from tools.sector_analysis import check_portfolio_allocation
        from tools.tax_loss import analyze_tax_loss_harvesting
        from tools.trade_journal import get_trade_history

        tasks.extend([
            ("get_portfolio_summary", get_portfolio_summary, (), {}),
            ("analyze_my_portfolio_fx", analyze_my_portfolio_fx, (), {}),
            ("analyze_tax_loss_harvesting", analyze_tax_loss_harvesting, (), {}),
            ("get_trade_history", get_trade_history, (), {}),
            ("calculate_portfolio_metrics", calculate_portfolio_metrics, (["AAPL", "MSFT"],), {}),
            ("calculate_var", calculate_var, (["AAPL"],), {}),
            ("analyze_correlation", analyze_correlation, (["AAPL", "MSFT", "SPY"],), {}),
            ("get_sector_exposure", get_sector_exposure, (["AAPL"],), {}),
            ("get_fee_income_analysis", get_fee_income_analysis, (["AAPL"],), {}),
            ("analyze_factors", analyze_factors, (["AAPL"],), {}),
            ("get_geographic_exposure", get_geographic_exposure, (["AAPL"],), {}),
            ("construct_bond_ladder", construct_bond_ladder, (100000,), {}),
            ("project_portfolio_income", project_portfolio_income, (["AAPL"], [100]), {}),
            ("check_portfolio_allocation", check_portfolio_allocation, (["AAPL", "NVDA", "XLE"],), {}),
            ("get_hypothetical_portfolio", get_hypothetical_portfolio, (), {}),
        ])
    except Exception as e:
        results.append({"tool": "Cat: Portfolio", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 2: Market Data
    try:
        from tools.alpha_vantage import get_company_overview, get_daily_prices
        from tools.alpha_vantage import get_quote as av_get_quote
        from tools.compare_assets import compare_assets
        from tools.market_data import (
            get_etf_holdings,
            get_fundamentals_detailed,
            get_historical_performance,
            get_realtime_quote,
            get_stock_data,
        )
        from tools.opportunity_scanner import scan_sector_opportunities
        from tools.price_targets import get_price_targets
        from tools.visualizer import generate_price_chart

        tasks.extend([
        ("get_stock_data (AAPL)", get_stock_data, ("AAPL",), {}),
        ("av_get_quote (AAPL)", av_get_quote, ("AAPL",), {}),
        ("get_daily_prices (AAPL, 5)", get_daily_prices, ("AAPL", 5), {}),
        ("get_company_overview (AAPL)", get_company_overview, ("AAPL",), {}),
        ("get_price_targets (AAPL)", get_price_targets, ("AAPL",), {}),
        ("compare_assets (AAPL, MSFT)", compare_assets, (["AAPL", "MSFT"],), {}),
        ("generate_price_chart (AAPL)", generate_price_chart, ("AAPL",), {}),
        ("get_realtime_quote (AAPL)", get_realtime_quote, ("AAPL",), {}),
        ("get_fundamentals_detailed (AAPL)", get_fundamentals_detailed, ("AAPL",), {}),
        ("get_etf_holdings (SPY)", get_etf_holdings, ("SPY",), {}),
        ("get_historical_performance (AAPL)", get_historical_performance, ("AAPL",), {}),
        # SKIP scan_sector_opportunities - medium runtime (~15-25s with batch architecture)
        # Verify import instead to avoid slowing down health check
    ])
        # Verify scan_sector_opportunities import without executing
        results.append({
            "tool": "scan_sector_opportunities",
            "status": "✅ OK" if callable(scan_sector_opportunities) else "❌ IMPORT ERROR",
            "latency_s": 0,
            "error": None,
            "sample": "Import verified (skipped execution — too slow for health check)"
        })
    except Exception as e:
        results.append({"tool": "Cat: Market Data", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 3: Technicals
    try:
        from tools.market_mechanics import detect_sector_rotation, predict_earnings_surprise, rank_relative_strength
        from tools.pattern_recognition import check_ma_crossover, detect_patterns, find_support_resistance
        from tools.screener import find_breakout_candidates
        from tools.technicals import get_comprehensive_technicals

        tasks.extend([
        ("get_comprehensive_technicals (AAPL)", get_comprehensive_technicals, ("AAPL",), {}),
        ("detect_patterns (AAPL)", detect_patterns, ("AAPL",), {}),
        ("find_support_resistance (AAPL)", find_support_resistance, ("AAPL",), {}),
        ("check_ma_crossover (AAPL)", check_ma_crossover, ("AAPL",), {}),
        ("find_breakout_candidates (AAPL)", find_breakout_candidates, ("AAPL",), {}),
        ("detect_sector_rotation", detect_sector_rotation, (), {}),
        ("rank_relative_strength", rank_relative_strength, (["AAPL", "MSFT"],), {}),
        ("predict_earnings_surprise (AAPL)", predict_earnings_surprise, ("AAPL",), {})
    ])
    except Exception as e:
        results.append({"tool": "Cat: Technicals", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 4: Macro
    try:
        from tools.boc_valet import get_boc_fed_divergence, get_canada_macro_snapshot
        from tools.fmp_api import get_economic_calendar
        from tools.fred_api import (
            get_all_macro_indicators,
            get_fed_funds_rate,
            get_gdp_growth,
            get_inflation_data,
            get_systemic_risk_indicators,
            get_treasury_yields,
        )
        from tools.macro_data import get_global_market_snapshot
        from tools.macro_strategy import analyze_macro_context
        from tools.market_data import get_dividend_analysis
        from tools.portfolio_analytics import calculate_currency_exposure

        tasks.extend([
        ("get_canada_macro_snapshot", get_canada_macro_snapshot, (), {}),
        ("get_boc_fed_divergence", get_boc_fed_divergence, (), {}),
        ("get_global_market_snapshot", get_global_market_snapshot, (), {}),
        ("get_fed_funds_rate", get_fed_funds_rate, (), {}),
        ("get_inflation_data", get_inflation_data, (), {}),
        ("get_treasury_yields", get_treasury_yields, (), {}),
        ("analyze_macro_context", analyze_macro_context, (), {}),
        ("get_all_macro_indicators", get_all_macro_indicators, (), {}),
        ("get_gdp_growth", get_gdp_growth, (), {}),
        ("get_systemic_risk_indicators", get_systemic_risk_indicators, (), {}),
        ("get_dividend_analysis (JNJ)", get_dividend_analysis, ("JNJ",), {}),
        ("calculate_currency_exposure", calculate_currency_exposure, (), {"holdings": {"AAPL": 50, "TSM": 30}}),
        ("get_economic_calendar", get_economic_calendar, (), {}),
    ])
    except Exception as e:
        results.append({"tool": "Cat: Macro", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 5: Options/Inst
    try:
        from tools.comprehensive_data import check_crowded_trade, get_institutional_ownership
        from tools.dark_pool import scan_dark_pool_proxy
        from tools.fmp_api import (
            get_earnings_transcript,
            get_fmp_insider_trades,
            get_fmp_senate_disclosures,
            get_short_interest,
        )
        from tools.insider_data import get_insider_and_short_data
        from tools.options import (
            analyze_options,
            calculate_dealer_gex,
            check_whale_accumulation,
            get_option_walls,
            scan_unusual_activity,
        )
        from tools.options_strategy import model_options_strategy

        tasks.extend([
        ("analyze_options (AAPL)", analyze_options, ("AAPL",), {}),
        ("scan_unusual_activity (AAPL)", scan_unusual_activity, ("AAPL",), {}),
        ("get_option_walls (AAPL)", get_option_walls, ("AAPL",), {}),
        ("scan_dark_pool_proxy (AAPL)", scan_dark_pool_proxy, ("AAPL",), {}),
        ("get_fmp_insider_trades (AAPL)", get_fmp_insider_trades, ("AAPL",), {}),
        ("get_earnings_transcript (AAPL)", get_earnings_transcript, ("AAPL",), {}),
        ("get_short_interest (AAPL)", get_short_interest, ("AAPL",), {}),
        ("check_whale_accumulation (AAPL)", check_whale_accumulation, ("AAPL",), {}),
        ("calculate_dealer_gex (AAPL)", calculate_dealer_gex, ("AAPL",), {}),
        ("check_crowded_trade (AAPL)", check_crowded_trade, ("AAPL",), {}),
        ("get_institutional_ownership (AAPL)", get_institutional_ownership, ("AAPL",), {}),
        ("get_fmp_senate_disclosures (AAPL)", get_fmp_senate_disclosures, ("AAPL",), {}),
        ("get_insider_and_short_data (AAPL)", get_insider_and_short_data, ("AAPL",), {}),
        ("model_options_strategy (AAPL)", model_options_strategy, ("AAPL",), {}),
    ])
    except Exception as e:
        results.append({"tool": "Cat: Options/Inst", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 6: Sentiment
    try:
        from tools.alternative_data import get_alternative_data_signal
        from tools.earnings_nlp import analyze_management_tone
        from tools.news_sources import get_company_news, get_market_news
        from tools.sentiment_analysis import (
            get_analyst_consensus,
            get_fear_greed_index,
            get_full_sentiment,
            get_news_sentiment,
            get_reddit_sentiment,
        )
        from tools.web_search import search_news

        tasks.extend([
        ("get_fear_greed_index", get_fear_greed_index, (), {}),
        ("get_reddit_sentiment (AAPL)", get_reddit_sentiment, ("AAPL",), {}),
        ("get_market_news", get_market_news, (3,), {}),
        ("get_company_news (AAPL)", get_company_news, ("AAPL",), {}),
        ("search_news (Nvidia AI)", search_news, ("Nvidia AI", 2), {}),
        ("get_analyst_consensus (AAPL)", get_analyst_consensus, ("AAPL",), {}),
        ("get_news_sentiment (AAPL)", get_news_sentiment, ("AAPL",), {}),
        ("get_full_sentiment (AAPL)", get_full_sentiment, ("AAPL",), {}),
        ("analyze_management_tone (AAPL)", analyze_management_tone, ("AAPL",), {}),
        ("get_alternative_data_signal (AAPL)", get_alternative_data_signal, ("AAPL",), {}),
    ])
    except Exception as e:
        results.append({"tool": "Cat: Sentiment", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 7: Planning/Simulation
    try:
        from tools.backtesting import backtest_strategy
        from tools.comprehensive_data import get_upcoming_ipos
        from tools.macro_analysis import run_stock_deep_dive
        from tools.market_scanner import scan_intraday_movers
        from tools.monte_carlo import run_monte_carlo
        from tools.position_sizing import calculate_position_size
        from tools.predictive import match_historical_regime
        from tools.seasonality import analyze_seasonality
        from tools.simulation import simulate_rebalancing, simulate_scenario
        from tools.trade_structuring import structure_trade_setup

        tasks.extend([
        ("run_monte_carlo", run_monte_carlo, (100000, 12000, 10), {}),
        ("scan_intraday_movers", scan_intraday_movers, (), {}),
        ("structure_trade_setup (AAPL)", structure_trade_setup, ("AAPL",), {}),
        ("analyze_seasonality (AAPL)", analyze_seasonality, ("AAPL",), {}),
        ("get_upcoming_ipos", get_upcoming_ipos, (), {}),
        # SKIP run_stock_deep_dive - too slow (calls 10+ tools internally)
        # SKIP assess_portfolio_risk - too slow (analyzes entire portfolio)
        ("simulate_rebalancing", simulate_rebalancing, ('{"AAPL": 50, "NVDA": 50}', '{"AAPL": 40, "NVDA": 40, "SPY": 20}'), {}),
        ("simulate_scenario", simulate_scenario, ("AAPL,MSFT", "recession"), {}),
        ("backtest_strategy (rsi)", backtest_strategy, (), {"strategy_type": "rsi", "symbols": ["AAPL"], "period": "1y"}),
        ("calculate_position_size", calculate_position_size, (100000,), {"entry_price": 150.0, "stop_loss_price": 140.0}),
    ])

    # Verify slow tools import without executing
        results.append({
        "tool": "run_stock_deep_dive",
        "status": "✅ OK" if callable(run_stock_deep_dive) else "❌ IMPORT ERROR",
        "latency_s": 0,
        "error": None,
        "sample": "Import verified (skipped execution — too slow for health check)"
    })

    # DSPy tools only if available
        try:
         tasks.append(("match_historical_regime", match_historical_regime, (3.2, 5.25, "bull"), {}))
        except Exception:
            pass
    except Exception as e:
        results.append({"tool": "Cat: Planning", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 8: Geopolitical & ESG
    try:
        from tools.esg_analytics import check_esg_scores
        from tools.fund_analytics import analyze_mutual_funds
        from tools.geopolitical_scanner import (
            get_supply_chain_exposure,
            get_ticker_geopolitical_context,
            scan_geopolitical_opportunities,
        )

        tasks.extend([
        # SKIP scan_geopolitical_opportunities - too slow (web scraping + analysis)
        ("get_ticker_geopolitical_context (AAPL)", get_ticker_geopolitical_context, ("AAPL",), {}),
        ("get_supply_chain_exposure (China)", get_supply_chain_exposure, ("China",), {}),
        ("check_esg_scores", check_esg_scores, (["AAPL", "MSFT"],), {}),
        ("analyze_mutual_funds", analyze_mutual_funds, (["VFIAX"],), {}),
    ])
    # Verify slow tool import without executing
        results.append({
        "tool": "scan_geopolitical_opportunities",
        "status": "✅ OK" if callable(scan_geopolitical_opportunities) else "❌ IMPORT ERROR",
        "latency_s": 0,
        "error": None,
        "sample": "Import verified (skipped execution — too slow for health check)"
    })
    except Exception as e:
        results.append({"tool": "Cat: Geopolitical/ESG", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 9: Earnings & Calendar
    try:
        from tools.earnings_calendar import get_earnings_info

        tasks.extend([
        ("get_earnings_info (AAPL)", get_earnings_info, ("AAPL",), {}),
    ])
    except Exception as e:
        results.append({"tool": "Cat: Earnings", "status": "💥 IMPORT ERROR", "error": str(e)})

    # Category 10: Web & Utilities
    try:
        from tools.memory import clean_memory
        from tools.web_reader import read_web_page

        tasks.extend([
        ("read_web_page (example.com)", read_web_page, ("https://example.com",), {}),
    ])
    # Note: clean_memory is destructive — only verify import, don't execute
        results.append({
        "tool": "clean_memory",
        "status": "✅ OK" if callable(clean_memory) else "❌ REGISTRY ERROR",
        "latency_s": 0,
        "error": None,
        "sample": "Import verified (skipped execution — destructive)"
    })
    except Exception as e:
        results.append({"tool": "Cat: Web/Utilities", "status": "💥 IMPORT ERROR", "error": str(e)})
# ─────────────────────────────────────────────────────────────────────────────
# Master Health Check
# ─────────────────────────────────────────────────────────────────────────────

@log_exceptions()
def run_tool_health_check() -> dict[str, Any]:
    """
    Runs a comprehensive probe of all tool capabilities in parallel.
    Uses ThreadPoolExecutor to prevent sequential API stalls.
    Thread-safe: Only one health check can run at a time.
    """
    global _logged_entries, _health_check_running

    # Check if health check is already running
    # If lock is held for more than 3 minutes, force release (likely stuck)
    lock_acquired = _health_check_lock.acquire(blocking=False)

    if not lock_acquired:
        # Check if the lock has been held too long (stuck health check)
        current_time = time.time()

        # Try to detect stuck lock by checking _health_check_running flag
        if _health_check_running:
            log_to_component("tools", "HealthCheck",
                           "Health check already in progress - skipping duplicate request",
                           {}, level=logging.WARNING)
            return {
                "health_summary": {
                    "overall_status": "⚠️ ALREADY RUNNING",
                    "operational": 0,
                    "failed": 0,
                    "total_checked": 0,
                    "prerequisites": {},
                    "missing_prerequisites": [],
                },
                "tool_results": [],
                "tool_inventory": {},
                "agent_instructions": "Health check already in progress. Please wait for it to complete."
            }
        else:
            # Lock is held but _health_check_running is False - likely stuck
            # This shouldn't happen, but if it does, log and try to recover
            log_to_component("tools", "HealthCheck",
                           "Detected stuck health check lock - attempting recovery",
                           {}, level=logging.ERROR)
            # Don't force release here - let it fail gracefully
            return {
                "health_summary": {
                    "overall_status": "⚠️ LOCK STUCK",
                    "operational": 0,
                    "failed": 0,
                    "total_checked": 0,
                    "prerequisites": {},
                    "missing_prerequisites": [],
                },
                "tool_results": [],
                "tool_inventory": {},
                "agent_instructions": "Health check lock is stuck. Please restart the server to recover."
            }

    try:
        # Clear global state from previous runs to prevent deadlocks
        _logged_entries.clear()
        _health_check_running = True

        log_to_component("tools", "HealthCheck",
                       "Starting health check",
                       {"timestamp": time.time()})

        results = []
        tasks = []

        # --- PREREQUISITE CHECKS (run before probes) ---
        # Required keys depend on the active LLM provider; optional data-provider
        # keys are classified separately and never counted as missing prerequisites.
        prerequisites, _missing_prereq_results, optional_not_configured = _classify_credentials()
        results.extend(_missing_prereq_results)

        # Check FAISS / dense tool retriever status
        try:
            from agent.tool_retriever import ToolRetriever

            retriever = ToolRetriever()
            dense_status = getattr(retriever, '_dense_status', 'unknown')
            dense_error = getattr(retriever, '_dense_error', None)

            # If still initializing, wait up to 5 seconds for it to complete
            if dense_status == "initializing":
                max_wait = 5  # seconds
                waited = 0
                while dense_status == "initializing" and waited < max_wait:
                    time.sleep(0.5)
                    waited += 0.5
                    dense_status = getattr(retriever, '_dense_status', 'unknown')
                    dense_error = getattr(retriever, '_dense_error', None)

            if dense_status == "ready":
                prerequisites["Tool_Retriever_Dense"] = f"✅ {dense_status}"
            elif dense_status == "initializing":
                # Transient — the background thread hasn't finished yet. Record in
                # prerequisites for visibility but do NOT add to results so it is
                # not counted as a broken tool (it will resolve on its own).
                prerequisites["Tool_Retriever_Dense"] = "⏳ initializing (background thread still running — BM25 active)"
            else:
                # Genuinely failed (e.g. embedding 404, missing model deployment).
                error_msg = dense_error or "Embedding index unavailable — using keyword fallback"
                prerequisites["Tool_Retriever_Dense"] = f"⚠️ {dense_status}" + (f" ({dense_error})" if dense_error else "")
                results.append({"tool": "Prereq: FAISS Dense Index", "status": f"⚠️ {dense_status.upper()}", "latency_s": 0, "error": error_msg, "sample": None})
        except Exception as e:
            prerequisites["Tool_Retriever_Dense"] = f"❌ Error: {e}"

        # --- PROBE REGISTRATION (per-category guarded imports) ---
        _register_probe_tasks(tasks, results)

        # --- EXECUTE IN PARALLEL ---
        MAX_WORKERS = 4  # Reduced to prevent [Errno 24] Too many open files and DNSError on macOS
        PROBE_TIMEOUT = 45  # Increased to 45s to allow for heavier tools under load
        EARLY_TERMINATION_THRESHOLD = 30
        OVERALL_TIMEOUT = 150  # Increased to 150s to prevent health check timeouts

        # Cancellation flag
        _cancelled = threading.Event()

        # Track execution timing
        parallel_start_time = time.time()

        # Log start of parallel execution
        log_to_component("tools", "HealthCheck",
                   f"Starting parallel execution of {len(tasks)} probes with {MAX_WORKERS} workers",
                   {"total_tasks": len(tasks), "max_workers": MAX_WORKERS, "timeout": OVERALL_TIMEOUT, "start_time": parallel_start_time})

        # Track which tools are currently running
        running_tools = set()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        log_to_component("tools", "HealthCheck",
                       "ThreadPoolExecutor created, submitting tasks",
                       {"max_workers": MAX_WORKERS})

        future_to_name = {}
        submission_start = time.time()

        try:
            from tools.user_profile import get_active_profile, run_under_profile
            active_profile = get_active_profile()
        except Exception:
            active_profile = None
            run_under_profile = None

        for (name, fn, args, kwargs) in tasks:
            try:
                if run_under_profile is not None and active_profile is not None:
                    future = executor.submit(run_under_profile, active_profile, _probe, name, fn, *args, **kwargs)
                else:
                    future = executor.submit(_probe, name, fn, *args, **kwargs)
                future_to_name[future] = name
                running_tools.add(name)
            except Exception as e:
                log_to_component("tools", "HealthCheck",
                               f"Failed to submit task {name}",
                               {"tool": name, "error": str(e)},
                               level=logging.ERROR)

        submission_elapsed = time.time() - submission_start
        log_to_component("tools", "HealthCheck",
                       f"All {len(future_to_name)} tasks submitted in {submission_elapsed:.2f}s",
                       {"submitted": len(future_to_name), "elapsed": submission_elapsed})

        completed_count = 0
        failed_count = 0
        last_log_time = time.time()

        try:
            log_to_component("tools", "HealthCheck",
                           "Entering as_completed loop",
                           {"timeout": OVERALL_TIMEOUT})

            for future in concurrent.futures.as_completed(future_to_name, timeout=OVERALL_TIMEOUT):
                # Check for cancellation
                if _cancelled.is_set():
                    log_to_component("tools", "HealthCheck",
                                   "Cancellation flag set, breaking loop",
                                   {"completed": completed_count, "failed": failed_count})
                    # Cancel remaining futures
                    for f in future_to_name:
                        f.cancel()
                    break

                name = future_to_name[future]
                running_tools.discard(name)
                completed_count += 1

                # Log progress every 5 seconds or every 20 tools
                current_time = time.time()
                elapsed_since_start = current_time - parallel_start_time
                should_log = (completed_count % 20 == 0) or (current_time - last_log_time >= 5)

                if should_log:
                    remaining = len(tasks) - completed_count
                    log_to_component("tools", "HealthCheck",
                                   f"Progress update: {completed_count}/{len(tasks)} completed",
                                   {
                                       "completed": completed_count,
                                       "total": len(tasks),
                                       "failed": failed_count,
                                       "remaining": remaining,
                                       "elapsed_seconds": round(elapsed_since_start, 2),
                                       "currently_running_count": len(running_tools),
                                       "currently_running_sample": list(running_tools)[:5]
                                   })
                    last_log_time = current_time

                try:
                    probe_res = future.result(timeout=PROBE_TIMEOUT)
                    results.append(probe_res)

                    # Track failures
                    if probe_res.get("status") not in ["✅ OK"]:
                        failed_count += 1

                    # Early termination if too many failures
                    if failed_count > EARLY_TERMINATION_THRESHOLD:
                        log_to_component("tools", "HealthCheck",
                                       f"Early termination: {failed_count} failures detected",
                                       {"failed": failed_count, "completed": completed_count})
                        _cancelled.set()
                        for f in future_to_name:
                            f.cancel()
                        break

                except Exception as e:
                    # Use standard error handling for timeouts/exceptions
                    failed_count += 1
                    error_type = "TIMEOUT" if isinstance(e, concurrent.futures.TimeoutError) else "EXCEPTION"

                    # Log which tool timed out
                    log_to_component("tools", "HealthCheck",
                                   f"Tool {name} {error_type}: {str(e)[:100]}",
                                   {"tool": name, "error_type": error_type, "error": str(e)[:200]},
                                   level=logging.ERROR)

                    results.append({
                        "tool": name,
                        "status": f"💥 {error_type}",
                        "latency_s": PROBE_TIMEOUT,
                        "error": str(e)[:150],
                        "error_type": error_type,
                        "sample": None
                    })

            # Log completion of as_completed loop
            loop_elapsed = time.time() - parallel_start_time
            log_to_component("tools", "HealthCheck",
                           "as_completed loop finished normally",
                           {
                               "completed": completed_count,
                               "failed": failed_count,
                               "elapsed_seconds": round(loop_elapsed, 2)
                           })

        except concurrent.futures.TimeoutError:
            # Overall timeout exceeded - cancel remaining futures and log which tools didn't complete
            timeout_elapsed = time.time() - parallel_start_time
            incomplete_tools = [name for future, name in future_to_name.items() if not future.done()]

            log_to_component("tools", "HealthCheck",
                           f"Overall timeout ({OVERALL_TIMEOUT}s) exceeded - {len(incomplete_tools)} tools incomplete",
                           {
                               "completed": completed_count,
                               "total": len(tasks),
                               "incomplete_count": len(incomplete_tools),
                               "incomplete_tools": incomplete_tools[:10],  # Log first 10
                               "elapsed_seconds": round(timeout_elapsed, 2),
                               "still_running": list(running_tools)[:10]
                           },
                           level=logging.ERROR)

            # Cancel all remaining futures
            cancel_start = time.time()
            cancelled_count = 0
            for f in future_to_name:
                if not f.done():
                    f.cancel()
                    cancelled_count += 1
            cancel_elapsed = time.time() - cancel_start

            log_to_component("tools", "HealthCheck",
                           f"Cancelled {cancelled_count} remaining futures",
                           {"cancelled": cancelled_count, "cancel_time": round(cancel_elapsed, 2)})

            # Add timeout entries for incomplete tools
            for tool_name in incomplete_tools:
                results.append({
                    "tool": tool_name,
                    "status": "💥 OVERALL_TIMEOUT",
                    "latency_s": OVERALL_TIMEOUT,
                    "error": f"Health check overall timeout ({OVERALL_TIMEOUT}s) exceeded",
                    "error_type": "OverallTimeout",
                    "sample": None
                })

        except Exception as e:
            # Unexpected exception in as_completed loop
            loop_elapsed = time.time() - parallel_start_time
            log_to_component("tools", "HealthCheck",
                           f"Unexpected exception in as_completed loop: {str(e)}",
                           {
                               "error": str(e),
                               "error_type": type(e).__name__,
                               "completed": completed_count,
                               "elapsed_seconds": round(loop_elapsed, 2)
                           },
                           level=logging.ERROR)
            raise

        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            # Log final state
            final_elapsed = time.time() - parallel_start_time
            log_to_component("tools", "HealthCheck",
                           "Parallel execution phase complete",
                           {
                               "completed": completed_count,
                               "failed": failed_count,
                               "total_tasks": len(tasks),
                               "elapsed_seconds": round(final_elapsed, 2)
                           })

        # --- REGISTRY INTEGRITY CHECK (Dynamic for all 109 tools) ---
        registry_start = time.time()
        log_to_component("tools", "HealthCheck",
                       "Starting registry integrity check",
                       {"total_tools_in_registry": len(ALL_TOOLS)})

        probed_names = {r['tool'].split(' ')[0] for r in results if not r.get('tool', '').startswith('Cat:')}
        registry_checked = 0

        for tool in ALL_TOOLS:
            if tool.name not in probed_names:
                # Soft probe for registered tools not in the hard-probe list
                results.append({
                    "tool": f"{tool.name} (Registry)",
                    "status": "✅ OK" if hasattr(tool, 'func') or hasattr(tool, 'invoke') else "❌ REGISTRY ERROR",
                    "latency_s": 0,
                    "error": None,
                    "sample": "Registered in ALL_TOOLS"
                })
                registry_checked += 1

        registry_elapsed = time.time() - registry_start
        log_to_component("tools", "HealthCheck",
                       "Registry integrity check complete",
                       {
                           "checked": registry_checked,
                           "elapsed_seconds": round(registry_elapsed, 2)
                       })

        # --- SYNTHESIZE REPORT ---
        synthesis_start = time.time()
        log_to_component("tools", "HealthCheck",
                       "Starting report synthesis",
                       {"total_results": len(results)})

        total = len(results)
        ok = sum(1 for r in results if r.get("status") == "✅ OK")
        failed = total - ok
        registry_count = len(ALL_TOOLS)

        # Build detailed broken tools list with error messages and types
        broken_tools_detailed = []
        for r in results:
            if r.get("status") != "✅ OK":
                tool_name = r["tool"]
                error_msg = r.get("error", "Unknown error")
                error_type = r.get("error_type", "Unknown")
                error_location = r.get("error_location", "")

                if error_msg:
                    detail = f"{tool_name}: [{error_type}] {error_msg}"
                    if error_location:
                        detail += f" (at {error_location})"
                    broken_tools_detailed.append(detail)
                else:
                    broken_tools_detailed.append(f"{tool_name}: [{error_type}] No error details")

        broken_tools = sorted([r["tool"] for r in results if r.get("status") != "✅ OK"])
        broken_tools_with_reasons = sorted(broken_tools_detailed)

        tool_inventory = _TOOL_INVENTORY

        status_str = "🟢 ALL SYSTEMS GO" if failed == 0 else ("🟡 DEGRADED" if failed <= 10 else "🔴 CRITICAL FAILURES")


        # Build prerequisite summary string
        prereq_lines = [f"  {k}: {v}" for k, v in prerequisites.items()]
        prereq_summary = "\n".join(prereq_lines)
        # Only GENUINELY missing REQUIRED prerequisites count as "missing" (the ❌
        # entries: the active LLM provider's credentials). Optional data-provider
        # keys (➖) and informational warnings like the dense-index status (⚠️) are
        # excluded so the agent never reports optional integrations as missing.
        missing_prereqs = [k for k, v in prerequisites.items() if "❌" in v]

        optional_summary = (
            ", ".join(optional_not_configured) if optional_not_configured else "None"
        )
        agent_instructions = (
        f"HEALTH CHECK STATUS: {status_str} ({ok}/{total} tools verified, registry size: {registry_count}).\n"
        f"PREREQUISITES:\n{prereq_summary}\n"
        f"OPTIONAL INTEGRATIONS NOT CONFIGURED (non-critical, graceful fallbacks active): {optional_summary}\n"
        f"BROKEN TOOLS ({len(broken_tools)}):\n"
        f"{chr(10).join('  - ' + bt for bt in broken_tools_with_reasons) if broken_tools_with_reasons else '  None'}.\n\n"
        "INSTRUCTIONS FOR AGENT:\n"
        "1. DO NOT use or cite data from tools marked ❌ ERROR or 💥 EXCEPTION.\n"
        "2. Use the 'tool_inventory' to understand your full capabilities.\n"
        "3. Always prefer tools over training data for live market metrics.\n"
        "4. Optional integrations that are not configured are NOT missing credentials and NOT errors — "
        "the app works fully without them via fallbacks (yfinance, web scraping). Do NOT present them "
        "as 'missing'; if relevant, describe them only as optional sources the user MAY enable."
        )

        synthesis_elapsed = time.time() - synthesis_start
        log_to_component("tools", "HealthCheck",
                       "Report synthesis complete",
                       {
                           "status": status_str,
                           "operational": ok,
                           "failed": failed,
                           "total": total,
                           "elapsed_seconds": round(synthesis_elapsed, 2)
                       })

        return {
        "health_summary": {
            "overall_status": status_str,
            "operational": ok,
            "failed": failed,
            "total_checked": total,
            "prerequisites": prerequisites,
            "missing_prerequisites": missing_prereqs,
            "optional_not_configured": optional_not_configured,
        },
        "tool_results": results,
        "tool_inventory": tool_inventory,
        "agent_instructions": agent_instructions
    }

    finally:
        # Always release the lock and reset running flag
        _health_check_running = False
        _health_check_lock.release()

        # Only log if variables are defined (avoid UnboundLocalError)
        try:
            log_to_component("tools", "HealthCheck",
                           "Health check completed",
                           {"status": status_str, "operational": ok, "failed": failed})
        except NameError:
            log_to_component("tools", "HealthCheck",
                           "Health check completed with early exit",
                           {})

if __name__ == "__main__":
    print("🚀 Running Concurrent System Health Check...")
    report = run_tool_health_check()
    log_to_component("tools", "HealthCheck", f"Completed: {report['health_summary']['overall_status']}", report["health_summary"])
    print(f"\n{report['agent_instructions']}")
