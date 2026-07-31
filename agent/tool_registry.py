from langchain_core.tools import tool

from tools.alpha_vantage import get_company_overview, get_daily_prices
from tools.alpha_vantage import get_quote as av_get_quote
from tools.alternative_data import get_alternative_data_signal
from tools.asset_location import analyze_asset_location
from tools.australian_market import get_australian_analyst_estimates, get_australian_quote
from tools.canadian_market import (
    get_canadian_analyst_estimates,
    get_canadian_quote,
)
from tools.canadian_market import (
    scan_tsx_movers as tool_scan_tsx_movers,
)
from tools.compare_assets import compare_assets
from tools.comprehensive_data import (
    check_crowded_trade,
    get_insider_trading,
    get_institutional_ownership,
    get_upcoming_ipos,
)
from tools.comprehensive_data import get_earnings_calendar as _raw_get_earnings_calendar
from tools.dark_pool import scan_dark_pool_proxy
from tools.earnings_calendar import get_earnings_info
from tools.earnings_nlp import analyze_management_tone
from tools.european_market import get_european_analyst_estimates, get_european_quote
from tools.event_radar import build_event_radar_cached
from tools.fmp_api import (
    get_earnings_transcript,
    get_economic_calendar,
    get_fmp_insider_trades,
    get_fmp_senate_disclosures,
    get_short_interest,
)
from tools.fred_api import get_all_macro_indicators
from tools.fund_flows import collect_active_profile_fund_universe, get_flow_series
from tools.fx_utils import analyze_my_portfolio_fx
from tools.geopolitical_scanner import (
    get_supply_chain_exposure,
    get_ticker_geopolitical_context,
    scan_geopolitical_opportunities,
)
from tools.insider_data import get_insider_and_short_data
from tools.macro_analysis import assess_portfolio_risk as macro_portfolio_risk
from tools.macro_analysis import run_stock_deep_dive as macro_stock_deep_dive
from tools.macro_data import get_global_market_snapshot
from tools.macro_strategy import analyze_macro_context
from tools.market_data import get_dividend_analysis, get_etf_holdings, get_stock_data
from tools.market_data import get_historical_performance as fetch_historical_perf
from tools.market_mechanics import detect_sector_rotation, predict_earnings_surprise, rank_relative_strength
from tools.market_scanner import scan_intraday_movers as tool_scan_intraday_movers
from tools.market_sentinel import get_market_regime, get_regime_history
from tools.memory import clean_memory
from tools.monte_carlo import run_monte_carlo as run_mc_engine
from tools.news_sources import get_company_news, get_market_news
from tools.opportunity_scanner import scan_sector_opportunities
from tools.options import (
    analyze_options,
    calculate_dealer_gex,
    scan_unusual_activity,
)
from tools.pattern_recognition import check_ma_crossover, detect_patterns, find_support_resistance
from tools.portfolio_analytics import (
    analyze_correlation,
    analyze_factors,
    calculate_portfolio_metrics,
    calculate_var,
    estimate_marginal_risk_contribution,
    generate_portfolio_charts,
    get_fee_income_analysis,
    get_geographic_exposure,
    get_sector_exposure,
)
from tools.portfolio_csv import get_portfolio_summary, get_tradeable_symbols
from tools.portfolio_reconciliation import get_reconciliation
from tools.position_sizing import calculate_position_size
from tools.screener import find_breakout_candidates
from tools.seasonality import analyze_seasonality
from tools.sentiment_analysis import (
    get_analyst_consensus,
    get_fear_greed_index,
    get_full_sentiment,
    get_news_sentiment,
    get_reddit_sentiment,
)
from tools.simulation import simulate_rebalancing, simulate_scenario
from tools.tax_loss import analyze_tax_loss_harvesting
from tools.technicals import get_comprehensive_technicals
from tools.trade_journal import close_trade, get_trade_history, log_trade
from tools.trade_structuring import structure_trade_setup as get_trade_setup
from tools.visualizer import generate_monte_carlo_chart, generate_price_chart
from tools.web_reader import read_web_page
from tools.web_search import search_news

try:
    from tools.price_targets import get_price_targets
except ImportError:
    get_price_targets = None
import json
import time

from agent.dspy_setup import DSPY_AVAILABLE
from agent.utils import is_cancelled, safe_print, send_status
from tools.cache import get_cached, set_cached
from tools.fixed_income import construct_bond_ladder as get_bond_ladder
from tools.options_strategy import model_options_strategy as get_options_strat
from tools.portfolio_manager import get_hypothetical_portfolio as get_portfolio_proxy

# Global Cache for Portfolio (15 min TTL)
PORTFOLIO_CACHE = {"data": None, "timestamp": 0}

def _get_portfolio_data():
    """Helper to fetch dynamic portfolio data for the active profile."""
    from tools.portfolio_csv import get_portfolio_summary, get_tradeable_symbols
    summary = get_portfolio_summary()
    symbols = get_tradeable_symbols()

    if isinstance(summary, dict) and "error" in summary:
        return symbols, {}, "No portfolio data available."

    # Context string for agents
    ctx = f"Total Value: {summary.get('summary', {}).get('current_value', 'Unknown')}\n"
    holdings_list = [f"{h['symbol']} ({h['gain_loss']})" for h in summary.get('holdings', [])]
    ctx += f"Holdings: {', '.join(holdings_list[:20])}"

    allocations = {h['symbol']: h['value_usd'] for h in summary.get('holdings', []) if 'symbol' in h and 'value_usd' in h}

    return symbols, allocations, ctx

@tool
def fetch_fundamentals(symbol: str):
    """Get price, PE ratio, dividends, 52-week position, and earnings date."""
    return get_stock_data(symbol)

@tool
def analyze_technicals(symbol: str):
    """
        [Backend-Only] Get comprehensive technical analysis (Support/Resist, Trend, Momentum, Patterns).
        Returns a detailed text summary of indicators. Does NOT generate a chart image.
        Use this to understand the technical setup (bullish/bearish) before making a recommendation.
        """
    return get_comprehensive_technicals(symbol)

@tool
def plot_chart(symbol: str):
    """Generate an ASCII price chart."""
    return generate_price_chart(symbol)

@tool
def fetch_comprehensive_analysis(symbol: str):
    """
        PREFERRED: Fetch fundamentals, technicals, and sentiment in PARALLEL for speed.
        Use this for comprehensive stock analysis - much faster than calling each tool separately.
        Results are cached for 5 minutes.
        """
    from tools.cache import fetch_stock_data_parallel
    return fetch_stock_data_parallel(symbol)

@tool
def analyze_portfolio_risk(symbols: str, investment_amount: int=1000000):
    """
        Analyze a portfolio for Institutional Risk Metrics.
        Use this when user provides a list of stocks (e.g. "analyze my portfolio of AAPL, MSFT").
        If user says "analyze MY PORTFOLIO" (without listing stocks), pass "PORTFOLIO" as the symbols argument.

        Args:
           symbols: Comma-separated list of tickers (e.g. "AAPL, MSFT") OR "PORTFOLIO" to load from CSV.
           investment_amount: Total portfolio value (default 1,000,000)
        """
    weights = None
    portfolio_mode = False
    if symbols.strip().upper() in ['PORTFOLIO', 'MY PORTFOLIO', 'ALL', 'MY STOCKS']:
        portfolio_mode = True
        # Use get_portfolio_summary() as single source of truth for values & currency
        summary = get_portfolio_summary()
        if isinstance(summary, dict) and "error" in summary:
            return f"Error loading portfolio: {summary.get('error')}"

        enriched_holdings = summary.get("holdings", [])
        usd_cad_rate = summary.get("usd_cad_rate", 1.40)

        tradable = []
        for h in enriched_holdings:
            sym = h.get('symbol', '').upper()
            val_usd = h.get('value_usd', 0)
            if val_usd > 1.0 and not h.get('is_private_asset', False) and sym:
                tradable.append(h)

        symbol_list = [h.get('symbol', '').upper() for h in tradable if h.get('symbol')]

        # Use current market value (in USD) for weighting — consistent with assess_portfolio_risk
        total_market_value_usd = sum(h.get('value_usd', 0) for h in tradable)
        if total_market_value_usd > 0:
            weights = [h.get('value_usd', 0) / total_market_value_usd for h in tradable]
            # Override investment_amount with actual portfolio value in CAD
            investment_amount = total_market_value_usd * usd_cad_rate

        if not symbol_list:
            return 'Error: No tradable stocks found in your portfolio CSV.'
        report_header = f'### 🏦 Institutional Report: My Portfolio (~${investment_amount:,.0f} CAD)'
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        report_header = f'### 🏦 Institutional Portfolio Report (${investment_amount:,.0f})'
    if not symbol_list:
        return 'Error: No valid symbols provided.'
    report = []
    report.append(report_header)
    report.append(f"**Assets:** {', '.join(symbol_list[:10])}..." if len(symbol_list) > 10 else f"**Assets:** {', '.join(symbol_list)}")
    report.append('')

    # Filter to tradeable symbols for risk metrics (exclude CASH, pensions, etc.)
    if portfolio_mode:
        from tools.portfolio_csv import get_tradeable_symbols
        tradeable_set = set(get_tradeable_symbols())
        tradeable_pairs = [(s, w) for s, w in zip(symbol_list, weights or [1.0/len(symbol_list)]*len(symbol_list)) if s in tradeable_set]
        if tradeable_pairs:
            risk_symbols, risk_weights = zip(*tradeable_pairs)
            risk_symbols = list(risk_symbols)
            risk_weights = list(risk_weights)
            # Renormalize weights after filtering
            w_sum = sum(risk_weights)
            if w_sum > 0:
                risk_weights = [w / w_sum for w in risk_weights]
        else:
            risk_symbols, risk_weights = symbol_list, weights
    else:
        risk_symbols, risk_weights = symbol_list, weights

    metrics_1y = calculate_portfolio_metrics(risk_symbols, weights=risk_weights, period="1y")
    metrics_10y = calculate_portfolio_metrics(risk_symbols, weights=risk_weights, period="10y")

    if 'error' not in metrics_1y:
        m1 = metrics_1y.get('metrics', {})
        m10 = metrics_10y.get('metrics', {}) if 'error' not in metrics_10y else {}
        interp = metrics_1y.get('interpretation', {})

        report.append('#### 📊 Core Risk Metrics (Trailing 1Y vs. Historical 10Y)')
        report.append('| Metric | Current (1Y) | Historical (10Y) | Rating (Current) |')
        report.append('| :--- | :--- | :--- | :--- |')
        report.append(f"| **Sharpe Ratio** | {m1.get('sharpe_ratio')} | {m10.get('sharpe_ratio', 'N/A')} | {interp.get('sharpe')} |")
        report.append(f"| **Beta (vs SPY)** | {m1.get('beta')} | {m10.get('beta', 'N/A')} | {interp.get('beta')} |")
        report.append(f"| **Max Drawdown** | {m1.get('max_drawdown')} | {m10.get('max_drawdown', 'N/A')} | ⚠️ Historic Max Loss |")
        report.append(f"| **Volatility** | {m1.get('annual_volatility')} | {m10.get('annual_volatility', 'N/A')} | {interp.get('volatility')} |")
        report.append(f"| **Exp. Return** | {m1.get('annual_return')} | {m10.get('annual_return', 'N/A')} | Annualized |")
        report.append('')

        if m10:
            # Add a small insight if there's a big gap
            try:
                ret_1y = float(m1.get('annual_return', '0%').replace('%', ''))
                ret_10y = float(m10.get('annual_return', '0%').replace('%', ''))
                if ret_1y > ret_10y + 10:
                    report.append(f"> 💡 **Note:** Recent performance ({m1.get('annual_return')}) is significantly higher than the 10-year historical average ({m10.get('annual_return')}). This may indicate a temporary bull run rather than a permanent portfolio characteristic.\n")
            except:
                pass
    var = calculate_var(risk_symbols, weights=risk_weights, investment=investment_amount)
    if 'error' not in var:
        v_data = var.get('value_at_risk', {})
        report.append('#### 🛡️ Stress Test (Value at Risk)')
        report.append(f"> **95% Confidence:** You should NOT lose more than **{v_data.get('daily_var_dollars')}** in a single day.")
        report.append('> *Worst-Case Note:* In extreme crashes (like 2020), losses can exceed this.\n')
    corr = analyze_correlation(risk_symbols)
    if 'error' not in corr:
        report.append('#### 🔗 Diversification Check')
        report.append(f"**Overall Quality:** {corr.get('diversification_quality')}")
        report.append(f"**Avg Correlation:** {corr.get('average_correlation')}")
        pairs = corr.get('correlation_pairs', [])[:3]
        if pairs:
            report.append('\n**⚠️ Highest Correlations (Risk of Moving Together):**')
            for p in pairs:
                report.append(f"- {p['pair']}: **{p['correlation']}**")
        report.append('')
    sect = get_sector_exposure(symbol_list, weights=weights, is_portfolio=portfolio_mode)
    if 'error' not in sect:
        report.append('#### 🏗️ Sector Allocation')
        for (s, pct) in sect.get('sector_breakdown', {}).items():
            report.append(f'- **{s}:** {pct}%')
        if sect.get('concentration_warning'):
            report.append(f"\n> ⚠️ {sect.get('concentration_warning')}")
    costs = get_fee_income_analysis(symbol_list, weights=weights)
    if 'error' not in costs:
        report.append('')
        report.append('#### 💰 Cost & Income Analysis')
        waer = costs.get('weighted_expense_ratio', 0)
        fee_rating = costs.get('fee_rating', 'Unknown')
        report.append(f'**Weighted Expense Ratio:** {waer * 100:.2f}% ({fee_rating})')
        high_fees = costs.get('high_fee_funds', [])
        if high_fees:
            report.append(f"> ⚠️ **High Fee Alert:** You are paying >0.50% fees on: {', '.join(high_fees)}")
        yield_pct = costs.get('expected_yield', 0)
        est_income = yield_pct * investment_amount
        report.append(f'**Est. Annual Dividend Income:** ${est_income:,.0f} ({yield_pct * 100:.2f}% Yield)')
        if costs.get('dividend_payers_count', 0) > 0:
            report.append(f"*(Passive income from {costs.get('dividend_payers_count')} dividend-paying assets)*")

    report.append('')
    report.extend(_build_canonical_metrics_table(
        metrics_1y if 'error' not in metrics_1y else {},
        var if 'error' not in var else {},
        corr if 'error' not in corr else {},
        costs if 'error' not in costs else {},
        investment_amount,
    ))

    macro = analyze_macro_context()
    if 'error' not in macro:
        report.append('')
        report.append(f"#### 🌍 Macro Strategy ({macro.get('current_regime')})")
        if macro.get('plain_english'):
            report.append(f'''> *"{macro.get('plain_english')}"*''')
        if macro.get('canadian_strategy'):
            report.append(f"> **{macro.get('canadian_strategy')}**")
        strat = macro.get('strategy', {})
        opps = strat.get('tactical_opportunity', [])
        if opps:
            report.append(f"**💡 Tactical Opportunity:** Consider adding {', '.join(opps)}")
        und = strat.get('sectors_to_underweight', [])
        if und:
            report.append(f"**⚠️ Defensive Move:** Consider trimming {', '.join(und)}")
        inds = macro.get('key_indicators', {})
        risk_alert = inds.get('Systemic Risk', 'Low')
        liq = inds.get('Liquidity (M2)', 'Neutral')
        report.append(f"> *Context: Risk {risk_alert} | Liquidity {liq} | Inflation {inds.get('Inflation (US)')}*")
    return '\n'.join(report)

@tool
def analyze_options_chain(symbol: str):
    """Analyze options chain (Put/Call Ratio, Volatility) for sentiment."""
    return analyze_options(symbol)

@tool
def search_stock_news(symbol: str):
    """Search for recent news about the stock - earnings, lawsuits, product launches, analyst ratings."""
    return search_news(f'{symbol} stock news latest')

@tool
def get_latest_trump_yaps(days: int = 15, max_posts: int = 30):
    """
    Fetch the latest raw social media posts/statements from Donald Trump's Truth Social account.
    Args:
        days: Fetch posts from within this number of days (default: 15).
        max_posts: Maximum number of posts to return (default: 30) to avoid overloading context.
    Use this to get the most recent, real-time political and policy yaps directly to analyze their market/supply-chain impact.
    """
    from tools.trump_tracker import get_latest_trump_posts
    return get_latest_trump_posts(days=days, max_posts=max_posts)

@tool
def analyze_sectors(symbols: str):
    """Analyze sector exposure and get rebalancing suggestions. Pass comma-separated symbols like 'AAPL,MSFT,JNJ'."""
    symbol_list = [s.strip() for s in symbols.split(',')]
    from tools.sector_analysis import check_portfolio_allocation
    return check_portfolio_allocation(symbol_list)

@tool
def compare_stocks(symbols: str, mode: str='fundamentals'):
    """
        Compare multiple stocks / competitors side by side. Useful for deciding
        between similar companies (e.g. "KO vs PEP" or "compare AAPL and MSFT").
        Args:
            symbols: comma-separated list, e.g. 'AAPL, MSFT'
            mode: 'fundamentals' (PE, Market Cap, Yield) OR 'performance' (Relative Strength vs SPY)
        """
    sym_list = [s.strip() for s in symbols.split(',')]
    from tools.compare_assets import compare_assets
    return compare_assets(sym_list, mode=mode)

@tool
def get_analyst_targets(symbol: str):
    """Get analyst price targets, upside potential, and recommended entry points."""
    return get_price_targets(symbol)

@tool
def run_retirement_simulation(years: int, annual_contribution: int=0, start_value: int=0, withdrawal: int=0):
    """
        Run a Monte Carlo simulation to forecast retirement savings.

        Args:
            years: Number of years to simulate (e.g. Retirement Age - Current Age).
            annual_contribution: How much the user saves per year.
            start_value: Current portfolio value. If 0, tries to use user memory.
            withdrawal: Annual spending (if already retired).
        """
    if start_value == 0:
        start_value = 100000
    result = run_mc_engine(current_portfolio_value=float(start_value), annual_contribution=float(annual_contribution), years=int(years), annual_withdrawal=float(withdrawal))
    chart_token = generate_monte_carlo_chart(result)
    return f"{result['interpretation']}\n\n{chart_token}"

@tool
def get_earnings_calendar(symbol: str):
    """Get upcoming earnings date, historical beat rate, and earnings warnings."""
    return get_earnings_info(symbol)

@tool
def calculate_position(portfolio_value: float, entry_price: float, stop_loss_price: float,
                       risk_per_trade_pct: float = 0.0):
    """Calculate recommended position size from the stop distance and the user's OWN stated max-risk rule.

    Leave risk_per_trade_pct at 0 to use the rule from the user's profile. If they have stated no
    risk limit there is no default (no '2% rule'): pass an explicit risk_per_trade_pct to get a
    size, and present it as your own assumption rather than as the user's limit. Check `risk_basis`
    in the result before attributing the number to the user.
    """
    return calculate_position_size(
        portfolio_value, risk_per_trade_pct or None, entry_price, stop_loss_price
    )

@tool
def preview_candidate_impact(symbol: str, size_pct: float = 0.0, size_usd: float = 0.0,
                             stop: float = 0.0, entry: float = 0.0):
    """"Should I add this?" — recompute the portfolio WITH this candidate at a proposed
    size and report the DELTA: IPS position/sector/dollar-at-risk checks plus before/after
    beta, volatility, and CVaR (tail risk). A portfolio-fit analysis, not a buy/sell verdict.
    Size defaults to an assumed 5% probe; pass size_pct (% of portfolio) or size_usd to be
    exact, and stop/entry to include the dollar-at-risk check."""
    from tools.candidate_impact import preview_candidate_impact as _preview
    return _preview(
        symbol,
        size_usd=size_usd or None,
        size_pct=size_pct or None,
        stop=stop or None,
        entry=entry or None,
    )

@tool
def optimize_portfolio(objective: str = "min_vol", target_vol_pct: float = 0.0,
                       period: str = "1y", risk_free_rate_pct: float = 0.0):
    """Optimize the live portfolio under the user's OWN stated IPS caps (position, fund,
    sector, restricted list). objective: 'min_vol' (lowest risk), 'max_sharpe' (best
    return per unit risk), or 'target_vol' (max return under target_vol_pct annual
    volatility). Weights come from the historical window — labelled estimation, never a
    forecast. If the user has stated no caps, the optimization is unconstrained and says
    so. Report current vs optimized weights and stats; do not present as a trade plan."""
    from tools.portfolio_optimizer import optimize_portfolio as _opt
    return _opt(
        objective=objective,
        target_vol_pct=target_vol_pct or None,
        period=period,
        risk_free_rate_pct=risk_free_rate_pct,
    )

@tool
def check_rebalance_drift(target_allocation: str = "", objective: str = "",
                          target_vol_pct: float = 0.0, period: str = "1y"):
    """"Should I rebalance?" — checks current weights against a target using the user's
    OWN drift band from their drawdown playbook (rebalance_drift_pct). If no band is
    stored the check is unavailable and says why — nothing invents one.

    CALL THIS WITH NO ARGUMENTS for "should I rebalance?". With no target_allocation
    and no objective, drift is measured against the allocation the user has STORED
    (Context › Target Allocation) — their own stated plan, which is what the question
    is about. If they have never stored one, the check says so and names that screen;
    it does not substitute a target of its own.

    The other two routes are OVERRIDES for what-if questions, and each REPLACES the
    stored plan for that call:
    - target_allocation: JSON, e.g. '{"AAPL": 30, "SCHD": 40, "XESG.TO": 30}'. Rescaled
      to total 100%, so a mix summing to 90% becomes fully invested — deliberate cash
      survives in the stored plan but NOT here.
    - objective: 'min_vol'/'max_sharpe'/'target_vol' — drift against a fresh optimizer
      solve, which is a computed mix, not something the user has stated.
    Pass one only when the user asked for that specific comparison. The result reports
    which basis was used (`target_basis`/`target_source`) and flags the substitution
    (`stored_target_overridden`) when a stored plan was overridden — relay that flag.

    When the band is breached, reports the trades back to target, turnover, and the
    taxable realized-gain exposure of the sells (the tax bill itself is withheld — no
    marginal rate is stated in the profile)."""
    from tools.portfolio_optimizer import check_rebalance_drift as _drift
    alloc = None
    if target_allocation and target_allocation.strip():
        try:
            alloc = json.loads(target_allocation)
        except (ValueError, TypeError):
            return {"available": False,
                    "reason": "target_allocation must be a JSON object like '{\"AAPL\": 30, \"SCHD\": 70}'"}
        if not isinstance(alloc, dict):
            return {"available": False, "reason": "target_allocation must decode to a JSON object of symbol -> weight"}
    return _drift(
        target_allocation=alloc,
        objective=objective or None,
        target_vol_pct=target_vol_pct or None,
        period=period,
    )

@tool
def get_insider_short_interest(symbol: str):
    """Get insider trading activity, short interest %, and institutional ownership."""
    return get_insider_and_short_data(symbol)

@tool
def get_realtime_quote(symbol: str):
    """Get the latest available price quote (US, Canadian .TO, and international stocks).

    NOT guaranteed to be a live intraday tick: the underlying feed is end-of-day on
    the current key tier, so during a session this often returns the PREVIOUS close.
    Always check the returned `is_stale` / `as_of` / `staleness_note` fields before
    describing the price as current or as today's move.
    """
    return av_get_quote(symbol)

@tool
def get_price_history(symbol: str, days: int=30):
    """Get historical daily prices (open, high, low, close, volume) for chart analysis."""
    return get_daily_prices(symbol, days)

@tool
def get_fundamentals_detailed(symbol: str):
    """Get detailed company fundamentals: PE, EPS, dividends, beta, analyst targets. US stocks only."""
    return get_company_overview(symbol)

@tool
def get_macro_overview():
    """
        Get all macro-economic indicators in one call: Federal Funds Rate (interest
        rates and their trend), inflation (CPI), GDP, unemployment, and Treasury
        yields / yield curve (10Y vs 2Y — inverted curve = recession warning).
        Use for any question about the Fed, rates, inflation, recession risk,
        bond yields, or general market context.
        """
    return get_all_macro_indicators()

@tool
def get_macro_strategy():
    """Get 'Market Regime' (Inflation/Recession) and tactical sector recommendations based on Fed data."""
    return analyze_macro_context()

@tool
def get_canada_macro():
    """
        Canadian macro straight from the Bank of Canada (Valet API): the policy rate
        (target for the overnight rate) with the date and size of its last move, CORRA
        and its spread to the target, CPI-trim / CPI-median core inflation vs the 2%
        target, posted chartered-bank prime / mortgage / GIC rates, and the BoC-vs-Fed
        policy divergence with its CAD implication.
        Use for any question about the Bank of Canada, CAD interest rates, Canadian
        inflation, Canadian mortgage or GIC rates, or why the loonie is moving.
        This is the source of record for Canada — get_macro_overview is US data.
        """
    from tools.boc_valet import get_canada_macro_snapshot
    return get_canada_macro_snapshot()

@tool
def get_boc_vs_fed():
    """
        Bank of Canada vs US Federal Reserve policy-rate divergence: the spread in basis
        points, how each has moved over the past year, and the carry mechanism that
        transmits it into USD/CAD. Use for 'is the BoC ahead of or behind the Fed',
        'why is the CAD weak', or currency-hedging context on US holdings.
        """
    from tools.boc_valet import get_boc_fed_divergence
    return get_boc_fed_divergence()

@tool
def analyze_patterns(symbol: str):
    """Detect chart patterns: support/resistance, MA crossovers, RSI divergence. Predicts likely price direction."""
    return detect_patterns(symbol)

@tool
def get_support_resistance(symbol: str):
    """Find key support & resistance price levels. Support = buy zone, Resistance = sell zone."""
    return find_support_resistance(symbol)

@tool
def get_ma_signals(symbol: str):
    """Check for moving average crossovers (Golden Cross = bullish, Death Cross = bearish)."""
    return check_ma_crossover(symbol)

@tool
def get_sentiment(symbol: str):
    """Get comprehensive sentiment: Fear/Greed, news, analyst ratings. Contrarian indicator."""
    return get_full_sentiment(symbol)

@tool
def get_analyst_ratings(symbol: str):
    """Get analyst Buy/Hold/Sell ratings and price targets."""
    return get_analyst_consensus(symbol)

@tool
def visualize_stock_chart(symbol: str, period: str='3mo'):
    """Generate an ASCII price chart for the terminal to visualize trends. Periods: 1mo, 3mo, 6mo, 1y."""
    return generate_price_chart(symbol, period)

@tool
def scan_opportunities(sector: str = 'All'):
    """
        Find timely investment opportunities. With sector='All' this runs the Opportunity
        Funnel V2 — a top-down, accumulation-first engine: it builds a live universe from
        sector rotation, movers, active geo/policy themes and guru picks, ranks themes by
        inflow + macro + catalyst, scores names on theme/relative-strength/forward pillars,
        and demotes already-extended (parabolic) names via an entry-stage gate. It catches
        early secular winners (e.g. a memory/semis name at the base) rather than just
        beaten-down value.

        Args:
            sector: 'All' (cross-sector funnel — the default for "scan the market" /
                    "find opportunities" / "what should I buy"),
                    'Growth Leaders' (Tech/Discretionary), 'Value & Defensive'
                    (Healthcare/Staples/Utilities), or a specific sector like
                    'Tech', 'Healthcare', 'Energy', 'Finance'.

        ROUTING:
        - "scan the market" / "find opportunities" / "what should I buy" / "what's
          inflecting now" → pass sector='All' (the funnel).
        - "what's cheap / on sale / beaten down / oversold dip" → a value question; pass
          a specific sector (or 'Value & Defensive'), which uses the legacy value rubric.
        - Only pass a specific sector if the user names one.
        """
    return scan_sector_opportunities(sector)

@tool
def scan_guru_picks():
    """
        📺 GURU SCANNER: Analyze the latest Media Guru picks through
        the full opportunity pipeline.

        Fetches recent Buy/Sell recommendations from financial TV personalities,
        and runs them through the identical rigorous 4-phase pipeline as the main
        opportunity scanner. It outputs:
        - Deep fundamentals and technicals
        - The Guru's signal (BUY/SELL/FEATURED)
        - Freshness tracking (to avoid stale picks)

        Trigger this tool for queries like:
        - 'Guru picks', 'what did the TV gurus say', 'Media sentiment'
        - 'TV recommendations', 'Media picks'
        - 'Lightning round picks', 'Guru analysis'
        """
    from tools.opportunity_scanner import scan_sector_opportunities
    return scan_sector_opportunities("Guru")

@tool
def get_funnel_scorecard():
    """
        📊 SCANNER TRACK RECORD: How accurate have the opportunity scanner's past
        picks actually been? Returns the realized performance of matured scan
        signals — hit rate and average alpha vs SPY AND vs each pick's sector
        benchmark — broken down by conviction tier and entry stage, at 14- and
        21-day horizons, with sample-size caveats.

        Also includes the MISS DETECTOR: how the names the scanner scored but
        REJECTED went on to perform (regret analysis per cut gate — did the
        entry/risk gates reject winners or dodge losers?).

        Call this alongside scan_opportunities so you can tell the user how the
        scanner's past picks performed (always cite the sample-size caveats).
        Also use for: 'how accurate is the scanner', 'scanner track record',
        'did your past picks work', 'backtest results', 'what did the scanner miss'.
        """
    from tools.funnel_backtest import get_funnel_scorecard_data
    return get_funnel_scorecard_data()

@tool
def get_advisor_performance_scorecard():
    """
        📊 ADVISOR TRACK RECORD: How accurate has your advice actually been?
        Returns the scorecard showing hit rates and average relative outperformance (alpha)
        versus the S&P 500 (SPY) for past BUY, SELL, HOLD, TRIM, or ADD advice.

        Use this to answer queries like:
        - "How good has your advice been?"
        - "What is your historical track record?"
        - "Show me your scorecard."
        - "Are your stock recommendations profitable?"
        """
    from tools.memory import get_advisor_scorecard
    return get_advisor_scorecard()

@tool
def record_recommendation_execution(ticker: str, executed: bool, note: str = ''):
    """Record whether the USER acted on prior advice for a ticker.

    Call this ONLY when the user states it themselves, e.g.:
    - "I bought AAPL yesterday"           -> executed=True
    - "I did not execute on the TSLA buy" -> executed=False
    - "I passed on that" / "never filled" -> executed=False

    The recommendation ledger is written by extracting the advisor's own text, so it
    records what was ADVISED and can never know what was DONE. This is the only way
    that fact enters memory.

    NEVER infer it from the portfolio: a name being absent cannot distinguish declined
    from unfilled from bought-and-since-sold, and guessing manufactures a false history.
    If the user has not said, leave it unrecorded and ask.

    Args:
        ticker: The ticker the user is reporting on (e.g. 'AAPL').
        executed: True if the user acted on the advice, False if they declined.
        note: Optional short verbatim context ("waiting for a pullback").
    """
    from tools.memory import set_recommendation_execution
    return set_recommendation_execution(ticker, executed, note or None)

@tool
def scan_geopolitical_events(event: str=''):
    """
        Scan for geopolitical event-driven investment opportunities.
        Detects wars, sanctions, supply chain disruptions and maps them to investable tickers.

        Use this when user asks about:
        - Geopolitical events, wars, sanctions, conflicts
        - Supply chain disruptions
        - Country-specific risks (e.g., 'Iran', 'Qatar', 'Taiwan', 'Russia')
        - 'What opportunities exist from current events'
        - Commodity price spikes from world events

        Args:
            event: Optional specific event description (e.g., 'Iran strike on Qatar').
                  If empty, auto-detects from latest news.
        """
    return scan_geopolitical_opportunities(event if event else None)

@tool
def check_ticker_geopolitical_context(symbol: str):
    """
        Check if a specific stock is exposed to geopolitical events or supply chain disruptions.
        Use this for commodity-linked stocks (energy, mining, agriculture, shipping, defense)
        to see if they benefit from a "conflict premium" or face supply risks.
        """
    return get_ticker_geopolitical_context(symbol)

@tool
def check_supply_chain(country: str):
    """
        Look up which commodities and tickers are exposed to a specific country.
        Use when user asks about a country's economic impact, e.g., 'what happens if Taiwan is invaded'.

        Args:
            country: Country name (e.g., 'Qatar', 'Taiwan', 'Russia', 'China')
        """
    return get_supply_chain_exposure(country)

@tool
def run_diagnostics():
    """
        Run a full system health check to see which tools are working and which are broken.
        Call this when the user asks 'what tools do you have', 'health check', 'run diagnostics', or 'system status'.
        """
    from tools.health_check import run_tool_health_check

    result = run_tool_health_check()

    # Return condensed summary to prevent overwhelming the LLM with massive JSON
    # Full report is still logged and available in the backend
    broken_tools = [r["tool"] for r in result.get("tool_results", []) if r.get("status") != "✅ OK"]

    return {
        "overall_status": result["health_summary"]["overall_status"],
        "operational_tools": result["health_summary"]["operational"],
        "failed_tools": result["health_summary"]["failed"],
        "total_checked": result["health_summary"]["total_checked"],
        "broken_tools": broken_tools[:10] if len(broken_tools) > 10 else broken_tools,
        "missing_prerequisites": result["health_summary"].get("missing_prerequisites", []),
        "optional_not_configured": result["health_summary"].get("optional_not_configured", {}),
        "summary": f"{result['health_summary']['operational']}/{result['health_summary']['total_checked']} tools operational. " +
                   (f"Broken: {', '.join(broken_tools[:5])}" if broken_tools else "All systems operational."),
        "note": "Full detailed report available in backend logs. This is a condensed summary for agent consumption. "
                "'optional_not_configured' lists optional data sources the user MAY enable — these are NOT missing credentials or errors."
    }

@tool
def dealer_gamma_exposure(symbol: str):
    """Analyze dealer Gamma Exposure (GEX) to predict explosive moves or squeezes."""
    return calculate_dealer_gex(symbol)

@tool
def get_alt_data(symbol: str):
    """Fetch alternative data (web traffic/downloads proxy) as leading earnings indicator."""
    return get_alternative_data_signal(symbol)

@tool
def check_management_tone(symbol: str):
    """Analyze earnings call transcript for management tone dispersion (Bullish/Bearish tilt)."""
    return analyze_management_tone(symbol)

@tool
def compare_management_tone_qoq(symbol: str):
    """Compare management tone on the latest earnings call against the PREVIOUS quarter's call.

    The change is the signal, not the level: a team that always hedges is not a sell.
    Reports per-1,000-word shifts in cautious, confident, hedging, legal and
    obligation language. Use when assessing whether a story is improving or
    deteriorating, before endorsing or exiting a position.
    """
    from tools.earnings_nlp import compare_management_tone
    return compare_management_tone(symbol)

@tool
def analyze_crowded_trade(symbol: str):
    """Detect if a stock is over-owned by institutions ('Crowded Trade') and vulnerable to a flash crash."""
    return check_crowded_trade(symbol)

@tool
def check_portfolio_correlation(symbols: str=''):
    """
            Analyze correlation between portfolio holdings to detect concentration risk.
            If no symbols provided, uses the user's current portfolio symbols.
            High correlation (>0.7) indicates you're effectively betting on the same thing.

            Args:
                symbols: Comma-separated list of symbols (optional, defaults to portfolio)
            """
    if symbols and not _portfolio_alias(symbols):
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
    else:
        p_symbols, _, _ = _get_portfolio_data()
        symbol_list = p_symbols[:40]
    if len(symbol_list) < 2:
        return {'error': 'Need at least 2 symbols for correlation analysis'}
    return analyze_correlation(symbol_list)

@tool
def assess_marginal_trade_risk(symbol: str, allocation_pct: float = 5.0):
    """
            Estimate how adding a ticker changes the current portfolio's volatility.
            Use this before endorsing a new high-beta or same-sector buy.

            Args:
                symbol: Proposed ticker to add.
                allocation_pct: Proposed portfolio allocation percentage for the new/additional position.
            """
    p_symbols, p_allocs, _ = _get_portfolio_data()
    if not p_symbols or not p_allocs:
        return {"error": "Portfolio data unavailable for marginal risk analysis"}
    total_value = sum(float(v) for v in p_allocs.values() if v is not None)
    if total_value <= 0:
        return {"error": "Portfolio value unavailable for marginal risk analysis"}
    symbols = [s for s in p_symbols if s in p_allocs and float(p_allocs.get(s, 0)) > 0]
    weights = [float(p_allocs[s]) / total_value for s in symbols]
    return estimate_marginal_risk_contribution(
        symbols,
        weights,
        symbol,
        candidate_weight=allocation_pct,
    )

@tool
def simulate_portfolio_rebalancing(adjustments: str):
    """
            Simulate a portfolio rebalancing to prove/disprove if a change would improve returns.
            Use this to test 'what if' scenarios like: 'What if I sold 50% of NVDA and bought SPY?'

            Args:
                adjustments: Natural language adjustments (e.g., "Sell 50% NVDA, Buy SPY")
                             or JSON allocations (e.g., '{"NVDA": 25, "AAPL": 30, "SPY": 45}')
            """
    p_symbols, p_allocs, _ = _get_portfolio_data()
    current_holdings_json = json.dumps(p_allocs)
    return simulate_rebalancing(current_holdings_json, adjustments)

@tool
def analyze_factor_exposures(period: str = "1y"):
    """
            Measure the portfolio's REAL factor exposures by regression (Roadmap 4.2).

            PREFER THIS over the factor_analysis block in analyze_portfolio_exposure,
            which only counts holdings by style label. This reports a beta per factor
            (market, size, value, momentum, quality, rates, and currency) WITH its
            t-statistic, so an exposure that cannot be distinguished from zero says so
            instead of reading as a tilt. Also reports alpha, R-squared, and each
            factor's share of explained variance.

            Portfolio returns are measured in the user's base currency; the currency
            move enters as its own factor, so `fx` answers "how much of my variance is
            the dollar".

            Args:
                period: lookback, e.g. "1y" (default), "2y", "5y".
            """
    from tools.factor_exposures import estimate_factor_exposures
    p_symbols, p_allocs, _ = _get_portfolio_data()
    if not p_symbols:
        return {"error": "No holdings to analyze"}
    weights = [float(p_allocs.get(s, 0) or 0) for s in p_symbols] if isinstance(p_allocs, dict) else None
    return estimate_factor_exposures(p_symbols, weights, period=period)


@tool
def replay_historical_episode(episode: str = "all"):
    """
            Replay a REAL historical crash against the user's CURRENT weights, using
            the actual daily prices from that window (Advisor Roadmap 4.3).

            PREFER THIS over run_stress_test. This measures what these holdings did
            in that episode; run_stress_test multiplies beta by a hardcoded drop.

            Reports peak-to-trough, how long this portfolio took to regain its prior
            level, the worst positions, and which pair stopped diversifying. It also
            reports COVERAGE: holdings with no price history in that window are named
            and excluded, and a replay covering too little of the portfolio refuses
            rather than returning a flattering number.

            Args:
                episode: "gfc", "covid", "bear_2022", "dotcom", or "all" (default).
            """
    from tools.episode_replay import replay_all_episodes, replay_episode
    p_symbols, p_allocs, _ = _get_portfolio_data()
    if not p_symbols:
        return {"error": "No holdings to replay"}
    weights = [float(p_allocs.get(s, 0) or 0) for s in p_symbols] if isinstance(p_allocs, dict) else None
    if str(episode).strip().lower() in ("all", "", "none"):
        return replay_all_episodes(p_symbols, weights)
    return replay_episode(p_symbols, weights, episode=episode)


@tool
def run_stress_test(scenario: str):
    """
            Estimate portfolio impact under a named scenario using ASSUMED market
            drops (recession -35%, tech_crash -45%) multiplied by each holding's beta.

            The drop figures are AUTHORED CONSTANTS, not measurements — prefer
            replay_historical_episode, which uses the real daily paths from the
            actual episode. Use this only for a hypothetical with no historical
            analogue, and say plainly that the magnitude was assumed.

            Args:
                scenario: One of "recession", "rate_hike", "tech_crash", or "bull_market"
            """
    p_symbols, _, _ = _get_portfolio_data()
    symbols_str = ','.join(p_symbols[:10])
    # Roadmap 2.7: the basis marker now comes from simulate_scenario itself, so every
    # caller inherits it rather than only this wrapper.
    return simulate_scenario(symbols_str, scenario)

@tool
def analyze_reddit_sentiment(symbol: str):
    """Scans Reddit for 'MEME' potential and retail hype (YOLO, Moon, Squeeze)."""
    return get_reddit_sentiment(symbol)

@tool
def scan_options_chain(symbol: str):
    """Scans options for 'Gamma Squeeze' signals, High IV, and unusual volume."""
    return scan_unusual_activity(symbol)

@tool
def scan_technical_breakouts(symbols: str):
    """Scans a list of stocks (comma-separated) for RSI oversold/breakout setups."""
    return find_breakout_candidates(symbols)

@tool
def get_earnings_data(symbol: str):
    """Get next earnings date, consensus estimates, and revenue projections."""
    return _raw_get_earnings_calendar(symbol)

@tool
def get_insider_activity(symbol: str):
    """Check if executives/insiders are buying or selling stock — US AND Canadian.

    US listings: SEC Form 4 (true open-market buy/sell coding + cluster buys).
    Canadian listings (.TO/.V/.CN/.NE): the same analysis over SEDI-sourced
    filings, since Canadian issuers file on SEDI and have no Form 4 on EDGAR.
    Either way you get open-market buys/sells separated from grants, option
    exercises and issuer buybacks, with dollar values and cluster detection."""
    from tools.insider_data import get_detailed_insider_activity, is_canadian_listing
    from tools.sec_edgar import get_form4_activity
    from tools.tool_errors import is_unavailable

    # A Canadian suffix settles it offline — EDGAR holds no CIK and no Form 4 for
    # these, so asking wastes a round-trip and returns a "not a US filer" note
    # that reads to the model like "no insider data exists". That is how a TSX
    # name ended up with no insider evidence to reason over at all.
    if not is_canadian_listing(symbol):
        try:
            result = get_form4_activity(symbol)
        except Exception:
            result = None
        # Fall through only when EDGAR says "wrong venue" (not_an_sec_filer) or
        # could not answer. A network blip must NOT silently downgrade every US
        # ticker to the weaker source — that path returns `unavailable`, and
        # dropping to yfinance is the right call only because EDGAR is truly out.
        if result and not is_unavailable(result) and not result.get("not_an_sec_filer"):
            return result

    detailed = get_detailed_insider_activity(symbol)
    if detailed and not is_unavailable(detailed):
        return detailed
    return get_insider_trading(symbol)


@tool
def get_material_events(symbol: str):
    """SEC 8-K material corporate events for a stock — bankruptcy, restatement,
    auditor change, delisting notice, M&A, executive departures — with per-filing
    severity, straight from EDGAR."""
    from tools.sec_edgar import get_recent_8k
    return get_recent_8k(symbol)


@tool
def get_institutional_moves(manager: str = ""):
    """Quarterly 13F position changes (new buys, exits, adds, trims) for tracked
    long-horizon institutional managers (Berkshire, Duquesne, Baupost...). Pass a
    manager name for one manager's full diff, or empty for the cross-manager view."""
    from tools.sec_edgar import get_institutional_moves as _edgar_moves
    return _edgar_moves(manager or None)

@tool
def get_institutional_data(symbol: str):
    """Check ownership by major funds and institutions."""
    return get_institutional_ownership(symbol)

@tool
def check_sector_rotation():
    """Analyze which sectors (Tech, Energy, etc.) are heating up or cooling down."""
    return detect_sector_rotation()

@tool
def get_relative_strength(symbols: str):
    """Rank a list of symbols by performance vs SPY (Leaders vs Laggards)."""
    return rank_relative_strength(symbols)

@tool
def predict_surprise(symbol: str):
    """Predict probability of earnings beat based on historical beat rate."""
    return predict_earnings_surprise(symbol)

@tool
def find_ipos():
    """Find upcoming IPOs (New Listings) for the current month."""
    return get_upcoming_ipos()

@tool
def run_stock_deep_dive(symbol: str):
    """
            🚀 MACRO TOOL: Run a complete 360-degree analysis on a stock.
            Fetches: Price, Valuation, Technicals (RSI/Trends), Analyst Ratings, Insider Selling, and Earnings.
            USE THIS INSTEAD of calling 6 separate tools.
            """
    return macro_stock_deep_dive(symbol)

@tool
def assess_portfolio_risk():
    """
            🚀 MACRO TOOL: Run a complete Portfolio Risk Audit.
            Checks: Concentration, Correlation, Sharpe/Beta, Sectors, and Currency Exposure.
            USE THIS for 'Health Checks' or 'Risk Analysis'.
            """
    return macro_portfolio_risk()

@tool
def scan_intraday_movers():
    """
            🚀 REAL-TIME US SCANNER: Get the 'Pulse' of the US market Right Now.
            Finds: US stocks moving >3%, Unusual Volume, and RSI Breakouts.
            Use for: 'What's moving?', 'Market pulse', 'Volatility check' (US/NYSE/Nasdaq).
            For Canadian/TSX movers use scan_tsx_movers instead.
            """
    return tool_scan_intraday_movers()

@tool
def scan_tsx_movers():
    """
            🍁 REAL-TIME TSX SCANNER: Market-wide Canadian (Toronto Stock Exchange) movers.
            Finds: top TSX gainers, top losers, and large-cap most-active names, plus
            TSX Composite and USD/CAD context. Screens the ENTIRE exchange — verified
            live data, not news articles.
            Use for: 'TSX movers', 'what's moving in Canada', 'Canadian market pulse',
            'TSX gainers/losers today'.
            """
    return tool_scan_tsx_movers()

@tool
def get_market_pulse_data():
    """
            🛰️ MARKET PULSE: Get the current market regime and daily briefing.
            Returns: Regime (CRISIS/FEAR/CAUTIOUS/NEUTRAL/BULLISH/EUPHORIA),
            Fear & Greed score, VIX, SPY drawdown, portfolio alerts, and top opportunities.
            Use for: 'What's the market regime?', 'Should I be worried?', 'Market pulse',
            'Are we in a crash?', 'Is now a good time to buy?'
            """
    return get_market_regime()

@tool
def get_market_regime_data(days: int = 30):
    """
            📊 REGIME HISTORY: Get the market regime trajectory over the last N days.
            Shows how the regime has shifted over time and how long we've been in the current state.
            Use for: 'How long have we been in fear?', 'Show me the regime history',
            'When was the last crisis?', 'Market regime over the past month'
            """
    return get_regime_history(days)

@tool
def generate_future_forecast(query: str):
    """
            🔮 SCENARIO ENGINE: Generates a forward-looking scenario using live macro signals plus historical analogues.
            Use this ONLY when the user asks for a market forecast, scenario, or long-term structural outlook.
            Output must be treated as scenario analysis, not a calibrated prediction.
            """
    from tools.fmp_api import get_economic_calendar
    from tools.fred_api import get_fed_funds_rate, get_inflation_data, get_systemic_risk_indicators, get_treasury_yields
    from tools.market_data import get_stock_data
    from tools.market_mechanics import detect_sector_rotation
    from tools.predictive import match_historical_regime

    _dspy_available = globals().get("DSPY_AVAILABLE", False)

    def _clean_note(result, fallback_message):
        if not isinstance(result, dict):
            return fallback_message
        if result.get("note"):
            return str(result["note"])
        if result.get("error"):
            return str(result["error"])
        return fallback_message

    def _evidence_quality_label(fresh_signal_count: int, used_fallback: bool) -> str:
        if fresh_signal_count >= 4 and not used_fallback:
            return "High"
        if fresh_signal_count >= 2:
            return "Medium"
        return "Low"

    send_status('📡 Fetching Advanced Signals (Yield Curve, Liquidity, VIX)...')
    current_inflation = 3.2
    current_rate = 5.25
    current_trend = 'bull'
    current_pe = 24.0
    risk_data = {}
    yield_data = {}
    evidence_notes = []
    fresh_signal_count = 0
    used_fallback = False

    try:
        cpi_data = get_inflation_data()
        if 'headline_inflation' in cpi_data:
            current_inflation = float(cpi_data['headline_inflation'].replace('%', ''))
            if cpi_data.get("note") or cpi_data.get("error"):
                used_fallback = True
                evidence_notes.append(_clean_note(cpi_data, "Inflation data relied on fallback assumptions."))
            else:
                fresh_signal_count += 1
        else:
            evidence_notes.append("Inflation data unavailable; using baseline assumption.")

        rate_data = get_fed_funds_rate()
        if 'current_rate' in rate_data:
            current_rate = float(rate_data['current_rate'].replace('%', ''))
            if rate_data.get("note") or rate_data.get("error"):
                used_fallback = True
                evidence_notes.append(_clean_note(rate_data, "Fed rate data relied on fallback assumptions."))
            else:
                fresh_signal_count += 1
        else:
            evidence_notes.append("Fed funds data unavailable; using baseline assumption.")

        risk_data = get_systemic_risk_indicators()
        if isinstance(risk_data, dict) and not risk_data.get("error"):
            fresh_signal_count += 1
        elif isinstance(risk_data, dict):
            evidence_notes.append(_clean_note(risk_data, "Systemic risk data unavailable."))

        yield_data = get_treasury_yields()
        if isinstance(yield_data, dict):
            if yield_data.get("note"):
                used_fallback = True
                evidence_notes.append(str(yield_data["note"]))
            elif not yield_data.get("error"):
                fresh_signal_count += 1
            else:
                evidence_notes.append(_clean_note(yield_data, "Yield curve data unavailable."))
    except Exception:
        pass

    vix_val = 'N/A'
    try:
        spy_data = get_stock_data('SPY')
        if 'error' not in spy_data:
            if spy_data.get('pe_ratio') != 'N/A':
                current_pe = float(spy_data['pe_ratio'])
            trend_str = spy_data.get('recent_trend', '0%')
            if 'N/A' in trend_str:
                trend_str = '0%'
            trend_val = float(trend_str.replace('%', '').split()[0])
            if trend_val > 5:
                current_trend = 'strong bull'
            elif trend_val > 0:
                current_trend = 'bull'
            elif trend_val > -10:
                current_trend = 'correction'
            else:
                current_trend = 'bear'
            fresh_signal_count += 1
        else:
            evidence_notes.append("SPY valuation/trend data unavailable; using baseline assumptions.")
        vix_data = get_stock_data('^VIX')
        if 'error' not in vix_data:
            vix_val = vix_data.get('current_price', 'N/A').replace('$', '')
        else:
            evidence_notes.append("VIX data unavailable.")
    except Exception:
        pass

    evidence_quality = _evidence_quality_label(fresh_signal_count, used_fallback)
    send_status(f'🕰️ Scenario: Matching historical analogues (CPI: {current_inflation}%, Rates: {current_rate}%, PE: {current_pe})')
    matches = match_historical_regime(current_inflation, current_rate, current_trend, current_pe)
    best_match = matches['matched_regime']
    similarity_score = matches.get('similarity_score', matches.get('match_score', 0))
    send_status('📡 Fetching Forward-Looking Signals (Sector Rotation & Calendar)...')
    rotation_data = detect_sector_rotation()
    calendar_events = get_economic_calendar()
    cal_str = 'None available.'
    if isinstance(calendar_events, list) and len(calendar_events) > 0:
        try:
            cal_str = '\n'.join([
                f"- {e.get('date', 'Unknown Date')}: {e.get('event', 'Unknown Event')} (Est: {e.get('estimate', 'N/A')})"
                for e in calendar_events[:5] if isinstance(e, dict)
            ])
        except Exception:
            cal_str = 'Data format error in calendar.'
    _, _, p_ctx = _get_portfolio_data()
    current_ctx_str = f"\nMACRO DATA:\n- Inflation (CPI): {current_inflation}%\n- Fed Rate: {current_rate}%\n- Treasury Yield Curve (10Y-2Y): {yield_data.get('yield_spread', 'N/A')} ({yield_data.get('curve_status', 'Unknown')})\n\nMARKET VITALS:\n- Trend: {current_trend.upper()}\n- S&P 500 P/E: {current_pe}\n- VIX (Fear Index): {vix_val}\n\nSYSTEMIC RISK:\n- Credit Spreads: {risk_data.get('credit_spread', 'N/A')} ({risk_data.get('crash_risk', 'Unknown')})\n- M2 Money Supply: {risk_data.get('liquidity_status', 'Unknown')} ({risk_data.get('m2_growth_yoy', 'N/A')})\n\nFORWARD-LOOKING SIGNALS:\n- Market Leaders (Sector Rotation): {(', '.join(rotation_data.get('leading_sectors', [])) if rotation_data else 'Unknown')}\n- Market Laggards: {(', '.join(rotation_data.get('lagging_sectors', [])) if rotation_data else 'Unknown')}\n- Upcoming Economic Events:\n{cal_str}\n\nUSER PORTFOLIO:\n{p_ctx}\n\nUSER REQUEST: {query}\n"
    # 2.7: the two scenario lines below are AUTHORED constants. The block says so
    # in the prompt itself, because the model cannot tell a hand-typed
    # "Crash (-50%)" from a replayed one, and the regime name asserts history.
    historical_match_str = (
        f"\nRegime: {best_match}\n"
        f"Similarity Score: {similarity_score}% (COMPUTED from the live macro inputs)\n"
        f"Description: {matches['description']}\n"
        f"Authored scenario (3mo): {matches['authored_scenario_3mo']}\n"
        f"Authored scenario (1yr): {matches['authored_scenario_1yr']}\n"
        f"Key Risks: {matches['key_risks']}\n"
        f"Method Note: {matches.get('methodology_note', 'Heuristic analogue matching.')}\n"
        f"BASIS: the two 'Authored scenario' lines and the Key Risks were typed into "
        f"the codebase by hand — they are NOT measured, NOT backtested, and NOT what "
        f"this regime provably did. Only the similarity score is computed. If you use "
        f"them, attribute them in the same sentence as an assumed analogue scenario; "
        f"never present them as what history shows. A measured answer to 'what would a "
        f"drawdown like this do to my holdings' comes from replay_historical_episode.\n"
    )

    evidence_block = [
        "### 📡 Evidence Quality",
        f"- Quality: {evidence_quality}",
        f"- Fresh live signal count: {fresh_signal_count}",
        f"- Historical analogue similarity: {similarity_score}%",
        f"- Method: {matches.get('methodology_note', 'Heuristic analogue matching.')}",
        # 2.7: the evidence header is where a reader decides how much to trust the
        # rest, so the authored half is declared there rather than in a footer.
        f"- Analogue outcome basis: {matches.get('basis', 'authored constant')} "
        f"(the scenario text is hand-written; only the similarity score is computed)",
    ]
    if evidence_notes:
        evidence_block.append(f"- Caveats: {'; '.join(dict.fromkeys(evidence_notes))}")
    else:
        evidence_block.append("- Caveats: None")
    evidence_prefix = "\n".join(evidence_block) + "\n\n"

    def _authored_scenario_body() -> str:
        """The two non-LLM paths render the authored strings straight to the user.

        2.7: they are labelled AT the figure rather than in a footnote, because a
        heading reading "3-Month Scenario" over "Crash (-50%)" is exactly the
        presentation that makes a hand-typed constant read as a finding.
        """
        return (
            f"**Closest Historical Analogue:** {best_match} "
            f"(similarity {similarity_score}% — computed)\n\n"
            f"**3-Month Scenario — authored analogue, not measured:** "
            f"{matches['authored_scenario_3mo']}\n\n"
            f"**1-Year Scenario — authored analogue, not measured:** "
            f"{matches['authored_scenario_1yr']}\n\n"
            f"**Key Risks (authored):** {matches['key_risks']}\n\n"
            "The scenario lines above are written into this tool by hand — they "
            "describe what one author expects to follow this regime, and are not "
            "measured from price history. For a measured answer, "
            "`replay_historical_episode` replays your actual holdings through the "
            "real daily paths of the GFC, COVID, 2022 and the dot-com bust.\n\n"
        )

    if evidence_quality == "Low" or not _dspy_available:
        return (
            f"{evidence_prefix}"
            f"### 🔮 Forward Scenario Analysis\n"
            f"{_authored_scenario_body()}"
            "This is a scenario assessment built from limited or partially fallback data. "
            "Treat it as directional context, not a forecast or recommendation."
        )

    try:
        from agent.modules import PredictionAnalyst
        predictor = PredictionAnalyst()
        send_status('🧠 Deep Reasoning: Generating Forward Scenario...')
        scenario_text = predictor.forward(current_context=current_ctx_str, historical_match=historical_match_str)
        return evidence_prefix + scenario_text
    except Exception as e:
        send_status(f'⚠️ Deep Reasoning Failed: {str(e)[:50]}... Falling back to heuristic.', degraded=True)
        return (
            f"{evidence_prefix}"
            f"### 🔮 Forward Scenario Analysis (Heuristic Fallback)\n"
            f"{_authored_scenario_body()}"
            "This heuristic scenario is built using historical analogues since the deep reasoning engine encountered an issue."
        )

@tool
def get_portfolio_snapshot():
    """Get current portfolio holdings with LIVE values, P&L, and daily performance."""
    global PORTFOLIO_CACHE
    if PORTFOLIO_CACHE['data'] and time.time() - PORTFOLIO_CACHE['timestamp'] < 900:
        return PORTFOLIO_CACHE['data']
    summary = get_portfolio_summary()
    if not (isinstance(summary, dict) and "error" in summary):
        PORTFOLIO_CACHE['data'] = summary
        PORTFOLIO_CACHE['timestamp'] = time.time()
    return summary

@tool
def get_stock_quote(symbol: str):
    """Get REAL-TIME price, change %, and volume for a stock (e.g., 'AMD')."""
    # get_realtime_quote is decorated with @tool in this module, so it is a StructuredTool.
    # We call its underlying function to avoid 'StructuredTool' object is not callable.
    func = getattr(get_realtime_quote, 'func', get_realtime_quote)
    if func is get_realtime_quote and hasattr(get_realtime_quote, 'invoke'):
        return get_realtime_quote.invoke({"symbol": symbol})
    return func(symbol)

@tool
def get_stock_news(symbol: str):
    """Get recent news sentiment and headlines explaining price moves."""
    return get_news_sentiment(symbol)

@tool
def get_valuation_metrics(symbol: str):
    """Get PE, PEG, Forward PE, Margins, and Analyst Targets for valuation."""
    # Import the actual function, not the tool
    from tools.market_data import get_fundamentals_detailed as get_fundamentals_func
    return get_fundamentals_func(symbol)

@tool
def get_dividend_data(symbol: str):
    """Get dividend yield, payout ratio, safety score and growth history."""
    if hasattr(get_dividend_analysis, 'invoke'):
        return get_dividend_analysis.invoke(symbol)
    return get_dividend_analysis(symbol)

@tool
def get_economic_calendar_tool():
    """Get schedule of major market moving events (CPI, Fed, Jobs)."""
    if hasattr(get_economic_calendar, 'invoke'):
        return get_economic_calendar.invoke({})
    return get_economic_calendar()

@tool
def screen_stocks(criteria: str='All'):
    """Scan market for opportunities. Pass 'All' for broad market scan, or a sector name like 'Tech', 'Energy', 'Healthcare'."""
    return scan_sector_opportunities(criteria)

@tool
def get_historical_performance(symbol: str):
    """Get 1Y, 3Y, 5Y annualized returns (CAGR) and total return."""
    return fetch_historical_perf(symbol)

@tool
def get_competitors(symbol: str):
    """Get a list of main competitors and how they compare (PE, Market Cap)."""
    return compare_assets([symbol])

@tool
def get_etf_holdings_data(symbol: str):
    """See what companies are inside an ETF (e.g. Top 10 holdings of QQQ)."""
    return get_etf_holdings(symbol)

@tool
def check_portfolio_earnings():
    """Check upcoming earnings dates for ALL stocks in your portfolio."""
    from tools.portfolio_csv import get_tradeable_symbols
    symbol_list = get_tradeable_symbols()
    if not symbol_list:
        return 'Portfolio empty or includes no tradeable positions.'
    report = []
    from tools.market_data import get_stock_data
    for sym in symbol_list:
        data = get_stock_data(sym)
        date = data.get('earnings_date', 'N/A')
        if date != 'N/A':
            report.append(f'{sym}: {date}')
    return '\n'.join(report) if report else 'No upcoming earnings found for portfolio positions.'

@tool
def structure_trade_setup(symbol: str):
    """
            📐 TRADE ARCHITECT: Get a professional Entry/Stop/Target plan.
            Calculates exact price levels using ATR volatility and Risk:Reward ratios.
            Use for: 'Where should I buy?', 'Stop loss for AAPL?', 'Trade plan'.
            """
    return get_trade_setup(symbol)

@tool
def get_hypothetical_portfolio(persona: str):
    """
            👤 PORTFOLIO GENERATOR: Get a proxy portfolio for a specific style.
            Use when user has NO connected portfolio but wants specific advice.
            Styles: 'Aggressive Growth', 'Conservative', 'Dividend Income', 'Balanced'.
            """
    return get_portfolio_proxy(persona)

@tool
def model_options_strategy(symbol: str, strategy_type: str='covered_call'):
    """
            🛡️ OPTIONS STRATEGY: Architect Income or Protection.
            Strategies: 'covered_call' (Generate Yield), 'protective_put' (Insurance), 'collar' (Low-Cost Protection).
            Use for: 'How to generate income on AAPL?', 'Protect my NVDA', 'Zero cost collar'.
            """
    return get_options_strat(symbol, strategy_type)

@tool
def construct_bond_ladder(amount: float=100000, currency: str='CAD'):
    """
            🪜 BOND LADDER: Construct a 5-Year GIC/Bond Ladder for safe income.
            Use when user asks for 'Safe investments', 'GICs', 'Guaranteed income', or 'Retirement bucket'.
            """
    return get_bond_ladder(amount, investment_type='GIC' if currency == 'CAD' else 'Treasury', currency=currency)

@tool
def run_technical_analysis(symbol: str):
    """
            Run comprehensive technical analysis (RSI, MACD, Bollinger Bands, Moving Averages).
            Use this for DEEP DIVE on a single stock's accumulation/distribution.
            """
    from tools.technicals import get_comprehensive_technicals
    return get_comprehensive_technicals(symbol)

@tool
def get_seasonality_data(symbol: str):
    """
            Analyzes historical monthly return patterns (seasonality) for a stock.
            Returns average return and win rate per month to identify best/worst times to buy.
            """
    from tools.seasonality import analyze_seasonality

    return analyze_seasonality(symbol)

# ── INTERNATIONAL MARKET TOOLS ───────────────────────────────────────────────

@tool
def get_tsx_stock_quote(symbol: str):
    """Get real-time quote, market cap, PE, and dividends for a Canadian TSX stock (e.g. RY.TO, SHOP.TO, VET.TO)."""
    return get_canadian_quote(symbol)

@tool
def get_tsx_stock_analyst(symbol: str):
    """Get analyst price targets and consensus for a Canadian TSX stock."""
    return get_canadian_analyst_estimates(symbol)

@tool
def get_asx_stock_quote(symbol: str):
    """Get real-time quote, market cap, PE, and dividends for an Australian ASX stock (e.g. AX1.AX, BHP.AX, CBA.AX)."""
    return get_australian_quote(symbol)

@tool
def get_asx_stock_analyst(symbol: str):
    """Get analyst price targets and consensus for an Australian ASX stock."""
    return get_australian_analyst_estimates(symbol)

@tool
def get_eu_stock_quote(symbol: str):
    """Get real-time quote, market cap, PE, and dividends for a European stock (e.g. SHEL.L, SAP.DE, MC.PA, ASML.AS)."""
    return get_european_quote(symbol)

@tool
def get_eu_stock_analyst(symbol: str):
    """Get analyst price targets and consensus for a European stock (LSE, XETRA, Euronext)."""
    return get_european_analyst_estimates(symbol)

@tool
def get_global_indices():
    """
            Get a snapshot of major global market INDEX levels: S&P 500, Nasdaq,
            TSX (Canada), FTSE (UK), Nikkei (Japan), and Bitcoin. Use for "how are
            markets doing globally" / "overnight markets" questions. For economic
            indicators (Fed rate, inflation, GDP) use get_macro_overview instead.
            """
    from tools.macro_data import get_global_market_snapshot

    return get_global_market_snapshot()

def _risk_tolerance_to_profile(default: str = "balanced") -> str:
    """Map the stored profile risk_tolerance onto a run_monte_carlo preset name.

    The profile vocabulary is conservative/moderate/aggressive; 'moderate' maps to
    the 'balanced' preset. Anything unset or unrecognised falls back to balanced.
    """
    try:
        from tools.memory import load_memory
        rt = (load_memory().get("user_profile", {}).get("risk_tolerance") or "").lower()
    except Exception:
        return default
    return {
        "conservative": "conservative",
        "moderate": "balanced",
        "balanced": "balanced",
        "aggressive": "aggressive",
    }.get(rt, default)


@tool
def project_retirement_goal(current_value: float=0, monthly_contribution: float=0, years: int=0, risk_profile: str=""):
    """
            Run a Monte Carlo simulation to test if the user can hit their wealth goal.
            Args:
                current_value: Optional. If 0, uses the current portfolio value in
                    the profile's base currency.
                monthly_contribution: Recurring monthly saving (default 0).
                years: Horizon. If 0, uses the stored goal's horizon, else 10.
                risk_profile: 'conservative' | 'balanced' | 'aggressive'. If blank,
                    derived from the profile's risk_tolerance.
            """
    from tools.memory import get_financial_goal
    from tools.monte_carlo import run_monte_carlo

    if current_value == 0:
        try:
            from tools.portfolio_csv import get_portfolio_summary
            summary = get_portfolio_summary()
            current_value = summary.get('total_value_base', 100000)
        except Exception:
            current_value = 100000

    goal = get_financial_goal() or {}
    years = int(years or goal.get('horizon_years') or 10)
    risk_profile = risk_profile or _risk_tolerance_to_profile()

    return run_monte_carlo(
        current_value,
        0,
        years,
        risk_profile=risk_profile,
        monthly_contribution=monthly_contribution,
        goal_target=goal.get('target_low'),
    )

@tool
def check_fx_impact(base_currency: str='CAD'):
    """
            Analyze the currency risk (USD vs CAD) in the user's specific portfolio.
            Quantifies the loss/gain if CAD strengthens or weakens.
            Use this when user asks about "Currency Risk", "USD exposure", or "Exchange rate impact".
            """
    from tools.fx_utils import analyze_my_portfolio_fx
    return analyze_my_portfolio_fx(base_currency)

@tool
def check_risk_metrics(symbols: str=''):
    """
            Calculate advanced risk metrics: Sharpe Ratio, Sortino, Beta, Max Drawdown, and Value-at-Risk (VaR).
            Use this when user asks about "Risk-adjusted returns", "Sharpe", "Beta", or "Is my portfolio safe?".
            Args:
                symbols: Comma-separated list of symbols (optional, defaults to portfolio).
            """
    from tools.portfolio_analytics import calculate_portfolio_metrics, calculate_var
    symbol_list, meta = _resolve_risk_symbols(symbols)
    if not symbol_list:
        return {'error': f"No symbols found for profile '{meta['profile']}'.", 'profile': meta['profile']}
    metrics = calculate_portfolio_metrics(symbol_list)
    var = calculate_var(symbol_list)
    result = {
        'profile': meta['profile'],
        'symbols': symbol_list,
        'scope': meta['scope'],
        'risk_metrics': metrics.get('metrics', metrics),
        'value_at_risk': var.get('value_at_risk', var),
        'interpretation': metrics.get('interpretation', {}),
    }
    if meta['note']:
        result['reconciliation_note'] = meta['note']
    return result

@tool
def check_esg_scores(symbols: str=''):
    """
            Analyze ESG (Environmental, Social, Governance) scores and flags.
            Use for: 'Is XESG greenwashed?', 'ESG score for AAPL', 'Ethical screening'.
            """
    from tools.esg_analytics import check_esg_scores
    if symbols and not _portfolio_alias(symbols):
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
    else:
        try:
            from tools.portfolio_csv import get_tradeable_symbols
            symbol_list = get_tradeable_symbols()
        except Exception:
            return {'error': 'Could not load portfolio symbols.'}
    return check_esg_scores(symbol_list)

@tool
def analyze_mutual_funds(symbols: str=''):
    """
            Analyze Mutual Funds and Pension holdings (Fees, Expense Ratios, Manager Performance).
            Use for: 'Analyze my pension', 'Fees for a fund', 'Compare index funds'.
            """
    from tools.fund_analytics import analyze_mutual_funds
    if symbols and not _portfolio_alias(symbols):
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
    else:
        try:
            from tools.portfolio_csv import get_tradeable_symbols
            symbol_list = get_tradeable_symbols()
        except Exception:
            return {'error': 'Could not load portfolio symbols.'}
    return analyze_mutual_funds(symbol_list)

@tool
def project_portfolio_income():
    """
            Project detailed portfolio income (Dividends & Distributions).
            Use for: 'How much income will I have?', 'Project my dividends', 'Yield on cost'.
            """
    from tools.income_analytics import project_portfolio_income
    from tools.portfolio_csv import load_portfolio

    try:
        port = load_portfolio()
        if isinstance(port, dict) and "error" in port:
            return port
        symbols = [p["symbol"] for p in port if isinstance(p, dict) and p.get("symbol")]
        amounts = [
            float(p.get("shares", p.get("quantity", 0)) or 0)
            for p in port
            if isinstance(p, dict) and p.get("symbol")
        ]
        return project_portfolio_income(symbols, amounts)
    except Exception as e:
        return {'error': f'Could not load portfolio for income projection: {e}'}

@tool
def backtest_strategy(strategy_type: str, symbols: str, period: str='2y', details: str=''):
    """
            Unified Backtester.
            Args:
                strategy_type: 'rsi' (Technical) OR 'dca'/'lump_sum' (Portfolio)
                symbols: 'AAPL' or 'AAPL, MSFT'
                period: '2y', '5y'
                details: For Portfolio, provide weights '0.6, 0.4'. For RSI, '30, 70' (buy/sell).
            """
    from tools.backtesting import backtest_strategy
    try:
        sym_list = [s.strip() for s in symbols.split(',')]
        params = {}
        if strategy_type in ['dca', 'lump_sum'] and details:
            params['allocations'] = [float(x) for x in details.split(',')]
        if strategy_type == 'rsi' and details:
            parts = details.split(',')
            if len(parts) == 2:
                params['buy_threshold'] = int(parts[0])
                params['sell_threshold'] = int(parts[1])
        return backtest_strategy(strategy_type, sym_list, period, params)
    except Exception as e:
        return {'error': f'Backtest input error: {e}'}

@tool
def check_portfolio_allocation():
    """
            Analyze TRUE portfolio sector exposure (decomposing ETFs/Funds).
            Use for: 'Am I overweight Tech?', 'Sector breakdown', 'Diversification check'.
            """
    from tools.portfolio_csv import get_portfolio_summary
    from tools.sector_analysis import check_portfolio_allocation
    try:
        summary = get_portfolio_summary()
        if isinstance(summary, dict) and "error" in summary:
            return summary

        # Unvalued holdings are dropped rather than passed at 0: they weigh nothing in
        # a sector breakdown either way, and sending them through would spend a
        # classification lookup on a position we cannot size.
        holdings = [
            p for p in summary.get("holdings", [])
            if isinstance(p, dict) and p.get("symbol") and not p.get("is_unvalued")
        ]
        symbols = [p["symbol"] for p in holdings]
        amounts = [float(p.get("value_usd", 0) or 0) for p in holdings]
        return check_portfolio_allocation(symbols, amounts)
    except Exception as e:
        return {'error': f'Could not load portfolio for sector analysis: {e}'}

@tool
def read_url(url: str):
    """Read content from a URL (news article, blog, earnings report)."""
    return read_web_page(url)

@tool
def run_health_check():
    """
            Runs a full diagnostic of all available tools. Returns status of each tool (OK / ERROR / EXCEPTION),
            response times, and agent instructions on which tools are broken and should NOT be cited.
            ALWAYS call this tool first when the user asks for a health check, system status, or 'what tools do you have'.
            """
    from tools.health_check import run_tool_health_check

    result = run_tool_health_check()

    # Return condensed summary to prevent overwhelming the LLM with massive JSON
    # Full report is still logged and available in the backend
    broken_tools_with_errors = []
    for r in result.get("tool_results", []):
        if r.get("status") != "✅ OK":
            tool_name = r["tool"]
            error_msg = r.get("error", "Unknown error")
            error_type = r.get("error_type", "Unknown")
            error_location = r.get("error_location")

            detail = {
                "tool": tool_name,
                "error": error_msg,
                "error_type": error_type
            }
            if error_location:
                detail["error_location"] = error_location

            broken_tools_with_errors.append(detail)

    broken_tools = [r["tool"] for r in result.get("tool_results", []) if r.get("status") != "✅ OK"]

    return {
        "overall_status": result["health_summary"]["overall_status"],
        "operational_tools": result["health_summary"]["operational"],
        "failed_tools": result["health_summary"]["failed"],
        "total_checked": result["health_summary"]["total_checked"],
        "broken_tools": broken_tools[:10] if len(broken_tools) > 10 else broken_tools,
        "broken_tools_details": broken_tools_with_errors[:10] if len(broken_tools_with_errors) > 10 else broken_tools_with_errors,
        "missing_prerequisites": result["health_summary"].get("missing_prerequisites", []),
        "optional_not_configured": result["health_summary"].get("optional_not_configured", {}),
        "summary": f"{result['health_summary']['operational']}/{result['health_summary']['total_checked']} tools operational. " +
                   (f"Broken: {', '.join(broken_tools[:5])}" if broken_tools else "All systems operational."),
        "note": "Full detailed report with stack traces available in backend logs. This is a condensed summary for agent consumption. "
                "'optional_not_configured' lists optional data sources the user MAY enable — these are NOT missing credentials or errors."
    }

@tool
def get_my_portfolio():
    """Get your portfolio holdings, current values, and gains/losses from your CSV file."""
    return get_portfolio_summary()

@tool
def verify_portfolio_holdings(symbols: str=''):
    """
        Verify whether tickers are actually held in the user's current portfolio.
        Use before recommending trims/sells/rebalances and whenever the user disputes holdings.

        Args:
            symbols: Optional comma-separated tickers to verify. Leave blank to return all verified holdings.
    """
    from tools.portfolio_csv import get_portfolio_decision_context

    symbol_list = (
        [s.strip().upper() for s in symbols.split(',') if s.strip()]
        if symbols and not _portfolio_alias(symbols)
        else None
    )
    context = get_portfolio_decision_context(symbols=symbol_list)
    if context.get("error"):
        return context

    if symbol_list:
        return {
            "as_of": context.get("as_of"),
            "is_stale": context.get("is_stale"),
            "sync_errors": context.get("sync_errors", []),
            "total_value_cad": context.get("total_value_cad"),
            "total_value_usd": context.get("total_value_usd"),
            "requested_symbols": context.get("requested_symbols", []),
            "owned_symbols": context.get("owned_symbols", []),
            "rule": "Only requested_symbols with owned=True may be treated as current holdings.",
        }

    return {
        "as_of": context.get("as_of"),
        "is_stale": context.get("is_stale"),
        "sync_errors": context.get("sync_errors", []),
        "total_value_cad": context.get("total_value_cad"),
        "total_value_usd": context.get("total_value_usd"),
        "owned_symbols": context.get("owned_symbols", []),
        "holdings": context.get("holdings", []),
        "rule": "Do not recommend trimming/selling any ticker absent from owned_symbols.",
    }


def _parse_metric_number(value, default: float = 0.0) -> float:
    """Parse currency, percent, and numeric strings from portfolio tool output."""
    import re

    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in {"", "-", "."}:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def _portfolio_alias(symbols: str) -> bool:
    return symbols.strip().upper() in ['PORTFOLIO', 'MY PORTFOLIO', 'ALL', 'MY STOCKS', 'RISK']


def _resolve_risk_symbols(symbols: str):
    """Resolve the symbol set for portfolio risk tools, scoped to the active profile.

    Returns (symbol_list, meta). ``meta`` records the profile the holdings were
    resolved under and any reconciliation notes. Defaulting an empty request to
    the tradeable portfolio and validating supplied tickers against the user's
    verified holdings prevents risk metrics from being computed on assets the
    user does not actually own — the failure mode behind the cross-profile and
    fabricated-symbol risk reports (one user's holdings, or invented tickers,
    silently driving another's risk numbers).
    """
    from tools.user_profile import get_active_profile
    profile = get_active_profile()

    owned_set: set[str] = set()
    tradeable: list[str] = []
    try:
        from tools.portfolio_csv import get_portfolio_decision_context, get_tradeable_symbols
        ctx = get_portfolio_decision_context()
        owned_set = {str(s).upper() for s in (ctx.get('owned_symbols') or [])}
        tradeable = [str(s).upper() for s in get_tradeable_symbols()]
    except Exception:
        pass

    raw = (symbols or '').strip()
    meta = {'profile': profile, 'scope': 'portfolio', 'not_held': [], 'note': None}

    if not raw or _portfolio_alias(raw):
        if not tradeable:
            meta['note'] = f"No tradeable holdings found for profile '{profile}'."
        return tradeable, meta

    supplied = [s.strip().upper() for s in raw.split(',') if s.strip()]
    if owned_set and supplied:
        not_held = [s for s in supplied if s not in owned_set]
        meta['not_held'] = not_held
        if not_held and len(not_held) == len(supplied):
            # None of the requested tickers are held: almost certainly the wrong
            # holdings set for a "my portfolio" risk question (a leaked profile or
            # fabricated symbols). Compute on the real portfolio and say so.
            meta['note'] = (
                f"None of the requested symbols ({', '.join(supplied)}) are in "
                f"profile '{profile}'s verified holdings; computed on the actual "
                f"portfolio instead."
            )
            return tradeable, meta
        if not_held:
            meta['note'] = (
                f"Requested symbols not in '{profile}'s verified holdings "
                f"(included as requested): {', '.join(not_held)}"
            )
    meta['scope'] = 'explicit'
    return supplied, meta


def _aggregate_symbol_weights(holdings, *, allowed_symbols=None, denominator=None):
    values = {}
    order = []
    allowed = {s.upper() for s in allowed_symbols} if allowed_symbols is not None else None

    for holding in holdings or []:
        if not isinstance(holding, dict):
            continue
        symbol = str(holding.get('symbol', '')).upper().strip()
        if not symbol or (allowed is not None and symbol not in allowed):
            continue
        value_usd = _parse_metric_number(holding.get('value_usd'), default=0.0)
        if value_usd <= 0:
            continue
        if symbol not in values:
            order.append(symbol)
            values[symbol] = 0.0
        values[symbol] += value_usd

    total = denominator if denominator and denominator > 0 else sum(values.values())
    symbols = [symbol for symbol in order if values.get(symbol, 0.0) > 0]
    weights = [values[symbol] / total for symbol in symbols] if total > 0 else None
    return symbols, weights, sum(values.values())


def _format_loss_dollars(value) -> str:
    if value in (None, "", "N/A"):
        return "Data Unavailable"
    text = str(value).strip()
    if text.startswith("-"):
        return text
    if text.startswith("$"):
        return f"-{text}"
    return f"-${text}"


def _return_assessment(value) -> str:
    pct = _parse_metric_number(value, default=None)
    if pct is None:
        return "Data Unavailable"
    if pct >= 20:
        return "High trailing return; verify sustainability"
    if pct >= 6:
        return "Reasonable"
    if pct >= 0:
        return "Low positive return"
    return "Negative trailing return"


def _build_canonical_metrics_table(metrics, var, corr, costs, investment_amount: float):
    m = metrics.get('metrics', {}) if isinstance(metrics, dict) else {}
    interp = metrics.get('interpretation', {}) if isinstance(metrics, dict) else {}
    v_data = var.get('value_at_risk', {}) if isinstance(var, dict) else {}

    expense_ratio = costs.get('weighted_expense_ratio') if isinstance(costs, dict) else None
    expected_yield = costs.get('expected_yield') if isinstance(costs, dict) else None
    expense_display = (
        f"{expense_ratio * 100:.2f}%"
        if isinstance(expense_ratio, (int, float))
        else "Data Unavailable"
    )
    dividend_display = "Data Unavailable"
    if isinstance(expected_yield, (int, float)) and investment_amount:
        dividend_display = f"${expected_yield * investment_amount:,.0f}/yr ({expected_yield * 100:.2f}% yield)"

    return [
        "<!-- CANONICAL_METRICS_START -->",
        "#### Key Metrics (Canonical Calculated Values)",
        "| Metric | Your Portfolio | Assessment |",
        "| :--- | :--- | :--- |",
        f"| Expected Return | {m.get('annual_return', 'Data Unavailable')} annualized | {_return_assessment(m.get('annual_return'))} |",
        f"| Volatility | {m.get('annual_volatility', 'Data Unavailable')} | {interp.get('volatility', 'Data Unavailable')} |",
        f"| Beta | {m.get('beta', 'Data Unavailable')} | {interp.get('beta', 'Data Unavailable')} |",
        f"| Max Drawdown | {m.get('max_drawdown', 'Data Unavailable')} | Manageable if aligned with your risk tolerance |",
        f"| Sharpe Ratio | {m.get('sharpe_ratio', 'Data Unavailable')} | {interp.get('sharpe', 'Data Unavailable')} |",
        f"| Avg Correlation | {corr.get('average_correlation', 'Data Unavailable') if isinstance(corr, dict) else 'Data Unavailable'} | {corr.get('diversification_quality', 'Data Unavailable') if isinstance(corr, dict) else 'Data Unavailable'} |",
        f"| Expense Ratio | {expense_display} | {costs.get('fee_rating', 'Data Unavailable') if isinstance(costs, dict) else 'Data Unavailable'} |",
        f"| Dividend Income | {dividend_display} | Estimated from weighted dividend yield |",
        f"| Daily VaR (95%) | {_format_loss_dollars(v_data.get('daily_var_dollars'))} | Worst expected single-day loss |",
        "<!-- CANONICAL_METRICS_END -->",
    ]


@tool
def get_portfolio_risk_metrics(symbols: str, investment: float=1000000):
    """
        Analyze a portfolio for Institutional Risk Metrics (Sharpe, VaR, Beta).
        If user says "analyze MY PORTFOLIO", pass "PORTFOLIO" as the symbols argument.

        Args:
           symbols: Comma-separated list of tickers (e.g. "AAPL, MSFT") OR "PORTFOLIO" to load from CSV.
           investment: Total portfolio value (default 1,000,000)
        """
    from tools.portfolio_analytics import (
        analyze_correlation,
        calculate_portfolio_metrics,
        calculate_var,
        generate_portfolio_charts,
        get_fee_income_analysis,
        get_sector_exposure,
    )
    weights = None
    all_symbols = None
    all_weights = None
    if _portfolio_alias(symbols):
        summary = get_portfolio_summary()
        if isinstance(summary, dict) and 'error' in summary:
            return f"Error loading portfolio: {summary['error']}"

        holdings = summary.get('holdings', [])
        tradeable_set = set(get_tradeable_symbols())
        symbol_list, weights, _ = _aggregate_symbol_weights(holdings, allowed_symbols=tradeable_set)

        total_value_cad = _parse_metric_number(summary.get('total_value_cad'))
        total_value_usd = _parse_metric_number(summary.get('total_value_usd'))
        if total_value_cad > 0:
            investment = total_value_cad
        all_symbols, all_weights, _ = _aggregate_symbol_weights(
            holdings,
            denominator=total_value_usd if total_value_usd > 0 else None,
        )
        report_header = f'### 🏦 Institutional Report: My Portfolio (~${investment:,.0f} CAD)'
    else:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        report_header = f'### 🏦 Institutional Portfolio Report (${investment:,.0f})'
        all_symbols = symbol_list
    if not symbol_list:
        return 'Error: No valid symbols provided.'
    report = []
    report.append(report_header)
    report.append(f"**Assets:** {', '.join(symbol_list[:10])}..." if len(symbol_list) > 10 else f"**Assets:** {', '.join(symbol_list)}")
    report.append('')

    metrics = calculate_portfolio_metrics(symbol_list, weights=weights)
    var = calculate_var(symbol_list, weights=weights, investment=investment)
    corr = analyze_correlation(symbol_list)
    costs = get_fee_income_analysis(all_symbols or symbol_list, weights=all_weights)

    canonical_metrics = _build_canonical_metrics_table(
        metrics if 'error' not in metrics else {},
        var if 'error' not in var else {},
        corr if 'error' not in corr else {},
        costs if 'error' not in costs else {},
        investment,
    )
    report.extend(canonical_metrics)
    report.append('')

    if 'error' not in metrics:
        m = metrics.get('metrics', {})
        interp = metrics.get('interpretation', {})
        report.append('#### 📊 Core Risk Metrics')
        report.append('| Metric | Value | Rating |')
        report.append('| :--- | :--- | :--- |')
        report.append(f"| **Sharpe Ratio** | {m.get('sharpe_ratio')} | {interp.get('sharpe')} |")
        report.append(f"| **Sortino Ratio** | {m.get('sortino_ratio')} | Downside Risk |")
        report.append(f"| **Beta (vs SPY)** | {m.get('beta')} | {interp.get('beta')} |")
        report.append(f"| **Max Drawdown** | {m.get('max_drawdown')} | ⚠️ Historic Max Loss |")
        report.append(f"| **Volatility** | {m.get('annual_volatility')} | {interp.get('volatility')} |")
        report.append('')
    if 'error' not in var:
        v_data = var.get('value_at_risk', {})
        report.append('#### 🛡️ Stress Test (Value at Risk)')
        report.append(f"> **95% Confidence:** You should NOT lose more than **{v_data.get('daily_var_dollars')}** in a single day.")
        report.append('')
    if 'error' not in corr:
        report.append('#### 🔗 Diversification Check')
        report.append(f"**Avg Correlation:** {corr.get('average_correlation')}")
        pairs = corr.get('correlation_pairs', [])[:3]
        if pairs:
            report.append('**⚠️ Highest Correlations:**')
            for p in pairs:
                report.append(f"- {p['pair']}: **{p['correlation']}**")
        report.append('')
    # Sector exposure uses the whole-portfolio weights (all_symbols/all_weights),
    # not the tradeable-only symbol_list/weights used for the return-series
    # metrics above. Renormalizing sector % to just the tradeable-equity subset
    # (excluding pensions/funds/cash) silently inflates whatever sector that
    # subset happens to concentrate in — the source of a real "Technology at
    # 100.0%" false concentration warning against a portfolio that was actually
    # ~34% tech. Mirrors the get_fee_income_analysis call above, which already
    # gets this right.
    sect = get_sector_exposure(all_symbols or symbol_list, weights=all_weights, is_portfolio=_portfolio_alias(symbols))
    if 'error' not in sect and sect.get('concentration_warning'):
        report.append(f"> ⚠️ {sect.get('concentration_warning')}")
    try:
        charts = generate_portfolio_charts(symbol_list)
        for (_chart_name, json_str) in charts.items():
            report.append(f'\n[PLOTLY_JSON:{json_str}]')
    except Exception as e:
        report.append(f'\n> Chart generation failed: {e}')
    if len(symbol_list) > 15:
        condensed = [report[0], report[1]]
        if 'error' not in metrics:
            condensed.append(f"- **Risk:** Sharpe {m.get('sharpe_ratio')}, Beta {m.get('beta')}, Vol {m.get('annual_volatility')}")
            condensed.append(f"- **Max Drawdown:** {m.get('max_drawdown')}")
        if 'error' not in var:
            condensed.append(f"- **Value at Risk (95%):** {v_data.get('daily_var_dollars')}")
        if 'error' not in corr:
            condensed.append(f"- **Avg Correlation:** {corr.get('average_correlation')}")
        condensed.append('')
        condensed.extend(canonical_metrics)
        for r in report:
            if '[PLOTLY_JSON:' in r:
                condensed.append(r)
        return '\n'.join(condensed)
    return '\n'.join(report)

@tool
def get_correlation_analysis(symbols: str):
    """Analyze correlation between assets to check diversification. Low correlation = better diversification. Pass comma-separated symbols."""
    symbol_list = [s.strip() for s in symbols.split(',')]
    return analyze_correlation(symbol_list)

@tool
def get_value_at_risk(symbols: str='', investment: float=100000):
    """Calculate Value at Risk - maximum expected loss at 95% confidence.
    Pass comma-separated symbols, or leave empty to use the user's portfolio.
    Symbols not in the user's verified holdings are flagged; if none are held,
    the calculation falls back to the actual portfolio."""
    symbol_list, meta = _resolve_risk_symbols(symbols)
    if not symbol_list:
        return {'error': f"No symbols found for profile '{meta['profile']}'.", 'profile': meta['profile']}
    var = calculate_var(symbol_list, investment=investment)
    if isinstance(var, dict):
        stamped = {'profile': meta['profile'], 'symbols': symbol_list, 'scope': meta['scope'], **var}
        if meta['note']:
            stamped['reconciliation_note'] = meta['note']
        return stamped
    return var

@tool
def get_portfolio_sectors(symbols: str):
    """
    Analyze sector concentration, geographic exposure, and factor style (Growth/Value).
    If user says "analyze MY PORTFOLIO", pass "PORTFOLIO" as the symbols argument.

    Args:
        symbols: Comma-separated list of tickers (e.g. "AAPL, MSFT") OR "PORTFOLIO"
                 to load the user's full verified, dollar-weighted holdings.
    """
    weights = None
    all_symbols = None
    all_weights = None
    if _portfolio_alias(symbols):
        summary = get_portfolio_summary()
        if isinstance(summary, dict) and 'error' in summary:
            return f"Error loading portfolio: {summary['error']}"
        holdings = summary.get('holdings', [])
        tradeable_set = set(get_tradeable_symbols())
        symbol_list, weights, _ = _aggregate_symbol_weights(holdings, allowed_symbols=tradeable_set)
        if not symbol_list:
            return 'Error: No tradeable holdings found in the verified portfolio.'
        # Whole-portfolio weights for sector exposure — pensions/funds/cash are
        # included (get_sector_exposure buckets non-tickers as "Private/Manual
        # Holding" via _is_plausible_ticker rather than erroring), so the reported
        # percentages reflect the actual portfolio instead of being renormalized
        # to 100% over just the tradeable-equity subset. geo/factor analysis below
        # still use the tradeable-only symbol_list — those need real price data
        # per symbol and can't be computed for a pension/fund pseudo-ticker.
        total_value_usd = _parse_metric_number(summary.get('total_value_usd'))
        all_symbols, all_weights, _ = _aggregate_symbol_weights(
            holdings,
            denominator=total_value_usd if total_value_usd > 0 else None,
        )
    else:
        symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        if not symbol_list:
            return 'Error: No valid symbols provided.'

    sectors = get_sector_exposure(all_symbols or symbol_list, weights=all_weights, is_portfolio=_portfolio_alias(symbols))
    geo = get_geographic_exposure(symbol_list)
    factors = analyze_factors(symbol_list)
    return {'sector_analysis': sectors, 'geographic_analysis': geo, 'factor_analysis': factors}

@tool
def clean_user_memory(target: str='all'):
    """
        Clean/reset the agent's memory of the user.
        target: 'all' (wipe everything), 'facts' (clear facts), 'history' (clear conversation), 'profile' (reset profile).
        Use this if the user asks to forget them or start over.
        """
    return clean_memory(target)

@tool
def analyze_technical_chart(symbol: str):
    """
        Performs full technical analysis (Trends, Momentum, Support/Resistance, Moving Averages).
        Use this when user asks for "Chart Analysis", "Technicals", or "Trends".
        Returns a text summary of indicators.
        """
    data = get_comprehensive_technicals(symbol)
    if 'llm_summary' in data:
        return data['llm_summary']
    return str(data)

@tool
def check_tax_loss_harvesting():
    """Scans the portfolio for positions with >10% losses that can be sold to offset taxes (Tax-Loss Harvesting)."""
    return analyze_tax_loss_harvesting()

@tool
def check_asset_location():
    """Evaluates portfolio asset location tax efficiency (0-100) across account types and recommends non-taxable asset swaps."""
    return analyze_asset_location()

@tool
def precheck_wash_sale(symbol: str, account: str, proposed_date: str = ""):
    """Check a proposed BUY against the wash-sale / superficial-loss window (4.7).

    CALL THIS BEFORE recommending any rebuy, re-entry or add to a position that
    was recently trimmed or sold at a loss. It is a required gate, not a nicety.

    `allowed: false` has three distinct reasons and they are not interchangeable:
      * `jurisdiction_unresolved` — the account name does not name a country. Say
        so and ask; do NOT assume the user's own country.
      * `not_covered` — this engine has no rules for that jurisdiction. It will
        NOT apply another country's. Say the check could not be run.
      * `repurchase_window` — a recorded disposition falls inside the window.

    `allowed: true` with `evidence_complete: false` is a WEAK pass and you must
    say so: the transaction record is empty or partial, so the engine failed to
    OBJECT rather than cleared the trade. Never report it as "no wash sale".

    `advice_ready` is false on every module today — no tax professional has
    reviewed them. Use this to stop and ask, never to state the user's tax
    treatment.
    """
    from tools.tax_policy import precheck_rebuy

    return precheck_rebuy(symbol, account, proposed_date or None)


@tool
def get_tax_dispositions(lookback_days: int = 400):
    """Dated dispositions on file, for loss-deferral checks (4.7).

    `status: "no_data"` means the RECORD is empty, not that nothing was sold.
    Two sources feed it: the trade journal (what a human typed) and 4.10a
    position changes whose cause a human stated as a trade. An unclassified
    position decrease is deliberately excluded — it might be a sale, a transfer,
    a fee or a corporate action, and inferring one is forbidden.
    """
    from tools.tax_policy import scan_dispositions

    return scan_dispositions(lookback_days=lookback_days)


@tool
def get_portfolio_attribution(window_days: int = 365):
    """Flow-adjusted time-weighted return vs a benchmark blended to the book's currency mix (4.10).

    READ `status` FIRST. Four of the five values are REFUSALS, and each names what
    would unblock it. Report the refusal; do NOT substitute another return figure:

      * `insufficient_coverage` — too few of the window's days carry a valuation.
        Quote `coverage.coverage_pct`. Do NOT quote `coverage.span_days` as if it
        were coverage; the span cannot see holes between its endpoints.
      * `flows_incomplete` — a position change in the window has no stated cause,
        or has one and no stated amount. The user can resolve it on the
        reconciliation screen. Until then there is NO return figure to give.
      * `flow_date_unvalued` — a flow landed on a day with no portfolio value.
      * `no_history` — no valuation series at all.
      * `measured` — `twr_pct` is the return with external flows removed, and
        `alpha_pct` is it minus the blended benchmark.

    When `status` is `measured`, also read `benchmark.benchmark_note`: the legs are
    PRICE series, so alpha is biased upward by roughly the benchmark's dividend
    yield. And `positions.complete` — if false, the contribution table excludes
    positions that could not be priced and does NOT sum to the portfolio.

    The `percent_return` on the dashboard is value-over-cost-basis and is NOT this
    number: it moves when a deposit lands. Never present the two as alternatives.
    """
    from tools.attribution import get_attribution_report

    return get_attribution_report(window_days=window_days)


@tool
def check_rate_sensitivity():
    """Fixed-income duration, convexity and the +/-100bp shock table for the portfolio (4.8).

    READ `status` BEFORE ANY NUMBER, and do not collapse these three into "no bonds":

      * `no_fixed_income` — every holding was classified and none is a bond. This
        IS a measured zero and you may say the book has no direct rate exposure.
      * `undetermined`    — nothing classified as a bond, but some holdings could
        not be classified at all. You may NOT say the book has no bonds; say the
        holdings named in `unclassified` could not be identified.
      * `yields_missing`  — bonds were found and no yield-to-maturity is on file
        for them. Duration is withheld on purpose. Do not substitute the yield
        curve's nearest tenor and quote the result as this holding's duration.
      * `measured`        — `modified_duration`, `convexity` and `shocks` apply to
        the FIXED-INCOME SLEEVE only, never to the whole book.
    """
    from tools.bond_analytics import portfolio_rate_sensitivity

    return portfolio_rate_sensitivity()


@tool
def analyze_bond_shock(coupon_pct: float, ytm_pct: float, years: float,
                       face: float = 100.0, frequency: int = 2):
    """Price, modified duration, convexity and a parallel-shift shock table for one bond (4.8).

    Rates in PERCENT (4.5 means 4.5%). `frequency` is coupons per year; pass 0 for
    a GIC, a strip or any pay-at-maturity instrument.

    Each shock row gives the EXACT reprice next to the duration-only and
    duration+convexity estimates. Quote the exact figure; use the gap between the
    two estimates to explain convexity. Every row assumes a PARALLEL shift of the
    whole curve — say so if the user is asking about a steepening or a flattening.
    """
    from tools.bond_analytics import shock_table

    return shock_table(coupon_rate=coupon_pct / 100.0, ytm=ytm_pct / 100.0,
                       years=years, face=face, frequency=frequency)


@tool
def analyze_ladder_rate_sensitivity(amount: float = 100000.0,
                                    investment_type: str = "GIC",
                                    currency: str = "CAD"):
    """Duration and convexity of the 5-year bond/GIC ladder, off the live curve (4.8).

    Read `marked_to_market` with the numbers. A non-redeemable GIC has no
    secondary market: the shock rows are the opportunity cost of being locked in
    while rates moved, NOT a loss the holder can take. Say that plainly rather
    than reporting a paper loss on a GIC ladder.
    """
    from tools.bond_analytics import ladder_rate_sensitivity

    return ladder_rate_sensitivity(amount, investment_type, currency)


@tool
def get_portfolio_reconciliation():
    """Returns per-account position changes observed since the previous daily snapshot.

    Read `status` first: `no_data`/`accruing` mean there is nothing to compare
    yet, NOT that the portfolio was unchanged.

    Then read `classified` on EVERY change before you read its `cause`:

      * `classified: false` — the cause is UNKNOWN. A quantity delta is equally
        consistent with a trade, a deposit, a transfer, a reinvested dividend, a
        fee or a corporate action. Do not name one, do not pick the likeliest,
        and do not treat it as an external flow or a tax event. Say it is
        unclassified and, if it matters to the answer, ask the user.
      * `classified: true` — the `cause` is what the USER stated, and
        `classified_at` is when. You may rely on it and should attribute it to
        them ("you recorded this as a contribution"), never to your own analysis.

    `classification.complete` says whether every change in the window has a
    stated cause. While it is false, any total of deposits or withdrawals is a
    LOWER BOUND — never present it as the amount contributed or withdrawn.

    You cannot classify a change. That store has exactly one author, and it is
    the user, through the portfolio page.
    """
    return get_reconciliation()

@tool
def get_catalyst_scoreboard():
    """Returns how often this system's own catalyst calls came true, by confidence and materiality.

    Use this when asked whether the catalyst signals can be trusted, or how well
    the advisor's own event calls have worked out.

    Read `overall.reportable` BEFORE `overall.hit_rate`. Below 20 scored calls the
    rate is null and you must say there is not enough evidence yet — never
    compute your own rate from the counts to fill the gap. A bucket's counts are
    real even when its rate is null.

    Only directional calls on named tickers are scored. Catalysts whose
    direction was "mixed" or "unclear", and ones whose prices could not be
    fetched, are counted but excluded from every rate — they are not failures.
    """
    from tools.catalyst_resolution import scoreboard
    return scoreboard()

@tool
def get_event_radar():
    """Returns the merged holdings event radar (upcoming earnings, ex-dividend, and FOMC dates for held names)."""
    return build_event_radar_cached()

@tool
def get_etf_flows(symbol: str = ""):
    """Returns creation/redemption share flow series and accrual status for held ETF funds.
    Args:
        symbol: Optional specific ETF symbol (e.g. 'SPY', 'QQQ'). If omitted, returns series for all held funds.
    """
    sym = symbol.strip().upper() if symbol else ""
    if sym:
        return get_flow_series(sym)
    universe = collect_active_profile_fund_universe()
    funds = universe.get("funds", [])
    return {
        "universe": universe,
        "fund_series": {f: get_flow_series(f) for f in funds},
    }

@tool
def get_contribution_sensitivity():
    """Returns how much saving more (or less) each year changes the user's goal odds and required return.

    Use this for "what if I contribute more", "should I save more", "how much
    would an extra $10K a year help", or whenever the goal reads off track and
    the user asks what to do about it.

    Read the `required_annual_return` on each row, not just the success rate.
    That is the actionable framing: an extra contribution LOWERS the return the
    plan needs, so "save more" and "earn more alpha" are the same row read two
    ways — and only one of them is under the user's control.

    Every row ran on the same simulated return paths, so DIFFERENCES between rows
    are real. A single row's absolute success rate carries about a percentage
    point of simulation error — do not quote one to more precision than that.

    Only the listed contribution levels were simulated. Do not interpolate
    between rows or extrapolate past the largest one, and do not recommend a
    level; report what each implies and let the user choose.
    """
    from tools.goal_projection import build_contribution_sensitivity
    return build_contribution_sensitivity()

@tool
def run_monte_carlo_simulation(current_value: float=100000, annual_contribution: float=12000, years: int=20):
    """
        Runs a Monte Carlo simulation to project portfolio growth/retirement success.
        Args:
            current_value: Current portfolio size
            annual_contribution: Yearly savings addition
            years: Time horizon
        Returns: Success rates and median/best/worst case outcomes.
        """
    from tools.monte_carlo import run_monte_carlo as run_mc_engine
    res = run_mc_engine(current_portfolio_value=current_value, annual_contribution=annual_contribution, years=years, num_simulations=5000)
    if 'error' in res:
        return res['error']
    return f"### 🎲 Monte Carlo Simulation Results ({years} Years)\n**Success Rate:** {res['success_rate']}%\n**Median Outcome:** ${res['median_result']:,}\n**Worst Case (10th %):** ${res['worst_case']:,}\n**Best Case (90th %):** ${res['best_case']:,}\n**Stress Test (Seq. of Returns):** {res['stress_test_success_rate']}% success if market crashes early.\n**Interpretation:** {res['interpretation']}"

@tool
def scan_dark_pool(symbol: str):
    """
        Scans for 'Dark Pool' signatures and Block Trades.
        Detects 1-minute volume spikes > 3 sigma indicative of institutional activity.
        """
    return scan_dark_pool_proxy(symbol)

@tool
def check_smart_money(symbol: str):
    """
        Checks for Insider Trading and US Senate Trading disclosures.
        Useful for tracking 'Smart Money' flows.
        """
    return {'insider_activity': get_fmp_insider_trades(symbol), 'senate_activity': get_fmp_senate_disclosures(symbol)}

@tool
def get_market_calendar():
    """
        Fetches the Economic Calendar for the next 14 days.
        Returns upcoming Fed meetings, CPI releases, GDP data, and Job reports.
        """
    events = get_economic_calendar()
    if events:
        return '\n'.join([f"- {e['date']}: {e['event']} (Est: {e['estimate']}, Prev: {e['previous']})" for e in events])
    from tools.web_search import search_news
    safe_print('⚠️ FMP Calendar API failed/limited. Falling back to Web Search.')
    return search_news('Upcoming US Economic Calendar dates next 2 weeks CPI Fed FOMC Jobs Report', max_results=3)

@tool
def analyze_fx_risks(base_currency: str='CAD'):
    """
        Analyze current portfolio for Foreign Exchange (Currency) risks.
        Calculates exposure to USD vs CAD and total equity sensitivity to rate changes.
        """
    return analyze_my_portfolio_fx(base_currency)

@tool
def get_my_trade_journal():
    """Get a history of your past trade decisions and outcomes."""
    return get_trade_history()

@tool
def log_investment_decision(symbol: str, action: str, price: float, thesis: str, quantity: float = 0.0, time_horizon: str='Medium Term', conviction: str='Medium'):
    """
        Log a new trade decision/thesis.
        action: BUY, SELL, HOLD, ADD, TRIM.
        thesis: Why are you doing this? (e.g. 'Undervalued based on DCF', 'Breaking out').
        price: Execution price.
        quantity: Number of shares/contracts involved. Use 0 for thesis-only notes.
        """
    return log_trade(
        symbol=symbol,
        action=action,
        price=price,
        quantity=quantity,
        thesis=thesis,
        time_horizon=time_horizon,
        conviction=conviction,
    )

@tool
def close_investment_decision(symbol: str, exit_price: float, outcome: str, lessons_learned: str):
    """
        Close the most recent OPEN trade journal entry for a symbol and record its outcome.
        symbol: Ticker to close.
        exit_price: Price at which the position was exited.
        outcome: e.g. 'Profit', 'Loss', 'Breakeven'.
        lessons_learned: What this trade taught you, for future reference.
        """
    return close_trade(
        symbol=symbol,
        exit_price=exit_price,
        outcome=outcome,
        lessons_learned=lessons_learned,
    )

@tool
def analyze_earnings_transcript(symbol: str, year: int=None, quarter: int=None):
    """
        Retrieves and reads the actual earnings call transcript for qualitative analysis.
        Use this to find management tone, future guidance, and key Q&A details.
        """
    return get_earnings_transcript(symbol, year, quarter)

@tool
def get_short_interest_data(symbol: str):
    """
        Get Short Interest data to assess squeeze risk or market sentiment.
        Returns Short % of Float, Short Interest, to gauge betting against the stock.
        """
    return get_short_interest(symbol)

@tool
def perform_search(query: str):
    """
        Plain single-query web search: news, world events, market trends, sector
        performance, investment ideas, or fact-checking (e.g. 'best safe stocks
        2026', 'undervalued tech stocks'). Use when the data isn't available from
        other tools (e.g. very recent events) — and for anything NOT about markets
        (evaluating software/tools, general how-to or fact questions, etc.), since
        this runs the query as-is instead of rewriting it into market-news angles.
        """
    return search_news(query)

@tool
def search_multi_source(topic: str):
    """
        PREFERRED for financial/market topics (a ticker, sector, macro theme, company):
        searches multiple angles sequentially for comprehensive coverage — breaking
        news, expert analysis, market impact, and Canadian market. Results are
        cached for 5 minutes.

        NOT for general/non-financial questions (e.g. evaluating a piece of software,
        a how-to question, fact-checking something unrelated to markets) — this tool
        appends "market analysis" / "impact stocks" / "TSX news" to every query, which
        pollutes non-financial searches with irrelevant finance keywords and returns
        unrelated stock news. Use perform_search for those instead.

        NOTE: Runs sequentially (not parallel) to avoid thread-pool stampede when
        Tavily is rate-limited and all searches fall back to DuckDuckGo simultaneously.
        """
    cache_key = f'news_multi:{topic}'
    cached = get_cached(cache_key)
    if cached:
        return cached

    # Run searches sequentially — avoids nested thread pools that cause deadlocks
    # when DDG fallback is active (global DDG lock serializes anyway)
    results = {}

    search_tasks = [
        ('breaking', f'{topic} breaking news', 'd'),
        ('canada',   f'{topic} Canada TSX news', 'w'),
        ('analysis', f'{topic} market analysis', 'w'),
        ('impact',   f'{topic} impact stocks', 'd'),
    ]

    for label, query, timelimit in search_tasks:
        if is_cancelled():
            results[label] = 'Search cancelled'
            continue
        try:
            results[label] = search_news(query, timelimit=timelimit)
        except Exception as e:
            results[label] = f'Search error: {e}'

    combined = f"\n### Breaking News\n{results.get('breaking', 'No results')}\n\n### Expert Analysis\n{results.get('analysis', 'No results')}\n\n### Market Impact\n{results.get('impact', 'No results')}\n\n### Canadian Market (TSX/Economy)\n{results.get('canada', 'No results')}\n"
    set_cached(cache_key, combined)
    return combined

@tool
def get_fear_greed():
    """
        Get the CNN Fear & Greed Index — overall market mood / sentiment gauge.
        Contrarian indicator: extreme fear = potential buy opportunity, extreme
        greed = caution. Use for 'market mood', 'sentiment', or 'fear/greed' queries.
        """
    return get_fear_greed_index()

@tool
def get_market_headlines(limit: int=10):
    """
        Get reliable financial news from major outlets (Yahoo Finance, etc.).
        Use this for 'market update', 'what happened today', or broad news.
        """
    return get_market_news(limit)

@tool
def get_specific_news(tickers: str=None):
    """
        Get company-specific news. Pass comma-separated tickers (e.g. 'AAPL, TSLA').
        Use this when user asks about specific stocks.
        """
    if not tickers:
        return 'Error: No tickers provided.'
    return get_company_news(tickers)

PORTFOLIO_TOOLS = [
  analyze_portfolio_risk,
  run_retirement_simulation,
  check_portfolio_correlation,
  assess_marginal_trade_risk,
  simulate_portfolio_rebalancing,
  assess_portfolio_risk,
  get_portfolio_snapshot,
  get_dividend_data,
  check_portfolio_earnings,
  get_hypothetical_portfolio,
  project_retirement_goal,
  check_fx_impact,
  project_portfolio_income,
  check_portfolio_allocation,
  get_my_portfolio,
  verify_portfolio_holdings,
  get_portfolio_risk_metrics,
  get_portfolio_sectors,
  check_tax_loss_harvesting,
  check_asset_location,
  precheck_wash_sale,
  get_tax_dispositions,
  get_portfolio_attribution,
  check_rate_sensitivity,
  analyze_bond_shock,
  analyze_ladder_rate_sensitivity,
  get_portfolio_reconciliation,
  get_event_radar,
  get_etf_flows,
  get_catalyst_scoreboard,
  get_contribution_sensitivity,
  analyze_fx_risks,
  preview_candidate_impact,
  optimize_portfolio,
  check_rebalance_drift
]
MACRO_TOOLS = [
  get_macro_overview,
  get_macro_strategy,
  get_canada_macro,
  get_boc_vs_fed,
  generate_future_forecast,
  get_economic_calendar_tool,
  model_options_strategy,
  get_global_indices,
  backtest_strategy
]
NEWS_TOOLS = [
  search_stock_news,
  get_sentiment,
  analyze_reddit_sentiment,
  get_stock_news,
  get_fear_greed,
  get_specific_news,
  get_latest_trump_yaps
]
RISK_TOOLS = [
  run_stress_test,
  replay_historical_episode,
  analyze_factor_exposures,
  check_risk_metrics,
  get_correlation_analysis,
  get_value_at_risk
]
DEEP_ALPHA_TOOLS = [
  get_insider_short_interest,
  dealer_gamma_exposure,
  get_alt_data,
  check_management_tone,
  compare_management_tone_qoq,
  analyze_crowded_trade,
  get_insider_activity,
  get_material_events,
  get_institutional_moves,
  scan_dark_pool,
  check_smart_money,
  scan_geopolitical_events,
  check_ticker_geopolitical_context
]
MARKET_TOOLS = [
  fetch_fundamentals,
  scan_guru_picks,
  analyze_technicals,
  plot_chart,
  fetch_comprehensive_analysis,
  analyze_options_chain,
  analyze_sectors,
  compare_stocks,
  get_analyst_targets,
  get_earnings_calendar,
  calculate_position,
  get_realtime_quote,
  get_price_history,
  get_fundamentals_detailed,
  analyze_patterns,
  get_support_resistance,
  get_ma_signals,
  get_analyst_ratings,
  visualize_stock_chart,
  scan_opportunities,
  get_funnel_scorecard,
  get_advisor_performance_scorecard,
  record_recommendation_execution,
  check_supply_chain,
  run_diagnostics,
  scan_options_chain,
  scan_technical_breakouts,
  get_earnings_data,
  get_institutional_data,
  check_sector_rotation,
  get_relative_strength,
  predict_surprise,
  find_ipos,
  run_stock_deep_dive,
  scan_intraday_movers,
  scan_tsx_movers,
  get_stock_quote,
  get_valuation_metrics,
  screen_stocks,
  get_historical_performance,
  get_competitors,
  get_etf_holdings_data,
  structure_trade_setup,
  construct_bond_ladder,
  run_technical_analysis,
  get_seasonality_data,
  check_esg_scores,
  analyze_mutual_funds,
  read_url,
  run_health_check,
  clean_user_memory,
  analyze_technical_chart,
  run_monte_carlo_simulation,
  get_market_calendar,
  get_my_trade_journal,
  log_investment_decision,
  close_investment_decision,
  analyze_earnings_transcript,
  get_short_interest_data,
  perform_search,
  search_multi_source,
  get_market_headlines,
  get_market_pulse_data,
  get_market_regime_data,
  get_tsx_stock_quote,
  get_tsx_stock_analyst,
  get_asx_stock_quote,
  get_asx_stock_analyst,
  get_eu_stock_quote,
  get_eu_stock_analyst
]

ALL_TOOLS = PORTFOLIO_TOOLS + MACRO_TOOLS + NEWS_TOOLS + RISK_TOOLS + DEEP_ALPHA_TOOLS + MARKET_TOOLS
