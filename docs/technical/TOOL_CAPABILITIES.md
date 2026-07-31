# CairnIQ - Tool Capabilities

This document outlines the library of ~130 registered tools powering CairnIQ. The tools are segmented by their primary function within the AI reasoning engine, providing a comprehensive "Toolbox" for the terminal's multi-agent system.

---

## 1. Broker & Data Integrations
These tools connect directly to external accounts and market data providers to fetch raw, real-time data securely.
- **`alpaca.py`**: Connects to Alpaca Markets for US equity data, paper trading, and live balance aggregation.
- **`questrade.py`**: Interfaces with Questrade (Canada) to pull live RRSP, TFSA, and Margin account holdings securely.
- **`alpha_vantage.py`**: Extracts core fundamental data, EPS, and daily time series.
- **`fmp_api.py`**: Pulls deep fundamental data, valuation metrics, and discounted cash flow (DCF) models from Financial Modeling Prep.
- **`fred_api.py`**: Scrapes Federal Reserve Economic Data (FRED) for macroeconomic indicators like inflation, unemployment, and yield curves.
- **`finnhub_api.py` / `polygon_api.py`**: Specialized connectors for tick data, institutional trades, and options flow.
- **`gdelt_api.py`**: Extracts real-time global event data for geopolitical sentiment analysis.

## 2. Portfolio & Wealth Management
These tools act on your specific holdings, whether fetched via APIs or inputted manually via CSV.
- **`portfolio_csv.py` / `portfolio_tracker.py`**: Deduplicates live broker data against your manual `my_portfolio.csv` files to create a single "Zero-Trust" unified portfolio view.
- **`portfolio_analytics.py`**: Calculates beta, Sharpe ratios, maximum drawdowns, and sector allocations of your current holdings. Return series are converted to your base currency first, so FX risk is priced into volatility, VaR, drawdown, beta and correlation rather than silently ignored on a mixed-currency book.
- **`covariance.py` / `fx_utils.py`**: Ledoit-Wolf (default), EWMA, and sample covariance estimators, plus currency inference and historical FX series — the estimation layer behind the risk numbers.
- **`position_sizing.py`**: Calculates trade entry sizes from your stated risk constraints. Every result carries a `risk_basis` so a figure is never attributed to a limit you didn't set.
- **`candidate_impact.py`**: Pre-trade "should I add this?" preview — recomputes the portfolio with a candidate at a proposed size and reports the delta (beta, volatility, 95% CVaR, correlation-to-book), never a verdict.
- **`portfolio_optimizer.py`**: SLSQP max-Sharpe / min-vol / target-vol optimization under the profile's *own* IPS caps, and `check_rebalance_drift` — "should I rebalance?" against the user's own stored drift band. With no band stored the check is unavailable and says why; nothing invents one. A breach returns the trade list, turnover in base currency, and realized-gain exposure for taxable accounts only (the tax bill itself is withheld — no marginal rate exists in the profile).
- **`drawdown_playbook.py`**: The store for rules written while calm — never-sell list, contribution priority, cash deployment ladder, drift band, note to self — and `evaluate_deployment_ladder`, which arms each rung against peak-to-date drawdown so the level the user named fires once, when reached. Nothing here is ever authored by the app.
- **`tax_loss.py`**: Scans the portfolio for tax-loss harvesting opportunities to optimize end-of-year tax positioning.

## 3. Advanced Analysis & Valuation
The core quantitative intelligence used by the Deep Reasoning Engine to evaluate individual assets.
- **`comprehensive_data.py`**: An aggregated pipeline that merges FMP, AlphaVantage, and sentiment data into a single, cohesive company profile.
- **`technicals.py` / `pattern_recognition.py`**: Generates moving averages, RSI, MACD, and algorithmically detects candlestick patterns.
- **`monte_carlo.py` / `simulation.py`**: Runs stochastic simulations on portfolio variance and generates probability distributions for future asset prices.
- **`compare_assets.py`**: Runs side-by-side fundamental and technical comparisons of industry peers (e.g., AAPL vs. MSFT).
- **`price_targets.py`**: Aggregates Wall Street analyst consensus targets and tracks historical estimate revisions.
- **`earnings_nlp.py`**: Scores management language on a Loughran-McDonald financial lexicon subset (finance-specific, because general lexicons misread *liability*, *cost* and *capital*) and reports the **quarter-over-quarter delta**, normalised per 1,000 words — the change is the signal, not the level. Detection of a real transcript is positive: a rate-limited web summary returns `unavailable` instead of a tone verdict for a call nobody read.

## 4. Macro & Market Intelligence
Broad market scanners that detect structural shifts in economic regimes.
- **`opportunity_scanner.py`**: Scans the broad market for high-conviction structural shifts, prioritizing AI growth leaders and cyclical sector rotations.
- **`market_sentinel.py` / `macro_analysis.py`**: Monitors Fear & Greed indices, VIX levels, and broader economic cycles to dynamically adjust the system's risk appetite.
- **`sector_rotation.py` / `sector_analysis.py`**: Identifies capital flow between sectors (e.g., Tech to Utilities) to predict near-term market trends.
- **`earnings_calendar.py` / `fed_calendar.py`**: Tracks upcoming catalysts that may trigger severe market volatility.
- **`news_sources.py` / `web_search.py`**: Aggregates live headlines and performs DuckDuckGo web searches (using thread-safe non-blocking timeouts) to inject real-time context into the AI's thesis.
- **`insider_data.py`**: Insider transactions for **both** venues. US rows carry Form 4 wording, TSX/TSXV rows carry SEDI wording that shares no keyword with it — an ordered, most-specific-first rule table separates conviction (open-market buys/sells) from mechanics (grants, exercises, ownership plans, gifts) and from issuer buybacks, so the most common Canadian row (a buyback whose text contains "repurchase") is no longer read as an insider buy. A Canadian suffix settles the venue offline rather than asking EDGAR, which holds no filings for those names.
- **`event_radar.py`**: One deterministic pass merging earnings, ex-dividend and FOMC dates against **held** names, nearest first, with T-3 / T-1 alerts only. A date the provider did not supply is listed as unknown, never as "nothing scheduled".
- **`fund_flows.py`**: Creation/redemption recorder for ETFs and mutual funds. No vendor sells the share-count history on these plans, so the series is accrued locally one dated point per day; until enough days exist it reports `accruing` with a day count rather than drawing a 0.0% flow. Shares not AUM (unit-free, so no FX enters the arithmetic), and one source per series — a series refuses to difference across a source change rather than manufacture a 15% phantom creation.
- **`sec_edgar.py`**: Keyless SEC filings pipeline — Form 4 insider activity with correct transaction coding (only code P is an open-market buy; grants and exercises are compensation mechanics, not conviction), 8-K material events with per-item severity, and 13F quarter-over-quarter institutional position diffs.
- **`boc_valet.py`**: Keyless Bank of Canada Valet feed — the policy rate with the date and size of its last move, CORRA and its spread to the target, CPI-trim/median (the core measures the BoC's own rate statements cite), posted chartered-bank prime/mortgage/GIC rates, and the BoC-vs-Fed divergence. Each series carries a freshness window sized to its own publication cadence; an observation past that window is returned as explicitly unavailable rather than as a current number.

## 5. Risk, Compliance & Monitoring
The layer that audits what the system is about to tell you, and watches the market when you aren't.
- **`ips_precheck.py`**: Deterministically extracts proposed trades from a draft and checks them numerically against your stated caps before the LLM judge ever sees it.
- **`risk_verdict_log.py`**: Persists every risk verdict to a per-profile audit trail for calibration.
- **`alerts.py`**: The single delivery path for unprompted output — persistent store, WebSocket push, and desktop notification.
- **`watch_conditions.py`**: Stores the trigger levels the advisor commits to and re-checks them on a schedule, so its own commitments don't expire silently.
- **`intraday_sentinel.py`**: Zero-LLM market-state change detector (VIX / drawdown bands, fresh crosses, volume spikes) with a state store and hysteresis, so it reports changes rather than standing conditions.
- **`freshness.py`**: Records fetch time inside a payload so a cache hit can't masquerade as a live quote — the gate that stops an automated alert firing on unprovable data.
- **`provenance.py`**: The turn-level view of what the evidence was worth — live / unavailable / stale / unverified source counts, read from the same rendered context the judge audits. Unstamped is `unverified`, never `fresh`; thin evidence caps the draft's confidence.
- **`profile_readiness.py`**: Which user-authored inputs are blank, and which shipped feature is inert as a result. Reports emptiness and names the consequence — never authors, defaults or exemplifies a value.
- **`weekly_review.py`**: The Sunday-evening one-page assembly over surfaces that already exist. Every section always renders, it reads only (no LLM, no scan, no network beyond a cached read), and it never invents a number.
- **`observations.py` / `observation_consolidation.py` / `pending_lessons.py`**: The behavioural memory chain — deterministic prompt-invisible observations, a gated consolidation pass that must cite its evidence, and a human confirm before anything reaches a prompt.
- **`housekeeping.py`**: Daily log rotation (copy-truncate, because writers hold the inode) and checkpoint pruning by whole thread, only where the age is provable from the UUID6 timestamp.
- **`scheduler.py`**: The background task loop (cooldowns, locking, timeouts) that drives all of the above.

## 6. Core Infrastructure & Agent Memory
The silent engines that keep the terminal fast, context-aware, and resilient against failure.
- **`memory.py` / `graph_memory.py`**: Persists your specific rules, goals, and facts across sessions using a JSON-backed knowledge graph, ensuring the AI "remembers" your risk tolerance.
- **`credential_manager.py`**: Securely manages multi-provider rate limits globally (handling HTTP 429 errors) so background scans don't exhaust your API quotas or hang the system.
- **`health_check.py`**: A background diagnostic tool that verifies the tool modules are responding properly without hanging the main FastAPI event loop.
- **`exception_logger.py`**: Provides standardized, context-aware error logging that feeds directly into the AI's diagnostic reasoning trace.
- **`cache.py` / `daily_cache.py`**: Ensures heavy API responses (like historical data or sector aggregates) are cached to disk to lower costs and speed up UI rendering.

---

### Security & Usage Note
*CairnIQ enforces a Bring Your Own Key (BYOK) architecture. The tools listed above execute entirely locally on your machine and only communicate with third-party APIs using the credentials provided in your environment settings.*
