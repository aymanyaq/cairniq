# CairnIQ

CairnIQ is a local-first AI portfolio intelligence console for personal market research, portfolio monitoring, and risk-aware decision support.

🌐 **Official Website:** [cairniq.com](https://www.cairniq.com)

📱 **iOS Companion App:** A native SwiftUI client (Chat, Market, and News) lives in its own repository — [github.com/aymanyaq/cairniq-ios](https://github.com/aymanyaq/cairniq-ios). It connects to your self-hosted CairnIQ server over your local network / VPN.

> [!CAUTION]
> **NOT FINANCIAL ADVICE**: This software is for educational and informational purposes only. Trading involves significant risk. The authors and contributors assume no liability for financial losses. Never make investment decisions based solely on this tool. Always consult with a certified financial advisor.

> [!IMPORTANT]
> 🇵🇸 **URGENT GAZA HUMANITARIAN EMERGENCY**
> This project is free for personal wealth management and source-available. If you find value in this software, please witness the documented ground reality and consider supporting vital civilian relief efforts directly through trusted global organizations:
> - **[Documented Ground Reality (Getty Images Photojournalism Archive)](https://www.gettyimages.ca/photos/gaza-destruction)** — Verified editorial photojournalism documenting the scale of civilian crisis and infrastructure destruction.
> - **[UNRWA Gaza Emergency](https://www.unrwa.org/gaza-emergency)** — Direct support for food, shelter, and medical care.
> - **[Islamic Relief Palestine Appeal](https://www.islamicreliefcanada.org/emergencies/palestine-appeal)** — Critical humanitarian aid and medical supplies.
>
> *Humanitarian Notice: This project is independent and has no political, governmental, or military affiliations. These links are provided solely for humanitarian purposes. Verified editorial archives are linked directly to respect copyright and maintain source authenticity.*

![Version](https://img.shields.io/badge/version-2.4.0-blue)
![Python](https://img.shields.io/badge/python-3.12--3.13-green)
![License](https://img.shields.io/badge/license-Custom_NonCommercial-red)


## 🌟 Capabilities & Features

CairnIQ is designed as a private command center for your personal finances, merging live broker data with deep AI reasoning.

### 1. Multi-Agent Artificial Intelligence
- **Deep Reasoning Engine:** Automatically breaks down complex financial queries into multi-step execution plans.
- **Market Pulse Analyst:** Interprets broad macroeconomic trends (Fear & Greed, VIX, S&P 500 regimes) to gauge overall market safety.
- **Opportunity Scanner:** Continuously scans the market for high-conviction structural shifts and cyclical opportunities.
- **Policy Signal Tracking ("Trump Yap"):** Pulls Donald Trump's latest Truth Social posts in real time to weigh potential market and supply-chain impact.

### 2. Financial Connectivity Hub
Manage your data feeds securely from a unified dashboard.
- **Alpaca Markets (US & Global):** Full integration for real-time market data and paper-trading capabilities.
- **Questrade (Canada):** Securely track your TFSA, RRSP, and Margin accounts in real-time.
- **Zero-Trust CSV Tracking:** Privacy-conscious? Keep your keys to yourself. Download our CSV template, fill in your holdings, and upload it directly via the UI.

### 3. Automated Valuation & Technicals
- **Fundamental Analysis:** The AI automatically pulls and synthesizes P/E, P/S, EV/EBITDA, and FCF Yields from external APIs (FMP, AlphaVantage).
- **Technical Indicators:** Real-time generation of moving averages, RSI, and MACD momentum signals.
- **SEC EDGAR Filings:** Keyless Form 4 insider activity with correct transaction coding (an open-market buy is not a stock grant), 8-K material events with per-item severity, and 13F quarter-over-quarter institutional position diffs.
- **Canadian Insider Filings (SEDI):** TSX/TSXV issuers file on SEDI, not EDGAR, in a vocabulary that shares no keyword with Form 4 — so a naive reader scores a Canadian table backwards, since the most common row is an *issuer buyback* whose description contains "repurchase". Canadian listings are classified on their own rule table, with conviction (open-market buys and sells) separated from mechanics (grants, option exercises, ownership plans) and from buybacks.
- **Earnings-Call Tone, Quarter-over-Quarter:** Management language scored on a Loughran-McDonald financial lexicon, reported as the **change** from the same team's previous call rather than a level — a team that always hedges is not a sell. If the transcript could not actually be read, the tool says so instead of scoring a search snippet.
- **Constrained Optimizer & Drift Rebalancing:** Max-Sharpe / min-vol / target-vol optimization under *your* stated position, sector and restricted-list caps, and a "should I rebalance?" check against *your* stored drift band — with the trade list, turnover, and the realized-gain exposure of the sells in taxable accounts only.
- **Bank of Canada Macro:** Keyless first-class CAD data from the institution that sets the rate — policy rate with the date and size of its last move, CORRA and its spread to the target, CPI-trim/median core inflation, posted mortgage and GIC rates, and the BoC-vs-Fed policy divergence with its effect on the loonie.

### 4. Grounded Advice, Not Just Answers
Every advice-generating turn passes a review gate before you see it.
- **Risk Judge:** A final challenge pass that audits the draft for unsourced numbers, claims about holdings you don't own, and fabricated history — with each verdict written to a per-profile audit trail.
- **Deterministic IPS Pre-Check:** Proposed trades are extracted and checked numerically against *your* stated constraints — position caps, sector caps, dollar-at-risk. There are no house defaults: a limit you have not stated enforces nothing and is never quoted back at you.
- **Candidate Impact Preview:** "Should I add this?" recomputes your portfolio *with* the candidate and reports the delta — beta, volatility, 95% CVaR, correlation — as a report, never a verdict.
- **Turn Provenance:** Every turn carries a record of what its evidence was actually worth — how many sources answered, how many were unavailable, stale, or unstamped. Data with no readable timestamp is `unverified`, never `fresh`, and a turn built on thin evidence has its confidence capped rather than shipping like a fully-sourced one.
- **Honest Substitution:** When a tool fails, a hand-curated equivalent that answers the *same* question can stand in — never a merely "related" tool, since a wrong source is more dangerous than a visible gap. Substitutions are always named in the output; where no honest equivalent exists, the Data Gap stands.
- **Regression Harness:** A golden-set corpus of engineered drafts that runs against the real judge, so a provider, model, or prompt change can't quietly regress the safety layer.

### 5. Monitoring That Speaks First
CairnIQ watches between conversations and comes to you.
- **Alerts Inbox:** A persistent, deduplicated inbox (`/alerts`) with severity levels, a nav badge, and macOS desktop notifications for warnings and above.
- **Watch Conditions:** Trigger levels the advisor commits to in a brief are stored and re-checked automatically every 30 minutes in market hours — so its own commitments don't quietly expire.
- **Intraday Sentinel:** A zero-LLM market-state change detector (VIX and drawdown band crossings, fresh death/golden crosses, volume spikes) that fires on *changes*, never on a level that has stood all session.
- **Holdings Event Radar:** Earnings, ex-dividend dates and FOMC meetings merged against the names you actually hold, with **T-3 and T-1 alerts only** — each firing once, because a daily countdown trains you to ignore it. A date the provider didn't give is reported as unknown, never as "nothing coming".
- **Armed Deployment Ladder:** The cash-deployment rungs you wrote while calm fire at the drawdown levels you named — once each, at the moment they're reached — instead of only surfacing on a crash deep enough to print the whole playbook.
- **Scheduled Morning Brief:** Today's Priority is precomputed before the open so it's ready when you are. All background automation is opt-in per profile via the `SCHEDULER_ENABLED` toggle in Settings.
- **Freshness Gates:** Automated alerts refuse to fire on a quote that can't be proven recent, and every fired alert carries its vintage ("as of 10:28, 2 min ago").

### 6. Surfaces You Actually Read
Correct-but-invisible is a failure mode. These pages exist to make shipped intelligence legible.
- **Weekly One-Page Review (`/review`):** Assembled and delivered Sunday evening so it's waiting Monday morning — goal headline, the week's market state, how past advice actually scored, what the advisor said, whether the background engines are alive, and what's still blank. **Every section always renders**: a section with nothing to say says so by name, because an omitted section is a silence that gets filled in with plausible-sounding history. It reads only — no LLM, no scan, nothing it could change while describing.
- **Advisor Ledger (`/recommendations`):** Past calls scored against outcomes at 2-week, 1-month and 3-month horizons versus SPY, with stated-confidence accuracy. Restatements collapse into the original call, and a call superseded before its horizon is graded at supersession rather than retiring unscored.
- **Profile Readiness Switchboard (`/context`):** Every input only *you* can state, whether it's on file, and **exactly which shipped feature is inert while it isn't**. It never authors, defaults, or even exemplifies a value — a blank is a valid answer, and nothing here is scored or chased.
- **Engine Health (`/api/engine_health`):** The one view that shows whether a background engine is quietly dead, reporting the count that proves the whole chain ran rather than a rare event that may legitimately be zero.

### 7. Your Rules, Never Ours
Every constraint the system enforces is one you stated. **Unstated means unconstrained** — nothing is defaulted on your behalf.
- **Drawdown Playbook:** A never-sell list, contribution priorities, a cash deployment ladder, a rebalance drift band and a note to your future self — written while calm, read back at −15% / −25%. The app never authors an entry: a rule invented by software and quoted back during a crash carries the authority of a promise you made yourself.
- **Wealth Goal & Projection:** Your own target, horizon and contribution feed the Monte Carlo goal projection — so the sentence at −25% can be "with contributions continuing, the goal is still funded in N% of paths" rather than "stay calm".
- **Behavioural Memory With a Human Gate:** The system observes what you *do* after a turn (deterministically — no model judges you), consolidates that into candidate rules on a gated pass, and **drafts** them for you to confirm. Nothing reaches a prompt without a click, and every candidate must cite the observations it came from.

---

## 🚀 Onboarding & Quick Start

### 1. Installation
Ensure you have Python 3.12 or 3.13 installed on your machine. Python 3.11 is no longer supported (numpy 2.5+ dropped it) and 3.14+ is not yet supported (transitive Pydantic V1 compat shim is incompatible).

```bash
# Clone the repository
git clone https://github.com/aymanyaq/cairniq.git
cd cairniq
```

**macOS / Linux** — run the automated installer:
```bash
./install.sh
```

**Windows** — open PowerShell in the project folder and run:
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

> [!TIP]
> **Guided Setup**: The installer will offer to launch an interactive **Setup Wizard** at the end. This wizard helps you configure your AI provider (Bedrock, Anthropic, OpenAI, or Azure OpenAI / AI Foundry), link your brokerage, and set up your personal financial profile without manually editing any files.

> [!TIP]
> **Safe Installation**: The installer automatically detects existing user data and creates a timestamped backup in the `backups/` directory before any changes are made. It will also migrate legacy data files into the protected `user_data/` zone automatically.

### 2. Launching CairnIQ
Once installed, double-click the desktop launcher the installer created, or run from the project folder:

| Platform | Command |
|----------|---------|
| macOS / Linux | `./CairnIQ.command` |
| Windows | `CairnIQ.bat` |

The application will automatically open in your web browser at `http://localhost:8000`.

### 3. Data Management & Backups
Your persistent state is stored in `user_data/`. The system includes utilities for maintenance:
- **Backup Location**: `backups/user_data_YYYYMMDD_HHMMSS.tar.gz`
- **Restore Data**: `./scripts/install/restore_backup.sh`
- **Verify Integrity**: `python3 scripts/install/verify_data.py`
- **Tune the opportunity scanner**: edit `user_data/funnel_config.json` (auto-created from `funnel_config.example.json` on install/first run). See the [Funnel Configuration Guide](docs/technical/FUNNEL_CONFIG.md).

### 4. "Bring Your Own Key" (BYOK) Setup
We believe your data and API usage should remain under your control. When you first launch the application:
1. Navigate to the **Settings** page in the left sidebar.
2. Enter your preferred **LLM Provider Keys** (AWS Bedrock, Anthropic, OpenAI, Azure, or Google) - **one is required**.
3. Enter your **Financial Data Keys** (recommended but optional with fallbacks):
   - **FMP (Recommended)** - Provides insider trades, senate data, and earnings transcripts (no fallback available)
   - **Alpha Vantage (Optional)** - Quotes and fundamentals (falls back to yfinance)
   - **FRED, Finnhub, Polygon, Tavily (Optional)** - Enhance capabilities but have fallbacks
4. Connect your **Brokerage** (Alpaca or Questrade) in the Financial Connectivity Hub - **optional, CSV alternative available**.
5. Click **Commit Infrastructure Sync**.

Your keys are saved *locally* and are never transmitted to any central server. On desktop platforms, CairnIQ stores secrets in the OS keychain when available and keeps only blank placeholders in `user_data/.env`.

---

## 💼 Managing Your Portfolio

The console aggregates your holdings into a single pane of glass. You have two ways to feed data into the engine:

**Method A: Live Broker Sync**
Enter your brokerage API credentials in the Settings menu. CairnIQ will automatically poll and synchronize your holdings.

**Method B: Manual Entry (Zero-Trust)**
1. Navigate to the **Portfolio** tab.
2. Click **Download CSV Template**.
3. Fill out your positions (Symbol, Shares, Entry Price) in Excel or Numbers.
4. Click **Upload CSV** to instantly populate your dashboard.

CairnIQ handles deduplication automatically—live syncs take precedence over manual CSV entries for the same account.

---

## 📜 License & Usage Policy

**Personal Wealth Management Non-Commercial License**

This software is licensed **strictly for personal, non-commercial use.**
- ✅ **Allowed:** Managing your personal portfolio, learning about the markets, personal financial research.
- ❌ **Prohibited:** Use within a hedge fund, family office, proprietary trading firm, or any corporate entity. You may not monetize this software or use it to manage third-party assets for a fee.

Please see the [LICENSE](LICENSE) file for the full legal text.

### API Data Compliance
This project does not redistribute or re-sell raw financial data. You are responsible for ensuring that your usage of third-party API keys (Alpha Vantage, FMP, etc.) complies with their respective Terms of Service.

---

## 📚 Documentation

Browse the full documentation in [`docs/`](docs/README.md). Highlights:

**For users:**
- 📦 [Installation & Provider Setup Guide](docs/user-guide/INSTALLATION.md) — installer walkthrough, Guided Setup Wizard, LLM/data provider comparison, fallback architecture
- 📖 [User Guide](docs/user-guide/USER_GUIDE.md) — chat interface, agent routing, portfolio management, memory system, thesis journal
- 🛠️ [Troubleshooting](docs/user-guide/TROUBLESHOOTING.md) — common install, startup, API, performance, and data issues
- 🚀 [Launcher Modes](docs/LAUNCHER_MODES.md) — production (`CairnIQ.command`) vs demo (`start_demo.sh`)

**For developers:**
- 🧭 [Project Structure](docs/PROJECT_STRUCTURE.md) — directory layout, core components, data flow
- 🏛️ [Architecture](docs/technical/ARCHITECTURE.md) — runtime shape, agent flow, operational constraints
- 🌐 [API Reference](docs/technical/API.md) — every REST/SSE endpoint with request/response shapes
- 🧪 [Adding Tools](docs/technical/ADDING_TOOLS.md) — extending the agent's capabilities
- 🔌 [Tool Capabilities](docs/technical/TOOL_CAPABILITIES.md) — inventory of available tools and their inputs

**Companion app:**
- 📱 [CairnIQ iOS](https://github.com/aymanyaq/cairniq-ios) — native SwiftUI client that talks to your self-hosted CairnIQ server

---

## 📋 Changelog

See the [CHANGELOG](docs/CHANGELOG.md) for a detailed history of changes.
