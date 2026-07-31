# Project Structure

Overview of the CairnIQ codebase.

## Directory Layout

```
cairniq/
├── agent/                      # AI Agent System
│   ├── nodes/                 # Agent nodes (reasoning, analysis, routing, risk)
│   ├── prompts/               # Node prompts
│   ├── eval/                  # Golden-set eval harness for the risk layer
│   ├── optimized/             # Compiled/optimized DSPy artifacts
│   ├── state.py              # Agent state management
│   ├── modules.py            # DSPy modules for structured outputs
│   ├── tool_registry.py      # Tool groups that compose ALL_TOOLS
│   ├── tool_retriever.py     # Semantic tool selection (FAISS, BM25 fallback)
│   ├── tool_substitution.py  # Curated equivalent for a failed tool (never a merely related one)
│   ├── checkpointer.py       # Per-profile LangGraph conversation store routing
│   ├── risk_rules.py         # Generator/judge rule text, built per request
│   ├── catalyst_engine.py    # News → ranked catalysts → escalated scenarios
│   ├── cost_tracker.py       # Per-slot LLM cost accounting
│   ├── llm_budget.py         # Persistent spend budget + runaway protection
│   ├── logger.py             # Logging utilities
│   └── memory.py             # User memory system
│
├── api/                        # HTTP layer
│   ├── routers/              # chat, dashboard, portfolio, memory, news,
│   │                         #   settings, alerts, auth, pages
│   ├── background.py         # Background task plumbing
│   └── dependencies.py       # Shared request dependencies
│
├── tools/                      # Analysis Tools (~130 registered agent tools)
│   ├── portfolio_csv.py      # Portfolio management
│   ├── opportunity_scanner.py # Market scanning
│   ├── scheduler.py          # Background task loop (briefs, scans, monitors)
│   ├── alerts.py             # Persistent alerts store + delivery
│   ├── watch_conditions.py   # Advisor-authored trigger levels + evaluator
│   ├── intraday_sentinel.py  # Zero-LLM market-state change detector
│   ├── event_radar.py        # Earnings / ex-div / FOMC T-3 + T-1 sweep on held names
│   ├── freshness.py          # As-of fetch stamps + staleness gates
│   ├── provenance.py         # Turn-level evidence quality (live/unavailable/stale/unverified)
│   ├── ips_precheck.py       # Deterministic compliance pre-check
│   ├── risk_verdict_log.py   # Per-profile verdict audit trail
│   ├── candidate_impact.py   # Pre-trade "should I add this?" delta report
│   ├── portfolio_optimizer.py # Constrained optimizer + drift-band rebalancing
│   ├── drawdown_playbook.py  # User-authored crash rules + armed deployment ladder
│   ├── covariance.py         # Ledoit-Wolf / EWMA / sample estimators
│   ├── fx_utils.py           # Currency inference + historical FX series
│   ├── sec_edgar.py          # Form 4 / 8-K / 13F pipeline
│   ├── insider_data.py       # Insider transactions, US (Form 4) + Canadian (SEDI) vocabularies
│   ├── earnings_nlp.py       # Loughran-McDonald transcript tone, quarter-over-quarter
│   ├── fund_flows.py         # ETF/fund share-count recorder (accrues its own history)
│   ├── weekly_review.py      # The Sunday one-page review (read-only assembly)
│   ├── profile_readiness.py  # Which blanks switch which shipped feature off
│   ├── observations.py       # Prompt-invisible behavioural log
│   ├── observation_consolidation.py # Gated pass: observations → drafted candidate rules
│   ├── pending_lessons.py    # Candidate rules awaiting a human confirm
│   ├── housekeeping.py       # Daily log rotation + checkpoint pruning
│   ├── health_check.py       # System diagnostics
│   ├── trump_tracker.py      # "Trump Yap": Truth Social post feed
│   ├── market_*.py           # Market data tools
│   ├── yf_utils.py           # Yahoo Finance utilities
│   └── alpha_vantage.py      # Alpha Vantage API
│
├── static/                     # Web UI Assets
│   ├── css/                  # Stylesheets
│   ├── js/                   # JavaScript
│   └── images/               # Images
│
├── templates/                  # HTML Templates
│   ├── index.html            # Main chat interface
│   ├── dashboard.html        # Portfolio dashboard
│   ├── context_and_graph.html # Context: profile + goal + playbook, learned, graph
│   ├── weekly_review.html    # The weekly one-page review
│   ├── alerts.html           # Alerts inbox
│   ├── recommendations.html  # Advisor Ledger scorecard
│   └── *.html                # Other pages
│
├── user_data/                  # All runtime state (gitignored)
│   ├── profiles/<name>/      # Per-profile portfolio, memory, alerts, ledgers
│   ├── daily_cache/          # API response cache
│   ├── tool_index/           # Tool-RAG vector index
│   └── funnel_config.json    # Opportunity scanner tuning
│
├── logs/                       # Application Logs
│   ├── server/               # Server logs
│   ├── tools/                # Tool execution logs
│   ├── chat_runtime/         # Chat stream / queue diagnostics
│   └── frontend/             # Frontend logs
│
├── docs/                       # Documentation
│   ├── user-guide/           # User documentation
│   ├── technical/            # Technical documentation
│   └── archive/              # Old/debug documentation
│
├── scripts/                    # Utility Scripts
│   ├── install/              # Installer helpers, backup/restore, verify
│   ├── docker/               # Docker build files
│   ├── package/              # Packaging
│   ├── local/                # Default-deny zone for personal scripts (gitignored)
│   ├── run_eval_harness.py   # Risk-layer regression gate (--live runs the real judge)
│   └── cairniq_watchdog.py   # Supervision for always-on deployments
│
├── assets/                     # Project Assets
│   └── icons/                # Application icons
│
├── tests/                      # Test Suite (~1,675 tests across 135 files)
│   ├── test_agent/           # Agent + node tests
│   ├── test_api/             # HTTP surface tests
│   ├── test_tools/           # Tool-level tests
│   ├── test_eval/            # Golden-set harness (deterministic scenarios)
│   └── test_*.py             # Cross-cutting tests
│
├── server.py                   # FastAPI Application
├── install.sh                 # Installation script
├── CairnIQ.command # Desktop launcher
├── requirements.txt           # Python dependencies (runtime)
├── requirements-optional.txt  # Optional runtime enhancers (e.g. faiss-cpu)
├── requirements-dev.txt       # Dev tooling (pre-commit)
├── .pre-commit-config.yaml    # Secret-scanning git hooks (gitleaks)
├── .env.example              # Environment template
├── my_portfolio.example.csv  # Portfolio template
├── CONTRIBUTING.md            # Contributor guide
└── README.md                  # Main documentation
```

## Core Components

### Agent System (`agent/`)

**Purpose**: Multi-agent AI system for financial analysis

**Key Files**:
- `state.py` - Manages conversation state and context
- `nodes/supervisor.py` - Routes queries to appropriate agents
- `nodes/deep_reasoning.py` - Strategic analysis and synthesis
- `nodes/market_analyst.py` - Market data gathering and analysis
- `modules/` - DSPy modules for structured outputs

**Flow**:
1. User query → Supervisor
2. Supervisor routes to appropriate agent
3. Agent uses tools to gather data
4. Deep Reasoning synthesizes final answer

### Tools System (`tools/`)

**Purpose**: ~130 registered agent tools (composed in `agent/tool_registry.py` from the portfolio, macro, news, risk, deep-alpha, and market groups), plus internal modules the agent never calls directly.

**Categories**:
- **Portfolio**: Position tracking, performance calculation, pre-trade impact preview
- **Market Data**: Real-time quotes, historical data
- **Technical Analysis**: Indicators, chart patterns
- **Fundamental Analysis**: Financial metrics, ratios
- **Filings**: SEC EDGAR Form 4 insider activity, 8-K material events, 13F institutional diffs; SEDI-sourced insider activity for TSX/TSXV listings
- **Macro**: FRED US indicators; Bank of Canada Valet for CAD policy rate, CORRA, core CPI, bank rates, and BoC-vs-Fed divergence
- **News & Sentiment**: News aggregation, sentiment scoring
- **Political/Policy Signals**: Truth Social "Trump Yap" feed for market/supply-chain impact
- **Screening**: Stock screening, opportunity scanning
- **Risk**: Risk metrics, correlation, covariance estimation, compliance pre-check

**Key Files**:
- `health_check.py` - System diagnostics
- `opportunity_scanner.py` - Market scanning engine
- `portfolio_csv.py` - Portfolio management
- `market_mechanics.py` - Market analysis utilities
- `trump_tracker.py` - "Trump Yap" Truth Social feed crawler (`get_latest_trump_yaps`)
- `sec_edgar.py` - Filings pipeline (`get_material_events`, `get_institutional_moves`, EDGAR-first `get_insider_activity`)

**Infrastructure modules** (not agent-callable):
- `scheduler.py` - Background task loop; see [Architecture](technical/ARCHITECTURE.md#background-layer) for the task table
- `alerts.py` - The single delivery path for unprompted output (store, WebSocket push, desktop notification)
- `watch_conditions.py` / `intraday_sentinel.py` / `event_radar.py` - The zero-LLM monitors
- `freshness.py` / `provenance.py` - Fetch-time stamps, and the turn-level summary of what the evidence was worth
- `ips_precheck.py` / `risk_verdict_log.py` - The deterministic half of the advice gate, and its audit trail
- `weekly_review.py` / `profile_readiness.py` - Read surfaces over intelligence that already shipped
- `observations.py` / `observation_consolidation.py` / `pending_lessons.py` - Behavioural memory, gated behind a human confirm
- `housekeeping.py` - Daily log rotation and checkpoint pruning
- `user_profile.py` - Per-profile path resolution (`get_data_path`)

### Web Interface (`static/`, `templates/`)

**Purpose**: Modern web UI for interaction

**Components**:
- Chat interface with streaming responses
- Real-time portfolio dashboard
- Interactive charts and visualizations
- News feed integration
- Mobile-responsive design

**Technology**:
- Vanilla CSS for styling
- Vanilla JavaScript (no framework)
- Server-Sent Events for streaming
- Chart.js for visualizations

### Server (`server.py`, `api/routers/`)

**Purpose**: FastAPI application serving the web interface and the iOS client

**Features**:
- RESTful API endpoints, grouped into routers under `api/routers/`
- WebSocket support for real-time updates (status, reasoning trace, alerts)
- Auth middleware + per-request profile binding
- Session management
- Request/response logging
- Error handling

**Key Endpoints**:
- `/api/chat` - Chat interface (NDJSON stream)
- `/api/dashboard-data` - Portfolio data
- `/api/news-feed` - Market news
- `/api/alerts` - Alerts inbox
- `/api/priority` - Today's Priority brief
- `/api/recommendations` - Advisor Ledger scorecard
- `/api/weekly_review` - The weekly one-page review
- `/api/profile_readiness` - Which blanks switch which feature off
- `/api/engine_health` - Background engine liveness
- `/api/health` - Public liveness probe
- `/ws` - WebSocket connection

Full reference: [API.md](technical/API.md).

## Data Flow

### Query Processing

```
User Input
    ↓
FastAPI Server (server.py)
    ↓
Supervisor Agent (agent/nodes/supervisor.py)
    ↓
Specialized Agent (DeepReasoning/MarketAnalyst)
    ↓
Tool Execution (tools/*.py)
    ↓
Data Gathering (APIs, Cache, Database)
    ↓
Response Synthesis (DSPy Modules)
    ↓
Streaming Response
    ↓
User Interface
```

### Portfolio Updates

```
CSV File / Questrade API
    ↓
Portfolio Loader (tools/portfolio_csv.py)
    ↓
Price Updates (Yahoo Finance)
    ↓
Performance Calculation
    ↓
Dashboard Update
    ↓
User Interface
```

## Configuration Files

### `.env`
Environment variables for API keys and settings

### `requirements.txt`
Python package dependencies (runtime)

### `requirements-optional.txt`
Optional runtime enhancers (e.g. `faiss-cpu` for dense tool retrieval; falls back to BM25 if absent)

### `requirements-dev.txt`
Development tooling. Install with `pip install -r requirements-dev.txt`, then run `pre-commit install`.

### `.pre-commit-config.yaml`
Local secret-scanning git hooks (gitleaks + detect-private-key) that block commits containing secrets before they enter git history. Complements GitHub's server-side secret scanning + push protection.

### `.gitignore`
Files to exclude from version control

## Data Files

Everything under `user_data/` is gitignored. State is **per profile** under `user_data/profiles/<profile>/`; resolve paths through `tools.user_profile.get_data_path(...)` rather than hardcoding them.

### Per-profile (`user_data/profiles/<profile>/`)

| File | Contents |
| :--- | :--- |
| `my_portfolio.csv` | Portfolio holdings |
| `portfolio_history.csv` | Daily post-close snapshots |
| `chat_history.json` | Conversation history |
| `user_memory.json` | Profile, facts, lessons (capped at 15, FIFO), active theses, recommendation ledger, wealth goal, drawdown playbook, and `risk_constraints` (the only source of risk limits) |
| `observations.json` | Prompt-invisible behavioural observation log |
| `pending_lessons.json` | Candidate rules awaiting a human confirm |
| `trade_journal.json` | Thesis journal entries |
| `knowledge_graph.json` | Semantic knowledge graph |
| `checkpoints.sqlite` | LangGraph checkpoints for conversation state |
| `alerts.jsonl` | Alerts inbox (capped at 500, atomic rewrites) |
| `watch_conditions.jsonl` | Pending + fired advisor trigger levels |
| `intraday_sentinel_state.json` | Last observed market state, so a tick fires only on a change |
| `risk_verdicts.jsonl` | Risk Judge verdict audit trail |
| `sentinel_history.json` | Market pulse history |
| `scheduler_runs.json` | Per-task cooldown / daily-marker registry |
| `feedback.json` | User feedback on responses |

### Shared (`user_data/`)

| File | Contents |
| :--- | :--- |
| `.env` | Non-secret config (secrets live in the OS keychain) |
| `auth.json` | User accounts for multi-user auth |
| `funnel_config.json` | Opportunity scanner tuning — see [Funnel Configuration Guide](technical/FUNNEL_CONFIG.md) |
| `funnel_signal_log/` | Walk-forward signal log for scanner backtesting |
| `fund_shares_history.csv` | Locally accrued fund share counts — one dated row per fund per day, with its source |
| `daily_cache/`, `tool_index/`, `catalyst_log/` | Caches and indexes |

## Logging

### Log Locations

- `logs/server/server.jsonl` - Server events
- `logs/tools/tools.jsonl` - Tool execution
- `logs/frontend/frontend.jsonl` - UI events

### Log Format

JSON Lines format for easy parsing:
```json
{"timestamp": "2026-04-08T21:00:00", "level": "INFO", "component": "Server", "message": "Request received"}
```

## Dependencies

### Core
- Python 3.12+
- FastAPI (web server)
- LangChain & LangGraph (agent framework)
- DSPy (structured outputs)

### AI/ML
- AWS Bedrock (Claude models)
- FAISS (vector search)
- sentence-transformers (embeddings)

### Data
- pandas (data manipulation)
- yfinance (market data)
- requests (API calls)

### Full list in `requirements.txt`

## Development

### Secret Scanning (pre-commit)

Commits are scanned locally for secrets via a [gitleaks](https://github.com/gitleaks/gitleaks) pre-commit hook (see `.pre-commit-config.yaml`). Activate it once after cloning:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

This is the local layer; GitHub secret scanning + push protection cover the server side.

### Adding New Tools

1. Create tool file in `tools/`
2. Implement tool function with proper signature
3. Add to tool registry in `agent/nodes/`
4. Add tests in `tests/`
5. Update documentation

### Adding New Agent Nodes

1. Create node file in `agent/nodes/`
2. Implement node function
3. Add to graph in supervisor
4. Update routing logic
5. Add tests

### Modifying UI

1. Edit templates in `templates/`
2. Update styles in `static/css/`
3. Add JavaScript in `static/js/`
4. Test in browser
5. Check mobile responsiveness

## Testing

### Run Tests
```bash
source .venv/bin/activate
python -m pytest tests/
```

### Test Coverage
```bash
pytest --cov=agent --cov=tools tests/
```

### Risk-Layer Regression Gate

The golden-set harness runs engineered drafts through the grounding and compliance
checks; its deterministic scenarios are part of the pytest suite. Run the live
variant — which puts the corpus through the **real** LLM judge — before changing
any provider, model, or prompt:

```bash
python scripts/run_eval_harness.py --live
```

It exits non-zero on any FAIL or ERROR, and never writes to the per-profile
verdict audit trail.

## Deployment

See [INSTALLATION.md](user-guide/INSTALLATION.md) for deployment instructions.

---

**Last Updated**: July 28, 2026
