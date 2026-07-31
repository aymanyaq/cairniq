# CairnIQ Installation & Provider Setup Guide

Welcome to the CairnIQ installation guide. This document provides step-by-step instructions for running the universal installer, navigating the Guided Setup Wizard, understanding the integration options for LLMs and financial APIs, and leveraging the system's failover and fallback systems.

---

## 🖥️ System Requirements

CairnIQ is designed to be cross-platform, running on macOS, Windows, and Linux.

| Requirement | macOS | Linux | Windows |
| :--- | :--- | :--- | :--- |
| **OS Version** | 11.0+ (Apple Silicon or Intel) | Ubuntu 20.04+ / Debian 11+ | 10 / 11 (64-bit) |
| **PowerShell** | *N/A* | *N/A* | 5.1+ (Built-in) |
| **Python** | **3.12 – 3.13** (See Note) | **3.12 – 3.13** (See Note) | **3.12 – 3.13** (See Note) |
| **RAM** | 4 GB min, 8 GB recommended | 4 GB min, 8 GB recommended | 4 GB min, 8 GB recommended |
| **Storage** | 2 GB free disk space | 2 GB free disk space | 2 GB free disk space |
| **Internet** | Required (high speed recommended) | Required (high speed recommended) | Required (high speed recommended) |

> [!IMPORTANT]
> **Python 3.14+ is NOT supported yet, and 3.11 no longer is either.**
> Downstream libraries in the LangChain ecosystem pull in the legacy `pydantic.v1` compatibility shim, which is incompatible with Python 3.14+. Separately, `numpy` (>=2.5) dropped Python 3.11 entirely. You must install Python **3.12 or 3.13** to run CairnIQ. The installer will halt if Python 3.14+ or 3.11 or earlier is detected.

> [!NOTE]
> **Windows Users**: Ensure that you check the **"Add Python to PATH"** option when installing Python. PowerShell 5.1+ is required to execute the setup scripts.

> [!NOTE]
> **OS build prerequisites (rarely needed)**: dependencies ship prebuilt wheels for Python 3.12–3.13, so `pip install` normally needs no compiler. If pip falls back to a *source* build of **`lxml` (≥6)** on Linux, install the XML headers first — Debian/Ubuntu: `sudo apt-get install libxml2-dev libxslt1-dev` (on macOS they come with the Xcode Command Line Tools). **`pyarrow` (≥24)** and **`numpy` (≥2.5)** are large prebuilt wheels published only for Python 3.12–3.13 (numpy dropped 3.11); on an unsupported interpreter pip will report "no matching distribution".

---

## 📦 Universal Installer Walkthrough

CairnIQ includes automated, universal installer scripts that set up the environment, verify dependencies, and handle configuration templates.

### Running the Installer

*   **macOS / Linux**:
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
*   **Windows (PowerShell)**:
    ```powershell
    powershell -ExecutionPolicy Bypass -File install.ps1
    ```

### Under the Hood: What the Installer Does

1.  **Data Preservation (Automatic Backups)**: If an existing `user_data/` directory is found, the script packages it into a timestamped archive (`backups/user_data_YYYYMMDD_HHMMSS.tar.gz` or `.zip` on Windows) to prevent accidental data loss.
2.  **Legacy Data Migration**: Migrates deprecated files from the project root (e.g. `checkpoints.sqlite`, `chat_history.json`, `.env`, and `my_portfolio.csv`) into the modern `user_data/` repository.
3.  **Python Version Enforcement**: Probes candidate paths (`python3.13` down to `python3`) and verifies the major/minor version matches the supported **3.12–3.13** range.
4.  **Port Conflict Detection**: Checks if the default port (`8000`) is in use by another service and issues a warning.
5.  **Virtual Environment Creation**: Sets up a Python virtual environment in `.venv/` and upgrades `pip`, `setuptools`, and `wheel`.
6.  **Dependency Isolation**: Installs packages listed in `requirements.txt`.
7.  **Optional Accelerators**: Installs secondary mathematical / database libraries (e.g. `faiss-cpu`) from `requirements-optional.txt`.
8.  **Structure Verification**: Initializes required folder hierarchies:
    *   Logs: `logs/{agent, chat_runtime, frontend, server, tools}`
    *   Data: `user_data/{cache, embeddings, profiles, daily_cache}` plus a scratch `tmp/`
9.  **Config Template Seeding**: Creates first-run files in `user_data/` if absent — `.env` (from `.env.example`), `my_portfolio.csv` (from `my_portfolio.example.csv`), `funnel_config.json` (from `funnel_config.example.json`), and empty `chat_history.json` / `user_memory.json` / `knowledge_graph.json` stores. Existing files are never overwritten.
10. **Desktop Launchers**: Creates a clickable desktop launcher (`CairnIQ.command` on macOS/Linux or `CairnIQ.bat` on Windows) referencing the absolute path of the project.
11. **Data Integrity Check**: Executes `scripts/install/verify_data.py` to validate that configuration directories, database connections, and templates are correctly configured.
12. **Launch Wizard**: Prompts the user to run the interactive setup wizard.

---

## 🧙 Guided Setup Wizard Walkthrough

The Guided Setup Wizard (`scripts/install/guided_setup.py`) is an interactive terminal wizard that configures identities, imports legacy setups, configures AI credentials, and establishes financial API access.

### Wizard Steps

#### **Step 0: Existing Configuration Import**
*   **Prompt**: *"Do you have a shared config bundle (.zip) to import?"*
*   **Details**: Allows importing a pre-packaged configuration zip file containing environment variables and portfolio data via `scripts/install/import_config.sh`.

#### **Step 1: Identity & Profile**
*   **Prompts**: Name, Age (Optional), and Primary Financial Goal.
*   **Details**: Initializes local profile configuration in `user_data/user_memory.json`.

#### **Step 2: AI Infrastructure (LLM Provider)**
*   **Prompt**: Choice of:
    1.  **AWS Bedrock** (Recommended for enterprise / data-sovereign setups)
    2.  **Anthropic** (Direct API)
    3.  **OpenAI** (Direct API)
    4.  **Azure OpenAI / AI Foundry** (gpt-5.x, DeepSeek, Grok, Kimi)
*   **Details**: Collects corresponding API keys, regions, and model preferences (e.g., Primary Model `AIDLC_MODEL_ID` and Fast Model `AIDLC_SONNET_MODEL_ID`). For Azure, supply the OpenAI-compatible `…/openai/v1` endpoint and your **deployment names** (not model names) — see the **Azure OpenAI** row in the *AI LLM Providers* comparison below.

#### **Step 3: Financial Data Access (Optional but Recommended)**
*   **Prompts**: AlphaVantage API Key and Financial Modeling Prep (FMP) API Key. The wizard then offers an optional follow-up — *"Add more optional data sources?"* — that collects **Tavily** (news/web search), **FRED** (macro), **Finnhub** (sentiment), and **Polygon** (options) keys in one pass.
*   **Details**: Configures primary market and fundamental data access. The system works without these keys but has reduced capabilities. All providers have fallback mechanisms (yfinance, web scraping) so the app remains functional. Any key you skip here can still be added later in **Settings**.

#### **Step 3b: Regional Preferences**
*   **Prompts**: Base currency (`BASE_CURRENCY`, e.g. `USD`, `CAD`, `EUR`, `GBP`) and regional locale (`REGIONAL_LOCALE`, e.g. `English (United States)`, `English (Canada)`).
*   **Details**: Controls how figures and currency are presented throughout the dashboard. Non-secret, so these are written to `user_data/.env` and can be changed any time in **Settings** or by editing that file.

#### **Step 4: Brokerage Connectivity**
*   **Prompt**: *"Do you want to connect a live brokerage?"*
*   **Options**:
    *   **Alpaca**: Configures API keys, secret keys, and establishes sandbox/paper trading toggle (`ALPACA_PAPER_MODE=true`).
    *   **Questrade**: Configures API credentials and imports refresh tokens for tracking Canadian portfolios.

#### **Step 5: Secure Save & Finalization**
*   **Details**: Distributes configurations. Non-secret options are written to `user_data/.env` in plaintext, while secret credentials are encrypted and stored in the OS-native keychain.

---

## 🧠 AI LLM Providers: Cost, Privacy, & Capabilities

CairnIQ utilizes a dual-model slot setup:
*   **Primary Reasoning Model (`AIDLC_MODEL_ID`)**: Configured with a high-capacity model (e.g. Claude Opus or GPT-4o) to execute complex, multi-agent reasoning, portfolio stress-testing, and strategy generation.
*   **Fast / Sonnet Model (`AIDLC_SONNET_MODEL_ID`)**: Configured with a cheaper, faster model (e.g. Claude Sonnet/Haiku or GPT-4o-mini) to handle routine vector lookups, risk pre-checks, data parsing, and summarization tasks.

### LLM Provider Direct Comparison

| Provider | Recommended Models | Data Privacy Profile | Pricing & Setup | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **AWS Bedrock** *(Recommended)* | **Primary**: `global.anthropic.claude-opus-4-8-v1`<br>**Fast**: `global.anthropic.claude-sonnet-4-6` | **Max Sovereignty**: Data stays within your virtual private AWS cloud environment. No model training on your prompts. | Paid via AWS billing. Requires an IAM user with Bedrock permissions. | Uses cross-region inference profiles (`global.*`) which automatically route queries across multiple regions to optimize rate limits. |
| **Anthropic (Direct API)** | **Primary**: `claude-opus-4-8`<br>**Fast**: `claude-sonnet-4-6` | Commercial terms prohibit training on API prompts. Data passes through Anthropic's endpoints. | Direct billing at console.anthropic.com. Requires prepaid developer balance. | Expects bare model IDs. Bedrock-style ARN or profile names will result in validation errors. |
| **OpenAI (Direct API)** | **Primary**: `gpt-4o`<br>**Fast**: `gpt-4o-mini` | Standard API terms prevent training. Data passes through OpenAI's endpoints. | Direct billing at platform.openai.com. Requires developer billing setup. | Lower latency but different reasoning structures compared to Claude. |
| **Azure OpenAI** | **Primary / Fast**: your *deployment names* (set in the Azure portal) | Enterprise data-residency and compliance controls under your Azure tenant. No training on prompts. | Billed via Azure. Requires an API key + endpoint per resource. | Accepts either the bare resource URL or the Azure AI Foundry `…/openai/v1` endpoint (for Foundry models like DeepSeek, Llama, Kimi). The model id is your **deployment name**, not a model name. |
| **Google (Gemini)** | **Primary**: `gemini-2.5-pro`<br>**Fast**: `gemini-2.5-flash` | Standard API terms. Data passes through Google's endpoints. | Direct billing at aistudio.google.com. Requires the `langchain-google-genai` package. | Expects Gemini model ids. |

> **Vendor-neutral by design.** Every capability — multi-agent reasoning, semantic tool routing (embeddings), and structured extraction (DSPy memory/thesis parsing) — follows whichever **LLM Provider** you select. Switching providers requires no code changes: set the provider, models, and credentials in Settings. Per-provider model names are remembered, so you can switch back and forth without re-entering them. (Anthropic-direct is the one exception for embeddings: it has no native embeddings API, so tool routing falls back to BM25 keyword search automatically.)

### ⚡ Throughput & Rate Limits (TPM Sizing)

CairnIQ's deep-reasoning runs are **token-heavy, not request-heavy**. A single synthesis call commonly sends **~15,000–20,000+ input tokens** (system prompt + portfolio + tool results + accumulated context), and the planner runs over multiple cycles. So throttling almost always appears as a **tokens-per-minute (TPM)** limit, *not* a requests-per-minute (RPM) one — and a client-side request limiter cannot fix a TPM ceiling.

**Sizing guidance (per deployment):**

| Deployment TPM | Behavior |
| :--- | :--- |
| 20,000 (a common default) | ❌ A single synthesis call can exceed the whole budget → repeated 429s, degraded/failed runs |
| **≥ 60,000** *(recommended)* | ✅ Comfortable headroom for a full reasoning cycle |
| 100,000+ | For heavier use or concurrent sessions |

*   **Quota is per-model.** The Primary (`AIDLC_MODEL_ID`) and Fast (`AIDLC_SONNET_MODEL_ID`) deployments have **independent** limits — raise both.
*   **Azure**: raise it in AI Foundry → your deployment → **Edit → Tokens-per-Minute**. In the capacity-units dialog, **1 unit = 1,000 TPM** (enter `60` for 60k). For Standard deployments this is a rate ceiling drawn from your subscription quota — you still pay per token *used*, not per unit allocated.
*   **AWS Bedrock** sidesteps per-region caps via cross-region inference profiles (`global.*`), which spread load automatically.

**Measure your real usage.** Every LLM call logs its token count and a rolling per-minute total to `logs/agent/agent.jsonl`:

```bash
grep '"TokenUsage"' logs/agent/agent.jsonl | tail -20
```

Inspect `input_tokens`, `tokens_last_60s`, and `throttle_risk` per `model_id` to see how close each model runs to its ceiling.

**Tuning knobs** (env vars / Settings, all optional — these shape how the app *waits*; raising deployment TPM is the durable fix):

| Var | Default | Purpose |
| :--- | :--- | :--- |
| `LLM_TPM_LIMIT` | `20000` | Sets the `throttle_risk` threshold in the token logs |
| `LLM_MAX_RETRIES` | `8` | Retry budget for non-streaming calls (honors `Retry-After`) |
| `LLM_STREAM_RETRIES` / `LLM_STREAM_RETRY_CAP` | `2` / `20` | Streaming retry attempts + max seconds to wait per retry on a 429 |
| `LLM_MAX_RPS` | `0` (off) | Optional per-endpoint request cap (only helps if you are RPM-bound) |
| `LLM_RETRY_RATE_LIMITS` | `0` (off) | By default a 429 (TPM/RPM) fails fast instead of burning the retry budget on a quota ceiling that won't clear in seconds. Set `1` to retry rate-limit errors too. |

> When a run can't recover from throttling, the header shows a permanent **DEGRADED** (amber) or **FAILED** (red) status so you know the result may be incomplete — see [Troubleshooting → Azure Throttling](TROUBLESHOOTING.md#azure-throttling-429-you-are-token-limited-not-request-limited).

---

## 📊 Market Data & Brokerage Integrations

CairnIQ integrates with several financial data providers to collect real-time quotes, historical pricing, fundamental reports, sentiment analytics, and macroeconomic datasets.

| Provider | Purpose | Data Offered | Free / Paid Tier |
| :--- | :--- | :--- | :--- |
| **FMP (Recommended)** | Fundamental & Insider Data | Financial ratios, SEC filings, **insider trading** (no fallback), **senate disclosures** (no fallback), **earnings transcripts** (no fallback). | Free trial has limited keys. Premium recommended. |
| **AlphaVantage** | Market Data | Real-time equity quotes, historical charts, currency FX rates. | Free tier available (25 requests/day). Paid tiers remove limits. |
| **FRED (St. Louis Fed)** | Macroeconomics | GDP growth rates, Fed funds rates, CPI, inflation metrics, and treasury yields. | Free API key (requires signup). |
| **Finnhub** | Sentiment & News | Analyst recommendations, social sentiment (Reddit/Twitter), company news, and earnings calendars. | Free tier (60 calls/minute limit) covers basic needs. |
| **Polygon.io** | Options & Technicals | Options chain prices, technical indicators, and intraday trades. | Free tier with daily limitations. |
| **Tavily** | News Search | Optimized AI search results for scanning current events, geopolitics, and regulatory actions. | Free tier (1,000 queries/month). |

### Data Provider Fallback Matrix

CairnIQ's architecture includes robust fallback mechanisms so you can start with minimal API keys and add more as needed:

| Data Type | Primary Provider(s) | Fallback Provider(s) | Works Without Any Key? |
| :--- | :--- | :--- | :--- |
| **Stock Quotes** | AlphaVantage → FMP → Polygon | yfinance (free, no key) | ✅ Yes |
| **Fundamentals** | FMP → AlphaVantage | yfinance (limited) | ⚠️ Reduced data |
| **Insider Trading** | FMP | None | ❌ Feature unavailable |
| **Earnings Calendar** | FMP | yfinance | ⚠️ Reduced data |
| **Macro/Economic** | FRED | None | ⚠️ Feature unavailable |
| **Sentiment/News** | Finnhub → Tavily | DDGS web search | ✅ Yes |
| **Options Chains** | Polygon | None | ❌ Feature unavailable |
| **Web Search** | Tavily | DDGS (free, no key) | ✅ Yes |

**Legend:**
- ✅ **Yes** - Full functionality without API key
- ⚠️ **Reduced data** - Basic functionality works, advanced features limited
- ❌ **Feature unavailable** - Feature disabled without API key
| **Questrade** | Brokerage Sync | Live balances, stock holdings, and cost-basis sync for Canadian accounts. | Free for Questrade account holders (uses oauth refresh tokens). |
| **Alpaca** | Brokerage Sync & Trade | US/Global stock and crypto execution, supporting real trading and sandbox paper accounts. | Free developer accounts. |

---

## 🛡️ Failures, Cooldowns, & Fallback Architecture

To ensure operational uptime when working with rate-limited free tiers or offline systems, CairnIQ implements robust fallback mechanisms.

### 1. Security: OS Keychain vs `.env` Fallback

CairnIQ uses the Python `keyring` package to secure credentials:
*   **Keychain Mode (Default)**: On macOS (Keychain Access), Windows (Credential Manager), and desktop Linux (Secret Service/kwallet), sensitive API keys are encrypted and stored inside the OS storage.
*   **Automatic Plaintext Fallback**: If keyring access is unavailable (e.g. running on headless servers, Docker containers, or environments lacking a DBus interface), the installer gracefully falls back to writing credentials in plaintext directly inside `user_data/.env`. The application remains fully functional in fallback mode.

### 2. Market Data API Rate Limit Handling

The CairnIQ credential manager (`tools/credential_manager.py`) tracks rate-limit state per provider so the app degrades gracefully when a free-tier quota is hit:

*   **Dynamic Cooldowns**: When a provider returns HTTP 429 (or an equivalent throttling response), the affected key is placed on a **300-second (5-minute)** cooldown. The credential manager surfaces a clear error to the requesting tool and avoids hammering the endpoint until the cooldown clears.
*   **Graceful Tool-Level Fallback**: Many analysis tools fall back to a secondary data source (e.g., yfinance) when the primary provider is rate-limited, so the user experience is "best-effort with a clear note" rather than a hard failure.
*   **Hitting a quota repeatedly?** Upgrade your plan in the provider's dashboard. Each provider's Terms of Service governs how its keys may be used — please don't run multiple free-tier keys to circumvent quotas.

### 3. Vector Database Search & Embedding Model

CairnIQ uses dense embeddings for semantic **Tool-RAG** (routing queries to the right tools) and memory log search. The embedding model follows your selected **LLM Provider** and can be customised in Settings under *Embedding Model*.

| LLM Provider | Default Embedding Model | Requirement |
|---|---|---|
| AWS Bedrock | `amazon.titan-embed-text-v2:0` | AWS credentials (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) |
| OpenAI | `text-embedding-3-small` | `OPENAI_API_KEY` |
| Azure OpenAI | *(none — set your deployment name)* | Azure API key + endpoint; deploy an embedding model and enter its deployment name, else BM25 is used |
| Google (Gemini) | `models/text-embedding-004` | `GOOGLE_API_KEY` + `langchain-google-genai` |
| Anthropic | *(not available)* | Falls back to BM25 automatically — no config needed |

*   **FAISS (Primary)**: If `faiss-cpu` is installed (see `requirements-optional.txt`) and the active provider supports embeddings, dense vector search is used for tool routing and log recall.
*   **BM25 (Fallback)**: Used automatically when FAISS is not installed, embeddings fail, or the provider is Anthropic. The system degrades gracefully with no crashes.
*   **Custom embedding model**: Leave the *Embedding Model* field blank in Settings to use the provider default, or enter a specific model id (e.g. `text-embedding-3-large` for OpenAI, a custom Azure deployment name). The value is stored as `AIDLC_EMBED_MODEL_ID` (or `AIDLC_EMBED_MODEL_ID_<PROVIDER>` for per-provider overrides).

### 4. Optional Computation Accelerators

*   If packages in `requirements-optional.txt` cannot be compiled or loaded on your machine, CairnIQ skips them and defaults to standard NumPy/CPU operations, preserving system execution.

---

## ⚙️ Advanced Configuration (Environment Variables)

The Guided Setup Wizard covers everything most users need. Beyond it, CairnIQ exposes a set of optional tuning knobs for power users. **Every value below is optional — leaving it unset reproduces the default behavior.** Set them in `user_data/.env` (or, where noted, in the in-app **Settings** page). A full annotated list lives in [`.env.example`](../../.env.example).

> [!TIP]
> Edit `user_data/.env` and restart CairnIQ for changes to take effect. Secrets (API keys) belong in the OS keychain — the wizard puts them there automatically; only non-secret knobs like the ones below are meant to be hand-edited in `.env`.

### Reasoning Depth ("Max Think")

One provider-agnostic knob maps to each backend's native reasoning mechanism (Bedrock `budget_tokens`, Anthropic adaptive thinking, Azure/OpenAI `reasoning_effort`, Google `thinking_budget`). There is **no prompt keyword** (e.g. "ultrathink") — depth is a request parameter. Levels: `off | low | medium | high | max`.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `AIDLC_REASONING_EFFORT` | `off` | Reasoning depth for **both** model tiers |
| `AIDLC_REASONING_EFFORT_PRIMARY` | *(inherits above)* | Per-tier override for the primary / Deep Reasoning model |
| `AIDLC_REASONING_EFFORT_FAST` | *(inherits above)* | Per-tier override for the planner / fast model |
| `AIDLC_REASONING_EXTRA_BODY` | *(empty)* | Escape hatch — raw JSON merged into the request body for OpenAI-compatible gateways (self-hosted vLLM/SGLang) that want a non-standard shape. When set, it **replaces** `reasoning_effort`. Azure AI Foundry's MaaS gateway rejects `chat_template_kwargs`, so leave this empty there and use the plain effort knob. |

### Output Token Budgets

`max_tokens` is a **ceiling, not a target** — you only pay for tokens actually emitted. With reasoning enabled, the chain-of-thought shares this budget, so defaults leave headroom. DeepSeek-V4-Pro / Kimi-K2.6 accept up to `131072`.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `AIDLC_MAX_TOKENS` *(alias `AIDLC_MAX_TOKENS_PRIMARY`)* | `16384` | Primary-tier output ceiling |
| `AIDLC_MAX_TOKENS_FAST` | `8192` | Fast-tier output ceiling |
| `AIDLC_MAX_TOKENS_DEEP` | `32768` | Deep Reasoning conclusion synthesis (multi-section verdicts) |
| `AIDLC_EMBED_BATCH_SIZE` | `96` | Texts per embeddings request. Azure Foundry MaaS embed deployments cap a request at 96; lower only if a deployment has a smaller limit. |

### Cost Tracking (optional)

Token usage is **always** tracked (accurate for any model). Estimated cost is shown only when you set a price per slot — USD per **1M tokens**, formatted `input/output[/cache]`. Leave unset to display tokens only.

| Variable | Example | Applies to |
| :--- | :--- | :--- |
| `AIDLC_PRICE_PRIMARY` | `5/25` | Primary-slot model |
| `AIDLC_PRICE_FAST` | `0.15/0.60` | Fast-slot model |
| `AIDLC_PRICE_EMBED` | `0.02/0` | Embedding model |

### Azure — Separate Fast Resource (optional)

By default the fast slot shares the primary Azure resource/key. Pin a separate one only if you want the fast model on a different deployment or quota pool.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `AZURE_OPENAI_ENDPOINT_FAST` / `AZURE_OPENAI_API_KEY_FAST` | *(shares primary)* | Dedicated resource + key for the fast slot |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | Only needed with a **bare** resource URL instead of `…/openai/v1`. Newer gpt-5.x / o-series models require a recent version. |

### System, Logging & Locale

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8000` | HTTP port the server binds |
| `LOG_LEVEL` | `INFO` | Log verbosity. `DEBUG` also enables developer mode (verbose tracebacks). |
| `BASE_CURRENCY` *(alias `CAIRNIQ_BASE_CURRENCY`)* | *(derived from locale; `CAD` fallback)* | Currency for valuations and portfolio math (`USD`, `CAD`, `EUR`, `GBP`, …) |
| `REGIONAL_LOCALE` | `English (Canada)` | Number/format locale for the dashboard |

### Multi-User Auth & Network Exposure (advanced)

Off by default — single-user local setups need none of this. Turn it on only once login accounts exist (`scripts/cairniq_user.py`) and you want to reach the server from another device (e.g. an iPhone over your VPN/LAN). See also [SECURITY.md](../../SECURITY.md).

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CAIRNIQ_HOST` | `127.0.0.1` | Bind address. Set `0.0.0.0` (or a specific LAN/VPN address) to reach CairnIQ from other devices — **enable auth first**. |
| `CAIRNIQ_AUTH_REQUIRED` | `false` | When `true`, every protected route requires a valid token. Only enable once a login UI/accounts exist. |
| `CAIRNIQ_TOKEN_TTL` | `2592000` | Access-token lifetime in seconds (default 30 days) |
| `CAIRNIQ_JWT_SECRET` | *(auto-generated in keychain)* | HS256 signing secret. Set explicitly only to share one secret across hosts; never commit a real value. |

### Private / Local-Only Features

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CAIRNIQ_ENABLE_GURU_PICKS` / `CAIRNIQ_GURU_PICKS_TOOL` | `false` / *(empty)* | Gate for private guru-picks tooling not shipped in source-available builds. Leave disabled unless you have the tool installed locally. |

---

## 📁 Portfolio Setup

Before launching, populate your portfolio so the dashboard, allocation views, and AI agents have something to analyze.

### Option A — Edit the example CSV (recommended for first-time users)

```bash
cp my_portfolio.example.csv user_data/my_portfolio.csv
```

Then edit `user_data/my_portfolio.csv` with your real holdings. The CSV schema is:

```csv
Symbol,Shares,Purchase Price,Account
AAPL,25,150.00,TFSA
MSFT,15,300.00,RRSP
VTI,50,200.00,Taxable
GOOGL,10,140.00,TFSA
```

- **Symbol** — Ticker (US, Canadian `.TO`, etc.)
- **Shares** — Quantity held
- **Purchase Price** — Average cost basis per share (in your base currency)
- **Account** — Free-text grouping label (`TFSA`, `RRSP`, `401k`, `IRA`, `Taxable`, etc.)

### Option B — Upload via the UI

Once the app is running, navigate to **Portfolio Editor** in the left sidebar and use the **Upload CSV** button. The page also exposes a **Download Template** button if you want the latest schema.

### Option C — Sync from a brokerage

If you connected Alpaca or Questrade during the Guided Setup Wizard, holdings are pulled automatically. You can still keep a manual CSV alongside brokerage holdings — they are merged.

---

## 🚀 Launching CairnIQ

CairnIQ ships with a desktop launcher and supports manual command-line startup.

### Option A — Desktop Launcher (Recommended)

- **macOS / Linux**: Double-click `CairnIQ.command` (on your Desktop after install, or in the project root).
- **Windows**: Double-click `CairnIQ.bat` in the project root.

The launcher activates the venv, runs the requirements check, and opens your browser to `http://localhost:8000` automatically.

### Option B — Command Line

```bash
# macOS / Linux
./CairnIQ.command

# Windows (Command Prompt or PowerShell)
.\CairnIQ.bat
```

### Option C — Manual Start (Power Users)

```bash
# macOS / Linux
source .venv/bin/activate
python server.py

# Windows
.venv\Scripts\activate
python server.py
```

Useful for piping logs, attaching a debugger, or running under a process supervisor.

### Demo Mode

To explore CairnIQ with synthetic data and no real credentials:

```bash
./start_demo.sh        # macOS / Linux
```

Demo mode forces `ALPACA_PAPER_MODE=true`, disables Questrade, and uses `demo_portfolio.example.csv`. See [LAUNCHER_MODES.md](../LAUNCHER_MODES.md) for the full feature isolation matrix.

---

## 🔍 System Verification & Diagnostics

To verify your installation and trace key permissions:

1.  Launch CairnIQ using your preferred launcher (e.g., `./CairnIQ.command` or `CairnIQ.bat`).
2.  Open the web interface at `http://localhost:8000`.
3.  Navigate to the Chat tab and type:
    ```text
    Run a full system health check
    ```
4.  The health suite checks:
    *   AI Provider initialization and access tokens.
    *   Financial data API response status and network latency.
    *   Vector database retrieval systems.
    *   Data storage file permissions.

You will receive an execution table detailing which services are fully functional and which ones are currently operating on fallback strategies.

---

## 🔧 Troubleshooting & Maintenance

| Need | Where to look |
| :--- | :--- |
| Common install / startup errors | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Daily usage walkthrough | [USER_GUIDE.md](USER_GUIDE.md) |
| Demo vs production launch modes | [LAUNCHER_MODES.md](../LAUNCHER_MODES.md) |
| Codebase / agent architecture | [ARCHITECTURE.md](../technical/ARCHITECTURE.md) |
| What changed between versions | [CHANGELOG.md](../CHANGELOG.md) |

### Updating

```bash
git pull
./install.sh        # macOS / Linux — recompiles the venv if requirements changed
.\install.ps1       # Windows
```

Your `user_data/` directory is preserved (a fresh tarball is dropped into `backups/` as a safety net before any destructive change).

### Uninstalling

```bash
rm -rf .venv                          # virtual environment
rm -rf logs/ backups/                 # optional: drop logs and backup archives
# user_data/ holds your portfolio + chat history — delete deliberately if you want a clean slate
rm -rf user_data/
```

Secrets stored in the OS keychain are NOT removed by deleting the project directory. To purge them, open the system Keychain UI (Keychain Access on macOS, Credential Manager on Windows) and delete the `cairniq` entries.
