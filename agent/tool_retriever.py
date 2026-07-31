import hashlib
import os
import re
import shutil
import threading
import time as _time
from pathlib import Path

from langchain_core.tools import BaseTool

try:
    import faiss as _faiss_lib  # noqa: F401 — verify native lib first
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except (ImportError, Exception):
    FAISS = None
    _FAISS_AVAILABLE = False
from typing import Any

from langchain_community.retrievers import BM25Retriever

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except ModuleNotFoundError:
    from langchain_classic.retrievers import EnsembleRetriever

from agent.logger import log_event
from agent.tool_registry import ALL_TOOLS
from agent.utils import get_embedding_identity, get_embeddings, safe_print

# On-disk cache for the dense (FAISS) tool index. Rebuilding the index re-embeds
# every tool doc through the provider API on each server start — with hot-reload
# restarts that meant most queries ran before the dense index was warm (BM25-only).
# The persisted index is fingerprinted on provider + embed model + tool docs, so
# changing the LLM provider, the embedding model, or any tool description
# invalidates it automatically and forces a clean rebuild.
# Anchored to the repo root via __file__ (same convention as tools/daily_cache.py)
# so the cache always lands in the project's user_data/ regardless of the CWD the
# server was launched from.
_TOOL_INDEX_DIR = Path(__file__).resolve().parent.parent / "user_data" / "tool_index"
_TOOL_INDEX_FINGERPRINT_FILE = "fingerprint.txt"
# After a dense-index init failure (e.g. a transient embeddings 400/outage at
# boot), wait this long before a query is allowed to retry it — so a recovered
# endpoint self-heals to FAISS+BM25 instead of staying BM25-only until restart,
# without hammering a still-down endpoint on every query.
_DENSE_RETRY_COOLDOWN_SEC = 120

# Wall-clock anchor for the dense-init diagnostics below: lets a failure log say
# whether it happened seconds into startup (secrets not yet loaded) or hours in.
_MODULE_IMPORTED_AT = _time.monotonic()

# Credential names each provider's embeddings path actually reads, used only to
# report presence/absence when dense init fails. Mirrors get_embeddings() in
# agent/utils.py — keep in sync if a provider's auth inputs change.
_PROVIDER_CREDENTIAL_KEYS = {
    "vertexai": ["GOOGLE_SERVICE_ACCOUNT_KEY", "GOOGLE_CLOUD_PROJECT"],
    "google": ["GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "azure": [
        "AZURE_OPENAI_API_KEY_EMBEDDING", "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT_EMBEDDING", "AZURE_OPENAI_ENDPOINT",
    ],
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
}


def _dense_credential_diagnostics() -> dict[str, Any]:
    """Non-secret snapshot of what the dense-init thread can see for credentials.

    Every value describes only presence or shape — never a secret's contents —
    so this is safe to persist to the plaintext JSONL logs.

    Exists because dense init has failed inside the server with
    "GOOGLE_SERVICE_ACCOUNT_KEY is not set" while the chat path resolved the very
    same key fine in the same process. The fields separate the candidate causes:
    a startup race (low uptime + env empty), a keychain the background thread
    can't reach (env empty + keychain empty + backend unavailable), or an
    env-only view that disagrees with the keychain (env empty, keychain present).
    """
    diag: dict[str, Any] = {
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "uptime_sec": round(_time.monotonic() - _MODULE_IMPORTED_AT, 1),
        "provider_env": os.environ.get("LLM_PROVIDER", ""),
        "active_profile": os.environ.get("ACTIVE_PROFILE", ""),
        "pytest_env": "PYTEST_CURRENT_TEST" in os.environ,
        "keyring_disabled_env": os.environ.get("CAIRNIQ_DISABLE_KEYRING") == "1",
    }

    try:
        from tools.secrets_store import KEYRING_SERVICE, keyring_status
        diag["keyring"] = keyring_status()
    except Exception as e:  # noqa: BLE001 — diagnostics must never raise
        diag["keyring"] = {"error": type(e).__name__}
        KEYRING_SERVICE = None

    provider = (os.environ.get("LLM_PROVIDER") or "bedrock").lower()
    sources: dict[str, Any] = {}
    for name in _PROVIDER_CREDENTIAL_KEYS.get(provider, []):
        entry = {"in_env": bool(os.environ.get(name)), "in_keychain": None}
        # Probe the keychain directly rather than through get_secret(), which
        # short-circuits on env and so can't tell the two sources apart.
        if KEYRING_SERVICE:
            try:
                import keyring as _kr
                entry["in_keychain"] = bool(_kr.get_password(KEYRING_SERVICE, name))
            except Exception as e:  # noqa: BLE001
                entry["in_keychain"] = f"error:{type(e).__name__}"
        sources[name] = entry
    diag["credential_sources"] = sources
    return diag

# Tool relationship graph for "Graph-RAG Lite" retrieval.
# Maps a primary tool name to a list of related tool names.
#
# EVERY name here must be a REGISTERED tool (agent.tool_registry.ALL_TOOLS), not
# the implementation function behind one. As first written the table named the
# `tools/` functions — `detect_patterns`, `calculate_var`, `get_news_sentiment` —
# and the registry wrapped them under different names. `_expand_with_relationships`
# skips an unknown name silently, so the rot was invisible: measured 2026-07-27,
# only 55 of 154 edges had both ends registered, i.e. the one-hop expansion was
# running at roughly a third of its apparent reach. Each dead name below was
# repointed at the registered tool that actually wraps it, and dropped where that
# wrapper was the key itself (a self-edge expands to nothing) or where the entry
# was already present. tests/test_agent/test_tool_retriever.py enforces this.
TOOL_RELATIONSHIPS = {
    # --- MACRO CLUSTER ---
    "get_macro_overview": ["get_macro_strategy", "get_global_indices", "get_economic_calendar_tool", "get_canada_macro"],
    "get_macro_strategy": ["get_macro_overview", "generate_future_forecast", "get_global_indices", "get_canada_macro"],
    "get_economic_calendar_tool": ["get_market_calendar", "get_market_headlines", "get_global_indices"],
    "get_canada_macro": ["get_boc_vs_fed", "get_macro_overview", "get_macro_strategy", "check_fx_impact"],
    "get_boc_vs_fed": ["get_canada_macro", "get_macro_overview", "analyze_fx_risks", "check_fx_impact"],

    # --- PORTFOLIO CLUSTER ---
    "get_portfolio_snapshot": ["get_portfolio_risk_metrics", "check_portfolio_correlation", "get_portfolio_sectors", "analyze_fx_risks"],
    "get_portfolio_risk_metrics": ["get_value_at_risk", "check_risk_metrics", "analyze_factor_exposures"],
    "check_portfolio_correlation": ["get_correlation_analysis", "assess_marginal_trade_risk", "get_portfolio_sectors"],
    "get_portfolio_sectors": ["check_portfolio_allocation"],
    "analyze_fx_risks": ["check_fx_impact"],
    "check_tax_loss_harvesting": ["get_portfolio_snapshot"],
    "check_asset_location": ["get_portfolio_snapshot", "check_tax_loss_harvesting"],
    "get_event_radar": ["get_portfolio_snapshot", "get_earnings_calendar"],
    "get_etf_flows": ["get_portfolio_snapshot"],
    "get_portfolio_reconciliation": ["get_portfolio_snapshot"],
    "project_portfolio_income": ["get_dividend_data", "get_portfolio_snapshot"],

    # --- TECHNICALS CLUSTER ---
    "analyze_technicals": ["plot_chart", "analyze_patterns", "get_support_resistance", "get_ma_signals", "run_technical_analysis"],
    "analyze_patterns": ["get_support_resistance", "get_ma_signals", "visualize_stock_chart"],
    "get_support_resistance": ["plot_chart", "analyze_technical_chart"],
    "run_technical_analysis": ["analyze_technicals", "get_ma_signals", "get_seasonality_data", "analyze_technical_chart"],
    "scan_technical_breakouts": ["scan_opportunities", "scan_intraday_movers"],

    # --- INSTITUTIONAL / ALPHA CLUSTER ---
    "scan_options_chain": ["dealer_gamma_exposure"],
    "dealer_gamma_exposure": ["scan_options_chain"],
    "check_smart_money": ["get_insider_activity"],
    "analyze_crowded_trade": ["get_institutional_data", "get_short_interest_data"],
    "get_short_interest_data": ["get_insider_short_interest", "analyze_crowded_trade"],

    # --- SENTIMENT CLUSTER ---
    "get_sentiment": ["get_fear_greed", "analyze_reddit_sentiment", "get_stock_news"],
    "get_analyst_ratings": ["get_analyst_targets"],
    "analyze_reddit_sentiment": ["scan_options_chain", "get_stock_news"],
    "get_fear_greed": ["get_sentiment", "get_market_headlines"],

    # --- DISCOVERY / SCANNER CLUSTER ---
    "scan_opportunities": ["screen_stocks", "scan_intraday_movers", "find_ipos", "scan_guru_picks", "get_funnel_scorecard"],
    "scan_guru_picks": ["scan_opportunities", "get_market_headlines"],
    "get_funnel_scorecard": ["scan_opportunities", "backtest_strategy"],
    "scan_intraday_movers": ["scan_technical_breakouts", "get_market_headlines", "get_specific_news"],
    "scan_geopolitical_events": ["check_ticker_geopolitical_context", "check_supply_chain"],
    "check_supply_chain": ["check_ticker_geopolitical_context"],

    # --- PLANNING / SIM CLUSTER ---
    "run_monte_carlo_simulation": ["run_retirement_simulation", "project_retirement_goal"],
    "simulate_portfolio_rebalancing": ["get_portfolio_risk_metrics", "assess_marginal_trade_risk", "structure_trade_setup"],
    "run_stress_test": ["get_value_at_risk", "check_risk_metrics"],
    "structure_trade_setup": ["calculate_position", "get_support_resistance"],
    "backtest_strategy": ["run_technical_analysis", "simulate_portfolio_rebalancing"],

    # --- FUNDAMENTALS CLUSTER ---
    "fetch_fundamentals": ["get_fundamentals_detailed", "get_valuation_metrics"],
    "get_dividend_data": ["project_portfolio_income", "get_portfolio_risk_metrics"],
    "compare_stocks": ["get_competitors"],
    "analyze_earnings_transcript": ["check_management_tone"],
    "get_etf_holdings_data": ["get_portfolio_sectors", "check_portfolio_allocation"],

    # --- INTERNATIONAL MARKET CLUSTER ---
    "get_tsx_stock_quote": ["get_tsx_stock_analyst", "get_realtime_quote", "fetch_fundamentals"],
    "get_tsx_stock_analyst": ["get_tsx_stock_quote", "get_analyst_ratings", "get_analyst_targets"],
    "get_eu_stock_quote": ["get_eu_stock_analyst", "get_realtime_quote", "fetch_fundamentals"],
    "get_eu_stock_analyst": ["get_eu_stock_quote", "get_analyst_ratings", "get_analyst_targets"],
}


# Tokens that look ticker-like ([A-Z]{1,5}) but are almost always ordinary
# English words or finance acronyms in user queries, not symbols. Without this
# filter the pronoun "I" or words like "ETF"/"US" matched the ticker regex and
# pinned the entire six-tool stock-decision suite onto unrelated queries
# (observed: ~3 tools pinned on average across 818 logged retrievals). A few of
# these are real tickers (AI, IT, EU) — in query text they're overwhelmingly the
# acronym, and semantic retrieval still surfaces the right tools for them;
# pinning is a recall bonus, not the only path.
_TICKER_STOPWORDS = {
    "I", "A", "OK", "US", "USA", "UK", "EU", "AI", "IT", "TV", "PC", "ESG",
    "CEO", "CFO", "IPO", "ETF", "ETFS", "GDP", "EPS", "PE", "PB", "PS",
    "ROI", "ROE", "RSI", "YTD", "FY", "FX", "AM", "PM", "FAQ", "LLM", "API",
    "USD", "CAD", "EUR", "GBP", "JPY", "TSX", "NYSE", "NASDA", "DOW",
    "TFSA", "RRSP", "RESP", "IRA", "DCA", "VAR", "ATR", "MACD",
}


def _query_mentions_ticker(query: str) -> bool:
    """True when the query contains a plausible ticker symbol, ignoring
    uppercase tokens that are really pronouns/acronyms (see _TICKER_STOPWORDS)."""
    for match in re.finditer(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", query or ""):
        if match.group().split(".")[0] not in _TICKER_STOPWORDS:
            return True
    return False


def _mandatory_tool_names_for_query(query: str) -> list[str]:
    """Pin a small set of must-have tools for high-impact investment intents."""
    query_lower = (query or "").lower()
    pinned: list[str] = []

    def add_names(*tool_names: str):
        for tool_name in tool_names:
            if tool_name not in pinned:
                pinned.append(tool_name)

    broad_market_phrases = [
        "all sectors", "broad market", "market overview", "everything",
        "scan for opportunities", "hidden gems", "what to buy", "golden opportunities"
    ]
    if any(phrase in query_lower for phrase in broad_market_phrases):
        add_names("scan_sector_opportunities", "scan_geopolitical_opportunities", "check_sector_rotation")

    guru_phrases = [
        "guru", "media guru", "tv picks", "tv sentiment", "guru picks",
        "lightning round", "media recommendations"
    ]
    if any(phrase in query_lower for phrase in guru_phrases):
        add_names("scan_guru_picks")

    investment_decision_phrases = [
        "good investment", "should i buy", "about to buy", "worth buying",
        "is this a good", "invest in", "buy this stock", "buy this company"
    ]
    if any(phrase in query_lower for phrase in investment_decision_phrases) or _query_mentions_ticker(query):
        add_names("run_stock_deep_dive", "predict_surprise", "get_earnings_data", "get_insider_activity", "structure_trade_setup", "assess_marginal_trade_risk")

    portfolio_risk_phrases = [
        "portfolio", "allocation", "risk assessment", "portfolio risk",
        "position size", "fit my portfolio"
    ]
    if any(phrase in query_lower for phrase in portfolio_risk_phrases):
        add_names("assess_portfolio_risk", "get_portfolio_snapshot", "assess_marginal_trade_risk")

    # International market detection
    if ".to" in query_lower or ".vn" in query_lower or "tsx" in query_lower:
        add_names("get_tsx_stock_quote", "get_tsx_stock_analyst")
    european_suffixes = [".l ", ".de", ".pa", ".as", ".mi", ".mc", ".sw", ".st"]
    if any(suffix in query_lower for suffix in european_suffixes) or "lse" in query_lower or "xetra" in query_lower or "euronext" in query_lower:
        add_names("get_eu_stock_quote", "get_eu_stock_analyst")

    return pinned


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _preview_tool_names(tool_names: list[str], max_items: int = 6) -> str:
    if not tool_names:
        return "none"
    preview = tool_names[:max_items]
    suffix = "" if len(tool_names) <= max_items else ", ..."
    return ", ".join(preview) + suffix


def format_tool_retrieval_status(metadata: dict[str, Any], label: str = "Tool Router") -> str:
    """Create a short human-readable status line for the UI trace."""
    tool_names = metadata.get("selected_tool_names", [])
    elapsed_ms = metadata.get("elapsed_ms", 0)
    strategy = metadata.get("strategy", "unknown")
    dense_status = metadata.get("dense_status", "unknown")
    warm_note = ""
    if dense_status == "initializing":
        warm_note = " Dense index warming in background."
    elif dense_status == "failed":
        warm_note = " Dense index unavailable."
    return (
        f"🧰 {label}: Selected {metadata.get('tool_count', 0)} candidates "
        f"in {elapsed_ms}ms via {strategy}. Top: {_preview_tool_names(tool_names)}."
        f"{warm_note}"
    )

class ToolRetriever:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def reset(cls):
        """Drop the singleton so the next ToolRetriever() rebuilds against the
        current provider's embeddings.

        Used on a runtime LLM_PROVIDER / embed-model change: the dense (FAISS)
        index is provider-specific, so the in-memory retriever must be rebuilt
        rather than keep serving vectors from the previous embedding backend. Any
        in-flight dense-init daemon thread on the old instance is left to finish
        harmlessly; callers always fetch a fresh ToolRetriever().
        """
        cls._instance = None

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self.all_tools = ALL_TOOLS
        self.tool_map = {t.name: t for t in self.all_tools}
        self.tool_docs = []
        self.tool_doc_lookup = {}
        self.tool_name_tokens = {}
        self.tool_description_tokens = {}
        self.query_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
        self.bm25_retriever = None
        self.faiss_retriever = None
        self.ensemble_retriever = None
        self._dense_status = "not_started"
        self._dense_error = None
        self._dense_failed_at = 0.0
        self._dense_thread = None
        self._init_lock = threading.Lock()

        self._prepare_tool_documents()
        self._initialize_sparse_retriever()
        self._start_dense_initialization()

    def _prepare_tool_documents(self):
        """Build cheap local indexes synchronously so the first request stays fast."""
        for tool in self.all_tools:
            content = f"Tool Name: {tool.name}\nDescription: {tool.description}"
            self.tool_docs.append(content)
            self.tool_doc_lookup[tool.name] = content
            self.tool_name_tokens[tool.name] = set(_tokenize_text(tool.name.replace("_", " ")))
            self.tool_description_tokens[tool.name] = set(_tokenize_text(tool.description))

    def _initialize_sparse_retriever(self):
        safe_print("🔧 Initializing Tool-RAG Semantic Router...")

        try:
            self.bm25_retriever = BM25Retriever.from_texts(self.tool_docs)
            self.bm25_retriever.k = 15
            safe_print(f"✅ Tool-RAG initialized with BM25 for {len(self.all_tools)} tools.")
        except Exception as e:
            safe_print(f"⚠️ BM25 Initialization failed: {e}")
            self.bm25_retriever = None

    def _start_dense_initialization(self):
        """Warm the dense index in the background instead of blocking the first user request.

        'ready'/'disabled' are terminal; 'initializing' is in-flight. 'failed' is
        retried after a cooldown so a transient embeddings outage at boot doesn't
        pin the whole process to BM25-only until a manual restart.
        """
        with self._init_lock:
            status = self._dense_status
            if status in {"initializing", "ready", "disabled"}:
                return
            if status == "failed":
                if (_time.monotonic() - self._dense_failed_at) < _DENSE_RETRY_COOLDOWN_SEC:
                    return
                safe_print("🔄 Tool-RAG retrying dense index init after earlier failure...")
            self._dense_status = "initializing"
            self._dense_thread = threading.Thread(
                target=self._initialize_dense_retriever,
                name="tool-retriever-dense-init",
                daemon=True,
            )
            self._dense_thread.start()

    def _tool_index_fingerprint(self) -> str:
        """Fingerprint of everything the persisted index depends on.

        Includes the embedding backend identity (provider + embed model), so a
        Settings change to LLM_PROVIDER or the embed model invalidates the cache,
        and the full tool-doc corpus, so adding/renaming a tool or editing any
        description forces a rebuild.
        """
        payload = get_embedding_identity() + "\x00" + "\x00".join(self.tool_docs)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_persisted_index(self, embeddings, fingerprint: str):
        """Return the cached FAISS store if its fingerprint matches, else None.

        On any mismatch or load failure the stale cache directory is removed so
        an index built against a different provider/embed model never lingers.
        """
        fp_path = _TOOL_INDEX_DIR / _TOOL_INDEX_FINGERPRINT_FILE
        try:
            if not fp_path.exists():
                return None
            if fp_path.read_text().strip() != fingerprint:
                shutil.rmtree(_TOOL_INDEX_DIR, ignore_errors=True)
                safe_print("🔄 Tool-RAG index cache invalidated (provider/embed model/tools changed) — rebuilding.")
                return None
            # Our own file written by _persist_index — pickle deserialization is safe.
            store = FAISS.load_local(
                str(_TOOL_INDEX_DIR), embeddings,
                allow_dangerous_deserialization=True,
            )
            # Dimension guard: the fingerprint keys on the embed model NAME, not its
            # vector dimension. A same-name deployment rebuilt at a different dim
            # (e.g. swapping the underlying model behind 'Cohere-embed-v3-english')
            # loads clean here, then throws a bare AssertionError at query time while
            # _dense_status stays 'ready'. Probe the live dim and rebuild on mismatch.
            try:
                live_dim = len(embeddings.embed_query("tool-rag dimension probe"))
                index_dim = getattr(getattr(store, "index", None), "d", None)
                if index_dim is not None and live_dim != index_dim:
                    safe_print(
                        f"🔄 Tool-RAG cached index dim ({index_dim}) ≠ current embed dim "
                        f"({live_dim}) — rebuilding."
                    )
                    shutil.rmtree(_TOOL_INDEX_DIR, ignore_errors=True)
                    return None
            except Exception as probe_err:
                # Can't probe (e.g. endpoint hiccup) → don't nuke a possibly-good
                # cache; keep current behavior and let query-time surface any issue.
                safe_print(f"ℹ️ Tool-RAG dim check skipped ({probe_err}); using cached index.")
            return store
        except Exception as e:
            safe_print(f"⚠️ Tool-RAG index cache unreadable ({e}) — rebuilding.")
            shutil.rmtree(_TOOL_INDEX_DIR, ignore_errors=True)
            return None

    def _persist_index(self, vectorstore, fingerprint: str) -> None:
        """Best-effort save; a failure here only costs a rebuild next start."""
        try:
            _TOOL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(_TOOL_INDEX_DIR))
            (_TOOL_INDEX_DIR / _TOOL_INDEX_FINGERPRINT_FILE).write_text(fingerprint)
            safe_print(f"💾 Tool-RAG dense index persisted to {_TOOL_INDEX_DIR}.")
        except Exception as e:
            safe_print(f"⚠️ Could not persist Tool-RAG index: {e}")

    def _initialize_dense_retriever(self):
        if not _FAISS_AVAILABLE:
            safe_print("ℹ️ FAISS not installed — using BM25 keyword retrieval only. Install with: pip install faiss-cpu")
            self._dense_status = "disabled"
            return

        # Resolve the embeddings object for the active LLM provider. Returns None
        # for providers with no native embeddings API (e.g. Anthropic), and raises
        # on misconfiguration (missing keys / missing package).
        try:
            embeddings = get_embeddings()
        except Exception as e:
            safe_print(f"⚠️ FAISS/Embeddings Initialization failed: {e}")
            self._dense_status = "failed"
            self._dense_error = str(e)
            self._dense_failed_at = _time.monotonic()
            log_event("ToolRetriever", "Dense semantic index failed", {
                "error": str(e),
                "has_bm25": bool(self.bm25_retriever),
                "diagnostics": _dense_credential_diagnostics(),
            })
            if not self.bm25_retriever:
                safe_print("⚠️ Tool-RAG using local keyword fallback only.")
            return

        if embeddings is None:
            safe_print("ℹ️ Dense embeddings not available for the current LLM provider — using BM25 keyword retrieval only.")
            self._dense_status = "disabled"
            return

        try:
            fingerprint = self._tool_index_fingerprint()
            vectorstore = self._load_persisted_index(embeddings, fingerprint)
            loaded_from_disk = vectorstore is not None

            if vectorstore is None:
                vectorstore = FAISS.from_texts(self.tool_docs, embeddings)
                self._persist_index(vectorstore, fingerprint)

                # Track embedding cost for initial index build (skipped on disk load)
                try:
                    from agent.cost_tracker import track_embedding_cost
                    total_chars = sum(len(doc) for doc in self.tool_docs)
                    track_embedding_cost(max(total_chars // 4, 1))
                except Exception:
                    pass

            self.faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
            if loaded_from_disk:
                safe_print("⚡ Tool-RAG dense index loaded from disk cache (no re-embedding).")

            if self.bm25_retriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.faiss_retriever],
                    weights=[0.4, 0.6]
                )
                safe_print(f"✅ Tool-RAG upgraded to Ensemble (FAISS + BM25) for {len(self.all_tools)} tools.")
            else:
                self.ensemble_retriever = self.faiss_retriever
                safe_print(f"✅ Tool-RAG initialized with FAISS only for {len(self.all_tools)} tools.")

            self._dense_status = "ready"
            self.query_cache.clear()
            log_event("ToolRetriever", "Dense semantic index ready", {
                "tool_count": len(self.all_tools),
                "has_bm25": bool(self.bm25_retriever),
                "loaded_from_disk": loaded_from_disk,
            })
        except Exception as e:
            safe_print(f"⚠️ FAISS/Embeddings Initialization failed: {e}")
            self._dense_status = "failed"
            self._dense_error = str(e)
            self._dense_failed_at = _time.monotonic()
            log_event("ToolRetriever", "Dense semantic index failed", {
                "error": str(e),
                "has_bm25": bool(self.bm25_retriever),
                "diagnostics": _dense_credential_diagnostics(),
            })
            if not self.bm25_retriever:
                safe_print("⚠️ Tool-RAG using local keyword fallback only.")

    def _expand_with_relationships(self, primary_tools: list[BaseTool], max_expansion: int = 5) -> list[BaseTool]:
        """
        One-hop expansion using the TOOL_RELATIONSHIPS graph.
        If a tool like 'fetch_fundamentals' is found, we also pull in related tools.
        """
        expanded_tools = list(primary_tools)
        seen_names = {t.name for t in primary_tools}
        added_count = 0

        for tool in primary_tools:
            if tool.name in TOOL_RELATIONSHIPS:
                related_names = TOOL_RELATIONSHIPS[tool.name]
                for r_name in related_names:
                    if r_name not in seen_names and r_name in self.tool_map:
                        expanded_tools.append(self.tool_map[r_name])
                        seen_names.add(r_name)
                        added_count += 1
                        if added_count >= max_expansion:
                            return expanded_tools
        return expanded_tools

    def _pin_mandatory_tools(self, query: str, retrieved_tools: list[BaseTool]) -> list[BaseTool]:
        """Guarantee high-value tools are available for known critical intents."""
        pinned_tools = list(retrieved_tools)
        seen_names = {tool.name for tool in retrieved_tools}

        for tool_name in _mandatory_tool_names_for_query(query):
            if tool_name in self.tool_map and tool_name not in seen_names:
                pinned_tools.append(self.tool_map[tool_name])
                seen_names.add(tool_name)

        return pinned_tools

    def _keyword_rank_tools(self, query: str, k: int) -> list[BaseTool]:
        """Fast local fallback when semantic retrievers are unavailable or still warming."""
        query_lower = (query or "").lower()
        query_tokens = set(_tokenize_text(query))
        scored_tools = []

        for index, tool in enumerate(self.all_tools):
            name_tokens = self.tool_name_tokens.get(tool.name, set())
            description_tokens = self.tool_description_tokens.get(tool.name, set())
            score = 0

            if tool.name.lower() in query_lower:
                score += 50

            score += len(query_tokens & name_tokens) * 12
            score += len(query_tokens & description_tokens) * 3

            if "portfolio" in query_tokens and "portfolio" in description_tokens:
                score += 6
            if "market" in query_tokens and "market" in description_tokens:
                score += 4
            if "news" in query_tokens and "news" in description_tokens:
                score += 4

            scored_tools.append((score, index, tool))

        scored_tools.sort(key=lambda item: (-item[0], item[1]))
        ranked = [tool for score, _, tool in scored_tools if score > 0]
        if len(ranked) < k:
            fallback_names = {tool.name for tool in ranked}
            for tool in self.all_tools:
                if tool.name not in fallback_names:
                    ranked.append(tool)
                if len(ranked) >= k:
                    break
        return ranked[:k]

    def _tools_from_docs(self, docs: list[Any], k: int) -> list[BaseTool]:
        retrieved_tools = []
        seen = set()

        for doc in docs:
            first_line = doc.page_content.split('\n')[0]
            if "Tool Name:" in first_line:
                t_name = first_line.replace("Tool Name:", "").strip()
                if t_name in self.tool_map and t_name not in seen:
                    retrieved_tools.append(self.tool_map[t_name])
                    seen.add(t_name)
                    if len(retrieved_tools) >= k:
                        break
        return retrieved_tools

    def get_tools_for_query(self, query: str, k: int = 20) -> list[BaseTool]:
        """
        Dynamically retrieve tools using semantic search + relationship graph expansion.
        """
        tools, _ = self.get_tools_for_query_with_metadata(query, k)
        return tools

    def get_tools_for_query_with_metadata(self, query: str, k: int = 20):
        self._start_dense_initialization()
        cache_key = ((query or "").strip().lower(), k, self._dense_status)
        cached = self.query_cache.get(cache_key)
        if cached:
            cached_metadata = dict(cached["metadata"])
            cached_metadata["cache_hit"] = True
            cached_metadata["elapsed_ms"] = 0
            return list(cached["tools"]), cached_metadata

        start_time = _time.perf_counter()
        strategy = "keyword_fallback"
        cache_hit = False
        dense_status = self._dense_status

        try:
            if self.ensemble_retriever:
                strategy = "ensemble"
                docs = self.ensemble_retriever.invoke(query)
                # Track embedding cost (~1 embed call per query, estimate tokens from chars)
                try:
                    from agent.cost_tracker import track_embedding_cost
                    track_embedding_cost(max(len(query) // 4, 1))
                except Exception:
                    pass
                retrieved_tools = self._tools_from_docs(docs, k)
            elif self.bm25_retriever:
                strategy = "bm25"
                docs = self.bm25_retriever.invoke(query)
                retrieved_tools = self._tools_from_docs(docs, k)
            else:
                retrieved_tools = self._keyword_rank_tools(query, k)

            if not retrieved_tools:
                strategy = "keyword_fallback"
                retrieved_tools = self._keyword_rank_tools(query, k)

            expanded_tools = self._expand_with_relationships(retrieved_tools, max_expansion=5)
            expanded_tools = self._pin_mandatory_tools(query, expanded_tools)

            if len(expanded_tools) < 5:
                existing_names = {tool.name for tool in expanded_tools}
                for fallback_tool in self._keyword_rank_tools(query, max(k, 5)):
                    if fallback_tool.name not in existing_names:
                        expanded_tools.append(fallback_tool)
                        existing_names.add(fallback_tool.name)
                    if len(expanded_tools) >= 5:
                        break

            selected_names = [tool.name for tool in expanded_tools]
            metadata = {
                "strategy": strategy,
                "elapsed_ms": int((_time.perf_counter() - start_time) * 1000),
                "dense_status": dense_status,
                "cache_hit": cache_hit,
                "tool_count": len(expanded_tools),
                "selected_tool_names": selected_names,
                "selected_tool_preview": _preview_tool_names(selected_names),
                "pinned_tool_names": [
                    name for name in _mandatory_tool_names_for_query(query)
                    if name in selected_names
                ],
            }
            self.query_cache[((query or "").strip().lower(), k, dense_status)] = {
                "tools": list(expanded_tools),
                "metadata": dict(metadata),
            }
            log_event("ToolRetriever", "Selected candidate tools", metadata)
            return expanded_tools, metadata
        except Exception as e:
            safe_print(f"⚠️ Tool Retrieval Error: {e}. Falling back to default list.")
            fallback_tools = self._keyword_rank_tools(query, min(k, len(self.all_tools)))
            metadata = {
                "strategy": "keyword_fallback_after_error",
                "elapsed_ms": int((_time.perf_counter() - start_time) * 1000),
                "dense_status": dense_status,
                "cache_hit": cache_hit,
                "tool_count": len(fallback_tools),
                "selected_tool_names": [tool.name for tool in fallback_tools],
                "selected_tool_preview": _preview_tool_names([tool.name for tool in fallback_tools]),
                "error": str(e),
                "pinned_tool_names": [],
            }
            log_event("ToolRetriever", "Retriever fallback after error", metadata)
            return fallback_tools, metadata

# Singleton accessor
def get_semantic_tools(query: str, k: int = 20) -> list[BaseTool]:
    retriever = ToolRetriever()
    return retriever.get_tools_for_query(query, k)


def get_semantic_tools_with_metadata(query: str, k: int = 20):
    retriever = ToolRetriever()
    return retriever.get_tools_for_query_with_metadata(query, k)

def get_tool_directory_string() -> str:
    """Returns a highly condensed catalog of all available tools for the Deep Reasoning agent."""
    import re

    from agent.tool_registry import ALL_TOOLS
    lines = []
    for tool in ALL_TOOLS:
        desc = getattr(tool, "description", "").split("\n")[0]
        # Clean up description
        desc = re.sub(r'^\s*-\s*', '', desc)
        lines.append(f"- {tool.name}: {desc}")
    return "\n".join(lines)
