import collections
import hashlib
import os
import re
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

STATUS_CALLBACK = None
STREAM_CALLBACK = None


@dataclass
class RunContext:
    """Per-run streaming and cancellation state for an active chat."""
    on_token: Callable[[str], None] | None = None
    # Called as on_status(msg, degraded=False); degraded=True is a deliberate
    # degradation signal (see send_status). Callable[..., None] so the optional
    # kwarg type-checks.
    on_status: Callable[..., None] | None = None
    # Reasoning-trace sink. Separate from on_token so a model's thought blocks
    # reach the UI's trace panel without ever mixing into the visible answer —
    # the spill that made filtering them necessary in the first place.
    on_thinking: Callable[[str], None] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    notices: list[str] = field(default_factory=list)


_CURRENT_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar("current_run_context", default=None)

def set_stream_callbacks(on_token=None, on_status=None):
    """Register global callbacks for real-time status and token updates."""
    global STATUS_CALLBACK, STREAM_CALLBACK
    if on_token: STREAM_CALLBACK = on_token
    if on_status: STATUS_CALLBACK = on_status


def build_run_context(on_token=None, on_status=None, cancel_event=None, on_thinking=None):
    """Create an isolated runtime context for a single chat run."""
    return RunContext(
        on_token=on_token,
        on_status=on_status,
        on_thinking=on_thinking,
        cancel_event=cancel_event or threading.Event()
    )


def activate_run_context(run_context: RunContext):
    """Bind a run context to the current thread/execution context."""
    return _CURRENT_RUN_CONTEXT.set(run_context)


def reset_run_context(token):
    """Restore the previous run context."""
    _CURRENT_RUN_CONTEXT.reset(token)


def get_run_context() -> RunContext | None:
    """Return the active per-run context for the current execution context."""
    return _CURRENT_RUN_CONTEXT.get()


def has_stream_callback() -> bool:
    """Whether the current run can stream tokens back to the caller."""
    run_context = get_run_context()
    return bool((run_context and run_context.on_token) or STREAM_CALLBACK)

def send_status(msg, *, degraded: bool = False):
    """Safely send a status update message to the UI or console.

    degraded=True marks a deliberate degradation signal: the run still produces
    a usable answer, but something was missing/timed-out/fell-back, so the chat
    header should land on DEGRADED. Default False — a benign, informational
    status never trips the badge even if its text carries a ⚠️/🚩 glyph. This
    replaces inferring degradation by regex-matching emoji in the status string,
    which false-positived on decorative glyphs (see static/js/chat.js).
    """
    run_context = get_run_context()
    callback = (run_context.on_status if run_context else None) or STATUS_CALLBACK
    if callback:
        try:
            callback(msg, degraded=degraded)
        except TypeError:
            # Legacy single-arg callback — degradation flag is dropped, but the
            # status text still reaches the UI/console.
            callback(msg)
    else:
        print(f"  [STATUS] {msg}")

def send_thinking(text):
    """Send reasoning-trace text to the UI's trace panel.

    Deliberately NOT routed through send_stream: the visible answer and the
    reasoning must stay on separate channels. Silently no-ops when the run has
    no trace sink (scheduler/background runs), so callers need no guard.
    """
    if not text:
        return
    run_context = get_run_context()
    if run_context and run_context.on_thinking:
        try:
            run_context.on_thinking(text)
        except Exception:
            # A trace is never worth failing a run over.
            pass


def send_stream(token):
    """Safely send a token stream update to the UI or console."""
    run_context = get_run_context()
    if run_context and run_context.on_token:
        run_context.on_token(token)
    elif STREAM_CALLBACK:
        STREAM_CALLBACK(token)
    else:
        # Avoid flooding console with individual tokens by default
        pass


def current_turn_key(messages) -> str:
    """Identify the current conversational turn from its message list.

    Nodes that hand evidence to a LATER node through ``data_context`` need a way
    for the reader to prove the writer wrote it THIS turn — ``data_context`` has
    no state reducer, so a key written on an earlier turn is indistinguishable
    from a fresh one by content alone. The turn's genuine user message is the
    only thing every node in one pass sees identically, so its id is the key.

    Synthetic ``<compliance_correction_required>`` directives are skipped: the
    retry gate injects one mid-turn, and a retry must resolve to the SAME key as
    the pass it is correcting or the evidence union silently drops to nothing.

    Returns "" when there is no user message to key off — an empty key never
    matches, so an unidentifiable turn falls back to the conservative path.
    """
    for msg in reversed(list(messages or [])):
        if not isinstance(msg, HumanMessage):
            continue
        text = stringify_message_content(getattr(msg, "content", ""))
        if text.lstrip().startswith("<compliance_correction_required>"):
            continue
        msg_id = getattr(msg, "id", None)
        if msg_id:
            return str(msg_id)
        # No id (a hand-built message, or state that never passed through
        # add_messages) — the content itself is the next-best identity.
        return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
    return ""


def stringify_message_content(content) -> str:
    """Normalize string/list message content into a plain text string."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in _TOOL_BLOCK_TYPES:
                    continue
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


# Content-block types that are tool-call machinery, NOT user-visible text. When an
# agentic, tool-bound LLM streams, its tool-call arguments arrive as these blocks
# (e.g. find_breakout_candidates' "NVDA, META, AVGO, ..." symbol list). They must
# never be forwarded to the UI stream — otherwise the user sees bare tickers and
# tool args ("All", "Aggressive") spill into the chat before the synthesized answer.
_TOOL_BLOCK_TYPES = {"tool_use", "tool_call", "input_json_delta", "tool_call_chunk", "thinking", "thought", "reasoning"}


# Some models (notably Kimi/Moonshot) serialize tool calls as chat-template special
# tokens — <|tool_calls_section_begin|>, <|tool_call_begin|>functions.NAME:idx,
# <|tool_call_argument_begin|>, <|tool_call_end|>, <|tool_calls_section_end|>. When
# such a model is prompted about tools with none bound, it emits those tokens as
# plain text instead of a structured tool_call, and they leak into the chat. This is
# a backstop — the real fix is to never advertise tools to a tool-less synthesis
# call — but it guarantees raw tokens never reach the UI or get persisted.
_TOOL_CALL_SPAN_RE = re.compile(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", re.DOTALL)
_TOOL_CALL_MARKER_RE = re.compile(r"<\|tool_call[a-z_]*\|>")


def strip_tool_call_tokens(text):
    """Remove model-native tool-call special tokens leaked into visible text.

    Type-safe: non-str input (e.g. a Bedrock content-block list) is returned
    unchanged so callers can wrap raw message content without a type guard.
    """
    if not isinstance(text, str) or "<|tool_call" not in text:
        return text
    text = _TOOL_CALL_SPAN_RE.sub("", text)  # drop each call (name + args) whole
    return _TOOL_CALL_MARKER_RE.sub("", text)  # drop section + any stray markers


# Every node prompt wraps its own instructions/context in snake_case pseudo-XML
# tags (<output_format>, <rules>, <role>, <market_pulse>, ...) and tells the model
# to "print only the Markdown blocks" / "omit XML tags in final output". Weaker or
# rushed completions sometimes copy a tag verbatim instead of just filling in the
# template (e.g. RiskManager's "<output_format strict=\"true\">" leaking ahead of
# the verdict). Since every synthesis prompt promises clean Markdown only, no
# legitimate answer ever contains a literal tag like this — strip the wrapper on
# sight, on any node's output, without needing to enumerate every tag name.
# ONE exemption: the watch-conditions side-channel (<watch>...</watch>, roadmap
# 3.3). It is tag-shaped but is NOT a leaked scaffold wrapper to be unwrapped --
# it carries machine-readable JSON a producer must PARSE (capture) before it is
# removed WHOLE by the dedicated tools.watch_conditions.strip_watch_blocks.
# Unwrapping it here (stripping the tags, keeping the JSON) is the 2026-07-23
# regression: it silently orphaned the JSON so capture found nothing and the raw
# object leaked into the visible brief. The '\b' keeps this exact -- <watchlist>
# and any other tag whose name merely starts with "watch" are still stripped.
_SCAFFOLD_TAG_RE = re.compile(r'</?(?!watch\b)[a-z][a-z0-9_]*(?:\s+[a-z_]+="[^"]*")*\s*>\n?', re.IGNORECASE)

# A weak completion once echoed RiskManager's tag WITHOUT the leading '<' — the
# literal text `output_format strict="true">` reached the UI because the regex
# above only recognizes the full <tag> form. Orphaned fragments are matched
# conservatively: anchored to the start of a line, either a known scaffold tag
# name from the node prompts or any snake_case name carrying at least one
# attr="..." pair, with the '>' glued on (no space before it) — so real prose
# like "margin > 20%" or "x > 5" is never eaten.
# Name list grep-derived from agent/nodes/ + agent/prompts:
#   grep -rhoE '<[a-z][a-z0-9_]*' agent/nodes/ agent/prompts* | sort -u
_SCAFFOLD_TAG_NAMES = (
    "analysis_focus|analyst_findings|boundaries|compliance_correction_required|"
    "context_review|current_conversation_summary|dashboard_data|data_boundary_rules|"
    "data_integrity|deterministic_audit|evidence_rules|execution_rules|"
    "hunter_seeker_protocol|instructions|length_instruction|lens_contract|"
    "market_pulse|mission|no_tools|objective|output_format|personalization|"
    "planner_notes|portfolio_context|portfolio_data|portfolio_verification|"
    "portfolio_verification_context|prior_verdict|recent_chat_history|"
    "recent_tool_results|rendering_constraint|report_structure|reversal_discipline|"
    "risk_flags|risk_prescreen|risk_report|role|rules|search_results|strategy_rules|"
    "structured_analysis|system|target_market_aspects|task|thinking|today|"
    "tone_and_style|tool_execution_context|tool_execution_protocol|tool_results|"
    "tool_selection_rules|user_context|user_framework|user_framework_rules|"
    "user_memory|user_portfolio_context|user_profile|user_profile_memory"
)
_ORPHAN_SCAFFOLD_TAG_RE = re.compile(
    r'^[ \t]*/?'
    r'(?:(?:' + _SCAFFOLD_TAG_NAMES + r')(?:\s+[a-z_]+="[^"]*")*'
    r'|[a-z][a-z0-9_]*(?:\s+[a-z_]+="[^"]*")+'
    r')>\n?',
    re.IGNORECASE | re.MULTILINE,
)


def strip_scaffold_tags(text):
    """Remove leaked prompt-scaffold tags (e.g. stray <output_format>/<rules>) from visible text.

    Also removes orphaned fragments that lost their leading '<' (e.g. a
    line-start `output_format strict="true">` or `/output_format>`), which the
    full-tag pattern can't see. Only strips the tag wrapper itself, keeping any
    real content between tags — unlike <thinking> blocks, whose content is
    meant to be hidden entirely. Type-safe: non-str input is returned unchanged.
    """
    if not isinstance(text, str) or ">" not in text:
        return text
    text = _SCAFFOLD_TAG_RE.sub("", text)
    return _ORPHAN_SCAFFOLD_TAG_RE.sub("", text)


def extract_stream_text(content) -> str:
    """
    Extract ONLY user-visible text from a streamed message chunk's content.

    Unlike stringify_message_content, this drops tool-call / tool-use content blocks
    (and anything that isn't an explicit text block), so streaming a tool-bound LLM
    never leaks tool arguments (ticker lists, sector/profile args) into the chat.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return strip_scaffold_tags(strip_tool_call_tokens(content))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Only forward explicit text blocks; skip tool_use/input_json_delta/etc.
                if item.get("type") in _TOOL_BLOCK_TYPES:
                    continue
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
            # Non-dict items (raw tool-call delta objects) are never visible text → skip.
        return strip_scaffold_tags(strip_tool_call_tokens("".join(parts)))
    return ""


# Reasoning-block types, as distinct from tool-call machinery. Both are filtered
# out of the visible stream by extract_stream_text, but these carry the model's
# actual chain of thought and belong in the trace panel rather than the bin.
_REASONING_BLOCK_TYPES = {"thinking", "thought", "reasoning"}


def extract_reasoning_text(content) -> str:
    """Extract ONLY the reasoning/thought text from a streamed chunk's content.

    The mirror of extract_stream_text: that function drops these blocks so the
    chain of thought never spills into the answer, this one recovers them for
    the trace panel. A plain string carries no block type, so it is never
    reasoning — returning "" there keeps the answer out of the trace.
    """
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in _REASONING_BLOCK_TYPES:
            continue
        # Providers disagree on the field name: Anthropic uses "thinking",
        # Gemini/OpenAI-style reasoning blocks use "text".
        for key in ("text", "thinking", "reasoning"):
            val = item.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
                break
    return "".join(parts)


def extract_visible_text(content, strip_node_prefix: bool = False) -> str:
    """
    Return only the user-visible portion of a model response.
    This removes internal <thinking> blocks and can optionally strip a leading
    "[NodeName]:" prefix that is used for internal bookkeeping.
    """
    text = stringify_message_content(content)
    if not text:
        return ""

    text = strip_tool_call_tokens(text)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>(?:(?!</thinking>).)*$", "", text, flags=re.DOTALL)
    text = strip_scaffold_tags(text)

    if strip_node_prefix:
        text = re.sub(r"^\[.*?\]:\s*", "", text)

    return text.strip()


# In-message privacy switches, equivalent to the Ghost toggle. Defined once here
# because more than one capture site has to honour them: the supervisor's memory
# capture and the chat router's feedback capture. A second, drifting copy of this
# list is how a turn ends up private to one store and recorded in another.
PRIVACY_TAGS = ("@Private", "@Ghost", "[Private]", "No capture")


def is_private_turn(content, state_ghost: bool = False) -> bool:
    """True when this turn must not be recorded to any persistent store."""
    if state_ghost:
        return True
    return any(tag in str(content or "") for tag in PRIVACY_TAGS)


import time
from functools import wraps

from dotenv import load_dotenv

# Load Environment Variables - CALL EARLY in this module
# This ensures that ALL constants (MODEL_ID, REGION) are correctly loaded
# even if this module is imported before load_dotenv() is called elsewhere.
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_data", ".env")
load_dotenv(_ENV_PATH, override=False)
try:
    from tools.secrets_store import (
        clear_incompatible_aws_session_token,
        load_secrets_into_env,
    )

    load_secrets_into_env()
    clear_incompatible_aws_session_token()
except Exception:
    pass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Cancellation Token (Thread-Safe) ---
# Global fallback used when code runs outside an isolated chat context.
_cancel_event = threading.Event()

def request_cancellation(cancel_event: threading.Event | None = None):
    """Signal all running tools to stop ASAP."""
    if cancel_event is not None:
        cancel_event.set()
        return

    run_context = get_run_context()
    if run_context:
        run_context.cancel_event.set()
    else:
        _cancel_event.set()

def reset_cancellation(cancel_event: threading.Event | None = None):
    """Clear the cancellation flag (call at the start of each new agent run)."""
    if cancel_event is not None:
        cancel_event.clear()
        return

    run_context = get_run_context()
    if run_context:
        run_context.cancel_event.clear()
    else:
        _cancel_event.clear()

def is_cancelled():
    """Check if cancellation has been requested. Fast, thread-safe, no-lock."""
    run_context = get_run_context()
    if run_context:
        return run_context.cancel_event.is_set()
    return _cancel_event.is_set()

# --- LLM Config ---
# Provider-aware default model IDs. Used when AIDLC_MODEL_ID is not set.
# These are sensible defaults; users can override in .env.
_PROVIDER_DEFAULT_MODELS = {
    "anthropic": {
        "primary": "claude-sonnet-4-6-20250929",
        "fast": "claude-haiku-4-6-20250929",
    },
    "openai": {
        "primary": "gpt-4o",
        "fast": "gpt-4o-mini",
    },
    "google": {
        "primary": "gemini-2.5-pro",
        "fast": "gemini-2.5-flash",
    },
    "vertexai": {
        "primary": "gemini-2.5-pro",
        "fast": "gemini-2.5-flash",
    },
    # bedrock has no sensible default — must be a region-specific ARN or profile ID
    "bedrock": {"primary": None, "fast": None},
    # azure has no sensible default — the "model id" is YOUR deployment name,
    # chosen when you deploy a model in the Azure OpenAI portal.
    "azure": {"primary": None, "fast": None},
}


def _current_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "bedrock").lower()


def _is_bedrock_provider() -> bool:
    return _current_provider() == "bedrock"


def _resolve_model_id(role: str = "primary") -> str | None:
    """Return the configured model ID for the current provider/role.

    Each provider remembers its own model ids via provider-scoped vars
    (e.g. AIDLC_MODEL_ID_AZURE) so switching LLM_PROVIDER never loses the
    models configured for another provider. The unscoped vars hold the
    ACTIVE provider's values (kept in sync by the Settings save), so every
    legacy reader of AIDLC_MODEL_ID keeps working.

    Order of precedence:
      1. AIDLC_MODEL_ID_<PROVIDER> (or AIDLC_SONNET_MODEL_ID_<PROVIDER> for role="fast")
      2. AIDLC_MODEL_ID (or AIDLC_SONNET_MODEL_ID for role="fast")
      3. Provider-specific default
      4. None (caller must handle)
    """
    provider = _current_provider()
    suffix = provider.upper()

    def _scoped_or_generic(name: str) -> str | None:
        return os.environ.get(f"{name}_{suffix}") or os.environ.get(name)

    if role == "fast":
        env_val = _scoped_or_generic("AIDLC_SONNET_MODEL_ID") or _scoped_or_generic("AIDLC_MODEL_ID")
    else:
        env_val = _scoped_or_generic("AIDLC_MODEL_ID")
    if env_val:
        # Tolerate stray whitespace in .env values (e.g. " gpt-5.4-mini" with a
        # leading space). An unstripped deployment name hits Azure as a literal
        # " gpt-5.4-mini" and comes back DeploymentNotFound / BadRequest.
        env_val = env_val.strip()
        if env_val:
            return env_val
    return _PROVIDER_DEFAULT_MODELS.get(provider, {}).get(role)


# Default embedding model per provider. Used when AIDLC_EMBED_MODEL_ID is not set.
# Anthropic has no native embeddings API; get_embeddings() returns None for it.
# Azure has no default either: the embedding "model id" is a DEPLOYMENT NAME you
# must create in the Azure portal (separate from your chat deployment). Guessing a
# name like text-embedding-3-small would 404 — so when unset, we fall back to BM25.
_PROVIDER_DEFAULT_EMBED_MODELS: dict[str, str | None] = {
    "bedrock":   "amazon.titan-embed-text-v2:0",
    "openai":    "text-embedding-3-small",
    "azure":     None,
    "google":    "models/text-embedding-004",
    "vertexai":  "text-embedding-005",
    "anthropic": None,
}


def _resolve_embed_model_id() -> str | None:
    """Return the configured embedding model ID for the current provider.

    Order of precedence:
      1. AIDLC_EMBED_MODEL_ID_<PROVIDER>  (scoped, per-provider)
      2. AIDLC_EMBED_MODEL_ID             (generic / active-provider mirror)
      3. Provider-specific default
      4. None (Anthropic — no native embeddings)
    """
    provider = _current_provider()
    suffix = provider.upper()
    env_val = os.environ.get(f"AIDLC_EMBED_MODEL_ID_{suffix}") or os.environ.get("AIDLC_EMBED_MODEL_ID")
    if env_val:
        return env_val
    return _PROVIDER_DEFAULT_EMBED_MODELS.get(provider)


def get_embedding_identity() -> str:
    """Stable identity string for the active embedding backend.

    Used to fingerprint persisted vector indexes (e.g. the Tool-RAG FAISS
    cache) so that switching LLM provider or embed model invalidates them.
    """
    return f"{_current_provider()}:{_resolve_embed_model_id() or 'default'}"


def _get_secret_or_env(key: str) -> str:
    """Read a configuration or secret variable, falling back to the OS keyring if needed.

    Values are trimmed: endpoints, API keys, and deployment names must never carry
    stray whitespace (a hand-edited .env or a pasted key with a trailing newline
    otherwise reaches the provider verbatim and fails as a 401/DeploymentNotFound).
    """
    val = os.environ.get(key)
    if val:
        return val.strip()
    try:
        from tools.secrets_store import get_secret
        return (get_secret(key) or "").strip()
    except Exception:
        return ""


# Minimum credentials each provider needs before _build_chat_llm can construct a
# client. Only providers whose auth is verifiable OFFLINE are listed; bedrock
# (IAM role / ~/.aws / env) and anything unknown are treated as ready so the gate
# never blocks a working provider it simply can't introspect.
_PROVIDER_REQUIRED_SECRETS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "vertexai": ("GOOGLE_SERVICE_ACCOUNT_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
}


def llm_ready() -> tuple[bool, str]:
    """Cheap, NETWORK-FREE check that the active provider's credentials are present.

    Lets a background job skip heavy work (graph build, tool calls, the model
    itself) when the LLM can't be constructed, instead of running the whole
    pipeline only to raise at first use — the failure mode behind the 2026-07-24
    priority/premarket retry storm (152 failed builds on a missing Vertex key).
    Returns (ready, reason); reason is set only when not ready. A missing secret
    does NOT self-heal within a process (secrets hydrate at startup), so a False
    here means "don't keep retrying until it is reconfigured and the server
    restarts". Providers whose auth can't be verified offline are treated as ready.
    """
    provider = _current_provider()
    required = _PROVIDER_REQUIRED_SECRETS.get(provider)
    if not required:
        return True, ""  # bedrock / unknown: not offline-verifiable — never block
    missing = [k for k in required if not _get_secret_or_env(k)]
    if missing:
        return False, f"LLM_PROVIDER={provider} but {', '.join(missing)} not set"
    return True, ""


def _vertex_credentials():
    """Build Vertex AI service-account credentials from the keychain SA-key JSON.

    The full service-account key (JSON) is stored as an ordinary keychain secret
    (GOOGLE_SERVICE_ACCOUNT_KEY) — pasted in Settings like any other key — and
    turned into credentials here at call time, so no key file is ever written to
    disk. Returns (credentials, project_id). The project defaults to the key's
    own project_id unless GOOGLE_CLOUD_PROJECT overrides it.
    """
    sa_json = _get_secret_or_env("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not sa_json:
        raise ValueError(
            "LLM_PROVIDER=vertexai but GOOGLE_SERVICE_ACCOUNT_KEY is not set. "
            "Paste your Vertex AI service-account key (JSON) in Settings."
        )
    import json
    try:
        info = json.loads(sa_json)
    except ValueError as exc:
        raise ValueError(
            "GOOGLE_SERVICE_ACCOUNT_KEY is not valid JSON — paste the full "
            "service-account key file contents in Settings."
        ) from exc
    try:
        from google.oauth2 import service_account
    except ImportError as exc:
        raise ValueError(
            "LLM_PROVIDER=vertexai requires google-auth (bundled with "
            "langchain-google-genai). Install it with: pip install langchain-google-genai"
        ) from exc
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project = _get_secret_or_env("GOOGLE_CLOUD_PROJECT") or info.get("project_id")
    if not project:
        raise ValueError(
            "GOOGLE_SERVICE_ACCOUNT_KEY has no project_id — set GOOGLE_CLOUD_PROJECT in Settings."
        )
    return creds, project


def _azure_embed_chunk_size() -> int:
    """Per-request text-batch size for Azure embeddings.

    Azure AI Foundry MaaS embed deployments cap a single request's input list
    (Cohere-embed-v3-english allows at most 96 texts). langchain's default
    chunk_size (1000) sends every text in ONE request, so a catalog larger than
    the cap — e.g. 122 tool docs — 400s with "total number of texts must be at
    most 96". Batching under the cap fixes it. Override via AIDLC_EMBED_BATCH_SIZE.
    NOTE: deliberately NOT routed through _env_int() (its 256-token floor is for
    output budgets and would force a batch size that re-triggers the 400).
    """
    raw = (os.environ.get("AIDLC_EMBED_BATCH_SIZE") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 96


def get_embeddings():
    """Return a LangChain Embeddings instance for the current provider/embed model.

    Returns None when the provider has no native embeddings API (Anthropic).
    Callers should fall back to BM25-only retrieval in that case.
    Raises ValueError on misconfiguration (missing API key / missing package).
    """
    provider = _current_provider()
    model = _resolve_embed_model_id()

    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(
            model_id=model or "amazon.titan-embed-text-v2:0",
            region_name=REGION,
            **get_bedrock_credential_kwargs(),
        )

    if provider == "openai":
        api_key = _get_secret_or_env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or "text-embedding-3-small", api_key=api_key)

    if provider == "azure":
        # No configured embedding deployment → fall back to BM25 (return None)
        # rather than guessing a deployment name that would 404.
        if not model:
            return None
        api_key = _get_secret_or_env("AZURE_OPENAI_API_KEY_EMBEDDING") or _get_secret_or_env("AZURE_OPENAI_API_KEY")
        endpoint = (_get_secret_or_env("AZURE_OPENAI_ENDPOINT_EMBEDDING") or _get_secret_or_env("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        if not api_key or not endpoint:
            raise ValueError(
                "LLM_PROVIDER=azure but AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT are not set. "
                "Configure them or set AZURE_OPENAI_ENDPOINT_EMBEDDING / AZURE_OPENAI_API_KEY_EMBEDDING."
            )
        # Mirror the chat path: the Foundry ".../openai/v1" surface is OpenAI-
        # compatible — AzureOpenAIEmbeddings would append its legacy deployments
        # path and 404, so use the plain OpenAI embeddings client against the
        # v1 base URL instead. The bare resource URL uses the legacy Azure route.
        # check_embedding_ctx_length=False: the OpenAI client otherwise tokenizes
        # inputs locally (tiktoken) and POSTs integer token-ID arrays. Real OpenAI
        # accepts those, but the Azure embeddings surface here only accepts strings
        # and rejects arrays with HTTP 422 ("Input should be a valid string"). Sending
        # raw text is safe — the only caller (tool_retriever's FAISS index over short
        # tool docs) stays far under the model's context window, so no chunking is lost.
        if endpoint.endswith("/openai/v1"):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=endpoint,
                check_embedding_ctx_length=False,
                # Foundry MaaS embed deployments cap a request at 96 texts; the
                # default chunk_size (1000) sends the whole tool catalog at once
                # and 400s once it exceeds the cap. See _azure_embed_chunk_size.
                chunk_size=_azure_embed_chunk_size(),
            )
        api_version = (os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip() or "2024-10-21"
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            model=model,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            check_embedding_ctx_length=False,
            chunk_size=_azure_embed_chunk_size(),
        )

    if provider == "google":
        api_key = _get_secret_or_env("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=google but GOOGLE_API_KEY is not set.")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ValueError(
                "LLM_PROVIDER=google requires the langchain-google-genai package. "
                "Install it with: pip install langchain-google-genai"
            ) from exc
        return GoogleGenerativeAIEmbeddings(
            model=model or "models/text-embedding-004",
            google_api_key=api_key,
        )

    if provider == "vertexai":
        creds, project = _vertex_credentials()
        location = _get_secret_or_env("GOOGLE_CLOUD_LOCATION") or "global"
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ValueError(
                "LLM_PROVIDER=vertexai requires the langchain-google-genai package (>=2.1). "
                "Install it with: pip install langchain-google-genai"
            ) from exc
        return GoogleGenerativeAIEmbeddings(
            model=model or "text-embedding-005",
            vertexai=True,
            project=project,
            location=location,
            credentials=creds,
        )

    if provider == "anthropic":
        return None  # Anthropic has no native embeddings API; caller falls back to BM25

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


# Read once at import; may be None for anthropic/openai users who rely on defaults.
# Bedrock callers will fail loudly inside get_llm() with a clearer error.
MODEL_ID = _resolve_model_id("primary")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_bedrock_credential_kwargs() -> dict[str, str]:
    """Return non-empty AWS credential kwargs for explicit Bedrock clients."""
    try:
        from tools.secrets_store import (
            clear_incompatible_aws_session_token,
            get_secret,
            load_secrets_into_env,
        )

        load_secrets_into_env()
        clear_incompatible_aws_session_token()
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or get_secret("AWS_ACCESS_KEY_ID") or None
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or get_secret("AWS_SECRET_ACCESS_KEY") or None
        session_token = os.environ.get("AWS_SESSION_TOKEN") or get_secret("AWS_SESSION_TOKEN") or None
    except Exception:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or None
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or None
        session_token = os.environ.get("AWS_SESSION_TOKEN") or None

    if access_key:
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    if access_key and access_key.startswith("AKIA"):
        session_token = None
        os.environ.pop("AWS_SESSION_TOKEN", None)
        os.environ.pop("AWS_PROFILE", None)

    if not access_key or not secret_key:
        return {}

    kwargs = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if session_token and not access_key.startswith("AKIA"):
        kwargs["aws_session_token"] = session_token
    return kwargs


def _create_bedrock_boto_client(service_name: str, boto_config: Any):
    import boto3

    return boto3.client(
        service_name,
        region_name=REGION,
        config=boto_config,
        **get_bedrock_credential_kwargs(),
    )

# --- Secret redaction for any output that may end up in logs/stdout ---
# Recognised secret-shaped tokens (provider prefixes, JWTs, PEM blocks). Anything that
# matches is replaced with [REDACTED] before being printed/logged.
_SECRET_REGEX = re.compile(
    r"("
    r"sk-[A-Za-z0-9_\-]{20,}"                  # OpenAI / Anthropic-style
    r"|ghp_[A-Za-z0-9]{20,}"                   # GitHub PATs
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"                       # AWS access key IDs
    r"|ASIA[0-9A-Z]{16}"                       # AWS STS tokens
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"           # Slack tokens
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"  # JWTs
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    r")"
)


def _redact_secrets(text: str) -> str:
    """Replace secret-shaped substrings and known env-var secret values with [REDACTED].

    Used as a sanitizer barrier before any text reaches stdout or a log sink.
    """
    if not isinstance(text, str) or not text:
        return text
    try:
        # Redact env-var values (key matches a "secret-like" name)
        sensitive_vals: set[str] = set()
        for env_k, env_v in os.environ.items():
            k_upper = env_k.upper()
            if any(term in k_upper for term in ("KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "AUTH")):
                if isinstance(env_v, str) and len(env_v) > 5:
                    sensitive_vals.add(env_v)
        for s_val in sorted(sensitive_vals, key=len, reverse=True):
            text = text.replace(s_val, "[REDACTED]")
        # Redact secret-shaped tokens regardless of env presence
        text = _SECRET_REGEX.sub("[REDACTED]", text)
    except Exception:
        pass
    return text


# Warning suppression configuration
_SUPPRESS_WARNINGS = os.environ.get("SUPPRESS_WARNINGS", "false").lower() in ("true", "1", "yes")
_WARNING_LEVEL = os.environ.get("WARNING_LEVEL", "normal").lower()  # "silent", "minimal", "normal", "verbose"

# Patterns to suppress based on warning level
_SUPPRESSED_PATTERNS = {
    "silent": [
        "⚠️", "❌", "🔴", "⚠", "WARNING", "WARN", "Failed", "failed", "Error", "error"
    ],
    "minimal": [
        "FMP.*failed", "FMP.*limited", "Falling back", "Switching to",
        "Cache Hit", "Refreshing token", "empty/failed"
    ],
    "normal": [
        "FMP.*failed", "FMP.*limited", "Falling back to search"
    ]
}

def safe_print(msg):
    """
    Print message safely, catching I/O errors from closed or null streams.
    Supports warning suppression via environment variables:
    - SUPPRESS_WARNINGS=true: Suppress all warnings
    - WARNING_LEVEL=silent: Suppress all warnings and errors
    - WARNING_LEVEL=minimal: Suppress common API fallback warnings
    - WARNING_LEVEL=normal: Suppress only repetitive API warnings (default)
    - WARNING_LEVEL=verbose: Show all messages
    """
    try:
        # Check for sys and stdout to avoid NameError/AttributeError
        import sys
        if sys is None or not hasattr(sys, 'stdout') or sys.stdout is None:
            return

        # Standard check for closed streams
        if hasattr(sys.stdout, 'closed') and sys.stdout.closed:
            return

        # _redact_secrets is the sanitizer barrier — it returns a fresh string with
        # any secret-shaped substring (env-resident values, provider tokens, JWTs,
        # PEM blocks) replaced by the literal "[REDACTED]". Any remaining match
        # against the secret regex means redaction failed; in that case we suppress.
        msg_str = _redact_secrets(str(msg))
        if _SECRET_REGEX.search(msg_str):
            return  # last-resort guard: never emit a still-secret-looking line

        # Apply warning suppression
        if _SUPPRESS_WARNINGS or _WARNING_LEVEL != "verbose":
            # Check if message should be suppressed
            patterns = _SUPPRESSED_PATTERNS.get(_WARNING_LEVEL, [])
            for pattern in patterns:
                if re.search(pattern, msg_str, re.IGNORECASE):
                    return  # Suppress this message

        # Use sys.stdout.write — at this point msg_str has passed through
        # _redact_secrets *and* a final regex-match guard, so secret-shaped tokens
        # cannot reach the stream.
        sys.stdout.write(msg_str + "\n")
        # Flush to force error to happen immediately if it's going to
        if hasattr(sys.stdout, 'flush'):
            sys.stdout.flush()

        # Collect warnings/errors into the active run context for header display.
        # Dedupe (retry loops emit near-identical lines) and cap the list so a
        # noisy run can't balloon the header pill or grow memory unbounded.
        ctx = _CURRENT_RUN_CONTEXT.get()
        if ctx is not None and re.search(r'[⚠❌🔴]|WARNING|WARN|\bFailed\b|\bError\b', msg_str):
            if msg_str not in ctx.notices:
                ctx.notices.append(msg_str)
                if len(ctx.notices) > 50:
                    del ctx.notices[0]
    except (OSError, ValueError, RuntimeError, BrokenPipeError, AttributeError):
        # Catch "I/O operation on closed file" and related stream errors
        pass
    except Exception:
        pass


# Sonnet/fast model ID for lighter tasks (data gathering, risk checks, summarization).
# Falls back to provider default if neither AIDLC_SONNET_MODEL_ID nor AIDLC_MODEL_ID is set.
SONNET_MODEL_ID = _resolve_model_id("fast")

def _get_model_config(model_id: str) -> dict[str, Any]:
    """
    Returns dict with 'model' and 'provider'.
    """
    # ARN, us.*, or global.* profile — always set provider for prompt caching to work
    if "inference-profile" in model_id or model_id.startswith(("us.", "global.")) or "anthropic" in model_id:
        return {"model": model_id, "provider": "anthropic"}

    return {"model": model_id}

def ensure_bedrock_sequence(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Ensures message history complies with AWS Bedrock rules:
    1. Every AIMessage with tool_calls MUST be followed by ToolMessages.
    2. If a tool_use is dangling (at end or followed by Human), fix the sequence.
    3. Maintains User -> Assistant alternation (ToolMessage = User role in Bedrock).
    """
    if not messages:
        return messages

    fixed = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        fixed.append(msg)

        # Check for AIMessage with tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Look ahead for ToolMessages
            tool_ids = {tc['id'] for tc in msg.tool_calls}
            found_ids = set()
            following_tool_messages = []

            j = i + 1
            while j < len(messages) and (isinstance(messages[j], ToolMessage) or (hasattr(messages[j], "tool_call_id") and messages[j].tool_call_id)):
                t_msg = messages[j]
                following_tool_messages.append(t_msg)
                if t_msg.tool_call_id in tool_ids:
                    found_ids.add(t_msg.tool_call_id)
                j += 1

            missing_ids = tool_ids - found_ids
            fixed.extend(following_tool_messages)

            # CRITICAL FIX: If ToolResults are missing, Bedrock fails.
            if missing_ids:
                for tid in missing_ids:
                    t_name = next((tc['name'] for tc in msg.tool_calls if tc['id'] == tid), "unknown_tool")
                    fixed.append(ToolMessage(
                        content=f"Error: Tool execution was interrupted or result not found for {t_name}.",
                        tool_call_id=tid,
                        name=t_name
                    ))
            if following_tool_messages or missing_ids:
                i = j - 1
        i += 1

    for m in fixed:
        if hasattr(m, "content") and m.content == "":
             if not (hasattr(m, "tool_calls") and m.tool_calls):
                 m.content = "."

    return fixed


def _normalize_bedrock_input(input_data):
    """
    Repair Bedrock-invalid message histories before invocation.
    This is especially important when a prior turn left dangling tool_calls
    in the in-memory LangGraph thread history.
    """
    if not _is_bedrock_provider():
        return input_data

    def _log_repair(original_count: int, normalized_count: int):
        inserted = max(0, normalized_count - original_count)
        if inserted <= 0:
            return
        try:
            from agent.logger import log_event
            log_event("BedrockSanitizer", "Inserted placeholder tool results for dangling Bedrock tool calls", {
                "original_message_count": original_count,
                "normalized_message_count": normalized_count,
                "inserted_tool_results": inserted,
            })
        except Exception:
            pass

    if isinstance(input_data, dict):
        messages = input_data.get("messages")
        if isinstance(messages, list):
            normalized_messages = ensure_bedrock_sequence(list(messages))
            _log_repair(len(messages), len(normalized_messages))
            normalized_input = dict(input_data)
            normalized_input["messages"] = normalized_messages
            return normalized_input
        return input_data

    if isinstance(input_data, list) and all(isinstance(msg, BaseMessage) for msg in input_data):
        normalized_messages = ensure_bedrock_sequence(list(input_data))
        _log_repair(len(input_data), len(normalized_messages))
        return normalized_messages

    return input_data

_ROLE_ENV_HINT = {
    "primary": "AIDLC_MODEL_ID",
    "fast": "AIDLC_SONNET_MODEL_ID (or AIDLC_MODEL_ID)",
}


# --- Throttling defense (HTTP 429) ------------------------------------------
# Azure/OpenAI return 429 when a deployment's per-minute quota (TPM/RPM) is
# exceeded. A client-side limiter cannot create capacity, so the PRIMARY fix is
# to raise the deployment's TPM in Azure. Two softer, code-side knobs back it up:
#   - max_retries: the SDK retries a 429 with exponential backoff, honoring the
#     server's Retry-After header.
#   - an optional per-endpoint request limiter, OFF by default (it adds latency
#     without adding quota). Opt in only if you must stay under a hard RPM cap.
#
# Tune without touching code (read live, so a Settings save takes effect):
#   LLM_MAX_RPS      per-endpoint requests/sec ceiling (float). Default 0 = off.
#   LLM_MAX_RETRIES  retry budget for the anchor region. Default 8.
_RATE_LIMITERS: dict[str, object] = {}
_RATE_LIMITER_RPS: float | None = None


def _max_retries() -> int:
    try:
        return max(0, int(os.environ.get("LLM_MAX_RETRIES", "8")))
    except ValueError:
        return 8


def _rate_limiter_for(endpoint_id: str):
    """Per-endpoint request limiter, or None when disabled (LLM_MAX_RPS<=0).

    Keyed per endpoint so each region gets its own RPS budget — a single shared
    bucket would cap total throughput and defeat multi-region load balancing.
    """
    global _RATE_LIMITER_RPS
    try:
        rps = float(os.environ.get("LLM_MAX_RPS", "0"))
    except ValueError:
        rps = 0.0
    if rps <= 0:
        return None
    if rps != _RATE_LIMITER_RPS:  # knob changed at runtime → rebuild all buckets
        _RATE_LIMITERS.clear()
        _RATE_LIMITER_RPS = rps
    if endpoint_id not in _RATE_LIMITERS:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        _RATE_LIMITERS[endpoint_id] = InMemoryRateLimiter(
            requests_per_second=rps,
            check_every_n_seconds=0.1,
            max_bucket_size=max(1, round(rps)),
        )
    return _RATE_LIMITERS[endpoint_id]


def _throttle_kwargs(endpoint_id: str, max_retries: int) -> dict:
    """Throttle-resistance kwargs (max_retries + optional rate limiter) for any
    LangChain chat client that accepts them — openai, azure, google, anthropic.
    Bedrock uses boto's adaptive retries instead and wires the limiter directly."""
    kwargs = {"max_retries": max(0, max_retries)}
    limiter = _rate_limiter_for(endpoint_id)
    if limiter is not None:
        kwargs["rate_limiter"] = limiter
    return kwargs


def _retry_after_seconds(exc, default: float) -> float:
    """Pull the server's Retry-After (seconds) off a 429, else `default`.

    A TPM (tokens-per-minute) throttle clears when the rolling window resets, and
    the server states exactly when via Retry-After — far better than a blind
    exponential backoff. Returns `default` when no usable header is present.
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        try:
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra is not None:
                return max(0.0, float(ra))
        except (TypeError, ValueError):
            pass
    return default


# Trailing-60s token meter, keyed per model. Azure quota (TPM) is per-model, so
# this shows directly in the logs when a model nears its ceiling — the real
# throttle trigger. Set the ceiling with AZURE_OPENAI_TPM_LIMIT (default 20000).
_TPM_WINDOW: dict[str, "collections.deque"] = collections.defaultdict(collections.deque)
_TPM_LOCK = threading.Lock()


def _record_tpm(model_id: str, total_tokens: int) -> dict:
    """Add a call's tokens to the trailing-60s window; return rolling stats."""
    now = time.monotonic()
    key = model_id or "unknown"
    try:
        # Provider-neutral knob; AZURE_OPENAI_TPM_LIMIT kept as a back-compat alias.
        limit = int(os.environ.get("LLM_TPM_LIMIT")
                    or os.environ.get("AZURE_OPENAI_TPM_LIMIT") or "20000")
    except ValueError:
        limit = 20000
    with _TPM_LOCK:
        win = _TPM_WINDOW[key]
        win.append((now, total_tokens))
        cutoff = now - 60.0
        while win and win[0][0] < cutoff:
            win.popleft()
        rolling = sum(t for _, t in win)
        calls = len(win)
    return {
        "tokens_last_60s": rolling,
        "calls_last_60s": calls,
        "tpm_limit": limit,
        "tpm_pct": round(100.0 * rolling / limit, 1) if limit else 0.0,
        "throttle_risk": bool(limit and rolling >= limit),
    }


# Dedupe agent/graph-state messages so the same AIMessage isn't counted twice
# when message history is re-passed across cycles. Bounded to avoid unbounded growth.
_LOGGED_USAGE_MSG_IDS: set[str] = set()


def _model_from_meta(meta) -> str:
    """Best-effort model name from response_metadata across providers.

    Bedrock exposes 'model_id'; Azure/OpenAI expose 'model_name'. The old code
    only read 'model_id', so every Azure call fell back to AIDLC_MODEL_ID (the
    PRIMARY model) — silently mislabeling fast-tier (Kimi) calls as the primary.
    """
    if isinstance(meta, dict):
        return meta.get("model_id") or meta.get("model_name") or ""
    return ""


def _log_token_usage(um: dict, meta, mode: str):
    """Accumulate cost and emit a TokenUsage log line for one LLM response."""
    if not um:
        return
    model = _model_from_meta(meta) or os.environ.get("AIDLC_MODEL_ID", "")
    cache_read = 0
    itd = um.get('input_token_details') or {}
    if isinstance(itd, dict):
        cache_read = itd.get('cache_read', 0) or 0
    try:
        from agent.cost_tracker import accumulate_cost
        accumulate_cost(
            um.get('input_tokens', 0), um.get('output_tokens', 0),
            model, cache_read_tokens=cache_read,
        )
    except Exception:
        pass
    try:
        from agent.logger import log_event
        tot = um.get('total_tokens', 0) \
            or (um.get('input_tokens', 0) + um.get('output_tokens', 0))
        log_event("TokenUsage", "LLM call token usage", {
            "mode": mode,
            "model_id": model,
            "input_tokens": um.get('input_tokens', 0),
            "output_tokens": um.get('output_tokens', 0),
            "total_tokens": tot,
            "cache_read_tokens": cache_read,
            **_record_tpm(model, tot),
        })
    except Exception:
        pass
    _log_grounding_usage(meta, model)


def _log_grounding_usage(meta, model: str):
    """Log whenever Gemini's native Google Search grounding actually fired.

    Gemini executes google_search server-side (not as a tool_call we dispatch),
    so ToolExecution logging never sees it — this is the only signal we have.
    langchain-google-genai surfaces it in response_metadata["grounding_metadata"]
    (the web_search_queries actually run + grounding_chunks citing the sources
    used) only when the model chose to ground itself, so absence of this log
    line for a given call means grounding did not fire on it.
    """
    if not isinstance(meta, dict):
        return
    grounding = meta.get("grounding_metadata")
    if not grounding:
        return
    queries = grounding.get("web_search_queries") or []

    # Grounding is billed per request, separately from tokens, so the ordinary
    # usage_metadata path above cannot account for it. Without this the meter
    # silently understates every grounded turn.
    try:
        from agent.cost_tracker import accumulate_grounding
        accumulate_grounding(requests=1, queries=len(queries))
    except Exception:
        pass

    try:
        from agent.logger import log_event
        chunks = grounding.get("grounding_chunks") or []
        sources = [
            {"title": (c.get("web") or {}).get("title"), "uri": (c.get("web") or {}).get("uri")}
            for c in chunks if isinstance(c, dict) and c.get("web")
        ]
        log_event("Grounding", "Google Search grounding fired", {
            "model_id": model,
            "web_search_queries": queries,
            "source_count": len(sources),
            "sources": sources[:10],
        })
    except Exception:
        pass


def _capture_usage(result, mode: str):
    """Log token usage from an AIMessage result OR an agent/graph state dict.

    Agents built via create_agent (prompt | llm) return an AIMessage directly,
    but full-graph invokes return {"messages": [...]}; we then sum usage across
    the AIMessages in that list (deduped by id so re-passed history isn't double
    counted). This is what lets fast-tier (Kimi) agent calls get logged at all.
    """
    try:
        um = getattr(result, 'usage_metadata', None)
        if um:
            _log_token_usage(um, getattr(result, 'response_metadata', {}), mode)
            return
        msgs = result.get("messages") if isinstance(result, dict) else None
        if isinstance(msgs, list):
            for m in msgs:
                mu = getattr(m, 'usage_metadata', None)
                if not mu:
                    continue
                mid = getattr(m, 'id', None)
                if mid:
                    if mid in _LOGGED_USAGE_MSG_IDS:
                        continue
                    if len(_LOGGED_USAGE_MSG_IDS) > 10000:
                        _LOGGED_USAGE_MSG_IDS.clear()
                    _LOGGED_USAGE_MSG_IDS.add(mid)
                _log_token_usage(mu, getattr(m, 'response_metadata', {}), mode)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Reasoning / "max think" control (provider-agnostic)
#
# Kimi, DeepSeek, GPT-5/o-series, Gemini, and Claude all support extended
# reasoning, but NONE of them expose a Claude-Code-style "think harder" prompt
# keyword — depth is a *request parameter* (or, on some gateways, a separate
# model deployment). This maps ONE knob, AIDLC_REASONING_EFFORT, onto each
# provider's native mechanism so no caller has to special-case a vendor:
#
#   AIDLC_REASONING_EFFORT            = off | low | medium | high | max  (both tiers)
#   AIDLC_REASONING_EFFORT_PRIMARY    = …                               (per-tier override)
#   AIDLC_REASONING_EFFORT_FAST       = …
#   AIDLC_REASONING_EXTRA_BODY        = raw JSON merged into the OpenAI/Azure
#                                       request body — escape hatch for gateways
#                                       that want e.g. {"chat_template_kwargs":
#                                       {"thinking": true}} (DeepSeek/Kimi on
#                                       vLLM/SGLang) instead of reasoning_effort.
#
# 'off' (the default) reproduces today's behavior byte-for-byte. The OpenAI/Azure
# mapping is the verified path (it backs the live Kimi + DeepSeek stack); the
# google/anthropic/bedrock mappings are best-effort — confirm the parameter
# against your installed LangChain version before relying on them.
# ---------------------------------------------------------------------------
_REASONING_LEVELS = ("off", "low", "medium", "high", "max")


def _reasoning_effort(role: str) -> str:
    """Configured reasoning effort for a tier; 'off' disables (the default)."""
    raw = (
        os.environ.get(f"AIDLC_REASONING_EFFORT_{role.upper()}")
        or os.environ.get("AIDLC_REASONING_EFFORT")
        or "off"
    ).strip().lower()
    return raw if raw in _REASONING_LEVELS else "off"


def _reasoning_temperature(provider: str, role: str, default: float = 0.0):
    """Temperature to use given the reasoning setting.

    Reasoning models constrain temperature: OpenAI o-series/gpt-5 and Anthropic
    Claude 4.7+ reject an explicit value, while Bedrock extended thinking
    requires 1.0. Returning None tells the LangChain client to omit the field.
    """
    if _reasoning_effort(role) == "off":
        return default
    if provider == "anthropic":
        return None        # Claude 4.7+ rejects an explicit temperature outright
    if provider == "bedrock":
        return 1.0         # Bedrock extended thinking requires temperature=1
    # openai/azure/google: verified live (2026-06-17) that DeepSeek-V4-Pro and
    # Kimi-K2.6 accept temperature=0.0 alongside reasoning_effort, so keep the
    # caller's temperature for determinism. (A future gpt-5/o-series azure
    # deployment rejects temperature — handle that if/when it's added.)
    return default


def _reasoning_kwargs(provider: str, role: str, max_tokens: int) -> dict:
    """Provider-native constructor kwargs for the configured reasoning effort.

    Empty dict when AIDLC_REASONING_EFFORT is off (today's behavior).
    """
    level = _reasoning_effort(role)
    if level == "off":
        return {}

    if provider in ("openai", "azure"):
        extra = os.environ.get("AIDLC_REASONING_EXTRA_BODY", "").strip()
        if extra:
            import json as _json
            try:
                return {"model_kwargs": {"extra_body": _json.loads(extra)}}
            except ValueError:
                import logging
                logging.getLogger(__name__).warning(
                    "AIDLC_REASONING_EXTRA_BODY is not valid JSON; ignoring it"
                )
                return {}
        # OpenAI's vocabulary is low|medium|high (no 'max'); gpt-5 adds 'minimal'.
        return {"reasoning_effort": "high" if level == "max" else level}

    if provider in ("google", "vertexai"):
        budget = {"low": 1024, "medium": 8192, "high": 24576, "max": -1}[level]
        # thinking_budget alone makes the model think in private: without
        # include_thoughts the API returns no thought parts, so langchain never
        # emits the {"type": "thinking"} blocks the UI's reasoning trace reads
        # (see extract_reasoning_text). Asking for the summaries is what makes
        # the trace non-empty; it does not change how much the model thinks.
        return {"thinking_budget": budget, "include_thoughts": True}

    if provider == "anthropic":
        # Adaptive thinking is the modern surface (Opus 4.6+/Sonnet 4.6); older
        # models would instead need {"type": "enabled", "budget_tokens": N}.
        return {"thinking": {"type": "adaptive"}}

    if provider == "bedrock":
        budget = {"low": 1024, "medium": 4096, "high": 16384, "max": 32768}[level]
        budget = max(1024, min(budget, max_tokens - 1))  # budget must be < max_tokens
        return {
            "additional_model_request_fields": {
                "reasoning_config": {"type": "enabled", "budget_tokens": budget}
            }
        }

    return {}


# ---------------------------------------------------------------------------
# Output-budget (max_tokens) defaults
#
# max_tokens is a CAP, not a target — you only pay for tokens actually emitted,
# so a generous cap just prevents truncation. With reasoning enabled the hidden
# chain-of-thought shares this budget (reasoning_content + answer), so the
# defaults below leave headroom for a reasoner. Override per tier via env.
#
#   AIDLC_MAX_TOKENS / AIDLC_MAX_TOKENS_PRIMARY  — primary tier  (default 16384)
#   AIDLC_MAX_TOKENS_FAST                        — fast tier     (default 8192)
#   AIDLC_MAX_TOKENS_DEEP                        — Deep Reasoning conclusion
#                                                  synthesis only (default 32768)
#
# Live-verified 2026-06-17: DeepSeek-V4-Pro and Kimi-K2.6 on Azure Foundry both
# accept up to 131072 output tokens, so these values are well within range.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_TOKENS = {"primary": 16384, "fast": 8192}
_DEEP_MAX_TOKENS_DEFAULT = 32768


def _env_int(*names: str) -> int | None:
    """First parseable positive int among the given env vars, else None."""
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if raw:
            try:
                return max(256, int(raw))
            except ValueError:
                continue
    return None


def _default_max_tokens(role: str) -> int:
    if role == "primary":
        return _env_int("AIDLC_MAX_TOKENS_PRIMARY", "AIDLC_MAX_TOKENS") or _DEFAULT_MAX_TOKENS["primary"]
    return _env_int("AIDLC_MAX_TOKENS_FAST") or _DEFAULT_MAX_TOKENS["fast"]


def deep_reasoning_max_tokens() -> int:
    """Output budget for the Deep Reasoning *conclusion* synthesis — larger than
    the primary default so a high-effort reasoner's chain-of-thought plus the
    multi-section verdict don't truncate. Override via AIDLC_MAX_TOKENS_DEEP."""
    return _env_int("AIDLC_MAX_TOKENS_DEEP") or max(_DEEP_MAX_TOKENS_DEFAULT, _default_max_tokens("primary"))


def _build_chat_llm(role: str, max_tokens: int):
    """Single vendor dispatch for both model tiers (role: 'primary' | 'fast').

    Every LLM call in the app funnels through get_llm()/get_sonnet_llm() into here,
    so supporting a new vendor is ONE branch in this function (plus a defaults entry
    above, settings-UI fields, and a pricing entry) — never a per-node change.
    """
    provider = _current_provider()
    model = _resolve_model_id(role)
    env_hint = _ROLE_ENV_HINT.get(role, "AIDLC_MODEL_ID")

    if provider == "openai":
        api_key = _get_secret_or_env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set. Add it to user_data/.env.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or _PROVIDER_DEFAULT_MODELS["openai"][role],
            api_key=api_key,
            temperature=_reasoning_temperature(provider, role),
            max_tokens=max_tokens,
            **_reasoning_kwargs(provider, role, max_tokens),
            **_throttle_kwargs("openai", _max_retries()),
        )

    if provider == "azure":
        # Azure OpenAI / AI Foundry: the "model id" fields hold your *deployment
        # names*, chosen when you deploy a model. A deployment lives on one resource;
        # the request must hit that resource's endpoint carrying that exact
        # deployment name, or Azure answers with DeploymentNotFound. Each tier
        # resolves ONE endpoint + key + deployment and builds ONE client.
        #
        # Throttling (429) is a QUOTA ceiling, not a transient blip — solved by
        # raising the deployment's capacity (TPM) in Azure, not by client retries.
        #
        # NOTE: embeddings are intentionally NOT resolved here. They are a separate
        # deployment, often on a different resource, and are built in get_embeddings()
        # via AZURE_OPENAI_ENDPOINT_EMBEDDING / AZURE_OPENAI_API_KEY_EMBEDDING.
        shared_endpoint = _get_secret_or_env("AZURE_OPENAI_ENDPOINT")
        shared_key = _get_secret_or_env("AZURE_OPENAI_API_KEY")

        if role == "fast":
            # The fast tier may pin its own resource via *_FAST; otherwise it shares
            # the primary chat resource. Endpoint and key fall back together.
            endpoint = _get_secret_or_env("AZURE_OPENAI_ENDPOINT_FAST") or shared_endpoint
            key = _get_secret_or_env("AZURE_OPENAI_API_KEY_FAST") or shared_key
        else:
            endpoint, key = shared_endpoint, shared_key

        if not endpoint or not key:
            raise ValueError(
                "LLM_PROVIDER=azure but AZURE_OPENAI_API_KEY and/or AZURE_OPENAI_ENDPOINT "
                "is not set. Add both in Settings (or user_data/.env)."
            )
        if not model:
            raise ValueError(
                f"LLM_PROVIDER=azure but no {role} deployment is configured. Set {env_hint}."
            )

        endpoint = endpoint.strip().rstrip("/")
        if endpoint.endswith("/anthropic"):
            # Azure AI Foundry's Claude (format="Anthropic") deployments speak the
            # Anthropic Messages API at {endpoint}/v1/messages — NOT the OpenAI
            # /openai/deployments/<name>/chat/completions path AzureChatOpenAI
            # builds, which 404s "Resource not found" on this surface. Route them
            # through ChatAnthropic with base_url (the anthropic SDK appends
            # /v1/messages and sends x-api-key + anthropic-version). The reasoning
            # knob and temperature follow ANTHROPIC vendor semantics (adaptive
            # thinking, no explicit temperature on Claude 4.7+), not the
            # azure/openai reasoning_effort path — pass "anthropic" to both helpers.
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                api_key=key,
                base_url=endpoint,
                temperature=_reasoning_temperature("anthropic", role),
                max_tokens=max_tokens,
                **_reasoning_kwargs("anthropic", role, max_tokens),
                **_throttle_kwargs(endpoint, _max_retries()),
            )
        if endpoint.endswith("/openai/v1"):
            # New Azure AI Foundry v1 surface — addressed like stock OpenAI.
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                api_key=key,
                base_url=endpoint,
                temperature=_reasoning_temperature(provider, role),
                max_tokens=max_tokens,
                **_reasoning_kwargs(provider, role, max_tokens),
                **_throttle_kwargs(endpoint, _max_retries()),
            )
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=endpoint,
            api_key=key,
            # gpt-5.x / o-series reject the old 2024-10-21 GA version with
            # "API version not supported". Default to a current preview that
            # covers them; override via AZURE_OPENAI_API_VERSION. (Better still,
            # use the /openai/v1 endpoint above, which needs no api-version.)
            api_version=(os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip() or "2024-12-01-preview",
            temperature=_reasoning_temperature(provider, role),
            max_tokens=max_tokens,
            **_reasoning_kwargs(provider, role, max_tokens),
            **_throttle_kwargs(endpoint, _max_retries()),
        )

    if provider == "google":
        api_key = _get_secret_or_env("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=google but GOOGLE_API_KEY is not set. Add it in Settings (or user_data/.env).")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ValueError(
                "LLM_PROVIDER=google requires the langchain-google-genai package. "
                "Install it with: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=model or _PROVIDER_DEFAULT_MODELS["google"][role],
            google_api_key=api_key,
            temperature=_reasoning_temperature(provider, role),
            max_output_tokens=max_tokens,
            **_reasoning_kwargs(provider, role, max_tokens),
            **_throttle_kwargs("google", _max_retries()),
        )

    if provider == "vertexai":
        # Gemini on Vertex AI, billed against the GCP project. Auth is a
        # service-account key pasted in Settings and stored in the keychain
        # (GOOGLE_SERVICE_ACCOUNT_KEY) — see _vertex_credentials(). Uses the
        # maintained ChatGoogleGenerativeAI with vertexai=True (langchain's
        # ChatVertexAI is deprecated, so we deliberately avoid it).
        creds, project = _vertex_credentials()
        location = _get_secret_or_env("GOOGLE_CLOUD_LOCATION") or "global"
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ValueError(
                "LLM_PROVIDER=vertexai requires the langchain-google-genai package (>=2.1). "
                "Install it with: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=model or _PROVIDER_DEFAULT_MODELS["vertexai"][role],
            vertexai=True,
            project=project,
            location=location,
            credentials=creds,
            temperature=_reasoning_temperature(provider, role),
            max_output_tokens=max_tokens,
            **_reasoning_kwargs(provider, role, max_tokens),
            **_throttle_kwargs("vertexai", _max_retries()),
        )

    if provider == "anthropic":
        api_key = _get_secret_or_env("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. Add it to user_data/.env.")
        if not model:
            raise ValueError(
                f"LLM_PROVIDER=anthropic but no model is configured. Set {env_hint} "
                "in user_data/.env to a valid Anthropic model id "
                "(e.g. 'claude-sonnet-4-6-20250929' or 'claude-haiku-4-6-20250929')."
            )
        # Reject Bedrock-style IDs that would 404 against the Anthropic API
        if model.startswith(("us.", "global.", "arn:aws")) or "inference-profile" in model:
            raise ValueError(
                f"{env_hint}='{model}' looks like a Bedrock ARN/profile, "
                "but LLM_PROVIDER=anthropic expects a bare Anthropic model id "
                "(e.g. 'claude-sonnet-4-6-20250929'). Update user_data/.env."
            )
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=_reasoning_temperature(provider, role),
            max_tokens=max_tokens,
            **_reasoning_kwargs(provider, role, max_tokens),
            **_throttle_kwargs("anthropic", _max_retries()),
        )

    # bedrock (default)
    if not model:
        raise ValueError(
            "LLM_PROVIDER=bedrock but AIDLC_MODEL_ID is not set. "
            "Set it to a Bedrock model id, inference profile (e.g. 'us.anthropic.claude-...'), "
            "or full ARN in user_data/.env."
        )
    from botocore.config import Config

    config = _get_model_config(model)

    # Adaptive retries for throttling/rate limiting; 5 attempts (excessive retries
    # just add latency). The primary tier supports long-running reasoning turns, so
    # it gets a longer read timeout than the high-volume fast tier.
    boto_config = Config(
        retries={'max_attempts': 5, 'mode': 'adaptive'},
        connect_timeout=10,
        read_timeout=300 if role == "primary" else 120
    )

    client = _create_bedrock_boto_client("bedrock-runtime", boto_config)
    bedrock_client = _create_bedrock_boto_client("bedrock", boto_config)

    # Bedrock keeps boto's adaptive retries (above), but still honors the optional
    # shared rate limiter so the throttle knob (LLM_MAX_RPS) is universal.
    bedrock_kwargs = {}
    _bedrock_limiter = _rate_limiter_for("bedrock")
    if _bedrock_limiter is not None:
        bedrock_kwargs["rate_limiter"] = _bedrock_limiter

    # Imported here, not at module scope: langchain_aws pulls in transformers and
    # costs ~4.7s of cold import — paid by every process that touches agent.utils,
    # including each pytest run, whether or not the active provider is Bedrock.
    # Matches the lazy `from langchain_aws import BedrockEmbeddings` below.
    from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse

    return ChatBedrockConverse(
        client=client,
        bedrock_client=bedrock_client,
        model=config["model"],
        provider=config.get("provider"),
        temperature=_reasoning_temperature(provider, role),
        max_tokens=max_tokens,
        **_reasoning_kwargs(provider, role, max_tokens),
        **bedrock_kwargs,
    )


def get_llm(max_tokens: int | None = None):
    """Primary LLM for core reasoning and synthesis.

    max_tokens caps the *output* budget. It's a ceiling, not a target — you only
    pay for tokens emitted — but with reasoning enabled the chain-of-thought
    shares it, so leave headroom. Omit to use the tier default (AIDLC_MAX_TOKENS,
    16384); callers emitting short structured results pass a smaller cap.
    """
    return _build_chat_llm("primary", max_tokens if max_tokens is not None else _default_max_tokens("primary"))


def get_sonnet_llm(max_tokens: int | None = None):
    """Fast, cheaper model for data gathering, risk checks, and synthesis.

    max_tokens caps the *output* budget; pass a smaller value for short,
    structured turns (e.g. tool-selection planning). Omit for the tier default
    (AIDLC_MAX_TOKENS_FAST, 8192).
    """
    return _build_chat_llm("fast", max_tokens if max_tokens is not None else _default_max_tokens("fast"))

def get_fast_llm(max_tokens: int | None = None):
    """Alias for Sonnet — used for quick summarization & memory tasks."""
    return get_sonnet_llm(max_tokens=max_tokens)


def _system_prompt_text(system_prompt: str | list[dict]) -> str:
    if isinstance(system_prompt, str):
        return system_prompt

    parts = []
    for item in system_prompt:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts)


# Anthropic allows at most four cache breakpoints per request; extras are an error.
_ANTHROPIC_MAX_CACHE_BREAKPOINTS = 4


def _anthropic_cache_blocks(system_prompt: list[dict]) -> list[dict]:
    """Translate our [cachePoint] markers into Anthropic cache_control blocks.

    The two providers express the same idea inversely. Bedrock inserts a marker
    BETWEEN blocks — everything before it is cached. Anthropic instead tags the
    LAST block of the cached prefix with `cache_control: ephemeral`. So a marker
    here attaches to the block preceding it rather than becoming a block itself.

    Capped at four breakpoints because Anthropic rejects more; the earliest are
    kept, which is what matters — they cover the largest stable prefix.
    """
    blocks: list[dict] = []
    breakpoints = 0
    for item in system_prompt:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict) and item.get("cachePoint"):
            if blocks and breakpoints < _ANTHROPIC_MAX_CACHE_BREAKPOINTS:
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
                breakpoints += 1
        elif isinstance(item, dict) and item.get("text"):
            blocks.append({"type": "text", "text": str(item["text"])})
    return blocks


def _system_prompt_message(system_prompt: str | list[dict]):
    if isinstance(system_prompt, list):
        provider = _current_provider()
        if provider == "bedrock":
            return SystemMessage(content=system_prompt)
        if provider == "anthropic":
            return SystemMessage(content=_anthropic_cache_blocks(system_prompt))
        # google/vertexai reach the flattening below on purpose. Gemini 2.5+/3.x
        # caches IMPLICITLY: there is no marker to send — a stable prefix over the
        # model's minimum is discounted automatically, and a literal "[cachePoint]"
        # in the text would only corrupt that prefix. Cache hits are not invisible:
        # ChatGoogleGenerativeAI maps cached_content_token_count into
        # usage_metadata["input_token_details"]["cache_read"], which _log_token_usage
        # already bills at the cache rate. Check the `cache_read_tokens` figure in
        # the session breakdown before concluding caching is or is not working.
    # Return a SystemMessage object, NEVER a ("system", text) tuple:
    # ChatPromptTemplate parses tuple content as an f-string template, so any
    # bare '{' in the prompt (tool-result JSON, portfolio dicts) raises
    # "unmatched '{' in format spec". Message objects pass through verbatim.
    # Our prompts are fully pre-formatted in Python — nothing needs template vars.
    return SystemMessage(content=_system_prompt_text(system_prompt))


def create_agent(llm, tools: list, system_prompt: str | list[dict]):
    # Bind tools if provided
    if tools:
        bind_tools = list(tools)
        # Gemini 3 (google/vertexai) supports combining Grounding with Google Search
        # with custom function-calling tools in one request — gives the model a
        # native fallback for topics none of our finance tools cover (general
        # knowledge, evaluating third-party software, etc.) instead of it either
        # misusing a finance-shaped tool like search_multi_source or fabricating an
        # answer. Other providers don't understand this special tool type, so it's
        # gated to google/vertexai only. Opt out with AIDLC_GOOGLE_SEARCH_GROUNDING=0.
        if _current_provider() in ("google", "vertexai") and os.environ.get("AIDLC_GOOGLE_SEARCH_GROUNDING", "1") != "0":
            bind_tools.append({"google_search": {}})
        llm = llm.bind_tools(bind_tools)

    # Handle structured system prompt for Caching
    # ChatPromptTemplate expects a list of messages.
    # If system_prompt is a list (JSON for caching), wrap it in SystemMessage

    sys_msg = _system_prompt_message(system_prompt)

    prompt = ChatPromptTemplate.from_messages([
        sys_msg,
        MessagesPlaceholder(variable_name="messages"),
    ])

    return prompt | llm

# Rate-limit / 429 errors are a hard per-minute quota (TPM/RPM) ceiling, not a
# transient blip. Retrying is pointless: a single request that exceeds the quota
# can never fit, and the per-minute window won't clear in the few seconds we'd
# wait. So these fail FAST into the caller's degraded/fallback path instead of
# burning the user's time on blind backoff. Opt back in with LLM_RETRY_RATE_LIMITS=1.
_RATE_LIMIT_ERROR_SUBSTRINGS = [
    "ratelimitreached", "rate limit", "rate_limit", "ratelimit",
    "429", "too many requests", "toomanyrequests", "too_many_requests",
    "throttling", "throttlingexception",
]

# Genuinely transient errors that DO benefit from a quick retry.
_RETRYABLE_ERROR_SUBSTRINGS = [
    "gzip", "content-length of 0", "empty response",
    "connection", "timeout", "503", "502", "500",
    "model_output_error", "model output must contain",
]


def _is_rate_limit_error(error_str: str) -> bool:
    """True if the (lowercased) error text is a TPM/RPM rate-limit / 429."""
    return any(s in error_str for s in _RATE_LIMIT_ERROR_SUBSTRINGS)


def _rate_limit_retries_enabled() -> bool:
    """Default: do NOT retry rate limits. Set LLM_RETRY_RATE_LIMITS=1 to opt in."""
    return os.environ.get("LLM_RETRY_RATE_LIMITS", "0").strip().lower() in ("1", "true", "yes", "on")


def safe_invoke(agent_or_llm, input_data, max_retries=5, initial_delay=3):
    """Invoke agent/LLM with retry logic for throttling and transient API errors.

    Retries honor the server's Retry-After header, capped per attempt by
    LLM_RETRY_CAP (default 15s), so a 429 (e.g. a stream falling back here on a
    rate limit) fails fast instead of stalling on a long blind backoff.
    """

    from agent.logger import log_event

    delay = initial_delay
    last_error = None

    log_event("LLM", "safe_invoke called", {"max_retries": max_retries})

    for attempt in range(max_retries):
        try:
            log_event("LLM", f"Invoking LLM (attempt {attempt + 1}/{max_retries})")
            normalized_input = _normalize_bedrock_input(input_data)
            result = agent_or_llm.invoke(normalized_input)
            log_event("LLM", "LLM invoke completed successfully")

            # Check for empty response (retry if empty AND no tools called)
            content = getattr(result, 'content', None)
            has_tools = False
            if hasattr(result, 'tool_calls') and result.tool_calls:
                has_tools = True
            if hasattr(result, 'additional_kwargs') and result.additional_kwargs.get('tool_calls'):
                has_tools = True

            if (content is None or (isinstance(content, str) and content.strip() == "")) and not has_tools:
                if attempt < max_retries - 1:
                    safe_print(f"⚠️ Empty response, retrying ({attempt + 1}/{max_retries})...")
                    log_event("LLM", "Empty response, retrying", {"attempt": attempt + 1})
                    time.sleep(delay)
                    delay *= 1.5
                    continue

            # --- Track token cost + usage (AIMessage or agent/graph state dict) ---
            _capture_usage(result, "invoke")

            return result

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            log_event("LLM", "LLM invoke failed", {
                "error": str(e),
                "error_type": type(e).__name__,
                "attempt": attempt + 1
            })

            # A TPM/RPM rate limit (429) is a hard quota ceiling — retrying it just
            # wastes the user's time on a window that won't clear in seconds. Fail
            # fast into the caller's fallback/degraded path.
            if _is_rate_limit_error(error_str) and not _rate_limit_retries_enabled():
                safe_print("⛔ Rate limit (TPM/RPM) — not retrying; failing over fast.")
                log_event("LLM", "Rate-limit error; failing fast (no retry)", {"error": str(e)})
                raise

            # Retry on gzip errors, connection issues, or Anthropic model_output_error
            # (empty content block — transient, safe to retry).
            should_retry = any(err in error_str for err in _RETRYABLE_ERROR_SUBSTRINGS)

            if should_retry and attempt < max_retries - 1:
                # Honor the server's Retry-After (a 429 is often a per-second/RPM
                # or per-minute TPM cap that states exactly when to retry), capped
                # so we never blind-wait the old 3→6→12→24→48s (~93s) ramp — which
                # just stalls the user when a stream falls back here on a throttle.
                try:
                    cap = float(os.environ.get("LLM_RETRY_CAP", "15"))
                except ValueError:
                    cap = 15.0
                wait = min(_retry_after_seconds(e, delay), cap)
                safe_print(f"⏳ API rate/token-limited, waiting {int(wait)}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                delay = min(delay * 2, cap)
            else:
                # CRITICAL: Do NOT use logger.exception here as it crashes on closed I/O
                # Just raise it and let the caller handle or print
                raise

    # If we exhausted retries due to empty responses
    if last_error:
        raise last_error
    return AIMessage(content="I apologize, but I'm having trouble connecting. Please try again.")

def safe_stream(llm, input_data, is_cancelled_func=None):
    """
    Stream from LLM with token cost tracking.
    Yields chunks and captures usage_metadata from the final chunk.
    """
    last_chunk = None
    # A 429 here is usually a TOKENS-per-minute (TPM) cap, not a request-rate one,
    # so a long blind backoff is pointless: if a single request exceeds the whole
    # quota it can never fit, and when the window WILL free up the server says so
    # via Retry-After. So: few retries, honor Retry-After (capped), then fail fast
    # into the caller's degraded/failed handling instead of stalling the user.
    #   LLM_STREAM_RETRIES   retry attempts (default 2)
    #   LLM_STREAM_RETRY_CAP max seconds to wait per retry (default 20)
    try:
        max_retries = max(1, int(os.environ.get("LLM_STREAM_RETRIES", "2")))
    except ValueError:
        max_retries = 2
    try:
        wait_cap = float(os.environ.get("LLM_STREAM_RETRY_CAP", "20"))
    except ValueError:
        wait_cap = 20.0
    delay = 3.0

    normalized_input = _normalize_bedrock_input(input_data)

    for attempt in range(max_retries):
        try:
            for chunk in llm.stream(normalized_input):
                if is_cancelled_func and is_cancelled_func():
                    break
                last_chunk = chunk
                yield chunk
            break # Success, exit retry loop to track costs
        except Exception as e:
            error_str = str(e).lower()
            # A TPM/RPM rate limit (429) can't clear in a few seconds — don't burn
            # retries on it. Fail fast so the caller falls back immediately.
            if _is_rate_limit_error(error_str) and not _rate_limit_retries_enabled():
                safe_print("⛔ Stream rate limit (TPM/RPM) — not retrying; falling back fast.")
                raise e
            # Retryable streaming errors: transient (gzip/connection/5xx) OR Anthropic
            # model_output_error (empty content block — safe to retry).
            is_retryable_stream_error = any(err in error_str for err in _RETRYABLE_ERROR_SUBSTRINGS)
            if attempt >= max_retries - 1 or not is_retryable_stream_error:
                raise e

            import time
            wait = min(_retry_after_seconds(e, delay), wait_cap)
            safe_print(
                f"⏳ Stream rate/token-limited, waiting {int(wait)}s before retry "
                f"({attempt + 1}/{max_retries})..."
            )
            time.sleep(wait)
            delay = min(delay * 2, wait_cap)

    # --- Track token cost from the final chunk ---
    if last_chunk:
        _capture_usage(last_chunk, "stream")

def retry_with_backoff(max_retries=3, initial_delay=2):
    """Decorator to retry function calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                        if attempt < max_retries - 1:
                            safe_print(f"⏳ Rate limited, waiting {delay}s before retry...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator

def get_st_aware_func(func):
    """Wrap a function to preserve Streamlit, run, and profile context in threads.

    copy_context() captures the active-profile ContextVar at wrap time, so tools
    submitted to a ThreadPoolExecutor from a correctly-bound thread keep that
    profile. We additionally re-bind it explicitly inside the worker: if the
    captured context is ever run on a thread whose profile drifts, the explicit
    bind guarantees portfolio tools still resolve the profile that was active
    when the work was scheduled — never a stale or process-global one.
    """
    execution_context = copy_context()
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        ctx = getattr(threading.current_thread(), "streamlit_script_run_ctx", None)
    except ImportError:
        ctx = None

    try:
        from tools.user_profile import reset_profile, set_active_profile
        captured_profile = execution_context.run(_capture_active_profile)
    except Exception:
        reset_profile = set_active_profile = None
        captured_profile = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        def invoke():
            if ctx is not None:
                add_script_run_ctx(threading.current_thread(), ctx)
            if set_active_profile is not None and captured_profile is not None:
                token = set_active_profile(captured_profile)
                try:
                    return func(*args, **kwargs)
                finally:
                    reset_profile(token)
            return func(*args, **kwargs)

        return execution_context.run(invoke)

    return wrapper


def _capture_active_profile():
    """Resolve the active profile (used inside a copied context at wrap time)."""
    from tools.user_profile import get_active_profile
    return get_active_profile()

def suppress_streamlit_context_warnings():
    """Globally suppress the annoying Streamlit missing context warnings from background threads."""
    try:
        import logging


        # Streamlit uses its own logger wrapper. We'll add a filter to the specific logger
        st_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")

        class SuppressContextWarningFilter(logging.Filter):
            def filter(self, record):
                if "missing ScriptRunContext" in record.getMessage():
                    return False
                return True

        st_logger.addFilter(SuppressContextWarningFilter())
    except Exception:
        pass

# Execute on import
suppress_streamlit_context_warnings()
