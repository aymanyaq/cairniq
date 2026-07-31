import os
import threading
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache" / "dspy"

# LiteLLM reads this flag during import time. Set it before importing DSPy/LiteLLM
# so offline environments use the bundled model cost map instead of attempting a
# remote fetch on every cold import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("DSPY_CACHEDIR", str(_DEFAULT_CACHE_DIR))
_DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# DSPy's evaluate module does `from IPython.display import HTML, display` at import
# time, which drags in the full IPython terminal stack (~6s on a cold start) purely
# for Jupyter rendering we never use in this server. Pre-seed a no-op stub so that
# import is satisfied cheaply and IPython.terminal never loads. Only effective when
# IPython hasn't already been imported (e.g. a notebook), so it's safe there too.
import sys as _sys
import types as _types

if "IPython.display" not in _sys.modules and "IPython" not in _sys.modules:
    _ipy = _types.ModuleType("IPython")
    _ipy_display = _types.ModuleType("IPython.display")
    _ipy_display.display = _ipy_display.HTML = lambda *a, **k: None
    _ipy.display = _ipy_display
    _sys.modules["IPython"] = _ipy
    _sys.modules["IPython.display"] = _ipy_display

try:
    import dspy as _dspy

    dspy = _dspy
    DSPY_AVAILABLE = True
except ImportError:
    dspy = None
    DSPY_AVAILABLE = False


_DSPY_CONFIG_LOCK = threading.Lock()


def _get_bedrock_litellm_kwargs(region_name: str) -> dict[str, str]:
    """Return LiteLLM Bedrock auth kwargs without relying on ambient profiles."""
    kwargs = {"aws_region_name": region_name}
    try:
        from tools.secrets_store import (
            clear_incompatible_aws_session_token,
            get_secret,
            load_secrets_into_env,
        )

        load_secrets_into_env()
        clear_incompatible_aws_session_token()
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or get_secret("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or get_secret("AWS_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN") or get_secret("AWS_SESSION_TOKEN")
    except Exception:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        session_token = os.environ.get("AWS_SESSION_TOKEN", "")

    if not access_key or not secret_key:
        return kwargs

    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    kwargs["aws_access_key_id"] = access_key
    kwargs["aws_secret_access_key"] = secret_key

    if session_token and not access_key.startswith("AKIA"):
        kwargs["aws_session_token"] = session_token
    elif access_key.startswith("AKIA"):
        os.environ.pop("AWS_SESSION_TOKEN", None)
        os.environ.pop("AWS_PROFILE", None)

    return kwargs


def get_bedrock_model_name(model_id: str | None = None) -> str:
    model_name = model_id or os.environ.get("AIDLC_MODEL_ID")
    if not model_name:
        raise ValueError("AIDLC_MODEL_ID environment variable is not set")
    if not model_name.startswith("bedrock/"):
        model_name = f"bedrock/{model_name}"
    return model_name


def _build_litellm_lm(model_id: str | None, region_name: str):
    """Build a DSPy LM for the active LLM provider.

    DSPy wraps LiteLLM, which supports every provider we target. The LiteLLM
    model-string + auth mapping mirrors agent.utils.get_llm() exactly so DSPy
    extraction works on the same provider as the rest of the app:

      bedrock        -> bedrock/<id>            (AWS creds)        [CachingBedrockLM]
      openai         -> openai/<model>          (OPENAI_API_KEY)
      anthropic      -> anthropic/<model>       (ANTHROPIC_API_KEY)
      google         -> gemini/<model>          (GOOGLE_API_KEY)
      vertexai       -> vertex_ai/<model>       (GOOGLE_SERVICE_ACCOUNT_KEY JSON)
      azure (claude) -> anthropic/<deployment>  (Foundry /anthropic Messages surface)
      azure (v1)     -> openai/<deployment>     (OpenAI-compatible Foundry surface)
      azure (legacy) -> azure/<deployment>      (api_base + api_version)

    Returns a configured dspy LM, or raises ValueError on misconfiguration.
    """
    provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()

    if provider == "bedrock":
        model_name = get_bedrock_model_name(model_id)
        return CachingBedrockLM(model_name, **_get_bedrock_litellm_kwargs(region_name))

    model = model_id or os.environ.get("AIDLC_MODEL_ID")

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return CleaningLM(f"openai/{model or 'gpt-4o-mini'}", api_key=api_key)

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        if not model:
            raise ValueError("LLM_PROVIDER=anthropic but AIDLC_MODEL_ID is not set")
        # Honours [cachePoint] rather than deleting it — see AnthropicCachingLM.
        return AnthropicCachingLM(f"anthropic/{model}", api_key=api_key)

    if provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER=google but GOOGLE_API_KEY is not set")
        return CleaningLM(f"gemini/{model or 'gemini-2.5-flash'}", api_key=api_key)

    if provider == "vertexai":
        # Auth is a service-account key stored in the keychain
        # (GOOGLE_SERVICE_ACCOUNT_KEY, JSON). LiteLLM's vertex_ai/ route accepts
        # the JSON string directly as vertex_credentials; project defaults to the
        # key's own project_id unless GOOGLE_CLOUD_PROJECT overrides it.
        import json as _json

        sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
        if not sa_json:
            try:
                from tools.secrets_store import get_secret
                sa_json = get_secret("GOOGLE_SERVICE_ACCOUNT_KEY")
            except Exception:
                sa_json = ""
        if not sa_json:
            raise ValueError(
                "LLM_PROVIDER=vertexai but GOOGLE_SERVICE_ACCOUNT_KEY is not set. "
                "Paste your Vertex AI service-account key (JSON) in Settings."
            )
        try:
            info = _json.loads(sa_json)
        except ValueError as exc:
            raise ValueError("GOOGLE_SERVICE_ACCOUNT_KEY is not valid JSON.") from exc
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or info.get("project_id")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        return CleaningLM(
            f"vertex_ai/{model or 'gemini-2.5-flash'}",
            vertex_project=project,
            vertex_location=location,
            vertex_credentials=sa_json,
        )

    if provider == "azure":
        api_key = (os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
        endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        if not api_key or not endpoint:
            raise ValueError("LLM_PROVIDER=azure but AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT are not set")
        if not model:
            raise ValueError("LLM_PROVIDER=azure but no deployment is configured (AIDLC_MODEL_ID)")
        # Foundry "/anthropic" surface serves Claude (format="Anthropic") via the
        # Anthropic Messages API at {endpoint}/v1/messages — route through
        # LiteLLM's anthropic/ provider with api_base (same surface the chat path
        # hands to ChatAnthropic). The azure/ route below would 404 here.
        if endpoint.endswith("/anthropic"):
            return CleaningLM(f"anthropic/{model}", api_key=api_key, api_base=endpoint)
        # Foundry "/openai/v1" surface is OpenAI-compatible — route through the
        # openai/ provider with a custom base, same as the chat path. The bare
        # resource URL uses LiteLLM's azure/ route with api_version.
        if endpoint.endswith("/openai/v1"):
            return CleaningLM(f"openai/{model}", api_key=api_key, api_base=endpoint)
        return CleaningLM(
            f"azure/{model}",
            api_key=api_key,
            api_base=endpoint,
            # gpt-5.x / o-series reject 2024-10-21 ("API version not supported").
            # Default to a current preview; override via AZURE_OPENAI_API_VERSION.
            api_version=(os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip() or "2024-12-01-preview",
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


class _MeteredLM(dspy.LM):
    """Shared base for our LM subclasses: prompt rewriting + cost metering.

    Subclasses override `_preprocess_messages` only.

    Metering lives here because DSPy never routes through agent.utils.safe_invoke,
    so `_capture_usage` — the path that feeds the cost meter for every other LLM
    call in the app — cannot see a single DSPy call. Hooking the LM itself catches
    them all regardless of which module issued the prediction.

    The flush hangs off `__call__`/`acall`, NOT `forward`/`aforward`:
    dspy.BaseLM.__call__ runs `forward()` first and only then appends the history
    entry (in `_process_lm_response`). Flushing inside `forward` would therefore
    always read a history that is missing the call just made — lagging one call
    behind forever and dropping the last call of every session.
    """

    def _preprocess_messages(self, messages):
        return messages

    def _flush_usage(self):
        try:
            from agent.cost_tracker import track_dspy_calls
            track_dspy_calls(self, getattr(self, "model", "") or "")
        except Exception:
            pass

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = self._preprocess_messages(messages)
        return super().forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        messages = self._preprocess_messages(messages)
        return await super().aforward(prompt=prompt, messages=messages, **kwargs)

    def __call__(self, prompt=None, messages=None, **kwargs):
        try:
            return super().__call__(prompt=prompt, messages=messages, **kwargs)
        finally:
            self._flush_usage()

    async def acall(self, prompt=None, messages=None, **kwargs):
        try:
            return await super().acall(prompt=prompt, messages=messages, **kwargs)
        finally:
            self._flush_usage()


class CleaningLM(_MeteredLM):
    """Strips literal [cachePoint] markers from system prompts.

    Used by providers whose API has no cache-marker concept, where the marker
    would otherwise reach the model as literal text.
    """
    def _preprocess_messages(self, messages):
        if not messages:
            return messages
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                content_str = msg["content"]
                if "[cachePoint]" in content_str:
                    new_messages.append({"role": "system", "content": content_str.replace("[cachePoint]", "")})
                    continue
            new_messages.append(msg)
        return new_messages


class AnthropicCachingLM(_MeteredLM):
    """Splits a system prompt on '[cachePoint]' into Anthropic cache_control blocks.

    Same markers, inverse encoding to Bedrock's: Anthropic tags the LAST block of
    the cached prefix rather than inserting a separator. Without this the markers
    were simply deleted (CleaningLM) and the stable prefix was re-billed on every
    DSPy call.
    """
    def _preprocess_messages(self, messages):
        if not messages:
            return messages
        from agent.utils import _anthropic_cache_blocks

        new_messages = []
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                content_str = msg["content"]
                if "[cachePoint]" in content_str:
                    parts = content_str.split("[cachePoint]")
                    structured: list = []
                    for i, part in enumerate(parts):
                        if part:
                            structured.append({"text": part})
                        if i < len(parts) - 1:
                            structured.append({"cachePoint": {"type": "default"}})
                    new_messages.append({
                        "role": "system",
                        "content": _anthropic_cache_blocks(structured),
                    })
                    continue
            new_messages.append(msg)
        return new_messages


class CachingBedrockLM(_MeteredLM):
    """Splits a system prompt on '[cachePoint]' into Bedrock cache-point blocks."""
    def _preprocess_messages(self, messages):
        if not messages:
            return messages
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                content_str = msg["content"]
                if "[cachePoint]" in content_str:
                    parts = content_str.split("[cachePoint]")
                    content_list = []
                    for i, part in enumerate(parts):
                        if part:
                            content_list.append({"type": "text", "text": part})
                        if i < len(parts) - 1:
                            content_list.append({"cachePoint": {"type": "default"}})
                    new_messages.append({"role": "system", "content": content_list})
                    continue
            new_messages.append(msg)
        return new_messages


def configure_dspy(
    model_id: str | None = None,
    region: str | None = None,
    error_callback: Callable[[str], None] | None = None,
) -> bool:
    """Configure the shared DSPy LM once for the active provider.

    Provider-agnostic: builds the right LiteLLM-backed LM for bedrock, openai,
    anthropic, google, vertexai, or azure (see _build_litellm_lm). Idempotent and safe to
    call from every node module at import time — the global LM is set once.
    """
    if not DSPY_AVAILABLE:
        return False

    region_name = region or os.environ.get("AWS_REGION", "us-east-1")

    try:
        with _DSPY_CONFIG_LOCK:
            if not getattr(dspy.settings, "lm", None):
                lm = _build_litellm_lm(model_id, region_name)
                dspy.settings.configure(lm=lm)
        return True
    except Exception as exc:
        if error_callback:
            error_callback(f"⚠️ Failed to configure DSPy: {exc}")
        return False


def reconfigure_dspy(
    region: str | None = None,
    error_callback: Callable[[str], None] | None = None,
) -> bool:
    """Rebuild the shared DSPy LM for the CURRENT provider/model, replacing any
    existing one.

    Unlike configure_dspy() (which is idempotent and set-once), this forces a
    swap. Call it after a runtime LLM_PROVIDER / model change so DSPy stops using
    the previously-configured provider without needing a process restart. Reads
    the active provider + model fresh from the environment via _build_litellm_lm.
    """
    if not DSPY_AVAILABLE:
        return False

    region_name = region or os.environ.get("AWS_REGION", "us-east-1")
    try:
        with _DSPY_CONFIG_LOCK:
            lm = _build_litellm_lm(None, region_name)
            dspy.settings.configure(lm=lm)
        # The fast LM is resolved from the same env; a provider swap invalidates
        # it too, or mechanical work keeps going to the previous provider.
        reset_fast_dspy_lm()
        return True
    except Exception as exc:
        if error_callback:
            error_callback(f"⚠️ Failed to reconfigure DSPy: {exc}")
        return False


_FAST_LM = None
_FAST_LM_RESOLVED = False
_FAST_LM_LOCK = threading.Lock()


def get_fast_dspy_lm():
    """A second DSPy LM bound to the FAST slot, built once and reused.

    DSPy has exactly ONE global LM, and it comes from the primary slot. That
    meant every DSPy call was billed at deep-tier rates — including
    ContextExtraction, which is mechanical JSON extraction over a single short
    message and runs on every non-ghost turn.

    Returns None when no distinct fast model is configured (or DSPy is
    unavailable, or the build fails), so callers fall back to the global LM
    rather than losing the extraction entirely. The failure is remembered so a
    misconfigured fast slot costs one build attempt per process, not one per turn.
    """
    global _FAST_LM, _FAST_LM_RESOLVED

    if _FAST_LM_RESOLVED:
        return _FAST_LM
    if not DSPY_AVAILABLE:
        return None

    with _FAST_LM_LOCK:
        if _FAST_LM_RESOLVED:
            return _FAST_LM
        try:
            from agent.utils import _resolve_model_id

            fast_id = _resolve_model_id("fast")
            primary_id = _resolve_model_id("primary")
            # No point paying for a second client when both slots are one model.
            if fast_id and fast_id != primary_id:
                region = os.environ.get("AWS_REGION", "us-east-1")
                _FAST_LM = _build_litellm_lm(fast_id, region)
        except Exception:
            _FAST_LM = None
        finally:
            _FAST_LM_RESOLVED = True

    return _FAST_LM


def reset_fast_dspy_lm() -> None:
    """Forget the cached fast LM so the next call rebuilds it.

    Needed after a runtime provider/model change, alongside reconfigure_dspy().
    """
    global _FAST_LM, _FAST_LM_RESOLVED
    with _FAST_LM_LOCK:
        _FAST_LM = None
        _FAST_LM_RESOLVED = False


@contextmanager
def fast_dspy_context():
    """Scope the enclosed DSPy calls to the fast slot, when one is configured.

    Use for mechanical work — extraction, classification, reformatting — where
    the deep tier buys nothing. Falls through to the global LM when no separate
    fast model exists, so the caller never has to branch.
    """
    lm = get_fast_dspy_lm()
    with (dspy.context(lm=lm) if lm is not None else nullcontext()):
        yield


# Backward-compatible alias. The configuration is now provider-aware, so the
# "_for_bedrock" name is a misnomer kept only to avoid breaking external imports.
configure_dspy_for_bedrock = configure_dspy
