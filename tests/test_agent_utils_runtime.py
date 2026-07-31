import os
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.dspy_setup import _get_bedrock_litellm_kwargs
from agent.utils import (
    _capture_usage,
    _system_prompt_message,
    activate_run_context,
    build_run_context,
    ensure_bedrock_sequence,
    extract_visible_text,
    get_bedrock_credential_kwargs,
    get_st_aware_func,
    request_cancellation,
    reset_run_context,
    safe_invoke,
    send_status,
    send_stream,
)
from tools.secrets_store import clear_incompatible_aws_session_token


class CaptureAgent:
    def __init__(self):
        self.captured = None

    def invoke(self, input_data):
        self.captured = input_data
        return AIMessage(content="ok")


def test_run_context_callbacks_propagate_to_background_threads():
    events = []
    run_context = build_run_context(
        on_token=lambda token: events.append(("token", token)),
        on_status=lambda status: events.append(("status", status)),
    )
    token = activate_run_context(run_context)

    try:
        wrapped = get_st_aware_func(
            lambda: (send_status("isolated-status"), send_stream("isolated-token"))
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(wrapped).result()
    finally:
        reset_run_context(token)

    assert ("status", "isolated-status") in events
    assert ("token", "isolated-token") in events


def test_request_cancellation_can_target_one_run_context():
    run_a = build_run_context()
    run_b = build_run_context()

    request_cancellation(run_a.cancel_event)

    assert run_a.cancel_event.is_set()
    assert not run_b.cancel_event.is_set()


def test_extract_visible_text_strips_hidden_reasoning_and_node_prefixes():
    rendered = extract_visible_text(
        "[DeepReasoning]: <thinking>internal debate</thinking>\nVisible conclusion",
        strip_node_prefix=True,
    )

    assert rendered == "Visible conclusion"
    assert extract_visible_text("<thinking>still hidden") == ""


def test_safe_invoke_repairs_dangling_bedrock_tool_calls(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    agent = CaptureAgent()
    tool_use_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tooluse_123", "name": "scan_market", "args": {}}],
    )
    input_messages = [
        HumanMessage(content="Run the scan."),
        tool_use_msg,
        HumanMessage(content="What did you find?"),
    ]

    safe_invoke(agent, {"messages": input_messages}, max_retries=1)

    repaired_messages = agent.captured["messages"]
    assert len(repaired_messages) == 4
    assert isinstance(repaired_messages[2], ToolMessage)
    assert repaired_messages[2].tool_call_id == "tooluse_123"
    assert "scan_market" in repaired_messages[2].content


def test_safe_invoke_leaves_non_bedrock_messages_unchanged(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    agent = CaptureAgent()
    tool_use_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tooluse_123", "name": "scan_market", "args": {}}],
    )
    input_messages = [
        HumanMessage(content="Run the scan."),
        tool_use_msg,
        HumanMessage(content="What did you find?"),
    ]

    safe_invoke(agent, {"messages": input_messages}, max_retries=1)

    assert agent.captured["messages"] == input_messages


def test_structured_system_prompts_are_provider_aware(monkeypatch):
    structured_prompt = [
        {"text": "Static instructions"},
        {"cachePoint": {"type": "default"}},
        {"text": "Dynamic context"},
    ]

    # Non-Bedrock: cachePoint blocks flatten to text, returned as a SystemMessage
    # OBJECT (never a ("system", text) tuple — tuples get f-string-template parsed
    # by ChatPromptTemplate, and tool-result JSON braces then crash the prompt).
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    flat_msg = _system_prompt_message(structured_prompt)
    assert isinstance(flat_msg, SystemMessage)
    assert flat_msg.content == "Static instructions\nDynamic context"

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    bedrock_msg = _system_prompt_message(structured_prompt)
    assert isinstance(bedrock_msg, SystemMessage)
    assert bedrock_msg.content == structured_prompt


def test_create_agent_survives_braces_in_system_prompt_on_non_bedrock(monkeypatch):
    """Regression: RiskManager embeds raw tool-result JSON (full of '{') in its
    system prompt. On azure/openai/google the old ("system", text) tuple path made
    ChatPromptTemplate f-string-parse it -> ValueError: unmatched '{' in format spec."""
    from langchain_core.runnables import RunnableLambda

    from agent.utils import create_agent

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    brace_heavy = (
        "You are the Risk Compliance Judge.\n"
        '<tool_execution_context>{"snapshot": {"total": 1, "items": [{"x": 1}]}}</tool_execution_context>'
    )
    agent = create_agent(RunnableLambda(lambda value: value), [], brace_heavy)  # must not raise
    rendered = agent.invoke({"messages": [HumanMessage(content="check")]})
    assert '{"snapshot"' in rendered.messages[0].content  # braces intact, not templated


def test_create_agent_adds_google_search_grounding_on_vertexai(monkeypatch):
    """Gemini 3 can natively combine Grounding with Google Search with our custom
    finance tools in one request — create_agent should append it for google/vertexai
    only, and leave other providers' tool lists untouched."""
    from langchain_core.runnables import RunnableLambda

    from agent.utils import create_agent

    class StubLLM:
        def __init__(self):
            self.bound_tools = None

        def bind_tools(self, tools):
            self.bound_tools = tools
            return RunnableLambda(lambda value: value)

    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    stub = StubLLM()
    create_agent(stub, ["some_tool"], "system prompt")
    assert stub.bound_tools == ["some_tool", {"google_search": {}}]

    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    monkeypatch.setenv("AIDLC_GOOGLE_SEARCH_GROUNDING", "0")
    stub_disabled = StubLLM()
    create_agent(stub_disabled, ["some_tool"], "system prompt")
    assert stub_disabled.bound_tools == ["some_tool"]
    monkeypatch.delenv("AIDLC_GOOGLE_SEARCH_GROUNDING", raising=False)

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    stub_other = StubLLM()
    create_agent(stub_other, ["some_tool"], "system prompt")
    assert stub_other.bound_tools == ["some_tool"]


def test_capture_usage_logs_when_google_search_grounding_fires(monkeypatch):
    """We have no other way to tell whether Gemini's native google_search
    grounding actually fired — it runs server-side, never as a dispatched
    tool_call, so ToolExecution logging never sees it. langchain-google-genai
    surfaces it in response_metadata["grounding_metadata"] only when used."""
    events = []
    monkeypatch.setattr("agent.logger.log_event", lambda phase, msg, data=None: events.append((phase, msg, data)))

    grounded_msg = AIMessage(
        content="SearXNG is a metasearch engine.",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={
            "model_name": "gemini-3.1-pro-preview",
            "grounding_metadata": {
                "web_search_queries": ["what is searxng"],
                "grounding_chunks": [
                    {"web": {"title": "SearXNG GitHub", "uri": "https://github.com/searxng/searxng"}}
                ],
            },
        },
    )
    _capture_usage(grounded_msg, "invoke")

    grounding_events = [e for e in events if e[0] == "Grounding"]
    assert len(grounding_events) == 1
    _, _, data = grounding_events[0]
    assert data["web_search_queries"] == ["what is searxng"]
    assert data["source_count"] == 1
    assert data["sources"][0]["uri"] == "https://github.com/searxng/searxng"


def test_capture_usage_does_not_log_grounding_when_absent(monkeypatch):
    events = []
    monkeypatch.setattr("agent.logger.log_event", lambda phase, msg, data=None: events.append((phase, msg, data)))

    ungrounded_msg = AIMessage(
        content="Some answer using a custom tool instead.",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "gemini-3.1-pro-preview"},
    )
    _capture_usage(ungrounded_msg, "invoke")

    assert not [e for e in events if e[0] == "Grounding"]


def test_bedrock_sequence_repair_preserves_existing_tool_results():
    tool_use_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "tooluse_1", "name": "scan_market", "args": {}},
            {"id": "tooluse_2", "name": "read_news", "args": {}},
        ],
    )
    existing_result = ToolMessage(
        content="scan complete",
        tool_call_id="tooluse_1",
        name="scan_market",
    )

    repaired = ensure_bedrock_sequence([
        HumanMessage(content="Run the scan."),
        tool_use_msg,
        existing_result,
        HumanMessage(content="What did you find?"),
    ])

    assert repaired[2] is existing_result
    assert isinstance(repaired[3], ToolMessage)
    assert repaired[3].tool_call_id == "tooluse_2"
    assert repaired[4].content == "What did you find?"


def test_bedrock_credential_kwargs_ignore_blank_placeholders(monkeypatch):
    monkeypatch.setattr("tools.secrets_store.get_secret", lambda k: "")
    monkeypatch.setattr("tools.secrets_store.load_secrets_into_env", lambda: 0)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale")

    assert get_bedrock_credential_kwargs() == {}


def test_static_aws_keys_drop_stale_session_token(monkeypatch):
    monkeypatch.setattr("tools.secrets_store.get_secret", lambda k: "")
    monkeypatch.setattr("tools.secrets_store.load_secrets_into_env", lambda: 0)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA1234567890ABCDEF")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s" * 40)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-token")
    monkeypatch.setenv("AWS_PROFILE", "stale-sso")

    cleanup = clear_incompatible_aws_session_token()

    assert cleanup["cleared"] is True
    assert "AWS_SESSION_TOKEN" not in os.environ
    assert "AWS_PROFILE" not in os.environ
    assert get_bedrock_credential_kwargs() == {
        "aws_access_key_id": "AKIA1234567890ABCDEF",
        "aws_secret_access_key": "s" * 40,
    }


def test_dspy_bedrock_kwargs_use_litellm_aws_names(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA1234567890ABCDEF")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s" * 40)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-token")
    monkeypatch.setenv("AWS_PROFILE", "stale-sso")

    kwargs = _get_bedrock_litellm_kwargs("us-east-1")

    assert kwargs == {
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "AKIA1234567890ABCDEF",
        "aws_secret_access_key": "s" * 40,
    }
    assert "AWS_SESSION_TOKEN" not in os.environ
    assert "AWS_PROFILE" not in os.environ


def test_extract_stream_text_suppresses_tool_call_args():
    """Streaming a tool-bound LLM must not leak tool-call args (ticker lists) to the UI."""
    from agent.utils import extract_stream_text

    # Plain text passes through
    assert extract_stream_text("hello") == "hello"

    # Anthropic-style list: text kept, tool_use dropped
    content = [
        {"type": "text", "text": "Running the screen."},
        {"type": "tool_use", "name": "find_breakout_candidates",
         "input": {"symbols_str": "NVDA, META, AVGO"}, "id": "1"},
    ]
    assert extract_stream_text(content) == "Running the screen."

    # Streaming tool-arg deltas and bare-string args are dropped (the observed leak)
    assert extract_stream_text([{"type": "input_json_delta", "partial_json": "NVDA, META"}]) == ""
    assert extract_stream_text(["All", "Aggressive", "momentum"]) == ""

    class _RawDelta:
        def __str__(self):
            return "NVDA, META, AVGO"
    assert extract_stream_text([_RawDelta()]) == ""


def test_strip_tool_call_tokens_removes_kimi_native_tool_syntax():
    """Kimi/Moonshot serialize tool calls as <|tool_call…|> tokens. When a tool-less
    call emits them as text, they must never reach the UI or get persisted."""
    from agent.utils import extract_stream_text, extract_visible_text, strip_tool_call_tokens

    leaked = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.check_portfolio_allocation:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
        "<|tool_call_begin|>functions.analyze_fx_risks:1<|tool_call_argument_begin|>{}<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    # The whole leaked block is removed, real prose around it is preserved
    assert strip_tool_call_tokens(leaked) == ""
    assert strip_tool_call_tokens(f"Tech leads.\n\n{leaked}\n\nStay put.") == "Tech leads.\n\n\n\nStay put."

    # The shared extractors apply the backstop on both the streaming and final paths
    assert extract_stream_text(leaked) == ""
    assert extract_visible_text(f"Verdict: hold.{leaked}") == "Verdict: hold."


def test_strip_scaffold_tags_removes_leaked_output_format_wrapper():
    """RiskManager's prompt wraps its template in a literal <output_format strict="true">
    tag and tells the model to omit it — weaker completions sometimes echo the tag anyway.
    The wrapper must be stripped while the real Markdown content inside is preserved."""
    from agent.utils import extract_stream_text, extract_visible_text, strip_scaffold_tags

    leaked = (
        '<output_format strict="true">\n'
        "⚖️ **Verdict: 10/10** — Clean portfolio audit.\n\n"
        "🔴 **Risks:** None flagged.\n\n"
        "🤔 **Devil's Advocate:** Some caveat.\n"
        "</output_format>"
    )
    cleaned = strip_scaffold_tags(leaked)
    assert "<output_format" not in cleaned
    assert "</output_format>" not in cleaned
    assert "⚖️ **Verdict: 10/10** — Clean portfolio audit." in cleaned
    assert "🤔 **Devil's Advocate:** Some caveat." in cleaned

    # The shared extractors apply the same backstop
    assert "<output_format" not in extract_visible_text(leaked)
    assert "<output_format" not in extract_stream_text(leaked)

    # Non-tag angle brackets in real prose are left alone
    assert strip_scaffold_tags("Revenue < guidance, margin > 20%") == "Revenue < guidance, margin > 20%"

    # Non-str content is returned unchanged (type-safe)
    assert strip_scaffold_tags(None) is None


def test_strip_scaffold_tags_removes_orphaned_fragment_missing_leading_bracket():
    """A weak completion once echoed RiskManager's tag WITHOUT the leading '<' —
    the literal text `output_format strict="true">` reached the chat UI because
    the full-tag regex requires the '<'. Orphaned fragments (known tag name, or
    any snake_case name with attrs, at line start with '>' glued on) must be
    stripped too, while ordinary prose containing '>' stays intact."""
    from agent.utils import extract_stream_text, extract_visible_text, strip_scaffold_tags

    # The exact leak observed in production: opening tag missing its '<'
    leaked = (
        'output_format strict="true">\n'
        "⚖️ **Verdict: 8/10** — Concentrated but coherent.\n\n"
        "🔴 **Risks:** Sector overlap.\n"
    )
    cleaned = strip_scaffold_tags(leaked)
    assert "output_format" not in cleaned
    assert cleaned.startswith("⚖️ **Verdict: 8/10**")
    assert "🔴 **Risks:** Sector overlap." in cleaned

    # Closing tag missing its '<' (mid-text, after a newline)
    assert strip_scaffold_tags("⚖️ Verdict stands.\n/output_format>\n") == "⚖️ Verdict stands.\n"

    # Bare known tag names without attrs are also caught
    assert strip_scaffold_tags("rules>\nHold the position.") == "Hold the position."

    # Unknown snake_case name is still caught when it carries the attr="..." shape
    assert strip_scaffold_tags('brand_new_tag strict="true">\nContent.') == "Content."

    # The shared extractors apply the same backstop on both paths
    assert extract_visible_text(leaked).startswith("⚖️ **Verdict: 8/10**")
    assert "output_format" not in extract_stream_text(leaked)

    # Ordinary prose with '>' is never eaten — even at line start, even when the
    # word before '>' is a known tag name (space before '>' breaks the match),
    # and even when an unknown word has '>' glued on (no attrs, not a known name).
    for prose in (
        "Revenue < guidance, margin > 20%",
        "x > 5 implies momentum",
        "today > yesterday for tech",
        "risk > reward on this entry",
        "margin>20% is healthy",
        "> quoted reply in markdown",
        "A -> B rotation continues",
    ):
        assert strip_scaffold_tags(prose) == prose


# --- Vendor-neutral LLM factory (provider dispatch) --------------------------
import pytest

from agent.utils import get_llm, get_sonnet_llm


def _clear_model_env(monkeypatch):
    # Clear generic AND provider-scoped model vars. The scoped vars
    # (AIDLC_MODEL_ID_<PROVIDER>) take precedence in _resolve_model_id, so a real
    # user's user_data/.env (e.g. AIDLC_MODEL_ID_AZURE=Kimi-K2.6) would otherwise
    # leak into tests that only set the generic var.
    monkeypatch.delenv("AIDLC_MODEL_ID", raising=False)
    monkeypatch.delenv("AIDLC_SONNET_MODEL_ID", raising=False)
    monkeypatch.delenv("AIDLC_EMBED_MODEL_ID", raising=False)
    for prov in ("BEDROCK", "OPENAI", "ANTHROPIC", "AZURE", "GOOGLE"):
        monkeypatch.delenv(f"AIDLC_MODEL_ID_{prov}", raising=False)
        monkeypatch.delenv(f"AIDLC_SONNET_MODEL_ID_{prov}", raising=False)
        monkeypatch.delenv(f"AIDLC_EMBED_MODEL_ID_{prov}", raising=False)


@pytest.fixture(autouse=True)
def _isolate_model_env(monkeypatch):
    """Strip any AIDLC_* model vars leaking from a real user_data/.env so each
    test controls its own model resolution. Tests set what they need explicitly.
    Also clears API keys and endpoints to prevent leaks from the user's OS keyring."""
    _clear_model_env(monkeypatch)

    # Clear provider keys and endpoints to prevent leaks from OS keyring / .env
    for key in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY_FAST",
        "AZURE_OPENAI_ENDPOINT_FAST",
        "AZURE_OPENAI_API_KEY_EMBEDDING",
        "AZURE_OPENAI_ENDPOINT_EMBEDDING",
        "AZURE_OPENAI_API_VERSION",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_get_llm_azure_requires_key_endpoint_then_deployment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        get_llm()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    _clear_model_env(monkeypatch)
    # No deployment configured → actionable error naming the env var to set.
    with pytest.raises(ValueError, match="deployment"):
        get_llm()


def test_get_llm_azure_builds_with_deployment_names(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AIDLC_MODEL_ID", "my-gpt-4o")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "my-gpt-4o-mini")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    from langchain_openai import AzureChatOpenAI

    primary = get_llm()
    fast = get_sonnet_llm()
    assert isinstance(primary, AzureChatOpenAI)
    assert isinstance(fast, AzureChatOpenAI)
    assert primary.deployment_name == "my-gpt-4o"
    assert fast.deployment_name == "my-gpt-4o-mini"


def test_get_llm_google_requires_key(monkeypatch):
    # Key check fires before the package import, so this test needs no
    # langchain-google-genai install.
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_llm()


def test_get_llm_google_builds_with_default_models(monkeypatch):
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    _clear_model_env(monkeypatch)

    primary = get_llm()
    fast = get_sonnet_llm()
    # ChatGoogleGenerativeAI may normalize ids to "models/gemini-..." — match loosely.
    assert "gemini-2.5-pro" in str(primary.model)
    assert "gemini-2.5-flash" in str(fast.model)


def test_anthropic_fast_tier_error_names_fast_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "us.anthropic.claude-haiku-4-5-v1:0")
    # Bedrock-style id on the anthropic provider → rejected, naming the fast env var.
    with pytest.raises(ValueError, match="AIDLC_SONNET_MODEL_ID"):
        get_sonnet_llm()


def test_cost_tracker_tracks_tokens_without_per_model_pricing(monkeypatch):
    # The tracker prices the *slot*, not the model: an unrecognized model id is
    # tracked by tokens and costs nothing until its slot has a configured price
    # (no more silently mispricing unknown models as Sonnet).
    from agent.cost_tracker import (
        accumulate_cost,
        get_session_stats,
        reset_session_cost,
    )
    for k in ("AIDLC_PRICE_PRIMARY", "AIDLC_PRICE_FAST", "AIDLC_PRICE_EMBED", "AIDLC_PRICE_OTHER"):
        monkeypatch.delenv(k, raising=False)
    reset_session_cost()
    cost = accumulate_cost(1000, 500, model_id="models/gemini-2.5-flash")
    stats = get_session_stats()
    reset_session_cost()
    assert cost == 0.0
    assert stats["input_tokens"] == 1000
    assert stats["output_tokens"] == 500
    assert stats["any_unpriced"] is True


def test_azure_v1_endpoint_routes_to_openai_compatible_client(monkeypatch):
    # Azure AI Foundry hands out ".../openai/v1" endpoints (serving Foundry models
    # like DeepSeek). AzureChatOpenAI would append its legacy deployments path on
    # top of that and 404 — so this shape must route to the plain OpenAI client.
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1")
    monkeypatch.setenv("AIDLC_MODEL_ID", "DeepSeek-V4-Flash")

    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    llm = get_llm()
    assert isinstance(llm, ChatOpenAI) and not isinstance(llm, AzureChatOpenAI)
    assert llm.model_name == "DeepSeek-V4-Flash"
    assert str(llm.openai_api_base).rstrip("/") == "https://example.openai.azure.com/openai/v1"


def test_azure_v1_endpoint_detected_with_trailing_slash(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AIDLC_MODEL_ID", "my-deployment")

    from langchain_openai import AzureChatOpenAI

    assert not isinstance(get_llm(), AzureChatOpenAI)


def test_azure_anthropic_endpoint_routes_to_chat_anthropic(monkeypatch):
    # Azure AI Foundry serves Claude (format="Anthropic") deployments on a
    # ".../anthropic" endpoint via the Anthropic Messages API. AzureChatOpenAI
    # would append its OpenAI /openai/deployments/.../chat/completions path and
    # 404 "Resource not found" — so this shape must route to ChatAnthropic with
    # the endpoint as base_url (the anthropic SDK appends /v1/messages).
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.services.ai.azure.com/anthropic")
    monkeypatch.setenv("AIDLC_MODEL_ID", "claude-opus-4-8")

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import AzureChatOpenAI

    llm = get_llm()
    assert isinstance(llm, ChatAnthropic) and not isinstance(llm, AzureChatOpenAI)
    assert llm.model == "claude-opus-4-8"
    assert str(llm.anthropic_api_url).rstrip("/") == "https://example.services.ai.azure.com/anthropic"


def test_azure_anthropic_reasoning_uses_thinking_not_reasoning_effort(monkeypatch):
    # The primary "max think" knob must map to Claude adaptive thinking on this
    # surface — NOT the openai/azure reasoning_effort param (which Claude rejects)
    # — and must omit explicit temperature (Claude 4.7+ rejects it).
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.services.ai.azure.com/anthropic")
    monkeypatch.setenv("AIDLC_MODEL_ID", "claude-opus-4-8")
    monkeypatch.setenv("AIDLC_REASONING_EFFORT_PRIMARY", "high")

    llm = get_llm()
    assert llm.thinking == {"type": "adaptive"}
    assert llm.temperature is None


def test_azure_empty_api_version_env_falls_back_to_default(monkeypatch):
    # AZURE_OPENAI_API_VERSION='' in .env must not produce an empty api-version.
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "")
    monkeypatch.setenv("AIDLC_MODEL_ID", "my-gpt-4o")

    llm = get_llm()
    assert llm.openai_api_version == "2024-12-01-preview"


def test_azure_fast_tier_endpoint_and_key_overrides(monkeypatch):
    """Deployments in DIFFERENT Azure resources: the fast tier may point at its own
    endpoint/key via *_FAST vars; the primary tier must ignore those overrides."""
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "primary-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://primary.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY_FAST", "fast-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT_FAST", "https://fast.openai.azure.com")
    monkeypatch.setenv("AIDLC_MODEL_ID", "primary-deploy")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "fast-deploy")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    primary = get_llm()
    fast = get_sonnet_llm()
    assert str(primary.azure_endpoint).rstrip("/") == "https://primary.openai.azure.com"
    assert str(fast.azure_endpoint).rstrip("/") == "https://fast.openai.azure.com"
    assert fast.deployment_name == "fast-deploy"


def test_azure_fast_tier_falls_back_to_primary_credentials(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "primary-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://primary.openai.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY_FAST", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT_FAST", raising=False)
    monkeypatch.setenv("AIDLC_MODEL_ID", "primary-deploy")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "fast-deploy")

    fast = get_sonnet_llm()
    assert str(fast.azure_endpoint).rstrip("/") == "https://primary.openai.azure.com"


def test_azure_fast_tier_v1_override_routes_to_openai_client(monkeypatch):
    # Mixed setup: primary on the legacy route, fast on a Foundry /openai/v1 resource.
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "primary-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://primary.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY_FAST", "fast-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT_FAST", "https://fast.openai.azure.com/openai/v1")
    monkeypatch.setenv("AIDLC_MODEL_ID", "primary-deploy")
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "DeepSeek-V4-Flash")

    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    primary = get_llm()
    primary = getattr(primary, "runnable", primary)
    assert isinstance(primary, AzureChatOpenAI)
    fast = get_sonnet_llm()
    fast = getattr(fast, "runnable", fast)
    assert isinstance(fast, ChatOpenAI) and not isinstance(fast, AzureChatOpenAI)
    assert fast.model_name == "DeepSeek-V4-Flash"


def test_resolve_model_id_prefers_provider_scoped_var(monkeypatch):
    # Provider-scoped vars let each provider remember its own model; switching
    # LLM_PROVIDER must pick up that provider's scoped value, not the generic one.
    from agent.utils import _resolve_model_id

    monkeypatch.setenv("AIDLC_MODEL_ID", "generic-active-model")
    monkeypatch.setenv("AIDLC_MODEL_ID_AZURE", "azure-deployment")
    monkeypatch.setenv("AIDLC_MODEL_ID_GOOGLE", "gemini-2.5-pro")

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    assert _resolve_model_id("primary") == "azure-deployment"

    monkeypatch.setenv("LLM_PROVIDER", "google")
    assert _resolve_model_id("primary") == "gemini-2.5-pro"


def test_resolve_model_id_falls_back_to_generic_when_no_scoped(monkeypatch):
    # Migration: existing users have only the generic var until their first save.
    from agent.utils import _resolve_model_id

    monkeypatch.delenv("AIDLC_MODEL_ID_BEDROCK", raising=False)
    monkeypatch.setenv("AIDLC_MODEL_ID", "us.anthropic.claude-opus")
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    assert _resolve_model_id("primary") == "us.anthropic.claude-opus"


def test_resolve_model_id_fast_tier_scoped_then_generic(monkeypatch):
    from agent.utils import _resolve_model_id

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("AIDLC_SONNET_MODEL_ID_OPENAI", raising=False)
    monkeypatch.delenv("AIDLC_MODEL_ID_OPENAI", raising=False)
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID", "gpt-4o-mini")
    assert _resolve_model_id("fast") == "gpt-4o-mini"
    # Scoped fast var wins when present.
    monkeypatch.setenv("AIDLC_SONNET_MODEL_ID_OPENAI", "gpt-4o-mini-scoped")
    assert _resolve_model_id("fast") == "gpt-4o-mini-scoped"


# ---------------------------------------------------------------------------
# get_embeddings / _resolve_embed_model_id
# ---------------------------------------------------------------------------

def test_resolve_embed_model_id_defaults_per_provider(monkeypatch):
    from agent.utils import _resolve_embed_model_id

    for provider, expected in [
        ("bedrock", "amazon.titan-embed-text-v2:0"),
        ("openai",  "text-embedding-3-small"),
        # Azure embedding "model id" is a deployment name the user must create —
        # no safe default, so unset → None (graceful BM25 fallback, no 404).
        ("azure",   None),
        ("google",  "models/text-embedding-004"),
        ("anthropic", None),
    ]:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        monkeypatch.delenv("AIDLC_EMBED_MODEL_ID", raising=False)
        monkeypatch.delenv(f"AIDLC_EMBED_MODEL_ID_{provider.upper()}", raising=False)
        assert _resolve_embed_model_id() == expected


def test_resolve_embed_model_id_scoped_var_wins(monkeypatch):
    from agent.utils import _resolve_embed_model_id

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("AIDLC_EMBED_MODEL_ID", "text-embedding-3-large")
    monkeypatch.setenv("AIDLC_EMBED_MODEL_ID_OPENAI", "text-embedding-ada-002")
    assert _resolve_embed_model_id() == "text-embedding-ada-002"


def test_resolve_embed_model_id_generic_fallback(monkeypatch):
    from agent.utils import _resolve_embed_model_id

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("AIDLC_EMBED_MODEL_ID_OPENAI", raising=False)
    monkeypatch.setenv("AIDLC_EMBED_MODEL_ID", "text-embedding-3-large")
    assert _resolve_embed_model_id() == "text-embedding-3-large"


def test_get_embeddings_returns_none_for_anthropic(monkeypatch):
    from agent.utils import get_embeddings

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert get_embeddings() is None


def test_get_embeddings_openai_requires_api_key(monkeypatch):
    from agent.utils import get_embeddings

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embeddings()


def test_get_embeddings_openai_builds_with_key(monkeypatch):
    from langchain_openai import OpenAIEmbeddings

    from agent.utils import get_embeddings

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("AIDLC_EMBED_MODEL_ID", raising=False)
    monkeypatch.delenv("AIDLC_EMBED_MODEL_ID_OPENAI", raising=False)

    emb = get_embeddings()
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.model == "text-embedding-3-small"


def test_get_embeddings_azure_returns_none_when_no_deployment(monkeypatch):
    # No embedding deployment configured → graceful BM25 fallback (None), never a
    # guessed deployment name that would 404 against Azure.
    from agent.utils import get_embeddings

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    assert get_embeddings() is None


def test_get_embeddings_azure_requires_key_and_endpoint(monkeypatch):
    from agent.utils import get_embeddings

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AIDLC_EMBED_MODEL_ID", "my-embed-deployment")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI"):
        get_embeddings()


def test_get_embeddings_azure_v1_endpoint_uses_openai_client(monkeypatch):
    # Foundry "/openai/v1" surface is OpenAI-compatible — must use the plain
    # OpenAI embeddings client, not AzureOpenAIEmbeddings (which would 404).
    from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

    from agent.utils import get_embeddings

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1")
    monkeypatch.setenv("AIDLC_EMBED_MODEL_ID", "my-embed-deployment")

    emb = get_embeddings()
    assert isinstance(emb, OpenAIEmbeddings) and not isinstance(emb, AzureOpenAIEmbeddings)
    # Must send raw strings, not locally-tokenized integer arrays — the Azure
    # embeddings surface rejects token-ID arrays with HTTP 422.
    assert emb.check_embedding_ctx_length is False


def test_get_embeddings_google_requires_api_key(monkeypatch):
    pytest.importorskip("langchain_google_genai")
    from agent.utils import get_embeddings

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_embeddings()


# ---------------------------------------------------------------------------
# DSPy provider-aware LM construction (_build_litellm_lm)
# ---------------------------------------------------------------------------

def test_dspy_lm_bedrock_prefixes_model(monkeypatch):
    from agent.dspy_setup import _build_litellm_lm

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("AIDLC_MODEL_ID", "us.anthropic.claude-opus")
    lm = _build_litellm_lm("us.anthropic.claude-opus", "us-east-1")
    assert lm.model == "bedrock/us.anthropic.claude-opus"


def test_dspy_lm_openai(monkeypatch):
    from agent.dspy_setup import _build_litellm_lm

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert _build_litellm_lm("gpt-4o", "us-east-1").model == "openai/gpt-4o"


def test_dspy_lm_anthropic(monkeypatch):
    from agent.dspy_setup import _build_litellm_lm

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert _build_litellm_lm("claude-sonnet-4-6", "us-east-1").model == "anthropic/claude-sonnet-4-6"


def test_dspy_lm_google(monkeypatch):
    from agent.dspy_setup import _build_litellm_lm

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert _build_litellm_lm("gemini-2.5-pro", "us-east-1").model == "gemini/gemini-2.5-pro"


def test_dspy_lm_azure_v1_routes_through_openai_provider(monkeypatch):
    # Foundry "/openai/v1" surface is OpenAI-compatible — LiteLLM hits it via the
    # openai/ provider with a custom base, mirroring the chat path.
    from agent.dspy_setup import _build_litellm_lm

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/openai/v1")
    lm = _build_litellm_lm("Kimi-K2.6", "us-east-1")
    assert lm.model == "openai/Kimi-K2.6"


def test_dspy_lm_azure_legacy_uses_azure_provider(monkeypatch):
    from agent.dspy_setup import _build_litellm_lm

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    lm = _build_litellm_lm("my-gpt-4o", "us-east-1")
    assert lm.model == "azure/my-gpt-4o"


def test_dspy_lm_azure_requires_deployment(monkeypatch):
    from agent.dspy_setup import _build_litellm_lm

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    with pytest.raises(ValueError, match="deployment"):
        _build_litellm_lm(None, "us-east-1")
