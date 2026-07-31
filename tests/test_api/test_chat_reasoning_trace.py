"""The reasoning trace must reach the UI without polluting the answer.

Regression cover for edaa8c1, which filtered thought/reasoning content blocks
out of the visible stream to stop them spilling into the chat — and, having no
other channel, dropped them entirely. The trace panel exists to receive them.
"""

import json

import pytest
from fastapi.testclient import TestClient

from agent.utils import extract_reasoning_text, extract_stream_text
from server import app


@pytest.fixture()
def client():
    from tools.user_profile import get_active_profile

    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


# The real shape langchain_google_genai emits for a Gemini thought part:
# {"type": "thinking", "thinking": part.text} — the text is under "thinking",
# NOT "text" (see _parse_response_candidate in its chat_models.py).
CHUNK = [
    {"type": "thinking", "thinking": "The portfolio is 34% semis. Checking correlation."},
    {"type": "text", "text": "Concentration risk is elevated."},
]


def test_gemini_reasoning_is_requested_from_the_api(monkeypatch):
    """thinking_budget alone makes the model think privately — without
    include_thoughts no thought parts come back and the trace is always empty."""
    from agent.utils import _reasoning_kwargs

    # Pin the effort rather than inheriting whatever this machine configures,
    # so the assertion means the same thing everywhere.
    monkeypatch.setenv("AIDLC_REASONING_EFFORT_PRIMARY", "max")

    for provider in ("google", "vertexai"):
        kwargs = _reasoning_kwargs(provider, "primary", 16384)
        assert kwargs.get("include_thoughts") is True, provider
        assert "thinking_budget" in kwargs, provider


def test_reasoning_off_stays_off(monkeypatch):
    """Effort 'off' must not smuggle include_thoughts in — no reasoning asked
    for, nothing billed for, no trace."""
    from agent.utils import _reasoning_kwargs

    monkeypatch.setenv("AIDLC_REASONING_EFFORT_PRIMARY", "off")
    assert _reasoning_kwargs("vertexai", "primary", 16384) == {}


def test_reasoning_and_visible_text_go_to_separate_channels():
    assert extract_stream_text(CHUNK) == "Concentration risk is elevated."
    assert extract_reasoning_text(CHUNK) == "The portfolio is 34% semis. Checking correlation."


@pytest.mark.parametrize("block_type", ["thinking", "thought", "reasoning"])
def test_every_reasoning_block_type_is_recovered(block_type):
    assert extract_reasoning_text([{"type": block_type, "text": "why"}]) == "why"
    # ...and none of them leak into the answer.
    assert extract_stream_text([{"type": block_type, "text": "why"}]) == ""


def test_plain_string_content_is_never_treated_as_reasoning():
    # A bare string chunk carries no block type — it is the answer, not a trace.
    assert extract_reasoning_text("Concentration risk is elevated.") == ""


def test_reasoning_reaches_the_stream_without_entering_the_answer(monkeypatch, client):
    """End-to-end: a node that streams a reasoning block must produce a
    {"thinking": ...} payload, and that text must never appear in {"text": ...}."""
    from agent.utils import send_stream, send_thinking

    class FakeAgent:
        def invoke(self, inputs, config=None):
            from langchain_core.messages import AIMessage

            # Mimic a node streaming one Gemini-shaped chunk.
            send_thinking(extract_reasoning_text(CHUNK))
            send_stream(extract_stream_text(CHUNK))
            return {"messages": [AIMessage(content="Concentration risk is elevated.")]}

    monkeypatch.setattr("api.routers.chat.get_agent", lambda: FakeAgent())

    with client.stream("POST", "/api/chat", json={"message": "check my risk"}) as resp:
        assert resp.status_code == 200
        payloads = [json.loads(line) for line in resp.iter_lines() if line.strip()]

    thinking = "".join(p["thinking"] for p in payloads if "thinking" in p)
    answers = [p["text"] for p in payloads if "text" in p]

    assert "34% semis" in thinking, f"reasoning never reached the trace: {payloads}"
    assert answers, "the answer never streamed"
    assert all("34% semis" not in a for a in answers), "reasoning leaked into the answer"
