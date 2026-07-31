"""The answer stream sends what is NEW, not the whole answer per token.

The stream used to re-send the entire accumulated answer on every single token:
append token -> sanitize the whole buffer -> json.dumps the complete answer. That
is O(n^2) on the wire — a 20,000-char answer cost ~50MB, ~550x the text itself —
and it got worse the longer the answer, over a VPN, every turn.

Deltas are OPT-IN (`stream_deltas`). The wire format is a published contract: the
iOS client documents `text` as "the FULL accumulated answer so far (replace,
don't append)", and it is sideloaded, so an older build can be talking to a newer
server. A client that does not ask keeps the old protocol byte-for-byte.

The property that makes deltas safe is that the server's sanitizers can REWRITE
history, not just extend it: a <thinking> or watch block that completes
retroactively removes text already sent. When that happens the server must fall
back to a full frame instead of emitting a delta against text that no longer
matches.
"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import api.routers.chat as chat_mod
from api.routers.chat import ChatRequest, chat_endpoint


@pytest.fixture(autouse=True)
def no_persistence(monkeypatch):
    async def _noop_coro(*args, **kwargs):
        return None

    monkeypatch.setattr("api.routers.chat.save_current_session", lambda *a, **k: None)
    monkeypatch.setattr("api.routers.chat.capture_watch_conditions", lambda *a, **k: None)
    monkeypatch.setattr("api.routers.chat._run_chat_post_processing", _noop_coro)


def _install_agent(monkeypatch, tokens, final_content):
    """An agent that emits `tokens` through the run context, then finishes."""
    class FakeAgent:
        def invoke(self, inputs, config=None):
            from agent.utils import get_run_context

            ctx = get_run_context()
            for tok in tokens:
                ctx.on_token(tok)
            return {"messages": [
                HumanMessage(content="q"),
                AIMessage(content=final_content, name="MarketAnalyst"),
            ]}

    monkeypatch.setattr(chat_mod, "get_agent", lambda: FakeAgent())


def _collect(monkeypatch, tokens, final_content, *, stream_deltas):
    """Drive one full turn and return (frames, wire_bytes)."""
    async def run():
        response = await chat_endpoint(ChatRequest(
            message="q", stream_deltas=stream_deltas,
        ))
        frames, wire = [], 0
        async for chunk in response.body_iterator:
            wire += len(chunk)
            for line in chunk.splitlines():
                if line.strip():
                    frames.append(json.loads(line))
        return frames, wire

    _install_agent(monkeypatch, tokens, final_content)
    return asyncio.run(run())


def _rebuild(frames):
    """Apply the client contract: append on delta, replace on text."""
    text = ""
    for f in frames:
        if "delta" in f:
            text += f["delta"]
        elif "text" in f:
            text = f["text"]
    return text


TOKENS = [f"word{i} " for i in range(300)]
FINAL = "".join(TOKENS)


def test_delta_stream_rebuilds_the_same_answer_as_the_legacy_stream(monkeypatch):
    legacy, _ = _collect(monkeypatch, TOKENS, FINAL, stream_deltas=False)
    delta, _ = _collect(monkeypatch, TOKENS, FINAL, stream_deltas=True)
    assert _rebuild(delta) == _rebuild(legacy)
    assert "word299" in _rebuild(delta)


def test_delta_stream_is_dramatically_smaller_on_the_wire(monkeypatch):
    _, legacy_bytes = _collect(monkeypatch, TOKENS, FINAL, stream_deltas=False)
    _, delta_bytes = _collect(monkeypatch, TOKENS, FINAL, stream_deltas=True)
    # Quadratic vs linear: the gap widens with answer length, so this bound is
    # deliberately loose — it is checking the shape of the growth, not a constant.
    assert delta_bytes * 5 < legacy_bytes, f"legacy={legacy_bytes} delta={delta_bytes}"


def test_legacy_clients_still_get_full_text_frames(monkeypatch):
    """A client that does not opt in must see the old protocol unchanged."""
    frames, _ = _collect(monkeypatch, TOKENS, FINAL, stream_deltas=False)
    assert not any("delta" in f for f in frames), "legacy stream leaked a delta frame"
    answer_frames = [f for f in frames if "text" in f]
    assert answer_frames, "no answer frames at all"
    # Every answer frame is the whole answer so far — monotonically growing.
    lengths = [len(f["text"]) for f in answer_frames]
    assert lengths == sorted(lengths)


def test_a_sanitizer_rewriting_history_forces_a_full_resync(monkeypatch):
    """<thinking> is stripped only once its closing tag arrives.

    Mid-block, the partial-tag regex hides the text; when the block closes the
    sanitized answer is REWRITTEN rather than extended. A delta measured against
    the old text would corrupt the answer, so the server must send a full frame.
    """
    tokens = ["Answer part one. ", "<thinking>", "hidden reasoning", "</thinking>", " Answer part two."]
    frames, _ = _collect(monkeypatch, tokens, "".join(tokens), stream_deltas=True)

    rebuilt = _rebuild(frames)
    assert "hidden reasoning" not in rebuilt
    assert "Answer part one." in rebuilt
    assert "Answer part two." in rebuilt


def test_delta_stream_ends_with_a_reconciling_full_frame(monkeypatch):
    frames, _ = _collect(monkeypatch, TOKENS, FINAL, stream_deltas=True)
    finals = [f for f in frames if f.get("final")]
    assert len(finals) == 1, "expected exactly one reconciling frame"
    # It must agree with what the deltas built, or the client would jump.
    assert finals[0]["text"] == _rebuild(frames)
