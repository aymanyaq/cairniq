"""A dropped stream must not cost the user the turn.

Regression for the 2026-07-29 market-dip run: the client's stream was torn down
mid-synthesis, the graph ran on for another nine minutes, and every message it
produced afterwards — a compliance-retry revision, its verdict, and its warning
banner — was suppressed by the post-invoke fallback as "already covered by the
visible stream". Nothing had covered it; the dedup was comparing against the
worker's own production buffer, which fills whether or not a client is reading.
The turn was never persisted either, because the auto-save lived in the
generator's `else:` branch, which an abandoned generator never reaches.

The disconnect is staged by closing the StreamingResponse's body iterator
mid-flight. That is the same teardown a real dropped connection produces — the
generator is resumed with GeneratorExit at its yield — and unlike driving it
through a socket it is deterministic: TestClient drains the response instead of
abandoning it, httpx's ASGITransport buffers the whole body and deadlocks, and a
killed socket under uvicorn is only noticed on some later write, well after the
run has ended.
"""

import asyncio
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import api.routers.chat as chat_mod
from api.routers.chat import ChatRequest, chat_endpoint

DELIVERED = "First draft: accumulate ASML and AMZN on the dip. "
# The revision the user never saw. Deliberately shares vocabulary with the first
# draft — a *revision* always does, which is what made the >50% word-overlap
# check in the dedup fire so reliably on the real run.
REVISED = "[DeepReasoning]: Position change on the ASML and AMZN dip draft: shifting to WAIT, accumulate nothing."
BANNER = "[RiskManager]: Compliance Review Warning: flagged on both attempts."


@pytest.fixture()
def captured_saves(monkeypatch):
    """Intercept persistence so the test never writes a real session file."""
    saves: list[tuple[str, list]] = []

    async def _noop_coro(*args, **kwargs):
        return None

    def fake_save(thread_id, messages, **kwargs):
        saves.append((thread_id, list(messages)))

    monkeypatch.setattr("api.routers.chat.save_current_session", fake_save)
    monkeypatch.setattr("api.routers.chat.capture_watch_conditions", lambda *a, **k: None)
    monkeypatch.setattr("api.routers.chat._run_chat_post_processing", _noop_coro)
    return saves


def _drive_disconnect(monkeypatch, agent_factory, until) -> None:
    """Abandon the stream mid-run, then let the agent finish behind it.

    `until` is polled to decide when the post-stream work has landed. It has to
    be a poll rather than a fixed wait: the worker schedules its persistence back
    onto this loop, so the loop must stay alive until that lands, but how long it
    takes is machine-dependent. A constant sleep is either wasteful (the old
    unconditional 10s) or flaky on a loaded box.
    """
    release = threading.Event()
    finished = threading.Event()

    monkeypatch.setattr(chat_mod, "get_agent", lambda: agent_factory(release, finished))

    async def scenario():
        response = await chat_endpoint(ChatRequest(message="build me a dip deployment plan"))
        body = response.body_iterator

        # The thread_id preamble, then at least one chunk of real answer, so the
        # turn is genuinely mid-stream when the client goes away.
        await asyncio.wait_for(body.__anext__(), timeout=10)
        await asyncio.wait_for(body.__anext__(), timeout=10)

        # The client is gone.
        await body.aclose()

        # Only now does the run produce the rest of the turn — nobody listening.
        release.set()
        loop = asyncio.get_running_loop()
        assert await loop.run_in_executor(None, finished.wait, 20), "agent never completed"
        # The worker persists after invoke returns. Stay on the loop while it
        # lands: it schedules work back onto this loop, so closing it early would
        # break the very path under test.
        deadline = loop.time() + 20
        while not until():
            assert loop.time() < deadline, "post-stream work never landed"
            await asyncio.sleep(0.01)

    asyncio.run(scenario())


def _fake_agent(messages):
    def factory(release, finished):
        class FakeAgent:
            def invoke(self, inputs, config=None):
                from agent.utils import get_run_context, is_cancelled

                ctx = get_run_context()
                for token in DELIVERED.split(" "):
                    ctx.on_token(token + " ")
                release.wait(timeout=20)
                FakeAgent.saw_cancelled = is_cancelled()
                finished.set()
                return {"messages": messages}

        FakeAgent.saw_cancelled = None
        factory.cls = FakeAgent
        return FakeAgent()

    return factory


def test_turn_survives_a_client_disconnect(monkeypatch, captured_saves):
    factory = _fake_agent([
        HumanMessage(content="build me a dip deployment plan"),
        AIMessage(content="[MarketAnalyst]: " + DELIVERED, name="MarketAnalyst"),
        AIMessage(content=REVISED, name="DeepReasoning"),
        AIMessage(content=BANNER, name="RiskManager"),
    ])
    _drive_disconnect(monkeypatch, factory, until=lambda: bool(captured_saves))

    assert captured_saves, "a disconnected turn was persisted nowhere"
    assistant = [m for m in captured_saves[-1][1] if m.get("role") == "assistant"]
    assert assistant, "no assistant message saved"
    saved = assistant[-1]["content"]

    # The whole point: messages produced after the stream died belong to the
    # saved turn, not dropped as duplicates of text nobody ever received.
    assert "shifting to WAIT" in saved, f"revision was lost; saved={saved!r}"
    assert "Compliance Review Warning" in saved, f"banner was lost; saved={saved!r}"


def test_disconnect_reaches_the_running_agent(monkeypatch, captured_saves):
    """The run must learn the client left; its log must not claim a clean finish."""
    factory = _fake_agent([
        HumanMessage(content="q"),
        AIMessage(content="[X]: done", name="X"),
    ])
    _drive_disconnect(monkeypatch, factory, until=lambda: bool(captured_saves))
    assert factory.cls.saw_cancelled is True, "the run never learned the client had gone"
