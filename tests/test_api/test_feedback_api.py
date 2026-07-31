"""Feedback wiring, end to end (Advisor Roadmap 1.5).

tools/feedback.py has been fully built and completely unused: the live profile
held 100 captured turns, none rated, the newest dated 2026-04-01. Two seams had
to exist for the store to be real — a capture at the chat finalize point, and an
endpoint that attaches a rating to a specific captured turn. Both are exercised
here through the REAL production path: a POST to /api/chat, streamed to
completion, then a POST to /api/feedback carrying only what the browser has (the
thread id from the response header).

That end-to-end shape is deliberate. This codebase has repeatedly shipped
features whose tests fed hand-built input straight into the component under test
and stayed green while the feature was dead in production (3.3's 56 tests through
a full trading day of outage). The capture site sits AFTER three sanitizers, so a
test that calls add_interaction directly would prove nothing about what actually
gets stored.

Every store is monkeypatched onto tmp_path — no test writes into a live profile.
"""
import pytest
from fastapi.testclient import TestClient

import tools.feedback as fb
import tools.memory as mem
import tools.pending_lessons as pl
from server import app

# A realistic final answer: prose the user reads, an internal reasoning block, a
# leaked prompt-scaffold tag, and the 3.3 watch side-channel. Everything but the
# prose is removed by the chain that runs before the capture point.
ANSWER = (
    "### PRIORITY\nHold. Cash is 12% of the book.\n"
    "<thinking>INTERNAL SCRATCH: user seems anxious</thinking>\n"
    '<output_format strict="true">\n'
    '<watch>{"conditions": [{"symbol": "NVDA", "metric": "price", "operator": "<=", '
    '"threshold": 165.0, "label": "NVDA reaches the entry zone", '
    '"action": "Execute the half-position entry", "direction": "entry", "expires_in_days": 30}]}</watch>'
)


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """Redirect the feedback / drafted-lesson / memory stores into tmp_path.

    Patched at module scope rather than via the profile, because
    profile_middleware re-resolves the profile on every request and would
    otherwise put these writes in a real profile directory.
    """
    monkeypatch.setattr(fb, "get_data_path", lambda filename: str(tmp_path / filename))
    monkeypatch.setattr(pl, "get_data_path", lambda filename: str(tmp_path / filename))
    monkeypatch.setattr(mem, "get_data_path", lambda filename: str(tmp_path / filename))
    return tmp_path


@pytest.fixture
def client(stores):
    from tools.user_profile import get_active_profile

    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


@pytest.fixture
def fake_agent(monkeypatch):
    """Stub only the model. Everything downstream of it is the real code path."""
    from langchain_core.messages import AIMessage, HumanMessage

    class _FakeAgent:
        def invoke(self, inputs, config=None):
            return {"messages": [HumanMessage(content="what should I do?"), AIMessage(content=ANSWER)]}

    monkeypatch.setattr("api.routers.chat.get_agent", lambda: _FakeAgent())

    # The advice-ledger / summary task is a separate concern and needs an LLM.
    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr("api.routers.chat._run_chat_post_processing", _noop)
    return _FakeAgent


def _run_turn(client, message="what should I do?", ghost=False):
    """Drive one real chat turn and return its thread id."""
    response = client.post("/api/chat", json={"message": message, "ghost": ghost})
    assert response.status_code == 200
    response.read()  # consume the stream so the finalize block runs
    return response.headers["X-Thread-ID"]


# ---------------------------------------------------------------------------
# Capture — the write that never happened
# ---------------------------------------------------------------------------

def test_a_real_chat_turn_is_captured_for_rating(client, fake_agent):
    thread_id = _run_turn(client)

    stored = fb.load_feedback()["interactions"]
    assert len(stored) == 1, "the finalize point did not write to the feedback store"
    turn = stored[0]
    assert turn["thread_id"] == thread_id
    assert turn["rating"] is None
    assert turn["user_query"] == "what should I do?"


def test_what_is_stored_is_what_the_user_SAW(client, fake_agent):
    """Capture sits downstream of the sanitizer chain on purpose: a rated example
    must be an example of the shipped answer, not of the pre-strip draft. This is
    the assertion a test that called add_interaction directly could not make."""
    _run_turn(client)

    answer = fb.load_feedback()["interactions"][0]["agent_response"]

    assert "Cash is 12% of the book." in answer
    assert "INTERNAL SCRATCH" not in answer      # <thinking> removed
    assert "<output_format" not in answer        # leaked scaffold tag removed
    assert "<watch>" not in answer               # 3.3 side-channel removed
    assert "expires_in_days" not in answer


def test_a_ghost_turn_is_never_captured(client, fake_agent):
    """Ghost mode exists to keep a turn out of persistent memory. The store keeps
    the query and answer verbatim, so capture must honour the same switch the
    supervisor's memory capture does."""
    _run_turn(client, ghost=True)

    assert fb.load_feedback()["interactions"] == []


def test_an_in_message_privacy_tag_is_honoured_too(client, fake_agent):
    """@Private is the typed equivalent of the Ghost toggle (agent.utils.PRIVACY_TAGS)."""
    _run_turn(client, message="@Private what should I do?")

    assert fb.load_feedback()["interactions"] == []


# ---------------------------------------------------------------------------
# Rating — the endpoint the thumbs call
# ---------------------------------------------------------------------------

def test_a_thumbs_up_makes_the_turn_a_few_shot_example(client, fake_agent):
    """The whole point of 1.5: capture alone feeds nothing, a rating does."""
    thread_id = _run_turn(client)
    assert fb.get_high_quality_interactions() == []

    response = client.post("/api/feedback", json={"rating": 5, "thread_id": thread_id})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["drafted_lesson"] is None
    pool = fb.get_high_quality_interactions()
    assert len(pool) == 1
    assert "Cash is 12% of the book." in pool[0]["agent_response"]


def test_feedback_with_no_captured_turn_is_a_404_not_a_guess(client, stores):
    response = client.post("/api/feedback", json={"rating": 5, "thread_id": "never-existed"})

    assert response.status_code == 404
    assert "No stored interaction" in response.json()["error"]


def test_an_out_of_range_rating_is_rejected(client, fake_agent):
    thread_id = _run_turn(client)

    response = client.post("/api/feedback", json={"rating": 11, "thread_id": thread_id})

    assert response.status_code == 400
    assert fb.load_feedback()["interactions"][0]["rating"] is None


def test_stats_report_the_pool_size_not_just_the_total(client, fake_agent):
    thread_id = _run_turn(client)
    client.post("/api/feedback", json={"rating": 5, "thread_id": thread_id})

    stats = client.get("/api/feedback/stats").json()

    assert stats["total"] == 1
    assert stats["rated"] == 1
    assert stats["high_quality"] == 1


# ---------------------------------------------------------------------------
# Low ratings draft corrective lessons — under 1.4's confirmation guard
# ---------------------------------------------------------------------------

def test_a_complaint_with_words_drafts_a_lesson_but_does_not_apply_it(client, fake_agent):
    thread_id = _run_turn(client)

    response = client.post("/api/feedback", json={
        "rating": 1,
        "thread_id": thread_id,
        "comment": "State the as-of time on every quote",
    })

    assert response.status_code == 200
    drafted = response.json()["drafted_lesson"]
    assert drafted["text"] == "State the as-of time on every quote"
    assert drafted["evidence"]["rating"] == 1
    # Pending, and NOT in the store that is injected into every prompt.
    assert len(pl.list_pending_lessons()) == 1
    assert mem.load_memory().get("lessons_learned", []) == []


def test_a_complaint_without_words_drafts_nothing(client, fake_agent):
    """A thumbs-down alone says the answer was bad, not what the rule should be.
    Manufacturing the 'why' from silence is the empty-block fabrication class
    that has already put invented history in front of this user."""
    thread_id = _run_turn(client)

    response = client.post("/api/feedback", json={"rating": 1, "thread_id": thread_id})

    assert response.status_code == 200
    assert response.json()["drafted_lesson"] is None
    assert pl.list_pending_lessons() == []
    # The rating itself is still recorded — the complaint is not lost.
    assert fb.load_feedback()["interactions"][0]["rating"] == 1


def test_the_confirm_endpoint_is_what_applies_a_draft(client, fake_agent):
    thread_id = _run_turn(client)
    client.post("/api/feedback", json={
        "rating": 1, "thread_id": thread_id, "comment": "State the as-of time on every quote",
    })
    listed = client.get("/api/memory/lessons/pending").json()["pending"]
    assert len(listed) == 1

    response = client.post(f"/api/memory/lessons/pending/{listed[0]['id']}/confirm")

    assert response.status_code == 200
    assert "State the as-of time on every quote" in mem.load_memory()["lessons_learned"]
    assert client.get("/api/memory/lessons/pending").json()["pending"] == []


def test_a_draft_can_be_reworded_at_the_gate(client, fake_agent):
    thread_id = _run_turn(client)
    client.post("/api/feedback", json={"rating": 2, "thread_id": thread_id, "comment": "this was wrong"})
    draft_id = client.get("/api/memory/lessons/pending").json()["pending"][0]["id"]

    client.post(f"/api/memory/lessons/pending/{draft_id}/confirm",
                json={"text": "Never present a cached quote as live"})

    assert mem.load_memory()["lessons_learned"] == ["Never present a cached quote as live"]


def test_a_draft_can_be_discarded_without_learning(client, fake_agent):
    thread_id = _run_turn(client)
    client.post("/api/feedback", json={"rating": 1, "thread_id": thread_id, "comment": "unhelpful"})
    draft_id = client.get("/api/memory/lessons/pending").json()["pending"][0]["id"]

    response = client.delete(f"/api/memory/lessons/pending/{draft_id}")

    assert response.status_code == 200
    assert mem.load_memory().get("lessons_learned", []) == []
    assert client.get("/api/memory/lessons/pending").json()["pending"] == []


def test_unknown_draft_ids_404(client, stores):
    assert client.post("/api/memory/lessons/pending/nope/confirm").status_code == 404
    assert client.delete("/api/memory/lessons/pending/nope").status_code == 404


# ---------------------------------------------------------------------------
# The UI can actually reach it
# ---------------------------------------------------------------------------

def test_the_chat_footer_carries_the_rating_control():
    """The endpoint is only useful if something calls it. An inline script is
    all-or-nothing (see tests/test_template_scripts.py), so the ids and the fetch
    are pinned here as well as parsed there."""
    from pathlib import Path

    html = (Path(__file__).resolve().parents[2] / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'id="btn-feedback-up"' in html
    assert 'id="btn-feedback-down"' in html
    assert "'/api/feedback'" in html
    # chat.js announces turn completion; the footer arms itself on that event.
    assert "cairniq:turn-complete" in html
    chat_js = (Path(__file__).resolve().parents[2] / "static" / "js" / "chat.js").read_text(encoding="utf-8")
    assert "cairniq:turn-complete" in chat_js


def test_the_context_page_exposes_drafts_for_confirmation():
    """A draft nobody can see is another dark store."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    html = (root / "templates" / "context_and_graph.html").read_text(encoding="utf-8")
    pages = (root / "api" / "routers" / "pages.py").read_text(encoding="utf-8")

    assert "pending_lessons" in html
    assert "confirmDraftedLesson" in html and "discardDraftedLesson" in html
    assert "list_pending_lessons" in pages
