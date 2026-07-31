"""The feedback store and the drafted-lesson queue (Advisor Roadmap 1.5 / 1.4).

tools/feedback.py was fully built and completely dark: the live profile held 100
captured interactions, the newest dated 2026-04-01, and **not one of them was
rated**. get_high_quality_interactions is the intended few-shot pool, so the pool
was not "small", it was EMPTY — and get_statistics reported total=100, which
reads healthy. These tests pin the two properties that were missing: a rating can
be attached to a specific turn, and a rated example survives.

Everything here is monkeypatched onto tmp_path. Nothing may touch a live profile.
"""
import ast
from pathlib import Path

import pytest

import tools.feedback as fb
import tools.memory as mem
import tools.pending_lessons as pl


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Point every per-profile store at tmp_path, including memory's."""
    monkeypatch.setattr(fb, "get_data_path", lambda filename: str(tmp_path / filename))
    monkeypatch.setattr(pl, "get_data_path", lambda filename: str(tmp_path / filename))
    monkeypatch.setattr(mem, "get_data_path", lambda filename: str(tmp_path / filename))
    return tmp_path


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def test_capture_returns_an_id_and_stores_the_turn(store):
    interaction_id = fb.add_interaction("what is my cash?", "Cash is 12%.", thread_id="t-1")

    assert interaction_id
    stored = fb.load_feedback()["interactions"]
    assert len(stored) == 1
    assert stored[0]["id"] == interaction_id
    assert stored[0]["thread_id"] == "t-1"
    assert stored[0]["source"] == "chat"
    assert stored[0]["rating"] is None  # unrated until a human says otherwise


def test_an_empty_answer_is_not_an_interaction(store):
    assert fb.add_interaction("hello?", "   ") is None
    assert fb.load_feedback()["interactions"] == []


# ---------------------------------------------------------------------------
# Rating — the half that never existed
# ---------------------------------------------------------------------------

def test_the_high_quality_pool_is_empty_until_something_is_rated(store):
    """The dark-store failure in one assertion: capture alone yields no pool."""
    for i in range(5):
        fb.add_interaction(f"q{i}", f"a{i}", thread_id="t-1")

    assert fb.get_statistics()["total"] == 5      # looks healthy...
    assert fb.get_high_quality_interactions() == []  # ...and feeds nothing

    fb.rate_interaction(fb.THUMBS_UP_RATING, thread_id="t-1")

    assert len(fb.get_high_quality_interactions()) == 1


def test_rating_by_id_targets_that_exact_turn(store):
    first = fb.add_interaction("q1", "a1", thread_id="t-1")
    fb.add_interaction("q2", "a2", thread_id="t-1")

    rated = fb.rate_interaction(4, interaction_id=first)

    assert rated["id"] == first
    assert rated["rated_at"]
    stored = fb.load_feedback()["interactions"]
    assert stored[0]["rating"] == 4 and stored[1]["rating"] is None


def test_rating_by_thread_targets_the_latest_turn_of_that_thread(store):
    fb.add_interaction("q1", "a1", thread_id="t-1")
    other = fb.add_interaction("q2", "a2", thread_id="t-2")
    latest = fb.add_interaction("q3", "a3", thread_id="t-1")

    rated = fb.rate_interaction(fb.THUMBS_DOWN_RATING, thread_id="t-1")

    assert rated["id"] == latest
    by_id = {i["id"]: i for i in fb.load_feedback()["interactions"]}
    assert by_id[other]["rating"] is None


def test_an_unmatched_thread_rates_nothing_rather_than_guessing(store):
    """Falling through to "most recent" would attach a verdict the user never
    gave about that turn — straight into the few-shot pool."""
    fb.add_interaction("q1", "a1", thread_id="t-1")

    assert fb.rate_interaction(1, thread_id="does-not-exist") is None
    assert fb.load_feedback()["interactions"][0]["rating"] is None


def test_rating_an_empty_store_returns_none(store):
    assert fb.rate_interaction(5, thread_id="t-1") is None


@pytest.mark.parametrize("bad", [0, 6, -1, 99])
def test_out_of_range_ratings_are_rejected(store, bad):
    fb.add_interaction("q", "a", thread_id="t-1")
    with pytest.raises(ValueError):
        fb.rate_interaction(bad, thread_id="t-1")


def test_a_comment_rides_along_with_the_rating(store):
    fb.add_interaction("q", "a", thread_id="t-1")

    rated = fb.rate_interaction(1, thread_id="t-1", comment="Quoted a stale price as live")

    assert rated["comment"] == "Quoted a stale price as live"


def test_thumb_ratings_line_up_with_the_pool_and_lesson_bars(store):
    """The two thumb values are only useful if they clear the bars that consume
    them: up must qualify for the few-shot pool, down must qualify as a complaint."""
    assert fb.THUMBS_UP_RATING >= 4          # get_high_quality_interactions default
    assert fb.THUMBS_DOWN_RATING <= fb.LOW_RATING_MAX


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_unrated_churn_never_evicts_a_rated_example(store):
    """Auto-capture makes this load-bearing. Under the previous flat 100-item
    FIFO trim, turning capture on would have deleted every rated example within
    ~100 turns of ordinary chat — quietly emptying the pool it exists to fill."""
    keeper = fb.add_interaction("good question", "excellent answer", thread_id="t-0")
    fb.rate_interaction(5, interaction_id=keeper)

    for i in range(fb.MAX_UNRATED + 40):
        fb.add_interaction(f"q{i}", f"a{i}", thread_id=f"t-{i}")

    ids = {i["id"] for i in fb.load_feedback()["interactions"]}
    assert keeper in ids
    assert len(fb.get_high_quality_interactions()) == 1
    # ...and the unrated stream is still capped.
    unrated = [i for i in fb.load_feedback()["interactions"] if not i.get("rating")]
    assert len(unrated) == fb.MAX_UNRATED


# ---------------------------------------------------------------------------
# Drafted lessons — roadmap 1.4's confirmation guard
# ---------------------------------------------------------------------------

def test_a_draft_does_not_become_a_lesson(store):
    draft = pl.add_pending_lesson("Never quote a cached price as live", source="feedback")

    assert draft["id"]
    assert [p["text"] for p in pl.list_pending_lessons()] == ["Never quote a cached price as live"]
    # The thing that actually reaches every prompt is untouched.
    assert mem.load_memory().get("lessons_learned", []) == []


def test_nothing_is_drafted_from_silence(store):
    assert pl.add_pending_lesson("", source="feedback") is None
    assert pl.add_pending_lesson("   ", source="feedback") is None
    assert pl.list_pending_lessons() == []


def test_drafts_dedupe_against_pending_and_confirmed(store):
    pl.add_pending_lesson("Always verify the quote timestamp", source="feedback")
    assert pl.add_pending_lesson("always   VERIFY the Quote Timestamp", source="feedback") is None

    mem.add_lesson("Never catch a falling knife")
    assert pl.add_pending_lesson("never catch a falling knife", source="feedback") is None
    assert len(pl.list_pending_lessons()) == 1


def test_confirming_promotes_the_draft_and_clears_it(store):
    draft = pl.add_pending_lesson("Never quote a cached price as live", source="feedback")

    promoted = pl.confirm_pending_lesson(draft["id"])

    assert promoted["text"] == "Never quote a cached price as live"
    assert "Never quote a cached price as live" in mem.load_memory()["lessons_learned"]
    assert pl.list_pending_lessons() == []


def test_the_confirming_human_can_rewrite_the_wording(store):
    draft = pl.add_pending_lesson("this was wrong lol", source="feedback")

    pl.confirm_pending_lesson(draft["id"], text="State the as-of time on every quote")

    lessons = mem.load_memory()["lessons_learned"]
    assert lessons == ["State the as-of time on every quote"]


def test_discarding_learns_nothing(store):
    draft = pl.add_pending_lesson("ignore me", source="feedback")

    assert pl.discard_pending_lesson(draft["id"]) is True
    assert pl.list_pending_lessons() == []
    assert mem.load_memory().get("lessons_learned", []) == []
    assert pl.discard_pending_lesson(draft["id"]) is False


def test_confirming_an_unknown_draft_is_a_miss_not_a_write(store):
    assert pl.confirm_pending_lesson("nope") is None
    assert mem.load_memory().get("lessons_learned", []) == []


def test_only_the_confirm_gate_may_call_add_lesson():
    """Structural guard for roadmap 1.4. The value of a confirmation gate is that
    it cannot be bypassed, so this asserts on the source: add_lesson is reachable
    from exactly one function in this module. A future 'auto-apply obvious ones'
    shortcut fails here rather than in production, where the symptom is a
    fabricated rule silently injected into every prompt.
    """
    source = Path(pl.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    callers = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "add_lesson":
                callers.add(node.name)

    assert callers == {"confirm_pending_lesson"}, (
        f"add_lesson must only be reachable from the confirmation gate, found: {sorted(callers)}"
    )
