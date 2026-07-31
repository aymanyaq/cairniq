"""
Feedback System
Collects user ratings on agent responses for continuous improvement.
Feedback is stored per-profile via get_data_path() so different users
don't share ratings.

Writers: ``api/routers/chat.py`` captures every non-ghost chat turn at finalize;
``POST /api/feedback`` (api/routers/feedback.py) attaches the rating a thumb
click carries. Readers: ``get_high_quality_interactions`` is the intended
few-shot pool, so an unrated store is an EMPTY pool, not a small one.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Any

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic
from tools.user_profile import get_data_path

# A thumb is a two-value control mapped onto the 1-5 scale this store has always
# used. Up lands at 5 so it clears get_high_quality_interactions' default bar of
# 4; down lands at 1 so it clears the LOW_RATING_MAX bar for drafting a lesson.
THUMBS_UP_RATING = 5
THUMBS_DOWN_RATING = 1

# At or below this, a rating counts as a complaint worth learning from.
LOW_RATING_MAX = 2

# Retention. Rated interactions are the product — they are the few-shot pool —
# so they get their own budget rather than sharing one FIFO queue with the
# unrated capture stream. Under the old flat 100-item trim, auto-capture would
# have evicted every rated example within ~100 turns of ordinary chat.
MAX_UNRATED = 100
MAX_RATED = 200


@log_exceptions()
def _feedback_file() -> str:
    """Return the profile-specific feedback file path."""
    return get_data_path("feedback.json")


@log_exceptions()
def load_feedback() -> dict[str, Any]:
    """Load feedback from disk (profile-scoped)."""
    try:
        path = _feedback_file()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        safe_print(f"⚠️ Error loading feedback: {e}")

    return {"interactions": []}


@log_exceptions()
def save_feedback(feedback: dict[str, Any]) -> bool:
    """Save feedback to disk (profile-scoped)."""
    try:
        write_json_atomic(_feedback_file(), feedback)
        return True
    except Exception as e:
        safe_print(f"⚠️ Error saving feedback: {e}")
        return False


def _trim(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap the store without letting unrated churn evict rated examples.

    Rated and unrated are trimmed against separate budgets, then re-interleaved
    in stored order so the file still reads chronologically.
    """
    rated_keep = [i for i in interactions if i.get("rating")][-MAX_RATED:]
    unrated_keep = [i for i in interactions if not i.get("rating")][-MAX_UNRATED:]
    keep = {id(i) for i in rated_keep} | {id(i) for i in unrated_keep}
    return [i for i in interactions if id(i) in keep]


@log_exceptions()
def add_interaction(
    user_query: str,
    agent_response: str,
    rating: int | None = None,
    tags: list[str] | None = None,
    thread_id: str | None = None,
    source: str = "chat",
) -> str | None:
    """
    Add a new interaction to the feedback log.

    Args:
        user_query: The user's question/request
        agent_response: The agent's full response
        rating: 1-5 rating (optional, attached later by POST /api/feedback)
        tags: Optional tags like ['helpful', 'accurate', 'confusing']
        thread_id: Chat thread this turn belongs to — how a later thumb click
            finds the turn it is rating without depending on global recency.
        source: What produced the interaction (e.g. "chat").

    Returns the new interaction's id, or None if nothing was written.
    """
    if not str(agent_response or "").strip():
        # Nothing was said. An empty answer is not an interaction to learn from,
        # and storing it would dilute the pool with blanks.
        return None

    feedback = load_feedback()

    interaction_id = uuid.uuid4().hex[:12]
    interaction = {
        "id": interaction_id,
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread_id,
        "source": source,
        "user_query": (user_query or "")[:500],  # Truncate long queries
        "agent_response": agent_response[:2000],  # Truncate long responses
        "rating": rating,
        "tags": tags or []
    }

    feedback["interactions"].append(interaction)
    feedback["interactions"] = _trim(feedback["interactions"])

    save_feedback(feedback)
    return interaction_id


@log_exceptions()
def update_last_rating(rating: int, tags: list[str] | None = None) -> bool:
    """Update the rating for the most recent interaction."""
    feedback = load_feedback()

    if not feedback["interactions"]:
        return False

    feedback["interactions"][-1]["rating"] = rating
    if tags:
        feedback["interactions"][-1]["tags"] = tags

    return save_feedback(feedback)


@log_exceptions()
def rate_interaction(
    rating: int,
    interaction_id: str | None = None,
    thread_id: str | None = None,
    tags: list[str] | None = None,
    comment: str | None = None,
) -> dict[str, Any] | None:
    """Attach a rating to one stored interaction. Returns it, or None if not found.

    Resolution is explicit and never guesses: an ``interaction_id`` addresses one
    record; a ``thread_id`` addresses the most recent turn of that thread; with
    neither, the most recent interaction overall. A ``thread_id`` that matches
    nothing returns None rather than falling through to "latest" — silently
    rating an unrelated turn would poison the few-shot pool with a verdict the
    user never gave about it.
    """
    if rating not in (1, 2, 3, 4, 5):
        raise ValueError(f"rating must be 1-5, got {rating!r}")

    feedback = load_feedback()
    interactions = feedback.get("interactions") or []
    if not interactions:
        return None

    target = None
    if interaction_id:
        target = next((i for i in reversed(interactions) if i.get("id") == interaction_id), None)
    elif thread_id:
        target = next((i for i in reversed(interactions) if i.get("thread_id") == thread_id), None)
    else:
        target = interactions[-1]

    if target is None:
        return None

    target["rating"] = rating
    if tags:
        target["tags"] = tags
    if comment and str(comment).strip():
        target["comment"] = str(comment).strip()[:1000]
    target["rated_at"] = datetime.now().isoformat()

    if not save_feedback(feedback):
        return None
    return target


@log_exceptions()
def get_high_quality_interactions(min_rating: int = 4) -> list[dict[str, Any]]:
    """Get interactions with rating >= min_rating for training."""
    feedback = load_feedback()
    return [
        interaction for interaction in feedback["interactions"]
        if interaction.get("rating") and interaction["rating"] >= min_rating
    ]


@log_exceptions()
def get_statistics() -> dict[str, Any]:
    """Get feedback statistics."""
    feedback = load_feedback()
    interactions = feedback["interactions"]

    if not interactions:
        return {"total": 0, "rated": 0, "average_rating": None}

    rated = [i for i in interactions if i.get("rating")]

    stats = {
        "total": len(interactions),
        "rated": len(rated),
        "average_rating": sum(i["rating"] for i in rated) / len(rated) if rated else None,
        "rating_distribution": {
            1: sum(1 for i in rated if i["rating"] == 1),
            2: sum(1 for i in rated if i["rating"] == 2),
            3: sum(1 for i in rated if i["rating"] == 3),
            4: sum(1 for i in rated if i["rating"] == 4),
            5: sum(1 for i in rated if i["rating"] == 5)
        }
    }

    return stats


if __name__ == "__main__":
    # Test
    print("=== Feedback System Test ===")
    add_interaction("What should I buy?", "Consider diversifying with VTI and SCHD.", 5, ["helpful"])
    stats = get_statistics()
    print(f"Stats: {stats}")
    print(f"High quality examples: {len(get_high_quality_interactions())}")
