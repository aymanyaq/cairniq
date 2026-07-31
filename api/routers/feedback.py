"""Feedback endpoints — the rating half of roadmap 1.5.

The capture half lives at the finalize point in ``api/routers/chat.py``: every
non-ghost chat turn is written to the per-profile feedback store as an UNRATED
interaction. This router is what turns one of those into a rated one when the
user clicks a thumb, and what draws a corrective lesson out of a complaint.

Nothing here invents a rating target: a thumb that cannot be matched to a stored
interaction is a 404, not a rating attached to whatever happened to be last.
"""
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.logger import log_to_component
from tools.feedback import (
    LOW_RATING_MAX,
    get_high_quality_interactions,
    get_statistics,
    rate_interaction,
)
from tools.pending_lessons import add_pending_lesson

router = APIRouter()


class FeedbackRequest(BaseModel):
    rating: int
    thread_id: str | None = None
    interaction_id: str | None = None
    tags: list[str] | None = None
    comment: str | None = None


@router.post("/api/feedback")
async def api_submit_feedback(req: FeedbackRequest):
    """Attach a rating (and optionally a written correction) to one chat turn.

    A low rating WITH a comment drafts a corrective lesson for confirmation —
    it is never written into ``lessons_learned`` here. A low rating WITHOUT a
    comment records the rating and drafts nothing: "that was bad" carries no
    statement of what the rule should be, and manufacturing one from silence is
    how a fabricated lesson gets into every future prompt.
    """
    if req.rating not in (1, 2, 3, 4, 5):
        return JSONResponse({"error": "rating must be an integer from 1 to 5"}, status_code=400)

    interaction = rate_interaction(
        rating=req.rating,
        interaction_id=req.interaction_id,
        thread_id=req.thread_id,
        tags=req.tags,
        comment=req.comment,
    )

    if interaction is None:
        return JSONResponse(
            {"error": "No stored interaction matches this feedback."},
            status_code=404,
        )

    drafted = None
    if req.rating <= LOW_RATING_MAX and (req.comment or "").strip():
        drafted = add_pending_lesson(
            text=req.comment,
            source="feedback",
            evidence={
                "interaction_id": interaction.get("id"),
                "rating": req.rating,
                "query": (interaction.get("user_query") or "")[:160],
            },
        )

    log_to_component("server", "Feedback", "Recorded chat feedback", {
        "interaction_id": interaction.get("id"),
        "rating": req.rating,
        "has_comment": bool((req.comment or "").strip()),
        "drafted_lesson": bool(drafted),
    }, level=logging.INFO)

    return {
        "status": "success",
        "interaction_id": interaction.get("id"),
        "rating": req.rating,
        "drafted_lesson": drafted,
    }


@router.get("/api/feedback/stats")
async def api_feedback_stats():
    """Rating counts plus the size of the high-quality pool.

    ``high_quality`` is the number of interactions a few-shot selector would
    actually have to draw on — the number that was silently 0 for the whole time
    this store was dark, while ``total`` looked healthy.
    """
    stats = get_statistics()
    stats["high_quality"] = len(get_high_quality_interactions())
    return stats
