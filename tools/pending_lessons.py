"""Drafted-but-unconfirmed corrective lessons.

Roadmap 1.4's guard, made structural: *every auto-drafted lesson requires human
confirmation, and nothing is drafted below the evidence bar.* This module is the
holding area that makes that enforceable — a draft lands here, never in
``lessons_learned``. The ONLY call into ``tools.memory.add_lesson`` from this
file is inside :func:`confirm_pending_lesson`, which exists to be reached by an
explicit human action (the confirm button on /context). Nothing else in the
codebase may promote a draft.

Why the guard matters: lessons are injected into every prompt, and the store is
capped (``tools.memory.LESSON_CAP``) and truncates from the front — so at cap an
unconfirmed draft does not merely add noise, confirming it RETIRES the oldest
rule the user wrote. Hence :func:`confirm_pending_lesson` reports what the
promotion cost, and the caller has to render it.

Stored per-profile as ``pending_lessons.json`` via get_data_path().
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

# Drafts are a queue for a human to read, not a log. Beyond this, the oldest
# untouched drafts fall off rather than growing an unreadable backlog.
MAX_PENDING = 25


@log_exceptions()
def _pending_file() -> str:
    """Return the profile-specific pending-lessons file path."""
    return get_data_path("pending_lessons.json")


@log_exceptions()
def load_pending() -> dict[str, Any]:
    """Load drafted lessons from disk (profile-scoped)."""
    try:
        path = _pending_file()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("pending"), list):
                    return data
    except Exception as e:
        safe_print(f"⚠️ Error loading pending lessons: {e}")

    return {"pending": []}


@log_exceptions()
def save_pending(data: dict[str, Any]) -> bool:
    """Persist drafted lessons to disk (profile-scoped)."""
    try:
        write_json_atomic(_pending_file(), data)
        return True
    except Exception as e:
        safe_print(f"⚠️ Error saving pending lessons: {e}")
        return False


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


@log_exceptions()
def add_pending_lesson(
    text: str,
    source: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Queue a drafted lesson for human confirmation.

    Returns the draft, or None when there is nothing to draft (empty text) or
    the same rule is already pending / already confirmed. Never writes to
    ``lessons_learned``.
    """
    body = str(text or "").strip()
    if not body:
        # Nothing to learn from. A thumbs-down with no words attached says the
        # answer was bad, not WHY — inventing the "why" here is exactly the
        # empty-block fabrication this codebase has been burned by.
        return None
    body = body[:400]

    data = load_pending()
    existing = data["pending"]

    key = _normalize(body)
    if any(_normalize(p.get("text", "")) == key for p in existing):
        return None

    try:
        from tools.memory import load_memory
        confirmed = load_memory().get("lessons_learned", []) or []
    except Exception:
        confirmed = []
    if any(_normalize(lesson) == key for lesson in confirmed):
        return None

    draft = {
        "id": uuid.uuid4().hex[:12],
        "text": body,
        "source": source,
        "drafted_at": datetime.now().isoformat(),
        "evidence": evidence or {},
    }
    existing.append(draft)
    data["pending"] = existing[-MAX_PENDING:]
    save_pending(data)
    safe_print(f"📝 Drafted lesson pending confirmation: {body[:80]}")
    return draft


@log_exceptions()
def list_pending_lessons() -> list[dict[str, Any]]:
    """Every drafted lesson awaiting a human decision, oldest first."""
    return list(load_pending()["pending"])


@log_exceptions()
def discard_pending_lesson(lesson_id: str) -> bool:
    """Drop a draft without learning from it."""
    data = load_pending()
    remaining = [p for p in data["pending"] if p.get("id") != lesson_id]
    if len(remaining) == len(data["pending"]):
        return False
    data["pending"] = remaining
    return save_pending(data)


@log_exceptions()
def confirm_pending_lesson(lesson_id: str, text: str | None = None) -> dict[str, Any] | None:
    """Promote one draft into the real lesson store. THE human-confirmation gate.

    ``text`` lets the confirming human edit the wording first — the draft is a
    proposal, not a verdict. Returns None if the id is unknown, otherwise a dict
    carrying ``status: confirmed`` and ``retired``: the user-written rules this
    promotion pushed out of the capped store, captured before the write because
    afterwards they are unrecoverable. ``retired`` is empty on the normal path.
    """
    from tools.memory import LESSON_EVICTED, add_lesson, lessons_pending_eviction

    data = load_pending()
    draft = next((p for p in data["pending"] if p.get("id") == lesson_id), None)
    if draft is None:
        return None

    final_text = str(text or draft.get("text") or "").strip()[:400]
    if not final_text:
        return None

    at_risk = lessons_pending_eviction()
    outcome = add_lesson(final_text)

    data["pending"] = [p for p in data["pending"] if p.get("id") != lesson_id]
    save_pending(data)
    return {
        **draft,
        "text": final_text,
        "status": "confirmed",
        "confirmed_at": datetime.now().isoformat(),
        # A duplicate is a no-op write: it retires nothing.
        "retired": at_risk if outcome == LESSON_EVICTED else [],
    }
