"""
Chat History Storage
Saves and loads conversation sessions for later retrieval.
"""
import json
import os
from datetime import datetime
from typing import Any

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.json_store import write_json_atomic
from tools.user_profile import get_data_path


@log_exceptions()
def load_all_chats() -> dict[str, Any]:
    """Load all chat sessions from disk."""
    try:
        history_file = get_data_path("chat_history.json")
        if os.path.exists(history_file):
            with open(history_file) as f:
                return json.load(f)
    except Exception as e:
        safe_print(f"⚠️ Error loading chat history: {e}")

    return {"sessions": []}


@log_exceptions()
def save_all_chats(data: dict[str, Any]) -> bool:
    """Save all chat sessions to disk."""
    try:
        history_file = get_data_path("chat_history.json")
        write_json_atomic(history_file, data)
        return True
    except Exception as e:
        safe_print(f"⚠️ Error saving chat history: {e}")
        return False


@log_exceptions()
def save_current_session(session_id: str, messages: list[dict[str, str]], title: str | None = None, session_cost_cad: float = 0.0, session_tokens: int = 0) -> bool:
    """
    Save the current chat session.

    Args:
        session_id: Unique session identifier
        messages: List of message dicts with 'role' and 'content'
        title: Optional custom title (defaults to first user message)
        session_cost_cad: Running session cost in CAD
        session_tokens: Running session token total (input+output)
    """
    if not messages:
        return False

    data = load_all_chats()

    # Generate title from first user message if not provided
    if not title:
        first_user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), "Untitled Chat")
        title = first_user_msg[:50] + ("..." if len(first_user_msg) > 50 else "")

    # Check if session already exists
    existing_idx = next((i for i, s in enumerate(data["sessions"]) if s["session_id"] == session_id), None)

    session = {
        "session_id": session_id,
        "title": title,
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages),
        "session_cost_cad": session_cost_cad,
        "session_tokens": session_tokens,
        "messages": messages
    }

    if existing_idx is not None:
        # Update existing session
        data["sessions"][existing_idx] = session
    else:
        # Add new session
        data["sessions"].append(session)

    # Keep only last 7 sessions to prevent file bloat
    data["sessions"] = data["sessions"][-7:]

    return save_all_chats(data)


@log_exceptions()
def load_session(session_id: str) -> dict[str, Any] | None:
    """Load a specific chat session by ID. Returns dict with 'messages' and 'session_cost_cad'."""
    data = load_all_chats()

    session = next((s for s in data["sessions"] if s["session_id"] == session_id), None)

    if session:
        return {
            "messages": session["messages"],
            "session_cost_cad": session.get("session_cost_cad", 0.0),
            "session_tokens": session.get("session_tokens", 0)
        }
    return None


@log_exceptions()
def get_session_list() -> list[dict[str, Any]]:
    """Get list of all saved sessions (metadata only, no messages)."""
    data = load_all_chats()

    # Return sessions sorted by timestamp (newest first)
    sessions = sorted(data["sessions"], key=lambda s: s["timestamp"], reverse=True)

    return [
        {
            "session_id": s["session_id"],
            "title": s["title"],
            "timestamp": s["timestamp"],
            "message_count": s["message_count"],
            "session_cost_cad": s.get("session_cost_cad", 0.0),
            "session_tokens": s.get("session_tokens", 0)
        }
        for s in sessions
    ]


@log_exceptions()
def delete_session(session_id: str) -> bool:
    """Delete a specific chat session."""
    data = load_all_chats()

    original_count = len(data["sessions"])
    data["sessions"] = [s for s in data["sessions"] if s["session_id"] != session_id]

    if len(data["sessions"]) < original_count:
        return save_all_chats(data)

    return False


@log_exceptions()
def delete_all_sessions() -> bool:
    """Delete all saved chat sessions."""
    return save_all_chats({"sessions": []})


if __name__ == "__main__":
    # Test
    print("=== Chat History System Test ===")
    test_messages = [
        {"role": "user", "content": "What about NVDA?"},
        {"role": "assistant", "content": "NVDA is trading at..."}
    ]
    save_current_session("test_123", test_messages)
    print(f"Sessions: {get_session_list()}")
    loaded = load_session("test_123")
    print(f"Loaded: {loaded}")
