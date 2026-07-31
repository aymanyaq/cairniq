import os
from datetime import datetime, timedelta

import pytest

from tools.chat_history import (
    delete_session,
    get_session_list,
    load_all_chats,
    load_session,
    save_current_session,
)
from tools.user_profile import get_data_path


@pytest.fixture
def clean_chat_history_file():
    history_path = get_data_path("chat_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    if os.path.exists(history_path):
        os.remove(history_path)

    yield history_path

    if os.path.exists(history_path):
        os.remove(history_path)


def test_save_and_load_chat_session_round_trip(clean_chat_history_file):
    messages = [
        {"role": "user", "content": "How exposed am I to semiconductors?"},
        {"role": "assistant", "content": "You have a concentrated chip exposure."},
    ]

    assert save_current_session("thread-1", messages, session_cost_cad=1.25) is True

    loaded = load_session("thread-1")
    sessions = get_session_list()

    assert loaded == {"messages": messages, "session_cost_cad": 1.25, "session_tokens": 0}
    assert sessions == [
        {
            "session_id": "thread-1",
            "title": "How exposed am I to semiconductors?",
            "timestamp": sessions[0]["timestamp"],
            "message_count": 2,
            "session_cost_cad": 1.25,
            "session_tokens": 0,
        }
    ]


def test_chat_history_retains_only_last_seven_sessions(clean_chat_history_file, monkeypatch):
    class IncrementingDatetime:
        counter = -1

        @classmethod
        def now(cls):
            cls.counter += 1
            return datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=cls.counter)

    monkeypatch.setattr("tools.chat_history.datetime", IncrementingDatetime)

    for index in range(9):
        save_current_session(
            f"thread-{index}",
            [{"role": "user", "content": f"Question {index}"}],
            session_cost_cad=float(index),
        )

    stored_ids = [session["session_id"] for session in load_all_chats()["sessions"]]
    listed_ids = [session["session_id"] for session in get_session_list()]

    assert stored_ids == [f"thread-{index}" for index in range(2, 9)]
    assert listed_ids[0] == "thread-8"
    assert set(listed_ids) == set(stored_ids)


def test_delete_session_removes_only_matching_thread(clean_chat_history_file):
    save_current_session("keep", [{"role": "user", "content": "Keep me"}])
    save_current_session("delete", [{"role": "user", "content": "Delete me"}])

    assert delete_session("delete") is True
    assert load_session("delete") is None
    assert load_session("keep") is not None
    assert delete_session("missing") is False
