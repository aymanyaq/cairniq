import inspect
import os
import sqlite3

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint

import tools.user_profile as up
from agent.checkpointer import CHECKPOINT_FILENAME, ProfileRoutingSaver


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point profile data at a temp dir and start from a clean guard state."""
    monkeypatch.setattr(up, "_user_data_root", lambda: str(tmp_path))
    monkeypatch.setattr(up, "_multiuser_guard", False)
    return tmp_path


def _store_path(root, profile: str) -> str:
    return os.path.join(root, "profiles", profile, CHECKPOINT_FILENAME)


def _write_checkpoint(saver: ProfileRoutingSaver, thread_id: str) -> None:
    saver.put(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        empty_checkpoint(),
        {"source": "input", "step": -1, "parents": {}},
        {},
    )


def _threads(path: str) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    finally:
        conn.close()


def test_each_profile_writes_to_its_own_store(data_root):
    """The bug this module exists to fix: one store captured every profile."""
    saver = ProfileRoutingSaver()

    with up.profile_context("alice"):
        _write_checkpoint(saver, "alice-thread")
    with up.profile_context("bob"):
        _write_checkpoint(saver, "bob-thread")

    assert _threads(_store_path(data_root, "alice")) == {"alice-thread"}
    assert _threads(_store_path(data_root, "bob")) == {"bob-thread"}
    # And nothing pooled into the sentinel.
    assert not os.path.exists(_store_path(data_root, up.UNBOUND_PROFILE))


def test_reads_are_routed_too_so_a_profile_cannot_see_another(data_root):
    saver = ProfileRoutingSaver()
    with up.profile_context("alice"):
        _write_checkpoint(saver, "shared-id")

    with up.profile_context("bob"):
        found = saver.get_tuple(
            {"configurable": {"thread_id": "shared-id", "checkpoint_ns": ""}}
        )

    # Same thread id, different profile — bob must not read alice's checkpoint.
    assert found is None


def test_one_connection_is_reused_per_profile(data_root):
    saver = ProfileRoutingSaver()
    with up.profile_context("alice"):
        first = saver.active_saver()
        second = saver.active_saver()
    with up.profile_context("bob"):
        other = saver.active_saver()

    assert first is second
    assert other is not first
    assert saver.open_profiles() == ["alice", "bob"]


def test_a_lost_contextvar_isolates_to_unbound_without_pooling(data_root, monkeypatch):
    """The fail-safe still works — it just no longer collects everyone."""
    monkeypatch.setattr(up, "_multiuser_guard", True)
    saver = ProfileRoutingSaver()

    with up.profile_context("alice"):
        _write_checkpoint(saver, "alice-thread")

    # Clear the ContextVar outright — the conftest binds a pytest_* profile for
    # every test, so simply leaving profile_context is not the detached-worker
    # state we need to exercise.
    token = up._profile_ctx.set("")
    try:
        _write_checkpoint(saver, "orphan-thread")
    finally:
        up._profile_ctx.reset(token)

    assert _threads(_store_path(data_root, "alice")) == {"alice-thread"}
    assert _threads(_store_path(data_root, up.UNBOUND_PROFILE)) == {"orphan-thread"}


def test_no_saver_method_falls_through_to_the_base_class(data_root):
    """A method left inherited would write to whatever store it resolved —
    the silent-wrong-profile failure this module was written to end."""
    inherited = [
        name
        for name in dir(BaseCheckpointSaver)
        if not name.startswith("_")
        and callable(getattr(BaseCheckpointSaver, name, None))
        and getattr(ProfileRoutingSaver, name) is getattr(BaseCheckpointSaver, name)
    ]
    assert inherited == []


def test_forwarders_keep_the_call_shape_of_the_interface(data_root):
    """alist is an async GENERATOR; awaiting it would break streaming."""
    for name in dir(BaseCheckpointSaver):
        if name.startswith("_"):
            continue
        base = getattr(BaseCheckpointSaver, name, None)
        if not callable(base):
            continue
        routed = getattr(ProfileRoutingSaver, name)
        assert inspect.isasyncgenfunction(routed) == inspect.isasyncgenfunction(base), name
        assert inspect.iscoroutinefunction(routed) == inspect.iscoroutinefunction(base), name


def test_close_releases_every_open_store(data_root):
    saver = ProfileRoutingSaver()
    with up.profile_context("alice"):
        _write_checkpoint(saver, "alice-thread")
    with up.profile_context("bob"):
        _write_checkpoint(saver, "bob-thread")

    saver.close()

    assert saver.open_profiles() == []
