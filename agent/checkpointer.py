"""Per-profile conversation checkpointing.

``server.py`` builds the LangGraph agent once, inside ``lifespan``, before any
request has bound a profile. ``get_data_path()`` therefore resolved against the
multi-user guard's ``_unbound`` sentinel, and the single ``SqliteSaver``
connection baked into the compiled graph became the conversation store for
EVERY profile for the life of the process.

On the production host that ran from 2026-06-28 — commit 70d09aa, which
introduced ``enable_multiuser_guard`` — until this fix. The evidence was a
clean handoff in the UUID6 checkpoint timestamps: the owner's own store holds
12,778 checkpoints ending 2026-06-28, and ``_unbound`` holds 1,578 beginning
that same day. Threads are UUID-keyed so no user ever read another's
conversation, but per-profile isolation of conversation state was gone, and the
warning meant to make it diagnosable never reached the logs.

The fix keeps the single compiled graph and instead resolves the profile per
CALL rather than once at build time. Each profile gets its own SQLite file,
opened lazily and cached for the process lifetime, so a request's conversation
state lands in that request's profile directory.
"""
import inspect
import sqlite3
import threading

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.logger import log_to_component
from tools.user_profile import get_active_profile, get_data_path

CHECKPOINT_FILENAME = "checkpoints.sqlite"


class ProfileRoutingSaver(BaseCheckpointSaver):
    """A checkpoint saver that picks its database from the active profile.

    One instance is compiled into the graph; it holds one ``SqliteSaver`` per
    profile behind a lock and dispatches every call to whichever profile is
    bound on the current request. A profile whose ContextVar was lost still
    resolves to the isolated ``_unbound`` store — that fail-safe is unchanged,
    it just no longer captures everyone else's conversations too.
    """

    def __init__(self) -> None:
        super().__init__()
        self._savers: dict[str, SqliteSaver] = {}
        self._lock = threading.Lock()

    def active_saver(self) -> SqliteSaver:
        """The checkpoint store for the profile bound to this call."""
        profile = get_active_profile()
        saver = self._savers.get(profile)
        if saver is not None:
            return saver

        with self._lock:
            # Re-check: another thread may have opened it while we waited.
            saver = self._savers.get(profile)
            if saver is None:
                path = get_data_path(CHECKPOINT_FILENAME)
                conn = sqlite3.connect(path, check_same_thread=False)
                saver = SqliteSaver(conn)
                saver.setup()
                self._savers[profile] = saver
                log_to_component(
                    "agent",
                    "Checkpointer",
                    f"Opened conversation store for profile '{profile}'",
                    {"profile": profile, "path": path},
                )
            return saver

    def open_profiles(self) -> list[str]:
        """Profiles with an open store — the check that this routes at all."""
        with self._lock:
            return sorted(self._savers)

    def close(self) -> None:
        """Close every open store. Called on shutdown."""
        with self._lock:
            for saver in self._savers.values():
                try:
                    saver.conn.close()
                except Exception:
                    pass
            self._savers.clear()


def _forward(name: str, base_method):
    """Build a method that hands ``name`` to the active profile's store.

    The saver interface mixes three call shapes — plain sync methods, coroutines
    and one async generator (``alist``) — and each needs its own wrapper, since
    awaiting an async generator or returning a bare coroutine from a sync def
    would break the caller in ways no test of ``put`` would catch.
    """
    if inspect.isasyncgenfunction(base_method):
        async def method(self, *args, **kwargs):
            async for item in getattr(self.active_saver(), name)(*args, **kwargs):
                yield item
    elif inspect.iscoroutinefunction(base_method):
        async def method(self, *args, **kwargs):
            return await getattr(self.active_saver(), name)(*args, **kwargs)
    else:
        def method(self, *args, **kwargs):
            return getattr(self.active_saver(), name)(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"ProfileRoutingSaver.{name}"
    method.__doc__ = f"Route {name}() to the active profile's checkpoint store."
    return method


# Generated from the base class rather than hand-listed: a LangGraph upgrade
# that adds a saver method would otherwise inherit the base implementation
# silently and write to the wrong profile's database — the exact failure this
# module exists to fix.
for _name in dir(BaseCheckpointSaver):
    if _name.startswith("_"):
        continue
    _base = getattr(BaseCheckpointSaver, _name, None)
    if callable(_base):
        setattr(ProfileRoutingSaver, _name, _forward(_name, _base))
