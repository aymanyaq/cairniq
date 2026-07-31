"""Disk housekeeping — rotate logs and prune aged conversation checkpoints.

Nothing in this tree rotated. On the production host the logs directory had
grown to ~72 MB of never-truncated JSONL plus an 8.7 MB stderr log carrying a
restart storm from months earlier, and the LangGraph checkpoint stores to
377 MB across two files. ``task_cache_cleanup`` only ever looked at
``user_data/daily_cache``, so none of that was any job's responsibility.

Two rules shape the implementation, and both are load-bearing:

  * Rotation is COPY-TRUNCATE, never rename. launchd holds the stdout/stderr
    files open for the life of the service, and ``agent.logger`` holds a
    ``FileHandler`` on every ``.jsonl``. Renaming the path leaves each writer
    bound to the old inode — the archive keeps growing and the fresh file
    stays empty forever. Truncating in place keeps the inode the writer
    already has, at the cost of losing the handful of bytes written between
    the copy and the truncate.

  * A checkpoint is pruned only when its age is KNOWN, and only a whole
    thread at a time. LangGraph keys checkpoints by a UUID6, whose 60-bit
    timestamp yields an exact write time without deserializing the msgpack
    blob — but an id that does not parse as a sane UUID6 date is KEPT, since
    deleting a row we could not date is deleting conversation history on a
    guess. Pruning by thread rather than by row keeps every surviving
    conversation's lineage intact: a thread is dropped entire, or not at all.
"""
import datetime
import glob
import gzip
import os
import shutil
import sqlite3
import uuid

from agent.logger import LOG_BASE_DIR, log_to_component
from tools.exception_logger import log_exceptions
from tools.user_profile import _user_data_root

# Rotate a log once it exceeds this, and keep rotations for this many days.
#
# This was 100 MB, justified by "the largest log on the production host was
# 33 MB after months". That basis went stale and the threshold was never
# revisited: measured 2026-07-31, agent.jsonl was 41 MB, chat_runtime.jsonl
# 34 MB, logs/ 103 MB in total — and ZERO .gz archives existed, meaning
# rotation had never once fired in the life of the job. prune_rotations() was
# therefore inert too: it only deletes archives, and there were none. The
# "backstop against unbounded growth" was a backstop that had never engaged.
#
# 16 MB makes rotation a routine event instead of a theoretical one. JSONL
# gzips roughly 10-20x, so an archive costs ~1 MB and the brief per-file lock
# is paid a few times a week rather than never.
LOG_MAX_BYTES = 16 * 1024 * 1024
LOG_KEEP_DAYS = 30

# Hard ceiling on the WHOLE logs tree, archives included. Per-file rotation
# plus a 30-day retention still has no aggregate bound — the only ceiling was
# LOG_MAX_BYTES x however many files happen to exist, which is a number nobody
# chose. When the tree exceeds this, the OLDEST ARCHIVES are deleted until it
# fits.
#
# Note this can delete archives younger than LOG_KEEP_DAYS, and that ordering is
# deliberate: a disk that fills takes the whole service down, while a lost
# 20-day-old compressed log costs a diagnostic. Live logs are never touched
# here — only .gz archives — so nothing a writer holds open is disturbed.
LOG_TOTAL_MAX_BYTES = 256 * 1024 * 1024

# A conversation untouched for this long stops being worth its disk.
CHECKPOINT_RETENTION_DAYS = 30

# A VACUUM holds an exclusive lock for its whole duration, so running one on a
# store a conversation is actively checkpointing into can fail that turn. Skip
# the reclaim on any store touched this recently and let a quieter pass take it:
# the row deletions still land (those are brief and serialized), and SQLite
# reuses the freed pages meanwhile, so the file stops growing either way.
VACUUM_IDLE_SECONDS = 300

# Only these get rotated. The glob must never match its own .gz output, or a
# rotation would compress its own archives on every pass.
_LOG_PATTERNS = ("*.log", "*/*.jsonl")

# UUID6 counts 100-nanosecond intervals from the Gregorian epoch.
_GREGORIAN_EPOCH = datetime.datetime(1582, 10, 15)

# A parsed checkpoint date outside this window means the id was not really a
# UUID6 timestamp — treat it as undatable and keep the row.
_SANE_EPOCH = datetime.datetime(2020, 1, 1)


def _uuid6_written_at(checkpoint_id: str) -> datetime.datetime | None:
    """Exact write time encoded in a LangGraph UUID6 checkpoint id.

    Returns None for anything that is not a UUID6 resolving to a plausible
    date. Callers must treat None as "do not touch this row".
    """
    try:
        parsed = uuid.UUID(str(checkpoint_id))
    except (ValueError, AttributeError, TypeError):
        return None
    if parsed.version != 6:
        return None
    digits = parsed.hex
    # time_high (8) + time_mid (4) + time_low (3, after the version nibble).
    ticks = int(digits[0:12] + digits[13:16], 16)
    try:
        written = _GREGORIAN_EPOCH + datetime.timedelta(microseconds=ticks / 10)
    except (OverflowError, OSError, ValueError):
        return None
    if not (_SANE_EPOCH <= written <= datetime.datetime.now() + datetime.timedelta(days=1)):
        return None
    return written


def _rotation_name(path: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{path}.{stamp}.gz"


@log_exceptions()
def rotate_log_file(path: str, max_bytes: int = LOG_MAX_BYTES, dry_run: bool = False) -> int:
    """Compress ``path`` aside and truncate it in place. Returns bytes reclaimed.

    A file under ``max_bytes`` is left alone and returns 0. The original is
    truncated rather than moved so the process writing it keeps its inode —
    see the module docstring.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0
    if size < max_bytes:
        return 0
    if dry_run:
        return size

    target = _rotation_name(path)
    try:
        with open(path, "rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    except OSError:
        # A half-written archive is worse than no archive: drop it and leave
        # the live log untouched so the next pass can retry cleanly.
        if os.path.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass
        raise

    os.truncate(path, 0)
    return size


@log_exceptions()
def prune_rotations(keep_days: int = LOG_KEEP_DAYS, dry_run: bool = False) -> int:
    """Delete rotated ``.gz`` archives older than ``keep_days``. Returns bytes freed."""
    cutoff = datetime.datetime.now().timestamp() - keep_days * 86400
    freed = 0
    for pattern in _LOG_PATTERNS:
        for archive in glob.glob(os.path.join(LOG_BASE_DIR, pattern + ".*.gz")):
            try:
                if os.path.getmtime(archive) >= cutoff:
                    continue
                freed += os.path.getsize(archive)
                if not dry_run:
                    os.remove(archive)
            except OSError:
                continue
    return freed


def _log_tree_bytes() -> int:
    """Total size of everything under LOG_BASE_DIR, archives included."""
    total = 0
    for root, _dirs, files in os.walk(LOG_BASE_DIR):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


@log_exceptions()
def enforce_total_log_budget(
    max_total_bytes: int = LOG_TOTAL_MAX_BYTES,
    dry_run: bool = False,
) -> int:
    """Delete oldest ARCHIVES until the logs tree fits its budget. Returns bytes freed.

    The aggregate ceiling per-file rotation does not give you. Only ``.gz``
    archives are eligible — a live log is held open by launchd or a FileHandler,
    and deleting it would leave the writer bound to an unlinked inode, silently
    writing to nowhere.

    If the live logs ALONE exceed the budget there is nothing further this can
    do; it deletes every archive and stops rather than looping. That state means
    LOG_MAX_BYTES is too high for the number of live logs, and the returned
    figure plus the caller's log line is how it becomes visible.
    """
    total = _log_tree_bytes()
    if total <= max_total_bytes:
        return 0

    archives = []
    for pattern in _LOG_PATTERNS:
        for archive in glob.glob(os.path.join(LOG_BASE_DIR, pattern + ".*.gz")):
            try:
                archives.append((os.path.getmtime(archive), os.path.getsize(archive), archive))
            except OSError:
                continue
    archives.sort()  # oldest first

    freed = 0
    for _mtime, size, path in archives:
        if total - freed <= max_total_bytes:
            break
        if not dry_run:
            try:
                os.remove(path)
            except OSError:
                continue
        freed += size
    return freed


@log_exceptions()
def rotate_logs(
    max_bytes: int = LOG_MAX_BYTES,
    keep_days: int = LOG_KEEP_DAYS,
    dry_run: bool = False,
    max_total_bytes: int = LOG_TOTAL_MAX_BYTES,
) -> dict:
    """Rotate every oversized log and expire old archives.

    ``scanned`` is the count that proves the sweep ran — it is non-zero on a
    healthy host even when nothing needed rotating, which ``reclaimed_bytes``
    is not.
    """
    scanned = 0
    rotated: list[str] = []
    reclaimed = 0

    for pattern in _LOG_PATTERNS:
        for path in sorted(glob.glob(os.path.join(LOG_BASE_DIR, pattern))):
            if path.endswith(".gz"):
                continue
            scanned += 1
            freed = rotate_log_file(path, max_bytes=max_bytes, dry_run=dry_run)
            if freed:
                rotated.append(os.path.relpath(path, LOG_BASE_DIR))
                reclaimed += freed

    reclaimed += prune_rotations(keep_days=keep_days, dry_run=dry_run)

    # Age-based pruning alone leaves the tree unbounded; this is the ceiling.
    over_budget = enforce_total_log_budget(max_total_bytes=max_total_bytes, dry_run=dry_run)
    reclaimed += over_budget

    return {
        "scanned": scanned,
        "rotated": rotated,
        "reclaimed_bytes": reclaimed,
        "budget_evicted_bytes": over_budget,
        "tree_bytes": _log_tree_bytes(),
    }


def _seconds_since_write(db_path: str) -> float:
    """How long ago anything last wrote this store.

    The WAL and shared-memory files carry the freshest mtime — a conversation
    checkpointing right now touches those, not the main database. Must be read
    BEFORE we open or modify the store, since our own deletions would otherwise
    make every store look busy and permanently suppress the reclaim.
    """
    newest = 0.0
    for suffix in ("", "-wal", "-shm"):
        try:
            newest = max(newest, os.path.getmtime(db_path + suffix))
        except OSError:
            continue
    if not newest:
        return float("inf")
    return max(0.0, datetime.datetime.now().timestamp() - newest)


def _aged_thread_ids(conn: sqlite3.Connection, cutoff: datetime.datetime) -> list[str]:
    """Threads whose NEWEST checkpoint predates ``cutoff``.

    A thread with even one undatable checkpoint is held back entirely: without
    a reliable newest-write time we cannot say the conversation is cold.
    """
    newest: dict[str, datetime.datetime] = {}
    undatable: set[str] = set()

    for thread_id, checkpoint_id in conn.execute(
        "SELECT thread_id, checkpoint_id FROM checkpoints"
    ):
        written = _uuid6_written_at(checkpoint_id)
        if written is None:
            undatable.add(thread_id)
            continue
        if thread_id not in newest or written > newest[thread_id]:
            newest[thread_id] = written

    return [t for t, when in newest.items() if when < cutoff and t not in undatable]


@log_exceptions()
def prune_checkpoints(
    retention_days: int = CHECKPOINT_RETENTION_DAYS,
    dry_run: bool = False,
) -> dict:
    """Drop conversations untouched for ``retention_days``, then reclaim the pages.

    Runs against every ``checkpoints.sqlite`` under ``user_data`` — including
    the one the live server holds open, which is safe: the cutoff is far
    outside any in-flight conversation, and SQLite serializes the writes. A
    VACUUM that cannot get its lock is reported, not raised.
    """
    cutoff = datetime.datetime.now() - datetime.timedelta(days=retention_days)
    stores: list[dict] = []
    scanned = 0

    pattern = os.path.join(_user_data_root(), "**", "checkpoints.sqlite")
    for db_path in sorted(glob.glob(pattern, recursive=True)):
        scanned += 1
        entry = {
            "store": os.path.relpath(db_path, _user_data_root()),
            "threads_pruned": 0,
            "rows_deleted": 0,
            "reclaimed_bytes": 0,
        }
        try:
            before = os.path.getsize(db_path)
            # Read liveness before connecting: opening the store, and certainly
            # deleting from it, refreshes the very mtimes this measures.
            idle_for = _seconds_since_write(db_path)
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA busy_timeout = 30000")
            try:
                aged = _aged_thread_ids(conn, cutoff)
                entry["threads_pruned"] = len(aged)
                if aged and not dry_run:
                    deleted = 0
                    for start in range(0, len(aged), 500):
                        batch = aged[start:start + 500]
                        # `marks` is a run of `?` placeholders sized to the batch —
                        # sqlite3 has no way to bind a variable-length IN list, so the
                        # COUNT is interpolated and every VALUE is still bound. No
                        # caller data reaches the statement text.
                        marks = ",".join("?" * len(batch))
                        cur = conn.execute(
                            f"DELETE FROM checkpoints WHERE thread_id IN ({marks})",  # nosec B608
                            batch,
                        )
                        deleted += cur.rowcount or 0
                        conn.execute(
                            f"DELETE FROM writes WHERE thread_id IN ({marks})",  # nosec B608
                            batch,
                        )
                    conn.commit()
                    entry["rows_deleted"] = deleted
                    if idle_for < VACUUM_IDLE_SECONDS:
                        # Someone is very likely mid-conversation in here. The
                        # rows are gone; the reclaim waits for a quieter pass
                        # rather than risk locking out their next checkpoint.
                        entry["vacuum_skipped"] = f"store written {idle_for:.0f}s ago"
                    else:
                        try:
                            conn.execute("VACUUM")
                        except sqlite3.OperationalError as exc:
                            # Free pages stay reusable; the file just does not
                            # shrink until a pass that can take the lock.
                            entry["vacuum_skipped"] = str(exc)[:80]
                elif aged and dry_run:
                    marks = ",".join("?" * len(aged))  # placeholders only — see above
                    entry["rows_deleted"] = conn.execute(
                        f"SELECT COUNT(*) FROM checkpoints WHERE thread_id IN ({marks})",  # nosec B608
                        aged,
                    ).fetchone()[0]
            finally:
                conn.close()
            if not dry_run:
                entry["reclaimed_bytes"] = max(0, before - os.path.getsize(db_path))
        except (sqlite3.Error, OSError) as exc:
            entry["error"] = str(exc)[:120]
        stores.append(entry)

    return {
        "scanned": scanned,
        "stores": stores,
        "threads_pruned": sum(s["threads_pruned"] for s in stores),
        "reclaimed_bytes": sum(s["reclaimed_bytes"] for s in stores),
    }


@log_exceptions()
def run_housekeeping(dry_run: bool = False) -> dict:
    """Rotate logs and prune cold conversations. Returns a report."""
    logs = rotate_logs(dry_run=dry_run)
    checkpoints = prune_checkpoints(dry_run=dry_run)

    report = {
        "scanned": logs["scanned"] + checkpoints["scanned"],
        "logs": logs,
        "checkpoints": checkpoints,
        "reclaimed_bytes": logs["reclaimed_bytes"] + checkpoints["reclaimed_bytes"],
        "dry_run": dry_run,
    }

    reclaimed_mb = report["reclaimed_bytes"] / (1024 * 1024)
    log_to_component(
        "tools",
        "Housekeeping",
        f"{'Would reclaim' if dry_run else 'Reclaimed'} {reclaimed_mb:.1f} MB "
        f"({len(logs['rotated'])} logs rotated, "
        f"{checkpoints['threads_pruned']} cold conversations pruned)",
        {
            "scanned": report["scanned"],
            "rotated": logs["rotated"],
            "threads_pruned": checkpoints["threads_pruned"],
            "reclaimed_bytes": report["reclaimed_bytes"],
        },
    )
    return report
