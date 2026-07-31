import datetime
import gzip
import os
import sqlite3
import uuid

import pytest

from tools.housekeeping import (
    _GREGORIAN_EPOCH,
    LOG_MAX_BYTES,
    _uuid6_written_at,
    enforce_total_log_budget,
    prune_checkpoints,
    prune_rotations,
    rotate_log_file,
    rotate_logs,
    run_housekeeping,
)


def _uuid6_at(when: datetime.datetime) -> str:
    """Build a UUID6 whose embedded timestamp is exactly ``when``."""
    delta = when - _GREGORIAN_EPOCH
    micros = delta.days * 86400 * 10**6 + delta.seconds * 10**6 + delta.microseconds
    ticks = micros * 10
    digits = f"{ticks:015x}"
    return f"{digits[0:8]}-{digits[8:12]}-6{digits[12:15]}-8000-000000000000"


def _make_store(
    path: str,
    threads: dict[str, list[datetime.datetime]],
    active: bool = False,
) -> None:
    """Create a LangGraph-shaped checkpoint DB with the given thread histories.

    Backdates the file mtimes unless ``active``: a fixture store is not a live
    conversation, and leaving it freshly-written would trip the VACUUM idle
    guard in every test that expects the reclaim to run.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE checkpoints (thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL "
        "DEFAULT '', checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT, type TEXT, "
        "checkpoint BLOB, metadata BLOB, PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
    )
    conn.execute(
        "CREATE TABLE writes (thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL "
        "DEFAULT '', checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL, idx INTEGER "
        "NOT NULL, channel TEXT NOT NULL, type TEXT, value BLOB)"
    )
    for thread_id, stamps in threads.items():
        for when in stamps:
            cid = when if isinstance(when, str) else _uuid6_at(when)
            conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                "checkpoint) VALUES (?, '', ?, ?)",
                (thread_id, cid, b"x" * 512),
            )
            conn.execute(
                "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, "
                "idx, channel) VALUES (?, '', ?, 't', 0, 'c')",
                (thread_id, cid),
            )
    conn.commit()
    conn.close()

    if not active:
        stale = datetime.datetime.now().timestamp() - 3600
        for suffix in ("", "-wal", "-shm"):
            try:
                os.utime(path + suffix, (stale, stale))
            except OSError:
                continue


# --- UUID6 dating -----------------------------------------------------------

def test_uuid6_roundtrips_to_its_own_timestamp():
    when = datetime.datetime(2026, 6, 28, 2, 30, 47)
    assert _uuid6_written_at(_uuid6_at(when)) == when


def test_undatable_ids_return_none_rather_than_a_guess():
    # A row we cannot date must never be treated as old.
    assert _uuid6_written_at(str(uuid.uuid4())) is None
    assert _uuid6_written_at("not-a-uuid") is None
    assert _uuid6_written_at(None) is None
    assert _uuid6_written_at("") is None


def test_uuid6_outside_a_plausible_window_is_rejected():
    assert _uuid6_written_at(_uuid6_at(datetime.datetime(1999, 1, 1))) is None
    future = datetime.datetime.now() + datetime.timedelta(days=400)
    assert _uuid6_written_at(_uuid6_at(future)) is None


# --- log rotation -----------------------------------------------------------

def test_rotation_truncates_in_place_and_keeps_the_inode(tmp_path):
    """The whole point: launchd and FileHandler hold these files open.

    If rotation renamed the path, the writer would keep appending to the old
    inode and the fresh log would stay empty forever.
    """
    log = tmp_path / "big.log"
    log.write_bytes(b"a" * 5000)
    inode_before = os.stat(log).st_ino

    # A writer holding the file open across the rotation, as in production.
    with open(log, "a") as writer:
        reclaimed = rotate_log_file(str(log), max_bytes=1000)
        writer.write("after rotation\n")
        writer.flush()

    assert reclaimed == 5000
    assert os.stat(log).st_ino == inode_before
    # Size, not just content: a writer holding the file WITHOUT O_APPEND keeps
    # its old offset and writes at byte 5000, leaving a sparse file that still
    # ends with the right text while reporting the size we just reclaimed.
    assert log.read_bytes() == b"after rotation\n"
    # The rotated bytes survive, compressed, alongside it.
    archives = list(tmp_path.glob("big.log.*.gz"))
    assert len(archives) == 1
    assert gzip.decompress(archives[0].read_bytes()) == b"a" * 5000


def test_a_log_under_the_cap_is_left_alone(tmp_path):
    log = tmp_path / "small.log"
    log.write_bytes(b"a" * 100)
    assert rotate_log_file(str(log), max_bytes=1000) == 0
    assert log.read_bytes() == b"a" * 100
    assert not list(tmp_path.glob("*.gz"))


def test_dry_run_reports_without_touching_the_file(tmp_path):
    log = tmp_path / "big.log"
    log.write_bytes(b"a" * 5000)
    assert rotate_log_file(str(log), max_bytes=1000, dry_run=True) == 5000
    assert log.stat().st_size == 5000
    assert not list(tmp_path.glob("*.gz"))


def test_rotate_logs_sweeps_both_layouts_and_skips_archives(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    (tmp_path / "cairniq.stderr.log").write_bytes(b"a" * 5000)
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.jsonl").write_bytes(b"b" * 5000)
    (tmp_path / "cairniq.stderr.log.20260101-000000.gz").write_bytes(b"old")

    report = rotate_logs(max_bytes=1000)

    # Two live logs scanned; the pre-existing .gz is not re-rotated.
    assert report["scanned"] == 2
    assert sorted(report["rotated"]) == ["agent/agent.jsonl", "cairniq.stderr.log"]


def test_prune_rotations_expires_only_old_archives(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    old = tmp_path / "a.log.20260101-000000.gz"
    new = tmp_path / "b.log.20260601-000000.gz"
    old.write_bytes(b"x" * 10)
    new.write_bytes(b"y" * 10)
    ancient = datetime.datetime.now().timestamp() - 60 * 86400
    os.utime(old, (ancient, ancient))

    freed = prune_rotations(keep_days=30)

    assert freed == 10
    assert not old.exists()
    assert new.exists()


def test_the_rotation_threshold_is_low_enough_to_actually_fire():
    """The bug was never the mechanism — it was the number.

    At 100 MB, rotation had never fired once in the life of the job: measured
    2026-07-31, agent.jsonl was 41 MB and there were ZERO .gz archives, which
    also left prune_rotations() inert (it only deletes archives, and none
    existed). A threshold no real log ever reaches is not a backstop, it is an
    off switch. This pins it against drifting back up.
    """
    assert LOG_MAX_BYTES <= 32 * 1024 * 1024


def test_the_budget_evicts_oldest_archives_first(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    (tmp_path / "live.log").write_bytes(b"L" * 100)

    now = datetime.datetime.now().timestamp()
    for i, name in enumerate(["a", "b", "c"]):
        p = tmp_path / f"{name}.log.2026010{i}-000000.gz"
        p.write_bytes(b"x" * 100)
        os.utime(p, (now - (10 - i) * 86400, now - (10 - i) * 86400))

    # 400 bytes on disk, budget 250 -> must shed 150, i.e. the two oldest.
    freed = enforce_total_log_budget(max_total_bytes=250)

    assert freed == 200
    assert not (tmp_path / "a.log.20260100-000000.gz").exists()
    assert not (tmp_path / "b.log.20260101-000000.gz").exists()
    assert (tmp_path / "c.log.20260102-000000.gz").exists(), "newest archive evicted first"


def test_the_budget_never_deletes_a_live_log(tmp_path, monkeypatch):
    """A live log is held open by launchd or a FileHandler.

    Unlinking it leaves the writer bound to an unlinked inode — still writing,
    to nowhere, with no error and no file. Archives only.
    """
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    live = tmp_path / "live.log"
    live.write_bytes(b"L" * 5000)

    enforce_total_log_budget(max_total_bytes=10)

    assert live.exists()
    assert live.read_bytes() == b"L" * 5000


def test_live_logs_over_budget_shed_every_archive_and_stop(tmp_path, monkeypatch):
    """Nothing further can be done, and it must not spin trying."""
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    (tmp_path / "live.log").write_bytes(b"L" * 5000)
    (tmp_path / "old.log.20260101-000000.gz").write_bytes(b"x" * 100)

    freed = enforce_total_log_budget(max_total_bytes=10)

    assert freed == 100
    assert not (tmp_path / "old.log.20260101-000000.gz").exists()
    assert (tmp_path / "live.log").exists()
    # Second pass has nothing left to take and returns cleanly.
    assert enforce_total_log_budget(max_total_bytes=10) == 0


def test_a_tree_inside_its_budget_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    archive = tmp_path / "a.log.20260101-000000.gz"
    archive.write_bytes(b"x" * 100)

    assert enforce_total_log_budget(max_total_bytes=10_000) == 0
    assert archive.exists()


def test_budget_dry_run_reports_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    archive = tmp_path / "a.log.20260101-000000.gz"
    archive.write_bytes(b"x" * 500)

    assert enforce_total_log_budget(max_total_bytes=10, dry_run=True) == 500
    assert archive.exists()


def test_rotate_logs_reports_the_tree_size_and_what_the_budget_took(tmp_path, monkeypatch):
    """reclaimed_bytes alone cannot show whether the ceiling is holding."""
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    (tmp_path / "live.log").write_bytes(b"L" * 50)
    old = tmp_path / "old.log.20260101-000000.gz"
    old.write_bytes(b"x" * 400)

    report = rotate_logs(max_bytes=10_000, max_total_bytes=100)

    assert report["budget_evicted_bytes"] == 400
    assert not old.exists()
    assert report["tree_bytes"] == 50
    assert report["reclaimed_bytes"] >= 400


# --- checkpoint pruning -----------------------------------------------------

@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.housekeeping._user_data_root", lambda: str(tmp_path))
    profile = tmp_path / "profiles" / "demo"
    profile.mkdir(parents=True)
    return profile


def test_cold_threads_are_pruned_and_warm_ones_survive_intact(store_root):
    now = datetime.datetime.now()
    _make_store(str(store_root / "checkpoints.sqlite"), {
        "cold": [now - datetime.timedelta(days=90), now - datetime.timedelta(days=60)],
        "warm": [now - datetime.timedelta(days=45), now - datetime.timedelta(days=1)],
    })

    report = prune_checkpoints(retention_days=30)

    assert report["threads_pruned"] == 1
    # "cannot VACUUM from within a transaction" is an OperationalError, so a
    # botched commit would be swallowed into vacuum_skipped and the file would
    # quietly never shrink — the whole point of the pass.
    assert "vacuum_skipped" not in report["stores"][0]
    conn = sqlite3.connect(str(store_root / "checkpoints.sqlite"))
    threads = {r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    # 'warm' keeps its FULL lineage, including the 45-day-old ancestor: pruning
    # is per-thread, so a live conversation is never left with holes.
    assert threads == {"warm"}
    assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 2
    conn.close()


def test_a_thread_with_an_undatable_checkpoint_is_never_pruned(store_root):
    now = datetime.datetime.now()
    _make_store(str(store_root / "checkpoints.sqlite"), {
        "opaque": [now - datetime.timedelta(days=90), str(uuid.uuid4())],
    })

    report = prune_checkpoints(retention_days=30)

    # Its datable rows are all cold, but one row cannot be dated — so the
    # conversation's real last-touch time is unknown and it stays.
    assert report["threads_pruned"] == 0
    conn = sqlite3.connect(str(store_root / "checkpoints.sqlite"))
    assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
    conn.close()


def test_dry_run_counts_what_it_would_delete_without_deleting(store_root):
    now = datetime.datetime.now()
    _make_store(str(store_root / "checkpoints.sqlite"), {
        "cold": [now - datetime.timedelta(days=90), now - datetime.timedelta(days=80)],
    })

    report = prune_checkpoints(retention_days=30, dry_run=True)

    assert report["threads_pruned"] == 1
    assert report["stores"][0]["rows_deleted"] == 2
    conn = sqlite3.connect(str(store_root / "checkpoints.sqlite"))
    assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
    conn.close()


def test_every_store_under_user_data_is_swept(store_root, tmp_path):
    now = datetime.datetime.now()
    other = tmp_path / "profiles" / "_unbound"
    other.mkdir(parents=True)
    _make_store(str(store_root / "checkpoints.sqlite"),
                {"cold": [now - datetime.timedelta(days=90)]})
    _make_store(str(other / "checkpoints.sqlite"),
                {"cold": [now - datetime.timedelta(days=90)]})

    report = prune_checkpoints(retention_days=30)

    assert report["scanned"] == 2
    assert report["threads_pruned"] == 2


def test_vacuum_is_skipped_on_a_store_someone_is_still_writing(store_root):
    """A VACUUM's exclusive lock can fail a live conversation's next checkpoint,
    so an actively-written store keeps its free pages until a quieter pass."""
    now = datetime.datetime.now()
    _make_store(str(store_root / "checkpoints.sqlite"), {
        "cold": [now - datetime.timedelta(days=90), now - datetime.timedelta(days=80)],
    }, active=True)

    report = prune_checkpoints(retention_days=30)
    store = report["stores"][0]

    assert "written" in store["vacuum_skipped"]
    # The deletions still landed — only the space reclaim waited.
    assert store["threads_pruned"] == 1
    assert store["rows_deleted"] == 2
    conn = sqlite3.connect(str(store_root / "checkpoints.sqlite"))
    assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
    conn.close()


def test_liveness_is_measured_before_our_own_deletes(store_root):
    """The guard reads mtime BEFORE opening the store. Measured after, our own
    DELETEs would make every store look busy and suppress the reclaim forever."""
    now = datetime.datetime.now()
    _make_store(str(store_root / "checkpoints.sqlite"), {
        "cold": [now - datetime.timedelta(days=90), now - datetime.timedelta(days=80)],
    })

    report = prune_checkpoints(retention_days=30)

    assert "vacuum_skipped" not in report["stores"][0]
    assert report["stores"][0]["reclaimed_bytes"] >= 0


def test_an_unreadable_store_is_reported_not_raised(store_root):
    (store_root / "checkpoints.sqlite").write_text("this is not a database")

    report = prune_checkpoints(retention_days=30)

    assert "error" in report["stores"][0]
    assert report["threads_pruned"] == 0


# --- the reported number ----------------------------------------------------

def test_scanned_is_nonzero_when_there_is_nothing_to_clean(tmp_path, monkeypatch):
    """The heartbeat counts paths scanned precisely because 0 bytes reclaimed
    is the healthy steady state — reporting bytes would flag a working engine."""
    monkeypatch.setattr("tools.housekeeping.LOG_BASE_DIR", str(tmp_path))
    monkeypatch.setattr("tools.housekeeping._user_data_root", lambda: str(tmp_path))
    (tmp_path / "cairniq.stderr.log").write_bytes(b"tiny")

    report = run_housekeeping()

    assert report["reclaimed_bytes"] == 0
    assert report["scanned"] >= 1
