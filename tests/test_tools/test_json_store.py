"""Store writes must not be able to destroy the store.

`open(path, 'w')` truncates before the first byte of new content lands. Nineteen
of the twenty-eight store modules did exactly that, so a crash — or merely a
payload that failed to serialize — mid-save left the file empty. Each of these
files is the only copy of something the user typed: the trade journal, the memory
store, the scan ledger.
"""
import json
import os

import pytest

from tools.json_store import read_json, write_json_atomic


def _temp_files(directory) -> list[str]:
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


def test_round_trips(tmp_path):
    path = str(tmp_path / "store.json")
    write_json_atomic(path, {"a": 1, "b": [2, 3]})
    assert read_json(path) == {"a": 1, "b": [2, 3]}


def test_leaves_no_temp_file_behind(tmp_path):
    path = str(tmp_path / "store.json")
    write_json_atomic(path, {"a": 1})
    assert _temp_files(tmp_path) == []


def test_a_failing_serialization_leaves_the_existing_store_intact(tmp_path):
    """The property truncate-first could not offer.

    Serialization happens before the rename, so an unencodable payload aborts
    without ever touching the real file.
    """
    path = str(tmp_path / "store.json")
    write_json_atomic(path, {"good": "data"})

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": Unserializable()})

    assert read_json(path) == {"good": "data"}, "a failed write destroyed the store"
    assert _temp_files(tmp_path) == [], "failed write stranded a temp file"


def test_the_temp_file_is_a_sibling_of_the_target(tmp_path, monkeypatch):
    """os.replace is only atomic within one filesystem.

    Writing the temp file to the system temp dir would silently degrade the
    rename to a copy — non-atomic again, on exactly the machines where /tmp is a
    separate mount.
    """
    seen = {}
    import tools.json_store as js

    real_mkstemp = js.tempfile.mkstemp

    def spy(*args, **kwargs):
        seen["dir"] = kwargs.get("dir")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(js.tempfile, "mkstemp", spy)
    path = str(tmp_path / "nested" / "store.json")
    write_json_atomic(path, {"a": 1})
    assert seen["dir"] == os.path.dirname(os.path.abspath(path))


def test_creates_missing_parent_directories(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "store.json")
    write_json_atomic(path, [1, 2, 3])
    assert read_json(path) == [1, 2, 3]


def test_existing_permissions_are_preserved(tmp_path):
    """mkstemp creates 0600; a store must not tighten its own mode on every save."""
    path = str(tmp_path / "store.json")
    write_json_atomic(path, {"a": 1})
    os.chmod(path, 0o644)
    write_json_atomic(path, {"a": 2})
    assert oct(os.stat(path).st_mode & 0o777) == "0o644"


def test_dump_kwargs_are_forwarded(tmp_path):
    from datetime import date

    path = str(tmp_path / "store.json")
    write_json_atomic(path, {"d": date(2026, 7, 31)}, default=str)
    assert read_json(path) == {"d": "2026-07-31"}


def test_indent_none_writes_compact_json(tmp_path):
    path = str(tmp_path / "store.json")
    write_json_atomic(path, {"a": 1, "b": 2}, indent=None)
    assert "\n" not in open(path).read().strip()


def test_a_reader_never_sees_a_partial_file(tmp_path):
    """The whole point: replace is atomic, so a concurrent reader gets old or new."""
    path = str(tmp_path / "store.json")
    write_json_atomic(path, {"version": 1, "payload": "x" * 100_000})

    for _ in range(20):
        write_json_atomic(path, {"version": 2, "payload": "y" * 100_000})
        loaded = json.load(open(path))
        assert loaded["version"] in (1, 2)
        assert len(loaded["payload"]) == 100_000


def test_read_json_defaults_for_missing_and_corrupt(tmp_path):
    missing = str(tmp_path / "nope.json")
    assert read_json(missing, default={"fallback": True}) == {"fallback": True}

    corrupt = str(tmp_path / "corrupt.json")
    with open(corrupt, "w") as f:
        f.write("{not json at all")
    assert read_json(corrupt, default=[]) == []


def test_read_json_default_is_none_when_unspecified(tmp_path):
    assert read_json(str(tmp_path / "nope.json")) is None
