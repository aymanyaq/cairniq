"""Durable JSON writes for the profile stores.

`open(path, 'w')` truncates the file before the first byte of new content is
written. Anything that stops the process in between — a crash, a SIGKILL, a full
disk, the machine losing power — leaves the store truncated or empty, and every
one of these files is the only copy of something the user typed: the trade
journal, the memory store, risk constraints, the scan ledger.

Ten modules already did the tmp-file + `os.replace` dance by hand and the other
nineteen did not, which was an accident of when each was written rather than a
judgement about which data mattered. `os.replace` is atomic on POSIX and Windows,
so a reader concurrently loading the file sees either the whole old version or
the whole new one — never a half-written one.

Use `write_json_atomic` for anything under `user_data/`. `read_json` is the
matching reader for the common "load it, or fall back to a default" shape; stores
with bespoke recovery semantics (graph_memory retries a decode race, portfolio
falls back to a last-known-good snapshot) keep their own loaders.
"""
import json
import os
import tempfile
from typing import Any


def write_json_atomic(path: str, data: Any, **dump_kwargs: Any) -> None:
    """Serialize `data` to `path` so a reader never observes a partial file.

    Writes to a temp file in the SAME directory — `os.replace` is only atomic
    within one filesystem, so /tmp would silently degrade to a copy — then fsyncs
    and renames over the target.

    `dump_kwargs` are passed to `json.dump` (`indent`, `default`, …). Serialization
    happens BEFORE the rename, so a payload that fails to encode leaves the
    existing file untouched instead of destroying it, which is the other half of
    what truncate-first got wrong.
    """
    dump_kwargs.setdefault("indent", 2)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        # Carry the original file's permissions across; mkstemp creates 0600 and
        # a store that silently tightened its own mode on every save would be a
        # surprise for anything else reading it.
        try:
            os.chmod(tmp_path, os.stat(path).st_mode & 0o777)
        except OSError:
            pass
        os.replace(tmp_path, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt or GeneratorExit here
        # would otherwise strand the temp file next to the real store forever.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json(path: str, default: Any = None) -> Any:
    """Load JSON from `path`, returning `default` if it is missing or unreadable.

    A store that does not exist yet and a store that is corrupt are deliberately
    the same answer here: both mean "there is nothing usable to read", and the
    caller's default is what it wants in either case. Callers needing to tell
    those apart should check existence themselves first.
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default
