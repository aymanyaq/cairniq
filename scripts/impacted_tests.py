#!/usr/bin/env python3
"""Print the tests a set of changed source files can actually affect.

The suite is ~1100 tests. Running all of them to check a two-line edit in
tools/freshness.py is why full runs feel mandatory. This selects the tests that
actually execute the changed code, so the edit-test loop costs a second or two.

    make test-changed                                  # the normal way to use this
    python scripts/impacted_tests.py --build-map       # (re)build the coverage map
    python scripts/impacted_tests.py --base main       # everything since main
    python scripts/impacted_tests.py --files tools/freshness.py

Selection is COVERAGE-based, not import-based. A static import graph was tried first
and is useless here: this is a monolith where nearly every test transitively imports
`server`, so editing one leaf module selected 85 of 85 test files. Recording which
lines each test actually runs cuts the same edit down to the ~20 tests that touch it.

The map lives in .testmap.json, built by one instrumented full run (`--build-map`,
~2 min). It goes stale as you add tests, so the tool is deliberately paranoid — it
falls back to running EVERYTHING whenever it cannot prove a smaller set is safe:

  * a changed .py file that appears nowhere in the map (new module, or one only
    imported at collection time);
  * conftest.py, pytest.ini, net_guard.py, requirements.txt, or this script, all of
    which change how every test behaves;
  * any changed non-Python file that is not obviously inert (a template or a JSON
    fixture can absolutely break a test; docs and markdown cannot);
  * no map on disk at all.

A changed test file always selects itself, whether or not it is in the map.

This is what you run on every save. It does not replace `make test` before you push —
a coverage map only knows about code paths that existed when it was built.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = REPO_ROOT / ".testmap.json"
COVERAGE_DB = REPO_ROOT / ".coverage"

MEASURED_PACKAGES = ("agent", "api", "tools", "lib", "server")

# Touching any of these changes how the whole suite runs, so no subset is safe.
RUN_EVERYTHING = {
    "pytest.ini",
    "tests/conftest.py",
    "tests/net_guard.py",
    "scripts/impacted_tests.py",
    "requirements.txt",
    "Makefile",
}

# Changes that cannot break a Python test. Everything else is treated as dangerous.
INERT_SUFFIXES = {".md", ".txt", ".rst", ".lock"}
INERT_PREFIXES = ("docs/", "landing_page/", ".github/", "backups/", "logs/")


# --------------------------------------------------------------------------- map


def build_map() -> int:
    """Run the suite under coverage with per-test contexts and persist file -> tests."""
    cov_args = [f"--cov={pkg}" for pkg in MEASURED_PACKAGES]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-q",
        "--cov-context=test",
        "--cov-report=",
        *cov_args,
    ]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    # Not parallelised: xdist shards the coverage DB per worker and combining it
    # correctly is more machinery than a map rebuild justifies. This runs rarely.
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode not in (0, 1):  # 1 == some tests failed; the map is still valid
        print("map build aborted: pytest could not run", file=sys.stderr)
        return result.returncode
    if not COVERAGE_DB.exists():
        print(f"map build failed: no coverage DB at {COVERAGE_DB}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(COVERAGE_DB)
    query = """
        SELECT f.path, ctx.context
        FROM line_bits lb
        JOIN file f ON f.id = lb.file_id
        JOIN context ctx ON ctx.id = lb.context_id
        WHERE ctx.context != ''
    """
    mapping: dict[str, set[str]] = collections.defaultdict(set)
    for path, context in conn.execute(query):
        try:
            rel = str(Path(path).resolve().relative_to(REPO_ROOT))
        except ValueError:
            continue  # site-packages
        # contexts look like "tests/test_x.py::test_y|run" — key by the test FILE,
        # not the individual node. Coverage's per-test context recording occasionally
        # drops a single node under a full run (a harmless ERROR line during the build,
        # the test itself still passes); keying by file means one lost node can't
        # remove a source->test edge as long as any sibling in the file recorded. It
        # also keeps the map small and lets pytest+xdist schedule whole files.
        test_file = context.rsplit("|", 1)[0].split("::", 1)[0]
        mapping[rel].add(test_file)
    conn.close()

    payload = {
        "built_at": time.time(),
        "head": _git("rev-parse", "HEAD").strip(),
        "files": {k: sorted(v) for k, v in sorted(mapping.items())},
    }
    MAP_PATH.write_text(json.dumps(payload, indent=0), encoding="utf-8")
    n_test_files = len({t for v in mapping.values() for t in v})
    print(
        f"wrote {MAP_PATH.name}: {len(mapping)} source files -> {n_test_files} test files",
        file=sys.stderr,
    )
    return 0


def load_map():
    if not MAP_PATH.exists():
        return None
    try:
        return json.loads(MAP_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ----------------------------------------------------------------------- changes


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout


def changed_files(base: str | None) -> list[str]:
    paths = set(_git("diff", "--name-only", "HEAD").split())
    paths |= set(_git("ls-files", "--others", "--exclude-standard").split())
    if base:
        paths |= set(_git("diff", "--name-only", f"{base}...HEAD").split())
    return sorted(paths)


def _is_inert(path: str) -> bool:
    return path.startswith(INERT_PREFIXES) or Path(path).suffix in INERT_SUFFIXES


# ---------------------------------------------------------------------- selection


def select(paths: list[str]) -> tuple[list[str] | None, str]:
    """Return (test file paths, reason). None means: run the whole suite."""
    testmap = load_map()
    if testmap is None:
        return None, "no .testmap.json (run: make test-map)"

    if any(p in RUN_EVERYTHING for p in paths):
        hit = next(p for p in paths if p in RUN_EVERYTHING)
        return None, f"{hit} affects every test"

    files = testmap["files"]
    selected: set[str] = set()
    for path in paths:
        if _is_inert(path):
            continue
        if path.startswith("tests/") and path.endswith(".py"):
            selected.add(path)  # a changed test always runs itself
            continue
        if not path.endswith(".py"):
            return None, f"cannot reason about non-Python change: {path}"
        if path not in files:
            return None, f"{path} is not in the map (new or collection-only)"
        selected.update(files[path])

    return sorted(selected), f"{len(paths)} changed file(s)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-map", action="store_true", help="rebuild .testmap.json")
    ap.add_argument("--base", help="also include everything changed since this git ref")
    ap.add_argument("--files", nargs="*", help="explicit file list instead of git")
    args = ap.parse_args()

    if args.build_map:
        return build_map()

    paths = args.files if args.files else changed_files(args.base)
    if not paths:
        print("# no changes detected", file=sys.stderr)
        return 0

    tests, reason = select(paths)
    if tests is None:
        print(f"# running everything: {reason}", file=sys.stderr)
        print("tests")
        return 0
    if not tests:
        print(f"# no tests reach the changed files ({reason})", file=sys.stderr)
        return 0

    print(f"# {len(tests)} test file(s) selected from {reason}", file=sys.stderr)
    # Selected values are test-file paths (no spaces); newline-separated for xargs.
    print("\n".join(tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
