"""Structural guards on the observation log's boundaries (Advisor Roadmap 1.7).

Two properties this feature is worthless without, asserted on the SOURCE rather
than on behaviour — because the value of a boundary is that it cannot be
bypassed, and a behavioural test only covers the paths someone thought to write.

1. **The log is prompt-invisible.** Nothing that assembles a prompt may read it.
   Observations are raw and unreviewed; the store they feed (`lessons_learned`)
   is injected into every prompt and capped, so an unreviewed entry does not add
   noise, it competes with a rule the user wrote.

2. **This pass drafts, it never learns.** `add_lesson` stays reachable from the
   confirmation gate alone. The sibling of this test —
   `test_only_the_confirm_gate_may_call_add_lesson` in
   `tests/test_tools/test_feedback_store.py` — exists for the same reason and is
   the pattern copied here.

A future "just auto-apply the obvious ones" shortcut fails HERE, rather than in
production, where the symptom is a rule nobody wrote silently displacing one they
did.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

OBSERVATION_MODULES = {"tools.observations", "tools.observation_consolidation"}

# The only modules allowed to touch the observation log, and why each one is on
# the list. Everything else in agent/, tools/ and api/ must not know it exists.
ALLOWED_READERS = {
    # the store itself, and the one consumer of it
    "tools/observations.py",
    "tools/observation_consolidation.py",
    # the post-turn WRITER seam
    "api/routers/chat.py",
    # the read surface (GET /api/observations) and the manual consolidate button
    "api/routers/memory.py",
    # the scheduled follow-through sweep + gated pass
    "tools/scheduler.py",
}


def _source_files():
    for package in ("agent", "tools", "api"):
        yield from sorted((REPO_ROOT / package).rglob("*.py"))


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every module named by an import in this file, including function-local ones."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_no_prompt_builder_can_read_the_observation_log():
    offenders = []
    for path in _source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_READERS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file is another test's problem
            continue
        if _imported_modules(tree) & OBSERVATION_MODULES:
            offenders.append(rel)

    assert offenders == [], (
        "The observation log is raw and unreviewed and must not reach a prompt. "
        f"These modules import it: {offenders}. If one of them legitimately needs "
        "it, adding it to ALLOWED_READERS is a decision about what the model can "
        "see, and should be made deliberately."
    )


def test_the_user_context_builder_does_not_mention_the_log():
    """get_user_context is what actually lands in the system prompt. Belt and
    braces alongside the import scan: a getattr-style read would slip past AST
    import matching, and this is the one function where that would matter."""
    source = (REPO_ROOT / "tools" / "memory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_user_context"
    )
    body = ast.get_source_segment(source, func) or ""

    assert "observation" not in body.lower()


def test_the_consolidation_pass_never_learns_a_lesson_itself():
    """It drafts into the pending queue; a human promotes. Roadmap 1.7's open
    decision was settled 2026-07-27: no rule ever auto-promotes, at any evidence
    count."""
    source = (REPO_ROOT / "tools" / "observation_consolidation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imported = _imported_modules(tree)

    assert "add_lesson" not in called
    assert not any(name.endswith("add_lesson") for name in imported), (
        "The consolidation pass must reach lessons_learned only through the human "
        "confirmation gate."
    )


@pytest.mark.parametrize("module", sorted(OBSERVATION_MODULES))
def test_the_allowlist_names_files_that_exist(module):
    """A stale allowlist entry would silently widen the guard."""
    assert (REPO_ROOT / (module.replace(".", "/") + ".py")).exists()
    for rel in ALLOWED_READERS:
        assert (REPO_ROOT / rel).exists(), f"ALLOWED_READERS names a missing file: {rel}"
