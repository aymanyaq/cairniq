"""Reversible compression for tool output fed into LLM reasoning context.

Several reasoning nodes inline raw tool results (quotes, fundamentals, filings,
news) into the prompt they send to the model. Large results must be trimmed to
stay within the token budget — but a blind head-chop (``result[:4000]``)
permanently drops the *tail*, where financial tool output often carries the
totals, latest rows, or summary that matter most.

This module formalizes the head+tail "compact but preserve" idiom the deep
reasoning planner already uses (``_record_and_compact``) into one reusable,
testable helper:

* whitespace bloat is collapsed first (often avoids truncation entirely);
* if still over budget, the head AND tail are kept and only the middle is
  elided, behind a transparent marker;
* the full original is stashed in a bounded in-process registry keyed by a
  short id embedded in the marker, so a retrieval path can pull it back via
  :func:`get_full_tool_result` — reversible, and nothing leaves the machine.

Inspired by the local prompt-compression proxy pattern (extraheadroom.com),
done in-process so the agent graph and structured-output calls stay untouched.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict

__all__ = ["annotate_authored_basis", "compress_tool_result", "get_full_tool_result"]

# --- Roadmap 2.7: attribute on the basis marker, don't merely carry it ---------
#
# Several tools now stamp their payload with `basis: "authored constant"` when the
# figures they return were typed into the codebase rather than measured
# (`run_stress_test`'s -35%/-45% drops, `match_historical_regime`'s scenario
# strings). Stamping alone changed nothing for the reader: the marker travelled
# inside a dict the model was free to skim past, and an assumed -35% still reached
# the user's screen sounding like a finding.
#
# This is the seam where a tool result becomes model context, so the instruction is
# attached HERE — once, for every tool that carries the marker now or later —
# rather than as a sentence bolted onto each node's prompt, where it would drift
# out of sync with the tools that need it.
_AUTHORED_BASIS_RE = re.compile(r"""['"]basis['"]\s*:\s*['"]authored constant['"]""")

_AUTHORED_BASIS_DIRECTIVE = (
    "\n\n[BASIS — AUTHORED, NOT MEASURED: the result above carries "
    "basis=\"authored constant\", meaning one or more of its figures was typed into "
    "this codebase by hand rather than computed from data. If you use any of them, "
    "attribute the assumption in the same sentence (\"an assumed 35% market drop\", "
    "\"an authored analogue, not a measurement\"). Never present it as what history "
    "shows, what the data says, or a measured result. Where the payload names a "
    "measured_alternative, prefer calling that tool over reporting the authored "
    "figure.]"
)


def annotate_authored_basis(text: str) -> str:
    """Append the attribution directive when a tool payload declares an authored basis.

    Idempotent and substring-gated, so it costs nothing on the overwhelming majority
    of tool results that carry no marker. Appended at the END deliberately: both
    compaction paths in this codebase preserve the tail, so the directive survives
    the truncation that a mid-payload note would not.
    """
    if not text or "authored constant" not in text:
        return text
    if "[BASIS — AUTHORED" in text:
        return text
    if not _AUTHORED_BASIS_RE.search(text):
        # Prose merely discussing authored constants (this module's own docstrings,
        # a tool explaining the concept) must not trip the directive.
        return text
    return text + _AUTHORED_BASIS_DIRECTIVE

# Default budget mirrors the historical blind-chop threshold so token economics
# don't shift when call sites switch to this helper.
_DEFAULT_MAX_CHARS = 4000
_HEAD_RATIO = 0.7  # keep more of the head than the tail by default

_INLINE_WS_RE = re.compile(r"[ \t\f\v]{2,}")
_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n[ \t]*(?:\n[ \t]*)+")

# Bounded registry of full originals so compressed markers stay reversible within
# a session without unbounded memory growth (FIFO eviction past the cap).
_REGISTRY: OrderedDict[str, str] = OrderedDict()
_REGISTRY_MAX = 128


def _collapse_whitespace(text: str) -> str:
    """Shrink repetitive whitespace — runs of spaces/tabs and blank-line
    stretches — that bloat tool output without carrying meaning."""
    text = _INLINE_WS_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text


def _stash(content: str) -> str:
    rid = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:10]
    _REGISTRY[rid] = content
    _REGISTRY.move_to_end(rid)
    while len(_REGISTRY) > _REGISTRY_MAX:
        _REGISTRY.popitem(last=False)
    return rid


def get_full_tool_result(rid: str) -> str | None:
    """Return the full original tool output for a compression id (taken from a
    marker), or ``None`` if it has aged out of the bounded registry."""
    return _REGISTRY.get(rid)


def compress_tool_result(
    content,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    head_ratio: float = _HEAD_RATIO,
) -> str:
    """Compress a tool result for inclusion in LLM reasoning context.

    Collapses whitespace first; only if still over ``max_chars`` does it keep the
    head and tail and elide the middle, leaving a transparent, reversible marker
    (``id=<rid>``) whose original is retrievable via :func:`get_full_tool_result`.
    A blind head-chop is never used, so trailing summaries/totals survive.
    """
    if not content:
        return ""
    text = content if isinstance(content, str) else str(content)

    # A 2.7 basis directive is an instruction ABOUT the payload, not payload — so it
    # is held out of the budget and re-appended, rather than competing with the data
    # for tail space. Caught by test: at a small budget the tail kept only the
    # directive's closing clause and elided the "AUTHORED, NOT MEASURED" headline,
    # which is the one part a reader needs.
    directive = ""
    if text.endswith(_AUTHORED_BASIS_DIRECTIVE):
        directive = _AUTHORED_BASIS_DIRECTIVE
        text = text[: -len(_AUTHORED_BASIS_DIRECTIVE)]

    collapsed = _collapse_whitespace(text)
    if len(collapsed) <= max_chars:
        return collapsed + directive

    rid = _stash(text + directive)
    head_len = max(0, int(max_chars * head_ratio))
    tail_len = max(0, max_chars - head_len)
    head = collapsed[:head_len].rstrip()
    tail = collapsed[-tail_len:].lstrip() if tail_len else ""
    omitted = len(collapsed) - len(head) - len(tail)
    marker = (
        f"\n\n[... {omitted} chars elided to fit the reasoning budget; "
        f"full result preserved, id={rid} ...]\n\n"
    )
    return head + marker + tail + directive
