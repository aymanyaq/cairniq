"""
Server-side quick-action prompts.

Large quick-action *system instructions* live as plain-text files in agent/prompts/
(one per action) instead of multi-KB JS strings buried in templates/index.html. A
button sends a slim marker — ``[QuickAction name=<x>]`` — and DeepReasoning rewrites it
in place to the full prompt. Same mechanism as the Catalyst Engine's ``[EventScenario]``
marker (agent/catalyst_engine.py), so there's one consistent way to host prompts.

To add a quick-action prompt:
  1. drop the instruction in ``agent/prompts/<name>.txt``
  2. register it in QUICK_ACTION_PROMPTS below (marker name -> file)
  3. point the button at ``[DeepReasoning] [QuickAction name=<name>]``
"""

from __future__ import annotations

import logging
import os
import re

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_logger = logging.getLogger(__name__)


def _load(filename: str) -> str:
    path = os.path.join(_PROMPT_DIR, f"{filename}.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        _logger.warning("Quick-action prompt file missing or unreadable: %s", path)
        return ""


# Marker name -> prompt text (loaded once at import).
QUICK_ACTION_PROMPTS: dict[str, str] = {
    "priority": _load("today_priority"),
}

# Tools a quick-action prompt mandates ("TOOLS YOU MUST CALL BEFORE ANSWERING"). Semantic
# tool retrieval can rank a mandated tool out of the top-k, and an unbound tool can never
# be called — so the prompt's MUST silently fails. DeepReasoning unions these into its
# signal-tool floor so the mandate is actually satisfiable. Use the REGISTERED tool name
# (e.g. check_sector_rotation, the @tool wrapper — NOT the underlying detect_sector_rotation).
QUICK_ACTION_REQUIRED_TOOLS: dict[str, list[str]] = {
    "priority": [
        "get_market_pulse_data",
        "check_portfolio_allocation",
        "check_sector_rotation",
        "scan_intraday_movers",
        "verify_portfolio_holdings",
        "check_portfolio_earnings",
    ],
}


def required_tools_for_action(name: str) -> list[str]:
    """Tools to floor for a given quick action. Unknown actions → empty list."""
    return list(QUICK_ACTION_REQUIRED_TOOLS.get((name or "").lower(), []))

# [QuickAction name=priority]  /  [QuickAction]  (name required to match a prompt)
_MARKER_RE = re.compile(
    r"\[\s*QuickAction(?:\s+name\s*=\s*['\"]?([A-Za-z_]+)['\"]?)?\s*\]", re.IGNORECASE
)
_DEEP_PREFIX_RE = re.compile(r"^\s*\[\s*DeepReasoning\s*\]\s*", re.IGNORECASE)


def parse_quick_action_marker(text: str) -> tuple[str, str] | None:
    """Return (name, remaining_text) if `text` carries a known QuickAction marker, else None."""
    if not text:
        return None
    stripped = _DEEP_PREFIX_RE.sub("", str(text))
    m = _MARKER_RE.search(stripped)
    if not m:
        return None
    name = (m.group(1) or "").lower()
    if not QUICK_ACTION_PROMPTS.get(name):
        return None
    remainder = (stripped[: m.start()] + stripped[m.end():]).strip()
    return name, remainder


# Quick actions that commit to checkable trigger levels, and so must also emit
# the watch-conditions side-channel (Roadmap 3.3). The spec is appended at build
# time from tools.watch_conditions rather than pasted into the .txt, so the
# prompt and the parser that reads it can never drift apart.
WATCH_SIDE_CHANNEL_ACTIONS = frozenset({"priority"})


def build_quick_action_prompt(name: str, extra: str = "") -> str:
    """Full prompt for a quick action, with any trailing passthrough (e.g. a FOCUS line) appended."""
    prompt = QUICK_ACTION_PROMPTS.get(name, "")
    if prompt and name in WATCH_SIDE_CHANNEL_ACTIONS:
        from tools.watch_conditions import WATCH_SIDE_CHANNEL_PROMPT
        prompt = f"{prompt}\n\n{WATCH_SIDE_CHANNEL_PROMPT.strip()}"
    if extra and extra.strip():
        prompt = f"{prompt}\n\n{extra.strip()}"
    return prompt


def maybe_rewrite_quick_action(messages: list) -> bool:
    """If the last message carries a ``[QuickAction name=X]`` marker, rewrite its content
    in place to the full server-side prompt. Returns True if it did."""
    if not messages:
        return False
    last = messages[-1]
    content = getattr(last, "content", None)
    if not isinstance(content, str):
        return False
    parsed = parse_quick_action_marker(content)
    if not parsed:
        return False
    name, extra = parsed
    last.content = build_quick_action_prompt(name, extra)
    return True
