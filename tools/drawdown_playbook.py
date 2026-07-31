"""Drawdown playbook — Advisor Roadmap 3.7.

A −30% tape inside a ten-year plan is close to certain. The plan does not die
from the drawdown; it dies if the holder sells into it. With a material
contribution stream a crash is a buying program, not a sell signal — but nobody
reasons their way to that at −25%, in the moment, from a red screen.

So the rules get written while calm and read back when it hurts. This module is
the store for those rules and the text the sentinel (3.4) surfaces when the
market crosses the bands they were written for.

**Nothing here is ever authored by the app.** No default "never sell" list, no
default deployment ladder, no suggested drift band. That is the same contract as
`risk_constraints` and the wealth goal, and it matters more here than anywhere
else in the codebase: a rule invented by software and read back during a crash
carries the full authority of a promise the user made to themselves, and they
will act on it. An unset playbook produces an alert that says the playbook is
missing — naming the absence, never filling it.

The pairing with 4.5's goal projection is the point of the whole item: at −25%
the useful sentence is not "stay calm", it is "at this level, with contributions
continuing, the goal is still funded in N% of paths". That is a fact about their
plan, computed from their own numbers — the only kind of reassurance worth
sending.

Roadmap 3.9 adds the other half. Storing a ladder and reciting it on a deep
crossing still leaves the rungs unarmed: a −5% rung was only ever seen if the
tape happened to fall far enough to surface the whole playbook, which is to say
the shallow rungs were decorative. `evaluate_deployment_ladder` arms them
against peak-to-date drawdown, so the level the user named delivers the action
the user pre-committed to, once, at the moment it is reached.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from tools.exception_logger import log_exceptions

logger = logging.getLogger(__name__)

_PLAYBOOK_BLOCK = "drawdown_playbook"

# Free-text fields the user writes. Ordered lists stay ordered — "what new
# contributions buy first" is a priority sequence, and reordering it silently
# would change the instruction.
_TEXT_LIST_KEYS = ("never_sell", "buy_first")
_TEXT_KEYS = ("notes",)
_NUMBER_KEYS = ("rebalance_drift_pct",)

# Sentinel band keys that surface the playbook. Deliberately only the deep ones:
# a 5% pullback is noise on a decade horizon, and an advisor that recites the
# crash plan every time the tape dips trains the user to ignore it — so by the
# time it matters, it is wallpaper.
PLAYBOOK_BANDS = ("deep", "severe")


@log_exceptions()
def get_playbook() -> dict[str, Any] | None:
    """The user's pre-agreed drawdown rules, or None if nothing is set.

    None is MEANINGFUL and must be surfaced as "no playbook on file" — never
    substituted with sensible-looking defaults. See the module docstring.
    """
    from tools.memory import load_memory

    block = load_memory().get(_PLAYBOOK_BLOCK)
    if not isinstance(block, dict) or not any(block.get(k) for k in block):
        return None
    return block


@log_exceptions()
def set_playbook(updates: dict[str, Any]) -> dict[str, Any] | None:
    """Set or clear playbook fields. Returns the resulting block, or None if empty.

    Keys: never_sell (list), buy_first (list, ORDER IS THE INSTRUCTION),
    deployment_levels (list of {drawdown_pct, action}), rebalance_drift_pct,
    notes. Passing None clears a field; a malformed value is rejected and leaves
    the existing entry standing, so a typo cannot quietly delete a rule the user
    will be relying on during the worst week of the decade.
    """
    from tools.memory import load_memory, save_memory

    memory = load_memory()
    block = memory.get(_PLAYBOOK_BLOCK)
    if not isinstance(block, dict):
        block = {}

    for key, value in (updates or {}).items():
        if key in _TEXT_LIST_KEYS:
            if value is None:
                block.pop(key, None)
            elif isinstance(value, (list, tuple)):
                cleaned = [str(v).strip() for v in value if str(v).strip()]
                if cleaned:
                    block[key] = cleaned
                else:
                    block.pop(key, None)
        elif key in _TEXT_KEYS:
            if value is None or not str(value).strip():
                block.pop(key, None)
            else:
                block[key] = str(value).strip()
        elif key in _NUMBER_KEYS:
            if value is None:
                block.pop(key, None)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue  # reject, keep what stands
            if number > 0:
                block[key] = number
        elif key == "deployment_levels":
            if value is None:
                block.pop(key, None)
                continue
            levels = _clean_deployment_levels(value)
            if levels:
                block["deployment_levels"] = levels
            else:
                block.pop(key, None)

    memory[_PLAYBOOK_BLOCK] = block
    save_memory(memory)
    return get_playbook()


def _clean_deployment_levels(value: Any) -> list[dict[str, Any]]:
    """Normalize the cash-deployment ladder, sorted shallowest-first.

    Sorted so the ladder reads in the order it will be hit. A malformed rung is
    dropped rather than defaulted — a deployment level with no depth is not a
    rule, and guessing one would put the user's money on a number nobody chose.
    """
    if not isinstance(value, (list, tuple)):
        return []
    levels = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        try:
            depth = abs(float(entry.get("drawdown_pct")))
        except (TypeError, ValueError):
            continue
        action = str(entry.get("action", "")).strip()
        if depth > 0 and action:
            levels.append({"drawdown_pct": depth, "action": action})
    return sorted(levels, key=lambda x: x["drawdown_pct"])


def describe_playbook(playbook: dict[str, Any] | None, drawdown_pct: float) -> str:
    """Render the playbook for an alert body at `drawdown_pct` (a positive depth).

    Rungs already breached are marked, because at −22% the user needs to know
    which of their own instructions are live NOW, not read a ladder and do the
    arithmetic while panicking.
    """
    if not playbook:
        return (
            "**No drawdown playbook is on file.**\n\n"
            "Nothing is being recommended here — these rules only mean anything if "
            "you wrote them yourself, and writing them now, mid-drawdown, is exactly "
            "the decision this instrument exists to avoid. Set them once the tape is "
            "calm (Context › Drawdown Playbook), and they will be read back to you "
            "the next time it is not."
        )

    lines: list[str] = []
    never_sell = playbook.get("never_sell")
    if never_sell:
        lines.append("**Never sold in a drawdown:** " + ", ".join(never_sell))

    buy_first = playbook.get("buy_first")
    if buy_first:
        ordered = " → ".join(f"{i}. {item}" for i, item in enumerate(buy_first, 1))
        lines.append(f"**New contributions buy, in order:** {ordered}")

    levels = playbook.get("deployment_levels") or []
    if levels:
        rungs = []
        for rung in levels:
            depth = rung["drawdown_pct"]
            hit = " ← **LIVE NOW**" if drawdown_pct >= depth else ""
            rungs.append(f"−{depth:g}%: {rung['action']}{hit}")
        lines.append("**Cash deployment ladder:**\n" + "\n".join(f"  • {r}" for r in rungs))

    drift = playbook.get("rebalance_drift_pct")
    if drift:
        lines.append(f"**Rebalance trigger:** any sleeve more than {drift:g}% off target.")

    notes = playbook.get("notes")
    if notes:
        lines.append(f"**Your note:** {notes}")

    return "\n\n".join(lines) if lines else describe_playbook(None, drawdown_pct)


def goal_status_line() -> str:
    """One sentence on whether the goal still funds AT TODAY'S depressed value.

    This is the behavioral half of 3.7 and the reason it pairs with 4.5: the
    projection runs off the CURRENT portfolio value, so during a drawdown it
    answers the only question that actually matters — does the plan still work
    from here, with contributions continuing?

    Returns "" when no goal is set. An unset goal must produce SILENCE, never a
    reassuring generality: "you're still on track" without a target behind it is
    a fabricated comfort, and this is the worst possible moment to fabricate one.
    """
    try:
        from tools.goal_projection import build_goal_projection

        projection = build_goal_projection(num_simulations=2000)
        if not projection.get("available"):
            return ""
        funded = projection.get("goal_success_rate")
        if funded is None:
            return ""
        currency = projection.get("currency") or ""
        contribution = projection.get("annual_contribution")
        years = projection.get("horizon_years")
        return (
            f"**From here, with contributions continuing:** the goal is still funded in "
            f"{funded:.0f}% of simulated paths over {years} years "
            f"(assumes {contribution:,.0f} {currency}/yr keeps going in). "
            f"Selling converts a paper drawdown into a permanent one and removes the "
            f"recovery those paths depend on."
        )
    except Exception as e:  # noqa: BLE001 — an alert must never fail on its optional half
        logger.debug(f"goal status line unavailable: {e}")
        return ""


# ---------------------------------------------------------------------------
# The armed ladder — Roadmap 3.9
# ---------------------------------------------------------------------------

# A rung fires ONCE per drawdown episode, and the episode ends when the index is
# back at (or within this much of) a new high — the user's call, 2026-07-27.
#
# The tolerance is the load-bearing half of that decision, in both directions.
# `drawdown_from_high` reads exactly 0 only on a day a new high actually prints,
# so a strict `== 0` test would leave the ladder spent through a recovery that
# never quite tags the old peak. Equally it has to stay well clear of the
# shallowest rung anyone would plausibly write: reset anywhere near −5% and a
# 1pp wobble would re-arm a rung the user had already deployed into, which is
# the machine-gunned inbox 3.4's hysteresis exists to prevent — except here each
# repeat costs real money rather than attention.
LADDER_RESET_PCT = 1.0

# Recorded in place of a timestamp for a rung that was already breached the
# first time the ladder was seen. Distinct from a real fire so the two can never
# be confused when reading the state file by hand.
LADDER_BASELINE = "baseline"


def _rung_key(rung: dict[str, Any]) -> str:
    """Stable state-file key for a rung: its depth, '5' rather than '5.0'."""
    return f"{rung['drawdown_pct']:g}"


def _rung_alert(rung: dict[str, Any], depth: float, episode: str, now: datetime) -> dict[str, Any]:
    """One alert spec for a rung that has just been crossed.

    The message says what the user wrote and that nothing was done for them.
    It deliberately adds no encouragement, no sizing and no view on the market:
    the entire authority of this alert comes from the fact that every word of
    the instruction is the user's own, and anything the app contributes beyond
    delivering it dilutes exactly that.
    """
    target = rung["drawdown_pct"]
    return {
        "summary": {
            "type": "deployment_rung",
            "drawdown_pct": target,
            "severity": "warning",
        },
        "raise": {
            "title": f"Deployment ladder: your −{target:g}% rung is live",
            "message": (
                f"SPY is {depth:.1f}% off its 6-month high, which reaches the −{target:g}% "
                f"rung of the cash-deployment ladder you wrote.\n\n"
                f"**Your pre-committed action:** {rung['action']}\n\n"
                "This is your own instruction, recorded while the tape was calm and "
                "delivered now because the level you named has been reached. Nothing has "
                "been done for you — the decision and the order are yours."
            ),
            "severity": "warning",
            "source": "sentinel",
            # Scoped to the episode, not the date: a rung is a once-per-episode
            # event, so a same-day duplicate is a bug rather than a refresh.
            "dedup_key": f"playbook:deployment:{target:g}:{episode}",
            "data": {
                "signal": "deployment_rung",
                "drawdown_pct": target,
                "observed_drawdown_pct": round(depth, 1),
                "action": rung["action"],
                "episode": episode,
            },
        },
    }


def evaluate_deployment_ladder(
    drawdown_pct: float,
    ladder_state: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    """Arm the user's cash-deployment rungs against peak-to-date drawdown (3.9).

    `drawdown_pct` is a POSITIVE depth (5.0 means 5% off the high) and must come
    from the caller's existing peak reading — 3.4 already tracks one, and a
    second tracker here could drift against the band alerts and deploy on a
    depth the user was never shown. `ladder_state` is the caller's persisted
    slot and is mutated in place.

    3.7 stores the rungs and recites them on a deep crossing; this is what
    actually pulls the trigger. Each rung delivers its action ONCE per drawdown
    episode, and the whole ladder re-arms only when the tape recovers to a new
    high (see LADDER_RESET_PCT) — by which point the episode is over and the
    cash it deployed has, by the plan's own logic, been rebuilt.

    A first observation is a SILENT BASELINE. Rungs already breached the first
    time the ladder is seen are recorded as passed and never fired, which is the
    sentinel's standing "void if already true when armed" rule: a ladder written
    on a Tuesday when SPY is already 12% down must not immediately fire three
    rungs for levels the tape crossed before anybody armed them. Those rungs are
    not lost — `describe_playbook` marks every breached rung LIVE NOW on a deep
    crossing, which is the surface built for precisely that case.

    Returns ``{"specs", "armed", "fired", "seeded", "levels", "evaluated"}``.
    The counts are the liveness report 2.5/2.6 require and `armed` is returned
    even when it is 0, because the playbook is empty on every profile today and
    an inert ladder that reports nothing is indistinguishable from a healthy
    quiet one.
    """
    stamp = now.isoformat(timespec="seconds")
    levels = (get_playbook() or {}).get("deployment_levels") or []
    result = {
        "specs": [],
        "armed": 0,
        "fired": 0,
        "seeded": 0,
        "levels": len(levels),
        "evaluated": True,
    }

    if not levels:
        # No ladder on file. Drop any state from a ladder since deleted, so
        # re-writing one later starts from a clean baseline rather than
        # inheriting fires recorded against rungs that no longer exist.
        ladder_state.clear()
        return result

    keys = [_rung_key(rung) for rung in levels]
    depth = max(0.0, float(drawdown_pct))

    stored = ladder_state.get("fired")
    # Forget rungs the user has since edited away; keeping them would let a
    # deleted −10% rung suppress a newly written one at the same depth.
    fired = {k: v for k, v in stored.items() if k in keys} if isinstance(stored, dict) else {}

    if depth <= LADDER_RESET_PCT:
        # At or near a new high: the episode is over and every rung re-arms.
        ladder_state.update(
            {"seen": True, "episode": "", "fired": {}, "depth": depth, "updated_at": stamp}
        )
        result["armed"] = len(levels)
        return result

    first_sight = not ladder_state.get("seen")
    episode = ladder_state.get("episode") or stamp

    if first_sight:
        for rung, key in zip(levels, keys):
            if depth >= rung["drawdown_pct"]:
                fired[key] = LADDER_BASELINE
                result["seeded"] += 1
    else:
        for rung, key in zip(levels, keys):
            if key in fired or depth < rung["drawdown_pct"]:
                continue
            fired[key] = stamp
            result["specs"].append(_rung_alert(rung, depth, episode, now))
        result["fired"] = len(result["specs"])

    ladder_state.update(
        {"seen": True, "episode": episode, "fired": fired, "depth": depth, "updated_at": stamp}
    )
    result["armed"] = len(levels) - len(fired)
    return result


def build_drawdown_message(drawdown_pct: float, band_label: str) -> str:
    """The full alert body the sentinel attaches when a deep band is crossed."""
    parts = [
        f"SPY is {drawdown_pct:.1f}% off its 6-month high — {band_label}. "
        f"This is the moment your pre-agreed rules exist for.",
        describe_playbook(get_playbook(), drawdown_pct),
    ]
    goal_line = goal_status_line()
    if goal_line:
        parts.append(goal_line)
    return "\n\n".join(parts)
