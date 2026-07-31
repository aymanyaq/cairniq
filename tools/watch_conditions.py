"""
Watch-conditions engine — Advisor Roadmap 3.3.

The advisor already commits to triggers in prose: Today's Priority's NEXT-CHECK
board, the catalyst engine's TRIGGER PLAN, a scenario's Confirms/Invalidates
column. Prose triggers are unenforceable — nothing evaluates them, so a
condition the advisor set is only ever re-checked if the user happens to ask
again. The advisor's own commitments quietly expire.

This module closes that loop. The prompts emit a machine-readable side-channel
beside the prose (a ``<watch>…</watch>`` block, stripped before display);
``parse_watch_block`` validates it, ``add_conditions`` stores it per profile,
and ``evaluate_conditions`` re-checks every armed condition on a market-hours
tick with a batched, zero-LLM price read. A satisfied condition fires ONCE into
the alerts inbox (3.2) and is then resolved, so it can never flap.

Deliberate limits, each chosen against a failure this codebase has already had:

- Only ``price`` and ``pct_change`` metrics, both served by ONE ``get_stock_data``
  read per distinct symbol per tick. Anything the parser does not fully
  understand is DROPPED, never guessed — an invented trigger level is the same
  class of failure as an invented portfolio total.
- A condition already satisfied at its FIRST evaluation is voided, not fired.
  The advisor wrote a trigger for something that had already happened; alerting
  would be noise, and it usually means the level was parsed off the wrong side.
- Firing is terminal. A fired condition leaves the active set, so a price
  oscillating around a threshold cannot produce a second alert — hysteresis by
  construction rather than by tuning.
- Quotes carry no as-of stamp yet (roadmap 5.8), so every record keeps the value
  and the timestamp it was judged on. A fire is auditable after the fact even
  though the freshness of its input is not yet provable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from agent.logger import log_to_component
from tools.user_profile import get_data_path

_WATCH_FILENAME = "watch_conditions.jsonl"
_MAX_RECORDS = 400
_MAX_ACTIVE_PER_TURN = 6        # one answer may not flood the store
_MAX_LABEL_CHARS = 200
_MAX_ACTION_CHARS = 300
_DEFAULT_EXPIRY_DAYS = 30
_MIN_EXPIRY_DAYS = 1
_MAX_EXPIRY_DAYS = 180

METRICS = ("price", "pct_change")
OPERATORS = ("<=", ">=", "<", ">")
DIRECTIONS = ("entry", "exit", "confirms", "invalidates", "watch")
STATUSES = ("active", "fired", "expired", "void", "cancelled")

# A ticker as the market data layer accepts it: plain symbols, index symbols
# (^VIX), FX/futures (ZQ=F), crypto pairs (BTC-USD), TSX suffixes (SHOP.TO).
_SYMBOL_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,11}$")

# The side-channel block. Tag-shaped rather than a markdown fence so the existing
# streaming sanitizer shape (see strip_watch_blocks) can drop it token by token —
# a ```watch fence would render visibly in the chat for the seconds it takes the
# JSON to stream.
_WATCH_BLOCK_RE = re.compile(r"<watch>(.*?)</watch>", re.DOTALL | re.IGNORECASE)
_WATCH_OPEN_TAIL_RE = re.compile(r"<watch>(?:(?!</watch>).)*$", re.DOTALL | re.IGNORECASE)
_WATCH_PARTIAL_TAG_RE = re.compile(r"<w(?:a(?:t(?:c(?:h>?)?)?)?)?$", re.IGNORECASE)


# The canonical side-channel spec, appended to every prompt that commits to
# triggers. It lives beside the parser on purpose: a prompt and a parser that
# drift apart fail SILENTLY — the block still streams, still gets stripped, and
# simply stores nothing.
WATCH_SIDE_CHANNEL_PROMPT = """
WATCH-CONDITIONS SIDE-CHANNEL (machine-readable, appended AFTER all prose):
Any trigger you commit to above — a next-check level, an entry/stop/target, a
condition that would confirm or invalidate a scenario — must ALSO be emitted as
a JSON block so the system can actually watch it for you between conversations.
A trigger stated only in prose is never checked again.

End your answer with, and nothing after it:

<watch>
{"conditions": [
  {"symbol": "NVDA", "metric": "price", "operator": "<=", "threshold": 165.00,
   "label": "NVDA back inside the accumulation zone", "action": "Execute the half-position entry, 2% at risk to the $148 stop",
   "direction": "entry", "expires_in_days": 30}
]}
</watch>

Field rules (a condition that breaks ANY of them is discarded silently, so be exact):
- symbol: one ticker exactly as it trades (AAPL, SHOP.TO, ^VIX, BTC-USD).
- metric: "price" (last price) or "pct_change" (today's % move). Nothing else exists.
- operator: "<=", ">=", "<", ">".
- threshold: a bare number — the level you named in the prose. Never a range, never a guess.
- label: what it means when it fires, in your own words.
- action: what you are instructing the user to DO when it fires. Required — a
  trigger with no committed action is just a notification, and this channel is
  not for notifications.
- direction: "entry", "exit", "confirms", "invalidates", or "watch".
- expires_in_days: how long the trigger stays valid (1-180).

HARD RULES:
- Emit ONLY triggers you actually stated in the prose above, at the SAME levels.
  Do not invent a level to fill the block, and do not round one you cited.
- Only levels anchored in this turn's tool output are eligible. If you had no
  real number, emit no condition for it.
- At most 4 conditions. Closest-to-firing first.
- If nothing in your answer commits to a checkable level, omit the block entirely.
  An empty block is correct and costs nothing; a fabricated one commits the user
  to a trade trigger that came from nowhere.
"""


# ---------------------------------------------------------------------------
# Visible-text sanitizer
# ---------------------------------------------------------------------------

def strip_watch_blocks(text: Any) -> Any:
    """Remove ``<watch>`` side-channel blocks from user-visible text.

    Handles three shapes so a streaming answer never flashes raw JSON: the
    complete block, an opened-but-unterminated block at the end of a partial
    stream, and a partial opening tag (``<wat``) at the very end of a chunk.
    Type-safe: non-str input is returned unchanged.
    """
    if not isinstance(text, str) or "<" not in text:
        return text
    text = _WATCH_BLOCK_RE.sub("", text)
    text = _WATCH_OPEN_TAIL_RE.sub("", text)
    return _WATCH_PARTIAL_TAG_RE.sub("", text)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _watch_file() -> str:
    return get_data_path(_WATCH_FILENAME)


def _load_all() -> list[dict[str, Any]]:
    """Read every condition, oldest → newest. Skips corrupt lines; [] on error."""
    try:
        path = _watch_file()
        if not os.path.exists(path):
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("id"):
                    records.append(rec)
        return records
    except Exception:
        return []


def _trim(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap the store without ever evicting a live condition.

    Same lesson as the recommendation ledger's flat 50-cap (142fc18): a blind
    tail-slice silently deletes the in-flight rows the feature exists to track.
    Active conditions are always kept; only resolved ones are trimmed.
    """
    active = [r for r in records if r.get("status") == "active"]
    resolved = [r for r in records if r.get("status") != "active"]
    keep_resolved = max(0, _MAX_RECORDS - len(active))
    return active + resolved[-keep_resolved:] if keep_resolved else active


def _write_all(records: list[dict[str, Any]]) -> bool:
    """Atomically replace the store (tmp + replace). Never raises."""
    try:
        path = _watch_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in _trim(records):
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
        return True
    except Exception as e:
        log_to_component("server", "Watch", f"watch_conditions.jsonl rewrite failed: {e}", level=logging.WARNING)
        return False


# ---------------------------------------------------------------------------
# Parsing — strict by design
# ---------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    try:
        return float(str(value).replace("$", "").replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _clean_condition(raw: Any) -> dict[str, Any] | None:
    """Validate ONE emitted condition. Returns None for anything not fully understood.

    Every field is required except `direction` and `expires_in_days`. There is no
    inference here on purpose: a condition whose operator or metric we had to
    guess would commit the advisor to a trigger it never wrote.
    """
    if not isinstance(raw, dict):
        return None

    symbol = str(raw.get("symbol", "")).strip().upper()
    if not symbol or not _SYMBOL_RE.match(symbol):
        return None

    metric = str(raw.get("metric", "")).strip().lower()
    if metric not in METRICS:
        return None

    operator = str(raw.get("operator", "")).strip()
    if operator not in OPERATORS:
        return None

    threshold = _num(raw.get("threshold"))
    if threshold is None:
        return None
    # A price threshold at or below zero is a parse artifact, not a level.
    if metric == "price" and threshold <= 0:
        return None

    label = str(raw.get("label", "")).strip()
    action = str(raw.get("action", "")).strip()
    if not label or not action:
        return None

    direction = str(raw.get("direction", "watch")).strip().lower()
    if direction not in DIRECTIONS:
        direction = "watch"

    days = _num(raw.get("expires_in_days"))
    days = _DEFAULT_EXPIRY_DAYS if days is None else int(days)
    days = max(_MIN_EXPIRY_DAYS, min(_MAX_EXPIRY_DAYS, days))

    return {
        "symbol": symbol,
        "metric": metric,
        "operator": operator,
        "threshold": round(threshold, 4),
        "label": label[:_MAX_LABEL_CHARS],
        "action": action[:_MAX_ACTION_CHARS],
        "direction": direction,
        "expires_in_days": days,
    }


def parse_watch_block(text: str) -> list[dict[str, Any]]:
    """Extract validated conditions from every ``<watch>`` block in `text`.

    Accepts either ``{"conditions": [...]}`` or a bare ``[...]`` inside the block.
    Malformed JSON yields nothing; malformed ITEMS are dropped individually, so
    one bad row never costs the good rows beside it.
    """
    if not isinstance(text, str) or "<watch>" not in text.lower():
        return []

    out: list[dict[str, Any]] = []
    for body in _WATCH_BLOCK_RE.findall(text):
        body = body.strip()
        # Tolerate a model wrapping the JSON in a code fence inside the tag.
        if body.startswith("```"):
            lines = [ln for ln in body.split("\n")]
            body = "\n".join(lines[1:-1] if len(lines) > 2 else lines).strip()
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            log_to_component("server", "Watch", "Discarded a malformed <watch> block (invalid JSON).")
            continue

        items = data.get("conditions") if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue
        for item in items:
            cleaned = _clean_condition(item)
            if cleaned:
                out.append(cleaned)
    return out


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def _dedup_key(cond: dict[str, Any]) -> tuple:
    return (cond["symbol"], cond["metric"], cond["operator"], cond["threshold"])


def add_conditions(conditions: list[dict[str, Any]], source: str = "") -> dict[str, int]:
    """Store validated conditions, collapsing restatements of a live one.

    The advisor restates the same trigger every time it re-runs the same brief.
    A matching ACTIVE condition (same symbol/metric/operator/threshold) is
    refreshed in place — expiry extended, wording updated, `restated_count`
    bumped — rather than appended, so a daily precompute cannot turn one
    commitment into thirty rows (the ledger's restatement bug, one layer over).
    """
    if not conditions:
        return {"added": 0, "refreshed": 0}

    try:
        records = _load_all()
        by_key = {
            _dedup_key(r): r for r in records
            if r.get("status") == "active" and all(k in r for k in ("symbol", "metric", "operator", "threshold"))
        }
        now = datetime.now()
        added = refreshed = 0

        for cond in conditions[:_MAX_ACTIVE_PER_TURN]:
            expires_at = (now + timedelta(days=cond["expires_in_days"])).isoformat(timespec="seconds")
            existing = by_key.get(_dedup_key(cond))
            if existing is not None:
                existing["label"] = cond["label"]
                existing["action"] = cond["action"]
                existing["direction"] = cond["direction"]
                existing["expires_at"] = expires_at
                existing["last_restated"] = now.isoformat(timespec="seconds")
                existing["restated_count"] = existing.get("restated_count", 0) + 1
                refreshed += 1
                continue

            record = {
                "id": "wc_" + uuid.uuid4().hex[:10],
                "created_at": now.isoformat(timespec="seconds"),
                "source": source,
                "status": "active",
                "symbol": cond["symbol"],
                "metric": cond["metric"],
                "operator": cond["operator"],
                "threshold": cond["threshold"],
                "label": cond["label"],
                "action": cond["action"],
                "direction": cond["direction"],
                "expires_at": expires_at,
                "armed_at": None,
                "checked_at": None,
                "last_value": None,
                "restated_count": 0,
            }
            records.append(record)
            by_key[_dedup_key(cond)] = record
            added += 1

        if added or refreshed:
            _write_all(records)
        return {"added": added, "refreshed": refreshed}
    except Exception as e:
        log_to_component("server", "Watch", f"add_conditions failed: {e}", level=logging.WARNING)
        return {"added": 0, "refreshed": 0}


def capture_watch_conditions(text: str, source: str = "") -> dict[str, int]:
    """Parse a finished answer's side-channel and store what it committed to.

    The single entry point for producers (chat post-processing, the Today's
    Priority precompute, an auto-escalated catalyst scenario). Network-free.
    """
    conditions = parse_watch_block(text)
    if not conditions:
        # Two very different silences hide here, and conflating them is how this
        # engine ran dead for a full day unnoticed (2026-07-23): an answer with
        # NO trigger block at all (correct, common, silent) vs. one that emitted
        # a <watch> block whose every condition was discarded -- malformed JSON,
        # a prompt/parser drift, or an upstream sanitizer that ate the tags
        # before they reached here. The second is a failure and gets a WARNING,
        # so a producer can never again harvest nothing without leaving a trace.
        if isinstance(text, str) and "<watch>" in text.lower():
            log_to_component(
                "server", "Watch",
                f"A <watch> block from {source or 'unknown'} yielded 0 usable "
                "conditions (malformed, or the tags were stripped upstream before "
                "capture). The advisor committed to a trigger that is NOT armed.",
                level=logging.WARNING,
            )
        return {"added": 0, "refreshed": 0}
    result = add_conditions(conditions, source=source)
    if result["added"] or result["refreshed"]:
        log_to_component(
            "server", "Watch",
            f"Captured {result['added']} new / {result['refreshed']} restated condition(s) from {source or 'unknown'}.",
        )
    return result


def cancel_condition(condition_id: str) -> bool:
    """Retire one active condition (user dismissal). Returns True if it changed."""
    try:
        records = _load_all()
        for rec in records:
            if rec.get("id") == condition_id and rec.get("status") == "active":
                rec["status"] = "cancelled"
                rec["resolved_at"] = datetime.now().isoformat(timespec="seconds")
                return _write_all(records)
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_conditions(status: str | None = "active", limit: int = 100) -> list[dict[str, Any]]:
    """Newest-first conditions, optionally filtered by status. [] on any error."""
    records = _load_all()
    if status:
        records = [r for r in records if r.get("status") == status]
    records.reverse()
    return records[:limit] if limit and limit > 0 else records


def get_watch_summary() -> dict[str, Any]:
    """Counts by status + the active conditions, for the API and the trigger board."""
    records = _load_all()
    counts: dict[str, int] = {}
    for rec in records:
        st = rec.get("status", "unknown")
        counts[st] = counts.get(st, 0) + 1
    return {
        "counts": counts,
        "active": [r for r in records if r.get("status") == "active"],
        "recently_fired": [r for r in records if r.get("status") == "fired"][-10:],
    }


# ---------------------------------------------------------------------------
# Evaluation — the zero-LLM tick
# ---------------------------------------------------------------------------

def _satisfied(value: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == ">":
        return value > threshold
    return False


# Roadmap 5.8: how old a quote may be and still justify firing a trigger.
# `get_stock_data` caches for 1 hour while this tick runs every 30 minutes, so a
# read here is often a replay of an earlier fetch — ages alternate roughly 0 and
# 30 minutes. 45 admits that normal cached replay (the alert states its own
# as-of, so the user sees the vintage) while blocking a genuinely ancient or
# end-of-day print from being presented as a live crossing. Tightening this
# further requires a cache-bypassing fresh read on the alert path.
MAX_QUOTE_AGE_MINUTES = 45


def _default_price_fn(symbol: str) -> dict[str, float | None]:
    """One market read per symbol per tick, shaped for both supported metrics.

    Carries through the as-of stamp (Roadmap 5.8). `get_stock_data` is cached
    with a 1-hour TTL while this tick runs every 30 minutes, so a read here is
    routinely a replay of an earlier fetch — the stamp is the only way the
    evaluator can tell a live level from an hour-old one.
    """
    from tools.freshness import AS_OF_KEY
    from tools.market_data import get_stock_data

    data = get_stock_data(symbol)
    if not isinstance(data, dict) or "error" in data:
        return {"price": None, "pct_change": None}
    price = _num(data.get("current_price") or data.get("price"))
    pct = data.get("day_change_pct")
    return {
        "price": price,
        "pct_change": _num(pct) if pct is not None else None,
        AS_OF_KEY: data.get(AS_OF_KEY),
    }


def _is_stale_quote(quote: dict[str, Any], now: datetime) -> bool:
    """True when a quote carries a stamp proving it is older than the limit.

    An UNSTAMPED quote is not treated as stale — it is unverified, and the alert
    says so rather than claiming freshness. Blocking every unstamped quote would
    switch the engine off wholesale the moment a stamp went missing upstream,
    and a silent no-op is worse than an honestly-labelled one.
    """
    from tools.freshness import is_stale
    return is_stale(quote, MAX_QUOTE_AGE_MINUTES, now)


def _describe_quote(quote: dict[str, Any], now: datetime) -> str:
    from tools.freshness import describe
    return describe(quote, now)


def _fire_alert(rec: dict[str, Any], value: float) -> None:
    from tools.alerts import raise_alert

    unit = "%" if rec["metric"] == "pct_change" else ""
    prefix = "" if rec["metric"] == "pct_change" else "$"
    raise_alert(
        title=f"{rec['symbol']}: watch condition fired — {rec['label'][:110]}",
        message=(
            f"{rec['label']}\n\n"
            f"Trigger: {rec['symbol']} {rec['metric']} {rec['operator']} {prefix}{rec['threshold']:g}{unit} "
            f"— now {prefix}{value:g}{unit}.\n"
            f"Pre-committed action: {rec['action']}\n\n"
            f"Set {str(rec.get('created_at', ''))[:10]} from {rec.get('source') or 'an earlier answer'}"
            f" ({rec.get('direction', 'watch')}).\n"
            f"Quote {rec.get('fired_as_of') or 'as-of unverified'}."
        ),
        severity="warning",
        source="watch",
        dedup_key=f"watch:{rec['id']}",
        data={
            "condition_id": rec["id"],
            "symbol": rec["symbol"],
            "metric": rec["metric"],
            "operator": rec["operator"],
            "threshold": rec["threshold"],
            "value": value,
            "direction": rec.get("direction"),
            "action": rec.get("action"),
            "as_of": rec.get("fired_as_of"),
        },
    )


def evaluate_conditions(
    now: datetime | None = None,
    price_fn: Callable[[str], dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Re-check every active condition. Zero LLM calls; one market read per symbol.

    `price_fn` is injectable so the whole engine is testable offline.

    Semantics per record:
    - past `expires_at`            → status "expired", no alert.
    - no live value for its symbol → left active, `last_error` stamped.
    - satisfied on the FIRST check → status "void": the trigger was already true
      when it was written, so firing would be noise rather than news.
    - satisfied on a later check   → status "fired" + one alert into the inbox.
    """
    now = now or datetime.now()
    fetch = price_fn or _default_price_fn

    try:
        records = _load_all()
        active = [r for r in records if r.get("status") == "active"]
        if not active:
            return {"checked": 0, "fired": 0, "expired": 0, "voided": 0, "unavailable": 0, "stale": 0}

        # Expire first so a dead condition never costs a market read.
        live: list[dict[str, Any]] = []
        expired = 0
        for rec in active:
            exp = str(rec.get("expires_at") or "")
            try:
                is_expired = bool(exp) and datetime.fromisoformat(exp) < now
            except ValueError:
                is_expired = False
            if is_expired:
                rec["status"] = "expired"
                rec["resolved_at"] = now.isoformat(timespec="seconds")
                expired += 1
            else:
                live.append(rec)

        # ONE read per distinct symbol, shared by every condition on that symbol.
        values: dict[str, dict[str, float | None]] = {}
        for symbol in {r["symbol"] for r in live}:
            try:
                values[symbol] = fetch(symbol)
            except Exception as e:  # noqa: BLE001 — one bad symbol must not abort the tick
                log_to_component("server", "Watch", f"Price read failed for {symbol}: {e}", level=logging.WARNING)
                values[symbol] = {"price": None, "pct_change": None}

        fired = voided = unavailable = stale = 0
        for rec in live:
            quote = values.get(rec["symbol"]) or {}
            value = quote.get(rec["metric"])
            if value is None:
                rec["last_error"] = f"no {rec['metric']} available at {now.isoformat(timespec='seconds')}"
                unavailable += 1
                continue

            # Freshness gate (Roadmap 5.8). A quote too old to prove it is live
            # cannot justify firing a trigger — that is the stale-quote failure (an
            # end-of-day print sold as the live tape), and here it would page the
            # user unprompted. Skip the record ENTIRELY rather than just muting
            # the alert: advancing armed_at/last_value against a stale quote would
            # consume the crossing, and the real one would never fire.
            if _is_stale_quote(quote, now):
                rec["last_error"] = f"quote too old to act on ({_describe_quote(quote, now)})"
                stale += 1
                continue

            rec.pop("last_error", None)
            rec["quote_as_of"] = _describe_quote(quote, now)
            rec["checked_at"] = now.isoformat(timespec="seconds")
            rec["last_value"] = value
            hit = _satisfied(value, rec["operator"], rec["threshold"])

            if not rec.get("armed_at"):
                # First real evaluation: arm it, unless it was already true.
                if hit:
                    rec["status"] = "void"
                    rec["resolved_at"] = now.isoformat(timespec="seconds")
                    rec["void_reason"] = "already satisfied when armed"
                    voided += 1
                else:
                    rec["armed_at"] = now.isoformat(timespec="seconds")
                continue

            if hit:
                rec["fired_as_of"] = _describe_quote(quote, now)
                rec["status"] = "fired"
                rec["fired_at"] = now.isoformat(timespec="seconds")
                rec["fired_value"] = value
                rec["resolved_at"] = rec["fired_at"]
                fired += 1
                try:
                    _fire_alert(rec, value)
                except Exception as e:  # noqa: BLE001 — a delivery failure must not lose the state change
                    log_to_component("server", "Watch", f"Alert for {rec['id']} failed: {e}", level=logging.WARNING)

        _write_all(records)
        return {
            "checked": len(live),
            "fired": fired,
            "expired": expired,
            "voided": voided,
            "unavailable": unavailable,
            "stale": stale,
        }
    except Exception as e:
        log_to_component("server", "Watch", f"evaluate_conditions failed: {e}", level=logging.WARNING)
        return {"checked": 0, "fired": 0, "expired": 0, "voided": 0, "unavailable": 0, "stale": 0, "error": str(e)}
