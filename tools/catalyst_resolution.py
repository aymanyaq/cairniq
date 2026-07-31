"""
1.3 — the catalyst resolution scoreboard, and the emitter it needs first.

The catalyst extractor produces a `direction_hint` and a `horizon` on every
catalyst, auto-escalates the ones above 0.8 confidence to an Opus scenario call,
and then never finds out whether any of it was right. `catalyst_log` records
``{ts, ids, escalated}`` — enough to dedup, and nothing that could ever be
scored. **The two thresholds this engine is tuned by (0.5 to survive the noise
cut, 0.8 to spend an Opus call) are authored constants that have never been
tested against an outcome.** This module is the missing half.

Two parts, in the only order they can be built:

1. **The emitter.** `record_predictions` writes what was claimed AT THE TIME —
   direction, horizon, confidence, materiality, tickers — because none of that
   is recoverable later. A scoreboard bolted onto the existing log would have to
   re-derive the prediction from a headline, which is guessing.
2. **The resolver.** Once a catalyst's horizon has elapsed, compare the named
   tickers' move against the direction that was claimed.

**Most catalysts are not scoreable, and saying so is the point.** A
`direction_hint` of "mixed" or "unclear" is not a prediction; a catalyst naming
only a sector is not attached to anything priceable. Scoring those as misses
would manufacture a hit rate out of statements nobody made, and scoring them as
hits would be worse. They resolve to `not_directional` and are excluded from
every rate this module reports — while still being COUNTED, because "what
fraction of our catalysts are even directional" is itself the calibration
evidence 1.3 was asked for.

**A small move is an outcome, not a failure.** A bullish call on a name that then
moves +0.3% did not come true; it also did not fail. `inconclusive` is a real
third state and it is reported separately, because folding it into either side
would let the noise band decide the hit rate.

**The store is global, not per-profile.** A catalyst's RELEVANCE is per-holder,
but "this event implied bullish for AAPL over days" is a claim about the world.
It is also the only way the corpus accrues fast enough to calibrate anything —
this project has several analytics blocked on n≥20 that would be blocked far
longer if split per profile.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from agent.logger import log_to_component
from tools.exception_logger import log_exceptions

_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "user_data", "catalyst_predictions.jsonl"
)

# AUTHORED CONSTANTS (2.7). Neither is measured — they are the reading this
# module gives to the extractor's own vocabulary, and the scoreboard exists
# partly to tell us whether they are wrong.
#
# The extractor emits a horizon WORD; scoring needs a number of days. These are
# the spans those words are taken to mean.
HORIZON_DAYS: dict[str, int] = {
    "intraday": 1,
    "days": 5,
    "weeks": 21,
    "structural": 90,
}
_DEFAULT_HORIZON = "days"

# How far a name must move before the move counts as evidence in either
# direction. Below it the call is `inconclusive` rather than confirmed or
# invalidated: a ±0.4% drift is not a catalyst playing out, and letting it score
# would make the hit rate a function of this constant rather than of the calls.
NOISE_BAND_PCT = 2.0

# Confidence buckets, chosen to straddle the two thresholds under test rather
# than to be evenly sized. `below_cut` should be EMPTY — threshold() drops
# anything under MIN_CONFIDENCE — so a non-zero count there is a finding about
# the pipeline, not about the calls.
CONFIDENCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("below_cut", 0.0, 0.5),
    ("cut_to_escalate", 0.5, 0.8),
    ("escalate_eligible", 0.8, 1.0001),
)

# Below this many SCORED calls, a rate is not reported. Same discipline as the
# 1.4 (n>=20) and 3.8 (n>=30) gates: a hit rate over four calls is a story.
MIN_SCOREABLE = 20

OUTCOMES = ("pending", "confirmed", "invalidated", "inconclusive",
            "not_directional", "unresolvable")

_DIRECTIONAL = {"bullish", "bearish"}


def store_path() -> str:
    return os.path.abspath(_STORE_PATH)


def horizon_days(horizon: str) -> int:
    return HORIZON_DAYS.get(str(horizon or "").strip().lower(),
                            HORIZON_DAYS[_DEFAULT_HORIZON])


def prediction_key(catalyst_id: str, recorded_on: str) -> str:
    """One prediction = one catalyst on one day.

    Dated, because the extractor's novelty dedup has a 7-day horizon: the same
    story re-entering three weeks later is a NEW claim about a new price, and
    collapsing the two would score one outcome twice.
    """
    return f"{catalyst_id}:{recorded_on}"


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
@log_exceptions()
def record_predictions(catalysts: list[dict[str, Any]],
                       escalated_ids: list[str] | None = None,
                       now: datetime | None = None,
                       path: str | None = None) -> dict[str, Any]:
    """Write what each catalyst CLAIMED, at the moment it claimed it.

    Records every catalyst, including the ones that can never be scored. The
    directional fraction is calibration evidence in its own right, and a store
    that quietly held only the scoreable ones would report a suspiciously
    complete-looking corpus.

    Idempotent per (catalyst id, day): a second scan on the same day re-records
    nothing, so the six-hourly proactive scan cannot inflate the denominator.
    """
    now = now or datetime.now()
    path = path or store_path()
    today = now.date().isoformat()
    escalated = set(escalated_ids or [])

    existing = {r["key"] for r in _read_records(path) if r.get("kind") == "prediction"}
    written = 0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for c in catalysts or []:
            cid = c.get("id")
            if not cid:
                continue
            key = prediction_key(cid, today)
            if key in existing:
                continue
            existing.add(key)
            direction = str(c.get("direction_hint") or "unclear").strip().lower()
            tickers = [str(t).upper().strip()
                       for t in (c.get("entities") or {}).get("tickers", []) if str(t).strip()]
            f.write(json.dumps({
                "kind": "prediction",
                "key": key,
                "catalyst_id": cid,
                "recorded_on": today,
                "recorded_at": now.isoformat(timespec="seconds"),
                "headline": str(c.get("headline") or "")[:300],
                "event_type": c.get("event_type"),
                "direction_hint": direction,
                "horizon": str(c.get("horizon") or _DEFAULT_HORIZON).strip().lower(),
                "confidence": c.get("confidence"),
                "materiality": c.get("materiality"),
                "portfolio_relevance": c.get("portfolio_relevance"),
                "tickers": tickers,
                "escalated": cid in escalated,
                # Whether this could EVER be scored, decided at write time from
                # what was claimed — not at read time from what we managed to
                # fetch. The two are different failures and must stay apart.
                "directional": bool(direction in _DIRECTIONAL and tickers),
            }) + "\n")
            written += 1

    if written:
        log_to_component("tools", "catalyst_resolution",
                         f"recorded {written} catalyst prediction(s)")
    return {"recorded": written, "date": today, "store": path}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def _read_records(path: str | None = None) -> list[dict[str, Any]]:
    path = path or store_path()
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # one bad line must not cost the ledger
    return out


@log_exceptions()
def read_predictions(path: str | None = None) -> list[dict[str, Any]]:
    """Every prediction with its latest outcome merged in, oldest first."""
    records = _read_records(path)
    predictions = {r["key"]: dict(r) for r in records if r.get("kind") == "prediction"}
    for r in records:
        if r.get("kind") == "outcome" and r.get("key") in predictions:
            predictions[r["key"]].update({
                "outcome": r.get("outcome"),
                "resolved_at": r.get("resolved_at"),
                "move_pct": r.get("move_pct"),
                "resolution_note": r.get("note"),
                "per_ticker": r.get("per_ticker"),
            })
    return sorted(predictions.values(), key=lambda r: (r.get("recorded_on", ""), r.get("key", "")))


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------
def _fetch_window(symbol: str, start: str, end: str) -> tuple[float, float] | None:
    """(first close, last close) over [start, end], or None if unavailable.

    None means UNKNOWN, never "flat". A missing quote resolves the prediction to
    `unresolvable`, which is excluded from the hit rate — a delisted or
    unfetchable name must not read as a failed call.
    """
    try:
        import yfinance as yf

        from tools.market_data import _safe_yf_call

        hist = _safe_yf_call(
            lambda: yf.Ticker(symbol).history(start=start, end=end, timeout=20)
        )
        if hist is None or hist.empty or "Close" not in hist:
            return None
        closes = [float(x) for x in hist["Close"].tolist() if x == x]
        if len(closes) < 2 or closes[0] <= 0:
            return None
        return closes[0], closes[-1]
    except Exception:  # noqa: BLE001 — an unfetchable name is unresolvable, not a miss
        return None


def resolve_prediction(rec: dict[str, Any], now: datetime | None = None,
                       price_fn: Callable[[str, str, str], tuple[float, float] | None] | None = None,
                       ) -> dict[str, Any]:
    """Score one prediction against what the named tickers actually did.

    Returns `{outcome, move_pct, note, per_ticker}`. Never raises, and never
    guesses: every path that cannot produce evidence returns a named non-outcome
    rather than a default of "invalidated".
    """
    now = now or datetime.now()
    price_fn = price_fn or _fetch_window

    if not rec.get("directional"):
        return {
            "outcome": "not_directional",
            "move_pct": None,
            "note": (
                f"direction_hint was {rec.get('direction_hint')!r}"
                + ("" if rec.get("tickers") else " and no ticker was named")
                + ". This is not a prediction, so it is counted but never scored."
            ),
            "per_ticker": {},
        }

    start = date.fromisoformat(rec["recorded_on"])
    window = horizon_days(rec.get("horizon"))
    ends = start + timedelta(days=window)
    if now.date() <= ends:
        return {
            "outcome": "pending",
            "move_pct": None,
            "note": f"horizon {rec.get('horizon')} ({window}d) ends {ends.isoformat()}.",
            "per_ticker": {},
        }

    per_ticker: dict[str, Any] = {}
    moves: list[float] = []
    for symbol in rec.get("tickers", []):
        pair = price_fn(symbol, start.isoformat(), (ends + timedelta(days=1)).isoformat())
        if not pair:
            per_ticker[symbol] = None
            continue
        first, last = pair
        pct = ((last - first) / first) * 100.0
        per_ticker[symbol] = round(pct, 4)
        moves.append(pct)

    if not moves:
        return {
            "outcome": "unresolvable",
            "move_pct": None,
            "note": ("No price data for any named ticker over the window. This is a "
                     "gap in evidence, NOT a failed call, and it is excluded from "
                     "the hit rate."),
            "per_ticker": per_ticker,
        }

    # Equal-weighted across the named tickers. A catalyst naming three names
    # made one claim about all of them, not three separate calls.
    move = sum(moves) / len(moves)
    bullish = rec.get("direction_hint") == "bullish"

    if abs(move) < NOISE_BAND_PCT:
        outcome = "inconclusive"
        note = (f"moved {move:+.2f}%, inside the ±{NOISE_BAND_PCT}% band. The call "
                "neither came true nor failed.")
    elif (move > 0) == bullish:
        outcome = "confirmed"
        note = f"moved {move:+.2f}% over {window}d, in the direction claimed."
    else:
        outcome = "invalidated"
        note = f"moved {move:+.2f}% over {window}d, against the direction claimed."

    return {"outcome": outcome, "move_pct": round(move, 4), "note": note,
            "per_ticker": per_ticker}


@log_exceptions()
def resolve_all(now: datetime | None = None, path: str | None = None,
                price_fn: Callable[..., Any] | None = None,
                limit: int = 200) -> dict[str, Any]:
    """Resolve every prediction whose horizon has elapsed and that has no outcome.

    Re-resolves nothing: an outcome is written once and then left alone, so a
    later price move cannot rewrite a call that already resolved. `pending` and
    `unresolvable` ARE retried, because both are statements that evidence was not
    available yet rather than conclusions.
    """
    now = now or datetime.now()
    path = path or store_path()
    settled = {"confirmed", "invalidated", "inconclusive", "not_directional"}

    todo = [r for r in read_predictions(path)
            if r.get("outcome") not in settled][:limit]
    if not todo:
        return {"resolved": 0, "checked": 0, "store": path}

    written = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in todo:
            result = resolve_prediction(rec, now=now, price_fn=price_fn)
            if result["outcome"] == "pending":
                continue
            f.write(json.dumps({
                "kind": "outcome",
                "key": rec["key"],
                "outcome": result["outcome"],
                "move_pct": result["move_pct"],
                "note": result["note"],
                "per_ticker": result["per_ticker"],
                "resolved_at": now.isoformat(timespec="seconds"),
            }) + "\n")
            written += 1

    if written:
        log_to_component("tools", "catalyst_resolution",
                         f"resolved {written} of {len(todo)} open prediction(s)")
    return {"resolved": written, "checked": len(todo), "store": path}


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------
def _bucket(confidence: Any) -> str:
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "unknown"
    for name, low, high in CONFIDENCE_BUCKETS:
        if low <= c < high:
            return name
    return "unknown"


def _tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and — only if there are enough of them — a hit rate.

    `hit_rate` is None below MIN_SCOREABLE and the reason is carried beside it,
    so a caller cannot read a small-sample rate without also reading that it is
    one. Denominator is confirmed + invalidated: `inconclusive` is excluded
    because a call that neither happened nor failed is not evidence either way,
    and folding it in would let NOISE_BAND_PCT set the score.
    """
    counts = {o: 0 for o in OUTCOMES}
    for r in rows:
        counts[r.get("outcome") or "pending"] = counts.get(r.get("outcome") or "pending", 0) + 1

    scored = counts["confirmed"] + counts["invalidated"]
    enough = scored >= MIN_SCOREABLE
    return {
        "total": len(rows),
        "counts": counts,
        "scored": scored,
        "hit_rate": round(counts["confirmed"] / scored, 4) if (enough and scored) else None,
        "reportable": enough,
        "note": ("" if enough else
                 f"{scored} scored call(s) — below the {MIN_SCOREABLE} needed to quote a "
                 "rate. Counts are real; a rate over this many would be a story."),
    }


@log_exceptions()
def scoreboard(path: str | None = None) -> dict[str, Any]:
    """Resolution outcomes overall and by the dimensions 1.3 asked to calibrate.

    The confidence breakdown is the one the item was opened for: MIN_CONFIDENCE
    (0.5) and ESCALATE_MIN_CONFIDENCE (0.8) are authored, and the only thing that
    can justify or move them is whether `escalate_eligible` actually hits harder
    than `cut_to_escalate`.
    """
    rows = read_predictions(path)
    if not rows:
        return {
            "status": "no_data",
            "note": ("No catalyst predictions have been recorded. The emitter runs "
                     "inside the catalyst scan — if scans are running and this is "
                     "empty, the emitter is not wired, which is a different problem "
                     "from a quiet news week."),
            "overall": _tally([]),
        }

    directional = [r for r in rows if r.get("directional")]

    by_confidence = {name: _tally([r for r in directional if _bucket(r.get("confidence")) == name])
                     for name, _, _ in CONFIDENCE_BUCKETS}
    by_materiality = {m: _tally([r for r in directional if r.get("materiality") == m])
                      for m in ("high", "medium", "low")}
    by_horizon = {h: _tally([r for r in directional if r.get("horizon") == h])
                  for h in HORIZON_DAYS}

    overall = _tally(directional)
    escalated = _tally([r for r in directional if r.get("escalated")])

    return {
        "status": "ready",
        "predictions": len(rows),
        # The directional fraction IS calibration evidence: an extractor whose
        # output is mostly "unclear" is not producing tradeable signal however
        # good its hit rate on the remainder looks.
        "directional": len(directional),
        "directional_pct": round(100.0 * len(directional) / len(rows), 2),
        "overall": overall,
        "by_confidence": by_confidence,
        "by_materiality": by_materiality,
        "by_horizon": by_horizon,
        "escalated": escalated,
        "thresholds_under_test": {
            "min_confidence": 0.5,
            "escalate_min_confidence": 0.8,
            "noise_band_pct": NOISE_BAND_PCT,
            "min_scoreable": MIN_SCOREABLE,
        },
        "note": (
            "Hit rates count confirmed / (confirmed + invalidated). Non-directional "
            "catalysts and unresolvable ones are counted but never scored. A bucket "
            "with `reportable: false` has a real count and no rate."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(scoreboard(), indent=2, default=str))
