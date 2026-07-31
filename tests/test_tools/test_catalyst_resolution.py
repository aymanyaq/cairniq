"""
1.3 — catalyst resolution (`tools/catalyst_resolution`).

The extractor's two thresholds (0.5 to survive the noise cut, 0.8 to spend an
Opus call) are authored constants that have never been tested against an outcome.
This module exists to test them, so its own honesty is what these tests guard.

Four things it must refuse, each with a way of failing that would look fine:

A non-prediction must never be scored. `direction_hint: "unclear"`, or a catalyst
naming only a sector, is not a claim — scoring those as misses manufactures a hit
rate out of statements nobody made, and the resulting number would look
respectably low rather than obviously wrong.

A missing quote must never read as a failed call. `unresolvable` is a gap in
evidence, and folding it into the denominator would penalise the extractor for a
delisting.

A small move must never be forced onto one side. If `inconclusive` were counted
as a miss, NOISE_BAND_PCT — an authored constant — would be setting the hit rate
rather than the calls.

And a rate must not be quoted from four observations. Same discipline as the
n>=20 and n>=30 gates elsewhere in this project.
"""
import json
from datetime import datetime

import pytest

from tools import catalyst_resolution as cr


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "catalyst_predictions.jsonl")


NOW = datetime(2026, 7, 29, 18, 0, 0)
LATER = datetime(2026, 9, 30, 18, 0, 0)   # past every horizon but structural


def _catalyst(cid="c1", direction="bullish", horizon="days", confidence=0.85,
              materiality="high", tickers=("AAPL",), headline="Big thing happens"):
    return {
        "id": cid, "headline": headline, "event_type": "earnings",
        "direction_hint": direction, "horizon": horizon,
        "confidence": confidence, "materiality": materiality,
        "portfolio_relevance": "held",
        "entities": {"tickers": list(tickers), "sectors": [], "commodities": []},
    }


def _prices(pct):
    """A price_fn that moves every symbol by `pct`."""
    return lambda sym, start, end: (100.0, 100.0 * (1 + pct / 100.0))


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------
def test_the_claim_is_recorded_at_the_time_it_was_made(store):
    cr.record_predictions([_catalyst()], escalated_ids=["c1"], now=NOW, path=store)
    rec = cr.read_predictions(store)[0]

    assert rec["direction_hint"] == "bullish"
    assert rec["horizon"] == "days"
    assert rec["confidence"] == 0.85
    assert rec["materiality"] == "high"
    assert rec["tickers"] == ["AAPL"]
    assert rec["escalated"] is True
    assert rec["recorded_on"] == "2026-07-29"
    assert rec["directional"] is True


def test_non_scoreable_catalysts_are_recorded_too(store):
    """The directional FRACTION is calibration evidence in its own right: an
    extractor mostly emitting "unclear" is not producing tradeable signal
    however good its hit rate on the remainder looks."""
    cr.record_predictions([
        _catalyst("a", direction="unclear"),
        _catalyst("b", direction="mixed"),
        _catalyst("c", direction="bullish", tickers=()),
        _catalyst("d", direction="bullish"),
    ], now=NOW, path=store)

    rows = cr.read_predictions(store)
    assert len(rows) == 4
    assert sum(r["directional"] for r in rows) == 1


def test_a_second_scan_the_same_day_records_nothing(store):
    """The proactive scan runs six-hourly; it must not inflate the denominator."""
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    second = cr.record_predictions([_catalyst()], now=NOW, path=store)
    assert second["recorded"] == 0
    assert len(cr.read_predictions(store)) == 1


def test_the_same_catalyst_on_a_later_day_is_a_new_prediction(store):
    """Novelty dedup has a 7-day horizon; a story re-entering later is a new
    claim about a new price, and collapsing them would score one outcome twice."""
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    cr.record_predictions([_catalyst()], now=datetime(2026, 8, 20, 9, 0), path=store)
    assert len(cr.read_predictions(store)) == 2


# ---------------------------------------------------------------------------
# Resolution — the refusals
# ---------------------------------------------------------------------------
def test_a_non_directional_catalyst_is_never_scored(store):
    cr.record_predictions([_catalyst(direction="unclear")], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(-20))

    rec = cr.read_predictions(store)[0]
    assert rec["outcome"] == "not_directional"
    assert rec["move_pct"] is None
    assert "not a prediction" in rec["resolution_note"]


def test_a_catalyst_with_no_ticker_is_never_scored(store):
    cr.record_predictions([_catalyst(direction="bearish", tickers=())], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(-20))
    assert cr.read_predictions(store)[0]["outcome"] == "not_directional"


def test_a_missing_quote_is_unresolvable_not_a_miss(store):
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=lambda *a: None)

    rec = cr.read_predictions(store)[0]
    assert rec["outcome"] == "unresolvable"
    assert "NOT a failed call" in rec["resolution_note"]


def test_a_move_inside_the_band_is_inconclusive(store):
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(0.4))

    rec = cr.read_predictions(store)[0]
    assert rec["outcome"] == "inconclusive"
    assert "neither came true nor failed" in rec["resolution_note"]


def test_an_unelapsed_horizon_stays_pending(store):
    cr.record_predictions([_catalyst(horizon="structural")], now=NOW, path=store)
    cr.resolve_all(now=datetime(2026, 8, 1), path=store, price_fn=_prices(30))
    assert cr.read_predictions(store)[0].get("outcome") in (None, "pending")


# ---------------------------------------------------------------------------
# Resolution — the scoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("direction,move,expected", [
    ("bullish", 8.0, "confirmed"),
    ("bullish", -8.0, "invalidated"),
    ("bearish", -8.0, "confirmed"),
    ("bearish", 8.0, "invalidated"),
])
def test_direction_is_scored_against_the_actual_move(store, direction, move, expected):
    cr.record_predictions([_catalyst(direction=direction)], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(move))
    assert cr.read_predictions(store)[0]["outcome"] == expected


def test_multiple_tickers_are_one_claim_not_several(store):
    """A catalyst naming three names made ONE claim about all of them."""
    cr.record_predictions([_catalyst(tickers=("AAPL", "MSFT", "NVDA"))],
                          now=NOW, path=store)
    moves = {"AAPL": 12.0, "MSFT": 6.0, "NVDA": -3.0}
    cr.resolve_all(now=LATER, path=store,
                   price_fn=lambda s, a, b: (100.0, 100.0 * (1 + moves[s] / 100)))

    rows = cr.read_predictions(store)
    assert len(rows) == 1, "three tickers must not become three scored calls"
    assert rows[0]["outcome"] == "confirmed"
    assert rows[0]["move_pct"] == pytest.approx(5.0)


def test_a_settled_outcome_is_never_rewritten(store):
    """A later price move must not rewrite a call that already resolved."""
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(10))
    assert cr.read_predictions(store)[0]["outcome"] == "confirmed"

    again = cr.resolve_all(now=LATER, path=store, price_fn=_prices(-40))
    assert again["resolved"] == 0
    assert cr.read_predictions(store)[0]["outcome"] == "confirmed"


def test_unresolvable_is_retried_because_it_is_not_a_conclusion(store):
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=lambda *a: None)
    assert cr.read_predictions(store)[0]["outcome"] == "unresolvable"

    cr.resolve_all(now=LATER, path=store, price_fn=_prices(9))
    assert cr.read_predictions(store)[0]["outcome"] == "confirmed"


def test_a_malformed_line_does_not_cost_the_ledger(store):
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    with open(store, "a", encoding="utf-8") as f:
        f.write("}{ not json\n")
    assert len(cr.read_predictions(store)) == 1


# ---------------------------------------------------------------------------
# The scoreboard
# ---------------------------------------------------------------------------
def _seed(store, n, direction="bullish", move=8.0, confidence=0.85, prefix="x"):
    cr.record_predictions(
        [_catalyst(f"{prefix}{i}", direction=direction, confidence=confidence)
         for i in range(n)], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(move), limit=1000)


def test_no_data_names_the_wiring_as_a_possible_cause(store):
    board = cr.scoreboard(store)
    assert board["status"] == "no_data"
    assert "emitter is not wired" in board["note"]


def test_a_rate_is_withheld_below_the_minimum(store):
    _seed(store, 4)
    board = cr.scoreboard(store)
    assert board["overall"]["scored"] == 4
    assert board["overall"]["hit_rate"] is None
    assert board["overall"]["reportable"] is False
    assert "below the 20 needed" in board["overall"]["note"]


def test_a_rate_appears_once_there_is_enough(store):
    _seed(store, cr.MIN_SCOREABLE)
    board = cr.scoreboard(store)
    assert board["overall"]["reportable"] is True
    assert board["overall"]["hit_rate"] == 1.0


def test_inconclusive_calls_are_out_of_the_denominator(store):
    """Otherwise NOISE_BAND_PCT — an authored constant — sets the hit rate."""
    _seed(store, 10, move=8.0, prefix="hit")
    _seed(store, 10, move=0.5, prefix="flat")

    overall = cr.scoreboard(store)["overall"]
    assert overall["counts"]["confirmed"] == 10
    assert overall["counts"]["inconclusive"] == 10
    assert overall["scored"] == 10, "inconclusive must not enter the denominator"


def test_unresolvable_calls_are_out_of_the_denominator(store):
    cr.record_predictions([_catalyst(f"g{i}") for i in range(5)], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=lambda *a: None, limit=1000)

    overall = cr.scoreboard(store)["overall"]
    assert overall["counts"]["unresolvable"] == 5
    assert overall["scored"] == 0


def test_non_directional_calls_are_out_of_every_rate_but_still_counted(store):
    cr.record_predictions([_catalyst(f"u{i}", direction="unclear") for i in range(8)],
                          now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(9), limit=1000)

    board = cr.scoreboard(store)
    assert board["predictions"] == 8
    assert board["directional"] == 0
    assert board["directional_pct"] == 0.0
    assert board["overall"]["total"] == 0, "non-directional rows must not enter the rate base"


def test_the_confidence_split_is_what_the_item_was_opened_for(store):
    """0.5 and 0.8 are authored; only this comparison can justify or move them."""
    _seed(store, 3, confidence=0.6, move=8.0, prefix="mid")
    _seed(store, 3, confidence=0.9, move=8.0, prefix="high")

    board = cr.scoreboard(store)
    assert board["by_confidence"]["cut_to_escalate"]["total"] == 3
    assert board["by_confidence"]["escalate_eligible"]["total"] == 3
    # Nothing should ever sit below the noise cut — threshold() drops it.
    assert board["by_confidence"]["below_cut"]["total"] == 0
    assert board["thresholds_under_test"]["escalate_min_confidence"] == 0.8


def test_escalated_calls_are_tallied_separately(store):
    """These are the ones that cost an Opus call, so they are the ones whose hit
    rate has to justify the spend."""
    cr.record_predictions([_catalyst("e1"), _catalyst("e2")],
                          escalated_ids=["e1"], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(9))
    assert cr.scoreboard(store)["escalated"]["total"] == 1


def test_horizon_words_map_to_the_spans_they_are_scored_over():
    assert cr.horizon_days("intraday") == 1
    assert cr.horizon_days("structural") == 90
    # An unrecognised word falls back rather than raising or scoring instantly.
    assert cr.horizon_days("eventually") == cr.HORIZON_DAYS["days"]


def test_the_scoreboard_never_reports_a_rate_it_did_not_compute(store):
    """A null rate beside real counts is the contract every reader depends on."""
    _seed(store, 3)
    board = cr.scoreboard(store)
    for block in (board["by_confidence"], board["by_materiality"], board["by_horizon"]):
        for name, tally in block.items():
            if not tally["reportable"]:
                assert tally["hit_rate"] is None, name
            assert isinstance(tally["total"], int)
    assert "counted but never scored" in board["note"]


def test_the_store_survives_a_round_trip_as_plain_jsonl(store):
    cr.record_predictions([_catalyst()], now=NOW, path=store)
    cr.resolve_all(now=LATER, path=store, price_fn=_prices(9))
    kinds = [json.loads(line)["kind"] for line in open(store) if line.strip()]
    assert kinds == ["prediction", "outcome"]


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------
def test_the_scoreboard_is_reachable_over_http():
    """A scoreboard nobody can read is a store with no consumer."""
    from fastapi.testclient import TestClient

    from server import app

    res = TestClient(app).get("/api/catalysts/scoreboard")
    assert res.status_code == 200
    assert res.json()["status"] in ("no_data", "ready")


def test_the_resolver_is_registered_as_a_scheduled_task():
    """1.3's emitter accrues nothing if the resolver never runs — and a store
    that fills but never resolves is exactly the shape 1.3 was opened about."""
    from tools.scheduler import SCHEDULED_TASKS

    tasks = {t[0]: t for t in SCHEDULED_TASKS}
    assert "catalyst_resolution" in tasks
    _, _, cooldown, timeout = tasks["catalyst_resolution"]
    assert cooldown == 86400
    assert timeout >= 60


def test_the_emitter_is_wired_into_the_catalyst_scan():
    """The claim is not recoverable after the fact: it is recorded during the
    scan or never. Asserted against the source because the scan needs a live
    news fetch to run."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "api" / "background.py"
    text = src.read_text(encoding="utf-8")
    assert "record_predictions" in text
    assert "from tools.catalyst_resolution import record_predictions" in text
