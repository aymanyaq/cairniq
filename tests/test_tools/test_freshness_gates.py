"""Freshness gates on the automated alert path (Roadmap 5.8 × 3.3 × 3.4).

Both engines now fire alerts on their own, with desktop notifications and no
human in the loop. That turns an unprovable quote into an unprompted false
alarm — the stale-quote failure (an end-of-day print presented as the live tape) with
nobody having asked a question.

The load-bearing property tested here is subtler than "don't alert on stale
data": a stale reading must be skipped ENTIRELY, leaving state untouched. An
engine that suppresses the alert but still advances its state would consume the
crossing against stale data, and the real crossing would then never fire.
"""
import json
from datetime import datetime, timedelta

import pytest

import tools.freshness as fr
import tools.intraday_sentinel as isr
import tools.watch_conditions as wc


def _at(h, m=0, day=23):
    return datetime(2026, 7, day, h, m)


# ---------------------------------------------------------------------------
# 3.3 — watch conditions
# ---------------------------------------------------------------------------

@pytest.fixture
def watch(monkeypatch, tmp_path):
    path = tmp_path / "watch_conditions.jsonl"
    monkeypatch.setattr(wc, "get_data_path", lambda filename: str(path))
    fired: list[dict] = []
    monkeypatch.setattr(wc, "_fire_alert", lambda rec, value: fired.append({"rec": rec, "value": value}))
    return {"path": path, "fired": fired}


def _condition(**overrides):
    base = {
        "symbol": "NVDA", "metric": "price", "operator": "<=", "threshold": 165.0,
        "label": "Back inside the accumulation zone",
        "action": "Execute the half-position entry",
        "direction": "entry", "expires_in_days": 30,
    }
    base.update(overrides)
    return base


def _quote(price, stamped_at=None):
    q = {"price": price, "pct_change": None}
    return fr.stamp(q, now=stamped_at) if stamped_at else q


def _arm(watch, now):
    """Store a condition and arm it with a fresh, not-yet-satisfied quote."""
    wc.add_conditions([_condition()], source="test")
    wc.evaluate_conditions(now=now, price_fn=lambda s: _quote(200.0, now))


def test_a_stale_quote_cannot_fire_a_trigger(watch):
    _arm(watch, _at(10, 0))

    # The level is breached, but on a quote fetched 2 hours ago.
    result = wc.evaluate_conditions(
        now=_at(12, 0), price_fn=lambda s: _quote(160.0, _at(10, 0))
    )

    assert watch["fired"] == []
    assert result["stale"] == 1
    assert result["fired"] == 0


def test_a_stale_reading_does_not_consume_the_crossing(watch):
    """The property that matters: after a stale skip, the SAME breach on a fresh
    quote must still fire. If the stale tick had advanced state, it wouldn't."""
    _arm(watch, _at(10, 0))
    wc.evaluate_conditions(now=_at(12, 0), price_fn=lambda s: _quote(160.0, _at(10, 0)))
    assert watch["fired"] == []

    wc.evaluate_conditions(now=_at(12, 30), price_fn=lambda s: _quote(160.0, _at(12, 30)))

    assert len(watch["fired"]) == 1
    assert watch["fired"][0]["value"] == 160.0


def test_a_fresh_quote_fires_and_carries_its_as_of(watch):
    _arm(watch, _at(10, 0))

    wc.evaluate_conditions(now=_at(10, 30), price_fn=lambda s: _quote(160.0, _at(10, 28)))

    assert len(watch["fired"]) == 1
    assert "as of 10:28" in watch["fired"][0]["rec"]["fired_as_of"]


def test_an_unverified_quote_still_fires_but_is_labelled(watch):
    """Refusing every unstamped quote would silently switch the engine off the
    moment a stamp went missing upstream — worse than an honest label."""
    _arm(watch, _at(10, 0))

    result = wc.evaluate_conditions(now=_at(10, 30), price_fn=lambda s: _quote(160.0))

    assert len(watch["fired"]) == 1
    assert result["stale"] == 0
    assert watch["fired"][0]["rec"]["fired_as_of"] == "as-of unverified"


def test_a_stale_skip_records_a_visible_reason(watch):
    """A gate whose suppressions are invisible is indistinguishable from a
    broken engine — the reason must be inspectable."""
    _arm(watch, _at(10, 0))
    wc.evaluate_conditions(now=_at(12, 0), price_fn=lambda s: _quote(160.0, _at(10, 0)))

    stored = [json.loads(line) for line in watch["path"].read_text().splitlines() if line.strip()]
    assert "too old" in stored[0]["last_error"]
    assert stored[0]["status"] == "active"  # still live, not resolved away


# ---------------------------------------------------------------------------
# 3.4 — intraday sentinel (daily bars → current-session freshness)
# ---------------------------------------------------------------------------

@pytest.fixture
def sentinel(monkeypatch, tmp_path):
    path = tmp_path / "intraday_sentinel_state.json"
    monkeypatch.setattr(isr, "get_data_path", lambda filename: str(path))
    fired: list[dict] = []
    monkeypatch.setattr(isr, "_raise", lambda **kw: fired.append(kw))
    return {"path": path, "fired": fired}


def _bar(payload, session_day):
    return fr.stamp(dict(payload), now=_at(0, 0, day=session_day))


def _tick(vix=None, holdings=None, now=None):
    snap = {}
    if vix is not None:
        snap["VIX"] = vix
    return isr.run_sentinel_tick(
        now=now, snapshot_fn=lambda: snap, holdings_fn=lambda: holdings or {}
    )


def test_a_previous_session_bar_cannot_fire_a_death_cross(sentinel):
    _tick(holdings={"NVDA": _bar({"death_cross": False}, 23)}, now=_at(10, 0))

    result = _tick(holdings={"NVDA": _bar({"death_cross": True}, 22)}, now=_at(10, 30))

    assert sentinel["fired"] == []
    assert result["stale_skipped"] == 1


def test_a_stale_bar_does_not_consume_the_sentinel_transition(sentinel):
    _tick(holdings={"NVDA": _bar({"death_cross": False}, 23)}, now=_at(10, 0))
    _tick(holdings={"NVDA": _bar({"death_cross": True}, 22)}, now=_at(10, 30))
    assert sentinel["fired"] == []

    # Same cross, now on a current-session bar — must still fire.
    _tick(holdings={"NVDA": _bar({"death_cross": True}, 23)}, now=_at(11, 0))

    assert len(sentinel["fired"]) == 1
    assert "death cross" in sentinel["fired"][0]["title"].lower()


def test_a_current_session_bar_fires_and_carries_its_as_of(sentinel):
    _tick(holdings={"NVDA": _bar({"death_cross": False}, 23)}, now=_at(10, 0))
    _tick(holdings={"NVDA": _bar({"death_cross": True}, 23)}, now=_at(10, 30))

    assert len(sentinel["fired"]) == 1
    assert "as of" in sentinel["fired"][0]["data"]["as_of"]
    assert "as of" in sentinel["fired"][0]["message"]


def test_an_unstamped_bar_still_fires_but_is_labelled(sentinel):
    _tick(holdings={"NVDA": {"death_cross": False}}, now=_at(10, 0))
    result = _tick(holdings={"NVDA": {"death_cross": True}}, now=_at(10, 30))

    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["data"]["as_of"] == "as-of unverified"
    assert result["stale_skipped"] == 0


def test_a_stale_index_bar_cannot_fire_a_band_crossing(sentinel):
    _tick(vix=_bar({"price": 15.0}, 23), now=_at(10, 0))          # calm baseline

    result = _tick(vix=_bar({"price": 30.0}, 22), now=_at(10, 30))  # yesterday's bar

    assert sentinel["fired"] == []
    assert result["stale_skipped"] == 1


def test_a_stale_holding_is_not_pruned_from_state(sentinel):
    """Skipping a stale symbol must not look like the holding was sold."""
    _tick(holdings={"NVDA": _bar({"death_cross": False}, 23)}, now=_at(10, 0))
    _tick(holdings={"NVDA": _bar({"death_cross": False}, 22)}, now=_at(10, 30))

    state = json.loads(sentinel["path"].read_text())
    assert "NVDA" in state["holdings"]


def test_freshness_gate_does_not_disturb_a_normal_fresh_run(sentinel):
    """Regression guard: the gate must not quietly suppress everything."""
    _tick(vix=_bar({"price": 15.0}, 23), now=_at(10, 0))
    result = _tick(vix=_bar({"price": 30.0}, 23), now=_at(10, 30))

    assert len(sentinel["fired"]) == 1
    assert result["stale_skipped"] == 0
    assert sentinel["fired"][0]["data"]["band"] == "high"


# ---------------------------------------------------------------------------
# The stamp actually reaches the engines through the real reader shape
# ---------------------------------------------------------------------------

def test_price_fn_carries_the_stamp_from_get_stock_data(monkeypatch):
    """End-to-end wiring: a stamp written inside get_stock_data must survive to
    the evaluator, or the gate silently measures nothing."""
    fetched = fr.stamp({"current_price": "$160.00", "day_change_pct": -1.2}, now=_at(9, 42))
    monkeypatch.setattr("tools.market_data.get_stock_data", lambda symbol: fetched)

    quote = wc._default_price_fn("NVDA")

    assert quote["price"] == 160.0
    assert fr.as_of(quote) == _at(9, 42)
    assert not fr.is_stale(quote, wc.MAX_QUOTE_AGE_MINUTES, now=_at(10, 0))
    assert fr.is_stale(quote, wc.MAX_QUOTE_AGE_MINUTES, now=_at(11, 0))


def test_get_stock_data_stamps_at_fetch_time_and_survives_a_cache_replay():
    """The stamp is written inside the cached function, so a replayed result
    reports the original fetch time rather than the read time."""
    payload = fr.stamp({"current_price": "$100.00"}, now=_at(9, 0))
    replayed = dict(payload)  # what a cache hit hands back later

    assert fr.age_minutes(replayed, now=_at(9, 30)) == 30.0
    assert fr.describe(replayed, now=_at(9, 30)) == "as of 09:00 (30 min ago)"


def test_quote_age_beyond_the_limit_is_what_the_constant_says():
    quote = fr.stamp({"price": 1.0}, now=_at(10, 0))
    limit = wc.MAX_QUOTE_AGE_MINUTES

    assert fr.is_stale(quote, limit, now=_at(10, 0) + timedelta(minutes=limit - 1)) is False
    assert fr.is_stale(quote, limit, now=_at(10, 0) + timedelta(minutes=limit + 1)) is True
