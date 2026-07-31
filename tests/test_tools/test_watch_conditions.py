"""Advisor-authored watch conditions (Advisor Roadmap Theme 3.3).

The engine's whole value is that it fires on a level the ADVISOR committed to —
so the tests are weighted toward everything that must NOT happen: no condition
invented from a malformed block, no alert on a trigger that was already true
when written, no second alert once one fired, no live row evicted by the cap.
"""
import json
from datetime import datetime, timedelta

import pytest

import tools.watch_conditions as wc


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Isolated per-profile store + captured alerts."""
    path = tmp_path / "watch_conditions.jsonl"
    monkeypatch.setattr(wc, "get_data_path", lambda filename: str(path))
    fired: list[dict] = []
    monkeypatch.setattr(wc, "_fire_alert", lambda rec, value: fired.append({"rec": rec, "value": value}))
    return {"path": path, "fired": fired}


def _block(*conditions):
    return "Prose above.\n\n<watch>\n" + json.dumps({"conditions": list(conditions)}) + "\n</watch>"


def _cond(**overrides):
    base = {
        "symbol": "NVDA", "metric": "price", "operator": "<=", "threshold": 165.0,
        "label": "Back inside the accumulation zone",
        "action": "Execute the half-position entry",
        "direction": "entry", "expires_in_days": 30,
    }
    base.update(overrides)
    return base


def _prices(mapping):
    """price_fn over a {symbol: {metric: value}} map; unknown symbols quote nothing."""
    return lambda symbol: mapping.get(symbol, {"price": None, "pct_change": None})


# ---------------------------------------------------------------------------
# Parsing — strict, and silent about what it drops
# ---------------------------------------------------------------------------

def test_parses_a_well_formed_block():
    parsed = wc.parse_watch_block(_block(_cond()))

    assert len(parsed) == 1
    assert parsed[0]["symbol"] == "NVDA"
    assert parsed[0]["threshold"] == 165.0
    assert parsed[0]["direction"] == "entry"


def test_accepts_index_tsx_and_crypto_symbols():
    parsed = wc.parse_watch_block(_block(
        _cond(symbol="^VIX", threshold=28, operator=">="),
        _cond(symbol="shop.to", threshold=90),
        _cond(symbol="BTC-USD", threshold=60000),
    ))

    assert [c["symbol"] for c in parsed] == ["^VIX", "SHOP.TO", "BTC-USD"]


@pytest.mark.parametrize("bad", [
    {"metric": "rsi"},                    # a metric the engine cannot evaluate
    {"operator": "crosses"},              # an operator it cannot evaluate
    {"threshold": "somewhere near 160"},  # not a level
    {"threshold": 0},                     # price parse artifact
    {"symbol": ""},                       # no instrument
    {"symbol": "the whole energy sector"},
    {"action": ""},                       # a trigger with no committed action
    {"label": ""},
])
def test_drops_conditions_it_cannot_fully_understand(bad):
    """Guessing a level would commit the user to a trade trigger nobody wrote."""
    assert wc.parse_watch_block(_block(_cond(**bad))) == []


def test_one_bad_row_does_not_cost_the_good_rows():
    parsed = wc.parse_watch_block(_block(_cond(), _cond(symbol="AAPL", metric="rsi"), _cond(symbol="MSFT")))

    assert [c["symbol"] for c in parsed] == ["NVDA", "MSFT"]


def test_malformed_json_yields_nothing():
    assert wc.parse_watch_block("<watch>\n{not json at all}\n</watch>") == []


def test_no_block_yields_nothing():
    assert wc.parse_watch_block("### TODAY'S PRIORITY\nDO NOTHING — everything inside bands.") == []


def test_expiry_is_clamped_to_the_supported_window():
    parsed = wc.parse_watch_block(_block(_cond(expires_in_days=9999), _cond(symbol="AMD", expires_in_days=0)))

    assert [c["expires_in_days"] for c in parsed] == [wc._MAX_EXPIRY_DAYS, wc._MIN_EXPIRY_DAYS]


# ---------------------------------------------------------------------------
# Visible-text stripping — the JSON must never reach the chat
# ---------------------------------------------------------------------------

def test_strips_a_complete_block():
    visible = wc.strip_watch_blocks("Answer text.\n<watch>{\"conditions\": []}</watch>\ntail")

    assert "watch" not in visible
    assert "conditions" not in visible
    assert "Answer text." in visible and "tail" in visible


def test_strips_a_half_streamed_block():
    """Mid-stream the closing tag has not arrived yet — the JSON must still be hidden."""
    visible = wc.strip_watch_blocks('Answer text.\n<watch>\n{"conditions": [{"symbol": "NV')

    assert visible.rstrip() == "Answer text."


def test_strips_a_partial_opening_tag():
    assert wc.strip_watch_blocks("Answer text.\n<wat").rstrip() == "Answer text."


def test_leaves_ordinary_text_and_comparisons_alone():
    text = "Trim if P/E < 20 and the 40-week MA breaks."

    assert wc.strip_watch_blocks(text) == text


def test_strip_is_type_safe():
    assert wc.strip_watch_blocks(None) is None
    assert wc.strip_watch_blocks(42) == 42


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def test_capture_stores_conditions_as_active(store):
    result = wc.capture_watch_conditions(_block(_cond()), source="priority")

    assert result == {"added": 1, "refreshed": 0}
    active = wc.get_conditions()
    assert len(active) == 1
    assert active[0]["status"] == "active"
    assert active[0]["source"] == "priority"
    assert active[0]["armed_at"] is None


def test_restating_the_same_trigger_refreshes_instead_of_duplicating(store):
    """The morning precompute restates its trigger board daily; 30 days of that
    must not become 30 rows (the recommendation ledger's bug, one layer over)."""
    wc.capture_watch_conditions(_block(_cond()), source="priority")
    wc.capture_watch_conditions(_block(_cond(label="Restated with fresher wording")), source="priority")
    wc.capture_watch_conditions(_block(_cond(label="And again")), source="priority")

    active = wc.get_conditions()
    assert len(active) == 1
    assert active[0]["restated_count"] == 2
    assert active[0]["label"] == "And again"


def test_a_different_level_on_the_same_symbol_is_a_new_condition(store):
    wc.capture_watch_conditions(_block(_cond(threshold=165.0)), source="priority")
    wc.capture_watch_conditions(_block(_cond(threshold=150.0)), source="priority")

    assert len(wc.get_conditions()) == 2


def test_one_turn_cannot_flood_the_store(store):
    many = [_cond(symbol=f"SYM{i}", threshold=10 + i) for i in range(20)]
    wc.capture_watch_conditions(_block(*many), source="chat")

    assert len(wc.get_conditions()) == wc._MAX_ACTIVE_PER_TURN


def test_trim_never_evicts_a_live_condition(store):
    """Active rows are the point of the store; only resolved ones may be capped."""
    records = [
        {"id": f"old{i}", "status": "fired", "symbol": "X", "metric": "price",
         "operator": "<=", "threshold": 1.0}
        for i in range(wc._MAX_RECORDS + 50)
    ]
    records += [
        {"id": f"live{i}", "status": "active", "symbol": "NVDA", "metric": "price",
         "operator": "<=", "threshold": float(i)}
        for i in range(5)
    ]

    wc._write_all(records)

    assert len(wc.get_conditions(status="active")) == 5
    assert len(wc._load_all()) <= wc._MAX_RECORDS


def test_cancel_retires_an_active_condition(store):
    wc.capture_watch_conditions(_block(_cond()), source="chat")
    cid = wc.get_conditions()[0]["id"]

    assert wc.cancel_condition(cid) is True
    assert wc.get_conditions(status="active") == []
    assert wc.get_conditions(status="cancelled")[0]["id"] == cid


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_first_check_arms_without_firing(store):
    wc.capture_watch_conditions(_block(_cond(threshold=165.0)), source="priority")

    result = wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 180.0}}))

    assert result["fired"] == 0
    rec = wc.get_conditions()[0]
    assert rec["armed_at"] is not None
    assert rec["last_value"] == 180.0
    assert store["fired"] == []


def test_fires_once_the_level_is_crossed(store):
    wc.capture_watch_conditions(_block(_cond(threshold=165.0)), source="priority")
    wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 180.0}}))

    result = wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 162.5}}))

    assert result["fired"] == 1
    assert store["fired"][0]["value"] == 162.5
    rec = wc.get_conditions(status="fired")[0]
    assert rec["fired_value"] == 162.5


def test_a_fired_condition_never_fires_again(store):
    """Firing is terminal, so a price oscillating around the level cannot spam."""
    wc.capture_watch_conditions(_block(_cond(threshold=165.0)), source="priority")
    wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 180.0}}))
    wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 162.5}}))

    for price in (170.0, 160.0, 171.0, 159.0):
        wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": price}}))

    assert len(store["fired"]) == 1


def test_a_condition_already_true_when_written_is_voided_not_fired(store):
    """The advisor wrote a trigger for something that had already happened —
    usually a level read off the wrong side. Alerting on it would be noise."""
    wc.capture_watch_conditions(_block(_cond(threshold=165.0)), source="priority")

    result = wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 140.0}}))

    assert result["voided"] == 1 and result["fired"] == 0
    assert store["fired"] == []
    assert wc.get_conditions(status="void")[0]["void_reason"] == "already satisfied when armed"


def test_missing_quote_leaves_the_condition_active(store):
    """A data gap must never resolve a commitment — it stays pending, and says why."""
    wc.capture_watch_conditions(_block(_cond()), source="priority")

    result = wc.evaluate_conditions(price_fn=_prices({}))

    assert result["unavailable"] == 1 and result["fired"] == 0
    rec = wc.get_conditions(status="active")[0]
    assert rec["armed_at"] is None
    assert "no price available" in rec["last_error"]


def test_expired_conditions_resolve_without_firing(store):
    wc.capture_watch_conditions(_block(_cond(expires_in_days=1)), source="priority")

    result = wc.evaluate_conditions(
        now=datetime.now() + timedelta(days=2),
        price_fn=_prices({"NVDA": {"price": 100.0}}),   # deep through the level
    )

    assert result["expired"] == 1 and result["fired"] == 0
    assert store["fired"] == []


def test_one_price_read_per_symbol_regardless_of_condition_count(store):
    """Three levels on one name is one quote, not three — the tick runs every
    30 minutes across every profile."""
    wc.capture_watch_conditions(_block(
        _cond(threshold=165.0), _cond(threshold=150.0), _cond(threshold=140.0),
        _cond(symbol="AAPL", threshold=200.0),
    ), source="priority")

    calls: list[str] = []

    def counting_fn(symbol):
        calls.append(symbol)
        return {"price": 300.0, "pct_change": 1.0}

    wc.evaluate_conditions(price_fn=counting_fn)

    assert sorted(calls) == ["AAPL", "NVDA"]


def test_pct_change_metric_fires_on_the_daily_move(store):
    wc.capture_watch_conditions(_block(_cond(
        symbol="AMD", metric="pct_change", operator="<=", threshold=-5.0,
        label="AMD gaps down hard", action="Do not average down; re-check the thesis",
    )), source="chat")
    wc.evaluate_conditions(price_fn=_prices({"AMD": {"price": 100.0, "pct_change": -1.0}}))

    result = wc.evaluate_conditions(price_fn=_prices({"AMD": {"price": 92.0, "pct_change": -8.0}}))

    assert result["fired"] == 1
    assert store["fired"][0]["value"] == -8.0


def test_evaluating_an_empty_store_is_free(store):
    called: list[str] = []
    wc.evaluate_conditions(price_fn=lambda s: called.append(s) or {"price": 1.0})

    assert called == []


def test_alert_carries_the_committed_action(monkeypatch, tmp_path):
    """The alert must restate what the advisor said to DO — a fired trigger with
    no instruction is the notification this channel exists to replace."""
    path = tmp_path / "watch_conditions.jsonl"
    monkeypatch.setattr(wc, "get_data_path", lambda filename: str(path))
    raised: list[dict] = []
    monkeypatch.setattr("tools.alerts.raise_alert", lambda **kw: raised.append(kw))

    wc.capture_watch_conditions(_block(_cond(action="Execute the half-position entry")), source="priority")
    wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 180.0}}))
    wc.evaluate_conditions(price_fn=_prices({"NVDA": {"price": 160.0}}))

    assert len(raised) == 1
    assert "Execute the half-position entry" in raised[0]["message"]
    assert raised[0]["severity"] == "warning"
    assert raised[0]["data"]["value"] == 160.0
