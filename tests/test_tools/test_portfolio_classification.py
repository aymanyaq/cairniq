"""
4.10a's classification half (`tools/portfolio_classification`).

The module's whole value is a set of refusals, so that is what is tested.

It never infers. There is no code path that reads a delta and decides what
caused it, and the test that matters most here is the negative one: a change
with no human record comes back `unclassified`, whatever its shape. A "cash down
and shares up on the same day is a buy" heuristic would pass every other test in
this file while inventing history, which is the failure 4.10a was written around.

It refuses an unknown cause rather than storing it, because a free-text cause is
invisible to `is_external_flow` — and a flow the TWR engine cannot recognise is
strictly worse than one that is openly unclassified.

It refuses to carry an answer onto numbers it was not given. Snapshots are
rewritten wholesale by `_write_all`, so a corrected import can change a delta
while its date, account and symbol stay put. The fingerprint catches that and
reverts to unclassified WITH an explanation, rather than either silently
re-pointing the old answer or silently erasing that anyone answered.

And it refuses to report partial flows as flows. `complete` is strict: a TWR
computed over three known deposits while a fourth sits unstated is not
approximately right, it is wrong by an unknown amount in an unknown direction.
"""
import json

import pytest

from tools import portfolio_classification as pc


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the classification store at a temp file for the whole test."""
    path = tmp_path / "position_classifications.jsonl"
    monkeypatch.setattr(pc, "store_path", lambda: str(path))
    return path


def _change(symbol="VOO", account="TFSA", prior=100.0, current=112.5,
            prior_date="2026-07-28", current_date="2026-07-29", **kw):
    delta = (current or 0.0) - (prior or 0.0)
    base = {
        "kind": "quantity_increase" if delta > 0 else "quantity_decrease",
        "account": account, "symbol": symbol, "currency": "CAD", "is_cash": False,
        "prior_shares": prior, "current_shares": current, "delta": delta,
        "prior_date": prior_date, "current_date": current_date,
        "cause": "unclassified", "spans_gap": False, "gap_days": 1,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# It does not infer
# ---------------------------------------------------------------------------
def test_nothing_is_classified_without_a_human_record(store):
    """The load-bearing negative. No shape of delta produces a cause."""
    changes = [
        _change("VOO", prior=100.0, current=112.5),                    # shares up
        _change("CAD", prior=5000.0, current=1200.0, is_cash=True),    # cash down
        _change("KO", prior=40.0, current=None),                     # position gone
        _change("NVDA", prior=None, current=10.0),                     # position new
    ]
    for c in pc.apply_classifications(changes):
        assert c["cause"] == pc.UNCLASSIFIED
        assert c["classified"] is False
        assert c["is_external_flow"] is False


def test_a_cash_decrease_beside_a_share_increase_is_still_not_a_trade(store):
    """The single most tempting inference, named explicitly so nobody adds it."""
    same_day = [
        _change("CAD", prior=10_000.0, current=4_000.0, is_cash=True),
        _change("VOO", prior=0.0, current=60.0),
    ]
    assert all(not c["classified"] for c in pc.apply_classifications(same_day))
    assert pc.flow_summary(same_day)["complete"] is False


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_a_stated_cause_is_applied_and_attributed(store):
    change = _change()
    assert pc.classify_change(change, "external_inflow", note="RRSP top-up")["ok"]

    out = pc.apply_classifications([change])[0]
    assert out["cause"] == "external_inflow"
    assert out["classified"] is True
    assert out["is_external_flow"] is True
    assert out["cause_label"] == "Money in"
    assert out["classification_note"] == "RRSP top-up"
    assert out["classified_by"] == "user"
    assert out["classified_at"]


def test_an_unknown_cause_is_refused_not_stored(store):
    res = pc.classify_change(_change(), "probably_a_buy")
    assert res["ok"] is False
    assert "unknown cause" in res["error"]
    assert not store.exists(), "a rejected cause must not reach the ledger"


def test_reclassification_appends_rather_than_overwrites(store):
    change = _change()
    pc.classify_change(change, "trade")
    pc.classify_change(change, "external_inflow")

    lines = [json.loads(x) for x in store.read_text().splitlines() if x.strip()]
    assert len(lines) == 2, "the earlier statement must survive as audit trail"
    assert [x["cause"] for x in lines] == ["trade", "external_inflow"]
    # Last write wins on read.
    assert pc.apply_classifications([change])[0]["cause"] == "external_inflow"


def test_retracting_is_a_write_not_a_delete(store):
    change = _change()
    pc.classify_change(change, "external_inflow")
    pc.classify_change(change, pc.UNCLASSIFIED)

    out = pc.apply_classifications([change])[0]
    assert out["cause"] == pc.UNCLASSIFIED
    assert out["classified"] is False
    assert len(store.read_text().strip().splitlines()) == 2


def test_the_same_ticker_in_two_accounts_classifies_independently(store):
    """Account is in the identity because collapsing it would let one leg of a
    transfer carry the other's cause — and the two legs net to zero."""
    out_leg = _change("VOO", account="TFSA", prior=100.0, current=0.0)
    in_leg = _change("VOO", account="RRSP", prior=0.0, current=100.0)
    pc.classify_change(out_leg, "internal_transfer")

    applied = {c["account"]: c for c in pc.apply_classifications([out_leg, in_leg])}
    assert applied["TFSA"]["cause"] == "internal_transfer"
    assert applied["RRSP"]["cause"] == pc.UNCLASSIFIED


def test_a_malformed_ledger_line_does_not_cost_the_rest(store):
    change = _change()
    pc.classify_change(change, "trade")
    with open(store, "a", encoding="utf-8") as f:
        f.write("{not json at all\n")
    assert pc.apply_classifications([change])[0]["cause"] == "trade"


# ---------------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------------
def test_a_rewritten_snapshot_does_not_inherit_the_old_answer(store):
    """A classification of "money in, 12.5 shares" must not attach itself to a
    delta that now reads 400."""
    original = _change(prior=100.0, current=112.5)
    pc.classify_change(original, "external_inflow")

    rewritten = _change(prior=100.0, current=500.0)
    out = pc.apply_classifications([rewritten])[0]

    assert out["cause"] == pc.UNCLASSIFIED
    assert out["classified"] is False
    assert out["is_external_flow"] is False
    stale = out["stale_classification"]
    assert stale["cause"] == "external_inflow"
    assert stale["against_delta"] == 12.5
    assert stale["now_delta"] == 400.0


def test_a_stale_answer_is_reported_rather_than_erased(store):
    """Reverting silently would look like the save had never happened."""
    pc.classify_change(_change(prior=100.0, current=112.5), "trade")
    out = pc.apply_classifications([_change(prior=100.0, current=113.0)])[0]
    assert "needs restating" in out["stale_classification"]["note"]


# ---------------------------------------------------------------------------
# What TWR reads
# ---------------------------------------------------------------------------
def test_only_the_two_external_causes_move_the_capital_base():
    """A trade or a DRIP counted as a flow would cancel out the return 4.10 is
    trying to measure."""
    assert pc.EXTERNAL_FLOW_CAUSES == {"external_inflow", "external_outflow"}
    for internal in ("trade", "drip", "income", "fee", "corporate_action",
                     "fx_conversion", "internal_transfer"):
        assert not pc.is_external_flow(internal), internal


def test_flow_summary_is_incomplete_while_any_change_is_unstated(store):
    a, b = _change("VOO"), _change("XIC", prior=50.0, current=20.0)
    pc.classify_change(a, "external_inflow")

    summary = pc.flow_summary([a, b])
    assert summary["complete"] is False
    assert summary["unclassified_count"] == 1
    assert summary["external_inflows"] == 1
    assert "LOWER BOUND" in summary["note"]
    assert "unknown amount" in summary["note"]


def test_flow_summary_is_complete_only_when_every_change_is_stated(store):
    a, b = _change("VOO"), _change("XIC", prior=50.0, current=20.0)
    pc.classify_change(a, "external_inflow")
    pc.classify_change(b, "trade")

    summary = pc.flow_summary([a, b])
    assert summary["complete"] is True
    assert summary["unclassified_count"] == 0
    assert summary["external_inflows"] == 1
    assert summary["external_outflows"] == 0   # a trade is not an outflow
    assert summary["inflow_units"] == 12.5


def test_an_empty_change_list_is_complete(store):
    """Nothing observed is nothing to classify — the correct answer is that the
    window is usable, not that it is blocked."""
    assert pc.flow_summary([])["complete"] is True


# ---------------------------------------------------------------------------
# Demand-driven
# ---------------------------------------------------------------------------
def test_410_is_blocked_by_any_unstated_change(store):
    changes = [_change("VOO"), _change("XIC", prior=50.0, current=20.0)]
    pending = pc.pending_for("4.10", changes)
    assert pending["blocked"] is True
    assert pending["pending_count"] == 2
    assert "chain-linked" in pending["requirement"]


def test_47_only_asks_about_decreases(store):
    """A filter on what to ASK, not an inference about what happened: an
    increase with no cash leg cannot be a disposition."""
    up = _change("VOO", prior=100.0, current=112.5)
    down = _change("XIC", prior=50.0, current=20.0)
    pending = pc.pending_for("4.7", [up, down])
    assert pending["pending_count"] == 1
    assert pending["pending"][0]["symbol"] == "XIC"


def test_a_fully_stated_window_blocks_nobody(store):
    change = _change()
    pc.classify_change(change, "trade")
    assert pc.pending_for("4.10", [change])["blocked"] is False


def test_an_unknown_consumer_is_named_not_silently_allowed(store):
    res = pc.pending_for("9.9", [_change()])
    assert "unknown consumer" in res["error"]
    assert res["known"] == ["4.10", "4.7"]


def test_pending_counts_stale_answers_as_needing_restatement(store):
    pc.classify_change(_change(prior=100.0, current=112.5), "trade")
    pending = pc.pending_for("4.10", [_change(prior=100.0, current=900.0)])
    assert pending["pending_count"] == 1
    assert pending["needs_restating"] == 1


# ---------------------------------------------------------------------------
# Tax
# ---------------------------------------------------------------------------
def test_tax_review_marks_changes_to_look_at_never_taxable_events(store):
    """Whether a disposition realises a gain depends on shelter and jurisdiction,
    neither of which this module encodes. See the standing lesson that tax rules
    are not parameters."""
    sale = _change("XIC", prior=50.0, current=20.0)
    drip = _change("VOO", prior=100.0, current=100.4)
    pc.classify_change(sale, "trade")
    pc.classify_change(drip, "drip")

    flagged = pc.tax_review_changes([sale, drip])
    assert [c["symbol"] for c in flagged] == ["XIC"]
    # The word the module must never use about these.
    assert all("taxable" not in json.dumps(c).lower() for c in flagged)


def test_an_unclassified_decrease_is_not_offered_to_tax_review(store):
    """It might be a sale. It might be a fee or a transfer. 4.7 gets told about
    it through `pending_for`, not through the reviewed list."""
    assert pc.tax_review_changes([_change("XIC", prior=50.0, current=20.0)]) == []


# ---------------------------------------------------------------------------
# The valued flow (4.10)
# ---------------------------------------------------------------------------
def test_a_flow_with_no_stated_amount_is_unpriced_rather_than_zero(store):
    """The second completeness axis, added for 4.10.

    This store records QUANTITIES — shares for a security, currency units for
    cash. A time-weighted return needs money, in base currency, on the flow's own
    date. A classified flow with no amount is therefore fully answered and still
    unusable, and `complete` alone reports that window as ready.
    """
    change = _change(prior=0.0, current=5000.0)
    pc.classify_change(change, "external_inflow")

    summary = pc.flow_summary([change])
    assert summary["complete"] is True          # the cause IS stated
    assert summary["priced"] is False           # and the amount is not
    assert summary["unpriced_flow_count"] == 1
    assert summary["flow_amount_base"] is None


def test_a_stated_amount_makes_the_window_priced(store):
    change = _change(prior=0.0, current=5000.0)
    pc.classify_change(change, "external_inflow", amount_base=5000.0,
                       base_currency="CAD")

    summary = pc.flow_summary([change])
    assert summary["priced"] is True
    assert summary["flow_amount_base"] == 5000.0

    applied = pc.apply_classifications([change])[0]
    assert applied["amount_base"] == 5000.0
    assert applied["amount_base_currency"] == "CAD"


def test_the_sign_comes_from_the_cause_not_from_what_was_typed(store):
    """A user entering "5000" for a withdrawal and another entering "-5000" must
    produce the same record. Otherwise the sign is a data-entry convention and
    the TWR engine inherits it."""
    out_a = _change("XIC", prior=100.0, current=0.0)
    out_b = _change("XIU", prior=100.0, current=0.0)
    pc.classify_change(out_a, "external_outflow", amount_base=5000.0)
    pc.classify_change(out_b, "external_outflow", amount_base=-5000.0)

    amounts = [c["amount_base"] for c in pc.apply_classifications([out_a, out_b])]
    assert amounts == [-5000.0, -5000.0]

    inflow = _change("VOO", prior=0.0, current=10.0)
    pc.classify_change(inflow, "external_inflow", amount_base=-2500.0)
    assert pc.apply_classifications([inflow])[0]["amount_base"] == 2500.0


def test_a_non_finite_amount_never_enters_the_store(store):
    """A bare NaN is not valid JSON and one of them took an endpoint down for a
    day. It is rejected at the writer, not filtered at the reader."""
    change = _change(prior=0.0, current=5000.0)
    res = pc.classify_change(change, "external_inflow", amount_base=float("nan"))
    assert res["ok"] is False
    assert not store.exists() or store.read_text().strip() == ""

    assert pc.classify_change(change, "external_inflow", amount_base="abc")["ok"] is False


def test_a_stale_fingerprint_drops_the_amount_along_with_the_cause(store):
    """A classification of "money in, $5,000" must not survive onto a delta that
    now reads $50,000 — and neither must its amount."""
    pc.classify_change(_change(prior=0.0, current=5000.0), "external_inflow",
                       amount_base=5000.0)
    restated = pc.apply_classifications([_change(prior=0.0, current=50000.0)])[0]
    assert restated["classified"] is False
    assert restated.get("amount_base") is None
    assert restated["stale_classification"]["cause"] == "external_inflow"
