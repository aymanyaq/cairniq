"""
4.10a classification: the API, and the contract between the engine and its
entry screen.

The reason this file exists rather than trusting the unit tests: `risk_constraints`
sat empty for months filed as "blocked on the user" while having no entry screen
at all, and the lesson recorded from it is that a store is not shipped until a
human can reach its writer. So the tests here are about REACHABILITY — that the
options the UI offers are exactly the causes the engine accepts, that the writer
answers, and that a bad write is refused loudly rather than absorbed.
"""
import pytest
from fastapi.testclient import TestClient

from server import app
from tools import portfolio_classification as pc


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pc, "store_path", lambda: str(tmp_path / "position_classifications.jsonl")
    )
    return TestClient(app)


def _change(symbol="VOO", prior=100.0, current=112.5):
    return {
        "kind": "quantity_increase", "account": "TFSA", "symbol": symbol,
        "currency": "CAD", "is_cash": False,
        "prior_shares": prior, "current_shares": current,
        "delta": (current or 0) - (prior or 0),
        "prior_date": "2026-07-28", "current_date": "2026-07-29",
        "cause": "unclassified", "spans_gap": False, "gap_days": 1,
    }


def test_classification_options_match_the_engine(client):
    """A cause the UI offers but the engine rejects is a dead button; one the UI
    omits is a cause nobody can give. Both are silent, so they are asserted."""
    res = client.get("/api/portfolio/classification-options")
    assert res.status_code == 200

    offered = {c["value"] for c in res.json()["causes"]}
    assert offered == set(pc.CAUSES), "the entry screen and the engine have drifted"

    for cause in res.json()["causes"]:
        assert cause["label"] and cause["description"], cause
        assert cause["is_external_flow"] == (cause["value"] in pc.EXTERNAL_FLOW_CAUSES)


def test_every_offered_cause_is_actually_accepted_by_the_writer(client):
    """The end-to-end version of the contract above: each option is POSTed."""
    for cause in client.get("/api/portfolio/classification-options").json()["causes"]:
        res = client.post("/api/portfolio/classify",
                          json={"change": _change(symbol=cause["value"]),
                                "cause": cause["value"]})
        assert res.status_code == 200, (cause["value"], res.text)
        assert res.json()["ok"] is True


def test_the_writer_records_a_cause_that_reads_back(client):
    change = _change()
    res = client.post("/api/portfolio/classify",
                      json={"change": change, "cause": "external_inflow",
                            "note": "July contribution"})
    assert res.status_code == 200
    record = res.json()["record"]
    assert record["cause"] == "external_inflow"
    assert record["note"] == "July contribution"

    applied = pc.apply_classifications([change])[0]
    assert applied["cause"] == "external_inflow"
    assert applied["is_external_flow"] is True


def test_an_unknown_cause_is_a_400_not_a_silent_success(client):
    res = client.post("/api/portfolio/classify",
                      json={"change": _change(), "cause": "vibes"})
    assert res.status_code == 400
    assert "unknown cause" in res.json()["error"]
    assert "valid_causes" in res.json()


def test_a_request_without_a_change_is_refused(client):
    res = client.post("/api/portfolio/classify", json={"cause": "trade"})
    assert res.status_code == 400
    assert "missing `change`" in res.json()["error"]


def test_pending_reports_the_recorder_state_rather_than_zero_when_not_ready(client):
    """`pending_count: 0` because nothing is blocked and `pending_count: 0`
    because no snapshot exists are different facts, so the not-ready reply
    carries the recorder's own status."""
    res = client.get("/api/portfolio/classification-pending?consumer=4.10")
    assert res.status_code == 200
    body = res.json()
    if body.get("status") in ("no_data", "accruing"):
        assert body["blocked"] is False
        assert body["note"], "a not-ready reply must say why it is empty"
    else:
        assert "requirement" in body


def test_pending_rejects_a_consumer_it_does_not_know(client):
    res = client.get("/api/portfolio/classification-pending?consumer=nope")
    body = res.json()
    # Either the recorder is not ready (no consumer check reached) or the
    # consumer is rejected by name — never a confident empty list.
    assert body.get("status") in ("no_data", "accruing") or "unknown consumer" in body.get("error", "")


def test_the_reconciliation_endpoint_carries_the_completeness_block(client):
    res = client.get("/api/portfolio/reconciliation")
    assert res.status_code == 200
    body = res.json()
    if body["status"] == "ready":
        cls = body["classification"]
        # `complete` must be present and boolean — a missing key would read as
        # falsy at the call site and quietly block forever.
        assert isinstance(cls["complete"], bool)
        assert cls["changes_seen"] == body["change_count"], \
            "completeness must be computed over ALL changes, not the truncated page"


# ---------------------------------------------------------------------------
# The AMOUNT — 4.10's second axis, and the writer it shipped without
# ---------------------------------------------------------------------------
# `amount_base` was added to `classify_change` on 2026-07-30 and nothing sent
# one, so 4.10 could only ever return `flows_incomplete → unpriced_flows` — and
# that reads as the user declining to answer a question no screen asked. Exactly
# the failure this file's docstring was written about, re-introduced the same
# morning it was cited. These tests are the reachability half.
def test_the_writer_accepts_an_amount_and_stores_it(client):
    change = _change(symbol="CASH", prior=1000.0, current=6000.0)
    res = client.post("/api/portfolio/classify",
                      json={"change": change, "cause": "external_inflow",
                            "amount_base": 5000.0, "base_currency": "CAD"})
    assert res.status_code == 200
    assert res.json()["record"]["amount_base"] == 5000.0
    assert res.json()["record"]["base_currency"] == "CAD"

    applied = pc.apply_classifications([change])[0]
    assert applied["amount_base"] == 5000.0
    assert applied["amount_base_currency"] == "CAD"


def test_a_cause_still_saves_with_no_amount(client):
    """The two are separate writes on purpose. "It was a deposit" is a real
    answer even when the figure is not to hand, and demanding both would make the
    cause unstatable — which is worse, because the cause is the harder half."""
    change = _change(symbol="CASH", prior=1000.0, current=6000.0)
    res = client.post("/api/portfolio/classify",
                      json={"change": change, "cause": "external_inflow"})
    assert res.status_code == 200
    assert res.json()["record"]["amount_base"] is None

    summary = pc.flow_summary([change])
    assert summary["complete"] is True     # the cause IS stated
    assert summary["priced"] is False      # and the window is still unusable


def test_the_amount_completes_the_window_for_4_10(client):
    change = _change(symbol="CASH", prior=1000.0, current=6000.0)
    client.post("/api/portfolio/classify",
                json={"change": change, "cause": "external_inflow"})
    assert pc.flow_summary([change])["priced"] is False

    client.post("/api/portfolio/classify",
                json={"change": change, "cause": "external_inflow",
                      "amount_base": 5000.0, "base_currency": "CAD"})
    summary = pc.flow_summary([change])
    assert summary["priced"] is True
    assert summary["flow_amount_base"] == 5000.0


def test_an_emptied_amount_retracts_it_rather_than_recording_zero(client):
    """Zero is a legitimate stated value; "I took that figure back" is not zero.
    The UI sends null for an emptied box and the ledger appends the retraction
    rather than deleting the earlier answer."""
    change = _change(symbol="CASH", prior=1000.0, current=6000.0)
    client.post("/api/portfolio/classify",
                json={"change": change, "cause": "external_inflow",
                      "amount_base": 5000.0})
    assert pc.flow_summary([change])["priced"] is True

    client.post("/api/portfolio/classify",
                json={"change": change, "cause": "external_inflow",
                      "amount_base": None})
    assert pc.flow_summary([change])["priced"] is False
    assert pc.apply_classifications([change])[0]["classified"] is True


def test_a_bad_amount_is_refused_loudly_rather_than_absorbed(client):
    change = _change(symbol="CASH", prior=1000.0, current=6000.0)
    res = client.post("/api/portfolio/classify",
                      json={"change": change, "cause": "external_inflow",
                            "amount_base": "abc"})
    assert res.status_code == 400
    assert res.json()["ok"] is False


def test_a_bare_nan_on_the_wire_never_reaches_the_store(client):
    """`JSON.stringify(NaN)` is `null`, so no browser sends this — but Python's
    `json.loads` accepts bare `NaN` and `Infinity` by default, so the body below
    parses into a real float before any validation runs. It is the one path by
    which a non-finite number can arrive, and a bare NaN in a durable store has
    already taken an endpoint of this app down for a full day."""
    import json

    change = _change(symbol="CASH", prior=1000.0, current=6000.0)
    for literal in ("NaN", "Infinity", "-Infinity"):
        body = ('{"change": %s, "cause": "external_inflow", "amount_base": %s}'
                % (json.dumps(change), literal))
        res = client.post("/api/portfolio/classify", content=body,
                          headers={"Content-Type": "application/json"})
        assert res.status_code == 400, literal
        assert res.json()["ok"] is False

    # And the store is untouched — refused at the writer, not filtered on read.
    assert pc.apply_classifications([change])[0]["classified"] is False


def test_the_sign_is_derived_from_the_cause_not_from_the_entry(client):
    """Whether a user types 5000 or -5000 for a withdrawal must not change the
    record. Otherwise the sign is a data-entry convention and the TWR engine
    inherits it."""
    typed_positive = _change(symbol="XIC", prior=100.0, current=0.0)
    typed_negative = _change(symbol="XIU", prior=100.0, current=0.0)
    client.post("/api/portfolio/classify",
                json={"change": typed_positive, "cause": "external_outflow",
                      "amount_base": 5000.0})
    client.post("/api/portfolio/classify",
                json={"change": typed_negative, "cause": "external_outflow",
                      "amount_base": -5000.0})

    amounts = [c["amount_base"]
               for c in pc.apply_classifications([typed_positive, typed_negative])]
    assert amounts == [-5000.0, -5000.0]
