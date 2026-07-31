"""The profile-readiness endpoint and its /context panel (Roadmap 2.8).

`tests/test_tools/test_profile_readiness.py` covers the contract — that a blank
names its consequence and that nothing here ever proposes a value. This file
covers the WIRING, because 2.8 exists to end a failure that was never a logic
bug: a store shipped correct and sat dark, and nobody was told.

An endpoint no page calls is that same failure one layer up, which is why 4.5's
panel carries the same mount-AND-fetch assertion (see
`test_the_dashboard_actually_calls_the_endpoint` in test_goal_api.py).
"""
import pytest
from fastapi.testclient import TestClient

import tools.memory as mem
from server import app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """An isolated profile. This surface reports on the user's OWN stated rules;
    a suite that wrote into the live memory would be corrupting the exact data
    the panel exists to report on."""
    from tools.user_profile import get_active_profile

    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


def test_the_endpoint_answers_with_every_input_and_the_contract(client):
    payload = client.get("/api/profile_readiness").json()

    assert payload["counts"]["total"] == len(payload["inputs"])
    keys = {i["key"] for i in payload["inputs"]}
    assert keys == {
        "drawdown_playbook",
        "risk_constraints",
        "target_allocation",
        "account_jurisdictions",
        "wealth_goal",
        "feedback_ratings",
    }
    # The contract travels WITH the payload rather than living only in the
    # template, so any second consumer inherits it instead of re-stating it.
    assert payload["contract"]


def test_a_blank_store_reports_the_feature_it_switches_off(client):
    """The whole product of this item: not "empty", but "empty, and here is what
    is not running because of it"."""
    payload = client.get("/api/profile_readiness").json()
    blanks = [i for i in payload["inputs"] if i["status"] == "empty"]

    assert blanks, "a bare profile must report blanks, or the surface proves nothing"
    for item in blanks:
        assert item["inert"], f"{item['key']} reports empty but names no consequence"
        assert all(text.strip() for text in item["inert"])
    assert payload["inert_count"] >= len(blanks)


def test_the_context_page_actually_calls_the_endpoint(client):
    """A dead panel and a working one are identical in a screenshot taken before
    the fetch resolves — so assert the block is MOUNTED and that something
    fetches it. This is the assertion that would have caught 4.5's endpoint
    sitting uncalled, and 2.8 is the item about exactly that class of silence."""
    html = client.get("/context").text

    assert 'id="readiness-content"' in html
    assert "/api/profile_readiness" in html
    assert "loadReadiness()" in html
