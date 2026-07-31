"""The observation endpoints, the /context panel, and the refusal at the cap (1.7).

`tests/test_tools/test_observations.py` and `..._consolidation.py` cover the
logic. This file covers the WIRING, for the reason 2.8's API test states: an
endpoint no page calls is a dark store one layer up, and 1.7 exists because a
memory write path was silent. So the panel has to be MOUNTED and FETCHED, and a
refused lesson has to be VISIBLE — a 409 nobody renders is the same silence.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tools.memory as mem
import tools.observations as obs
import tools.pending_lessons as pl
from server import app

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "context_and_graph.html"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """An isolated profile. These stores hold evidence about the user; a suite
    that wrote into the live ones would be corrupting the data the surface exists
    to report on."""
    from tools.user_profile import get_active_profile

    for module in (mem, obs, pl):
        monkeypatch.setattr(module, "get_data_path", lambda name: str(tmp_path / name))
    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------

def test_an_empty_log_answers_with_zeros_and_the_contract(client):
    """Zero has to be reportable. A log nobody writes to and a log nobody reads
    are the same silence otherwise — the failure this item was opened over."""
    payload = client.get("/api/observations").json()

    assert payload["stats"]["total"] == 0
    assert payload["stats"]["unconsolidated"] == 0
    assert payload["stats"]["gate_met"] is False
    assert payload["recent"] == []
    # The contract travels with the payload, so a second consumer inherits it.
    assert "never injected into a prompt" in payload["contract"]


def test_recorded_turns_show_up_with_their_evidence(client):
    obs.observe_turn("How is YYYY looking?", thread_id="t-1", interaction_id="i-1")

    payload = client.get("/api/observations").json()

    assert payload["stats"]["total"] == 1
    assert payload["stats"]["by_kind"][obs.KIND_ASKED] == 1
    row = payload["recent"][0]
    assert row["tickers"] == ["YYYY"]
    assert row["thread_id"] == "t-1"
    assert row["interaction_id"] == "i-1"


def test_consolidating_an_almost_empty_log_refuses_rather_than_inventing(client):
    """The button skips the CADENCE gate, not the floor: a rule needs two
    citations, so one row cannot produce a valid one and no model is called. The
    surface says that instead of returning an empty success."""
    obs.observe_turn("How is YYYY looking?", thread_id="t-1")

    report = client.post("/api/observations/consolidate").json()

    assert report["drafted"] == 0
    assert report["gated"] is True
    assert "to cite" in report["reason"]


# ---------------------------------------------------------------------------
# The panel is mounted AND fetched
# ---------------------------------------------------------------------------

def test_the_context_page_mounts_the_panel():
    html = TEMPLATE.read_text(encoding="utf-8")
    for element_id in ("observations-counts", "observations-recent", "observations-summary"):
        assert f'id="{element_id}"' in html


def test_the_context_page_actually_calls_the_endpoint():
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "fetch('/api/observations')" in html
    assert "loadObservations();" in html, "the panel is mounted but never populated"


def test_the_page_can_reach_the_consolidate_button():
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "/api/observations/consolidate" in html
    assert "consolidateObservations()" in html


# ---------------------------------------------------------------------------
# Truncation at the cap (user's call 2026-07-27)
# ---------------------------------------------------------------------------
#
# The store retires its oldest rule to fit a new one. That is only acceptable
# because the response NAMES what it retired: the text is destroyed by the write
# and no later read can recover it, so a bare 200 here would lose a rule the user
# wrote with nothing on screen to show for it.

_DISTINCT_RULES = [
    "Size every position before adding risk.",
    "Never catch a falling knife.",
    "Quote fixed income in yield, not price.",
    "Ignore analyst price targets entirely.",
    "Check the ex-dividend date ahead of any trim.",
    "Treat pension holdings as bonds.",
    "Flag anything above 8% of the book.",
    "Do not chase 52-week highs.",
    "Say when a data source is stale.",
    "Prefer index funds inside registered accounts.",
    "Convert every foreign holding to the base currency.",
    "Warn before any trade inside a locked-in account.",
    "Rebalance only on the first business day of a quarter.",
    "Name the broker whenever you quote a commission.",
    "Show the cost basis next to each unrealized gain.",
    "Skip the pre-market tape when sizing an entry.",
    "Assume no options overlay unless I say otherwise.",
    "Round share counts down, never up.",
    "Keep six months of spending in cash outside the market.",
    "Report sector weights before recommending a swap.",
]


def _fill_lessons():
    # The cap tests are only meaningful if this list can actually reach it.
    assert len(_DISTINCT_RULES) >= mem.LESSON_CAP
    for rule in _DISTINCT_RULES[:mem.LESSON_CAP]:
        mem.add_lesson(rule)


def test_a_new_rule_at_the_cap_names_the_rule_it_retired(client):
    _fill_lessons()
    oldest = mem.load_memory()["lessons_learned"][0]

    res = client.post("/api/memory/lessons", json={"text": "A brand new instruction."})

    assert res.status_code == 200
    body = res.json()
    assert body["retired"] == [oldest]
    assert oldest in body["notice"]
    assert body["lesson_count"] == mem.LESSON_CAP
    stored = mem.load_memory()["lessons_learned"]
    assert oldest not in stored
    assert "A brand new instruction." in stored


def test_a_write_with_room_to_spare_announces_nothing(client):
    """The notice has to mean something. If every success carried one the user
    would stop reading the one that costs them a rule."""
    res = client.post("/api/memory/lessons", json={"text": "A brand new instruction."})

    assert res.status_code == 200
    assert "notice" not in res.json()
    assert "retired" not in res.json()


def test_a_confirmed_draft_at_the_cap_reports_what_it_cost(client):
    """Promoting a drafted rule into a full store retires one the user wrote
    themselves — the response has to say which, not just that the draft landed."""
    _fill_lessons()
    oldest = mem.load_memory()["lessons_learned"][0]
    draft = pl.add_pending_lesson("A rule drafted from behaviour.", source="observation_consolidation")

    res = client.post(f"/api/memory/lessons/pending/{draft['id']}/confirm")

    assert res.status_code == 200
    assert res.json()["retired"] == [oldest]
    assert oldest in res.json()["notice"]
    stored = mem.load_memory()["lessons_learned"]
    assert "A rule drafted from behaviour." in stored
    assert oldest not in stored
    assert pl.list_pending_lessons() == []


def test_a_server_refusal_is_rendered_rather_than_swallowed():
    """apiCall used to show 'Sync Error: Check Console' for every non-OK
    response, which would make a reason the user needs to see look like a
    network fault."""
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "payload.error" in html
    assert "showToast(message" in html


def test_the_retirement_notice_survives_the_reload():
    """A 200 that quietly reloads the page would drop the one thing the response
    exists to say. The toast has to fire, and the reload has to wait for it."""
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "payload.notice" in html
    assert "showToast(notice" in html
    assert "lessons[0]" in html  # /context warns which rule is next BEFORE the add
