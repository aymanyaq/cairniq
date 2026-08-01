"""The structural-convictions editor and its endpoints (Roadmap 3.1's overlay).

`tests/test_tools/test_secular_themes.py` covers the store's contract — that no
theme is ever authored, that clearing means none rather than a reset, and which
half-written rows are refused. This file covers the WIRING, which is the half
that was actually missing and the reason the store is being reopened at all.

The sequence is worth stating once, because it is the same one risk_constraints
and the target allocation each went through, and this is the fourth. The field
shipped with a house thesis as its DEFAULT and back-filled it into live profiles,
where it was read back as the user's own conviction. Emptying the default fixed
that and left the store permanently empty — no setter, no endpoint, no screen —
so the overlay that reads it could never fire, and nothing said so. An engine
that is correct and unreachable is indistinguishable from one that is broken.

So the assertions here are about reachability: the page mounts the editor, the
editor posts to the writer, and a save comes back through the reader the advisor
uses.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tools.memory as mem
from server import app

THEME = {
    "theme": "Grid / Electrification",
    "conviction": "high",
    "horizon": "10 years",
    "rationale": "Stated by the user in their own words.",
    "trim_triggers": ["Close below the 40-week MA"],
    "do_not_trim_for": ["RSI > 70 alone"],
}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """An isolated profile. These are the user's own convictions, and a suite
    that wrote into the live memory would be stating one on their behalf —
    which is the exact failure this store exists to make impossible."""
    from tools.user_profile import get_active_profile

    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


def _template() -> str:
    return Path(__file__).resolve().parents[2].joinpath(
        "templates", "context_and_graph.html"
    ).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The way in — the half that did not exist
# ---------------------------------------------------------------------------

def test_the_context_page_mounts_the_editor_and_posts_to_the_writer():
    """The item in one assertion. A store with no entry screen is how this block
    stayed empty while everything that reads it shipped complete."""
    html = _template()

    assert 'id="st-rows"' in html
    assert "/api/memory/secular_themes" in html
    assert "saveSecularThemes()" in html
    assert "loadSecularThemes()" in html
    assert "addSecularTheme()" in html


def test_the_editor_offers_no_conviction_until_one_is_picked():
    """A pre-selected level would be the page stating how strongly the user
    believes something — the shipped default's mistake in miniature."""
    html = _template()

    assert "ST_CONVICTIONS = ['high', 'medium', 'low']" in html
    # The blank option exists and the value is only set when it matches a level
    # the user picked, so a new card opens on nothing.
    assert "ST_CONVICTIONS.includes(value) ? value : ''" in html


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def test_the_endpoint_reports_no_theme_on_a_fresh_profile(client):
    payload = client.get("/api/memory/secular_themes").json()

    assert payload["secular_themes"] == []


def test_a_saved_conviction_comes_back_through_the_reader(client):
    client.post("/api/memory/secular_themes", json={"themes": [THEME]})

    payload = client.get("/api/memory/secular_themes").json()

    assert [t["theme"] for t in payload["secular_themes"]] == ["Grid / Electrification"]
    assert payload["secular_themes"][0]["trim_triggers"] == ["Close below the 40-week MA"]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def test_a_save_lands_in_the_store_the_advisor_reads(client):
    """Round-tripping through the endpoint proves nothing on its own; the
    accessor the injected context reads is what has to see it."""
    response = client.post("/api/memory/secular_themes", json={"themes": [THEME]})

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(mem.get_secular_themes()) == 1


def test_clearing_through_the_endpoint_leaves_none_rather_than_a_default(client):
    client.post("/api/memory/secular_themes", json={"themes": [THEME]})

    response = client.post("/api/memory/secular_themes", json={"themes": None})

    assert response.status_code == 200
    assert response.json()["cleared"] is True
    assert client.get("/api/memory/secular_themes").json()["secular_themes"] == []
    assert mem.get_secular_themes() == []


def test_a_refused_theme_answers_with_the_reason_rather_than_a_generic_failure(client):
    """The page renders `error` verbatim in a toast. A refusal the user cannot
    act on sends them back to retype the same list."""
    response = client.post("/api/memory/secular_themes", json={
        "themes": [{"theme": "Grid / Electrification", "conviction": "high"}],
    })

    assert response.status_code == 400
    assert "trim" in response.json()["error"].lower()
    assert mem.get_secular_themes() == []


def test_a_refusal_leaves_what_was_already_stated_untouched(client):
    client.post("/api/memory/secular_themes", json={"themes": [THEME]})

    client.post("/api/memory/secular_themes", json={
        "themes": [THEME, {"theme": "Half-written", "conviction": "high"}],
    })

    assert [t["theme"] for t in mem.get_secular_themes()] == ["Grid / Electrification"]


# ---------------------------------------------------------------------------
# The readiness row: reported, never chased
# ---------------------------------------------------------------------------

def test_the_readiness_row_reports_an_unstated_conviction_as_complete(client):
    """A profile with no conviction is finished. The row exists so that the
    blank is VISIBLE and its entry screen nameable, not so it can be chased."""
    payload = client.get("/api/profile_readiness").json()
    row = next(i for i in payload["inputs"] if i["key"] == "secular_themes")

    assert row["status"] == "not_stated"
    assert row["cost"] == ""
    assert row["capabilities_dark"] == []
    assert row["entry"]


def test_an_unstated_conviction_is_not_counted_against_the_profile(client):
    payload = client.get("/api/profile_readiness").json()

    assert payload["counts"]["required_total"] == payload["counts"]["total"] - 1
    assert payload["counts"]["not_stated"] == 1


def test_a_stated_conviction_lights_its_own_row(client):
    client.post("/api/memory/secular_themes", json={"themes": [THEME]})

    payload = client.get("/api/profile_readiness").json()
    row = next(i for i in payload["inputs"] if i["key"] == "secular_themes")

    assert row["status"] == "set"
    assert row["observed"]["themes"] == ["Grid / Electrification"]
