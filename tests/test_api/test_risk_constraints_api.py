"""The risk-limits editor and its execution-readiness gate (Roadmap 2.2 / 4.4).

`tests/test_tools/test_risk_constraints_memory.py` covers the store's contract —
that no cap is ever authored, that only an explicit clear removes one, and that
a confirmation records the axes it was given about. This file covers the WIRING,
which is the half that was actually missing: the store, its accessor and both
consumers all shipped and were correct, and for months there was no way to reach
any of it. `risk_constraints` was `{}` on the live profile with no screen to fill
it in from, so the gate this codebase calls mandatory had nothing to enforce and
said nothing about that.

So the assertions here are deliberately about reachability: the page mounts the
editor, the editor posts to the writer, and the readiness the engines act on is
the readiness the page shows.
"""
import pytest
from fastapi.testclient import TestClient

import tools.memory as mem
from server import app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """An isolated profile — these are the user's own stated limits, and a suite
    that wrote into the live memory would be editing real risk rules."""
    from tools.user_profile import get_active_profile

    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


# ---------------------------------------------------------------------------
# The way in
# ---------------------------------------------------------------------------

def test_the_context_page_mounts_the_editor_and_posts_to_the_writer():
    """The item in one assertion. A store with no entry screen is how this block
    stayed empty while everything that reads it shipped complete."""
    from pathlib import Path

    html = Path(__file__).resolve().parents[2].joinpath(
        "templates", "context_and_graph.html"
    ).read_text(encoding="utf-8")

    for field in ("rc-max-position", "rc-max-fund", "rc-max-sector", "rc-max-risk"):
        assert f'id="{field}"' in html, f"no input for {field}"
    assert 'id="rc-restricted"' in html
    assert 'id="rc-ack"' in html
    assert "/api/memory/risk_constraints" in html
    assert "saveRiskLimits()" in html
    assert "loadRiskReadiness()" in html
    # The consequence is shown while the answer is being decided, not only in
    # the report underneath — and it is server prose, not a second copy.
    assert 'id="rc-consequences"' in html
    assert "consequence_by_field" in html


def test_the_editors_live_preview_shows_what_a_blank_box_switches_off(client):
    """Same sentences the report uses, reachable by the editor for a box that is
    filled in — which is what makes the preview update as you clear it."""
    row = next(
        i for i in client.get("/api/profile_readiness").json()["inputs"]
        if i["key"] == "risk_constraints"
    )

    assert set(row["consequence_by_field"]) == set(row["required"])
    assert all(text.strip() for text in row["consequence_by_field"].values())


def test_the_page_renders_the_stored_limits_rather_than_placeholders(client):
    client.post("/api/memory/risk_constraints", json={"updates": {
        "max_position_pct": 12, "restricted_symbols": ["NVDA"],
    }})

    html = client.get("/context").text

    assert 'value="12.0"' in html
    assert 'value="NVDA"' in html


def test_a_bare_profile_renders_every_box_empty(client):
    """No house defaults reach the form either. A pre-filled figure would be
    read back later as the user's own limit with one extra click of consent."""
    html = client.get("/context").text

    editor = html[html.index('id="rc-max-position"'):html.index('id="rc-restricted"')]
    assert 'value=""' in editor
    assert "Not Set" in editor


# ---------------------------------------------------------------------------
# The write path, end to end
# ---------------------------------------------------------------------------

def test_saving_a_limit_round_trips_through_the_reader_the_gate_uses(client):
    client.post("/api/memory/risk_constraints", json={"updates": {
        "max_position_pct": 10, "max_sector_pct": 30,
    }})

    payload = client.get("/api/memory/risk_constraints").json()

    assert payload["stated"] == {"max_position_pct": 10.0, "max_sector_pct": 30.0}
    assert payload["execution_readiness"]["execution_ready"] is False
    assert sorted(payload["execution_readiness"]["unanswered"]) == [
        "max_fund_position_pct", "max_risk_per_trade_pct",
    ]


def test_a_blank_box_clears_the_limit_rather_than_being_ignored(client):
    client.post("/api/memory/risk_constraints", json={"updates": {"max_position_pct": 10}})

    client.post("/api/memory/risk_constraints", json={"updates": {"max_position_pct": None}})

    assert client.get("/api/memory/risk_constraints").json()["stated"] == {}


def test_confirming_the_blanks_makes_the_profile_execution_ready(client):
    """The gap the roadmap could name but not close: "no limits" was
    indistinguishable from "never asked", and only the first is an answer."""
    before = client.get("/api/memory/risk_constraints").json()
    assert before["execution_readiness"]["execution_ready"] is False

    saved = client.post("/api/memory/risk_constraints", json={
        "updates": {"acknowledge_unconstrained": True}
    }).json()

    assert saved["execution_readiness"]["execution_ready"] is True
    after = client.get("/api/memory/risk_constraints").json()
    assert after["execution_readiness"]["execution_ready"] is True
    # And it authored nothing to get there.
    assert after["stated"] == {}


def test_unticking_the_box_withdraws_the_confirmation(client):
    client.post("/api/memory/risk_constraints", json={
        "updates": {"acknowledge_unconstrained": True}
    })

    client.post("/api/memory/risk_constraints", json={
        "updates": {"acknowledge_unconstrained": False}
    })

    payload = client.get("/api/memory/risk_constraints").json()
    assert payload["execution_readiness"]["execution_ready"] is False


# ---------------------------------------------------------------------------
# One state, one answer
# ---------------------------------------------------------------------------

def test_the_readiness_surface_agrees_with_the_editors_own_endpoint(client):
    """Two panels on one page reading two accessors is how a surface starts
    contradicting itself about the user's data."""
    client.post("/api/memory/risk_constraints", json={"updates": {"max_position_pct": 10}})

    endpoint = client.get("/api/memory/risk_constraints").json()["execution_readiness"]
    row = next(
        i for i in client.get("/api/profile_readiness").json()["inputs"]
        if i["key"] == "risk_constraints"
    )

    assert sorted(row["missing"]) == sorted(endpoint["unanswered"])
    assert row["stated"] == endpoint["stated"]


def test_the_readiness_row_points_at_the_screen_that_now_exists(client):
    """It used to say there was no entry screen, which was true and was the
    finding. A row that still said so would be reporting a gap it no longer has."""
    row = next(
        i for i in client.get("/api/profile_readiness").json()["inputs"]
        if i["key"] == "risk_constraints"
    )

    assert row["entry"]
    assert "no entry screen" not in row["entry"].lower()
