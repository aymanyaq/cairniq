"""Goal entry + goal projection endpoints (Roadmap 4.5 slice 4).

4.5 shipped the storage, the accessors, and 24 tests -- and the goal itself was
still unset in production months later, because nothing could WRITE one. These
tests cover the write path and the read path together, since either alone is
what produced that gap.

Every test runs against an isolated user_memory.json. The goal is a real user's
long-horizon plan; a test suite that writes a target into the live profile would
be corrupting the exact data the panel reports on.
"""
import pytest
from fastapi.testclient import TestClient

import tools.memory as mem
from server import app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from tools.user_profile import get_active_profile

    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


@pytest.fixture()
def funded_portfolio(monkeypatch):
    """A portfolio value without touching the network or a real CSV."""
    import tools.portfolio_csv as pcsv

    monkeypatch.setattr(
        pcsv, "get_portfolio_summary", lambda *a, **k: {"total_value_base": 1_500_000.0}
    )


def test_an_unset_goal_reads_as_null_not_as_a_plausible_default(client):
    """Unset is MEANINGFUL — the same contract as risk_constraints. A default
    target here would be a number nobody chose, anchoring a decade of decisions."""
    response = client.get("/api/memory/financial_goal")

    assert response.status_code == 200
    assert response.json()["goal"] is None


def test_the_goal_can_be_written_and_read_back(client):
    """The gap this slice closes: get/set_financial_goal existed with no HTTP
    surface and no UI, so the goal stayed empty in production."""
    saved = client.post("/api/memory/financial_goal", json={"updates": {
        "target_low": 3_000_000,
        "target_high": 5_000_000,
        "horizon_years": 10,
        "annual_contribution": 65_000,
    }})

    assert saved.status_code == 200
    goal = client.get("/api/memory/financial_goal").json()["goal"]
    assert goal["target_low"] == 3_000_000
    assert goal["horizon_years"] == 10
    assert goal["annual_contribution"] == 65_000


def test_a_blank_field_clears_it_and_leaves_the_rest_standing(client):
    """What the UI sends when someone empties one box. It must clear that
    figure only -- never wipe the goal."""
    client.post("/api/memory/financial_goal", json={"updates": {
        "target_low": 3_000_000, "horizon_years": 10, "annual_contribution": 65_000,
    }})
    client.post("/api/memory/financial_goal", json={"updates": {"annual_contribution": None}})

    goal = client.get("/api/memory/financial_goal").json()["goal"]
    assert goal["annual_contribution"] is None
    assert goal["target_low"] == 3_000_000
    assert goal["horizon_years"] == 10


def test_a_malformed_figure_does_not_erase_the_existing_goal(client):
    """A typo in a long-horizon target must never destroy it silently."""
    client.post("/api/memory/financial_goal", json={"updates": {"target_low": 3_000_000}})
    client.post("/api/memory/financial_goal", json={"updates": {"target_low": "three million"}})

    assert client.get("/api/memory/financial_goal").json()["goal"]["target_low"] == 3_000_000


def test_projection_is_unavailable_until_a_goal_exists(client, funded_portfolio):
    """An empty chart with a reason beats a confident chart of an assumed plan."""
    payload = client.get("/api/goal_projection").json()

    assert payload["available"] is False
    assert "no wealth goal" in payload["reason"]


def test_projection_names_the_missing_field_rather_than_assuming_it(client, funded_portfolio):
    """The contribution decides whether the goal is reachable, so the endpoint
    must prompt for it -- the UI needs to know WHICH box to highlight."""
    client.post("/api/memory/financial_goal", json={"updates": {
        "target_low": 3_000_000, "horizon_years": 10,
    }})

    payload = client.get("/api/goal_projection").json()

    assert payload["available"] is False
    assert payload["missing"] == ["annual_contribution"]


def test_a_complete_goal_projects_end_to_end(client, funded_portfolio):
    client.post("/api/memory/profile", json={"updates": {
        "risk_tolerance": "Aggressive", "base_currency": "CAD",
    }})
    client.post("/api/memory/financial_goal", json={"updates": {
        "target_low": 3_000_000, "target_high": 5_000_000,
        "horizon_years": 10, "annual_contribution": 65_000,
    }})

    payload = client.get("/api/goal_projection?simulations=1000").json()

    assert payload["available"] is True
    assert payload["currency"] == "CAD"
    assert payload["current_value"] == 1_500_000.0
    assert payload["goal_success_rate"] is not None
    assert payload["required_annual_return"]["low"] is not None
    assert len(payload["bands"]["p50"]) == 11
    # The assumption travels with the answer, never as an invisible default.
    assert payload["assumptions"]["risk_profile"] == "aggressive"


def test_the_dashboard_actually_calls_the_endpoint(client):
    """The failure this whole feature came out of: the goal store shipped with
    accessors, tests, and no caller, so it stayed empty for months. An endpoint
    with no invocation is the same shape one layer up. This asserts the panel is
    mounted AND fetched -- a dead panel and a working one render identically in
    a screenshot taken before the fetch resolves."""
    html = client.get("/dashboard").text

    assert 'id="goal-content"' in html
    assert "/api/goal_projection" in html
    assert "loadGoalProjection()" in html


def test_the_simulation_count_is_bounded(client, funded_portfolio):
    """A query string must not be able to pin the CPU on a read-only panel."""
    client.post("/api/memory/financial_goal", json={"updates": {
        "target_low": 3_000_000, "horizon_years": 10, "annual_contribution": 65_000,
    }})

    payload = client.get("/api/goal_projection?simulations=999999").json()

    assert payload["assumptions"]["num_simulations"] == 20000
