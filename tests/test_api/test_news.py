import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture()
def client():
    from tools.user_profile import get_active_profile

    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


def test_news_feed_endpoint(client):
    # Calling without force
    response = client.get("/api/news-feed")
    assert response.status_code == 200
    data = response.json()
    # It either returns fetching status or cached markdown
    assert "status" in data or "markdown" in data

def test_market_pulse_endpoint(client):
    response = client.get("/api/market-pulse")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "regime" in data

def test_market_pulse_history(client):
    response = client.get("/api/market-pulse/history?days=5")
    assert response.status_code == 200
    data = response.json()
    # It should return a list of history points
    assert isinstance(data, dict) or isinstance(data, list)


def test_catalysts_endpoint_never_autospends(client):
    response = client.get("/api/catalysts")
    assert response.status_code == 200
    data = response.json()
    # Plain GET returns cache, in-progress status, or empty — never starts a scan.
    assert "catalysts" in data or data.get("status") in ("fetching", "empty")


def test_catalyst_scenario_endpoint_read_only(client):
    # Unknown id → clean not_found (no LLM call, no error).
    response = client.get("/api/catalysts/scenario/definitely-not-a-real-id")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "not_found")
    assert data["id"] == "definitely-not-a-real-id"
