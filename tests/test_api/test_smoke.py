import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture()
def client():
    from tools.user_profile import get_active_profile

    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_journal_endpoint(client):
    response = client.get("/journal")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_context_endpoint(client):
    response = client.get("/context")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_portfolio_endpoint(client):
    response = client.get("/portfolio")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_settings_endpoint(client):
    response = client.get("/settings")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_dashboard_endpoint(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
