import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture()
def client():
    from tools.user_profile import get_active_profile

    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


def test_api_update_profile(client):
    response = client.post("/api/memory/profile", json={"updates": {"risk_tolerance": "Aggressive"}})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_api_upsert_thesis(client):
    response = client.post("/api/memory/theses", json={
        "symbol": "TEST",
        "action": "BUY",
        "catalyst": "Earnings",
        "target_price": "150"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_api_upsert_lesson(client):
    response = client.post("/api/memory/lessons", json={
        "text": "Never catch a falling knife."
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_api_add_fact(client):
    response = client.post("/api/memory/facts", json={
        "text": "User is a long-term investor."
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_api_sync_from_facts(client):
    response = client.post("/api/memory/sync_from_facts")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "profile" in data

def test_extract_thesis_no_text(client):
    response = client.post("/api/memory/extract_thesis", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["error"] == "No text provided"
