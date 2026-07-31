import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture()
def client():
    from tools.user_profile import get_active_profile

    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


def test_chat_list_endpoint(client):
    response = client.get("/api/chats")
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)

def test_chat_stop_endpoint_invalid_thread(client):
    # Calling stop with a non-existent thread should return not_found
    response = client.post("/api/chat/stop", json={"thread_id": "non_existent_thread_123"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_found"

def test_chat_stop_endpoint_no_thread(client):
    # Calling stop with no thread should return cancelled (cancels all active)
    response = client.post("/api/chat/stop", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"
    assert "cancelled_threads" in data


def test_chat_endpoint_returns_530_when_agent_unavailable(monkeypatch, client):
    def raise_unavailable():
        raise RuntimeError("Agent not initialized")

    monkeypatch.setattr("api.routers.chat.get_agent", raise_unavailable)

    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 530
    assert "starting up" in response.json()["message"]


def test_process_attachments():
    from api.routers.chat import AttachmentModel, process_attachments
    attachments = [
        AttachmentModel(
            name="test.txt",
            type="text/plain",
            data="data:text/plain;base64,SGVsbG8="
        ),
        AttachmentModel(
            name="chart.png",
            type="image/png",
            data="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
    ]
    extra_text, image_blocks = process_attachments(attachments)
    assert "test.txt" in extra_text
    assert "Hello" in extra_text
    assert len(image_blocks) == 1
    assert image_blocks[0]["type"] == "image_url"


def test_process_attachments_docx_xlsx_routing():
    from api.routers.chat import AttachmentModel, process_attachments
    attachments = [
        AttachmentModel(
            name="dummy.docx",
            type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            data="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,SGVsbG8="
        ),
        AttachmentModel(
            name="dummy.xlsx",
            type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            data="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,SGVsbG8="
        )
    ]
    extra_text, image_blocks = process_attachments(attachments)
    assert "dummy.docx" in extra_text
    assert "Error parsing Word document" in extra_text
    assert "dummy.xlsx" in extra_text
    assert "Error parsing Excel spreadsheet" in extra_text


