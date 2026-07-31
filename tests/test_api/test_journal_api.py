"""
Unit tests for /api/journal REST API endpoints and Zero-Effort Auto-Sync Thesis System.
"""
from fastapi.testclient import TestClient

from server import app
from tools.trade_journal import get_trade_history, log_trade, reconcile_with_holdings

client = TestClient(app)


def test_journal_api_crud():
    # 1. Create a new thesis
    response = client.post(
        "/api/journal",
        json={
            "symbol": "TEST_TICKER",
            "action": "BUY",
            "price": 100.0,
            "quantity": 10,
            "thesis": "Test thesis for automated API test",
            "target_price": 120.0,
            "stop_loss": 90.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # 2. Get list of journal entries
    get_res = client.get("/api/journal")
    assert get_res.status_code == 200
    res_json = get_res.json()
    assert "active_theses" in res_json
    assert "archived_history" in res_json

    active_symbols = [t["symbol"] for t in res_json["active_theses"]]
    assert "TEST_TICKER" in active_symbols

    # 3. Close the thesis
    close_res = client.post(
        "/api/journal/close",
        json={
            "symbol": "TEST_TICKER",
            "exit_price": 125.0,
            "outcome": "Profit",
            "lessons_learned": "Target reached successfully"
        }
    )
    assert close_res.status_code == 200

    # 4. Check archived history
    get_res_after = client.get("/api/journal")
    res_json_after = get_res_after.json()
    closed_symbols = [t["symbol"] for t in res_json_after["archived_history"]]
    assert "TEST_TICKER" in closed_symbols


def test_reconcile_with_holdings_auto_archives_exited_positions():
    # Log an open trade for a symbol that will NOT be in holdings
    log_trade("RECON_EXITED", "BUY", price=50.0, thesis="Temporary test symbol")

    # Simulate holdings without RECON_EXITED
    holdings = [
        {"symbol": "AAPL", "shares": 10},
        {"symbol": "MSFT", "shares": 5}
    ]

    result = reconcile_with_holdings(holdings)
    assert result["reconciled"] is True
    assert "RECON_EXITED" in result["auto_archived_symbols"]

    # Verify status is closed
    history = get_trade_history("RECON_EXITED")
    assert len(history) > 0
    assert history[-1]["status"] == "CLOSED"
    assert "Auto-Archived" in history[-1]["outcome"]
