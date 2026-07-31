"""
Tests for Visibility Surfaces (Themes 3.5b Holdings Event Radar & 5.5 ETF Fund Flows).
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_get_event_radar_endpoint():
    """Verify GET /api/portfolio/event-radar endpoint."""
    with patch("tools.event_radar.build_event_radar") as mock_radar:
        mock_radar.return_value = {
            "as_of": "2026-07-28T22:00:00",
            "checked": 5,
            "events": [
                {"kind": "earnings", "symbol": "AAPL", "date": "2026-08-01", "days_until": 4, "label": "AAPL reports earnings"}
            ],
            "unknown": ["CASH"],
        }
        res = client.get("/api/portfolio/event-radar")
        assert res.status_code == 200
        data = res.json()
        assert data["checked"] == 5
        assert len(data["events"]) == 1
        assert data["events"][0]["symbol"] == "AAPL"


def test_get_fund_flows_endpoint():
    """Verify GET /api/portfolio/fund-flows endpoint."""
    with patch("tools.fund_flows.collect_active_profile_fund_universe") as mock_universe, \
         patch("tools.fund_flows.collect_fund_universe") as mock_global_universe, \
         patch("tools.fund_flows.get_flow_series") as mock_series:
        mock_universe.return_value = {"funds": ["SPY"], "non_funds": 2, "unresolved": [], "profiles_read": 1}
        mock_series.return_value = {
            "symbol": "SPY",
            "status": "accruing",
            "days_recorded": 1,
            "wow": None,
        }

        res = client.get("/api/portfolio/fund-flows")
        assert res.status_code == 200
        data = res.json()
        assert "universe" in data
        assert "fund_series" in data
        assert "SPY" in data["fund_series"]
        assert data["fund_series"]["SPY"]["status"] == "accruing"
        assert not mock_global_universe.called


def test_dashboard_api_includes_event_radar_summary():
    """Verify GET /api/dashboard-data attaches event_radar_summary."""
    with patch("api.routers.dashboard.get_portfolio_summary") as mock_sum, \
         patch("tools.event_radar.build_event_radar") as mock_radar:
        mock_sum.return_value = {
            "summary": {"total_value_cad": 100000.0},
            "holdings": [{"symbol": "AAPL", "value_cad": 100000.0}],
            "accounts": [],
            "liquidity": {},
            "top_winners": [],
            "top_losers": [],
            "sync_errors": [],
        }
        mock_radar.return_value = {
            "events": [
                {"kind": "earnings", "symbol": "AAPL", "date": "2026-08-01", "days_until": 4, "label": "AAPL reports earnings"}
            ]
        }

        res = client.get("/api/dashboard-data")
        assert res.status_code == 200
        data = res.json()
        assert "event_radar_summary" in data
        assert data["event_radar_summary"]["total_upcoming_7d"] == 1


def test_agent_tools_event_radar_and_etf_flows():
    """Verify LangGraph registered agent tools for event_radar and etf_flows."""
    from agent.tool_registry import get_etf_flows, get_event_radar

    with patch("tools.event_radar.build_event_radar") as mock_radar, \
         patch("tools.fund_flows.collect_active_profile_fund_universe") as mock_universe, \
         patch("tools.fund_flows.get_flow_series") as mock_series:
        mock_radar.return_value = {"events": []}
        mock_universe.return_value = {"funds": ["QQQ"]}
        mock_series.return_value = {"symbol": "QQQ", "status": "accruing"}

        radar_res = get_event_radar.invoke({})
        assert "events" in radar_res

        flows_res = get_etf_flows.invoke({"symbol": "QQQ"})
        assert flows_res["symbol"] == "QQQ"


def test_get_portfolio_reconciliation_endpoint():
    """Verify GET /api/portfolio/reconciliation (4.10a)."""
    with patch("tools.portfolio_reconciliation.get_reconciliation") as mock_recon:
        mock_recon.return_value = {
            "status": "ready",
            "coverage": {"observed_days": 3, "calendar_days": 5, "missing_days": 2, "gaps": []},
            "prior_date": "2026-07-27",
            "current_date": "2026-07-29",
            "spans_gap": True,
            "changes": [{"kind": "quantity_increase", "symbol": "AAPL", "account": "TFSA",
                         "delta": 2.0, "cause": "unclassified", "spans_gap": True, "gap_days": 2}],
            "change_count": 1,
            "note": "1 change(s) observed",
        }
        res = client.get("/api/portfolio/reconciliation")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        # Coverage travels with the answer — the existing 365-day gate measures
        # span and cannot see the two missing days.
        assert data["coverage"]["missing_days"] == 2
        # The cause must survive serialization: no consumer may render this as
        # a trade or a cash flow.
        assert data["changes"][0]["cause"] == "unclassified"
        assert data["spans_gap"] is True


def test_reconciliation_endpoint_reports_accruing_without_a_change_list():
    """An accruing record must not reach the wire as an unchanged portfolio."""
    with patch("tools.portfolio_reconciliation.get_reconciliation") as mock_recon:
        mock_recon.return_value = {
            "status": "accruing",
            "coverage": {"observed_days": 1, "calendar_days": 1, "missing_days": 0, "gaps": []},
            "changes": [],
            "snapshots": 1,
            "note": "this is an accruing record, not an unchanged portfolio.",
        }
        res = client.get("/api/portfolio/reconciliation")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "accruing"
        assert data["changes"] == []
        assert "not an unchanged portfolio" in data["note"]
