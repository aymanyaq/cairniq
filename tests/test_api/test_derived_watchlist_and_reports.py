"""
Tests for Derived Watchlist (Roadmap 3.6) & Report Export Endpoints (Product Surface).
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_export_weekly_review_endpoint():
    """Verify GET /api/export/weekly-review returns formatted markdown."""
    with patch("api.routers.reports.build_weekly_review") as mock_review:
        mock_review.return_value = {
            "generated_at": "2026-07-28T22:00:00",
            "period": {"label": "Jul 21 – Jul 28, 2026"},
            "sections": [
                {
                    "key": "goal",
                    "title": "Wealth goal",
                    "status": "ok",
                    "current_value": 1500000.0,
                    "goal": 3500000.0,
                },
                {
                    "key": "advice",
                    "title": "What the advisor said this week",
                    "status": "ok",
                    "calls": [{"symbol": "AAPL", "action": "BUY"}],
                }
            ],
        }

        res = client.get("/api/export/weekly-review")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/markdown")
        assert "# CairnIQ Weekly One-Page Review" in res.text
        assert "AAPL" in res.text
        assert "Wealth goal" in res.text


def test_export_advisor_scorecard_csv():
    """Verify GET /api/export/advisor-scorecard returns CSV file."""
    with patch("tools.memory.load_memory") as mock_mem:
        mock_mem.return_value = {
            "past_recommendations": [
                {"symbol": "MU", "action": "SELL", "date": "2026-07-20", "entry_price": 120.0, "exit_price": 110.0, "alpha_pct": "+8.3%", "status": "GRADED"}
            ]
        }

        res = client.get("/api/export/advisor-scorecard?format=csv")
        assert res.status_code == 200
        assert "text/csv" in res.headers["content-type"]
        assert "Symbol,Action,Stated Date" in res.text
        assert "MU,SELL" in res.text


def test_export_advisor_scorecard_escapes_spreadsheet_formulas():
    """Ensure exported recommendation data cannot execute spreadsheet formulas."""
    with patch("tools.memory.load_memory") as mock_mem:
        mock_mem.return_value = {
            "past_recommendations": [
                {"symbol": "=HYPERLINK(\"https://example.test\")", "action": "+SUM(1,1)"}
            ]
        }

        res = client.get("/api/export/advisor-scorecard?format=csv")

        assert res.status_code == 200
        assert "'=HYPERLINK" in res.text
        assert "'+SUM(1,1)" in res.text


def test_export_advisor_scorecard_json():
    """Verify GET /api/export/advisor-scorecard?format=json returns JSON."""
    with patch("tools.memory.load_memory") as mock_mem:
        mock_mem.return_value = {"past_recommendations": [{"symbol": "MU"}]}

        res = client.get("/api/export/advisor-scorecard?format=json")
        assert res.status_code == 200
        data = res.json()
        assert "count" in data
        assert data["count"] == 1


def test_derived_watchlist_in_catalyst_scan():
    """Verify background catalyst scan derives watchlist from WATCHING theses."""
    from api.background import run_catalyst_scan_in_background

    with patch("agent.nodes.news_analyst.gather_news_tool_outputs") as mock_news, \
         patch("tools.catalyst_extractor.extract_catalysts") as mock_extract, \
         patch("tools.portfolio_csv.get_portfolio_summary") as mock_sum, \
         patch("tools.memory.get_active_theses") as mock_theses, \
         patch("tools.memory._thesis_position_state") as mock_state:

        mock_news.return_value = ({}, [], [{"tool": "news", "output": "test"}])
        mock_sum.return_value = {"holdings": [{"symbol": "AAPL"}]}
        mock_theses.return_value = [{"symbol": "NVDA"}, {"symbol": "AAPL"}]
        mock_state.side_effect = lambda t, held: "watching" if t["symbol"] == "NVDA" else "held"
        mock_extract.return_value = {}

        run_catalyst_scan_in_background()

        assert mock_extract.called
        call_kwargs = mock_extract.call_args.kwargs
        assert call_kwargs["watchlist"] == ["NVDA"]
        assert call_kwargs["holdings"] == ["AAPL"]
