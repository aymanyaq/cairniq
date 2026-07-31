import os
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from tools.portfolio_tracker import get_portfolio_history, snapshot_portfolio


@pytest.fixture
def mock_history_file(tmp_path):
    """Fixture to mock the history file path to a temp directory."""
    with patch("tools.portfolio_tracker.get_history_file", return_value=str(tmp_path / "portfolio_history.csv")):
        yield tmp_path / "portfolio_history.csv"

@patch("tools.portfolio_csv.get_portfolio_summary")
def test_snapshot_portfolio_new(mock_summary, mock_history_file):
    # Setup mock summary
    mock_summary.return_value = {
        "total_value_cad": 13000.0,
        "total_value_usd": 10000.0,
        "total_invested_cad": 11000.0,
        "total_invested_usd": 8500.0,
        "percent_return": 17.6
    }

    # Run snapshot
    snapshot_portfolio()

    # Verify file exists and has data
    assert os.path.exists(mock_history_file)
    df = pd.read_csv(mock_history_file)
    assert len(df) == 1
    assert df.iloc[0]["total_value_usd"] == 10000.0
    assert df.iloc[0]["date"] == date.today().isoformat()

@patch("tools.portfolio_csv.get_portfolio_summary")
def test_snapshot_portfolio_idempotency(mock_summary, mock_history_file):
    mock_summary.return_value = {"total_value_usd": 10000.0}

    # Snapshot twice
    snapshot_portfolio()
    snapshot_portfolio() # Should skip

    df = pd.read_csv(mock_history_file)
    assert len(df) == 1

@patch("tools.portfolio_csv.get_portfolio_summary")
def test_snapshot_portfolio_force(mock_summary, mock_history_file):
    mock_summary.return_value = {"total_value_usd": 10000.0}
    snapshot_portfolio()

    # Update value and force snapshot
    mock_summary.return_value = {"total_value_usd": 11000.0}
    snapshot_portfolio(force=True)

    df = pd.read_csv(mock_history_file)
    assert len(df) == 1
    assert df.iloc[0]["total_value_usd"] == 11000.0

def test_get_portfolio_history_empty(mock_history_file):
    # No file yet
    df = get_portfolio_history()
    assert df.empty

def test_get_portfolio_history_filtering(mock_history_file):
    # Create fake history spanning a year
    dates = [
        (pd.Timestamp.now() - pd.DateOffset(days=400)).date().isoformat(), # > 1y
        (pd.Timestamp.now() - pd.DateOffset(days=200)).date().isoformat(), # < 1y
        (pd.Timestamp.now() - pd.DateOffset(days=15)).date().isoformat(),  # < 1m
    ]
    df_data = pd.DataFrame({
        "date": dates,
        "total_value_usd": [1000, 1100, 1200],
        "total_value_cad": [1350, 1485, 1620],
        "invested_cad": [1000, 1000, 1000],
        "invested_usd": [740, 740, 740],
        "percent_return": [35, 48, 62]
    })
    df_data.to_csv(mock_history_file, index=False)

    # Test 'all'
    assert len(get_portfolio_history(period="all")) == 3

    # Test '1y'
    assert len(get_portfolio_history(period="1y")) == 2

    # Test '1m'
    assert len(get_portfolio_history(period="1m")) == 1
