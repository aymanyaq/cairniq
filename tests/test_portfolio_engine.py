import os

import pytest

from tools.portfolio_csv import load_portfolio
from tools.user_profile import get_data_path, set_active_profile


@pytest.fixture
def test_profile_environment(tmp_path):
    """Fixture to set up a test profile with a mock CSV."""
    profile_name = "pytest_portfolio_user"
    set_active_profile(profile_name)

    # Create mock user_data directory structure
    csv_path = get_data_path("my_portfolio.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Write mock CSV
    mock_csv_content = "symbol,shares,purchase_price,purchase_date\nAAPL,10,150.0,2023-01-01\nMSFT,5,250.0,2023-01-02"
    with open(csv_path, "w") as f:
        f.write(mock_csv_content)

    yield csv_path

    # Teardown
    if os.path.exists(csv_path):
        os.remove(csv_path)

def test_load_portfolio_csv_only(test_profile_environment, monkeypatch):
    """Verify that portfolio engine loads manual CSV holdings correctly."""
    monkeypatch.setenv("QUESTRADE_ENABLED", "false")
    monkeypatch.setenv("ALPACA_API_KEY", "")

    holdings = load_portfolio(test_profile_environment)
    valid_holdings = [h for h in holdings if '_sync_errors' not in h]

    assert len(valid_holdings) == 2

    aapl = next((h for h in valid_holdings if h["symbol"] == "AAPL"), None)
    assert aapl is not None
    assert aapl["shares"] == 10.0
    assert aapl["purchase_price"] == 150.0
    assert aapl["source"] == "Manual"

def test_load_portfolio_empty_csv(tmp_path, monkeypatch):
    """Verify that the engine handles an empty or missing CSV gracefully."""
    monkeypatch.setenv("QUESTRADE_ENABLED", "false")
    monkeypatch.setenv("ALPACA_API_KEY", "")

    empty_csv_path = str(tmp_path / "empty_portfolio.csv")

    # Missing file
    holdings = load_portfolio(empty_csv_path)
    valid_holdings = [h for h in holdings if '_sync_errors' not in h]
    assert len(valid_holdings) == 0

    # Empty file with headers
    with open(empty_csv_path, "w") as f:
        f.write("symbol,shares,purchase_price\n")

    holdings = load_portfolio(empty_csv_path)
    valid_holdings = [h for h in holdings if '_sync_errors' not in h]
    assert len(valid_holdings) == 0
