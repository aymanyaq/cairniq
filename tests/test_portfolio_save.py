import asyncio

import pytest

from api.routers import portfolio as portfolio_router
from tools import portfolio_csv


@pytest.fixture
def csv_path(tmp_path, monkeypatch):
    """Point the save endpoint and the loader at a throwaway portfolio CSV.

    Everything the endpoint touches beyond the CSV is stubbed out, because none of it
    is redirected by the tmp_path patch above: it clears the active profile's daily
    cache by globbing CACHE_DIR, and syncs to the knowledge graph, which binds its own
    get_data_path at import time and would prune and rewrite the real profile's graph
    against this one-row portfolio.
    """
    import tools.cache
    import tools.daily_cache
    import tools.user_profile

    path = tmp_path / "my_portfolio.csv"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setattr(tools.user_profile, "get_data_path", lambda *a, **k: str(path))
    monkeypatch.setattr(tools.daily_cache, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(tools.daily_cache, "get_active_profile", lambda: "test")
    monkeypatch.setattr(tools.cache, "clear_cache", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_csv, "sync_portfolio_to_graph", lambda *a, **k: None)
    # Demo mode makes load_portfolio a pure CSV read, with no broker sync.
    monkeypatch.setattr(portfolio_csv, "is_demo_mode", lambda: True)
    return path


def save(payload):
    return asyncio.run(portfolio_router.save_portfolio_data(payload))


def test_manual_current_price_survives_save_and_reload(csv_path):
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Current Price,Account,Currency,Return Pct\n"
        "AAPL,30,170.00,195.00,Demo Brokerage,USD,\n"
    )

    # The editor renders price read-only, so its payload carries no current_price.
    result = save([{
        "symbol": "AAPL",
        "shares": 30,
        "purchase_price": 170.0,
        "account": "Demo Brokerage",
        "currency": "USD",
        "asset_type": "Public",
        "return_pct": None,
    }])
    assert result["status"] == "success"

    assert "Current Price" in csv_path.read_text().splitlines()[0]

    holdings = portfolio_csv.load_portfolio(str(csv_path))
    assert len(holdings) == 1
    assert holdings[0]["current_price"] == 195.0


def test_manual_market_value_survives_save_and_reload(csv_path):
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Market Value,Account,Currency,Return Pct\n"
        "PENSION,5000,1.00,5250.00,Employer Plan,CAD,5\n"
    )

    save([{
        "symbol": "PENSION",
        "shares": 5000,
        "purchase_price": 1.0,
        "account": "Employer Plan",
        "currency": "CAD",
        "asset_type": "Private",
        "return_pct": 5,
    }])

    holdings = portfolio_csv.load_portfolio(str(csv_path))
    assert holdings[0]["market_value"] == 5250.0
    assert holdings[0]["return_pct"] == 5.0


def test_live_quoted_rows_stay_blank(csv_path):
    """A row with no manual price must not acquire one.

    _compute_portfolio_summary skips the live fetch for any row carrying a
    current_price, so writing a quote here would freeze the row at that price.
    """
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Account,Currency,Return Pct\n"
        "MSFT,10,250.00,Demo Brokerage,USD,\n"
    )

    save([{
        "symbol": "MSFT",
        "shares": 10,
        "purchase_price": 250.0,
        "account": "Demo Brokerage",
        "currency": "USD",
        "asset_type": "Public",
        "return_pct": None,
    }])

    header = csv_path.read_text().splitlines()[0]
    assert "Current Price" not in header
    assert "Market Value" not in header

    holdings = portfolio_csv.load_portfolio(str(csv_path))
    assert "current_price" not in holdings[0]


def test_mixed_rows_blank_the_column_only_where_unpriced(csv_path):
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Current Price,Account,Currency,Return Pct\n"
        "AAPL,30,170.00,195.00,Demo Brokerage,USD,\n"
        "MSFT,10,250.00,,Demo Brokerage,USD,\n"
    )

    save([
        {"symbol": "AAPL", "shares": 30, "purchase_price": 170.0, "account": "Demo Brokerage",
         "currency": "USD", "asset_type": "Public", "return_pct": None},
        {"symbol": "MSFT", "shares": 10, "purchase_price": 250.0, "account": "Demo Brokerage",
         "currency": "USD", "asset_type": "Public", "return_pct": None},
    ])

    holdings = {h["symbol"]: h for h in portfolio_csv.load_portfolio(str(csv_path))}
    assert holdings["AAPL"]["current_price"] == 195.0
    assert "current_price" not in holdings["MSFT"]


def test_renaming_a_holding_drops_its_manual_price(csv_path):
    """Symbol+account identifies the holding; an edit to either makes it a new one."""
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Current Price,Account,Currency,Return Pct\n"
        "AAPL,30,170.00,195.00,Demo Brokerage,USD,\n"
    )

    save([{
        "symbol": "GOOG",
        "shares": 30,
        "purchase_price": 170.0,
        "account": "Demo Brokerage",
        "currency": "USD",
        "asset_type": "Public",
        "return_pct": None,
    }])

    holdings = portfolio_csv.load_portfolio(str(csv_path))
    assert holdings[0]["symbol"] == "GOOG"
    assert "current_price" not in holdings[0]


def test_typed_total_is_written_for_a_holding_with_no_quote(csv_path):
    """The editor now sends the statement total for rows the market cannot price."""
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Account,Currency,Return Pct,Asset Type\n"
        "GRPPEN,1240.5678,0,Workplace Pension,CAD,8.4,Public\n"
    )

    save([{
        "symbol": "GRPPEN",
        "shares": 1240.5678,
        "purchase_price": 0.0,
        "account": "Workplace Pension",
        "currency": "CAD",
        "asset_type": "Public",
        "return_pct": 11.2,
        "market_value": 47111.10,
    }])

    holdings = portfolio_csv.load_portfolio(str(csv_path))
    assert holdings[0]["market_value"] == 47111.10
    assert holdings[0]["return_pct"] == 11.2


def test_clearing_the_total_reverts_the_row_to_units_times_price(csv_path):
    """An empty total is an instruction, not an absent field — it must erase."""
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Market Value,Account,Currency,Return Pct,Asset Type\n"
        "GRPPEN,1240.5678,0,45678.90,Workplace Pension,CAD,8.4,Public\n"
    )

    save([{
        "symbol": "GRPPEN",
        "shares": 1240.5678,
        "purchase_price": 0.0,
        "account": "Workplace Pension",
        "currency": "CAD",
        "asset_type": "Public",
        "return_pct": 8.4,
        "market_value": "",
    }])

    holdings = portfolio_csv.load_portfolio(str(csv_path))
    assert "market_value" not in holdings[0]


def test_a_market_priced_row_cannot_be_pinned_to_a_typed_total(csv_path):
    """A stated total suppresses the live quote, so it must never reach a real ticker.

    The editor does not render the input on these rows, but the server is the seam
    that decides what the CSV holds — a payload that carries one anyway would freeze
    the holding at today's number for good.
    """
    csv_path.write_text(
        "Symbol,Shares,Purchase Price,Account,Currency,Return Pct,Asset Type\n"
        "MSFT,10,250.00,Demo Brokerage,USD,,Public\n"
    )

    save([{
        "symbol": "MSFT",
        "shares": 10,
        "purchase_price": 250.0,
        "account": "Demo Brokerage",
        "currency": "USD",
        "asset_type": "Public",
        "return_pct": None,
        "market_value": 9999.0,
    }])

    holdings = portfolio_csv.load_portfolio(str(csv_path))
    assert "market_value" not in holdings[0]
