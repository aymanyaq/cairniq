"""Day move and native-currency value on portfolio holdings.

The load-bearing property here is negative: a row only claims a direction when a live
quote actually backed its price. Everything else — a broker price, a manual CSV price,
a Last-Known-Good snapshot, a provider serving the prior session's close — must report
no direction rather than a flat 0.00%, which would read as "unchanged today".
"""
import pytest

from tools import portfolio_csv


@pytest.fixture
def rates(monkeypatch):
    def fake_get_cached(key, ttl_seconds=None):
        if key == "usd_cad_rate":
            return 1.4
        return None

    monkeypatch.setattr(portfolio_csv, "get_cached", fake_get_cached)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *a, **k: None)


def summary_for(monkeypatch, holdings, stock_data):
    monkeypatch.setattr(portfolio_csv, "load_portfolio", lambda: holdings)
    monkeypatch.setattr(
        "tools.market_data.get_stock_data",
        lambda sym: stock_data.get(sym, {"error": "not found"}),
    )
    return portfolio_csv._compute_portfolio_summary()


def holding(**kw):
    base = {
        "symbol": "AAPL",
        "shares": 10.0,
        "purchase_price": 100.0,
        "account": "Manual Brokerage",
        "currency": "USD",
        "source": "Manual",
        "is_private_asset": False,
    }
    base.update(kw)
    return base


def test_live_quote_carries_day_move_and_native_value(monkeypatch, rates):
    summary = summary_for(
        monkeypatch,
        [holding()],
        {"AAPL": {"current_price": "$110.00", "day_change_pct": 2.5, "previous_close": 107.32}},
    )
    row = summary["holdings"][0]
    assert row["day_change_pct"] == pytest.approx(2.5)
    # Native value stays in the row's own currency, unconverted.
    assert row["value_native"] == pytest.approx(1100.0)
    assert row["value_cad"] == pytest.approx(1540.0)


def test_quote_without_day_move_reports_no_direction(monkeypatch, rates):
    """A provider that has a price but no usable previous close (e.g. Polygon)."""
    summary = summary_for(
        monkeypatch,
        [holding()],
        {"AAPL": {"current_price": "$110.00", "day_change_pct": None, "previous_close": None}},
    )
    assert summary["holdings"][0]["day_change_pct"] is None


def test_live_quote_beats_a_pinned_csv_price(monkeypatch, rates):
    """The pin is a fallback, not an override: a live quote is newer and wins.

    Skipping the fetch made a pinned price permanent — the row reported it long after
    the market moved, and could never show a day move.
    """
    summary = summary_for(
        monkeypatch,
        [holding(current_price=195.0)],
        {"AAPL": {"current_price": "$110.00", "day_change_pct": 9.9}},
    )
    row = summary["holdings"][0]
    assert row["current_price_raw"] == pytest.approx(110.0)
    assert row["day_change_pct"] == pytest.approx(9.9)


def test_demo_seeded_price_is_not_overridden_by_a_quote(monkeypatch, rates):
    """The demo ships seeded prices so the sample portfolio is deterministic and offline."""
    monkeypatch.setattr(portfolio_csv, "is_demo_mode", lambda: True)
    summary = summary_for(
        monkeypatch,
        [holding(current_price=195.0)],
        {"AAPL": {"current_price": "$110.00", "day_change_pct": 9.9}},
    )
    row = summary["holdings"][0]
    assert row["current_price_raw"] == pytest.approx(195.0)
    assert row["day_change_pct"] is None


def test_pinned_csv_price_is_used_when_no_quote_exists(monkeypatch, rates):
    """An untradable fund or a failed lookup falls back to the pin, with no direction."""
    summary = summary_for(
        monkeypatch,
        [holding(symbol="ACMEFUND", current_price=195.0)],
        {"ACMEFUND": {"current_price": "N/A"}},
    )
    row = summary["holdings"][0]
    assert row["current_price_raw"] == pytest.approx(195.0)
    assert row["day_change_pct"] is None


def test_broker_row_gets_day_move_but_keeps_broker_price(monkeypatch, rates):
    """A broker price is genuinely current, so today's move belongs beside it.

    The quote is fetched for the move only — the displayed price must still be the
    broker's, never the quote's.
    """
    summary = summary_for(
        monkeypatch,
        [holding(source="API", current_price=110.0, market_value=1100.0)],
        {"AAPL": {"current_price": "$999.00", "day_change_pct": 2.5}},
    )
    row = summary["holdings"][0]
    assert row["day_change_pct"] == pytest.approx(2.5)
    assert row["current_price_raw"] == pytest.approx(110.0)


def test_same_ticker_held_manually_and_via_broker_agree(monkeypatch, rates):
    """The reported symptom: one SCHB row showed a move, the synced copy showed none."""
    summary = summary_for(
        monkeypatch,
        [
            holding(symbol="SCHB", purchase_price=50.00, shares=40.0),
            holding(symbol="SCHB", source="API", shares=80.0, purchase_price=52.00,
                    current_price=45.00, market_value=3600.0),
        ],
        {"SCHB": {"current_price": "$45.00", "day_change_pct": -0.48}},
    )
    moves = [r["day_change_pct"] for r in summary["holdings"]]
    assert moves == [pytest.approx(-0.48), pytest.approx(-0.48)]


def test_broker_cash_row_reports_no_direction(monkeypatch, rates):
    """Broker cash is still cash — a $1.00 unit has no market move."""
    summary = summary_for(
        monkeypatch,
        [holding(symbol="CASH", source="API", shares=5000.0, purchase_price=1.0,
                 current_price=1.0, market_value=5000.0, return_pct=0.0)],
        {"CASH": {"current_price": "$1.00", "day_change_pct": 3.3}},
    )
    assert summary["holdings"][0]["day_change_pct"] is None


def test_cash_and_private_rows_report_no_direction(monkeypatch, rates):
    summary = summary_for(
        monkeypatch,
        [
            holding(symbol="CASH", shares=5000.0, purchase_price=1.0, return_pct=0.0),
            holding(symbol="HOUSE", purchase_price=500000.0, shares=1.0, is_private_asset=True),
        ],
        {},
    )
    assert [r["day_change_pct"] for r in summary["holdings"]] == [None, None]


def test_lkg_fallback_price_reports_no_direction(monkeypatch, rates, tmp_path):
    """A stale snapshot price must not be paired with today's move."""
    import json

    import tools.user_profile

    lkg = tmp_path / "test_portfolio_lkg.json"
    lkg.write_text(json.dumps({"holdings": [{"symbol": "AAPL", "current_price": "$99.00"}]}))
    monkeypatch.setattr(portfolio_csv, "get_active_profile", lambda: "test")
    monkeypatch.setattr(tools.user_profile, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(portfolio_csv, "get_data_path", lambda name: str(tmp_path / name))

    # Quote fails, so the price falls back to the LKG snapshot.
    summary = summary_for(monkeypatch, [holding()], {"AAPL": {"current_price": "N/A"}})
    row = summary["holdings"][0]
    assert row["current_price_raw"] == pytest.approx(99.0)
    assert row["day_change_pct"] is None
