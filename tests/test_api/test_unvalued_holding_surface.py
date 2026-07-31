"""The read surface for a holding the engine cannot value (`templates/portfolio_editor.html`).

Why this is executed rather than eyeballed: an unvalued row carries None in every
value field, and the cells that render it were written against numbers. Jinja's
``default(0)`` substitutes only for UNDEFINED, not for None, so ``{% if vperf > 0 %}``
on a None raises TypeError and takes the entire page down — a 500, not a wrong
number. No Python test of the summary would notice, because the summary is correct;
the failure lives entirely in the template. So the page is served here.

The second guard is the notice. The whole design of the fix is that the position is
excluded from the total and the holder is told which row is missing and how to fix
it; a silently-dropped position would be a worse bug than the $0.00 it replaced. The
banner is therefore checked for the symbol, the account and both repairs by name.
"""
import pytest
from fastapi.testclient import TestClient

import tools.portfolio_csv as portfolio_csv
from server import app


def _summary_with_unvalued_row():
    """One priceable holding and one that cannot be priced at all."""
    return {
        "base_currency": "CAD",
        "total_value_base": 20000.0,
        "total_value_cad": 20000.0,
        "total_value_usd": 13600.0,
        "percent_return": 5.0,
        "last_sync_time": "2026-07-31T09:00:00",
        "sync_errors": [],
        "integration_notices": [],
        "unvalued_holdings": [{
            "symbol": "GRPPEN",
            "account": "Workplace Pension",
            "shares": 1240.0,
            "reason": "no entry price and no stated total value",
        }],
        "unvalued_notice": "GRPPEN (Workplace Pension) excluded from every total.",
        "holdings": [
            {
                "symbol": "GRPPEN", "name": "Workplace Pension", "shares": 1240.0,
                "purchase_price": "—", "purchase_price_raw": 0.0,
                "current_price": "—", "current_price_raw": 0.0,
                "day_change_pct": None, "gain_loss": "—", "gain_loss_pct": None,
                "status": "⚠️ Unvalued — needs an entry price or a total",
                "is_unvalued": True, "account": "Workplace Pension", "currency": "CAD",
                "is_cash_or_pension": False, "is_private_asset": False,
                "return_pct": 8.4, "stated_total": None, "source": "Manual",
                "value_native": None, "value_base": None,
                "value_usd": None, "value_cad": None,
            },
            {
                "symbol": "PENSION-OK", "name": "Group RRSP", "shares": 1000.0,
                "purchase_price": "$19.05", "purchase_price_raw": 19.047,
                "current_price": "$20.00", "current_price_raw": 20.0,
                "day_change_pct": None, "gain_loss": "+5.0%", "gain_loss_pct": 5.0,
                "status": "💰 +5.0% return", "is_unvalued": False,
                "account": "Workplace Pension", "currency": "CAD",
                "is_cash_or_pension": True, "is_private_asset": False,
                "return_pct": 5.0, "stated_total": 20000.0, "source": "Manual",
                "value_native": 20000.0, "value_base": 20000.0,
                "value_usd": 13600.0, "value_cad": 20000.0,
            },
        ],
        "accounts": [], "liquidity": {}, "summary": {},
    }


@pytest.fixture
def page(monkeypatch):
    monkeypatch.setattr(
        portfolio_csv, "get_portfolio_summary",
        lambda force=False: _summary_with_unvalued_row(),
    )
    res = TestClient(app).get("/portfolio")
    # A 500 here IS the bug this file exists to catch: None reaching a numeric
    # comparison in a Jinja expression aborts the whole template, so the symptom is
    # a dead page rather than a wrong figure.
    assert res.status_code == 200, res.text[:2000]
    return res.text


def test_the_page_survives_a_holding_with_no_value(page):
    """Both rows render. The unvalued one must not take the valued one with it."""
    assert "GRPPEN" in page
    assert "PENSION-OK" in page


def test_the_banner_names_the_excluded_position_and_both_repairs(page):
    assert "excluded from your total" in page
    assert "Workplace Pension" in page
    # Naming the row is the point — "1 holding excluded" would leave the holder
    # hunting for which one.
    assert "1,240 units" in page or "1240 units" in page
    # Either input alone is enough, so the notice must offer both rather than
    # sending the holder looking for a unit cost their statement does not carry.
    assert "Total Value column" in page
    assert "Entry Price" in page


def test_the_unvalued_row_shows_no_number_anywhere(page):
    """Not $0.00, and not a placeholder total the holder might mistake for real.

    The row's own cells are the last place a fabricated zero could survive the fix.
    """
    row = page.split("GRPPEN", 1)[1].split("PENSION-OK", 1)[0]
    assert "$0.00" not in row
    assert "+0.0%" not in row
    # The input is still present: this cell is where the holder fixes it.
    assert 'name="market_value"' in row
    assert 'placeholder="Needs a total"' in row


def test_a_complete_portfolio_shows_no_banner(monkeypatch):
    """The banner must appear only when something is actually missing."""
    summary = _summary_with_unvalued_row()
    summary["unvalued_holdings"] = []
    summary["unvalued_notice"] = ""
    summary["holdings"] = [summary["holdings"][1]]
    monkeypatch.setattr(portfolio_csv, "get_portfolio_summary", lambda force=False: summary)

    res = TestClient(app).get("/portfolio")

    assert res.status_code == 200
    assert "excluded from your total" not in res.text
