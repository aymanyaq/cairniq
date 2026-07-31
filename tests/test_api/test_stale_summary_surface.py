"""The portfolio page against a summary cached by an EARLIER build.

Every field the template reads is a field some cached summary predates. The
summary cache has a 900s TTL and the Last-Known-Good snapshot on disk has none
at all, so after any deploy that adds a field the page is served — to the live
holder, on the machine that was just restarted — against dicts written before
that field existed.

Jinja turns a missing key into Undefined, and `Undefined is not none` is TRUE.
So the natural-looking guard `{% if h.stated_total is not none %}` does not
guard: it passes Undefined straight into the formatter and takes the whole page
down with a 500. `is defined` has to come first.
"""
import pytest
from fastapi.testclient import TestClient

import tools.portfolio_csv as portfolio_csv
from server import app


def _holding_as_an_older_build_cached_it():
    """A pension row with NO `stated_total` key — the shape before it existed."""
    return {
        "symbol": "PENSION-DEMO", "name": "Group Pension", "shares": 59000.0,
        "purchase_price": "$1.00", "purchase_price_raw": 1.0,
        "current_price": "$1.06", "current_price_raw": 1.055,
        "day_change_pct": None, "gain_loss": "+5.5%", "gain_loss_pct": 5.5,
        "status": "  +5.5% return", "account": "Demo Pension", "currency": "CAD",
        "is_cash_or_pension": True, "is_private_asset": False,
        "return_pct": 5.5, "source": "Manual",
        "value_native": 62245.0, "value_base": 62245.0,
        "value_usd": 43200.0, "value_cad": 62245.0,
    }


@pytest.fixture
def page(monkeypatch):
    summary = {
        "base_currency": "CAD", "total_value_base": 62245.0,
        "total_value_cad": 62245.0, "total_value_usd": 43200.0,
        "percent_return": 5.5, "last_sync_time": "2026-07-31T09:00:00",
        "sync_errors": [], "integration_notices": [],
        "holdings": [_holding_as_an_older_build_cached_it()],
        "accounts": [], "liquidity": {}, "summary": {},
    }
    monkeypatch.setattr(
        portfolio_csv, "get_portfolio_summary", lambda force=False: summary,
    )
    res = TestClient(app).get("/portfolio")
    assert res.status_code == 200, res.text[:2000]
    return res.text


def test_a_summary_without_stated_total_still_renders(page):
    """The row survives, rather than 500ing the page until the cache expires."""
    assert "PENSION-DEMO" in page


def test_the_row_falls_back_to_an_editable_entry_price(page):
    """With no stated total there is nothing to derive an entry price FROM.

    So the cell must stay the ordinary input, not the read-only derived display —
    which is also the branch that would have raised on the Undefined.
    """
    row = page.split("PENSION-DEMO", 1)[1].split("</tr>", 1)[0]
    assert 'name="purchase_price"' in row
    assert "Derived:" not in row


def test_the_total_input_renders_empty_rather_than_printing_undefined(page):
    row = page.split("PENSION-DEMO", 1)[1].split("</tr>", 1)[0]
    assert 'name="market_value"' in row
    assert "Undefined" not in row
