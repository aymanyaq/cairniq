"""POST /api/portfolio/parse-statement — a read surface, and only a read surface.

The endpoint hands drafts back to the editor. It is the one place a model's
reading of a financial document meets an HTTP route, and the property worth
pinning is what it does NOT do: no file, no cache, no store. Rows reach
my_portfolio.csv through /api/portfolio/save, pressed by a human, or not at all.
"""
import pytest

from api.routers import portfolio as portfolio_router


def call(payload):
    # A plain `def` route, deliberately: the body is entirely synchronous, so
    # declaring it `async` would run that work ON the event loop and stall every
    # other request (including /api/health, which the watchdog reads). Starlette
    # runs a sync handler in its threadpool instead. Nothing here to await.
    return portfolio_router.parse_statement(payload)


@pytest.fixture
def parser(monkeypatch):
    """Stand in for the extraction, and record what the route passed it."""
    import tools.statement_parser as sp

    seen = {}

    def _parse(text, default_account="", default_currency=""):
        seen.update(text=text, default_account=default_account, default_currency=default_currency)
        return {
            "ok": True,
            "rows": [{"symbol": "AAPL", "shares": 30, "purchase_price": 170.25,
                      "account": "Pension", "currency": "CAD", "asset_type": "Public",
                      "source_text": "AAPL 30 170.25", "name": "", "warnings": []}],
            "row_count": 1,
            "dropped": [],
            "reason": "Drafted 1 position(s) from the pasted text.",
        }

    monkeypatch.setattr(sp, "parse_statement_text", _parse)
    return seen


def test_the_route_passes_the_text_and_the_users_defaults_through(parser):
    result = call({"text": "AAPL 30 170.25", "default_account": "Pension", "default_currency": "CAD"})

    assert parser["text"] == "AAPL 30 170.25"
    assert parser["default_account"] == "Pension"
    assert parser["default_currency"] == "CAD"
    assert result["row_count"] == 1
    assert result["rows"][0]["symbol"] == "AAPL"


def test_missing_fields_do_not_crash_the_route(parser):
    result = call({"text": "AAPL 30 170.25"})

    assert parser["default_account"] == ""
    assert parser["default_currency"] == ""
    assert result["ok"] is True


def test_a_refusal_comes_back_as_a_stated_reason_not_an_empty_list(monkeypatch):
    import tools.statement_parser as sp
    monkeypatch.setattr(sp, "parse_statement_text", lambda *a, **k: {
        "ok": False, "rows": [], "row_count": 0, "dropped": [],
        "reason": "No language model is configured, so the text cannot be read: key not set.",
    })

    result = call({"text": "AAPL 30 170.25"})

    # The editor branches on `ok` before it renders anything about the user's
    # text: "we could not look" must never surface as "we found nothing".
    assert result["ok"] is False
    assert "no language model is configured" in result["reason"].lower()


def test_the_pasted_document_is_never_written_to_the_log(monkeypatch, parser):
    """Only the size of the paste is logged, never its contents.

    A brokerage statement is a private financial document with account names and
    balances in it. The logs are JSONL on disk and are read by other tooling.
    """
    logged = []
    monkeypatch.setattr(portfolio_router, "log_to_component",
                        lambda *a, **k: logged.append((a, k)))

    secret = "RRSP 123-456789  AAPL  30  170.25"
    call({"text": secret})

    assert logged, "the parse should be logged at all"
    blob = repr(logged)
    assert secret not in blob
    assert "123-456789" not in blob
    assert "'chars': 33" in blob or '"chars": 33' in blob


def test_the_route_touches_no_store(parser, monkeypatch):
    """Asserted on the source: this endpoint must stay a pure read.

    The guard is structural rather than behavioural because the failure it
    prevents is a later edit, not this one — a save call added here would put a
    model's reading of a PDF straight into the ledger with no human in between.
    """
    import inspect

    source = inspect.getsource(portfolio_router.parse_statement)
    for forbidden in ("open(", "get_data_path", "clear_cache", "sync_portfolio_to_graph", "writer"):
        assert forbidden not in source, f"parse-statement must not {forbidden!r}"
