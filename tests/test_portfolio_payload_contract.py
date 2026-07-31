"""The shape load_portfolio() hands back, and what a broker's silence means.

Two defects are pinned here, and they compounded:

  1. load_portfolio() appended {"_sync_errors": [...]} into the holdings LIST, so
     an N-position portfolio returned N+1 entries and every caller's count was one
     too many. Every consumer survived it by accident (an empty-symbol `continue`),
     which is why it went unnoticed — but any new consumer that counts entries
     inherits an off-by-one that is constant, and therefore looks stable.

  2. A broker that was never asked — Questrade switched off, no tokens saved —
     reported that as a sync ERROR. Since that is the default state, sync_errors
     was non-empty on every single load, forever. That is what kept the sentinel
     permanently present, and it also parked get_portfolio_summary in its failure
     branch: save_lkg() could never run, so the Last-Known-Good snapshot that
     exists to bridge a real outage was never written.
"""
import pytest

from tools import observations, portfolio_csv, trade_journal
from tools.portfolio_csv import PortfolioPayload, split_portfolio_payload

# Bound at import, before the autouse disable_external_brokers fixture replaces
# tools.questrade.QuestradeAPI — this is the only handle on the real class.
from tools.questrade import QuestradeAPI as _RealQuestradeAPI


def _write_csv(tmp_path, rows):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "\n".join(["symbol,shares,purchase_price,account,currency", *rows])
    )
    return str(path)


# ---------------------------------------------------------------------------
# 1. The list is positions, and only positions
# ---------------------------------------------------------------------------

def test_loaded_list_length_is_the_position_count(tmp_path, monkeypatch):
    """The regression: 29 positions must not come back as 30 entries.

    The autouse broker stubs report an unlinked Questrade, which is exactly the
    condition that used to append the metadata entry.
    """
    monkeypatch.setattr(portfolio_csv, "is_demo_mode", lambda: False)
    csv_path = _write_csv(tmp_path, [
        "AAPL,10,150.0,Manual Brokerage,USD",
        "MSFT,5,250.0,Manual Brokerage,USD",
        "VTI,20,180.0,Manual Brokerage,USD",
    ])

    payload = portfolio_csv.load_portfolio(csv_path)

    assert len(payload) == 3
    assert all("symbol" in row for row in payload)
    assert sorted(r["symbol"] for r in payload) == ["AAPL", "MSFT", "VTI"]
    # Counting entries and counting positions are now the same question.
    assert len([r for r in payload if r.get("symbol")]) == len(payload)


def test_metadata_rides_beside_the_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio_csv, "is_demo_mode", lambda: False)
    csv_path = _write_csv(tmp_path, ["AAPL,10,150.0,Manual Brokerage,USD"])

    payload = portfolio_csv.load_portfolio(csv_path)

    assert isinstance(payload, PortfolioPayload)
    assert payload.sync_errors == []
    assert payload.integration_notices == ["Questrade integration is disabled in Settings."]


def test_no_position_carries_the_old_sentinel_key(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio_csv, "is_demo_mode", lambda: False)
    csv_path = _write_csv(tmp_path, ["AAPL,10,150.0,Manual Brokerage,USD"])

    payload = portfolio_csv.load_portfolio(csv_path)

    assert not any("_sync_errors" in row for row in payload)


# ---------------------------------------------------------------------------
# 2. "Never asked" is not "asked and failed"
# ---------------------------------------------------------------------------

def test_disabled_broker_is_a_notice_not_a_sync_error(tmp_path, monkeypatch):
    """The load that made every load look degraded.

    An unlinked Questrade must leave sync_errors empty — otherwise the LKG
    snapshot is suppressed on every run and a real outage has nothing to fall
    back on, while the dashboard shows a broker-failure warning that never clears.
    """
    monkeypatch.setattr(portfolio_csv, "is_demo_mode", lambda: False)
    csv_path = _write_csv(tmp_path, ["AAPL,10,150.0,Manual Brokerage,USD"])

    payload = portfolio_csv.load_portfolio(csv_path)

    assert payload.sync_errors == []
    assert payload.integration_notices  # recorded, not dropped


def test_real_broker_failure_still_reaches_sync_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio_csv, "is_demo_mode", lambda: False)

    class FailingQuestradeAPI:
        enabled = True
        clients = ["one"]

        def __init__(self, *args, **kwargs):
            pass

        def get_all_holdings(self):
            return {"holdings": [], "errors": ["Questrade token refresh failed"],
                    "notices": []}

    monkeypatch.setattr("tools.questrade.QuestradeAPI", FailingQuestradeAPI)
    csv_path = _write_csv(tmp_path, ["AAPL,10,150.0,Manual Brokerage,USD"])

    payload = portfolio_csv.load_portfolio(csv_path)

    assert payload.sync_errors == ["Questrade token refresh failed"]
    assert len(payload) == 1  # the error did not become a 2nd "holding"


def test_questrade_disabled_reports_notice_at_the_source():
    """Classify where the fact is known, rather than string-matching downstream."""
    qt = _RealQuestradeAPI.__new__(_RealQuestradeAPI)
    qt.enabled = False
    qt.clients = []
    result = qt.get_all_holdings()

    assert result["errors"] == []
    assert result["notices"] == ["Questrade integration is disabled in Settings."]

    qt.enabled = True
    result = qt.get_all_holdings()
    assert result["errors"] == []
    assert result["notices"] == ["No Questrade tokens configured."]


# ---------------------------------------------------------------------------
# 3. The summary still resolves both the new shape and the old one
# ---------------------------------------------------------------------------

def test_split_accepts_the_legacy_inline_sentinel():
    """Hand-built holdings lists (tests, older callers) must still resolve errors.

    Dropping support silently would report "no sync errors" for a payload that
    plainly has them — the failure mode this whole change exists to remove.
    """
    legacy = [
        {"symbol": "AAPL", "shares": 1},
        {"_sync_errors": ["Questrade Global Error: token refresh failed"]},
    ]

    positions, sync_errors, notices = split_portfolio_payload(legacy)

    assert [p["symbol"] for p in positions] == ["AAPL"]
    assert sync_errors == ["Questrade Global Error: token refresh failed"]
    assert notices == []


def test_split_reads_attributes_off_the_payload():
    payload = PortfolioPayload(
        [{"symbol": "AAPL"}], sync_errors=["boom"], integration_notices=["off"]
    )

    assert split_portfolio_payload(payload) == ([{"symbol": "AAPL"}], ["boom"], ["off"])


def test_summary_separates_notices_from_errors(monkeypatch):
    monkeypatch.setattr(
        portfolio_csv,
        "load_portfolio",
        lambda: PortfolioPayload(
            [{
                "symbol": "VOO", "shares": 10.0, "purchase_price": 20.0,
                "current_price": 25.0, "market_value": 250.0,
                "account": "Manual Brokerage", "currency": "USD",
                "source": "Manual", "is_private_asset": False,
            }],
            sync_errors=[],
            integration_notices=["Questrade integration is disabled in Settings."],
        ),
    )
    monkeypatch.setattr(portfolio_csv, "get_cached", lambda key, ttl_seconds=None: None)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_csv, "get_profile_base_currency", lambda: "USD")
    monkeypatch.setattr(portfolio_csv, "get_exchange_rate", lambda f, t: 1.0)

    summary = portfolio_csv._compute_portfolio_summary()

    # This is what unblocks save_lkg() and the knowledge-graph sync in
    # get_portfolio_summary — both gate on sync_errors being empty.
    assert summary["sync_errors"] == []
    assert summary["integration_notices"] == ["Questrade integration is disabled in Settings."]
    assert summary["summary"]["number_of_positions"] == 1


def test_summary_returns_the_error_envelope_without_iterating_it(monkeypatch):
    """The error shape is a dict; iterating it walks its keys as if they were rows."""
    monkeypatch.setattr(
        portfolio_csv, "load_portfolio", lambda: {"error": "Failed to read portfolio file: boom"}
    )
    monkeypatch.setattr(portfolio_csv, "get_cached", lambda key, ttl_seconds=None: None)
    monkeypatch.setattr(portfolio_csv, "set_cached", lambda *a, **k: None)

    assert portfolio_csv._compute_portfolio_summary() == {
        "error": "Failed to read portfolio file: boom"
    }


# ---------------------------------------------------------------------------
# 4. Consumers that were counting the sentinel, or misreading the error dict
# ---------------------------------------------------------------------------

def test_holdings_map_counts_only_positions(monkeypatch):
    monkeypatch.setattr(
        "tools.portfolio_csv.load_portfolio",
        lambda: PortfolioPayload(
            [{"symbol": "AAPL", "shares": 10}, {"symbol": "MSFT", "shares": 5}],
            sync_errors=["Questrade token refresh failed"],
        ),
    )

    assert observations.load_holdings_map() == {"AAPL": 10.0, "MSFT": 5.0}


def test_holdings_map_reports_unreadable_rather_than_empty(monkeypatch):
    """None and {} are different answers, and only one of them is honest here.

    load_portfolio() signals an unreadable file by returning {"error": ...} rather
    than raising, so the except clause never fired: the dict's keys were iterated
    as rows, every .get() raised, the per-row except swallowed it, and the caller
    got {} — "you hold nothing". resolve_rec_follow_through then marks every aged
    SELL as followed and writes that behavioural claim to the store permanently.
    """
    monkeypatch.setattr(
        "tools.portfolio_csv.load_portfolio",
        lambda: {"error": "Failed to read portfolio file: boom"},
    )

    assert observations.load_holdings_map() is None


def test_follow_through_resolves_nothing_when_the_portfolio_is_unreadable(monkeypatch):
    monkeypatch.setattr(
        "tools.portfolio_csv.load_portfolio",
        lambda: {"error": "Failed to read portfolio file: boom"},
    )
    monkeypatch.setattr(
        observations, "load_observations", lambda: pytest.fail("must not read the store")
    )

    assert observations.resolve_rec_follow_through() == {
        "resolved": 0, "followed": 0, "ignored": 0, "pending": 0
    }


def test_tradeable_symbols_ignore_metadata_from_either_shape(monkeypatch):
    monkeypatch.setattr(
        portfolio_csv,
        "load_portfolio",
        lambda: PortfolioPayload(
            [{"symbol": "AAPL", "purchase_price": 150.0},
             {"symbol": "CASH", "purchase_price": 1.0}],
            sync_errors=["Questrade token refresh failed"],
        ),
    )

    assert portfolio_csv.get_tradeable_symbols() == ["AAPL"]


# ---------------------------------------------------------------------------
# 5. The guard the sentinel used to make impossible
# ---------------------------------------------------------------------------

def test_reconcile_refuses_to_archive_against_zero_positions(monkeypatch):
    """A one-entry list of pure metadata used to read as "you hold nothing".

    reconcile_with_holdings closes a thesis when its symbol is ABSENT, so an empty
    holdings list archives every open thesis at once — permanently, with a
    fabricated "exited during portfolio sync" reason. While the sentinel was in
    the list, `if not holdings` could never catch that case: the list was never
    empty. It can now, so it does.
    """
    monkeypatch.setattr(
        trade_journal, "_load_journal", lambda: pytest.fail("must not touch the journal")
    )

    result = trade_journal.reconcile_with_holdings(PortfolioPayload([], sync_errors=["boom"]))

    assert result["reconciled"] is False
    assert result["auto_archived_count"] == 0
    assert "skipped_reason" in result


def test_reconcile_refuses_an_error_envelope(monkeypatch):
    monkeypatch.setattr(
        trade_journal, "_load_journal", lambda: pytest.fail("must not touch the journal")
    )

    result = trade_journal.reconcile_with_holdings({"error": "Failed to read portfolio file"})

    assert result["reconciled"] is False
    assert result["auto_archived_count"] == 0


def test_reconcile_still_archives_a_genuinely_exited_position(monkeypatch, tmp_path):
    """The refusal must not cost the feature: a readable portfolio still reconciles."""
    saved = {}
    monkeypatch.setattr(trade_journal, "_load_journal", lambda: [
        {"id": "1", "symbol": "AAPL", "status": "OPEN"},
        {"id": "2", "symbol": "GONE", "status": "OPEN"},
    ])
    monkeypatch.setattr(trade_journal, "_save_journal", lambda e: saved.update({"entries": e}))
    monkeypatch.setattr(trade_journal, "_remove_from_memory", lambda _id: None)

    result = trade_journal.reconcile_with_holdings(
        PortfolioPayload([{"symbol": "AAPL", "shares": 10}])
    )

    assert result["reconciled"] is True
    assert result["auto_archived_symbols"] == ["GONE"]
    assert saved["entries"][1]["status"] == "CLOSED"
