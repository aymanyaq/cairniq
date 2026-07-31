"""
Tests for portfolio-change reconciliation (Roadmap 4.10a).

The contracts under test are mostly NEGATIVE — what this module must refuse to
say. A reconciliation ledger that guesses a cause, or that reports an empty
change list before it has anything to compare, produces a number downstream
consumers cannot distinguish from a measured one.
"""

import csv
from unittest.mock import patch

import pytest

from tools import portfolio_reconciliation as pr


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the module at an isolated per-test store."""
    path = tmp_path / "position_history.csv"
    monkeypatch.setattr(pr, "history_path", lambda: str(path))
    return path


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pr._FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in pr._FIELDS})


def _row(date, symbol, shares, account="TFSA", currency="USD"):
    return {"date": date, "account": account, "symbol": symbol, "currency": currency,
            "shares": shares, "private": "", "as_of": f"{date}T17:00:00"}


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def test_snapshot_records_one_row_per_account_and_symbol(store):
    holdings = [
        {"symbol": "AAPL", "shares": 10, "account": "TFSA", "currency": "USD"},
        {"symbol": "AAPL", "shares": 5, "account": "RRSP", "currency": "USD"},
        {"symbol": "CASH", "shares": 2500.0, "account": "TFSA", "currency": "CAD"},
    ]
    with patch("tools.portfolio_csv.load_portfolio", return_value=holdings):
        report = pr.snapshot_positions()

    assert report["recorded"] == 3
    rows = pr.read_history()
    # The SAME ticker in two accounts stays two positions. Collapsing them would
    # hide an inter-account transfer entirely — the legs would net to zero.
    assert {(r["account"], r["symbol"]) for r in rows} == {
        ("TFSA", "AAPL"), ("RRSP", "AAPL"), ("TFSA", "CASH"),
    }


def test_snapshot_is_idempotent_per_day(store):
    holdings = [{"symbol": "AAPL", "shares": 10, "account": "TFSA", "currency": "USD"}]
    with patch("tools.portfolio_csv.load_portfolio", return_value=holdings):
        first = pr.snapshot_positions()
        second = pr.snapshot_positions()

    assert first["recorded"] == 1
    assert second["recorded"] == 0
    assert second["declined"] == "already recorded today"
    assert len(pr.read_history()) == 1


def test_an_unreadable_holding_is_skipped_not_recorded_as_zero(store):
    """A row we cannot parse is not a zero position.

    Recording it as zero would manufacture a full disposal on the next
    reconciliation — an invented sale of the entire position.
    """
    holdings = [
        {"symbol": "AAPL", "shares": 10, "account": "TFSA", "currency": "USD"},
        {"symbol": "BROKEN", "shares": "not-a-number", "account": "TFSA"},
    ]
    with patch("tools.portfolio_csv.load_portfolio", return_value=holdings):
        report = pr.snapshot_positions()

    assert report["recorded"] == 1
    assert report["skipped_unreadable"] == 1
    assert {r["symbol"] for r in pr.read_history()} == {"AAPL"}


def test_a_portfolio_read_error_records_nothing_and_says_so(store):
    with patch("tools.portfolio_csv.load_portfolio", return_value={"error": "csv unreadable"}):
        report = pr.snapshot_positions()

    assert report["recorded"] == 0
    assert "csv unreadable" in report["error"]
    assert pr.read_history() == []


# ---------------------------------------------------------------------------
# The accruing contract — the reason this module exists in this shape
# ---------------------------------------------------------------------------
def test_no_snapshots_reports_no_data_never_an_empty_change_list(store):
    res = pr.get_reconciliation()
    assert res["status"] == "no_data"
    assert res["changes"] == []
    assert "NOT a report" in res["note"]


def test_one_snapshot_reports_accruing_and_withholds_the_change_list(store):
    """A first snapshot and an unchanged portfolio are DIFFERENT claims.

    Returning `changes: []` for both is the failure this repo has shipped in
    Market Pulse, in 5.4's tone verdict and in 5.9's Unknown rows.
    """
    _write(store, [_row("2026-07-20", "AAPL", 10)])
    res = pr.get_reconciliation()

    assert res["status"] == "accruing"
    assert res["changes"] == []
    assert res["snapshots"] == 1
    assert "not an unchanged portfolio" in res["note"]


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------
def test_detects_opened_closed_and_resized_positions(store):
    _write(store, [
        _row("2026-07-20", "AAPL", 10),
        _row("2026-07-20", "MSFT", 4),
        _row("2026-07-21", "AAPL", 12),      # increased
        _row("2026-07-21", "NVDA", 3),       # opened
        # MSFT absent on the 21st -> closed
    ])
    changes = {c["symbol"]: c for c in pr.detect_changes("2026-07-20", "2026-07-21")}

    assert changes["AAPL"]["kind"] == "quantity_increase"
    assert changes["AAPL"]["delta"] == pytest.approx(2)
    assert changes["NVDA"]["kind"] == "position_opened"
    assert changes["MSFT"]["kind"] == "position_closed"
    assert changes["MSFT"]["delta"] == pytest.approx(-4)


def test_every_change_is_unclassified(store):
    """The central contract. A delta is equally consistent with a trade, a
    deposit, a transfer, a DRIP, a fee or a corporate action."""
    _write(store, [
        _row("2026-07-20", "AAPL", 10),
        _row("2026-07-21", "AAPL", 12),
        _row("2026-07-20", "CASH", 5000.0),
        _row("2026-07-21", "CASH", 9000.0),
    ])
    changes = pr.detect_changes("2026-07-20", "2026-07-21")

    assert changes, "expected changes to compare"
    assert all(c["cause"] == "unclassified" for c in changes)
    # And nothing anywhere calls a $4000 cash increase a deposit.
    cash = next(c for c in changes if c["symbol"] == "CASH")
    assert cash["kind"] == "cash_increase"
    assert "deposit" not in str(cash).lower()


def test_a_fractional_share_increase_is_reported(store):
    """No floor on share changes — a fractional increase IS the DRIP signal."""
    _write(store, [_row("2026-07-20", "VTI", 100.0), _row("2026-07-21", "VTI", 100.0731)])
    changes = pr.detect_changes("2026-07-20", "2026-07-21")

    assert len(changes) == 1
    assert changes[0]["delta"] == pytest.approx(0.0731)


def test_sub_floor_cash_noise_is_not_reported_as_a_movement(store):
    _write(store, [
        _row("2026-07-20", "CASH", 5000.00, currency="CAD"),
        _row("2026-07-21", "CASH", 5000.40, currency="CAD"),
    ])
    assert pr.detect_changes("2026-07-20", "2026-07-21") == []


def test_an_unchanged_portfolio_produces_no_changes_once_two_snapshots_exist(store):
    _write(store, [_row("2026-07-20", "AAPL", 10), _row("2026-07-21", "AAPL", 10)])
    res = pr.get_reconciliation()

    # This is the ONLY state in which an empty change list is a real answer,
    # and it is distinguishable from the accruing one by `status`.
    assert res["status"] == "ready"
    assert res["changes"] == []
    assert res["change_count"] == 0


# ---------------------------------------------------------------------------
# Coverage, not span — the finding this item was resized around
# ---------------------------------------------------------------------------
def test_coverage_counts_observed_days_and_names_the_gaps(store):
    """`(max - min).days` cannot see a hole; this must.

    Mirrors the live shape measured 2026-07-29: fewer rows than the calendar
    window, with the missing days sitting between the endpoints.
    """
    _write(store, [
        _row("2026-07-01", "AAPL", 10),
        _row("2026-07-02", "AAPL", 10),
        # 3rd and 4th missing
        _row("2026-07-05", "AAPL", 10),
    ])
    cov = pr.get_coverage()

    assert cov["observed_days"] == 3
    assert cov["calendar_days"] == 5
    assert cov["missing_days"] == 2
    assert cov["gaps"] == [{"after": "2026-07-02", "before": "2026-07-05", "missing_days": 2}]


def test_a_change_observed_across_a_gap_is_flagged_as_unattributable(store):
    """A delta spanning missing days cannot be dated to one of them — and a
    chain-linked return would otherwise span it silently."""
    _write(store, [_row("2026-07-01", "AAPL", 10), _row("2026-07-05", "AAPL", 25)])
    res = pr.get_reconciliation()

    assert res["spans_gap"] is True
    assert res["changes"][0]["gap_days"] == 4
    assert "cannot be attributed to a single date" in res["note"]


def test_adjacent_snapshots_are_not_flagged_as_spanning_a_gap(store):
    _write(store, [_row("2026-07-01", "AAPL", 10), _row("2026-07-02", "AAPL", 25)])
    res = pr.get_reconciliation()

    assert res["spans_gap"] is False
    assert res["changes"][0]["gap_days"] == 1


def test_coverage_on_an_empty_store_is_zero_not_an_error(store):
    cov = pr.get_coverage()
    assert cov["observed_days"] == 0
    assert cov["calendar_days"] == 0
    assert cov["gaps"] == []


def test_the_sync_error_sentinel_is_not_counted_as_an_unreadable_holding(store):
    """`load_portfolio` appends `{"_sync_errors": [...]}` to the holdings LIST.

    It is metadata, not a position. Counting it as an unreadable holding would
    add exactly one phantom parse failure every single day — which is precisely
    how a real parse failure stops being visible. Measured on the live portfolio:
    30 list entries, 29 positions, and the 30th was this.
    """
    holdings = [
        {"symbol": "AAPL", "shares": 10, "account": "TFSA", "currency": "USD"},
        {"_sync_errors": ["Questrade integration is disabled in Settings."]},
    ]
    with patch("tools.portfolio_csv.load_portfolio", return_value=holdings):
        report = pr.snapshot_positions()

    assert report["recorded"] == 1
    assert report["skipped_unreadable"] == 0, "the sentinel is not an unreadable holding"
    assert report["positions"] == 1, "the sentinel is not a position"
    # And the errors are surfaced, because a failed broker sync means the day's
    # snapshot may be PARTIAL — which reconciles as a mass disposal tomorrow.
    assert report["sync_errors"] == ["Questrade integration is disabled in Settings."]


def test_a_partial_sync_still_records_what_it_could_read(store):
    """The snapshot is best-effort by design: a broker outage must not cost the
    day entirely, since nothing can backfill it."""
    holdings = [
        {"symbol": "AAPL", "shares": 10, "account": "TFSA", "currency": "USD"},
        {"_sync_errors": ["Questrade token expired"]},
    ]
    with patch("tools.portfolio_csv.load_portfolio", return_value=holdings):
        report = pr.snapshot_positions()

    assert report["recorded"] == 1
    assert report["sync_errors"]
