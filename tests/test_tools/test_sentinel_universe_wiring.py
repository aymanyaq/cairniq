"""The Sentinel's lite scan must actually reach the scan.

`_scan_opportunities_lite` wraps its scanner imports in a bare `except` that
returns []. When Funnel V2 M1 retired the static universe file, the
`_load_universe` import inside that block started raising ImportError on every
run — and the except swallowed it, so the Sentinel reported zero opportunities
unconditionally and looked merely quiet rather than broken. These tests fail
loudly on the import contract instead.
"""
import pytest

import tools.market_sentinel as ms


def test_scanner_symbols_the_sentinel_imports_still_exist():
    """Named explicitly so a scanner refactor breaks the test, not just prod."""
    scanner = pytest.importorskip("tools.opportunity_scanner")

    for name in (
        "_assemble_dynamic_universe",
        "_batch_download",
        "_compute_technicals_batch",
        "_fast_score",
        "_get_sector_for_ticker",
        "_get_thematic_tags",
        "_stable_unique_symbols",
    ):
        assert hasattr(scanner, name), f"opportunity_scanner lost {name}, used by market_sentinel"

    mechanics = pytest.importorskip("tools.market_mechanics")
    assert hasattr(mechanics, "detect_sector_rotation")


def test_lite_scan_reaches_the_download_stage(monkeypatch, capsys):
    """Proves the import block completed — the failure mode was never getting here."""
    import tools.market_mechanics as mm
    import tools.opportunity_scanner as sc

    monkeypatch.setattr(mm, "detect_sector_rotation", lambda: {})
    monkeypatch.setattr(sc, "_assemble_dynamic_universe", lambda *a, **k: (["NVDA", "AAPL"], {}))

    reached = {}

    def _fake_download(tickers, period=None):
        reached["tickers"] = list(tickers)
        return None  # empty frame -> early return, no network

    monkeypatch.setattr(sc, "_batch_download", _fake_download)

    assert ms._scan_opportunities_lite() == []
    assert reached["tickers"] == ["NVDA", "AAPL"]
    assert "Opportunity scan failed" not in capsys.readouterr().out


def test_empty_universe_short_circuits(monkeypatch):
    """No candidates is a clean skip, not a download of nothing."""
    import tools.market_mechanics as mm
    import tools.opportunity_scanner as sc

    monkeypatch.setattr(mm, "detect_sector_rotation", lambda: {})
    monkeypatch.setattr(sc, "_assemble_dynamic_universe", lambda *a, **k: ([], {}))

    def _never(*a, **k):
        raise AssertionError("_batch_download must not run on an empty universe")

    monkeypatch.setattr(sc, "_batch_download", _never)

    assert ms._scan_opportunities_lite() == []


def test_rotation_failure_does_not_abort_the_scan(monkeypatch):
    """Rotation is one of several universe sources — losing it must not be fatal."""
    import tools.market_mechanics as mm
    import tools.opportunity_scanner as sc

    def _boom():
        raise RuntimeError("rotation feed down")

    seen = {}

    monkeypatch.setattr(mm, "detect_sector_rotation", _boom)
    monkeypatch.setattr(sc, "_assemble_dynamic_universe",
                        lambda rotation_data, *a, **k: (seen.setdefault("rotation", rotation_data), ["NVDA"])[1])
    monkeypatch.setattr(sc, "_batch_download", lambda *a, **k: None)

    assert ms._scan_opportunities_lite() == []
    assert seen["rotation"] == {}
