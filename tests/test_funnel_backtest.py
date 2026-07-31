"""Tests for the M5 Tier-1 backtest harness (tools.funnel_backtest)."""
import json
from datetime import UTC, datetime, timedelta, timezone

from tools.funnel_backtest import evaluate_signal_log, get_funnel_scorecard_data


def _write_snapshot(log_dir, ts, picks, near_misses=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"{ts.strftime('%Y-%m-%d')}.jsonl"
    with open(fname, "a") as f:
        f.write(json.dumps({
            "ts": ts.isoformat(),
            "market_status": "Risk-On",
            "picks": picks,
            "near_misses": near_misses or [],
        }) + "\n")


def test_no_data(tmp_path):
    r = evaluate_signal_log(log_dir=str(tmp_path / "empty"))
    assert r["status"] == "no_data"


def test_insufficient_maturity(tmp_path):
    # Snapshot from today → not matured for a 21-day window
    now = datetime.now(UTC)
    _write_snapshot(tmp_path, now, [{"symbol": "MU", "conviction": "High Conviction"}])
    r = evaluate_signal_log(days_forward=21, log_dir=str(tmp_path), as_of=now)
    assert r["status"] == "insufficient_maturity"
    assert r["snapshots"] == 1


def test_evaluation_with_mocked_returns(tmp_path):
    as_of = datetime(2026, 6, 1, tzinfo=UTC)
    snap_ts = as_of - timedelta(days=30)  # matured for 21d window
    _write_snapshot(tmp_path, snap_ts, [
        {"symbol": "MU",  "conviction": "High Conviction", "entry_stage": "accumulation_base"},
        {"symbol": "WDC", "conviction": "Qualified",       "entry_stage": "early_breakout"},
    ])

    # Mock forward returns: MU +20%, WDC -5%, SPY +3%
    returns = {"MU": 20.0, "WDC": -5.0, "SPY": 3.0}

    def fake_fwd(symbol, start, days_forward):
        return returns.get(symbol)

    r = evaluate_signal_log(days_forward=21, as_of=as_of, log_dir=str(tmp_path),
                            forward_return_fn=fake_fwd)
    assert r["status"] == "ok"
    assert r["evaluated_picks"] == 2
    # MU alpha = 20 - 3 = +17; WDC alpha = -5 - 3 = -8
    overall = r["overall"]
    assert overall["n"] == 2
    assert overall["hit_rate_vs_spy"] == 0.5     # MU wins, WDC loses
    assert overall["avg_alpha_pct"] == round((17 + -8) / 2, 2)
    # Per-conviction breakdown
    assert r["by_conviction"]["High Conviction"]["avg_alpha_pct"] == 17.0
    assert r["by_conviction"]["Qualified"]["avg_alpha_pct"] == -8.0
    # Per-entry-stage breakdown present
    assert "accumulation_base" in r["by_entry_stage"]


def test_unpriceable_picks_are_skipped(tmp_path):
    as_of = datetime(2026, 6, 1, tzinfo=UTC)
    snap_ts = as_of - timedelta(days=30)
    _write_snapshot(tmp_path, snap_ts, [{"symbol": "DEAD", "conviction": "High Conviction"}])

    def fake_fwd(symbol, start, days_forward):
        return None  # no price data for anything

    r = evaluate_signal_log(days_forward=21, as_of=as_of, log_dir=str(tmp_path),
                            forward_return_fn=fake_fwd)
    assert r["status"] == "no_priceable_picks"


def test_dedupe_scores_each_symbol_once_from_first_appearance(tmp_path):
    """Daily scans re-surface the same names; dedupe must keep only the first
    matured appearance so one signal can't pseudo-replicate 10x."""
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    for day in (1, 2, 3):
        _write_snapshot(tmp_path, datetime(2026, 6, day, tzinfo=UTC), [
            {"symbol": "MU", "conviction": "High Conviction", "entry_stage": "accumulation_base"},
        ])

    calls = []

    def fake_fwd(symbol, start, days_forward):
        calls.append((symbol, start.date().isoformat()))
        return {"MU": 10.0, "SPY": 2.0}.get(symbol)

    r = evaluate_signal_log(days_forward=14, as_of=as_of, log_dir=str(tmp_path),
                            forward_return_fn=fake_fwd)
    assert r["status"] == "ok"
    assert r["evaluated_picks"] == 1
    assert r["duplicate_rows_dropped"] == 2
    assert r["detail"][0]["snapshot"] == "2026-06-01"  # first surfacing wins

    # dedupe=False keeps the raw every-row view
    r_all = evaluate_signal_log(days_forward=14, as_of=as_of, log_dir=str(tmp_path),
                                forward_return_fn=fake_fwd, dedupe=False)
    assert r_all["evaluated_picks"] == 3
    assert r_all["duplicate_rows_dropped"] == 0


def test_sector_relative_alpha_uses_theme_benchmark(tmp_path):
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    _write_snapshot(tmp_path, datetime(2026, 6, 1, tzinfo=UTC), [
        {"symbol": "NVDA", "conviction": "High Conviction",
         "entry_stage": "early_breakout", "theme": "Technology"},
        {"symbol": "XOM", "conviction": "Qualified",
         "entry_stage": "mid_trend"},  # no theme -> no sector benchmark
    ])

    returns = {"NVDA": 10.0, "XOM": 4.0, "SPY": 2.0, "XLK": 6.0}

    def fake_fwd(symbol, start, days_forward):
        return returns.get(symbol)

    r = evaluate_signal_log(days_forward=14, as_of=as_of, log_dir=str(tmp_path),
                            forward_return_fn=fake_fwd)
    by_sym = {row["symbol"]: row for row in r["detail"]}
    nvda = by_sym["NVDA"]
    assert nvda["sector_benchmark"] == "XLK"
    assert nvda["alpha_pct"] == 8.0          # 10 - SPY 2
    assert nvda["sector_alpha_pct"] == 4.0   # 10 - XLK 6
    xom = by_sym["XOM"]
    assert xom["sector_benchmark"] is None
    assert xom["sector_alpha_pct"] is None
    # aggregates expose the sector-relative view where available
    assert r["overall"]["avg_sector_alpha_pct"] == 4.0
    assert r["overall"]["hit_rate_vs_sector"] == 1.0


def test_near_misses_scored_with_regret(tmp_path):
    """Miss detector: cut names are scored like picks, grouped by cut gate,
    and the regret delta vs picks is computed."""
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    _write_snapshot(
        tmp_path, datetime(2026, 6, 1, tzinfo=UTC),
        picks=[{"symbol": "NUE", "conviction": "High Conviction",
                "entry_stage": "accumulation_base", "theme": "Materials"}],
        near_misses=[
            {"symbol": "MU", "cut_reason": "entry_gate_demotion",
             "entry_stage": "extended", "theme": "Technology"},
            {"symbol": "OXY", "cut_reason": "risk_flag",
             "entry_stage": "early_breakout", "theme": "Energy"},
        ],
    )

    returns = {"NUE": 2.0, "MU": 30.0, "OXY": -6.0,
               "SPY": 1.0, "XLB": 1.0, "XLK": 5.0, "XLE": -2.0}

    def fake_fwd(symbol, start, days_forward):
        return returns.get(symbol)

    r = evaluate_signal_log(days_forward=14, as_of=as_of, log_dir=str(tmp_path),
                            forward_return_fn=fake_fwd)
    nm = r["near_misses"]
    # MU sector alpha = 30 - 5 = 25; OXY = -6 - (-2) = -4 → avg 10.5
    assert nm["overall"]["avg_sector_alpha_pct"] == 10.5
    assert nm["by_cut_reason"]["entry_gate_demotion"]["avg_sector_alpha_pct"] == 25.0
    assert nm["by_cut_reason"]["risk_flag"]["avg_sector_alpha_pct"] == -4.0
    # picks sector alpha: NUE 2 - XLB 1 = 1.0 → regret = 10.5 - 1.0 = 9.5
    assert nm["regret_vs_picks_sector_alpha_pct"] == 9.5
    assert "REGRET" in nm["regret_note"]

    # Near misses never contaminate the picks aggregates
    assert r["evaluated_picks"] == 1
    assert r["overall"]["n"] == 1


def test_no_near_misses_key_when_none_logged(tmp_path):
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    _write_snapshot(tmp_path, datetime(2026, 6, 1, tzinfo=UTC),
                    picks=[{"symbol": "MU", "conviction": "Qualified"}])

    def fake_fwd(symbol, start, days_forward):
        return {"MU": 5.0, "SPY": 1.0}.get(symbol)

    r = evaluate_signal_log(days_forward=14, as_of=as_of, log_dir=str(tmp_path),
                            forward_return_fn=fake_fwd)
    assert "near_misses" not in r


def test_collect_near_misses_attributes_cut_gates():
    from tools.opportunity_scanner import _collect_near_misses

    selected = [{"symbol": "WIN", "score": 80}]
    ranked_finalists = [
        {"symbol": "WIN", "score": 80},
        {"symbol": "GATED", "score": 35, "entry_multiplier": 0.6,
         "entry_stage": "extended", "theme": "Technology"},
        {"symbol": "RISKY", "score": 55, "entry_multiplier": 1.0,
         "risk_flags": ["headwind"], "score_breakdown": {"risk_adjust": -8}},
        {"symbol": "LOWSCORE", "score": 30, "entry_multiplier": 1.0},
        {"symbol": "CROWDED", "score": 62, "entry_multiplier": 1.0},
    ]
    all_scored = ranked_finalists + [
        {"symbol": "NEXTUP", "score": 58},
    ]
    misses = _collect_near_misses(ranked_finalists, selected, all_scored)
    by_sym = {m["symbol"]: m for m in misses}
    assert by_sym["GATED"]["cut_reason"] == "entry_gate_demotion"
    assert by_sym["RISKY"]["cut_reason"] == "risk_flag"
    assert by_sym["LOWSCORE"]["cut_reason"] == "below_score_threshold"
    assert by_sym["CROWDED"]["cut_reason"] == "outside_top_n"
    assert by_sym["NEXTUP"]["cut_reason"] == "not_finalist"
    assert "WIN" not in by_sym  # selected picks are never near-misses


def test_scorecard_compacts_and_caveats(tmp_path):
    _write_snapshot(tmp_path, datetime(2026, 6, 1, tzinfo=UTC), [
        {"symbol": "MU", "conviction": "High Conviction",
         "entry_stage": "accumulation_base", "theme": "Technology"},
    ])

    import tools.funnel_backtest as fb

    def fake_fwd(symbol, start, days_forward):
        return {"MU": 10.0, "SPY": 2.0, "XLK": 6.0}.get(symbol)

    orig = fb._forward_return
    fb._forward_return = fake_fwd
    try:
        # patch "now" indirectly: signals from 2026-06-01 matured long ago in test time?
        # get_funnel_scorecard_data uses real now(); the 2026-06-01 snapshot is matured
        # for both horizons relative to any later date, so this holds.
        sc = get_funnel_scorecard_data(horizons=(14, 21), log_dir=str(tmp_path))
    finally:
        fb._forward_return = orig

    assert sc["status"] == "ok"
    h14 = sc["horizons"]["14d"]
    assert h14["unique_signals"] == 1
    assert "detail" not in h14
    assert any("Small sample" in c for c in sc["caveats"])
    assert any("regime" in c for c in sc["caveats"])
