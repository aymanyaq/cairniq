import copy
from datetime import datetime, timedelta
from typing import Any

import pytest

from tools.memory import (
    DEFAULT_MEMORY,
    add_conversation_summary,
    add_recommendation,
    get_advisor_scorecard,
    get_user_context,
    load_memory,
    save_memory,
    score_past_recommendations,
)


def test_advice_ledger_flow(monkeypatch):
    # Setup clean temp memory
    test_memory = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "key_facts": [],
        "lessons_learned": [],
        "past_recommendations": []
    }

    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)

    saved_memories = []
    def mock_save(m):
        saved_memories.append(m)
        test_memory.update(m)

    monkeypatch.setattr("tools.memory.save_memory", mock_save)

    # 1. Add recommendations
    # We add one recommendation from 20 days ago (should be scored at 2w)
    date_20d_ago = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    # We add one recommendation from 5 days ago (should be pending)
    date_5d_ago = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    add_recommendation(
        ticker="AAPL",
        action="BUY",
        reason="Strong cash flow",
        price_at_advice=150.0,
        confidence_grade="HIGH",
        horizon="Medium Term"
    )

    # Overwrite the date manually to simulate elapsed time
    test_memory["past_recommendations"][0]["date"] = date_20d_ago

    add_recommendation(
        ticker="TSLA",
        action="SELL",
        reason="High valuation",
        price_at_advice=200.0,
        confidence_grade="MEDIUM",
        horizon="Short Term"
    )
    test_memory["past_recommendations"][1]["date"] = date_5d_ago

    assert len(test_memory["past_recommendations"]) == 2
    assert test_memory["past_recommendations"][0]["ticker"] == "AAPL"
    assert test_memory["past_recommendations"][1]["ticker"] == "TSLA"

    # 2. Mock _forward_return
    # AAPL (BUY): AAPL returned +10%, SPY returned +5% -> alpha = +5% -> HIT
    # TSLA (SELL): TSLA returned -5%, SPY returned +2% -> alpha = -7% -> HIT (SELL with negative alpha is a hit since it saved money)
    def mock_forward_return(symbol: str, start: datetime, days_forward: int) -> float | None:
        if symbol == "AAPL":
            return 10.0
        elif symbol == "TSLA":
            return -5.0
        elif symbol == "SPY":
            if days_forward == 14:
                return 5.0
            return 2.0
        return None

    monkeypatch.setattr("tools.funnel_backtest._forward_return", mock_forward_return)

    # 3. Run scoring
    updated = score_past_recommendations(test_memory)
    assert updated is True

    # Verify AAPL (2w) has been scored
    scores_aapl = test_memory["past_recommendations"][0]["scores"]
    assert "2w" in scores_aapl
    assert scores_aapl["2w"]["perf"] == 10.0
    assert scores_aapl["2w"]["spy_perf"] == 5.0
    assert scores_aapl["2w"]["alpha"] == 5.0

    # Verify TSLA (2w) is not scored because only 5 days elapsed
    scores_tsla = test_memory["past_recommendations"][1]["scores"]
    assert "2w" not in scores_tsla

    # 4. Generate scorecard
    scorecard = get_advisor_scorecard()
    assert "### 📊 Advisor Performance Scorecard" in scorecard
    assert "AAPL" in scorecard
    assert "TSLA" in scorecard
    assert "HIGH" in scorecard
    assert "100.0%" in scorecard # 1 hit out of 1 scored call

    # 5. Verify context injection
    context = get_user_context()
    assert "-- ADVISOR CALIBRATION TRACK RECORD --" in context
    assert "HIGH-confidence calls hit 100.0%" in context


def test_hold_recommendation_scored_with_long_bias(monkeypatch):
    """Regression: HOLD is a bullish "keep your exposure" stance, not a bearish
    one — a HOLD on a name that goes on to beat SPY must score as a HIT, not a
    miss (the prior bug graded HOLD with SELL semantics)."""
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "key_facts": [],
        "lessons_learned": [],
        "past_recommendations": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_recommendation(
        ticker="NVDA",
        action="HOLD",
        reason="Stay the course",
        price_at_advice=900.0,
        confidence_grade="HIGH",
        horizon="Medium Term",
    )
    date_20d_ago = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    test_memory["past_recommendations"][0]["date"] = date_20d_ago

    # NVDA beats SPY: a correct "keep holding" call.
    def mock_forward_return(symbol: str, start: datetime, days_forward: int) -> float | None:
        if symbol == "NVDA":
            return 8.0
        if symbol == "SPY":
            return 2.0
        return None

    monkeypatch.setattr("tools.funnel_backtest._forward_return", mock_forward_return)

    assert score_past_recommendations(test_memory) is True
    assert test_memory["past_recommendations"][0]["scores"]["2w"]["alpha"] == 6.0

    scorecard = get_advisor_scorecard()
    assert "100.0%" in scorecard  # 1/1 hit, not 0/1

    context = get_user_context()
    assert "HIGH-confidence calls hit 100.0%" in context


def test_add_recommendation_dedupes_same_day_restatement(monkeypatch):
    """Regression: restating the same ticker+action later in the same
    conversation must update the existing entry, not append a duplicate that
    would get scored (and counted toward the hit rate) multiple times."""
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "key_facts": [],
        "lessons_learned": [],
        "past_recommendations": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_recommendation(ticker="AAPL", action="BUY", reason="Strong cash flow", price_at_advice=150.0)
    add_recommendation(ticker="AAPL", action="BUY", reason="Still bullish, cash flow holds", price_at_advice=152.0)
    add_recommendation(ticker="AAPL", action="BUY", reason="Third mention, same day", price_at_advice=151.0)

    recs = test_memory["past_recommendations"]
    assert len(recs) == 1
    assert recs[0]["reason"] == "Third mention, same day"
    assert recs[0]["price_at_advice"] == 151.0

    # A different action on the same ticker the same day is a genuinely new call.
    add_recommendation(ticker="AAPL", action="SELL", reason="Changed my mind")
    assert len(test_memory["past_recommendations"]) == 2


def test_opposite_bias_call_supersedes_prior_open_call(monkeypatch):
    """A SELL/TRIM closes out a prior open BUY/ADD/HOLD for the same ticker so the
    contradicted long call is flagged superseded (and dropped from injected context)
    instead of coexisting with the exit that closed it."""
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [],
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_recommendation(ticker="YYYY", action="BUY", reason="AI memory tailwind", price_at_advice=940.0)
    add_recommendation(ticker="YYYY", action="SELL", reason="Insider selling invalidates entry", price_at_advice=1006.0)

    recs = {(r["action"], r.get("superseded", False)) for r in test_memory["past_recommendations"]}
    assert ("BUY", True) in recs        # the prior long call is closed out
    assert ("SELL", False) in recs      # the exit itself stays open

    # A consistent same-bias follow-up (HOLD after BUY) must NOT supersede.
    test_memory["past_recommendations"] = []
    add_recommendation(ticker="NVDA", action="BUY", reason="x", price_at_advice=100.0)
    add_recommendation(ticker="NVDA", action="HOLD", reason="stay", price_at_advice=105.0)
    assert all(not r.get("superseded") for r in test_memory["past_recommendations"])


def test_reduce_call_stamps_exit_signal_on_matching_active_thesis(monkeypatch):
    """A logged SELL/TRIM for a ticker with an open user-pinned LONG thesis on a HELD
    name stamps an exit_signal on that thesis (base-symbol match across exchange
    suffixes) — without deleting the user's entry — so the contradiction is surfaced,
    not silently kept. Only a held name can be contradicted; the not-held case is
    covered in tests/test_tools/test_thesis_position_state.py."""
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [],
        "active_theses": [
            {"id": "a1", "symbol": "KEEL", "action": "BUY", "stop_loss": "2.50"},
            {"id": "a2", "symbol": "AAPL", "action": "BUY", "stop_loss": "150"},
        ],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))
    monkeypatch.setattr("tools.memory._held_base_symbols", lambda: {"KEEL", "AAPL"})

    # Exchange-suffixed ticker must still reconcile the bare-symbol thesis.
    add_recommendation(ticker="KEEL.TO", action="SELL", reason="Thesis broken", price_at_advice=6.9)

    theses = {t["symbol"]: t for t in test_memory["active_theses"]}
    assert theses["KEEL"].get("exit_signal", {}).get("action") == "SELL"
    assert "exit_signal" not in theses["AAPL"]  # untouched ticker unaffected
    assert theses["KEEL"] in test_memory["active_theses"]  # thesis NOT deleted

    # A user edit (the reconciliation act) clears the pending exit_signal so the
    # CONTRADICTED flag stops firing once they've addressed it.
    from tools.memory import update_active_thesis
    assert update_active_thesis("a1", {"stop_loss": "6.00"}) is True
    keel_after = next(t for t in test_memory["active_theses"] if t["symbol"] == "KEEL")
    assert "exit_signal" not in keel_after


def test_superseded_recommendations_excluded_from_context(monkeypatch):
    """Superseded (closed-out) calls must not appear in the injected prior-calls
    block, and the block must be framed as past calls to re-verify."""
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [],
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_recommendation(ticker="YYYY", action="BUY", reason="entry", price_at_advice=940.0)
    add_recommendation(ticker="YYYY", action="SELL", reason="exit", price_at_advice=1006.0)

    context = get_user_context()
    assert "RE-VERIFY" in context
    # The closed BUY line is gone; only the live SELL survives in the block.
    assert "BUY YYYY" not in context
    assert "SELL YYYY" in context


def test_superseded_calls_excluded_from_scoring_and_calibration(monkeypatch):
    """A call reversed by a later opposite-bias call must be excluded from the
    scorer, the scorecard hit-rate, and the calibration block injected into every
    prompt — otherwise a call we exited early distorts the self-grade the model
    calibrates against."""
    date_20d_ago = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [
            # A superseded (reversed) HIGH-confidence BUY, already scored as a "hit".
            {"date": date_20d_ago, "ticker": "YYYY", "action": "BUY", "price_at_advice": 940.0,
             "confidence_grade": "HIGH", "horizon": "Medium Term", "superseded": True,
             "scores": {"2w": {"perf": 10.0, "spy_perf": 2.0, "alpha": 8.0}}},
            # A standing HIGH-confidence BUY that is a genuine miss (lagged SPY).
            {"date": date_20d_ago, "ticker": "AAPL", "action": "BUY", "price_at_advice": 150.0,
             "confidence_grade": "HIGH", "horizon": "Medium Term",
             "scores": {"2w": {"perf": 1.0, "spy_perf": 5.0, "alpha": -4.0}}},
        ],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    # Scorecard: only the standing AAPL miss counts → 0% hit rate, not 50%.
    scorecard = get_advisor_scorecard()
    assert "YYYY (closed)" in scorecard        # shown for history
    assert "0.0%" in scorecard               # superseded YYYY hit is NOT counted
    assert "50.0%" not in scorecard

    # Calibration block injected into every prompt reflects the same exclusion.
    context = get_user_context()
    assert "HIGH-confidence calls hit 0.0%" in context


def test_stale_flag_survives_price_fetch_failure(monkeypatch):
    """A live-price lookup failure must not wipe the network-free STALE flag — a
    months-old thesis has to keep warning even when the quote can't be fetched."""
    from tools import memory as memory_module

    old_date = (datetime.now() - timedelta(days=120)).isoformat()
    thesis = {"symbol": "ZZZ", "action": "BUY", "conditions": "core", "stop_loss": "10",
              "created_at": old_date}

    def boom(symbol):
        raise RuntimeError("quote service down")

    monkeypatch.setattr("tools.market_data.get_stock_data", boom)
    enriched = memory_module._enrich_thesis_with_price_context(dict(thesis))

    assert any("STALE" in f for f in enriched["_health_flags"])
    assert enriched.get("_live_price") is None


def test_add_conversation_summary_replaces_per_thread(monkeypatch):
    """Regression: re-summarizing the same thread every turn must replace that
    thread's entry, not append a new near-duplicate each time (which would flood
    the 20-entry cap and evict every other session's summary)."""
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "conversation_summaries": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_conversation_summary("Turn 1: discussed AAPL", thread_id="thread-a")
    add_conversation_summary("Turn 2: discussed AAPL and TSLA", thread_id="thread-a")
    add_conversation_summary("Turn 1: discussed a rate-cut thesis", thread_id="thread-b")

    summaries = test_memory["conversation_summaries"]
    assert len(summaries) == 2
    by_thread = {s["thread_id"]: s["summary"] for s in summaries}
    assert by_thread["thread-a"] == "Turn 2: discussed AAPL and TSLA"
    assert by_thread["thread-b"] == "Turn 1: discussed a rate-cut thesis"


def test_cross_day_restatement_collapses_into_anchor(monkeypatch):
    """Re-affirming an unchanged stance on a later day must update the existing
    entry (stamping last_affirmed) while keeping the ORIGINAL date and price as
    the scoring anchor — daily restated HOLDs were appending ~5 rows/day, which
    churned the capped ledger in ~9 days so no entry ever reached the 14-day
    scoring horizon (live scored=0 root cause)."""
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [],
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_recommendation(ticker="AAPL", action="HOLD", reason="Core position", price_at_advice=100.0, confidence_grade="MEDIUM")
    anchor_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    test_memory["past_recommendations"][0]["date"] = anchor_date

    add_recommendation(ticker="AAPL", action="HOLD", reason="Still core", price_at_advice=110.0, confidence_grade="HIGH")

    recs = test_memory["past_recommendations"]
    assert len(recs) == 1, "cross-day restatement must collapse, not append"
    assert recs[0]["date"] == anchor_date, "scoring anchor date must not reset"
    assert recs[0]["price_at_advice"] == 100.0, "anchor price must not reset"
    assert recs[0]["last_affirmed"] == datetime.now().strftime("%Y-%m-%d")
    assert recs[0]["confidence_grade"] == "HIGH"
    assert recs[0]["reason"] == "Still core"

    # A reversal closes the stance; a LATER re-entry of the same action starts a
    # fresh entry with a fresh anchor — it must NOT collapse into the closed one.
    add_recommendation(ticker="AAPL", action="SELL", reason="Exit", price_at_advice=120.0)
    add_recommendation(ticker="AAPL", action="HOLD", reason="Back in", price_at_advice=125.0)
    recs = test_memory["past_recommendations"]
    holds = [r for r in recs if r["action"] == "HOLD"]
    assert len(holds) == 2
    assert holds[0].get("superseded") is True
    assert holds[1].get("superseded") is None and holds[1]["price_at_advice"] == 125.0


def test_trim_keeps_maturing_entries_beyond_completed_cap(monkeypatch):
    """The ledger cap must never evict an entry the scorer hasn't finished:
    fully-scored entries rotate out at 50, but unscored in-flight entries all
    survive regardless of how many completed rows exist."""
    today = datetime.now()
    completed = [
        {
            "date": (today - timedelta(days=20)).strftime("%Y-%m-%d"),
            "ticker": f"C{i:02d}", "action": "BUY", "price_at_advice": 10.0,
            "scores": {"2w": {}, "1m": {}, "3m": {}},
        }
        for i in range(60)
    ]
    in_flight = [
        {
            "date": (today - timedelta(days=i + 1)).strftime("%Y-%m-%d"),
            "ticker": f"F{i:02d}", "action": "BUY", "price_at_advice": 20.0,
            "scores": {},
        }
        for i in range(5)
    ]
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": completed + in_flight,
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_recommendation(ticker="NEWCO", action="BUY", reason="fresh call", price_at_advice=30.0)

    recs = test_memory["past_recommendations"]
    tickers = [r["ticker"] for r in recs]
    # All 5 in-flight + the new call survive.
    for i in range(5):
        assert f"F{i:02d}" in tickers
    assert "NEWCO" in tickers
    # Completed rows are capped at 50 — the 10 oldest rotate out.
    assert len(recs) == 56
    assert "C00" not in tickers and "C09" not in tickers
    assert "C10" in tickers and "C59" in tickers


def test_stale_marker_uses_last_affirmed(monkeypatch):
    """A stance anchored days ago but re-affirmed today was re-derived today —
    the injected context must not flag it 'likely stale'. An old entry with no
    re-affirmation still gets the stale marker."""
    today = datetime.now()
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [
            {
                "date": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
                "ticker": "FRESH", "action": "HOLD", "price_at_advice": 50.0,
                "last_affirmed": today.strftime("%Y-%m-%d"), "scores": {},
            },
            {
                "date": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
                "ticker": "OLDCALL", "action": "BUY", "price_at_advice": 60.0,
                "scores": {},
            },
        ],
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)

    context = get_user_context()
    fresh_line = next(line for line in context.splitlines() if "FRESH" in line)
    old_line = next(line for line in context.splitlines() if "OLDCALL" in line)
    assert "likely stale" not in fresh_line
    assert "likely stale" in old_line


def test_scorer_memoizes_forward_return_per_symbol_date(monkeypatch):
    """Two recommendations sharing an advice date must fetch the SPY benchmark
    once per horizon, not once per recommendation."""
    date_20d_ago = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    test_memory: dict[str, Any] = {
        "past_recommendations": [
            {"date": date_20d_ago, "ticker": "AAPL", "action": "BUY", "scores": {}},
            {"date": date_20d_ago, "ticker": "MSFT", "action": "BUY", "scores": {}},
        ],
    }
    calls: list[str] = []

    def mock_forward_return(symbol: str, start: datetime, days_forward: int) -> float:
        calls.append(symbol)
        return 3.0

    monkeypatch.setattr("tools.funnel_backtest._forward_return", mock_forward_return)

    assert score_past_recommendations(test_memory) is True
    assert calls.count("SPY") == 1, f"SPY must be memoized per (date, horizon), got {calls}"
    assert calls.count("AAPL") == 1 and calls.count("MSFT") == 1


def test_trim_heals_preexisting_duplicate_stances(monkeypatch):
    """Ledgers written before the cross-day collapse hold the same open stance
    duplicated across days (live: HOLD AAPL re-logged ~daily). The next
    add_recommendation folds those into the earliest anchor — original
    date/price kept, newest restatement date recorded — while a supersede
    boundary keeps a genuine re-entry separate."""
    today = datetime.now()
    def d(n):
        return (today - timedelta(days=n)).strftime("%Y-%m-%d")
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [
            {"date": d(9), "ticker": "AAPL", "action": "HOLD", "price_at_advice": 100.0,
             "confidence_grade": "MEDIUM", "scores": {}},
            {"date": d(8), "ticker": "AAPL", "action": "HOLD", "price_at_advice": 104.0,
             "confidence_grade": "HIGH", "scores": {}},
            {"date": d(7), "ticker": "AAPL", "action": "HOLD", "price_at_advice": 108.0,
             "confidence_grade": "HIGH", "scores": {}},
            # Closed stance + re-entry: must stay TWO rows, not collapse across it.
            {"date": d(6), "ticker": "YYYY", "action": "BUY", "price_at_advice": 900.0,
             "superseded": True, "scores": {}},
            {"date": d(2), "ticker": "YYYY", "action": "BUY", "price_at_advice": 950.0,
             "scores": {}},
        ],
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    add_recommendation(ticker="NEWCO", action="BUY", reason="trigger trim", price_at_advice=10.0)

    recs = test_memory["past_recommendations"]
    holds = [r for r in recs if r["ticker"] == "AAPL"]
    assert len(holds) == 1, "duplicate open stances must fold into one anchor"
    assert holds[0]["date"] == d(9) and holds[0]["price_at_advice"] == 100.0
    assert holds[0]["last_affirmed"] == d(7)
    assert holds[0]["confidence_grade"] == "HIGH"
    mu = [r for r in recs if r["ticker"] == "YYYY"]
    assert len(mu) == 2, "a re-entry after a supersede is a new stance, not a restatement"


def test_get_scored_recommendations_data(monkeypatch):
    from tools.memory import get_scored_recommendations_data

    # Setup clean temp memory with scored recommendations
    test_memory = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "key_facts": [],
        "lessons_learned": [],
        "past_recommendations": [
            {
                "date": "2026-07-01",
                "ticker": "AAPL",
                "action": "BUY",
                "price_at_advice": 150.0,
                "confidence_grade": "HIGH",
                "horizon": "Medium Term",
                "reason": "Reason AAPL",
                "scores": {
                    "2w": {"perf": 10.0, "spy_perf": 5.0, "alpha": 5.0}
                }
            },
            {
                "date": "2026-07-02",
                "ticker": "TSLA",
                "action": "SELL",
                "price_at_advice": 200.0,
                "confidence_grade": "MEDIUM",
                "horizon": "Short Term",
                "reason": "Reason TSLA",
                "scores": {
                    "2w": {"perf": -5.0, "spy_perf": 2.0, "alpha": -7.0}
                }
            }
        ]
    }

    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))
    # Mock score_past_recommendations so it doesn't try to query online API or other metrics
    monkeypatch.setattr("tools.memory.score_past_recommendations", lambda m: False)

    data = get_scored_recommendations_data()

    assert "stats" in data
    assert "confidence_stats" in data
    assert "recommendations" in data

    # AAPL BUY: alpha +5.0 is a hit
    # TSLA SELL: alpha -7.0 is a hit (long_bias=False, hit is alpha < 0, rec_alpha = -alpha = 7.0)
    stats_2w = data["stats"]["2w"]
    assert stats_2w["total"] == 2
    assert stats_2w["hits"] == 2
    assert stats_2w["hit_rate"] == 100.0
    assert stats_2w["avg_alpha"] == 6.0  # (5.0 alpha + 7.0 alpha) / 2 = 6.0

    conf_high = data["confidence_stats"]["HIGH"]
    assert conf_high["total"] == 1
    assert conf_high["hits"] == 1
    assert conf_high["hit_rate"] == 100.0



def test_restatement_turn_also_heals_unrelated_duplicates(monkeypatch):
    """The healing collapse must run on the restatement write path too — the
    common daily turn IS a restatement, so waiting for the next genuinely-new
    ticker call would leave pre-collapse duplicate stances (live: 8) polluting
    the first scored cohort for days."""
    today = datetime.now()

    def d(n: int) -> str:
        return (today - timedelta(days=n)).strftime("%Y-%m-%d")

    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [
            {"date": d(9), "ticker": "MSFT", "action": "HOLD", "price_at_advice": 300.0, "scores": {}},
            {"date": d(8), "ticker": "MSFT", "action": "HOLD", "price_at_advice": 305.0, "scores": {}},
            {"date": d(7), "ticker": "AAPL", "action": "HOLD", "price_at_advice": 100.0, "scores": {}},
        ],
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))

    # A pure restatement of AAPL — no new stance appended anywhere.
    add_recommendation(ticker="AAPL", action="HOLD", reason="still fine", price_at_advice=101.0)

    recs = test_memory["past_recommendations"]
    msft = [r for r in recs if r["ticker"] == "MSFT"]
    assert len(msft) == 1, "unrelated duplicate stances must heal on a restatement turn"
    assert msft[0]["date"] == d(9) and msft[0]["price_at_advice"] == 300.0
    assert msft[0]["last_affirmed"] == d(8)
    aapl = [r for r in recs if r["ticker"] == "AAPL"]
    assert len(aapl) == 1 and aapl[0]["last_affirmed"] == today.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Roadmap 1.8 — grade a call before supersession retires it
#
# Measured on the live ledger 2026-07-26: 13 superseded recs had scored ZERO
# times, and 6 of the 8 mature ACTIONABLE calls were among them (YYYY SELL x3,
# ABCD.TO TRIM x3). A daily restatement reverses the open call before its horizon
# elapses, so the leg was retired unscored and the ledger recycled instead of
# accruing. These tests cover the leg being graded over the horizon ACTUALLY
# held, and — just as important — every case where it must be withheld.
# ---------------------------------------------------------------------------


class _FrozenClock(datetime):
    """A datetime whose now() is settable, so a test can drive the REAL write
    path across several days instead of hand-editing the fields under test."""

    _now = datetime(2026, 6, 1)

    @classmethod
    def now(cls, tz=None):  # noqa: D102 - drop-in for datetime.now
        return cls._now

    @classmethod
    def set(cls, y: int, m: int, d: int) -> None:
        cls._now = datetime(y, m, d)


def _fake_tape(bars: dict[str, list[tuple[str, float]]], calls: list | None = None):
    """A stand-in for yfinance.Ticker serving fixed daily bars.

    Only the NETWORK boundary is faked — window selection, entry/exit bar choice
    and the freshness gate all run for real inside _realized_return_between.
    """
    import pandas as pd

    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start=None, end=None, **kwargs):
            if calls is not None:
                calls.append((self.symbol, start, end))
            rows = bars.get(self.symbol, [])
            frame = pd.DataFrame(
                {"Close": [c for _, c in rows]},
                index=pd.to_datetime([d for d, _ in rows]),
            )
            if start:
                frame = frame[frame.index >= pd.Timestamp(start)]
            if end:  # yfinance treats `end` as exclusive
                frame = frame[frame.index < pd.Timestamp(end)]
            return frame

    return _FakeTicker


def _series(day_prices: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(day_prices.items())


# YYYY: flat at 100 into the reversal, 110 on the exit date, then a wild print that
# must NOT be picked up (proves the exit bar is the exit DATE, not the last row).
_YYYY_BARS = _series({
    "2026-05-27": 95.0, "2026-05-29": 98.0,
    "2026-06-01": 100.0, "2026-06-05": 104.0, "2026-06-10": 108.0,
    "2026-06-11": 110.0,
    "2026-06-12": 999.0, "2026-06-19": 999.0,
})
_SPY_BARS = _series({
    "2026-05-27": 99.0, "2026-05-29": 99.5,
    "2026-06-01": 100.0, "2026-06-05": 101.0, "2026-06-10": 101.5,
    "2026-06-11": 102.0,
    "2026-06-12": 500.0, "2026-06-19": 500.0,
})


def _ledger(monkeypatch) -> dict[str, Any]:
    test_memory: dict[str, Any] = {
        "user_profile": {"name": "Test User", "base_currency": "USD"},
        "past_recommendations": [],
        "active_theses": [],
    }
    monkeypatch.setattr("tools.memory.load_memory", lambda: test_memory)
    monkeypatch.setattr("tools.memory.save_memory", lambda m: test_memory.update(m))
    monkeypatch.setattr("tools.memory.datetime", _FrozenClock)
    return test_memory


def test_superseded_leg_is_graded_at_supersession_end_to_end(monkeypatch):
    """THE end-to-end test: add_recommendation -> real reconciliation/supersession
    -> scorer, with only the price feed faked.

    Nothing is hand-written into the ledger: the BUY, the SELL that supersedes it
    and the `superseded_by` stamp the scorer reads all come from the production
    write path, driven across three days by a frozen clock. A test that fed a
    synthetic superseded row straight to the scorer would pass even if
    add_recommendation stopped stamping the date the score depends on.
    """
    from tools.memory import get_scored_recommendations_data

    test_memory = _ledger(monkeypatch)

    _FrozenClock.set(2026, 6, 1)
    add_recommendation(ticker="YYYY", action="BUY", reason="entry", price_at_advice=100.0,
                       confidence_grade="HIGH", horizon="Medium Term")

    _FrozenClock.set(2026, 6, 11)
    add_recommendation(ticker="YYYY", action="SELL", reason="exit", price_at_advice=110.0,
                       confidence_grade="HIGH", horizon="Short Term")

    recs = test_memory["past_recommendations"]
    closed = [r for r in recs if r["action"] == "BUY"][0]
    assert closed["superseded"] is True
    assert closed["superseded_by"]["date"] == "2026-06-11"  # written by the real path
    assert closed["scores"] == {}

    # 19 days after the anchor: without 1.8 this leg is silently skipped forever,
    # and it is old enough that the 2w horizon would otherwise have fired.
    _FrozenClock.set(2026, 6, 20)
    monkeypatch.setattr("yfinance.Ticker", _fake_tape({"YYYY": _YYYY_BARS, "SPY": _SPY_BARS}))

    assert score_past_recommendations(test_memory) is True

    held = closed["scores"]["held"]
    assert held["held_days"] == 10                    # the horizon ACTUALLY held
    assert held["from"] == "2026-06-01" and held["to"] == "2026-06-11"
    assert held["perf"] == 10.0                       # 100 -> 110 on the exit date
    assert held["spy_perf"] == 2.0                    # 100 -> 102
    assert held["alpha"] == 8.0
    assert held["partial"] is True
    # A partial hold is NOT a full-horizon result: it never occupies a 2w/1m/3m slot.
    assert not {"2w", "1m", "3m"} & set(closed["scores"])

    # ... and it stays out of the pooled horizon stats and the prompt self-grade.
    data = get_scored_recommendations_data()
    assert data["stats"]["2w"]["total"] == 0
    assert data["partial_stats"]["total"] == 1
    assert data["partial_stats"]["hits"] == 1
    assert data["partial_stats"]["actionable"] == 1
    assert data["partial_stats"]["avg_held_days"] == 10.0

    scorecard = get_advisor_scorecard()
    assert "Closed Early" in scorecard
    assert "held 10d: +10.0% (+8.0% vs SPY)" in scorecard

    context = get_user_context()
    assert "ADVISOR CALIBRATION TRACK RECORD" not in context


def test_partial_hold_withheld_without_a_confirming_bar(monkeypatch):
    """5.8: a stale quote must not consume a scoring event.

    When the tape stops AT the exit date, the newest bar is not proof of the exit
    price — it is just where the data ends. The leg must stay unscored and be
    retried, not be graded against the last row that happens to be there.
    """
    test_memory = _ledger(monkeypatch)

    _FrozenClock.set(2026, 6, 1)
    add_recommendation(ticker="YYYY", action="BUY", reason="entry", price_at_advice=100.0)
    _FrozenClock.set(2026, 6, 11)
    add_recommendation(ticker="YYYY", action="SELL", reason="exit", price_at_advice=110.0)

    truncated = {
        "YYYY": [b for b in _YYYY_BARS if b[0] <= "2026-06-11"],
        "SPY": [b for b in _SPY_BARS if b[0] <= "2026-06-11"],
    }
    monkeypatch.setattr("yfinance.Ticker", _fake_tape(truncated))
    _FrozenClock.set(2026, 6, 12)

    score_past_recommendations(test_memory)
    closed = [r for r in test_memory["past_recommendations"] if r["action"] == "BUY"][0]
    assert "held" not in closed["scores"]

    # The very next day, once the tape has printed past the exit, it grades.
    monkeypatch.setattr("yfinance.Ticker", _fake_tape({"YYYY": _YYYY_BARS, "SPY": _SPY_BARS}))
    _FrozenClock.set(2026, 6, 13)
    assert score_past_recommendations(test_memory) is True
    assert closed["scores"]["held"]["perf"] == 10.0


def test_partial_hold_withheld_when_price_is_unavailable(monkeypatch):
    """Withhold rather than estimate: no bars, no score. Never an interpolated
    or carried price to complete the row."""
    test_memory = _ledger(monkeypatch)

    _FrozenClock.set(2026, 6, 1)
    add_recommendation(ticker="ZZZZ", action="BUY", reason="entry", price_at_advice=10.0)
    _FrozenClock.set(2026, 6, 11)
    add_recommendation(ticker="ZZZZ", action="SELL", reason="exit", price_at_advice=12.0)

    _FrozenClock.set(2026, 6, 20)
    monkeypatch.setattr("yfinance.Ticker", _fake_tape({"SPY": _SPY_BARS}))  # no ZZZZ bars

    score_past_recommendations(test_memory)
    closed = [r for r in test_memory["past_recommendations"] if r["action"] == "BUY"][0]
    assert closed["scores"] == {}


def test_partial_hold_withheld_without_a_supersession_date(monkeypatch):
    """A row that cannot say WHEN it was closed has no measurable holding period.
    The window is never inferred from last_affirmed or from 'now' — and the price
    feed must not even be consulted."""
    test_memory = _ledger(monkeypatch)
    test_memory["past_recommendations"] = [
        # Shape written before superseded_by existed: flag set, no closing date.
        {"date": "2026-06-01", "ticker": "YYYY", "action": "BUY", "price_at_advice": 100.0,
         "confidence_grade": "HIGH", "superseded": True, "last_affirmed": "2026-06-09",
         "scores": {}},
    ]
    calls: list = []
    monkeypatch.setattr("yfinance.Ticker", _fake_tape({"YYYY": _YYYY_BARS, "SPY": _SPY_BARS}, calls))
    _FrozenClock.set(2026, 6, 20)

    assert score_past_recommendations(test_memory) is False
    assert test_memory["past_recommendations"][0]["scores"] == {}
    assert calls == [], "an ungradeable leg must not reach the price feed at all"


def test_partial_hold_withheld_when_the_leg_was_barely_held(monkeypatch):
    """A reversal a day later is restatement flap, not a position. Grading it
    would drop a near-zero-information row into a small-n population — the same
    shape as the 4.3b zero-capital leg that moved a pooled hit rate."""
    test_memory = _ledger(monkeypatch)

    _FrozenClock.set(2026, 6, 1)
    add_recommendation(ticker="YYYY", action="BUY", reason="entry", price_at_advice=100.0)
    _FrozenClock.set(2026, 6, 2)
    add_recommendation(ticker="YYYY", action="SELL", reason="oops", price_at_advice=101.0)

    calls: list = []
    monkeypatch.setattr("yfinance.Ticker", _fake_tape({"YYYY": _YYYY_BARS, "SPY": _SPY_BARS}, calls))
    _FrozenClock.set(2026, 6, 20)

    score_past_recommendations(test_memory)
    closed = [r for r in test_memory["past_recommendations"] if r["action"] == "BUY"][0]
    assert "held" not in closed["scores"]
    # The guard must short-circuit BEFORE pricing, not price and then discard —
    # this scorer runs daily over the whole ledger and every flap leg would
    # otherwise cost two fetches against a rate-limited feed. The partial path
    # always fetches from the leg's ANCHOR date, so no call may start there.
    # (The surviving SELL is a live, un-superseded call: the regular scorer
    # pricing it from 06-02 is correct and is not what this test forbids.)
    assert all(start != "2026-06-01" for _, start, _ in calls)


def test_partial_backfill_is_idempotent_and_leaves_scored_rows_alone(monkeypatch):
    """The back-fill runs over a ledger that already holds real scores, daily.
    Re-running it must not rewrite a single existing number."""
    test_memory = _ledger(monkeypatch)
    already_scored = {
        "date": "2026-05-01", "ticker": "AAPL", "action": "BUY", "price_at_advice": 150.0,
        "confidence_grade": "HIGH", "horizon": "Medium Term",
        "scores": {"2w": {"perf": 1.0, "spy_perf": 5.0, "alpha": -4.0}},
    }
    test_memory["past_recommendations"] = [
        already_scored,
        {"date": "2026-06-01", "ticker": "YYYY", "action": "BUY", "price_at_advice": 100.0,
         "confidence_grade": "HIGH", "superseded": True,
         "superseded_by": {"date": "2026-06-11", "action": "SELL"}, "scores": {}},
    ]
    untouched = copy.deepcopy(already_scored)

    monkeypatch.setattr("yfinance.Ticker", _fake_tape({"YYYY": _YYYY_BARS, "SPY": _SPY_BARS}))
    _FrozenClock.set(2026, 6, 20)

    assert score_past_recommendations(test_memory) is True
    first = copy.deepcopy(test_memory["past_recommendations"][1]["scores"])

    # Second pass: nothing left to grade, nothing rewritten.
    assert score_past_recommendations(test_memory) is False
    assert test_memory["past_recommendations"][1]["scores"] == first
    assert test_memory["past_recommendations"][0] == untouched


def test_scheduler_reports_legs_graded_through_note_production(monkeypatch):
    """An engine that grades zero and one that never ran must not emit the same
    silence. The scoring task reports what it GRADED on every pass — including
    the pass that grades nothing — counted off the ledger, not self-reported."""
    import asyncio

    import tools.engine_heartbeat as heartbeat
    import tools.memory as mem
    import tools.scheduler as sched
    import tools.user_profile as up

    ledger = {
        "past_recommendations": [
            {"date": "2026-06-01", "ticker": "YYYY", "action": "BUY", "price_at_advice": 100.0,
             "superseded": True, "superseded_by": {"date": "2026-06-11", "action": "SELL"},
             "scores": {}},
        ]
    }
    reported: list[tuple[int, str]] = []
    monkeypatch.setattr(up, "list_available_profiles", lambda: [{"name": "default"}])
    monkeypatch.setattr(mem, "load_memory", lambda: ledger)
    monkeypatch.setattr(mem, "save_memory", lambda m: True)
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: True)
    monkeypatch.setattr(heartbeat, "note_production", lambda n, detail="": reported.append((n, detail)))
    monkeypatch.setattr("yfinance.Ticker", _fake_tape({"YYYY": _YYYY_BARS, "SPY": _SPY_BARS}))
    monkeypatch.setattr("tools.memory.datetime", _FrozenClock)
    _FrozenClock.set(2026, 6, 20)

    asyncio.run(sched.task_score_recommendations())
    assert reported, "the scorer must report to the heartbeat"
    produced, detail = reported[-1]
    assert produced == 1                       # 2.6: ledger rows walked
    assert "1 legs graded (1 at supersession)" in detail
    assert ledger["past_recommendations"][0]["scores"]["held"]["alpha"] == 8.0

    # Second run: nothing left to grade. The engine must still say so.
    reported.clear()
    asyncio.run(sched.task_score_recommendations())
    assert reported, "a pass that grades nothing must still report"
    assert "0 legs graded (0 at supersession)" in reported[-1][1]


# ---------------------------------------------------------------------------
# Partial-hold population: rows vs distinct calls (follow-on to 1.8, 2026-07-27)
# ---------------------------------------------------------------------------
#
# The live ledger's first graded partial population was 9 rows and 4 calls: YYYY
# SELL had been restated on five consecutive days before one reversal closed all
# five. Counting rows credited a single correct call five times.


def _graded(date, ticker, action, alpha, exit_date, held_days=5):
    return {
        "date": date, "ticker": ticker, "action": action, "price_at_advice": 100.0,
        "superseded": True, "superseded_by": {"date": exit_date, "action": "HOLD"},
        "scores": {"held": {
            "perf": alpha, "spy_perf": 0.0, "alpha": alpha,
            "held_days": held_days, "from": date, "to": exit_date, "partial": True,
        }},
    }


def test_a_call_restated_on_five_days_is_one_call_not_five():
    """The real shape from the live ledger. Five YYYY SELL rows, one reversal
    closing all of them: one decision, and a hit rate that counts it once."""
    from tools.memory import get_partial_hold_stats

    recs = [
        _graded("2026-07-10", "YYYY", "SELL", -11.77, "2026-07-17"),
        _graded("2026-07-11", "YYYY", "SELL", -8.61, "2026-07-17"),
        _graded("2026-07-12", "YYYY", "SELL", -8.61, "2026-07-17"),
        _graded("2026-07-13", "YYYY", "SELL", -8.61, "2026-07-17"),
        _graded("2026-07-14", "YYYY", "SELL", -12.51, "2026-07-17"),
    ]
    stats = get_partial_hold_stats(recs)

    assert stats["total"] == 1, "five restatements of one stance are one call"
    assert stats["graded_rows"] == 5, "the row count is still reported, never hidden"
    assert stats["actionable"] == 1
    # A SELL is right when the name UNDERperforms, so this is one hit, not five.
    assert stats["hits"] == 1
    assert stats["hit_rate"] == 100.0


def test_the_earliest_row_anchors_the_group():
    """The first time a call was made is the point the user could have acted."""
    from tools.memory import get_partial_hold_stats

    recs = [
        _graded("2026-07-14", "YYYY", "SELL", -12.51, "2026-07-17", held_days=3),
        _graded("2026-07-10", "YYYY", "SELL", -11.77, "2026-07-17", held_days=7),
    ]
    stats = get_partial_hold_stats(recs)

    assert stats["total"] == 1
    assert stats["avg_held_days"] == 7.0, "the anchor is the earliest row's window"
    assert stats["avg_alpha"] == 11.77


def test_a_re_entry_after_a_reversal_stays_its_own_call():
    """The distinction that makes the grouping a fact rather than a heuristic:
    same ticker and same action, but closed by DIFFERENT reversals, so the user
    made the call twice and was right or wrong twice."""
    from tools.memory import get_partial_hold_stats

    recs = [
        _graded("2026-07-09", "ABCD.TO", "TRIM", -0.72, "2026-07-10"),
        _graded("2026-07-10", "ABCD.TO", "TRIM", -0.50, "2026-07-14"),
        _graded("2026-07-11", "ABCD.TO", "TRIM", 0.46, "2026-07-14"),
    ]
    stats = get_partial_hold_stats(recs)

    assert stats["total"] == 2, "two separate stances, the second one restated"
    assert stats["graded_rows"] == 3


def test_a_row_with_no_exit_stamp_is_never_merged_into_another():
    """A missing superseded_by must not become a shared grouping key — that
    would fold unrelated calls together on the strength of a null."""
    from tools.memory import get_partial_hold_stats

    a = _graded("2026-07-10", "YYYY", "SELL", -5.0, "2026-07-17")
    b = _graded("2026-07-11", "YYYY", "SELL", -5.0, "2026-07-17")
    a["superseded_by"] = None
    b["superseded_by"] = None
    stats = get_partial_hold_stats([a, b])

    assert stats["total"] == 2
    assert stats["graded_rows"] == 2


def test_the_scorecard_says_calls_and_rows_when_they_differ(monkeypatch):
    """Say it where the number is READ. A reader shown only the row count is
    being told the sample is larger than it is."""
    test_memory = _ledger(monkeypatch)
    test_memory["past_recommendations"] = [
        _graded("2026-07-10", "YYYY", "SELL", -11.77, "2026-07-17"),
        _graded("2026-07-11", "YYYY", "SELL", -8.61, "2026-07-17"),
    ]
    monkeypatch.setattr(
        "tools.memory.score_past_recommendations", lambda m: False
    )

    md = get_advisor_scorecard()

    assert "Distinct Calls" in md
    assert "1 distinct calls** from 2 graded ledger rows" in md
