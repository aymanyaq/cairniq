"""The observation log and its deterministic detectors (Advisor Roadmap 1.7).

The store this replaces was near-mute: 2 key facts against 20 conversation
summaries on the live profile, because the extractor judged one isolated message
on the FIRST supervisor pass — before any tool ran and before the answer existed.
These tests pin the properties that failure argues for: the writers are
deterministic, they carry evidence back to the turn they came from, they refuse
to invent a behavioural record when the underlying data is unreadable, and a
private turn writes nothing at all.

Everything here is monkeypatched onto tmp_path. Nothing may touch a live profile.
"""
from datetime import datetime, timedelta

import pytest

import tools.observations as obs


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Point the observation log at tmp_path."""
    monkeypatch.setattr(obs, "get_data_path", lambda filename: str(tmp_path / filename))
    return tmp_path


def _kinds(rows):
    return [r["kind"] for r in rows]


# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------

def test_uppercase_tickers_are_found_and_prose_words_are_not():
    found = obs.extract_tickers("Should I sell ABCD.TO and add YYYY? The CEO said no.")
    assert found == ["ABCD.TO", "YYYY"]


def test_a_held_name_is_matched_case_insensitively():
    """Users type their own holdings in lower case constantly; without this the
    log would attribute none of it."""
    assert obs.extract_tickers("whats up with pltr today", {"PLTR": 10.0}) == ["PLTR"]


def test_an_unheld_lowercase_word_is_not_promoted_to_a_ticker():
    assert obs.extract_tickers("should i buy more", {"PLTR": 10.0}) == []


# ---------------------------------------------------------------------------
# Per-turn writer
# ---------------------------------------------------------------------------

def test_an_ordinary_turn_records_one_asked_observation(store):
    written = obs.observe_turn("How is YYYY looking?", thread_id="t-1", interaction_id="i-1")

    assert _kinds(written) == [obs.KIND_ASKED]
    row = obs.load_observations()["observations"][0]
    assert row["tickers"] == ["YYYY"]
    assert row["thread_id"] == "t-1"
    assert row["interaction_id"] == "i-1"        # evidence pointer into feedback.json
    assert row["consolidated"] is False
    assert row["span"] == "How is YYYY looking?"


def test_pushback_is_recorded_only_when_there_was_an_answer_to_push_back_on(store):
    """A correction opening a thread is about something outside this app.
    Recording it would fabricate a grievance against an answer never given."""
    first = obs.observe_turn("No, that's wrong.", thread_id="t-1")
    assert _kinds(first) == [obs.KIND_ASKED]

    second = obs.observe_turn(
        "No, that's wrong. I hold it in the registered account.",
        thread_id="t-1",
        prior_answer="YYYY sits in your taxable account.",
    )
    assert obs.KIND_PUSHBACK in _kinds(second)


def test_a_pushback_row_carries_the_cue_and_the_users_own_words(store):
    written = obs.observe_turn(
        "You forgot the cost basis again.",
        thread_id="t-1",
        prior_answer="Here is the position.",
    )
    row = next(r for r in written if r["kind"] == obs.KIND_PUSHBACK)
    assert row["detail"]["cue"] == "You forgot"
    assert row["span"] == "You forgot the cost basis again."


def test_a_declined_suggestion_is_recorded(store):
    written = obs.observe_turn("I'm not selling ABCD.TO, leave it alone.", thread_id="t-1")
    assert obs.KIND_DECLINE in _kinds(written)


def test_an_ordinary_question_is_not_read_as_a_complaint(store):
    """Precision bar: a false pushback becomes fabricated evidence in a drafted
    rule, which is worse than missing the real ones."""
    written = obs.observe_turn(
        "What is the outlook for YYYY into earnings?",
        thread_id="t-1",
        prior_answer="Some earlier answer.",
    )
    assert _kinds(written) == [obs.KIND_ASKED]


def test_a_private_turn_writes_nothing(store):
    assert obs.observe_turn("@Private what do you think of YYYY?", thread_id="t-1") == []
    assert obs.load_observations()["observations"] == []


def test_an_empty_message_writes_nothing(store):
    assert obs.observe_turn("   ", thread_id="t-1") == []
    assert obs.load_observations()["observations"] == []


def test_the_log_is_capped(store, monkeypatch):
    monkeypatch.setattr(obs, "MAX_OBSERVATIONS", 5)
    for i in range(8):
        obs.observe_turn(f"question {i}", thread_id="t-1")
    rows = obs.load_observations()["observations"]
    assert len(rows) == 5
    assert rows[-1]["span"] == "question 7"


# ---------------------------------------------------------------------------
# rec_issued anchoring
# ---------------------------------------------------------------------------

def test_an_actionable_call_anchors_the_position_size(store):
    row = obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0, thread_id="t-1")
    assert row["kind"] == obs.KIND_REC_ISSUED
    assert row["detail"]["shares_at_advice"] == 40.0
    assert row["detail"]["resolved_by"] is None


def test_a_hold_is_not_anchored(store):
    """A HOLD has no observable follow-through — 'did nothing' is
    indistinguishable from 'ignored it', and recording one would manufacture an
    ignored-advice signal out of agreement."""
    assert obs.record_rec_issued("YYYY", "HOLD", shares_at_advice=40.0) is None
    assert obs.load_observations()["observations"] == []


# ---------------------------------------------------------------------------
# Follow-through sweep
# ---------------------------------------------------------------------------

def _age(store_path, observation_id, days):
    data = obs.load_observations()
    for row in data["observations"]:
        if row["id"] == observation_id:
            row["timestamp"] = (datetime.now() - timedelta(days=days)).isoformat()
    obs.save_observations(data)


def test_a_sell_the_user_acted_on_resolves_as_followed(store):
    row = obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0)
    _age(store, row["id"], days=10)

    summary = obs.resolve_rec_follow_through(holdings={"YYYY": 0.0})

    assert summary == {"resolved": 1, "followed": 1, "ignored": 0, "pending": 0}
    resolution = obs.load_observations()["observations"][-1]
    assert resolution["kind"] == obs.KIND_REC_FOLLOWED
    assert resolution["detail"]["shares_at_advice"] == 40.0
    assert resolution["detail"]["shares_now"] == 0.0
    assert resolution["detail"]["from_observation"] == row["id"]


def test_an_untouched_position_resolves_as_ignored(store):
    row = obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0)
    _age(store, row["id"], days=10)

    summary = obs.resolve_rec_follow_through(holdings={"YYYY": 40.0})

    assert summary["ignored"] == 1
    assert obs.load_observations()["observations"][-1]["kind"] == obs.KIND_REC_IGNORED


def test_a_position_moved_the_wrong_way_is_not_credited_as_followed(store):
    row = obs.record_rec_issued("YYYY", "BUY", shares_at_advice=40.0)
    _age(store, row["id"], days=10)

    obs.resolve_rec_follow_through(holdings={"YYYY": 10.0})

    assert obs.load_observations()["observations"][-1]["kind"] == obs.KIND_REC_IGNORED


def test_dust_is_not_a_decision(store):
    """A DRIP drip is not the user acting on advice."""
    row = obs.record_rec_issued("YYYY", "BUY", shares_at_advice=100.0)
    _age(store, row["id"], days=10)

    obs.resolve_rec_follow_through(holdings={"YYYY": 100.4})

    assert obs.load_observations()["observations"][-1]["kind"] == obs.KIND_REC_IGNORED


def test_a_call_inside_its_window_stays_open(store):
    obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0)

    summary = obs.resolve_rec_follow_through(holdings={"YYYY": 0.0})

    assert summary == {"resolved": 0, "followed": 0, "ignored": 0, "pending": 1}


def test_an_unreadable_portfolio_resolves_nothing(store, monkeypatch):
    """The guard that matters most. Reading a failed portfolio load as 'no
    position' would mark every open SELL as followed — a fabricated behavioural
    record about the user, in the store whose entire purpose is holding evidence
    about the user."""
    row = obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0)
    _age(store, row["id"], days=10)
    monkeypatch.setattr(obs, "load_holdings_map", lambda: None)

    summary = obs.resolve_rec_follow_through()

    assert summary["resolved"] == 0
    assert _kinds(obs.load_observations()["observations"]) == [obs.KIND_REC_ISSUED]


def test_an_unknown_starting_size_is_not_guessed(store):
    """shares_at_advice=None means the portfolio was unreadable at the time of
    the call. Treating it as zero would invent the starting position."""
    row = obs.record_rec_issued("YYYY", "BUY", shares_at_advice=None)
    _age(store, row["id"], days=10)

    summary = obs.resolve_rec_follow_through(holdings={"YYYY": 50.0})

    assert summary == {"resolved": 0, "followed": 0, "ignored": 0, "pending": 1}


def test_a_resolved_call_is_not_resolved_twice(store):
    row = obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0)
    _age(store, row["id"], days=10)

    obs.resolve_rec_follow_through(holdings={"YYYY": 0.0})
    second = obs.resolve_rec_follow_through(holdings={"YYYY": 0.0})

    assert second["resolved"] == 0
    assert len(obs.load_observations()["observations"]) == 2


def test_an_empty_portfolio_is_a_real_answer_not_an_unreadable_one(store):
    """{} and None are different states and the sweep has to tell them apart."""
    row = obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0)
    _age(store, row["id"], days=10)

    summary = obs.resolve_rec_follow_through(holdings={})

    assert summary["followed"] == 1


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------

def test_stats_report_the_gate_distance(store):
    for i in range(3):
        obs.observe_turn(f"question {i}", thread_id="t-1")

    stats = obs.get_observation_stats()

    assert stats["total"] == 3
    assert stats["unconsolidated"] == 3
    assert stats["gate_met"] is False
    assert stats["by_kind"][obs.KIND_ASKED] == 3
    assert stats["last_consolidated_at"] is None


def test_stats_report_zero_rather_than_nothing_on_an_empty_log(store):
    """A log nobody writes to and a log nobody reads have to look different."""
    stats = obs.get_observation_stats()
    assert stats["total"] == 0
    assert stats["by_kind"] == {kind: 0 for kind in obs.OBSERVATION_KINDS}


def test_marking_consolidated_moves_rows_out_of_the_unread_pool(store):
    rows = [obs.observe_turn(f"question {i}", thread_id="t-1")[0] for i in range(3)]

    marked = obs.mark_consolidated([rows[0]["id"], rows[1]["id"]])

    assert marked == 2
    assert len(obs.get_unconsolidated()) == 1
    assert obs.get_observation_stats()["last_consolidated_at"] is not None
