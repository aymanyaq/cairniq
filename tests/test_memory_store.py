import json
import os

import pytest

from tools import memory as memory_module
from tools.user_profile import ensure_demo_profile, get_data_path, reset_profile, set_active_profile


@pytest.fixture
def clean_memory_file():
    memory_path = get_data_path("user_memory.json")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    if os.path.exists(memory_path):
        os.remove(memory_path)

    yield memory_path

    if os.path.exists(memory_path):
        os.remove(memory_path)


def test_load_memory_returns_isolated_default_structures(clean_memory_file):
    first = memory_module.load_memory()
    first["key_facts"].append("mutated fact")
    first["user_profile"]["investment_goals"].append("mutated goal")

    second = memory_module.load_memory()

    assert second["key_facts"] == []
    assert second["user_profile"]["investment_goals"] == []


def test_load_memory_backfills_old_files_with_isolated_defaults(clean_memory_file):
    with open(clean_memory_file, "w") as f:
        json.dump({"user_profile": {"name": "TestUser"}}, f)

    first = memory_module.load_memory()
    first["key_facts"].append("mutated fact")

    second = memory_module.load_memory()

    assert second["user_profile"]["name"] == "TestUser"
    assert "base_currency" in second["user_profile"]
    assert second["key_facts"] == []


def test_memory_crud_round_trip(clean_memory_file):
    memory_module.update_profile(
        {"age": "42", "annual_income": "$150,000", "base_currency": "CAD", "unknown_field": "ignored"}
    )
    memory_module.add_fact("I prefer Canadian-listed ETFs")
    memory_module.add_fact("I prefer Canadian-listed ETFs")
    memory_module.add_lesson("Always size positions before adding risk.")
    memory_module.add_lesson("Always size positions before adding risk.")
    memory_module.add_active_thesis(
        {"symbol": "NVDA", "action": "watch", "catalyst": "earnings"}
    )

    stored = memory_module.load_memory()
    assert stored["user_profile"]["age"] == "42"
    assert stored["user_profile"]["annual_income"] == "$150,000"
    assert stored["user_profile"]["base_currency"] == "CAD"
    assert "unknown_field" not in stored["user_profile"]
    assert stored["key_facts"] == ["I prefer Canadian-listed ETFs"]
    assert stored["lessons_learned"] == ["Always size positions before adding risk."]
    assert len(stored["active_theses"]) == 1

    thesis_id = stored["active_theses"][0]["id"]
    assert memory_module.update_lesson(0, "Size positions before adding risk.") is True
    assert memory_module.update_active_thesis(thesis_id, {"action": "trim"}) is True

    updated = memory_module.load_memory()
    assert updated["lessons_learned"] == ["Size positions before adding risk."]
    assert updated["active_theses"][0]["action"] == "trim"
    assert "updated_at" in updated["active_theses"][0]

    assert memory_module.delete_lesson(0) is True
    assert memory_module.delete_active_thesis(thesis_id) is True
    final = memory_module.load_memory()
    assert final["lessons_learned"] == []
    assert final["active_theses"] == []


def test_active_theses_do_not_cross_demo_boundary(tmp_path, monkeypatch):
    base_dir = tmp_path
    monkeypatch.setattr("tools.user_profile.os.path.dirname", lambda x: str(base_dir))
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("CAIRNIQ_FORCE_DEMO", raising=False)

    token = set_active_profile("default")
    try:
        memory_module.add_active_thesis({"symbol": "REAL", "action": "hold"})
        normal_memory = memory_module.load_memory()
        assert [thesis["symbol"] for thesis in normal_memory["active_theses"]] == ["REAL"]

        monkeypatch.setenv("DEMO_MODE", "true")
        ensure_demo_profile(reset=True)
        memory_module.add_active_thesis({"symbol": "DEMO", "action": "watch"})
        demo_memory = memory_module.load_memory()
        assert [thesis["symbol"] for thesis in demo_memory["active_theses"]] == ["DEMO"]

        monkeypatch.delenv("DEMO_MODE", raising=False)
        normal_memory_after_demo = memory_module.load_memory()
        assert [thesis["symbol"] for thesis in normal_memory_after_demo["active_theses"]] == ["REAL"]
    finally:
        reset_profile(token)


def test_enrich_thesis_surfaces_levels_from_bare_numeric_fields(monkeypatch):
    # stop_loss/target_price are stored as bare numeric strings (no leading "$") by
    # both the UI form and add_active_thesis. The enrichment must parse them (not
    # silently no-op) AND keep them distinct: a stop must never be mistaken for the
    # entry price (which would emit a bogus "ENTRY MISSED" measured off the stop).
    thesis = {
        "symbol": "MU",
        "action": "BUY",
        "conditions": "AI Growth",   # no explicit entry price recorded
        "stop_loss": "600",
        "target_price": "950",
        "created_at": "2026-07-06T21:43:02.616719",
    }
    monkeypatch.setattr(
        "tools.market_data.get_stock_data",
        lambda symbol: {"current_price": "1014.15", "beta": "1.4"},
    )

    # Held — the exit lifecycle under test only applies to an open position; a
    # not-held BUY thesis is an entry plan with its own flag vocabulary
    # (see tests/test_tools/test_thesis_position_state.py).
    enriched = memory_module._enrich_thesis_with_price_context(dict(thesis), held={"MU"})
    flags = enriched["_health_flags"]

    assert enriched["_live_price"] == 1014.15
    # Price is above the $950 target → TARGET REACHED, needs review.
    assert any("TARGET REACHED" in f for f in flags)
    # The $600 stop must NOT be treated as an entry price.
    assert not any("ENTRY MISSED" in f for f in flags)


def test_enrich_thesis_entry_missed_and_stop_breached(monkeypatch):
    # Both theses below are HELD: STOP BREACHED / "close it" is the open-position
    # lifecycle. The not-held equivalents are covered in
    # tests/test_tools/test_thesis_position_state.py.
    # Explicit entry price in conditions → ENTRY MISSED when price runs past it.
    entry_thesis = {
        "symbol": "X", "action": "BUY", "conditions": "Entry $100-110",
        "stop_loss": "90", "target_price": "200", "created_at": "2026-07-08T00:00:00",
    }
    monkeypatch.setattr(
        "tools.market_data.get_stock_data",
        lambda s: {"current_price": "140", "beta": "1.4"},
    )
    entry_flags = memory_module._enrich_thesis_with_price_context(
        dict(entry_thesis), held={"X"}
    )["_health_flags"]
    assert any("ENTRY MISSED" in f for f in entry_flags)
    assert not any("STOP BREACHED" in f for f in entry_flags)

    # Price at/below the stop → STOP BREACHED (thesis invalidated).
    stop_thesis = {
        "symbol": "Y", "action": "BUY", "conditions": "core hold",
        "stop_loss": "90", "target_price": "200", "created_at": "2026-07-08T00:00:00",
    }
    monkeypatch.setattr(
        "tools.market_data.get_stock_data",
        lambda s: {"current_price": "80", "beta": "1.0"},
    )
    stop_flags = memory_module._enrich_thesis_with_price_context(
        dict(stop_thesis), held={"Y"}
    )["_health_flags"]
    assert any("STOP BREACHED" in f for f in stop_flags)


# ---------------------------------------------------------------------------
# The lesson cap (user's call 2026-07-27: 15, truncating from the front)
# ---------------------------------------------------------------------------
#
# The cap evicts again, but the reason 1.7 removed eviction still stands and is
# what these tests actually pin: the old version dropped a rule the user wrote
# months ago out of every prompt and said so only via a safe_print on a server
# nobody reads. So the eviction has to be REPORTABLE — a distinct return code,
# and the outgoing text readable before the write that destroys it.

# Deliberately unalike: add_lesson's near-duplicate guard collapses reworded
# repeats, so a filler list of "rule 1 / rule 2 / ..." would never reach the cap.
_DISTINCT_RULES = [
    "Size every position before adding risk.",
    "Never catch a falling knife.",
    "Quote fixed income in yield, not price.",
    "Ignore analyst price targets entirely.",
    "Check the ex-dividend date ahead of any trim.",
    "Treat pension holdings as bonds.",
    "Flag anything above 8% of the book.",
    "Do not chase 52-week highs.",
    "Say when a data source is stale.",
    "Prefer index funds inside registered accounts.",
    "Convert every foreign holding to the base currency.",
    "Warn before any trade inside a locked-in account.",
    "Rebalance only on the first business day of a quarter.",
    "Name the broker whenever you quote a commission.",
    "Show the cost basis next to each unrealized gain.",
    "Skip the pre-market tape when sizing an entry.",
    "Assume no options overlay unless I say otherwise.",
    "Round share counts down, never up.",
    "Keep six months of spending in cash outside the market.",
    "Report sector weights before recommending a swap.",
]


def _fill_lessons(n):
    assert n <= len(_DISTINCT_RULES)
    for rule in _DISTINCT_RULES[:n]:
        memory_module.add_lesson(rule)


def test_a_lesson_at_the_cap_retires_the_oldest(clean_memory_file):
    _fill_lessons(memory_module.LESSON_CAP)
    oldest, second = memory_module.load_memory()["lessons_learned"][:2]

    result = memory_module.add_lesson("A brand new rule about currency reporting.")

    assert result == memory_module.LESSON_EVICTED
    stored = memory_module.load_memory()["lessons_learned"]
    assert len(stored) == memory_module.LESSON_CAP
    assert oldest not in stored  # front of the queue is what goes
    assert stored[0] == second
    assert stored[-1] == "A brand new rule about currency reporting."


def test_the_outgoing_rule_is_readable_before_it_is_destroyed(clean_memory_file):
    """The whole reason eviction is allowed back: a caller can name the casualty.

    After the write the retired text exists in no store, so anything that wants
    to tell the user what an add cost has to ask first.
    """
    _fill_lessons(memory_module.LESSON_CAP - 1)
    assert memory_module.lessons_pending_eviction() == []

    memory_module.add_lesson(_DISTINCT_RULES[memory_module.LESSON_CAP - 1])
    at_risk = memory_module.lessons_pending_eviction()

    assert at_risk == [_DISTINCT_RULES[0]]
    memory_module.add_lesson("A brand new rule about currency reporting.")
    assert at_risk[0] not in memory_module.load_memory()["lessons_learned"]


def test_a_lowered_cap_reports_every_rule_one_add_truncates(clean_memory_file):
    """A store left above the cap sheds several rules on the next add, and all of
    them have to be nameable — not just the first."""
    over = memory_module.LESSON_CAP + 2
    assert over <= len(_DISTINCT_RULES)
    memory_module.save_memory(
        {**memory_module.load_memory(), "lessons_learned": list(_DISTINCT_RULES[:over])}
    )

    at_risk = memory_module.lessons_pending_eviction()
    assert at_risk == _DISTINCT_RULES[:3]

    assert memory_module.add_lesson("A brand new rule about currency reporting.") == memory_module.LESSON_EVICTED
    stored = memory_module.load_memory()["lessons_learned"]
    assert len(stored) == memory_module.LESSON_CAP
    assert not any(rule in stored for rule in at_risk)


def test_add_lesson_reports_what_happened(clean_memory_file):
    assert memory_module.add_lesson("Report totals in the base currency.") == memory_module.LESSON_ADDED
    assert memory_module.add_lesson("Report totals in the base currency.") == memory_module.LESSON_DUPLICATE
    assert memory_module.add_lesson("   ") == memory_module.LESSON_EMPTY


def test_a_duplicate_at_the_cap_costs_nothing(clean_memory_file):
    """A no-op write must not retire a rule to make room for something it is not
    going to store."""
    _fill_lessons(memory_module.LESSON_CAP)
    before = memory_module.load_memory()["lessons_learned"]

    assert memory_module.add_lesson(_DISTINCT_RULES[3]) == memory_module.LESSON_DUPLICATE

    assert memory_module.load_memory()["lessons_learned"] == before
