"""A secular theme is the user's conviction or it does not exist.

Regression suite for a shipped default: DEFAULT_MEMORY carried a fully-formed
AI / semiconductors thesis — conviction "high", horizon "5-10 years", its own
trim_triggers and do_not_trim_for lists. Because load_memory() back-fills every
missing top-level key, a live profile that had never had the field acquired that
entry the first time it was read and saved, and the advisor then quoted it back as
the user's own stated view and shielded whatever it tagged from a trim.

Two blanks are not the same blank, and the file already draws the distinction for
risk limits (`unconstrained_ack`): a limit the user CHOSE to leave open is an
answer, a blank nobody was ever asked about is not. Emptying the default fixed the
fabrication and left every blank here of the second kind, because the store had no
writer of any sort — no setter, no endpoint, no screen. So the second half of this
suite covers the entry path that closes that: what it accepts, what it refuses,
and the two things it may never do, which are to author a theme and to turn a
clearing into a reset.
"""

import json
from typing import Any

import pytest

import tools.memory as mem
from tools.memory import get_secular_themes, get_user_context, set_secular_themes

THEME = {
    "theme": "Grid / Electrification",
    "conviction": "high",
    "horizon": "10 years",
    "rationale": "Stated by the user in their own words.",
    "trim_triggers": ["Close below the 40-week MA"],
    "do_not_trim_for": ["RSI > 70 alone"],
}


@pytest.fixture
def profile(monkeypatch, tmp_path):
    """A real on-disk profile, so the load_memory() back-fill is exercised."""
    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    return tmp_path / "user_memory.json"


def _memory(themes: Any) -> dict[str, Any]:
    return {
        "user_profile": {"name": "Test User", "base_currency": "CAD"},
        "key_facts": [],
        "conversation_summaries": [],
        "past_recommendations": [],
        "active_theses": [],
        "secular_themes": themes,
    }


@pytest.fixture
def context(monkeypatch):
    """Render the injected memory context over a profile with the given themes."""
    def _render(themes: Any = None) -> str:
        monkeypatch.setattr(mem, "load_memory", lambda: _memory(themes or []))
        return get_user_context()
    return _render


# --- the default ----------------------------------------------------------------


def test_shipped_default_states_no_theme():
    """The house view is not the user's view. Nothing may be seeded here."""
    assert mem.DEFAULT_MEMORY["secular_themes"] == []


def test_a_fresh_profile_gets_no_theme(profile):
    assert mem.load_memory()["secular_themes"] == []


def test_a_profile_predating_the_field_is_not_seeded_with_one(profile):
    """The actual incident: the key was absent, the back-fill supplied a conviction."""
    profile.write_text(json.dumps({"user_profile": {"name": "Test User"}}))

    assert mem.load_memory()["secular_themes"] == []


def test_a_stated_theme_survives_the_back_fill(profile):
    """Back-fill fills gaps; it must never touch what the user actually stated."""
    profile.write_text(json.dumps({"secular_themes": [THEME]}))

    assert mem.load_memory()["secular_themes"] == [THEME]


# --- the empty block ------------------------------------------------------------
#
# The block was gated on truthiness and emitted NOTHING when the list was empty.
# That was survivable only while the list was never empty — the shipped default
# saw to that — so removing the default turned silence into the normal case, and
# silence is what gets filled in. The negative has to be on the page, and it has
# to close BOTH doors at once: an absent conviction may not read as one the model
# supplies, nor as the user having none.


def test_empty_block_is_stated_not_omitted(context):
    """Silence is what gets back-filled — the negative has to be on the page."""
    rendered = context()

    assert "STRUCTURAL CONVICTION (SECULAR THEMES)" in rendered
    assert "NONE ON RECORD — the user has stated no secular theme" in rendered
    assert "complete and authoritative" in rendered


def test_empty_block_forbids_promoting_a_holding_to_a_theme(context):
    """The fabrication this prevents: "your secular AI position" over a growth name."""
    rendered = context()

    assert "do NOT name, infer, or reconstruct one" in rendered
    assert "secular / structural / long-term-conviction position" in rendered
    assert "no trim_trigger" in rendered and "to claim was met" in rendered


def test_empty_block_is_not_read_as_permission_to_trim(context):
    """The inverse fabrication: reading an unanswered question as "no conviction"."""
    rendered = context()

    assert "UNANSWERED question, not an answer" in rendered
    assert "NOT evidence for trimming" in rendered
    assert "never becomes a reason to sell" in rendered


# --- a stated theme -------------------------------------------------------------


def test_stated_theme_renders_with_its_own_rules(context):
    rendered = context([THEME])

    assert "Grid / Electrification" in rendered
    assert "conviction=high" in rendered
    assert "Close below the 40-week MA" in rendered
    assert "RSI > 70 alone" in rendered


def test_stated_theme_suppresses_the_negative(context):
    """Both branches must never render — that would assert and deny in one context."""
    rendered = context([THEME])

    assert "NONE ON RECORD — the user has stated no secular theme" not in rendered


def test_unreadable_rows_do_not_render_a_phantom_theme(context):
    """A junk row must not become an "Unnamed theme" that shields a position."""
    rendered = context(["AI / Semiconductors", None, {}, {"theme": "   "}])

    assert "Unnamed theme" not in rendered
    assert "AI / Semiconductors" not in rendered
    assert "NONE ON RECORD — the user has stated no secular theme" in rendered


def test_a_conviction_level_is_never_invented_for_a_hand_edited_row(context):
    """The renderer used to default a missing level to "medium", which states the
    user's strength of belief for them. The writer refuses such a row, but a
    hand-edited file can still carry one."""
    rendered = context([{"theme": "Grid / Electrification",
                         "trim_triggers": ["Close below the 40-week MA"]}])

    assert "conviction=unstated" in rendered
    assert "conviction=medium" not in rendered


# --- the reader -----------------------------------------------------------------


def test_the_reader_reports_no_theme_rather_than_a_house_one(profile):
    assert get_secular_themes() == []


def test_an_unreadable_row_is_dropped_rather_than_named(profile):
    """A junk row must not surface as an unnamed theme. Whatever position the
    reader then decided it referred to would be shielded by a row nobody wrote."""
    profile.write_text(json.dumps({"secular_themes": ["AI / Semiconductors", None, {}]}))

    assert get_secular_themes() == []


def test_a_store_that_is_not_a_list_reads_as_no_theme(profile):
    profile.write_text(json.dumps({"secular_themes": {"theme": "Something"}}))

    assert get_secular_themes() == []


# --- the writer: what it accepts ------------------------------------------------


def test_a_stated_theme_round_trips(profile):
    result = set_secular_themes([THEME])

    assert result["ok"] is True
    stored = get_secular_themes()
    assert len(stored) == 1
    assert stored[0]["theme"] == "Grid / Electrification"
    assert stored[0]["conviction"] == "high"
    assert stored[0]["horizon"] == "10 years"
    assert stored[0]["rationale"] == "Stated by the user in their own words."
    assert stored[0]["trim_triggers"] == ["Close below the 40-week MA"]
    assert stored[0]["do_not_trim_for"] == ["RSI > 70 alone"]


def test_the_optional_fields_may_be_left_blank(profile):
    """Horizon, rationale and the never-trim list change nothing that is
    enforced, so a user is entitled to skip them."""
    assert set_secular_themes([{
        "theme": "Grid / Electrification",
        "conviction": "low",
        "trim_triggers": ["The thesis I wrote down stops being true"],
    }])["ok"] is True

    stored = get_secular_themes()[0]
    assert stored["horizon"] == ""
    assert stored["rationale"] == ""
    assert stored["do_not_trim_for"] == []


def test_rules_may_be_typed_as_one_block_of_lines(profile):
    """The editor hands over a textarea. Splitting it here rather than in the
    page keeps one definition of what a rule is."""
    set_secular_themes([{
        "theme": "Grid / Electrification",
        "conviction": "medium",
        "trim_triggers": "  Close below the 40-week MA \n\n The capex cycle rolls over \n",
    }])

    assert get_secular_themes()[0]["trim_triggers"] == [
        "Close below the 40-week MA", "The capex cycle rolls over",
    ]


def test_conviction_is_stored_as_the_user_picked_it(profile):
    for level in ("high", "medium", "low"):
        set_secular_themes([{"theme": "T", "conviction": level.upper(),
                             "trim_triggers": ["A rule"]}])
        assert get_secular_themes()[0]["conviction"] == level


def test_saving_replaces_the_whole_list_rather_than_merging(profile):
    """A merge would make a deleted conviction un-deletable, and a stale theme
    keeps vetoing sell advice on a position the user has stopped believing in."""
    set_secular_themes([THEME])

    set_secular_themes([{"theme": "Something else", "conviction": "low",
                         "trim_triggers": ["A rule"]}])

    assert [t["theme"] for t in get_secular_themes()] == ["Something else"]


def test_an_untouched_theme_keeps_the_date_it_was_stated(profile):
    """Re-stamping every row on every save would turn "stated on" into "last
    time you opened the editor", which is a different fact."""
    set_secular_themes([THEME])
    stated_at = get_secular_themes()[0]["set_at"]

    set_secular_themes([THEME, {"theme": "A second one", "conviction": "low",
                                "trim_triggers": ["A rule"]}])

    stored = {t["theme"]: t for t in get_secular_themes()}
    assert stored["Grid / Electrification"]["set_at"] == stated_at


def test_editing_a_theme_restamps_only_that_one(profile):
    set_secular_themes([THEME])
    original = get_secular_themes()[0]["set_at"]

    edited = dict(THEME, rationale="I changed my mind about why.")
    set_secular_themes([edited])

    stored = get_secular_themes()[0]
    assert stored["rationale"] == "I changed my mind about why."
    assert stored["set_at"] >= original


# --- the writer: clearing means NO theme ----------------------------------------


def test_clearing_leaves_no_theme_rather_than_a_house_one(profile):
    """The bug this store was reopened over, in its most direct form: a reset
    that restores a default is how the fabricated thesis got in front of people."""
    set_secular_themes([THEME])

    result = set_secular_themes(None)

    assert result["ok"] is True
    assert result["cleared"] is True
    assert get_secular_themes() == []
    assert mem.load_memory()["secular_themes"] == []


def test_clearing_with_an_empty_list_is_the_same_thing(profile):
    set_secular_themes([THEME])

    assert set_secular_themes([])["ok"] is True
    assert get_secular_themes() == []


def test_a_cleared_store_is_not_refilled_by_the_next_read(profile):
    """The back-fill runs on every load. An empty list must survive it as an
    empty list — this is the exact path the original default travelled."""
    set_secular_themes([THEME])
    set_secular_themes(None)

    mem.load_memory()
    mem.load_memory()

    assert get_secular_themes() == []


# --- the writer: what it refuses, and why each refusal is not a default ----------


def test_a_theme_with_no_name_is_refused(profile):
    result = set_secular_themes([{"conviction": "high", "trim_triggers": ["A rule"]}])

    assert result["ok"] is False
    assert "name" in result["error"].lower()
    assert get_secular_themes() == []


def test_a_missing_conviction_is_refused_rather_than_filled_in(profile):
    """Conviction is the dial that decides how hard a theme argues against a
    trim. A level this function picked would be quoted back as the user's own
    strength of belief — the same class of invention as the shipped default."""
    result = set_secular_themes([{"theme": "Grid", "trim_triggers": ["A rule"]}])

    assert result["ok"] is False
    assert "conviction" in result["error"]
    assert get_secular_themes() == []


def test_an_unrecognised_conviction_is_refused_rather_than_rounded(profile):
    result = set_secular_themes([{"theme": "Grid", "conviction": "extremely high",
                                  "trim_triggers": ["A rule"]}])

    assert result["ok"] is False
    assert get_secular_themes() == []


def test_a_theme_with_no_trim_rule_is_refused(profile):
    """The strongest standing order in the app: a theme with no exit rule holds
    the position through anything. Reaching that by leaving a box empty is the
    silence-becomes-instruction failure the whole store exists to avoid, so it is
    accepted only as a sentence the user wrote."""
    result = set_secular_themes([{"theme": "Grid", "conviction": "high"}])

    assert result["ok"] is False
    assert "trim" in result["error"].lower()
    assert get_secular_themes() == []


def test_a_trim_rule_of_only_whitespace_is_not_a_trim_rule(profile):
    result = set_secular_themes([{"theme": "Grid", "conviction": "high",
                                  "trim_triggers": ["", "   ", "\n"]}])

    assert result["ok"] is False
    assert get_secular_themes() == []


def test_one_theme_named_twice_is_refused_rather_than_silently_merged(profile):
    """Two rows under one name carry two sets of exit rules. Keeping either one
    drops a rule the user wrote while the save reports success."""
    result = set_secular_themes([
        dict(THEME, trim_triggers=["Rule one"]),
        dict(THEME, trim_triggers=["Rule two"]),
    ])

    assert result["ok"] is False
    assert "twice" in result["error"]
    assert get_secular_themes() == []


def test_a_bad_row_leaves_the_stored_themes_exactly_as_they_were(profile):
    """All-or-nothing. A partial write leaves the user believing a theme is
    protecting a position when the row naming it was the one that failed."""
    set_secular_themes([THEME])

    result = set_secular_themes([
        THEME,
        {"theme": "Half-written", "conviction": "high"},
    ])

    assert result["ok"] is False
    assert [t["theme"] for t in get_secular_themes()] == ["Grid / Electrification"]


def test_every_rejected_row_is_named_rather_than_dropped(profile):
    """A refusal the user cannot act on is the silent drop wearing a different
    hat: they retype the same list and it fails again for a reason nobody said."""
    result = set_secular_themes([
        {"theme": "First", "conviction": "high"},
        {"theme": "Second", "conviction": "sideways", "trim_triggers": ["A rule"]},
    ])

    assert result["ok"] is False
    assert len(result["rejected"]) == 2
    assert any("First" in line for line in result["rejected"])
    assert any("Second" in line for line in result["rejected"])


def test_a_theme_is_never_written_by_anything_but_this_function(profile):
    """The invariant, asserted where it can be checked: the store the advisor
    reads holds exactly the themes that came through the setter. Nothing derives
    one from holdings, and no model-drafted theme lands without a call here."""
    set_secular_themes([THEME])

    assert mem.load_memory()["secular_themes"] == get_secular_themes()
    assert len(get_secular_themes()) == 1
