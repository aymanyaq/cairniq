"""Profile readiness surface (Advisor Roadmap 2.8).

The item exists because five shipped slices went dark on an empty store while
the app was *correct* to refuse to fill it in. So the tests that matter are not
"does it list four rows". They are:

  - does it ever author, suggest or exemplify a value for a blank it reports on
    (the 3.7 empty-state contract, generalized — see
    test_drawdown_playbook.py::test_the_message_for_an_unset_playbook_names_the_absence);
  - can it tell a stated value from a blank, rather than only listing blanks;
  - does it count the number the CONSUMER reads, not the store's own total;
  - does an unreadable store read as unverified rather than as empty.
"""
import re

import pytest

import tools.asset_location as al
import tools.feedback as fb
import tools.memory as mem
import tools.profile_readiness as pr


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Every store this surface reads, homed in a tmp profile.

    A test that wrote a cap or a target into the live profile would be
    corrupting the exact data the page reports on.
    """
    monkeypatch.setattr(mem, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(fb, "get_data_path", lambda name: str(tmp_path / name))
    # The account-jurisdiction row (4.7a) is the one input here whose gap is
    # per-ACCOUNT, so it reads the portfolio. Pinned to empty rather than left to
    # whatever the box happens to hold: a suite that reads the real portfolio is
    # both non-deterministic and touching the user's data, which is the leak the
    # conftest fixtures were written to close.
    monkeypatch.setattr(al, "load_portfolio", lambda: [])
    return tmp_path


def _by_key(readiness):
    return {row["key"]: row for row in readiness["inputs"]}


# ---------------------------------------------------------------------------
# THE contract: it reports the blank, it never fills it
# ---------------------------------------------------------------------------

# Anything that would read as the app proposing a figure or a starting point.
# Word boundaries matter: "entry" is not "try", and "Roadmap 4.4" is not a cap.
_VALUE_SHAPED = (
    re.compile(r"\d+(?:\.\d+)?\s*%"),          # a percentage — every cap here is one
    re.compile(r"[$€£¥]\s*\d"),                # a money figure
)
_SUGGESTION_WORDS = (
    "suggest", "suggested", "recommend", "recommended", "advise", "consider",
    "typical", "typically", "usually", "commonly", "example", "such as",
    "try", "start with", "rule of thumb", "ballpark", "sensible",
)


def _prose(row):
    """Every string the surface AUTHORS for one row.

    ``stated`` and ``observed`` are excluded on purpose: those are the user's
    own figures read back out of their own store, and echoing them is the
    opposite of inventing one.

    Dict VALUES are walked for the same reason lists are: the per-field
    consequence map hands the editor a sentence for a field that is not
    currently missing, so its prose reaches a human without passing through
    ``inert`` — and prose this contract does not scan is prose that can quietly
    start proposing a figure.
    """
    out = []
    for key, value in row.items():
        if key in ("stated", "observed"):
            continue
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str))
        elif isinstance(value, dict):
            out.extend(v for v in value.values() if isinstance(v, str))
    return out


def test_nothing_the_surface_says_about_a_blank_proposes_a_value():
    """The whole point of the item. A cap, a rung or a target that the software
    supplied gets quoted back later as the user's own stated rule, with the
    authority of a promise they made to themselves — so this page may name the
    blank and the cost, and nothing else."""
    readiness = pr.build_profile_readiness()

    for row in readiness["inputs"]:
        if row["status"] == pr.STATUS_SET:
            continue
        for text in _prose(row) + [readiness["contract"]]:
            for pattern in _VALUE_SHAPED:
                assert not pattern.search(text), (
                    f"{row['key']} offered a value-shaped figure: {text!r}"
                )
            lowered = text.lower()
            for word in _SUGGESTION_WORDS:
                assert not re.search(rf"\b{re.escape(word)}\b", lowered), (
                    f"{row['key']} suggested a value ({word!r}): {text!r}"
                )


# Prose the surface authors for a human to read, as opposed to the field keys
# and slice IDs it also carries for the page and for a developer.
_PROSE_KEYS = ("cost", "entry")
_PROSE_LISTS = ("inert", "feeds", "capabilities", "capabilities_dark")

_CODE_SHAPED = (
    (re.compile(r"roadmap", re.I), "a roadmap label"),
    (re.compile(r"\b\d+\.\d+[a-z]?\b"), "a slice number"),
    (re.compile(r"\b[a-z]+_[a-z_]+\b"), "a code identifier"),
    (re.compile(r"\b(?:GET|POST|PUT|DELETE)\b|/api/"), "an endpoint"),
)


def _authored_prose(row):
    out = [row[k] for k in _PROSE_KEYS if row.get(k)]
    for key in _PROSE_LISTS:
        out.extend(row.get(key) or [])
    return out


def test_no_line_the_user_reads_names_a_roadmap_slice_or_a_symbol():
    """The page is read by someone deciding whether to fill a box in, and a
    roadmap number tells them nothing about that. Slice IDs belong in `roadmap`,
    endpoints and function names in the module's own comments — the prose names
    the capability that goes dark. Field keys are exempt: `missing` is a list of
    boxes on a form, not a sentence."""
    readiness = pr.build_profile_readiness()

    for row in readiness["inputs"]:
        for text in _authored_prose(row) + [readiness["contract"]]:
            for pattern, what in _CODE_SHAPED:
                assert not pattern.search(text), (
                    f"{row['key']} put {what} in a line the user reads: {text!r}"
                )


def test_a_gap_carries_one_summary_line_and_the_detail_behind_it():
    """The volume complaint that prompted this: with nothing on file, every
    field's consequence fires at once. `cost` is the one line the collapsed row
    shows; `inert` is the same gap at full detail, and they are built from the
    same `missing` list so they cannot disagree."""
    readiness = pr.build_profile_readiness()

    for row in readiness["inputs"]:
        # STATUS_NOT_STATED joins SET here rather than the gap branch below, and
        # the reason is the whole point of the fourth word: an OPTIONAL input
        # nobody has stated has no cost to summarize. Charging one would make the
        # collapsed line say a feature is off when none is.
        if row["status"] in (pr.STATUS_SET, pr.STATUS_NOT_STATED):
            assert row["cost"] == "", f"{row['key']} charged a cost with nothing missing"
            assert row["inert"] == []
        else:
            assert row["cost"], f"{row['key']} reported a gap with no summary line"
            assert row["cost"].count(".") == 1, (
                f"{row['key']} put more than one sentence in the collapsed line: "
                f"{row['cost']!r}"
            )
            assert row["inert"], f"{row['key']} has a summary with no detail behind it"


def test_one_capability_lost_to_several_blanks_is_named_once():
    """Three playbook fields feed the same alert. Listing it three times in the
    collapsed line would reintroduce exactly the repetition the line exists to
    replace."""
    row = _by_key(pr.build_profile_readiness())["drawdown_playbook"]

    assert row["cost"].count("deep-drawdown alerts") == 1


# ---------------------------------------------------------------------------
# The switchboard counts CAPABILITIES, not consequence sentences
# ---------------------------------------------------------------------------

def test_the_switchboard_counts_distinct_capabilities_not_blank_fields():
    """The header used to sum `inert_count` and call the result capabilities.
    That figure is one sentence per blank FIELD, and a fully blank playbook has
    four fields naming three features — so it reported four things dark when
    three were. Row-versus-thing, the same error the recommendation ledger had.
    """
    readiness = pr.build_profile_readiness()
    caps = readiness["capabilities"]
    playbook = _by_key(readiness)["drawdown_playbook"]

    assert len(playbook["missing"]) == 4
    assert playbook["capabilities_dark"] == [
        "deep-drawdown alerts", "the automated deployment ladder", "drift checks",
    ]
    # Every store blank: distinct features, against one sentence per blank field.
    # `total` stayed 8 when target_allocation was added and inert_count went to
    # 13, which is the distinction this test exists for: the target allocation
    # switches off "drift checks", a capability the playbook's band already
    # darkens. A second input on the same feature is another SENTENCE about why,
    # not another dark FEATURE.
    #
    # account_jurisdictions (4.7a) DOES add two, because nothing else feeds
    # asset-location scoring or the sell-and-rebuy tax check — the same rule
    # producing the opposite answer.
    assert caps["total"] == 10
    assert len(caps["dark"]) == 10
    assert readiness["inert_count"] == 14
    assert len(caps["dark"]) == len(set(caps["dark"]))


def test_a_stated_store_lights_its_capabilities_and_leaves_the_rest_dark():
    """The lit half has to be knowable, or the page can only ever show failures
    and there is no way to tell a working feature from an untracked one."""
    mem.set_financial_goal({
        "target_low": 3_000_000, "horizon_years": 10, "annual_contribution": 65_000,
    })

    caps = pr.build_profile_readiness()["capabilities"]

    assert caps["live"] == [
        "the goal projection", "the drawdown alert's still-on-track line",
    ]
    assert "the goal projection" not in caps["dark"]
    assert "deep-drawdown alerts" in caps["dark"]
    assert len(caps["live"]) + len(caps["dark"]) == caps["total"]


def test_an_unreadable_store_is_counted_as_unverified_not_as_dark(monkeypatch):
    """Marking an unread store's features dark would assert a fact about data
    nobody managed to read — the same invention `unreadable` exists to prevent."""
    monkeypatch.setattr(
        pr, "_BUILDERS", (_broken("_playbook_input"), pr._wealth_goal_input)
    )

    caps = pr.build_profile_readiness()["capabilities"]

    assert caps["unverified_stores"] == 1
    assert "deep-drawdown alerts" not in caps["dark"]
    assert "deep-drawdown alerts" not in caps["live"]


def test_every_row_carries_the_field_list_the_page_draws_its_marks_from():
    """The page renders one mark per required field. A row without `required`
    renders an empty strip, which reads as "nothing to state" — the opposite of
    what a blank means, and exactly backwards for the ratings row."""
    for row in pr.build_profile_readiness()["inputs"]:
        assert "required" in row, f"{row['key']} has no field list to draw"
        assert "optional_present" in row
        assert set(row["missing"]) <= set(row["required"]), (
            f"{row['key']} reported a gap in a field it never listed as required"
        )


def test_every_blank_names_a_consequence_and_where_it_is_stated():
    """Reporting emptiness without naming what it costs is the status quo this
    item replaces — the blank was always visible, the cost never was."""
    readiness = pr.build_profile_readiness()

    for row in readiness["inputs"]:
        # Where it is stated is required of EVERY blank, optional or not: a row
        # the user cannot act on is the same dead end whether or not it costs
        # them anything. Only the consequence is exempt, and only because an
        # optional blank genuinely has none.
        if row["status"] != pr.STATUS_SET:
            assert row["entry"], f"{row['key']} reported a blank with nowhere to state it"
        if row["status"] in (pr.STATUS_SET, pr.STATUS_NOT_STATED):
            continue
        assert row["inert"], f"{row['key']} reported a gap with no consequence"


def test_a_bare_profile_reports_every_required_input_as_empty():
    readiness = pr.build_profile_readiness()
    rows = _by_key(readiness)

    assert set(rows) == {
        "drawdown_playbook", "risk_constraints", "target_allocation",
        "account_jurisdictions", "wealth_goal", "secular_themes",
        "feedback_ratings",
    }
    required = {k: v for k, v in rows.items() if k not in pr.OPTIONAL_KEYS}
    assert all(row["status"] == pr.STATUS_EMPTY for row in required.values())
    assert readiness["counts"]["empty"] == 6
    assert readiness["counts"]["set"] == 0
    assert readiness["inert_count"] >= 4


# ---------------------------------------------------------------------------
# The fourth state: OPTIONAL, and blank
#
# Every row above reports a blank that switches a shipped feature off, so
# "empty" carries a cost by construction. Structural convictions do not: with
# none on file the daily priority still runs and still recommends, it just
# weighs every holding as tactical. That is a different default, not a dark
# feature — and the store it reports on is the one where an invented value did
# the most damage, so the row has to hold both halves at once.
# ---------------------------------------------------------------------------

def test_an_unstated_optional_input_is_complete_not_empty():
    """The nag test. A profile with no conviction is finished, and a row that
    reported it as `empty` would paint it red and chase an answer the user is
    entitled not to have."""
    row = _by_key(pr.build_profile_readiness())["secular_themes"]

    assert row["status"] == pr.STATUS_NOT_STATED
    assert row["status"] != pr.STATUS_EMPTY
    assert row["missing"] == []
    assert row["required"] == []


def test_an_unstated_optional_input_charges_nothing_and_darkens_nothing():
    """It must not reach the switchboard in either direction: a capability
    listed here would render lit or dark, and neither is true of a feature that
    behaves the same way with the store empty."""
    readiness = pr.build_profile_readiness()
    row = _by_key(readiness)["secular_themes"]

    assert row["cost"] == ""
    assert row["inert"] == []
    assert row["capabilities"] == []
    assert row["capabilities_dark"] == []
    caps = readiness["capabilities"]
    assert "the daily priority" not in caps["dark"] + caps["live"]


def test_the_stated_fraction_excludes_the_optional_row():
    """The figure the page puts at the top. Counting an optional input in the
    denominator would report a complete profile as incomplete forever, because
    the only way to close it is to state a conviction the user may not hold."""
    counts = pr.build_profile_readiness()["counts"]

    assert counts["total"] == 7
    assert counts["required_total"] == 6
    assert counts["required_set"] == 0
    assert counts["not_stated"] == 1


def test_a_stated_conviction_reads_as_set_and_echoes_the_users_own_words():
    mem.set_secular_themes([{
        "theme": "Grid / Electrification",
        "conviction": "high",
        "trim_triggers": ["Close below the 40-week moving average"],
    }])

    readiness = pr.build_profile_readiness()
    row = _by_key(readiness)["secular_themes"]

    assert row["status"] == pr.STATUS_SET
    assert row["observed"]["themes"] == ["Grid / Electrification"]
    assert row["optional_present"] == ["themes"]
    # Still outside the fraction. Stating one is the user raising their own bar,
    # not repairing something that was broken.
    assert readiness["counts"]["required_total"] == 6
    assert readiness["counts"]["required_set"] == 0


def test_the_optional_row_never_proposes_a_conviction():
    """The failure this store actually shipped, in the one place built to report
    on it. A page that named a theme while reporting that none is on file would
    be reintroducing the default through the instrument that caught it."""
    row = _by_key(pr.build_profile_readiness())["secular_themes"]

    prose = " ".join(_prose(row)).lower()

    for word in ("ai", "semiconductor", "compute", "energy", "grid"):
        assert not re.search(rf"\b{word}\b", prose), (
            f"the readiness row named a theme: {word!r}"
        )


def test_the_optional_blank_is_not_reported_as_an_absence_of_conviction():
    """The inverse fabrication, and the one that ends in a sell: an unanswered
    question read as "this user holds nothing for the long term" is evidence for
    trimming that nobody supplied."""
    row = _by_key(pr.build_profile_readiness())["secular_themes"]

    consequence = row["consequence_by_field"]["themes"]

    assert "complete answer" in consequence
    assert "nothing is switched off" in consequence.lower()
    assert row["stated"] == {}


# ---------------------------------------------------------------------------
# It must distinguish SET from unset, not merely list blanks
# ---------------------------------------------------------------------------

def test_a_stated_goal_reads_as_set_and_echoes_the_users_own_figures():
    """The one input that has ever closed. If the surface cannot show it as
    stated it is a blank-lister, not a readiness report."""
    mem.set_financial_goal({
        "target_low": 3_000_000, "target_high": 5_000_000,
        "horizon_years": 10, "annual_contribution": 65_000,
    })

    row = _by_key(pr.build_profile_readiness())["wealth_goal"]

    assert row["status"] == pr.STATUS_SET
    assert row["inert"] == []
    assert row["missing"] == []
    assert row["stated"]["target_low"] == 3_000_000
    assert row["stated"]["annual_contribution"] == 65_000
    # The stretch target is optional to 4.5's projection, so it is reported as
    # present rather than as a requirement.
    assert row["optional_present"] == ["target_high"]
    # What it powers is still named — the coupling is the product here. The
    # slice IDs live in `roadmap`; the prose names the capability instead.
    assert "4.5" in row["roadmap"]
    assert any("goal" in line.lower() for line in row["feeds"])


def test_a_goal_missing_only_its_contribution_names_that_one_gap():
    """Matches build_goal_projection's own required set: a target with no inflow
    is a real partially-specified state, and the surface must say WHICH box."""
    mem.set_financial_goal({"target_low": 3_000_000, "horizon_years": 10})

    row = _by_key(pr.build_profile_readiness())["wealth_goal"]

    assert row["status"] == pr.STATUS_PARTIAL
    assert row["missing"] == ["annual_contribution"]
    assert len(row["inert"]) == 1
    # The drawdown line drops too, not just the projection.
    assert "drawdown" in row["inert"][0].lower()
    assert row["cost"] == (
        "Switched off while blank: the goal projection, "
        "the drawdown alert's still-on-track line."
    )


# ---------------------------------------------------------------------------
# Consequences attach to FIELDS — half a playbook leaves a specific half dark
# ---------------------------------------------------------------------------

def test_a_half_written_playbook_names_only_the_half_that_is_missing():
    import tools.drawdown_playbook as pb

    pb.set_playbook({
        "never_sell": ["Core index ETFs"],
        "buy_first": ["Total market index"],
        "deployment_levels": [{"drawdown_pct": 20, "action": "deploy half"}],
    })

    row = _by_key(pr.build_profile_readiness())["drawdown_playbook"]

    assert row["status"] == pr.STATUS_PARTIAL
    assert row["missing"] == ["rebalance_drift_pct"]
    assert len(row["inert"]) == 1
    assert "drift" in row["inert"][0].lower()
    assert row["stated"]["never_sell"] == ["Core index ETFs"]
    # Only the half that is missing appears in the collapsed line either: the
    # three stated fields' alert is not reported as switched off.
    assert row["cost"] == "Switched off while blank: drift checks."


def test_a_note_to_your_future_self_is_not_a_missing_requirement():
    """`notes` gates nothing, so its absence is not reported as a gap. A surface
    that chases fields nothing reads is the nag this item refuses to be."""
    import tools.drawdown_playbook as pb

    pb.set_playbook({
        "never_sell": ["Core index ETFs"],
        "buy_first": ["Total market index"],
        "deployment_levels": [{"drawdown_pct": 20, "action": "deploy half"}],
        "rebalance_drift_pct": 5,
    })

    row = _by_key(pr.build_profile_readiness())["drawdown_playbook"]

    assert row["status"] == pr.STATUS_SET
    assert row["missing"] == []
    assert row["inert"] == []


def test_partial_risk_limits_name_only_the_unstated_caps():
    """2.2's gate enforces each axis independently: a stated single-name cap
    does not make the sector axis enforced."""
    mem.set_risk_constraints({"max_position_pct": 12})

    row = _by_key(pr.build_profile_readiness())["risk_constraints"]

    assert row["status"] == pr.STATUS_PARTIAL
    assert row["stated"] == {"max_position_pct": 12.0}
    assert "max_position_pct" not in row["missing"]
    assert "max_sector_pct" in row["missing"]
    assert not any("single-name" in line for line in row["inert"])
    assert any("sector" in line for line in row["inert"])


def test_an_empty_restricted_list_is_reported_but_is_not_a_gap():
    """"I restrict nothing" is a complete answer. Nothing is switched off by it,
    so it is measured and never chased."""
    row = _by_key(pr.build_profile_readiness())["risk_constraints"]

    assert row["observed"]["restricted_symbols"] == []
    assert "restricted_symbols" not in row["missing"]


# ---------------------------------------------------------------------------
# The third state: blank ON PURPOSE
#
# A field can be deliberately left empty as the user's real answer. That is
# neither stated nor missing, and this surface had only those two words for it —
# so a decision the user had already made kept being reported as a gap.
# ---------------------------------------------------------------------------

def test_a_confirmed_unlimited_axis_stops_being_reported_as_a_gap():
    """Otherwise the confirmation is worth nothing: the row keeps charging a
    cost for a question the user has closed, which is the nag this surface is
    built not to be."""
    mem.set_risk_constraints({"acknowledge_unconstrained": True})

    row = _by_key(pr.build_profile_readiness())["risk_constraints"]

    assert row["status"] == pr.STATUS_SET
    assert row["missing"] == []
    assert row["inert"] == []
    assert row["cost"] == ""


def test_a_confirmed_blank_is_never_reported_as_a_stated_figure():
    """The other direction, and the worse one: nothing may put a number in the
    user's mouth. The axis is answered, and `stated` stays empty."""
    mem.set_risk_constraints({"acknowledge_unconstrained": True})

    row = _by_key(pr.build_profile_readiness())["risk_constraints"]

    assert row["stated"] == {}
    assert sorted(row["answered_blank"]) == sorted(row["required"])
    assert sorted(row["observed"]["unconstrained_by_choice"]) == sorted(row["required"])


def test_an_axis_nobody_has_been_asked_about_is_still_a_gap():
    """A confirmation given about three axes says nothing about the fourth."""
    mem.set_risk_constraints({"max_position_pct": 12, "acknowledge_unconstrained": True})
    mem.set_risk_constraints({"max_position_pct": None})

    row = _by_key(pr.build_profile_readiness())["risk_constraints"]

    assert row["status"] == pr.STATUS_PARTIAL
    assert row["missing"] == ["max_position_pct"]
    assert "max_sector_pct" in row["answered_blank"]


def test_the_editor_can_read_the_cost_of_a_box_that_is_currently_filled():
    """The live preview's whole requirement. `inert` holds only what is MISSING,
    so an editor asking "what does clearing this cost?" about a stated cap gets
    nothing from it — and the fix that suggests itself, a copy of the sentence in
    the template, is how a page starts contradicting the report beneath it."""
    mem.set_risk_constraints({"max_sector_pct": 30})

    row = _by_key(pr.build_profile_readiness())["risk_constraints"]

    assert "max_sector_pct" not in row["missing"]
    assert row["consequence_by_field"]["max_sector_pct"]
    # Same sentence in both places, so they cannot drift apart.
    assert row["consequence_by_field"]["max_position_pct"] in row["inert"]


def test_every_row_carries_the_third_state_even_when_it_cannot_have_one():
    """The page draws one strip of marks for every row from these three lists.
    A row missing the key renders as having nothing to answer — the same silent
    blank the whole surface exists to end."""
    for row in pr.build_profile_readiness()["inputs"]:
        assert isinstance(row["answered_blank"], list), row["key"]


# ---------------------------------------------------------------------------
# Count what the CONSUMER reads
# ---------------------------------------------------------------------------

def test_a_full_but_unrated_feedback_store_is_an_empty_pool():
    """The exact shape of the 07-26 reading: `total` said 100 and the few-shot
    pool held zero. The pool is the number that belongs on this page."""
    for i in range(100):
        fb.add_interaction(f"q{i}", f"a{i}")

    row = _by_key(pr.build_profile_readiness())["feedback_ratings"]

    assert row["observed"] == {"captured": 100, "rated": 0, "high_quality_pool": 0}
    assert row["status"] == pr.STATUS_EMPTY
    assert "1.5b" in row["roadmap"]
    assert "pool" in row["inert"][0].lower()


def test_a_rating_below_the_pool_bar_does_not_make_a_pool():
    """Rated is not the same as usable: the selector reads a min_rating bar, so
    a store of complaints is still zero rows to the consumer."""
    fb.add_interaction("q", "a")
    fb.rate_interaction(rating=1)

    row = _by_key(pr.build_profile_readiness())["feedback_ratings"]

    assert row["observed"]["rated"] == 1
    assert row["observed"]["high_quality_pool"] == 0
    assert row["status"] == pr.STATUS_PARTIAL
    assert row["inert"]


def test_a_rated_answer_fills_the_pool():
    fb.add_interaction("q", "a")
    fb.rate_interaction(rating=5)

    row = _by_key(pr.build_profile_readiness())["feedback_ratings"]

    assert row["observed"]["high_quality_pool"] == 1
    assert row["status"] == pr.STATUS_SET
    assert row["inert"] == []


# ---------------------------------------------------------------------------
# The instrument must not acquire the failure it was built to catch
# ---------------------------------------------------------------------------

def _broken(builder_name):
    """A builder that throws but keeps its identity, so the row it fails to
    produce is still recognisable as the input it was reporting on."""
    def _boom():
        raise RuntimeError("store unreadable")
    _boom.__name__ = builder_name
    return _boom


def test_an_unreadable_store_is_unverified_not_empty(monkeypatch):
    """Reporting a store that threw as "empty" would be this surface inventing a
    fact about the user in the one place that must not."""
    monkeypatch.setattr(
        pr, "_BUILDERS", (_broken("_playbook_input"), pr._wealth_goal_input)
    )

    readiness = pr.build_profile_readiness()
    row = _by_key(readiness)["drawdown_playbook"]

    assert row["status"] == pr.STATUS_UNREADABLE
    assert row["status"] != pr.STATUS_EMPTY
    assert "UNKNOWN" in row["inert"][0]
    # The rest of the surface still renders.
    assert "wealth_goal" in _by_key(readiness)


def test_one_broken_store_never_takes_the_surface_down(monkeypatch):
    """The instrument built to catch silent gaps must not go silent itself."""
    monkeypatch.setattr(pr, "_BUILDERS", tuple(
        _broken(name) for name in (
            "_playbook_input", "_risk_constraints_input",
            "_wealth_goal_input", "_feedback_ratings_input",
        )
    ))

    readiness = pr.build_profile_readiness()

    assert readiness["counts"]["unreadable"] == 4
    assert readiness["counts"]["empty"] == 0
    assert {row["key"] for row in readiness["inputs"]} == {
        "drawdown_playbook", "risk_constraints", "wealth_goal", "feedback_ratings"
    }
