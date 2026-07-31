"""
4.7a — the tax jurisdiction of an ACCOUNT, stated rather than inferred.

Until this store existed the only evidence in the codebase was the account NAME.
That heuristic is load-bearing and has been wrong twice in ways that INVERTED the
answer — `"ISA" in acc_upper` matched *Visa*, `"REGISTERED" in acc_upper` matched
*Non-Registered* — and it is simply silent for "Brokerage", "Joint" or "Pension",
which name a class without naming a country. `REGIONAL_LOCALE` was never an
option: it is a display setting, and one household can hold accounts in two
countries, so jurisdiction belongs on the account.

Four properties carry the weight, and each has a failure it prevents:

**What the user stated beats what the name implies.** The name stays as the
fallback — it is right far more often than not and it needs nobody to type
anything — but it is no longer the ceiling, and `jurisdiction_source` says which
answered. A stated country and a substring match are not equally good evidence
for a rule that decides whether a loss is deferred or destroyed.

**UNKNOWN is an ANSWER.** An account the user has marked UNKNOWN fails closed
exactly like an unanswered one and is reported differently. This is 2.9's
`unconstrained_ack` lesson applied to a second store: without the third state, a
finished profile and an untouched one are identical from downstream.

**A code with no policy module is stored, not refused.** Which jurisdictions the
engines cover is the engines' statement to make. Refusing "DE" at entry would
make an uncovered country indistinguishable from a typo and would push the user
toward naming a country we happen to support rather than the one their account is
in.

**The screen is drawn from the PORTFOLIO, not from the store.** A store keyed by
free text can be full and match nothing, and from inside the store that is
indistinguishable from working — this repo's most repeated failure. Entries that
have stopped matching an account are named rather than swept up.
"""
import pytest

import tools.asset_location as al
import tools.memory as mem
import tools.tax_policy as tp


@pytest.fixture
def profile(monkeypatch):
    """An isolated in-memory store standing in for the profile's memory file."""
    state: dict = {}

    def _save(m):
        # REPLACE, not merge: `set_account_jurisdictions(None)` clears by popping
        # the key, and a merging fake would keep it so the clear would look like
        # a no-op that passed.
        state.clear()
        state.update(m)

    monkeypatch.setattr(mem, "load_memory", lambda: dict(state))
    monkeypatch.setattr(mem, "save_memory", _save)
    return state


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

def test_a_stated_code_round_trips_under_a_normalized_key(profile):
    result = mem.set_account_jurisdictions({"  rrsp   Spousal ": "ca"})
    assert result["ok"] is True
    # Upper-cased, trimmed, internal whitespace collapsed. The same account
    # arrives as "RRSP  Spousal" from one source and "rrsp spousal" from another,
    # and a key that fails to match its own account is a filled-in store that
    # changes nothing.
    assert mem.get_account_jurisdictions() == {"RRSP SPOUSAL": "CA"}


def test_unknown_is_stored_as_an_answer_rather_than_dropped(profile):
    mem.set_account_jurisdictions({"Pension": mem.JURISDICTION_UNKNOWN})
    assert mem.get_account_jurisdictions() == {"PENSION": "UNKNOWN"}


def test_a_code_with_no_policy_module_is_accepted(profile):
    # Germany has no module here. Storing it is correct: the ENGINE reports
    # `not_covered`, and refusing entry would make an uncovered jurisdiction look
    # like a typo and nudge the user toward a country we happen to support.
    assert mem.set_account_jurisdictions({"Depot": "DE"})["ok"] is True
    assert mem.get_account_jurisdictions()["DEPOT"] == "DE"


def test_junk_is_refused_by_name_and_nothing_is_written(profile):
    result = mem.set_account_jurisdictions({"TFSA": "CA", "Roth": "Canada"})
    assert result["ok"] is False
    assert "ROTH → CANADA" in result["rejected"]
    # The valid half of the payload must not land either. A partial write here
    # stores a plan the user did not type, and the refusal they saw would be
    # about a store that had already changed.
    assert mem.get_account_jurisdictions() == {}


def test_a_blank_value_removes_that_account_rather_than_storing_empty(profile):
    mem.set_account_jurisdictions({"TFSA": "CA", "Roth": "US"})
    mem.set_account_jurisdictions({"TFSA": "CA", "Roth": ""})
    # Back to unanswered, which means the name inference applies again — this is
    # how a stated country is taken back.
    assert mem.get_account_jurisdictions() == {"TFSA": "CA"}


def test_clearing_every_account_removes_the_block(profile):
    mem.set_account_jurisdictions({"TFSA": "CA"})
    assert mem.set_account_jurisdictions(None)["cleared"] is True
    assert mem.get_account_jurisdictions() == {}
    assert mem.get_account_jurisdictions_record() is None


def test_an_unreadable_store_reads_as_nothing_stated(profile, monkeypatch):
    monkeypatch.setattr(mem, "load_memory", lambda: {"account_jurisdictions": "junk"})
    # The safe direction: fall back to name inference, never to a guess.
    assert mem.get_account_jurisdictions() == {}


# ---------------------------------------------------------------------------
# Resolution — stated beats inferred
# ---------------------------------------------------------------------------

def test_the_name_still_answers_when_nothing_is_stated():
    resolved = al.classify_account("TFSA", jurisdictions={})
    assert resolved["jurisdiction"] == "CA"
    assert resolved["jurisdiction_source"] == "inferred_from_name"
    assert resolved["jurisdiction_conflict"] is False


def test_a_stated_country_overrides_the_name_and_says_they_disagree():
    # A "TFSA"-named account the user says is US. Theirs wins — they know what
    # the account is — and the disagreement is published rather than resolved,
    # because one of the two is wrong and only they know which.
    resolved = al.classify_account("TFSA", jurisdictions={"TFSA": "US"})
    assert resolved["jurisdiction"] == "US"
    assert resolved["jurisdiction_source"] == "stated"
    assert resolved["jurisdiction_inferred"] == "CA"
    assert resolved["jurisdiction_conflict"] is True


def test_a_stated_country_answers_a_name_that_names_no_country():
    # The case the store exists for: "Pension" is a class, not a country.
    named = al.classify_account("Company Pension", jurisdictions={})
    assert named["jurisdiction"] is None
    assert named["tax_class"] == "TAX_DEFERRED"

    stated = al.classify_account(
        "Company Pension", jurisdictions={"COMPANY PENSION": "GB"}
    )
    assert stated["jurisdiction"] == "GB"
    assert stated["jurisdiction_source"] == "stated"
    # No conflict: the name never made a competing claim.
    assert stated["jurisdiction_conflict"] is False


def test_declared_unknown_fails_closed_and_is_not_an_open_question():
    resolved = al.classify_account(
        "Brokerage One", jurisdictions={"BROKERAGE ONE": "UNKNOWN"}
    )
    assert resolved["jurisdiction"] is None
    # Fails closed like an unanswered account, and reads differently — which is
    # the entire point of the third state.
    assert resolved["jurisdiction_source"] == "declared_unknown"


def test_declared_unknown_overrides_a_name_that_did_infer_a_country():
    # "I know it says TFSA, and I cannot tell you the country." The user's
    # uncertainty must beat the substring match, or the answer they gave is
    # silently discarded in favour of a guess.
    resolved = al.classify_account("TFSA", jurisdictions={"TFSA": "UNKNOWN"})
    assert resolved["jurisdiction"] is None
    assert resolved["jurisdiction_source"] == "declared_unknown"


def test_the_shelter_and_tax_class_still_come_from_the_name():
    # Only the COUNTRY is overridden. A TFSA is a TFSA whatever country it is in,
    # and letting a stated jurisdiction rewrite the shelter would let a typo turn
    # a sheltered account into a taxable one.
    resolved = al.classify_account("TFSA", jurisdictions={"TFSA": "US"})
    assert resolved["shelter"] == "TFSA"
    assert resolved["tax_class"] == "TAX_FREE"


def test_the_two_name_defects_stay_fixed_with_the_store_in_play():
    # Regression pins. Both of these scored a fully taxable account as sheltered.
    visa = al.classify_account("Visa Infinite Card", jurisdictions={})
    assert visa["tax_class"] == "TAXABLE"
    assert visa["shelter"] is None

    non_reg = al.classify_account("Non-Registered", jurisdictions={})
    assert non_reg["tax_class"] == "TAXABLE"


def test_a_stated_entry_for_an_unrelated_account_changes_nothing():
    # The quiet failure this design has: keys that match nothing. It must not
    # leak onto a different account.
    resolved = al.classify_account("TFSA", jurisdictions={"OLD RRSP": "US"})
    assert resolved["jurisdiction"] == "CA"
    assert resolved["jurisdiction_source"] == "inferred_from_name"


def test_classify_account_reads_the_store_when_no_map_is_passed(profile):
    # The single-account callers in tax_policy pass nothing, so the default path
    # must consult the store. `{}` means "name inference only"; `None` means read.
    mem.set_account_jurisdictions({"Pension": "AU"})
    assert al.classify_account("Pension")["jurisdiction"] == "AU"
    assert al.classify_account("Pension", jurisdictions={})["jurisdiction"] is None


# ---------------------------------------------------------------------------
# The engines pick it up
# ---------------------------------------------------------------------------

def test_tax_policy_resolves_a_stated_jurisdiction_and_names_its_basis(profile):
    mem.set_account_jurisdictions({"Family Pension": "US"})
    resolved = tp.resolve_jurisdiction("Family Pension")
    assert resolved["resolved"] is True
    assert resolved["jurisdiction"] == "US"
    assert resolved["basis"] == "stated on the account"
    # Covered by a policy module, so a rebuy pre-check can actually run — this
    # account resolved to nothing at all before the store existed.
    assert resolved["covered"] is True


def test_tax_policy_marks_a_name_derived_country_as_an_inference(profile):
    resolved = tp.resolve_jurisdiction("RRSP")
    assert resolved["jurisdiction"] == "CA"
    assert resolved["basis"] == "account name"
    assert "not a stated fact" in resolved["basis_note"]


def test_tax_policy_reports_declared_unknown_as_answered_not_missing(profile):
    mem.set_account_jurisdictions({"Brokerage": mem.JURISDICTION_UNKNOWN})
    resolved = tp.resolve_jurisdiction("Brokerage")
    assert resolved["resolved"] is False
    assert resolved["basis"] == "declared_unknown"
    assert "recorded answer, not a gap" in resolved["note"]


def test_an_uncovered_stated_country_resolves_but_is_not_covered(profile):
    # The two states this must keep apart: we know the country, and we have no
    # rules for it. `not_covered` BLOCKS; it is not the same as "unknown".
    mem.set_account_jurisdictions({"Depot": "DE"})
    resolved = tp.resolve_jurisdiction("Depot")
    assert resolved["resolved"] is True
    assert resolved["jurisdiction"] == "DE"
    assert resolved["covered"] is False


# ---------------------------------------------------------------------------
# The entry screen's view
# ---------------------------------------------------------------------------

@pytest.fixture
def holdings(monkeypatch):
    """Two sheltered accounts and one taxable one, as the portfolio names them."""
    rows = [
        {"symbol": "VTI", "account": "TFSA", "value_base": 100.0},
        {"symbol": "XIC.TO", "account": "TFSA", "value_base": 100.0},
        {"symbol": "AAPL", "account": "Company Pension", "value_base": 100.0},
        {"symbol": "MSFT", "account": "Joint Margin", "value_base": 100.0},
    ]
    monkeypatch.setattr(al, "load_portfolio", lambda: rows)
    return rows


def test_the_view_lists_accounts_the_portfolio_names_not_the_store(profile, holdings):
    view = al.portfolio_account_jurisdictions()
    assert [a["account"] for a in view["accounts"]] == [
        "Company Pension", "Joint Margin", "TFSA",
    ]
    # Deduped: TFSA holds two positions and is one account to answer for.
    assert view["counts"]["accounts"] == 3


def test_a_taxable_account_is_not_reported_as_an_open_question(profile, holdings):
    view = al.portfolio_account_jurisdictions()
    margin = next(a for a in view["accounts"] if a["account"] == "Joint Margin")
    # Income taxed at the marginal rate has no shelter rule that could be wrong,
    # so there is no country to ask for. Counting it would invent a requirement.
    assert margin["jurisdiction_needed"] is False
    assert view["counts"]["need_jurisdiction"] == 2


def test_the_view_separates_stated_from_inferred_from_unanswered(profile, holdings):
    mem.set_account_jurisdictions({"Company Pension": "GB"})
    counts = al.portfolio_account_jurisdictions()["counts"]
    assert counts["stated"] == 1          # Company Pension, by the user
    assert counts["inferred_from_name"] == 1   # TFSA, by its name
    assert counts["unanswered"] == 0


def test_an_account_with_no_country_from_either_source_counts_as_unanswered(
    profile, holdings
):
    counts = al.portfolio_account_jurisdictions()["counts"]
    # "Company Pension" names a class without a country and nobody has said.
    assert counts["unanswered"] == 1


def test_a_stored_entry_matching_no_account_is_named(profile, holdings):
    mem.set_account_jurisdictions({"TFSA": "CA", "Closed RRSP 2019": "CA"})
    view = al.portfolio_account_jurisdictions()
    # Renamed, closed, or a typo. From inside the store it is indistinguishable
    # from a working entry, which is exactly why it is reported.
    assert view["stated_unmatched"] == ["CLOSED RRSP 2019"]


def test_the_view_carries_the_inference_so_a_blank_row_shows_what_it_keeps(
    profile, holdings
):
    tfsa = next(
        a for a in al.portfolio_account_jurisdictions()["accounts"]
        if a["account"] == "TFSA"
    )
    assert tfsa["stated"] is None
    assert tfsa["inferred_from_name"] == "CA"
    assert tfsa["jurisdiction"] == "CA"


# ---------------------------------------------------------------------------
# 2.8 reports it
# ---------------------------------------------------------------------------

def test_readiness_reports_an_unanswered_account_as_a_gap(profile, holdings):
    from tools.profile_readiness import build_profile_readiness

    row = next(
        i for i in build_profile_readiness()["inputs"]
        if i["key"] == "account_jurisdictions"
    )
    assert row["status"] == "partial"
    assert row["entry"] == "Context › Account Jurisdictions"
    # The consequence is the half anyone acts on.
    assert "SKIPPED" in row["consequence_by_field"]["jurisdictions"]


def test_readiness_names_the_accounts_running_on_a_guess(profile, holdings):
    from tools.profile_readiness import build_profile_readiness

    mem.set_account_jurisdictions({"Company Pension": "GB"})
    row = next(
        i for i in build_profile_readiness()["inputs"]
        if i["key"] == "account_jurisdictions"
    )
    # Nothing is unanswered now, so the row is SET — and the TFSA is still
    # running on a substring match, which is a weaker claim than the status
    # shows. Naming it is how the row avoids overstating what is on file.
    assert row["status"] == "set"
    assert row["observed"]["inferred_from_name"] == ["TFSA"]
