"""
4.4's drift target — the store, and the reason it had to exist at all.

`check_rebalance_drift` had two routes to a target: an inline dict passed
per-call, or an optimizer objective that spends a solve. Neither is a plan a
person keeps. `target_allocation` had no store, no endpoint and no entry screen
anywhere in the codebase — it existed only as a function parameter — so the drift
check reported "nothing to drift against" as though the user had declined to
state one, when in fact nobody could tell us. That is the shape `risk_constraints`
was in before 2.9, misfiled the same way.

Two refusals carry the weight here, and both exist because the failure they
prevent is SILENT and produces real trade instructions:

**Weights are never rescaled.** A mix summing to 90% is refused, not stretched.
Stretching turns a deliberate 10% cash position into "spread that cash across
everything", and the drift check then emits BUY instructions for money the user
meant to hold. The remainder has to be named as its own sleeve.

**A zero-weight sleeve is refused, not dropped.** `CASH 0` vanishing while the
remaining sleeves happen to total 100% would store a plan the user did not type —
and the sum check that would catch it passes precisely because the dropped line
contributed nothing. (This one was a real defect in the first cut of this module,
caught by exercising it rather than by reading it.)

A third failure, found after the store shipped: the store being the DEFAULT is
only true if the caller knows it is. The agent-facing docstring still described
the two override routes and never mentioned the no-argument one, so the LLM would
supply an inline mix or an objective — and the drift check would answer against
that substitute while the user's own stored plan sat unread. `target_basis`
recorded which was used, but nothing said so where the answer gets read. Hence
the override reporting below, and the wrapper tests: the marshalling between the
LLM's string arguments and the core's typed ones had no coverage at all.
"""
import pytest

import agent.tool_registry as reg
import tools.memory as mem
import tools.portfolio_optimizer as po


@pytest.fixture
def profile(monkeypatch):
    """An isolated in-memory store standing in for the profile's memory file."""
    state: dict = {}

    def _save(m):
        # REPLACE, not merge. `set_target_allocation(None)` clears by popping the
        # key and saving; a merging fake would keep the popped key and the clear
        # would silently look like a no-op.
        state.clear()
        state.update(m)

    monkeypatch.setattr(mem, "load_memory", lambda: dict(state))
    monkeypatch.setattr(mem, "save_memory", _save)
    return state


VALID = {"VTI": 40, "VXUS": 30, "BND": 20, "CASH": 10}


# ---------------------------------------------------------------------------
# Unset is meaningful
# ---------------------------------------------------------------------------
def test_unset_is_none_and_is_never_filled_in(profile):
    """A target this app invented would be quoted back as the user's own plan
    and then used to generate BUY and SELL instructions against it."""
    assert mem.get_target_allocation() is None
    assert mem.get_target_allocation_record() is None


def test_clearing_removes_the_block(profile):
    mem.set_target_allocation(VALID)
    assert mem.get_target_allocation() is not None
    assert mem.set_target_allocation(None)["cleared"] is True
    assert mem.get_target_allocation() is None


# ---------------------------------------------------------------------------
# The two refusals
# ---------------------------------------------------------------------------
def test_a_mix_that_does_not_total_100_is_refused_not_rescaled(profile):
    result = mem.set_target_allocation({"VTI": 40, "VXUS": 30, "BND": 20})
    assert result["ok"] is False
    assert "sum to 90" in result["error"]
    assert "cash" in result["error"].lower(), "the fix must be named, not just the fault"
    assert mem.get_target_allocation() is None, "a refused write must not partially store"


def test_the_rescale_that_is_refused_would_have_invented_a_trade(profile):
    """Concretely: rescaling 90% to 100% moves 10 points of cash into equities."""
    raw = {"VTI": 40, "VXUS": 30, "BND": 20}
    total = sum(raw.values())
    rescaled = {k: v / total for k, v in raw.items()}
    assert rescaled["VTI"] == pytest.approx(0.4444, abs=1e-4)
    # 4.4 points of drift on one sleeve, out of nothing the user typed.
    assert (rescaled["VTI"] - 0.40) * 100 == pytest.approx(4.44, abs=0.01)
    assert mem.set_target_allocation(raw)["ok"] is False


def test_a_zero_weight_sleeve_is_refused_not_silently_dropped(profile):
    """The sum check cannot catch this: the dropped line contributes nothing, so
    the remainder totals 100% and the write looks clean."""
    result = mem.set_target_allocation({"VTI": 100, "CASH": 0})
    assert result["ok"] is False
    assert "CASH" in result["error"]
    assert result["rejected"] == ["CASH"]
    assert mem.get_target_allocation() is None


def test_an_unparseable_weight_is_refused_by_name(profile):
    result = mem.set_target_allocation({"VTI": 90, "VXUS": "ten"})
    assert result["ok"] is False
    assert "VXUS" in result["error"]


def test_an_all_empty_mix_is_refused(profile):
    result = mem.set_target_allocation({"VTI": 0})
    assert result["ok"] is False
    assert "no usable positive weights" in result["error"]


def test_a_tiny_rounding_gap_is_tolerated(profile):
    """A human splitting thirds types 33.33/33.33/33.34, and refusing that would
    make the store unusable by hand."""
    assert mem.set_target_allocation({"A": 33.33, "B": 33.33, "C": 33.34})["ok"] is True
    # 99.9 is 0.1 off — inside the 0.5 tolerance.
    assert mem.set_target_allocation({"A": 33.3, "B": 33.3, "C": 33.3})["ok"] is True
    # 99.0 is 1.0 off — outside it, and a whole missing point is a missing sleeve.
    assert mem.set_target_allocation({"A": 33.0, "B": 33.0, "C": 33.0})["ok"] is False


# ---------------------------------------------------------------------------
# What is stored
# ---------------------------------------------------------------------------
def test_a_valid_mix_stores_symbols_uppercased_with_its_metadata(profile):
    result = mem.set_target_allocation({" vti ": 40, "vxus": 30, "bnd": 20, "cash": 10},
                                       note="core + cash sleeve")
    assert result["ok"] is True
    record = mem.get_target_allocation_record()
    assert set(record["weights"]) == {"VTI", "VXUS", "BND", "CASH"}
    assert record["total_pct"] == 100.0
    assert record["note"] == "core + cash sleeve"
    assert record["set_at"]


def test_weights_are_stored_as_percentages_not_decimals(profile):
    """The consumer divides by the total; storing decimals would make a 100%
    mix read as 1% and the drift calculation silently wrong."""
    mem.set_target_allocation(VALID)
    assert mem.get_target_allocation()["VTI"] == 40.0


def test_a_suffixed_ticker_survives(profile):
    mem.set_target_allocation({"SHOP.TO": 50, "BRK.B": 50})
    assert set(mem.get_target_allocation()) == {"SHOP.TO", "BRK.B"}


# ---------------------------------------------------------------------------
# The consumer
# ---------------------------------------------------------------------------
def test_drift_names_the_entry_screen_when_nothing_is_stored(monkeypatch):
    """The 2.9 lesson: a store belongs on 'blocked on you' only once a human can
    reach it. Before this, the reason said nothing about where to go."""
    monkeypatch.setattr(po, "_playbook", lambda: {"rebalance_drift_pct": 10.0})
    monkeypatch.setattr(po, "_stored_target_allocation", lambda: None)
    monkeypatch.setattr(po, "_decision_context", lambda: {
        "holdings": [{"symbol": "VTI", "value_base": 1000.0, "is_cash_or_pension": False}]
    })

    result = po.check_rebalance_drift()
    assert result["available"] is False
    assert "never invented" in result["reason"]
    assert "Target Allocation" in result["reason"]
    assert result["entry_screen"] == "/context › Target Allocation"


def test_drift_uses_the_stored_target_when_no_argument_is_given(monkeypatch):
    """The branch that makes the drift check reachable without a caller
    supplying a target per call."""
    monkeypatch.setattr(po, "_playbook", lambda: {"rebalance_drift_pct": 10.0})
    monkeypatch.setattr(po, "_stored_target_allocation", lambda: {"VTI": 50.0, "VXUS": 50.0})
    monkeypatch.setattr(po, "_decision_context", lambda: {
        "holdings": [
            {"symbol": "VTI", "value_base": 8000.0, "is_cash_or_pension": False},
            {"symbol": "VXUS", "value_base": 2000.0, "is_cash_or_pension": False},
        ]
    })

    result = po.check_rebalance_drift()
    assert result["available"] is True
    assert result["target_basis"] == "stored"
    # 80/20 held against a 50/50 target is 30 points of drift, past a 10% band.
    assert result["drift_pct_by_symbol"]["VTI"] == pytest.approx(30.0, abs=0.01)
    assert result["breached"] is True


def test_an_explicit_argument_still_wins_over_the_store(monkeypatch):
    """A caller asking a what-if must not silently get the saved plan instead —
    and must not silently get the what-if reported AS the saved plan either."""
    monkeypatch.setattr(po, "_playbook", lambda: {"rebalance_drift_pct": 10.0})
    monkeypatch.setattr(po, "_stored_target_allocation", lambda: {"VTI": 50.0, "VXUS": 50.0})
    monkeypatch.setattr(po, "_decision_context", lambda: {
        "holdings": [
            {"symbol": "VTI", "value_base": 8000.0, "is_cash_or_pension": False},
            {"symbol": "VXUS", "value_base": 2000.0, "is_cash_or_pension": False},
        ]
    })

    result = po.check_rebalance_drift(target_allocation={"VTI": 80, "VXUS": 20})
    assert result["target_basis"] == "explicit"
    assert result["breached"] is False
    assert result["stored_target_overridden"]["stored_weights_pct"] == {"VTI": 50.0, "VXUS": 50.0}


# ---------------------------------------------------------------------------
# Naming the basis — an override is not the plan
# ---------------------------------------------------------------------------
def _drift_seams(monkeypatch, stored, band=10.0):
    """Holdings at 80/20, so a 50/50 stored plan is 30 points out of band."""
    monkeypatch.setattr(po, "_playbook", lambda: {"rebalance_drift_pct": band})
    monkeypatch.setattr(po, "_stored_target_allocation", lambda: stored)
    monkeypatch.setattr(po, "_decision_context", lambda: {
        "holdings": [
            {"symbol": "VTI", "value_base": 8000.0, "is_cash_or_pension": False},
            {"symbol": "VXUS", "value_base": 2000.0, "is_cash_or_pension": False},
        ]
    })


def test_the_verdict_says_when_a_stored_plan_was_overridden(monkeypatch):
    """`target_basis` is a machine field; the verdict is the line that gets read
    back. "Within band" against a substituted target, next to a stored plan the
    book is 30 points away from, is a wrong answer to the question asked."""
    _drift_seams(monkeypatch, {"VTI": 50.0, "VXUS": 50.0})

    result = po.check_rebalance_drift(target_allocation={"VTI": 80, "VXUS": 20})
    assert result["breached"] is False
    assert "Within band" in result["verdict"]
    assert "stored target allocation was overridden" in result["verdict"]
    assert "supplied with this call" in result["verdict"]


def test_the_overridden_plan_is_quoted_so_the_substitution_is_visible(monkeypatch):
    """Reporting "you overrode something" without saying what is not enough to
    tell whether the override mattered."""
    _drift_seams(monkeypatch, {"VTI": 50.0, "VXUS": 50.0})

    result = po.check_rebalance_drift(target_allocation={"VTI": 80, "VXUS": 20})
    flag = result["stored_target_overridden"]
    assert flag["stored_weights_pct"] == {"VTI": 50.0, "VXUS": 50.0}
    assert flag["measured_against"] == result["target_source"]
    assert "re-run with no target_allocation and no objective" in flag["note"]


def test_an_optimizer_objective_overrides_the_stored_plan_too(monkeypatch):
    """The optimizer route substitutes a computed mix for a stated one, which is
    the substitution 4.4 exists to prevent — it is reported the same way."""
    _drift_seams(monkeypatch, {"VTI": 50.0, "VXUS": 50.0})
    monkeypatch.setattr(po, "optimize_portfolio", lambda **kw: {
        "available": True,
        "optimized_weights_pct": {"VTI": 70.0, "VXUS": 30.0},
        "held_constant_pct": {},
    })

    result = po.check_rebalance_drift(objective="min_vol")
    assert result["target_basis"] == "optimizer:min_vol"
    assert "not a plan the user has stated" in result["target_source"]
    assert result["stored_target_overridden"]["stored_weights_pct"] == {"VTI": 50.0, "VXUS": 50.0}
    assert "stored target allocation was overridden" in result["verdict"]


def test_nothing_is_flagged_when_there_was_no_stored_plan_to_override(monkeypatch):
    """A user who never stated a target has had nothing substituted, and a flag
    saying otherwise would send the agent to a screen with nothing on it."""
    _drift_seams(monkeypatch, None)

    result = po.check_rebalance_drift(target_allocation={"VTI": 80, "VXUS": 20})
    assert result["available"] is True
    assert "stored_target_overridden" not in result
    assert "overridden" not in result["verdict"]


def test_the_stored_route_names_its_own_source_and_flags_no_override(monkeypatch):
    _drift_seams(monkeypatch, {"VTI": 50.0, "VXUS": 50.0})

    result = po.check_rebalance_drift()
    assert result["target_basis"] == "stored"
    assert "Context › Target Allocation" in result["target_source"]
    assert "stored_target_overridden" not in result


def test_an_unreadable_store_cannot_break_a_check_that_did_not_need_it(monkeypatch):
    """The override note reads the store only to report on it. If that read
    fails, the explicit-target answer is still correct and still owed — setting
    a target allocation must not be able to BREAK the what-if path."""
    def _boom():
        raise OSError("memory file is gone")

    _drift_seams(monkeypatch, None)
    monkeypatch.setattr(po, "_stored_target_allocation", _boom)

    result = po.check_rebalance_drift(target_allocation={"VTI": 80, "VXUS": 20})
    assert result["available"] is True
    assert result["target_basis"] == "explicit"
    assert "stored_target_overridden" not in result


# ---------------------------------------------------------------------------
# The agent path — the wrapper the LLM actually calls
# ---------------------------------------------------------------------------
# The store, the endpoint and the entry screen were all covered; the marshalling
# between the LLM's string arguments and the core's typed ones was not. It is the
# only code between a model that reads the docstring and the check that answers.


@pytest.fixture
def drift_spy(monkeypatch):
    """Captures what the registry wrapper marshals into the core."""
    calls = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return {"available": True, "target_basis": "spy"}

    monkeypatch.setattr(po, "check_rebalance_drift", _fake)
    return calls


def test_the_no_argument_call_reaches_the_stored_route(drift_spy):
    """The documented default. All three overrides must arrive as None, because
    any one of them being "" or 0.0 instead would silently take a branch of its
    own before the stored plan is ever looked at."""
    reg.check_rebalance_drift.invoke({})
    assert drift_spy == [{"target_allocation": None, "objective": None,
                          "target_vol_pct": None, "period": "1y"}]


def test_a_json_target_is_parsed_into_the_dict_the_core_expects(drift_spy):
    reg.check_rebalance_drift.invoke({"target_allocation": '{"VTI": 60, "VXUS": 40}'})
    assert drift_spy[0]["target_allocation"] == {"VTI": 60, "VXUS": 40}


def test_a_whitespace_only_target_is_treated_as_no_target(drift_spy):
    reg.check_rebalance_drift.invoke({"target_allocation": "   "})
    assert drift_spy[0]["target_allocation"] is None


def test_an_override_that_is_explicitly_empty_still_routes_to_the_store(drift_spy):
    """`{}` reaches the core as an empty dict, which is falsy there, so it falls
    through to the stored plan rather than being resolved as a target of no
    sleeves. The answer says `target_basis: stored`, so the route is reported
    rather than guessed at silently."""
    reg.check_rebalance_drift.invoke({"target_allocation": "{}"})
    assert drift_spy[0]["target_allocation"] == {}


def test_malformed_json_is_refused_before_the_check_runs(drift_spy):
    out = reg.check_rebalance_drift.invoke({"target_allocation": "VTI 60, VXUS 40"})
    assert out["available"] is False
    assert "JSON object" in out["reason"]
    assert drift_spy == [], "a target that could not be read must not fall back to another one"


def test_valid_json_that_is_not_an_object_is_refused_by_shape(drift_spy):
    """A list of sleeves, a bare number, a quoted string and `null` all parse.
    None of them is a symbol -> weight mapping, and the last would otherwise
    become "no target supplied" and quietly measure against the stored plan."""
    for payload in ('[{"VTI": 60}]', "null", '"VTI 60"', "60"):
        out = reg.check_rebalance_drift.invoke({"target_allocation": payload})
        assert out["available"] is False, payload
        assert "JSON object" in out["reason"], payload
    assert drift_spy == []


def test_empty_scalars_become_none_and_real_ones_pass_through(drift_spy):
    """`objective=""` and `target_vol_pct=0.0` are the schema's defaults, not
    choices: passed through as-is, "" would take the optimizer branch and solve
    for an objective nobody named."""
    reg.check_rebalance_drift.invoke({"objective": "", "target_vol_pct": 0.0})
    assert drift_spy[0]["objective"] is None
    assert drift_spy[0]["target_vol_pct"] is None

    reg.check_rebalance_drift.invoke(
        {"objective": "target_vol", "target_vol_pct": 12.0, "period": "3y"}
    )
    assert drift_spy[1]["objective"] == "target_vol"
    assert drift_spy[1]["target_vol_pct"] == 12.0
    assert drift_spy[1]["period"] == "3y"


def test_the_agent_path_end_to_end_measures_against_the_stored_plan(monkeypatch):
    """The whole chain the docstring now points at: no arguments from the LLM,
    through the wrapper, to drift against the user's own stated mix."""
    _drift_seams(monkeypatch, {"VTI": 50.0, "VXUS": 50.0})

    result = reg.check_rebalance_drift.invoke({})
    assert result["available"] is True
    assert result["target_basis"] == "stored"
    assert result["drift_pct_by_symbol"]["VTI"] == pytest.approx(30.0, abs=0.01)


def test_the_agent_path_names_the_entry_screen_when_nothing_is_stored(monkeypatch):
    _drift_seams(monkeypatch, None)

    result = reg.check_rebalance_drift.invoke({})
    assert result["available"] is False
    assert result["entry_screen"] == "/context › Target Allocation"


def test_the_docstring_documents_the_no_argument_route(monkeypatch):
    """The stale docstring named only the two override routes, so an LLM reading
    it would infer it had to supply one — and the stored branch, the reason 4.4
    was built, would never be exercised from the agent path."""
    doc = reg.check_rebalance_drift.description
    assert "NO ARGUMENTS" in doc
    assert "Target Allocation" in doc
    assert "OVERRIDES" in doc
    assert doc.index("NO ARGUMENTS") < doc.index("OVERRIDES"), \
        "the default has to be read before the overrides it is being contrasted with"


def test_the_stored_target_still_needs_the_playbook_band(monkeypatch):
    """Two independent inputs. Supplying one does not switch the check on, and
    the roadmap's claim that the band alone unblocked this was incomplete."""
    monkeypatch.setattr(po, "_playbook", lambda: {})
    monkeypatch.setattr(po, "_stored_target_allocation", lambda: {"VTI": 100.0})
    result = po.check_rebalance_drift()
    assert result["available"] is False
    assert "rebalance_drift_pct" in result["reason"]


# ---------------------------------------------------------------------------
# Reachability — the whole point of the item
# ---------------------------------------------------------------------------
def test_the_store_has_a_writer_a_human_can_reach(tmp_path, monkeypatch):
    """`risk_constraints` sat empty for months filed as blocked-on-the-user while
    having no entry screen at all. The rule learned from it: a store is not
    shipped until a human can reach its writer without curl."""
    from fastapi.testclient import TestClient

    from server import app

    state: dict = {}

    def _save(m):
        state.clear()
        state.update(m)

    monkeypatch.setattr(mem, "load_memory", lambda: dict(state))
    monkeypatch.setattr(mem, "save_memory", _save)
    client = TestClient(app)

    assert client.get("/api/memory/target_allocation").json()["target_allocation"] is None

    ok = client.post("/api/memory/target_allocation",
                     json={"weights": VALID, "note": "core"})
    assert ok.status_code == 200
    assert ok.json()["target_allocation"]["total_pct"] == 100.0

    read = client.get("/api/memory/target_allocation").json()["target_allocation"]
    assert set(read["weights"]) == set(VALID)


def test_a_refused_mix_is_a_400_carrying_the_reason(tmp_path, monkeypatch):
    """apiCall surfaces `error` from a non-OK body straight into a toast, so the
    sum-mismatch message is what the user actually reads."""
    from fastapi.testclient import TestClient

    from server import app

    state: dict = {}
    monkeypatch.setattr(mem, "load_memory", lambda: dict(state))
    monkeypatch.setattr(mem, "save_memory", lambda m: (state.clear(), state.update(m)))

    res = TestClient(app).post("/api/memory/target_allocation",
                               json={"weights": {"VTI": 40, "VXUS": 30}})
    assert res.status_code == 400
    assert "sum to 70" in res.json()["error"]


def test_the_entry_screen_is_on_the_context_page():
    from fastapi.testclient import TestClient

    from server import app

    page = TestClient(app).get("/context").text
    assert 'id="ta-weights"' in page
    assert "saveTargetAllocation()" in page
    assert "/api/memory/target_allocation" in page
