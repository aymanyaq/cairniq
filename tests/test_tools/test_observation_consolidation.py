"""The gated consolidation pass (Advisor Roadmap 1.7).

Consolidation is the deliverable, not the log: a write-only tier with no consumer
is this codebase's default outcome (feedback.json — 100 rows, 0 rated, unread by
the thing it exists to feed). These tests pin the three guards that let a
summarizer run over a behavioural store at all — the n gate, citation-or-discard,
and the per-pass cap — plus the boundary that matters most: this pass drafts,
it never learns.

Everything here is monkeypatched onto tmp_path, and no test reaches a model.
"""
import json
import types

import pytest

import tools.memory as mem
import tools.observation_consolidation as oc
import tools.observations as obs
import tools.pending_lessons as pl


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Point every per-profile store at tmp_path."""
    for module in (obs, pl, mem):
        monkeypatch.setattr(module, "get_data_path", lambda filename: str(tmp_path / filename))
    return tmp_path


@pytest.fixture
def llm(monkeypatch):
    """A stand-in model. Set `llm.reply` to the JSON the pass should receive."""
    import agent.utils as utils

    holder = types.SimpleNamespace(reply='{"proposals": []}', calls=0)

    def _safe_invoke(_client, _messages, **_kwargs):
        holder.calls += 1
        return types.SimpleNamespace(content=holder.reply)

    monkeypatch.setattr(utils, "llm_ready", lambda: (True, "ok"))
    monkeypatch.setattr(utils, "get_llm", lambda *a, **k: object())
    monkeypatch.setattr(utils, "safe_invoke", _safe_invoke)
    return holder


def _fill(n, kind=obs.KIND_ASKED):
    """n observations in the log, returned newest last."""
    return [
        obs.record_observation(kind, thread_id="t-1", span=f"question {i}", tickers=["YYYY"])
        for i in range(n)
    ]


def _proposal_json(*proposals):
    return json.dumps({"proposals": list(proposals)})


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_an_empty_log_drafts_nothing_and_says_so(store, llm):
    report = oc.consolidate_observations()

    assert report["gated"] is True
    assert report["drafted"] == 0
    assert "no unconsolidated observations" in report["reason"]
    assert llm.calls == 0


def test_below_the_gate_no_model_is_called(store, llm):
    """A summarizer pointed at a nearly-empty store is how this codebase produced
    invented history twice. The gate is what stops that."""
    _fill(oc.CONSOLIDATION_GATE_N - 1)

    report = oc.consolidate_observations()

    assert report["gated"] is True
    assert f"of {oc.CONSOLIDATION_GATE_N}" in report["reason"]
    assert llm.calls == 0
    assert len(obs.get_unconsolidated()) == oc.CONSOLIDATION_GATE_N - 1


def test_force_skips_the_gate_but_nothing_else(store, llm):
    rows = _fill(3)
    llm.reply = _proposal_json(
        {"rule": "Lead with the cost basis on YYYY.", "evidence_ids": [rows[0]["id"], rows[1]["id"]]},
    )

    report = oc.consolidate_observations(force=True)

    assert report["gated"] is False
    assert report["drafted"] == 1
    assert llm.calls == 1


def test_force_on_an_empty_log_still_drafts_nothing(store, llm):
    report = oc.consolidate_observations(force=True)

    assert report["drafted"] == 0
    assert llm.calls == 0


def test_an_unavailable_model_leaves_the_evidence_unread(store, llm, monkeypatch):
    """A provider outage must not silently burn a week of accumulated behaviour."""
    import agent.utils as utils
    monkeypatch.setattr(utils, "llm_ready", lambda: (False, "no credentials"))
    _fill(oc.CONSOLIDATION_GATE_N)

    report = oc.consolidate_observations()

    assert report["drafted"] == 0
    assert "LLM unavailable" in report["reason"]
    assert len(obs.get_unconsolidated()) == oc.CONSOLIDATION_GATE_N


# ---------------------------------------------------------------------------
# Citation or discard
# ---------------------------------------------------------------------------

def test_a_supported_proposal_becomes_a_draft_with_its_evidence(store, llm):
    rows = _fill(oc.CONSOLIDATION_GATE_N)
    cited = [rows[0]["id"], rows[1]["id"]]
    llm.reply = _proposal_json({"rule": "Always show YYYY's cost basis first.", "evidence_ids": cited})

    report = oc.consolidate_observations()

    assert report["drafted"] == 1
    drafts = pl.list_pending_lessons()
    assert len(drafts) == 1
    assert drafts[0]["text"] == "Always show YYYY's cost basis first."
    assert drafts[0]["source"] == "observation_consolidation"
    assert drafts[0]["evidence"]["observation_ids"] == sorted(cited)
    assert drafts[0]["evidence"]["kinds"] == [obs.KIND_ASKED]


def test_a_proposal_citing_an_id_that_was_never_shown_is_dropped(store, llm):
    """The anti-fabrication guard. An uncheckable claim about the user's own
    behaviour is worse than no rule at all — and it is dropped, never repaired:
    trimming the bad id off would leave a rule whose stated support is not the
    support it was written from."""
    rows = _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = _proposal_json(
        {"rule": "The user always sells into strength.", "evidence_ids": [rows[0]["id"], "ghost-id"]},
    )

    report = oc.consolidate_observations()

    assert report["drafted"] == 0
    assert report["dropped"] == 1
    assert pl.list_pending_lessons() == []


def test_a_proposal_resting_on_a_single_observation_is_dropped(store, llm):
    """One row is not a pattern; a rule drafted off it is a rule about one turn."""
    rows = _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = _proposal_json({"rule": "Never mention YYYY.", "evidence_ids": [rows[0]["id"]]})

    report = oc.consolidate_observations()

    assert report["drafted"] == 0
    assert report["dropped"] == 1


def test_an_empty_proposal_list_is_a_correct_answer(store, llm):
    """Ordinary use contains no durable rule most weeks. Drafting nothing is the
    expected outcome, not a failure."""
    _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = '{"proposals": []}'

    report = oc.consolidate_observations()

    assert report["drafted"] == 0
    assert report["dropped"] == 0
    assert pl.list_pending_lessons() == []


def test_unparseable_output_drafts_nothing(store, llm):
    _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = "I think the user prefers dividends."

    report = oc.consolidate_observations()

    assert report["drafted"] == 0
    assert pl.list_pending_lessons() == []


def test_one_pass_cannot_flood_the_confirm_queue(store, llm):
    rows = _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = _proposal_json(*[
        {"rule": f"Rule number {i}.", "evidence_ids": [rows[0]["id"], rows[1]["id"]]}
        for i in range(oc.MAX_DRAFTS_PER_PASS + 3)
    ])

    report = oc.consolidate_observations()

    assert report["drafted"] == oc.MAX_DRAFTS_PER_PASS
    assert len(pl.list_pending_lessons()) == oc.MAX_DRAFTS_PER_PASS


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def test_a_drafted_rule_is_not_in_effect(store, llm):
    """The whole contract. lessons_learned is injected into every prompt and
    capped, so an unreviewed entry does not add noise — it competes with a rule
    the user wrote."""
    rows = _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = _proposal_json(
        {"rule": "Lead with the cost basis.", "evidence_ids": [rows[0]["id"], rows[1]["id"]]},
    )

    oc.consolidate_observations()

    assert mem.load_memory().get("lessons_learned", []) == []
    assert len(pl.list_pending_lessons()) == 1


def test_the_batch_is_marked_read_even_when_it_produced_nothing(store, llm):
    """Evidence that produced no rule has still been considered. Leaving it
    unread would re-propose the same non-pattern on every future pass."""
    _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = '{"proposals": []}'

    oc.consolidate_observations()

    assert obs.get_unconsolidated() == []
    assert obs.get_observation_stats()["last_consolidated_at"] is not None


def test_a_second_pass_over_the_same_evidence_is_gated(store, llm):
    rows = _fill(oc.CONSOLIDATION_GATE_N)
    llm.reply = _proposal_json(
        {"rule": "Lead with the cost basis.", "evidence_ids": [rows[0]["id"], rows[1]["id"]]},
    )
    oc.consolidate_observations()

    second = oc.consolidate_observations()

    assert second["gated"] is True
    assert llm.calls == 1


def test_a_forced_pass_still_refuses_below_the_citation_floor(store, llm):
    """`force` skips the cadence gate, not the floor. Below MIN_CITATIONS rows
    every possible proposal fails validation, so calling a model would spend
    money to produce discarded output — and hand it the almost-empty store that
    has produced invented history in this codebase before."""
    _fill(oc.MIN_CITATIONS - 1)

    report = oc.consolidate_observations(force=True)

    assert report["gated"] is True
    assert "to cite" in report["reason"]
    assert llm.calls == 0


# ---------------------------------------------------------------------------
# The scheduled job
# ---------------------------------------------------------------------------

@pytest.fixture
def one_profile(monkeypatch):
    """A single profile, run in place. The real runner would iterate the live
    profile list, which no test may touch."""
    import tools.user_profile as up

    monkeypatch.setattr(up, "list_available_profiles", lambda: [{"name": "tester"}])
    monkeypatch.setattr(up, "run_under_profile", lambda name, fn, *a, **k: fn(*a, **k))


def _drive_task(monkeypatch, llm_ok):
    import asyncio

    import tools.scheduler as sched

    reported = {}
    monkeypatch.setattr(sched, "_skip_if_llm_unready", lambda name: not llm_ok)
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: True)
    monkeypatch.setattr(
        sched, "_note_engine_outcome",
        lambda worked, produced, declined, detail="": reported.update(
            worked=worked, produced=produced, detail=detail
        ),
    )
    asyncio.run(sched.task_observation_consolidation())
    return reported


def test_the_job_sweeps_follow_through_even_with_no_model_configured(store, one_profile, monkeypatch):
    """The two halves ride different rails on purpose: the follow-through sweep
    is zero-LLM and must not be held hostage by a provider outage."""
    row = obs.record_rec_issued("YYYY", "SELL", shares_at_advice=40.0)
    data = obs.load_observations()
    data["observations"][0]["timestamp"] = "2020-01-01T00:00:00"
    obs.save_observations(data)
    monkeypatch.setattr(obs, "load_holdings_map", lambda: {"YYYY": 0.0})

    reported = _drive_task(monkeypatch, llm_ok=False)

    assert reported["worked"] == 1
    assert "1 calls resolved" in reported["detail"]
    assert "sweep only" in reported["detail"]
    assert obs.load_observations()["observations"][-1]["kind"] == obs.KIND_REC_FOLLOWED
    assert row["id"]


def test_the_job_reports_the_chain_not_the_rare_event(store, one_profile, monkeypatch):
    """2.6: production is OBSERVATIONS WALKED. Past the gate most passes draft
    nothing — ordinary use contains no durable rule — so counting drafts would
    put a working engine on an idle streak within a week."""
    _fill(3)
    monkeypatch.setattr(obs, "load_holdings_map", lambda: {})

    reported = _drive_task(monkeypatch, llm_ok=False)

    assert reported["produced"] == 3
    assert "0 rules drafted" in reported["detail"]


def test_a_model_that_fails_to_build_is_a_clean_skip_not_a_500(store, llm, monkeypatch):
    """Caught live on 2026-07-27: llm_ready() cleared while get_llm() raised
    (provider selected without its model id), which 500'd the endpoint and would
    have errored every profile on the scheduled pass. A credential fault is a
    skip — and the unread evidence has to survive it."""
    import agent.utils as utils
    monkeypatch.setattr(utils, "get_llm", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("LLM_PROVIDER=bedrock but AIDLC_MODEL_ID is not set.")
    ))
    _fill(oc.CONSOLIDATION_GATE_N)

    report = oc.consolidate_observations()

    assert report["drafted"] == 0
    assert "LLM unavailable" in report["reason"]
    assert len(obs.get_unconsolidated()) == oc.CONSOLIDATION_GATE_N
