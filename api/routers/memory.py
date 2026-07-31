from typing import Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tools.graph_memory import graph_memory
from tools.memory import (
    add_active_thesis,
    add_fact,
    add_lesson,
    delete_active_thesis,
    delete_key_fact,
    delete_lesson,
    extract_thesis_from_text,
    get_scored_recommendations_data,
    load_memory,
    save_memory,
    set_risk_constraints,
    update_active_thesis,
    update_lesson,
    update_profile,
)

router = APIRouter()

class ProfileUpdateRequest(BaseModel):
    updates: dict[str, Any]

class ThesisRequest(BaseModel):
    id: str | None = None
    symbol: str
    action: str
    catalyst: str | None = None
    conditions: str | None = None
    stop_loss: str | None = None
    expiry_date: str | None = None
    target_price: str | None = None
    notes: str | None = None

class LessonRequest(BaseModel):
    index: int | None = None
    text: str

class NodeRequest(BaseModel):
    name: str
    type: str

class EdgeRequest(BaseModel):
    source: str
    target: str
    relation: str

@router.post("/api/memory/profile")
async def api_update_profile(req: ProfileUpdateRequest):
    update_profile(req.updates)
    return {"status": "success"}

@router.get("/api/memory/risk_constraints")
async def api_get_risk_constraints():
    """The user's stated risk limits, and whether the blanks among them are
    blank ON PURPOSE.

    An empty block still means no limits are set — that contract does not
    change. What `execution_readiness` adds is the second question the store
    could not previously answer: whether the user has been asked. A cap nobody
    ever put to them and a cap they deliberately declined produce the same empty
    block, and only the second makes a sized proposal execution-ready.
    """
    from tools.ips_precheck import execution_readiness, load_ips_constraints, stated_caps

    constraints = load_ips_constraints()
    return {
        "stated": stated_caps(constraints),
        "restricted_symbols": constraints.get("restricted_symbols", []),
        "execution_readiness": execution_readiness(constraints),
    }


@router.post("/api/memory/risk_constraints")
async def api_set_risk_constraints(req: ProfileUpdateRequest):
    """Set or clear risk limits. A null value clears that limit (unconstrained).

    `acknowledge_unconstrained: true` additionally records that every axis left
    blank BY THIS WRITE is blank by choice; `false` withdraws that. It is the
    only way the app learns the difference, and it is a user action by
    construction — nothing here infers it, and no cap is ever authored either
    way.
    """
    from tools.ips_precheck import execution_readiness

    constraints = set_risk_constraints(req.updates)
    return {
        "status": "success",
        "risk_constraints": constraints,
        "execution_readiness": execution_readiness(),
    }


@router.get("/api/memory/financial_goal")
async def api_get_financial_goal():
    """The user's stated wealth goal. `null` means nothing is set.

    Same contract as risk_constraints: unset is MEANINGFUL and is never filled
    in with a plausible-looking default. Everything downstream (required return,
    goal-funded probability, the goal panel) reads as unavailable until the user
    states a target — a projection against an assumed goal would be a decade of
    decisions anchored to a number nobody chose.
    """
    from tools.memory import get_financial_goal

    return {"goal": get_financial_goal()}


@router.post("/api/memory/financial_goal")
async def api_set_financial_goal(req: ProfileUpdateRequest):
    """Set or clear the wealth goal. Keys: target_low, target_high,
    horizon_years, annual_contribution. A null value clears that field; a
    malformed one leaves the existing figure standing, so a typo cannot silently
    erase a goal."""
    from tools.memory import set_financial_goal

    return {"status": "success", "goal": set_financial_goal(req.updates)}


@router.get("/api/memory/drawdown_playbook")
async def api_get_drawdown_playbook():
    """The user's pre-agreed drawdown rules (Roadmap 3.7). `null` = none on file.

    Same contract as risk_constraints and the wealth goal, and it matters more
    here: these rules get read back during a crash, when they carry the full
    authority of a promise the user made to themselves. Nothing may be defaulted
    — an unset playbook surfaces as "none on file", never as suggestions.
    """
    from tools.drawdown_playbook import get_playbook

    return {"playbook": get_playbook()}


@router.post("/api/memory/drawdown_playbook")
async def api_set_drawdown_playbook(req: ProfileUpdateRequest):
    """Set or clear playbook fields: never_sell, buy_first (ORDER IS THE
    INSTRUCTION), deployment_levels, rebalance_drift_pct, notes. None clears a
    field; a malformed value is rejected and leaves the existing rule standing."""
    from tools.drawdown_playbook import set_playbook

    return {"status": "success", "playbook": set_playbook(req.updates)}


@router.get("/api/memory/target_allocation")
async def api_get_target_allocation():
    """The user's stated target sleeve mix. `null` means they have never set one.

    Same contract as the wealth goal and risk_constraints: unset is MEANINGFUL
    and is never filled in with a plausible-looking default. A target allocation
    this app invented would be quoted back as the user's own plan and then used
    to generate BUY and SELL instructions against it.
    """
    from tools.memory import get_target_allocation_record

    return {"target_allocation": get_target_allocation_record()}


@router.post("/api/memory/target_allocation")
async def api_set_target_allocation(payload: dict = Body(...)):
    """Store or clear the target sleeve mix. `weights: null` or `{}` clears it.

    Weights are percentages and are NOT rescaled. A mix that does not sum to
    ~100% is refused with its own total quoted back, because silently rescaling a
    90% mix turns a deliberate 10% cash position into "spread it across
    everything" — which then emits real BUY instructions for money the user meant
    to hold. Name cash as its own sleeve instead.
    """
    from tools.memory import set_target_allocation

    result = set_target_allocation(payload.get("weights"), note=payload.get("note", ""))
    if not result.get("ok"):
        return JSONResponse(result, status_code=400)
    return {"status": "success", **result}


@router.get("/api/memory/account_jurisdictions")
async def api_get_account_jurisdictions():
    """Every account the portfolio names, and how its tax jurisdiction resolves.

    Not just the stored map (4.7a): the accounts the holdings ACTUALLY name,
    each with what you stated, what the account's name implies, which of the two
    is in force, and whether they disagree. A store keyed by free text can be
    full and still match nothing, so the screen is drawn from the portfolio
    rather than from the store — `stated_unmatched` names any entry that has
    stopped matching an account.
    """
    from tools.asset_location import portfolio_account_jurisdictions

    return portfolio_account_jurisdictions()


@router.post("/api/memory/account_jurisdictions")
async def api_set_account_jurisdictions(payload: dict = Body(...)):
    """Store or clear the per-account tax jurisdictions. `accounts: null` clears.

    Values are two-letter country codes, or `UNKNOWN` to record that the question
    was asked and the account has no jurisdiction the user can name — an ANSWER,
    which fails closed exactly like an unanswered account but is reported
    differently. A blank value removes that account, returning it to unanswered.

    A code with no policy module behind it is STORED, not refused: which
    jurisdictions the engines cover is the engines' statement to make, and
    refusing entry here would push the user toward naming a country we happen to
    support rather than the one the account is in.
    """
    from tools.memory import set_account_jurisdictions

    result = set_account_jurisdictions(
        payload.get("accounts"), note=payload.get("note", "")
    )
    if not result.get("ok"):
        return JSONResponse(result, status_code=400)
    return {"status": "success", **result}


@router.get("/api/profile_readiness")
async def api_profile_readiness():
    """Which user-authored inputs are on file, and what each blank switches off.

    Roadmap 2.8. The stores above (playbook, risk_constraints, the wealth goal)
    plus the feedback pool each gate a shipped feature, and every one of them has
    at some point sat empty while the feature that reads it quietly did nothing.
    This is the one place that says so.

    Reports emptiness and names the consequence — it never proposes a value. See
    tools/profile_readiness.py for the contract and the test that enforces it.
    """
    from tools.profile_readiness import build_profile_readiness

    return build_profile_readiness()


@router.post("/api/memory/theses")
async def api_upsert_thesis(req: ThesisRequest):
    from tools.trade_journal import log_trade
    data = req.model_dump(exclude_none=True)
    sym = data.get("symbol", "")
    if sym:
        log_trade(
            symbol=sym,
            action=data.get("action", "BUY"),
            price=float(data.get("price", 0) or 0),
            thesis=data.get("conditions") or data.get("catalyst") or "",
            target_price=float(data.get("target_price") or 0) if data.get("target_price") else None,
            stop_loss=float(data.get("stop_loss") or 0) if data.get("stop_loss") else None
        )
    return {"status": "success"}

@router.delete("/api/memory/theses/{thesis_id}")
async def api_delete_thesis(thesis_id: str):
    from tools.trade_journal import delete_trade
    if not thesis_id or thesis_id == "undefined" or thesis_id == "None":
        return JSONResponse({"error": "Invalid thesis ID"}, status_code=400)
    delete_active_thesis(thesis_id)
    delete_trade(thesis_id)
    return {"status": "success"}

def _lesson_write_response(retired: list[str]) -> dict:
    """200 for a lesson write, naming any rule the cap cost the user.

    The store truncates from the front at ``LESSON_CAP`` (user's call
    2026-07-27), so a write can drop a rule the user wrote — and an unannounced
    drop is the invisible blank 2.8 exists to end, one layer up. ``notice`` is
    what the client toasts; the retired text is unrecoverable from any later
    read, so it has to travel in this response or nowhere.
    """
    from tools.memory import LESSON_CAP, load_memory

    count = len(load_memory().get("lessons_learned", []) or [])
    payload: dict = {"status": "success", "lesson_count": count, "lesson_cap": LESSON_CAP}
    if retired:
        extra = f" (and {len(retired) - 1} more)" if len(retired) > 1 else ""
        payload["retired"] = retired
        payload["notice"] = (
            f"At {LESSON_CAP} rules — retired the oldest to make room: "
            f"“{retired[0]}”{extra}"
        )
    return payload


@router.post("/api/memory/lessons")
async def api_upsert_lesson(req: LessonRequest):
    from tools.memory import LESSON_EVICTED, lessons_pending_eviction

    if req.index is not None:
        update_lesson(req.index, req.text)
        return {"status": "success"}

    at_risk = lessons_pending_eviction()
    outcome = add_lesson(req.text)
    return _lesson_write_response(at_risk if outcome == LESSON_EVICTED else [])

@router.get("/api/memory/lessons/pending")
async def api_list_pending_lessons():
    """Lessons drafted from feedback, awaiting human confirmation (roadmap 1.4 guard).

    These are NOT in effect: nothing here is injected into a prompt until a human
    confirms it below.
    """
    from tools.pending_lessons import list_pending_lessons
    return {"pending": list_pending_lessons()}


@router.post("/api/memory/lessons/pending/{lesson_id}/confirm")
async def api_confirm_pending_lesson(lesson_id: str, req: LessonRequest | None = None):
    """THE human-confirmation gate. Promotes one draft into the real lesson store.

    An optional `text` edits the wording first — a draft is a proposal, and the
    confirming human owns the final rule.
    """
    from tools.pending_lessons import confirm_pending_lesson

    promoted = confirm_pending_lesson(lesson_id, text=(req.text if req else None))
    if promoted is None:
        return JSONResponse({"error": "Draft not found"}, status_code=404)
    # At the cap this promotion retired a rule the user wrote themselves; that
    # has to reach them, not just the drafted rule that replaced it.
    return {**_lesson_write_response(promoted.get("retired") or []), "lesson": promoted}


@router.delete("/api/memory/lessons/pending/{lesson_id}")
async def api_discard_pending_lesson(lesson_id: str):
    from tools.pending_lessons import discard_pending_lesson
    if not discard_pending_lesson(lesson_id):
        return JSONResponse({"error": "Draft not found"}, status_code=404)
    return {"status": "success"}


@router.get("/api/observations")
async def api_observations(limit: int = 25):
    """The observation log — roadmap 1.7's read surface.

    Nothing here is in effect. These are raw behavioural observations recorded
    after each turn; they reach a prompt only if the consolidation pass drafts a
    rule from them AND a human confirms it above.

    The counts are the point. A log that has never been written to and a log
    nobody reads produce identical silence, and this codebase has shipped that
    outcome before (feedback.json: 100 rows, 0 rated, unread).
    """
    from tools.observations import get_observation_stats, get_recent_observations

    return {
        "stats": get_observation_stats(),
        "recent": get_recent_observations(limit=limit),
        "contract": (
            "Recorded after each turn, never injected into a prompt. A rule drafted from "
            "these still needs your confirmation before it takes effect. Ghost/@Private "
            "turns are not recorded at all."
        ),
    }


@router.post("/api/observations/consolidate")
async def api_consolidate_observations():
    """Run the consolidation pass now, drafting into the pending queue only.

    The scheduled job waits for the n gate; this is the explicit human click, so
    it skips the gate — and nothing else. Every citation check still applies, and
    an empty log still drafts nothing.
    """
    import asyncio

    from tools.observation_consolidation import consolidate_observations

    return await asyncio.to_thread(consolidate_observations, True)


@router.delete("/api/memory/lessons/{index}")
async def api_delete_lesson(index: int):
    delete_lesson(index)
    return {"status": "success"}

@router.delete("/api/memory/facts/{index}")
async def api_delete_fact(index: int):
    delete_key_fact(index)
    return {"status": "success"}

@router.post("/api/memory/facts")
async def api_add_fact(req: LessonRequest):
    add_fact(req.text)
    return {"status": "success"}

@router.post("/api/memory/sync_from_facts")
async def api_sync_from_facts(request: Request):
    """Force re-extraction of profile fields from key_facts."""
    memory = load_memory()
    facts = memory.get("key_facts", [])
    profile = memory.get("user_profile", {})

    import re
    # Extract Age
    for fact in facts:
        match = re.search(r"(?i)(?:I\s+am|age\s+is|age)\s+(\d+)", fact)
        if not match: match = re.search(r"(?i)(\d+)\s+years\s+old", fact)
        if match:
            profile["age"] = match.group(1)
            break
    # Extract Income
    for fact in facts:
        match = re.search(r"(?i)(?:income|make)\s+(?:of|close\s+to|around)?\s*[\$]?\s*([\d,]+)", fact)
        if match:
            profile["annual_income"] = f"${match.group(1)}"
            break

    memory["user_profile"] = profile
    save_memory(memory)
    return {"status": "success", "profile": profile}

@router.post("/api/memory/graph/node")
def api_add_node(req: NodeRequest):
    graph_memory.load()
    graph_memory.add_entity(req.name, req.type)
    return {"status": "success"}

@router.delete("/api/memory/graph/node/{name}")
def api_delete_node(name: str):
    graph_memory.load()
    success = graph_memory.delete_entity(name)
    if success:
        return {"status": "success"}
    return JSONResponse({"error": "Node not found"}, status_code=404)

@router.post("/api/memory/graph/edge")
def api_add_edge(req: EdgeRequest):
    graph_memory.load()
    graph_memory.add_relationship(req.source, req.target, req.relation)
    return {"status": "success"}

@router.delete("/api/memory/graph/edge")
def api_delete_edge(req: EdgeRequest):
    graph_memory.load()
    success = graph_memory.delete_relationship(req.source, req.target, req.relation)
    if success:
        return {"status": "success"}
    return JSONResponse({"error": "Relationship not found"}, status_code=404)

@router.post("/api/memory/extract_thesis")
async def extract_thesis_from_chat(request: dict[str, str]):
    text = request.get("text", "")
    if not text:
        return {"error": "No text provided"}

    extracted = extract_thesis_from_text(text)
    return extracted

@router.get("/api/recommendations")
async def api_get_recommendations():
    """Retrieve scored past recommendations and performance statistics."""
    data = get_scored_recommendations_data()
    return JSONResponse(data)

