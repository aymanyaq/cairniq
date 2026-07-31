"""Watch-conditions wiring (Advisor Roadmap 3.3).

The engine in tools/watch_conditions.py is only useful if three things hold at
the seams: the prompts that commit to levels actually ASK for the side-channel,
the producers HARVEST it before the sanitizer strips it, and the scheduler
CHECKS it. Each of those has failed silently before in this codebase — a prompt
and parser that drift, a signal log nothing writes to, a ledger nothing scores.
"""
import asyncio
from datetime import datetime

import pytest

import tools.scheduler as sched
import tools.watch_conditions as wc

# ---------------------------------------------------------------------------
# The prompts ask for it
# ---------------------------------------------------------------------------

def test_priority_prompt_carries_the_side_channel_spec():
    from agent.quick_actions import build_quick_action_prompt

    prompt = build_quick_action_prompt("priority")

    assert "<watch>" in prompt
    assert "WATCH-CONDITIONS SIDE-CHANNEL" in prompt
    # The prose trigger board is what the side-channel makes enforceable; losing
    # either half makes the other pointless.
    assert "NEXT-CHECK TRIGGER" in prompt


def test_catalyst_scenario_instruction_carries_the_side_channel_spec():
    from agent.catalyst_engine import build_event_scenario_instruction

    instruction = build_event_scenario_instruction("generic")

    assert "<watch>" in instruction
    assert "TRIGGER PLAN" in instruction


def test_spec_and_parser_agree_on_the_example():
    """The spec's own example must survive the parser it documents — if the two
    drift, the block still streams and still gets stripped, and simply stores
    nothing. Nothing errors; the feature just quietly stops working."""
    example = wc.WATCH_SIDE_CHANNEL_PROMPT

    parsed = wc.parse_watch_block(example)

    assert len(parsed) == 1
    assert parsed[0]["symbol"] == "NVDA"
    assert parsed[0]["metric"] in wc.METRICS
    assert parsed[0]["operator"] in wc.OPERATORS
    assert parsed[0]["direction"] in wc.DIRECTIONS


def test_spec_names_only_metrics_the_engine_can_evaluate():
    spec = wc.WATCH_SIDE_CHANNEL_PROMPT
    for metric in wc.METRICS:
        assert f'"{metric}"' in spec


# ---------------------------------------------------------------------------
# The producers harvest it
# ---------------------------------------------------------------------------

ANSWER = (
    "### ⭐ TODAY'S PRIORITY\n**DO NOTHING**\n\n"
    "### \U0001f514 NEXT-CHECK TRIGGER\nNVDA $181.20 → $165.00 entry → 9.0% away\n\n"
    '<watch>{"conditions": [{"symbol": "NVDA", "metric": "price", "operator": "<=", '
    '"threshold": 165.0, "label": "NVDA reaches the entry zone", '
    '"action": "Execute the half-position entry", "direction": "entry", "expires_in_days": 30}]}</watch>'
)


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(wc, "get_data_path", lambda filename: str(tmp_path / "watch_conditions.jsonl"))
    return tmp_path


def test_capture_then_strip_is_the_required_order(store):
    """Stripping first would leave nothing to capture — the exact silent no-op
    this ordering exists to prevent. Assert both halves on one answer."""
    captured = wc.capture_watch_conditions(ANSWER, source="priority")
    visible = wc.strip_watch_blocks(ANSWER)

    assert captured == {"added": 1, "refreshed": 0}
    assert wc.get_conditions()[0]["threshold"] == 165.0
    assert "<watch>" not in visible and "conditions" not in visible
    # The prose the user reads is untouched.
    assert "NEXT-CHECK TRIGGER" in visible and "$165.00 entry" in visible


def test_priority_precompute_harvests_before_caching(monkeypatch, store):
    """The morning brief runs whether or not anyone opens the app; it is the
    engine's highest-volume producer, so its harvest is wired at the cache write."""
    import api.background as bg

    monkeypatch.setattr(bg, "_compose_priority_markdown", lambda state: ANSWER)
    monkeypatch.setattr(bg, "get_active_profile", lambda: "pytest_watch")

    class _FakeGraph:
        def invoke(self, state, config=None):
            return {"messages": []}

    monkeypatch.setattr("agent.graph.build_graph", lambda **kw: _FakeGraph())
    cached: dict = {}
    monkeypatch.setattr("tools.daily_cache.set_cached", lambda key, value: cached.update({key: value}))

    assert bg.run_priority_precompute_in_background() is True

    assert wc.get_conditions()[0]["source"] == "priority"
    assert "<watch>" not in cached["today_priority"]["markdown"]


# ---------------------------------------------------------------------------
# The sanitizer the producers run FIRST must not eat the side-channel
# ---------------------------------------------------------------------------
# Regression guard for 2026-07-23: every producer composes the visible answer
# through extract_visible_text / strip_scaffold_tags BEFORE it harvests. When the
# generic scaffold-tag stripper ALSO matched <watch>, it unwrapped the block --
# deleting the tags, orphaning the JSON -- so capture parsed nothing and the raw
# object leaked into the brief. The harvest tests above missed it entirely by
# feeding raw tagged text straight into capture, skipping the mangling step. The
# feature ran dead for a full trading day: 0 conditions armed, JSON in the brief.

def test_scaffold_stripper_preserves_the_side_channel():
    from agent.utils import extract_visible_text, strip_scaffold_tags

    # Survives the generic scaffold stripper AND the full visible-text pass...
    for cleaned in (strip_scaffold_tags(ANSWER), extract_visible_text(ANSWER)):
        assert "<watch>" in cleaned
        assert len(wc.parse_watch_block(cleaned)) == 1
    # ...while genuine leaked scaffold wrappers are still removed,
    assert "<output_format>" not in strip_scaffold_tags('<output_format strict="true">x</output_format>')
    # and a tag that merely *starts with* "watch" is not the side-channel.
    assert "<watchlist>" not in strip_scaffold_tags("<watchlist>x</watchlist>")


def test_full_visible_pipeline_captures_then_strips(store):
    """The real production order: compose the visible text (extract_visible_text),
    THEN harvest, THEN strip for display. If any step eats the tags early, capture
    gets nothing -- asserted here end to end, the seam the mocked tests skip."""
    from agent.utils import extract_visible_text

    visible = extract_visible_text(ANSWER, strip_node_prefix=True)
    captured = wc.capture_watch_conditions(visible, source="priority")
    display = wc.strip_watch_blocks(visible)

    assert captured == {"added": 1, "refreshed": 0}
    assert wc.get_conditions()[0]["threshold"] == 165.0
    assert "<watch>" not in display and "conditions" not in display


def test_priority_precompute_captures_through_real_compose(monkeypatch, store):
    """Drives the REAL _compose_priority_markdown on a graph result carrying the
    tagged answer, so extract_visible_text runs for real. This is the seam that
    failed in production while test_priority_precompute_harvests_before_caching --
    which mocks compose out -- stayed green."""
    from langchain_core.messages import AIMessage, HumanMessage

    import api.background as bg

    monkeypatch.setattr(bg, "get_active_profile", lambda: "pytest_watch")

    class _FakeGraph:
        def invoke(self, state, config=None):
            return {"messages": [HumanMessage(content="go"), AIMessage(content=ANSWER)]}

    monkeypatch.setattr("agent.graph.build_graph", lambda **kw: _FakeGraph())
    cached: dict = {}
    monkeypatch.setattr("tools.daily_cache.set_cached", lambda key, value: cached.update({key: value}))

    assert bg.run_priority_precompute_in_background() is True

    # The condition was actually armed, and the cached brief is clean of the block.
    assert wc.get_conditions()[0]["threshold"] == 165.0
    assert wc.get_conditions()[0]["source"] == "priority"
    brief = cached["today_priority"]["markdown"]
    assert "<watch" not in brief.lower() and "conditions" not in brief


def test_capture_warns_when_a_block_is_present_but_nothing_parses(store, monkeypatch):
    """A <watch> block that parses to zero usable conditions is a failure, not
    silence: drift, malformed JSON, or an upstream strip. It must leave a trace so
    the engine can never again harvest nothing unnoticed."""
    import logging as _logging

    logs: list = []
    monkeypatch.setattr(wc, "log_to_component",
                        lambda comp, phase, msg, **kw: logs.append((msg, kw.get("level"))))
    # Valid JSON, but the one condition is invalid (unknown metric) -> discarded.
    bad = ('<watch>{"conditions": [{"symbol": "NVDA", "metric": "vibes", '
           '"operator": "<=", "threshold": 1, "label": "x", "action": "y", '
           '"direction": "entry", "expires_in_days": 5}]}</watch>')

    result = wc.capture_watch_conditions(bad, source="priority")

    assert result == {"added": 0, "refreshed": 0}
    assert any(lvl == _logging.WARNING and "0 usable" in m for m, lvl in logs)


def test_capture_is_silent_when_there_is_no_block(store, monkeypatch):
    """The common case -- an answer with no trigger at all -- must NOT warn, or
    the signal drowns. Only a present-but-unusable block is noise worth flagging."""
    logs: list = []
    monkeypatch.setattr(wc, "log_to_component", lambda comp, phase, msg, **kw: logs.append(msg))

    result = wc.capture_watch_conditions("### DO NOTHING\nNo triggers today.", source="priority")

    assert result == {"added": 0, "refreshed": 0}
    assert logs == []


def test_chat_router_strips_the_side_channel_from_the_visible_stream():
    """chat.py must import the sanitizer, not re-implement it — a second regex
    is a second thing to forget when the tag shape changes."""
    from api.routers import chat

    assert chat.strip_watch_blocks is wc.strip_watch_blocks
    assert chat.capture_watch_conditions is wc.capture_watch_conditions


def test_catalyst_escalation_raises_an_alert(monkeypatch):
    """3.2 left this producer open: an escalation spends real money deciding
    something matters, then landed only in a cache the user must think to open."""
    import api.background as bg

    raised: list[dict] = []
    monkeypatch.setattr("tools.alerts.raise_alert", lambda **kw: raised.append(kw))

    bg._alert_catalyst_escalation(
        {"id": "cat1", "headline": "Chip export ban widened", "portfolio_relevance": "portfolio_impact",
         "entities": {"tickers": ["NVDA", "AMD"]}},
        "### CATALYST\nExport controls extended to two more nodes.\n",
    )

    assert len(raised) == 1
    assert raised[0]["severity"] == "warning"           # touches held names
    assert raised[0]["dedup_key"] == "catalyst_escalation:cat1"
    assert "NVDA" in raised[0]["message"]


def test_market_wide_escalation_is_info_not_warning(monkeypatch):
    import api.background as bg

    raised: list[dict] = []
    monkeypatch.setattr("tools.alerts.raise_alert", lambda **kw: raised.append(kw))

    bg._alert_catalyst_escalation(
        {"id": "cat2", "headline": "ECB holds", "portfolio_relevance": "opportunity", "entities": {}},
        "### CATALYST\nNo change to the deposit rate.\n",
    )

    assert raised[0]["severity"] == "info"


def test_escalation_alert_never_breaks_the_scan(monkeypatch):
    import api.background as bg

    def _boom(**kw):
        raise RuntimeError("inbox unavailable")

    monkeypatch.setattr("tools.alerts.raise_alert", _boom)

    bg._alert_catalyst_escalation({"id": "c", "headline": "x"}, "body")  # must not raise


# ---------------------------------------------------------------------------
# The safety layer must not audit the machinery
# ---------------------------------------------------------------------------

def test_judge_never_sees_the_side_channel(monkeypatch):
    """Every deterministic audit scans free text for numbers. A stored trigger
    level left in the draft would read as an unsourced price claim or a phantom
    trade — the false-flag class this layer has repeatedly shipped (7816c1a,
    cd833c7). Strip happens inside the seam, so the node and the 2.4 harness are
    both covered."""
    from agent.nodes import risk_manager as rm

    seen: list[str] = []
    for audit in ("run_deterministic_grounding_audit", "run_deterministic_total_audit",
                  "run_deterministic_price_audit", "run_deterministic_allocation_audit"):
        monkeypatch.setattr(rm, audit, lambda text, _s=seen: (_s.append(text), [])[1])
    monkeypatch.setattr("tools.ips_precheck.run_ips_precheck",
                        lambda text, tickers: (seen.append(text), {"trades": [], "rows": [], "violations": [], "block": ""})[1])
    # Stop at the LLM boundary — the assertion is about what the audits received.
    monkeypatch.setattr(rm, "get_llm", lambda: object())
    monkeypatch.setattr(rm, "create_agent", lambda llm, tools, system: object())
    monkeypatch.setattr(rm, "safe_invoke", lambda *a, **k: type("R", (), {"content": "✅ Risk Check Passed"})())
    monkeypatch.setattr(rm, "has_stream_callback", lambda: False)

    rm.judge_advice(ANSWER, judge_messages=[])

    assert seen, "the audits must still run on the prose"
    for audited in seen:
        assert "<watch>" not in audited
        assert '"threshold"' not in audited
        # ...but the prose the judge is actually there to audit survives intact.
        assert "NEXT-CHECK TRIGGER" in audited


def test_judge_context_strips_the_block_from_the_message_window(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from agent.nodes import risk_manager as rm

    window = rm._build_judge_context([
        HumanMessage(content="What should I do today?"),
        AIMessage(content=ANSWER, name="PortfolioManager"),
    ])

    drafts = [m.content for m in window if isinstance(m, AIMessage)]
    assert drafts and all("<watch>" not in d for d in drafts)


# ---------------------------------------------------------------------------
# The scheduler checks it
# ---------------------------------------------------------------------------

def test_task_is_registered_with_a_thirty_minute_cooldown():
    entry = next((t for t in sched.SCHEDULED_TASKS if t[0] == "watch_conditions"), None)

    assert entry is not None, "watch_conditions task must be registered or nothing ever evaluates"
    _, factory, cooldown, timeout = entry
    assert factory is sched.task_watch_conditions
    assert cooldown == 1800
    assert timeout <= 300


@pytest.mark.parametrize("when,expected", [
    (datetime(2026, 7, 22, 9, 0), False),    # pre-market
    (datetime(2026, 7, 22, 9, 30), True),    # the open
    (datetime(2026, 7, 22, 13, 15), True),
    (datetime(2026, 7, 22, 16, 0), True),    # the close
    (datetime(2026, 7, 22, 16, 1), False),   # after hours
])
def test_market_hours_window(when, expected):
    assert sched._in_market_hours(when) is expected


def test_task_does_not_evaluate_outside_market_hours(monkeypatch):
    """An after-hours tick would re-read the same close print all evening and
    could fire on a thin-liquidity level the advisor never meant."""
    calls: list[int] = []
    monkeypatch.setattr(sched, "_eastern_now", lambda: datetime(2026, 7, 22, 20, 0))
    monkeypatch.setattr(wc, "evaluate_conditions", lambda *a, **k: calls.append(1))

    asyncio.run(sched.task_watch_conditions())

    assert calls == []


def test_task_does_not_evaluate_on_a_weekend(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(sched, "_eastern_now", lambda: datetime(2026, 7, 26, 13, 0))  # a Sunday, mid-session clock
    monkeypatch.setattr(wc, "evaluate_conditions", lambda *a, **k: calls.append(1))

    asyncio.run(sched.task_watch_conditions())

    assert calls == []


def test_task_evaluates_each_profile_in_market_hours(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(sched, "_eastern_now", lambda: datetime(2026, 7, 22, 13, 0))
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: True)
    monkeypatch.setattr(
        "tools.user_profile.list_available_profiles",
        lambda: [{"name": "alpha"}, {"name": "beta"}, {"name": "pytest_skipme"}, {"name": "_unbound"}],
    )
    monkeypatch.setattr(
        "tools.user_profile.run_under_profile",
        lambda name, fn, *a, **k: (seen.append(name), fn(*a, **k))[1],
    )
    monkeypatch.setattr(
        wc, "evaluate_conditions",
        lambda *a, **k: {"checked": 1, "fired": 0, "voided": 0, "expired": 0, "unavailable": 0},
    )

    asyncio.run(sched.task_watch_conditions())

    # 'default' is the profile-listing read; the pytest_ and _unbound profiles
    # are excluded exactly as every other per-profile task excludes them.
    assert seen == ["default", "alpha", "beta"]
