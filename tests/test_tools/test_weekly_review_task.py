"""The weekly review's scheduler task — specifically its window gate.

A time-window gate is the part of a scheduled task most likely to be wrong in
the direction nobody notices: too narrow and it never fires, and "never fired"
looks exactly like "nothing to report" on a weekly cadence. 3.9 shipped armed
and inert on purpose and had to SAY so; this one is supposed to fire, so the
gate is tested from both sides.
"""
import asyncio
from datetime import datetime

import pytest

import tools.scheduler as sched


@pytest.fixture()
def captured(monkeypatch):
    """Record what the task reports to the heartbeat, and stub delivery."""
    outcomes = []
    alerts = []

    monkeypatch.setattr(
        sched, "_note_engine_outcome",
        lambda worked, produced, declined, detail="": outcomes.append(
            {"worked": worked, "produced": produced, "declined": declined, "detail": detail}
        ),
    )
    monkeypatch.setattr("tools.alerts.raise_alert", lambda **kw: alerts.append(kw))
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: True)
    monkeypatch.setattr(
        "tools.user_profile.list_available_profiles",
        lambda: [{"name": "alice"}, {"name": "_unbound"}, {"name": "pytest_x"}],
    )
    monkeypatch.setattr(
        "tools.user_profile.run_under_profile",
        lambda name, fn, *a, **k: fn(*a, **k),
    )
    return {"outcomes": outcomes, "alerts": alerts}


def _run_at(monkeypatch, when: datetime):
    monkeypatch.setattr(sched, "_eastern_now", lambda: when)
    asyncio.run(sched.task_weekly_review())


# --- the gate ---------------------------------------------------------------

@pytest.mark.parametrize("when,label", [
    (datetime(2026, 7, 27, 19, 0), "Monday evening"),
    (datetime(2026, 8, 1, 20, 0), "Saturday evening"),
    (datetime(2026, 8, 2, 11, 0), "Sunday morning"),
    (datetime(2026, 8, 2, 17, 59), "Sunday, one minute early"),
])
def test_it_stays_shut_outside_the_window(monkeypatch, captured, when, label):
    _run_at(monkeypatch, when)

    assert not captured["alerts"], f"delivered on {label}"
    assert captured["outcomes"][0]["worked"] == 0
    # A skip must be reported as a skip, not as a run that produced nothing —
    # otherwise every hourly tick outside the window accrues an idle streak.
    assert "outside" in captured["outcomes"][0]["declined"]


def test_it_fires_in_the_sunday_evening_window(monkeypatch, captured):
    _run_at(monkeypatch, datetime(2026, 8, 2, 18, 30))

    assert len(captured["alerts"]) == 1
    assert captured["outcomes"][0]["worked"] == 1


# --- what it delivers -------------------------------------------------------

def test_the_alert_is_deduped_per_iso_week(monkeypatch, captured):
    """Ticking hourly through the window must refresh one inbox entry, not stack
    five copies of the same weekly notice."""
    _run_at(monkeypatch, datetime(2026, 8, 2, 18, 30))
    _run_at(monkeypatch, datetime(2026, 8, 2, 21, 30))

    keys = {a["dedup_key"] for a in captured["alerts"]}
    assert len(keys) == 1
    assert "2026-W31" in keys.pop()


def test_a_later_week_gets_its_own_alert(monkeypatch, captured):
    _run_at(monkeypatch, datetime(2026, 8, 2, 18, 30))
    _run_at(monkeypatch, datetime(2026, 8, 9, 18, 30))

    assert len({a["dedup_key"] for a in captured["alerts"]}) == 2


def test_it_skips_profiles_that_are_not_real_users(monkeypatch, captured):
    """`_unbound` is the scheduler's own binding and `pytest_*` are test
    residue; delivering a weekly review to either is noise."""
    _run_at(monkeypatch, datetime(2026, 8, 2, 18, 30))

    assert captured["outcomes"][0]["worked"] == 1


def test_production_is_sections_assembled_not_interesting_findings(monkeypatch, captured):
    """A quiet week is the normal outcome. Counting only weeks with something to
    report would put a working reporter on an idle streak within a month."""
    _run_at(monkeypatch, datetime(2026, 8, 2, 18, 30))

    outcome = captured["outcomes"][0]
    assert outcome["produced"] == 6
    assert "sections" in outcome["detail"]


def test_a_disabled_scheduler_delivers_nothing(monkeypatch, captured):
    monkeypatch.setattr(sched, "is_scheduler_enabled", lambda: False)

    _run_at(monkeypatch, datetime(2026, 8, 2, 18, 30))

    assert not captured["alerts"]
    assert captured["outcomes"][0]["produced"] == 0
