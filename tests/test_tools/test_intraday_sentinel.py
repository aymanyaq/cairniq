"""Intraday sentinel tick (Advisor Roadmap Theme 3.4).

The engine's value is that it fires on a market-state *change*, never on a level
that has been standing all session — so, like the watch-conditions tests, these
are weighted toward everything that must NOT happen: no alert on a pre-existing
band, no flapping across a boundary, no re-fire until a signal fully clears, no
state wiped by a transient download failure.
"""
import json

import pytest

import tools.intraday_sentinel as isr


@pytest.fixture
def sentinel(monkeypatch, tmp_path):
    """Isolated per-profile state file + captured alerts."""
    path = tmp_path / "intraday_sentinel_state.json"
    monkeypatch.setattr(isr, "get_data_path", lambda filename: str(path))
    fired: list[dict] = []
    monkeypatch.setattr(isr, "_raise", lambda **kw: fired.append(kw))
    return {"path": path, "fired": fired}


def _snap(vix=None, spy_dd=None):
    s: dict = {}
    if vix is not None:
        s["VIX"] = {"price": vix}
    if spy_dd is not None:
        s["SPY"] = {"drawdown_from_high": spy_dd}
    return s


def _tick(vix=None, spy_dd=None, holdings=None, now=None):
    return isr.run_sentinel_tick(
        now=now,
        snapshot_fn=lambda: _snap(vix, spy_dd),
        holdings_fn=lambda: holdings or {},
    )


def _load(sentinel):
    return json.loads(sentinel["path"].read_text())


# ---------------------------------------------------------------------------
# Baseline — a first observation is recorded silently, never alerted
# ---------------------------------------------------------------------------

def test_first_vix_observation_is_a_silent_baseline(sentinel):
    result = _tick(vix=30.0)  # already in the "high" band at startup

    assert sentinel["fired"] == []
    assert result["fired"] == 0
    assert _load(sentinel)["market"]["VIX_BAND"]["band"] == 2  # remembered as high


def test_preexisting_death_cross_is_never_alerted(sentinel):
    _tick(holdings={"NVDA": {"death_cross": True}})   # true on the first look
    _tick(holdings={"NVDA": {"death_cross": True}})   # and still true

    assert sentinel["fired"] == []


def test_preexisting_volume_spike_is_never_alerted(sentinel):
    _tick(holdings={"NVDA": {"vol_spike": 3.0}})
    _tick(holdings={"NVDA": {"vol_spike": 3.1}})

    assert sentinel["fired"] == []


# ---------------------------------------------------------------------------
# VIX bands — upward crossings fire, with hysteresis
# ---------------------------------------------------------------------------

def test_vix_crossing_up_fires_a_warning(sentinel):
    _tick(vix=15.0)              # calm baseline
    _tick(vix=30.0)             # into the high band

    assert len(sentinel["fired"]) == 1
    alert = sentinel["fired"][0]
    assert alert["severity"] == "warning"
    assert alert["source"] == "sentinel"
    assert "high" in alert["title"].lower()
    assert alert["data"]["band"] == "high"


def test_vix_extreme_is_critical_and_a_multiband_jump_fires_once(sentinel):
    _tick(vix=15.0)             # calm
    _tick(vix=45.0)            # calm -> extreme in one tick

    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["severity"] == "critical"
    assert sentinel["fired"][0]["data"]["band"] == "extreme"


def test_vix_wobble_inside_the_deadzone_does_not_flap(sentinel):
    _tick(vix=22.0)            # elevated baseline
    _tick(vix=25.5)           # inside the 25±1 dead-zone — must not cross
    assert sentinel["fired"] == []

    _tick(vix=26.5)           # decisively above 25 — now it crosses
    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["data"]["band"] == "high"


def test_vix_must_fully_exit_a_band_before_it_can_refire(sentinel):
    _tick(vix=15.0)            # calm
    _tick(vix=30.0)           # -> high (fires)
    assert len(sentinel["fired"]) == 1

    _tick(vix=27.0)           # eases but stays high — silent, no re-fire
    assert len(sentinel["fired"]) == 1

    _tick(vix=23.0)           # drops out of high, back to elevated — silent
    assert len(sentinel["fired"]) == 1

    _tick(vix=30.0)           # climbs into high again — a genuinely new event
    assert len(sentinel["fired"]) == 2


def test_vix_relaxing_is_silent(sentinel):
    _tick(vix=40.0)           # extreme baseline
    _tick(vix=12.0)           # collapses to calm — the sentinel does not celebrate

    assert sentinel["fired"] == []


# ---------------------------------------------------------------------------
# SPY drawdown bands
# ---------------------------------------------------------------------------

def test_spy_drawdown_deepening_fires(sentinel):
    _tick(spy_dd=-2.0)        # near the high, baseline
    _tick(spy_dd=-12.0)      # into correction territory

    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["severity"] == "warning"
    assert sentinel["fired"][0]["data"]["band"] == "correction"


def test_spy_deep_correction_is_critical(sentinel):
    _tick(spy_dd=-2.0)
    _tick(spy_dd=-18.0)

    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["severity"] == "critical"
    assert sentinel["fired"][0]["data"]["band"] == "deep"


# ---------------------------------------------------------------------------
# Fresh crosses — the death-cross producer 3.2 left open
# ---------------------------------------------------------------------------

def test_fresh_death_cross_fires_a_warning(sentinel):
    _tick(holdings={"NVDA": {"death_cross": False}})   # baseline
    _tick(holdings={"NVDA": {"death_cross": True}})    # freshly formed

    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["severity"] == "warning"
    assert "death cross" in sentinel["fired"][0]["title"].lower()
    assert sentinel["fired"][0]["data"]["symbol"] == "NVDA"


def test_death_cross_can_clear_and_refire(sentinel):
    _tick(holdings={"NVDA": {"death_cross": False}})
    _tick(holdings={"NVDA": {"death_cross": True}})    # fires
    _tick(holdings={"NVDA": {"death_cross": False}})   # clears — silent
    _tick(holdings={"NVDA": {"death_cross": True}})    # fires again

    assert len(sentinel["fired"]) == 2


def test_fresh_golden_cross_is_info(sentinel):
    _tick(holdings={"NVDA": {"golden_cross": False}})
    _tick(holdings={"NVDA": {"golden_cross": True}})

    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["severity"] == "info"
    assert "golden cross" in sentinel["fired"][0]["title"].lower()


# ---------------------------------------------------------------------------
# Volume spike — armed with a re-arm floor
# ---------------------------------------------------------------------------

def test_volume_spike_arms_once_and_rearms_below_floor(sentinel):
    _tick(holdings={"NVDA": {"vol_spike": 1.2}})   # baseline, quiet
    _tick(holdings={"NVDA": {"vol_spike": 3.0}})   # spike — fires
    assert len(sentinel["fired"]) == 1
    assert sentinel["fired"][0]["severity"] == "info"

    _tick(holdings={"NVDA": {"vol_spike": 2.8}})   # still elevated — no re-fire
    assert len(sentinel["fired"]) == 1

    _tick(holdings={"NVDA": {"vol_spike": 1.5}})   # drops below 2.0 — re-arms silently
    _tick(holdings={"NVDA": {"vol_spike": 2.6}})   # a new spike — fires again
    assert len(sentinel["fired"]) == 2


def test_volume_spike_between_arm_and_disarm_does_not_flap(sentinel):
    _tick(holdings={"NVDA": {"vol_spike": 1.0}})
    _tick(holdings={"NVDA": {"vol_spike": 3.0}})   # fires, armed
    _tick(holdings={"NVDA": {"vol_spike": 2.2}})   # in the 2.0–2.5 hold zone — no re-fire
    _tick(holdings={"NVDA": {"vol_spike": 2.7}})   # still armed — no re-fire

    assert len(sentinel["fired"]) == 1


# ---------------------------------------------------------------------------
# Resilience — bad data never crashes, never fabricates, never wipes state
# ---------------------------------------------------------------------------

def test_empty_snapshot_and_holdings_is_a_clean_noop(sentinel):
    result = isr.run_sentinel_tick(snapshot_fn=lambda: {}, holdings_fn=lambda: {})

    assert "error" not in result
    assert result == {
        "checked_holdings": 0, "fired": 0, "stale_skipped": 0, "alerts": [],
        # No SPY reading at all, so 3.9's ladder was never looked at. That is
        # NOT the same as an empty ladder, and the heartbeat reports them apart.
        "ladder": dict(isr._LADDER_UNEVALUATED),
    }
    assert sentinel["fired"] == []


def test_missing_vix_key_does_not_crash(sentinel):
    result = _tick(spy_dd=-2.0)  # snapshot has SPY but no VIX
    assert "error" not in result
    assert sentinel["fired"] == []


def test_snapshot_failure_does_not_block_holdings(sentinel):
    def _boom():
        raise RuntimeError("market read down")

    isr.run_sentinel_tick(snapshot_fn=_boom, holdings_fn=lambda: {"NVDA": {"death_cross": False}})
    isr.run_sentinel_tick(snapshot_fn=_boom, holdings_fn=lambda: {"NVDA": {"death_cross": True}})

    assert len(sentinel["fired"]) == 1  # the death cross still fired
    assert "death cross" in sentinel["fired"][0]["title"].lower()


def test_holdings_failure_does_not_block_market_bands(sentinel):
    def _boom():
        raise RuntimeError("batch download failed")

    isr.run_sentinel_tick(snapshot_fn=lambda: _snap(vix=15.0), holdings_fn=_boom)
    isr.run_sentinel_tick(snapshot_fn=lambda: _snap(vix=30.0), holdings_fn=_boom)

    assert len(sentinel["fired"]) == 1  # VIX band still fired
    assert sentinel["fired"][0]["data"]["band"] == "high"


def test_a_departed_holding_is_pruned_from_state(sentinel):
    _tick(holdings={"NVDA": {"death_cross": False}, "AMD": {"death_cross": False}})
    _tick(holdings={"NVDA": {"death_cross": False}})  # AMD sold

    holdings_state = _load(sentinel)["holdings"]
    assert "NVDA" in holdings_state
    assert "AMD" not in holdings_state


def test_a_transient_empty_scan_keeps_remembered_state(sentinel):
    _tick(holdings={"NVDA": {"death_cross": False}})   # baseline: not crossed
    _tick(holdings={})                                 # download failed → empty
    assert "NVDA" in _load(sentinel)["holdings"]       # state must survive

    _tick(holdings={"NVDA": {"death_cross": True}})    # fresh cross vs the kept baseline
    assert len(sentinel["fired"]) == 1


def test_state_persists_across_ticks(sentinel):
    _tick(vix=15.0, holdings={"NVDA": {"death_cross": False}})
    state = _load(sentinel)

    assert state["market"]["VIX_BAND"]["band"] == 0
    assert state["holdings"]["NVDA"]["death_cross"] is False
    assert "updated_at" in state


# ---------------------------------------------------------------------------
# Drawdown playbook surfacing (Roadmap 3.7) — driven through the REAL tick
# ---------------------------------------------------------------------------
# The playbook only earns its keep if it reaches the user at the moment they are
# most likely to sell. These drive run_sentinel_tick itself rather than the
# formatter, so a producer that stops calling the attach step fails here.

def test_a_deep_drawdown_crossing_carries_the_playbook(sentinel, monkeypatch):
    import tools.drawdown_playbook as pb

    monkeypatch.setattr(pb, "get_playbook", lambda: {"never_sell": ["Core index ETFs"]})
    monkeypatch.setattr(pb, "goal_status_line", lambda: "")

    _tick(spy_dd=-3.0)          # baseline: near its high
    _tick(spy_dd=-17.0)         # crosses into "deep"

    assert len(sentinel["fired"]) == 1
    fired = sentinel["fired"][0]
    assert "Core index ETFs" in fired["message"]
    assert fired["data"]["playbook_surfaced"] is True


def test_a_shallow_pullback_does_not_recite_the_crash_plan(sentinel, monkeypatch):
    """Reciting it on every 5% dip makes it wallpaper by the time it matters —
    the whole value of this text is that the user has not seen it recently."""
    import tools.drawdown_playbook as pb

    monkeypatch.setattr(pb, "get_playbook", lambda: {"never_sell": ["Core index ETFs"]})

    _tick(spy_dd=-1.0)          # baseline
    _tick(spy_dd=-7.0)          # crosses into "pullback" only

    assert len(sentinel["fired"]) == 1
    assert "Core index ETFs" not in sentinel["fired"][0]["message"]
    assert "playbook_surfaced" not in sentinel["fired"][0]["data"]


def test_the_25pct_band_exists_and_fires_beyond_deep(sentinel, monkeypatch):
    """3.7 asks for −15% AND −25%. Without the second rung the worst tape of the
    decade would produce no new alert at all — the band would already be maxed."""
    import tools.drawdown_playbook as pb

    monkeypatch.setattr(pb, "get_playbook", lambda: None)
    monkeypatch.setattr(pb, "goal_status_line", lambda: "")

    _tick(spy_dd=-17.0)         # baseline inside "deep"
    _tick(spy_dd=-28.0)         # crosses into "severe"

    assert len(sentinel["fired"]) == 1
    assert "severe drawdown" in sentinel["fired"][0]["title"]


def test_appending_the_new_band_did_not_relabel_remembered_state(sentinel):
    """Persisted state stores the band INDEX. Inserting the 25% rung anywhere but
    the end would have re-labelled every remembered band on existing profiles —
    a machine that thought it was at 'correction' would wake at 'pullback' and
    re-fire a crossing it had already reported."""
    _tick(spy_dd=-12.0)

    assert _load(sentinel)["market"]["SPY_DRAWDOWN_BAND"]["band"] == 2  # correction
    assert isr.SPY_DD_BANDS[2]["key"] == "correction"
    assert isr.SPY_DD_BANDS[3]["key"] == "deep"


def test_a_missing_playbook_still_delivers_the_drawdown_alert(sentinel, monkeypatch):
    """No rules on file must not mean no alert — the drawdown is real either way,
    and the message says the playbook is missing rather than inventing one."""
    import tools.drawdown_playbook as pb

    monkeypatch.setattr(pb, "get_playbook", lambda: None)
    monkeypatch.setattr(pb, "goal_status_line", lambda: "")

    _tick(spy_dd=-3.0)
    _tick(spy_dd=-17.0)

    assert len(sentinel["fired"]) == 1
    assert "No drawdown playbook is on file" in sentinel["fired"][0]["message"]


# ---------------------------------------------------------------------------
# The armed deployment ladder (Roadmap 3.9) — wired to THIS tick's peak
# ---------------------------------------------------------------------------

_LADDER = [
    {"drawdown_pct": 5, "action": "Deploy the first 25% of cash into VTI."},
    {"drawdown_pct": 10, "action": "Deploy the next 25%."},
]


@pytest.fixture
def ladder(monkeypatch):
    """A ladder on file, isolated from the profile store."""
    import tools.drawdown_playbook as pb

    monkeypatch.setattr(pb, "get_playbook", lambda: {"deployment_levels": _LADDER})
    monkeypatch.setattr(pb, "goal_status_line", lambda: "")
    return _LADDER


def test_a_crossed_rung_delivers_the_users_action_into_the_inbox(sentinel, ladder):
    _tick(spy_dd=-0.5)   # at a new high: every rung armed
    _tick(spy_dd=-6.0)   # crosses −5%

    rungs = [a for a in sentinel["fired"] if a["data"].get("signal") == "deployment_rung"]
    assert len(rungs) == 1
    assert "Deploy the first 25% of cash into VTI." in rungs[0]["message"]


def test_the_ladder_uses_the_same_peak_as_the_band_alert(sentinel, ladder):
    """One peak, not two. A rung armed off its own tracker could deploy on a
    depth that disagrees with the band alert the user was just shown."""
    _tick(spy_dd=-0.5)
    _tick(spy_dd=-11.0)

    rungs = [a for a in sentinel["fired"] if a["data"].get("signal") == "deployment_rung"]
    assert {r["data"]["observed_drawdown_pct"] for r in rungs} == {11.0}
    assert _load(sentinel)["deployment_ladder"]["depth"] == 11.0


def test_a_stale_spy_bar_does_not_consume_a_rung(sentinel, ladder):
    """The 5.8 gate, and here it guards real money: a rung spent against an
    earlier session's bar would not be there to fire when the tape genuinely
    crossed it."""
    from datetime import datetime

    from tools.freshness import AS_OF_KEY

    now = datetime(2026, 3, 3, 11, 0)
    _tick(spy_dd=-0.5, now=now)

    stale = {"SPY": {"drawdown_from_high": -6.0, AS_OF_KEY: "2026-03-02T16:00:00"}}
    result = isr.run_sentinel_tick(now=now, snapshot_fn=lambda: stale, holdings_fn=lambda: {})

    assert result["stale_skipped"] == 1
    assert result["ladder"]["evaluated"] is False
    assert [a for a in sentinel["fired"] if a["data"].get("signal") == "deployment_rung"] == []

    # And the rung is still armed when a fresh bar arrives.
    fresh = {"SPY": {"drawdown_from_high": -6.0, AS_OF_KEY: now.isoformat()}}
    isr.run_sentinel_tick(now=now, snapshot_fn=lambda: fresh, holdings_fn=lambda: {})

    assert len([a for a in sentinel["fired"] if a["data"].get("signal") == "deployment_rung"]) == 1


def test_ladder_state_survives_a_tick_that_lost_the_market_read(sentinel, ladder):
    """A transient snapshot failure must not re-arm a spent rung — that would
    re-deploy a tranche on the next tick."""
    _tick(spy_dd=-0.5)
    _tick(spy_dd=-6.0)

    isr.run_sentinel_tick(snapshot_fn=lambda: {}, holdings_fn=lambda: {})
    _tick(spy_dd=-6.0)

    assert len([a for a in sentinel["fired"] if a["data"].get("signal") == "deployment_rung"]) == 1


def test_an_empty_playbook_reports_an_evaluated_but_inert_ladder(sentinel, monkeypatch):
    """The inert case, which is every profile today. It must be distinguishable
    from a ladder that was never looked at."""
    import tools.drawdown_playbook as pb

    monkeypatch.setattr(pb, "get_playbook", lambda: None)
    monkeypatch.setattr(pb, "goal_status_line", lambda: "")

    result = _tick(spy_dd=-6.0)

    assert result["ladder"] == {"armed": 0, "fired": 0, "seeded": 0, "levels": 0, "evaluated": True}


def test_a_broken_playbook_store_still_delivers_the_band_alert(sentinel, monkeypatch):
    """The band alert is the load-bearing part. 3.9 is a passenger on this tick
    and must never be able to take it down."""
    import tools.drawdown_playbook as pb

    def _boom():
        raise RuntimeError("playbook store unreadable")

    _tick(spy_dd=-3.0)
    monkeypatch.setattr(pb, "get_playbook", _boom)
    monkeypatch.setattr(pb, "goal_status_line", lambda: "")
    result = _tick(spy_dd=-17.0)

    assert "error" not in result
    assert result["ladder"]["evaluated"] is False
    assert any("deep correction" in a["title"] for a in sentinel["fired"])
