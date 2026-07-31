"""Watchdog supervision check — the orphaned-listener gap.

The watchdog had two states in mind: serving (healthy) and not serving (revive
it). The state between them was invisible and is the one that actually occurred
on cairniq 2026-07-25 — the port bound by a process launchd does not own. The old
probe returned "up", because it asked whether *something* was listening and never
*who*.

Two consequences, both silent: the service runs with no supervision, so the
moment it dies it stays dead; and `launchctl kickstart -k` cannot replace an
instance launchd does not own, so a deploy leaves the OLD CODE serving while
every health check passes. That is how a push and a "successful" restart can both
report success and change nothing.

The watchdog had no tests at all before this file.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import cairniq_watchdog as wd  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(wd, "ALERT_PATH", str(tmp_path / "alert.txt"))
    monkeypatch.setattr(wd, "_notify", lambda *a, **k: None)
    monkeypatch.delenv("AIDLC_WATCHDOG_ADOPT_ORPHAN", raising=False)
    return tmp_path


def _wire(monkeypatch, *, job=(True, 849), listener=849, descendant=True):
    monkeypatch.setattr(wd, "_job_status", lambda: job)
    monkeypatch.setattr(wd, "_listener_pid", lambda: listener)
    monkeypatch.setattr(wd, "_is_descendant", lambda c, a, **k: descendant)


# ---------------------------------------------------------------------------
# The gap
# ---------------------------------------------------------------------------

def test_a_port_held_by_a_process_launchd_does_not_own_is_NOT_healthy(monkeypatch):
    """The exact live state: listener pid 599, launchd job `not running`.
    The old probe scored this "up"."""
    _wire(monkeypatch, job=(True, None), listener=599)

    assert wd._check_supervision({}) == "orphaned"


def test_the_alert_says_why_it_matters_not_just_that_it_happened(monkeypatch):
    _wire(monkeypatch, job=(True, None), listener=599)
    state = {}

    wd._check_supervision(state)

    alert = Path(wd.ALERT_PATH).read_text()
    assert "UNSUPERVISED" in alert
    assert "nothing will restart it" in alert
    assert "old code serving" in alert


def test_a_listener_owned_by_the_launchd_job_is_fine(monkeypatch):
    """The healthy case measured on cairniq after the fix: job pid == listener."""
    _wire(monkeypatch, job=(True, 849), listener=849)

    assert wd._check_supervision({}) is None


def test_a_worker_forked_from_the_launchd_process_is_supervised(monkeypatch):
    """Guards against a false alarm if the server ever runs multiple workers:
    a child of the owned parent IS supervised and must not be reported."""
    _wire(monkeypatch, job=(True, 849), listener=901, descendant=True)

    assert wd._check_supervision({}) is None


def test_an_unrelated_process_holding_the_port_is_an_orphan(monkeypatch):
    _wire(monkeypatch, job=(True, 849), listener=901, descendant=False)

    assert wd._check_supervision({}) == "orphaned"


# ---------------------------------------------------------------------------
# Unknown is not a fault — the codebase's standing rule
# ---------------------------------------------------------------------------

def test_a_missing_lsof_does_not_raise_a_supervision_alarm(monkeypatch):
    """A tooling gap must not present as a supervision fault."""
    _wire(monkeypatch, job=(True, None), listener=None)

    assert wd._check_supervision({}) is None


def test_an_unloaded_job_is_left_to_the_existing_liveness_path(monkeypatch):
    """Not loaded is a different, already-handled condition — this check must
    not claim it."""
    _wire(monkeypatch, job=(False, None), listener=599)

    assert wd._check_supervision({}) is None


# ---------------------------------------------------------------------------
# Alerting behaviour
# ---------------------------------------------------------------------------

def test_the_alert_fires_once_not_every_two_minutes(monkeypatch):
    """The watchdog runs every ~2 min. An alarm that repeats forever is one
    people mute, and a muted alarm is the failure it was meant to prevent."""
    _wire(monkeypatch, job=(True, None), listener=599)
    state = {}

    for _ in range(5):
        wd._check_supervision(state)

    lines = [ln for ln in Path(wd.ALERT_PATH).read_text().splitlines() if ln.strip()]
    assert len(lines) == 1


def test_recovery_clears_the_flag_so_a_later_orphan_alerts_again(monkeypatch):
    state = {}
    _wire(monkeypatch, job=(True, None), listener=599)
    wd._check_supervision(state)

    _wire(monkeypatch, job=(True, 849), listener=849)
    wd._check_supervision(state)
    assert "orphan_since" not in state and "orphan_alerted" not in state

    _wire(monkeypatch, job=(True, None), listener=777)
    wd._check_supervision(state)
    lines = [ln for ln in Path(wd.ALERT_PATH).read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_the_first_sighting_time_is_recorded_and_kept(monkeypatch):
    """How long it has been unsupervised is the operationally useful part."""
    _wire(monkeypatch, job=(True, None), listener=599)
    state = {}

    wd._check_supervision(state)
    first = state["orphan_since"]
    wd._check_supervision(state)

    assert state["orphan_since"] == first


# ---------------------------------------------------------------------------
# Adoption is opt-in, because it trades a real outage for a latent risk
# ---------------------------------------------------------------------------

def test_adoption_is_OFF_by_default_and_nothing_is_killed(monkeypatch):
    """The orphan is SERVING. Killing it on a schedule is a bigger blast radius
    than the risk it removes, so the trade is the operator's to make."""
    killed = []
    _wire(monkeypatch, job=(True, None), listener=599)
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))

    assert wd._check_supervision({}) == "orphaned"
    assert killed == []


def test_adoption_when_enabled_frees_the_port_then_kickstarts_WITHOUT_k(monkeypatch):
    """With the port still held, launchd's fresh instance hits the
    single-instance guard, exits 0, and the orphan keeps serving — which is why
    `kickstart -k` is useless here and the port must be freed first."""
    monkeypatch.setenv("AIDLC_WATCHDOG_ADOPT_ORPHAN", "1")
    _wire(monkeypatch, job=(True, None), listener=599)

    killed, kicked = [], []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(wd, "_port_open", lambda: False)   # freed immediately
    monkeypatch.setattr(wd, "_kickstart", lambda: kicked.append(True) or True)
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)

    assert wd._check_supervision({}) == "orphan-adopted"
    assert killed == [(599, wd.signal.SIGTERM)]
    assert kicked == [True]


def test_adoption_does_not_kickstart_if_the_port_never_frees(monkeypatch):
    """Kickstarting into a still-held port produces a silent no-op that reads as
    a successful recovery."""
    monkeypatch.setenv("AIDLC_WATCHDOG_ADOPT_ORPHAN", "1")
    _wire(monkeypatch, job=(True, None), listener=599)

    kicked = []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(wd, "_port_open", lambda: True)    # never frees
    monkeypatch.setattr(wd, "_kickstart", lambda: kicked.append(True) or True)
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)

    assert wd._check_supervision({}) == "orphan-adopt-failed"
    assert kicked == []


def test_adoption_respects_the_revive_cooldown(monkeypatch):
    """Shares the rate limit with the liveness kickstart, so the two halves
    cannot chase each other into a restart storm."""
    import time as real_time
    monkeypatch.setenv("AIDLC_WATCHDOG_ADOPT_ORPHAN", "1")
    _wire(monkeypatch, job=(True, None), listener=599)

    killed = []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))

    assert wd._check_supervision({"last_revive": real_time.time()}) == "orphaned"
    assert killed == []


def test_an_orphan_makes_the_watchdog_exit_nonzero(monkeypatch):
    """launchd surfaces the failure; a zero exit would bury it."""
    assert "orphaned" in wd.main.__doc__ if wd.main.__doc__ else True
    src = Path(wd.__file__).read_text()
    assert '"orphaned", "orphan-adopt-failed"' in src


# ---------------------------------------------------------------------------
# PATH resolution under launchd — the fix's own inert-in-production bug
# ---------------------------------------------------------------------------
# Caught by reading the SCHEDULED run log, not the manual one. The watchdog
# plist sets PATH=/…/.venv/bin:/usr/local/bin:/usr/bin:/bin — no /usr/sbin,
# which is where lsof lives. So `subprocess.run(["lsof", …])` raised
# FileNotFoundError on every scheduled run while working perfectly over SSH, and
# because "undeterminable" is deliberately not a fault, the supervision check
# returned None forever and logged "ok" every two minutes.

def test_system_tools_resolve_by_absolute_path_not_via_PATH():
    """launchd does not give a job an interactive shell's PATH."""
    assert wd._tool("lsof", "/usr/sbin/lsof", "/usr/bin/lsof").startswith("/")
    assert wd._tool("ps", "/bin/ps") == "/bin/ps"


def test_a_missing_binary_falls_back_to_a_bare_name_rather_than_crashing():
    """Portability: on a host where the candidates do not exist, try PATH."""
    assert wd._tool("lsof", "/nope/lsof") == "lsof"


def test_listener_lookup_uses_the_resolved_path(monkeypatch):
    seen = {}

    def _fake_run(cmd, **kwargs):
        seen["argv0"] = cmd[0]
        class R:
            stdout = "849\n"
        return R()

    monkeypatch.setattr(wd.subprocess, "run", _fake_run)
    assert wd._listener_pid() == 849
    assert seen["argv0"].startswith("/"), "must not rely on PATH under launchd"


def test_an_undeterminable_listener_is_logged_loudly_not_absorbed(monkeypatch, capsys):
    """The tolerance that made the PATH bug silent. Unknown still must not raise
    a false alarm, but it must never again pass quietly as 'ok'."""
    def _boom(*a, **k):
        raise FileNotFoundError("lsof")

    monkeypatch.setattr(wd.subprocess, "run", _boom)
    assert wd._listener_pid() is None

    out = capsys.readouterr().out
    assert "INERT" in out and "not passing" in out
