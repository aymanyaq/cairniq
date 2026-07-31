"""7.1 Step 3 — the serving probe, and the outage it exists to have caught.

`/context` returned HTTP 500 eight times over roughly five hours on cairniq while
the watchdog logged `ok` every two minutes, because checks 3 and 4 both ask who
holds the port and neither ever sent a request. The contract tests here are:

  * a bound, launchd-owned port is NOT reported healthy without a request,
  * a 5xx is reported and never acted on — this watchdog does not kill running
    processes, and a half-deploy looks exactly like a 5xx,
  * the log strings are a vocabulary `tools.availability` parses, so they are
    pinned here rather than left to prose.
"""

import json
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


def _serve(monkeypatch, status, payload=None):
    """Wire the HTTP probe to a canned response without touching a socket."""
    monkeypatch.setattr(wd, "_http_status", lambda *a, **k: (status, payload or {}))


# ---------------------------------------------------------------------------
# The gap this closes
# ---------------------------------------------------------------------------
def test_a_bound_supervised_port_serving_500s_is_NOT_up(monkeypatch):
    """THE REGRESSION. Every check before this one passed throughout the real
    five-hour outage: the port was bound, launchd owned it, the box was up."""
    _serve(monkeypatch, 500)
    assert wd._check_serving({}) == "serving-errors"


def test_liveness_no_longer_calls_a_bound_port_healthy_on_its_own(monkeypatch):
    """The wiring, not just the helper. A port that is open and supervised must
    now reach the serving probe rather than returning 'up' directly."""
    monkeypatch.setattr(wd, "_port_open", lambda: True)
    monkeypatch.setattr(wd, "_check_supervision", lambda state: None)
    _serve(monkeypatch, 503)
    assert wd._check_liveness({}) == "serving-errors"


def test_a_healthy_surface_is_up(monkeypatch):
    monkeypatch.setattr(wd, "_port_open", lambda: True)
    monkeypatch.setattr(wd, "_check_supervision", lambda state: None)
    _serve(monkeypatch, 200, {"status": "ok", "code_stale": False})
    assert wd._check_liveness({}) == "up"


# ---------------------------------------------------------------------------
# A 5xx is reported, never acted on
# ---------------------------------------------------------------------------
def test_a_5xx_never_restarts_anything(monkeypatch):
    """The watchdog's standing rule. The process is RUNNING — killing it is the
    operator's call (the post-receive hook decision), not this script's."""
    killed, kicked = [], []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(wd, "_kickstart", lambda: kicked.append(1) or True)
    monkeypatch.setattr(wd, "_disable_server", lambda: kicked.append("disable") or True)
    _serve(monkeypatch, 500)

    wd._check_serving({})

    assert killed == []
    assert kicked == []


def test_a_5xx_alerts_once_not_every_two_minutes(monkeypatch):
    _serve(monkeypatch, 500)
    state = {}
    alerts = []
    monkeypatch.setattr(wd, "_raise_alert", lambda r: alerts.append(r))

    for _ in range(4):
        assert wd._check_serving(state) == "serving-errors"

    assert len(alerts) == 1


def test_recovery_clears_the_flag_so_a_later_outage_alerts_again(monkeypatch):
    state = {}
    alerts = []
    monkeypatch.setattr(wd, "_raise_alert", lambda r: alerts.append(r))

    _serve(monkeypatch, 500)
    wd._check_serving(state)
    _serve(monkeypatch, 200, {"code_stale": False})
    assert wd._check_serving(state) == "up"
    _serve(monkeypatch, 502)
    wd._check_serving(state)

    assert len(alerts) == 2


def test_the_alert_says_a_half_deploy_looks_like_this(monkeypatch):
    """The operator reading this line at 3am needs the likely cause in it."""
    _serve(monkeypatch, 500)
    alerts = []
    monkeypatch.setattr(wd, "_raise_alert", lambda r: alerts.append(r))
    wd._check_serving({})
    assert "half-deploy" in alerts[0]


# ---------------------------------------------------------------------------
# What counts as serving
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [200, 204, 302, 401, 403, 404])
def test_any_non_5xx_answer_means_the_surface_is_serving(monkeypatch, status):
    """A 401 is the app working correctly — auth is on. Only 5xx is a fault."""
    _serve(monkeypatch, status, {"code_stale": False})
    assert wd._check_serving({}) == "up"


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_every_5xx_is_a_fault(monkeypatch, status):
    _serve(monkeypatch, status)
    assert wd._check_serving({}) == "serving-errors"


def test_a_bound_port_that_completes_no_http_exchange_is_not_serving(monkeypatch):
    _serve(monkeypatch, None)
    state = {}
    assert wd._check_serving(state) == "not-serving"
    assert state["not_serving_checks"] == 1


def test_a_recovered_exchange_clears_the_not_serving_count(monkeypatch):
    state = {}
    _serve(monkeypatch, None)
    wd._check_serving(state)
    _serve(monkeypatch, 200, {"code_stale": False})
    wd._check_serving(state)
    assert "not_serving_checks" not in state


# ---------------------------------------------------------------------------
# Stale code — the half-deploy, caught before it renders as a 500
# ---------------------------------------------------------------------------
def test_stale_code_is_reported_and_not_acted_on(monkeypatch):
    kicked = []
    monkeypatch.setattr(wd, "_kickstart", lambda: kicked.append(1) or True)
    _serve(monkeypatch, 200, {"code_stale": True, "code_stale_detail": "newest server.py"})

    assert wd._check_serving({}) == "up-stale-code"
    assert kicked == []


def test_stale_code_never_logs_ok(monkeypatch, capsys):
    """`ok` is the line the five-hour outage hid behind. A process serving code
    older than the disk must not emit it."""
    monkeypatch.setattr(wd, "_port_open", lambda: True)
    monkeypatch.setattr(wd, "_check_supervision", lambda state: None)
    monkeypatch.setattr(wd, "_count_starts", lambda: 0)
    monkeypatch.setattr(wd, "_load_state", lambda: {})
    monkeypatch.setattr(wd, "_save_state", lambda d: None)
    _serve(monkeypatch, 200, {"code_stale": True, "code_stale_detail": "x"})

    rc = wd.main()
    out = capsys.readouterr().out

    assert wd.STALE_CODE_MARKER in out
    # The whole point: not a failure exit (the surface answers), but not "ok".
    assert rc == 0
    assert not any(line.strip().endswith(" ok") for line in out.splitlines())


def test_an_unknown_staleness_is_not_treated_as_stale(monkeypatch):
    """`code_stale: None` means the emitter could not look. Unknown is not a
    fault — the same rule the lsof path already follows."""
    _serve(monkeypatch, 200, {"code_stale": None})
    assert wd._check_serving({}) == "up"


# ---------------------------------------------------------------------------
# The log vocabulary is a contract with tools.availability
# ---------------------------------------------------------------------------
def test_the_5xx_line_carries_the_marker_and_the_raw_status_code(monkeypatch, capsys):
    """`tools.availability` greps for this marker and parses the code out of it,
    so the string and the three digits beside it are both load-bearing."""
    _serve(monkeypatch, 503)
    monkeypatch.setattr(wd, "_raise_alert", lambda r: print(f"x {r}"))
    wd._check_serving({})
    out = capsys.readouterr().out
    assert f"{wd.SERVING_ERROR_MARKER} 503" in out


def test_the_healthy_line_names_the_status_code_it_saw(monkeypatch, capsys):
    """REGRESSION. A bare `ok` is what the old tcp-only probe wrote, so it is not
    evidence a request was made — which left the serving measurement able to prove
    itself only by failing."""
    monkeypatch.setattr(wd, "_port_open", lambda: True)
    monkeypatch.setattr(wd, "_check_supervision", lambda state: None)
    monkeypatch.setattr(wd, "_count_starts", lambda: 0)
    monkeypatch.setattr(wd, "_load_state", lambda: {})
    monkeypatch.setattr(wd, "_save_state", lambda d: None)
    _serve(monkeypatch, 200, {"code_stale": False})

    assert wd.main() == 0
    out = capsys.readouterr().out
    assert f"{wd.HEALTHY_MARKER} 200" in out
    # And never the bare form, which is the line that proved nothing.
    assert not any(line.strip().endswith(" ok") for line in out.splitlines())


def test_availability_reads_the_healthy_line_this_script_writes(monkeypatch, capsys):
    """Both halves of the healthy path, wired. The writer's line must satisfy the
    reader's regex, or the measurement silently stays dark on a working host."""
    from tools import availability

    monkeypatch.setattr(wd, "_port_open", lambda: True)
    monkeypatch.setattr(wd, "_check_supervision", lambda state: None)
    monkeypatch.setattr(wd, "_count_starts", lambda: 0)
    monkeypatch.setattr(wd, "_load_state", lambda: {})
    monkeypatch.setattr(wd, "_save_state", lambda d: None)
    _serve(monkeypatch, 200, {"code_stale": False})
    wd.main()

    # The exact text the watchdog emitted, minus its timestamp prefix.
    emitted = capsys.readouterr().out.strip().splitlines()[-1].split(" ", 1)[1]
    assert availability._HEALTHY_RE.match(emitted), emitted


def test_availability_reads_the_markers_this_script_writes(monkeypatch, tmp_path):
    """The two halves wired together — a probe log written in this vocabulary must
    measure as a 5xx on the report. A contract asserted on only one side is how a
    measurement goes quietly dark."""
    from tools import availability

    _serve(monkeypatch, 500)
    lines = []
    monkeypatch.setattr(wd, "log", lambda m: lines.append(m))
    monkeypatch.setattr(wd, "_raise_alert", lambda r: lines.append(r))
    wd._check_serving({})

    log = tmp_path / "w.log"
    log.write_text(
        "2026-07-30T09:00:00 ok\n"
        + f"2026-07-30T09:02:00 {lines[0]}\n"
        + "2026-07-30T09:04:00 ok\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(log))

    assert report["serving_probe_active"] is True
    assert report["serving_error_probes"] == 1
    assert report["serving_error_codes"] == {"500": 1}
    assert report["serving_errors_in_window"] == 1


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------
def test_an_http_error_status_is_returned_not_raised(monkeypatch):
    """urllib raises HTTPError on 5xx. Letting that propagate as an exception
    would send a 500 down the same path as an unreachable host — which is the
    difference between 'up and broken' and 'down', and the whole point here."""
    import urllib.error

    def _boom(url, timeout=None):
        raise urllib.error.HTTPError(
            url, 500, "Internal Server Error", {}, None  # type: ignore[arg-type]
        )

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    status, payload = wd._http_status()
    assert status == 500
    assert payload == {}


def test_an_unreachable_server_is_none_not_a_status(monkeypatch):
    def _boom(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert wd._http_status() == (None, {})


def test_the_probe_parses_the_health_payload(monkeypatch):
    class _Resp:
        status = 200

        def read(self, n=None):
            return json.dumps({"status": "ok", "code_stale": False}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: _Resp())
    status, payload = wd._http_status()
    assert status == 200
    assert payload["code_stale"] is False
