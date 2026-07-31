"""7.1 — the availability read surface and its scheduler reporting.

Two contracts beyond "the endpoint answers": the figure must never travel
without the caveat that makes it a bound, and an unreadable probe log must reach
the heartbeat as a FINDING rather than as a skip. The second is the one that
matters operationally — an instrument that goes blind and reports a clean skip
is how a dark engine stays dark, which this codebase has now done six times.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def _write_probes(path, start, count, minutes=2):
    path.write_text(
        "".join(
            f"{(start + timedelta(minutes=minutes * i)).isoformat(timespec='seconds')} ok\n"
            for i in range(count)
        ),
        encoding="utf-8",
    )
    return str(path)


def test_availability_endpoint_returns_a_measured_report(tmp_path):
    log = _write_probes(tmp_path / "w.log", datetime(2026, 7, 6, 9, 0), 60)
    with patch("tools.availability.probe_log_path", return_value=log):
        res = client.get("/api/availability")

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "measured"
    assert body["window_coverage_pct"] == 100.0
    assert body["probes"] == 60


def test_endpoint_reports_a_missing_probe_log_as_no_data(tmp_path):
    """Not 100%. An absent instrument and a perfect record are opposite claims."""
    with patch("tools.availability.probe_log_path", return_value=str(tmp_path / "absent.log")):
        body = client.get("/api/availability").json()

    assert body["status"] == "no_data"
    assert body.get("window_coverage_pct") is None


def test_the_figure_never_ships_without_its_blind_spots(tmp_path):
    log = _write_probes(tmp_path / "w.log", datetime(2026, 7, 6, 9, 0), 30)
    with patch("tools.availability.probe_log_path", return_value=log):
        body = client.get("/api/availability").json()

    assert "upper bound" in body["summary"]
    states = {m["number"]: m["state"] for m in body["open_measurements"]}
    assert "NOT MEASURED" in states["surfaces returning 5xx"]


def test_response_is_json_safe(tmp_path):
    """A bare NaN is not valid JSON and once took an endpoint down for a day."""
    log = _write_probes(tmp_path / "w.log", datetime(2026, 7, 6, 9, 0), 10)
    with patch("tools.availability.probe_log_path", return_value=log):
        res = client.get("/api/availability")
    assert "NaN" not in res.text
    assert "Infinity" not in res.text


# ---------------------------------------------------------------------------
# 2.6 reporting
# ---------------------------------------------------------------------------
def test_scheduler_task_reports_probes_as_production(tmp_path):
    from tools import scheduler

    log = _write_probes(tmp_path / "w.log", datetime(2026, 7, 6, 9, 0), 45)
    with patch("tools.availability.probe_log_path", return_value=log), \
         patch.object(scheduler, "_note_engine_outcome") as noted:
        asyncio.run(scheduler.task_availability_report())

    noted.assert_called_once()
    worked, produced, declined, detail = noted.call_args[0]
    assert worked == 1
    # Probes read is the count that proves the chain: log found, parsed, coverage
    # computed. Not "gaps found" — zero gaps is the healthy state.
    assert produced == 45
    assert declined == ""
    assert "window coverage" in detail
    assert "upper bound" in detail


def test_a_blind_instrument_is_a_finding_not_a_skip(tmp_path):
    """Zero production against an unreadable log, so it accrues an idle streak.
    A skip would tell the ops view the task declined to work by instruction."""
    from tools import scheduler

    with patch("tools.availability.probe_log_path", return_value=str(tmp_path / "gone.log")), \
         patch.object(scheduler, "_note_engine_outcome") as noted:
        asyncio.run(scheduler.task_availability_report())

    worked, produced, declined, detail = noted.call_args[0]
    assert worked == 1
    assert produced == 0
    assert declined == ""  # NOT a skip
    assert "UNKNOWN" in detail


# ---------------------------------------------------------------------------
# /api/health — the emitter the 5xx axis needs (7.1 Step 3)
# ---------------------------------------------------------------------------
def test_health_stays_public_and_cheap():
    """The watchdog probes this every ~2 minutes and the iOS client hits it before
    login, so it must answer without a token and must never touch the agent."""
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "code_stale" in body


def test_health_reports_uptime_and_staleness_fields():
    body = client.get("/api/health").json()
    assert body["uptime_s"] >= 0
    # In the test process the source predates start, so this is a real False.
    assert body["code_stale"] in (True, False, None)
    assert "code_stale_detail" in body


def test_a_stale_process_still_answers_200():
    """`code_stale` is a FIELD, not a status. A non-2xx here would make the
    watchdog treat a working server as down and put it on the revive path."""
    import server

    with patch.object(server, "_PROCESS_STARTED", 0.0):  # older than any file
        res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["code_stale"] is True


def test_staleness_is_unknown_not_false_when_nothing_can_be_stat_ed():
    """An instrument that cannot look must say so. False here would be a clean
    reading from a probe that never ran."""
    import server

    with patch.object(server, "_DEPLOY_WATCH_PATHS", ("does-not-exist-anywhere",)):
        body = client.get("/api/health").json()
    assert body["code_stale"] is None
    assert "no source path" in body["code_stale_detail"]


def test_a_fresh_process_is_not_stale():
    import time as _time

    import server

    with patch.object(server, "_PROCESS_STARTED", _time.time() + 3600):
        assert client.get("/api/health").json()["code_stale"] is False


# ---------------------------------------------------------------------------
# 7.1 number 4 — delivery latency is its own per-profile surface
# ---------------------------------------------------------------------------
def test_delivery_endpoint_answers_and_states_its_status():
    body = client.get("/api/alerts/delivery").json()
    assert body["status"] in ("measured", "no_data")
    # The counts that make an empty result legible rather than reassuring.
    for key in ("alerts_total", "alerts_read", "timed_reads", "bulk_read",
                "unmeasurable_reads"):
        assert key in body


def test_latency_is_NOT_folded_into_the_global_availability_report(tmp_path):
    """An inbox belongs to one person and `/api/availability` is deliberately
    global, so a single latency figure across profiles would mean nothing. The row
    must point at the per-profile seam instead of carrying a number."""
    log = _write_probes(tmp_path / "w.log", datetime(2026, 7, 6, 9, 0), 20)
    with patch("tools.availability.probe_log_path", return_value=log):
        body = client.get("/api/availability").json()

    states = {m["number"]: m for m in body["open_measurements"]}
    row = states["delivery latency (produced-at vs read-at)"]
    assert "MEASURED PER-PROFILE" in row["state"]
    assert "/api/alerts/delivery" in row["how"]
    # And no latency number leaked onto the global report.
    assert "median_seconds" not in body


def test_task_is_registered_with_the_scheduler():
    from tools.scheduler import SCHEDULED_TASKS

    entry = next((t for t in SCHEDULED_TASKS if t[0] == "availability_report"), None)
    assert entry is not None, "availability_report must be in SCHEDULED_TASKS to ever run"
    assert entry[2] == 86400
