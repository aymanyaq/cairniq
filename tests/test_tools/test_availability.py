"""7.1 Step 1 — the availability measurement.

The contracts worth pinning here are mostly about what the report REFUSES to
claim: an absent probe log is not perfect uptime, a gap on a closed market is
not a miss, and a coverage figure never ships without the blind spots that make
it a bound. The holiday case is a regression test — the first version of this
measurement scored a 57.8h Independence Day gap as the largest miss in the
report, and it was worth nothing.
"""

from datetime import datetime, timedelta

import pytest

from tools import availability


def _write_probes(path, stamps, status="ok"):
    """Write a watchdog-shaped log: one `<isoformat> <status>` line per stamp."""
    path.write_text(
        "".join(f"{s.isoformat(timespec='seconds')} {status}\n" for s in stamps),
        encoding="utf-8",
    )
    return str(path)


def _cadence(start, count, minutes=2):
    """`count` probes at `minutes` spacing from `start`."""
    return [start + timedelta(minutes=minutes * i) for i in range(count)]


# ---------------------------------------------------------------------------
# Absent / unusable instrument
# ---------------------------------------------------------------------------
def test_missing_log_is_no_data_not_perfect_uptime(tmp_path):
    report = availability.measure_availability(str(tmp_path / "nope.log"))
    assert report["status"] == "no_data"
    assert "coverage" not in report
    assert report.get("window_coverage_pct") is None
    # The distinction the whole module exists to protect.
    assert "not" in report["note"].lower()
    assert "100%" in report["note"]


def test_single_probe_cannot_measure_an_interval(tmp_path):
    log = _write_probes(tmp_path / "w.log", [datetime(2026, 7, 6, 9, 0)])
    assert availability.measure_availability(log)["status"] == "no_data"


def test_unparseable_lines_are_dropped_not_guessed(tmp_path):
    p = tmp_path / "w.log"
    p.write_text(
        "not a probe line\n"
        "2026-07-06T09:00:00 ok\n"
        "garbage 2026\n"
        "2026-07-06T09:02:00 ok\n",
        encoding="utf-8",
    )
    probes = availability.read_probes(str(p))
    assert len(probes) == 2
    assert [s.minute for s, _ in probes] == [0, 2]


# ---------------------------------------------------------------------------
# Trading calendar — the denominator
# ---------------------------------------------------------------------------
def test_market_holiday_is_not_a_trading_day():
    # 2026-07-03: Independence Day observed. A Friday, and the market is shut.
    assert availability.is_trading_day(datetime(2026, 7, 3).date()) is False
    assert availability.is_trading_day(datetime(2026, 7, 6).date()) is True  # Monday


def test_weekend_is_not_a_trading_day():
    assert availability.is_trading_day(datetime(2026, 7, 18).date()) is False  # Sat
    assert availability.is_trading_day(datetime(2026, 7, 19).date()) is False  # Sun


def test_gap_across_a_market_holiday_costs_zero_window_minutes():
    """REGRESSION. The first version of this measurement counted 412 lost
    window-minutes for a gap that spanned a closed market, and that single
    non-miss was the largest entry in the report."""
    # Friday 2026-07-03 09:38 -> Sunday 2026-07-05 19:24, the real recorded gap.
    lost = availability.window_minutes(
        datetime(2026, 7, 3, 9, 38), datetime(2026, 7, 5, 19, 24)
    )
    assert lost == 0.0


def test_gap_on_a_trading_day_costs_its_in_window_minutes():
    # Friday 2026-07-10 09:35 -> 13:37 is fully inside 07:00-16:30.
    lost = availability.window_minutes(
        datetime(2026, 7, 10, 9, 35), datetime(2026, 7, 10, 13, 37)
    )
    assert lost == pytest.approx(242.0, abs=0.5)


def test_overnight_gap_outside_the_window_costs_nothing():
    lost = availability.window_minutes(
        datetime(2026, 7, 8, 18, 0), datetime(2026, 7, 9, 6, 0)
    )
    assert lost == 0.0


def test_window_is_clipped_at_both_edges():
    # 06:00 -> 08:00 on a trading day contributes only 07:00-08:00.
    lost = availability.window_minutes(
        datetime(2026, 7, 8, 6, 0), datetime(2026, 7, 8, 8, 0)
    )
    assert lost == pytest.approx(60.0, abs=0.5)


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------
def test_normal_cadence_produces_no_gaps(tmp_path):
    log = _write_probes(tmp_path / "w.log", _cadence(datetime(2026, 7, 6, 9, 0), 60))
    report = availability.measure_availability(log)
    assert report["status"] == "measured"
    assert report["gap_count"] == 0
    assert report["window_coverage_pct"] == 100.0


def test_a_gap_over_the_threshold_is_reported_with_its_cost(tmp_path):
    stamps = _cadence(datetime(2026, 7, 6, 9, 0), 10)
    stamps += _cadence(datetime(2026, 7, 6, 12, 0), 10)  # ~2h42m hole
    log = _write_probes(tmp_path / "w.log", stamps)
    report = availability.measure_availability(log)

    assert report["gap_count"] == 1
    assert report["gaps_in_window"] == 1
    gap = report["gaps"][0]
    assert gap["start"].startswith("2026-07-06T09:18")
    assert gap["hours"] == pytest.approx(2.7, abs=0.05)
    assert gap["window_minutes_lost"] == pytest.approx(162.0, abs=0.5)
    assert report["window_coverage_pct"] < 100.0


def test_jitter_under_the_threshold_is_not_an_outage(tmp_path):
    stamps = _cadence(datetime(2026, 7, 6, 9, 0), 5)
    stamps.append(stamps[-1] + timedelta(minutes=4))  # under the 5-min floor
    log = _write_probes(tmp_path / "w.log", stamps)
    assert availability.measure_availability(log)["gap_count"] == 0


def test_find_gaps_does_not_name_a_cause():
    """Asleep, powered off, and a dead watchdog are indistinguishable here, and
    the module must not pick one — the same contract as 4.10a's `unclassified`."""
    gaps = availability.find_gaps(
        [datetime(2026, 7, 6, 9, 0), datetime(2026, 7, 6, 11, 0)]
    )
    assert len(gaps) == 1
    assert set(gaps[0]) == {"start", "end", "hours", "window_minutes_lost"}
    assert "cause" not in gaps[0]
    assert "reason" not in gaps[0]


# ---------------------------------------------------------------------------
# Wall clock vs window — the headline must be the window
# ---------------------------------------------------------------------------
def test_a_nightly_sleep_scores_poor_uptime_and_full_coverage(tmp_path):
    """The distinction the whole item rests on: a box that sleeps every night is
    fully available for this product and looks broken on a wall-clock figure."""
    stamps = []
    for day in (6, 7, 8):  # Mon-Wed
        stamps += _cadence(datetime(2026, 7, day, 6, 30), 330)  # 06:30 -> 17:30
    log = _write_probes(tmp_path / "w.log", stamps)
    report = availability.measure_availability(log)

    assert report["window_coverage_pct"] == 100.0
    assert report["wall_clock_uptime_pct"] < 60.0
    assert report["gap_count"] == 2  # the two nights


# ---------------------------------------------------------------------------
# Honesty of the report itself
# ---------------------------------------------------------------------------
def test_unhealthy_probe_statuses_are_surfaced(tmp_path):
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 ok\n"
        "2026-07-06T09:02:00 server down on 127.0.0.1:8000 — kickstarting\n"
        "2026-07-06T09:04:00 ok\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(p))
    assert report["unhealthy_probes"] == {
        "server down on <n>.<n>.<n>.<n>:<n> — kickstarting": 1
    }
    assert "ok" not in report["unhealthy_probes"]


def test_repeated_events_group_despite_their_pids(tmp_path):
    """REGRESSION, found on the real log and invisible to the fixtures. Probe
    messages carry a pid, so keying the tally on the raw string turned seven
    restarts into seven unrelated events each with a count of 1."""
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 startup in progress (pid 460)\n"
        "2026-07-06T09:02:00 startup in progress (pid 10432)\n"
        "2026-07-06T09:04:00 startup in progress (pid 1210)\n",
        encoding="utf-8",
    )
    unhealthy = availability.measure_availability(str(p))["unhealthy_probes"]
    assert len(unhealthy) == 1
    assert list(unhealthy.values()) == [3]


def test_a_second_writer_is_deduped_and_still_reported(tmp_path):
    """REGRESSION, found on the real log and invisible to the fixtures. Two
    watchdog instances appended to the probe log from 06-27 to 07-18, so `probes`
    — the count 2.6 reports as production — read 2x high, and the cadence of a
    2-minute probe rendered as 1.38 minutes. Coverage never moved: a 0-minute
    interval is below any gap threshold."""
    stamps = _cadence(datetime(2026, 7, 6, 9, 0), 30)
    doubled = [s for s in stamps for _ in (0, 1)]  # every line written twice
    log = _write_probes(tmp_path / "w.log", doubled)
    report = availability.measure_availability(log)

    assert report["probes"] == 30
    assert report["duplicate_probe_lines"] == 30
    assert report["probe_cadence_minutes"] == pytest.approx(2.0, abs=0.01)
    # The duplicate writer must not invent an outage or move the figure.
    assert report["gap_count"] == 0
    assert report["window_coverage_pct"] == 100.0


def test_a_single_writer_reports_no_duplicates(tmp_path):
    log = _write_probes(tmp_path / "w.log", _cadence(datetime(2026, 7, 6, 9, 0), 20))
    assert availability.measure_availability(log)["duplicate_probe_lines"] == 0


def test_duplicate_statuses_are_not_double_counted(tmp_path):
    """The unhealthy tally is keyed on the message, so a second writer would
    otherwise report every restart twice."""
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 ok\n"
        "2026-07-06T09:00:00 ok\n"
        "2026-07-06T09:02:00 server down on 127.0.0.1:8000 — kickstarting\n"
        "2026-07-06T09:02:00 server down on 127.0.0.1:8000 — kickstarting\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(p))
    assert report["unhealthy_probes"] == {
        "server down on <n>.<n>.<n>.<n>:<n> — kickstarting": 1
    }


def test_every_report_carries_its_open_measurements(tmp_path):
    absent = availability.measure_availability(str(tmp_path / "nope.log"))
    log = _write_probes(tmp_path / "w.log", _cadence(datetime(2026, 7, 6, 9, 0), 10))
    measured = availability.measure_availability(log)

    for report in (absent, measured):
        numbers = {m["number"]: m["state"] for m in report["open_measurements"]}
        assert numbers["process availability"] == "MEASURED"
        # What this module cannot deliver, named rather than folded into the
        # coverage figure.
        assert "NOT MEASURED" in numbers["surfaces returning 5xx"]
        # Number 3 has a replay now (`tools.missed_alerts`), which reads THESE
        # gaps — so it stops being EXPOSURE ONLY and takes number 4's shape
        # instead: measured, per-profile, and deliberately not folded in here. A
        # watch condition belongs to one person and this report is global.
        missed = numbers["alerts that should have fired and did not"]
        assert "MEASURED PER-PROFILE" in missed
        # The exposure claim itself must survive the change: the gaps are still
        # all this report carries, and the count lives elsewhere.
        assert "EXPOSURE" in next(
            m["how"] for m in report["open_measurements"]
            if m["number"] == "alerts that should have fired and did not")
        # Number 4 has a field now (`read_at`), so it is no longer NOT COMPUTABLE —
        # but it stays OFF this report, which is global, because an inbox is one
        # person's. The row points at the per-profile seam instead of carrying a
        # number; `tests/test_api/test_availability_api.py` holds that boundary.
        assert "MEASURED PER-PROFILE" in numbers["delivery latency (produced-at vs read-at)"]


# ---------------------------------------------------------------------------
# Step 2 — the stated SLO, and the asymmetry of judging a bound
# ---------------------------------------------------------------------------
def test_the_stated_slo_is_98_percent():
    """Pinned. Step 2's whole purpose is that the bar was written down BEFORE the
    remedy, so a later cycle cannot quietly move it to whatever was achieved."""
    assert availability.SLO_WINDOW_COVERAGE_PCT == 98.0


def test_a_bound_below_the_floor_is_a_definite_breach():
    verdict = availability.evaluate_slo(97.93)
    assert verdict["slo_state"] == "breached"
    # The direction of the bound is what makes this certain rather than probable.
    assert "DEFINITE" in verdict["slo_note"]


def test_a_bound_above_the_floor_is_never_reported_as_met():
    """The asymmetry. Coverage is an UPPER bound, so clearing the bar proves
    nothing on its own — the unmeasured 5xx time sits underneath the figure."""
    verdict = availability.evaluate_slo(99.5)
    assert verdict["slo_state"] == "not_proven_met"
    assert verdict["slo_state"] != "met"
    assert "upper bound" in verdict["slo_note"]


def test_an_absent_figure_is_unknown_not_passing():
    verdict = availability.evaluate_slo(None)
    assert verdict["slo_state"] == "unknown"
    assert "not passing" in verdict["slo_note"]


def test_the_slo_rides_on_the_report(tmp_path):
    stamps = _cadence(datetime(2026, 7, 6, 9, 0), 10)
    stamps += _cadence(datetime(2026, 7, 6, 14, 0), 10)  # a big in-window hole
    log = _write_probes(tmp_path / "w.log", stamps)
    report = availability.measure_availability(log)
    assert report["slo_target_pct"] == 98.0
    assert report["slo_state"] == "breached"


# ---------------------------------------------------------------------------
# Step 3 — the 5xx axis
# ---------------------------------------------------------------------------
def test_a_log_with_no_serving_probe_does_not_claim_zero_errors(tmp_path):
    """The boundary that keeps this honest. Weeks of the real log predate the
    serving probe, and zero 5xx found where nothing was looking is not a clean
    record — it is no record."""
    log = _write_probes(tmp_path / "w.log", _cadence(datetime(2026, 7, 6, 9, 0), 20))
    report = availability.measure_availability(log)

    assert report["serving_probe_active"] is False
    assert report["first_serving_probe"] is None
    state = {m["number"]: m["state"] for m in report["open_measurements"]}
    assert "NOT MEASURED" in state["surfaces returning 5xx"]


def test_a_healthy_serving_log_proves_the_probe_ran(tmp_path):
    """REGRESSION, found by deploying it. `serving_probe_active` could only turn
    true when something BROKE, because a healthy probe wrote a bare `ok` — the
    same line the old tcp-only probe wrote. A permanently healthy host reported
    its perfectly working instrument as absent, and the claim was unfalsifiable in
    exactly the case you most want to trust."""
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 ok HTTP 200\n"
        "2026-07-06T09:02:00 ok HTTP 200\n"
        "2026-07-06T09:04:00 ok HTTP 200\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(p))

    assert report["serving_probe_active"] is True
    assert report["first_serving_probe"] == "2026-07-06T09:00:00"
    assert report["serving_ok_probes"] == 3
    assert report["serving_error_probes"] == 0
    state = {m["number"]: m["state"] for m in report["open_measurements"]}
    assert state["surfaces returning 5xx"] == "MEASURED — from the serving probe"


def test_a_bare_ok_still_proves_nothing(tmp_path):
    """The other half of the same contract: `ok` is what the tcp-only probe wrote,
    so a log of them must NOT be read as a measured 5xx axis."""
    log = _write_probes(tmp_path / "w.log", _cadence(datetime(2026, 7, 6, 9, 0), 20))
    report = availability.measure_availability(log)
    assert report["serving_probe_active"] is False
    assert report["serving_ok_probes"] == 0


def test_the_healthy_line_is_not_filed_as_unhealthy(tmp_path):
    """`ok HTTP 200` is healthy. Left out of the skip list it would file the entire
    healthy majority under `unhealthy_probes` as one huge `ok HTTP <n>` entry."""
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 ok HTTP 200\n"
        "2026-07-06T09:02:00 ok\n"
        "2026-07-06T09:04:00 ok HTTP 200\n",
        encoding="utf-8",
    )
    assert availability.measure_availability(str(p))["unhealthy_probes"] == {}


def test_a_serving_error_is_measured_and_dated(tmp_path):
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 ok\n"
        "2026-07-06T09:02:00 SERVING-ERRORS HTTP 500 from /api/health — serving errors\n"
        "2026-07-06T09:04:00 SERVING-ERRORS HTTP 502 from /api/health — serving errors\n"
        "2026-07-06T09:06:00 ok\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(p))

    assert report["serving_probe_active"] is True
    assert report["first_serving_probe"] == "2026-07-06T09:02:00"
    assert report["serving_error_probes"] == 2
    assert report["serving_error_codes"] == {"500": 1, "502": 1}
    state = {m["number"]: m["state"] for m in report["open_measurements"]}
    assert state["surfaces returning 5xx"] == "MEASURED — from the serving probe"


def test_the_status_code_survives_digit_normalisation(tmp_path):
    """`unhealthy_probes` deliberately rewrites digits to <n> to group restarts by
    message. The 5xx axis is parsed BEFORE that, or every code would read <n>."""
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 ok\n"
        "2026-07-06T09:02:00 SERVING-ERRORS HTTP 503 from /api/health — serving errors\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(p))
    assert "503" in report["serving_error_codes"]
    # And the grouped tally still normalises, as it must.
    assert any("<n>" in key for key in report["unhealthy_probes"])


def test_errors_outside_the_window_are_counted_separately(tmp_path):
    p = tmp_path / "w.log"
    p.write_text(
        # 22:00 is outside 07:00-16:30, so it is an error but not a window loss.
        "2026-07-06T22:00:00 SERVING-ERRORS HTTP 500 from /api/health — serving errors\n"
        "2026-07-07T10:00:00 SERVING-ERRORS HTTP 500 from /api/health — serving errors\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(p))
    assert report["serving_error_probes"] == 2
    assert report["serving_errors_in_window"] == 1


def test_stale_code_probes_are_counted(tmp_path):
    p = tmp_path / "w.log"
    p.write_text(
        "2026-07-06T09:00:00 ok\n"
        "2026-07-06T09:02:00 STALE-CODE: /api/health answers 200, but the running "
        "process started before the newest source on disk\n",
        encoding="utf-8",
    )
    report = availability.measure_availability(str(p))
    assert report["stale_code_probes"] == 1
    assert report["serving_probe_active"] is True


def test_coverage_outside_the_holiday_table_is_flagged_as_a_bound(tmp_path):
    """2031 is not in the authored table, so any window minutes there rest on the
    weekday fallback and the report has to say so."""
    log = _write_probes(tmp_path / "w.log", _cadence(datetime(2031, 7, 8, 9, 0), 30))
    report = availability.measure_availability(log)
    assert report["window_minutes_outside_holiday_table"] > 0
    assert 2031 not in report["holiday_table_years"]

    log2026 = _write_probes(tmp_path / "y.log", _cadence(datetime(2026, 7, 8, 9, 0), 30))
    assert availability.measure_availability(log2026)[
        "window_minutes_outside_holiday_table"
    ] == 0


def test_summary_states_the_figure_is_an_upper_bound(tmp_path):
    stamps = _cadence(datetime(2026, 7, 6, 9, 0), 10)
    stamps += _cadence(datetime(2026, 7, 6, 12, 0), 10)
    log = _write_probes(tmp_path / "w.log", stamps)
    report = availability.get_availability_report(log)
    assert "upper bound" in report["summary"]
    assert str(report["window_coverage_pct"]) in report["summary"]


def test_report_never_raises_on_a_binary_log(tmp_path):
    p = tmp_path / "w.log"
    p.write_bytes(b"\x00\xff\xfe not a log at all \x00")
    assert availability.get_availability_report(str(p))["status"] == "no_data"
