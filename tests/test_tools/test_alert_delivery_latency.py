"""7.1 number 4 — delivery latency, and the button press it must not average in.

Availability answers whether the box was up to send. This answers whether anyone
was there to receive, which is the only one of 7.1's four numbers that measures
Theme 3's actual promise.

The contract that matters is not the median. It is that a mark-all click — which
stamps every unread alert with one instant — is counted and never timed, and that
alerts read before the stamp existed are reported as unmeasurable rather than
quietly shrinking the denominator.
"""

import json
from datetime import datetime, timedelta

from tools import alerts


def _store(tmp_path, monkeypatch, records):
    path = tmp_path / "alerts.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    monkeypatch.setattr(alerts, "_alerts_file", lambda: str(path))
    return path


def _alert(aid, ts, read=False, read_at=None, read_via=None):
    rec = {
        "id": aid, "ts": ts, "severity": "info", "title": "t",
        "message": "m", "source": "s", "read": read, "dedup_key": None, "data": {},
    }
    if read_at:
        rec["read_at"] = read_at
    if read_via:
        rec["read_via"] = read_via
    return rec


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------
def test_marking_by_id_stamps_the_clock_and_the_manner(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch, [_alert("a1", "2026-07-29T09:00:00")])

    assert alerts.mark_read(alert_ids=["a1"]) == 1

    rec = alerts._load_all()[0]
    assert rec["read"] is True
    assert rec["read_via"] == "id"
    assert rec["read_at"]  # an ISO stamp, not a boolean
    datetime.fromisoformat(rec["read_at"])


def test_mark_all_records_itself_as_bulk(tmp_path, monkeypatch):
    """The distinction the measurement rests on. Without `read_via` a single click
    is indistinguishable from N people reading N alerts at the same instant."""
    _store(tmp_path, monkeypatch, [
        _alert("a1", "2026-07-29T09:00:00"),
        _alert("a2", "2026-07-29T09:30:00"),
    ])

    assert alerts.mark_read(all_alerts=True) == 2

    assert {r["read_via"] for r in alerts._load_all()} == {"all"}


def test_an_already_read_alert_is_not_restamped(tmp_path, monkeypatch):
    """Re-marking must not move the clock — that would reset the age of an alert
    that was read days ago and make late reads look instant."""
    _store(tmp_path, monkeypatch, [
        _alert("a1", "2026-07-29T09:00:00", read=True,
               read_at="2026-07-29T09:05:00", read_via="id"),
    ])

    assert alerts.mark_read(alert_ids=["a1"]) == 0
    assert alerts._load_all()[0]["read_at"] == "2026-07-29T09:05:00"


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------
def test_latency_is_the_difference_between_raised_and_read(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch, [
        _alert("a1", "2026-07-29T09:00:00", read=True,
               read_at="2026-07-29T09:10:00", read_via="id"),  # 600s
    ])

    out = alerts.get_delivery_latency()
    assert out["status"] == "measured"
    assert out["timed_reads"] == 1
    assert out["median_seconds"] == 600.0


def test_a_mark_all_click_is_counted_but_never_timed(tmp_path, monkeypatch):
    """THE CONTRACT. Three alerts of very different ages marked at one instant is
    one button press, not three fast reads — and the oldest one's age is not a
    reading time."""
    _store(tmp_path, monkeypatch, [
        _alert("a1", "2026-07-20T09:00:00", read=True,
               read_at="2026-07-29T09:00:00", read_via="all"),
        _alert("a2", "2026-07-25T09:00:00", read=True,
               read_at="2026-07-29T09:00:00", read_via="all"),
        _alert("a3", "2026-07-29T08:55:00", read=True,
               read_at="2026-07-29T09:00:00", read_via="id"),  # 300s, real
    ])

    out = alerts.get_delivery_latency()
    assert out["bulk_read"] == 2
    assert out["timed_reads"] == 1
    assert out["median_seconds"] == 300.0
    assert "mark-all" in out["note"]


def test_alerts_read_before_the_stamp_existed_are_unmeasurable(tmp_path, monkeypatch):
    """The live store's 42 alerts are all `read: True` with no clock. They must
    read as UNKNOWN, not as a clean record — the same rule the serving probe
    follows about the span predating it."""
    _store(tmp_path, monkeypatch, [
        _alert("a1", "2026-07-20T09:00:00", read=True),
        _alert("a2", "2026-07-21T09:00:00", read=True),
    ])

    out = alerts.get_delivery_latency()
    assert out["status"] == "no_data"
    assert out["unmeasurable_reads"] == 2
    assert out["timed_reads"] == 0
    assert "UNKNOWN" in out["note"]
    assert "median_seconds" not in out


def test_no_stamped_read_is_unknown_not_fast(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch, [_alert("a1", "2026-07-29T09:00:00")])
    out = alerts.get_delivery_latency()
    assert out["status"] == "no_data"
    assert out["alerts_unread"] == 1


def test_a_negative_delta_is_not_a_zero_second_read(tmp_path, monkeypatch):
    """A clock change or a hand-edited record must not be averaged as an
    instantaneous read."""
    _store(tmp_path, monkeypatch, [
        _alert("a1", "2026-07-29T09:00:00", read=True,
               read_at="2026-07-29T08:00:00", read_via="id"),
    ])

    out = alerts.get_delivery_latency()
    assert out["timed_reads"] == 0
    assert out["unmeasurable_reads"] == 1


def test_percentiles_come_from_the_timed_reads_only(tmp_path, monkeypatch):
    base = datetime(2026, 7, 29, 9, 0, 0)
    records = [
        _alert(f"a{i}", base.isoformat(timespec="seconds"), read=True,
               read_at=(base + timedelta(seconds=60 * (i + 1))).isoformat(timespec="seconds"),
               read_via="id")
        for i in range(10)
    ]
    # One bulk click with an enormous age, which must not move max_seconds.
    records.append(_alert("bulk", "2026-01-01T00:00:00", read=True,
                          read_at="2026-07-29T09:00:00", read_via="all"))
    _store(tmp_path, monkeypatch, records)

    out = alerts.get_delivery_latency()
    assert out["timed_reads"] == 10
    assert out["max_seconds"] == 600.0  # 10 minutes, not seven months
    assert out["median_seconds"] <= out["p90_seconds"] <= out["max_seconds"]


def test_the_measurement_survives_a_corrupt_record(tmp_path, monkeypatch):
    path = tmp_path / "alerts.jsonl"
    path.write_text(
        json.dumps(_alert("a1", "2026-07-29T09:00:00", read=True,
                          read_at="not-a-date", read_via="id")) + "\n"
        + "{ not json at all\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(alerts, "_alerts_file", lambda: str(path))

    out = alerts.get_delivery_latency()
    assert out["status"] == "no_data"
    assert out["unmeasurable_reads"] == 1


# ---------------------------------------------------------------------------
# Round trip through the real write path
# ---------------------------------------------------------------------------
def test_a_raised_then_read_alert_measures_end_to_end(tmp_path, monkeypatch):
    """No hand-built fixture: raise through the real producer, mark through the
    real writer, then measure. A contract asserted only on fixtures is how the
    two halves drift apart."""
    _store(tmp_path, monkeypatch, [])
    monkeypatch.setattr(alerts, "_broadcast", lambda rec: None)
    monkeypatch.setattr(alerts, "_notify_desktop", lambda t, b: None)

    rec = alerts.raise_alert("t", "m", severity="info")
    assert rec and rec["read"] is False

    assert alerts.mark_read(alert_ids=[rec["id"]]) == 1

    out = alerts.get_delivery_latency()
    assert out["status"] == "measured"
    assert out["timed_reads"] == 1
    assert out["median_seconds"] >= 0
