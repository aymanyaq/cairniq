"""Per-profile alerts.jsonl inbox (Advisor Roadmap Theme 3.2)."""
import json

import tools.alerts as alerts


def _use_tmp_file(monkeypatch, tmp_path):
    path = tmp_path / "alerts.jsonl"
    monkeypatch.setattr(alerts, "get_data_path", lambda filename: str(path))
    # Deliveries are side effects; capture instead of executing.
    monkeypatch.setattr(alerts, "_broadcast", lambda record: None)
    notified = []
    monkeypatch.setattr(alerts, "_notify_desktop", lambda t, b: notified.append((t, b)))
    return path, notified


def _read_all(path):
    return [json.loads(ln) for ln in path.read_text().strip().split("\n") if ln.strip()]


def test_raise_appends_record_with_defaults(monkeypatch, tmp_path):
    path, _ = _use_tmp_file(monkeypatch, tmp_path)

    rec = alerts.raise_alert("Regime flip: NEUTRAL → FEAR", "VIX spiking", source="market_sentinel")

    assert rec is not None
    assert rec["severity"] == "info"
    assert rec["read"] is False
    assert rec["ts"].startswith("20")
    stored = _read_all(path)
    assert len(stored) == 1
    assert stored[0]["title"] == "Regime flip: NEUTRAL → FEAR"
    assert alerts.get_unread_count() == 1


def test_warning_and_critical_notify_desktop_by_default(monkeypatch, tmp_path):
    _, notified = _use_tmp_file(monkeypatch, tmp_path)

    alerts.raise_alert("fyi", "m", severity="info")
    alerts.raise_alert("heads up", "m", severity="warning")
    alerts.raise_alert("act now", "m", severity="critical")

    assert [t for t, _ in notified] == ["CairnIQ — heads up", "CairnIQ — act now"]


def test_notify_param_overrides_default(monkeypatch, tmp_path):
    _, notified = _use_tmp_file(monkeypatch, tmp_path)

    alerts.raise_alert("quiet critical", "m", severity="critical", notify=False)
    alerts.raise_alert("loud info", "m", severity="info", notify=True)

    assert [t for t, _ in notified] == ["CairnIQ — loud info"]


def test_dedup_key_refreshes_unread_instead_of_duplicating(monkeypatch, tmp_path):
    path, _ = _use_tmp_file(monkeypatch, tmp_path)

    first = alerts.raise_alert("Action required", "v1", dedup_key="ar-2026-07-19")
    second = alerts.raise_alert("Action required", "v2", dedup_key="ar-2026-07-19")

    stored = _read_all(path)
    assert len(stored) == 1
    assert stored[0]["id"] == first["id"] == second["id"]
    assert stored[0]["message"] == "v2"
    assert stored[0]["refreshed_count"] == 1


def test_dedup_allows_new_alert_once_read(monkeypatch, tmp_path):
    path, _ = _use_tmp_file(monkeypatch, tmp_path)

    alerts.raise_alert("Action required", "v1", dedup_key="k")
    alerts.mark_read(all_alerts=True)
    alerts.raise_alert("Action required", "v2", dedup_key="k")

    assert len(_read_all(path)) == 2
    assert alerts.get_unread_count() == 1


def test_mark_read_by_ids_and_all(monkeypatch, tmp_path):
    _use_tmp_file(monkeypatch, tmp_path)
    a = alerts.raise_alert("a", "m")
    b = alerts.raise_alert("b", "m")
    c = alerts.raise_alert("c", "m")

    assert alerts.mark_read(alert_ids=[a["id"], b["id"]]) == 2
    assert alerts.get_unread_count() == 1
    assert alerts.mark_read(all_alerts=True) == 1
    assert alerts.get_unread_count() == 0
    # Idempotent: nothing left to change.
    assert alerts.mark_read(alert_ids=[c["id"]]) == 0


def test_get_alerts_newest_first_with_unread_filter(monkeypatch, tmp_path):
    _use_tmp_file(monkeypatch, tmp_path)
    alerts.raise_alert("first", "m")
    second = alerts.raise_alert("second", "m")
    alerts.mark_read(alert_ids=[second["id"]])

    all_alerts = alerts.get_alerts()
    assert [a["title"] for a in all_alerts] == ["second", "first"]
    unread = alerts.get_alerts(unread_only=True)
    assert [a["title"] for a in unread] == ["first"]


def test_store_is_capped(monkeypatch, tmp_path):
    path, _ = _use_tmp_file(monkeypatch, tmp_path)
    monkeypatch.setattr(alerts, "_MAX_RECORDS", 10)

    for i in range(15):
        alerts.raise_alert(f"a{i}", "m")

    stored = _read_all(path)
    assert len(stored) == 10
    assert stored[0]["title"] == "a5"


def test_raise_never_raises_on_write_failure(monkeypatch):
    monkeypatch.setattr(alerts, "get_data_path", lambda filename: "/nonexistent-dir-\x00/x.jsonl")
    monkeypatch.setattr(alerts, "_broadcast", lambda record: None)
    monkeypatch.setattr(alerts, "_notify_desktop", lambda t, b: None)

    assert alerts.raise_alert("t", "m") is None
    assert alerts.get_alerts() == []
    assert alerts.get_unread_count() == 0
    assert alerts.mark_read(all_alerts=True) == 0
