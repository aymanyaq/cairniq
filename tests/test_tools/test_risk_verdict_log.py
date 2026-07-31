"""Per-profile risk_verdicts.jsonl audit trail (Advisor Roadmap Theme 2.1)."""
import json

import tools.risk_verdict_log as rvl


def _use_tmp_file(monkeypatch, tmp_path):
    path = tmp_path / "risk_verdicts.jsonl"
    monkeypatch.setattr(rvl, "get_data_path", lambda filename: str(path))
    return path


def test_log_appends_json_lines_with_timestamp(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)

    assert rvl.log_risk_verdict({"event": "verdict", "score": 2, "risk_result": "CRITICAL_FAIL"})
    assert rvl.log_risk_verdict({"event": "verdict", "score": 10, "risk_result": "PASS"})

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["score"] == 2
    assert first["risk_result"] == "CRITICAL_FAIL"
    assert "ts" in first and first["ts"].startswith("20")


def test_log_truncates_verdict_text_and_query(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)

    rvl.log_risk_verdict({
        "event": "verdict",
        "verdict_text": "v" * 10_000,
        "query": "q" * 2_000,
    })

    rec = json.loads(path.read_text().strip())
    assert len(rec["verdict_text"]) == rvl._MAX_TEXT_CHARS
    assert len(rec["query"]) == rvl._MAX_QUERY_CHARS


def test_log_never_raises_on_write_failure(monkeypatch):
    def boom(filename):
        raise OSError("disk on fire")

    monkeypatch.setattr(rvl, "get_data_path", boom)
    assert rvl.log_risk_verdict({"event": "verdict", "score": 5}) is False


def test_get_recent_verdicts_orders_and_limits(monkeypatch, tmp_path):
    _use_tmp_file(monkeypatch, tmp_path)

    for i in range(5):
        rvl.log_risk_verdict({"event": "verdict", "score": i})

    recent = rvl.get_recent_verdicts(limit=3)
    assert [r["score"] for r in recent] == [2, 3, 4]  # oldest → newest tail


def test_get_recent_verdicts_skips_corrupt_lines(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)

    rvl.log_risk_verdict({"event": "verdict", "score": 7})
    with open(path, "a") as f:
        f.write("{not valid json\n")
        f.write('"a bare string, not an object"\n')
    rvl.log_risk_verdict({"event": "verdict", "score": 9})

    scores = [r["score"] for r in rvl.get_recent_verdicts()]
    assert scores == [7, 9]


def test_get_recent_verdicts_filters_bypassed_by_default(monkeypatch, tmp_path):
    _use_tmp_file(monkeypatch, tmp_path)

    rvl.log_risk_verdict({"event": "verdict", "score": 10})
    rvl.log_risk_verdict({"event": "bypassed", "risk_result": "PASS"})

    assert len(rvl.get_recent_verdicts()) == 1
    assert len(rvl.get_recent_verdicts(include_bypassed=True)) == 2


def test_get_recent_verdicts_empty_when_no_file(monkeypatch, tmp_path):
    _use_tmp_file(monkeypatch, tmp_path)
    assert rvl.get_recent_verdicts() == []
