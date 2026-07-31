"""Turn-wide data provenance (Advisor Roadmap 2.3 × 5.8).

The module exists to stop degraded evidence reading as complete evidence, so the
tests that matter are the ones about which way it errs. An optimistic provenance
summary is worse than none at all: it puts "all sources live" underneath advice
built on an unavailable feed, and it does so in the one place a reader would go
to check.

So: unstamped must never read as fresh, one fresh call among five must never make
the set read as live, and a parse failure must degrade to silence rather than to
a reassuring default.
"""
from datetime import datetime, timedelta

import pytest

import tools.provenance as prov

_NOW = datetime(2026, 7, 27, 15, 0, 0)


def _block(name, payload):
    return f"### Tool Call: {name}({{}})\nResult:\n{payload}"


def _ctx(*blocks):
    return "\n\n".join(blocks)


def _stamped(minutes_ago, extra=""):
    at = (_NOW - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    return "{'price': 101.5, '_as_of': '" + at + "'" + extra + "}"


# ---------------------------------------------------------------------------
# Never guess in the reassuring direction
# ---------------------------------------------------------------------------

def test_an_unstamped_payload_is_unverified_never_fresh():
    """Absence of proof is not proof of freshness — the standing 5.8 policy."""
    summary = prov.summarize_tool_context(_ctx(_block("get_quote", "{'price': 101.5}")), _NOW)

    assert summary["sources"][0]["status"] == prov.STATUS_UNVERIFIED
    assert summary["counts"]["fresh"] == 0


def test_an_unparseable_stamp_is_unverified_not_fresh():
    payload = "{'price': 1, '_as_of': 'not-a-timestamp-at-all'}"
    summary = prov.summarize_tool_context(_ctx(_block("get_quote", payload)), _NOW)

    assert summary["sources"][0]["status"] == prov.STATUS_UNVERIFIED


def test_the_footer_reports_the_oldest_reading_not_the_freshest():
    """A footer saying 'live' because one of six calls was fresh is the exact
    overclaim this line exists to prevent."""
    summary = prov.summarize_tool_context(
        _ctx(_block("get_quote", _stamped(1)), _block("get_macro", _stamped(180))), _NOW
    )

    assert "3h" in summary["footer"]
    assert "live" not in summary["footer"]


def test_a_garbled_context_degrades_to_silence_not_to_a_default():
    summary = prov.summarize_tool_context("\x00 not a tool context at all", _NOW)

    assert summary["sources"] == []
    assert summary["footer"] == ""
    assert summary["degraded"] is False


# ---------------------------------------------------------------------------
# unavailable() — the case nobody writes by hand
# ---------------------------------------------------------------------------

def test_an_unavailable_payload_is_named_with_its_source():
    payload = (
        "{'status': 'unavailable', 'source': 'FMP', "
        "'reason': 'FMP_API_KEY not configured — add it in Settings → API Keys.'}"
    )
    summary = prov.summarize_tool_context(_ctx(_block("get_insider_trades", payload)), _NOW)

    entry = summary["sources"][0]
    assert entry["status"] == prov.STATUS_UNAVAILABLE
    assert entry["source"] == "FMP"
    assert "FMP_API_KEY" in entry["reason"]
    assert summary["degraded"] is True
    assert "FMP unavailable" in summary["footer"]


def test_json_quoting_is_recognized_as_well_as_repr_quoting():
    """Which quoting reaches the judge depends on which node stringified the
    payload. Both must parse, or provenance silently disagrees with the evidence
    it is describing."""
    payload = '{"status": "unavailable", "source": "Tavily", "reason": "quota exhausted"}'
    summary = prov.summarize_tool_context(_ctx(_block("web_search", payload)), _NOW)

    assert summary["sources"][0]["status"] == prov.STATUS_UNAVAILABLE
    assert summary["sources"][0]["source"] == "Tavily"


def test_an_unavailable_tool_without_a_source_falls_back_to_the_tool_name():
    payload = "{'status': 'unavailable', 'reason': 'upstream outage'}"
    summary = prov.summarize_tool_context(_ctx(_block("get_esg_scores", payload)), _NOW)

    assert "get_esg_scores unavailable" in summary["footer"]


# ---------------------------------------------------------------------------
# Fresh / stale / degraded
# ---------------------------------------------------------------------------

def test_a_recent_fetch_is_fresh_and_not_degraded():
    summary = prov.summarize_tool_context(_ctx(_block("get_quote", _stamped(2))), _NOW)

    assert summary["sources"][0]["status"] == prov.STATUS_FRESH
    assert summary["degraded"] is False
    assert "live" in summary["footer"]


def test_a_fetch_past_the_threshold_is_stale_and_degrades_the_turn():
    summary = prov.summarize_tool_context(
        _ctx(_block("get_quote", _stamped(prov.STALE_AFTER_MINUTES + 10))), _NOW
    )

    assert summary["sources"][0]["status"] == prov.STATUS_STALE
    assert summary["degraded"] is True


def test_unverified_alone_does_not_mark_the_turn_degraded():
    """Almost nothing outside the cached surface carries a stamp yet. Treating
    unverified as degraded would cap essentially every verdict, and a signal that
    fires on everything is worthless within a day."""
    summary = prov.summarize_tool_context(
        _ctx(_block("a", "{'x': 1}"), _block("b", "{'y': 2}")), _NOW
    )

    assert summary["counts"]["unverified"] == 2
    assert summary["degraded"] is False


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_every_tool_call_in_the_turn_is_accounted_for():
    summary = prov.summarize_tool_context(
        _ctx(
            _block("get_quote", _stamped(1)),
            _block("get_insider_trades", "{'status': 'unavailable', 'source': 'FMP', 'reason': 'no key'}"),
            _block("get_news", "{'articles': []}"),
        ),
        _NOW,
    )

    assert summary["counts"]["total"] == 3
    assert [s["tool"] for s in summary["sources"]] == ["get_quote", "get_insider_trades", "get_news"]
    # Exhaustive on purpose: the statuses must partition the turn, so a new
    # bucket has to be added here deliberately rather than slipping in. 6.2's
    # `substituted` is the one non-status count — a recovered call still carries
    # a status of its own, which is why it does not partition with the others.
    assert summary["counts"] == {
        "total": 3, "unavailable": 1, "stale": 0, "fresh": 1, "unverified": 1,
        "substituted": 0,
    }


def test_no_tool_calls_produces_no_footer():
    summary = prov.summarize_tool_context("No tool calls executed in recent context.", _NOW)

    assert summary["counts"]["total"] == 0
    assert summary["footer"] == ""


def test_a_repeated_unavailable_source_is_named_once():
    payload = "{'status': 'unavailable', 'source': 'FMP', 'reason': 'no key'}"
    summary = prov.summarize_tool_context(
        _ctx(_block("get_insider_trades", payload), _block("get_price_targets", payload)), _NOW
    )

    assert summary["footer"].count("FMP unavailable") == 1
