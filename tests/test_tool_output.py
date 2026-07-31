"""Tests for agent.tool_output — reversible tool-result compression.

Covers the behavior the two reasoning nodes rely on: small results pass through,
oversized results keep BOTH head and tail (the old blind head-chop dropped the
tail), and the full original stays retrievable by the id in the marker.
"""
import re

from agent.tool_output import (
    annotate_authored_basis,
    compress_tool_result,
    get_full_tool_result,
)


def test_short_content_passes_through_without_marker():
    text = "AAPL last=150.00 vol=1.2M"
    out = compress_tool_result(text)
    assert out == text
    assert "elided" not in out


def test_repetitive_whitespace_is_collapsed():
    text = "col1" + " " * 50 + "col2" + "\n" * 6 + "row"
    out = compress_tool_result(text)
    assert "  " not in out          # runs of spaces collapsed to one
    assert "\n\n\n" not in out      # blank-line stretches collapsed to one


def test_empty_or_none_returns_empty_string():
    assert compress_tool_result("") == ""
    assert compress_tool_result(None) == ""


def test_non_string_is_coerced():
    out = compress_tool_result({"price": 1})
    assert "price" in out


def test_large_content_keeps_head_and_tail_and_is_reversible():
    head_sentinel = "HEAD_SENTINEL_START"
    tail_sentinel = "TAIL_SENTINEL_END"
    text = head_sentinel + ("x" * 9000) + tail_sentinel

    out = compress_tool_result(text, max_chars=4000)

    # Old blind head-chop would have dropped the tail entirely; we keep both.
    assert head_sentinel in out
    assert tail_sentinel in out
    assert "elided" in out
    # Near the budget, not a multiple of it (marker adds only a little overhead).
    assert len(out) < 4000 + 200

    # Reversible: the id embedded in the marker recovers the full original.
    rid = re.search(r"id=([0-9a-f]+)", out).group(1)
    assert get_full_tool_result(rid) == text


def test_unknown_id_returns_none():
    assert get_full_tool_result("0000nope00") is None


# --- Roadmap 2.7: authored-basis attribution at the model-context seam ---------


def test_authored_basis_payload_gets_the_attribution_directive():
    """A stamped payload must arrive with its instruction, not just its marker.

    Stamping alone was the state 2.7 shipped in and it changed nothing for the
    reader: the marker sat inside a dict the model could skim past.
    """
    payload = str({"scenario": "recession", "impact_pct": -35, "basis": "authored constant"})
    out = annotate_authored_basis(payload)
    assert "[BASIS — AUTHORED" in out
    assert out.startswith(payload)  # the payload itself is never altered


def test_measured_payload_is_untouched():
    payload = str({"episode": "COVID crash", "peak_to_trough_pct": -17.7, "basis": "measured"})
    assert annotate_authored_basis(payload) == payload


def test_prose_about_authored_constants_does_not_trip_the_directive():
    """Only the structured marker fires it — a tool EXPLAINING the concept must not.

    `run_stress_test`'s own docstring and basis_note both contain the phrase, so a
    bare substring match would have annotated results that carry no marker at all.
    """
    prose = "This tool's drop magnitudes are an authored constant, described in the docs."
    assert annotate_authored_basis(prose) == prose


def test_annotation_is_idempotent():
    """Both nodes annotate, and deep reasoning re-feeds its own recorded text to the
    planner on cycle 2 — so a second pass must not stack a second directive."""
    payload = str({"basis": "authored constant"})
    once = annotate_authored_basis(payload)
    assert annotate_authored_basis(once) == once


def test_directive_survives_compaction_because_it_is_appended():
    """The directive is appended, not inserted, precisely so truncation keeps it.

    Both compaction paths in this codebase preserve the tail; a mid-payload note
    would be the first thing elided on a large result.
    """
    big = str({"rows": ["x" * 200] * 60, "basis": "authored constant"})
    out = compress_tool_result(annotate_authored_basis(big), max_chars=500)
    assert "elided" in out
    assert "[BASIS — AUTHORED" in out


def test_json_style_marker_is_recognized():
    """Tool payloads reach the seam as str(dict) OR as json.dumps output."""
    assert "[BASIS — AUTHORED" in annotate_authored_basis('{"basis": "authored constant"}')
