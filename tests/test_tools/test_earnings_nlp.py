"""Earnings-call tone analysis (Advisor Roadmap 5.4).

This module had NO tests, and that is how it shipped a tone verdict for calls it
never read: `get_earnings_transcript` returns *"⚠️ API Limit Reached … here is a
web summary instead"* on a rate limit — truthy, and containing no "Error" — so
the old guard passed it straight into the word counter, which found fewer than
five sentiment words and returned "Neutral". Measured live against MSFT on
2026-07-28.

So the first test here drives the REAL producer's REAL failure path into the
REAL consumer. A test that hand-wrote the fallback string would keep passing
after that string changed, while production quietly went back to scoring
rate-limit notices.
"""
import pytest

import tools.earnings_nlp as nlp
from tools.fmp_api import TRANSCRIPT_HEADER, is_real_transcript, parse_transcript_period
from tools.tool_errors import is_unavailable

_BODY = (
    "We delivered strong growth with record revenue and improving margins. "
    "Momentum accelerated across every segment and we are confident in the outlook. "
) * 40


def _transcript(body: str = _BODY, quarter: int = 2, year: int = 2026) -> str:
    return f"{TRANSCRIPT_HEADER} MSFT (Q{quarter} {year})\n**Date:** 2026-04-25\n\n{body}"


# ---------------------------------------------------------------------------
# THE regression: a call that was never read is never a tone reading
# ---------------------------------------------------------------------------

def test_a_rate_limited_fetch_is_unavailable_not_neutral(monkeypatch):
    """Drives the real fallback through the real fetcher.

    `_fmp_get` fails, so `get_earnings_transcript` builds its own fallback
    string, and the tone analyser must refuse it. This is the exact production
    path that returned 'Neutral' for MSFT.
    """
    import tools.fmp_api as fmp

    monkeypatch.setattr(fmp, "_fmp_get", lambda *a, **k: (None, "429 rate limited"))
    monkeypatch.setattr("tools.web_search.search_news", lambda *a, **k: "Some web summary text.")

    payload = fmp.get_earnings_transcript.__wrapped__.__wrapped__("MSFT")

    # The producer's own failure output, not a string this test invented.
    assert not is_real_transcript(payload)
    assert "Error" not in payload, "the old guard's assumption — kept as a live check"

    monkeypatch.setattr(nlp, "get_earnings_transcript", lambda *a, **k: payload)
    result = nlp.analyze_management_tone.__wrapped__.__wrapped__("MSFT")

    assert is_unavailable(result)
    # No tone key at all — the caller cannot accidentally read a verdict off it.
    assert "tone_status" not in result


def test_an_empty_transcript_db_response_is_also_unavailable(monkeypatch):
    """The other fallback branch: FMP answers, with no filings."""
    import tools.fmp_api as fmp

    monkeypatch.setattr(fmp, "_fmp_get", lambda *a, **k: ([], None))
    monkeypatch.setattr("tools.web_search.search_news", lambda *a, **k: "Web summary.")

    payload = fmp.get_earnings_transcript.__wrapped__.__wrapped__("MSFT")
    monkeypatch.setattr(nlp, "get_earnings_transcript", lambda *a, **k: payload)

    assert is_unavailable(nlp.analyze_management_tone.__wrapped__.__wrapped__("MSFT"))


def test_the_unavailable_payload_names_the_source_and_the_reason(monkeypatch):
    """So the turn-provenance summary can count it and say which feed died."""
    monkeypatch.setattr(nlp, "get_earnings_transcript", lambda *a, **k: "⚠️ API Limit Reached")

    result = nlp.analyze_management_tone.__wrapped__.__wrapped__("NVDA")

    assert result["source"] == "FMP earnings transcript"
    assert "NVDA" in result["reason"]


def test_a_short_transcript_is_unavailable_rather_than_neutral(monkeypatch):
    """"Too short to characterise" is not "balanced"."""
    monkeypatch.setattr(nlp, "get_earnings_transcript", lambda *a, **k: _transcript("Thanks everyone."))

    assert is_unavailable(nlp.analyze_management_tone.__wrapped__.__wrapped__("MSFT"))


def test_neutral_is_reserved_for_a_call_that_was_actually_read(monkeypatch):
    """The complement, and the reason the bug was invisible: Neutral must still
    be reachable, but only from a real transcript."""
    balanced = ("The quarter showed growth and improving demand. "
                "We also saw pressure and softness in some regions, with weakness in pricing. ") * 30
    monkeypatch.setattr(nlp, "get_earnings_transcript", lambda *a, **k: _transcript(balanced))

    result = nlp.analyze_management_tone.__wrapped__.__wrapped__("MSFT")

    assert not is_unavailable(result)
    assert result["tone_status"] == "Neutral"
    assert result["transcript_words"] > nlp.MIN_SCORABLE_WORDS


def test_it_no_longer_predicts_what_the_tone_precedes(monkeypatch):
    """The old strings asserted confident language 'historically precedes upward
    earnings revisions' and cautious language means 'high risk of a guidance
    cut'. Neither was ever measured here."""
    monkeypatch.setattr(nlp, "get_earnings_transcript", lambda *a, **k: _transcript())

    reading = nlp.analyze_management_tone.__wrapped__.__wrapped__("MSFT")["interpretation"].lower()

    for claim in ("historically", "precedes", "guidance cut", "setting the stage"):
        assert claim not in reading


# ---------------------------------------------------------------------------
# Transcript identification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    "⚠️ API Limit Reached for full transcript. Here is a web summary instead:\n\nstuff",
    "⚠️ No transcript found in DB. Here is a web summary instead:\n\nstuff",
    "", None, {}, [], 42,
])
def test_only_a_real_transcript_passes_identification(payload):
    """Positive detection: an unrecognised payload is 'not a transcript', never
    'a transcript with nothing in it'."""
    assert not is_real_transcript(payload)


def test_a_real_transcript_is_identified_and_dated():
    payload = _transcript(quarter=3, year=2025)

    assert is_real_transcript(payload)
    assert parse_transcript_period(payload) == (2025, 3)


def test_a_fallback_payload_has_no_period():
    assert parse_transcript_period("⚠️ API Limit Reached") is None


def test_our_own_header_is_not_scored_as_management_speech():
    """The header is this codebase's metadata. Leaving it in adds a fixed token
    count to every transcript, which dilutes per-1,000-word rates more in a short
    call than a long one — a length dependency inside a comparison built not to
    have one."""
    from tools.fmp_api import transcript_body

    body = transcript_body(_transcript("We saw growth."))

    assert body.strip() == "We saw growth."
    assert "Earnings Call Transcript" not in body
    assert transcript_body("⚠️ API Limit Reached") == ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_categories_are_counted_separately():
    scored = nlp.score_text(
        "growth momentum strong. headwinds weakness decline. "
        "uncertain perhaps possibly. lawsuit litigate court. covenant obligation restricted."
    )

    assert scored["counts"]["positive"] >= 3
    assert scored["counts"]["negative"] >= 3
    assert scored["counts"]["uncertainty"] >= 3
    assert scored["counts"]["litigious"] >= 3
    assert scored["counts"]["constraining"] >= 3


def test_rates_are_per_thousand_words():
    scored = nlp.score_text("growth " + "filler " * 999)

    assert scored["total_words"] == 1000
    assert scored["per_1k"]["positive"] == pytest.approx(1.0)


def test_the_lexicon_is_declared_as_a_subset():
    """Absolute counts are lower bounds, and the module has to say so — claiming
    the full Loughran-McDonald dictionary would be the overclaim."""
    assert "subset" in nlp.LEXICON_NOTE
    assert "lower bounds" in nlp.LEXICON_NOTE


# ---------------------------------------------------------------------------
# The quarter-over-quarter delta — 5.4's actual signal
# ---------------------------------------------------------------------------

def _two_quarters(monkeypatch, latest_body, prior_body, quarter=2, year=2026):
    def fetch(symbol, year=None, quarter=None):
        if year is None:
            return _transcript(latest_body, quarter=2, year=2026)
        return _transcript(prior_body, quarter=quarter, year=year)

    monkeypatch.setattr(nlp, "get_earnings_transcript", fetch)


def test_the_delta_is_length_invariant(monkeypatch):
    """A longer call is not a more negative one. Raw counts would say otherwise;
    per-1,000-word rates must not."""
    # Both comfortably over MIN_SCORABLE_WORDS; the point is that one is 3x the
    # other with the same word MIX.
    body = "headwinds pressure weakness filler filler filler filler filler "
    _two_quarters(monkeypatch, body * 120, body * 40)

    result = nlp.compare_management_tone.__wrapped__.__wrapped__("MSFT")

    assert not is_unavailable(result)
    assert result["delta_per_1k"]["negative"] == pytest.approx(0.0, abs=0.01)
    assert result["material_shifts"] == []


def test_a_real_tone_shift_is_reported_with_its_direction(monkeypatch):
    confident = "growth momentum strong confident record improving " * 40
    cautious = "headwinds weakness decline pressure softness slowdown " * 40
    _two_quarters(monkeypatch, cautious, confident)

    result = nlp.compare_management_tone.__wrapped__.__wrapped__("MSFT")
    categories = {s["category"] for s in result["material_shifts"]}

    assert "negative" in categories and "positive" in categories
    assert result["delta_per_1k"]["negative"] > 0
    assert result["delta_per_1k"]["positive"] < 0
    assert "more cautious language" in result["interpretation"]


def test_no_material_shift_is_stated_as_a_result(monkeypatch):
    """A quiet quarter must not look like a dead signal."""
    body = "growth momentum strong improving demand " * 40
    _two_quarters(monkeypatch, body, body)

    result = nlp.compare_management_tone.__wrapped__.__wrapped__("MSFT")

    assert result["material_shifts"] == []
    assert "No material change" in result["interpretation"]
    assert "read and compared" in result["interpretation"]


def test_the_prior_quarter_rolls_back_across_the_year(monkeypatch):
    captured = {}

    def fetch(symbol, year=None, quarter=None):
        if year is None:
            return _transcript(quarter=1, year=2026)
        captured["asked"] = (year, quarter)
        return _transcript(quarter=quarter, year=year)

    monkeypatch.setattr(nlp, "get_earnings_transcript", fetch)
    nlp.compare_management_tone.__wrapped__.__wrapped__("MSFT")

    assert captured["asked"] == (2025, 4)


def test_a_missing_prior_quarter_yields_unavailable_not_a_one_sided_delta(monkeypatch):
    """A delta against a quarter that could not be fetched is a different number
    wearing the same name."""
    def fetch(symbol, year=None, quarter=None):
        return _transcript() if year is None else "⚠️ API Limit Reached"

    monkeypatch.setattr(nlp, "get_earnings_transcript", fetch)

    result = nlp.compare_management_tone.__wrapped__.__wrapped__("MSFT")

    assert is_unavailable(result)
    assert "quarter-over-quarter" in result["reason"]


def test_an_undated_transcript_is_not_guessed_at(monkeypatch):
    """Assuming the calendar quarter would silently compare a company's Q2
    against its own Q2 whenever reporting lags."""
    monkeypatch.setattr(
        nlp, "get_earnings_transcript",
        lambda *a, **k: f"{TRANSCRIPT_HEADER} MSFT (no period)\n\n{_BODY}",
    )

    result = nlp.compare_management_tone.__wrapped__.__wrapped__("MSFT")

    assert is_unavailable(result)
    assert "quarter" in result["reason"]


def test_the_delta_is_marked_as_measured(monkeypatch):
    """2.7: every figure here was counted from two transcripts actually read."""
    body = "growth momentum strong improving " * 50
    _two_quarters(monkeypatch, body, body)

    assert nlp.compare_management_tone.__wrapped__.__wrapped__("MSFT")["basis"] == "measured"
