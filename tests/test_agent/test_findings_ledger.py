"""
6.4 — the cross-specialist findings ledger (`agent/findings`).

What this replaced, in DeepReasoning's risk pre-screen:

    if 'top_picks' in msg_content and 'score' in msg_content:
        for match in re.findall(r"'symbol':\\s*'([A-Z]{1,5})'", msg_content):

A Python ``repr`` parsed with a regex, feeding the headwind screen that
DeepReasoning is instructed to address every flag from. Its four failure modes
are each a test below, because all four fail SILENTLY — no tickers matched means
the pre-screen does not run and nothing says so.

The other half is turn-stamping. ``data_context`` has no state reducer, so a
finding written on an earlier turn is indistinguishable by content from a fresh
one — the same trap 2.3's evidence union hit when the deep path's publication
kept losing to a stale key.
"""
from agent.findings import (
    MAX_FINDINGS,
    extract_tool_findings,
    findings_status,
    findings_symbols,
    make_finding,
    publish_findings,
    read_findings,
)

TURN = "turn-abc"
OTHER_TURN = "turn-xyz"


def _scan(picks):
    return {"sector": "All", "top_picks": picks, "summary": "..."}


def _pick(symbol, score=88, conviction="High", risk_flag=None, theme="AI"):
    p = {"symbol": symbol, "price": 100.0, "score": score,
         "conviction": conviction, "theme": theme, "entry_stage": "base"}
    if risk_flag:
        p["risk_flag"] = risk_flag
    return p


# ---------------------------------------------------------------------------
# The four things the regex got wrong
# ---------------------------------------------------------------------------
def test_suffixed_and_share_class_tickers_survive():
    """`[A-Z]{1,5}` could not match either, so the whole Canadian and
    share-class side of the book was invisible to the headwind screen."""
    found = extract_tool_findings("scan_opportunities",
                                  _scan([_pick("SHOP.TO"), _pick("BRK.B"), _pick("NVDA")]))
    assert findings_symbols([f for f in found if f["kind"] == "candidate"]) == \
        ["SHOP.TO", "BRK.B", "NVDA"]


def test_extraction_does_not_depend_on_how_the_payload_would_be_rendered():
    """The regex needed single quotes. This reads the object, so JSON, repr and
    any other rendering are all the same input."""
    import json

    picks = [_pick("AAPL")]
    from_obj = extract_tool_findings("scan_opportunities", _scan(picks))
    from_roundtrip = extract_tool_findings(
        "scan_opportunities", json.loads(json.dumps(_scan(picks))))
    assert findings_symbols(from_obj) == findings_symbols(from_roundtrip) == ["AAPL"]


def test_a_scan_that_picked_nothing_is_distinguishable_from_an_unparsed_one():
    """The regex returned [] for both, and the caller could not tell them apart."""
    empty = extract_tool_findings("scan_opportunities", _scan([]))
    unregistered = extract_tool_findings("some_other_tool", _scan([_pick("AAPL")]))
    assert empty == [] and unregistered == []
    # The distinction is made by findings_status, not by the extractor.
    ctx_ran = publish_findings({}, [make_finding("observation", "OpportunityScanner",
                                                 "scan completed, no picks")], TURN)
    assert findings_status(ctx_ran, TURN)["status"] == "ready"
    assert findings_status({}, TURN)["status"] == "no_producer"


def test_numbers_come_across_as_numbers():
    """The regex recovered a symbol and dropped everything else, so score and
    conviction had to be re-read from prose downstream."""
    found = extract_tool_findings("scan_opportunities",
                                  _scan([_pick("NVDA", score=91, conviction="High")]))
    payload = found[0]["payload"]
    assert payload["score"] == 91
    assert payload["conviction"] == "High"
    assert payload["symbol"] == "NVDA"


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------
def test_a_surfaced_risk_flag_becomes_its_own_finding():
    """DeepReasoning is instructed to address every scanner risk_flag; making it
    re-read the pick list to find them is how one gets missed."""
    found = extract_tool_findings(
        "scan_opportunities",
        _scan([_pick("AAPL"), _pick("XYZ", risk_flag="Insider selling cluster")]))
    flags = [f for f in found if f["kind"] == "risk_flag"]
    assert len(flags) == 1
    assert flags[0]["symbols"] == ["XYZ"]
    assert "Insider selling" in flags[0]["summary"]


def test_a_pick_without_a_symbol_is_skipped_not_recorded_as_blank():
    found = extract_tool_findings("scan_opportunities",
                                  _scan([{"score": 90}, _pick("AAPL")]))
    assert findings_symbols(found) == ["AAPL"]


def test_a_producer_that_raises_does_not_take_the_tool_result_with_it():
    """This runs inside the tool-recording path: throwing here would lose the
    result it was describing, which is worse than the regex it replaces."""
    assert extract_tool_findings("scan_opportunities", {"top_picks": "not a list"}) == []
    assert extract_tool_findings("scan_opportunities", None) == []
    assert extract_tool_findings(None, _scan([_pick("AAPL")])) == []


def test_holdings_verification_publishes_both_sides():
    found = extract_tool_findings("verify_portfolio_holdings",
                                  {"held": ["AAPL", "VOO"], "not_held": ["TSLA"]})
    by_held = {f["payload"]["held"]: f for f in found}
    assert by_held[True]["symbols"] == ["AAPL", "VOO"]
    assert by_held[False]["symbols"] == ["TSLA"]
    assert "NOT held" in by_held[False]["summary"]


# ---------------------------------------------------------------------------
# Turn-stamping
# ---------------------------------------------------------------------------
def test_a_finding_from_another_turn_is_not_read_as_this_turn_s():
    ctx = publish_findings({}, [make_finding("candidate", "S", "old", ["OLD"])], OTHER_TURN)
    ctx = publish_findings(ctx, [make_finding("candidate", "S", "new", ["NEW"])], TURN)

    assert findings_symbols(read_findings(ctx, turn_key=TURN)) == ["NEW"]
    assert findings_symbols(read_findings(ctx, all_turns=True)) == ["OLD", "NEW"]


def test_an_unidentifiable_turn_reads_nothing_rather_than_someone_elses():
    """current_turn_key returns "" when there is no user message to key off, and
    an empty key must never match."""
    ctx = publish_findings({}, [make_finding("candidate", "S", "x", ["AAPL"])], TURN)
    assert read_findings(ctx, turn_key="") == []


def test_publishing_does_not_mutate_the_context_it_was_given():
    """data_context has no reducer; mutating in place writes into shared state."""
    original = {"tool_execution_context": "evidence"}
    published = publish_findings(original, [make_finding("candidate", "S", "x")], TURN)
    assert "findings" not in original
    assert published["tool_execution_context"] == "evidence"


def test_findings_survive_alongside_other_context_keys():
    ctx = {"tool_execution_context": "e", "tool_execution_turn": TURN}
    ctx = publish_findings(ctx, [make_finding("candidate", "S", "x", ["AAPL"])], TURN)
    assert ctx["tool_execution_turn"] == TURN
    assert len(read_findings(ctx, turn_key=TURN)) == 1


def test_the_reader_accepts_state_or_a_bare_context():
    """Callers hold one or the other; making them unwrap it is how a reader looks
    in the wrong place and concludes there is nothing there."""
    ctx = publish_findings({}, [make_finding("candidate", "S", "x", ["AAPL"])], TURN)
    assert read_findings(ctx, turn_key=TURN)
    assert read_findings({"data_context": ctx}, turn_key=TURN)


def test_the_ledger_is_capped():
    """It rides in checkpointed graph state; unbounded growth bloats every later
    turn's payload."""
    many = [make_finding("candidate", "S", str(i), [f"S{i}"]) for i in range(MAX_FINDINGS + 50)]
    ctx = publish_findings({}, many, TURN)
    rows = read_findings(ctx, turn_key=TURN)
    assert len(rows) == MAX_FINDINGS
    # Oldest-first trim: the most recent survive.
    assert rows[-1]["symbols"] == [f"S{MAX_FINDINGS + 49}"]


# ---------------------------------------------------------------------------
# Filtering and status
# ---------------------------------------------------------------------------
def test_findings_filter_by_kind_and_source():
    ctx = publish_findings({}, [
        make_finding("candidate", "OpportunityScanner", "a", ["AAPL"]),
        make_finding("risk_flag", "OpportunityScanner", "b", ["AAPL"]),
        make_finding("candidate", "NewsAnalyst", "c", ["MSFT"]),
    ], TURN)
    assert len(read_findings(ctx, TURN, kind="candidate")) == 2
    assert len(read_findings(ctx, TURN, source="NewsAnalyst")) == 1
    assert len(read_findings(ctx, TURN, kind="candidate", source="NewsAnalyst")) == 1


def test_symbol_order_is_producer_rank_not_set_order():
    """The scanner's first pick is its best one, and a caller takes the top five."""
    found = extract_tool_findings(
        "scan_opportunities",
        _scan([_pick("AAA", 99), _pick("BBB", 88), _pick("CCC", 77)]))
    assert findings_symbols(found, limit=2) == ["AAA", "BBB"]


def test_duplicate_symbols_collapse_once():
    found = extract_tool_findings(
        "scan_opportunities",
        _scan([_pick("AAPL", risk_flag="crowded"), _pick("MSFT")]))
    # AAPL appears as a candidate AND a risk_flag; it is one symbol.
    assert findings_symbols(found) == ["AAPL", "MSFT"]


def test_empty_and_no_producer_are_different_answers():
    """A consumer that renders both as "no risks found" asserts a clean bill of
    health it never received."""
    never = findings_status({}, TURN)
    assert never["status"] == "no_producer"
    assert "not a statement that nothing was found" in never["note"]

    other_turn_only = publish_findings({}, [make_finding("candidate", "S", "x")], OTHER_TURN)
    stale = findings_status(other_turn_only, TURN)
    assert stale["status"] == "empty"
    assert "none from this one" in stale["note"]


def test_an_unknown_kind_falls_back_rather_than_widening_the_set():
    """An open kind set becomes a second regex problem — consumers go back to
    substring-matching `kind`."""
    assert make_finding("whatever", "S", "x")["kind"] == "observation"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
def test_deep_reasoning_reads_the_ledger_before_scraping():
    """The scrape survives as a LAST RESORT that logs — keeping it silent is what
    made its failures invisible."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "agent" / "nodes" / "deep_reasoning.py")
    text = src.read_text(encoding="utf-8")

    # The EXECUTABLE occurrence, not the comment above it that quotes the old
    # pattern in order to explain what it got wrong.
    ledger_at = text.index("candidates = read_findings(")
    scrape_at = text.index("for match in re.findall(r\"'symbol':")
    assert ledger_at < scrape_at, "the ledger must be consulted before the scrape"
    # And exactly one executable scrape survives — a second would be a path that
    # never learned about the ledger.
    assert text.count("for match in re.findall(r\"'symbol':") == 1
    assert "risk pre-screen fell back to scraping message text" in text
    assert "publish_findings(" in text
