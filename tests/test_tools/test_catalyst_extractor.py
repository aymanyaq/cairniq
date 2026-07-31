"""Tests for tools/catalyst_extractor.py — deterministic post-processing.

The LLM step is mocked via the injectable `classifier`, so these exercise the
real threshold / relevance / dedup / lane-routing / auto-escalation logic without
a live model. See docs/technical/CATALYST_ENGINE_SPEC.md.
"""

from tools import catalyst_extractor as ce


def _cat(**overrides):
    """A valid high-materiality catalyst dict; override fields per-test."""
    base = {
        "headline": "Explosion at XYZ refinery halts output",
        "source_url": "http://example.com/x",
        "event_type": "outage_disruption",
        "summary": "A refinery fire halted production.",
        "entities": {"tickers": ["XOM"], "sectors": ["Energy"], "commodities": ["oil"]},
        "exposure_basis": "sourced",
        "direction_hint": "bullish",
        "materiality": "high",
        "confidence": 0.9,
        "horizon": "days",
    }
    base.update(overrides)
    return base


# --- JSON parsing -----------------------------------------------------------
def test_parse_json_array_strips_prose_and_fences():
    text = 'Here you go:\n[{"headline": "A", "event_type": "m_and_a"}]\nThanks!'
    out = ce._parse_llm_json_array(text)
    assert len(out) == 1 and out[0]["headline"] == "A"


def test_parse_json_array_bad_input_returns_empty():
    assert ce._parse_llm_json_array("not json at all") == []
    assert ce._parse_llm_json_array("") == []
    assert ce._parse_llm_json_array('{"not": "a list"}') == []


# --- structured extraction (forced tool call) --------------------------------
class _FakeResponse:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


def test_catalysts_from_tool_call_reads_forced_tool():
    resp = _FakeResponse([{"name": "submit_catalysts", "args": {"catalysts": [{"headline": "A"}]}}])
    assert ce._catalysts_from_tool_call(resp) == [{"headline": "A"}]


def test_catalysts_from_tool_call_empty_list_is_explicit_no_catalysts():
    # {"catalysts": []} is a real "quiet news day" — NOT a parse failure (None).
    resp = _FakeResponse([{"name": "submit_catalysts", "args": {"catalysts": []}}])
    assert ce._catalysts_from_tool_call(resp) == []


def test_catalysts_from_tool_call_none_when_unusable():
    assert ce._catalysts_from_tool_call(_FakeResponse([])) is None
    assert ce._catalysts_from_tool_call(_FakeResponse([{"name": "other_tool", "args": {}}])) is None
    assert ce._catalysts_from_tool_call(
        _FakeResponse([{"name": "submit_catalysts", "args": {"catalysts": "not a list"}}])
    ) is None
    assert ce._catalysts_from_tool_call(object()) is None  # no tool_calls attr at all


def test_submit_catalysts_tool_schema_shape():
    params = ce._SUBMIT_CATALYSTS_TOOL["parameters"]
    assert params["required"] == ["catalysts"]
    item_props = params["properties"]["catalysts"]["items"]["properties"]
    # Schema and prompt must agree on the fields downstream normalization expects.
    for field in ("headline", "event_type", "entities", "event_date", "materiality", "confidence", "horizon"):
        assert field in item_props
    assert item_props["event_type"]["enum"] == sorted(ce._VALID_EVENT_TYPES)


# --- normalization ----------------------------------------------------------
def test_normalize_clamps_and_defaults():
    c = ce.normalize_catalyst({
        "headline": "Deal announced",
        "event_type": "not_a_type",      # -> other
        "materiality": "bogus",          # -> medium
        "confidence": 5,                 # -> clamped to 1.0
        "entities": {"tickers": "aapl"}, # str -> ["AAPL"]
    })
    assert c["event_type"] == "other"
    assert c["materiality"] == "medium"
    assert c["confidence"] == 1.0
    assert c["entities"]["tickers"] == ["AAPL"]


def test_normalize_rejects_headless_catalyst():
    assert ce.normalize_catalyst({"headline": "  "}) is None
    assert ce.normalize_catalyst("not a dict") is None


def test_normalize_event_date_validation():
    assert ce.normalize_catalyst(_cat(event_date="2026-06-09"))["event_date"] == "2026-06-09"
    # ISO datetime → date part kept; garbage / missing → None
    assert ce.normalize_catalyst(_cat(event_date="2026-06-09T14:00:00"))["event_date"] == "2026-06-09"
    assert ce.normalize_catalyst(_cat(event_date="last Tuesday"))["event_date"] is None
    assert ce.normalize_catalyst(_cat())["event_date"] is None


# --- threshold --------------------------------------------------------------
def test_threshold_drops_low_materiality_and_low_confidence():
    cats = [
        _cat(materiality="high", confidence=0.9),   # keep
        _cat(materiality="low", confidence=0.9),    # drop (low materiality)
        _cat(materiality="medium", confidence=0.4), # drop (low confidence)
        _cat(materiality="medium", confidence=0.5), # keep (boundary)
    ]
    assert len(ce.threshold(cats)) == 2


# --- relevance (deterministic, no LLM knows holdings) -----------------------
def test_classify_relevance_tiers():
    held = {"XOM"}
    watch = {"CVX"}
    assert ce.classify_relevance({"tickers": ["XOM"]}, held, watch) == "held"
    assert ce.classify_relevance({"tickers": ["CVX"]}, held, watch) == "watchlist"
    assert ce.classify_relevance({"tickers": ["BP"], "sectors": ["Energy"]}, held, watch) == "sector"
    assert ce.classify_relevance({"tickers": ["BP"], "sectors": []}, held, watch) == "none"


def test_classify_relevance_normalizes_exchange_suffix():
    # Headline names the bare ticker; holding carries a TSX suffix — must still match held.
    assert ce.classify_relevance({"tickers": ["SHOP"]}, {"SHOP.TO"}, set()) == "held"
    # And the reverse: suffixed headline ticker vs a bare holding.
    assert ce.classify_relevance({"tickers": ["RY.TO"]}, {"RY"}, set()) == "held"
    # Watchlist tier normalizes too (TSXV .V suffix).
    assert ce.classify_relevance({"tickers": ["ABC"]}, set(), {"ABC.V"}) == "watchlist"


def test_classify_relevance_preserves_us_share_class():
    # BRK.B is a US share class, not an exchange suffix — must NOT collapse onto BRK.
    assert ce.classify_relevance({"tickers": ["BRK.B"]}, {"BRK"}, set()) == "none"


def test_normalize_ticker_unit():
    assert ce._normalize_ticker("shop.to") == "SHOP"   # case-folded + suffix stripped
    assert ce._normalize_ticker("RY.TO") == "RY"
    assert ce._normalize_ticker("AAPL") == "AAPL"       # no dot, unchanged
    assert ce._normalize_ticker("BRK.B") == "BRK.B"     # share class preserved
    assert ce._normalize_ticker("T") == "T"             # AT&T: no dot, never stripped


# --- dedup / novelty --------------------------------------------------------
def test_dedup_marks_seen_as_duplicate():
    c = _cat()
    cid = ce.catalyst_id(c)
    out = ce.apply_dedup([c], seen_ids={cid})
    assert out[0]["novelty"] == "duplicate"
    out2 = ce.apply_dedup([_cat()], seen_ids=set())
    assert out2[0]["novelty"] == "new"


def test_catalyst_id_stable_and_ticker_order_independent():
    a = _cat(entities={"tickers": ["AAPL", "MSFT"], "sectors": [], "commodities": []})
    b = _cat(entities={"tickers": ["MSFT", "AAPL"], "sectors": [], "commodities": []})
    assert ce.catalyst_id(a) == ce.catalyst_id(b)


def test_catalyst_id_rephrase_resistant():
    # Same event, two outlets' wordings → same id (the second wording must not
    # re-enter as 'new' and re-bill auto-escalation).
    a = _cat(headline="Explosion halts output at XYZ refinery")
    b = _cat(headline="XYZ refinery output halted after explosion")
    assert ce.catalyst_id(a) == ce.catalyst_id(b)
    # Genuinely different event on the same ticker/type → different id.
    c = _cat(headline="XYZ refinery announces record quarterly production")
    assert ce.catalyst_id(a) != ce.catalyst_id(c)


def test_stale_event_detection():
    from datetime import datetime
    now = datetime(2026, 6, 10)
    assert ce.is_stale_event("2026-06-01", now) is True       # 9 days old
    assert ce.is_stale_event("2026-06-09", now) is False      # yesterday
    assert ce.is_stale_event(None, now) is False              # undated = benefit of the doubt
    assert ce.is_stale_event("not-a-date", now) is False      # malformed = undated
    assert ce.is_stale_event("2026-06-01", now, stale_after_days=30) is False  # tunable


def test_stale_catalyst_displays_but_never_escalates():
    from datetime import datetime
    now = datetime(2026, 6, 10)
    raw = [
        _cat(headline="Old refinery fire resurfaces", event_date="2026-06-02",
             materiality="high", confidence=0.95),
        _cat(headline="Fresh megadeal announced today", event_date="2026-06-10",
             event_type="m_and_a", materiality="high", confidence=0.95),
    ]
    result = ce.extract_catalysts({"h": "x"}, holdings=["XOM"], now=now, classifier=lambda _o: raw)
    assert len(result["catalysts"]) == 2                      # both still shown
    stale_flags = {c["headline"]: c["stale"] for c in result["catalysts"]}
    assert stale_flags["Old refinery fire resurfaces"] is True
    assert stale_flags["Fresh megadeal announced today"] is False
    # Only the fresh one bills the scenario engine.
    assert [c["headline"] for c in result["auto_escalate"]] == ["Fresh megadeal announced today"]


# --- lane routing -----------------------------------------------------------
def test_route_lanes_splits_and_ranks():
    cats = [
        _cat(portfolio_relevance="held", confidence=0.7, materiality="high"),
        _cat(portfolio_relevance="none", confidence=0.95, materiality="high"),
        _cat(portfolio_relevance="sector", confidence=0.6, materiality="medium"),
        _cat(portfolio_relevance="watchlist", confidence=0.9, materiality="high"),
    ]
    lanes = ce.route_lanes(cats)
    assert {c["portfolio_relevance"] for c in lanes["portfolio_impact"]} == {"held", "watchlist"}
    assert {c["portfolio_relevance"] for c in lanes["opportunity"]} == {"none", "sector"}
    # opportunity lane ranked: high-materiality 0.95 before medium 0.6
    assert lanes["opportunity"][0]["confidence"] == 0.95


# --- auto-escalation (cost-bounded) -----------------------------------------
def test_auto_escalation_eligibility_and_priority():
    cats = [
        _cat(id="op", portfolio_relevance="none", novelty="new", materiality="high", confidence=0.99),
        _cat(id="held", portfolio_relevance="held", novelty="new", materiality="high", confidence=0.85),
        _cat(id="lowconf", portfolio_relevance="held", novelty="new", materiality="high", confidence=0.7),  # below 0.8
        _cat(id="med", portfolio_relevance="held", novelty="new", materiality="medium", confidence=0.95),   # not high
        _cat(id="dup", portfolio_relevance="held", novelty="duplicate", materiality="high", confidence=0.95),  # not new
    ]
    picked = ce.select_for_auto_escalation(cats, cap=3)
    ids = [c["id"] for c in picked]
    # held (portfolio) prioritized over opportunity despite lower confidence
    assert ids == ["held", "op"]


def test_auto_escalation_respects_cap_and_prior_escalations():
    cats = [
        _cat(id=f"c{i}", portfolio_relevance="held", novelty="new", materiality="high", confidence=0.9)
        for i in range(5)
    ]
    assert len(ce.select_for_auto_escalation(cats, cap=2)) == 2
    # already-escalated ids are excluded
    picked = ce.select_for_auto_escalation(cats, cap=5, already_escalated={"c0", "c1", "c2"})
    assert {c["id"] for c in picked} == {"c3", "c4"}


# --- orchestrator end-to-end (mocked classifier) ----------------------------
def test_extract_catalysts_end_to_end_with_mock_classifier():
    raw = [
        _cat(headline="Refinery fire", entities={"tickers": ["XOM"], "sectors": ["Energy"], "commodities": []},
             materiality="high", confidence=0.9),                       # held → portfolio lane + escalate
        _cat(headline="Startup IPO buzz", entities={"tickers": ["NEWCO"], "sectors": [], "commodities": []},
             materiality="low", confidence=0.9),                        # dropped (low materiality)
        _cat(headline="Rival lands megadeal", entities={"tickers": ["RIVL"], "sectors": ["Tech"], "commodities": []},
             event_type="m_and_a", materiality="high", confidence=0.95), # not owned → opportunity lane + escalate
    ]
    result = ce.extract_catalysts(
        tool_outputs={"get_market_headlines": "..."},
        holdings=["xom"],            # lowercase on purpose → normalized
        watchlist=[],
        classifier=lambda _outputs: raw,
    )
    assert len(result["catalysts"]) == 2  # low-materiality one dropped
    assert len(result["lanes"]["portfolio_impact"]) == 1
    assert result["lanes"]["portfolio_impact"][0]["entities"]["tickers"] == ["XOM"]
    assert len(result["lanes"]["opportunity"]) == 1
    # both surviving high-conviction catalysts auto-escalate, portfolio first
    assert [c["entities"]["tickers"][0] for c in result["auto_escalate"]] == ["XOM", "RIVL"]


def test_extract_catalysts_swallows_empty_classifier():
    result = ce.extract_catalysts({}, holdings=["AAPL"], classifier=lambda _o: [])
    assert result["catalysts"] == []
    assert result["auto_escalate"] == []


# --- dedup log round-trip ---------------------------------------------------
def test_dedup_log_round_trip(tmp_path):
    log_dir = str(tmp_path / "catalyst_log")
    cats = [{"id": "abc", "headline": "x"}, {"id": "def", "headline": "y"}]
    ce.record_catalyst_ids(cats, escalated_ids=["abc"], log_dir=log_dir)
    seen = ce.load_seen_ids(log_dir=log_dir)
    assert seen == {"abc", "def"}
    # The escalation set reads back too (spec §3.6: a recurring story never re-bills).
    assert ce.load_escalated_ids(log_dir=log_dir) == {"abc"}


def test_load_escalated_ids_missing_dir_is_empty():
    assert ce.load_escalated_ids(log_dir="/nonexistent/path/xyz") == set()


# --- escalation passthrough (already_escalated + cap) -------------------------
def test_extract_catalysts_excludes_already_escalated():
    raw = [_cat(headline="Refinery fire", materiality="high", confidence=0.9)]
    cid = ce.catalyst_id(ce.normalize_catalyst(raw[0]))
    result = ce.extract_catalysts(
        {"h": "x"}, holdings=["XOM"],
        already_escalated={cid},
        classifier=lambda _o: raw,
    )
    assert len(result["catalysts"]) == 1        # still displayed
    assert result["auto_escalate"] == []        # but never re-billed


def test_extract_catalysts_escalation_cap_zero_disables():
    raw = [_cat(headline="Refinery fire", materiality="high", confidence=0.9)]
    result = ce.extract_catalysts(
        {"h": "x"}, holdings=["XOM"], escalation_cap=0, classifier=lambda _o: raw,
    )
    assert result["auto_escalate"] == []


# --- escalation settings (funnel_config.json `catalyst` block) ----------------
def test_escalation_settings_defaults_when_config_missing():
    settings = ce.get_escalation_settings(config_path="/nonexistent/funnel_config.json")
    assert settings == ce.DEFAULT_ESCALATION_SETTINGS
    assert settings["auto_escalation_enabled"] is True
    assert settings["max_auto_escalations"] == ce.MAX_AUTO_ESCALATIONS


def test_escalation_settings_partial_block_merges_defaults(tmp_path):
    import json
    cfg = tmp_path / "funnel_config.json"
    cfg.write_text(json.dumps({"catalyst": {"auto_escalation_enabled": False, "max_auto_escalations": 1}}))
    settings = ce.get_escalation_settings(config_path=str(cfg))
    assert settings["auto_escalation_enabled"] is False
    assert settings["max_auto_escalations"] == 1
    # Keys absent from the block keep their defaults.
    assert settings["auto_scan_after_news"] is True
    assert settings["auto_scan_min_interval_hours"] == 6


def test_escalation_settings_malformed_config_falls_back(tmp_path):
    cfg = tmp_path / "funnel_config.json"
    cfg.write_text("{not valid json")
    assert ce.get_escalation_settings(config_path=str(cfg)) == ce.DEFAULT_ESCALATION_SETTINGS


def test_load_seen_ids_missing_dir_is_empty():
    assert ce.load_seen_ids(log_dir="/nonexistent/path/xyz") == set()


def test_record_catalyst_ids_noop_on_empty():
    # No ids → no file created, no error.
    ce.record_catalyst_ids([], log_dir="/nonexistent/path/xyz")


def test_seen_ids_tag_novelty_but_do_not_drop_from_display():
    """A catalyst already in the log is STILL shown on a re-scan (refresh must show the
    current list), but tagged novelty='duplicate' and excluded from auto-escalation."""
    raw = [_cat(headline="Refinery fire", materiality="high", confidence=0.9)]
    cid = ce.catalyst_id(ce.normalize_catalyst(raw[0]))
    result = ce.extract_catalysts(
        {"get_market_headlines": "..."},
        holdings=["XOM"],
        seen_ids={cid},
        classifier=lambda _o: raw,
    )
    assert len(result["catalysts"]) == 1                 # shown, NOT dropped
    assert result["catalysts"][0]["novelty"] == "duplicate"
    assert result["auto_escalate"] == []                 # but not re-escalated


def test_first_scan_then_refresh_keeps_list_non_empty():
    """Reproduces the refresh bug fix end-to-end: scan -> record ids -> re-scan same
    news still returns the catalyst (previously it returned [])."""
    raw = [_cat(headline="Refinery fire", materiality="high", confidence=0.9)]
    r1 = ce.extract_catalysts({"h": "x"}, holdings=["XOM"], seen_ids=set(), classifier=lambda _o: raw)
    seen = {c["id"] for c in r1["catalysts"]}
    r2 = ce.extract_catalysts({"h": "x"}, holdings=["XOM"], seen_ids=seen, classifier=lambda _o: raw)
    assert len(r1["catalysts"]) == 1 and len(r2["catalysts"]) == 1  # refresh no longer empties
