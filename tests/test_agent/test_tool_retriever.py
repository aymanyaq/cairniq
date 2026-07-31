
import pytest

from agent.tool_retriever import ToolRetriever, format_tool_retrieval_status, get_tool_directory_string


@pytest.fixture
def retriever():
    # Force a fresh instance or at least clear cache for testing
    r = ToolRetriever()
    r.query_cache.clear()
    return r

def test_singleton_pattern():
    r1 = ToolRetriever()
    r2 = ToolRetriever()
    assert r1 is r2

def test_keyword_fallback(retriever):
    # Query for "portfolio" should return portfolio tools
    tools = retriever.get_tools_for_query("portfolio analysis", k=5)
    tool_names = [t.name for t in tools]
    # Check if any portfolio tool is present
    assert any("portfolio" in name for name in tool_names)

def test_mandatory_pinning(retriever):
    # Query with "should I buy" should pin deep dive
    tools = retriever.get_tools_for_query("should I buy AAPL", k=5)
    tool_names = [t.name for t in tools]
    assert "run_stock_deep_dive" in tool_names

def test_relationship_expansion(retriever):
    # 'fetch_fundamentals' is related to 'get_fundamentals_detailed'
    # Mocking self.all_tools is hard since it's in __init__, so we test real values
    tools = retriever.get_tools_for_query("fundamentals", k=2)
    # Expansion happens after retrieval
    [t.name for t in tools]
    # If fetch_fundamentals was retrieved, expansion might add more
    # We just verify that the expansion function itself works
    from agent.tool_registry import fetch_fundamentals
    expanded = retriever._expand_with_relationships([fetch_fundamentals], max_expansion=1)
    assert len(expanded) > 1

def test_format_status():
    metadata = {
        "selected_tool_names": ["tool1", "tool2"],
        "elapsed_ms": 10,
        "strategy": "bm25",
        "tool_count": 2
    }
    status = format_tool_retrieval_status(metadata)
    assert "Selected 2 candidates" in status
    assert "via bm25" in status

def test_get_tool_directory_string():
    dir_str = get_tool_directory_string()
    assert "- fetch_fundamentals:" in dir_str
    assert "- plot_chart:" in dir_str

def test_query_cache(retriever):
    query = "market trends"
    # First call
    tools1, meta1 = retriever.get_tools_for_query_with_metadata(query)
    assert meta1.get("cache_hit") is False

    # Second call
    tools2, meta2 = retriever.get_tools_for_query_with_metadata(query)
    assert meta2.get("cache_hit") is True
    assert meta2["elapsed_ms"] == 0
    assert len(tools1) == len(tools2)

def test_tokenize_text():
    from agent.tool_retriever import _tokenize_text
    tokens = _tokenize_text("Portfolio_Risk analysis!")
    assert "portfolio" in tokens
    assert "risk" in tokens
    assert "analysis" in tokens


def test_query_mentions_ticker():
    from agent.tool_retriever import _query_mentions_ticker
    assert _query_mentions_ticker("should I buy AAPL") is True
    assert _query_mentions_ticker("what should I do with my cash") is False
    assert _query_mentions_ticker("should I buy an ETF") is False
    assert _query_mentions_ticker("tell me about US markets") is False


# ---------------------------------------------------------------------------
# TOOL_RELATIONSHIPS — the table itself
#
# Measured 2026-07-27: only 55 of 154 edges had both ends registered. The table
# had been written against the `tools/` implementation functions rather than the
# registered tool names, and _expand_with_relationships skips an unknown name
# without logging, so a shipped feature ran at a third of its apparent reach for
# as long as nobody counted. These assertions are the count, made permanent.
# ---------------------------------------------------------------------------

def test_every_name_in_the_table_is_a_registered_tool():
    from agent.tool_registry import ALL_TOOLS
    from agent.tool_retriever import TOOL_RELATIONSHIPS

    registered = {t.name for t in ALL_TOOLS}
    names = set(TOOL_RELATIONSHIPS) | {v for vs in TOOL_RELATIONSHIPS.values() for v in vs}
    dead = sorted(n for n in names if n not in registered)
    assert not dead, (
        "TOOL_RELATIONSHIPS names tools that are not registered — one-hop "
        f"expansion will skip these silently: {dead}"
    )


def test_no_tool_is_related_to_itself():
    """A self-edge and a duplicate both expand to nothing.

    _expand_with_relationships counts a related tool against max_expansion only
    when it is actually added, so neither shrinks the result — but both make the
    table claim reach it does not have, which is the failure being fixed here.
    """
    from agent.tool_retriever import TOOL_RELATIONSHIPS

    for name, related in TOOL_RELATIONSHIPS.items():
        assert name not in related, f"{name} lists itself as a related tool"
        assert len(set(related)) == len(related), f"{name} has duplicate related tools"


def test_expansion_actually_reaches_every_related_tool(retriever):
    """Drives the real expansion over the real registry, key by key.

    The check above proves the names resolve in ALL_TOOLS; this one proves they
    resolve in the map the retriever itself consults, which is what the live
    `if r_name in self.tool_map` guard silently filtered against.
    """
    from agent.tool_retriever import TOOL_RELATIONSHIPS

    for name, related in TOOL_RELATIONSHIPS.items():
        primary = retriever.tool_map[name]
        expanded = retriever._expand_with_relationships([primary], max_expansion=len(related))
        added = [t.name for t in expanded[1:]]
        assert added == list(related), f"{name} expanded to {added}, expected {list(related)}"

