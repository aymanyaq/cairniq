from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from agent.nodes.market_analyst import _build_portfolio_dashboard, market_analyst_node
from agent.state import AgentState


def test_build_portfolio_dashboard_macro_backdrop():
    """get_macro_overview output renders a Macro Backdrop table; errored/empty
    indicators are skipped, and an absent correlation tool renders no section."""
    macro = {
        "fed_funds": {"current_rate": "3.63%", "change_1y": "-0.70%", "as_of": "2026-05-01"},
        "inflation": {"headline_inflation": "3.0%", "core_inflation": "3.3%",
                      "status": "Above Target", "as_of": "2026-05-01"},
        "gdp": {"current_rate": "1.6%", "trend": "Accelerating", "as_of": "2026-01-01"},
        "unemployment": {"current_rate": "4.3%", "trend": "Worsening", "as_of": "2026-05-01"},
        "treasury_yields": {"10_year_yield": "4.48%", "2_year_yield": "4.09%",
                            "yield_spread": "0.39%", "curve_status": "Normal", "as_of": "2026-06-12"},
        "summary": "irrelevant",
    }
    out = "\n".join(_build_portfolio_dashboard({"get_macro_overview": macro}))
    assert "### 🏦 Macro Backdrop" in out
    assert "| Fed Funds Rate | 3.63% (1y Δ -0.70%) | 2026-05-01 |" in out
    assert "3.0% headline · 3.3% core — Above Target" in out
    assert "| Real GDP Growth | 1.6% (Accelerating) | 2026-01-01 |" in out
    assert "| Unemployment | 4.3% (Worsening) | 2026-05-01 |" in out
    assert "10Y 4.48% · 2Y 4.09% · spread 0.39% (Normal)" in out
    # No correlation tool ran -> no (empty) Correlation section bleeds in.
    assert "Correlation" not in out

    # Errored / empty indicators contribute no rows -> no table at all. Mix an errored
    # treasury block in to confirm per-row error guards (not just fed_funds) are honored.
    out2 = "\n".join(_build_portfolio_dashboard(
        {"get_macro_overview": {"fed_funds": {"error": "rate limited"}, "inflation": {},
                                 "treasury_yields": {"error": "fred down"}, "unemployment": {}}}))
    assert "Macro Backdrop" not in out2
    # A whole-tool error response (top-level 'error') also renders nothing.
    assert _build_portfolio_dashboard({"get_macro_overview": {"error": "fred unavailable"}}) == []


class DummyTool:
    def __init__(self, name, return_value):
        self.name = name
        self.return_value = return_value

    def invoke(self, args):
        return self.return_value

def test_market_analyst_scan_opportunities_dashboard(monkeypatch):
    """Verify that scan_opportunities runs the scan dashboard and does not trigger single symbol mode."""
    # 1. Mock state and user query
    state = AgentState(messages=[HumanMessage(content="[MarketAnalyst]: scan the market for opportunities")])

    # 2. Mock LLMs and agent creation
    mock_llm = MagicMock()
    monkeypatch.setattr("agent.nodes.market_analyst.get_sonnet_llm", lambda: mock_llm)

    mock_user_context = "Risk Tolerance: Aggressive"
    monkeypatch.setattr("agent.nodes.market_analyst.get_user_context", lambda: mock_user_context)

    # Mock tool metadata retrieval at source (agent.tool_retriever)
    monkeypatch.setattr(
        "agent.tool_retriever.get_semantic_tools_with_metadata",
        lambda query, k: ([], {"tool_count": 0, "selected_tool_names": []})
    )

    # Mock planner response with tool call to scan_opportunities
    mock_planner_res = MagicMock()
    mock_planner_res.tool_calls = [
        {"name": "scan_opportunities", "args": {"sector": "All"}, "id": "call_1"}
    ]
    mock_planner_res.content = "Planning to scan the market."

    # Mock synthesis response from LLM
    mock_synthesis_res = MagicMock()
    mock_synthesis_res.content = "Here is the synthesis of the scan."

    # Intercept safe_invoke to return planner response, then synthesis response
    invoke_count = 0
    def mock_safe_invoke(agent_or_llm, state_or_msgs):
        nonlocal invoke_count
        invoke_count += 1
        if invoke_count == 1:
            return mock_planner_res
        else:
            return mock_synthesis_res

    monkeypatch.setattr("agent.nodes.market_analyst.safe_invoke", mock_safe_invoke)

    # 3. Mock ToolRetriever map to return mock tools
    mock_scan_data = {
        "sector": "All Sectors",
        "top_picks": [
            {"symbol": "AAPL", "score": 120, "price": 175.0, "reason": "Strong momentum", "risk_flags": "None", "description": "Apple Inc."}
        ]
    }
    mock_tool_map = {
        "scan_opportunities": DummyTool("scan_opportunities", mock_scan_data)
    }

    class MockToolRetriever:
        @property
        def tool_map(self):
            return mock_tool_map

    monkeypatch.setattr("agent.tool_retriever.ToolRetriever", MockToolRetriever)

    # Ensure clean stream callback mock
    monkeypatch.setattr("agent.nodes.market_analyst.has_stream_callback", lambda: False)

    # 4. Invoke node
    final_output_state = market_analyst_node(state)

    # 5. Assertions
    assert isinstance(final_output_state, dict)
    response_content = final_output_state['messages'][-1].content

    # Verify Opportunity Scan dashboard is present
    assert "### 🔭 Opportunity Scan: All Sectors" in response_content
    assert "AAPL" in response_content
    # Verify Single Symbol dashboard (and its N/A values) is NOT present
    assert "📋 Current Snapshot" not in response_content
    assert "Price\tN/A" not in response_content


def test_market_analyst_portfolio_only_renders_dashboard(monkeypatch):
    """Regression: a pure portfolio query (no fundamentals/macro DSPy context) must
    still render the portfolio-aggregate dashboard. Previously the dashboard block was
    gated on ``stock_analyst and (fundamentals or macro)``, so portfolio-only queries
    skipped it, left final_output empty, and fell through to the raw tool-dump fallback
    ("The narrative above didn't include the underlying data tables.")."""
    state = AgentState(messages=[HumanMessage(content="[MarketAnalyst]: audit my portfolio for uncompensated risk")])

    mock_llm = MagicMock()
    monkeypatch.setattr("agent.nodes.market_analyst.get_sonnet_llm", lambda: mock_llm)
    monkeypatch.setattr("agent.nodes.market_analyst.get_user_context", lambda: "Risk Tolerance: Aggressive")
    monkeypatch.setattr(
        "agent.tool_retriever.get_semantic_tools_with_metadata",
        lambda query, k: ([], {"tool_count": 0, "selected_tool_names": []})
    )

    mock_planner_res = MagicMock()
    mock_planner_res.tool_calls = [
        {"name": "assess_portfolio_risk", "args": {}, "id": "call_r"},
        {"name": "check_portfolio_correlation", "args": {}, "id": "call_c"},
        {"name": "get_macro_overview", "args": {}, "id": "call_m"},
    ]
    mock_planner_res.content = "Planning portfolio risk audit."

    # Deliberately short synthesis (<200 chars): the OLD code would append the raw-dump
    # fallback here; the fix makes final_output non-empty so the dashboard is shown instead.
    mock_synthesis_res = MagicMock()
    mock_synthesis_res.content = "Concentrated in tech/ESG."

    invoke_count = 0
    def mock_safe_invoke(agent_or_llm, state_or_msgs):
        nonlocal invoke_count
        invoke_count += 1
        return mock_planner_res if invoke_count == 1 else mock_synthesis_res

    monkeypatch.setattr("agent.nodes.market_analyst.safe_invoke", mock_safe_invoke)

    mock_tool_map = {
        "assess_portfolio_risk": DummyTool("assess_portfolio_risk", {
            "snapshot": {
                "total_value_usd": "$500,000 USD",
                "total_value_cad": "$700,000 CAD",
                "exchange_rate": "1 USD = 1.40 CAD",
                "total_gain_loss_pct": 25.0,
            }
        }),
        "check_portfolio_correlation": DummyTool("check_portfolio_correlation", {
            "correlation_pairs": [{"pair": "VEA vs VOO", "correlation": 0.97}],
            "average_correlation": 0.85,
            "diversification_quality": "Poor (>0.5)",
        }),
        "get_macro_overview": DummyTool("get_macro_overview", {
            "fed_funds": {"current_rate": "3.63%", "change_1y": "-0.70%", "as_of": "2026-05-01"},
            "inflation": {"headline_inflation": "3.0%", "core_inflation": "3.3%",
                          "status": "Above Target", "as_of": "2026-05-01"},
            "gdp": {"current_rate": "1.6%", "trend": "Accelerating", "as_of": "2026-01-01"},
        }),
    }

    class MockToolRetriever:
        @property
        def tool_map(self):
            return mock_tool_map

    monkeypatch.setattr("agent.tool_retriever.ToolRetriever", MockToolRetriever)
    monkeypatch.setattr("agent.nodes.market_analyst.has_stream_callback", lambda: False)

    final_output_state = market_analyst_node(state)

    assert isinstance(final_output_state, dict)
    response_content = final_output_state['messages'][-1].content

    # Portfolio dashboard tables render from the aggregate tool outputs...
    assert "### 💼 Portfolio Snapshot" in response_content
    assert "### 🔗 Correlation" in response_content
    assert "VEA vs VOO" in response_content
    # ...including the macro backdrop wired in from get_macro_overview...
    assert "### 🏦 Macro Backdrop" in response_content
    assert "Fed Funds Rate" in response_content
    # ...and the raw tool-dump fallback is NOT triggered.
    assert "The narrative above didn't include" not in response_content


def test_market_analyst_single_symbol_dashboard(monkeypatch):
    """Verify that fetch_fundamentals runs the single stock dashboard and triggers Single Symbol Mode."""
    # 1. Mock state and user query for a single symbol
    state = AgentState(messages=[HumanMessage(content="[MarketAnalyst]: analyze AAPL")])

    # 2. Mock LLMs and agent creation
    mock_llm = MagicMock()
    monkeypatch.setattr("agent.nodes.market_analyst.get_sonnet_llm", lambda: mock_llm)

    mock_user_context = "Risk Tolerance: Aggressive"
    monkeypatch.setattr("agent.nodes.market_analyst.get_user_context", lambda: mock_user_context)

    # Mock tool metadata retrieval at source (agent.tool_retriever)
    monkeypatch.setattr(
        "agent.tool_retriever.get_semantic_tools_with_metadata",
        lambda query, k: ([], {"tool_count": 0, "selected_tool_names": []})
    )

    # Mock planner response with tool call to fetch_fundamentals
    mock_planner_res = MagicMock()
    mock_planner_res.tool_calls = [
        {"name": "fetch_fundamentals", "args": {"symbol": "AAPL"}, "id": "call_2"}
    ]
    mock_planner_res.content = "Planning to fetch fundamentals for AAPL."

    # Mock synthesis response from LLM
    mock_synthesis_res = MagicMock()
    mock_synthesis_res.content = "Here is the synthesis of AAPL."

    # Intercept safe_invoke
    invoke_count = 0
    def mock_safe_invoke(agent_or_llm, state_or_msgs):
        nonlocal invoke_count
        invoke_count += 1
        if invoke_count == 1:
            return mock_planner_res
        else:
            return mock_synthesis_res

    monkeypatch.setattr("agent.nodes.market_analyst.safe_invoke", mock_safe_invoke)

    # 3. Mock ToolRetriever map to return mock tools
    mock_fundamentals_data = {
        "symbol": "AAPL",
        "price": "$175.00",
        "market_cap": "$3.0T",
        "pe_ratio": "28.5",
        "52_week_low": "$165.00",
        "52_week_high": "$195.00",
        "description": "Apple Inc. designs consumer electronics."
    }
    mock_tool_map = {
        "fetch_fundamentals": DummyTool("fetch_fundamentals", mock_fundamentals_data)
    }

    class MockToolRetriever:
        @property
        def tool_map(self):
            return mock_tool_map

    monkeypatch.setattr("agent.tool_retriever.ToolRetriever", MockToolRetriever)
    monkeypatch.setattr("agent.nodes.market_analyst.has_stream_callback", lambda: False)

    # 4. Invoke node
    final_output_state = market_analyst_node(state)

    # 5. Assertions
    assert isinstance(final_output_state, dict)
    response_content = final_output_state['messages'][-1].content

    # Verify Single Symbol dashboard is present with correct mocked values
    assert "### 📋 Current Snapshot" in response_content
    assert "| **Price** | $175.00 |" in response_content
    assert "| **Market Cap** | $3.0T |" in response_content
    assert "| **P/E Ratio** | 28.5 |" in response_content
    # Verify Opportunity Scan dashboard is NOT present
    assert "🔭 Opportunity Scan" not in response_content


@pytest.mark.parametrize(
    "lens,expected_handoff",
    [
        ("external_screen", True),   # Scan button — ranked external picks
        ("guru_validation", True),   # Guru Pick button — validated picks
        ("market_dip", True),        # Dip Plan button — staged entry/deployment plan
        ("portfolio_audit", False),  # Analyze button — descriptive, no picks to reconcile
    ],
)
def test_market_analyst_handoff_flag_by_lens(monkeypatch, lens, expected_handoff):
    """MarketAnalyst must flag market_analyst_handoff=True for lenses that make an
    implicit buy/sell/entry-timing call (external_screen, guru_validation, market_dip),
    so after_market_analyst routes through DeepReasoning before RiskManager instead of
    shipping the pick straight to the compliance gate with no portfolio-aware review.
    portfolio_audit stays False — its contract is descriptive, not pick-generating."""
    state = AgentState(messages=[HumanMessage(content=f"[MarketAnalyst lens={lens}]: run it")])

    monkeypatch.setattr("agent.nodes.market_analyst.get_sonnet_llm", lambda: MagicMock())
    monkeypatch.setattr("agent.nodes.market_analyst.get_user_context", lambda: "Risk Tolerance: Aggressive")
    monkeypatch.setattr(
        "agent.tool_retriever.get_semantic_tools_with_metadata",
        lambda query, k: ([], {"tool_count": 0, "selected_tool_names": []})
    )

    mock_planner_res = MagicMock()
    mock_planner_res.tool_calls = [
        {"name": "scan_opportunities", "args": {"sector": "All"}, "id": "call_1"}
    ]
    mock_planner_res.content = "Planning to scan the market."

    mock_synthesis_res = MagicMock()
    mock_synthesis_res.content = "Synthesis text."

    invoke_count = 0
    def mock_safe_invoke(agent_or_llm, state_or_msgs):
        nonlocal invoke_count
        invoke_count += 1
        return mock_planner_res if invoke_count == 1 else mock_synthesis_res

    monkeypatch.setattr("agent.nodes.market_analyst.safe_invoke", mock_safe_invoke)

    mock_tool_map = {
        "scan_opportunities": DummyTool("scan_opportunities", {
            "sector": "All Sectors",
            "top_picks": [{"symbol": "MU", "score": 87, "price": 948.80, "reasons": [], "description": "Micron"}],
        })
    }

    class MockToolRetriever:
        @property
        def tool_map(self):
            return mock_tool_map

    monkeypatch.setattr("agent.tool_retriever.ToolRetriever", MockToolRetriever)
    monkeypatch.setattr("agent.nodes.market_analyst.has_stream_callback", lambda: False)

    final_output_state = market_analyst_node(state)

    assert final_output_state.get("market_analyst_handoff") is expected_handoff
