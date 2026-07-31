from unittest.mock import patch

from agent.tool_registry import (
    analyze_technicals,
    calculate_position,
    fetch_fundamentals,
    get_sentiment,
    log_investment_decision,
    verify_portfolio_holdings,
)


def test_registry_metadata():
    # Test that the functions are decorated with LangChain's @tool
    # which adds 'args_schema', 'name', 'description'
    assert hasattr(fetch_fundamentals, "name")
    assert fetch_fundamentals.name == "fetch_fundamentals"
    assert "PE ratio" in fetch_fundamentals.description

@patch("agent.tool_registry.get_stock_data")
def test_tool_execution(mock_get_data):
    mock_get_data.return_value = {"symbol": "AAPL", "price": 150.0}

    # Tool execution via .invoke
    res = fetch_fundamentals.invoke({"symbol": "AAPL"})
    assert res["symbol"] == "AAPL"
    mock_get_data.assert_called_with("AAPL")

@patch("agent.tool_registry.get_comprehensive_technicals")
def test_analyze_technicals(mock_tech):
    mock_tech.return_value = "Bullish Setup"
    res = analyze_technicals.invoke({"symbol": "AAPL"})
    assert res == "Bullish Setup"

@patch("agent.tool_registry.get_full_sentiment")
def test_get_sentiment(mock_sent):
    mock_sent.return_value = {"overall_sentiment": "Positive"}
    res = get_sentiment.invoke({"symbol": "AAPL"})
    assert res["overall_sentiment"] == "Positive"

@patch("agent.tool_registry.calculate_position_size")
def test_calculate_position(mock_calc):
    mock_calc.return_value = {"shares": 100}
    res = calculate_position.invoke({
        "portfolio_value": 100000,
        "entry_price": 150,
        "stop_loss_price": 140
    })
    assert res["shares"] == 100
    # None, not 2.0: the risk % is resolved from the user's own profile. Passing
    # a default here is what put a phantom "your 2% risk limit" into the advice.
    mock_calc.assert_called_with(100000, None, 150, 140)

    calculate_position.invoke({
        "portfolio_value": 100000,
        "entry_price": 150,
        "stop_loss_price": 140,
        "risk_per_trade_pct": 1.5,
    })
    mock_calc.assert_called_with(100000, 1.5, 150, 140)


@patch("agent.tool_registry.log_trade")
def test_log_investment_decision_preserves_trade_arguments(mock_log_trade):
    mock_log_trade.return_value = "ok"

    res = log_investment_decision.invoke({
        "symbol": "AAPL",
        "action": "BUY",
        "price": 100.0,
        "quantity": 5.0,
        "thesis": "undervalued",
        "time_horizon": "Long",
        "conviction": "High",
    })

    assert res == "ok"
    mock_log_trade.assert_called_once_with(
        symbol="AAPL",
        action="BUY",
        price=100.0,
        quantity=5.0,
        thesis="undervalued",
        time_horizon="Long",
        conviction="High",
    )


def test_verify_portfolio_holdings_marks_absent_tickers(monkeypatch):
    monkeypatch.setattr(
        "tools.portfolio_csv.get_portfolio_decision_context",
        lambda symbols=None: {
            "as_of": "2026-05-14T10:00:00",
            "is_stale": False,
            "sync_errors": [],
            "total_value_cad": 10000.0,
            "total_value_usd": 7000.0,
            "owned_symbols": ["T"],
            "holdings": [{"symbol": "T", "value_cad": 720.0, "allocation_pct": 7.2}],
            "requested_symbols": [
                {"symbol": "T", "owned": True, "matches": [{"symbol": "T"}]},
                {"symbol": "DAL", "owned": False, "matches": []},
            ],
        },
    )

    result = verify_portfolio_holdings.invoke({"symbols": "T,DAL"})

    assert result["requested_symbols"][0]["owned"] is True
    assert result["requested_symbols"][1]["owned"] is False
    assert "Only requested_symbols with owned=True" in result["rule"]
