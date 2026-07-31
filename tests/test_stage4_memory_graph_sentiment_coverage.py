import json
import sys
import types

import networkx as nx
import pandas as pd


def test_memory_context_includes_profile_graph_macro_and_theses(monkeypatch):
    import tools.memory as memory

    fake_macro = types.ModuleType("tools.macro_strategy")
    fake_macro.analyze_macro_context = lambda: {
        "current_regime": "Expansion",
        "plain_english": "Tactical: add quality growth.",
        "key_indicators": {"Systemic Risk": "Low"},
    }
    fake_mechanics = types.ModuleType("tools.market_mechanics")
    fake_mechanics.detect_sector_rotation = lambda: {
        "leading_sectors": ["Technology"],
        "lagging_sectors": ["Utilities"],
    }
    monkeypatch.setitem(sys.modules, "tools.macro_strategy", fake_macro)
    monkeypatch.setitem(sys.modules, "tools.market_mechanics", fake_mechanics)
    # Set up a real graph with edges for the new prompt injection path
    import networkx as nx
    test_graph = nx.MultiDiGraph()
    test_graph.add_node("AAPL", type="Stock", owned=True)
    test_graph.add_node("Technology", type="Sector")
    test_graph.add_node("User", type="Person")
    test_graph.add_edge("AAPL", "Technology", relation="IN_SECTOR")
    test_graph.add_edge("User", "ExampleBroker", relation="HAS_ACCOUNT_AT")
    # The `graph` property getter calls _ensure_profile_sync() on every access,
    # which reloads the graph from disk (clobbering our injected test_graph)
    # whenever the singleton's cached `current_profile` has drifted from the
    # active profile — a state that can leak in from earlier tests. With no graph
    # file present (as in CI) that reload yields an empty graph, so the PORTFOLIO
    # GRAPH block silently disappears and this test fails non-deterministically.
    # Pin the sync to a no-op so the injected graph is what get_user_context reads.
    monkeypatch.setattr(memory.graph_memory, "_ensure_profile_sync", lambda: None)
    monkeypatch.setattr(memory.graph_memory, "graph", test_graph)
    monkeypatch.setattr(memory, "load_memory", lambda: {
        "lessons_learned": ["Do not chase gaps"],
        "user_profile": {
            "name": "TestUser",
            "age": 42,
            "risk_tolerance": "aggressive",
            "retirement_age": 60,
            "annual_income": "not-a-number",
            "investment_goals": ["income", "growth"],
            "accounts": ["TFSA", "RRSP"],
        },
        "key_facts": [f"fact {i}" for i in range(7)],
        "conversation_summaries": [
            {"date": "2026-01-01", "summary": "old"},
            {"date": "2026-01-02", "summary": "recent"},
            {"date": "2026-01-03", "summary": "latest"},
        ],
        "past_recommendations": [
            {"date": "2026-01-01", "action": "buy", "ticker": "AAPL"},
            {"date": "2026-01-02", "action": "hold", "ticker": "MSFT"},
            {"date": "2026-01-03", "action": "trim", "ticker": "NVDA"},
            {"date": "2026-01-04", "action": "watch", "ticker": "GOOG"},
        ],
        "active_theses": [{
            "symbol": "NVDA",
            "action": "WATCH",
            "stop_loss": "$800",
            "catalyst": "earnings",
            "expiry_date": "2026-05-01",
            "conditions": "hold support",
            "notes": "AI demand",
        }],
    })

    context = memory.get_user_context()

    assert "PROFILE_FIELDS:" in context
    assert "profile_status: available" in context
    assert "risk_tolerance: aggressive" in context
    assert "years_to_retirement: 18" in context
    assert "annual_income: not-a-number" in context
    assert "base_currency:" in context
    assert "accounts: TFSA, RRSP" in context
    assert "TestUser" in context
    assert "not-a-number" in context
    assert "fact 2" in context and "fact 1" not in context
    assert "latest" in context and "2026-01-01" not in context
    assert "PORTFOLIO GRAPH" in context  # New graph block header
    assert "AAPL(Tech)" in context  # IN_SECTOR edge rendered compactly
    assert "Expansion" in context
    assert "NVDA [WATCH]" in context


def test_memory_llm_extraction_and_message_processing(monkeypatch):
    import tools.memory as memory

    class FakeContextResult:
        profile_updates = "{'age': 44, 'risk_tolerance': 'moderate'}"
        new_facts = "['Has a TFSA', 'Prefers CAD ETFs']"
        new_relationships = "[['User', 'INTERESTED_IN', 'Quantum Computing'], ['bad']]"

    class FakeThesisResult:
        symbol = "MSFT"
        action = "BUY"
        catalyst = "earnings"
        catalyst_date = "2026-05-01"
        stop_loss = "$380"
        conditions = "breakout"
        expiry_date = "2026-06-01"
        notes = "margin expansion"

    def fake_chain(signature):
        return lambda **kwargs: FakeThesisResult() if signature is memory.ActiveThesisExtraction else FakeContextResult()

    relationships = []
    monkeypatch.setattr(memory.dspy, "ChainOfThought", fake_chain)
    monkeypatch.setattr(memory, "update_profile", lambda updates: relationships.append(("profile", updates)))
    monkeypatch.setattr(memory, "add_fact", lambda fact: relationships.append(("fact", fact)))
    monkeypatch.setattr(memory.graph_memory, "add_relationship", lambda s, t, r, **kw: relationships.append((s, r, t)))
    monkeypatch.setattr(memory, "safe_print", lambda message: None)

    profile, facts, rels = memory.extract_context_with_llm("I own AAPL")
    thesis = memory.extract_thesis_from_text("Buy MSFT into earnings")
    memory.process_user_message("I own AAPL")

    assert profile["age"] == 44
    assert facts == ["Has a TFSA", "Prefers CAD ETFs"]
    assert rels[0] == ["User", "INTERESTED_IN", "Quantum Computing"]
    assert thesis["symbol"] == "MSFT"
    assert ("fact", "Has a TFSA") in relationships
    assert ("User", "INTERESTED_IN", "Quantum Computing") in relationships


def test_memory_clean_and_delete_branches(monkeypatch, tmp_path):
    import tools.memory as memory

    saved = []
    graph_file = tmp_path / "knowledge_graph.json"
    graph_file.write_text("{}", encoding="utf-8")

    fake_memory = {
        "user_profile": {"name": "TestUser"},
        "key_facts": ["one"],
        "conversation_summaries": [{"summary": "hello"}],
        "past_recommendations": [{"ticker": "AAPL"}],
        "active_theses": [],
    }
    monkeypatch.setattr(memory, "load_memory", lambda: json.loads(json.dumps(fake_memory)))
    monkeypatch.setattr(memory, "save_memory", lambda data: saved.append(data) or True)
    monkeypatch.setattr(memory, "get_data_path", lambda filename: str(graph_file if filename == "knowledge_graph.json" else tmp_path / filename))
    monkeypatch.setattr("tools.user_profile.get_data_path", lambda filename: str(graph_file if filename == "knowledge_graph.json" else tmp_path / filename))

    assert memory.clean_memory("facts") == "Key facts cleared."
    assert saved[-1]["key_facts"] == []
    assert memory.clean_memory("profile") == "User profile reset."
    assert saved[-1]["user_profile"]["name"] is None
    assert memory.clean_memory("history") == "Conversation history and recommendations cleared."
    assert saved[-1]["past_recommendations"] == []
    assert memory.clean_memory("all").startswith("Memory completely wiped")
    assert not graph_file.exists()
    assert memory.clean_memory("mystery") == "Unknown target: mystery"
    assert memory.delete_lesson(0) is False
    assert memory.delete_key_fact(99) is False
    assert memory.update_lesson(0, "x") is False
    assert memory.get_active_theses() == []


def test_graph_memory_round_trip_context_and_portfolio_summary(monkeypatch, tmp_path):
    import tools.graph_memory as graph_mod

    graph_path = tmp_path / "knowledge_graph.json"
    monkeypatch.setattr(graph_mod, "get_data_path", lambda filename: str(graph_path))
    monkeypatch.setattr(graph_mod, "get_active_profile", lambda: "stage4")

    gm = graph_mod.GraphMemory()
    callback_payloads = []
    gm.on_save_callback = callback_payloads.append

    gm.add_entity("Portfolio", "Unknown")
    gm.add_entity("Portfolio", "UserPortfolio")
    assert gm.graph.nodes["Portfolio"]["type"] == "UserPortfolio"

    gm.add_entity("User", "Person")
    gm.add_entity("AAPL", "Stock", {"sector": "Technology"})
    gm.add_relationship("User", "AAPL", "interested_in", {"source": "chat"})
    gm.add_relationship("User", "AAPL", "interested_in", {"confidence": "high"})

    assert graph_path.exists()
    assert callback_payloads
    assert "INTERESTED_IN" in gm.get_context(["User"], depth=1)
    assert gm.graph.edges["User", "AAPL", 0]["confidence"] == "high"
    assert gm.delete_relationship("User", "AAPL", "INTERESTED_IN") is True
    assert gm.delete_relationship("User", "AAPL", "MISSING") is False
    assert gm.delete_entity("AAPL") is True
    assert gm.delete_entity("AAPL") is False

    gm.add_portfolio_context(
        [{"symbol": "MSFT", "sector": "Technology"}, {"symbol": "", "sector": "Cash"}],
        sector_exposure={"Technology": 72.25, "": 10},
        correlations=[("MSFT", "NVDA", 0.83), ("MSFT", "BND", 0.2)],
    )
    summary = gm.get_portfolio_summary()

    assert "Technology: 72.2%" in summary
    assert "MSFT" in summary and "0.83" in summary

    reloaded = graph_mod.GraphMemory()
    assert reloaded.graph.has_node("MSFT")


def test_graph_memory_load_error_and_fallback_paths(monkeypatch, tmp_path):
    import tools.graph_memory as graph_mod

    graph_path = tmp_path / "knowledge_graph.json"
    graph_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(graph_mod, "get_data_path", lambda filename: str(graph_path))
    monkeypatch.setattr(graph_mod, "get_active_profile", lambda: "broken")
    monkeypatch.setattr(graph_mod, "safe_print", lambda message: None)

    gm = graph_mod.GraphMemory()
    assert isinstance(gm.graph, nx.MultiDiGraph)
    assert gm.get_context(["Missing"]) == ""

    gm.graph = nx.MultiDiGraph()
    gm.add_entity("User", "Person")
    monkeypatch.setattr(graph_mod.nx, "ego_graph", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert gm.get_context(["User"]) == ""

    monkeypatch.setattr(graph_mod.nx, "node_link_data", lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("old api")))
    monkeypatch.setattr(graph_mod.json, "dump", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    gm.save()


def test_sentiment_news_analyst_and_buzz_branches(monkeypatch):
    import tools.sentiment_analysis as sentiment

    class Response:
        status_code = 200

        def json(self):
            return {
                "feed": [{
                    "title": "Company beats estimates",
                    "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "0.30"}],
                }]
            }

    monkeypatch.setattr(sentiment.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sentiment, "get_api_key", lambda name, default=None: "demo")
    sentiment_result = sentiment.get_news_sentiment.__wrapped__("AAPL")
    assert sentiment_result["news_sentiment"].startswith("Very Positive")

    class SummaryTicker:
        recommendations = pd.DataFrame()
        info = {"recommendationKey": "strong_buy", "targetMeanPrice": 210.0}

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: SummaryTicker())
    analyst_summary = sentiment.get_analyst_consensus.__wrapped__("AAPL")
    assert analyst_summary["consensus"] == "Strong Buy"

    summary_table = pd.DataFrame([{"strongBuy": 7, "buy": 3, "hold": 1, "sell": 0, "strongSell": 0}])
    class TableTicker:
        recommendations = summary_table
        info = {"targetMeanPrice": 220.0, "currentPrice": 200.0}

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: TableTicker())
    analyst_table = sentiment.get_analyst_consensus.__wrapped__("MSFT")
    assert analyst_table["analyst_consensus"].startswith("Strong Buy")
    assert analyst_table["upside_potential"] == "+10.0%"

    hist = pd.DataFrame({"Volume": [1000] * 25 + [300] * 5})
    class LowBuzzTicker:
        def history(self, *args, **kwargs):
            return hist

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: LowBuzzTicker())
    assert sentiment.get_social_buzz("QUIET")["social_buzz_estimate"].startswith("Low")


def test_sentiment_core_branches_without_cache(monkeypatch):
    import tools.sentiment_analysis as sentiment

    class GreedResponse:
        status_code = 200

        def json(self):
            return {"fear_and_greed": {"score": 82, "rating": "Extreme Greed"}}

    monkeypatch.setattr(sentiment.requests, "get", lambda *args, **kwargs: GreedResponse())
    fear = sentiment.get_fear_greed_index.__wrapped__()
    assert fear["rating"] == "Extreme Greed"
    assert "taking profits" in fear["suggested_action"]

    class NoFeedResponse:
        def json(self):
            return {"Information": "rate limited"}

    monkeypatch.setattr(sentiment.requests, "get", lambda *args, **kwargs: NoFeedResponse())
    monkeypatch.setattr(sentiment, "_get_yfinance_news_sentiment", lambda symbol: {"symbol": symbol, "fallback": True})
    assert sentiment.get_news_sentiment.__wrapped__("AAPL")["fallback"] is True

    class EmptyFeedResponse:
        def json(self):
            return {"feed": []}

    monkeypatch.setattr(sentiment.requests, "get", lambda *args, **kwargs: EmptyFeedResponse())
    assert "No news found" in sentiment.get_news_sentiment.__wrapped__("AAPL")["error"]


def test_sentiment_yfinance_analyst_social_and_full_branches(monkeypatch):
    import tools.sentiment_analysis as sentiment

    class NewsTicker:
        news = [
            {"title": "Shares drop after earnings miss"},
            {"title": "Company announces routine update"},
        ]

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: NewsTicker())
    news = sentiment._get_yfinance_news_sentiment("AAPL")
    assert news["recent_headlines"][0]["sentiment"] == "Negative"

    class NoNewsTicker:
        news = []

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: NoNewsTicker())
    assert "No recent news" in sentiment._get_yfinance_news_sentiment("EMPTY")["note"]

    class LegacyAnalystTicker:
        recommendations = pd.DataFrame({"To Grade": ["Buy", "Underperform", "Neutral"]})
        info = {"targetMeanPrice": 90.0, "currentPrice": 100.0}

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: LegacyAnalystTicker())
    legacy = sentiment.get_analyst_consensus.__wrapped__("AAPL")
    assert legacy["ratings"] == {"Buy": 1, "Hold": 1, "Sell": 1}
    assert legacy["analyst_consensus"].startswith("Hold")

    class UnknownAnalystTicker:
        recommendations = pd.DataFrame({"firm": ["Underweight Bank", "Overweight Shop"]})
        info = {"previousClose": 100.0}

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: UnknownAnalystTicker())
    unknown = sentiment.get_analyst_consensus.__wrapped__("MIX")
    assert unknown["ratings"]["Buy"] == 1
    assert unknown["ratings"]["Sell"] == 1

    empty_hist = pd.DataFrame()
    class EmptyHistoryTicker:
        def history(self, *args, **kwargs):
            return empty_hist

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: EmptyHistoryTicker())
    assert "No data available" in sentiment.get_social_buzz("NONE")["note"]

    normal_hist = pd.DataFrame({"Volume": [1000] * 30})
    class NormalBuzzTicker:
        def history(self, *args, **kwargs):
            return normal_hist

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda symbol: NormalBuzzTicker())
    assert sentiment.get_social_buzz("AVG")["social_buzz_estimate"] == "Normal"

    monkeypatch.setattr(sentiment, "get_fear_greed_index", lambda: {"score": 80})
    monkeypatch.setattr(sentiment, "get_news_sentiment", lambda symbol: {"sentiment_score": -0.3})
    monkeypatch.setattr(sentiment, "get_analyst_consensus", lambda symbol: {"buy_percentage": 10})
    monkeypatch.setattr(sentiment, "get_social_buzz", lambda symbol: {"social_buzz_estimate": "Low"})
    monkeypatch.setattr(sentiment, "get_reddit_sentiment", lambda symbol: {"reddit_hype_score": 20})
    full = sentiment.get_full_sentiment.__wrapped__("AAPL")
    assert "BEARISH" in full["overall_sentiment"]
    assert full["signal_breakdown"]["bearish"] > full["signal_breakdown"]["bullish"]
