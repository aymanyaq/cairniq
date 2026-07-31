"""Today's Priority precompute (Theme 3.1): worker extraction/caching + endpoint."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from server import app


@pytest.fixture()
def client():
    from tools.user_profile import get_active_profile

    test_client = TestClient(app)
    test_client.cookies.set("profile", get_active_profile())
    return test_client


# ------------------------------------------------------------------
# Markdown composition from a finished graph run
# ------------------------------------------------------------------

def test_compose_priority_markdown_strips_scaffolding():
    """The cached brief must be the user-visible chat product: node prefixes and
    <thinking> blocks stripped, DeepReasoning brief + RiskManager verdict joined,
    and nothing from before the current turn."""
    from api.background import _compose_priority_markdown

    final_state = {
        "messages": [
            AIMessage(content="[DeepReasoning]: stale earlier-turn output", name="DeepReasoning"),
            HumanMessage(content="You are the Today's Priority engine... (full rewritten prompt)"),
            AIMessage(
                content="[DeepReasoning]: <thinking>internal chain</thinking>## Today's Priority\nTrim NVDA into strength.",
                name="DeepReasoning",
            ),
            AIMessage(content="", name="DeepReasoning"),  # empty — must be skipped
            AIMessage(content="[RiskManager]: \n\n---\n### 🛡️ Risk Assessment\nPASS", name="RiskManager"),
        ]
    }

    md = _compose_priority_markdown(final_state)

    assert "Trim NVDA" in md
    assert "Risk Assessment" in md
    assert "internal chain" not in md, "<thinking> content must never reach the cache"
    assert "[DeepReasoning]" not in md and "[RiskManager]" not in md
    assert "stale earlier-turn output" not in md, "only messages after the last human turn count"


# ------------------------------------------------------------------
# Worker: run → cache
# ------------------------------------------------------------------

class _FakeGraph:
    def __init__(self, final_state):
        self.final_state = final_state
        self.seen_inputs = None
        self.seen_config = None

    def invoke(self, inputs, config=None):
        self.seen_inputs = inputs
        self.seen_config = config
        return self.final_state


def test_run_priority_precompute_caches_brief(monkeypatch):
    import agent.graph as graph_mod
    import api.background as background
    from tools.daily_cache import get_cached

    fake = _FakeGraph({
        "messages": [
            HumanMessage(content="rewritten prompt"),
            AIMessage(content="[DeepReasoning]: The single priority is X.", name="DeepReasoning"),
        ]
    })
    monkeypatch.setattr(graph_mod, "build_graph", lambda use_memory=True: fake)

    assert background.run_priority_precompute_in_background() is True

    # The worker must drive the SAME marker the dashboard button sends.
    sent = fake.seen_inputs["messages"][0].content
    assert "[QuickAction name=priority]" in sent
    assert fake.seen_inputs["risk_retry_count"] == 0

    cached = get_cached("today_priority")
    assert cached and "The single priority is X." in cached["markdown"]
    assert cached.get("generated_at")
    assert not background.is_priority_running(), "running flag must clear on completion"


def test_run_priority_precompute_failure_caches_nothing(monkeypatch):
    import agent.graph as graph_mod
    import api.background as background
    from tools.daily_cache import get_cached

    def _boom(use_memory=True):
        raise RuntimeError("provider down")

    monkeypatch.setattr(graph_mod, "build_graph", _boom)

    assert background.run_priority_precompute_in_background() is False
    assert get_cached("today_priority") is None
    assert not background.is_priority_running(), "running flag must clear on failure"


def test_run_priority_precompute_empty_output_is_failure(monkeypatch):
    import agent.graph as graph_mod
    import api.background as background
    from tools.daily_cache import get_cached

    fake = _FakeGraph({"messages": [HumanMessage(content="rewritten prompt")]})
    monkeypatch.setattr(graph_mod, "build_graph", lambda use_memory=True: fake)

    assert background.run_priority_precompute_in_background() is False
    assert get_cached("today_priority") is None


# ------------------------------------------------------------------
# Endpoint: cost-respectful reads
# ------------------------------------------------------------------

def test_priority_endpoint_plain_get_never_autospends(client, monkeypatch):
    import api.routers.news as news_router

    started = []
    monkeypatch.setattr(news_router, "start_priority_precompute", lambda: started.append(1))

    response = client.get("/api/priority")
    assert response.status_code == 200
    data = response.json()
    assert "markdown" in data or data.get("status") in ("fetching", "empty")
    assert started == [], "a plain GET must never start a Deep Reasoning run"


def test_priority_endpoint_force_starts_run(client, monkeypatch):
    import api.routers.news as news_router

    started = []
    monkeypatch.setattr(news_router, "start_priority_precompute", lambda: started.append(1))

    response = client.get("/api/priority?force=true")
    assert response.status_code == 200
    assert response.json().get("status") == "fetching"
    assert started == [1]


def test_priority_endpoint_serves_cached_brief(client):
    from tools.daily_cache import set_cached

    set_cached("today_priority", {
        "markdown": "## Today's Priority\nHold the line.",
        "generated_at": "2026-07-13T07:45:00",
    })

    response = client.get("/api/priority")
    assert response.status_code == 200
    data = response.json()
    assert "Hold the line." in data["markdown"]
    assert data["generated_at"] == "2026-07-13T07:45:00"
