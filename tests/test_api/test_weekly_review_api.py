"""The weekly review's endpoint and page.

The module's own tests cover the assembly contract. These cover the two things
only a wired-up app can prove: that the page renders every section server-side
(it is meant to be read and printed, so it must not depend on client JS), and
that neither surface starts work — a report that generates is a report that can
block, cost money, or change the state it is describing.
"""
import pytest
from fastapi.testclient import TestClient

_SECTION_TITLES = (
    "Wealth goal",
    "Market state",
    "How past advice scored",
    "What the advisor said this week",
    "Background engines",
    "Inputs still needed",
)


@pytest.fixture()
def client():
    from server import app

    return TestClient(app)


def test_the_endpoint_returns_every_section(client):
    body = client.get("/api/weekly_review").json()

    assert body["counts"]["total"] == len(_SECTION_TITLES)
    assert {s["title"] for s in body["sections"]} == set(_SECTION_TITLES)


def test_the_page_renders_every_section_without_javascript(client):
    """Server-rendered on purpose: a single parse error in an inline script takes
    the whole block down, which is how a panel ends up stuck on 'Loading …' with
    a healthy backend behind it."""
    html = client.get("/review").text

    for title in _SECTION_TITLES:
        assert title in html, f"{title} is missing from the rendered page"


def test_a_blank_section_says_so_in_the_rendered_page(client):
    """The rendered proof of the module's contract: on a profile with nothing to
    report, the page must still show the section and label it."""
    html = client.get("/review").text

    assert "nothing to report" in html


def test_the_page_carries_the_contract_line(client):
    html = client.get("/review").text

    assert "nothing here is inferred, defaulted or filled in" in html


def test_neither_surface_starts_a_pulse_generation(client, monkeypatch):
    """The market section reads cache only. If it ever starts a generation, a
    weekly page becomes a multi-minute network job on every cold open."""
    started = []
    monkeypatch.setattr(
        "tools.market_sentinel.generate_market_pulse",
        lambda *a, **k: started.append(True),
    )

    client.get("/api/weekly_review")
    client.get("/review")

    assert not started


def test_the_review_is_reachable_from_the_navigation(client):
    """A page nobody can navigate to is its own kind of dark surface."""
    html = client.get("/alerts").text

    assert 'href="/review"' in html
