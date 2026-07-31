"""Dashboard context must survive partially-filled profiles.

A new user whose profile has an age but no income (the common shape right
after a household member is added) once 500'd every page that renders from
get_dashboard_context — `/`, `/settings`, `/portfolio`, `/memory` and the
rest — because a local `import re` inside the age branch made `re`
function-local for the whole function, leaving it unbound when the age
branch was skipped and the income branch ran.
"""

import pytest

from api.routers import pages


class _FakeURL:
    path = "/"


class _FakeState:
    user = None


class _FakeRequest:
    """Minimal stand-in for the Request fields get_dashboard_context reads."""

    url = _FakeURL()
    state = _FakeState()


# (age, annual_income, retirement_age) combinations a real profile can hold.
# The (set, unset, unset) row is the one that took the server down.
@pytest.mark.parametrize(
    "age,income,retirement_age",
    [
        (30, None, None),          # age set, nothing else — the regression
        ("37", "$95000", "65"),    # fully populated
        (None, None, None),        # brand-new empty profile
        (None, "200000", None),    # income only
        ("41", None, "67"),        # age + retirement, income missing
        (0, None, None),           # falsy-but-present age
    ],
)
def test_dashboard_context_survives_partial_profile(monkeypatch, age, income, retirement_age):
    profile = {}
    if age is not None:
        profile["age"] = age
    if income is not None:
        profile["annual_income"] = income
    if retirement_age is not None:
        profile["retirement_age"] = retirement_age

    monkeypatch.setattr(
        "tools.memory.load_memory",
        lambda *a, **k: {"user_profile": profile, "key_facts": []},
    )
    monkeypatch.setattr("tools.memory.get_active_theses", lambda *a, **k: [])

    context = pages.get_dashboard_context(_FakeRequest())

    assert context["profile"]["base_currency"]
    # An absent income renders as a placeholder rather than raising.
    assert context["profile"]["annual_income_display"]


def test_dashboard_context_extracts_income_from_facts_when_unset(monkeypatch):
    """The income fallback still works — the fix removed only the shadowing import."""
    monkeypatch.setattr(
        "tools.memory.load_memory",
        lambda *a, **k: {
            "user_profile": {"age": 30},
            "key_facts": ["I make close to $85,000 a year"],
        },
    )
    monkeypatch.setattr("tools.memory.get_active_theses", lambda *a, **k: [])

    context = pages.get_dashboard_context(_FakeRequest())

    assert "85" in context["profile"]["annual_income"]
