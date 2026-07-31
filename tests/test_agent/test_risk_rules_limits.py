"""The risk rules must name the user's OWN limit — or admit there is none.

Regression cover for the phantom "2% risk rule": these prompts hardcoded that
figure for every profile, so the judge enforced a limit no profile contained
and quoted it back to the user as "your 2% limit" — the invented-rule
attribution its own rule 8 forbids.
"""
import agent.risk_rules as rr


def test_stated_limit_is_named_in_both_perspectives():
    caps = {"max_risk_per_trade_pct": 1.5}

    generator = rr.risk_rules_generator(caps)
    judge = rr.risk_rules_judge(caps)

    assert "user's 1.5% max risk rule" in generator
    assert "user's 1.5% risk limit" in judge
    assert "2%" not in generator
    assert "2%" not in judge


def test_integral_limit_renders_without_decimal_noise():
    assert "user's 3% max risk rule" in rr.risk_rules_generator({"max_risk_per_trade_pct": 3.0})


def test_no_stated_limit_states_the_absence_instead_of_inventing_one():
    generator = rr.risk_rules_generator({})
    judge = rr.risk_rules_judge({})

    for rules in (generator, judge):
        assert "NO maximum-risk limit" in rules
        # The specific failure mode: any surviving percentage would be read as
        # the user's rule, and 2% is the one the model has seen a thousand times.
        assert "2% risk" not in rules
        assert "2% max risk" not in rules
    assert "never invent, assume, or cite a percentage risk cap" in judge
    assert "There is no default 2% rule." in judge


def test_sizing_disclosure_survives_an_absent_limit():
    """No cap to enforce is not licence to stop reporting the exposure."""
    generator = rr.risk_rules_generator({})
    judge = rr.risk_rules_judge({})

    assert "dollar-at-risk" in generator
    assert "percent of portfolio" in judge
    assert "proposed dollar/share size" in judge


def test_slot_is_always_filled():
    for caps in ({}, {"max_risk_per_trade_pct": 2.0}):
        assert rr._MAGNITUDE_SLOT not in rr.risk_rules_generator(caps)
        assert rr._MAGNITUDE_SLOT not in rr.risk_rules_judge(caps)


def test_unreadable_profile_yields_the_no_limit_wording(monkeypatch):
    """A failed profile read must not silently fall back to a house default."""
    import tools.ips_precheck as ips

    def boom():
        raise OSError("profile unreadable")
    monkeypatch.setattr(ips, "_load_memory", boom)

    assert "NO maximum-risk limit" in rr.risk_rules_judge()


def test_caps_are_read_per_call_not_frozen_at_import(monkeypatch):
    """Judge rules are built per request; a profile edit must take effect."""
    import tools.ips_precheck as ips

    monkeypatch.setattr(ips, "_load_memory", lambda: {"risk_constraints": {"max_risk_per_trade_pct": 4.0}})
    assert "user's 4% risk limit" in rr.risk_rules_judge()

    monkeypatch.setattr(ips, "_load_memory", lambda: {"risk_constraints": {}})
    assert "NO maximum-risk limit" in rr.risk_rules_judge()


def test_other_rules_are_untouched_by_the_slot_swap():
    judge = rr.risk_rules_judge({})

    assert "SOURCE FRAUD" in judge
    assert "CURRENCY HEADLINE MISMATCH" in judge
    assert judge.count("\n  1. SYMBOL MISMATCH") == 1
