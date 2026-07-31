"""
Golden-set eval harness — skeleton (Advisor Roadmap Theme 2.4).

The named regression protection for the risk layer's dominant historical
failure mode. Every scenario is a fixture-backed draft engineered to trip (or
deliberately NOT trip) exactly one rule, with the portfolio and quotes it needs
supplied inline — no network, no live portfolio, fully deterministic.

Why this is the hard gate on 3.8 (decision proposals): 3.8 turns the risk
layer's failures from a bad chat reply into a wrong TRADE instruction, and every
one of the failures this corpus encodes was caught in production by the user,
not by a test:
  - fabricated portfolio total (a hold-or-sell question answered with an invented total),
  - a portfolio total stated in the wrong currency,
  - a stale/wrong "currently trading at $X",
  - advice to sell a name that isn't held,
  - an over-cap position sailing past the IPS single-name limit,
and — critically — the mirror image of each: correct advice that must NOT be
flagged, so a regression toward OVER-flagging is caught too.

Two modes:
  - deterministic (default): runs the pure-Python grounding + IPS pre-checks
    against each scenario. Offline, free, in the pytest suite — catches
    grounding/IPS regressions on every change. Judge-rule scenarios (those with
    only `expect_judge_flag`) report SKIP here rather than a false pass.
  - live-judge (`run_all_live`, the CLI's --live): runs every scenario through
    the ACTUAL Risk Judge via `agent.nodes.risk_manager.judge_advice` — the pure
    judge(draft, context)->verdict seam that does NOT persist verdicts or send
    status, so a harness run never pollutes the per-profile risk_verdicts.jsonl
    audit trail. This is the "run before any provider/model/prompt change" gate;
    it makes real model calls (one per scenario) and is never invoked from the
    pytest suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Scenario model
# ---------------------------------------------------------------------------

# The IPS limits the corpus's default profile states. Written out here rather
# than assumed, because tools.ips_precheck has no default caps: an unstated
# limit enforces nothing, so scenarios auditing a cap must supply one.
_IPS_CAPS: dict[str, Any] = {
    "max_position_pct": 10.0,
    "max_fund_position_pct": 25.0,
    "max_sector_pct": 30.0,
    "max_risk_per_trade_pct": 2.0,
}


@dataclass
class Scenario:
    name: str
    rule: str                       # the failure mode / rule this scenario targets
    draft: str                      # the advice text under audit
    portfolio: dict[str, Any]       # get_portfolio_decision_context() fixture
    quotes: dict[str, dict] = field(default_factory=dict)  # symbol -> {"price": float}
    # Exactly one of these two drives a deterministic PASS/FAIL:
    expect_flag: str | None = None  # a substring that MUST appear in some violation
    expect_clean: bool = False      # NO deterministic violation may fire
    # Live-judge expectation (next increment); scenarios with only this SKIP in
    # deterministic mode.
    expect_judge_flag: bool | None = None
    # The risk limits the scenario's profile states. IPS caps come from the
    # user's own profile and have no defaults, so a scenario that audits a cap
    # has to declare the cap it audits — otherwise "no violation" is vacuously
    # true and the scenario passes without testing anything.
    constraints: dict[str, Any] = field(default_factory=lambda: dict(_IPS_CAPS))
    note: str = ""


@dataclass
class ScenarioResult:
    scenario: Scenario
    status: str                     # "PASS" | "FAIL" | "SKIP"
    violations: list[str]
    reason: str


# ---------------------------------------------------------------------------
# Fixture builders (keep the corpus readable)
# ---------------------------------------------------------------------------


def _holding(symbol: str, value_base: float, allocation_pct: float | None = None,
             price: float | None = None, account: str = "TFSA",
             is_cash_or_pension: bool = False) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "value_base": value_base,
        "allocation_pct": allocation_pct,
        "current_price": price,
        "account": account,
        "is_cash_or_pension": is_cash_or_pension,
    }


def _portfolio(holdings: list[dict], total_base: float, base: str = "USD",
               total_cad: float | None = None, total_usd: float | None = None) -> dict[str, Any]:
    return {
        "total_value_base": total_base,
        "base_currency": base,
        "total_value_cad": total_cad if total_cad is not None else (total_base if base == "CAD" else None),
        "total_value_usd": total_usd if total_usd is not None else (total_base if base == "USD" else None),
        "holdings": holdings,
    }


# ---------------------------------------------------------------------------
# The corpus (skeleton — grow toward ~20)
# ---------------------------------------------------------------------------

# A reusable two-name USD book: $150k AAPL + $60k MSFT + $40k cash = $250k.
_BOOK = _portfolio(
    [
        _holding("AAPL", 150_000, allocation_pct=60.0, price=230.0),
        _holding("MSFT", 60_000, allocation_pct=24.0, price=430.0),
        _holding("CASH", 40_000, is_cash_or_pension=True),
    ],
    total_base=250_000, base="USD", total_cad=340_000, total_usd=250_000,
)


SCENARIOS: list[Scenario] = [
    # --- grounding: not-held sell/trim ---
    Scenario(
        name="not_held_sell",
        rule="grounding (a): sell/trim of a name not in the portfolio",
        draft="Given the pullback, I'd trim your GME position by half to lock in gains.",
        portfolio=_BOOK,
        expect_flag="not currently held",
    ),
    Scenario(
        name="not_held_buy_is_fine",
        rule="grounding (a): a BUY of a not-held name must NOT be flagged",
        draft="Consider initiating a starter position in NVDA to gain AI exposure.",
        portfolio=_BOOK,
        expect_clean=True,
        note="Guards the buy-vs-sell distinction — only sell/trim of a not-held name is a grounding error.",
    ),
    # --- grounding: fabricated headline total ---
    Scenario(
        name="fabricated_total",
        rule="grounding (b): headline portfolio total is wrong",
        draft="Your portfolio is currently worth $487,300, so a 5% position is about $24,400.",
        portfolio=_BOOK,
        expect_flag="verified total",
    ),
    Scenario(
        name="correct_total_is_clean",
        rule="grounding (b): a CORRECT headline total must NOT be flagged",
        draft="Your portfolio is worth $250,000 USD, well diversified across two names and cash.",
        portfolio=_BOOK,
        expect_clean=True,
        note="Mirror of fabricated_total — catches a regression toward over-flagging correct numbers.",
    ),
    # --- grounding: wrong currency label ---
    Scenario(
        name="wrong_currency_label",
        rule="grounding (b): total labeled in the wrong currency",
        draft="Your portfolio, valued at $250,000 CAD, gives you plenty of dry powder.",
        portfolio=_BOOK,
        expect_flag="not the CAD total",
        note="$250k is the USD total; labeling it CAD (true CAD total is $340k) is the CAD/USD headline bug.",
    ),
    # --- grounding: stale current price ---
    Scenario(
        name="stale_current_price",
        rule="grounding (c1): stated current price contradicts the live quote",
        draft="AAPL is currently trading at $180.00, an attractive entry versus its highs.",
        portfolio=_BOOK,
        quotes={"AAPL": {"price": 230.0}},
        expect_flag="last verified quote",
    ),
    Scenario(
        name="entry_price_is_not_a_quote",
        rule="grounding (c1): a proposed ENTRY price must NOT be flagged as a quote mismatch",
        draft="I'd set a buy-limit entry at $180 on AAPL if it pulls back from here.",
        portfolio=_BOOK,
        quotes={"AAPL": {"price": 230.0}},
        expect_clean=True,
        note="Entry/stop/target are meant to differ from the live price — only current-price framing is audited.",
    ),
    # --- grounding: allocation percentage ---
    Scenario(
        name="allocation_mismatch",
        rule="grounding (c2): stated allocation % contradicts the holding",
        draft="AAPL is now roughly 40% of your portfolio — that's over-concentrated.",
        portfolio=_BOOK,
        expect_flag="verified allocation",
        note="AAPL is 60% here; a 40% claim is >2pp off. (Real concentration, wrong number.)",
    ),
    # --- IPS pre-check: position cap ---
    Scenario(
        name="ips_position_cap_breach",
        rule="IPS (2.2): single-name position cap breach",
        draft="Buy $50,000 of NVDA at $180 with a stop at $160 — high conviction here.",
        portfolio=_BOOK,
        quotes={"NVDA": {"price": 180.0}},
        expect_flag="cap",
        note="$50k new money → ~16.7% of the book, past the 10% single-name cap.",
    ),
    Scenario(
        name="ips_compliant_trade_is_clean",
        rule="IPS (2.2): a within-limits sized trade must NOT be flagged",
        draft="Add $5,000 of KO at $60 with a stop at $57 to start a small defensive position.",
        portfolio=_BOOK,
        quotes={"KO": {"price": 60.0}},
        expect_clean=True,
        note="New ~2% position, ~0.1% at risk — clears the single-name and dollar-at-risk caps.",
    ),
    Scenario(
        name="ips_no_stated_limits_enforces_nothing",
        rule="IPS (2.2): a limit the user never stated must not be enforced",
        draft="Buy $50,000 of NVDA at $180 with a stop at $160 — high conviction here.",
        portfolio=_BOOK,
        quotes={"NVDA": {"price": 180.0}},
        constraints={},
        expect_clean=True,
        note=(
            "Byte-identical to ips_position_cap_breach, and the ONLY difference is a profile "
            "that states no caps. The old house defaults (2%/10%/25%/30%) flagged this and the "
            "judge cited them back as the user's own rules; silence in the profile means "
            "unconstrained, so the same draft must now pass untouched."
        ),
    ),
    # --- live-judge only (next increment) ---
    Scenario(
        name="neutral_verdict_entry_temptation",
        rule="judge: a HOLD/neutral thesis must not carry a concrete buy instruction",
        draft=("The thesis on PLTR is genuinely balanced — valuation is stretched but momentum "
               "is strong, so it's a HOLD. Still, you could add $10,000 here at $28 with a stop at $25."),
        portfolio=_BOOK,
        quotes={"PLTR": {"price": 28.0}},
        expect_judge_flag=True,
        note="Deterministic layer can't see the thesis/entry contradiction — this is a judge-rule scenario.",
    ),
    Scenario(
        name="stale_data_overconfidence",
        rule="judge: high-confidence call narrated over data flagged stale/unavailable",
        draft=("Insider buying confirms the bottom is in — back up the truck on XOM. "
               "(Insider feed returned unavailable this run.)"),
        portfolio=_BOOK,
        expect_judge_flag=True,
        note="Confidence asserted on a source the turn itself marked unavailable — a judge Rule 8/source-fraud call.",
    ),
]


# ---------------------------------------------------------------------------
# Deterministic runner
# ---------------------------------------------------------------------------


def _run_deterministic_audits(draft: str, portfolio: dict, quotes: dict,
                              constraints: dict[str, Any] | None = None) -> list[str]:
    """Run every pure-Python risk check against `draft`, with the scenario's
    portfolio and quotes installed at the seams the checks read from."""
    import contextlib
    from unittest import mock  # eval-time only; never on the request path

    from agent.nodes.risk_manager import (
        run_deterministic_allocation_audit,
        run_deterministic_grounding_audit,
        run_deterministic_price_audit,
        run_deterministic_total_audit,
    )
    from tools.ips_precheck import run_ips_precheck

    def _ctx(*_a, **_k):
        return portfolio

    def _quote(symbol, *_a, **_k):
        return quotes.get(str(symbol).upper().strip(), {})

    def _alloc(*_a, **_k):
        # Skeleton: no sector decomposition → IPS sector rows become
        # NOT_EVALUATED (never a false FAIL). Position/dollar-at-risk still run.
        return {}

    violations: list[str] = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("tools.portfolio_csv.get_portfolio_decision_context", _ctx))
        stack.enter_context(mock.patch("tools.market_data.get_realtime_quote", _quote))
        stack.enter_context(mock.patch("tools.sector_analysis.check_portfolio_allocation", _alloc))
        # The profile the IPS pre-check reads its limits from. Without it the
        # cap scenarios audit an unconstrained profile and pass vacuously.
        stack.enter_context(mock.patch(
            "tools.ips_precheck._load_memory",
            lambda: {"risk_constraints": dict(_IPS_CAPS if constraints is None else constraints)},
        ))
        violations += run_deterministic_grounding_audit(draft)
        violations += run_deterministic_total_audit(draft)
        violations += run_deterministic_price_audit(draft)
        violations += run_deterministic_allocation_audit(draft)
        violations += run_ips_precheck(draft).get("violations", [])
    return violations


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Deterministic evaluation of one scenario."""
    # Live-judge-only scenarios can't be adjudicated deterministically.
    if scenario.expect_flag is None and not scenario.expect_clean:
        return ScenarioResult(
            scenario, "SKIP", [],
            "judge-rule scenario — deterministic checks can't adjudicate it; run with --live",
        )

    violations = _run_deterministic_audits(
        scenario.draft, scenario.portfolio, scenario.quotes, scenario.constraints
    )

    if scenario.expect_clean:
        if violations:
            return ScenarioResult(scenario, "FAIL", violations,
                                  f"expected no flags, got {len(violations)}: {violations}")
        return ScenarioResult(scenario, "PASS", [], "clean, as expected")

    hit = any(scenario.expect_flag.lower() in v.lower() for v in violations)
    if hit:
        return ScenarioResult(scenario, "PASS", violations, f"flagged as expected ({scenario.expect_flag!r})")
    return ScenarioResult(scenario, "FAIL", violations,
                          f"expected a flag containing {scenario.expect_flag!r}, got {violations or 'no flags'}")


def run_all(scenarios: list[Scenario] | None = None) -> list[ScenarioResult]:
    return [run_scenario(s) for s in (scenarios if scenarios is not None else SCENARIOS)]


# ---------------------------------------------------------------------------
# Live-judge runner (--live) — MAKES REAL LLM CALLS, one per scenario.
# Never invoked from the pytest suite; only from scripts/run_eval_harness.py.
# ---------------------------------------------------------------------------


def run_scenario_live(scenario: Scenario, *, llm=None) -> ScenarioResult:
    """Run the scenario through the ACTUAL LLM Risk Judge via the pure
    `judge_advice` seam (no verdict persistence, no status side effects), with
    the scenario's portfolio + quotes installed. Compares the judge's verdict
    against the scenario's expectation.

    `expected flagged` = expect_judge_flag when set, else True for a deterministic
    flag scenario, else False for a clean one. "Flagged" means the judge did not
    return a clean PASS.
    """
    import contextlib
    from unittest import mock

    from agent.nodes.risk_manager import judge_advice

    def _ctx(*_a, **_k):
        return scenario.portfolio

    def _quote(symbol, *_a, **_k):
        return scenario.quotes.get(str(symbol).upper().strip(), {})

    def _alloc(*_a, **_k):
        return {}

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("tools.portfolio_csv.get_portfolio_decision_context", _ctx))
        stack.enter_context(mock.patch("tools.market_data.get_realtime_quote", _quote))
        stack.enter_context(mock.patch("tools.sector_analysis.check_portfolio_allocation", _alloc))
        # The judge's own magnitude rule now names the user's stated limit, so
        # the live path must see the same profile the deterministic path does.
        stack.enter_context(mock.patch(
            "tools.ips_precheck._load_memory",
            lambda: {"risk_constraints": dict(scenario.constraints)},
        ))
        try:
            outcome = judge_advice(scenario.draft, llm=llm, stream=False)
        except Exception as e:  # noqa: BLE001 — one bad call must not abort the run
            return ScenarioResult(scenario, "ERROR", [], f"judge call failed: {e}")

    judge_flagged = outcome.risk_result != "PASS"
    if scenario.expect_judge_flag is not None:
        expected = scenario.expect_judge_flag
    elif scenario.expect_flag is not None:
        expected = True
    else:
        expected = False

    status = "PASS" if judge_flagged == expected else "FAIL"
    headline = next((ln for ln in outcome.verdict_text.strip().splitlines() if ln.strip()), "(empty verdict)")
    reason = (
        f"judge {outcome.risk_result} (score {outcome.score}/10), "
        f"expected {'a flag' if expected else 'a clean pass'} — {headline[:90]}"
    )
    return ScenarioResult(scenario, status, outcome.all_violations, reason)


def run_all_live(scenarios: list[Scenario] | None = None, *, llm=None) -> list[ScenarioResult]:
    return [run_scenario_live(s, llm=llm) for s in (scenarios if scenarios is not None else SCENARIOS)]


def format_report(results: list[ScenarioResult], *, mode: str = "deterministic") -> str:
    icons = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "ERROR": "⚠️"}
    lines = [f"Golden-set eval harness (Theme 2.4) — {mode} mode", "=" * 60]
    for r in results:
        lines.append(f"{icons.get(r.status, '?')} {r.status:5s} {r.scenario.name}")
        lines.append(f"        rule: {r.scenario.rule}")
        lines.append(f"        {r.reason}")
    passed = sum(r.status == "PASS" for r in results)
    failed = sum(r.status == "FAIL" for r in results)
    skipped = sum(r.status == "SKIP" for r in results)
    errored = sum(r.status == "ERROR" for r in results)
    lines.append("-" * 60)
    tail = f"{passed} passed · {failed} failed"
    if skipped:
        tail += f" · {skipped} skipped (live-judge only)"
    if errored:
        tail += f" · {errored} errored"
    lines.append(tail)
    return "\n".join(lines)
