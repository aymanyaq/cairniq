#!/usr/bin/env python3
"""
Recompile PortfolioAdvisor's few-shot demos (agent/optimized/portfolio_advisor.json).

The committed artifact was compiled against an older single-predictor
PortfolioAdvisor (a lone "advisor" ChainOfThought) and no longer matches the
current two-predictor shape (self.analyst + self.strategist) — loading it
raises `KeyError: 'analyst.predict'`. This script recompiles fresh demos
against the *current* module shape with BootstrapFewShot, scored by a metric
built entirely from checks already in the repo (the deterministic grounding/
total/allocation audits in agent/nodes/risk_manager.py) plus a lightweight
format-compliance and risk-keyword-coverage signal.

The synthetic training scenarios below are fabricated data (no real holdings
or account info) — this repo's policy is to never commit real personal
portfolio data.

Usage:
    LLM_PROVIDER=vertexai AIDLC_MODEL_ID=gemini-2.5-flash \\
        python scripts/optimize_portfolio_advisor.py

Any provider configure_dspy() supports works (bedrock/openai/anthropic/google/
vertexai/azure) — the compiled artifact stores plain-text few-shot demos, not
weights, so it isn't tied to whichever provider compiled it.
"""
import re
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.nodes.risk_manager as risk_manager  # noqa: E402
from agent.dspy_setup import DSPY_AVAILABLE, configure_dspy, dspy  # noqa: E402
from agent.modules import PortfolioAdvisor  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "agent" / "optimized" / "portfolio_advisor.json"

# Fabricated scenarios spanning the risk shapes RiskManager/grounding checks
# care about: concentration, leverage, cash drag, unrealized losses, a small
# starter account, and a plain diversified retirement account as a control.
SCENARIOS = [
    dict(
        name="concentrated_tech_growth",
        portfolio_data=(
            "Holdings: NVDA 420 shares @ $135.20 (cost $52.10, +159.5% unrealized gain), value $56,784; "
            "MSFT 30 shares @ $410.00, value $12,300; Cash $2,100. Account: TFSA (registered, tax-free).\n"
            "Total portfolio value: $71,184 USD."
        ),
        risk_metrics=(
            "Sharpe: 1.1 | Beta: 1.65 | 30d VaR (95%): -$6,200 | "
            "Top-1 concentration: NVDA 79.8% of portfolio | Sector concentration: Technology 100%."
        ),
        user_context="Age 29, software engineer, income $115K, high risk tolerance, saving for a house in 5 years.",
        fixture_ctx={
            "holdings": [
                {"symbol": "NVDA", "allocation_pct": 79.8},
                {"symbol": "MSFT", "allocation_pct": 17.3},
            ],
            "total_value_base": 71184.0,
            "total_value_cad": 71184.0,
            "total_value_usd": 71184.0,
            "base_currency": "USD",
        },
        expect_keywords=["concentrat", "NVDA", "diversif"],
    ),
    dict(
        name="diversified_retirement",
        portfolio_data=(
            "Holdings: VTI 150 shares @ $265.00, value $39,750; BND 300 shares @ $72.50, value $21,750; "
            "VXUS 100 shares @ $58.00, value $5,800; Cash $4,200. Account: RRSP (registered).\n"
            "Total portfolio value: $71,500 CAD."
        ),
        risk_metrics=(
            "Sharpe: 0.85 | Beta: 0.62 | 30d VaR (95%): -$2,100 | "
            "Top-1 concentration: VTI 55.6% of portfolio | Sector concentration: broad-market diversified."
        ),
        user_context="Age 61, retiring in 4 years, income $88K, moderate risk tolerance, wants capital preservation.",
        fixture_ctx={
            "holdings": [
                {"symbol": "VTI", "allocation_pct": 55.6},
                {"symbol": "BND", "allocation_pct": 30.4},
                {"symbol": "VXUS", "allocation_pct": 8.1},
            ],
            "total_value_base": 71500.0,
            "total_value_cad": 71500.0,
            "total_value_usd": 52400.0,
            "base_currency": "CAD",
        },
        expect_keywords=["bond", "retir", "preserv"],
    ),
    dict(
        name="margin_leverage",
        portfolio_data=(
            "Holdings: TSLA 80 shares @ $310.00, value $24,800; SPY 50 shares @ $560.00, value $28,000; "
            "Margin loan balance: -$18,000. Account: Margin (non-registered).\n"
            "Total portfolio value: $34,800 USD (net of margin loan)."
        ),
        risk_metrics=(
            "Sharpe: 0.4 | Beta: 1.4 | 30d VaR (95%): -$9,500 | "
            "Leverage ratio: 1.52x | Margin maintenance cushion: 22% above call level."
        ),
        user_context="Age 35, day-trades occasionally, income $150K, aggressive risk tolerance.",
        fixture_ctx={
            "holdings": [
                {"symbol": "TSLA", "allocation_pct": 47.0},
                {"symbol": "SPY", "allocation_pct": 53.0},
            ],
            "total_value_base": 34800.0,
            "total_value_cad": 34800.0,
            "total_value_usd": 34800.0,
            "base_currency": "USD",
        },
        expect_keywords=["margin", "leverage", "call"],
    ),
    dict(
        name="high_cash_drag",
        portfolio_data=(
            "Holdings: AAPL 20 shares @ $230.00, value $4,600; Cash $45,000. Account: Non-registered.\n"
            "Total portfolio value: $49,600 USD."
        ),
        risk_metrics=(
            "Sharpe: N/A (insufficient invested history) | Beta: 0.05 (cash-dominated) | "
            "Cash allocation: 90.7% of portfolio."
        ),
        user_context=(
            "Age 42, recently sold a business, income $0 currently, low risk tolerance, "
            "unsure how to deploy cash."
        ),
        fixture_ctx={
            "holdings": [{"symbol": "AAPL", "allocation_pct": 9.3}],
            "total_value_base": 49600.0,
            "total_value_cad": 49600.0,
            "total_value_usd": 49600.0,
            "base_currency": "USD",
        },
        expect_keywords=["cash", "deploy", "invest"],
    ),
    dict(
        name="losses_tax_loss_harvest",
        portfolio_data=(
            "Holdings: META 60 shares @ $280.00 (cost $410.00, -31.7% unrealized loss), value $16,800; "
            "GOOGL 40 shares @ $175.00 (cost $140.00, +25% unrealized gain), value $7,000; Cash $1,200. "
            "Account: Non-registered.\nTotal portfolio value: $25,000 USD."
        ),
        risk_metrics=(
            "Sharpe: 0.3 | Beta: 1.2 | 30d VaR (95%): -$2,400 | Top-1 concentration: META 67.2% of portfolio."
        ),
        user_context="Age 50, high income $220K, moderate risk tolerance, asked about reducing this year's tax bill.",
        fixture_ctx={
            "holdings": [
                {"symbol": "META", "allocation_pct": 67.2},
                {"symbol": "GOOGL", "allocation_pct": 28.0},
            ],
            "total_value_base": 25000.0,
            "total_value_cad": 25000.0,
            "total_value_usd": 25000.0,
            "base_currency": "USD",
        },
        expect_keywords=["tax", "loss", "concentrat"],
    ),
    dict(
        name="small_starter_portfolio",
        portfolio_data=(
            "Holdings: VOO 5 shares @ $520.00, value $2,600; Cash $400. Account: TFSA (registered).\n"
            "Total portfolio value: $3,000 CAD."
        ),
        risk_metrics="Sharpe: N/A (too little history) | Beta: 1.0 (index-tracking) | Diversification: single ETF.",
        user_context=(
            "Age 24, first job, income $52K, building an emergency fund alongside investing, "
            "low risk tolerance."
        ),
        fixture_ctx={
            "holdings": [{"symbol": "VOO", "allocation_pct": 86.7}],
            "total_value_base": 3000.0,
            "total_value_cad": 3000.0,
            "total_value_usd": 2200.0,
            "base_currency": "CAD",
        },
        expect_keywords=["emergency", "start", "contribut"],
    ),
]

_ESCAPED_CURRENCY_RE = re.compile(r"\\\$")


def build_trainset() -> list["dspy.Example"]:
    examples = []
    for sc in SCENARIOS:
        ex = dspy.Example(
            portfolio_data=sc["portfolio_data"],
            risk_metrics=sc["risk_metrics"],
            user_context=sc["user_context"],
        ).with_inputs("portfolio_data", "risk_metrics", "user_context")
        ex.fixture_ctx = sc["fixture_ctx"]
        ex.expect_keywords = sc["expect_keywords"]
        examples.append(ex)
    return examples


def portfolio_advisor_metric(example, pred, trace=None) -> float:
    """
    Composite score built from repo assets already used to police live advice
    output, evaluated against this scenario's synthetic fixture instead of the
    live portfolio:
      - the three text-only deterministic grounding audits from
        agent/nodes/risk_manager.py (not-held sell targets, headline total/
        currency, allocation-percent consistency)
      - a verdict-format proxy (non-empty advice, no backslash-escaped
        currency — mirroring PortfolioStrategy's own formatting rule)
      - keyword coverage of the risk concern each scenario is designed to
        surface (concentration, margin, cash drag, ...)
    """
    analysis_summary = getattr(pred, "analysis_summary", "") or ""
    risk_flags = getattr(pred, "risk_flags", "") or ""
    advice = getattr(pred, "advice", "") or ""
    combined = "\n".join([analysis_summary, risk_flags, advice])

    with mock.patch(
        "tools.portfolio_csv.get_portfolio_decision_context",
        return_value=example.fixture_ctx,
    ), mock.patch(
        "tools.market_data.get_realtime_quote",
        return_value={"price": 0},
    ):
        violations = (
            risk_manager.run_deterministic_grounding_audit(combined)
            + risk_manager.run_deterministic_total_audit(combined)
            + risk_manager.run_deterministic_allocation_audit(combined)
        )

    score = 1.0 - 0.25 * len(violations)

    if advice.strip():
        if _ESCAPED_CURRENCY_RE.search(advice):
            score -= 0.2
    else:
        score -= 0.4

    expect = example.expect_keywords
    if expect:
        haystack = f"{analysis_summary} {risk_flags}".lower()
        hits = sum(1 for kw in expect if kw.lower() in haystack)
        score += 0.5 * (hits / len(expect))

    return max(0.0, min(1.5, score))


def main() -> None:
    if not DSPY_AVAILABLE:
        print("DSPy is not installed in this environment.")
        sys.exit(1)

    if not configure_dspy(error_callback=print):
        print(
            "Failed to configure DSPy LM — set LLM_PROVIDER/AIDLC_MODEL_ID "
            "(and provider credentials) before running this script."
        )
        sys.exit(1)

    trainset = build_trainset()

    optimizer = dspy.teleprompt.BootstrapFewShot(
        metric=portfolio_advisor_metric,
        metric_threshold=0.7,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1,
    )

    print(f"Compiling PortfolioAdvisor against {len(trainset)} synthetic scenarios...")
    compiled = optimizer.compile(student=PortfolioAdvisor(), trainset=trainset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(OUTPUT_PATH))
    print(f"Saved compiled demos to {OUTPUT_PATH}")

    # Sanity check: the artifact must load cleanly against a fresh instance of
    # the *current* module shape — this is the exact failure being fixed.
    check = PortfolioAdvisor()
    check.load(str(OUTPUT_PATH))
    print("Reload sanity check passed: compiled artifact loads against current PortfolioAdvisor shape.")


if __name__ == "__main__":
    main()
