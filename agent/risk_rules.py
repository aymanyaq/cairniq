"""
Canonical risk compliance rules — single source of truth.

Two perspectives on the same 10 rules:
  • risk_rules_generator()  – for nodes that *produce* advice (DeepReasoning, PortfolioManager)
  • risk_rules_judge()      – for the Risk Manager node that *audits* advice

Both are FUNCTIONS, not constants, because the magnitude rule has to name the
user's own maximum-risk limit — and most profiles set none. These rules used to
hardcode "the user's 2% max risk rule" for everyone, so the judge enforced a
figure no profile contained and quoted it back as "your 2% limit": the exact
invented-rule attribution judge rule 8 exists to prevent. The limit now comes
from `risk_constraints` in the profile's user_memory.json via
tools.ips_precheck.stated_caps, and when the user has stated none, the rules
say so explicitly rather than falling back to a house default.
"""
from collections.abc import Mapping


def _resolve_caps(caps: Mapping[str, float] | None) -> Mapping[str, float]:
    """The user's stated caps; an unreadable profile means no caps, never defaults."""
    if caps is not None:
        return caps
    try:
        from tools.ips_precheck import stated_caps
        return stated_caps()
    except Exception:
        return {}


def _generator_magnitude_clause(caps: Mapping[str, float]) -> str:
    limit = caps.get("max_risk_per_trade_pct")
    if limit is None:
        return (
            "dollar-at-risk, stated both in the base currency and as a percent of the portfolio. "
            "The user's profile sets NO maximum-risk limit: report that exposure and stop there. Do "
            "not assert a limit, and do not describe a size as within, near, or over budget — there "
            "is no budget to be within"
        )
    return f"dollar-at-risk against the user's {limit:g}% max risk rule"


def _judge_magnitude_clause(caps: Mapping[str, float]) -> str:
    limit = caps.get("max_risk_per_trade_pct")
    if limit is None:
        return (
            "position sizing (current value, percent of portfolio, proposed dollar/share size, and "
            "dollar-at-risk in both currency and percent terms). The user's profile sets NO "
            "maximum-risk limit, so nothing can violate one: never invent, assume, or cite a "
            "percentage risk cap, and never write that a trade breaches or complies with the user's "
            "risk limit. There is no default 2% rule. Flag only sizing figures that are genuinely "
            "absent from the draft"
        )
    return (
        "position sizing (current value, percent of portfolio, proposed dollar/share size, or "
        f"dollar-at-risk against the user's {limit:g}% risk limit)"
    )


_MAGNITUDE_SLOT = "__MAGNITUDE_CLAUSE__"

# ---------------------------------------------------------------------------
# Generator perspective: instructions for analysis-producing nodes
# (DeepReasoning, PortfolioManager)
# ---------------------------------------------------------------------------
_GENERATOR_TEMPLATE = (
    "- RISK MANAGER ALIGNMENT (CRITICAL COMPLIANCE):\n"
    "  1. PORTFOLIO SYMBOLS: If recommending a trim/sell/rebalance action, ensure the symbol is in portfolio_verification. If a symbol is NOT in verified holdings, you must label it Not Held and do not suggest trimming/selling it.\n"
    "  2. CONVERGENCE RULES: If your Strategic Verdict is Neutral, Wait, Preserve Cash, or Avoid, you MUST NOT suggest a starter/half-position entry or FOMO hedge. Provide only watch-and-see conditions.\n"
    "  3. MAGNITUDE CHECK: For tactical, speculative, or short-to-medium-term single-stock trades, explicitly present: current position value, percent of total portfolio, proposed dollar/share size, and " + _MAGNITUDE_SLOT + ". Exclude long-term passive investments, core index/dividend ETFs (e.g., SCHD, VTI, SPY), passive rebalancing, and idle cash allocations from this magnitude check.\n"
    "  4. STRUCTURAL STOPS: Anchor suggested stop-losses to technical support, ATR, or moving averages first, and derive your proposed position size from that distance. Exclude long-term passive investments, core index/dividend ETFs, and passive rebalancing/idle cash allocations, as stop-losses are irrelevant and counterproductive for long-term core ETF holdings.\n"
    "  5. SHORT INTEREST: Do not frame high short interest as bullish or recommend trading short squeezes unless days-to-cover, borrow-cost/utilization, and a clear near-term catalyst are cited.\n"
    "  6. CORRELATION & VOLATILITY: For new high-beta or same-sector additions, address sector concentration and marginal portfolio volatility/correlation impact against the current portfolio.\n"
    "  7. TRADE EDGE: Explicitly define the Trade Edge (Time Horizon, Analytical, or Behavioral) when endorsing a thesis. If no unique edge exists, label it a Consensus/Beta Trade and note that we are just riding market beta.\n"
    "  8. SOURCE TRANSPARENCY: For earnings, event catalysts, or consensus figures, specify source freshness and assign a confidence grade (High/Medium/Low). Do not claim sync timestamps or cost bases that are absent from the context. Verify cited metrics against tool results. Do not assert that a name was previously recommended, evaluated, rejected, entered, or passed over unless that record is in context — a ticker seen in a scan, screen, or watchlist was only SEEN, and a name's absence from holdings never reveals WHY it is absent. Media/guru sentiment is an OPTIONAL overlay: when a scan result has guru_enabled=false (or no guru fields are present at all), do NOT treat the absence of guru/headline data as a data gap or downgrade confidence for that reason — only flag genuinely missing per-ticker data among the sources that are actually enabled.\n"
    "  9. RECONCILIATION: If the user disputes holdings or data, verify against portfolio_verification and tool results, or explicitly acknowledge uncertainty.\n"
    " 10. PROFILE & COMPLIANCE: Ensure recommendations align with the user's profile (age, retirement timeline, risk tolerance, account tax constraints, custom sector/exposure limits, and custom rules/lessons from the user profile).\n"
    " 11. CURRENCY HEADLINE: When stating the user's total portfolio value, always use the figure explicitly marked as the headline/base-currency total in portfolio_verification (labeled with the user's configured base currency). Never substitute a different-currency equivalent (e.g. a USD figure for a CAD-based user, or vice versa) as the total portfolio value, even if unlabeled dollar amounts elsewhere in the context suggest it.\n"
)

# ---------------------------------------------------------------------------
# Judge perspective: violation rules for the Risk Manager compliance node
# ---------------------------------------------------------------------------
_JUDGE_TEMPLATE = (
    "- Flag the following CRITICAL VIOLATIONS for trade recommendations and scans:\n"
    "  1. SYMBOL MISMATCH: Any trim/sell recommendation for a ticker absent from verified holdings (portfolio_verification_context) must be flagged as Not Held.\n"
    "  2. CONVERGENCE FAIL: Suggesting a buy entry or starter/half-position when the thesis is Neutral/Wait/Avoid/Preserve Cash.\n"
    "  3. MAGNITUDE MISS: Any tactical, short-to-medium term, single-stock, or speculative trade recommendation that lacks " + _MAGNITUDE_SLOT + ". Exclude long-term passive investments, core index/dividend ETFs (e.g., SCHD, VTI, SPY), passive portfolio rebalancing, idle cash allocations, and pure watchlist/screener ideas from this requirement.\n"
    "  4. INVALID STOPS: Stop-losses that are not anchored to support, ATR, or technical structure, or where the proposed position size is not derived from that stop distance. Exclude long-term passive investments, core index/dividend ETFs, and passive rebalancing/idle cash allocations, as stop-losses are irrelevant and counterproductive for core long-term holdings. Exclude pure watchlist/screener ideas.\n"
    "  5. SHORT SQUEEZES: Framing short interest as bullish without days-to-cover, borrow-cost, and a clear catalyst.\n"
    "  6. CORRELATION RISK: High-beta or same-sector additions lacking correlation or sector-weight warnings.\n"
    "  7. TRADE EDGE: Recommending a trade that lacks a defined Time Horizon, Analytical, or Behavioral Edge (label it a Consensus Trade).\n"
    "  8. SOURCE FRAUD: Citing brokerage sync, cost basis, or metrics/numbers not present in portfolio_verification_context, conversation evidence, or tool_execution_context. Verify standard market data (prices, valuations, technicals, or insider filing metrics) directly against tool_execution_context; flag them as unverifiable or hallucinated if they mismatch or are fabricated.\n"
    "     THIS APPLIES TO RULES, NOT ONLY NUMBERS. You may cite a user profile rule, mandate, constraint or preference ONLY if it appears in user_profile_memory — quote the wording that supports you. Never write 'per your profile rules' (or 'your mandate', 'your stated preference') for a rule the user did not write. If a draft looks wrong but no profile rule covers it, raise the objection in your OWN voice, as your judgement. Attributing an invented rule to the user is the most damaging error you can make here: the advisor treats your critique as authoritative and will rewrite its next draft to obey it, so a fabricated rule does not just misjudge one answer, it corrupts the next one.\n"
    "     IT ALSO APPLIES TO HISTORY. What was previously recommended, evaluated, screened, entered, rejected, or acted on is a factual claim needing a source in context exactly like a price does — the prior-recommendations block, the active-theses block, portfolio_verification_context, or this conversation. A ticker appearing in a scan, screen, funnel, or watchlist is evidence only that it was SEEN: never that it was evaluated against entry rules, recommended, rejected, or passed over. Flag any narrative that assigns a past decision or rejection reason to a name with no such record (e.g. 'past rotation targets X and Y failed strict entry rules'). Quoting a REAL user rule as the reason does not ground it — grafting a genuine rule name onto an evaluation that never happened is fabrication wearing the costume of evidence, and it is harder to catch than an invented number. Likewise, a name's absence from holdings tells you only that it is not held: never-recommended, recommended-and-declined, and still-pending are indistinguishable from that fact alone, so flag any draft that picks one.\n"
    "  9. RECONCILIATION: Failing to verify or acknowledge uncertainty if the user disputes holdings.\n"
    " 10. PROFILE AND COMPLIANCE ALIGNMENT: Advice conflicting with the user's profile (such as age, retirement timeline, risk tolerance, account tax constraints, custom sector/exposure limits, or custom rules and lessons learned from the user profile). Check all recommendations against the user's specific guidelines dynamically.\n"
    " 11. CURRENCY HEADLINE MISMATCH: The stated total portfolio value uses a currency other than the user's configured base currency shown in portfolio_verification_context (e.g. citing a USD figure as the total for a CAD-based user, or vice versa).\n"
)


def risk_rules_generator(caps: Mapping[str, float] | None = None) -> str:
    """Generator-side rules, with the magnitude rule bound to the user's own limit."""
    return _GENERATOR_TEMPLATE.replace(
        _MAGNITUDE_SLOT, _generator_magnitude_clause(_resolve_caps(caps))
    )


def risk_rules_judge(caps: Mapping[str, float] | None = None) -> str:
    """Judge-side rules, with the magnitude rule bound to the user's own limit."""
    return _JUDGE_TEMPLATE.replace(
        _MAGNITUDE_SLOT, _judge_magnitude_clause(_resolve_caps(caps))
    )
