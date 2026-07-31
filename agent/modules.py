"""
DSPy Modules for CairnIQ
Chain-of-thought modules that use the defined signatures.
"""
import concurrent.futures
import re

from agent.dspy_setup import dspy
from agent.signatures import (
    BearThesisGeneration,
    BullThesisGeneration,
    FuturePrediction,
    MarketAnalysis,
    NeutralThesisGeneration,
    NewsAnalysis,
    PortfolioAnalysis,
    PortfolioStrategy,
    RiskAssessment,
    ThesisEvaluation,
)


class PortfolioAdvisor(dspy.Module):
    """
    Portfolio advisor module using chain-of-thought reasoning.
    Takes portfolio data and user context, returns personalized advice.
    """
    def __init__(self):
        super().__init__()
        self.analyst = dspy.ChainOfThought(PortfolioAnalysis)
        self.strategist = dspy.ChainOfThought(PortfolioStrategy)

    def forward(self, portfolio_data: str, risk_metrics: str, user_context: str) -> "dspy.Prediction":
        # Step 1: Analyze
        analysis = self.analyst(
            portfolio_data=portfolio_data,
            risk_metrics=risk_metrics
        )

        # Step 2: Strategize
        result = self.strategist(
            analysis_summary=analysis.analysis_summary,
            risk_flags=analysis.risk_flags,
            user_context=user_context
        )
        # Prediction (not a bare string) so callers — and optimizer metrics —
        # can see the intermediate analyst fields alongside the final advice.
        return dspy.Prediction(
            analysis_summary=analysis.analysis_summary,
            risk_flags=analysis.risk_flags,
            advice=result.advice,
        )


class StockAnalyst(dspy.Module):
    """
    Stock analyst module using chain-of-thought reasoning.
    Analyzes fundamentals, technicals, and news to form a recommendation.
    """
    def __init__(self):
        super().__init__()
        self.analyst = dspy.ChainOfThought(MarketAnalysis)

    def forward(self, symbol: str, fundamentals: str, technicals: str, news: str) -> str:
        result = self.analyst(
            symbol=symbol,
            fundamentals=fundamentals,
            technicals=technicals,
            news=news
        )

        # Format the structured reasoning into a single string
        output = f"""
### ⚙️ Data Synthesis
{result.summary}

### 📰 Recent Catalysts
{result.catalysts}

### ⚖️ Bull vs Bear Case
**🐂 Bull Case:**
{result.bull_case}

**🐻 Bear Case:**
{result.bear_case}

### ⚠️ Risk Factors
{result.risk_factors}

### 🎯 Strategic Verdict
Based on the synthesis above, here is the final recommendation:
{result.analysis}
"""
        return output


# --- Evidence grounding for ThesisEvaluation (roadmap 6.3) -------------------
#
# The thesis judge used to see only the three theses. With no evidence in its
# window it could only reward the most persuasive PROSE. It now receives the same
# evidence the theses were generated from, and must name the strongest objection
# to whichever thesis it picks.
#
# A free-text "strongest objection" is precisely the field shape this codebase has
# been burned by before (2026-07-21: an advisor invented past rotation failures
# from ledgers that were empty). So the field is NOT trusted on its own: an
# evidence block with no substance forces the explicit insufficiency sentinel, and
# any financial figure in the objection that cannot be found in the evidence is
# flagged rather than presented as fact.

NO_EVIDENCE_MARKER = "(none available)"

INSUFFICIENT_OBJECTION = (
    "INSUFFICIENT EVIDENCE — no fundamentals, technicals, news, or macro data was available "
    "to argue against the selected thesis. Absence of an objection here is absence of evidence, "
    "not confirmation of the thesis."
)

UNVERIFIED_FIGURE_NOTICE = "⚠️ Unverified figures (not found in the evidence)"

_EVIDENCE_SECTIONS = ("FUNDAMENTALS", "TECHNICALS", "NEWS & SENTIMENT", "MACRO")

# A bare "LABEL:" line, or a "LABEL:" prefix on a line that carries content.
_LABEL_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9 &/_.-]{1,40}:\s*")
# "Symbol: AAPL" names the subject; a ticker is not evidence about the ticker.
_SYMBOL_LINE_RE = re.compile(r"^\s*symbol\s*:", re.IGNORECASE)
# Prompt directives ("CRITICAL INSTRUCTION: ...") ride along in some contexts.
# They are instructions to the model, never evidence.
_DIRECTIVE_LINE_RE = re.compile(r"^\s*(?:[A-Z]+\s+)?INSTRUCTIONS?\b", re.IGNORECASE)

# Figures that make a financial claim: currency amounts, percentages, multiples,
# basis points, and magnitude words. Deliberately NOT every integer — "over the
# next 2 quarters" is not a fabricated metric, and this codebase has already paid
# for false grounding errors that forced retries of correct output.
_MAGNITUDE = r"(?:[KMBT]\b|bn\b|mn\b|tn\b|billion\b|million\b|trillion\b|thousand\b)"
_CLAIM_FIGURE_RE = re.compile(
    rf"[$€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s*{_MAGNITUDE})?"
    r"|\d[\d,]*(?:\.\d+)?\s*%"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:x\b|bps\b|bp\b)"
    rf"|\d[\d,]*(?:\.\d+)?\s*{_MAGNITUDE}",
    re.IGNORECASE,
)
# The evidence side is read permissively — ANY number counts as attestation, because
# evidence arrives as raw tool dumps ("P/E 28.4", "rsi: 71") that rarely carry the
# units the claim regex looks for. Matching is on the numeric VALUE, unit-blind: the
# question this answers is "did this number come from somewhere", and answering it
# leniently is the direction that avoids false accusations.
_ANY_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def build_evidence_context(symbol, fundamentals=None, technicals=None, news=None, macro=None) -> str:
    """Render the thesis evidence block in ONE place, shared by every call site.

    Missing sections are rendered as an explicit ``(none available)`` rather than
    omitted. A truthiness-gated block that simply disappears when empty is what
    the model back-fills with invented specifics; stating the absence out loud is
    what stops it.
    """
    sections = zip(_EVIDENCE_SECTIONS, (fundamentals, technicals, news, macro))
    parts = [f"Symbol: {symbol or 'Unknown'}"]
    for label, values in sections:
        if isinstance(values, str):
            values = [values]
        body = "\n".join(str(v).strip() for v in (values or []) if str(v).strip())
        parts.append(f"{label}:\n{body}" if body else f"{label}:\n{NO_EVIDENCE_MARKER}")
    return "\n\n".join(parts) + "\n"


def evidence_is_empty(evidence_context) -> bool:
    """True when the context carries no substantive evidence.

    Section labels, the ``Symbol:`` line, explicit ``(none available)`` markers and
    prompt directives are all scaffolding — a context made only of those is empty
    no matter how many characters long it is.
    """
    for raw in str(evidence_context or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SYMBOL_LINE_RE.match(line) or _DIRECTIVE_LINE_RE.match(line):
            continue
        line = _LABEL_PREFIX_RE.sub("", line).strip()
        if not line or line.lower() == NO_EVIDENCE_MARKER:
            continue
        return False
    return True


def _numeric_key(text):
    """'$1,234.50' -> '1234.5'. Token-level, so 4.1 never matches inside 44.1."""
    match = _ANY_NUMBER_RE.search(str(text))
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    # A digit run long enough to overflow to inf (an id, a hash) is not a figure.
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return str(int(value)) if value == int(value) else str(value)


def ungrounded_figures(objection, evidence_context) -> list:
    """Financial figures cited in the objection that are absent from the evidence."""
    attested = {
        _numeric_key(m.group(0)) for m in _ANY_NUMBER_RE.finditer(str(evidence_context or ""))
    }
    missing, seen = [], set()
    for match in _CLAIM_FIGURE_RE.finditer(str(objection or "")):
        raw = match.group(0).strip()
        key = _numeric_key(raw)
        if key is None or key in seen:
            continue
        seen.add(key)
        if key not in attested:
            missing.append(raw)
    return missing


def ground_strongest_objection(objection, evidence_context) -> str:
    """Gate the judge's ``strongest_objection`` on the evidence it was given.

    Idempotent: re-running on an already-grounded objection returns it unchanged.
    """
    text = str(objection or "").strip()

    if evidence_is_empty(evidence_context):
        # Hard gate. Whatever the model wrote, it had nothing to write it from.
        return INSUFFICIENT_OBJECTION

    if not text or text.upper().rstrip(".") in {"N/A", "NA", "NONE", "NULL", "UNKNOWN"}:
        return INSUFFICIENT_OBJECTION

    if UNVERIFIED_FIGURE_NOTICE in text:
        return text

    missing = ungrounded_figures(text, evidence_context)
    if missing:
        text += f"\n\n{UNVERIFIED_FIGURE_NOTICE}: {', '.join(missing)} — treat as unsupported."
    return text


class TreeOfThoughtAnalyst(dspy.Module):
    """
    Tree-of-Thought analyst: Generates multiple investment theses (Bull/Bear/Neutral)
    IN PARALLEL to reduce latency, evaluates them against each other,
    and selects the strongest with confidence scoring.
    """
    def __init__(self):
        super().__init__()
        # Parallel generators
        self.bull_gen = dspy.ChainOfThought(BullThesisGeneration)
        self.bear_gen = dspy.ChainOfThought(BearThesisGeneration)
        self.neutral_gen = dspy.ChainOfThought(NeutralThesisGeneration)
        # Evaluator (Still sequential as it needs all inputs)
        self.thesis_evaluator = dspy.ChainOfThought(ThesisEvaluation)

    def thesis_generator(self, symbol: str, context: str):
        """Generate the bull, bear, and neutral theses in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_bull = executor.submit(self.bull_gen, symbol=symbol, context=context)
            future_bear = executor.submit(self.bear_gen, symbol=symbol, context=context)
            future_neutral = executor.submit(self.neutral_gen, symbol=symbol, context=context)

            res_bull = future_bull.result()
            res_bear = future_bear.result()
            res_neutral = future_neutral.result()

        class Theses:
            pass

        theses = Theses()
        theses.thesis_bull = res_bull.thesis
        theses.edge_bull = getattr(res_bull, "identified_edge", "Consensus Trade")
        theses.thesis_bear = res_bear.thesis
        theses.edge_bear = getattr(res_bear, "identified_edge", "Consensus Trade")
        theses.thesis_neutral = res_neutral.thesis
        return theses

    def evaluate_theses(self, symbol: str, theses, evidence_context: str):
        """Judge the three theses AGAINST THE EVIDENCE, then ground the objection.

        This is the single production entry point for the thesis judge — both graph
        nodes and ``forward`` route through it. Calling ``self.thesis_evaluator``
        directly skips the grounding gate, so don't: the evidence field is required
        by the signature precisely so a caller that forgets it fails loudly.
        """
        evaluation = self.thesis_evaluator(
            symbol=symbol,
            evidence_context=evidence_context,
            thesis_bull=theses.thesis_bull,
            thesis_bear=theses.thesis_bear,
            thesis_neutral=theses.thesis_neutral,
        )
        grounded = ground_strongest_objection(
            getattr(evaluation, "strongest_objection", ""), evidence_context
        )
        try:
            evaluation.strongest_objection = grounded
        except Exception:
            pass
        return evaluation

    def format_result(self, symbol, theses, evaluation):
        # Step 3: Format output with all theses and final selection
        selected_thesis = str(evaluation.selected_thesis).upper()
        is_bullish_selection = "BULL" in selected_thesis
        confidence_emoji = {
            "HIGH": "🟢",
            "MEDIUM": "🟡",
            "LOW": "🔴"
        }.get(evaluation.confidence_level.upper(), "⚪")

        # Step 4: Position sizing for BUY recommendations
        position_sizing_section = ""
        if is_bullish_selection:
            try:
                from tools.portfolio_csv import get_portfolio_summary
                from tools.position_sizing import calculate_position_size

                portfolio = get_portfolio_summary()
                if isinstance(portfolio, dict) and "total_value_usd" in portfolio:
                    raw_total = portfolio["total_value_usd"]
                    if isinstance(raw_total, str):
                        raw_total = raw_total.replace("$", "").replace(",", "").replace(" USD", "")
                    portfolio_value = float(raw_total)

                    # No risk % passed: use the user's own stated max-risk rule, and
                    # say so in the heading. Never label this "2% Risk Rule" — that
                    # heading claimed a rule the profile need not contain.
                    sizing = calculate_position_size(portfolio_value)
                    tiers = sizing.get("generic_allocations") or {}
                    tier_text = ", ".join(f"{k} = {v}" for k, v in tiers.items()) or "N/A"
                    if sizing.get("sizing_unavailable"):
                        heading = "Position Sizing (no risk limit set)"
                        max_risk = f"undefined — {sizing.get('risk_basis')}"
                    else:
                        heading = f"Position Sizing ({sizing.get('base_risk_pct')} of portfolio at risk)"
                        max_risk = sizing.get("volatility_adjusted_risk", "N/A")
                    position_sizing_section = f"""

### 💰 {heading}

| Portfolio Value | Max Risk Amount | Suggested Allocations |
|---|---|---|
| {sizing.get('portfolio_value', 'N/A')} | {max_risk} | {tier_text} |
"""
            except Exception as e:
                position_sizing_section = f"\n\n> ⚠️ Position sizing unavailable: {e}\n"

        # Never render an empty/absent objection as blank space — a missing
        # objection reads as "there is no case against this", which is the exact
        # claim the field exists to stop the report from making silently.
        strongest_objection = str(getattr(evaluation, "strongest_objection", "") or "").strip()
        if not strongest_objection:
            strongest_objection = INSUFFICIENT_OBJECTION

        stop_loss = evaluation.stop_loss
        profit_target = evaluation.profit_target
        action_summary = evaluation.action_summary
        if not is_bullish_selection:
            stop_loss = "N/A — no new long entry under the selected thesis"
            profit_target = "N/A — no new long entry under the selected thesis"
            action_summary = (
                f"No new buy entry for {symbol} while the selected thesis is "
                f"{selected_thesis}; use the neutral/bear conditions above as the watchlist trigger."
            )

        output = f"""
## 🌳 Tree-of-Thought Analysis: {symbol}

### Alternative Theses Considered

#### 🐂 Bullish Thesis
{theses.thesis_bull}
**Identified Edge:** {theses.edge_bull}

#### 🐻 Bearish Thesis
{theses.thesis_bear}
**Identified Edge:** {theses.edge_bear}

#### ⚖️ Neutral Thesis
{theses.thesis_neutral}

---

### 🎯 Selected Thesis: **{selected_thesis}**

**Justification:**
{evaluation.justification}

---

### 🥊 Strongest Objection
{strongest_objection}

---

### {confidence_emoji} Confidence: **{evaluation.confidence_level.upper()}**

{evaluation.confidence_reasoning}

---

### 🛡️ Risk Management

| Entry | Stop-Loss | Profit Target |
|---|---|---|
| Current Price | {stop_loss} | {profit_target} |
{position_sizing_section}
---

### 📋 Action Summary
> {action_summary}
"""
        return output

    def forward(self, symbol: str, context: str) -> str:
        theses = self.thesis_generator(symbol=symbol, context=context)

        # Step 2: Evaluate and select the best thesis (Sequential naturally).
        # The judge sees the SAME context the theses were generated from.
        evaluation = self.evaluate_theses(symbol=symbol, theses=theses, evidence_context=context)

        return self.format_result(symbol, theses, evaluation)


class RiskManager(dspy.Module):
    """
    Risk manager module for assessing investment recommendations.
    """
    def __init__(self):
        super().__init__()
        self.risk_checker = dspy.ChainOfThought(RiskAssessment)

    def forward(self, portfolio_context: str, proposed_action: str, user_profile: str) -> str:
        result = self.risk_checker(
            portfolio_context=portfolio_context,
            proposed_action=proposed_action,
            user_profile=user_profile
        )
        return result.assessment


class NewsAnalyst(dspy.Module):
    """
    News analyst module for summarizing market news.
    """
    def __init__(self):
        super().__init__()
        self.analyst = dspy.ChainOfThought(NewsAnalysis)

    def forward(self, news_articles: str, focus_area: str) -> str:
        result = self.analyst(news_articles=news_articles, focus_area=focus_area)
        return result.summary



class PredictionAnalyst(dspy.Module):
    """
    Predictive analyst module for forward-looking scenario analysis.
    """
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(FuturePrediction)

    def forward(self, current_context: str, historical_match: str) -> str:
        result = self.predictor(current_context=current_context, historical_match=historical_match)
        # Return a structured string for the UI
        return f"""
### 🔮 Forward Scenario Analysis: {result.scenario_description}

**Scenario Confidence:** {result.confidence_score}

#### 📅 Short Term (3 Months)
{result.forecast_3mo}

#### 🗓️ Long Term (1 Year)
{result.forecast_1yr}

#### 💰 Portfolio Impact
{result.portfolio_impact}

#### ⚠️ Key Risks
{result.key_risks}

---
**⚠️ IMPORTANT DISCLAIMER:** This analysis is a **hypothetical scenario assessment** for INFORMATIONAL AND EDUCATIONAL PURPOSES ONLY. It does NOT constitute financial advice, a recommendation, or a solicitation to buy or sell any securities. Specific ETFs and sectors mentioned are illustrative examples of historically correlated assets — NOT specific purchase recommendations. Percentage estimates are based on historical analogues and are NOT predictions or guarantees of future performance. Trading based on speculative geopolitical events carries extreme risk, including the total loss of capital. Geopolitical events are inherently unpredictable. **ALWAYS consult a qualified, licensed financial advisor before making any investment decisions.**
"""

# Convenience function to get all modules
def get_all_modules():
    return {
        "portfolio_advisor": PortfolioAdvisor(),
        "stock_analyst": StockAnalyst(),
        "tree_of_thought_analyst": TreeOfThoughtAnalyst(),
        "risk_manager": RiskManager(),
        "news_analyst": NewsAnalyst(),
        "prediction_analyst": PredictionAnalyst()
    }
